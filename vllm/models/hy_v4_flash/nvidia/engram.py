# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).
"""Flash Engram n-gram hashing and TP-sharded host embeddings."""

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.hy_v4_flash.common.ngram_hash import (
    hash_ngrams_torch,
    hash_ngrams_triton,
)
from vllm.models.hy_v4_flash.nvidia.boundary_ar import small_rows
from vllm.models.hy_v4_flash.profile import speed_setting
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    _e4m3fn_byte_to_float,
    _pin_host_tensor_exact,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_accelerator_view_from_cpu_tensor,
)


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )


class FlashNgramHashState(nn.Module):
    def __init__(self, config, *, dummy=False):
        super().__init__()
        self._packed_engram = speed_setting("HY4_NODES_ENGRAM")
        self.layout = FlashEngramLayout(config)
        self.visual_token_ids = ()
        self.lookback_depth = self.layout.max_ngram_size - 1
        layers = len(self.layout.layer_ids)
        self.register_buffer(
            "token_maps",
            torch.zeros((layers, config.vocab_size), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "map_sizes", torch.zeros(layers, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "multipliers",
            torch.zeros((layers, self.layout.max_ngram_size), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(self.layout.primes, dtype=torch.int64).flatten(1),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor(self.layout.offsets, dtype=torch.int64),
            persistent=False,
        )
        if dummy:
            identity = torch.arange(config.vocab_size, dtype=torch.int32)
            for layer_id in self.layout.layer_ids:
                self.load_lookup(identity, layer_id)

    def load_lookup(self, value, layer_id):
        index = self.layout.layer_ids.index(layer_id)
        compressed_vocab_size = int(value.max().item()) + 1
        bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
        values = (
            np.random.default_rng(self.layout.seed + 10007 * layer_id).integers(
                0, bound, size=self.layout.max_ngram_size, dtype=np.int64
            )
            * 2
            + 1
        )
        self.token_maps[index, : value.numel()].copy_(value)
        self.map_sizes[index] = value.numel()
        self.multipliers[index].copy_(torch.from_numpy(values))

    @eager_break_during_capture
    def prepare_hashes_with_output(
        self, input_ids, lookback_token_ids, layer_name, out
    ):
        metadata = get_forward_context().attn_metadata
        layer_metadata = metadata[layer_name] if isinstance(metadata, dict) else None
        if (
            self._packed_engram
            and input_ids.is_cuda
            and not self.visual_token_ids
            and out.is_contiguous()
        ):
            self.prepare_hashes(input_ids, lookback_token_ids, layer_metadata, out=out)
        else:
            out.copy_(self.prepare_hashes(input_ids, lookback_token_ids, layer_metadata))

    def prepare_hashes(self, input_ids, lookback_token_ids, metadata, *, out=None):
        if metadata is None or metadata.num_actual_tokens == 0:
            if out is not None:
                return out.fill_(-1)
            return input_ids.new_full(
                (len(self.layout.layer_ids), input_ids.numel(), self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        fn = hash_ngrams_triton if input_ids.is_cuda else hash_ngrams_torch
        hashes = fn(
            input_ids,
            metadata.query_start_loc,
            lookback_token_ids,
            self.token_maps,
            self.map_sizes,
            self.multipliers,
            self.primes,
            self.offsets,
            self.layout.pad_token_id,
            metadata.num_actual_tokens,
            **({"out": out} if out is not None else {}),
        )
        # Later n-grams still see the raw id. This row gathers as zero.
        for token_id in self.visual_token_ids:
            hashes[:, input_ids == token_id, :] = -1
        return hashes


def head_range(sizes, tp, rank):
    """Partition complete hash heads, balancing their row counts."""
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


@triton.jit
def _select_ptr(ptrs, index):
    # Choose a checkpoint parameter without stacking weights at runtime.
    ptr = ptrs[0]
    for i in tl.static_range(1, len(ptrs)):
        ptr = tl.where(index == i, ptrs[i], ptr)
    return ptr


@triton.jit
def _row_rrms(row, dim: tl.constexpr, eps: tl.constexpr):
    return tl.rsqrt(tl.sum(row * row, axis=0) / dim + eps)


@triton.jit
def _flash_engram_gather_kernel(
    storage_ptr,
    ids_ptr,
    output_ptr,
    norm_ptrs,
    vocab_start,
    local_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    id_offset = token * ids_stride_t + (HEAD_START + head) * ids_stride_h
    index = tl.load(ids_ptr + id_offset).to(tl.int64) - vocab_start
    owned = (index >= 0) & (index < local_rows)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < DIM
    storage = storage_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    raw = tl.load(storage + index * DIM + offsets, owned & mask, other=0)
    scales = (storage + local_rows.to(tl.int64) * DIM).to(tl.pointer_type(tl.bfloat16))
    scale = tl.load(scales + index, owned, other=0).to(tl.float32)
    row = _e4m3fn_byte_to_float(raw) * scale
    if len(norm_ptrs) > 0:
        # The checkpoint dequantizes to BF16 before normalizing in FP32.
        row = row.to(tl.bfloat16).to(tl.float32)
        gamma = tl.load(_select_ptr(norm_ptrs, head) + offsets, mask, other=0)
        row = row * _row_rrms(row, DIM, EPS) * gamma.to(tl.float32)
    tl.store(output_ptr + (token * LOCAL_HEADS + head) * DIM + offsets, row, mask)


@triton.jit
def _flash_engram_channel_gate(
    query, key_ptrs, key_gamma_ptrs, query_gamma_ptrs, token, channel, d, mask,
    key_stride_t: tl.constexpr, DIM: tl.constexpr, EPS: tl.constexpr,
    DOT_SCALE: tl.constexpr,
):
    key = tl.load(_select_ptr(key_ptrs, channel) + token * key_stride_t + d, mask, 0)
    key = key.to(tl.float32)
    query = query.to(tl.float32)
    key_gamma = tl.load(_select_ptr(key_gamma_ptrs, channel) + d, mask, 0)
    query_gamma = tl.load(_select_ptr(query_gamma_ptrs, channel) + d, mask, 0)
    key = key * _row_rrms(key, DIM, EPS) * key_gamma.to(tl.float32)
    query = query * _row_rrms(query, DIM, EPS) * query_gamma.to(tl.float32)
    dot = tl.sum(key * query, axis=0) * DOT_SCALE
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    signed = tl.where(dot > 0, magnitude, tl.where(dot < 0, -magnitude, 0.0))
    return tl.sigmoid(signed)


@triton.jit
def _flash_engram_gate_kernel(
    hidden_ptr,
    key_ptrs,
    key_gamma_ptrs,
    query_gamma_ptrs,
    gate_ptr,
    stride_t,
    stride_c,
    stride_d,
    key_stride_t: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    DOT_SCALE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < DIM
    query = tl.load(
        hidden_ptr + token * stride_t + channel * stride_c + d * stride_d, mask, 0
    )
    gate = _flash_engram_channel_gate(
        query, key_ptrs, key_gamma_ptrs, query_gamma_ptrs, token, channel, d, mask,
        key_stride_t, DIM, EPS, DOT_SCALE,
    )
    tl.store(gate_ptr + token * HC + channel, gate)


@triton.jit
def _flash_engram_value_kernel(
    hidden_ptr,
    ids_ptr,
    value_ptr,
    gate_ptr,
    gamma_ptr,
    scale_ptrs,
    output_ptr,
    key_ptrs,
    key_gamma_ptrs,
    query_gamma_ptrs,
    stride_t,
    stride_c,
    stride_d,
    ids_stride_t,
    ids_stride_h,
    value_stride_t: tl.constexpr,
    DOT_SCALE: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    HASH_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1)
    h = tl.arange(0, BLOCK_H)
    ids = tl.load(ids_ptr + token * ids_stride_t + h * ids_stride_h, h < HASH_COLS, -1)
    valid = tl.sum((ids >= 0).to(tl.int32), axis=0) > 0
    d = tl.arange(0, BLOCK_D)
    mask = d < DIM
    value = tl.load(value_ptr + token * value_stride_t + d, mask, 0).to(tl.float32)
    gamma = tl.load(gamma_ptr + d, mask, 0).to(tl.float32)
    scale = tl.load(_select_ptr(scale_ptrs, channel) + d, mask, 0).to(tl.float32)
    residual = tl.load(
        hidden_ptr + token * stride_t + channel * stride_c + d * stride_d, mask, 0
    ).to(tl.float32)
    if FUSE_GATE:
        gate = _flash_engram_channel_gate(
            residual, key_ptrs, key_gamma_ptrs, query_gamma_ptrs, token, channel,
            d, mask, value_stride_t, DIM, EPS, DOT_SCALE,
        )
    else:
        gate = tl.load(gate_ptr + token * HC + channel)
    signal = gamma * value * _row_rrms(value, DIM, EPS)
    # Round the Engram delta to the stream dtype before adding the residual.
    delta = ((gate + scale) * signal).to(hidden_ptr.dtype.element_ty).to(tl.float32)
    output = residual + tl.where(valid, delta, 0.0)
    tl.store(output_ptr + (token * HC + channel) * DIM + d, output, mask)


def flash_engram_gather(
    hashes: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    norm_weights: list[torch.Tensor],
) -> None:
    table = get_forward_context().no_compile_layers[layer_name]
    local_heads = table.head_end - table.head_start
    _flash_engram_gather_kernel[(hashes.shape[0], local_heads)](
        table._uva_ptr,
        hashes,
        output,
        tuple(norm_weights),
        table.vocab_start,
        table.local_rows,
        *hashes.stride(),
        HEAD_START=table.head_start,
        LOCAL_HEADS=local_heads,
        DIM=table.dim,
        BLOCK_D=triton.next_power_of_2(table.dim),
        EPS=torch.finfo(torch.float32).eps,
        num_warps=1,
        enable_fp_fusion=False,
    )


def flash_engram_gather_fake(
    hashes: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    norm_weights: list[torch.Tensor],
) -> None:
    return


direct_register_custom_op(
    op_name="hy4_flash_engram_gather",
    op_func=flash_engram_gather,
    mutates_args=["output"],
    fake_impl=flash_engram_gather_fake,
)


def flash_engram_apply(
    hidden: torch.Tensor,
    hashes: torch.Tensor,
    keys: list[torch.Tensor],
    value: torch.Tensor,
    key_weights: list[torch.Tensor],
    query_weights: list[torch.Tensor],
    gamma: torch.Tensor,
    scales: list[torch.Tensor],
) -> torch.Tensor:
    # Keep the Triton tuples of pointers inside the opaque op for Inductor.
    num_tokens, hc_mult, dim = hidden.shape
    hash_cols = hashes.shape[1]
    output = hidden.new_empty(hidden.shape)
    grid = (num_tokens, hc_mult)
    block_d = triton.next_power_of_2(dim)
    eps = torch.finfo(torch.float32).eps
    # Packed outputs are slices with a wider row stride than a standalone
    # projection. Each channel can apply its own gate before writing the residual.
    fused_gate = value.stride(0) != dim
    gate = None
    if not fused_gate:
        gate = hidden.new_empty((num_tokens, hc_mult), dtype=torch.float32)
        _flash_engram_gate_kernel[grid](
            hidden,
            tuple(keys),
            tuple(key_weights),
            tuple(query_weights),
            gate,
            *hidden.stride(),
            keys[0].stride(0),
            DIM=dim,
            HC=hc_mult,
            BLOCK_D=block_d,
            EPS=eps,
            DOT_SCALE=dim**-0.5,
            enable_fp_fusion=False,
        )
    _flash_engram_value_kernel[grid](
        hidden,
        hashes,
        value,
        gate,
        gamma,
        tuple(scales),
        output,
        tuple(keys) if fused_gate else (),
        tuple(key_weights) if fused_gate else (),
        tuple(query_weights) if fused_gate else (),
        *hidden.stride(),
        *hashes.stride(),
        value.stride(0),
        DOT_SCALE=dim**-0.5,
        FUSE_GATE=fused_gate,
        DIM=dim,
        HC=hc_mult,
        HASH_COLS=hash_cols,
        BLOCK_D=block_d,
        BLOCK_H=triton.next_power_of_2(hash_cols),
        EPS=eps,
        enable_fp_fusion=False,
    )
    return output


def flash_engram_apply_fake(
    hidden: torch.Tensor,
    hashes: torch.Tensor,
    keys: list[torch.Tensor],
    value: torch.Tensor,
    key_weights: list[torch.Tensor],
    query_weights: list[torch.Tensor],
    gamma: torch.Tensor,
    scales: list[torch.Tensor],
) -> torch.Tensor:
    return hidden.new_empty(hidden.shape)


direct_register_custom_op(
    op_name="hy4_flash_engram_apply",
    op_func=flash_engram_apply,
    fake_impl=flash_engram_apply_fake,
)


class FlashEmbedding(nn.Module):
    """Rank-local complete heads in one precisely registered host allocation."""

    def __init__(self, sizes, dim, vllm_config, prefix):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.local_rows = self.vocab_end - self.vocab_start
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.layer_name = prefix
        self.use_cuda = vllm_config.device_config.device_type == "cuda"
        self._uva_ptr = 0
        self._uva_view = None
        if self.use_cuda:
            vllm_config.compilation_config.static_forward_context[prefix] = self
        self.storage = torch.empty(0, dtype=torch.uint8, device="cpu")
        if vllm_config.load_config.load_format == "dummy":
            self._allocate()
            self.storage.zero_()

    def _allocate(self):
        self.storage = torch.empty(
            self.local_rows * (self.dim + 2), dtype=torch.uint8, device="cpu"
        )
        if self.use_cuda:
            _pin_host_tensor_exact(self.storage)
            self._uva_view = get_accelerator_view_from_cpu_tensor(self.storage)
            self._uva_ptr = self._uva_view.data_ptr()

    def load_from_safetensors(self, path, weight_name):
        """Read only this rank's row range, in bounded blocks, from one shard."""
        with safe_open(path, framework="pt", device="cpu") as reader:
            weight_slice = reader.get_slice(weight_name)
            if weight_slice.get_dtype() == "BF16":
                raise ValueError(
                    "Engram tables require FP8 weights and BF16 row scales"
                )
            scale_slice = reader.get_slice(weight_name + "_scale_inv")
            self._allocate()
            weight = self.storage[: self.local_rows * self.dim].view(
                self.local_rows, self.dim
            )
            scale = self.storage[self.local_rows * self.dim :].view(torch.bfloat16)
            for start in range(self.vocab_start, self.vocab_end, 131072):
                end = min(start + 131072, self.vocab_end)
                local_start = start - self.vocab_start
                local_end = end - self.vocab_start
                weight[local_start:local_end].copy_(
                    weight_slice[start:end, :].view(torch.uint8)
                )
                scale[local_start:local_end].copy_(scale_slice[start:end])

    def forward(self, hashes):
        if hashes.is_cuda or hashes.shape[0] == 0:
            output = hashes.new_empty(
                (hashes.shape[0], self.head_end - self.head_start, self.dim),
                dtype=torch.float32,
            )
            if hashes.shape[0]:
                torch.ops.vllm.hy4_flash_engram_gather(
                    hashes, output, self.layer_name, []
                )
            return output
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        local = indices - self.vocab_start
        valid = (local >= 0) & (local < self.local_rows)
        local = torch.where(valid, local, 0)
        weights = self.storage[: self.local_rows * self.dim].view(
            self.local_rows, self.dim
        )
        scales = self.storage[self.local_rows * self.dim :].view(torch.bfloat16)
        rows = weights[local].view(torch.float8_e4m3fn).float()
        output = rows * scales[local].float().unsqueeze(-1)
        return output.masked_fill_(~valid.unsqueeze(-1), 0)


class FlashEngram(nn.Module):
    """Recall and gate a rank-sharded Flash Engram table into iHC streams."""

    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        self.multi_head_embedding = FlashEmbedding(
            sizes, layout.head_dim, vllm_config, f"{prefix}.multi_head_embedding"
        )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = self.multi_head_embedding
        local_heads = local.head_end - local.head_start
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local_heads)]
        )
        self.value_proj = nn.Linear(local_heads * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local_heads * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = local.head_start * layout.head_dim
        self.stop = local.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(
                projection.weight, {"weight_loader": self._load_projection}
            )
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})
        self._packed_engram = speed_setting("HY4_NODES_ENGRAM")
        self.register_buffer("_projection_weight", None, persistent=False)
        self.register_buffer("_projection_bias", None, persistent=False)
        if self._packed_engram:
            self.prepare_projection_bundle()

    @torch.no_grad()
    def prepare_projection_bundle(self) -> None:
        weight = self.value_proj.weight
        if (
            self._projection_weight is not None
            or self.multi_head_embedding.tp_size != 4
            or self.hc_mult != 4
            or self.hidden_size != 3072
            or weight.dtype != torch.float16
            or not weight.is_cuda
            or envs.VLLM_BATCH_INVARIANT
            or torch.cuda.get_device_capability(weight.device) != (7, 0)
        ):
            return
        projections = (*self.key_projs, self.value_proj)
        packed_weight = torch.cat([proj.weight for proj in projections], dim=0)
        packed_bias = torch.cat([proj.bias for proj in projections], dim=0)
        for proj, w, b in zip(
            projections,
            packed_weight.split(self.hidden_size),
            packed_bias.split(self.hidden_size),
            strict=True,
        ):
            # Preserve each checkpoint parameter and loader. Both paths use
            # the same storage for the original five matrices and biases.
            proj.weight.set_(w)
            proj.bias.set_(b)
        self._projection_weight = packed_weight
        self._projection_bias = packed_bias

    def _load_projection(self, parameter, value):
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(
            parameter, value if self.rank == 0 else torch.zeros_like(value)
        )

    def forward(self, hidden: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        if not hidden.is_cuda:
            return self.forward_native(hidden, hashes)
        num_tokens = hidden.shape[0]
        if num_tokens == 0:
            return hidden

        local = self.multi_head_embedding
        x = hidden.new_empty((num_tokens, self.stop - self.start))
        torch.ops.vllm.hy4_flash_engram_gather(
            hashes,
            x,
            local.layer_name,
            [norm.weight for norm in self.lookup_per_head_norms],
        )
        if (
            self._packed_engram
            and (num_tokens <= 4 or num_tokens <= small_rows())
            and self._projection_weight is not None
            and self._projection_weight.device == x.device
            and self._projection_weight.dtype == x.dtype
        ):
            partial = F.linear(x, self._projection_weight, self._projection_bias)
            packed = tensor_model_parallel_all_reduce(partial)
            *keys, value = packed.split(self.hidden_size, dim=1)
        else:
            keys = [self._project(proj, x) for proj in self.key_projs]
            value = self._project(self.value_proj, x)
        return torch.ops.vllm.hy4_flash_engram_apply(
            hidden,
            hashes,
            keys,
            value,
            [norm.weight for norm in self.norm_key],
            [norm.weight for norm in self.norm_query],
            self.unified_pre_gamma,
            list(self.unified_per_hc_scales),
        )

    def _project(self, projection: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        result = projection(x)
        if self.multi_head_embedding.tp_size > 1:
            result = tensor_model_parallel_all_reduce(result)
        return result

    def forward_native(
        self, hidden: torch.Tensor, hashes: torch.Tensor
    ) -> torch.Tensor:
        local = self.multi_head_embedding
        rows = local(hashes).to(torch.bfloat16).float()
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack(
            [norm.weight for norm in self.lookup_per_head_norms]
        ).float()
        rows = rows.masked_fill(
            (hashes[:, local.head_start : local.head_end] < 0).unsqueeze(-1), 0
        )
        x = (
            (rows * (rows.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma)
            .to(hidden.dtype)
            .flatten(1)
        )
        gates = []
        for channel in range(self.hc_mult):
            key = self._project(self.key_projs[channel], x).float()
            query = hidden[:, channel].float()
            key = key * (key.square().mean(-1, keepdim=True) + eps).rsqrt()
            query = query * (query.square().mean(-1, keepdim=True) + eps).rsqrt()
            key = key * self.norm_key[channel].weight.float()
            query = query * self.norm_query[channel].weight.float()
            gate = (key * query).sum(-1) * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self._project(self.value_proj, x).float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(hidden.dtype)
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
