# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).

"""Hy4 Flash text MLA with a shared paged reader for both layer types."""

from functools import partial
from math import lcm
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from .cache import POOL_PAGE, FlashCache, FlashMetadata, FlashSWABackend
from .dcp import apply_attn_sink
from .decode import decode_slots
from ..nvidia.boundary_ar import can_fuse_reduce_norm, reduce_norm
from ..profile import speed_setting
from . import w8
from .dense import enable_skinny
from .f7_attn import (
    F7_ATTN_SHORT_SPLITS,
    F7_FP8_DS_MLA_BYTES,
    can_use_f7_attn_out,
    can_use_f7_attn_prep,
    ensure_f7_prep_weights,
    get_f7_attn_scratch,
    load_hy4_f7_attn_operator,
    run_f7_attn_out,
    run_f7_attn_prep,
)
from .indexpool import FlashCompressedIndexer
from .kernels import (
    cache_fp8_ds_mla,
    pack_fp8_ds_mla_torch,
    paged_attention,
    paged_attention_torch,
)
from .rotary import apply_rotary, rotary_phases, rotate_and_join, take_tokens
from .splitk import decode_workspace_specs, paged_attention_splitk
from .swa import window_indices


if TYPE_CHECKING:
    from .verification import VerificationGraph


class HYV4FlashMLA(nn.Module):
    """Projections shared by target and read-only DFly attention."""

    def __init__(
        self, config, layer_idx, prefix, quant, post_proj_norm=None, *, allow_w8=False
    ):
        super().__init__()
        # Projection names and TP axes follow vLLM PR #68296. The KV-A and
        # IndexPool projections are replicated, so every rank sees the same KV.
        self.is_window_layer = config.layer_types[layer_idx] == "sliding_attention"
        self.layer_idx = self.layer_id = layer_idx
        self.prefix = prefix
        self._fuse_text_positions = speed_setting("VLLM_HY4_POSITION_FUSION")
        self._f7_attn_prep = speed_setting("HY4_F7_ATTN_PREP")
        self._prep_split_w8 = speed_setting("HY4_PREP_SPLIT_W8")
        self._f7_attn_out = speed_setting("HY4_F7_ATTN_OUT")
        self._attn3_fp8_prmt = speed_setting("HY4_ATTN3_FP8_PRMT")
        self._attn4b_swa_tensor = self.is_window_layer and speed_setting(
            "HY4_ATTN4B_SWA_TENSOR"
        )
        self._attn3_w8_merge = speed_setting("HY4_ATTN3_W8_MERGE")
        self._boundary_ar_norm = speed_setting("HY4_BOUNDARY_AR_NORM")
        attn_prefix = "swa_" if self.is_window_layer else ""
        self.num_heads = getattr(config, f"{attn_prefix}num_attention_heads")
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = getattr(config, f"{attn_prefix}qk_nope_head_dim")
        self.qk_rope_head_dim = getattr(config, f"{attn_prefix}qk_rope_head_dim")
        self.v_head_dim = getattr(config, f"{attn_prefix}v_head_dim")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.qk_head_dim**-0.5
        hidden_size, q_rank = config.hidden_size, config.q_lora_rank
        eps = config.rms_norm_eps
        self.q_a_proj = ReplicatedLinear(
            hidden_size,
            q_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(q_rank, eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps)
        self.q_b_proj = ColumnParallelLinear(
            q_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        if not (allow_w8 and w8.enable_linear(self.o_proj, "O_PROJ")):
            enable_skinny(self.o_proj)
        if allow_w8:
            for projection in (
                self.q_a_proj,
                self.kv_a_proj_with_mqa,
                self.q_b_proj,
                self.linear_gate,
            ):
                w8.enable_linear(projection, "ATTN")
            w8.enable_absorbed(
                self.kv_b_proj, self.num_local_heads, self.qk_nope_head_dim, self.v_head_dim
            )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            self.learnable_sink_param, {"weight_loader": sharded_weight_loader(0)}
        )
        if post_proj_norm is None:
            post_proj_norm = config.sandwich_norm and (
                self.is_window_layer or not config.dsa_skip_attn_post_norm
            )
        self.post_proj_norm = RMSNorm(hidden_size, eps) if post_proj_norm else None
        rope_params = config.rope_for_layer(layer_idx)
        theta = rope_params["rope_theta"]
        self._rope_theta = float(theta)
        self.mrope_section = rope_params.get("mrope_section")
        self.register_buffer(
            "inv_freq",
            1.0
            / (
                theta
                ** (
                    torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                    / self.qk_rope_head_dim
                )
            ),
            persistent=False,
        )

    def _project(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
        kv_c = self.kv_a_layernorm(kv_c.contiguous())
        cos, sin = rotary_phases(positions, self.inv_freq, self.mrope_section)
        q_nope, q_pe = q.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
        # Three-coordinate image positions and bulk prefill retain the
        # original path. The launch width is stable within each CUDA graph.
        fuse_positions = (
            self._fuse_text_positions
            and q.is_cuda
            and q.dtype == torch.float16
            and self.qk_rope_head_dim == 64
            and positions.ndim == 1
            and 1 <= q.shape[0] <= 32
        )
        if not fuse_positions:
            q_pe = apply_rotary(q_pe, cos[:, None], sin[:, None])
            k_pe = apply_rotary(k_pe, cos, sin)
        w_uk_t = getattr(self.kv_b_proj, "_hy4_w8_uk", None)
        if w_uk_t is None:
            kv_b = self.kv_b_proj.weight.view(self.num_local_heads, -1, self.kv_lora_rank)
            w_uk, w_uv = kv_b.split((self.qk_nope_head_dim, self.v_head_dim), dim=1)
            latent_q = torch.bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)
        else:
            w_uv = self.kv_b_proj._hy4_w8_uv
            latent_q = w8.apply_absorbed(
                q_nope.transpose(0, 1), w_uk_t, self.kv_lora_rank,
                self.qk_nope_head_dim,
            ).transpose(0, 1)
        if fuse_positions:
            q, k_pe = rotate_and_join(q_pe, k_pe, latent_q, cos, sin)
        else:
            q = torch.cat((latent_q, q_pe), dim=-1)

        return q, kv_c, k_pe, q_c, cos, sin, w_uv

    def _project_with_gate(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        if can_use_f7_attn_prep(self, positions, hidden_states):
            return run_f7_attn_prep(self, positions, hidden_states)
        q, kv_c, k_pe, q_c, cos, sin, w_uv = self._project(positions, hidden_states)
        return q, kv_c, k_pe, q_c, cos, sin, w_uv, None

    def _project_output(
        self,
        latent: torch.Tensor,
        w_uv: torch.Tensor,
        hidden_states: torch.Tensor,
        gate_sig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = w8.apply_absorbed(
            latent.transpose(0, 1), w_uv, self.v_head_dim, self.kv_lora_rank
        )
        out = out.transpose(0, 1).reshape(-1, self.num_local_heads * self.v_head_dim)
        if gate_sig is None:
            gate_sig = torch.sigmoid(self.linear_gate(hidden_states)[0])
        out = out * gate_sig
        return self._output_projection(out)

    def _output_projection(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.post_proj_norm
        if self._boundary_ar_norm and can_fuse_reduce_norm(x, norm):
            assert norm is not None
            return self.o_proj(x, reduce_output=partial(reduce_norm, norm=norm))[0]
        out = self.o_proj(x)[0]
        return norm(out) if norm is not None else out


class HYV4FlashAttention(HYV4FlashMLA):
    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        super().__init__(
            config, layer_idx, prefix, vllm_config.quant_config, allow_w8=True
        )
        self._index_bypass = speed_setting("VLLM_HY4_INDEX_BYPASS", vllm_config)
        self._attncore_prep = speed_setting("HY4_ATTNCORE_PREP", vllm_config)
        self._attncore_split = speed_setting("HY4_ATTNCORE_SPLIT", vllm_config)
        self._pf_attn_tc = (
            speed_setting("HY4_PF_ATTN_TC", vllm_config)
            and vllm_config.device_config.device_type == "cuda"
            and current_platform.is_device_capability(70)
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
        )
        if self._pf_attn_tc:
            load_hy4_f7_attn_operator("paged_prefill_sm70")
        self._pf_attn_fix = (
            speed_setting("HY4_PF_ATTN_FIX", vllm_config) if self._pf_attn_tc else 0
        )
        self._short_graph_width = (
            getattr(config, "compressed_indexer_group_topk", 0)
            * config.compressed_indexer_compress_ratio
        )
        self._capture_dfly = False
        self.window = config.sliding_window
        world = vllm_config.parallel_config.decode_context_parallel_size
        global_page = POOL_PAGE * config.compressed_indexer_compress_ratio
        block_size = (
            global_page // world
            if world > 1
            else lcm(vllm_config.cache_config.block_size, global_page)
        )
        sharded = world > 1 and not self.is_window_layer
        if sharded:
            layers, _, attention = prefix.rsplit(".", 2)
            window_idx = config.layer_types.index("sliding_attention")
            self._swa_prefix = f"{layers}.{window_idx}.{attention}.attn"
        fp8_kv = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        main_cache_dtype = "fp8_ds_mla" if fp8_kv else "auto"
        cache_args = dict(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=torch.uint8 if fp8_kv else torch.float16,
            cache_dtype_str=main_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(main_cache_dtype),
            dcp_sharded=sharded,
        )
        if self.is_window_layer:
            self.attn = FlashCache(
                vllm_config,
                f"{prefix}.attn",
                SlidingWindowMLASpec(
                    **cache_args,
                    sliding_window=self.window,
                ),
                FlashSWABackend,
            )
            self.indexer = None
            self.width = self.window
        else:
            self.attn = FlashCache(
                vllm_config, f"{prefix}.attn", MLAAttentionSpec(**cache_args)
            )
            self.indexer = FlashCompressedIndexer(
                vllm_config,
                config,
                f"{prefix}.indexer",
                block_size,
            )
            self.width = self.indexer.width
        self.topk_indices_buffer = topk_indices_buffer
        self.dcp_group = self.indexer.dcp_group if self.indexer is not None else None
        self.dcp_world = world if sharded else 1
        self.dcp_rank = self.dcp_group.rank_in_group if self.dcp_group else 0
        spec = vllm_config.speculative_config
        verify_rows = 1 + spec.num_speculative_tokens if spec is not None else 1
        compilation = vllm_config.compilation_config
        if (
            spec is not None
            and spec.method == "dfly"
            and compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
            and envs.VLLM_HY4_DFLY_VERIFY_INDEX == "batched"
        ):
            # Size the shared split-K arena during forward profiling so
            # capture never has to grow it to fit the largest exact graph.
            capture_rows = compilation.cudagraph_capture_sizes or ()
            verify_rows = max(verify_rows, max(capture_rows, default=0))
        self._decode_workspace_specs = decode_workspace_specs(
            max(vllm_config.scheduler_config.max_num_seqs, verify_rows),
            self.num_local_heads * self.dcp_world,
            self.width // self.dcp_world,
            self.kv_lora_rank,
        )
        self._short_split_specs = decode_workspace_specs(
            1,
            self.num_local_heads,
            self.width,
            self.kv_lora_rank,
            minimum_splits=F7_ATTN_SHORT_SPLITS,
        )
        self.register_buffer(
            "_k_scale", torch.ones((), dtype=torch.float32), persistent=False
        )

    def _aggregate(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
        *,
        prep_writes_cache: bool = False,
        prep_writes_indices: bool = False,
        w_uv: torch.Tensor | None = None,
        gate_sig: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        main = metadata[self.attn.prefix]
        tokens = main.num_actual_tokens
        cache = self.attn.kv_cache
        page = cache.shape[1]
        indices = self.topk_indices_buffer[:tokens, : self.width]
        slots = main.slot_mapping[:tokens]
        local_width = self.width // self.dcp_world
        context = get_forward_context()
        graph = context.additional_kwargs.get("hy4_verify_graph")
        if graph is not None:
            counts = self._verification_slots(
                hidden_states[:tokens],
                q_c[:tokens],
                take_tokens(positions, tokens),
                cos[:tokens],
                sin[:tokens],
                metadata,
                indices,
                page,
                graph,
            )
        elif q.is_cuda and main.decode:
            counts = self._decode_slots(
                hidden_states,
                q_c,
                positions,
                cos,
                sin,
                metadata,
                indices,
                page,
                slots_ready=prep_writes_indices,
            )
        else:
            counts = self._eager_slots(
                hidden_states,
                q_c,
                positions,
                cos,
                sin,
                metadata,
                indices,
                page,
                decode_verification=q.is_cuda,
            )
        if self._capture_dfly:
            step = (
                graph
                or context.additional_kwargs.get("hy4_dfly_plain")
                or context.additional_kwargs.get("hy4_dfly")
            )
            if step is not None:
                step.capture(self.layer_idx, cache, indices, counts)
        if self.dcp_group is not None:
            indices = indices[:, :local_width]
            q = self.dcp_group.all_gather(q[:tokens].contiguous(), dim=1)
            lse = q.new_empty(q.shape[:2], dtype=torch.float32)
            sinks = None
        else:
            lse = None
            sinks = self.learnable_sink_param
        if q.is_cuda:
            if not prep_writes_cache:
                if cache.dtype == torch.uint8:
                    cache_fp8_ds_mla(kv_c[:tokens], k_pe[:tokens], cache, slots)
                else:
                    ops.concat_and_cache_mla(
                        kv_c[:tokens],
                        k_pe[:tokens],
                        cache,
                        slots,
                        "auto",
                        self._k_scale,
                    )
            verification_width = graph.width if graph is not None else 0
            batch_verification = (
                graph is not None and graph.index_workspace is not None
            )
            single_decode = (
                (verification_width > 0 and batch_verification)
                or (
                    verification_width == 0
                    and (main.verification is None or not main.verification.pending)
                    and main.decode
                )
            )
            if (
                single_decode
                and w_uv is not None
                and can_use_f7_attn_out(
                    self, q[:tokens], cache, indices, counts, sinks, lse, w_uv
                )
            ):
                descriptor = context.batch_descriptor
                short_splits = (
                    self._attncore_split
                    and q.shape[0] == 1
                    and tokens == 1
                    and graph is None
                    and main.verification is None
                    and not self._capture_dfly
                    and descriptor is not None
                    and descriptor.attention_context_bucket == self._short_graph_width
                    and cache.dtype == torch.uint8
                    and q.shape[1] in (8, 16)
                )
                specs = (
                    self._short_split_specs
                    if short_splits
                    else self._decode_workspace_specs
                )
                if tokens > specs[0][0][0]:
                    specs = decode_workspace_specs(
                        tokens, q.shape[1], indices.shape[1], self.kv_lora_rank
                    )
                state, acc = current_workspace_manager().get_simultaneous(*specs)
                gate_tok = (
                    gate_sig[:tokens]
                    if gate_sig is not None
                    else torch.sigmoid(self.linear_gate(hidden_states[:tokens])[0])
                )
                assert sinks is not None
                gated = run_f7_attn_out(
                    q[:tokens],
                    cache,
                    indices,
                    counts,
                    sinks,
                    self.scaling,
                    state,
                    acc,
                    w_uv,
                    gate_tok,
                    short_splits=short_splits,
                    prmt_fp8=self._attn3_fp8_prmt,
                    swa_tensor=self._attn4b_swa_tensor and 2 <= tokens <= 4,
                    narrow_w8=self._attn3_w8_merge,
                    legacy_max_splits=self._decode_workspace_specs[0][0][2],
                )
                return gated, True
            latent = self._read_cache(
                q[:tokens],
                cache,
                indices,
                counts,
                sinks,
                lse,
                main,
                verification_width=verification_width,
                batch_verification=batch_verification,
            )
            return (
                self._merge_dcp(latent, lse) if lse is not None else latent,
                False,
            )
        if cache.dtype == torch.uint8:
            kv = pack_fp8_ds_mla_torch(kv_c[:tokens], k_pe[:tokens])
        else:
            kv = torch.cat((kv_c[:tokens], k_pe[:tokens]), dim=-1)
        cache[slots // page, slots % page] = kv
        return (
            paged_attention_torch(
                q[:tokens],
                cache,
                indices,
                counts,
                self.learnable_sink_param,
                self.scaling,
                self.kv_lora_rank,
            ),
            False,
        )

    def _bypass_index_history(self) -> bool:
        # The dispatcher checks the current upper bound on each replay.
        # The existing slot kernel still reads each live length on device.
        if not self._index_bypass or not is_forward_context_available():
            return False
        descriptor = get_forward_context().batch_descriptor
        return (
            descriptor is not None and descriptor.attention_context_bucket == self.width
        )

    def _decode_slots(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
        indices: torch.Tensor,
        page: int,
        *,
        slots_ready: bool = False,
    ) -> torch.Tensor:
        main = metadata[self.attn.prefix]
        tokens = main.num_actual_tokens
        slots = main.slot_mapping[:tokens]
        local_width = self.width // self.dcp_world
        active_slots = (
            slots
            if self.dcp_group is None
            else metadata[self._swa_prefix].slot_mapping[:tokens]
        )
        counts = main.counts
        assert counts is not None
        groups = None
        ratio = 1
        if self.indexer is not None:
            ratio = self.indexer.ratio
            groups = self.indexer.decode_groups(
                hidden_states[:tokens],
                q_c[:tokens],
                take_tokens(positions, tokens),
                self.inv_freq,
                cos[:tokens],
                sin[:tokens],
                metadata[self.indexer.k_cache.prefix],
                metadata[self.indexer.tail_cache.prefix],
                active_slots,
                bypass_history=self._bypass_index_history(),
            )
        counts = counts[:tokens]
        if slots_ready:
            # The prep kernel wrote these using the same live metadata.
            assert groups is None
            return counts
        decode_slots(
            main.seq_lens,
            active_slots,
            main.block_table,
            groups,
            indices[:, :local_width],
            counts,
            page,
            ratio,
            self.dcp_world,
            self.dcp_rank,
        )
        return counts

    def _verification_slots(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
        indices: torch.Tensor,
        page: int,
        graph: "VerificationGraph",
    ) -> torch.Tensor:
        main = metadata[self.attn.prefix]
        width = graph.width
        tokens = hidden.shape[0]
        batch = tokens // width
        offsets = torch.arange(1 - width, 1, device=positions.device, dtype=torch.int32)
        lengths = (main.seq_lens[:batch, None] + offsets).flatten()
        slots = main.slot_mapping[:tokens]
        groups = None
        ratio = 1
        if self.indexer is not None:
            indexer = self.indexer
            ratio = indexer.ratio
            tail_name = indexer.tail_cache.prefix
            groups = indexer.verify_groups(
                hidden,
                q_c,
                positions,
                self.inv_freq,
                cos,
                sin,
                metadata[indexer.k_cache.prefix],
                metadata[tail_name],
                lengths,
                slots,
                graph.layer_tails[tail_name],
                width,
                graph.index_workspace,
                bypass_history=self._bypass_index_history(),
            )
        counts = torch.empty_like(lengths)
        decode_slots(
            lengths,
            slots,
            main.block_table,
            groups,
            indices,
            counts,
            page,
            ratio,
            query_width=width,
        )
        return counts

    def _read_cache(
        self,
        q: torch.Tensor,
        cache: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        sinks: torch.Tensor | None,
        lse: torch.Tensor | None,
        main: FlashMetadata,
        *,
        verification_width: int = 0,
        batch_verification: bool = False,
    ) -> torch.Tensor:
        verification = main.verification
        if verification_width and batch_verification:
            ranges = [(slice(None), True)]
        elif verification_width:
            ranges = [
                (slice(start, start + verification_width), True)
                for start in range(0, q.shape[0], verification_width)
            ]
        elif verification is None or not verification.pending:
            ranges = [(slice(None), main.decode)]
        else:
            ranges = [
                (slice(span.begin, span.end), verification.verifies(span))
                for span in main.spans
            ]

        outputs = []
        for rows, decode in ranges:
            row_q = q[rows]
            row_lse = None if lse is None else lse[rows]
            if decode:
                specs = self._decode_workspace_specs
                if len(row_q) > specs[0][0][0]:
                    specs = decode_workspace_specs(
                        len(row_q), q.shape[1], indices.shape[1], self.kv_lora_rank
                    )
                state, acc = current_workspace_manager().get_simultaneous(*specs)
                output = paged_attention_splitk(
                    row_q,
                    cache,
                    indices[rows],
                    counts[rows],
                    sinks,
                    self.scaling,
                    state,
                    acc,
                    lse=row_lse,
                )
            else:
                output = paged_attention(
                    row_q,
                    cache,
                    indices[rows],
                    counts[rows],
                    sinks,
                    self.scaling,
                    self.kv_lora_rank,
                    lse=row_lse,
                    tensor_cores=self._pf_attn_tc,
                    tensor_core_mode=self._pf_attn_fix,
                )
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def _merge_dcp(self, latent: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        assert self.dcp_group is not None
        out, global_lse = cast(
            tuple[torch.Tensor, torch.Tensor],
            dcp_a2a_lse_reduce(
                latent, lse, self.dcp_group, return_lse=True, is_lse_base_on_e=True
            ),
        )
        return apply_attn_sink(out.half(), global_lse, self.learnable_sink_param)

    def _profile_dcp(self, q: torch.Tensor) -> None:
        assert self.dcp_group is not None and self.indexer is not None
        # The profile has no cache. Exercise the real exchanges with empty
        # partials and keep gathered Q live until A2A scratch has been used.
        gathered = self.dcp_group.all_gather(q.contiguous(), dim=1)
        rows, heads, _ = gathered.shape
        self._merge_dcp(
            q.new_zeros((rows, heads, self.kv_lora_rank), dtype=torch.float32),
            q.new_full((rows, heads), -torch.inf, dtype=torch.float32),
        )
        del gathered

    def _eager_slots(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
        indices: torch.Tensor,
        page: int,
        *,
        decode_verification: bool = False,
    ) -> torch.Tensor:
        main = metadata[self.attn.prefix]
        tokens = main.num_actual_tokens
        if self.indexer is not None:
            self.indexer(
                hidden_states[:tokens],
                q_c[:tokens],
                take_tokens(positions, tokens),
                self.inv_freq,
                cos[:tokens],
                sin[:tokens],
                metadata[self.indexer.k_cache.prefix],
                metadata[self.indexer.tail_cache.prefix],
                indices,
                decode_verification=decode_verification,
            )
        elif self.is_window_layer:
            for begin, end, prefix, length in main.spans:
                indices[begin:end] = window_indices(
                    torch.arange(prefix, length, device=indices.device),
                    self.window,
                )
        # Both selectors return compact, increasing logical slots with -1 padding.
        # Translate in place using the main cache table (not the IndexPool table).
        counts = torch.empty(tokens, dtype=torch.int32, device=indices.device)
        for req, (begin, end, _, _) in enumerate(main.spans):
            rows = indices[begin:end]
            if self.dcp_group is not None:
                # All complete groups and the open group start on rank zero.
                logical = rows[:, self.dcp_rank :: self.dcp_world] // self.dcp_world
                rows = rows[:, : self.width // self.dcp_world]
            else:
                logical = rows
            live = logical >= 0
            counts[begin:end] = live.sum(dim=-1, dtype=torch.int32)
            blocks = main.block_table[req][logical.clamp_min(0) // page]
            rows.copy_((blocks * page + logical % page).masked_fill(~live, -1))

        return counts

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        context = get_forward_context()
        metadata = cast(dict[str, FlashMetadata] | None, context.attn_metadata)
        main = None if metadata is None else metadata[self.attn.prefix]
        tokens = 0 if main is None else main.num_actual_tokens
        prep_args = ()
        prep_writes_indices = False
        if can_use_f7_attn_prep(self, positions, hidden_states):
            cache = self.attn.kv_cache
            if (
                self._attncore_prep
                and main is not None
                and tokens > 0
                and main.decode
                and main.verification is None
                and not self._capture_dfly
                and "hy4_verify_graph" not in context.additional_kwargs
                and self.dcp_world == 1
                and cache.dtype == torch.uint8
                and cache.shape[-1] == F7_FP8_DS_MLA_BYTES
                and cache.stride(-1) == 1
            ):
                prep_writes_indices = (
                    self.is_window_layer or self._bypass_index_history()
                )
                prep_args = (cache, main.slot_mapping[:tokens])
                if prep_writes_indices:
                    assert main.counts is not None
                    prep_args += (
                        main.seq_lens,
                        main.block_table,
                        self.topk_indices_buffer[:tokens, : self.width],
                        main.counts,
                    )
            q, kv_c, k_pe, q_c, cos, sin, w_uv, gate_sig = run_f7_attn_prep(
                self, positions, hidden_states, prep_args=prep_args
            )
        else:
            q, kv_c, k_pe, q_c, cos, sin, w_uv = self._project(positions, hidden_states)
            gate_sig = None

        if tokens:
            assert metadata is not None
            if (
                metadata[self.attn.prefix].verification is not None
                and context.cudagraph_runtime_mode != CUDAGraphMode.NONE
            ):
                raise RuntimeError("Hy4 target verification requires eager dispatch")
            agg_out, fused_out = self._aggregate(
                q,
                kv_c,
                k_pe,
                hidden_states,
                q_c,
                positions,
                cos,
                sin,
                metadata,
                prep_writes_cache=bool(prep_args),
                prep_writes_indices=prep_writes_indices,
                w_uv=w_uv if self._f7_attn_out else None,
                gate_sig=gate_sig,
            )
        else:
            # Profile runs have no bound cache or scheduled slots.
            if metadata is None and q.is_cuda:
                # Reserve decode scratch before sizing the KV cache.
                current_workspace_manager().get_simultaneous(
                    *self._decode_workspace_specs
                )
                if self._attncore_split and self._f7_attn_out and self.dcp_world == 1:
                    current_workspace_manager().get_simultaneous(
                        *self._short_split_specs
                    )
                if self._f7_attn_prep and self.kv_b_proj.weight.is_cuda:
                    ensure_f7_prep_weights(self)
                if self._f7_attn_prep or self._f7_attn_out:
                    get_f7_attn_scratch(q.device)
                if self.dcp_group is not None:
                    self._profile_dcp(q)
                if self.indexer is not None:
                    self.indexer.profile(q)
            agg_out = q.new_zeros((*q.shape[:2], self.kv_lora_rank))
            fused_out = False
        if fused_out:
            if 0 < tokens < q.shape[0]:
                agg_out = torch.nn.functional.pad(
                    agg_out, (0, 0, 0, q.shape[0] - tokens)
                )
            return self._output_projection(agg_out)
        latent = agg_out
        if 0 < tokens < q.shape[0]:
            latent = torch.nn.functional.pad(
                latent, (0, 0, 0, 0, 0, q.shape[0] - tokens)
            )
        return self._project_output(latent, w_uv, hidden_states, gate_sig=gate_sig)
