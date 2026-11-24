# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F7 Stage 1 (HY4_F7_ATTN_PREP) and Stage 2 (HY4_F7_ATTN_OUT) SM70 attention kernels."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

import torch

from vllm import envs
from vllm.forward_context import get_forward_context, is_forward_context_available

from .w8 import can_use_projection_weight


F7_ATTN_MAX_ROWS = 24
F7_ATTN_HIDDEN = 3072
F7_ATTN_Q_RANK = 1024
F7_ATTN_KV_RANK = 512
F7_ATTN_QK_NOPE = 192
F7_ATTN_ROPE = 64
F7_ATTN_V_DIM = 256
F7_ATTN_MAX_HEADS = 16
F7_SWA_TAIL_TILES = 36
F7_SWA_TAIL_SPLITS = 4
F7_TILE_N = 16

_LOCAL_PHASE_CACHE: ContextVar[dict | None] = ContextVar(
    "_hy4_f7_local_phase_cache", default=None
)
_SCRATCH_BY_DEVICE: dict[int, dict[str, torch.Tensor]] = {}


def load_hy4_f7_attn_operator(name: str) -> None:
    if hasattr(torch.ops.hy4_f7_attn, name):
        return
    library = next(
        Path(__file__).resolve().parents[3].glob("_sm70_hy4_attn_C*.so"), None
    )
    if library is None:
        raise RuntimeError(
            f"{name} requires the bundled vllm._sm70_hy4_attn_C extension. "
            "Build vLLM for CUDA SM70."
        )
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.hy4_f7_attn, name):
        raise RuntimeError(
            f"The SM70 Hy4 F7 attention extension did not register {name}."
        )


@contextmanager
def f7_rope_phase_scope() -> Iterator[dict]:
    """Scope a per-forward K1 rotary phase cache outside ForwardContext (e.g. DFly)."""
    cache: dict = {}
    token = _LOCAL_PHASE_CACHE.set(cache)
    try:
        yield cache
    finally:
        _LOCAL_PHASE_CACHE.reset(token)


def _get_phase_cache() -> dict | None:
    local = _LOCAL_PHASE_CACHE.get()
    if local is not None:
        return local
    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    cache = ctx.additional_kwargs.get("_hy4_f7_rope_phases")
    if cache is None:
        cache = {}
        ctx.additional_kwargs["_hy4_f7_rope_phases"] = cache
    return cache


def get_f7_attn_scratch(device: torch.device) -> dict[str, torch.Tensor]:
    """Return address-stable per-device scratch buffers shared across all 42 layers."""
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _SCRATCH_BY_DEVICE.get(dev_idx)
    if bufs is not None:
        return bufs
    bufs = {
        "q_a_raw": torch.empty(
            (F7_ATTN_MAX_ROWS, F7_ATTN_Q_RANK), dtype=torch.float16, device=device
        ),
        "kv_a_raw": torch.empty(
            (F7_ATTN_MAX_ROWS, F7_ATTN_KV_RANK + F7_ATTN_ROPE),
            dtype=torch.float16,
            device=device,
        ),
        "q_nope": torch.empty(
            (F7_ATTN_MAX_HEADS, F7_ATTN_MAX_ROWS, F7_ATTN_QK_NOPE),
            dtype=torch.float16,
            device=device,
        ),
        "tail_partials": torch.empty(
            (F7_SWA_TAIL_TILES, F7_SWA_TAIL_SPLITS, F7_ATTN_MAX_ROWS, F7_TILE_N),
            dtype=torch.float32,
            device=device,
        ),
        "o_latent": torch.empty(
            (32, F7_ATTN_MAX_ROWS, F7_ATTN_KV_RANK),
            dtype=torch.float16,
            device=device,
        ),
    }
    _SCRATCH_BY_DEVICE[dev_idx] = bufs
    return bufs


def can_use_f7_attn_prep(
    module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> bool:
    """Static shape/dtype/device admission check for Stage 1 (K1 + K2).

    Explicitly falls back to today's path for:
    - 3-coordinate image prompts (positions.ndim != 1)
    - bulk prefill / batches above 24 rows (M > 24)
    - CPU or non-FP16 activations
    - non-Hy4-flash head/rank dimensions
    """
    if not getattr(module, "_f7_attn_prep", False):
        return False
    if not (
        hidden_states.is_cuda
        and positions.is_cuda
        and hidden_states.dtype == torch.float16
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == F7_ATTN_HIDDEN
        and 1 <= hidden_states.shape[0] <= F7_ATTN_MAX_ROWS
        and positions.ndim == 1
        and positions.shape[0] == hidden_states.shape[0]
        and positions.dtype in (torch.int64, torch.int32)
    ):
        return False
    heads = module.num_local_heads
    if not (
        heads in (8, 16)
        and module.qk_nope_head_dim == F7_ATTN_QK_NOPE
        and module.qk_rope_head_dim == F7_ATTN_ROPE
        and module.kv_lora_rank == F7_ATTN_KV_RANK
        and module.v_head_dim == F7_ATTN_V_DIM
        and module.inv_freq.is_cuda
        and module.inv_freq.dtype == torch.float32
        and module.inv_freq.shape == (F7_ATTN_ROPE // 2,)
        and module.q_a_layernorm.weight.is_cuda
        and module.q_a_layernorm.weight.dtype == torch.float16
        and module.kv_a_layernorm.weight.is_cuda
        and module.kv_a_layernorm.weight.dtype == torch.float16
        and module.q_a_layernorm.variance_epsilon == module.kv_a_layernorm.variance_epsilon
    ):
        return False
    return bool(
        can_use_projection_weight(module.q_a_proj, F7_ATTN_Q_RANK, F7_ATTN_HIDDEN)
        and can_use_projection_weight(
            module.kv_a_proj_with_mqa,
            F7_ATTN_KV_RANK + F7_ATTN_ROPE,
            F7_ATTN_HIDDEN,
        )
        and can_use_projection_weight(
            module.linear_gate, heads * F7_ATTN_V_DIM, F7_ATTN_HIDDEN
        )
        and can_use_projection_weight(
            module.q_b_proj, heads * (F7_ATTN_QK_NOPE + F7_ATTN_ROPE), F7_ATTN_Q_RANK
        )
        and (
            getattr(module.kv_b_proj, "_hy4_w8_uk", None) is not None
            or can_use_projection_weight(
                module.kv_b_proj,
                heads * (F7_ATTN_QK_NOPE + F7_ATTN_V_DIM),
                F7_ATTN_KV_RANK,
            )
        )
    )


def _tensor_version(t: torch.Tensor) -> int | None:
    return None if t.is_inference() else t._version


def ensure_f7_prep_weights(module) -> torch.Tensor:
    """Prepare contiguous transposed W_uk_T [H, 512, 192] lazily from kv_b_proj.weight."""
    prepared = getattr(module.kv_b_proj, "_hy4_w8_uk", None)
    if prepared is not None:
        return prepared
    weight = module.kv_b_proj.weight
    sig = (weight.data_ptr(), _tensor_version(weight), str(weight.device))
    cached = getattr(module, "_f7_w_uk_t", None)
    if cached is not None and getattr(module, "_f7_w_uk_t_sig", None) == sig:
        return cached
    kv_b = weight.view(module.num_local_heads, -1, module.kv_lora_rank)
    w_uk = kv_b[:, : module.qk_nope_head_dim, :]
    w_uk_t = w_uk.transpose(1, 2).contiguous()
    module._f7_w_uk_t = w_uk_t
    module._f7_w_uk_t_sig = sig
    return w_uk_t


def compute_f7_rope_phases(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K1: compute (or reuse within the current forward) FP32 cos/sin phases."""
    load_hy4_f7_attn_operator("rope_phases_sm70")
    pos_c = positions if positions.is_contiguous() else positions.contiguous()
    cache = _get_phase_cache()
    key = (id(positions), _tensor_version(positions), rope_theta, str(pos_c.device))
    if cache is not None and key in cache:
        return cache[key]
    inv_c = inv_freq if inv_freq.is_contiguous() else inv_freq.contiguous()
    cos_out = torch.empty(
        (pos_c.shape[0], inv_c.shape[0]), dtype=torch.float32, device=pos_c.device
    )
    sin_out = torch.empty_like(cos_out)
    torch.ops.hy4_f7_attn.rope_phases_sm70(pos_c, inv_c, cos_out, sin_out)
    if cache is not None:
        cache[key] = (cos_out, sin_out)
    return cos_out, sin_out


def run_f7_attn_prep(
    module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    prep_args: tuple[torch.Tensor, ...] = (),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Execute Stage 1 (K1 + K2) and return (q, kv_c, k_pe, q_c, cos, sin, w_uv, gate_sig)."""
    load_hy4_f7_attn_operator("attn_qkv_prep_sm70")
    x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    m = x_c.shape[0]
    heads = module.num_local_heads
    dev = x_c.device

    cos, sin = compute_f7_rope_phases(positions, module.inv_freq, module._rope_theta)
    w_uk_t = ensure_f7_prep_weights(module)
    scratch = get_f7_attn_scratch(dev)

    q_out = torch.empty(
        (m, heads, F7_ATTN_KV_RANK + F7_ATTN_ROPE), dtype=torch.float16, device=dev
    )
    kv_c_out = torch.empty((m, F7_ATTN_KV_RANK), dtype=torch.float16, device=dev)
    k_pe_out = torch.empty((m, F7_ATTN_ROPE), dtype=torch.float16, device=dev)
    q_c_out = torch.empty((m, F7_ATTN_Q_RANK), dtype=torch.float16, device=dev)
    gate_sig_out = torch.empty(
        (m, heads * F7_ATTN_V_DIM), dtype=torch.float16, device=dev
    )

    options = {"split_w8": True} if module._prep_split_w8 else {}
    torch.ops.hy4_f7_attn.attn_qkv_prep_sm70(
        x_c,
        None,
        cos,
        sin,
        module.q_a_proj.weight,
        module.kv_a_proj_with_mqa.weight,
        module.linear_gate.weight,
        module.q_a_layernorm.weight,
        module.kv_a_layernorm.weight,
        module.q_b_proj.weight,
        w_uk_t,
        None,
        scratch["q_a_raw"],
        scratch["kv_a_raw"],
        scratch["q_nope"],
        scratch["tail_partials"],
        q_out,
        kv_c_out,
        k_pe_out,
        q_c_out,
        gate_sig_out,
        float(module.q_a_layernorm.variance_epsilon),
        *prep_args,
        **options,
    )

    w_uv = getattr(module.kv_b_proj, "_hy4_w8_uv", None)
    if w_uv is None:
        kv_b = module.kv_b_proj.weight.view(heads, -1, F7_ATTN_KV_RANK)
        w_uv = kv_b[:, F7_ATTN_QK_NOPE:, :]
    return q_out, kv_c_out, k_pe_out, q_c_out, cos, sin, w_uv, gate_sig_out


F7_ATTN_OUT_MAX_SPLITS = 32
F7_ATTN_SHORT_SPLITS = 64
F7_FP8_DS_MLA_BYTES = 656


def can_use_f7_attn_out(
    module,
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sinks: torch.Tensor | None,
    lse: torch.Tensor | None,
    w_uv: torch.Tensor,
    *,
    draft_kv: torch.Tensor | None = None,
) -> bool:
    """Static admission check for Stage 2 (K4 + K5).

    Explicitly falls back to today's path for:
    - four cache shards / DCP (lse is not None or sinks is None)
    - bulk prefill / batches above 24 rows (M > 24)
    - CPU or non-FP16 Q
    """
    if not getattr(module, "_f7_attn_out", False):
        return False
    if lse is not None or sinks is None:
        return False
    if not (
        q.is_cuda
        and cache.is_cuda
        and indices.is_cuda
        and counts.is_cuda
        and sinks.is_cuda
        and w_uv.is_cuda
        and q.dtype == torch.float16
        and q.ndim == 3
        and q.shape[2] == F7_ATTN_KV_RANK + F7_ATTN_ROPE
        and q.stride(2) == 1
        and 1 <= q.shape[0] <= F7_ATTN_MAX_ROWS
        and 1 <= q.shape[1] <= 32
        and indices.dtype == torch.int32
        and counts.dtype == torch.int32
        and sinks.dtype == torch.float32
        and sinks.shape == (q.shape[1],)
        and (
            (
                w_uv.dtype == torch.float16
                and w_uv.shape == (q.shape[1], F7_ATTN_V_DIM, F7_ATTN_KV_RANK)
                and w_uv.stride(2) == 1
                and w_uv.stride(1) == F7_ATTN_KV_RANK
            )
            or (
                w_uv.dtype == torch.uint8
                and w_uv.shape == (q.shape[1], F7_ATTN_V_DIM, F7_ATTN_KV_RANK + 8)
                and w_uv.is_contiguous()
            )
        )
    ):
        return False
    if cache.dtype == torch.uint8:
        if cache.shape[2] != F7_FP8_DS_MLA_BYTES:
            return False
    elif cache.dtype == torch.float16:
        if cache.shape[2] != F7_ATTN_KV_RANK + F7_ATTN_ROPE:
            return False
    else:
        return False
    if draft_kv is not None:
        if not (
            draft_kv.is_cuda
            and draft_kv.dtype == cache.dtype
            and draft_kv.is_contiguous()
            and draft_kv.ndim == 3
            and draft_kv.shape[1] > 0
            and q.shape[0] % draft_kv.shape[1] == 0
        ):
            return False
    return True


def run_f7_attn_out(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    acc: torch.Tensor,
    w_uv: torch.Tensor,
    gate_sig: torch.Tensor,
    *,
    draft_kv: torch.Tensor | None = None,
    short_splits: bool = False,
    legacy_max_splits: int | None = None,
    prmt_fp8: bool | None = None,
    swa_tensor: bool = False,
    narrow_w8: bool | None = None,
) -> torch.Tensor:
    """Execute Stage 2 (K4 split-K reader + K5 cooperative merge/W_uv/gate)."""
    load_hy4_f7_attn_operator("attn_splitk_sm70")
    load_hy4_f7_attn_operator("attn_merge_v_gate_sm70")
    ws_splits = int(acc.shape[-2])
    max_splits = min(ws_splits, F7_ATTN_OUT_MAX_SPLITS)
    if short_splits:
        assert legacy_max_splits is not None
        max_splits = min(max_splits, legacy_max_splits)
    short_options = {"short_splits": True} if short_splits else {}
    query_width = int(draft_kv.shape[1]) if draft_kv is not None else 0
    indices_c = indices if indices.stride(-1) == 1 else indices.contiguous()
    counts_c = counts if counts.is_contiguous() else counts.contiguous()
    sinks_c = sinks if sinks.is_contiguous() else sinks.contiguous()
    gate_c = gate_sig if gate_sig.is_contiguous() else gate_sig.contiguous()

    split_options = dict(short_options)
    if prmt_fp8 is None:
        prmt_fp8 = envs.HY4_ATTN3_FP8_PRMT
    if w_uv.dtype == torch.uint8 and prmt_fp8:
        split_options["prmt_fp8"] = True
        if swa_tensor:
            split_options["swa_tensor"] = True
    torch.ops.hy4_f7_attn.attn_splitk_sm70(
        q,
        cache,
        draft_kv,
        indices_c,
        counts_c,
        state,
        acc,
        float(scale),
        max_splits,
        query_width,
        **split_options,
    )
    scratch = get_f7_attn_scratch(q.device)
    out = torch.empty(
        (q.shape[0], q.shape[1] * F7_ATTN_V_DIM),
        dtype=torch.float16,
        device=q.device,
    )
    if narrow_w8 is None:
        narrow_w8 = envs.HY4_ATTN3_W8_MERGE
    if w_uv.dtype == torch.uint8 and narrow_w8:
        short_options["narrow_w8"] = True
    torch.ops.hy4_f7_attn.attn_merge_v_gate_sm70(
        state,
        acc,
        counts_c,
        sinks_c,
        w_uv,
        gate_c,
        scratch["o_latent"],
        out,
        max_splits,
        **short_options,
    )
    return out
