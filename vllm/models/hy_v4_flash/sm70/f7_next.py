# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM70 IndexPool updates and query preparation."""

from __future__ import annotations

from pathlib import Path

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from .w8 import can_use_projection_weight

F7_NEXT_MAX_ROWS = 24
F7_NEXT_HIDDEN = 3072
F7_IDX_DIM = 128
F7_IDX_HEADS = 16
F7_IDX_ROPE = 64
F7_IDX_RATIO = 4
F7_IDX_TK_SPLITS = 16

def load_hy4_f7_next_operator(name: str) -> None:
    if hasattr(torch.ops.hy4_f7_next, name):
        return
    library = next(
        Path(__file__).resolve().parents[3].glob("_sm70_hy4_next_C*.so"), None
    )
    if library is None:
        raise RuntimeError(
            f"{name} requires the bundled vllm._sm70_hy4_next_C extension. "
            "Build vLLM for CUDA SM70."
        )
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.hy4_f7_next, name):
        raise RuntimeError(
            f"The SM70 Hy4 F7 next extension did not register {name}."
        )



def can_use_f7_idx_update(
    indexer,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    width: int = 1,
    staged: torch.Tensor | None = None,
) -> bool:
    """Static shape/dtype/device admission check for Stage 5 K3a."""
    if not getattr(indexer, "_f7_idx_fuse", False):
        return False
    if not (
        hidden.is_cuda
        and positions.is_cuda
        and hidden.dtype == torch.float16
        and torch.cuda.get_device_capability(hidden.device) == (7, 0)
        and hidden.ndim == 2
        and hidden.shape[1] == F7_NEXT_HIDDEN
        and 1 <= hidden.shape[0] <= F7_NEXT_MAX_ROWS
        and 1 <= width <= 6
        and hidden.shape[0] % width == 0
        and indexer.ratio == F7_IDX_RATIO
        and indexer.dim == F7_IDX_DIM
        and indexer.rope_dim == F7_IDX_ROPE
        and indexer.pos_dims in (1, 3)
        and positions.dtype == torch.int64
    ):
        return False
    if indexer.pos_dims == 1:
        if positions.ndim != 1 or positions.shape[0] != hidden.shape[0]:
            return False
    else:
        if (
            positions.ndim != 2
            or positions.shape != (3, hidden.shape[0])
            or indexer.mrope_section is None
        ):
            return False
    if width != 1 and (
        indexer.dcp_world != 1
        or staged is None
        or staged.dtype != torch.float32
        or staged.shape != (hidden.shape[0], F7_IDX_DIM + indexer.pos_dims)
        or not staged.is_contiguous()
    ):
        return False
    projection = indexer.token_key_projection
    if not isinstance(projection.quant_method, UnquantizedLinearMethod):
        return False
    if not can_use_projection_weight(projection, F7_IDX_DIM, F7_NEXT_HIDDEN):
        return False
    for w in (
        indexer.key_value_projection.weight,
        indexer.key_gate_projection.weight,
    ):
        if not (w.is_cuda and w.dtype == torch.float32 and w.shape == (F7_IDX_DIM, F7_IDX_DIM) and w.is_contiguous()):
            return False
    if not (
        indexer.intra_group_ape.is_cuda
        and indexer.intra_group_ape.dtype == torch.float32
        and indexer.intra_group_ape.shape == (F7_IDX_RATIO, F7_IDX_DIM)
        and indexer.intra_group_ape.is_contiguous()
    ):
        return False
    pool, ring = indexer.k_cache.kv_cache, indexer.tail_cache.kv_cache
    gamma_tk, gamma_pool = indexer.token_key_norm.weight, indexer.key_norm.weight
    return (
        pool.dtype == torch.float16
        and ring.dtype == torch.float32
        and pool.ndim == ring.ndim == 3
        and pool.shape[-1] == F7_IDX_DIM
        and ring.shape[1:] == (F7_IDX_RATIO, F7_IDX_DIM + indexer.pos_dims)
        and pool.stride(-1) == ring.stride(-1) == 1
        and gamma_tk.dtype == torch.float16
        and gamma_pool.dtype == torch.float32
        and gamma_tk.is_contiguous()
        and gamma_pool.is_contiguous()
    )


def run_f7_idx_update(
    indexer,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    metadata,
    tail_metadata,
    slots: torch.Tensor,
    *,
    staged: torch.Tensor | None = None,
    width: int = 1,
) -> None:
    load_hy4_f7_next_operator("idx_pool_update_sm70")
    hidden_c = hidden if hidden.is_contiguous() else hidden.contiguous()
    pos_c = positions if positions.is_contiguous() else positions.contiguous()
    inv_f = inv_freq if (inv_freq.dtype == torch.float32 and inv_freq.is_contiguous()) else inv_freq.float().contiguous()
    pool = indexer.k_cache.kv_cache
    ring = indexer.tail_cache.kv_cache
    page = int(pool.shape[1])
    pages_per_block = int(indexer.k_cache.spec.storage_block_size // page)
    sec1 = int(indexer.mrope_section[1]) if (indexer.pos_dims == 3 and indexer.mrope_section is not None) else 0
    sec2 = int(indexer.mrope_section[2]) if (indexer.pos_dims == 3 and indexer.mrope_section is not None) else 0
    # The allocator scopes this buffer to the graph/stream that owns it.
    tk_partials = hidden.new_empty(
        (hidden.shape[0], F7_IDX_TK_SPLITS, F7_IDX_DIM), dtype=torch.float32
    )
    torch.ops.hy4_f7_next.idx_pool_update_sm70(
        hidden_c,
        indexer.token_key_projection.weight,
        indexer.token_key_norm.weight,
        indexer.key_value_projection.weight,
        indexer.key_gate_projection.weight,
        indexer.intra_group_ape,
        indexer.key_norm.weight,
        inv_f,
        pos_c,
        metadata.seq_lens,
        slots,
        metadata.block_table,
        tail_metadata.block_table,
        tk_partials,
        staged,
        pool,
        ring,
        int(width),
        page,
        pages_per_block,
        int(indexer.dcp_world),
        int(indexer.dcp_rank),
        sec1,
        sec2,
        float(indexer.eps),
    )


def can_use_f7_idx_query_prep(
    indexer,
    hidden: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if not getattr(indexer, "_f7_idx_fuse", False):
        return False
    if not (
        hidden.is_cuda
        and q_c.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and hidden.dtype == torch.float16
        and torch.cuda.get_device_capability(hidden.device) == (7, 0)
        and q_c.dtype == torch.float16
        and cos.dtype == torch.float32
        and sin.dtype == torch.float32
        and hidden.ndim == 2
        and q_c.ndim == 2
        and 1 <= hidden.shape[0] <= F7_NEXT_MAX_ROWS
        and hidden.shape[1] == F7_NEXT_HIDDEN
        and q_c.shape == (hidden.shape[0], 1024)
        and cos.shape == (hidden.shape[0], F7_IDX_ROPE // 2)
        and sin.shape == (hidden.shape[0], F7_IDX_ROPE // 2)
        and indexer.num_heads == F7_IDX_HEADS
        and indexer.dim == F7_IDX_DIM
        and indexer.rope_dim == F7_IDX_ROPE
    ):
        return False
    if not all(
        isinstance(layer.quant_method, UnquantizedLinearMethod)
        for layer in (indexer.query_projection, indexer.head_weight_projection)
    ):
        return False
    return (
        can_use_projection_weight(
            indexer.query_projection, F7_IDX_HEADS * F7_IDX_DIM, 1024
        )
        and can_use_projection_weight(
            indexer.head_weight_projection, F7_IDX_HEADS, F7_NEXT_HIDDEN
        )
    )


def run_f7_idx_query_prep(
    indexer,
    hidden: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_hy4_f7_next_operator("idx_query_prep_sm70")
    m = hidden.shape[0]
    q_out = torch.empty(
        (m, F7_IDX_HEADS, F7_IDX_DIM), dtype=torch.float16, device=hidden.device
    )
    w_out = torch.empty((m, F7_IDX_HEADS), dtype=torch.float32, device=hidden.device)
    torch.ops.hy4_f7_next.idx_query_prep_sm70(
        q_c if q_c.is_contiguous() else q_c.contiguous(),
        hidden if hidden.is_contiguous() else hidden.contiguous(),
        indexer.query_projection.weight,
        indexer.head_weight_projection.weight,
        cos if cos.is_contiguous() else cos.contiguous(),
        sin if sin.is_contiguous() else sin.contiguous(),
        q_out,
        w_out,
    )
    return q_out, w_out
