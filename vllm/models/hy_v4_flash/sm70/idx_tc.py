# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM70 Tensor Core IndexPool decode scoring and history selection."""

from __future__ import annotations

from pathlib import Path

import torch

IDX_TC_HEADS = 16
IDX_TC_DIM = 128
IDX_TC_MAX_WIDTH = 8


def load_idx_tc() -> None:
    if hasattr(torch.ops, "hy4_idx_tc") and hasattr(torch.ops.hy4_idx_tc, "scores"):
        return
    library = next(Path(__file__).resolve().parents[3].glob("_sm70_idx_tc_C*.so"), None)
    if library is None:
        raise RuntimeError(
            "HY4_IDX_TC requires the bundled vllm._sm70_idx_tc_C extension. "
            "Build vLLM for CUDA SM70."
        )
    torch.ops.load_library(str(library))


def can_use_idx_tc(
    query: torch.Tensor, weights: torch.Tensor, pool: torch.Tensor, query_width: int
) -> bool:
    return (
        query.is_cuda
        and query.dtype == torch.float16
        and query.ndim == 3
        and query.shape[1:] == (IDX_TC_HEADS, IDX_TC_DIM)
        and query.stride(2) == 1
        and query.stride(1) % 8 == 0
        and query.stride(0) % 8 == 0
        and query.data_ptr() % 16 == 0
        and weights.dtype == torch.float32
        and weights.is_contiguous()
        and weights.shape == (query.shape[0], IDX_TC_HEADS)
        and pool.dtype == torch.float16
        and pool.ndim == 3
        and pool.shape[2] == IDX_TC_DIM
        and pool.stride(2) == 1
        and pool.stride(1) % 8 == 0
        and pool.stride(0) % 8 == 0
        and pool.data_ptr() % 16 == 0
        and 1 <= query_width <= IDX_TC_MAX_WIDTH
        and query.shape[0] % query_width == 0
        and torch.cuda.get_device_capability(query.device) == (7, 0)
    )


def run_idx_tc(
    query: torch.Tensor,
    weights: torch.Tensor,
    pool: torch.Tensor,
    table: torch.Tensor,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    slots: torch.Tensor,
    sorted_groups: torch.Tensor,
    *,
    ratio: int,
    group_topk: int,
    pages_per_block: int,
    query_width: int,
) -> torch.Tensor:
    """Score the live groups and return them as ascending top-history IDs."""
    load_idx_tc()
    ops = torch.ops.hy4_idx_tc
    # The selection reads only each row's live prefix, so skip the -inf fill.
    ops.scores(
        query, weights, pool, table, logits, lengths, slots,
        ratio, group_topk, pages_per_block, 1, 0, query_width, False,
    )
    ops.select(logits, lengths, slots, sorted_groups, ratio, group_topk, 1, 0)
    return sorted_groups
