# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4-flash TP4 native-g32 AWQ grouped-union MoE admission and loader."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

HY4_UNION_PLAN_INTS = 2056
HY4_UNION_MAX_TOKENS = 32
HY4_UNION_FOLD_MAX_TOKENS = 4
HY4_EXPERTS4B_MAX_TOKENS = 8


def load_hy4_sm70_operator(name: str) -> None:
    if hasattr(torch.ops.hy4_sm70, name):
        return
    library = next(
        Path(__file__).resolve().parents[3].glob("_sm70_hy4_moe_C*.so"), None
    )
    if library is None:
        raise RuntimeError(
            f"{name} requires the bundled vllm._sm70_hy4_moe_C extension. "
            "Build vLLM for CUDA SM70."
        )
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.hy4_sm70, name):
        raise RuntimeError(f"The SM70 Hy4 MoE extension did not register {name}.")


def hy4_union_layer_contract(layer, group_size: int) -> bool:
    return bool(
        int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_num_experts) == 512
        and int(group_size) == 32
        and int(layer.sm70_awq_checkpoint_group_size) == 32
        and int(layer.sm70_hidden_logical_size) == 1536
        and int(layer.sm70_w13_k_dim) == 1536
        and int(layer.sm70_w13_n_dim) == 768
        and int(layer.sm70_w2_k_dim) == 384
        and int(layer.sm70_w2_n_dim) == 1536
        and not bool(getattr(layer, "sm70_awq_moe_w13_interleaved", False))
    )


def initialize_hy4_union(layer, group_size: int) -> bool:
    if getattr(layer, "hy4_expert_kernel", envs.VLLM_HY4_EXPERT_KERNEL) != "union":
        return False
    if not hy4_union_layer_contract(layer, group_size):
        return False
    load_hy4_sm70_operator("awq_union_moe_out")
    layer.sm70_awq_hy4_union_min_rows = int(
        getattr(layer, "hy4_expert_union_min_rows", envs.VLLM_HY4_EXPERT_UNION_MIN_ROWS)
    )
    logger.info_once(
        "Hy4-flash SM70 AWQ grouped-union expert kernel enabled for M=%d..%d.",
        layer.sm70_awq_hy4_union_min_rows,
        HY4_UNION_MAX_TOKENS,
    )
    return True


def use_hy4_union(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    if not getattr(layer, "sm70_awq_hy4_union", False):
        return False
    num_tokens = x.shape[0]
    return bool(
        layer.sm70_awq_hy4_union_min_rows <= num_tokens <= HY4_UNION_MAX_TOKENS
        and x.ndim == 2
        and x.shape[1] == 1536
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.ndim == 2
        and topk_ids.shape == (num_tokens, 16)
        and topk_weights.ndim == 2
        and topk_weights.shape == (num_tokens, 16)
        and topk_weights.dtype == torch.float32
        and topk_weights.is_contiguous()
    )


def hy4_union_ring_fold(
    layer, num_tokens: int, max_tokens: int = HY4_UNION_FOLD_MAX_TOKENS
) -> bool:
    """Whether run_hy4_union takes the EXPERTS3 ring fold for this many rows.

    EXPERTS4B passes max_tokens=8: the ring settings, not the fold's row cap.
    """
    return bool(
        getattr(layer, "hy4_expert_union_fold", envs.VLLM_HY4_EXPERT_UNION_FOLD)
        and getattr(layer, "hy4_experts3_ring", False)
        and num_tokens <= max_tokens
        and int(os.getenv("VLLM_HY4_EXPERT_UNION_SPLIT", "0")) in (0, 4)
        and int(os.getenv("VLLM_HY4_EXPERT_UNION_W2_MODE", "2")) == 2
        and int(os.getenv("VLLM_HY4_EXPERT_UNION_PACK_ROWS", "0")) == 0
    )


def run_hy4_experts4(
    layer,
    output: torch.Tensor,
    intermediate: torch.Tensor,
    route_out: torch.Tensor,
    plan: torch.Tensor,
    x: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_x: torch.Tensor,
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
) -> torch.Tensor:
    """EXPERTS3 ring fold plus the W8 shared expert in the same two launches.

    Writes the routed result to output and returns the shared expert output.
    """
    limit = getattr(layer, "swiglu_limit", None)
    return torch.ops.hy4_sm70.experts4_ring_shared_out(
        output,
        intermediate,
        route_out,
        plan,
        x,
        layer.w13_tm_weight,
        layer.w13_tm_scales,
        layer.w2_tm_weight,
        layer.w2_tm_scales,
        topk_ids_i32,
        topk_weights,
        float(limit) if limit is not None else 0.0,
        shared_x,
        shared_gate_up,
        shared_down,
        layer._awq_moe_buf_hy4_e4_tickets,
    )


def run_hy4_union(
    layer,
    output: torch.Tensor,
    intermediate: torch.Tensor,
    route_out: torch.Tensor,
    plan: torch.Tensor,
    x: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    limit = getattr(layer, "swiglu_limit", None)
    limit_f = float(limit) if limit is not None else 0.0
    split = int(os.getenv("VLLM_HY4_EXPERT_UNION_SPLIT", "0"))
    w2_mode = int(os.getenv("VLLM_HY4_EXPERT_UNION_W2_MODE", "2"))
    pack_rows = int(os.getenv("VLLM_HY4_EXPERT_UNION_PACK_ROWS", "0"))
    if (
        getattr(layer, "hy4_expert_union_fold", envs.VLLM_HY4_EXPERT_UNION_FOLD)
        and x.shape[0] <= HY4_UNION_FOLD_MAX_TOKENS
        and split in (0, 4)
        and w2_mode == 2
        and pack_rows == 0
    ):
        # Same output bits as the default schedule below, without the plan and
        # weighted-reduce launches. The EXPERTS3 ring kernels keep the bits.
        fold = (
            torch.ops.hy4_sm70.experts3_ring_fold_out
            if getattr(layer, "hy4_experts3_ring", False)
            else torch.ops.hy4_sm70.awq_union_moe_fold_out
        )
        fold(
            output,
            intermediate,
            route_out,
            plan,
            x,
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            topk_ids_i32,
            topk_weights,
            limit_f,
        )
        return
    torch.ops.hy4_sm70.awq_union_moe_out(
        output,
        intermediate,
        route_out,
        plan,
        x,
        layer.w13_tm_weight,
        layer.w13_tm_scales,
        layer.w2_tm_weight,
        layer.w2_tm_scales,
        topk_ids_i32,
        topk_weights,
        limit_f,
        split,
        w2_mode,
        pack_rows,
    )
