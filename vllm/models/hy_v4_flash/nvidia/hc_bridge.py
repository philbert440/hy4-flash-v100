# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    load_hy4_sm70_operator,
)
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer, triton_ihc_supported
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.boundary_ar import small_rows
from vllm.platforms import current_platform


def initialize_bridge(enabled: bool) -> None:
    if (
        enabled
        and current_platform.is_cuda()
        and current_platform.is_device_capability(70)
        and not envs.VLLM_BATCH_INVARIANT
    ):
        load_hy4_sm70_operator("f7_hc_bridge")


def can_bridge(
    x: torch.Tensor, config: HYV4FlashConfig, enabled: bool | None = None
) -> bool:
    if enabled is None:
        enabled = envs.HY4_F7_HC_BRIDGE
    return (
        enabled
        and triton_ihc_supported(x)
        and current_platform.is_device_capability(70)
        and x.dtype == torch.float16
        and 1 <= x.shape[0] <= 24
        and config.hc_mult == 4
        and config.hidden_size == 3072
    )


def finish_post(
    hc: HYV4HCLayer,
    streams: torch.Tensor,
    previous: torch.Tensor | None,
    previous_post: torch.Tensor | None,
) -> torch.Tensor:
    return streams if previous is None else hc.post(previous, streams, previous_post)


def bridge(
    hc: HYV4HCLayer,
    streams: torch.Tensor,
    previous: torch.Tensor | None,
    previous_post: torch.Tensor | None,
    *,
    norm: RMSNorm | None = None,
    split_big_mix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = hc.hc_pre
    if (
        not streams.is_contiguous()
        or any(
            t.dtype != torch.float32 or not t.is_contiguous()
            for t in (pre.hc_fn.weight, pre.hc_scale, pre.hc_base)
        )
        or (previous is not None and not previous.is_contiguous())
    ):
        reduced, post, streams = hc.pre(
            finish_post(hc, streams, previous, previous_post)
        )
        return reduced if norm is None else norm(reduced), post, streams
    args = (
        streams,
        previous,
        previous_post,
        pre.hc_fn.weight,
        pre.hc_scale,
        pre.hc_base,
        pre.magnitude,
        pre.hc_eps,
        pre.layernorm_epsilon,
    )

    if (
        norm is not None
        and (streams.shape[0] <= 4 or streams.shape[0] <= small_rows(exact=True))
        and norm.weight.dtype == streams.dtype
        and norm.weight.device == streams.device
        and norm.weight.is_contiguous()
        and norm.weight.storage_offset() % 8 == 0
        and norm.variance_size_override is None
    ):
        args = (*args, norm.weight, norm.variance_epsilon)
        op = torch.ops.hy4_sm70.nodes_hc_rms
        return op(*args, split_big_mix=True) if split_big_mix else op(*args)
    op = torch.ops.hy4_sm70.f7_hc_bridge
    reduced, post, streams = (
        op(*args, split_big_mix=True) if split_big_mix else op(*args)
    )
    return reduced if norm is None else norm(reduced), post, streams
