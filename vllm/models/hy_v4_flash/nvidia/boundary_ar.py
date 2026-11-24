# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    load_hy4_sm70_operator,
)
from vllm.models.hy_v4_flash.profile import speed_setting
from vllm.platforms import current_platform

logger = init_logger(__name__)


def small_rows(exact: bool = False) -> int:
    """Largest batch on the small-row boundary paths; 8 covers the DFly verify.

    HY4_DREV2_WIDE_NORM widens only the exact paths (all-reduce RMSNorm, bridge
    RMSNorm), whose 5-8 row results equal the unfused path bit for bit.
    Only captured batches widen; eager prefill chunks keep the stock paths.
    """
    wide = speed_setting("HY4_DREV2_WIDE") or (
        exact and speed_setting("HY4_DREV2_WIDE_NORM")
    )
    return 8 if wide and torch.cuda.is_current_stream_capturing() else 4


def can_fuse_reduce_norm(x: torch.Tensor, norm: RMSNorm | None) -> bool:
    return (
        norm is not None
        and not envs.VLLM_BATCH_INVARIANT
        and x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 2
        and 1 <= x.shape[0]
        and (x.shape[0] <= 4 or x.shape[0] <= small_rows(exact=True))
        and norm.weight.shape == (3072,)
        and norm.weight.dtype == x.dtype
        and norm.weight.is_contiguous()
        and norm.weight.storage_offset() % 8 == 0
        and current_platform.is_device_capability(70)
        and torch.cuda.is_current_stream_capturing()
        and get_tp_group().world_size == 4
    )


def reduce_norm(
    x: torch.Tensor, norm: RMSNorm, other: torch.Tensor | None = None
) -> torch.Tensor:
    comm = get_tp_group().device_communicator
    ca = getattr(comm, "ca_comm", None)
    if (
        ca is not None
        and comm.use_custom_allreduce
        and not ca.disabled
        and ca.fully_connected
        and ca.sm70_tp4_push_buffer_ptrs is not None
        and ca._policy.native.raw("custom_allreduce_algo") is None
        and ca._policy.native.raw("sm70_tp4_push_allreduce_small_messages") == "1"
        and not comm.ring_comm.status["enabled"]
        and x.is_contiguous()
        and x.storage_offset() % 8 == 0
        and x.shape[-1] == 3072
        and (
            other is None
            or (other.is_contiguous() and other.storage_offset() % 8 == 0)
        )
    ):
        load_hy4_sm70_operator("boundary_ar_norm")
        logger.info_once(
            "Hy4 SM70 TP4 push RMSNorm active for %d rows, local add=%s",
            x.shape[0],
            other is not None,
        )
        return torch.ops.hy4_sm70.boundary_ar_norm(
            x,
            other,
            norm.weight,
            norm.variance_epsilon,
            ca.sm70_tp4_push_buffer_ptrs,
            ca.rank,
        )

    if other is not None:
        x = x + other
    return norm(tensor_model_parallel_all_reduce(x))
