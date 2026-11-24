# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in Hy4-flash dense projections on Volta."""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    load_hy4_sm70_operator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.models.hy_v4_flash.profile import speed_setting
from vllm.platforms import current_platform


logger = init_logger(__name__)


def _enabled(flag: bool) -> bool:
    return (
        flag
        and current_platform.is_cuda()
        and current_platform.is_device_capability(70)
        and not envs.VLLM_BATCH_INVARIANT
    )


def fp16_storage_enabled() -> bool:
    return _enabled(speed_setting("HY4_DENSE_FP16_STORAGE"))


def _check_fp16_range(weight: torch.Tensor) -> None:
    lo, hi = torch.aminmax(weight)
    if not (-65504 <= lo.item() and hi.item() <= 65504):
        raise ValueError("Hy4 FP16 storage received a non-finite or out-of-range weight.")


def _can_use_skinny(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4
        and x.is_contiguous()
        and weight.dtype == torch.float16
        and weight.is_contiguous()
        and weight.storage_offset() % 8 == 0
        and weight.shape[1] == x.shape[1]
        and weight.shape[1] % 8 == 0
    )


class FP16StorageGate(GateLinear):
    def __init__(self, *args, **kwargs):
        kwargs["params_dtype"] = torch.float16
        super().__init__(*args, **kwargs)
        load_hy4_sm70_operator("dense_gemv")
        logger.info_once("Hy4 router weights use FP16 storage and FP32 logits.")

    def weight_loader(self, param, loaded_weight):
        _check_fp16_range(loaded_weight)
        return super().weight_loader(param, loaded_weight)

    def forward(self, x):
        if _can_use_skinny(x, self.weight):
            return torch.ops.hy4_sm70.dense_gemv(x.float(), self.weight), None
        # Keep FP32 logits for prefill as well.
        return torch.mm(x.half(), self.weight.T, out_dtype=torch.float32), None


class _FP16HeadMethod(UnquantizedEmbeddingMethod):
    def process_weights_after_loading(self, layer):
        # Use the full FP32 logit path for sampling and logprob requests.
        pass

    def apply(self, layer, x, bias=None):
        x_2d = x.reshape(-1, x.shape[-1]).half()
        logits = torch.mm(x_2d, layer.weight.T, out_dtype=torch.float32)
        logits = logits.reshape(*x.shape[:-1], layer.weight.shape[0])
        return logits if bias is None else logits + bias


class FP16StorageHead(ParallelLMHead):
    def __init__(self, *args, **kwargs):
        kwargs["params_dtype"] = torch.float16
        super().__init__(*args, **kwargs)
        self.quant_method = _FP16HeadMethod()
        logger.info_once(
            "Hy4 target head uses FP16 storage and FP32 logits, %d bytes per rank.",
            self.weight.nbytes,
        )

    def weight_loader(self, param, loaded_weight):
        _check_fp16_range(loaded_weight)
        return super().weight_loader(param, loaded_weight)


def w8_head_enabled() -> bool:
    return _enabled(speed_setting("HY4_W8_HEAD"))


class _W8HeadMethod(_FP16HeadMethod):
    def process_weights_after_loading(self, layer):
        load_hy4_sm70_operator("w8_head_projection")
        layer.packed_head = torch.ops.hy4_sm70.w8_pack_fp16_grouped(
            layer.weight, layer.w8_group_size
        )
        logger.info_once(
            "Hy4 W8 target head ready: group size %d, %d packed bytes per rank.",
            layer.w8_group_size, layer.packed_head.nbytes,
        )

    def apply(self, layer, x, bias=None):
        rows = x.numel() // x.shape[-1]
        if not x.is_cuda or not 1 <= rows <= 8 or layer.packed_head is None:
            return super().apply(layer, x, bias)

        x_2d = x.reshape(rows, x.shape[-1]).half().contiguous()
        if x_2d.storage_offset() % 8:
            x_2d = x_2d.clone()
        logits = torch.ops.hy4_sm70.w8_head_projection(
            x_2d, layer.packed_head, layer.weight.shape[0], layer.w8_group_size
        )
        logits = logits.reshape(*x.shape[:-1], layer.weight.shape[0])
        return logits if bias is None else logits + bias


class W8StorageHead(FP16StorageHead):
    def __init__(self, *args, group_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.w8_group_size = (
            envs.HY4_W8_HEAD_GROUP_SIZE if group_size is None else group_size
        )
        if self.w8_group_size not in (64, 128):
            raise ValueError("HY4_W8_HEAD_GROUP_SIZE must be 64 or 128.")
        self.register_buffer("packed_head", None, persistent=False)
        self.quant_method = _W8HeadMethod()


_SKINNY_PROJECTIONS = frozenset({
    (1536, 3072),  # Latent down.
    (3072, 1536),  # Latent up.
    (3072, 4096),  # SWA o_proj.
    (3072, 2048),  # Sparse o_proj.
    (768, 3072),   # Shared gate/up.
    (3072, 384),   # Shared down.
})


class _SkinnyMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if (
            layer._hy4_dense_skinny_active
            and bias is None
            and x.dtype == torch.float16
            and x.storage_offset() % 8 == 0
            and _can_use_skinny(x, layer.weight)
        ):
            return torch.ops.hy4_sm70.skinny_projection(x, layer.weight)
        return super().apply(layer, x, bias)


def enable_skinny(layer: torch.nn.Module | None) -> None:
    if (
        layer is not None
        and _enabled(speed_setting("HY4_DENSE_SKINNY"))
        and type(layer.quant_method) is UnquantizedLinearMethod
        and tuple(layer.weight.shape) in _SKINNY_PROJECTIONS
    ):
        load_hy4_sm70_operator("skinny_projection")
        layer._hy4_dense_skinny_active = True
        layer.quant_method = _SkinnyMethod()
