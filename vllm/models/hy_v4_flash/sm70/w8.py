# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load-time symmetric grouped int8 storage for Hy4-flash dense weights."""

import os

import torch
from torch import nn

from vllm import envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    load_hy4_sm70_operator,
)
from vllm.platforms import current_platform

from ..profile import speed_setting


def enabled(weight_class: str) -> bool:
    # Storage changes at load, so these switches require a fresh worker.
    name = f"HY4_W8_{weight_class}"
    active = (
        speed_setting(name)
        if name in envs.environment_variables
        else os.environ.get(name) == "1"
    )
    return (
        active
        and current_platform.is_cuda()
        and current_platform.is_device_capability(70)
        and not envs.VLLM_BATCH_INVARIANT
    )


def can_use_projection_weight(linear, out_dim: int, in_dim: int) -> bool:
    weight = getattr(linear, "weight", None)
    if (
        isinstance(weight, torch.Tensor) and weight.is_cuda
        and weight.dtype == torch.uint8 and weight.is_contiguous()
    ):
        return getattr(linear, "_hy4_w8_shape", None) == (out_dim, in_dim)
    return bool(
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.dtype == torch.float16
        and weight.shape == (out_dim, in_dim)
        and weight.is_contiguous()
    )


def pack(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pack [..., N, K] into [..., Npad, K + 2 * ceil(K/group_size)]."""
    n, k = weight.shape[-2:]
    n_pad = (n + 7) // 8 * 8
    groups = (k + group_size - 1) // group_size
    rows = weight.float()
    if n_pad != n or groups * group_size != k:
        rows = torch.nn.functional.pad(rows, (0, groups * group_size - k, 0, n_pad - n))
    rows = rows.reshape(*weight.shape[:-2], n_pad, groups, group_size)
    scales = (rows.abs().amax(dim=-1) / 127).half()
    scales = torch.where(scales > 0, scales, torch.full_like(scales, 2**-24))
    biased = ((rows / scales.float().unsqueeze(-1)).round().clamp(-127, 127) + 128)
    q = biased.to(torch.uint8).flatten(-2)[..., :k]
    q = q.reshape(*weight.shape[:-2], n_pad // 8, 8, k // 16, 16)
    q = q.transpose(-3, -2).contiguous().flatten(-4)
    s = scales.reshape(*weight.shape[:-2], n_pad // 8, 8, groups)
    s = s.transpose(-1, -2).contiguous().flatten(-3).view(torch.uint8)
    return torch.cat((q, s), dim=-1).view(*weight.shape[:-2], n_pad, k + 2 * groups)


def unpack(
    weight: torch.Tensor, n: int, k: int, dtype=torch.float16, group_size: int = 128
) -> torch.Tensor:
    """Materialize an ordinary matrix for bulk prefill and alternate layouts."""
    n_pad = weight.shape[-2]
    leading = weight.shape[:-2]
    groups = (k + group_size - 1) // group_size
    q, s = weight.flatten(-2).split((n_pad * k, n_pad * groups * 2), dim=-1)
    q = q.reshape(*leading, n_pad // 8, k // 16, 8, 16).transpose(-3, -2)
    q = q.reshape(*leading, n_pad, k)[..., :n, :]
    s = s.view(torch.float16).reshape(*leading, n_pad // 8, groups, 8)
    s = s.transpose(-1, -2).reshape(*leading, n_pad, groups)[..., :n, :]
    scales = s.repeat_interleave(group_size, dim=-1)[..., :k]
    return ((q.half() - 128) * scales).to(dtype)


def project(
    x: torch.Tensor, weight: torch.Tensor, n: int, k: int, group_size: int = 128
) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.float16 and 1 <= x.shape[-2] <= 8:
        x = x.contiguous()
        if x.storage_offset() % 8 != 0:
            x = x.clone()
        if group_size == 128:
            return torch.ops.hy4_sm70.w8_projection(x, weight, n)
        return torch.ops.hy4_sm70.w8_projection_grouped(x, weight, n, group_size)
    return x @ unpack(weight, n, k, x.dtype, group_size).transpose(-1, -2)


def _pack_at_load(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    if weight.is_cuda and weight.dtype == torch.float16:
        if group_size == 128:
            return torch.ops.hy4_sm70.w8_pack_fp16(weight)
        return torch.ops.hy4_sm70.w8_pack_fp16_grouped(weight, group_size)
    return pack(weight, group_size)


def _replace_weight(layer: nn.Module, weight: torch.Tensor) -> None:
    layer.weight = nn.Parameter(weight, requires_grad=False)


class _W8Linear(UnquantizedLinearMethod):
    def __init__(self, group_size: int, packed_cache: str | None):
        self.group_size = group_size
        self.packed_cache = packed_cache

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.dtype == torch.uint8:
            return
        layer._hy4_w8_shape = tuple(layer.weight.shape)
        if self.packed_cache:
            from .w8_cache import cached_linear_weight

            packed = cached_linear_weight(layer, self.packed_cache, self.group_size)
        else:
            packed = _pack_at_load(layer.weight, self.group_size)
        _replace_weight(layer, packed)

    def apply(self, layer, x, bias=None):
        n, k = layer._hy4_w8_shape
        result = project(x.reshape(-1, k), layer.weight, n, k, self.group_size)
        result = result.view(*x.shape[:-1], n)
        return result if bias is None else result + bias


def enable_linear(layer: nn.Module | None, weight_class: str) -> bool:
    if (
        layer is not None
        and enabled(weight_class)
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and layer.weight.dtype in (torch.float16, torch.bfloat16)
        and layer.weight.shape[1] % 16 == 0
    ):
        # Only standalone consumers can change the packed group layout;
        # the fused attention and indexer operators retain their g128 ABI.
        group_size = 128
        if weight_class in ("LATENT", "SHARED", "MLP", "O_PROJ"):
            group_var = f"HY4_W8_{weight_class}_GROUP_SIZE"
            if weight_class == "SHARED":
                group_size = speed_setting(group_var)
            else:
                group_size = int(os.environ.get(group_var, "128"))
            if group_size not in (32, 64, 128):
                raise ValueError(f"{group_var} must be 32, 64, or 128")
        load_hy4_sm70_operator(
            "w8_projection" if group_size == 128 else "w8_projection_grouped"
        )
        packed_cache = (
            os.getenv("HY4_W8_SHARED_GPTQ_CACHE") if weight_class == "SHARED" else None
        )
        if (
            weight_class == "SHARED"
            and envs.VLLM_HY4_PROFILE in ("fast3", "fast4", "fast5", "fast6", "fast7", "fast8")
            and not packed_cache
        ):
            raise ValueError(
                f"The GPTQ {envs.VLLM_HY4_PROFILE} candidate requires "
                "HY4_W8_SHARED_GPTQ_CACHE."
            )
        layer.quant_method = _W8Linear(group_size, packed_cache)
        return True
    return False


class _W8Absorbed(UnquantizedLinearMethod):
    """Pack W_UK in its transposed projection direction and W_UV head-wise."""

    def __init__(self, heads: int, nope: int, value: int, uk: bool, uv: bool):
        self.heads = heads
        self.nope = nope
        self.value = value
        self.uk = uk
        self.uv = uv

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if hasattr(layer, "_hy4_w8_uv"):
            return
        w = layer.weight.view(self.heads, self.nope + self.value, -1)
        uk, uv = w.split((self.nope, self.value), dim=1)
        uk = uk.transpose(1, 2)
        layer.register_buffer(
            "_hy4_w8_uk", _pack_at_load(uk) if self.uk else uk.contiguous(), persistent=False
        )
        layer.register_buffer(
            "_hy4_w8_uv", _pack_at_load(uv) if self.uv else uv.contiguous(), persistent=False
        )
        _replace_weight(layer, w.new_empty(0))


def enable_absorbed(layer: nn.Module, heads: int, nope: int, value: int) -> None:
    uk, uv = enabled("ATTN"), enabled("UV")
    if (
        (uk or uv)
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and layer.weight.dtype in (torch.float16, torch.bfloat16)
        and nope % 16 == 0
        and value % 16 == 0
        and layer.weight.shape[1] % 16 == 0
    ):
        load_hy4_sm70_operator("w8_projection")
        layer.quant_method = _W8Absorbed(heads, nope, value, uk, uv)


def apply_absorbed(
    x: torch.Tensor, weight: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    if weight.dtype == torch.uint8:
        return project(x, weight, n, k)
    return torch.bmm(x, weight.transpose(1, 2))
