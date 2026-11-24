# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).
# Interleaved MRoPE from transformers/models/hy_v4_flash/modeling_hy_v4_flash.py at b865850d05326cf20e306b31f14b0027de802da3.

import torch

from vllm.triton_utils import tl, triton


def take_tokens(positions: torch.Tensor, count: int) -> torch.Tensor:
    """Slice the sequence axis. A [3, S] tensor must not be cut on axis 0."""
    if positions.ndim == 2 and positions.shape[0] == 3:
        return positions[:, :count]
    return positions[:count]


def span_tokens(positions: torch.Tensor, begin: int, end: int) -> torch.Tensor:
    """Slice a token span on the sequence axis."""
    if positions.ndim == 2 and positions.shape[0] == 3:
        return positions[:, begin:end]
    return positions[begin:end]


def rotary_phases(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, int, int] | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim == 1:
        angles = positions.float()[:, None] * inv_freq
    elif positions.ndim == 2 and positions.shape[0] == 3:
        if mrope_section is None:
            raise ValueError("3-axis Hy4 positions need mrope_section")
        freq = inv_freq.to(dtype=torch.float32, device=positions.device)
        planes = positions.to(dtype=torch.float32)[:, :, None] * freq
        angles = planes[0].clone()
        half = angles.shape[-1]
        for axis, offset in ((1, 1), (2, 2)):
            length = min(int(mrope_section[axis]) * 3, half)
            if offset < length:
                angles[..., offset:length:3] = planes[axis, ..., offset:length:3]
    else:
        raise ValueError(
            f"positions shape {tuple(positions.shape)} is not [S] or [3, S]"
        )
    return angles.cos(), angles.sin()


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved even/odd rotation for one position axis or for MRoPE."""
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return (
        torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        .flatten(-2)
        .to(x.dtype)
    )


@triton.jit
def _rotate_and_join(
    QPe,
    KPe,
    Latent,
    Cos,
    Sin,
    Q,
    K,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_ROW: tl.constexpr,
    L_ROW: tl.constexpr,
    L_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, ROPE)
    pair = r // 2
    if head == HEADS:
        src = KPe + row * K_ROW
        dst = K + row * ROPE
    else:
        src = QPe + row * Q_ROW + head * Q_HEAD
        dst_q = Q + (row * HEADS + head) * (RANK + ROPE)
        d = tl.arange(0, BLOCK)
        latent = tl.load(
            Latent + row * L_ROW + head * L_HEAD + d, d < RANK, other=0
        )
        tl.store(dst_q + d, latent, d < RANK)
        dst = dst_q + RANK

    cos = tl.load(Cos + row * (ROPE // 2) + pair)
    sin = tl.load(Sin + row * (ROPE // 2) + pair)
    even = tl.load(src + 2 * pair).to(tl.float32)
    odd = tl.load(src + 2 * pair + 1).to(tl.float32)
    rot_even = even * cos - odd * sin
    rot_odd = odd * cos + even * sin
    tl.store(dst + r, tl.where(r % 2 == 0, rot_even, rot_odd))


def rotate_and_join(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    latent: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, heads, rank = latent.shape
    rope = k_pe.shape[-1]
    q = latent.new_empty((rows, heads, rank + rope))
    k = k_pe.new_empty((rows, rope))
    # Match the original FP32 intermediates and single final FP16 cast.
    _rotate_and_join[(rows, heads + 1)](
        q_pe, k_pe, latent, cos, sin, q, k,
        q_pe.stride(0), q_pe.stride(1), k_pe.stride(0),
        latent.stride(0), latent.stride(1), heads, rank, rope,
        triton.next_power_of_2(rank), num_warps=4, enable_fp_fusion=False,
    )
    return q, k
