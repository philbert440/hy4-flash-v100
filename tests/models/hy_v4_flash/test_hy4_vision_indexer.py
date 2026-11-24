# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""An indexer group that straddles an image keeps all three rotary coordinates."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.hy_v4_flash.sm70.cache import QuerySpan
from vllm.models.hy_v4_flash.sm70.indexpool import FlashCompressedIndexer
from vllm.models.hy_v4_flash.sm70.kernels import pack_fp8_ds_mla_torch
from vllm.models.hy_v4_flash.sm70.rotary import apply_rotary, rotary_phases
from vllm.models.hy_v4_flash.sm70.swa import window_indices
from vllm.models.hy_v4_flash.sm70.vision_rope import mrope_phases

SECTION = (12, 10, 10)
DIM = 64
RATIO = 4


def _angles(position: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Independent T/H/W interleave. Index i is time unless i % 3 selects H or W."""
    planes = position.to(torch.float32)[:, None] * inv_freq.to(torch.float32)
    mixed = planes[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = min(SECTION[axis] * 3, mixed.shape[-1])
        mixed[offset:length:3] = planes[axis, offset:length:3]
    return mixed


def _rotate(vector: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos, sin = angles.cos(), angles.sin()
    even, odd = vector.float()[..., 0::2], vector.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)


def _inv_freq(theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))


def _indexer():
    indexer = FlashCompressedIndexer.__new__(FlashCompressedIndexer)
    nn.Module.__init__(indexer)
    indexer.pos_dims = 3
    indexer.ratio = RATIO
    indexer.dim = DIM
    indexer.rope_dim = DIM
    indexer.mrope_section = SECTION
    indexer.dcp_group = None
    value = nn.Linear(DIM, DIM, bias=False)
    gate = nn.Linear(DIM, DIM, bias=False)
    nn.init.eye_(value.weight)
    nn.init.zeros_(gate.weight)
    indexer.key_value_projection = value
    indexer.key_gate_projection = gate
    indexer.intra_group_ape = nn.Parameter(torch.zeros(RATIO, DIM))
    indexer.key_norm = nn.Identity()
    page = 16
    pool = torch.zeros(1, page, DIM)
    tail = torch.zeros(1, RATIO, DIM + 3)
    indexer.k_cache = SimpleNamespace(
        kv_cache=pool, spec=SimpleNamespace(storage_block_size=page)
    )
    indexer.tail_cache = SimpleNamespace(kv_cache=tail)
    return indexer, pool


def test_mix_agrees_with_mrope_phases():
    inv = _inv_freq(1.0e7)
    position = torch.tensor([5, 7, 9])
    mixed = _angles(position, inv)
    cos, sin = mrope_phases(position[:, None], inv, SECTION)
    assert torch.equal(cos[0], mixed.cos())
    assert torch.equal(sin[0], mixed.sin())
    same = torch.tensor([4, 4, 4])
    assert torch.equal(_angles(same, inv), same[0].float() * inv)


def test_open_group_straddle_matches_three_axis_rotation():
    torch.manual_seed(4)
    indexer, pool = _indexer()
    keys = torch.randn(RATIO, DIM)
    positions = torch.tensor(
        [
            [5, 5, 5, 5],
            [7, 7, 8, 8],
            [9, 10, 10, 11],
        ],
        dtype=torch.long,
    )
    inv = _inv_freq(1.0e7)
    table = torch.zeros(1, dtype=torch.long)
    ring = torch.zeros(1, dtype=torch.long)
    # The group stays open across the image boundary, then closes.
    indexer.prepare_groups(
        keys[:2], positions[:, :2], inv, QuerySpan(0, 2, 0, 2), table, ring
    )
    indexer.prepare_groups(
        keys[2:], positions[:, 2:], inv, QuerySpan(0, 2, 2, RATIO), table, ring
    )
    raw = keys.float().reshape(1, RATIO, DIM)
    values = indexer.key_value_projection(raw)
    gates = indexer.key_gate_projection(raw) + indexer.intra_group_ape
    pooled = indexer.key_norm((values * gates.softmax(dim=1)).sum(dim=1))[0]
    expected = _rotate(pooled, _angles(positions[:, 0], inv))
    time_only = _rotate(pooled, positions[0, 0].float() * inv)
    closed = pool[0, 0]
    assert not torch.allclose(expected, time_only)
    assert torch.allclose(closed, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(closed, time_only, atol=1e-4, rtol=1e-4)


def test_sliding_window_and_mla_do_not_store_a_coordinate():
    slots = torch.arange(8)
    by_slot = window_indices(slots, 4)
    # An image row shares one time coordinate. The window is over slots, so
    # that coordinate must not select the window.
    shared_time = torch.tensor([5, 5, 5, 5, 8, 8, 8, 9])
    assert not torch.equal(window_indices(shared_time, 4), by_slot)
    latent = torch.zeros(1, 512)
    rope = torch.randn(1, DIM, dtype=torch.float16)
    assert pack_fp8_ds_mla_torch(latent, rope).shape[-1] == 656
    inv = _inv_freq(1.0e4)
    text = torch.arange(1)
    stacked = text.view(1, -1).expand(3, -1).contiguous()
    cos, sin = rotary_phases(stacked, inv, SECTION)
    one_cos, one_sin = rotary_phases(text, inv)
    assert torch.equal(cos, one_cos)
    assert torch.equal(sin, one_sin)
    rotated = apply_rotary(rope, one_cos, one_sin)
    assert rotated.shape == rope.shape
    assert pack_fp8_ds_mla_torch(latent, rotated).shape[-1] == 656
