# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compile the Hy4 decode kernels for SM70 without a GPU.

The vision server passes three position coordinates and the text server
passes one. Both specializations have to compile. Triton rejects
``tl.arange(0, 3)`` because the length is not a power of two, which is
how the previous kernels failed offline.
"""

import os
from pathlib import Path

import pytest
import torch

from vllm.models.hy_v4_flash.sm70 import decode as decode_mod
from vllm.models.hy_v4_flash.sm70.decode import (
    gather_decode_group,
    gather_verify_group,
    store_decode_group,
)
from vllm.triton_utils import tl, triton

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

_GATHER_TYPES = ["*fp16", "*i32", "*i64", "*i32", "*fp32", "*fp32", "*i64"] + ["i32"] * 3
_STORE_TYPES = [
    "*fp16",
    "*i64",
    "*fp16",
    "*i32",
    "*i64",
    "*i32",
    "*i32",
    "*fp16",
    "*fp32",
] + ["i32"] * 6
_STORE_CONST = dict(RATIO=4, DIM=128, PAGE=64, PAGES_PER_BLOCK=1, WORLD=1, RANK=0)


def _compile(kernel, types, constexprs):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    signature = dict(zip(kernel.arg_names, types, strict=False))
    source = ASTSource(kernel, signature, constexprs=constexprs)
    return compile(source, target=GPUTarget("cuda", 70, 32))


@triton.jit
def _arange_span(Out, NPOS: tl.constexpr):
    coord = tl.arange(0, NPOS)
    tl.store(Out + coord, coord)


@pytest.mark.skipif(INTERPRET, reason="SM70 cubin compile uses the device compiler")
@pytest.mark.parametrize("npos", (1, 3))
def test_decode_kernels_compile_for_sm70(npos):
    source = Path(decode_mod.__file__).read_text()
    assert "tl.arange(0, NPOS)" not in source
    gather = _compile(gather_decode_group, _GATHER_TYPES, dict(RATIO=4, DIM=128, NPOS=npos))
    store = _compile(store_decode_group, _STORE_TYPES, _STORE_CONST | dict(NPOS=npos))
    assert gather.asm["cubin"]
    assert store.asm["cubin"]


@pytest.mark.skipif(INTERPRET, reason="SM70 cubin compile uses the device compiler")
def test_reintroduced_arange_of_three_does_not_compile():
    with pytest.raises(Exception, match="power of 2"):
        _compile(_arange_span, ["*i32"], dict(NPOS=3))


@pytest.mark.skipif(INTERPRET, reason="SM70 cubin compile uses the device compiler")
def test_arange_of_one_coordinate_still_compiles():
    binary = _compile(_arange_span, ["*i32"], dict(NPOS=1))
    assert binary.asm["cubin"]


def _store_then_gather(npos: int) -> None:
    dim = 16
    ratio = 4
    rows = 1
    page = 64
    expected = torch.tensor([77] if npos == 1 else [11, 22, 33], dtype=torch.float32)
    positions = expected.to(torch.int64).view(1, npos) if npos == 3 else expected.to(torch.int64)
    ring = torch.full((1, ratio, dim + npos), -7.0)
    keys = (torch.arange(dim, dtype=torch.float16) + 1).view(1, dim)
    pooled = torch.arange(dim, dtype=torch.float16) + 5
    pool = torch.full((1, page, dim), -3.0, dtype=torch.float16)
    lengths = torch.tensor([1], dtype=torch.int32)
    slots = torch.tensor([0], dtype=torch.int64)
    ring_table = torch.zeros(1, dtype=torch.int32)
    pool_table = torch.zeros(1, 4, dtype=torch.int32)
    store_decode_group[(rows,)](
        keys,
        positions,
        pooled,
        lengths,
        slots,
        pool_table,
        ring_table,
        pool,
        ring,
        pool_table.stride(0),
        ring_table.stride(0),
        pool.stride(0),
        pool.stride(1),
        ring.stride(0),
        ring.stride(1),
        ratio,
        dim,
        page,
        1,
        1,
        0,
        npos,
    )
    assert torch.equal(ring[0, 0, dim : dim + npos], expected)
    assert ring[0, 1, 0].item() == -7.0
    assert torch.equal(ring[0, 0, :dim], keys[0].float())

    if npos == 1:
        group_positions = torch.full((2,), -1, dtype=torch.int64)
    else:
        group_positions = torch.full((2, npos), -1, dtype=torch.int64)
    raw = torch.empty(rows, ratio, dim)
    gather_decode_group[(rows,)](
        keys,
        torch.tensor([ratio], dtype=torch.int32),
        slots,
        ring_table,
        ring,
        raw,
        group_positions,
        ring_table.stride(0),
        ring.stride(0),
        ring.stride(1),
        ratio,
        dim,
        npos,
    )
    assert torch.equal(group_positions[0].view(-1), expected.to(torch.int64))
    assert torch.all(group_positions[1] == -1)

    store_decode_group[(rows,)](
        keys,
        positions,
        pooled,
        torch.tensor([ratio], dtype=torch.int32),
        slots,
        pool_table,
        ring_table,
        pool,
        ring,
        pool_table.stride(0),
        ring_table.stride(0),
        pool.stride(0),
        pool.stride(1),
        ring.stride(0),
        ring.stride(1),
        ratio,
        dim,
        page,
        1,
        1,
        0,
        npos,
    )
    assert torch.equal(pool[0, 0], pooled)
    assert torch.equal(ring[0, 0, dim : dim + npos], expected)
    assert ring[0, 1, 0].item() == -7.0


@pytest.mark.skipif(not INTERPRET, reason="coordinate values run in the Triton interpreter")
@pytest.mark.parametrize("npos", (1, 3))
def test_coordinates_round_trip_without_spilling(npos):
    _store_then_gather(npos)


@pytest.mark.skipif(INTERPRET, reason="SM70 cubin compile uses the device compiler")
@pytest.mark.parametrize("npos", (1, 3))
def test_verify_gather_compiles_for_sm70(npos):
    types = ["*fp32", "*i32", "*i64", "*i32", "*fp32", "*fp32", "*i64"] + ["i32"] * 3
    binary = _compile(
        gather_verify_group, types, dict(WIDTH=6, CLOSES=2, RATIO=4, DIM=128, NPOS=npos)
    )
    assert binary.asm["cubin"]


@pytest.mark.skipif(not INTERPRET, reason="coordinate values run in the Triton interpreter")
def test_verify_gather_reads_three_positions_from_ring_and_staged_tail():
    from vllm.models.hy_v4_flash.sm70.cache import QuerySpan
    from vllm.models.hy_v4_flash.sm70.verification import (
        VerificationForward,
        _PendingTail,
    )

    dim, ratio, npos, width, closes = 16, 4, 3, 6, 2
    span = QuerySpan(0, width, 2, 8)
    pending = _PendingTail(span.length)
    context = VerificationForward({span: pending})
    keys = torch.arange(width * dim, dtype=torch.float32).reshape(width, dim)
    positions = torch.tensor([
        [1, 1, 1, 1, 5, 6],
        [7, 7, 8, 8, 5, 6],
        [8, 9, 7, 8, 5, 6],
    ])
    assert context.stage_tail(span, "layer", keys, positions)
    staged = pending.rows["layer"]
    torch.testing.assert_close(staged[:, dim:], positions.T.float())

    ring = torch.full((1, ratio, dim + npos), -99.0)
    ring[0, 0, dim:] = torch.tensor([1, 7, 6])
    ring[0, 1, dim:] = torch.tensor([1, 7, 7])
    raw = torch.empty((closes, ratio, dim))
    group_positions = torch.empty((closes, npos), dtype=torch.long)
    gather_verify_group[(1, closes)](
        staged, torch.tensor([8], dtype=torch.int32),
        torch.zeros(width, dtype=torch.long), torch.zeros((1, 1), dtype=torch.int32),
        ring, raw, group_positions, 1, ring.stride(0), ring.stride(1),
        width, closes, ratio, dim, npos,
    )
    torch.testing.assert_close(group_positions, torch.tensor([[1, 7, 6], [1, 8, 7]]))
    torch.testing.assert_close(raw.reshape(-1, dim)[2:], keys)
    torch.testing.assert_close(raw[0, :2], ring[0, :2, :dim])
