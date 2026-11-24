# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).

"""Fixed-shape SM70 decode kernels for Hy4 Flash."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def gather_decode_group(
    Keys,
    Lengths,
    Slots,
    RingTable,
    Ring,
    Raw,
    GroupPositions,
    ring_table_stride,
    ring_page_stride,
    ring_row_stride,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    NPOS: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    active = tl.load(Slots + row) >= 0
    close = active & (length % RATIO == 0)
    member = tl.arange(0, RATIO)
    d = tl.arange(0, DIM)
    ring_id = tl.load(RingTable + row * ring_table_stride, close, other=0)
    base = Ring + ring_id.to(tl.int64) * ring_page_stride
    current = tl.load(Keys + row * DIM + d, close, other=0).to(tl.float32)
    saved = tl.load(
        base + member[:, None] * ring_row_stride + d[None, :],
        close & (member[:, None] < RATIO - 1),
        other=0,
    )
    raw = tl.where(member[:, None] == RATIO - 1, current[None, :], saved)
    tl.store(Raw + (row * RATIO + member[:, None]) * DIM + d[None, :], raw)
    # One coordinate still reads the scalar stored at base + DIM.
    # Three coordinates use four lanes: arange rejects any other length,
    # and the unused lane must not touch the next ring row.
    tl.static_assert(NPOS == 1 or NPOS == 3, "Hy4 indexer coordinates are 1 or 3")
    if NPOS == 1:
        coord = tl.arange(0, 1)
        first_position = tl.load(base + DIM + coord, close, other=0).to(tl.int64)
        tl.store(GroupPositions + row * NPOS + coord, first_position)
    else:
        coord = tl.arange(0, 4)
        live = coord < NPOS
        first_position = tl.load(base + DIM + coord, close & live, other=0).to(tl.int64)
        tl.store(GroupPositions + row * NPOS + coord, first_position, live)


@triton.jit
def store_decode_group(
    Keys,
    Positions,
    Pooled,
    Lengths,
    Slots,
    PoolTable,
    RingTable,
    Pool,
    Ring,
    pool_table_stride,
    ring_table_stride,
    pool_page_stride,
    pool_row_stride,
    ring_page_stride,
    ring_row_stride,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    NPOS: tl.constexpr = 1,
):
    row = tl.program_id(0)
    if tl.load(Slots + row) >= 0:
        logical = tl.load(Lengths + row) - 1
        member = logical % RATIO
        d = tl.arange(0, DIM)
        group = logical // RATIO
        if (member == RATIO - 1) & (group % WORLD == RANK):
            group = group // WORLD
            page = group // PAGE
            block = tl.load(
                PoolTable + row * pool_table_stride + page // PAGES_PER_BLOCK
            )
            physical = block * PAGES_PER_BLOCK + page % PAGES_PER_BLOCK
            pool_dst = Pool + physical.to(tl.int64) * pool_page_stride
            pool_dst += (group % PAGE) * pool_row_stride
            tl.store(pool_dst + d, tl.load(Pooled + row * DIM + d))
        elif member != RATIO - 1:
            ring_id = tl.load(RingTable + row * ring_table_stride)
            ring_dst = Ring + ring_id.to(tl.int64) * ring_page_stride
            ring_dst += member * ring_row_stride
            tl.store(ring_dst + d, tl.load(Keys + row * DIM + d).to(tl.float32))
            tl.static_assert(NPOS == 1 or NPOS == 3, "Hy4 indexer coordinates are 1 or 3")
            if NPOS == 1:
                coord = tl.arange(0, 1)
                tl.store(
                    ring_dst + DIM + coord,
                    tl.load(Positions + row * NPOS + coord).to(tl.float32),
                )
            else:
                coord = tl.arange(0, 4)
                live = coord < NPOS
                tl.store(
                    ring_dst + DIM + coord,
                    tl.load(Positions + row * NPOS + coord, live, other=0).to(tl.float32),
                    live,
                )


@triton.jit
def gather_verify_group(
    Staged,
    Lengths,
    Slots,
    RingTable,
    Ring,
    Raw,
    GroupPositions,
    ring_table_stride,
    ring_page_stride,
    ring_row_stride,
    WIDTH: tl.constexpr,
    CLOSES: tl.constexpr,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    NPOS: tl.constexpr,
):
    request = tl.program_id(0)
    close = tl.program_id(1)
    length = tl.load(Lengths + request)
    prefix = length - WIDTH
    first = (prefix // RATIO + close) * RATIO
    live = (first + RATIO <= length) & (tl.load(Slots + request * WIDTH) >= 0)
    member = tl.arange(0, RATIO)
    d = tl.arange(0, DIM)
    row = (first - prefix) + member
    ring_id = tl.load(RingTable + request * ring_table_stride, live, other=0)
    ring = Ring + ring_id.to(tl.int64) * ring_page_stride
    staged = Staged + request * WIDTH * (DIM + NPOS)
    saved = tl.load(
        ring + member[:, None] * ring_row_stride + d[None, :],
        live & (row[:, None] < 0),
        other=0,
    )
    new = tl.load(
        staged + row[:, None] * (DIM + NPOS) + d[None, :],
        live & (row[:, None] >= 0),
        other=0,
    )
    slot = request * CLOSES + close
    tl.store(
        Raw + (slot * RATIO + member[:, None]) * DIM + d[None, :],
        tl.where(row[:, None] < 0, saved, new),
    )
    first_row = first - prefix
    coords = tl.arange(0, 1 if NPOS == 1 else 4)
    valid = coords < NPOS
    old_position = tl.load(ring + DIM + coords, live & (first_row < 0) & valid, other=0)
    new_position = tl.load(
        staged + first_row * (DIM + NPOS) + DIM + coords,
        live & (first_row >= 0) & valid,
        other=0,
    )
    tl.store(
        GroupPositions + slot * NPOS + coords,
        tl.where(first_row < 0, old_position, new_position),
        valid,
    )


@triton.jit
def store_verify_group(
    Pooled,
    Lengths,
    Slots,
    Table,
    Pool,
    table_stride,
    pool_page_stride,
    pool_row_stride,
    WIDTH: tl.constexpr,
    CLOSES: tl.constexpr,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    close = tl.program_id(1)
    length = tl.load(Lengths + request)
    group = (length - WIDTH) // RATIO + close
    if (group + 1) * RATIO <= length and tl.load(Slots + request * WIDTH) >= 0:
        logical_page = group // PAGE
        block = tl.load(Table + request * table_stride + logical_page // PAGES_PER_BLOCK)
        page = block * PAGES_PER_BLOCK + logical_page % PAGES_PER_BLOCK
        dst = Pool + page.to(tl.int64) * pool_page_stride
        dst += (group % PAGE) * pool_row_stride
        d = tl.arange(0, DIM)
        tl.store(dst + d, tl.load(Pooled + (request * CLOSES + close) * DIM + d))


@triton.jit
def _decode_slots(
    Lengths,
    Slots,
    Table,
    Groups,
    Indices,
    Counts,
    table_stride,
    indices_stride,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    active = tl.load(Slots + row) >= 0
    columns = tl.arange(0, BLOCK)
    # Complete groups start at a multiple of WORLD, including the causal tail.
    global_columns = columns * WORLD + RANK
    global_width = WIDTH * WORLD
    count = tl.minimum(length, global_width)
    if RATIO > 1:
        logical = global_columns
        if active & (length > global_width):
            history = global_width // RATIO - 1
            group = tl.load(
                Groups + row * history + global_columns // RATIO,
                global_columns < history * RATIO,
                other=0,
            )
            current = (length - 1) // RATIO
            group = tl.where(global_columns < history * RATIO, group, current)
            logical = group.to(tl.int32) * RATIO + global_columns % RATIO
            count = history * RATIO + (length - 1) % RATIO + 1
    else:
        logical = tl.maximum(length - global_width, 0) + global_columns
    live = active & (global_columns < count)
    logical = logical // WORLD
    page = tl.load(
        Table + (row // QUERY_WIDTH) * table_stride + logical // PAGE, live, other=0
    )
    physical = page * PAGE + logical % PAGE
    tl.store(
        Indices + row * indices_stride + columns,
        tl.where(live, physical, -1),
        columns < WIDTH,
    )
    tl.store(Counts + row, tl.where(active, (count + WORLD - 1 - RANK) // WORLD, 0))


def decode_slots(
    lengths: torch.Tensor,
    slots: torch.Tensor,
    table: torch.Tensor,
    groups: torch.Tensor | None,
    indices: torch.Tensor,
    counts: torch.Tensor,
    page: int,
    ratio: int,
    world: int = 1,
    rank: int = 0,
    *,
    query_width: int = 1,
) -> None:
    rows, width = indices.shape
    _decode_slots[(rows,)](
        lengths,
        slots,
        table,
        groups,
        indices,
        counts,
        table.stride(0),
        indices.stride(0),
        page,
        width,
        ratio if groups is not None else 1,
        triton.next_power_of_2(width),
        world,
        rank,
        query_width,
        num_warps=4,
    )
