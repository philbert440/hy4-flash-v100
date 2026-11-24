# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team.
# Adapted from transformers/models/hy_v4_flash/modeling_hy_v4_flash.py at b865850d05326cf20e306b31f14b0027de802da3.

"""Positions, interleaved rotary phases, and the image-embedding merge.

Hy4MergedSequence is input_ids int64 [S], inputs_embeds [S, 3072],
positions int64 [3, S], mrope_delta, and lookback int64 [S, K]. Text is
(p, p, p). Image pads, including the newline column, use (T, H, W), and
later text resumes at max(position) + 1.

The indexer open group is a separate tail, not this record. A text server
stores float32 [dim + 1] (key and one position). A vision server stores
float32 [dim + 3] (key and T, H, W of the group's first token).
"""

import itertools
from dataclasses import dataclass

import torch

from vllm.models.hy_v4_flash.sm70.image_process import (
    IMAGE_PAD_ID,
    MERGE_SIZE,
    VIDEO_REJECTION,
)
from vllm.models.hy_v4_flash.sm70.rotary import (
    rotary_phases as mrope_phases,
    span_tokens,
    take_tokens,
)


def vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = MERGE_SIZE,
    device: torch.device | None = None,
) -> torch.Tensor:
    """(3, slots) positions for one image, including one newline per row."""
    llm_t = int(grid_thw[0]) // 1
    llm_h = int(grid_thw[1]) // spatial_merge_size
    llm_w = int(grid_thw[2]) // spatial_merge_size
    block = max(llm_h, llm_w + 1)
    temporal = torch.arange(llm_t, device=device) * block
    height = torch.arange(llm_h, device=device)
    width = torch.arange(llm_w + 1, device=device)
    time_grid, height_grid, width_grid = torch.meshgrid(
        temporal, height, width, indexing="ij"
    )
    positions = torch.stack(
        [time_grid, height_grid + time_grid, width_grid + time_grid], dim=0
    ).reshape(3, -1)
    return positions + start_position


def rope_index(
    input_ids: torch.Tensor,
    mm_types: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int = MERGE_SIZE,
) -> tuple[torch.Tensor, int]:
    """Return [3, S] positions and mrope_delta for one sequence.

    mm_types is 0 for text and 1 for an image pad. Video (2) is rejected.
    A pure text sequence is arange on all three axes and delta 0.
    """
    device = input_ids.device
    pieces = []
    current = 0
    grids = iter(image_grid_thw if image_grid_thw is not None else [])
    grouped = itertools.groupby(enumerate(mm_types.tolist()), lambda item: item[1])
    for modality, group in grouped:
        group = list(group)
        start = group[0][0]
        end = group[-1][0] + 1
        if modality == 0:
            length = end - start
            pieces.append(
                torch.arange(length, device=device).view(1, -1).expand(3, -1) + current
            )
            current += length
            continue
        if modality != 1:
            raise ValueError(VIDEO_REJECTION)
        grid = next(grids)
        vision = vision_position_ids(current, grid, spatial_merge_size, device)
        pieces.append(vision)
        current = int(vision.max().item()) + 1
    positions = torch.cat(pieces, dim=1).to(dtype=torch.long)
    delta = int(positions.max().item()) + 1 - int(input_ids.shape[0])
    return positions, delta


def mm_types_for_ids(input_ids: torch.Tensor, image_pad_id: int = IMAGE_PAD_ID) -> torch.Tensor:
    return (input_ids == image_pad_id).to(dtype=torch.int)


def raw_lookback(
    input_ids: torch.Tensor,
    ngram: int,
    engram_pad_id: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """[S, ngram] raw ids. Column 0 is the current token. Image ids stay."""
    if positions is None:
        offsets = torch.arange(input_ids.shape[0], device=input_ids.device)
    elif positions.ndim == 2:
        offsets = positions[0]
    else:
        offsets = positions
    columns = [input_ids]
    for offset in range(1, ngram):
        previous = torch.nn.functional.pad(input_ids, (offset, 0), value=engram_pad_id)[
            : input_ids.shape[0]
        ]
        columns.append(torch.where(offsets < offset, engram_pad_id, previous))
    return torch.stack(columns, dim=-1)


def merge_image_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_pad_id: int = IMAGE_PAD_ID,
) -> torch.Tensor:
    """Scatter tower rows onto image-pad positions. Other rows stay put."""
    mask = input_ids == image_pad_id
    count = int(mask.sum())
    if count != image_embeds.shape[0]:
        raise ValueError(
            f"Image features and image tokens do not match, tokens: {count}, "
            f"features: {image_embeds.shape[0]}"
        )
    merged = inputs_embeds.clone()
    merged[mask] = image_embeds.to(dtype=merged.dtype, device=merged.device)
    return merged


@dataclass(frozen=True)
class Hy4MergedSequence:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    positions: torch.Tensor
    mrope_delta: int
    lookback_token_ids: torch.Tensor


def build_merged_sequence(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_embeds: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    *,
    image_pad_id: int = IMAGE_PAD_ID,
    engram_pad_id: int,
    ngram: int,
    spatial_merge_size: int = MERGE_SIZE,
) -> Hy4MergedSequence:
    if image_embeds is None:
        merged = inputs_embeds
        positions, delta = rope_index(
            input_ids, torch.zeros_like(input_ids, dtype=torch.int), None, spatial_merge_size
        )
    else:
        merged = merge_image_embeddings(
            inputs_embeds, input_ids, image_embeds, image_pad_id
        )
        positions, delta = rope_index(
            input_ids,
            mm_types_for_ids(input_ids, image_pad_id),
            image_grid_thw,
            spatial_merge_size,
        )
    lookback = raw_lookback(input_ids, ngram, engram_pad_id, positions)
    return Hy4MergedSequence(input_ids, merged, positions, delta, lookback)
