# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team.
# Adapted from transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py at b865850d05326cf20e306b31f14b0027de802da3.

"""One Hy4 image, resized and packed the way the pinned processor packs it.

Hy4ImageRecord is pixel_values float32 [P, 768], image_grid_thw int64 [1, 3],
and num_slots = rows * (cols + 1). Rows and cols are the merged grid.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF

from vllm.multimodal.image import convert_image_mode

PIXEL_CAP = 1_048_576
PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 1
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 4096
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
RESCALE_FACTOR = 1.0 / 255.0

IMAGE_START_ID = 120118
IMAGE_END_ID = 120119
IMAGE_PAD_ID = 120120
IMAGE_NEWLINE_ID = 120121
VIDEO_START_ID = 120122
VIDEO_END_ID = 120123
VIDEO_PAD_ID = 120124

IMAGE_START_TOKEN = "<｜hy_image▁start:opensource｜>"
IMAGE_END_TOKEN = "<｜hy_image▁end:opensource｜>"
IMAGE_PAD_TOKEN = "<｜hy_image▁pad:opensource｜>"
VIDEO_START_TOKEN = "<｜hy_video▁start:opensource｜>"
VIDEO_END_TOKEN = "<｜hy_video▁end:opensource｜>"
VIDEO_PAD_TOKEN = "<｜hy_video▁pad:opensource｜>"

VIDEO_REJECTION = (
    "Hy4 Flash rejects video. At most 0 video(s) may be provided in one prompt."
)
IMAGE_LIMIT_REJECTION = "At most 1 image(s) may be provided in one prompt."


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = PIXEL_CAP,
) -> tuple[int, int]:
    """Match HYV4VLImageProcessor.smart_resize, including its rounding."""
    if min(height, width) <= 0:
        raise ValueError(f"image size must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def reject_video() -> None:
    raise ValueError(VIDEO_REJECTION)


def check_image_limits(
    height: int,
    width: int,
    *,
    max_pixels: int = PIXEL_CAP,
    pixel_cap: int = PIXEL_CAP,
    min_pixels: int | None = None,
) -> None:
    """Reject a raw image, or a pixel request, that exceeds the cap.

    The engine sizes its image slots to ``pixel_cap``. A ``min_pixels``
    above that cap, or above ``max_pixels``, cannot be honored: the
    reference resize upscales in a separate branch and can land past the
    maximum. Name the limit instead of resizing past it.
    """
    if max_pixels > pixel_cap:
        raise ValueError(
            f"Hy4 Flash rejects max_pixels {max_pixels}. "
            f"At most {pixel_cap} pixels may be provided in one prompt."
        )
    limit = min(pixel_cap, max_pixels)
    if min_pixels is not None and min_pixels > limit:
        raise ValueError(
            f"Hy4 Flash rejects min_pixels {min_pixels}. "
            f"At most {limit} pixels may be requested."
        )
    raw = height * width
    if raw > pixel_cap:
        raise ValueError(
            f"Hy4 Flash rejects an image of {raw} pixels. "
            f"At most {pixel_cap} pixels may be provided in one prompt."
        )


def slot_count(grid_h: int, grid_w: int, merge_size: int = MERGE_SIZE) -> int:
    """Merged rows times (merged columns plus the learned newline)."""
    return (grid_h // merge_size) * (grid_w // merge_size + 1)


def max_slots_under_cap(pixel_cap: int = PIXEL_CAP, factor: int = FACTOR) -> int:
    """Most newline-padded slots a legal image can occupy.

    The square cap is not the longest sequence. Each merged row adds a
    newline, so a tall image under the same pixel cap uses more slots.
    """
    best = 0
    height = factor
    while height * factor <= pixel_cap:
        limit = min(pixel_cap // height, height * 200)
        width = (limit // factor) * factor
        if width >= factor and max(height, width) / min(height, width) <= 200:
            best = max(best, (height // factor) * (width // factor + 1))
        height += factor
    return best



def size_with_most_slots(
    pixel_cap: int = PIXEL_CAP, factor: int = FACTOR
) -> tuple[int, int]:
    """Resized (height, width) that occupies max_slots_under_cap.

    The search matches max_slots_under_cap, so the two cannot drift.
    Ties keep the later, taller image.
    """
    best_slots = -1
    best = (factor, factor)
    height = factor
    while height * factor <= pixel_cap:
        limit = min(pixel_cap // height, height * 200)
        width = (limit // factor) * factor
        if width >= factor and max(height, width) / min(height, width) <= 200:
            slots = (height // factor) * (width // factor + 1)
            if slots >= best_slots:
                best_slots = slots
                best = (height, width)
        height += factor
    return best


def patchify(
    images: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> tuple[torch.Tensor, int, int]:
    """Pack [B, C, H, W] into [B, patches, C * patch * patch], row-major."""
    batch, channel, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(
        batch,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 3, 5, 6, 1, 4, 7)
    flat = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            batch,
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
    )
    return flat, grid_h, grid_w


@dataclass(frozen=True)
class Hy4ImageRecord:
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    num_slots: int


def process_pil(
    image: Image.Image,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = PIXEL_CAP,
    pixel_cap: int = PIXEL_CAP,
    mean: tuple[float, ...] | float = IMAGE_MEAN,
    std: tuple[float, ...] | float = IMAGE_STD,
) -> Hy4ImageRecord:
    """Resize one RGB image and return the processor's pixel record."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL image, got {type(image).__name__}")
    # Same white composite the OpenAI media path uses. A bare RGB convert
    # puts transparent pixels on black.
    rgb = convert_image_mode(image, "RGB")
    width, height = rgb.size
    check_image_limits(
        height,
        width,
        max_pixels=max_pixels,
        pixel_cap=pixel_cap,
        min_pixels=min_pixels,
    )
    resized_h, resized_w = smart_resize(
        height, width, FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )
    aligned = resized_h * resized_w
    if aligned > pixel_cap:
        raise ValueError(
            f"Hy4 Flash rejects a resized image of {aligned} pixels "
            f"requested with min_pixels {min_pixels}. "
            f"At most {pixel_cap} pixels may be provided in one prompt."
        )
    array = np.array(rgb)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    resized = tvF.resize(
        tensor,
        [resized_h, resized_w],
        interpolation=tvF.InterpolationMode.BICUBIC,
        antialias=True,
    )
    # Match the fused rescale: multiply by 255, which is exact. Dividing by
    # 1/255 in float32 is one ulp off.
    scale = 1.0 / RESCALE_FACTOR
    fused_mean = torch.tensor(mean, dtype=torch.float32) * scale
    fused_std = torch.tensor(std, dtype=torch.float32) * scale
    normalized = tvF.normalize(resized.to(torch.float32), fused_mean, fused_std)
    flat, grid_h, grid_w = patchify(normalized)
    grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return Hy4ImageRecord(flat[0], grid, slot_count(grid_h, grid_w))


def expand_image_span(
    input_ids: torch.Tensor,
    num_slots: int,
    start_id: int = IMAGE_START_ID,
    pad_id: int = IMAGE_PAD_ID,
    end_id: int = IMAGE_END_ID,
) -> torch.Tensor:
    """Replace the single start/pad/end triple with start, num_slots pads, end."""
    ids = input_ids.tolist()
    found = []
    index = 0
    while index + 2 < len(ids):
        if ids[index] == start_id and ids[index + 1] == pad_id and ids[index + 2] == end_id:
            found.append(index)
            index += 3
        else:
            index += 1
    if len(found) == 0:
        raise ValueError(
            "Hy4 image prompt is missing "
            f"[{start_id}, {pad_id}, {end_id}]"
        )
    if len(found) != 1:
        raise ValueError(IMAGE_LIMIT_REJECTION)
    at = found[0]
    expanded = ids[: at + 1] + [pad_id] * num_slots + ids[at + 2 :]
    return torch.tensor(expanded, dtype=input_ids.dtype)
