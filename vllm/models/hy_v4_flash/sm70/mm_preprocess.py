# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/model_executor/models/qwen2_vl.py in this tree and vllm-upstream vllm/models/hy_v4/nvidia/mm_preprocess.py at 596455c7f.

"""Admit one Hy4 image through the engine's multimodal processor."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from .image_process import (
    IMAGE_END_ID,
    IMAGE_END_TOKEN,
    IMAGE_LIMIT_REJECTION,
    IMAGE_PAD_ID,
    IMAGE_PAD_TOKEN,
    IMAGE_START_ID,
    IMAGE_START_TOKEN,
    PIXEL_CAP,
    check_image_limits,
    max_slots_under_cap,
    process_pil,
    reject_video,
    size_with_most_slots,
    slot_count,
)

_PLACEHOLDER = (IMAGE_START_ID, IMAGE_PAD_ID, IMAGE_END_ID)
_PLACEHOLDER_TEXT = IMAGE_START_TOKEN + IMAGE_PAD_TOKEN + IMAGE_END_TOKEN


def grid_rows(grid: torch.Tensor) -> torch.Tensor:
    """Return image grids as int64 [N, 3]."""
    rows = torch.as_tensor(grid, dtype=torch.long).reshape(-1, 3)
    if rows.numel() == 0:
        raise ValueError("Hy4 image grid is empty")
    return rows


def slots_for_grid(grid: torch.Tensor) -> int:
    row = grid_rows(grid)[0]
    return slot_count(int(row[1]), int(row[2]))


class Hy4FlashProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1, "video": 0}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"image": max_slots_under_cap(), "video": 0}

    def get_image_size_with_most_features(self) -> ImageSize:
        height, width = size_with_most_slots()
        return ImageSize(width=width, height=height)

    def validate_num_items(self, modality: str, num_items: int) -> None:
        if modality == "video" and num_items > 0:
            reject_video()
        super().validate_num_items(modality, num_items)

    def get_hf_processor(self, **kwargs: object):
        raise RuntimeError(
            "Hy4 image pixels are packed by process_pil, not the Hugging Face processor"
        )


class Hy4FlashDummyInputsBuilder(BaseDummyInputsBuilder[Hy4FlashProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        if mm_counts.get("video", 0):
            reject_video()
        return _PLACEHOLDER_TEXT * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        if mm_counts.get("video", 0):
            reject_video()
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class Hy4FlashMultiModalProcessor(BaseMultiModalProcessor[Hy4FlashProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return False

    def _tokenize(self, prompt: str, tok_kwargs: Mapping[str, object]) -> list[int]:
        kwargs: dict[str, object] = {
            "add_special_tokens": tok_kwargs.get("add_special_tokens", False)
        }
        encoded = self.info.get_tokenizer().encode(prompt, **kwargs)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        return list(encoded)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if any(key in mm_data for key in ("video", "videos")):
            reject_video()
        images = mm_data.get("images")
        input_ids = self._tokenize(prompt, tok_kwargs)
        if not images:
            return BatchFeature(data={"input_ids": torch.tensor([input_ids])})
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise TypeError(f"Hy4 images must be a sequence, got {type(images).__name__}")
        if len(images) > 1:
            raise ValueError(IMAGE_LIMIT_REJECTION)
        image = images[0]
        max_pixels = int(mm_kwargs.get("max_pixels", PIXEL_CAP))
        min_pixels = int(mm_kwargs.get("min_pixels", 4096))
        check_image_limits(1, 1, max_pixels=max_pixels)
        record = process_pil(image, min_pixels=min_pixels, max_pixels=max_pixels)
        return BatchFeature(
            data={
                "input_ids": torch.tensor([input_ids]),
                "pixel_values": record.pixel_values,
                "image_grid_thw": record.image_grid_thw,
            }
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs

        def replacement(item_idx: int):
            item = out_mm_kwargs["image"][item_idx]
            grid = item["image_grid_thw"].data
            if grid is None:
                raise ValueError("Hy4 prompt update needs image_grid_thw")
            slots = slots_for_grid(grid)
            tokens = [IMAGE_START_ID, *([IMAGE_PAD_ID] * slots), IMAGE_END_ID]
            return PromptUpdateDetails.select_token_id(tokens, IMAGE_PAD_ID)

        return [
            PromptReplacement(
                modality="image",
                target=list(_PLACEHOLDER),
                replacement=replacement,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        grid = hf_inputs.get("image_grid_thw")
        if grid is None:
            return {}
        rows = grid_rows(grid)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", rows.prod(-1)),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }
