# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/models/hy_v4_flash/nvidia/vl_model.py at 596455c7f.

"""Hy4-flash decoder with one image tower on the SM70 port."""

import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors

from vllm.models.hy_v4_flash.sm70.image_process import (
    IMAGE_END_TOKEN,
    IMAGE_PAD_ID,
    IMAGE_PAD_TOKEN,
    IMAGE_START_TOKEN,
    VIDEO_PAD_ID,
    reject_video,
    slot_count,
)
from vllm.models.hy_v4_flash.sm70.mm_preprocess import (
    Hy4FlashDummyInputsBuilder,
    Hy4FlashMultiModalProcessor,
    Hy4FlashProcessingInfo,
    grid_rows,
)
from vllm.models.hy_v4_flash.sm70.vision_rope import mm_types_for_ids, rope_index

from .model import HYV4FlashForCausalLM
from .vision import Hy4VisionTransformer

_VISUAL_TOKEN_IDS = (IMAGE_PAD_ID, VIDEO_PAD_ID)


def load_vision_config(model_path: str) -> SimpleNamespace:
    """Read vision_config from the checkpoint file.

    HYV4FlashConfig.from_dict drops vision_config. The tower reads the file
    again so the text config class stays unchanged.
    """
    path = Path(model_path) / "config.json"
    raw = json.loads(path.read_text())
    vision = raw.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError(f"Hy4 checkpoint {path} has no vision_config")
    config = SimpleNamespace(**vision)
    if not hasattr(config, "layer_norm_eps"):
        config.layer_norm_eps = config.rms_norm_eps
    return config


def arm_visual_engram(language_model: nn.Module, enabled: bool) -> None:
    """Zero Engram gathers at image pads only when the tower is loaded."""
    hasher = getattr(getattr(language_model, "model", None), "engram_hash", None)
    if hasher is None:
        return
    hasher.visual_token_ids = _VISUAL_TOKEN_IDS if enabled else ()


def _feature_grid(feature: MultiModalFeatureSpec) -> torch.Tensor:
    if feature.modality == "video":
        reject_video()
    item = feature.data
    if item is None or "image_grid_thw" not in item or item["image_grid_thw"].data is None:
        raise ValueError("Hy4 image positions need image_grid_thw on the multimodal feature")
    return grid_rows(item["image_grid_thw"].data)[0]


@MULTIMODAL_REGISTRY.register_processor(
    Hy4FlashMultiModalProcessor,
    info=Hy4FlashProcessingInfo,
    dummy_inputs=Hy4FlashDummyInputsBuilder,
)
class HYV4FlashVLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsMRoPE
):
    requires_raw_input_tokens = True
    packed_modules_mapping = dict(HYV4FlashForCausalLM.packed_modules_mapping)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        quant = vllm_config.quant_config
        if quant is not None:
            quant.packed_modules_mapping = self.packed_modules_mapping
        vision_config = load_vision_config(vllm_config.model_config.model)
        with self._mark_tower_model(vllm_config, "image"):
            self.visual = Hy4VisionTransformer(
                vision_config,
                dtype=vllm_config.model_config.dtype,
            )
        with self._mark_language_model(vllm_config):
            self.language_model = HYV4FlashForCausalLM(
                vllm_config=vllm_config,
                prefix="",
            )
        self._vision_enabled = not isinstance(self.visual, StageMissingLayer)
        arm_visual_engram(self.language_model, self._vision_enabled)
        self.token_lookback_depth = self.language_model.token_lookback_depth
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return IMAGE_START_TOKEN + IMAGE_PAD_TOKEN + IMAGE_END_TOKEN
        if modality == "video":
            reject_video()
        raise ValueError(f"Hy4 Flash has no placeholder for {modality}")

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        if any(key in kwargs for key in ("pixel_values_videos", "video_grid_thw", "videos")):
            reject_video()
        pixel_values = kwargs.get("pixel_values")
        grid = kwargs.get("image_grid_thw")
        if pixel_values is None or grid is None:
            return ()
        if not self._vision_enabled:
            raise RuntimeError(
                "Hy4 vision tower is not loaded. "
                "At most 0 image(s) may be provided in one prompt."
            )
        grid_thw = grid_rows(grid)
        features = self.visual(pixel_values, grid_thw)
        sizes = [slot_count(int(row[1]), int(row[2])) for row in grid_thw.tolist()]
        if sum(sizes) != features.shape[0]:
            raise ValueError(
                f"Image features and image slots do not match, slots: {sum(sizes)}, "
                f"features: {features.shape[0]}"
            )
        return tuple(features.split(sizes))

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        ids = torch.tensor(input_tokens, dtype=torch.long)
        rows = [
            _feature_grid(feature)
            for feature in mm_features
            if feature.modality != "prompt_embeds"
        ]
        if not rows:
            types = torch.zeros(ids.shape[0], dtype=torch.int)
            return rope_index(ids, types, None)
        return rope_index(ids, mm_types_for_ids(ids), torch.stack(rows))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Hy4 vision forward got unexpected arguments: {unexpected}")
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    @staticmethod
    def skip_checkpoint_weight(name: str) -> bool:
        # get_all_weights asks the top module. The tower is a parameter here,
        # so vit.* stays in this loader. Engram tables and dfly follow the
        # text class: the tables are read by _load_engram_tables, and dfly
        # is not a parameter of this port. Any other name still has to match
        # a parameter or the load raises.
        if name.startswith("vit."):
            return False
        return HYV4FlashForCausalLM.skip_checkpoint_weight(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vision: list[tuple[str, torch.Tensor]] = []

        def text_weights():
            for name, tensor in weights:
                if name.startswith("vit."):
                    if self._vision_enabled:
                        vision.append((name, tensor))
                    continue
                yield name, tensor

        loaded = set(self.language_model.load_weights(text_weights()))
        if self._vision_enabled:
            seen = self.visual.load_weights(vision)
            loaded.update(f"vit.{name}" for name in seen)
        return loaded
