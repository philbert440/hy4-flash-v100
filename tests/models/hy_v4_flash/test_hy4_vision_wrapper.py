# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Image admission, merged embeddings, and three-axis positions for the wrapper."""

import json
import os

import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.vl_model import (
    HYV4FlashVLForConditionalGeneration,
    arm_visual_engram,
)
from vllm.models.hy_v4_flash.sm70.image_process import (
    IMAGE_END_ID,
    IMAGE_LIMIT_REJECTION,
    IMAGE_PAD_ID,
    IMAGE_START_ID,
    PIXEL_CAP,
    VIDEO_PAD_ID,
    VIDEO_REJECTION,
    max_slots_under_cap,
    process_pil,
    size_with_most_slots,
    slot_count,
)
from vllm.models.hy_v4_flash.sm70.vision_rope import merge_image_embeddings, rope_index
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.transformers_utils.config import uses_mrope

pytestmark = pytest.mark.skip_global_cleanup

CHECKPOINT = os.environ.get("HY4_FLASH_DIR", "hy4-flash-awq-g32")
METADATA = os.environ.get("HYV4_FLASH_CHECKPOINT_METADATA", "")


def _picture(width: int, height: int) -> Image.Image:
    import numpy as np

    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)
    array[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return Image.fromarray(array, "RGB")


def _model_config(path: str, *, architecture: str, limit: int, tokenizer: bool) -> ModelConfig:
    overrides = {"architectures": [architecture]} if architecture else {}
    return ModelConfig(
        model=path,
        hf_overrides=overrides,
        limit_mm_per_prompt={"image": limit, "video": 0},
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
        skip_tokenizer_init=not tokenizer,
        trust_remote_code=False,
    )


def test_tallest_legal_image_matches_the_slot_cap():
    height, width = size_with_most_slots()
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= PIXEL_CAP
    assert slot_count(height // 16, width // 16) == max_slots_under_cap()
    record = process_pil(_picture(width, height))
    assert record.num_slots == max_slots_under_cap()
    assert tuple(record.image_grid_thw.tolist()[0]) == (1, height // 16, width // 16)


def test_from_dict_stays_text_only():
    if not METADATA:
        pytest.skip("set HYV4_FLASH_CHECKPOINT_METADATA")
    raw = json.loads(
        open(os.path.join(METADATA, "config.json"), encoding="utf-8").read()
    )
    config = HYV4FlashConfig.from_dict(raw)
    assert config.architectures == ["HYV4FlashForCausalLM"]
    assert not hasattr(config, "vision_config")
    assert not uses_mrope(config)


def test_text_server_does_not_widen_positions():
    if not METADATA:
        pytest.skip("set HYV4_FLASH_CHECKPOINT_METADATA")
    config = _model_config(
        METADATA,
        architecture="HYV4FlashForCausalLM",
        limit=0,
        tokenizer=False,
    )
    assert config.architectures == ["HYV4FlashForCausalLM"]
    assert not config.uses_mrope
    assert not hasattr(config.hf_config, "vision_config")


def test_vision_server_marks_mrope_only_when_an_image_is_allowed():
    if not METADATA:
        pytest.skip("set HYV4_FLASH_CHECKPOINT_METADATA")
    closed = _model_config(
        METADATA,
        architecture="HYV4FlashVLForConditionalGeneration",
        limit=0,
        tokenizer=False,
    )
    assert not closed.uses_mrope
    open_server = _model_config(
        METADATA,
        architecture="HYV4FlashVLForConditionalGeneration",
        limit=1,
        tokenizer=False,
    )
    assert open_server.architectures == ["HYV4FlashVLForConditionalGeneration"]
    assert open_server.uses_mrope
    assert open_server.hf_config.rope_parameters["mrope_section"] == [12, 10, 10]
    assert not uses_mrope(HYV4FlashConfig.from_dict(
        json.loads(open(os.path.join(METADATA, "config.json"), encoding="utf-8").read())
    ))


def test_engram_arms_only_while_the_tower_is_loaded():
    class Hasher:
        visual_token_ids = ()

    class Language:
        def __init__(self):
            self.model = type("Model", (), {"engram_hash": Hasher()})()

    armed = Language()
    arm_visual_engram(armed, True)
    assert armed.model.engram_hash.visual_token_ids == (IMAGE_PAD_ID, VIDEO_PAD_ID)
    quiet = Language()
    arm_visual_engram(quiet, False)
    assert quiet.model.engram_hash.visual_token_ids == ()


def _bare_wrapper():
    model = HYV4FlashVLForConditionalGeneration.__new__(
        HYV4FlashVLForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    return model


def test_positions_follow_the_image_and_stay_equal_for_text():
    model = _bare_wrapper()
    text = list(range(6))
    positions, delta = model.get_mrope_input_positions(text, [])
    assert delta == 0
    assert torch.equal(positions[0], positions[1]) and torch.equal(positions[0], positions[2])
    pads = [IMAGE_PAD_ID] * 6
    tokens = [11, 12, IMAGE_START_ID, *pads, IMAGE_END_ID, 13]
    grid = torch.tensor([1, 4, 4])

    class Field:
        def __init__(self, data):
            self.data = data

    feature = MultiModalFeatureSpec(
        data={"image_grid_thw": Field(grid)},
        modality="image",
        identifier="same",
        mm_position=PlaceholderRange(offset=3, length=6),
    )
    got, got_delta = model.get_mrope_input_positions(tokens, [feature])
    expected, expected_delta = rope_index(
        torch.tensor(tokens),
        torch.tensor([0, 0, 0, *([1] * 6), 0, 0]),
        grid.reshape(1, 3),
    )
    assert got_delta == expected_delta
    assert torch.equal(got, expected)
    assert not torch.equal(got[1, 3:9], got[2, 3:9])


def test_embeddings_land_on_image_pads():
    model = _bare_wrapper()

    class Tower(torch.nn.Module):
        def forward(self, pixel_values, grid_thw):
            del pixel_values
            rows = grid_thw.reshape(-1, 3)
            slots = sum(slot_count(int(row[1]), int(row[2])) for row in rows.tolist())
            return torch.arange(slots * 4, dtype=torch.float32).reshape(slots, 4)

    class Language(torch.nn.Module):
        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 4)

    model.visual = Tower()
    model.language_model = Language()
    model._vision_enabled = True
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.zeros(16, 768)
    (features,) = model.embed_multimodal(pixel_values=pixels, image_grid_thw=grid)
    assert features.shape == (6, 4)
    ids = torch.tensor([3, IMAGE_START_ID, *([IMAGE_PAD_ID] * 6), IMAGE_END_ID])
    merged = model.embed_input_ids(
        ids,
        multimodal_embeddings=[features],
        is_multimodal=ids == IMAGE_PAD_ID,
    )
    expected = merge_image_embeddings(torch.zeros(ids.shape[0], 4), ids, features)
    assert torch.equal(merged, expected)


def test_load_weights_keeps_vision_tensors_out_of_the_decoder():
    model = _bare_wrapper()
    seen = {"text": [], "vision": []}

    class Language(torch.nn.Module):
        def load_weights(self, weights):
            names = []
            for name, _tensor in weights:
                names.append(name)
            seen["text"] = names
            return set(names)

    class Tower(torch.nn.Module):
        def load_weights(self, weights):
            names = [name for name, _tensor in weights]
            seen["vision"] = names
            return set(names)

    model.language_model = Language()
    model.visual = Tower()
    model._vision_enabled = True
    scalar = torch.zeros(1)
    model.load_weights(
        [
            ("model.layers.0.weight", scalar),
            ("vit.embeddings.weight", scalar),
            ("model.norm.weight", scalar),
        ]
    )
    assert seen["text"] == ["model.layers.0.weight", "model.norm.weight"]
    assert seen["vision"] == ["vit.embeddings.weight"]


def test_placeholder_names_the_image_token_and_video_is_rejected():
    assert HYV4FlashVLForConditionalGeneration.get_placeholder_str("image", 0).count(
        "<｜hy_image▁pad:opensource｜>"
    ) == 1
    with pytest.raises(ValueError, match="At most 0 video") as raised:
        HYV4FlashVLForConditionalGeneration.get_placeholder_str("video", 0)
    assert "rejects video" in str(raised.value)


@pytest.fixture(scope="module")
def vision_processor():
    if not os.path.isdir(CHECKPOINT):
        pytest.skip("set HY4_FLASH_DIR")
    config = _model_config(
        CHECKPOINT,
        architecture="HYV4FlashVLForConditionalGeneration",
        limit=1,
        tokenizer=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(config), config


def test_processor_matches_the_reference_prompt_and_pixels(vision_processor):
    processor, _config = vision_processor
    from transformers import AutoProcessor

    reference_processor = AutoProcessor.from_pretrained(CHECKPOINT)
    image = _picture(173, 113)
    prompt = reference_processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image"},
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    reference = reference_processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        max_pixels=PIXEL_CAP,
    )
    items = processor.info.parse_mm_data({"image": image})
    processed = processor(
        prompt,
        items,
        hf_processor_mm_kwargs={"max_pixels": PIXEL_CAP},
    )
    assert processed["prompt_token_ids"] == reference.input_ids[0].tolist()
    image_range = processed["mm_placeholders"]["image"][0]
    pads = [
        token
        for token, embed in zip(
            processed["prompt_token_ids"][image_range.offset : image_range.offset + image_range.length],
            image_range.is_embed.tolist(),
        )
        if embed
    ]
    assert pads == [IMAGE_PAD_ID] * int((reference.input_ids[0] == IMAGE_PAD_ID).sum())
    assert processed["prompt_token_ids"][image_range.offset] == IMAGE_START_ID
    record = process_pil(image, max_pixels=PIXEL_CAP)
    pixels = processed["mm_kwargs"]["image"][0]["pixel_values"].data
    grid = processed["mm_kwargs"]["image"][0]["image_grid_thw"].data
    assert torch.equal(pixels, record.pixel_values)
    assert torch.equal(grid.reshape(-1), record.image_grid_thw.reshape(-1))


def test_processor_rejects_a_second_image_video_and_the_pixel_cap(vision_processor):
    processor, _config = vision_processor
    image = _picture(32, 32)
    items = processor.info.parse_mm_data({"image": [image, image]}, validate=False)
    with pytest.raises(ValueError, match="At most 1 image") as raised:
        processor.info.validate_num_items("image", 2)
    assert str(raised.value) == IMAGE_LIMIT_REJECTION or "At most 1 image" in str(raised.value)
    with pytest.raises(ValueError, match="At most 0 video") as video:
        processor.info.parse_mm_data({"video": [image]})
    assert VIDEO_REJECTION.split(".")[0] in str(video.value) or "At most 0 video" in str(video.value)
    with pytest.raises(ValueError, match="1048576"):
        processor._call_hf_processor(
            "plain",
            {"images": [_picture(1025, 1024)]},
            {"max_pixels": PIXEL_CAP},
            {},
        )
    with pytest.raises(ValueError, match="1048576"):
        processor._call_hf_processor(
            "plain",
            {"images": [image]},
            {"max_pixels": PIXEL_CAP + 1},
            {},
        )
    with pytest.raises(ValueError, match="At most 1 image"):
        processor._call_hf_processor("plain", {"images": [image, image]}, {}, {})
    tiny = _picture(3, 2)
    with pytest.raises(ValueError, match="1048576"):
        processor._call_hf_processor(
            "plain",
            {"images": [tiny]},
            {"min_pixels": 2_097_152, "max_pixels": PIXEL_CAP},
            {},
        )
    with pytest.raises(ValueError, match="1048576"):
        processor._call_hf_processor(
            "plain",
            {"images": [tiny]},
            {"min_pixels": PIXEL_CAP, "max_pixels": PIXEL_CAP},
            {},
        )
    del items


def test_wrapped_model_constructs_on_meta(cpu_vllm):
    if not METADATA:
        pytest.skip("set HYV4_FLASH_CHECKPOINT_METADATA")
    cpu_vllm.model_config = _model_config(
        METADATA,
        architecture="HYV4FlashVLForConditionalGeneration",
        limit=1,
        tokenizer=False,
    )
    cpu_vllm.scheduler_config.max_num_batched_tokens = 2048
    with torch.device("meta"):
        model = HYV4FlashVLForConditionalGeneration(vllm_config=cpu_vllm)
    assert model._vision_enabled
    assert model.token_lookback_depth == model.language_model.token_lookback_depth
    assert model.language_model.model.engram_hash.visual_token_ids[0] == IMAGE_PAD_ID
    assert any(parameter.is_meta for parameter in model.visual.parameters())
