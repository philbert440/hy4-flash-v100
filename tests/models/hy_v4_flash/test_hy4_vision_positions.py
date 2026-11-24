# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Processor pixels, prompt tokens, and MRoPE positions against the pinned oracle."""

import os

import pytest
import torch
from PIL import Image

from vllm.models.hy_v4_flash.sm70.image_process import (
    IMAGE_END_ID,
    IMAGE_LIMIT_REJECTION,
    IMAGE_PAD_ID,
    IMAGE_START_ID,
    PIXEL_CAP,
    VIDEO_END_ID,
    VIDEO_PAD_ID,
    VIDEO_REJECTION,
    VIDEO_START_ID,
    check_image_limits,
    expand_image_span,
    max_slots_under_cap,
    process_pil,
    reject_video,
    slot_count,
)
from vllm.models.hy_v4_flash.sm70.vision_rope import (
    build_merged_sequence,
    merge_image_embeddings,
    mrope_phases,
    rope_index,
    span_tokens,
    take_tokens,
)

CHECKPOINT = os.environ.get("HY4_FLASH_DIR", "hy4-flash-awq-g32")
SIZES = [(32, 32), (173, 113), (320, 256), (1024, 1024)]


def _picture(width: int, height: int) -> Image.Image:
    import numpy as np

    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)
    array[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array[..., 2] = (np.arange(height)[:, None] + np.arange(width)[None, :]) % 256
    return Image.fromarray(array, "RGB")


def _same(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    if got.shape != ref.shape or not torch.equal(got, ref):
        detail = "shape"
        if got.numel() == ref.numel() and got.shape == ref.shape:
            detail = f"max_abs {(got.float() - ref.float()).abs().max().item()}"
        raise AssertionError(f"{name} {tuple(got.shape)} vs {tuple(ref.shape)} {detail}")


@pytest.fixture(scope="module")
def hf_bundle():
    from transformers import AutoProcessor
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import (
        HYV4FlashConfig,
    )
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        HYV4FlashModel,
        HYV4FlashRotaryEmbedding,
        HYV4FlashTextModel,
    )

    # vLLM registers its text config on AutoConfig during ModelConfig setup.
    config = HYV4FlashConfig.from_pretrained(CHECKPOINT)
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    return {
        "config": config,
        "processor": processor,
        "model": HYV4FlashModel,
        "text": HYV4FlashTextModel,
        "rotary": HYV4FlashRotaryEmbedding(config.text_config),
    }


def _prompt(processor, content):
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _ours_and_reference(bundle, content, image):
    processor = bundle["processor"]
    prompt = _prompt(processor, content)
    image_processor = processor.image_processor
    kwargs = {
        "min_pixels": image_processor.size["shortest_edge"],
        "max_pixels": PIXEL_CAP,
    }
    reference = processor(text=prompt, images=image, return_tensors="pt", **kwargs)
    record = process_pil(
        image,
        min_pixels=kwargs["min_pixels"],
        max_pixels=kwargs["max_pixels"],
        mean=tuple(image_processor.image_mean),
        std=tuple(image_processor.image_std),
    )
    bare = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    expanded = expand_image_span(bare.input_ids[0], record.num_slots)
    return record, expanded, reference, prompt


def test_token_ids_match_the_checkpoint_strings(hf_bundle):
    tokenizer = hf_bundle["processor"].tokenizer
    pairs = {
        "<｜hy_image▁start:opensource｜>": IMAGE_START_ID,
        "<｜hy_image▁end:opensource｜>": IMAGE_END_ID,
        "<｜hy_image▁pad:opensource｜>": IMAGE_PAD_ID,
        "<｜hy_video▁start:opensource｜>": VIDEO_START_ID,
        "<｜hy_video▁end:opensource｜>": VIDEO_END_ID,
        "<｜hy_video▁pad:opensource｜>": VIDEO_PAD_ID,
    }
    for text, token_id in pairs.items():
        assert tokenizer.convert_tokens_to_ids(text) == token_id


def test_limits_name_the_cap_and_video():
    with pytest.raises(ValueError, match="1048576"):
        check_image_limits(1025, 1024)
    with pytest.raises(ValueError, match="1048576"):
        check_image_limits(32, 32, max_pixels=PIXEL_CAP + 1)
    check_image_limits(1024, 1024)
    with pytest.raises(ValueError, match="rejects video") as raised:
        reject_video()
    assert "At most 0 video" in str(raised.value)
    with pytest.raises(ValueError, match="At most 1 image"):
        expand_image_span(
            torch.tensor(
                [IMAGE_START_ID, IMAGE_PAD_ID, IMAGE_END_ID] * 2
            ),
            4,
        )
    tiny = Image.new("RGB", (3, 2), (1, 2, 3))
    with pytest.raises(ValueError, match="1048576"):
        process_pil(tiny, min_pixels=2_097_152)
    with pytest.raises(ValueError, match="1048576"):
        process_pil(tiny, min_pixels=PIXEL_CAP)
    with pytest.raises(ValueError, match="4096"):
        process_pil(tiny, min_pixels=8000, max_pixels=4096)
    default = process_pil(Image.new("RGB", (32, 32), (4, 5, 6)))
    assert default.image_grid_thw.tolist() == [[1, 4, 4]]
    assert default.num_slots == slot_count(4, 4)
    explicit = process_pil(Image.new("RGB", (32, 32), (4, 5, 6)), min_pixels=4096)
    assert explicit.image_grid_thw.tolist() == [[1, 4, 4]]


def test_rgba_composites_on_white():
    from vllm.multimodal.image import rgba_to_rgb

    clear = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    white = Image.new("RGB", (32, 32), (255, 255, 255))
    black = Image.new("RGB", (32, 32), (0, 0, 0))
    assert torch.equal(process_pil(clear).pixel_values, process_pil(white).pixel_values)
    assert not torch.equal(process_pil(clear).pixel_values, process_pil(black).pixel_values)
    partial = Image.new("RGBA", (64, 64), (0, 0, 255, 128))
    assert torch.equal(
        process_pil(partial).pixel_values,
        process_pil(rgba_to_rgb(partial)).pixel_values,
    )


def test_fourteen_slots_match_a_real_grid():
    assert slot_count(4, 12) == 14
    record = process_pil(Image.new("RGB", (192, 64), (8, 8, 8)))
    assert record.image_grid_thw.tolist() == [[1, 4, 12]]
    assert record.num_slots == 14


@pytest.mark.parametrize("size", SIZES)
def test_pixels_grid_and_prompt_match_the_reference(hf_bundle, size):
    image = _picture(*size)
    record, expanded, reference, prompt = _ours_and_reference(
        hf_bundle,
        [{"type": "image"}, {"type": "text", "text": "What is this?"}],
        image,
    )
    assert "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>What is this?" in prompt
    _same("pixels", record.pixel_values, reference.pixel_values)
    _same("grid", record.image_grid_thw, reference.image_grid_thw)
    assert record.num_slots == int((reference.input_ids[0] == IMAGE_PAD_ID).sum())
    _same("tokens", expanded, reference.input_ids[0])
    if size == (32, 32):
        assert record.image_grid_thw.tolist() == [[1, 4, 4]]
        assert record.num_slots == 6
    if size == (320, 256):
        assert record.image_grid_thw.tolist() == [[1, 16, 20]]
        assert record.num_slots == 88
    if size == (1024, 1024):
        assert record.image_grid_thw.tolist() == [[1, 64, 64]]
        assert record.num_slots == slot_count(64, 64) == 1056


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image"}, {"type": "text", "text": "before"}],
        [
            {"type": "text", "text": "before"},
            {"type": "image"},
            {"type": "text", "text": "after"},
        ],
        [{"type": "text", "text": "before"}, {"type": "image"}],
    ],
)
def test_positions_and_merge_match_the_reference(hf_bundle, content):
    bundle = hf_bundle
    image = _picture(320, 256)
    record, expanded, reference, _prompt_text = _ours_and_reference(bundle, content, image)
    _same("tokens", expanded, reference.input_ids[0])
    oracle = type(
        "Oracle",
        (),
        {
            "config": bundle["config"],
            "get_vision_position_ids": bundle["model"].get_vision_position_ids,
        },
    )()
    ref_pos, ref_delta = bundle["model"].get_rope_index(
        oracle,
        reference.input_ids,
        reference.mm_token_type_ids,
        reference.image_grid_thw,
    )
    positions, delta = rope_index(
        expanded,
        (expanded == IMAGE_PAD_ID).to(torch.int),
        record.image_grid_thw,
    )
    _same("positions", positions, ref_pos[:, 0])
    assert delta == int(ref_delta.reshape(-1)[0])
    hidden = bundle["config"].text_config.hidden_size
    base = torch.arange(expanded.shape[0] * hidden, dtype=torch.float32).reshape(
        expanded.shape[0], hidden
    )
    image_rows = torch.arange(record.num_slots * hidden, dtype=torch.float32).reshape(
        record.num_slots, hidden
    ) + 0.25
    mask, _video = bundle["model"].get_placeholder_mask(
        oracle,
        expanded.unsqueeze(0),
        base.unsqueeze(0),
        image_features=image_rows,
    )
    reference_merged = base.unsqueeze(0).masked_scatter(mask, image_rows)[0]
    merged = build_merged_sequence(
        expanded,
        base,
        image_rows,
        record.image_grid_thw,
        engram_pad_id=bundle["config"].text_config.engram_pad_id,
        ngram=bundle["config"].text_config.engram_max_ngram_size,
    )
    _same("merged", merged.inputs_embeds, reference_merged)
    assert torch.equal(merged.inputs_embeds[expanded != IMAGE_PAD_ID], base[expanded != IMAGE_PAD_ID])
    text_stub = type("Text", (), {"config": bundle["config"].text_config})()
    ref_lookback = bundle["text"].build_ngram_tokens(
        text_stub, expanded.unsqueeze(0), positions.unsqueeze(1)
    )
    _same("lookback", merged.lookback_token_ids, ref_lookback[0])
    pad_rows = expanded == IMAGE_PAD_ID
    assert torch.equal(merged.lookback_token_ids[pad_rows, 0], expanded[pad_rows])
    after = int(pad_rows.nonzero()[-1]) + 1
    if after < expanded.shape[0]:
        assert IMAGE_PAD_ID in merged.lookback_token_ids[after].tolist()


def test_text_only_positions_are_one_dimensional(hf_bundle):
    processor = hf_bundle["processor"]
    prompt = _prompt(hf_bundle["processor"], [{"type": "text", "text": "hello"}])
    ids = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert IMAGE_PAD_ID not in ids.tolist()
    positions, delta = rope_index(ids, torch.zeros_like(ids, dtype=torch.int), None)
    expected = torch.arange(ids.shape[0]).view(1, -1).expand(3, -1)
    _same("text positions", positions, expected)
    assert delta == 0
    section = hf_bundle["config"].text_config.rope_parameters["indexed_attention"]["mrope_section"]
    inv = hf_bundle["rotary"].indexed_attention_inv_freq
    one = positions[0]
    cos_1d, sin_1d = mrope_phases(one, inv)
    cos_3d, sin_3d = mrope_phases(positions, inv, section)
    _same("cos", cos_3d, cos_1d)
    _same("sin", sin_3d, sin_1d)
    probe = torch.zeros(1, ids.shape[0], 64)
    ref_cos, ref_sin = hf_bundle["rotary"](probe, positions.unsqueeze(1), "indexed_attention")
    _same("hf cos", cos_3d, ref_cos[0, :, :32].float())
    _same("hf sin", sin_3d, ref_sin[0, :, :32].float())
    swa = hf_bundle["rotary"].sliding_attention_inv_freq
    swa_section = hf_bundle["config"].text_config.rope_parameters["sliding_attention"]["mrope_section"]
    swa_cos, _swa_sin = mrope_phases(positions, swa, swa_section)
    swa_1d, _swa_1d_sin = mrope_phases(one, swa)
    _same("swa", swa_cos, swa_1d)


def test_distinct_axes_match_the_reference_rotary(hf_bundle):
    image = _picture(173, 113)
    _record, expanded, reference, _prompt_text = _ours_and_reference(
        hf_bundle,
        [{"type": "text", "text": "left"}, {"type": "image"}, {"type": "text", "text": "right"}],
        image,
    )
    positions, _delta = rope_index(
        expanded, (expanded == IMAGE_PAD_ID).to(torch.int), reference.image_grid_thw
    )
    assert not torch.equal(positions[0], positions[1])
    for layer in ("indexed_attention", "sliding_attention"):
        section = hf_bundle["config"].text_config.rope_parameters[layer]["mrope_section"]
        assert section == [12, 10, 10]
        inv = getattr(hf_bundle["rotary"], f"{layer}_inv_freq")
        cos, sin = mrope_phases(positions, inv, section)
        probe = torch.zeros(1, positions.shape[1], 64)
        ref_cos, ref_sin = hf_bundle["rotary"](probe, positions.unsqueeze(1), layer)
        _same(layer + " cos", cos, ref_cos[0, :, :32].float())
        _same(layer + " sin", sin, ref_sin[0, :, :32].float())


def test_sequence_slice_keeps_three_axes():
    positions = torch.arange(30).reshape(3, 10)
    assert take_tokens(positions, 4).shape == (3, 4)
    assert span_tokens(positions, 2, 6).shape == (3, 4)
    flat = torch.arange(10)
    assert take_tokens(flat, 4).shape == (4,)
    with pytest.raises(ValueError, match="mrope_section"):
        mrope_phases(positions, torch.ones(4), None)


def test_tall_cap_has_more_slots_than_the_square():
    assert max_slots_under_cap() == 1364
    assert max_slots_under_cap() > slot_count(64, 64)


def test_merge_rejects_a_short_feature():
    ids = torch.tensor([1, IMAGE_PAD_ID, IMAGE_PAD_ID, 2])
    base = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="do not match"):
        merge_image_embeddings(base, ids, torch.ones(1, 3))
    assert IMAGE_LIMIT_REJECTION.startswith("At most 1 image")
