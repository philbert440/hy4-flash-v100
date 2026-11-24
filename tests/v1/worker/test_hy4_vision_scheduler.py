# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Image requests through the CPU scheduler and the runner's position copy."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.worker.test_hy4_engram_scheduler import make_runner
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.models.hy_v4_flash.sm70.image_process import (
    IMAGE_END_ID,
    IMAGE_PAD_ID,
    IMAGE_START_ID,
    slot_count,
)
from vllm.models.hy_v4_flash.sm70.vision_rope import rope_index

pytestmark = pytest.mark.skip_global_cleanup


class _Grid:
    def __init__(self, data):
        self.data = data


def _feature(identifier: str, offset: int, slots: int, grid: tuple[int, int, int]):
    length = slots + 2
    mask = torch.zeros(length, dtype=torch.bool)
    mask[1 : 1 + slots] = True
    return MultiModalFeatureSpec(
        data={"image_grid_thw": _Grid(torch.tensor([grid], dtype=torch.long))},
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=mask),
    )


def _request(req_id: str, prompt: list[int], features: list, max_tokens: int = 1):
    create_requests(0)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
        pooling_params=None,
        mm_features=features,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _scheduler(*, prefix: bool, threshold: int, seqs: int = 2):
    create_requests(0)
    scheduler = create_scheduler(
        max_num_seqs=seqs,
        max_num_batched_tokens=128,
        max_model_len=96,
        long_prefill_token_threshold=threshold,
        enable_prefix_caching=prefix,
        num_blocks=64,
        block_size=16,
    )
    # OPT has no encoder budget. Install the image budget the vision server
    # would have computed from get_mm_max_tokens_per_item.
    scheduler.max_num_encoder_input_tokens = 128
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=128)
    return scheduler


def _advance(scheduler, output, requests: dict[str, Request]) -> None:
    ids = list(output.num_scheduled_tokens)
    sampled = []
    for req_id in ids:
        request = requests[req_id]
        tokens = [7] if request.num_computed_tokens >= request.num_tokens else []
        sampled.append(tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _image_prompt(offset: int, slots: int, tail: int) -> list[int]:
    prompt = list(range(1, offset + 1))
    prompt.append(IMAGE_START_ID)
    prompt.extend([IMAGE_PAD_ID] * slots)
    prompt.append(IMAGE_END_ID)
    prompt.extend(range(1000, 1000 + tail))
    return prompt


def test_chunked_prefill_stops_inside_the_image():
    slots = 6
    offset = 4
    prompt = _image_prompt(offset, slots, 8)
    image_end = offset + slots + 2
    feature = _feature("chunk-image", offset, slots, (1, 4, 4))
    request = _request("image", prompt, [feature], max_tokens=1)
    # The chunk cap is applied only while another request is waiting.
    other = _request("text", list(range(30, 34)), [], max_tokens=1)
    scheduler = _scheduler(prefix=False, threshold=8, seqs=2)
    scheduler.add_request(request)
    scheduler.add_request(other)
    requests = {item.request_id: item for item in (request, other)}

    first = scheduler.schedule()
    assert first.num_scheduled_tokens["image"] == 8
    assert first.scheduled_encoder_inputs["image"] == [0]
    assert request.num_computed_tokens == 8
    assert offset < 8 < image_end
    _advance(scheduler, first, requests)

    second = scheduler.schedule()
    resumed = request.num_computed_tokens - second.num_scheduled_tokens["image"]
    assert resumed == 8
    assert resumed < image_end


def test_prefix_cache_hits_the_same_image_and_misses_a_different_one():
    offset = 16
    slots = 14
    grid = (1, 4, 12)
    assert slot_count(grid[1], grid[2]) == slots
    prompt = _image_prompt(offset, slots, 16)
    assert len(prompt) == 48
    warm = _request("warm", prompt, [_feature("pic-a", offset, slots, grid)])
    same = _request("same", list(prompt), [_feature("pic-a", offset, slots, grid)])
    other = _request(
        "other", list(prompt), [_feature("pic-b", offset, slots, grid)]
    )
    scheduler = _scheduler(prefix=True, threshold=0, seqs=1)
    requests = {item.request_id: item for item in (warm, same, other)}
    scheduler.add_request(warm)
    for _ in range(8):
        if warm.request_id not in scheduler.requests:
            break
        _advance(scheduler, scheduler.schedule(), requests)
    assert warm.request_id not in scheduler.requests

    scheduler.add_request(same)
    same_out = scheduler.schedule()
    same_cursor = same.num_computed_tokens - same_out.num_scheduled_tokens["same"]
    # The image occupies tokens [16, 32). Same identifier reuses through
    # that block. A different identifier reuses only the text block before it.
    assert same_cursor == 32, (
        f"cursor={same_cursor} hashes={len(same.block_hashes)} "
        f"computed={same.num_computed_tokens} "
        f"scheduled={same_out.num_scheduled_tokens['same']}"
    )
    _advance(scheduler, same_out, requests)
    for _ in range(4):
        if same.request_id not in scheduler.requests:
            break
        _advance(scheduler, scheduler.schedule(), requests)

    scheduler.add_request(other)
    other_out = scheduler.schedule()
    other_cursor = other.num_computed_tokens - other_out.num_scheduled_tokens["other"]
    assert other_cursor == 16


def test_mixed_batch_and_runner_keep_text_axes_equal(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    slots = 6
    offset = 4
    image_prompt = _image_prompt(offset, slots, 8)
    text_prompt = list(range(30, 42))
    image = _request(
        "image", image_prompt, [_feature("mixed-image", offset, slots, (1, 4, 4))]
    )
    text = _request("text", text_prompt, [])
    scheduler = _scheduler(prefix=False, threshold=0, seqs=2)
    scheduler.add_request(image)
    scheduler.add_request(text)
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"image", "text"}

    from vllm.models.hy_v4_flash.nvidia.vl_model import (
        HYV4FlashVLForConditionalGeneration,
    )

    runner = make_runner(monkeypatch)
    runner.uses_mrope = True
    runner.uses_flash_engram = False
    runner.input_batch = type(runner.input_batch)(
        max_num_reqs=4,
        max_model_len=96,
        max_num_batched_tokens=128,
        device=runner.device,
        pin_memory=False,
        vocab_size=200000,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    model = HYV4FlashVLForConditionalGeneration.__new__(
        HYV4FlashVLForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    runner.get_model = lambda: model
    buffer = torch.full((3, 128), -1, dtype=torch.long)
    runner.mrope_positions = SimpleNamespace(cpu=buffer, gpu=buffer, np=buffer.numpy())
    runner._update_states(output)
    runner._calc_mrope_positions(output)

    image_state = runner.requests["image"]
    expected, _delta = rope_index(
        torch.tensor(image_prompt),
        torch.tensor(
            [0] * offset + [0] + [1] * slots + [0] + [0] * (len(image_prompt) - offset - slots - 2)
        ),
        torch.tensor([[1, 4, 4]]),
    )
    assert torch.equal(image_state.mrope_positions, expected)
    pads = slice(offset + 1, offset + 1 + slots)
    assert not torch.equal(
        image_state.mrope_positions[1, pads], image_state.mrope_positions[2, pads]
    )
    text_state = runner.requests["text"]
    assert torch.equal(text_state.mrope_positions[0], text_state.mrope_positions[1])
    assert torch.equal(text_state.mrope_positions[0], text_state.mrope_positions[2])
    scheduled = output.num_scheduled_tokens["image"]
    assert torch.equal(buffer[:, :scheduled], expected[:, :scheduled])
