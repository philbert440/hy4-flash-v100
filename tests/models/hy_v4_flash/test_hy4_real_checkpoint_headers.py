# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve the real Hy4 checkpoint's safetensors headers against both classes.

No tensor payload is read. Headers supply names, shapes and dtypes. The
text class and the vision class must each claim every key and fill every
parameter. An unknown key still raises.
"""

import json
import os
from collections import Counter
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm.config import ModelConfig, VllmConfig
from vllm.models.hy_v4_flash.nvidia import model as causal_module
from vllm.models.hy_v4_flash.nvidia.engram import FlashEmbedding
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM
from vllm.models.hy_v4_flash.nvidia.vl_model import (
    HYV4FlashVLForConditionalGeneration,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.config import get_config, uses_mrope
from vllm.utils.torch_utils import set_default_torch_dtype

CHECKPOINT = Path(os.environ.get("HY4_FLASH_DIR", "hy4-flash-awq-g32"))
_ENGRAM_WEIGHT = ".engram_layer.multi_head_embedding.embedding.weight"


def _headers(checkpoint: Path) -> dict[str, str]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def _vit_shapes(checkpoint: Path, weight_map: dict[str, str]) -> dict[str, tuple[int, ...]]:
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        if name.startswith("vit."):
            by_file.setdefault(filename, []).append(name)
    shapes: dict[str, tuple[int, ...]] = {}
    for filename, names in by_file.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as reader:
            for name in names:
                shapes[name] = tuple(reader.get_slice(name).get_shape())
    return shapes


def _arm_sm70_awq(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_DISABLE", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(7, 0)
    )
    monkeypatch.setattr(current_platform, "has_device_capability", lambda cap: cap <= 70)


def _model_config(checkpoint: Path, *, architecture: str, limit: int) -> ModelConfig:
    return ModelConfig(
        model=str(checkpoint),
        hf_overrides={"architectures": [architecture]},
        limit_mm_per_prompt={"image": limit, "video": 0},
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
        skip_tokenizer_init=True,
        trust_remote_code=False,
    )


def _build(cpu_vllm, checkpoint: Path, *, vision: bool):
    architecture = (
        "HYV4FlashVLForConditionalGeneration"
        if vision
        else "HYV4FlashForCausalLM"
    )
    cpu_vllm.model_config = _model_config(
        checkpoint, architecture=architecture, limit=1 if vision else 0
    )
    cpu_vllm.quant_config = VllmConfig.get_quantization_config(
        cpu_vllm.model_config, cpu_vllm.load_config
    )
    cpu_vllm.scheduler_config.max_num_batched_tokens = 2048
    with torch.device("meta"), set_default_torch_dtype(torch.float16):
        if vision:
            model = HYV4FlashVLForConditionalGeneration(vllm_config=cpu_vllm)
        else:
            model = HYV4FlashForCausalLM(vllm_config=cpu_vllm)
    return model


def _record_loaders(causal, monkeypatch):
    counts: Counter[tuple[str, str | int | None, int | None]] = Counter()

    def record(name, param, value, dest=None, shard_id=None, expert_id=None):
        del param, value
        shard = dest if isinstance(dest, int) else shard_id
        counts[name, shard, expert_id] += 1

    for name, param in causal.named_parameters():
        param.weight_loader = partial(record, name)

    lookups: list[int] = []

    def record_lookup(value, layer_id):
        del value
        lookups.append(layer_id)

    causal.model.engram_hash.load_lookup = record_lookup
    tables: list[str] = []

    def header_only(self, path, weight_name):
        del self
        with safe_open(path, framework="pt", device="cpu") as reader:
            if reader.get_slice(weight_name).get_dtype() == "BF16":
                raise ValueError(
                    "Engram tables require FP8 weights and BF16 row scales"
                )
            reader.get_slice(weight_name + "_scale_inv")
        tables.append(weight_name)
        tables.append(weight_name + "_scale_inv")

    monkeypatch.setattr(FlashEmbedding, "load_from_safetensors", header_only)
    return counts, lookups, tables


def _expected_visits(params) -> Counter:
    expected: Counter[tuple[str, str | int | None, int | None]] = Counter()
    suffixes = ("qweight", "qzeros", "scales")
    w13_suffixes = tuple(f".w13_{suffix}" for suffix in suffixes)
    w2_suffixes = tuple(f".w2_{suffix}" for suffix in suffixes)
    for name in params:
        if name.endswith(w13_suffixes):
            expected.update(
                (name, shard, expert)
                for expert in range(512)
                for shard in ("w1", "w3")
            )
        elif name.endswith(w2_suffixes):
            expected.update((name, "w2", expert) for expert in range(512))
        elif name.endswith(".gate_up_proj.weight"):
            expected.update((name, shard, None) for shard in (0, 1))
        else:
            expected[name, None, None] = 1
    return expected


def _language(model):
    if isinstance(model, HYV4FlashVLForConditionalGeneration):
        return model.language_model
    return model


def _reconcile(model, weight_map, vit_shapes, monkeypatch) -> None:
    causal = _language(model)
    counts, lookups, tables = _record_loaders(causal, monkeypatch)
    fed_language = 0
    fed_vision = 0

    def weights():
        nonlocal fed_language, fed_vision
        for name in weight_map:
            if model.skip_checkpoint_weight(name):
                continue
            if name.startswith("vit."):
                fed_vision += 1
                yield name, torch.empty(vit_shapes[name], device="meta")
            else:
                # A fresh tensor. The mapper writes shard_id onto the object,
                # and one shared tensor would mark every later key as sharded.
                fed_language += 1
                yield name, torch.empty((), device="meta")

    loaded = model.load_weights(weights())
    assert sorted(lookups) == sorted(index + 1 for index in causal.model.engram_indices)
    expected_tables = []
    for layer_idx in causal.model.engram_indices:
        base = f"model.layers.{layer_idx}{_ENGRAM_WEIGHT}"
        expected_tables.extend((base, base + "_scale_inv"))
    assert sorted(tables) == sorted(expected_tables)
    assert set(expected_tables) <= set(weight_map)
    assert all(model.skip_checkpoint_weight(name) for name in expected_tables)
    assert sum(counts.values()) + len(lookups) == fed_language
    assert counts == _expected_visits(dict(causal.named_parameters()))
    if isinstance(model, HYV4FlashVLForConditionalGeneration):
        assert model._vision_enabled
        assert not any(model.skip_checkpoint_weight(name) for name in vit_shapes)
        vit_loaded = {name[4:] for name in loaded if name.startswith("vit.")}
        assert vit_loaded == {name[4:] for name in vit_shapes}
        assert fed_vision == len(vit_shapes)
    else:
        assert all(model.skip_checkpoint_weight(name) for name in vit_shapes)
        assert fed_vision == 0
        assert loaded == dict(causal.named_parameters()).keys()

    def unknown_language():
        yield "model.layers.0.not_a_real_parameter.weight", torch.empty((), device="meta")

    with pytest.raises((KeyError, ValueError), match="not_a_real_parameter"):
        model.load_weights(unknown_language())

    if isinstance(model, HYV4FlashVLForConditionalGeneration):

        def unknown_vision():
            yield "vit.not_a_real_parameter.weight", torch.empty(1, device="meta")

        with pytest.raises((KeyError, ValueError), match="not_a_real_parameter"):
            model.load_weights(unknown_vision())


def _rejects_stray_tables(model, weight_map, tmp_path, monkeypatch) -> None:
    # A name the predicate drops but the table loader does not read.
    strays = (
        f"model.layres.1{_ENGRAM_WEIGHT}",
        f"model.layers.0{_ENGRAM_WEIGHT}",
        f"model.layers.0{_ENGRAM_WEIGHT}_scale_inv",
    )
    index = tmp_path / "index.json"
    for stray in strays:
        assert model.skip_checkpoint_weight(stray)
        index.write_text(
            json.dumps({"weight_map": {**weight_map, stray: "absent.safetensors"}})
        )
        with monkeypatch.context() as patch:
            patch.setattr(causal_module, "cached_file", lambda *a, **k: str(index))
            with pytest.raises(ValueError, match="Unexpected Engram table"):
                _language(model)._load_engram_tables()


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="real Hy4 checkpoint is not mounted")
def test_real_checkpoint_headers_match_text_and_vision(cpu_vllm, monkeypatch, tmp_path):
    weight_map = _headers(CHECKPOINT)
    assert len(weight_map) == 190814
    assert sum(name.startswith("vit.") for name in weight_map) == 441
    assert sum("dfly" in name for name in weight_map) == 87
    text_config = get_config(str(CHECKPOINT), trust_remote_code=False)
    assert text_config.architectures == ["HYV4FlashForCausalLM"]
    assert not uses_mrope(text_config)

    _arm_sm70_awq(monkeypatch)
    vit_shapes = _vit_shapes(CHECKPOINT, weight_map)
    text = _build(cpu_vllm, CHECKPOINT, vision=False)
    assert not cpu_vllm.model_config.uses_mrope
    _reconcile(text, weight_map, vit_shapes, monkeypatch)
    _rejects_stray_tables(text, weight_map, tmp_path, monkeypatch)
    del text
    # Both classes register the same decoder prefixes on this config.
    cpu_vllm.compilation_config.static_forward_context.clear()
    cpu_vllm.compilation_config.static_all_moe_layers.clear()

    vision_config = _model_config(
        CHECKPOINT,
        architecture="HYV4FlashVLForConditionalGeneration",
        limit=1,
    )
    assert vision_config.uses_mrope
    assert vision_config.hf_config.rope_parameters["mrope_section"] == [12, 10, 10]
    cpu_vllm.model_config = vision_config
    vision = _build(cpu_vllm, CHECKPOINT, vision=True)
    _reconcile(vision, weight_map, vit_shapes, monkeypatch)
    _rejects_stray_tables(vision, weight_map, tmp_path, monkeypatch)
