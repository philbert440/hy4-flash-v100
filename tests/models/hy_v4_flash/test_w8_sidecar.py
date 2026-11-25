# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.models.hy_v4_flash.sm70 import w8, w8_cache, w8_sidecar

G = w8_sidecar.GROUP_SIZE


@pytest.mark.parametrize(("n", "k", "group_size"), [(13, 96, 32), (16, 384, 32), (24, 256, 128)])
def test_layout_matches_pack(n, k, group_size):
    weight = torch.randn(n, k, generator=torch.Generator().manual_seed(n * k)).half()
    packed = w8.pack(weight, group_size)
    q, scales = w8_sidecar.split_packed(packed, k, group_size)
    assert torch.equal(w8_sidecar.layout(q[:n], scales[:n], group_size), packed)


def _checkpoint(tmp_path, tp_size=4, inner=256, hidden=128):
    """A checkpoint with one shared expert and its sidecar made from w8.pack codes."""
    gen = torch.Generator().manual_seed(7)
    weights = {
        "gate_proj": torch.randn(inner, hidden, generator=gen).half(),
        "up_proj": torch.randn(inner, hidden, generator=gen).half(),
        "down_proj": torch.randn(hidden, inner, generator=gen).half(),
    }
    root = tmp_path / "checkpoint"
    root.mkdir()
    prefix = "model.layers.1.mlp.shared_experts"
    save_file({f"{prefix}.{p}.weight": w for p, w in weights.items()}, str(root / "base.safetensors"))
    (root / "config.json").write_text("{}")
    tensors = {}
    for name, weight in weights.items():
        q, scales = w8_sidecar.split_packed(w8.pack(weight, G), weight.shape[1])
        tensors[f"{prefix}.{name}.w8_codes"] = (q ^ 0x80).view(torch.int8).contiguous()
        tensors[f"{prefix}.{name}.w8_scales"] = scales
    metadata = {"format": w8_sidecar.FORMAT, "version": w8_sidecar.VERSION,
                "group_size": str(G), "checkpoint_content": w8_sidecar.checkpoint_content(root)}
    save_file(tensors, str(root / w8_sidecar.SIDECAR_NAME), metadata=metadata)
    return root, prefix, weights


def _clear():
    w8_sidecar.verified_sidecar.cache_clear()
    w8_sidecar._handle.cache_clear()


def test_rank_weight_matches_packing_each_rank_shard(tmp_path):
    root, prefix, weights = _checkpoint(tmp_path)
    _clear()
    sidecar = w8_sidecar.verified_sidecar(str(root))
    tp = 4
    rows = weights["gate_proj"].shape[0] // tp
    cols = weights["down_proj"].shape[1] // tp
    for rank in range(tp):
        gate_up = torch.cat([weights[p][rank * rows:(rank + 1) * rows] for p in ("gate_proj", "up_proj")])
        down = weights["down_proj"][:, rank * cols:(rank + 1) * cols]
        for name, expected in (("gate_up_proj", gate_up), ("down_proj", down)):
            got = w8_sidecar.rank_weight(sidecar, f"{prefix}.{name}", rank, tp)
            assert torch.equal(got, w8.pack(expected, G)), (rank, name)


def test_sidecar_is_bound_to_checkpoint_content(tmp_path):
    root, _, _ = _checkpoint(tmp_path)
    _clear()
    assert w8_sidecar.verified_sidecar(str(root)) is not None
    (root / "config.json").write_text(json.dumps({"changed": True}))
    _clear()
    with pytest.raises(ValueError, match="was made for checkpoint content"):
        w8_sidecar.verified_sidecar(str(root))
    (root / w8_sidecar.SIDECAR_NAME).unlink()
    _clear()
    assert w8_sidecar.verified_sidecar(str(root)) is None


def test_cache_key_ignores_sidecar(tmp_path):
    root, _, _ = _checkpoint(tmp_path)
    with_sidecar = w8_cache.pack_identity(root)[0]
    (root / w8_sidecar.SIDECAR_NAME).unlink()
    assert w8_cache.pack_identity(root)[0] == with_sidecar


@pytest.mark.parametrize(("sidecar_weight", "expected"), [("sidecar", "sidecar"), (None, "cache")])
def test_loader_prefers_sidecar_then_cache(monkeypatch, sidecar_weight, expected):
    from vllm.models.hy_v4_flash.sm70 import w8_cache as cache_module

    packs = {"sidecar": torch.full((8, 34), 1, dtype=torch.uint8),
             "cache": torch.full((8, 34), 2, dtype=torch.uint8)}
    monkeypatch.setattr(w8_sidecar, "linear_weight",
                        lambda layer, g: None if sidecar_weight is None else packs["sidecar"])
    monkeypatch.setattr(cache_module, "cached_linear_weight", lambda layer, cache, g: packs["cache"])
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(8, 32, dtype=torch.float16), requires_grad=False)
    w8._W8Linear(G, "/cache", sidecar=True).process_weights_after_loading(layer)
    assert torch.equal(layer.weight, packs[expected])
    assert layer._hy4_w8_shape == (8, 32)


def test_loader_without_sidecar_switch_uses_cache(monkeypatch):
    from vllm.models.hy_v4_flash.sm70 import w8_cache as cache_module

    def fail(layer, g):
        raise AssertionError("sidecar read with the switch off")

    monkeypatch.setattr(w8_sidecar, "linear_weight", fail)
    pack = torch.full((8, 34), 3, dtype=torch.uint8)
    monkeypatch.setattr(cache_module, "cached_linear_weight", lambda layer, cache, g: pack)
    layer = SimpleNamespace(weight=torch.zeros(8, 32, dtype=torch.float16))
    method = w8._W8Linear(G, "/cache")
    monkeypatch.setattr(w8, "_replace_weight", lambda layer, weight: setattr(layer, "weight", weight))
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight, pack)
