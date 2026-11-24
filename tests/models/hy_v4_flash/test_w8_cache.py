# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.models.hy_v4_flash.sm70 import w8_cache


def test_rank_manifest_rejects_replaced_projection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(w8_cache, "pack_identity", lambda _: ("checkpoint", "build"))
    manifest = {
        "checkpoint": "checkpoint",
        "build": "build",
        "group_size": w8_cache.GROUP_SIZE,
        "ranks": [{"rank": "rank-0"}],
    }
    rank_path = cache / "rank-0.pt"
    projection = torch.ones((8, 34), dtype=torch.uint8)
    torch.save({"projection": projection}, rank_path)
    size = rank_path.stat().st_size

    w8_cache.write_pack_manifest(cache, manifest)
    pack = w8_cache._rank_pack(str(cache), str(checkpoint), 0)
    assert torch.equal(pack["projection"], projection)
    w8_cache._rank_pack.cache_clear()
    del pack

    torch.save({"projection": torch.zeros_like(projection)}, rank_path)
    assert rank_path.stat().st_size == size
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        w8_cache._rank_pack(str(cache), str(checkpoint), 0)

    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing or ambiguous Hy4 W8 cache SHA-256"):
        w8_cache._rank_pack(str(cache), str(checkpoint), 0)
