# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and read an offline Hy4 shared-expert W8 pack."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

from .w8_sidecar import SIDECAR_NAME

GROUP_SIZE = 32

# The original fast3 calibration predates rank hashes. Bind its exact
# checkpoint and build identity to the published payloads so it can be used
# on compatible descendants without modifying the shared cache.
_CANONICAL_CHECKPOINT = (
    "c413bb1f9d47499cff5283687e415f164a8030ee9b73891147503e8c6ab7aacf"
)
_CANONICAL_BUILD = (
    "a9c1293a0f25f21ba03a2677544d1d5153dc299cf88fb584184e54924dec8daa"
)
_CANONICAL_RANK_SHA256 = {
    "rank-0": "3a6f09f529d0ed57ac78e98909fc06de7d749f754c3e7c90bb1f0fd0477f60ca",
    "rank-1": "7d5530719ae809f73621c4c9172ebb579cb16817fb0e0c0e3af89f7b265aa5e1",
    "rank-2": "a08e9e35a1e3b81654425ac62e2f7c07c523dfc741683f691b1be74bbf1cf967",
    "rank-3": "661f6f138dfad3fbfe284d59cebbf01163cae6679d65bdc7cd209f2facb811ae",
}


def pack_identity(checkpoint: Path) -> tuple[str, str]:
    files = {}
    for path in sorted(checkpoint.iterdir()):
        if path.name == SIDECAR_NAME:
            continue  # Keep cache keys stable when a checkpoint gains the sidecar.
        if path.suffix == ".json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.suffix == ".safetensors":
            metadata = path.stat()
            with path.open("rb") as reader:
                size = int.from_bytes(reader.read(8), "little")
                if not 0 < size < 100_000_000:
                    raise ValueError(f"Invalid safetensors header length in {path}")
                header = hashlib.sha256(reader.read(size)).hexdigest()
            files[path.name] = [metadata.st_size, metadata.st_mtime_ns, header]
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Hy4 W8 cached packs require a local safetensors checkpoint.")
    checkpoint_key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    model_root = Path(__file__).resolve().parents[1]
    vllm_root = model_root.parents[1]
    build_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("w8.py"),
        model_root / "profile.py",
        *sorted(vllm_root.glob("_sm70_hy4*.so")),
    )
    digest = hashlib.sha256()
    for path in build_files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return checkpoint_key, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pack_manifest(root: Path, manifest: dict) -> None:
    """Record rank hashes before publishing a completed staged pack."""
    ranks = [
        {**entry, "sha256": _file_sha256(root / f"{entry['rank']}.pt")}
        for entry in manifest["ranks"]
    ]
    (root / "manifest.json").write_text(
        json.dumps({**manifest, "ranks": ranks}, indent=2) + "\n"
    )


@lru_cache(maxsize=4)
def _rank_pack(cache: str, checkpoint: str, rank: int) -> dict[str, torch.Tensor]:
    checkpoint_key, build_key = pack_identity(Path(checkpoint))
    root = Path(cache)
    manifest = json.loads((root / "manifest.json").read_text())
    canonical = (
        checkpoint_key == _CANONICAL_CHECKPOINT
        and manifest["build"] == _CANONICAL_BUILD
    )
    if (
        manifest["checkpoint"] != checkpoint_key
        or manifest["group_size"] != GROUP_SIZE
        or (manifest["build"] != build_key and not canonical)
    ):
        raise ValueError(f"Hy4 W8 cache does not match checkpoint, build or group size: {root}")
    rank_name = f"rank-{rank}"
    path = root / f"{rank_name}.pt"
    entries = [entry for entry in manifest.get("ranks", ()) if entry["rank"] == rank_name]
    expected = _CANONICAL_RANK_SHA256.get(rank_name) if canonical else (
        entries[0].get("sha256") if len(entries) == 1 else None
    )
    if len(entries) != 1 or not expected:
        raise ValueError(f"Missing or ambiguous Hy4 W8 cache SHA-256 for {path}")
    if _file_sha256(path) != expected:
        raise ValueError(f"Hy4 W8 cache SHA-256 mismatch for {path}")
    return torch.load(path, weights_only=True, mmap=True, map_location="cpu")


def cached_linear_weight(layer, cache: str, group_size: int) -> torch.Tensor:
    from vllm.config.vllm import get_current_vllm_config

    if group_size != GROUP_SIZE:
        raise ValueError(f"The Hy4 W8 packed cache requires group size {GROUP_SIZE}.")
    checkpoint = get_current_vllm_config().model_config.model
    packed = _rank_pack(cache, checkpoint, layer.tp_rank)[layer.prefix]
    n, k = layer.weight.shape
    shape = ((n + 7) // 8 * 8, k + 2 * ((k + group_size - 1) // group_size))
    if packed.dtype != torch.uint8 or packed.shape != shape or not packed.is_contiguous():
        raise ValueError(f"Invalid cached Hy4 W8 layout for {layer.prefix}.")
    return packed.to(device=layer.weight.device)
