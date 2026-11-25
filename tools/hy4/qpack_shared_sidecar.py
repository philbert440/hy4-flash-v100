# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store the Hy4-flash shared-expert W8 packs inside a checkpoint.

  make    export a verified per-rank cache (HY4_W8_SHARED_GPTQ_CACHE layout) to
          logical int8 codes and FP16 scales, and write
          MODEL/hy4-v100-shared-w8.safetensors. MODEL may be any checkpoint
          whose shared-expert BF16 tensors are byte-identical to those of the
          checkpoint the cache was calibrated on (--cache-model). The sidecar
          records MODEL's content hash, so it must be made after every other
          file in MODEL is final.
  verify  check the sidecar against MODEL's content hash and compare every
          rank's loaded bytes with the cache's rank files.

Serve with HY4_W8_SHARED_SIDECAR=1; HY4_W8_SHARED_GPTQ_CACHE is then unused.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.models.hy_v4_flash.sm70 import w8_cache, w8_sidecar

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
METHOD = "w8b-gptq-g32-int8-sym"
MAX_REL_ERR = 1e-3


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def tensor(self, key: str) -> torch.Tensor:
        shard = self.index[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.root / shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(key)

    def shared_layers(self) -> list[int]:
        suffix = ".mlp.shared_experts.down_proj.weight"
        return sorted(int(k.split(".")[2]) for k in self.index
                      if k.startswith("model.layers.") and k.endswith(suffix))


def save_deterministic(tensors: dict[str, torch.Tensor], path: Path, metadata: dict) -> None:
    """safetensors with sorted names and metadata, so equal inputs give equal file bytes.

    safetensors.torch.save_file serializes __metadata__ from a hash map, whose
    order changes from run to run.
    """
    dtypes = {torch.int8: "I8", torch.float16: "F16"}
    header, offset = {"__metadata__": dict(sorted(metadata.items()))}, 0
    for name in sorted(tensors):
        t = tensors[name]
        size = t.numel() * t.element_size()
        header[name] = {"dtype": dtypes[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for name in sorted(tensors):
            f.write(tensors[name].contiguous().view(torch.uint8).numpy().tobytes())
        f.flush()
        os.fsync(f.fileno())


def tensor_digest(digest, name: str, tensor: torch.Tensor) -> None:
    digest.update(f"{name}|{tensor.dtype}|{list(tensor.shape)}|".encode())
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def shared_source(model: Checkpoint, reference: Checkpoint | None) -> str:
    """Hash MODEL's shared-expert BF16 tensors; require equality with the reference's."""
    digest = hashlib.sha256()
    for layer in model.shared_layers():
        for projection in PROJECTIONS:
            name = f"model.layers.{layer}.mlp.shared_experts.{projection}.weight"
            tensor = model.tensor(name)
            if reference is not None and not torch.equal(
                tensor.view(torch.uint8), reference.tensor(name).view(torch.uint8)
            ):
                sys.exit(f"{name} differs from the checkpoint the cache was calibrated on")
            tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def load_cache(cache: Path, cache_model: Path) -> tuple[dict, list[dict]]:
    """Rank packs after checking the cache's checkpoint binding and rank SHA-256s."""
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["group_size"] != w8_sidecar.GROUP_SIZE:
        sys.exit(f"{cache} has group size {manifest['group_size']}")
    content = manifest.get("checkpoint_content")
    if content is not None:
        if w8_sidecar.checkpoint_content(cache_model) != content:
            sys.exit(f"{cache} was not made for {cache_model} (checkpoint content)")
    elif w8_cache.pack_identity(cache_model)[0] != manifest["checkpoint"]:
        sys.exit(f"{cache} was not made for {cache_model} (checkpoint key)")
    canonical = manifest["checkpoint"] == w8_cache._CANONICAL_CHECKPOINT
    ranks, hashes = [], {}
    for index, entry in enumerate(sorted(manifest["ranks"], key=lambda e: e["rank"])):
        assert entry["rank"] == f"rank-{index}", manifest["ranks"]
        expected = entry.get("sha256") or (
            w8_cache._CANONICAL_RANK_SHA256.get(entry["rank"]) if canonical else None)
        path = cache / f"{entry['rank']}.pt"
        if not expected or w8_cache._file_sha256(path) != expected:
            sys.exit(f"Missing or mismatched SHA-256 for {path}")
        hashes[entry["rank"]] = expected
        ranks.append(torch.load(path, weights_only=True, mmap=True, map_location="cpu"))
    return {"manifest": manifest, "rank_sha256": hashes}, ranks


def logical(ranks: list[dict], layer: int, shapes: dict[str, tuple[int, int]]) -> dict:
    """Undo TP sharding and the W8 interleave for one layer: name -> (biased q, scales)."""
    tp = len(ranks)
    root = f"model.layers.{layer}.mlp.shared_experts"
    gate_n, k = shapes["gate_proj"]
    gate, up = [], []
    for pack in ranks:
        q, s = w8_sidecar.split_packed(pack[f"{root}.gate_up_proj"], k)
        rows = gate_n // tp
        assert q.shape[0] == 2 * rows
        gate.append((q[:rows], s[:rows]))
        up.append((q[rows:], s[rows:]))
    n, down_k = shapes["down_proj"]
    down = [w8_sidecar.split_packed(pack[f"{root}.down_proj"], down_k // tp) for pack in ranks]
    out = {}
    for name, parts, dim in (("gate_proj", gate, 0), ("up_proj", up, 0), ("down_proj", down, 1)):
        out[name] = (torch.cat([q for q, _ in parts], dim), torch.cat([s for _, s in parts], dim))
        assert tuple(out[name][0].shape) == shapes[name], (name, out[name][0].shape)
    return out


def dequant_rel_err(q: torch.Tensor, scales: torch.Tensor, weight: torch.Tensor) -> float:
    w = weight.float()
    d = (q.float() - 128) * scales.float().repeat_interleave(w8_sidecar.GROUP_SIZE, 1)
    return float((d - w).pow(2).sum() / w.pow(2).sum())


def make(args) -> None:
    model = Checkpoint(args.model)
    reference = None if args.cache_model.resolve() == args.model.resolve() else Checkpoint(args.cache_model)
    identity, ranks = load_cache(args.cache, args.cache_model)
    source = shared_source(model, reference)
    tensors, worst = {}, 0.0
    layers = model.shared_layers()
    if sorted(ranks[0]) != sorted(
        f"model.layers.{l}.mlp.shared_experts.{p}" for l in layers for p in ("gate_up_proj", "down_proj")
    ):
        sys.exit("The cache and the checkpoint name different shared-expert layers")
    for layer in layers:
        root = f"model.layers.{layer}.mlp.shared_experts"
        weights = {p: model.tensor(f"{root}.{p}.weight") for p in PROJECTIONS}
        parts = logical(ranks, layer, {p: tuple(w.shape) for p, w in weights.items()})
        for projection, (q, scales) in parts.items():
            err = dequant_rel_err(q, scales, weights[projection])
            worst = max(worst, err)
            if not err < MAX_REL_ERR:
                sys.exit(f"{root}.{projection}: dequantized codes differ from the BF16 weight "
                         f"(relative error power {err:.3e})")
            tensors[f"{root}.{projection}.w8_codes"] = (q ^ 0x80).view(torch.int8).contiguous()
            tensors[f"{root}.{projection}.w8_scales"] = scales.contiguous()
    metadata = {
        "format": w8_sidecar.FORMAT,
        "version": w8_sidecar.VERSION,
        "group_size": str(w8_sidecar.GROUP_SIZE),
        "method": METHOD,
        "checkpoint_content": w8_sidecar.checkpoint_content(args.model),
        "shared_source_sha256": source,
        "cache_checkpoint": identity["manifest"]["checkpoint"],
        "cache_checkpoint_content": identity["manifest"].get("checkpoint_content", ""),
        "cache_rank_sha256": json.dumps(identity["rank_sha256"], sort_keys=True),
        "cache_tp_size": str(len(ranks)),
        "layers": json.dumps(layers),
    }
    out = args.model / w8_sidecar.SIDECAR_NAME
    tmp = out.with_name(f".{out.name}.{os.getpid()}")
    save_deterministic(tensors, tmp, metadata)
    os.chmod(tmp, 0o644)
    os.replace(tmp, out)
    print(json.dumps({"sidecar": str(out), "bytes": out.stat().st_size,
                      "sha256": w8_cache._file_sha256(out), "tensors": len(tensors),
                      "max_dequant_rel_err": worst, **metadata}, indent=1))
    args.sidecar = out
    verify(args, ranks)


def verify(args, ranks=None) -> None:
    w8_sidecar.verified_sidecar.cache_clear()
    w8_sidecar._handle.cache_clear()
    sidecar = w8_sidecar.verified_sidecar(str(args.model))
    if sidecar is None:
        sys.exit(f"{args.model} has no {w8_sidecar.SIDECAR_NAME}")
    if ranks is None:
        ranks = load_cache(args.cache, args.cache_model)[1]
    compared = 0
    for rank, pack in enumerate(ranks):
        for prefix, expected in sorted(pack.items()):
            got = w8_sidecar.rank_weight(sidecar, prefix, rank, len(ranks))
            if got.shape != expected.shape or not torch.equal(got, expected):
                sys.exit(f"MISMATCH rank-{rank} {prefix}")
            compared += expected.numel()
    print(f"G2 OK: {sum(len(p) for p in ranks)} rank tensors, {compared} bytes byte-equal to {args.cache}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, run in (("make", make), ("verify", verify)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", type=Path, required=True, help="checkpoint directory to hold the sidecar")
        cmd.add_argument("--cache", type=Path, required=True, help="verified shared-expert W8 cache directory")
        cmd.add_argument("--cache-model", type=Path, required=True,
                         help="checkpoint directory the cache was calibrated on")
        cmd.set_defaults(run=run)
    args = parser.parse_args()
    with torch.inference_mode():
        args.run(args)


if __name__ == "__main__":
    main()
