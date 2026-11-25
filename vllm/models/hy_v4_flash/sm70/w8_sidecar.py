# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read the Hy4 shared-expert W8 codes stored inside the checkpoint.

The sidecar holds logical, unsharded int8 codes and FP16 group scales for
every shared-expert projection, named after the checkpoint's BF16 tensors:

  model.layers.L.mlp.shared_experts.{gate,up,down}_proj.w8_codes   int8 [N, K]
  model.layers.L.mlp.shared_experts.{gate,up,down}_proj.w8_scales  fp16 [N, K/32]

It is not listed in model.safetensors.index.json, so the weight loader and
other engines ignore it. Its metadata names the content hash of the
checkpoint it was made for (file names, sizes, safetensors headers and JSON
files, without modification times). Loading only slices and interleaves, so
the packed bytes are a pure function of the file.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

SIDECAR_NAME = "hy4-v100-shared-w8.safetensors"
FORMAT = "hy4-v100-shared-w8"
VERSION = "1"
GROUP_SIZE = 32


def checkpoint_content(checkpoint: Path) -> str:
    """Hash file names, sizes, safetensors headers and JSON files; skip the sidecar."""
    files = {}
    for path in sorted(Path(checkpoint).iterdir()):
        if path.name == SIDECAR_NAME:
            continue
        if path.suffix == ".json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.suffix == ".safetensors":
            with path.open("rb") as reader:
                size = int.from_bytes(reader.read(8), "little")
                if not 0 < size < 100_000_000:
                    raise ValueError(f"Invalid safetensors header length in {path}")
                header = hashlib.sha256(reader.read(size)).hexdigest()
            files[path.name] = [path.stat().st_size, header]
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def layout(q: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Interleave biased uint8 codes [N, K] and FP16 scales [N, G] as w8.pack does."""
    n, k = q.shape
    groups = scales.shape[1]
    if q.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError("W8 layout needs uint8 codes and float16 scales")
    if k % 16 or groups != (k + group_size - 1) // group_size or scales.shape[0] != n:
        raise ValueError(f"Invalid W8 code/scale shapes {tuple(q.shape)} {tuple(scales.shape)}")
    n_pad = (n + 7) // 8 * 8
    if n_pad != n:
        q = torch.nn.functional.pad(q, (0, 0, 0, n_pad - n), value=128)
        scales = torch.cat((scales, scales.new_full((n_pad - n, groups), 2**-24)))
    q = q.reshape(n_pad // 8, 8, k // 16, 16).transpose(1, 2).contiguous().flatten()
    s = scales.reshape(n_pad // 8, 8, groups).transpose(1, 2).contiguous().flatten()
    return torch.cat((q, s.view(torch.uint8))).view(n_pad, k + 2 * groups)


def split_packed(
    packed: torch.Tensor, k: int, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of layout: biased uint8 codes [Npad, K] and FP16 scales [Npad, G]."""
    n_pad = packed.shape[0]
    groups = (k + group_size - 1) // group_size
    if packed.dtype != torch.uint8 or packed.shape[1] != k + 2 * groups:
        raise ValueError(f"Invalid W8 pack {tuple(packed.shape)} for K={k}")
    q, s = packed.flatten().split((n_pad * k, n_pad * groups * 2))
    q = q.reshape(n_pad // 8, k // 16, 8, 16).transpose(1, 2).reshape(n_pad, k)
    s = s.view(torch.float16).reshape(n_pad // 8, groups, 8).transpose(1, 2)
    return q.contiguous(), s.reshape(n_pad, groups).contiguous()


@lru_cache(maxsize=4)
def _handle(path: str):
    from safetensors import safe_open

    return safe_open(path, framework="pt", device="cpu")


def rank_weight(sidecar: Path, prefix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Packed W8 bytes for one vLLM shared-expert projection on one TP rank."""
    root, projection = prefix.rsplit(".", 1)
    handle = _handle(str(sidecar))

    def tensors(name):
        return (handle.get_slice(f"{root}.{name}.w8_codes"),
                handle.get_slice(f"{root}.{name}.w8_scales"))

    if projection == "gate_up_proj":
        # MergedColumnParallelLinear: each rank holds its row slice of gate, then of up.
        parts = []
        for name in ("gate_proj", "up_proj"):
            codes, scales = tensors(name)
            rows = codes.get_shape()[0]
            if rows % tp_size:
                raise ValueError(f"{root}.{name} has {rows} rows for TP {tp_size}")
            lo, hi = tp_rank * rows // tp_size, (tp_rank + 1) * rows // tp_size
            parts.append((codes[lo:hi], scales[lo:hi]))
        q = torch.cat([codes for codes, _ in parts])
        s = torch.cat([scales for _, scales in parts])
    elif projection == "down_proj":
        # RowParallelLinear: each rank holds a group-aligned column slice.
        codes, scales = tensors("down_proj")
        cols = codes.get_shape()[1]
        if cols % (tp_size * GROUP_SIZE):
            raise ValueError(f"{root}.down_proj K={cols} is not group-aligned for TP {tp_size}")
        width = cols // tp_size
        q = codes[:, tp_rank * width:(tp_rank + 1) * width]
        g = width // GROUP_SIZE
        s = scales[:, tp_rank * g:(tp_rank + 1) * g]
    else:
        raise ValueError(f"No shared-expert W8 sidecar layout for {prefix}")
    biased = q.contiguous().view(torch.uint8) ^ 0x80
    return layout(biased, s.contiguous())


@lru_cache(maxsize=4)
def verified_sidecar(checkpoint: str) -> Path | None:
    """The checkpoint's sidecar after checking its format and checkpoint content hash."""
    root = Path(checkpoint)
    path = root / SIDECAR_NAME
    if not path.is_file():
        return None
    metadata = _handle(str(path)).metadata() or {}
    if (
        metadata.get("format") != FORMAT
        or metadata.get("version") != VERSION
        or metadata.get("group_size") != str(GROUP_SIZE)
    ):
        raise ValueError(f"Unsupported Hy4 W8 sidecar format in {path}: {metadata}")
    content = checkpoint_content(root)
    if metadata.get("checkpoint_content") != content:
        raise ValueError(
            f"{path} was made for checkpoint content "
            f"{metadata.get('checkpoint_content')}, but this checkpoint is {content}. "
            "Regenerate it with tools/hy4/qpack_shared_sidecar.py."
        )
    return path


def linear_weight(layer, group_size: int) -> torch.Tensor | None:
    """Packed W8 weight for a shared-expert projection, or None without a sidecar."""
    from vllm.config.vllm import get_current_vllm_config

    sidecar = verified_sidecar(get_current_vllm_config().model_config.model)
    if sidecar is None:
        return None
    if group_size != GROUP_SIZE:
        raise ValueError(f"The Hy4 W8 sidecar requires group size {GROUP_SIZE}.")
    packed = rank_weight(sidecar, layer.prefix, layer.tp_rank, layer.tp_size)
    n, k = layer.weight.shape
    shape = ((n + 7) // 8 * 8, k + 2 * ((k + group_size - 1) // group_size))
    if packed.shape != shape:
        raise ValueError(f"Invalid sidecar Hy4 W8 layout for {layer.prefix}: {tuple(packed.shape)}")
    return packed.to(device=layer.weight.device)
