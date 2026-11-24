# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the Hy4 tower with the pinned HF vision tower. Not part of pytest.

Bounds, fixed before this run: FP32 max abs <= 1e-3 and
max abs / max(abs(reference)) <= 1e-3. FP16 max abs <= 2e-2 against that
FP32 reference. SIZES includes the 1,048,576 pixel cap.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from vllm.models.hy_v4_flash.nvidia.vision import Hy4VisionTransformer
from vllm.models.hy_v4_flash.sm70.image_process import process_pil

ALIASES = (
    ("perceive.norm", "patch_merger.before_rms"),
    ("perceive.proj.0", "patch_merger.proj_conv"),
    ("perceive.proj.2", "patch_merger.proj_out"),
    ("perceive.image_newline", "patch_merger.image_newline"),
    ("input_layernorm", "layer_norm1"),
    ("post_attention_layernorm", "layer_norm2"),
    ("mlp.dense_h_to_4h", "mlp.fc1"),
    ("mlp.dense_4h_to_h", "mlp.fc2"),
)
SIZES = ((32, 32), (173, 113), (320, 256), (1024, 1024))
FP32_ABS = 1e-3
FP32_REL = 1e-3
FP16_ABS = 2e-2


def _alias(name: str) -> str:
    for original, replacement in ALIASES:
        name = name.replace(original, replacement)
    return name


def _load_vit(checkpoint: Path) -> dict[str, torch.Tensor]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = {
        index["weight_map"][key]
        for key in index["weight_map"]
        if key.startswith("vit.")
    }
    weights = {}
    for shard in sorted(shards):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith("vit."):
                    weights[name[4:]] = handle.get_tensor(name)
    return weights


def _picture(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    picture = np.empty((height, width, 3), dtype=np.uint8)
    picture[..., 0] = x.astype(np.uint8)
    picture[..., 1] = y[:, None].astype(np.uint8)
    picture[..., 2] = ((x + y[:, None]) * 0.5).astype(np.uint8)
    return Image.fromarray(picture, "RGB")



def _matmul_work(height: int, width: int, config) -> None:
    """Count FP32 matmul FLOPs. One multiply-add is two FLOPs."""
    patch = config.patch_size
    merge = config.spatial_merge_size
    patches_h = height // patch
    patches_w = width // patch
    patches = patches_h * patches_w
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layers = config.num_hidden_layers
    linear = layers * 2 * patches * (4 * hidden * hidden + 2 * hidden * intermediate)
    attention = layers * 4 * patches * patches * hidden
    merged_h = patches_h // merge
    merged_w = patches_w // merge
    stem = 2 * patches_h * patches_w * config.num_channels * hidden * patch * patch
    project_in = (
        2 * merged_h * merged_w * hidden * (hidden * merge * merge) * merge * merge
    )
    project_out = (
        2 * merged_h * merged_w * (hidden * merge * merge) * config.out_hidden_size
    )
    total = linear + attention + stem + project_in + project_out
    print(
        f"fp32_matmul_flops size={width}x{height} patches={patches} "
        f"linear={linear} attention={attention} stem={stem} "
        f"project_in={project_in} project_out={project_out} total={total}",
        flush=True,
    )


def _error(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    gap = (actual.float() - reference.float()).abs()
    max_abs = float(gap.max())
    scale = float(reference.float().abs().max())
    relative = max_abs / scale if scale else 0.0
    return max_abs, relative


def main() -> None:
    torch.set_num_threads(4)
    checkpoint = Path(os.environ["HY4_FLASH_DIR"])
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLVisionModel

    probe = torch.linspace(-2, 2, 9)
    if not torch.equal(ACT2FN["gelu"](probe), torch.nn.functional.gelu(probe)):
        raise SystemExit("GELU does not match the pinned activation")

    weights = _load_vit(checkpoint)
    print("vit_tensors", len(weights), flush=True)
    if len(weights) != 441:
        raise SystemExit(f"expected 441 vit tensors, found {len(weights)}")

    config = AutoConfig.from_pretrained(checkpoint).vision_config
    config._attn_implementation = "sdpa"
    fp32 = Hy4VisionTransformer(config, dtype=torch.float32, tp_size=1, tp_rank=0).eval()
    fp16 = Hy4VisionTransformer(config, dtype=torch.float16, tp_size=1, tp_rank=0).eval()
    # Copy while the checkpoint tensors are still BF16. The HF assign shares storage.
    print("loaded_fp32", len(fp32.load_weights(weights.items())), flush=True)
    print("loaded_fp16", len(fp16.load_weights(weights.items())), flush=True)
    reference = HYV4VLVisionModel(config).eval()
    aliased = {_alias(name): tensor for name, tensor in weights.items()}
    print(reference.load_state_dict(aliased, strict=True, assign=True), flush=True)
    reference.to(dtype=torch.float32)
    print("hf_attn", reference.config._attn_implementation, flush=True)
    del weights, aliased

    failed = False
    for width, height in SIZES:
        started = time.perf_counter()
        record = process_pil(_picture(width, height))
        pixels = record.pixel_values
        grid = record.image_grid_thw
        processed_h = int(grid[0, 1]) * config.patch_size
        processed_w = int(grid[0, 2]) * config.patch_size
        _matmul_work(processed_h, processed_w, config)
        with torch.inference_mode():
            hf = reference(pixels, grid).pooler_output[0]
            ours32 = fp32(pixels, grid)
            ours16 = fp16(pixels, grid)
        abs32, rel32 = _error(ours32, hf)
        abs16, rel16 = _error(ours16, hf)
        ok32 = abs32 <= FP32_ABS and rel32 <= FP32_REL and torch.isfinite(ours32).all()
        ok16 = abs16 <= FP16_ABS and torch.isfinite(ours16).all()
        failed = failed or not (ok32 and ok16)
        elapsed = time.perf_counter() - started
        print(
            f"size={width}x{height} grid={grid.tolist()} slots={tuple(ours32.shape)} "
            f"fp32_abs={abs32:.6e} fp32_rel={rel32:.6e} fp16_abs={abs16:.6e} "
            f"fp16_rel={rel16:.6e} seconds={elapsed:.3f} ok={bool(ok32 and ok16)}",
            flush=True,
        )

    full_bytes = fp16.rank_parameter_bytes()
    del reference, fp32
    rank0 = Hy4VisionTransformer(config, dtype=torch.float16, tp_size=4, tp_rank=0).eval()
    fresh = _load_vit(checkpoint)
    print("loaded_rank0", len(rank0.load_weights(fresh.items())), flush=True)
    rank_bytes = rank0.rank_parameter_bytes()
    print("full_fp16_parameter_bytes", full_bytes, flush=True)
    print("rank0_fp16_parameter_bytes", rank_bytes, flush=True)
    print("rank0_mib", round(rank_bytes / (1024 * 1024), 3), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
