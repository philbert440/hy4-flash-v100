# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build or rebind the Hy4-flash shared-expert W8 cache.

The fast profiles (fast3 and later) store the shared expert as int8 with
group-32 GPTQ error compensation. The packs are made offline by this tool and
passed to the server with HY4_W8_SHARED_GPTQ_CACHE. A cache is keyed by the
local checkpoint (file names, sizes, modification times and safetensors
headers) and by the build (w8.py, w8_cache.py, profile.py and the
_sm70_hy4*.so extensions), so build it against the checkpoint directory you
serve from and the install you serve with.

  calibrate  run the checkpoint on 4 GPUs over calibration text and write the
             packs to OUT/<checkpoint key>/<build key>/
  rebind     reuse verified packs: after a rebuild that changed the build
             key, or with downloaded packs on a checkpoint with the same
             content (the checkpoint key also covers modification times)
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

CALIBRATION_ENV = {
    "VLLM_HY4_PROFILE": "fast2",
    "HY4_W8_ATTN": "0",
    "HY4_W8_O_PROJ": "0",
    "HY4_W8_LATENT": "0",
    "HY4_W8_SHARED": "0",
    "HY4_W8_MLP": "0",
    "HY4_W8_INDEXER": "0",
    "HY4_W8_UV": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "VLLM_SM70_QUANT_BACKEND": "turbomind",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
SHARED_PROJECTIONS = 82
MIN_TOKENS = 128


def select(tokenizer, path: Path, docs: int, max_tokens: int, seed: int) -> list[list[int]]:
    texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
    texts = list(dict.fromkeys(texts))
    random.Random(seed).shuffle(texts)
    selected = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= MIN_TOKENS:
            selected.append(ids[:max_tokens])
        if len(selected) == docs:
            return selected
    sys.exit(f"{path} has {len(selected)} documents of at least {MIN_TOKENS} tokens; need {docs}")


def checkpoint_content(checkpoint: Path) -> str:
    """Checkpoint identity without modification times, so packs can move between copies."""
    files = {}
    for path in sorted(checkpoint.iterdir()):
        if path.suffix == ".json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.suffix == ".safetensors":
            with path.open("rb") as reader:
                size = int.from_bytes(reader.read(8), "little")
                header = hashlib.sha256(reader.read(size)).hexdigest()
            files[path.name] = [path.stat().st_size, header]
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def calibrate(args) -> None:
    os.environ.update(CALIBRATION_ENV)
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.models.hy_v4_flash.sm70.w8_cache import (
        GROUP_SIZE,
        pack_identity,
        write_pack_manifest,
    )

    checkpoint_key, build_key = pack_identity(args.model)
    cache = args.out / checkpoint_key / build_key
    if (cache / "manifest.json").is_file():
        print(cache)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    selected = select(tokenizer, args.calib, args.docs, args.max_tokens, args.seed)
    staging = cache.with_name(f".{cache.name}.{os.getpid()}")
    staging.mkdir(parents=True)

    llm = LLM(str(args.model), tensor_parallel_size=4, dtype="half",
              kv_cache_dtype="fp8_ds_mla", max_num_batched_tokens=2048,
              max_num_seqs=4, max_model_len=1024, gpu_memory_utilization=0.90,
              enforce_eager=True, enable_chunked_prefill=True,
              enable_prefix_caching=False,
              worker_extension_cls="vllm.models.hy_v4_flash.sm70.w8_calibrate.CalibrationWorker")
    try:
        found = llm.collective_rpc("w8_calibration_begin")
        assert found == [SHARED_PROJECTIONS] * 4, found
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        for start in range(0, len(selected), 8):
            llm.generate([{"prompt_token_ids": ids} for ids in selected[start:start + 8]],
                         params, use_tqdm=False)
        ranks = llm.collective_rpc("w8_calibration_finish", args=(str(staging),))
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)

    write_pack_manifest(staging, {
        "checkpoint": checkpoint_key,
        "checkpoint_content": checkpoint_content(args.model),
        "build": build_key,
        "group_size": GROUP_SIZE,
        "calibration_token_counts": [len(ids) for ids in selected],
        "calibration_token_sha256": [
            hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in selected
        ],
        "ranks": ranks,
    })
    os.replace(staging, cache)
    print(cache)


def rebind(args) -> None:
    from vllm.models.hy_v4_flash.sm70.w8_cache import _file_sha256, pack_identity

    source = args.cache.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    checkpoint_key, build_key = pack_identity(args.model)
    content = manifest.get("checkpoint_content")
    if content is not None:
        if checkpoint_content(args.model) != content:
            sys.exit("The checkpoint content differs from the one the packs were made for "
                     "(file names, sizes, safetensors headers or JSON files). Calibrate again.")
    elif manifest["checkpoint"] != checkpoint_key:
        sys.exit("The checkpoint key differs: the packs belong to another checkpoint "
                 "directory or its files changed. Calibrate again.")
    for entry in manifest["ranks"]:
        path = source / f"{entry['rank']}.pt"
        if "sha256" not in entry:
            sys.exit(f"{source / 'manifest.json'} has no SHA-256 for {entry['rank']}")
        if _file_sha256(path) != entry["sha256"]:
            sys.exit(f"SHA-256 mismatch for {path}")
    dest = args.out / checkpoint_key / build_key
    if dest.resolve() == source:
        print(dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    for entry in manifest["ranks"]:
        link = dest / f"{entry['rank']}.pt"
        if not link.exists():
            link.symlink_to((source / link.name).resolve())
    rebound = {**manifest, "checkpoint": checkpoint_key, "build": build_key,
               "rebound_from": str(source)}
    (dest / "manifest.json").write_text(json.dumps(rebound, indent=2) + "\n")
    print(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--model", type=Path, required=True,
                     help="local checkpoint directory, the one you serve from")
    cal.add_argument("--calib", type=Path, required=True,
                     help='JSONL calibration text, one {"text": ...} per line')
    cal.add_argument("--out", type=Path, required=True, help="cache root")
    cal.add_argument("--docs", type=int, default=24)
    cal.add_argument("--max-tokens", type=int, default=640)
    cal.add_argument("--seed", type=int, default=20261122)
    cal.set_defaults(run=calibrate)
    reb = sub.add_parser("rebind")
    reb.add_argument("--model", type=Path, required=True)
    reb.add_argument("--cache", type=Path, required=True,
                     help="existing or downloaded pack directory with manifest.json")
    reb.add_argument("--out", type=Path, required=True, help="cache root")
    reb.set_defaults(run=rebind)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
