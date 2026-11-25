# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-encode the Hy4-flash Engram tables as Tencent's exact FP8 round trip.

Tencent's runtime reads each BF16 Engram row through a per-row E4M3 round trip
(modeling_hy_v4_flash._fp8_round_trip). This tool stores those codes, with the
row scale in BF16, in the release's FP8 layout: each output file copies the
layout checkpoint's safetensors header byte for byte (names, dtypes, shapes,
offsets, metadata), so file sizes and model.safetensors.index.json are
unchanged. Per row: s = max(amax, 1e-12) / 448 (fp32), codes = e4m3(x / s),
weight_scale_inv = bf16(s).

  encode  stream each BF16 table and write OUT/hy4-engram-fp8-NN.safetensors.
          Resumable: progress is kept next to the .partial file.
          --sample-rows N writes only N random rows (runs of 500) into a sparse
          OUT/hy4-engram-fp8-NN.safetensors.sample, for testing.
  verify  gate G1: header and size identical to the layout file, and on random
          rows the normed relative error against Tencent's round trip of the
          BF16 rows is below 1e-12.

The BF16 source (--src) is Tencent's checkpoint as a local directory, or as
HOST:DIR on another machine, read over one ssh stream.
"""

import argparse
import bisect
import json
import os
import random
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch

LAYERS = (1, 11, 21, 31)
TABLE = "model.layers.{}.engram_layer.multi_head_embedding.embedding.weight"
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_ROWS = 131072
RUN = 500
G1_MAX = 1e-12
DTYPES = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}

READER = r"""
import json, os, struct, sys
root = sys.argv[1]
out = sys.stdout.buffer
fds = {}
for message in sys.stdin:
    name, ranges = json.loads(message)
    assert os.path.basename(name) == name and name.endswith((".safetensors", ".json"))
    if name not in fds:
        fds[name] = os.open(os.path.join(root, name), os.O_RDONLY)
    fd = fds[name]
    if ranges is None:
        size = os.fstat(fd).st_size
        out.write(struct.pack("<Q", size))
        ranges = [(0, size)]
    for start, length in ranges:
        while length:
            block = os.pread(fd, min(length, 8 << 20), start)
            if not block:
                raise EOFError(name)
            out.write(block)
            start += len(block)
            length -= len(block)
    out.flush()
"""


class Source:
    """Range reads from Tencent's BF16 checkpoint, local or on HOST:DIR over one ssh stream."""

    def __init__(self, src: str):
        remote = ":" in src and not Path(src).exists()
        self.local = None if remote else Path(src)
        self.process = None
        self.headers = {}
        if remote:
            host, root = src.split(":", 1)
            self.process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "Compression=no",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", host,
                 f"ionice -c2 -n7 nice -n 10 python3 -u -c {shlex.quote(READER)} {shlex.quote(root)}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.index = json.loads(self.whole("model.safetensors.index.json"))["weight_map"]

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait(timeout=30)

    def submit(self, name, ranges):
        self.process.stdin.write(json.dumps([name, ranges]).encode() + b"\n")
        self.process.stdin.flush()

    def receive(self, size) -> bytearray:
        buffer = bytearray(size)
        view, got = memoryview(buffer), 0
        while got < size:
            n = self.process.stdout.readinto(view[got:])
            if not n:
                raise EOFError("short read from the remote source")
            got += n
        return buffer

    def whole(self, name) -> bytes:
        if self.local is not None:
            return (self.local / name).read_bytes()
        self.submit(name, None)
        size = struct.unpack("<Q", self.receive(8))[0]
        return bytes(self.receive(size))

    def tensor_meta(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            if self.local is not None:
                with open(self.local / shard, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    self.headers[shard] = (8 + n, json.loads(f.read(n)))
            else:
                self.submit(shard, [(0, 8)])
                n = struct.unpack("<Q", self.receive(8))[0]
                self.submit(shard, [(8, n)])
                self.headers[shard] = (8 + n, json.loads(self.receive(n)))
        base, header = self.headers[shard]
        meta = header[key]
        return shard, base + meta["data_offsets"][0], meta

    def ranges(self, shard, ranges, submit_only=False):
        """Read [(offset, length)] in order. Remote reads may be submitted ahead."""
        if self.local is not None:
            with open(self.local / shard, "rb") as f:
                return b"".join(os.pread(f.fileno(), n, o) for o, n in ranges)
        if submit_only:
            self.submit(shard, ranges)
            return None
        return self.receive(sum(n for _, n in ranges))


def leoncca_header(layout: Path, layer: int):
    path = layout / f"hy4-engram-fp8-{layer:02d}.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    header = json.loads(raw)
    key = TABLE.format(layer)
    codes, scales = header[key], header[key + "_scale_inv"]
    rows, dim = codes["shape"]
    assert codes["dtype"] == "F8_E4M3" and scales["dtype"] == "BF16" and scales["shape"] == [rows], header
    assert set(header) == {"__metadata__", key, key + "_scale_inv"}, sorted(header)
    return path, n, raw, header, rows, dim


def encode_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 rows -> (E4M3 codes, BF16 scale_inv) of Tencent's FP8 round trip."""
    x = x.float()
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
    codes = (x / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes, s.squeeze(-1).to(torch.bfloat16)


def tencent_round_trip(x: torch.Tensor) -> torch.Tensor:
    """fp32 reference: Tencent's per-row E4M3 round trip."""
    x = x.float()
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
    return (x / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s


def source_table(source: Source, layer: int, rows: int, dim: int):
    shard, start, meta = source.tensor_meta(TABLE.format(layer))
    if meta["dtype"] != "BF16" or meta["shape"] != [rows, dim]:
        sys.exit(f"Source {TABLE.format(layer)} is {meta['dtype']} {meta['shape']}, layout has [{rows}, {dim}]")
    return shard, start


def sample_runs(rows: int, count: int, seed: int, limit: tuple[int, int] | None = None) -> list[int]:
    """Sorted starts of count // RUN non-overlapping runs of RUN rows in [lo, hi)."""
    lo, hi = limit or (0, rows)
    rng = random.Random(seed)
    starts = []
    while len(starts) < count // RUN:
        a = rng.randrange(lo, hi - RUN)
        i = bisect.bisect(starts, a)
        if (i == 0 or a >= starts[i - 1] + RUN) and (i == len(starts) or a + RUN <= starts[i]):
            starts.insert(i, a)
    return starts


def write_block(fd, header_len, header, key, row0, codes, scales):
    base = 8 + header_len
    dim = codes.shape[1]
    os.pwrite(fd, codes.view(torch.uint8).numpy().tobytes(),
              base + header[key]["data_offsets"][0] + row0 * dim)
    os.pwrite(fd, scales.view(torch.uint8).numpy().tobytes(),
              base + header[key + "_scale_inv"]["data_offsets"][0] + row0 * 2)


def encode_layer(args, source: Source, layer: int):
    path, n, raw, header, rows, dim = leoncca_header(args.layout, layer)
    key = TABLE.format(layer)
    shard, start = source_table(source, layer, rows, dim)
    final = args.out / path.name
    row_bytes = dim * 2
    if args.sample_rows:
        target = final.with_name(final.name + ".sample")
        runs = sample_runs(rows, args.sample_rows, args.seed + layer)
        blocks = [(a, RUN) for a in runs]
        progress_path, next_row = None, 0
    else:
        target = final.with_name(final.name + ".partial")
        if final.exists():
            print(f"{final} exists; skipping layer {layer}", flush=True)
            return
        progress_path = target.with_name(target.name + ".progress")
        next_row = 0
        if target.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text())
            assert progress["header_len"] == n and progress["rows"] == rows
            next_row = progress["next_row"]
        stop = min(rows, args.stop_row or rows)
        blocks = [(a, min(BLOCK_ROWS, stop - a)) for a in range(next_row, stop, BLOCK_ROWS)]
    fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.pwrite(fd, struct.pack("<Q", n) + raw, 0)
        os.ftruncate(fd, os.path.getsize(path))
        t0, done = time.monotonic(), 0
        if blocks and source.local is None:
            source.ranges(shard, [(start + blocks[0][0] * row_bytes, blocks[0][1] * row_bytes)], True)
        for i, (row0, count) in enumerate(blocks):
            if source.local is None and i + 1 < len(blocks):
                a, c = blocks[i + 1]  # Keep the next block in flight.
                source.ranges(shard, [(start + a * row_bytes, c * row_bytes)], True)
            data = source.ranges(shard, [(start + row0 * row_bytes, count * row_bytes)])
            x = torch.frombuffer(data, dtype=torch.bfloat16).reshape(count, dim)
            codes, scales = encode_rows(x)
            write_block(fd, n, header, key, row0, codes, scales)
            done += count
            if progress_path is not None and (i % 16 == 15 or i + 1 == len(blocks)):
                os.fsync(fd)
                progress_path.write_text(json.dumps(
                    {"header_len": n, "rows": rows, "next_row": row0 + count}))
                rate = done * row_bytes / (time.monotonic() - t0) / 1e6
                print(f"layer {layer}: {row0 + count}/{rows} rows, {rate:.1f} MB/s read", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    elapsed = time.monotonic() - t0
    print(f"layer {layer}: wrote {done} rows to {target} in {elapsed:.1f}s", flush=True)
    if args.sample_rows:
        target.with_name(target.name + ".rows.json").write_text(json.dumps({"run": RUN, "starts": runs}))
    elif blocks and blocks[-1][0] + blocks[-1][1] == rows:
        os.replace(target, final)
        progress_path.unlink()
        print(f"layer {layer}: complete {final}", flush=True)


def read_file_rows(path: Path, header_len: int, header: dict, key: str, starts: list[int], dim: int):
    base = 8 + header_len
    c0 = base + header[key]["data_offsets"][0]
    s0 = base + header[key + "_scale_inv"]["data_offsets"][0]
    with open(path, "rb") as f:
        fd = f.fileno()
        codes = b"".join(os.pread(fd, RUN * dim, c0 + a * dim) for a in starts)
        scales = b"".join(os.pread(fd, RUN * 2, s0 + a * 2) for a in starts)
    codes = torch.frombuffer(bytearray(codes), dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.frombuffer(bytearray(scales), dtype=torch.bfloat16)
    return codes.reshape(-1, dim), scales


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rel(a, b):
    return float((a - b).pow(2).sum() / b.pow(2).sum())


def verify_layer(args, source: Source, layer: int) -> dict:
    lpath, n, raw, header, rows, dim = leoncca_header(args.layout, layer)
    key = TABLE.format(layer)
    path = args.file_for(layer)
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header_identical = size == n and f.read(size) == raw
    size_identical = os.path.getsize(path) == os.path.getsize(lpath)
    sample = path.with_name(path.name + ".rows.json")
    if sample.exists():
        starts = json.loads(sample.read_text())["starts"]
    else:
        starts = sample_runs(rows, args.rows, args.seed * 7919 + layer, args.within)
    shard, start = source_table(source, layer, rows, dim)
    row_bytes = dim * 2
    bf = []
    for i in range(0, len(starts), 64):
        part = [(start + a * row_bytes, RUN * row_bytes) for a in starts[i:i + 64]]
        if source.local is None:
            source.ranges(shard, part, True)
        bf.append(torch.frombuffer(source.ranges(shard, part), dtype=torch.bfloat16).reshape(-1, dim))
    bf = torch.cat(bf).float()
    codes, scales = read_file_rows(path, n, header, key, starts, dim)
    ours = codes.float() * scales.float().reshape(-1, 1)
    leon_codes, leon_scales = read_file_rows(lpath, n, header, key, starts, dim)
    leon = leon_codes.float() * leon_scales.float().reshape(-1, 1)
    tencent = tencent_round_trip(bf)
    want_codes, _ = encode_rows(bf)
    result = {
        "file": str(path), "rows_checked": int(bf.shape[0]), "runs": len(starts),
        "header_identical": header_identical, "size_identical": size_identical,
        "reencoded_vs_tencent_normed": rel(rms(ours), rms(tencent)),
        "code_match_frac": float((codes.view(torch.uint8) == want_codes.view(torch.uint8)).float().mean()),
        "leoncca_vs_tencent_normed": rel(rms(leon), rms(tencent)),
        "leoncca_code_mismatch_frac": float((leon_codes.view(torch.uint8)
                                             != want_codes.view(torch.uint8)).float().mean()),
    }
    result["G1"] = bool(header_identical and size_identical
                        and result["reencoded_vs_tencent_normed"] < G1_MAX)
    print(json.dumps({f"layer {layer}": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--layout", type=Path, required=True,
                         help="checkpoint whose hy4-engram-fp8-NN.safetensors headers to copy")
        cmd.add_argument("--src", required=True,
                         help="Tencent's BF16 checkpoint: a local directory or HOST:DIR (read over ssh)")
        cmd.add_argument("--layers", default=",".join(map(str, LAYERS)))
        cmd.add_argument("--seed", type=int, default=20261124)
    enc = sub.choices["encode"]
    enc.add_argument("--out", type=Path, required=True)
    enc.add_argument("--sample-rows", type=int, default=0)
    enc.add_argument("--stop-row", type=int, default=0, help="testing: stop (resumably) at this row")
    ver = sub.choices["verify"]
    ver.add_argument("--dir", type=Path, required=True, help="directory with the re-encoded files")
    ver.add_argument("--suffix", default="", help='".sample" or ".partial" for test outputs')
    ver.add_argument("--rows", type=int, default=1_000_000)
    ver.add_argument("--within", default=None, help="A:B row range for random rows (partial files)")
    ver.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    source = Source(args.src)
    try:
        with torch.inference_mode():
            if args.command == "encode":
                args.out.mkdir(parents=True, exist_ok=True)
                for layer in layers:
                    encode_layer(args, source, layer)
            else:
                args.within = tuple(map(int, args.within.split(":"))) if args.within else None
                args.file_for = lambda l: args.dir / f"hy4-engram-fp8-{l:02d}.safetensors{args.suffix}"
                results = {str(l): verify_layer(args, source, l) for l in layers}
                if args.json:
                    args.json.write_text(json.dumps(results, indent=1) + "\n")
                ok = all(r["G1"] for r in results.values())
                print("G1 PASS" if ok else "G1 FAIL", flush=True)
                sys.exit(0 if ok else 1)
    finally:
        source.close()


if __name__ == "__main__":
    main()
