# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collect shared-expert inputs and build the groupwise GPTQ int8 W8 pack.

Used by tools/hy4/shared_w8_cache.py through vLLM's worker_extension_cls.
"""
import os
from pathlib import Path

import torch
from torch.nn import functional as F

GROUP = 32


def quantize(weight, gram):
    n, k = weight.shape
    assert k % GROUP == 0 and gram.shape == (k // GROUP, GROUP, GROUP)
    diag = gram.diagonal(dim1=-2, dim2=-1)
    diag.add_(0.01 * diag.mean(dim=-1, keepdim=True).clamp_min(1e-9))
    inv = torch.cholesky_inverse(torch.linalg.cholesky(gram))
    upper = torch.linalg.cholesky(inv, upper=True)

    n_pad = (n + 7) // 8 * 8
    rows = F.pad(weight.float(), (0, 0, 0, n_pad - n)).view(n_pad, -1, GROUP)
    scales = (rows.abs().amax(dim=-1) / 127).half()
    scales = torch.where(scales > 0, scales, torch.full_like(scales, 2**-24))
    fs = scales.float()
    codes = torch.empty_like(rows, dtype=torch.uint8)
    for col in range(GROUP):
        target = rows[..., col]
        code = (target / fs).round().clamp(-127, 127)
        codes[..., col] = (code + 128).to(torch.uint8)
        restored = (code.half() * scales).float()
        residual = (target - restored) / upper[:, col, col]
        rows[..., col + 1:].sub_(residual.unsqueeze(-1) * upper[:, col, col + 1:])

    codes = codes.reshape(n_pad // 8, 8, k // 16, 16).transpose(1, 2)
    scales = scales.view(n_pad // 8, 8, -1).transpose(1, 2)
    return torch.cat((codes.contiguous().flatten(),
                      scales.contiguous().flatten().view(torch.uint8))).view(n_pad, -1)


class CalibrationWorker:
    @torch.inference_mode()
    def w8_calibration_begin(self):
        projections = []
        for layer in self.model_runner.get_model().modules():
            if type(layer).__name__ != "HYV4FlashMoE" or layer.shared_experts is None:
                continue
            projections.extend((layer.shared_experts.gate_up_proj,
                                layer.shared_experts.down_proj))
        assert len(projections) == 82, len(projections)
        self._w8_probes = []
        for layer in projections:
            weight = layer.weight
            assert weight.ndim == 2 and weight.dtype == torch.float16
            k = weight.shape[1]
            gram = torch.zeros((k // GROUP, GROUP, GROUP),
                               dtype=torch.float32, device=weight.device)
            tokens = torch.zeros((), dtype=torch.int64, device=weight.device)

            def observe(_, args, gram=gram, k=k, tokens=tokens):
                x = args[0].reshape(-1, k).float().view(-1, k // GROUP, GROUP).transpose(0, 1)
                gram.baddbmm_(x.transpose(-2, -1), x)
                tokens.add_(x.shape[1])

            self._w8_probes.append((layer, layer.register_forward_pre_hook(observe), gram, tokens))
        return len(self._w8_probes)

    @torch.inference_mode()
    def w8_calibration_finish(self, destination):
        from vllm.distributed import get_tensor_model_parallel_rank

        pack = {}
        min_tokens = None
        for layer, hook, gram, tokens in self._w8_probes:
            hook.remove()
            count = int(tokens.item())
            assert count > 0, layer.prefix
            min_tokens = count if min_tokens is None else min(min_tokens, count)
            pack[layer.prefix] = quantize(layer.weight, gram / count).cpu()
        assert len(pack) == 82
        path = Path(destination) / f"rank-{get_tensor_model_parallel_rank()}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(pack, temporary)
        os.replace(temporary, path)
        self._w8_probes.clear()
        return {"rank": path.stem, "projections": len(pack), "min_tokens": min_tokens,
                "bytes": path.stat().st_size}
