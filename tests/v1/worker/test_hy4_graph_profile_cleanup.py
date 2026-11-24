# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.worker import gpu_model_runner as runner


def test_resident_graph_profile_releases_blas_workspaces(monkeypatch):
    events = []
    fake = Mock()
    fake._freeze_gc.return_value = nullcontext()
    fake.cudagraph_dispatcher.get_capture_descs.return_value = [
        (SimpleNamespace(name="FULL"), [SimpleNamespace(num_tokens=1)])
    ]
    fake.cudagraph_dispatcher.cudagraph_keys = {}
    fake._cleanup_profiling_kv_cache.side_effect = lambda: events.append("cleanup")
    monkeypatch.setattr(runner, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(runner, "graph_capture", lambda **_: nullcontext())
    monkeypatch.setattr(runner.current_platform, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(runner.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(runner, "set_cudagraph_capturing_enabled", lambda _: None)
    monkeypatch.setattr(
        runner.CUDAGraphWrapper, "clear_all_graphs", lambda: events.append("graphs")
    )
    monkeypatch.setattr(runner.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(runner.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner.torch._C, "_cuda_clearCublasWorkspaces", lambda: events.append("blas")
    )

    def snapshot(**_):
        events.append("snapshot")
        return SimpleNamespace(
            cuda_memory=100 if "blas" not in events else 25,
            torch_allocated_memory=100 if "blas" not in events else 25,
            torch_memory=100 if "blas" not in events else 25,
        )

    monkeypatch.setattr(runner, "MemorySnapshot", snapshot)

    released = runner.GPUModelRunner.profile_cudagraph_memory(fake, measure_residency=True)

    assert released == 75
    assert events == ["snapshot", "graphs", "blas", "cleanup", "snapshot"]
