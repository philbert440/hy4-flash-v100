# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from itertools import accumulate
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.test_hy4_flash_verification import (
    forward_scope,
    keys,
    new_indexer,
)
from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.worker.test_hy4_engram_scheduler import make_request, make_runner
from vllm.models.hy_v4_flash.sm70.cache import QuerySpan
from vllm.models.hy_v4_flash.sm70.dfly import DraftWorkspace
from vllm.models.hy_v4_flash.sm70.dfly_proposer import DFlyProposer
from vllm.models.hy_v4_flash.sm70.indexpool import FlashCompressedIndexer
from vllm.models.hy_v4_flash.sm70.verification import VerificationState
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.sample.rejection_sampler import RejectionSampler, rejection_sample

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        os.getenv("TRITON_INTERPRET") != "1", reason="Triton interpreter required"
    ),
]


def test_point_mass_draft_rejection_distribution():
    torch.manual_seed(31)
    n = 4096
    probs = torch.tensor([0.42, 0.33, 0.25])
    logits = probs.log().expand(n, -1).contiguous()
    draft = torch.zeros(n, dtype=torch.int32)
    metadata = SimpleNamespace(
        temperature=torch.ones(n), all_greedy=False, all_random=True, generators={}
    )
    result = rejection_sample(
        draft,
        [1] * n,
        1,
        torch.arange(1, n + 1, dtype=torch.int32),
        None,
        logits,
        torch.zeros(n, 1, dtype=torch.int32),
        metadata,
    )
    frequencies = result[:, 0].bincount(minlength=3).float() / n
    accepted = (result[:, 1] >= 0).float().mean()
    print(
        f"target={probs.tolist()} observed={frequencies.tolist()} "
        f"accept={accepted.item()}"
    )
    torch.testing.assert_close(frequencies, probs, atol=0.026, rtol=0)
    assert (result[:, 0].eq(0) == result[:, 1].ge(0)).all()


@pytest.mark.parametrize("load_transition", [False, True])
def test_dfly_scheduler_lifecycle(monkeypatch, load_transition):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    create_requests(0)
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=12,
        max_model_len=96,
        num_speculative_tokens=5,
        block_size=16,
        long_prefill_token_threshold=5,
        enable_prefix_caching=True,
        num_blocks=32,
        async_scheduling=False,
    )
    runner = make_runner(monkeypatch)
    runner.num_spec_tokens = 5
    runner.speculative_config = SimpleNamespace(
        method="dfly", use_ngram_gpu=lambda: False
    )
    drafter_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=96))
    if load_transition:
        monkeypatch.setenv("VLLM_HY4_DFLY_GATE", "ewma")
        monkeypatch.setenv("VLLM_HY4_DFLY_GATE_MAX_RUNNING", "1")
        scheduler.hy4_dfly_gate_max_running = 1
        drafter_config.speculative_config = runner.speculative_config
    runner.drafter = DFlyProposer(drafter_config, "cpu")
    runner.drafter.settings = SimpleNamespace(target_layers=(25, 29), mask_id=999)
    runner.drafter.workspace = DraftWorkspace(torch.zeros(1), torch.arange(12))
    runner.drafter.embed = runner.drafter.logits = None
    idx = new_indexer(blocks=3)
    idx.verification = runner._hy4_verification = VerificationState()
    ring = idx.tail_cache
    runner.compilation_config = SimpleNamespace(
        static_forward_context={ring.prefix: ring}
    )
    prompt = list(range(20))
    warm = make_request("warm", prompt, 1)
    follower = make_request(
        "follower", prompt[:16] + [20, 30, 40], 40 if load_transition else 22
    )
    other = make_request("other", list(range(8)), 6 if load_transition else 17)
    reqs = {req.request_id: req for req in (warm, follower, other)}
    table_ids = {"warm": 1, "follower": 2, "other": 3}
    seen_acceptance = set()
    first_cursor = {}
    mixed = nonfinal = resumed = False
    cycles = 0
    transitions = []

    def next_id(position):
        return 300 + position

    def draft_forward(tokens, positions, hidden, histories, workspace, *_):
        nonlocal cycles
        anchors = positions.reshape(-1, 6)[:, 0]
        assert torch.equal(hidden[:, 0] + 1, anchors)
        assert torch.equal(tokens, next_id(anchors - 1))
        for offset, history in enumerate(histories):
            assert torch.equal(history.slots[:, 0], anchors - 1 + 100 * offset)
            assert history.counts.tolist() == [1] * anchors.numel()
        drafts = next_id(anchors[:, None] + torch.arange(5))
        keep = (5, 2, 0)[cycles % 3]
        if keep < 5:
            drafts[:, keep] = 990
        cycles += 1
        workspace.scratch.fill_(cycles)
        return drafts

    runner.drafter.backbone = SimpleNamespace(propose=draft_forward)

    def step():
        nonlocal mixed, nonfinal, resumed
        out = scheduler.schedule()
        assert out.num_scheduled_tokens
        runner._update_states(out)
        ids = list(runner.input_batch.req_ids)
        widths = [out.num_scheduled_tokens[r] for r in ids]
        starts = list(accumulate(widths, initial=0))
        prefixes = [runner.requests[r].num_computed_tokens for r in ids]
        lengths = [p + w for p, w in zip(prefixes, widths)]
        counts = [len(out.scheduled_spec_decode_tokens.get(r, [])) for r in ids]
        discard = [n < reqs[r].num_tokens for n, r in zip(lengths, ids)]
        for r, p in zip(ids, prefixes):
            first_cursor.setdefault(r, p)
        mixed |= len(set(lengths)) == 2
        nonfinal |= any(discard)
        resumed |= "follower" in out.scheduled_cached_reqs.resumed_req_ids

        runner.query_start_loc = SimpleNamespace(np=np.array(starts))
        runner.optimistic_seq_lens_cpu = torch.tensor(lengths)
        runner.discard_request_mask = SimpleNamespace(np=np.array(discard))
        record = runner._prepare_hy4_dfly_target(counts, out.hy4_dfly_draft_load)
        assert (record is None) == all(discard)
        positions = torch.cat([torch.arange(p, n) for p, n in zip(prefixes, lengths)])
        hidden = positions[:, None]
        verify = {r for r, count in zip(ids, counts) if count}
        spans = [
            QuerySpan(b, e, p, n)
            for b, e, p, n in zip(starts, starts[1:], prefixes, lengths)
        ]
        pages = torch.tensor([[table_ids[r]] for r in ids])
        _, meta = forward_scope(idx, list(zip(ids, spans)), pages, verify)
        with runner._hy4_verification_forward({ring.prefix: meta}, ids, widths, verify):
            for span, table in zip(spans, pages):
                data, pos = keys(span.prefix, span.length, 1, table.device)
                FlashCompressedIndexer.prepare_groups(
                    idx,
                    data,
                    pos,
                    torch.ones(32),
                    span,
                    table,
                    table,
                    meta.verification,
                )
            if record is not None:
                slots = positions[:, None].clone()
                live = torch.ones(positions.numel(), dtype=torch.int)
                for layer in runner.drafter.settings.target_layers:
                    record.capture(layer, ring.kv_cache, slots, live)
                    slots.add_(100)
        if load_transition and "follower" in ids and out.hy4_dfly_draft_load > 1:
            row = ids.index("follower")
            position = lengths[row]
            tail = idx.verification._requests["follower"]
            assert tail.position == position and tail.pending is None
            live_positions = torch.arange(position - position % 4, position)
            ring_rows = ring.kv_cache[table_ids["follower"], :len(live_positions)]
            assert torch.equal(ring_rows[:, -1], live_positions.float())
            assert torch.equal(ring_rows[:, 0], (live_positions + 1).float())
        sampled = []
        for r, skip, n, count in zip(ids, discard, lengths, counts):
            if skip:
                sampled.append([])
            elif not count:
                sampled.append([next_id(n - 1)])
            else:
                draft = torch.tensor(
                    out.scheduled_spec_decode_tokens[r], dtype=torch.int32
                )
                greedy = torch.arange(n - count - 1, n - 1) + 300
                logits = torch.zeros(count, 1000).scatter_(1, greedy[:, None], 1)
                result = rejection_sample(
                    draft,
                    [count],
                    5,
                    torch.tensor([count], dtype=torch.int32),
                    None,
                    logits,
                    torch.tensor([[next_id(n - 1)]], dtype=torch.int32),
                    SimpleNamespace(all_greedy=True, all_random=False),
                )
                tokens = RejectionSampler.parse_output(result, 1000)[0][0]
                seen_acceptance.add(len(tokens) - 1)
                assert tokens == list(
                    range(next_id(n - count - 1), next_id(n - count - 1) + len(tokens))
                )
                sampled.append(tokens)
        for i, (r, tokens) in enumerate(zip(ids, sampled)):
            start = runner.input_batch.num_tokens_no_spec[i]
            end = start + len(tokens)
            runner.input_batch.token_ids_cpu[i, start:end] = tokens
            runner.input_batch.is_token_ids[i, start:end] = True
            runner.input_batch.num_tokens_no_spec[i] = end
            runner.requests[r].output_token_ids.extend(tokens)
        scratch_before = runner.drafter.workspace.scratch.clone()
        runner._draft_token_ids = runner.propose_draft_token_ids(
            out, sampled, None, hidden, hidden, None, None, None, None
        )
        runner._copy_draft_token_ids_to_cpu(out)
        published = runner.take_draft_token_ids()
        assert published is not None
        overloaded = load_transition and out.hy4_dfly_draft_load > 1
        for skip, drafts in zip(discard, published.draft_token_ids):
            assert len(drafts) == (0 if skip or overloaded else 5)
        if overloaded:
            assert not verify
            assert torch.equal(runner.drafter.workspace.scratch, scratch_before)
        if "follower" in ids:
            transitions.append((bool(verify), overloaded))
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        scheduler.update_draft_token_ids(published)

    scheduler.add_request(warm)
    while warm.request_id in scheduler.requests:
        step()
    scheduler.add_request(follower)
    if not load_transition:
        scheduler.add_request(other)
    peer_added = False
    preempted = False
    for _ in range(60):
        if not scheduler.requests:
            break
        step()
        if load_transition and not peer_added and seen_acceptance:
            scheduler.add_request(other)
            peer_added = True
        if not load_transition and not preempted and len(follower.output_token_ids) >= 7:
            assert follower.status == RequestStatus.RUNNING
            scheduler.running.remove(follower)
            scheduler._preempt_request(follower, 0.0)
            scheduler.prev_step_scheduled_req_ids.discard(follower.request_id)
            # Simulate the worker's receipt of the scheduler preemption here.
            runner._hy4_verification.forget((follower.request_id,))
            preempted = True
    assert not scheduler.requests
    assert first_cursor["warm"] == 0 and first_cursor["follower"] == 16
    if load_transition:
        first_skip = transitions.index((False, True))
        assert (True, False) in transitions[:first_skip]
        assert (True, False) in transitions[first_skip + 1:]
    else:
        assert mixed and nonfinal and resumed and follower.num_preemptions == 1
    assert {0, 2, 5} <= seen_acceptance
    for request in reqs.values():
        first = next_id(len(request.prompt_token_ids) - 1)
        assert list(request.output_token_ids) == list(
            range(first, first + request.max_tokens)
        )


@pytest.mark.parametrize("hard_cap", [True, False])
def test_load_gated_steps_emit_plain_decode_ids(monkeypatch, hard_cap):
    from vllm.config import CUDAGraphMode
    from vllm.v1.outputs import DraftTokenIds

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    monkeypatch.setenv("VLLM_HY4_DFLY_GATE", "off")
    monkeypatch.setenv("VLLM_HY4_DFLY_GATE_MAX_RUNNING", "2" if hard_cap else "0")
    monkeypatch.setenv("VLLM_HY4_DFLY_GRAPHS", "both")
    create_requests(0)

    def execute(speculative):
        scheduler = create_scheduler(
            max_num_seqs=3,
            max_num_batched_tokens=32,
            max_model_len=96,
            num_speculative_tokens=5 if speculative else None,
            block_size=16,
            num_blocks=32,
            async_scheduling=False,
        )
        proposer = DFlyProposer(
            SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=96),
                speculative_config=SimpleNamespace(method="dfly"),
                compilation_config=SimpleNamespace(
                    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                    cudagraph_capture_sizes=[6, 12, 24],
                ) if not hard_cap else None,
            ),
            "cpu",
        ) if speculative else None
        requests = [make_request(str(i), list(range(8)), 13) for i in range(3)]
        for request in requests:
            scheduler.add_request(request)
        if speculative:
            # The generic CPU scheduler fixture configures a different drafter.
            scheduler.hy4_dfly_gate_max_running = 2 if hard_cap else None
            scheduler.hy4_dfly_graph_sizes = None if hard_cap else {1, 2, 4}
            if not hard_cap:
                assert not proposer.load_allows(3, 3)
                proposer.verify_graphs[18] = None
                assert not proposer.load_allows(3, 3)
                proposer.draft_graphs[3] = None
                assert proposer.load_allows(3, 3)
                assert not proposer.load_allows(3, 2)
                del proposer.draft_graphs[3]

        steps = []
        while scheduler.requests:
            out = scheduler.schedule()
            ids = list(out.num_scheduled_tokens)
            widths = list(out.num_scheduled_tokens.values())
            lengths = [
                scheduler.requests[req].num_computed_tokens
                for req in ids
            ]
            sampled = [[300 + n] for n in lengths]
            if steps:
                assert widths == [1, 1, 1]
            if proposer is not None:
                count = len(ids)
                assert out.hy4_dfly_draft_load == count == 3
                assert not out.scheduled_spec_decode_tokens
                proposer.prepare_target(
                    list(accumulate(widths, initial=0)), lengths,
                    [False] * count, [0] * count, ids, out.hy4_dfly_draft_load,
                )
                proposals = proposer.propose(sampled, torch.empty(count, 1))
                assert proposals == [[], [], []]
            scheduler.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=ids, req_id_to_index={req: i for i, req in enumerate(ids)},
                    sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
            if proposer is not None:
                scheduler.update_draft_token_ids(DraftTokenIds(ids, proposals))
            steps.append(sampled)
        assert len(steps) == 13
        return [list(request.output_token_ids) for request in requests]

    assert execute(True) == execute(False)


def test_gate_probes_after_eight_plain_steps_and_recovers():
    from vllm.models.hy_v4_flash.dfly_gate import DFlyGate

    gate = DFlyGate(threshold=0.9)
    for _ in range(24):
        assert gate.allow(["r", "peer"])
        gate.observe("r", 0)
    assert [gate.allow(["r", "peer"]) for _ in range(8)] == [False] * 8
    assert gate.allow(["r", "peer"])
    gate.observe("r", 0)
    assert [gate.allow(["r", "peer"]) for _ in range(8)] == [False] * 8
    assert gate.allow(["r", "peer"])
    gate.observe("r", 5)
    assert gate.allow(["r", "peer"])

    # Recover from a low-acceptance stretch even if each probe accepts one.
    for _ in range(32):
        gate.observe("r", 0)
    for _ in range(16):
        assert [gate.allow(["r", "peer"]) for _ in range(8)] == [False] * 8
        assert gate.allow(["r", "peer"])
        gate.observe("r", 1)
        if not gate.requests["r"].paused:
            break
    assert not gate.requests["r"].paused


def test_dfly_request_boundaries_clear_gate_and_confidence(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    monkeypatch.setenv("VLLM_HY4_DFLY_GATE", "ewma")
    monkeypatch.setenv("VLLM_HY4_DFLY_LOG_CONFIDENCE", "1")
    runner = make_runner(monkeypatch)
    runner.speculative_config = SimpleNamespace(
        method="dfly", use_ngram_gpu=lambda: False
    )
    runner.drafter = DFlyProposer(
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=96),
            speculative_config=runner.speculative_config,
        ),
        "cpu",
    )
    for req in ("preempted", "finished", "resumed", "live"):
        runner.drafter.gate.observe(req, 0)
        runner.drafter.pending_confidence[req] = (16, [0.5] * 5)

    output = SchedulerOutput.make_empty()
    output.finished_req_ids.add("finished")
    output.preempted_req_ids = {"preempted"}
    output.scheduled_cached_reqs.resumed_req_ids.add("resumed")
    runner._update_states(output)

    assert runner.drafter.gate.requests.keys() == {"live"}
    assert runner.drafter.pending_confidence.keys() == {"live"}


def test_context_limit_confidence_does_not_request_target_history(monkeypatch):
    for confidence in ("1", "0"):
        monkeypatch.setenv("VLLM_HY4_DFLY_LOG_CONFIDENCE", confidence)
        proposer = DFlyProposer(
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=96)), "cpu"
        )
        assert proposer.prepare_target(
            [0, 6], [96], [False], [5], ["r"]
        ) is None
        assert (proposer.step is not None) == (confidence == "1")
        assert proposer.propose([[123]], torch.empty(0)) == [[]]


@pytest.mark.parametrize("verify", [False, True])
def test_kv_scales_fallback_drops_hy4_history_snapshot(monkeypatch, verify):
    from vllm.config import CUDAGraphMode

    runner = make_runner(monkeypatch)
    graph = object()
    hy4_graph, plain_snapshot = (graph, None) if verify else (None, graph)
    runner.calculate_kv_scales = True
    assert runner._finalize_forward_graphs(
        CUDAGraphMode.FULL, hy4_graph, plain_snapshot, False
    ) == (CUDAGraphMode.NONE, None, None, True)
    assert not runner.calculate_kv_scales
    assert runner._finalize_forward_graphs(
        CUDAGraphMode.FULL, hy4_graph, plain_snapshot, False
    ) == (CUDAGraphMode.FULL, hy4_graph, plain_snapshot, False)
