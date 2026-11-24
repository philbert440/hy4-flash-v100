# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import HYV4FlashForCausalLMConfig
from vllm.models.hy_v4_flash.sm70.cache import FlashMetadata, QuerySpan
from vllm.models.hy_v4_flash.sm70.decode import (
    decode_slots,
    gather_decode_group,
    store_decode_group,
)
from vllm.models.hy_v4_flash.sm70.indexpool import (
    FlashCompressedIndexer,
    select_tokens,
)
from vllm.models.hy_v4_flash.sm70.kernels import (
    cache_fp8_ds_mla,
    index_scores,
    pack_fp8_ds_mla_torch,
    paged_attention,
)
from vllm.models.hy_v4_flash.sm70.splitk import (
    decode_workspace_specs,
    paged_attention_splitk,
)
from vllm.models.hy_v4_flash.sm70.verification import (
    VerificationGraph,
    VerificationState,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import CircularBufferSpec

RATIO, DIM, PAGE, WINDOW = 4, 128, 256, 512
pytestmark = pytest.mark.skip_global_cleanup
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def new_indexer(device="cpu", pool_block_size=64, blocks=1):
    return SimpleNamespace(
        ratio=RATIO,
        pos_dims=1,
        mrope_section=None,
        dim=DIM,
        k_cache=SimpleNamespace(
            kv_cache=torch.zeros(
                blocks + 1, 64, DIM, dtype=torch.float16, device=device
            ),
            spec=SimpleNamespace(storage_block_size=pool_block_size),
        ),
        tail_cache=SimpleNamespace(
            prefix="tail",
            spec=ring_spec(DIM),
            kv_cache=torch.zeros(blocks + 1, RATIO, DIM + 1, device=device),
        ),
        dcp_group=None,
        key_value_projection=lambda rows: rows,
        key_gate_projection=torch.zeros_like,
        intra_group_ape=torch.zeros(RATIO, DIM, device=device),
        key_norm=lambda rows: rows,
        _rotate=lambda rows, *_: rows,
    )


def keys(start, stop, offset, device):
    positions = torch.arange(start, stop, device=device)
    return (positions[:, None] + offset).expand(-1, DIM).half().contiguous(), positions


def ring_spec(dim):
    return CircularBufferSpec(
        block_size=RATIO, num_kv_heads=1, head_size=dim + 1, dtype=torch.float32
    )


def forward_scope(indexer, requests, ring_table, verify=(), finalized=None):
    spans = tuple(span for _, span in requests)
    meta = FlashMetadata(
        spans,
        ring_table,
        ring_table[:0],
        spans[-1].end,
        ring_table.new_tensor([0, *(span.end for span in spans)]),
        ring_table.new_tensor([span.length for span in spans]),
        False,
        None,
        None,
    )
    name = indexer.tail_cache.prefix
    scope = indexer.verification.forward(
        {name: meta},
        {name: indexer.tail_cache},
        requests,
        verify,
        (
            {req_id: span.prefix + 1 for req_id, span in requests}
            if finalized is None
            else finalized
        ),
    )
    return scope, meta


def prepare(
    indexer,
    table,
    start,
    stop,
    *,
    offset=1,
    verify=False,
    ring_id=None,
):
    ring_id = table[:1] if ring_id is None else ring_id
    inputs, positions = keys(start, stop, offset, table.device)
    span = QuerySpan(0, stop - start, start, stop)
    request = str(int(ring_id[0]))
    if verify or hasattr(indexer, "verification"):
        if not hasattr(indexer, "verification"):
            indexer.verification = VerificationState()
        scope, meta = forward_scope(
            indexer,
            [(request, span)],
            ring_id[None],
            (request,) if verify else (),
        )
    else:
        scope, meta = nullcontext(), None
    with scope:
        FlashCompressedIndexer.prepare_groups(
            indexer,
            inputs,
            positions,
            torch.ones(32, device=table.device),
            span,
            table,
            ring_id,
            None if meta is None else meta.verification,
        )


@pytest.mark.parametrize("accepted", range(6))
def test_stage1_index_ring(accepted):
    table = torch.tensor([1])
    verified, reference = new_indexer(), new_indexer()
    prepare(verified, table, 0, 1)
    prepare(reference, table, 0, 1)
    prepare(verified, table, 1, 6, verify=True)
    committed = 1 + accepted
    for pos in range(1, committed):
        prepare(reference, table, pos, pos + 1)
    stop = (committed // RATIO + 1) * RATIO
    for indexer in (reference, verified):
        prepare(indexer, table, committed, stop, offset=101)
    expected = reference.k_cache.kv_cache[1, : stop // RATIO]
    actual = verified.k_cache.kv_cache[1, : stop // RATIO]
    assert torch.equal(expected, actual)
    if accepted == 0:
        assert actual[0, 0].item() == 77.5
        logits = torch.full((1, 512), 78.0)
        logits[0, 0] = actual[0, 0]
        selected = select_tokens(logits, torch.tensor([512 * RATIO]), RATIO, 511)
        assert not (selected == 0).any()


def kernel_store(pool, tail, table, ring_table, active, raw_keys, pos, pooled):
    store_decode_group[(len(pos),)](
        raw_keys,
        pos,
        pooled,
        (pos + 1).int(),
        active,
        table,
        ring_table,
        pool,
        tail,
        table.stride(0),
        ring_table.stride(0),
        *pool.stride()[:2],
        *tail.stride()[:2],
        RATIO,
        DIM,
        pool.shape[1],
        1,
    )


def kernel_gather(tail, ring_table, active, raw_keys, pos):
    raw = torch.empty(len(pos), RATIO, DIM, device=tail.device)
    first = torch.empty_like(pos)
    gather_decode_group[(len(pos),)](
        raw_keys,
        (pos + 1).int(),
        active,
        ring_table,
        tail,
        raw,
        first,
        ring_table.stride(0),
        *tail.stride()[:2],
        RATIO,
        DIM,
        1,
    )
    return raw, first


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stage1_decode_gather(device):
    if device == "cpu" and not INTERPRET:
        pytest.skip("Triton CPU interpreter required")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = torch.tensor([1], device=device)
    for n in (0, 5):
        idx = new_indexer(device)
        prepare(idx, table, 0, 1)
        if n:
            prepare(idx, table, 1, 6, verify=True)
        pool, tail = idx.k_cache.kv_cache, idx.tail_cache.kv_cache
        scope = (
            forward_scope(idx, [("1", QuerySpan(0, 1, 1, 2))], table[None])[0]
            if n
            else nullcontext()
        )
        with scope:
            key, pos = keys(1, 2, 101, device)
            kernel_store(pool, tail, table[None], table[None], table, key, pos, key)
        key, pos = keys(2, 3, 101, device)
        kernel_store(pool, tail, table[None], table[None], table, key, pos, key)
        key, pos = keys(3, 4, 101, device)
        raw, first = kernel_gather(tail, table[None], table, key, pos)
        assert raw[0, :, 0].tolist() == [1, 102, 103, 104]
        assert first.item() == 0


def test_forward_uses_authoritative_single_row_cuda_decode_spans():
    span = QuerySpan(0, 1, 3, 4)
    other = QuerySpan(1, 2, 10, 11)
    table = torch.tensor([[1], [2]])
    lengths = torch.tensor([4, 11])
    meta = FlashMetadata(
        (),
        table,
        torch.tensor([3, 10]),
        2,
        torch.arange(3),
        lengths,
        True,
        None,
        None,
    )
    state = VerificationState()
    requests = [("request", span), ("other", other)]
    with (
        pytest.raises(ValueError, match="metadata disagrees"),
        state.forward({"index": meta}, {}, requests[:1], {"request"}, {}),
    ):
        pytest.fail("invalid metadata reached forward")
    with state.forward({"index": meta, "alias": meta}, {}, requests, {"request"}, {}):
        assert not meta.decode and meta.verification is not None
        assert meta.spans == (span, other)
        assert meta.seq_lens is lengths and meta.block_table is table
    assert meta.decode and meta.spans == () and meta.verification is None


def test_hy4_startup_still_rejects_speculative_config(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        device_config=SimpleNamespace(device_type="cuda"),
        speculative_config=object(),
    )
    with pytest.raises(
        NotImplementedError, match="does not support --speculative-config"
    ):
        HYV4FlashForCausalLMConfig.verify_and_update_config(config)


@pytest.mark.parametrize("boundary", [0, 252, 256, 508, 512, 764, 768, 2044, 2048])
def test_accepted_prefix_matrix(boundary):
    if not INTERPRET:
        pytest.skip("Triton CPU interpreter required")
    # Sweep both sides of page, window, window reclamation and sparse cutoff
    # boundaries. With ratio 4 and page 256, not every short span can cross.
    cases = [
        (boundary + residue, width, accepted)
        for residue in range(RATIO)
        for width in range(1, 7)
        for accepted in range(width + 1)
    ]
    requests = len(cases)
    pages = (boundary + 16) // PAGE + 1
    table = torch.arange(1, requests * pages + 1).flip(0).view(requests, pages)
    ring_table = torch.arange(1, requests + 1)[:, None]
    logical = torch.arange(pages * PAGE)
    slots = table[:, logical // PAGE] * PAGE + logical % PAGE
    source_main = logical + 1
    source_swa = logical + 9001
    actual_main = torch.zeros((requests * pages + 1) * PAGE, dtype=torch.int32)
    actual_swa = torch.zeros_like(actual_main)
    expected_main = torch.zeros_like(actual_main)
    expected_swa = torch.zeros_like(actual_main)
    actual, expected = (new_indexer(blocks=requests * pages) for _ in range(2))
    all_lengths = []
    for row, (prefix, width, accepted) in enumerate(cases):
        ring = ring_table[row]
        t = table[row]
        # Seed a common committed prefix using the unchanged chunked path.
        if prefix:
            for idx in (actual, expected):
                prepare(idx, t, 0, prefix, ring_id=ring)
        selected = slots[row, :prefix]
        actual_main[selected] = expected_main[selected] = source_main[:prefix].int()
        actual_swa[selected] = expected_swa[selected] = source_swa[:prefix].int()

        prepare(actual, t, prefix, prefix + width, verify=True, ring_id=ring)
        selected = slots[row, prefix : prefix + width]
        actual_main[selected] = source_main[prefix : prefix + width].int()
        actual_swa[selected] = source_swa[prefix : prefix + width].int()
        end = prefix + accepted
        all_lengths.append(end)
        for pos in range(prefix, end):
            prepare(expected, t, pos, pos + 1, ring_id=ring)
            expected_main[slots[row, pos]] = source_main[pos]
            expected_swa[slots[row, pos]] = source_swa[pos]

        groups = torch.arange(end // RATIO)
        assert torch.equal(
            actual.k_cache.kv_cache[t[groups // 64], groups % 64],
            expected.k_cache.kv_cache[t[groups // 64], groups % 64],
        ), (prefix, width, accepted)
    positions = torch.tensor(all_lengths)
    active = torch.ones(requests, dtype=torch.int64)
    pool_a, pool_b = actual.k_cache.kv_cache, expected.k_cache.kv_cache
    ring_a, ring_b = actual.tail_cache.kv_cache, expected.tail_cache.kv_cache
    row_ids = torch.arange(requests)
    # Replay distinct inputs. Exercise two closes and a later score and slot
    # read after each committed token, including at a previously rejected row.
    for replay in range(8):
        pos = positions + replay
        physical = slots[row_ids, pos]
        key = (pos + 1500)[:, None].expand(-1, DIM).half().contiguous()
        batch = [
            (str(row + 1), QuerySpan(row, row + 1, end + replay, end + replay + 1))
            for row, end in enumerate(all_lengths)
        ]
        scope = (
            forward_scope(actual, batch, ring_table)[0]
            if replay == 0
            else actual.verification.forward(None, {}, batch, (), {})
        )
        with scope:
            if replay == 0:
                for row, end in enumerate(all_lengths, 1):
                    live = end % RATIO
                    assert torch.equal(ring_a[row, :live], ring_b[row, :live])
            raw_a, first_a = kernel_gather(ring_a, ring_table, active, key, pos)
            raw_b, first_b = kernel_gather(ring_b, ring_table, active, key, pos)
            assert torch.equal(raw_a, raw_b)
            assert torch.equal(first_a, first_b)
            kernel_store(
                pool_a,
                ring_a,
                table,
                ring_table,
                active,
                key,
                pos,
                raw_a.mean(1).half(),
            )
            kernel_store(
                pool_b,
                ring_b,
                table,
                ring_table,
                active,
                key,
                pos,
                raw_b.mean(1).half(),
            )
        lengths = (pos + 1).int()
        query = torch.ones((requests, 1, DIM), dtype=torch.float16)
        weights = torch.ones(requests, 1)
        scores_a, scores_b = (
            index_scores(
                query,
                weights,
                pool,
                table,
                0,
                pages * 64,
                RATIO,
                1,
                lengths=lengths,
                slots=active,
                group_topk=4,
            )
            for pool in (pool_a, pool_b)
        )
        assert torch.equal(scores_a, scores_b)
        groups = scores_a.topk(3, sorted=False).indices.sort().values

        actual_main[physical] = expected_main[physical] = (pos + 12000).int()
        actual_swa[physical] = expected_swa[physical] = (pos + 22000).int()
        for width, ratio, history, left, right in (
            (16, RATIO, groups, actual_main, expected_main),
            (WINDOW, 1, None, actual_swa, expected_swa),
        ):
            indices = torch.empty(requests, width, dtype=torch.int32)
            counts = torch.empty(requests, dtype=torch.int32)
            decode_slots(lengths, active, table, history, indices, counts, PAGE, ratio)
            live = torch.arange(width) < counts[:, None]
            assert (indices[~live] == -1).all()
            assert torch.equal(left[indices[live].long()], right[indices[live].long()])


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_uniform_graph_verification_lifecycle(batch):
    if not INTERPRET:
        pytest.skip("Triton CPU interpreter required")

    width = 6
    eager, replay = [new_indexer(blocks=batch * 2) for _ in range(2)]
    table = torch.arange(1, batch * 2 + 1).reshape(batch, 2)
    ring_ids = torch.arange(1, batch + 1)[:, None]
    graph = VerificationGraph(
        ("tail",), torch.empty(1, batch * width, DIM + 1), {}, width
    )
    request_ids = [str(row) for row in range(batch)]
    eager_state, graph_state = VerificationState(), VerificationState()

    # Use the production graph assembler with a deterministic projection.
    replay.token_key_projection = lambda row: (row,)
    replay.token_key_norm = lambda row: row
    replay._decode_history = lambda *args, **kwargs: None

    def verify(cursors, iteration):
        spans = tuple(
            QuerySpan(row * width, (row + 1) * width, start, start + width)
            for row, start in enumerate(cursors)
        )
        requests = list(zip(request_ids, spans, strict=True))
        lengths = torch.tensor([span.length for span in spans], dtype=torch.int32)
        entries = [
            keys(span.prefix, span.length, 1 + row + iteration * 80, "cpu")
            for row, span in enumerate(spans)
        ]
        meta = FlashMetadata(
            spans, table, torch.zeros(batch * width, dtype=torch.int64),
            batch * width, torch.arange(batch + 1) * width, lengths,
            False, None, None,
        )
        ring = FlashMetadata(
            spans, ring_ids, meta.slot_mapping, batch * width, meta.query_start_loc,
            lengths, False, None, None,
        )
        finalized = {request_id: span.prefix + 1 for request_id, span in requests}

        with eager_state.forward(
            {"tail": ring}, {"tail": eager.tail_cache}, requests, request_ids, finalized
        ):
            for row, (span, (key, pos)) in enumerate(zip(spans, entries, strict=True)):
                FlashCompressedIndexer.prepare_groups(
                    eager, key, pos, torch.ones(32), span, table[row], ring_ids[row],
                    ring.verification,
                )
        with graph_state.forward(
            {"tail": ring}, {"tail": replay.tail_cache},
            requests, request_ids, finalized, graph,
        ):
            key, pos = (torch.cat(rows) for rows in zip(*entries, strict=True))
            FlashCompressedIndexer.verify_groups(
                replay, key, key, pos, torch.ones(32), pos, pos, meta, ring,
                pos.int() + 1, meta.slot_mapping, graph.layer_tails["tail"], width,
            )
        # The next forward must read an owned staging copy, even if the
        # captured storage has since been replayed for a different request.
        graph.tails.fill_(-1000)
        assert torch.equal(eager.tail_cache.kv_cache, replay.tail_cache.kv_cache)
        assert torch.equal(eager.k_cache.kv_cache, replay.k_cache.kv_cache)

    cursors = [252 + row for row in range(batch)]
    for row, cursor in enumerate(cursors):
        for indexer in (eager, replay):
            prepare(indexer, table[row], 0, cursor, ring_id=ring_ids[row])
    for iteration, advance in enumerate((6, 3, 1, 0, 2)):
        verify(cursors, iteration)
        # 6, 3 and 1 inputs correspond to full, partial and zero proposal
        # acceptance respectively. A same-cursor retry discards all staging.
        cursors = [cursor + advance for cursor in cursors]
    for state in (eager_state, graph_state):
        state.forget(request_ids)
    # Resume at an aligned cache hit with new private ring pages.
    ring_ids.add_(batch)
    cursors = [cursor // RATIO * RATIO for cursor in cursors]
    verify(cursors, 5)
    verify([cursor + 3 for cursor in cursors], 6)

    # Switch from a verification graph to one ordinary row per request,
    # then back. Compare the committed ring and the just-closed groups.
    cursors = [cursor + 4 for cursor in cursors]
    spans = [
        QuerySpan(row, row + 1, cursor, cursor + 1)
        for row, cursor in enumerate(cursors)
    ]
    requests = list(zip(request_ids, spans, strict=True))
    for indexer, state, plain_graph in (
        (eager, eager_state, False), (replay, graph_state, True)
    ):
        indexer.verification = state
        _, meta = forward_scope(indexer, requests, ring_ids)
        with state.forward(
            {"tail": meta}, {"tail": indexer.tail_cache}, requests, (),
            {req_id: span.length for req_id, span in requests},
            plain_decode_graph=plain_graph,
        ):
            for row, span in enumerate(spans):
                data, positions = keys(span.prefix, span.length, 47 + row, "cpu")
                FlashCompressedIndexer.prepare_groups(
                    indexer, data, positions, torch.ones(32), span,
                    table[row], ring_ids[row], meta.verification,
                )
    assert torch.equal(eager.tail_cache.kv_cache, replay.tail_cache.kv_cache)
    assert torch.equal(eager.k_cache.kv_cache, replay.k_cache.kv_cache)
    verify([cursor + 1 for cursor in cursors], 7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_paged_reads_after_rejection(device):
    if device == "cpu" and not INTERPRET:
        pytest.skip("Triton CPU interpreter required")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # The SM70 writer uses PTX inline assembly. On CPU, pack the same layout
    # with Torch and run the actual paged reader in the interpreter.
    table = torch.tensor([[3, 1, 2]], device=device)
    start = PAGE - 2
    latent = torch.arange(512, device=device).half().remainder(17) / 32
    phase = torch.arange(64, device=device).half().remainder(7) / 32
    caches = [
        torch.zeros(4, PAGE, 656, device=device, dtype=torch.uint8) for _ in range(2)
    ]

    def write(cache, positions, offset):
        pos = torch.tensor(positions, device=device)
        values = (pos % 17).half()[:, None] / 16 + offset
        slots = table[0, pos.clamp_min(0) // PAGE] * PAGE + pos % PAGE
        if device == "cuda":
            cache_fp8_ds_mla(
                latent + values, phase + values, cache, slots.masked_fill(pos < 0, -1)
            )
        else:
            live = pos >= 0
            cache[slots[live] // PAGE, slots[live] % PAGE] = pack_fp8_ds_mla_torch(
                latent + values[live], phase + values[live]
            )

    for cache in caches:
        write(cache, list(range(start)), 1)
    write(caches[0], list(range(start, start + 6)) + [-1], 2)
    for p in range(start, start + 2):
        write(caches[1], [p], 2)
    for cache in caches:
        write(cache, [start + 2], 3)
    indices = torch.empty(1, 16, device=device, dtype=torch.int32)
    counts = torch.empty(1, device=device, dtype=torch.int32)
    length = torch.tensor([start + 3], device=device, dtype=torch.int32)
    decode_slots(length, length, table, None, indices, counts, PAGE, 1)
    live = indices[0, :16].long()
    payloads = [cache[live // PAGE, live % PAGE] for cache in caches]
    assert torch.equal(*payloads)
    probe = torch.ones(1, 1, 576, device=device, dtype=torch.float16)
    sink = torch.zeros(1, device=device)
    outputs = [
        paged_attention(probe, cache, indices, counts, sink, 0.05, 512)
        for cache in caches
    ]
    assert torch.equal(*outputs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_scratch_ignores_stale_and_padding_rows(device):
    if device == "cpu" and not INTERPRET:
        pytest.skip("Triton CPU interpreter required")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    lengths = torch.tensor([17, 0], dtype=torch.int32, device=device)
    active = torch.tensor([PAGE + 16, -1], device=device)
    table = torch.tensor([[1], [0]], device=device)
    cache = torch.ones(2, PAGE, 64, dtype=torch.float16, device=device)
    q = torch.ones(2, 1, 64, dtype=torch.float16, device=device)
    sink = torch.zeros(1, device=device)

    def replay(fill):
        indices = torch.full((2, 64), 999, dtype=torch.int32, device=device)
        counts = torch.full((2,), 64, dtype=torch.int32, device=device)
        scratch = [
            torch.full(shape, fill, dtype=dtype, device=device)
            for shape, dtype in decode_workspace_specs(2, 1, 64, 32)
        ]
        decode_slots(lengths, active, table, None, indices, counts, PAGE, 1)
        assert counts.tolist() == [17, 0]
        assert (indices[1] == -1).all()
        return paged_attention_splitk(q, cache, indices, counts, sink, 0.05, *scratch)

    expected = replay(0.0)
    actual = replay(float("nan"))
    assert torch.equal(expected, actual)
    assert (actual[1] == 0).all()


@pytest.mark.parametrize("rank", range(4))
def test_sharded_pool_ownership_preserves_accepted_prefix_near_1m(rank):
    # Model the local DCP4 group table at the last global page. Only touch
    # that page and check each rank's privately owned group and replicated tail.
    start = 1048563
    block = start // PAGE
    table = torch.zeros(block + 1, dtype=torch.int64)
    table[block] = 1
    ring = torch.tensor([1])
    for accepted in range(7):
        actual, expected = (new_indexer() for _ in range(2))
        for indexer in (actual, expected):
            indexer.dcp_group = object()
            indexer.dcp_world = 4
            indexer.dcp_rank = rank
            # Keep raw keys small. Logical positions still drive page ownership.
            prepare(indexer, table, start - 3, start, offset=-start, ring_id=ring)
        prepare(
            actual, table, start, start + 6, offset=-start, verify=True, ring_id=ring
        )
        committed = start + accepted
        for p in range(start, committed):
            prepare(expected, table, p, p + 1, offset=-start, ring_id=ring)
        visible = (committed // RATIO + 3 - rank) // 4
        groups = torch.arange((start - 3) // (4 * RATIO), visible)
        owned = groups % 64
        assert torch.equal(
            actual.k_cache.kv_cache[1, owned], expected.k_cache.kv_cache[1, owned]
        )
        for p in range(committed, start + 9):
            for indexer in (actual, expected):
                prepare(indexer, table, p, p + 1, offset=50 - start, ring_id=ring)
        assert torch.equal(actual.k_cache.kv_cache, expected.k_cache.kv_cache)
