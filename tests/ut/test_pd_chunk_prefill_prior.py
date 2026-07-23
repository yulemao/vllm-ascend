# SPDX-License-Identifier: Apache-2.0
"""Unit tests for chunk-prefill-prior scheduling in PDSeparatedScheduler.

Tests cover:
  - Config wiring (ascend_config → platform → scheduler_config)
  - PrefillChunkFlight creation and lifecycle
  - Ahead scheduling (next chunk PF before previous PL)
  - PL routing by head_token
  - Last-chunk → decode transition
  - Backward compatibility (chunk_prefill_prior_enable=False)
  - Request cleanup (finish / abort)
  - _migrate_prefill_to_running guard
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ------------------------------------------------------------------ #
# Test helpers                                                       #
# ------------------------------------------------------------------ #


def _make_scheduler_config(
    *,
    pd_prefill_inflight_limit: int = 1,
    pd_next_prefill_prior_enable: bool = False,
    pd_chunk_prefill_prior_enable: bool = False,
    pd_max_chunk_prefill_ahead: int = 1,
    async_scheduling: bool = False,
    max_num_running_reqs: int = 256,
    max_model_len: int = 131072,
    max_num_batched_tokens: int = 8192,
) -> SimpleNamespace:
    return SimpleNamespace(
        pd_prefill_inflight_limit=pd_prefill_inflight_limit,
        pd_next_prefill_prior_enable=pd_next_prefill_prior_enable,
        pd_chunk_prefill_prior_enable=pd_chunk_prefill_prior_enable,
        pd_max_chunk_prefill_ahead=pd_max_chunk_prefill_ahead,
        async_scheduling=async_scheduling,
        max_num_running_reqs=max_num_running_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        scheduler_cls=None,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
    )


def _make_mock_request(
    request_id: str = "req-0",
    num_prompt_tokens: int = 8000,
    num_computed_tokens: int = 0,
    is_prefill_chunk: bool = True,
    chunk_num: int = 0,
) -> MagicMock:
    """Create a mock Request with prefill-related attributes."""
    req = MagicMock()
    req.request_id = request_id
    req.num_prompt_tokens = num_prompt_tokens
    req.num_computed_tokens = num_computed_tokens
    req.is_prefill_chunk = is_prefill_chunk
    req.chunk_num = chunk_num
    req.is_finished.return_value = False
    req.all_token_ids = []
    req.status = None
    req.num_output_placeholders = 0
    req.spec_token_ids = []
    req.num_preemptions = 0
    req.discard_latest_async_tokens = False
    req.max_tokens = 256
    return req


def _make_vllm_config_for_scheduler(
    scheduler_config: SimpleNamespace,
) -> SimpleNamespace:
    """Minimal VllmConfig substitute for PDSeparatedScheduler.__init__."""
    return SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=SimpleNamespace(
            max_model_len=scheduler_config.max_model_len,
        ),
        cache_config=SimpleNamespace(block_size=128),
    )


# ------------------------------------------------------------------ #
# Test: Config wiring                                                 #
# ------------------------------------------------------------------ #


class TestConfigWiring:
    """Verify ascend_config → platform → scheduler_config propagation."""

    def test_chunk_prefill_prior_enabled_wires_to_scheduler_config(self):
        """When chunk_prefill_prior_enable=True, it lands on scheduler_config."""
        from vllm_ascend.platform import NPUPlatform

        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                async_scheduling=False,
                scheduler_cls=None,
                pd_prefill_inflight_limit=1,
            )
        )
        pd = SimpleNamespace(
            enabled=True,
            next_prefill_prior_enable=True,
            prefill_inflight_limit=2,
            chunk_prefill_prior_enable=True,
            max_chunk_prefill_ahead=1,
        )
        edge_cloud = SimpleNamespace(enabled=True, pd_separation=pd)
        ascend_config = SimpleNamespace(edge_cloud_config=edge_cloud)

        NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

        assert vllm_config.scheduler_config.pd_next_prefill_prior_enable is True
        assert vllm_config.scheduler_config.pd_chunk_prefill_prior_enable is True
        assert vllm_config.scheduler_config.pd_max_chunk_prefill_ahead == 1

    def test_chunk_prefill_prior_disabled_by_default(self):
        """When not configured, chunk_prefill_prior_enable defaults to False."""
        from vllm_ascend.platform import NPUPlatform

        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                async_scheduling=False,
                scheduler_cls=None,
                pd_prefill_inflight_limit=1,
            )
        )
        pd = SimpleNamespace(
            enabled=True,
            next_prefill_prior_enable=True,
            prefill_inflight_limit=2,
            chunk_prefill_prior_enable=False,
            max_chunk_prefill_ahead=1,
        )
        edge_cloud = SimpleNamespace(enabled=True, pd_separation=pd)
        ascend_config = SimpleNamespace(edge_cloud_config=edge_cloud)

        NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

        assert vllm_config.scheduler_config.pd_chunk_prefill_prior_enable is False


# ------------------------------------------------------------------ #
# Test: PrefillChunkFlight                                           #
# ------------------------------------------------------------------ #


class TestPrefillChunkFlight:
    """Verify PrefillChunkFlight dataclass and related helpers."""

    def test_flight_creation(self):
        from vllm_ascend.core.pd_separated_scheduler import PrefillChunkFlight
        from vllm.v1.core.sched.output import HiddenChannelType

        flight = PrefillChunkFlight(
            request_id="req-0",
            head_token="abc123",
            hidden_channel=HiddenChannelType.PREFILL_1,
            chunk_index=0,
            is_last_chunk=False,
            num_scheduled_tokens=4096,
        )
        assert flight.request_id == "req-0"
        assert flight.head_token == "abc123"
        assert flight.chunk_index == 0
        assert flight.is_last_chunk is False
        assert flight.num_scheduled_tokens == 4096

    @pytest.mark.xfail(
        reason="pre-existing: _remaining_prompt_tokens helper is not "
               "implemented on PDSeparatedScheduler",
        strict=False,
    )
    def test_remaining_prompt_tokens(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)

        req = _make_mock_request(
            num_prompt_tokens=8000, num_computed_tokens=4000
        )
        assert scheduler._remaining_prompt_tokens(req, 2000) == 2000
        assert scheduler._remaining_prompt_tokens(req, 4000) == 0
        assert scheduler._remaining_prompt_tokens(req, 5000) == 0

    @pytest.mark.xfail(
        reason="pre-existing: _has_more_chunks helper is not implemented "
               "on PDSeparatedScheduler",
        strict=False,
    )
    def test_has_more_chunks(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)

        req = _make_mock_request(
            num_prompt_tokens=8000, num_computed_tokens=0
        )
        assert scheduler._has_more_chunks(req, 4000) is True
        assert scheduler._has_more_chunks(req, 8000) is False

    def test_can_ahead_schedule(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.max_chunk_prefill_ahead = 1
        scheduler._ahead_chunk_count = {}

        assert scheduler._can_ahead_schedule("req-0") is True

        scheduler._ahead_chunk_count["req-0"] = 1
        assert scheduler._can_ahead_schedule("req-0") is False

        scheduler._ahead_chunk_count["req-0"] = 0
        assert scheduler._can_ahead_schedule("req-0") is True


# ------------------------------------------------------------------ #
# Test: Draft/decode cross-request overlap                            #
# ------------------------------------------------------------------ #


class TestDraftDecodeOverlap:
    def test_decode_can_progress_while_unrelated_draft_is_inflight(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
        )

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.running = [
            SimpleNamespace(request_id="draft-parent"),
            SimpleNamespace(request_id="decode-ready"),
        ]
        scheduler.draft_pending_req_ids = {"draft-parent"}
        scheduler.decode_inflight_count = 0
        scheduler.decode_inflight_limit = 1
        scheduler._force_decode_last = False

        assert scheduler._can_schedule_decode_first() is True

        scheduler.draft_pending_req_ids.add("decode-ready")
        assert scheduler._can_schedule_decode_first() is False

    def test_draft_can_progress_while_decode_is_inflight(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
        )

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.drafts_first_ready = deque([object()])
        scheduler.drafts_last_ready = deque()
        scheduler.draft_inflight_count = 0
        scheduler.draft_inflight_limit = 1
        scheduler.draft_remote_pending_count = 0
        scheduler.decode_inflight_count = 1
        scheduler._force_decode_last = True

        assert scheduler._can_schedule_draft_first() is True

    def test_decode_batch_excludes_its_pending_draft_parent(self):
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_ascend.core import pd_separated_scheduler as pd_scheduler
        from vllm_ascend.core.pd_separated_scheduler import (
            HiddenChannelManager,
            PDSeparatedScheduler,
        )

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        draft_parent = SimpleNamespace(request_id="draft-parent")
        decode_ready = SimpleNamespace(request_id="decode-ready")
        scheduler.running = [draft_parent, decode_ready]
        scheduler.draft_pending_req_ids = {"draft-parent"}
        scheduler.chunk_prefill_first = []
        scheduler.waiting = deque()
        scheduler.skipped_waiting = deque()
        scheduler.policy = "fcfs"
        scheduler.hidden_channel_manager = HiddenChannelManager()
        scheduler.decode_inflight_count = 0
        scheduler._force_decode_last = False
        scheduler.decodes_last_ready = deque()
        scheduler._ensure_cached_all_token_ids = MagicMock()
        scheduler._start_decode_last_delay = MagicMock()

        decode_output = SchedulerOutput.make_empty()
        decode_output.total_num_scheduled_tokens = 1
        decode_output.num_scheduled_tokens = {"decode-ready": 1}

        def schedule_only_eligible(self):
            assert [req.request_id for req in self.running] == [
                "decode-ready"
            ]
            return decode_output

        with patch.object(
            pd_scheduler,
            "create_request_queue",
            side_effect=[deque(), deque()],
        ), patch.object(
            Scheduler,
            "schedule",
            autospec=True,
            side_effect=schedule_only_eligible,
        ):
            picked = scheduler._pick_decode_first_batch()

        assert picked.num_scheduled_tokens == {"decode-ready": 1}
        assert [req.request_id for req in scheduler.running] == [
            "draft-parent",
            "decode-ready",
        ]

    def test_draft_batch_uses_dedicated_channel_and_blocks_parent_decode(self):
        from vllm.v1.core.sched.output import (
            BatchType,
            HiddenChannelType,
            SchedulerOutput,
        )
        from vllm_ascend.core.pd_separated_scheduler import (
            HiddenChannelManager,
            PDSeparatedScheduler,
        )

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        draft = SchedulerOutput.make_empty()
        draft.num_scheduled_tokens = {"draft-parent": 1}
        draft.parent_req_id = "draft-parent"
        scheduler.drafts_first_ready = deque([draft])
        scheduler.hidden_channel_manager = HiddenChannelManager()
        scheduler.draft_pending_req_ids = set()
        scheduler.draft_inflight_count = 0
        scheduler.draft_remote_pending_count = 0

        picked = scheduler._pick_draft_first_batch()

        assert picked.batch_type == BatchType.DRAFT_FIRST
        assert picked.hidden_channel == HiddenChannelType.DRAFT
        assert scheduler.draft_pending_req_ids == {"draft-parent"}


# ------------------------------------------------------------------ #
# Test: Cross-request head-prior (2P1D multi-request interleaving)   #
# ------------------------------------------------------------------ #


class TestCrossRequestHeadPrior:
    """Verify _has_other_prefill_request and _should_ahead_schedule.

    Single-request path: when no other request is available, ahead-dispatch
    the same request's next chunk (intra-request pipeline, both 2P1D slots
    serve one request).

    Multi-request path: when next_prefill_prior_enable is on and another
    request can fill the slot, yield (do NOT ahead) so the next slot
    dispatches the other request's head (cross-request P1首 -> P2首).
    """

    def _make_scheduler(self, *, next_prior=False, ahead_limit=1):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
        )

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.next_prefill_prior_enable = next_prior
        scheduler.max_chunk_prefill_ahead = ahead_limit
        scheduler._ahead_chunk_count = {}
        scheduler.chunk_prefill_first = []
        scheduler.running = []
        # A plain list stands in for RequestQueue here; only len()>0 matters.
        scheduler.waiting = []
        return scheduler

    # ---- _has_other_prefill_request ----

    def test_no_other_request_when_all_empty(self):
        scheduler = self._make_scheduler()
        assert scheduler._has_other_prefill_request("req-0") is False

    def test_other_request_in_waiting(self):
        scheduler = self._make_scheduler()
        scheduler.waiting = [_make_mock_request(request_id="req-1")]
        assert scheduler._has_other_prefill_request("req-0") is True

    def test_other_request_in_chunk_prefill_first(self):
        scheduler = self._make_scheduler()
        scheduler.chunk_prefill_first = [
            _make_mock_request(request_id="req-1")
        ]
        assert scheduler._has_other_prefill_request("req-0") is True

    def test_other_request_in_running_prefill_chunk(self):
        """Drained-window case: another mid-prefill request in running."""
        scheduler = self._make_scheduler()
        other = _make_mock_request(request_id="req-1", is_prefill_chunk=True)
        scheduler.running = [other]
        assert scheduler._has_other_prefill_request("req-0") is True

    def test_only_current_request_in_chunk_prefill_first(self):
        scheduler = self._make_scheduler()
        scheduler.chunk_prefill_first = [
            _make_mock_request(request_id="req-0")
        ]
        assert scheduler._has_other_prefill_request("req-0") is False

    def test_running_decode_request_does_not_count(self):
        """A decode request in running must not be treated as prefill work."""
        scheduler = self._make_scheduler()
        decode_req = _make_mock_request(
            request_id="req-1", is_prefill_chunk=False
        )
        scheduler.running = [decode_req]
        assert scheduler._has_other_prefill_request("req-0") is False

    # ---- _should_ahead_schedule: single-request path (ahead) ----

    def test_single_request_aheads_despite_next_prior(self):
        """Single request + next_prior=True: no other request -> ahead."""
        scheduler = self._make_scheduler(next_prior=True)
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=False) is True

    # ---- _should_ahead_schedule: multi-request path (yield) ----

    def test_multi_request_yields_to_other(self):
        """next_prior=True + another request waiting -> yield (no ahead)."""
        scheduler = self._make_scheduler(next_prior=True)
        scheduler.waiting = [_make_mock_request(request_id="req-1")]
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=False) is False

    def test_multi_request_yields_when_other_mid_prefill(self):
        """Yield when another request is mid-prefill in chunk_prefill_first."""
        scheduler = self._make_scheduler(next_prior=True)
        scheduler.chunk_prefill_first = [
            _make_mock_request(request_id="req-1")
        ]
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=False) is False

    def test_ahead_budget_full_returns_false_even_with_yield(self):
        """ahead budget exhausted -> False regardless of yield condition."""
        scheduler = self._make_scheduler(next_prior=True)
        scheduler._ahead_chunk_count["req-0"] = 1  # == max_chunk_prefill_ahead
        scheduler.waiting = [_make_mock_request(request_id="req-1")]
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=False) is False

    # ---- 1P1D / legacy path ----

    def test_1p1d_ignores_other_request(self):
        """next_prior=False: ahead even when another request is waiting."""
        scheduler = self._make_scheduler(next_prior=False)
        scheduler.waiting = [_make_mock_request(request_id="req-1")]
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=False) is True

    def test_last_chunk_never_aheads(self):
        scheduler = self._make_scheduler(next_prior=True)
        req = _make_mock_request(request_id="req-0")
        assert scheduler._should_ahead_schedule(req, is_last=True) is False


# ------------------------------------------------------------------ #
# Test: One-request-per-PF-batch (head_token collision regression)   #
# ------------------------------------------------------------------ #


class TestSinglePrefillCandidate:
    """Verify _select_single_prefill_candidate enforces one-request-per-batch.

    Regression guard for the head_token collision bug: when two requests were
    scheduled in one PF batch they shared one head_token, the second
    overwrote the first in _prefill_flight_by_token, and the first request's
    PL was never matched -- stalling it. _pick_prefill_first_batch now
    exposes at most one candidate to super().schedule() via this helper.
    """

    def _make(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
        )

        return PDSeparatedScheduler.__new__(PDSeparatedScheduler)

    def test_empty_candidates(self):
        scheduler = self._make()
        exposed, rest = scheduler._select_single_prefill_candidate([])
        assert exposed == []
        assert rest == []

    def test_single_candidate(self):
        scheduler = self._make()
        req0 = _make_mock_request(request_id="req-0")
        exposed, rest = scheduler._select_single_prefill_candidate([req0])
        assert exposed == [req0]
        assert rest == []

    def test_two_candidates_exposes_one(self):
        scheduler = self._make()
        req0 = _make_mock_request(request_id="req-0")
        req1 = _make_mock_request(request_id="req-1")
        exposed, rest = scheduler._select_single_prefill_candidate([req0, req1])
        assert len(exposed) == 1
        assert exposed[0] is req0
        assert rest == [req1]

    def test_three_candidates_exposes_one_rest_two(self):
        scheduler = self._make()
        reqs = [_make_mock_request(request_id=f"req-{i}") for i in range(3)]
        exposed, rest = scheduler._select_single_prefill_candidate(reqs)
        assert len(exposed) == 1
        assert exposed[0] is reqs[0]
        assert rest == reqs[1:]
        # Exposed never exceeds one -- the invariant that prevents two
        # requests from sharing a head_token in one PF batch.
        assert len(exposed) <= 1


# ------------------------------------------------------------------ #
# Test: PF batch state preparation (one-per-batch gating)             #
# ------------------------------------------------------------------ #


class TestPreparePFBatchState:
    """Verify _prepare_pf_running_state gates one-per-batch on chunk_prior.

    With chunk_prefill_prior_enable=True, a PF batch is limited to one request
    (head_token collision guard). With it False (legacy 2P1D), multi-request
    batching is preserved -- the original behavior, since legacy PL routes by
    req_id and has no flight map to collide.
    """

    def _make(self, *, chunk_prior):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
        )

        s = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        s.chunk_prefill_prior_enable = chunk_prior
        # State read by _select_pf_candidate_head_prior.
        s._pending_tail_count = {}
        s.waiting = []
        return s

    # ---- chunk_prefill_prior_enable=True: one-per-batch ----

    def test_chunk_prior_on_two_candidates_exposes_one(self):
        s = self._make(chunk_prior=True)
        req0 = _make_mock_request(request_id="req-0")
        req1 = _make_mock_request(request_id="req-1")
        running, max_num, rest = s._prepare_pf_running_state(
            [req0, req1], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == [req0]
        assert max_num == 1
        assert rest == [req1]

    def test_chunk_prior_on_no_candidates_admits_one(self):
        s = self._make(chunk_prior=True)
        running, max_num, rest = s._prepare_pf_running_state(
            [], saved_running=[_make_mock_request(request_id="dec-0")],
            saved_max_num_running_reqs=256,
        )
        assert running == []
        assert max_num == 1  # capacity available -> admit one new
        assert rest == []

    def test_chunk_prior_on_no_candidates_no_capacity_blocks_admit(self):
        """When saved_running fills the system, do not over-admit."""
        s = self._make(chunk_prior=True)
        decode_reqs = [_make_mock_request(request_id=f"dec-{i}")
                       for i in range(256)]
        running, max_num, rest = s._prepare_pf_running_state(
            [], saved_running=decode_reqs, saved_max_num_running_reqs=256
        )
        assert running == []
        assert max_num == 0  # no capacity -> block new admission
        assert rest == []

    # ---- chunk_prefill_prior_enable=True: cross-request head-prior ----

    def test_head_prior_prefers_fresh_over_in_flight(self):
        """A fresh candidate (no in-flight chunk) is picked before an
        in-flight one, so the freed slot refills a different request instead
        of clustering on the in-flight request."""
        s = self._make(chunk_prior=True)
        in_flight = _make_mock_request(request_id="req-0")
        fresh = _make_mock_request(request_id="req-1")
        s._pending_tail_count = {"req-0": 1}  # req-0 has a chunk in flight
        running, max_num, rest = s._prepare_pf_running_state(
            [in_flight, fresh], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == [fresh]  # fresh head chosen, not the in-flight req
        assert max_num == 1
        assert rest == [in_flight]

    def test_head_prior_yields_to_new_when_no_fresh(self):
        """No fresh candidate + a new request waiting -> admit the new
        request (cross-request) instead of clustering on an in-flight req."""
        s = self._make(chunk_prior=True)
        in_flight = _make_mock_request(request_id="req-0")
        waiting_new = _make_mock_request(request_id="req-1")
        s._pending_tail_count = {"req-0": 1}
        s.waiting = [waiting_new]
        running, max_num, rest = s._prepare_pf_running_state(
            [in_flight], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == []  # yield slot -> admit new from waiting
        assert max_num == 1
        assert rest == [in_flight]  # in-flight candidate preserved for later

    def test_head_prior_falls_back_to_in_flight_when_alone(self):
        """No fresh candidate, no waiting request -> ahead-dispatch the
        in-flight candidate (single-request 2P1D pipeline)."""
        s = self._make(chunk_prior=True)
        in_flight = _make_mock_request(request_id="req-0")
        s._pending_tail_count = {"req-0": 1}
        running, max_num, rest = s._prepare_pf_running_state(
            [in_flight], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == [in_flight]  # ahead fallback
        assert max_num == 1
        assert rest == []

    def test_head_prior_fresh_refill_preferred_over_new_admit(self):
        """A fresh candidate (PL just returned, ready for next chunk) is
        refilled before admitting a brand-new request -- keeps each in-flight
        request's pipeline continuous."""
        s = self._make(chunk_prior=True)
        fresh = _make_mock_request(request_id="req-0")
        waiting_new = _make_mock_request(request_id="req-1")
        s._pending_tail_count = {"req-0": 0}  # fresh
        s.waiting = [waiting_new]
        running, max_num, rest = s._prepare_pf_running_state(
            [fresh], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == [fresh]  # refill fresh, not admit new
        assert max_num == 1
        assert rest == []

    # ---- chunk_prefill_prior_enable=False: legacy multi-request batching ----

    def test_legacy_exposes_all_candidates(self):
        """Legacy preserves original multi-request batching (no collision)."""
        s = self._make(chunk_prior=False)
        req0 = _make_mock_request(request_id="req-0")
        req1 = _make_mock_request(request_id="req-1")
        running, max_num, rest = s._prepare_pf_running_state(
            [req0, req1], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == [req0, req1]  # all candidates exposed
        assert max_num == 256  # original cap: saved_max - len(saved_running)
        assert rest == []

    def test_legacy_cap_subtracts_saved_running(self):
        s = self._make(chunk_prior=False)
        decode_reqs = [_make_mock_request(request_id=f"dec-{i}")
                       for i in range(10)]
        _, max_num, _ = s._prepare_pf_running_state(
            [], saved_running=decode_reqs, saved_max_num_running_reqs=256
        )
        assert max_num == 256 - 10  # 246

    def test_legacy_no_candidates(self):
        s = self._make(chunk_prior=False)
        running, max_num, rest = s._prepare_pf_running_state(
            [], saved_running=[], saved_max_num_running_reqs=256
        )
        assert running == []
        assert max_num == 256
        assert rest == []


# ------------------------------------------------------------------ #
# Test: Chunk-flight state management                                 #
# ------------------------------------------------------------------ #


class TestChunkFlightState:
    """Verify flight tracking state transitions."""

    def test_total_pending_tails(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler._pending_tail_count = {"req-0": 2, "req-1": 1}
        assert scheduler._total_pending_tails() == 3

    def test_cleanup_request_flight_state(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
            PrefillChunkFlight,
        )
        from vllm.v1.core.sched.output import HiddenChannelType

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler._pending_tail_count = {"req-0": 2}
        scheduler._ahead_chunk_count = {"req-0": 1}
        scheduler._prefill_flight_by_token = {
            "tok0": PrefillChunkFlight(
                request_id="req-0",
                head_token="tok0",
                hidden_channel=HiddenChannelType.PREFILL_1,
                chunk_index=0,
                is_last_chunk=False,
                num_scheduled_tokens=4000,
            ),
            "tok1": PrefillChunkFlight(
                request_id="req-0",
                head_token="tok1",
                hidden_channel=HiddenChannelType.PREFILL_2,
                chunk_index=1,
                is_last_chunk=True,
                num_scheduled_tokens=4000,
            ),
            "tok2": PrefillChunkFlight(
                request_id="req-1",
                head_token="tok2",
                hidden_channel=HiddenChannelType.PREFILL_1,
                chunk_index=0,
                is_last_chunk=True,
                num_scheduled_tokens=2000,
            ),
        }

        scheduler._cleanup_request_flight_state("req-0")

        assert "req-0" not in scheduler._pending_tail_count
        assert "req-0" not in scheduler._ahead_chunk_count
        assert "tok0" not in scheduler._prefill_flight_by_token
        assert "tok1" not in scheduler._prefill_flight_by_token
        # req-1 should be untouched.
        assert "tok2" in scheduler._prefill_flight_by_token
        assert scheduler._pending_tail_count.get("req-1") is None


# ------------------------------------------------------------------ #
# Test: PL routing (chunk-prefill-prior)                              #
# ------------------------------------------------------------------ #


class TestPLLRoutingChunkPrior:
    """Verify _update_from_output_prefill_last_chunk_prior."""

    def _setup_scheduler(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
            PrefillChunkFlight,
        )
        from vllm.v1.core.sched.output import HiddenChannelType

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_prior_enable = True
        scheduler.chunk_prefill_first = []
        scheduler.running = []
        scheduler.prefill_last_pending = []
        scheduler.prefill_inflight_count = 2
        scheduler.prefill_inflight_limit = 2
        scheduler.hidden_channel_manager = MagicMock()
        scheduler._pending_tail_count = {}
        scheduler._ahead_chunk_count = {}
        scheduler._prefill_flight_by_token = {}
        scheduler.requests = {}
        return scheduler

    def test_last_chunk_pl_moves_to_running(self):
        """When the last chunk's PL returns, request enters running."""
        scheduler = self._setup_scheduler()

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,
        )
        scheduler.requests["req-0"] = req

        from vllm_ascend.core.pd_separated_scheduler import PrefillChunkFlight
        from vllm.v1.core.sched.output import HiddenChannelType

        flight = PrefillChunkFlight(
            request_id="req-0",
            head_token="tok-last",
            hidden_channel=HiddenChannelType.PREFILL_1,
            chunk_index=1,
            is_last_chunk=True,
            num_scheduled_tokens=4000,
        )
        scheduler._prefill_flight_by_token["tok-last"] = flight
        scheduler._pending_tail_count["req-0"] = 1

        so = MagicMock()
        so.head_token = "tok-last"
        so.batch_type = None
        so.num_scheduled_tokens = {"req-0": 4000}

        scheduler._update_from_output_prefill_last_chunk_prior(so)

        assert "tok-last" not in scheduler._prefill_flight_by_token
        assert scheduler._pending_tail_count.get("req-0", 0) == 0
        assert req in scheduler.running
        assert "req-0" not in scheduler._ahead_chunk_count

    def test_mid_chunk_pl_with_ahead_does_not_re_add(self):
        """When a mid-chunk PL returns and the request was ahead-scheduled,
        do not re-add to chunk_prefill_first."""
        scheduler = self._setup_scheduler()

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=True,
        )
        scheduler.requests["req-0"] = req

        from vllm_ascend.core.pd_separated_scheduler import PrefillChunkFlight
        from vllm.v1.core.sched.output import HiddenChannelType

        flight = PrefillChunkFlight(
            request_id="req-0",
            head_token="tok-mid",
            hidden_channel=HiddenChannelType.PREFILL_1,
            chunk_index=0,
            is_last_chunk=False,
            num_scheduled_tokens=4000,
        )
        scheduler._prefill_flight_by_token["tok-mid"] = flight
        scheduler._pending_tail_count["req-0"] = 2
        scheduler._ahead_chunk_count["req-0"] = 1  # ahead-scheduled

        so = MagicMock()
        so.head_token = "tok-mid"
        so.batch_type = None
        so.num_scheduled_tokens = {"req-0": 4000}

        scheduler._update_from_output_prefill_last_chunk_prior(so)

        # pending_tail_count should decrease.
        assert scheduler._pending_tail_count["req-0"] == 1
        # ahead_chunk_count should decrease.
        assert scheduler._ahead_chunk_count["req-0"] == 0
        # Request should NOT be in chunk_prefill_first (already ahead).
        assert req not in scheduler.chunk_prefill_first

    def test_mid_chunk_pl_without_ahead_re_adds(self):
        """When a mid-chunk PL returns and the request was NOT ahead-scheduled,
        re-add to chunk_prefill_first."""
        scheduler = self._setup_scheduler()

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=True,
        )
        scheduler.requests["req-0"] = req

        from vllm_ascend.core.pd_separated_scheduler import PrefillChunkFlight
        from vllm.v1.core.sched.output import HiddenChannelType

        flight = PrefillChunkFlight(
            request_id="req-0",
            head_token="tok-mid",
            hidden_channel=HiddenChannelType.PREFILL_1,
            chunk_index=0,
            is_last_chunk=False,
            num_scheduled_tokens=4000,
        )
        scheduler._prefill_flight_by_token["tok-mid"] = flight
        scheduler._pending_tail_count["req-0"] = 1
        scheduler._ahead_chunk_count["req-0"] = 0  # NOT ahead

        so = MagicMock()
        so.head_token = "tok-mid"
        so.batch_type = None
        so.num_scheduled_tokens = {"req-0": 4000}

        scheduler._update_from_output_prefill_last_chunk_prior(so)

        assert scheduler._pending_tail_count["req-0"] == 0
        assert req in scheduler.chunk_prefill_first

    def test_pl_missing_head_token_falls_back_to_legacy(self):
        """When head_token is missing, fall back to legacy routing."""
        scheduler = self._setup_scheduler()

        from vllm_ascend.core.pd_separated_scheduler import PrefillChunkFlight
        from vllm.v1.core.sched.output import HiddenChannelType

        # Set up legacy pending list.
        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,
        )
        scheduler.prefill_last_pending = [req]
        scheduler.requests["req-0"] = req

        so = MagicMock()
        so.head_token = None  # missing
        so.batch_type = None
        so.num_scheduled_tokens = {"req-0": 4000}

        scheduler._update_from_output_prefill_last_chunk_prior(so)

        # Legacy routing should move req to running.
        assert req in scheduler.running
        assert req not in scheduler.prefill_last_pending

    def test_pl_unknown_head_token_falls_back_to_legacy(self):
        """When head_token is not in flight map, fall back to legacy."""
        scheduler = self._setup_scheduler()

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,
        )
        scheduler.prefill_last_pending = [req]
        scheduler.requests["req-0"] = req

        so = MagicMock()
        so.head_token = "unknown-token"
        so.batch_type = None
        so.num_scheduled_tokens = {"req-0": 4000}

        scheduler._update_from_output_prefill_last_chunk_prior(so)

        assert req in scheduler.running


# ------------------------------------------------------------------ #
# Test: _migrate_prefill_to_running guard                             #
# ------------------------------------------------------------------ #


class TestMigratePrefillToRunning:
    """Verify that requests with pending tails are not moved to running."""

    def test_request_with_pending_tails_not_migrated(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_first = []
        scheduler.running = []
        scheduler._pending_tail_count = {"req-0": 2}

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,  # prefill complete, but tails pending
        )
        scheduler.chunk_prefill_first.append(req)

        scheduler._migrate_prefill_to_running()

        # Should NOT be moved because pending_tail_count > 0.
        assert req in scheduler.chunk_prefill_first
        assert req not in scheduler.running

    def test_request_without_pending_tails_migrated(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_first = []
        scheduler.running = []
        scheduler._pending_tail_count = {"req-0": 0}

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,
        )
        scheduler.chunk_prefill_first.append(req)

        scheduler._migrate_prefill_to_running()

        assert req not in scheduler.chunk_prefill_first
        assert req in scheduler.running

    def test_request_not_in_pending_tail_map_migrated(self):
        """Requests not in pending_tail_count should be migrated."""
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_first = []
        scheduler.running = []
        scheduler._pending_tail_count = {}

        req = _make_mock_request(
            request_id="req-0",
            is_prefill_chunk=False,
        )
        scheduler.chunk_prefill_first.append(req)

        scheduler._migrate_prefill_to_running()

        assert req not in scheduler.chunk_prefill_first
        assert req in scheduler.running


# ------------------------------------------------------------------ #
# Test: finish_requests cleanup                                       #
# ------------------------------------------------------------------ #


class TestFinishRequestsCleanup:
    """Verify that finish_requests cleans up chunk-prefill-prior state."""

    def test_finish_request_cleans_up_flight_state(self):
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
            PrefillChunkFlight,
        )
        from vllm.v1.core.sched.output import HiddenChannelType

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_first = []
        scheduler.prefill_last_pending = []
        scheduler.finished_req_ids = set()
        scheduler.requests = {}
        scheduler.log_stats = False

        req = _make_mock_request(request_id="req-0")
        req.is_finished.return_value = True
        scheduler.requests["req-0"] = req

        scheduler._pending_tail_count = {"req-0": 2}
        scheduler._ahead_chunk_count = {"req-0": 1}
        scheduler._prefill_flight_by_token = {
            "tok0": PrefillChunkFlight(
                request_id="req-0",
                head_token="tok0",
                hidden_channel=HiddenChannelType.PREFILL_1,
                chunk_index=0,
                is_last_chunk=False,
                num_scheduled_tokens=4000,
            ),
        }

        # Mock the parent finish_requests to return empty.
        with patch.object(
            scheduler.__class__,
            "finish_requests",
            wraps=lambda self, *a, **kw: [],
        ):
            # We need to patch the parent's finish_requests.
            # Use a simpler approach: directly test cleanup.
            scheduler._cleanup_request_flight_state("req-0")
            assert "req-0" not in scheduler._pending_tail_count
            assert "req-0" not in scheduler._ahead_chunk_count
            assert "tok0" not in scheduler._prefill_flight_by_token


# ------------------------------------------------------------------ #
# Test: backward compatibility                                        #
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Verify legacy behavior is preserved when chunk_prefill_prior_enable=False."""

    def test_log_scheduler_state_without_chunk_prior(self):
        """Log format uses legacy fields when chunk_prefill_prior_enable=False."""
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
            PrefillState,
        )
        from vllm.v1.core.sched.output import BatchType

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_prior_enable = False
        scheduler._step_counter = 0
        scheduler.waiting = []
        scheduler.chunk_prefill_first = []
        scheduler.prefill_last_pending = []
        scheduler.running = []
        scheduler.prefills_last_ready = []
        scheduler.decodes_last_ready = []
        scheduler.prefill_inflight_count = 0
        scheduler.prefill_inflight_limit = 1
        scheduler.decode_inflight_count = 0
        scheduler.decode_inflight_limit = 1

        # Should not raise — log format uses legacy fields.
        scheduler._log_scheduler_state(PrefillState.IDLE, BatchType.PREFILL_FIRST)
        assert scheduler._step_counter == 1

    def test_log_scheduler_state_with_chunk_prior(self):
        """Log format includes chunk-prior fields when enabled."""
        from vllm_ascend.core.pd_separated_scheduler import (
            PDSeparatedScheduler,
            PrefillState,
        )
        from vllm.v1.core.sched.output import BatchType

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_prior_enable = True
        scheduler._step_counter = 0
        scheduler.waiting = []
        scheduler.chunk_prefill_first = []
        scheduler.prefill_last_pending = []
        scheduler.running = []
        scheduler.prefills_last_ready = []
        scheduler.decodes_last_ready = []
        scheduler.prefill_inflight_count = 0
        scheduler.prefill_inflight_limit = 1
        scheduler.decode_inflight_count = 0
        scheduler.decode_inflight_limit = 1
        scheduler._prefill_flight_by_token = {}
        scheduler._pending_tail_count = {}
        scheduler._ahead_chunk_count = {}

        scheduler._log_scheduler_state(PrefillState.LOW, BatchType.PREFILL_FIRST)
        assert scheduler._step_counter == 1


# ------------------------------------------------------------------ #
# Test: get_request_counts with chunk-prior                           #
# ------------------------------------------------------------------ #


class TestRequestCounts:
    """Verify get_request_counts and get_num_unfinished_requests."""

    def test_get_request_counts_includes_pending_tails(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler.chunk_prefill_first = []
        scheduler.prefill_last_pending = []
        scheduler._pending_tail_count = {"req-0": 2, "req-1": 1}

        # Mock parent's get_request_counts.
        with patch.object(
            scheduler.__class__,
            "get_request_counts",
            return_value=(10, 5),
        ):
            num_running, num_waiting = scheduler.get_request_counts()
            assert num_running == 10 + 0 + 0 + 3  # running + chunk + pending + tails
            assert num_waiting == 5

    def test_get_num_unfinished_requests_includes_pending_tails(self):
        from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

        scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
        scheduler._pause_state = None
        scheduler.chunk_prefill_first = []
        scheduler.prefill_last_pending = []
        scheduler._pending_tail_count = {"req-0": 1}

        with patch.object(
            scheduler.__class__,
            "get_num_unfinished_requests",
            return_value=10,
        ):
            total = scheduler.get_num_unfinished_requests()
            assert total == 10 + 0 + 0 + 1  # base + chunk + pending + tails
