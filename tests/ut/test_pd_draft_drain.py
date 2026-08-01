# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DRAFT_FIRST -> DRAFT_LAST alternation invariant and the
drain path that pairs the cloud's DRAFT_LAST response when the owning request
finishes or is aborted mid-chain.

Regression coverage for the edge-side MTP draft deadlock fixes:

  * ``_can_schedule_draft_first`` (pre-generated branch) now respects
    ``_force_draft_last`` -- a second DRAFT_FIRST may not be picked until the
    preceding DRAFT_LAST is picked (which clears the flag).  Previously the
    pre-generated branch omitted this guard, so a DRAFT_LAST dropped without
    clearing the flag let a second DRAFT_FIRST through (two heads with no tail
    on the shared DECODE channel -> cloud ``irecv`` deadlock).
  * ``_is_stale_draft_output`` no longer exempts pre-generated dispatched
    chains: once the owning request is gone, future (not-yet-dispatched)
    DRAFT_FIRST heads are stale and must be skipped -- the edge can no longer
    produce their payload (draft context cleared) and the cloud would otherwise
    wait forever for data.
  * ``_pick_draft_last_batch`` never drops a dispatched DRAFT_LAST; it drains
    it (the cloud always ``isend``s a response, so the edge must ``irecv`` to
    keep the DECODE channel paired) and only spawns a verify placeholder for a
    live request.
  * ``_run_edge_cloud_draft_last_segment`` drains (recv already done by the
    caller, skip tail compute, return a token-less placeholder) when the draft
    context is gone, instead of raising.
  * Every middle prefill chunk runs a complete draft chain to populate MTP KV,
    while its proposals are discarded and no target verify placeholder is
    created.
  * The verify DECODE_FIRST placeholder is created at the final DRAFT_FIRST
    pick (not the wall-clock-delayed final DRAFT_LAST pick) and published to
    the cloud together with that DRAFT_FIRST; local dispatch stays behind
    the final DRAFT_LAST, and an already-published placeholder is kept for
    drain if its requests finish before local dispatch.
"""

from collections import deque
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.output import (
    BatchType,
    HiddenChannelType,
    SchedulerOutput,
)


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #


def _make_bare_scheduler():
    from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

    s = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
    s.drafts_first_ready = deque()
    s.drafts_last_ready = deque()
    s.requests = {}
    s._pregenerated_draft_task_ids = set()
    s._pregenerated_draft_req_ids = {}
    s._draft_first_dispatched = False
    s._draft_first_cloud_publish_pending = None
    s._draft_first_scalars_patched = False
    s._draft_remote_pending_limit = 2
    s.draft_remote_pending_count = 0
    s.decode_or_draft_inflight_count = 0
    s.decode_or_draft_inflight_limit = 1
    s.decode_head_inflight_count = 0
    s._force_draft_last = False
    s._force_decode_last = False
    s.num_spec_tokens = 3
    return s


def _make_draft_first(task_id="task-0", req_id="req-0", step=0):
    so = MagicMock()
    so.batch_type = BatchType.DRAFT_FIRST
    so.draft_task_id = task_id
    so.draft_step_idx = step
    so.head_token = f"tok-{task_id}-{step}"
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 1}
    so.parent_req_id = req_id
    so.num_accepted_tokens = None
    so.valid_sampled_token_count = None
    so.is_last_prefill_chunk = True
    so.draft_output_req_ids = (req_id,)
    return so


def _make_draft_last(task_id="task-0", req_id="req-0", step=0):
    so = MagicMock()
    so.batch_type = BatchType.DRAFT_LAST
    so.draft_task_id = task_id
    so.draft_step_idx = step
    so.head_token = f"tok-{task_id}-{step}"
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 1}
    so.parent_req_id = req_id
    so.is_last_prefill_chunk = True
    so.draft_output_req_ids = (req_id,)
    return so


def _make_real_output(
    batch_type=BatchType.PREFILL_LAST,
    task_id="task-0",
    req_id="req-0",
):
    so = SchedulerOutput.make_empty()
    so.batch_type = batch_type
    so.head_token = task_id
    so.num_scheduled_tokens = {req_id: 8}
    so.total_num_scheduled_tokens = 8
    return so


# ------------------------------------------------------------------ #
# Test: _can_schedule_draft_first honors _force_draft_last (fix ①)   #
# ------------------------------------------------------------------ #


class TestCanScheduleDraftFirstForceGuard:
    """The pre-generated branch must gate on _force_draft_last just like the
    legacy branch, so DRAFT_FIRST -> DRAFT_LAST alternation is guaranteed."""

    def _setup(self, pregenerated=True):
        s = _make_bare_scheduler()
        drf = _make_draft_first()
        s.drafts_first_ready.append(drf)
        if pregenerated:
            s._pregenerated_draft_task_ids.add(drf.draft_task_id)
        # Conditions that would otherwise allow scheduling.
        s.drafts_last_ready = deque()
        s._force_decode_last = False
        s._force_draft_last = False
        s.draft_remote_pending_count = 0
        s.decode_or_draft_inflight_count = 0
        return s

    def test_preGen_blocked_when_force_draft_last_true(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = True
        assert s._can_schedule_draft_first() is False

    def test_preGen_allowed_when_force_draft_last_false(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        assert s._can_schedule_draft_first() is True

    def test_preGen_blocked_by_drafts_last_ready(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        s.drafts_last_ready.append(_make_draft_last())
        assert s._can_schedule_draft_first() is False

    def test_preGen_blocked_when_decode_head_in_flight(self):
        """Regression for decode_or_draft_inflight=2/1: DRAFT_FIRST and
        DECODE_FIRST use different recv primitives but share the DECODE
        stream, so a DRAFT_FIRST must not be dispatched while a DECODE_FIRST
        head is in flight (the cloud's recv order could mismatch the edge's
        send order).  Gate on decode heads, not total heads."""
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        s.decode_head_inflight_count = 1  # a DECODE_FIRST in flight
        assert s._can_schedule_draft_first() is False
        s.decode_head_inflight_count = 0
        assert s._can_schedule_draft_first() is True

    def test_preGen_allows_draft_pipeline_while_draft_in_flight(self):
        """The next DRAFT_FIRST MAY be dispatched while a previous DRAFT_FIRST
        is still in flight (draft pipelining): DRAFT_FIRST is an edge->cloud
        send while DRAFT_LAST is a cloud->edge recv (opposite stream
        directions), and draft+draft uses the same recv primitive (FIFO).
        Only a DECODE_FIRST head blocks it, not another DRAFT_FIRST."""
        s = self._setup(pregenerated=True)
        s._force_draft_last = False  # previous DRAFT_LAST already picked
        s.drafts_last_ready = deque()  # previous DRAFT_LAST popped (in flight)
        s.decode_head_inflight_count = 0  # no DECODE_FIRST in flight
        s.decode_or_draft_inflight_count = 1  # a DRAFT_FIRST in flight
        s.draft_remote_pending_count = 1  # under the pipeline credit (<2)
        assert s._can_schedule_draft_first() is True

    def test_legacy_branch_also_blocked_by_force_draft_last(self):
        """Non-pre-generated branch already had the guard (unchanged)."""
        s = self._setup(pregenerated=False)
        s._force_draft_last = True
        assert s._can_schedule_draft_first() is False


class TestDraftFirstLastAlternation:
    """End-to-end gate check: a second DRAFT_FIRST is blocked while the first
    DRAFT_LAST is pending, and admitted only after the tail is picked."""

    def test_second_head_blocked_until_tail_picked(self):
        s = _make_bare_scheduler()
        s._pregenerated_draft_task_ids.add("task-0")
        # step-0 head already picked; step-1 is next in drafts_first_ready.
        s.drafts_first_ready.append(_make_draft_first(step=1))
        s._force_decode_last = False
        s.draft_remote_pending_count = 0

        # DRAFT_FIRST step-0 in force + its DRAFT_LAST pending -> step-1 blocked.
        s._force_draft_last = True
        s.drafts_last_ready.append(_make_draft_last(step=0))
        assert s._can_schedule_draft_first() is False

        # DRAFT_LAST step-0 picked -> flag cleared, no tail pending -> allowed.
        s._force_draft_last = False
        s.drafts_last_ready.clear()
        assert s._can_schedule_draft_first() is True


class TestMidPrefillDraftChain:
    """Every chunk warms MTP KV, but only the last chunk may seed verify."""

    def test_pick_mid_prefill_tail_starts_draft_chain(self):
        s = _make_bare_scheduler()
        target = _make_real_output()
        flight = MagicMock()
        flight.is_last_chunk = False
        s.prefills_last_ready = deque([target])
        s._prefill_flight_by_token = {target.head_token: flight}
        s.chunk_prefill_first = []
        s._validate_prefill_tail_channel = MagicMock()
        s._pregenerate_draft_chain = MagicMock()

        result = s._pick_prefill_last_batch()

        assert result is target
        assert result.is_last_prefill_chunk is False
        assert result.draft_output_req_ids == ()
        s._pregenerate_draft_chain.assert_called_once_with(target)

    def test_pregenerates_mid_chunk_and_preserves_marker(self):
        s = _make_bare_scheduler()
        request = MagicMock()
        request.is_finished.return_value = False
        s.requests["req-0"] = request
        s._uses_async_scheduled_mtp_placeholders = MagicMock(
            return_value=True
        )

        target = _make_real_output()
        target.is_last_prefill_chunk = False
        target.draft_output_req_ids = ()
        s._pregenerate_draft_chain(target)

        assert len(s.drafts_first_ready) == s.num_spec_tokens
        assert all(
            getattr(output, "is_last_prefill_chunk", True) is False
            for output in s.drafts_first_ready
        )
        assert all(
            output.draft_output_req_ids == ()
            for output in s.drafts_first_ready
        )

    def test_mid_chunk_draft_tail_does_not_prepare_verify(self):
        s = _make_bare_scheduler()
        request = MagicMock()
        request.is_finished.return_value = False
        s.requests["req-0"] = request
        tail = _make_draft_last()
        tail.is_last_prefill_chunk = False
        tail.draft_output_req_ids = ()
        s.drafts_last_ready.append(tail)
        s._force_draft_last = True
        s._validate_draft_tail_channel = MagicMock()
        s._start_decode_or_draft_first_only_window = MagicMock()
        s._prepare_next_decode_first_placeholder = MagicMock()

        assert s._pick_draft_last_batch() is tail
        s._prepare_next_decode_first_placeholder.assert_not_called()

    def test_engine_advances_mid_prefill_draft(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _advance_edge_cloud_draft,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        finalized = _make_real_output(BatchType.DRAFT_FIRST)
        engine.scheduler.finalize_pre_generated_draft_first.return_value = (
            finalized
        )
        completed = _make_real_output()
        completed.is_last_prefill_chunk = False
        model_output = MagicMock()
        model_output.edge_cloud_draft_state = {
            "draft_task_id": completed.head_token,
            "draft_step_idx": 0,
        }
        model_output.sampled_token_ids = [[]]

        _advance_edge_cloud_draft(engine, completed, model_output)

        engine.scheduler.finalize_pre_generated_draft_first.assert_called_once_with(
            draft_task_id=completed.head_token,
            num_accepted_tokens=[0],
            valid_sampled_token_count=[0],
        )
        engine._release_deferred_draft_pre_out.assert_called_once_with(
            completed.head_token
        )
        engine.scheduler.enqueue_draft_first.assert_not_called()

    def test_registers_worker_created_fallback_draft(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _register_edge_cloud_draft_parent,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        engine._uses_scheduled_edge_cloud_draft.return_value = True
        completed = _make_real_output()
        model_output = MagicMock()
        model_output.edge_cloud_draft_state = {
            "draft_task_id": completed.head_token,
            "draft_step_idx": 0,
        }

        _register_edge_cloud_draft_parent(engine, completed, model_output)

        engine.scheduler.register_edge_cloud_draft_task.assert_called_once_with(
            completed.head_token, {"req-0"}
        )

    def test_does_not_register_without_worker_draft_state(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _register_edge_cloud_draft_parent,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        engine._uses_scheduled_edge_cloud_draft.return_value = True
        completed = _make_real_output()
        model_output = MagicMock(spec=[])

        _register_edge_cloud_draft_parent(engine, completed, model_output)

        engine.scheduler.register_edge_cloud_draft_task.assert_not_called()


# ------------------------------------------------------------------ #
# Test: _draft_output_reqs_live / _is_stale_draft_output (fix ②d)    #
# ------------------------------------------------------------------ #


class TestDraftReqsLiveAndStale:
    def _setup(self, req_present=True, req_id="req-0"):
        s = _make_bare_scheduler()
        if req_present:
            s.requests[req_id] = MagicMock()
        so = _make_draft_last(req_id=req_id)
        return s, so

    def test_live_request_reqs_live_true(self):
        s, so = self._setup(req_present=True)
        assert s._draft_output_reqs_live(so) is True

    def test_dead_request_reqs_live_false(self):
        s, so = self._setup(req_present=False)
        assert s._draft_output_reqs_live(so) is False

    def test_live_request_not_stale(self):
        s, so = self._setup(req_present=True)
        assert s._is_stale_draft_output(so) is False

    def test_dead_request_stale(self):
        s, so = self._setup(req_present=False)
        assert s._is_stale_draft_output(so) is True

    def test_preGen_dispatched_dead_request_still_stale(self):
        """Regression for the removed override: a pre-generated, step-0-
        dispatched chain whose request has since gone must still be treated
        as stale for not-yet-dispatched heads.  Otherwise the edge picks and
        dispatches heads it can no longer produce payload for (context
        cleared) -> cloud waits forever for data that never arrives."""
        s, so = self._setup(req_present=False)
        s._pregenerated_draft_task_ids.add(so.draft_task_id)
        s._draft_first_dispatched = True
        assert s._is_stale_draft_output(so) is True


# ------------------------------------------------------------------ #
# Test: _pick_draft_last_batch drains instead of dropping (fix ②a)   #
# ------------------------------------------------------------------ #


class TestPickDraftLastBatchDrain:
    """A DRAFT_LAST in drafts_last_ready always has its DRAFT_FIRST already
    dispatched to the cloud, so the cloud will isend a response -- the edge
    must execute (drain) the tail to pair it, never drop it."""

    def _setup(self, req_present=True):
        s = _make_bare_scheduler()
        if req_present:
            s.requests["req-0"] = MagicMock()
        s.drafts_last_ready.append(_make_draft_last())
        s._force_draft_last = True
        s.draft_remote_pending_count = 1
        # Stub side-effecting helpers to isolate the drain decision.
        s._validate_draft_tail_channel = MagicMock()
        s._start_decode_or_draft_first_only_window = MagicMock()
        s._prepare_next_decode_first_placeholder = MagicMock()
        s._make_empty_batch = MagicMock(return_value="EMPTY")
        return s

    def test_dead_request_drained_not_dropped(self):
        s = self._setup(req_present=False)
        before = s.draft_remote_pending_count
        result = s._pick_draft_last_batch()

        # The tail is returned (dispatched to the worker for drain), not
        # dropped -- the cloud's response must be paired on the DECODE channel.
        assert result.batch_type == BatchType.DRAFT_LAST
        assert len(s.drafts_last_ready) == 0
        # _force_draft_last is always reset now (the old stale-drop skipped it).
        assert s._force_draft_last is False
        # The decrement moved to update_from_output; _pick_draft_last_batch no
        # longer decrements (the old stale-drop did).
        assert s.draft_remote_pending_count == before
        # No verify placeholder for a gone request.
        s._prepare_next_decode_first_placeholder.assert_not_called()
        s._validate_draft_tail_channel.assert_called_once()

    def test_live_request_prepares_placeholder(self):
        s = self._setup(req_present=True)
        result = s._pick_draft_last_batch()
        assert result.batch_type == BatchType.DRAFT_LAST
        assert s._force_draft_last is False
        s._prepare_next_decode_first_placeholder.assert_called_once()

    def test_empty_returns_empty_batch(self):
        s = self._setup(req_present=True)
        s.drafts_last_ready.clear()
        assert s._pick_draft_last_batch() == "EMPTY"
        s._make_empty_batch.assert_called_once()


# ------------------------------------------------------------------ #
# Test: worker _run_edge_cloud_draft_last_segment drain (fix ②c)    #
# ------------------------------------------------------------------ #


class TestRunDraftLastSegmentDrain:
    """When the draft context is gone (request finished/aborted after its
    DRAFT_FIRST was dispatched), the tail segment must drain (the recv in
    _execute_model_edge_draft_tail already paired the cloud response) and
    return a token-less placeholder instead of raising."""

    def _make_runner(self, context_present=False):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._pending_edge_cloud_draft_contexts = {}
        if context_present:
            runner._pending_edge_cloud_draft_contexts["task-0"] = MagicMock()
        return runner

    @staticmethod
    def _make_so(task_id="task-0"):
        so = MagicMock()
        so.draft_task_id = task_id
        so.draft_step_idx = 0
        so.num_scheduled_tokens = {"req-0": 1}
        return so

    def test_drains_when_context_gone(self):
        from vllm.v1.outputs import ModelRunnerOutput

        runner = self._make_runner(context_present=False)
        result = runner._run_edge_cloud_draft_last_segment(
            self._make_so(), MagicMock()
        )
        assert isinstance(result, ModelRunnerOutput)
        assert result.req_ids == ["req-0"]
        assert result.req_id_to_index == {"req-0": 0}

    def test_drains_when_task_id_none(self):
        from vllm.v1.outputs import ModelRunnerOutput

        runner = self._make_runner(context_present=False)
        result = runner._run_edge_cloud_draft_last_segment(
            self._make_so(task_id=None), MagicMock()
        )
        assert isinstance(result, ModelRunnerOutput)
        assert result.req_ids == ["req-0"]


# ------------------------------------------------------------------ #
# Test: early DECODE_FIRST placeholder at the final DRAFT_FIRST pick   #
# ------------------------------------------------------------------ #


def _make_real_draft_first(task_id="task-0", req_id="req-0", step=2):
    so = SchedulerOutput.make_empty()
    so.batch_type = BatchType.DRAFT_FIRST
    so.draft_task_id = task_id
    so.draft_step_idx = step
    so.head_token = None
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 1}
    so.total_num_scheduled_tokens = 1
    so.parent_req_id = req_id
    so.is_last_prefill_chunk = True
    so.draft_output_req_ids = (req_id,)
    return so


def _make_real_decode_first(task_id="task-0", req_id="req-0"):
    so = SchedulerOutput.make_empty()
    so.batch_type = BatchType.DECODE_FIRST
    so.head_token = f"df-{task_id}"
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 4}
    so.total_num_scheduled_tokens = 4
    return so


class TestEarlyDecodeFirstPlaceholder:
    """The verify DECODE_FIRST placeholder is created at the final
    DRAFT_FIRST pick (instead of the wall-clock-delayed final DRAFT_LAST
    pick) and published to the cloud together with that DRAFT_FIRST, so
    the cloud can pre-post its verify recv one draft step earlier.  Local
    edge dispatch still stays behind the final DRAFT_LAST."""

    def _setup(self):
        s = _make_bare_scheduler()
        s.decodes_first_ready = deque()
        s._decode_first_placeholder_parent = None
        s._edge_cloud_draft_task_reqs = {}
        s.finished_req_ids = set()
        request = MagicMock()
        request.is_finished.return_value = False
        s.requests["req-0"] = request
        s.running = [request]
        s._pregenerated_draft_task_ids.add("task-0")
        s._uses_async_scheduled_mtp_placeholders = MagicMock(
            return_value=True
        )
        s._validate_draft_tail_channel = MagicMock()
        return s

    def test_final_draft_first_pick_creates_placeholder(self):
        s = self._setup()
        df = _make_real_decode_first()
        s._pick_decode_first_batch = MagicMock(return_value=df)
        drf = _make_real_draft_first(step=2)  # final of num_spec_tokens=3
        s.drafts_first_ready.append(drf)

        picked = s._pick_draft_first_batch()

        assert picked is drf
        assert len(s.drafts_last_ready) == 1
        assert list(s.decodes_first_ready) == [df]
        assert df.parent_draft_task_id == "task-0"
        assert not getattr(df, "cloud_published_with_draft_chain", False)

    def test_non_final_draft_first_pick_creates_nothing(self):
        s = self._setup()
        s._pick_decode_first_batch = MagicMock()
        s.drafts_first_ready.append(_make_real_draft_first(step=0))

        s._pick_draft_first_batch()

        s._pick_decode_first_batch.assert_not_called()
        assert not s.decodes_first_ready

    def test_placeholder_held_behind_pending_draft_last(self):
        from vllm_ascend.core.pd_separated_scheduler import PrefillState

        s = self._setup()
        df = _make_real_decode_first()
        s._pick_decode_first_batch = MagicMock(return_value=df)
        s.drafts_first_ready.append(_make_real_draft_first(step=2))
        s._pick_draft_first_batch()
        # State machine stubs: no other work than the pending tail.
        s._pick_decode_or_draft_first_only_or_empty = MagicMock(
            return_value=None
        )
        s._can_schedule_prefill_first = MagicMock(return_value=False)
        s._can_schedule_draft_last = MagicMock(return_value=True)
        s._can_schedule_draft_first = MagicMock(return_value=False)
        s._can_schedule_decode_last = MagicMock(return_value=False)
        s._can_schedule_decode_first = MagicMock(return_value=False)
        s._log_scheduler_state = MagicMock()
        s._start_decode_or_draft_first_only_window = MagicMock()
        s.prefill_inflight_count = 0

        # The pending DRAFT_LAST is picked before the queued DECODE_FIRST.
        out = s._pick_by_state(PrefillState.IDLE)
        assert out.batch_type == BatchType.DRAFT_LAST
        # Once the tail is picked, the verify head follows immediately.
        out = s._pick_by_state(PrefillState.IDLE)
        assert out is df

    def test_take_early_publish_decode_first(self):
        s = _make_bare_scheduler()
        s.decodes_first_ready = deque()
        df = _make_real_decode_first()
        df.parent_draft_task_id = "task-0"
        s.decodes_first_ready.append(df)

        assert s.take_early_publish_decode_first("task-x") is None
        assert s.take_early_publish_decode_first("task-0") is df
        assert df.cloud_published_with_draft_chain is True
        # One-shot: a second take for the same chain returns None.
        assert s.take_early_publish_decode_first("task-0") is None
        # Still queued for local dispatch after the final DRAFT_LAST.
        assert list(s.decodes_first_ready) == [df]

    def test_early_published_placeholder_kept_for_drain(self):
        """A DECODE_FIRST already published to the cloud must not be
        dropped when its requests finish before local dispatch: the cloud
        will run the verify middle and isend its response, so the edge
        keeps it (and its self-posted DECODE_LAST) to pair the channel --
        the same drain rule as DRAFT_LAST."""
        s = _make_bare_scheduler()
        s.decodes_first_ready = deque()
        s._decode_first_placeholder_parent = None
        s._dropped_draft_task_ids_to_report = []
        published = _make_real_decode_first(task_id="task-0")
        published.parent_draft_task_id = "task-0"
        published.cloud_published_with_draft_chain = True
        unpublished = _make_real_decode_first(task_id="task-1")
        unpublished.parent_draft_task_id = "task-1"
        s.decodes_first_ready.extend([published, unpublished])

        s._drop_stale_drafts_for_req_ids({"req-0"})

        assert list(s.decodes_first_ready) == [published]


class TestEarlyDecodeFirstPublish:
    """EngineCore publishes the verify DECODE_FIRST to the cloud together
    with the final DRAFT_FIRST, preserving wire order on the shared DECODE
    hidden channel (DRAFT_FIRSTs first, DECODE_FIRST last)."""

    def _make_engine(self, opened):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _publish_early_decode_first,
        )

        engine = MagicMock()
        engine.scheduler.num_spec_tokens = 3
        engine.scheduler.is_pre_generated_draft.return_value = True
        engine._pd_draft_pre_out_open_tasks = opened
        engine._pd_deferred_draft_pre_out = {}
        engine._publish_early_decode_first = (
            lambda drf, defer: _publish_early_decode_first(
                engine, drf, defer=defer
            )
        )
        return engine

    def test_publishes_decode_first_with_final_draft_first(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _maybe_publish_pre_out,
        )

        engine = self._make_engine(opened={"task-0"})
        df = _make_real_decode_first()
        df.parent_draft_task_id = "task-0"
        engine.scheduler.take_early_publish_decode_first.return_value = df
        drf = _make_real_draft_first(step=2)

        _maybe_publish_pre_out(engine, drf)

        published = [
            call.args[0]
            for call in engine._pp_pd_channel.publish.call_args_list
        ]
        # Wire order: the final DRAFT_FIRST first, DECODE_FIRST behind it.
        assert published == [drf, df]

    def test_local_dispatch_does_not_republish(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _maybe_publish_pre_out,
        )

        engine = self._make_engine(opened={"task-0"})
        df = _make_real_decode_first()
        df.parent_draft_task_id = "task-0"
        df.cloud_published_with_draft_chain = True

        _maybe_publish_pre_out(engine, df)

        engine._pp_pd_channel.publish.assert_not_called()

    def test_deferred_stream_keeps_draft_then_decode_order(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _maybe_publish_pre_out,
        )

        engine = self._make_engine(opened=set())
        df = _make_real_decode_first()
        df.parent_draft_task_id = "task-0"
        engine.scheduler.take_early_publish_decode_first.return_value = df
        drf = _make_real_draft_first(step=2)

        _maybe_publish_pre_out(engine, drf)

        # Stream not open: nothing published yet, and the DECODE_FIRST
        # rides the deferred queue behind the final DRAFT_FIRST.
        engine._pp_pd_channel.publish.assert_not_called()
        assert engine._pd_deferred_draft_pre_out["task-0"] == [drf, df]

