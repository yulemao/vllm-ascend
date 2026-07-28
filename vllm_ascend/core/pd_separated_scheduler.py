# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import BatchType, HiddenChannelType, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


class PrefillState(enum.Enum):
    """Edge-side prefill in-flight state machine.

    See Phase5 design in ``PDbatch分离边云协同Phase5&7详细设计.md``.
    """
    IDLE = "idle"       # prefill_inflight_count == 0
    LOW = "low"         # prefill_inflight_count == 1
    HIGH = "high"       # prefill_inflight_count >= prefill_inflight_limit


@dataclass
class PrefillChunkFlight:
    """Per-chunk in-flight tracking for chunk-prefill-prior scheduling.

    When ``chunk_prefill_prior_enable`` is True, each PREFILL_FIRST batch
    creates one ``PrefillChunkFlight`` keyed by its ``head_token``.  This
    allows the same request to have multiple chunks in-flight
    simultaneously — the next chunk's PF can be dispatched before the
    previous chunk's PL returns.

    Fields
    ------
    request_id : str
        The owning request.
    head_token : str
        Unique token assigned to this chunk's PREFILL_FIRST batch.
        PREFILL_LAST echoes it back so the flight can be located.
    hidden_channel : HiddenChannelType
        Prefill data-plane channel allocated for this chunk.
    chunk_index : int
        0-based index of this chunk within the request.
    is_last_chunk : bool
        True when this chunk consumes the last remaining prompt tokens.
    num_scheduled_tokens : int
        Number of tokens scheduled in this chunk.
    """
    request_id: str
    head_token: str
    hidden_channel: HiddenChannelType
    chunk_index: int
    is_last_chunk: bool
    num_scheduled_tokens: int


class HiddenChannelManager:
    """Manages data-plane hidden tensor channels for edge-cloud PD separation.

    Two prefill channels (PREFILL_1 / PREFILL_2) support 2P1D; one decode
    channel (DECODE) supports single in-flight decode. Channels are allocated
    in FIFO order and freed when the tail segment completes.
    """

    def __init__(self) -> None:
        self._free_prefills: deque[HiddenChannelType] = deque([
            HiddenChannelType.PREFILL_1,
            HiddenChannelType.PREFILL_2,
        ])
        # Mapping from head_token to the allocated channel.  Only prefill
        # batches are recorded here; decode batches always use DECODE and
        # do not need a mapping.
        self._head_token_to_channel: dict[str, HiddenChannelType] = {}

    # ------------------------------------------------------------------ #
    # Prefill channel allocation / release                               #
    # ------------------------------------------------------------------ #
    def allocate_prefill(self, head_token: str) -> HiddenChannelType:
        """Allocate a free prefill channel for the batch identified by
        ``head_token``. Raises if none available."""
        if not self._free_prefills:
            raise RuntimeError(
                "No free prefill hidden channel available"
            )
        channel = self._free_prefills.popleft()
        self._head_token_to_channel[head_token] = channel
        return channel

    def release_prefill(self, head_token: str) -> HiddenChannelType | None:
        """Release the prefill channel previously allocated for
        ``head_token``. Returns the freed channel (or None if not found)."""
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is None:
            return None
        self._free_prefills.append(channel)
        return channel

    def has_free_prefill(self) -> bool:
        return bool(self._free_prefills)

    # ------------------------------------------------------------------ #
    # Decode channel (always DECODE, no free-list)                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def decode_channel() -> HiddenChannelType:
        return HiddenChannelType.DECODE

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def get_channel(self, head_token: str) -> HiddenChannelType | None:
        return self._head_token_to_channel.get(head_token)

    @property
    def in_use_prefills(self) -> list[HiddenChannelType]:
        return list(self._head_token_to_channel.values())


class PDSeparatedScheduler(Scheduler):
    """Scheduler that separates prefill and decode into distinct steps.

    In edge-cloud PD-separated mode the four cardinal phases are:
      - PREFILL_FIRST  (edge head segment)
      - PREFILL_LAST   (edge tail segment, sourced from cloud-returned outputs)
      - DECODE_FIRST   (Phase 4)
      - DECODE_LAST    (Phase 4)

    This class owns the request bookkeeping for *first* segments
    (``chunk_prefill_first`` + parent's ``waiting`` / ``running``) and the
    ready queues for *last* segments (``prefills_last_ready`` /
    ``decodes_last_ready``), which are filled by the EngineCore from the
    POST_OUT channel before each ``schedule()`` call.

    Chunk-prefill-prior (Phase 1)
    -----------------------------
    When ``chunk_prefill_prior_enable`` is True, per-chunk flight tracking
    replaces the request-granularity ``prefill_last_pending`` list.  This
    allows the next chunk's PREFILL_FIRST to be dispatched before the
    previous chunk's PREFILL_LAST returns, achieving the same pipeline
    interleaving that MindIE's ``generate_send_metadata_to_queue()``
    provides.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Requests that have started their P-first segment but have not yet
        # been fully consumed (still chunking, or still in flight on cloud).
        self.chunk_prefill_first: list[Request] = []

        # SchedulerOutputs returned from cloud (POST_OUT channel) carrying
        # the metadata needed to execute the edge tail segment.
        # Populated by EngineCore.step() before calling self.schedule().
        self.prefills_last_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()
        self.drafts_first_ready: deque[SchedulerOutput] = deque()
        self.drafts_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches).
        self.prefill_inflight_limit: int = getattr(
            self.scheduler_config, "pd_prefill_inflight_limit", 1
        )
        self.prefill_inflight_count: int = 0
        self.decode_inflight_limit: int = 1
        self.decode_inflight_count: int = 0
        self.draft_inflight_limit: int = 1
        self.draft_inflight_count: int = 0
        self.draft_remote_pending_count: int = 0

        # Phase6 data-plane channel manager.  Two prefill hidden channels are
        # available for 2P1D; decode uses a dedicated fixed channel.
        self.hidden_channel_manager = HiddenChannelManager()

        # Buffer queue: requests whose P-first segment is done but P-last
        # segment has not yet returned from the cloud.  Not eligible for
        # decode scheduling until PL completes and they are moved to running.
        # When chunk_prefill_prior_enable is True, this is supplemented by
        # per-chunk flight tracking.
        self.prefill_last_pending: list[Request] = []

        # ------------------------------------------------------------------ #
        # Chunk-prefill-prior fields                                          #
        # ------------------------------------------------------------------ #
        # Enabled via pd_separation.next_prefill_prior_enable. When True the
        # scheduler yields a freed prefill slot to a *different* request
        # (cross-request head-prior, MindIE-style P1首->P2首) instead of
        # ahead-dispatching the same request's next chunk.
        self.next_prefill_prior_enable: bool = getattr(
            self.scheduler_config, "pd_next_prefill_prior_enable", False
        )
        # Enabled via pd_separation.chunk_prefill_prior_enable.
        self.chunk_prefill_prior_enable: bool = getattr(
            self.scheduler_config, "pd_chunk_prefill_prior_enable", False
        )
        self.max_chunk_prefill_ahead: int = getattr(
            self.scheduler_config, "pd_max_chunk_prefill_ahead", 1
        )

        # Per-chunk flight tracking: head_token → PrefillChunkFlight.
        # Populated on PF, consumed on PL.
        self._prefill_flight_by_token: dict[str, PrefillChunkFlight] = {}

        # Per-request count of chunks still waiting for PL.
        # request_id → count.  When count reaches 0, the request is
        # eligible to enter decode.
        self._pending_tail_count: dict[str, int] = {}

        # Per-request count of chunks whose PF was dispatched ahead
        # (before the previous chunk's PL returned).  Decremented on
        # PL return so the request is not re-added to chunk_prefill_first.
        self._ahead_chunk_count: dict[str, int] = {}

        # 限制 PREFILL_FIRST 每个 batch 最多只组 1 个请求。
        # 配置路径: additional_config.edge_cloud_config.pd_separation.limit_prefill_batch_size
        self.limit_prefill_batch_size: bool = False
        _additional = getattr(self.vllm_config, "additional_config", None)
        if isinstance(_additional, dict):
            _ec = _additional.get("edge_cloud_config", {})
            _pd = _ec.get("pd_separation", {})
            self.limit_prefill_batch_size = bool(
                _pd.get("limit_prefill_batch_size", False)
            )

        # [新增] DECODE_LAST 延迟调度计时器。
        # D首 pick 后启动，D尾 在延迟到期前不可被调度。
        self._decode_last_delay_start_ts: float | None = None
        self._decode_last_delay_schedule_ms: int = 30

        # [新增] 强制调度 D尾 标记。
        # D首 调度后置 True，D尾 调度后置 False。
        # True 时禁止调度 D首，严格保证 DF -> DL 交替时序。
        self._force_decode_last: bool = False

        self._layer_slice_config_path: str | None = None
        self._layer_slice_config_mtime: float = 0.0
        self._load_layer_slice_config()
        # After scheduling a DECODE_LAST, briefly reserve the next scheduling
        # opportunity for DECODE_FIRST only.  This keeps decode middle work fed
        # without blocking; if no decode head can be scheduled within the window,
        # normal scheduling resumes.
        self._decode_first_only_start_ts: float | None = None
        self._decode_first_only_window_ms: int = 10

        # ------------------------------------------------------------------ #
        # Pre-generated draft chain (decode/draft pipelining)                 #
        # ------------------------------------------------------------------ #
        # When a DECODE_LAST is picked, the whole DRAFT_FIRST chain for the
        # next speculative round is already determined by the tail's
        # SchedulerOutput, except the step-0 sampling scalars
        # (num_accepted_tokens / valid_sampled_token_count) which only the
        # cloud draft middle consumes.  The chain is therefore enqueued at
        # tail-pick time (see _pregenerate_draft_chain) so draft batches can
        # be dispatched back-to-back behind the tail without an engine round
        # trip per step; the engine patches the scalars into the step-0 copy
        # and publishes it to the cloud when the tail completes.
        #
        # _draft_first_cloud_publish_pending: the pre-generated step-0
        #   DRAFT_FIRST awaiting its cloud publish (scalars not yet known).
        # _draft_first_scalars_patched: finalize already merged the scalars
        #   into the pending SO (sync-step mode publishes at pick time).
        # _draft_first_dispatched: the pending SO has been picked and
        #   dispatched to the edge worker; the cloud copy must then be
        #   published even if the request finished, so the cloud posts the
        #   matching recv for the already-issued edge isend.
        self._draft_first_cloud_publish_pending: SchedulerOutput | None = None
        self._draft_first_scalars_patched: bool = False
        self._draft_first_dispatched: bool = False
        # draft_task_ids whose chain was pre-generated; guards
        # _enqueue_next_draft_first from re-enqueuing chain links.
        self._pregenerated_draft_task_ids: set[str] = set()
        # Max DRAFT_FIRST heads whose DRAFT_LAST has not yet completed.
        # Worker and cloud dispatch strictly in order and every DRAFT_LAST
        # is picked before the next DRAFT_FIRST (drafts_last has priority in
        # _pick_by_state), so at most two links are ever outstanding: the
        # DRL queued behind the DRF being picked.
        self._draft_remote_pending_limit: int = 2

    # ------------------------------------------------------------------ #
    # Chunk-prefill-prior helpers                                         #
    # ------------------------------------------------------------------ #
    def _can_ahead_schedule(self, req_id: str) -> bool:
        """True when the request can have one more chunk PF dispatched ahead."""
        return (
            self._ahead_chunk_count.get(req_id, 0) < self.max_chunk_prefill_ahead
        )

    def _has_other_prefill_request(self, current_req_id: str) -> bool:
        """True if a different request has prefill work ready to fill the next
        prefill slot (a cross-request head-prior candidate).

        Only called from the ahead-decision point inside
        ``_pick_prefill_first_batch``, where ``self.running`` temporarily
        holds the drained ``chunk_prefill_first`` (so other mid-prefill
        requests appear there); the ``is_prefill_chunk`` filter excludes
        decode requests that normally live in ``running``.

        Returns True when any of the following holds:
          - ``chunk_prefill_first`` contains a request with a different id;
          - ``running`` contains a still-prefilling request with a different
            id (drained-window case);
          - ``waiting`` is non-empty (a new request is available).
        """
        for req in self.chunk_prefill_first:
            if req.request_id != current_req_id:
                return True
        running = getattr(self, "running", None)
        if running:
            for req in running:
                if (req.request_id != current_req_id
                        and getattr(req, "is_prefill_chunk", False)):
                    return True
        return len(self.waiting) > 0

    def _should_ahead_schedule(self, req: Request, is_last: bool) -> bool:
        """Decide whether to ahead-dispatch ``req``'s next chunk (intra-request
        pipeline) or yield the prefill slot to another request (cross-request
        head-prior, MindIE-style P1首 -> P2首).

        Ahead (re-add ``req`` to ``chunk_prefill_first``) iff:
          - this is not the last chunk, AND
          - the request's ahead budget allows it, AND
          - next_prefill_prior_enable is off, OR no other request is
            available to fill the slot (single-request pipeline).

        Yield (send ``req`` to ``prefill_last_pending``; the PL-return path
        re-adds it for its next chunk when the tail returns) when
        ``next_prefill_prior_enable`` is on and another request has prefill
        work ready.
        """
        if is_last or not self._can_ahead_schedule(req.request_id):
            return False
        if (self.next_prefill_prior_enable
                and self._has_other_prefill_request(req.request_id)):
            return False
        return True

    def _select_single_prefill_candidate(
        self, candidates: list[Request],
    ) -> tuple[list[Request], list[Request]]:
        """Pick at most one prefill candidate to expose to ``super().schedule()``.

        Enforces the one-request-per-PF-batch invariant. chunk-prefill-prior
        keys one ``PrefillChunkFlight`` per ``head_token`` and the sampler
        consumes a batch-level ``is_last_prefill_chunk`` flag, so a PF batch
        must contain exactly one request. If two requests shared a batch they
        would share one ``head_token`` (the second overwrites the first in
        ``_prefill_flight_by_token``, losing its PL tracking) and the
        batch-level ``is_last`` flag could not represent both requests'
        last-chunk state, stalling the overwritten request.

        Returns ``(exposed, rest)`` where ``exposed`` has 0 or 1 request and
        ``rest`` holds the remaining candidates to be scheduled in their own
        PF batches on subsequent calls.
        """
        if not candidates:
            return [], []
        return [candidates[0]], list(candidates[1:])

    def _select_pf_candidate_head_prior(
        self, candidates: list[Request],
    ) -> tuple[Request | None, list[Request]]:
        """Pick at most one PF candidate for cross-request head-prior.

        Prefers a *fresh* candidate -- one with no in-flight chunk
        (``_pending_tail_count[req] == 0``), i.e. its previous chunk's PL has
        returned and it has no other chunk on the cloud. Refilling a freed
        slot with a fresh candidate keeps one in-flight chunk per request, so
        the two 2P1D prefill slots spread across different requests
        (MindIE-style ``P1首 / P2首`` interleaving).

        Decision order:
          1. First fresh candidate in ``candidates`` -> expose it (refill its
             slot with its next chunk).
          2. No fresh candidate but ``waiting`` is non-empty -> return
             ``(None, candidates)`` so the caller admits a *new* request from
             waiting instead of clustering another chunk on an already
             in-flight request (cross-request head-prior).
          3. No fresh candidate and no waiting request -> fall back to the
             first (in-flight) candidate: ahead-dispatch its next chunk so
             both 2P1D slots serve the single request (intra-request
             pipeline).

        Returns ``(exposed_or_None, rest)`` where ``exposed_or_None`` is 0 or
        1 request. ``rest`` holds the remaining candidates (untouched when
        admitting new, so they stay eligible for later batches).
        """
        for req in candidates:
            if self._pending_tail_count.get(req.request_id, 0) == 0:
                return req, [r for r in candidates if r is not req]
        if len(self.waiting) > 0:
            return None, list(candidates)
        exposed_list, rest = self._select_single_prefill_candidate(candidates)
        return (exposed_list[0] if exposed_list else None), rest

    def _prepare_pf_running_state(
        self,
        saved_chunk_prefill_first: list[Request],
        saved_running: list[Request],
        saved_max_num_running_reqs: int,
    ) -> tuple[list[Request], int, list[Request]]:
        """Decide ``(running, max_num_running_reqs, rest_candidates)`` for the
        ``super().schedule()`` call in ``_pick_prefill_first_batch``.

        - ``chunk_prefill_prior_enable``: enforce one-request-per-PF-batch (the
          flight map keys one flight per ``head_token`` and the sampler reads
          a batch-level ``is_last_prefill_chunk`` flag, so a PF batch must
          contain exactly one request). Candidate selection prefers a fresh
          head (no in-flight chunk) or a new waiting request over clustering
          on an in-flight request -- cross-request head-prior. ``max`` is
          capped at 1 (continue one candidate, no new admission) or a
          capacity-gated 0/1 (admit one new request from waiting).
        - Legacy (``chunk_prefill_prior_enable`` off): no per-chunk flight
          tracking, so multi-request PF batches are safe (PL routes by
          ``req_id``) and preferred for token-budget utilization. Expose all
          candidates and cap by system capacity -- the original behavior.

        The base scheduler caps scheduled running reqs by
        ``max_num_running_reqs`` (vllm ``Scheduler.schedule`` line ~390), so
        the value returned here directly bounds the PF batch size.
        """
        if self.chunk_prefill_prior_enable:
            exposed, rest_candidates = self._select_pf_candidate_head_prior(
                saved_chunk_prefill_first
            )
            if exposed is not None:
                # Continue one candidate; cap at 1 so the base does not admit
                # a new request alongside it (one-per-batch).
                return [exposed], 1, rest_candidates
            # No candidate to continue: admit at most one new request from
            # waiting, gated by system capacity (saved_running occupy their
            # slots) so we never exceed max_num_running_reqs system-wide.
            available = saved_max_num_running_reqs - len(saved_running)
            max_num_running_reqs = 1 if available >= 1 else 0
            return [], max_num_running_reqs, rest_candidates
        if self.limit_prefill_batch_size:
            if saved_chunk_prefill_first:
                return (
                    [saved_chunk_prefill_first[0]],
                    saved_max_num_running_reqs - len(saved_running),
                    [],
                )
            else:
                return (
                    [],
                    saved_max_num_running_reqs - len(saved_running),
                    [],
                )
        return (
            list(saved_chunk_prefill_first),
            saved_max_num_running_reqs - len(saved_running),
            [],
        )

    def _total_pending_tails(self) -> int:
        """Total number of chunks waiting for PL across all requests."""
        return sum(self._pending_tail_count.values())

    def _cleanup_request_flight_state(self, req_id: str) -> None:
        """Remove all tracking state for a finished request."""
        self._pending_tail_count.pop(req_id, None)
        self._ahead_chunk_count.pop(req_id, None)
        # Remove flights for this request.
        to_remove = [
            token for token, flight in self._prefill_flight_by_token.items()
            if flight.request_id == req_id
        ]
        for token in to_remove:
            self._prefill_flight_by_token.pop(token, None)

    def schedule(self) -> SchedulerOutput:
        return self._schedule_pd_separated()

    def _make_empty_batch(self) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        return scheduler_output

    def _schedule_pd_separated(self) -> SchedulerOutput:
        state = self._prefill_state()
        scheduler_output = self._pick_by_state(state)
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        is_tail = scheduler_output.batch_type in (
            BatchType.PREFILL_LAST,
            BatchType.DECODE_LAST,
            BatchType.DRAFT_LAST,
        )
        if has_work or is_tail:
            self._log_scheduler_state(state, scheduler_output.batch_type)
        return scheduler_output

    def _decode_first_only_active(self) -> bool:
        started_at = self._decode_first_only_start_ts
        if started_at is None:
            return False
        elapsed_ms = (time.monotonic() - started_at) * 1000
        if elapsed_ms >= self._decode_first_only_window_ms:
            self._decode_first_only_start_ts = None
            return False
        return True

    def _start_decode_first_only_window(self) -> None:
        self._decode_first_only_start_ts = time.monotonic()

    def _clear_decode_first_only_window(self) -> None:
        self._decode_first_only_start_ts = None

    def _pick_decode_first_only_or_empty(self) -> SchedulerOutput | None:
        if self._has_draft_work():
            return None
        if not self._decode_first_only_active():
            return None
        if self._can_schedule_decode_first():
            self._clear_decode_first_only_window()
            return self._pick_decode_first_batch()
        return self._make_empty_batch()

    def _pick_by_state(self, state: PrefillState) -> SchedulerOutput:
        decode_first_only = self._pick_decode_first_only_or_empty()
        if decode_first_only is not None:
            return decode_first_only

        # D尾必须无条件优先于 D首，防止 decode_inflight_count 在 D首
        # 完成后立即释放导致 D尾 starvation。
        if state == PrefillState.IDLE:
            # IDLE: P首/chunk0首 > D尾 > D首 > Empty.
            if self._can_schedule_prefill_first():
                so = self._pick_prefill_first_batch()
                if so.total_num_scheduled_tokens > 0:
                    return so
                # P首 returned empty (e.g. KV cache exhausted). Preserve
                # finished_req_ids and fall through to decode tasks to avoid
                # a tight busy-loop where the edge spins on empty prefill.
                logger.warning(
                    "PREFILL_FIRST returned empty batch (total_num_scheduled_tokens=0). "
                    "This usually means KV cache blocks are exhausted by running decode "
                    "requests. Prefill work will be deferred until resources are freed."
                )
                self.finished_req_ids.update(so.finished_req_ids)
            if self.prefills_last_ready:
                return self._pick_prefill_last_batch()
            if self.drafts_last_ready:
                return self._pick_draft_last_batch()
            if self._can_schedule_draft_first():
                return self._pick_draft_first_batch()
            if self.decodes_last_ready and self._can_schedule_decode_last():
                return self._pick_decode_last_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            return self._make_empty_batch()

        if state == PrefillState.LOW:
            # LOW: chunk/P首(when slot available) > D尾 > D首 > P尾 > Empty.
            if self._can_schedule_prefill_first():
                so = self._pick_prefill_first_batch()
                if so.total_num_scheduled_tokens > 0:
                    return so
                logger.warning(
                    "PREFILL_FIRST returned empty batch (total_num_scheduled_tokens=0). "
                    "This usually means KV cache blocks are exhausted by running decode "
                    "requests. Prefill work will be deferred until resources are freed."
                )
                self.finished_req_ids.update(so.finished_req_ids)
            if self.decodes_last_ready and self._can_schedule_decode_last():
                return self._pick_decode_last_batch()
            if self.prefills_last_ready:
                return self._pick_prefill_last_batch()
            if self.drafts_last_ready:
                return self._pick_draft_last_batch()
            if self._can_schedule_draft_first():
                return self._pick_draft_first_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            return self._make_empty_batch()

        # HIGH: D尾 > D首 > P尾 > Empty. New P首 is forbidden.
        if self.decodes_last_ready and self._can_schedule_decode_last():
            return self._pick_decode_last_batch()
        if self.prefills_last_ready:
            return self._pick_prefill_last_batch()
        if self.drafts_last_ready:
            return self._pick_draft_last_batch()
        if self._can_schedule_draft_first():
            return self._pick_draft_first_batch()
        if self._can_schedule_decode_first():
            return self._pick_decode_first_batch()
        return self._make_empty_batch()

    def is_waiting_for_remote_tail(self) -> bool:
        """True when local requests exist only as remote in-flight work.

        In this state the edge has no local batch to execute until POST_OUT
        returns a PREFILL_LAST/DECODE_LAST, so the EngineCore should yield
        instead of tight-loop scheduling EMPTY batches.
        """
        return bool(
            (
                self.prefill_inflight_count > 0
                or self.decode_inflight_count > 0
                or self.draft_inflight_count > 0
                or self.draft_remote_pending_count > 0
            )
            and not self.prefills_last_ready
            and not self.decodes_last_ready
            and not self.drafts_last_ready
            and not self._can_schedule_prefill_first()
            and not self._can_schedule_draft_first()
            and not self._can_schedule_decode_first()
        )

    def _prefill_state(self) -> PrefillState:
        if self.prefill_inflight_count <= 0:
            return PrefillState.IDLE
        if (
            self.prefill_inflight_limit > 1
            and self.prefill_inflight_count >= self.prefill_inflight_limit
        ):
            return PrefillState.HIGH
        return PrefillState.LOW

    def _has_prefill_work(self) -> bool:
        return bool(self.chunk_prefill_first or self.waiting)

    def _can_schedule_prefill_first(self) -> bool:
        # When running decode requests already fill max_num_running_reqs,
        # super().schedule() inside _pick_prefill_first_batch will return an
        # empty batch, causing a tight-loop.  Pre-check capacity to avoid
        # the useless call.
        effective_capacity = self.max_num_running_reqs - len(self.running)
        return (
            self._has_prefill_work()
            and self.prefill_inflight_count < self.prefill_inflight_limit
            and self.hidden_channel_manager.has_free_prefill()
            and effective_capacity > 0
        )

    def _can_schedule_decode_first(self) -> bool:
        return bool(
            self.running
            and self.decode_inflight_count < self.decode_inflight_limit
            and self.draft_inflight_count == 0
            and self.draft_remote_pending_count == 0
            and not self.drafts_first_ready
            and not self.drafts_last_ready
            and not self._force_decode_last
        )

    def _can_schedule_draft_first(self) -> bool:
        # Scheduled draft head/tail payloads share the DECODE channel.
        # Ordering on the channel is preserved without serialising against
        # DECODE_FIRST / remote drafts because both workers dispatch in
        # strict FIFO order (edge: DF isend, DL recv, DRF isend, DRL recv;
        # cloud mirrors it) and every DRAFT_LAST is picked before the next
        # DRAFT_FIRST (drafts_last has priority in _pick_by_state), so
        # send/recv pairs stay matched.  draft_remote_pending_count is only
        # capped to bound the in-flight chain length.
        return bool(
            self.drafts_first_ready
            and self.draft_inflight_count < self.draft_inflight_limit
            and self.draft_remote_pending_count
            < self._draft_remote_pending_limit
            and not self.drafts_last_ready
            and not self._force_decode_last
        )

    def _log_scheduler_state(self, state: PrefillState, batch_type: BatchType) -> None:
        self._step_counter += 1
        if self.chunk_prefill_prior_enable:
            logger.info(
                f"[PD] Step{self._step_counter}, state is {state}, batch_type is {batch_type}, "
                f"waiting[]: {len(self.waiting)}, "
                f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
                f"running[]: {len(self.running)}, "
                f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
                f"drafts_first_ready[]: {len(self.drafts_first_ready)}, "
                f"drafts_last_ready[]: {len(self.drafts_last_ready)}, "
                f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"draft_inflight: {self.draft_inflight_count}/{self.draft_inflight_limit}, "
                f"draft_remote_pending: {self.draft_remote_pending_count}, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}, "
                f"chunk_flights: {len(self._prefill_flight_by_token)}, "
                f"pending_tails: {self._total_pending_tails()}, "
                f"ahead_chunks: {sum(self._ahead_chunk_count.values())}",
            )
        else:
            logger.info(
                f"[PD] Step{self._step_counter}, state is {state}, batch_type is {batch_type}, "
                f"waiting[]: {len(self.waiting)}, "
                f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
                f"running[]: {len(self.running)}, "
                f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
                f"drafts_first_ready[]: {len(self.drafts_first_ready)}, "
                f"drafts_last_ready[]: {len(self.drafts_last_ready)}, "
                f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"draft_inflight: {self.draft_inflight_count}/{self.draft_inflight_limit}, "
                f"draft_remote_pending: {self.draft_remote_pending_count}, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}",
            )

    # ------------------------------------------------------------------ #
    # Layer-slice config loading (Edge side)                             #
    # ------------------------------------------------------------------ #
    def _load_layer_slice_config(self) -> None:
        """Load decode_last_delay_schedule_ms from layer_slice_config.yaml."""
        yaml_path = os.environ.get("VLLM_LAYER_SLICE_CONFIG")
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "layer_slice_config.yaml"
            )
        if not os.path.exists(yaml_path):
            self._layer_slice_config_path = None
            self._layer_slice_config_mtime = 0.0
            return
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                logger.warning(
                    "Layer-slice config %s is not a dict; ignoring.", yaml_path
                )
                return
            _key = "decode_last_delay_schedule_ms"
            if _key in raw:
                try:
                    self._decode_last_delay_schedule_ms = int(raw[_key])
                    logger.info(
                        "[PDSeparatedScheduler] %s set to %d from %s",
                        _key, self._decode_last_delay_schedule_ms, yaml_path,
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid %s value %r in %s; keeping %d",
                        _key, raw[_key], yaml_path,
                        self._decode_last_delay_schedule_ms,
                    )
            self._layer_slice_config_path = yaml_path
            self._layer_slice_config_mtime = os.path.getmtime(yaml_path)
        except Exception:
            logger.exception("Failed to load layer-slice config from %s", yaml_path)

    def _maybe_hot_reload_layer_slice_config(self) -> None:
        """Check whether the YAML config file has changed on disk and reload."""
        path = self._layer_slice_config_path
        if path is None:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime != self._layer_slice_config_mtime:
            old_value = self._decode_last_delay_schedule_ms
            self._load_layer_slice_config()
            if self._decode_last_delay_schedule_ms != old_value:
                logger.info(
                    "[PDSeparatedScheduler] Layer-slice config hot-reloaded: "
                    "%s=%d",
                    "decode_last_delay_schedule_ms",
                    self._decode_last_delay_schedule_ms,
                )

    # ------------------------------------------------------------------ #
    # DECODE_LAST delay scheduling                                       #
    # ------------------------------------------------------------------ #
    def _start_decode_last_delay(self) -> None:
        """Start the timer when DECODE_FIRST is picked.
        DECODE_LAST cannot be scheduled until the delay expires."""
        self._decode_last_delay_start_ts = time.monotonic()

    def _can_schedule_decode_last(self) -> bool:
        """Return True if the delay since DECODE_FIRST has elapsed."""
        if self._decode_last_delay_start_ts is None:
            return True
        elapsed_ms = (time.monotonic() - self._decode_last_delay_start_ts) * 1000
        if elapsed_ms >= self._decode_last_delay_schedule_ms:
            self._decode_last_delay_start_ts = None
            return True
        logger.info(
            "[PD] DECODE_LAST delayed: elapsed=%.1f ms < limit=%d ms",
            elapsed_ms, self._decode_last_delay_schedule_ms,
        )
        return False

    def _pick_prefill_first_batch(self) -> SchedulerOutput:
        saved_running = self.running
        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_max_num_running_reqs = self.max_num_running_reqs

        # Decide what to expose to super().schedule() and the running cap.
        # With chunk_prefill_prior_enable this enforces one-request-per-PF-batch
        # (prevents head_token collision / is_last ambiguity); legacy keeps the
        # original multi-request batching. See _prepare_pf_running_state.
        self.running, self.max_num_running_reqs, rest_candidates = self._prepare_pf_running_state(
            saved_chunk_prefill_first, saved_running, saved_max_num_running_reqs
        )

        if self.limit_prefill_batch_size:
            # 限制每个 P首 batch 最多只包含 1 个请求。
            # 1. chunk_prefill_first 非空时，取第一个请求到 running，
            #    清空 waiting 防止 super().schedule() 再取更多。
            # 2. chunk_prefill_first 为空时，从 waiting 中只保留 1 个请求，
            #    让 super().schedule() 在 waiting 中正常调度（生成 NewRequestData）。
            #    其余 waiting 请求暂存，在 finally 中恢复。
            if saved_chunk_prefill_first:
                self.chunk_prefill_first = list(saved_chunk_prefill_first[1:])
                saved_waiting_rest = []
            else:
                self.chunk_prefill_first = []
                if len(self.waiting) > 0:
                    first_req = self.waiting.pop_request()
                    saved_waiting_rest = list(self.waiting)
                    self.waiting.clear()
                    self.waiting.append(first_req)
                else:
                    saved_waiting_rest = []
        else:
            self.chunk_prefill_first = []
            saved_waiting_rest = []


        # Snapshot num_computed_tokens before super().schedule() so that
        # the is_last computation below uses the pre-schedule value.
        # (super().schedule() → _update_after_schedule increments
        # num_computed_tokens; without the snapshot we would double-count
        # the current chunk's tokens.)
        _num_computed_before: dict[str, int] = {
            req.request_id: req.num_computed_tokens
            for req in self.running
        }

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs

            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                    for req in self.running:
                        if req.is_prefill_chunk:
                            self.chunk_prefill_first.append(req)
                        else:
                            self.prefill_last_pending.append(req)
                else:
                    scheduler_output.batch_type = BatchType.PREFILL_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.hidden_channel = (
                        self.hidden_channel_manager.allocate_prefill(
                            scheduler_output.head_token
                        )
                    )
                    self.prefill_inflight_count += 1

                    scheduled_req_ids = set(
                        scheduler_output.num_scheduled_tokens.keys()
                    )

                    if self.chunk_prefill_prior_enable:
                        # === Chunk-prefill-prior routing ===
                        # Each scheduled request gets a PerfillChunkFlight.
                        # If the request still has more chunks after this
                        # PF, it may be re-added to chunk_prefill_first
                        # immediately (ahead), allowing the next chunk's PF
                        # before the current chunk's PL returns.
                        for req in self.running:
                            if req.request_id in scheduled_req_ids:
                                num_scheduled = (
                                    scheduler_output.num_scheduled_tokens[
                                        req.request_id
                                    ]
                                )
                                # Use pre-schedule num_computed_tokens
                                # to avoid double-counting the current
                                # chunk's tokens.
                                num_comp_before = (
                                    _num_computed_before.get(
                                        req.request_id, 0
                                    )
                                )
                                remaining = (
                                    req.num_prompt_tokens
                                    - num_comp_before
                                    - num_scheduled
                                )
                                is_last = remaining <= 0
                                flight = PrefillChunkFlight(
                                    request_id=req.request_id,
                                    head_token=scheduler_output.head_token,
                                    hidden_channel=(
                                        scheduler_output.hidden_channel
                                    ),
                                    chunk_index=max(0, req.chunk_num - 1),
                                    is_last_chunk=is_last,
                                    num_scheduled_tokens=num_scheduled,
                                )
                                self._prefill_flight_by_token[
                                    scheduler_output.head_token
                                ] = flight
                                self._pending_tail_count[req.request_id] = (
                                    self._pending_tail_count.get(
                                        req.request_id, 0
                                    )
                                    + 1
                                )

                                if self._should_ahead_schedule(req, is_last):
                                    # Ahead: re-add to chunk_prefill_first
                                    # so the next chunk PF can be dispatched
                                    # before this chunk's PL returns. Used for
                                    # the single-request pipeline (both 2P1D
                                    # slots serve the same request).
                                    self.chunk_prefill_first.append(req)
                                    self._ahead_chunk_count[
                                        req.request_id
                                    ] = (
                                        self._ahead_chunk_count.get(
                                            req.request_id, 0
                                        )
                                        + 1
                                    )
                                    logger.info(
                                        "[PD-CHUNK-PRIOR] Ahead-scheduled "
                                        "chunk %d of request %s "
                                        "(head_token=%s, %d tokens, "
                                        "ahead_count=%d)",
                                        flight.chunk_index,
                                        req.request_id,
                                        scheduler_output.head_token,
                                        num_scheduled,
                                        self._ahead_chunk_count[
                                            req.request_id
                                        ],
                                    )
                                else:
                                    # Wait for PL before next chunk. Reasons:
                                    #   - last chunk (is_last);
                                    #   - ahead budget exhausted;
                                    #   - yield: next_prefill_prior_enable is
                                    #     on and another request can fill the
                                    #     slot (cross-request head-prior).
                                    if (
                                        self.next_prefill_prior_enable
                                        and not is_last
                                        and self._can_ahead_schedule(
                                            req.request_id
                                        )
                                        and self._has_other_prefill_request(
                                            req.request_id
                                        )
                                    ):
                                        wait_reason = "yield"
                                    elif is_last:
                                        wait_reason = "last"
                                    else:
                                        wait_reason = "ahead_full"
                                    self.prefill_last_pending.append(req)
                                    logger.info(
                                        "[PD-CHUNK-PRIOR] Chunk %d of "
                                        "request %s waiting for PL "
                                        "(head_token=%s, %d tokens, "
                                        "is_last=%s, "
                                        "pending_tails=%d, reason=%s)",
                                        flight.chunk_index,
                                        req.request_id,
                                        scheduler_output.head_token,
                                        num_scheduled,
                                        is_last,
                                        self._pending_tail_count.get(
                                            req.request_id, 0
                                        ),
                                        wait_reason,
                                    )
                            elif req.is_prefill_chunk:
                                # Not scheduled this round (token budget
                                # exhausted), keep for next round.
                                self.chunk_prefill_first.append(req)
                            else:
                                # Completed but not scheduled – defensive.
                                self.prefill_last_pending.append(req)
                    else:
                        # === Legacy routing (no chunk-prefill-prior) ===
                        for req in self.running:
                            if req.request_id in scheduled_req_ids:
                                self.prefill_last_pending.append(req)
                            elif req.is_prefill_chunk:
                                # Not scheduled this round (e.g. token budget
                                # exhausted), keep in chunk_prefill_first.
                                self.chunk_prefill_first.append(req)
                            else:
                                # Completed but not scheduled – defensive.
                                self.prefill_last_pending.append(req)

                # Restore prefill candidates not exposed to super() this round
                # so each is scheduled in its own PF batch (one-per-batch).
                self.chunk_prefill_first = (
                    rest_candidates + self.chunk_prefill_first
                )
                self.running = saved_running

                # [方案B] Edge 侧建议 Cloud 是否切层。
                # 必须在 self.running 恢复为 saved_running 之后检查，
                # 否则 self.running 被临时替换为 prefill 请求，永远为 False。
                if scheduler_output.total_num_scheduled_tokens > 0:
                    suggest = len(self.running) > 0
                    scheduler_output.cloud_suggest_slicing = suggest
                    if not suggest:
                        logger.error(
                            "[PD-EDGE-NO-SLICE] PREFILL_FIRST "
                            "cloud_suggest_slicing=False, running=%d, "
                            "chunk_prefill_first=%d, total_tokens=%d",
                            len(self.running),
                            len(self.chunk_prefill_first),
                            scheduler_output.total_num_scheduled_tokens,
                        )

            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.running = saved_running

            # 恢复 waiting 中其余请求（仅在 limit_prefill_batch_size 时）
            if self.limit_prefill_batch_size and saved_waiting_rest:
                for req in saved_waiting_rest:
                    self.waiting.append(req)

        if (
            self.limit_prefill_batch_size
            and scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens > 0
        ):
            logger.info(
                "[PD-LIMIT-PREFILL] PREFILL_FIRST batch_size=%d, "
                "total_tokens=%d, chunk_first_remaining=%d, waiting_remaining=%d",
                len(scheduler_output.num_scheduled_tokens),
                scheduler_output.total_num_scheduled_tokens,
                len(self.chunk_prefill_first),
                len(self.waiting),
            )

        return scheduler_output  # type: ignore[return-value]

    def _pick_prefill_last_batch(self) -> SchedulerOutput:
        """Pop one cloud-returned SchedulerOutput from prefills_last_ready.

        The cloud has already rewritten ``batch_type=PREFILL_LAST`` and kept
        all original KV / sampling metadata intact, so the edge worker can
        directly run segment_e + sampler on it. We also remove the involved
        requests from ``chunk_prefill_first`` so the parent class's
        ``update_from_output`` does not double-account them.
        """
        if not self.prefills_last_ready:
            return self._make_empty_batch()
        so = self.prefills_last_ready.popleft()
        assert so.batch_type == BatchType.PREFILL_LAST, (
            f"prefills_last_ready expects PREFILL_LAST, got {so.batch_type}"
        )
        # [ascend insert] Mark whether this PL is the request's last
        # prefill chunk.  Mid-chunk PL must not sample: prefill is still
        # incomplete, and the would-be sampled token actually predicts a
        # prompt token belonging to the next chunk.  sample_tokens() reads
        # this flag to skip sampling, which also avoids decrementing
        # num_output_placeholders (only incremented for the last chunk)
        # below zero.  The flight is still in the map here; it is popped
        # later in _update_from_output_prefill_last_chunk_prior.
        flight = (
            self._prefill_flight_by_token.get(so.head_token)
            if so.head_token else None
        )
        so.is_last_prefill_chunk = True if flight is None else flight.is_last_chunk
        # Drop these reqs from chunk_prefill_first. Keep them in
        # prefill_last_pending until update_from_output() moves them to running.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        self._validate_prefill_tail_channel(so)
        return so

    def _validate_prefill_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        token = scheduler_output.head_token
        channel = scheduler_output.hidden_channel
        if not token:
            raise RuntimeError("PREFILL_LAST missing head_token")
        if channel not in (HiddenChannelType.PREFILL_1, HiddenChannelType.PREFILL_2):
            raise RuntimeError(
                f"PREFILL_LAST expects a prefill hidden channel, got {channel}"
            )
        expected = self.hidden_channel_manager.get_channel(token)
        if expected != channel:
            raise RuntimeError(
                f"PREFILL_LAST hidden channel mismatch: expected {expected}, "
                f"got {channel}, head_token={token}"
            )

    def _validate_decode_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        if scheduler_output.hidden_channel != HiddenChannelType.DECODE:
            raise RuntimeError(
                "DECODE_LAST expects decode hidden channel, got "
                f"{scheduler_output.hidden_channel}"
            )

    def _validate_draft_tail_channel(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        if not scheduler_output.head_token:
            raise RuntimeError("DRAFT_LAST missing head_token")
        if not scheduler_output.draft_task_id:
            raise RuntimeError("DRAFT_LAST missing draft_task_id")
        if scheduler_output.draft_step_idx is None:
            raise RuntimeError("DRAFT_LAST missing draft_step_idx")
        if scheduler_output.hidden_channel != HiddenChannelType.DECODE:
            raise RuntimeError(
                "DRAFT_LAST expects decode hidden channel, got "
                f"{scheduler_output.hidden_channel}"
            )

    def _pick_draft_first_batch(self) -> SchedulerOutput:
        if not self.drafts_first_ready:
            return self._make_empty_batch()
        scheduler_output = self.drafts_first_ready.popleft()
        if scheduler_output is self._draft_first_cloud_publish_pending:
            # Track dispatch of the pre-generated step-0 copy so finalize
            # knows the edge isend will be issued (and the cloud copy must
            # be published even if the request finishes in between).
            self._draft_first_dispatched = True
            if self._draft_first_scalars_patched:
                # Sync-step mode: the tail completed before this pick, so
                # the scalars are already on the SO and the engine
                # publishes it right away; nothing left to defer.
                self._draft_first_cloud_publish_pending = None
                self._draft_first_scalars_patched = False
        scheduler_output.batch_type = BatchType.DRAFT_FIRST
        if scheduler_output.head_token is None:
            scheduler_output.head_token = uuid4().hex
        scheduler_output.hidden_channel = HiddenChannelType.DECODE
        # Draft-first self-posting mirrors the decode path below. Every
        # field needed by DRAFT_LAST is already known on the edge; the cloud
        # used to echo the same SchedulerOutput back through POST_OUT only
        # after its worker acked DRAFT_FIRST. Pre-generating the tail here
        # removes that per-draft-step control-plane round trip and lets the
        # edge post the matching receive as soon as DRAFT_FIRST has completed
        # locally. Worker FIFO ordering plus the per-channel send-work wait
        # preserves DRAFT_FIRST -> DRAFT_LAST data-plane ordering.
        draft_last = replace(
            scheduler_output,
            batch_type=BatchType.DRAFT_LAST,
            num_accepted_tokens=None,
            valid_sampled_token_count=None,
        )
        self._validate_draft_tail_channel(draft_last)
        self.drafts_last_ready.append(draft_last)
        self.draft_inflight_count += 1
        self.draft_remote_pending_count += 1
        return scheduler_output

    def enqueue_draft_first(
        self,
        source: SchedulerOutput,
        *,
        draft_task_id: str,
        draft_step_idx: int,
        num_accepted_tokens: list[int] | None = None,
        valid_sampled_token_count: list[int] | None = None,
    ) -> bool:
        """Generate a draft head locally, mirroring decode head generation.

        The worker owns the mutable draft tensors, but the scheduler owns all
        draft control-plane SchedulerOutputs. The initial step receives only
        the rejection-corrected scalar state from the worker; follow-up steps
        are derived directly from the completed DRAFT_LAST.
        """
        req_ids = list(source.num_scheduled_tokens)
        # Worker draft contexts are batch-scoped and are cleared as a whole
        # when any member request finishes or is aborted.
        if not req_ids or any(
            req_id not in self.requests for req_id in req_ids
        ):
            return False

        draft_first = replace(
            source,
            batch_type=BatchType.DRAFT_FIRST,
            head_token=None,
            hidden_channel=HiddenChannelType.DECODE,
            parent_req_id=req_ids[0],
            draft_task_id=draft_task_id,
            draft_step_idx=draft_step_idx,
            num_accepted_tokens=num_accepted_tokens,
            valid_sampled_token_count=valid_sampled_token_count,
        )
        self.drafts_first_ready.append(draft_first)
        return True

    def _enqueue_next_draft_first(
        self, draft_last: SchedulerOutput
    ) -> bool:
        draft_step_idx = int(draft_last.draft_step_idx or 0)
        next_step_idx = draft_step_idx + 1
        task_id = draft_last.draft_task_id
        if task_id in self._pregenerated_draft_task_ids:
            # The chain was pre-generated at tail-pick time; the next
            # DRAFT_FIRST is already queued.  Clean up when it completes.
            if next_step_idx >= self.num_spec_tokens:
                self._pregenerated_draft_task_ids.discard(task_id)
            return False
        if next_step_idx >= self.num_spec_tokens:
            return False
        if task_id is None:
            raise RuntimeError("DRAFT_LAST missing draft_task_id")
        return self.enqueue_draft_first(
            draft_last,
            draft_task_id=task_id,
            draft_step_idx=next_step_idx,
        )

    def _pick_draft_last_batch(self) -> SchedulerOutput:
        while self.drafts_last_ready:
            scheduler_output = self.drafts_last_ready.popleft()
            if scheduler_output.batch_type != BatchType.DRAFT_LAST:
                raise RuntimeError(
                    "drafts_last_ready expects DRAFT_LAST, got "
                    f"{scheduler_output.batch_type}"
                )
            self._validate_draft_tail_channel(scheduler_output)
            if self._is_stale_draft_output(scheduler_output):
                self.draft_remote_pending_count = max(
                    0, self.draft_remote_pending_count - 1
                )
                logger.info(
                    "[PD] drop stale DRAFT_LAST task_id=%s step=%s",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
                continue
            return scheduler_output
        return self._make_empty_batch()

    @staticmethod
    def _scheduler_output_intersects_req_ids(
        scheduler_output: SchedulerOutput, req_ids: set[str]
    ) -> bool:
        if scheduler_output.parent_req_id in req_ids:
            return True
        return any(
            req_id in req_ids
            for req_id in scheduler_output.num_scheduled_tokens
        )

    def _drop_stale_drafts_for_req_ids(self, req_ids: set[str]) -> None:
        if not req_ids:
            return
        kept_first: deque[SchedulerOutput] = deque()
        for output in self.drafts_first_ready:
            if self._scheduler_output_intersects_req_ids(output, req_ids):
                if output.draft_task_id:
                    self._pregenerated_draft_task_ids.discard(
                        output.draft_task_id
                    )
            else:
                kept_first.append(output)
        self.drafts_first_ready = kept_first
        pending = self._draft_first_cloud_publish_pending
        if (
            pending is not None
            and not self._draft_first_dispatched
            and self._scheduler_output_intersects_req_ids(pending, req_ids)
        ):
            # The pre-generated step-0 copy was dropped before dispatch;
            # it must never be published to the cloud (the edge will not
            # issue the matching isend).  A dispatched copy is kept so
            # finalize can still publish it and the cloud posts the recv
            # that pairs the already-issued edge isend.
            self._draft_first_cloud_publish_pending = None
            self._draft_first_scalars_patched = False
        kept_last: deque[SchedulerOutput] = deque()
        dropped_last = 0
        for output in self.drafts_last_ready:
            if self._scheduler_output_intersects_req_ids(output, req_ids):
                dropped_last += 1
            else:
                kept_last.append(output)
        self.drafts_last_ready = kept_last
        self.draft_remote_pending_count = max(
            0, self.draft_remote_pending_count - dropped_last
        )

    def _is_stale_draft_output(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        req_ids = set(scheduler_output.num_scheduled_tokens)
        if scheduler_output.parent_req_id:
            req_ids.add(scheduler_output.parent_req_id)
        return bool(req_ids) and all(
            req_id not in self.requests for req_id in req_ids
        )

    def _pick_decode_last_batch(self) -> SchedulerOutput:
        if not self.decodes_last_ready:
            return self._make_empty_batch()
        so = self.decodes_last_ready.popleft()
        assert so.batch_type == BatchType.DECODE_LAST, (
            f"decodes_last_ready expects DECODE_LAST, got {so.batch_type}"
        )
        self._validate_decode_tail_channel(so)
        self._start_decode_first_only_window()
        self._force_decode_last = False
        self._pregenerate_draft_chain(so)
        return so

    # ------------------------------------------------------------------ #
    # Pre-generated draft chain                                           #
    # ------------------------------------------------------------------ #
    def _uses_scheduled_draft_pregen(self) -> bool:
        """True when the draft chain can be pre-generated at tail-pick time.

        Limited to the MTP family: eagle3's cloud draft middle consumes
        extra per-task state that has not been validated against early
        dispatch, so eagle3 stays on the legacy completion-time enqueue
        path (enqueue_draft_first with scalars already attached).
        """
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or self.num_spec_tokens <= 0:
            return False
        method = getattr(speculative_config, "method", None)
        if method in ("qwen3_5_mtp", "qwen_mtp"):
            return True
        if method != "mtp":
            return False
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        return "qwen" in str(getattr(hf_config, "model_type", "")).lower()

    def _pregenerate_draft_chain(self, tail_so: SchedulerOutput) -> None:
        """Pre-generate the DRAFT_FIRST chain when a target tail is picked.

        Every field on a draft SchedulerOutput is already known when its
        parent DECODE_LAST is picked, except the step-0 sampling scalars
        (num_accepted_tokens / valid_sampled_token_count), which only the
        cloud draft middle consumes -- the edge worker derives all draft
        inputs from its stashed draft context.  Enqueueing the whole chain
        now lets DRAFT_FIRST batches be dispatched back-to-back behind the
        tail instead of waiting for an engine round trip per step.  The
        engine defers the step-0 cloud publish and patches the scalars in
        when the tail completes (finalize_pre_generated_draft_first).
        """
        if not self._uses_scheduled_draft_pregen():
            return
        if (
            self.drafts_first_ready
            or self._draft_first_cloud_publish_pending is not None
        ):
            # A previous chain has not drained; the completion-time enqueue
            # path (enqueue_draft_first) will handle this tail instead.
            return
        req_ids = list(tail_so.num_scheduled_tokens)
        if not req_ids or any(
            req_id not in self.requests for req_id in req_ids
        ):
            return
        task_id = tail_so.head_token
        if not task_id:
            return
        for step_idx in range(self.num_spec_tokens):
            draft_first = replace(
                tail_so,
                batch_type=BatchType.DRAFT_FIRST,
                head_token=None,
                hidden_channel=HiddenChannelType.DECODE,
                parent_req_id=req_ids[0],
                draft_task_id=task_id,
                draft_step_idx=step_idx,
                num_accepted_tokens=None,
                valid_sampled_token_count=None,
            )
            self.drafts_first_ready.append(draft_first)
            if step_idx == 0:
                self._draft_first_cloud_publish_pending = draft_first
                self._draft_first_scalars_patched = False
                self._draft_first_dispatched = False
        self._pregenerated_draft_task_ids.add(task_id)
        logger.info(
            "[PD] Pre-generated draft chain: task_id=%s steps=%d parent=%s",
            task_id,
            self.num_spec_tokens,
            req_ids[0],
        )

    def finalize_pre_generated_draft_first(
        self,
        *,
        num_accepted_tokens: list[int] | None,
        valid_sampled_token_count: list[int] | None,
    ) -> SchedulerOutput | None:
        """Patch the sampling scalars into the pre-generated step-0
        DRAFT_FIRST once its parent tail has completed sampling.

        Returns the SO for immediate cloud publish when it has already been
        dispatched to the edge worker (batch-queue mode): the cloud copy
        must go out even if the request just finished, because the edge
        isend is already issued and the cloud must post the matching recv.
        Returns None when the SO has not been dispatched yet (sync-step
        mode: the engine publishes it at pick time, scalars already
        patched) or when there is no pending pre-generated copy.
        """
        pending = self._draft_first_cloud_publish_pending
        if pending is None:
            return None
        pending.num_accepted_tokens = num_accepted_tokens
        pending.valid_sampled_token_count = valid_sampled_token_count
        if not self._draft_first_dispatched:
            self._draft_first_scalars_patched = True
            return None
        self._draft_first_cloud_publish_pending = None
        self._draft_first_scalars_patched = False
        return pending

    def _ensure_cached_all_token_ids(
        self, scheduler_output: SchedulerOutput,
    ) -> None:
        """Ensure every cached decode req carries all_token_ids.

        In PD-separated mode the edge worker's persistent input_batch may not
        retain a request across the PF → PL → DF transitions (the tail segment
        may have been skipped by ``_update_states``), yet the async-scheduling
        model runner requires ``all_token_ids`` to reconstruct
        ``output_token_ids`` for any resumed (req_index is None) request with
        ``num_output_tokens > 0``.

        The base ``_make_cached_request_data`` only fills ``all_token_ids``
        when the request was *not* scheduled in the previous step
        (``prev_step_scheduled_req_ids``).  Because PD separation interleaves
        PF / PL / DF / DL phases — and PL/DL are popped from cloud-returned
        queues without going through ``super().schedule()`` — the scheduler's
        ``prev_step_scheduled_req_ids`` can be stale or misleading, causing
        ``all_token_ids`` to be omitted for requests that the worker actually
        needs it for.

        This helper unconditionally back-fills ``all_token_ids`` for any
        cached request that has output tokens but is missing from the dict.
        The extra payload is cheap (control-plane only) and avoids the
        ``KeyError`` in ``gpu_model_runner._update_states``.
        """
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id, num_output_tokens in zip(
            cached_reqs.req_ids,
            cached_reqs.num_output_tokens,
        ):
            if num_output_tokens > 0 and req_id not in cached_reqs.all_token_ids:
                # Use the Request-level cached np.ndarray to avoid repeated
                # np.asarray() conversion of the Python list (dominant
                # bottleneck on long-sequence decode batches).
                cached_reqs.all_token_ids[req_id] = (
                    self.requests[req_id].cached_all_token_ids_np)

    def _pick_decode_first_batch(self) -> SchedulerOutput:
        if not self.running:
            return self._make_empty_batch()

        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_waiting = self.waiting
        saved_skipped = self.skipped_waiting

        self.chunk_prefill_first = []
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                    logger.debug(
                        "DECODE_FIRST race: empty batch due to async "
                        "update_from_output delay, running=%d",
                        len(self.running),
                    )
                else:
                    scheduler_output.batch_type = BatchType.DECODE_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.hidden_channel = (
                        self.hidden_channel_manager.decode_channel()
                    )
                    self._ensure_cached_all_token_ids(scheduler_output)
                    self.decode_inflight_count += 1
                    self._force_decode_last = True
                    self._start_decode_last_delay()

                    # === Decode-first self-posting optimization ===
                    # Cloud's _maybe_publish_post_out merely replaces
                    # batch_type with DECODE_LAST.  We pre-generate it on
                    # the edge side and stash it in decodes_last_ready so
                    # that scheduling DECODE_LAST needs no round-trip
                    # through POST_OUT.  The cloud unconditionally skips
                    # POST_OUT for all DECODE_FIRST batches.
                    decode_last = replace(
                        scheduler_output,
                        batch_type=BatchType.DECODE_LAST,
                    )
                    self.decodes_last_ready.append(decode_last)
                    # ===============================================
                for req in list(self.waiting):
                    saved_waiting.prepend_request(req)
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped

        return scheduler_output  # type: ignore[return-value]

    def _migrate_prefill_to_running(self) -> None:
        """Move fully-prefilled requests from chunk_prefill_first to running.

        When chunk_prefill_prior is enabled, a request stays in
        chunk_prefill_first even after ``is_prefill_chunk`` becomes False
        if it still has pending PL returns.  Only requests with zero
        pending tails are eligible to enter decode.
        """
        completed = [
            req for req in self.chunk_prefill_first
            if not req.is_prefill_chunk
            and self._pending_tail_count.get(req.request_id, 0) == 0
        ]
        for req in completed:
            self.chunk_prefill_first.remove(req)
            self.running.append(req)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        if request.is_prefill_chunk:
            self.chunk_prefill_first.append(request)
        else:
            request.num_computed_tokens = 0
            self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        super()._update_after_schedule(scheduler_output)

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    # ------------------------------------------------------------------ #
    # update_from_output — chunk-prefill-prior routing                    #
    # ------------------------------------------------------------------ #
    def _update_from_output_prefill_last_legacy(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Legacy PL routing: request-granularity pending list."""
        completed_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        newly_running: list[Request] = []
        newly_chunked: list[Request] = []
        remaining_pending: list[Request] = []
        for req in self.prefill_last_pending:
            if req.request_id in completed_req_ids:
                if req.is_prefill_chunk:
                    self.chunk_prefill_first.append(req)
                    newly_chunked.append(req)
                else:
                    self.running.append(req)
                    newly_running.append(req)
            else:
                remaining_pending.append(req)
        self.prefill_last_pending = remaining_pending

        logger.info(
            f"[PD] update_from_output PREFILL_LAST done, "
            f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
            f"moved {len(newly_running)} reqs to running[], "
            f"moved {len(newly_chunked)} reqs to chunk_prefill_first[], "
            f"running[]: {len(self.running)}, "
            f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}",
        )

    def _update_from_output_prefill_last_chunk_prior(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Chunk-prefill-prior PL routing: head_token → flight lookup."""
        head_token = scheduler_output.head_token
        if not head_token:
            logger.warning(
                "[PD-CHUNK-PRIOR] PREFILL_LAST missing head_token; "
                "falling back to legacy routing."
            )
            self._update_from_output_prefill_last_legacy(scheduler_output)
            return

        flight = self._prefill_flight_by_token.pop(head_token, None)
        if flight is None:
            logger.warning(
                "[PD-CHUNK-PRIOR] PREFILL_LAST head_token=%s not found "
                "in flight map; falling back to legacy routing.",
                head_token,
            )
            self._update_from_output_prefill_last_legacy(scheduler_output)
            return

        req_id = flight.request_id
        req = self.requests.get(req_id)

        # Decrement pending tail count.
        prev_count = self._pending_tail_count.get(req_id, 0)
        if prev_count > 0:
            self._pending_tail_count[req_id] = prev_count - 1
        remaining = self._pending_tail_count.get(req_id, 0)

        # Decrement ahead count if this chunk was pre-scheduled.
        ahead_before = self._ahead_chunk_count.get(req_id, 0)
        if ahead_before > 0:
            self._ahead_chunk_count[req_id] = ahead_before - 1

        logger.info(
            "[PD-CHUNK-PRIOR] PL returned: request=%s chunk=%d/%s "
            "head_token=%s tokens=%d "
            "pending_tails: %d→%d ahead: %d→%d",
            req_id,
            flight.chunk_index,
            "last" if flight.is_last_chunk else "mid",
            head_token,
            flight.num_scheduled_tokens,
            prev_count,
            remaining,
            ahead_before,
            self._ahead_chunk_count.get(req_id, 0),
        )

        if flight.is_last_chunk and remaining == 0:
            # All chunks complete → request enters decode.
            if req is not None:
                self.running.append(req)
            self._cleanup_request_flight_state(req_id)
            logger.info(
                "[PD-CHUNK-PRIOR] Request %s all chunks done, "
                "moved to running[] (%d total).",
                req_id,
                len(self.running),
            )
        else:
            # Mid-chunk PL returned, or last chunk but other tails still
            # pending.  Re-add for the next chunk IF there are more chunks
            # to schedule AND the request is not already queued.  This keeps
            # the pipeline continuous across >2 chunks: the next chunk's PF
            # fills the prefill slot freed by this PL, overlapping other
            # in-flight PLs (e.g. 4 chunks -> chunk2 PF starts as soon as
            # chunk0 PL returns, overlapping chunk1's PL, instead of waiting
            # for chunk1's PL).  The old logic skipped re-add whenever
            # ahead_before > 0, which forced pair-wise scheduling
            # ((0,1) then (2,3)) and left a slot idle between pairs.
            # Do NOT call _cleanup_request_flight_state here: ahead count
            # and in-flight flights are still needed for outstanding chunks.
            has_more_chunks = (
                req is not None
                and req.num_computed_tokens < req.num_prompt_tokens
            )
            already_queued = (
                req is not None and req in self.chunk_prefill_first
            )
            if has_more_chunks and not already_queued:
                self.chunk_prefill_first.append(req)
                logger.info(
                    "[PD-CHUNK-PRIOR] Request %s chunk %d PL: "
                    "re-added to chunk_prefill_first[] for next chunk "
                    "(remaining=%d, ahead=%d).",
                    req_id,
                    flight.chunk_index,
                    remaining,
                    self._ahead_chunk_count.get(req_id, 0),
                )
            else:
                logger.info(
                    "[PD-CHUNK-PRIOR] Request %s chunk %d PL: "
                    "skip re-add (remaining=%d, ahead=%d, more=%s, "
                    "queued=%s).",
                    req_id,
                    flight.chunk_index,
                    remaining,
                    self._ahead_chunk_count.get(req_id, 0),
                    has_more_chunks,
                    already_queued,
                )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        if scheduler_output.batch_type == BatchType.PREFILL_LAST:
            if self.prefill_inflight_count > 0:
                self.prefill_inflight_count -= 1
            if scheduler_output.head_token:
                self.hidden_channel_manager.release_prefill(
                    scheduler_output.head_token
                )

            if self.chunk_prefill_prior_enable:
                self._update_from_output_prefill_last_chunk_prior(
                    scheduler_output
                )
            else:
                self._update_from_output_prefill_last_legacy(
                    scheduler_output
                )

        if scheduler_output.batch_type == BatchType.DECODE_FIRST:
            # D首完成后立即释放 inflight 计数，使下一个 D首可以
            # 在 D尾仍在 batch_queue 中时就被调度，消除 Cloud idle gap。
            if self.decode_inflight_count > 0:
                self.decode_inflight_count -= 1
            logger.info(
                f"[PD] update_from_output DECODE_FIRST done, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}",
            )
        if scheduler_output.batch_type == BatchType.DRAFT_FIRST:
            self.draft_inflight_count = max(
                0, self.draft_inflight_count - 1
            )
            logger.info(
                "[PD] update_from_output DRAFT_FIRST done, "
                "draft_inflight: %d/%d",
                self.draft_inflight_count,
                self.draft_inflight_limit,
            )
        enqueue_next_draft = (
            scheduler_output.batch_type == BatchType.DRAFT_LAST
        )
        if enqueue_next_draft:
            self.draft_remote_pending_count = max(
                0, self.draft_remote_pending_count - 1
            )
        if scheduler_output.batch_type == BatchType.DECODE_LAST:
            # decode_inflight_count 已在 DECODE_FIRST 的 update_from_output
            # 中释放，此处不再重复减 1。
            logger.info(
                f"[PD] update_from_output DECODE_LAST done, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}",
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if enqueue_next_draft:
            next_draft_ready = self._enqueue_next_draft_first(
                scheduler_output
            )
            logger.info(
                "[PD] update_from_output DRAFT_LAST done, "
                "draft_remote_pending: %d, next_draft_ready: %s",
                self.draft_remote_pending_count,
                next_draft_ready,
            )
        self.chunk_prefill_first = [
            req for req in self.chunk_prefill_first if not req.is_finished()
        ]
        self.prefill_last_pending = [
            req for req in self.prefill_last_pending if not req.is_finished()
        ]
        return outputs

    def get_request_counts(self) -> tuple[int, int]:
        num_running, num_waiting = super().get_request_counts()
        if self.chunk_prefill_prior_enable:
            # Use a set to avoid double-counting requests that appear in
            # multiple tracking structures (e.g. a request in
            # prefill_last_pending also has _pending_tail_count > 0).
            pending_ids: set[str] = set()
            pending_ids.update(
                req.request_id for req in self.chunk_prefill_first
            )
            pending_ids.update(
                req.request_id for req in self.prefill_last_pending
            )
            pending_ids.update(self._pending_tail_count.keys())
            return (num_running + len(pending_ids), num_waiting)
        return (
            num_running
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending),
            num_waiting,
        )

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        base = super().get_num_unfinished_requests()
        if self.chunk_prefill_prior_enable:
            pending_ids: set[str] = set()
            pending_ids.update(
                req.request_id for req in self.chunk_prefill_first
            )
            pending_ids.update(
                req.request_id for req in self.prefill_last_pending
            )
            pending_ids.update(self._pending_tail_count.keys())
            return base + len(pending_ids)
        return (
            base
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending)
        )

    def _has_draft_work(self) -> bool:
        return bool(
            self.drafts_first_ready
            or self.drafts_last_ready
            or self.draft_inflight_count > 0
            or self.draft_remote_pending_count > 0
        )

    def has_requests(self) -> bool:
        return super().has_requests() or self._has_draft_work()

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        result = super().finish_requests(request_ids, finished_status)
        finished_req_ids = {req_id for req_id, _client_index in result}
        self._drop_stale_drafts_for_req_ids(finished_req_ids)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        to_remove = set()
        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req and req.is_finished():
                to_remove.add(req)
                # Clean up chunk-prefill-prior flight state.
                self._cleanup_request_flight_state(req_id)

        if to_remove:
            self.chunk_prefill_first = remove_all(
                self.chunk_prefill_first, to_remove
            )

        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.chunk_prefill_first:
                request = self.chunk_prefill_first.pop()
                self.kv_cache_manager.free(request)
                self.encoder_cache_manager.free(request)
                request.status = RequestStatus.PREEMPTED
                request.num_computed_tokens = 0
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.num_preemptions += 1
                if self.log_stats:
                    request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
                self.waiting.prepend_request(request)

            # Also clean up chunk-prefill-prior flight state.
            for req_id in list(self._pending_tail_count.keys()):
                self._cleanup_request_flight_state(req_id)

        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.num_running_reqs += len(self.chunk_prefill_first)
        return stats

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        saved_running = self.running
        self.running = list(self.running) + [
            r for r in self.chunk_prefill_first if r not in self.running
        ]
        try:
            result = super()._handle_invalid_blocks(invalid_block_ids)
        finally:
            self.running = saved_running
        return result


class AsyncPDSeparatedScheduler(AsyncScheduler, PDSeparatedScheduler):
    """Async scheduler with PD separation."""
    pass
