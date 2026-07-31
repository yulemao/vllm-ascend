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


# Configurable constants for DP-scalable channel management.
_PREFILL_CHANNELS_PER_DP = 2
_DECODE_CHANNELS_PER_DP = 1


class HiddenChannelManager:
    """Manages data-plane hidden tensor channels for edge-cloud PD separation.

    Each EngineCore_DP owns an independent instance managing only the
    channel slice for its dp_rank::

        dp_rank=0 -> PREFILL_1..2, DECODE_1
        dp_rank=1 -> PREFILL_3..4, DECODE_2

    Channels are allocated in FIFO order and freed when the tail segment
    completes.
    """

    def __init__(
        self,
        dp_rank: int = 0,
        prefill_per_dp: int = _PREFILL_CHANNELS_PER_DP,
        is_shared_model_edge: bool = False,
    ) -> None:
        # In the per-rank edge topology (edge_npu_count > 1, i.e. NOT
        # shared-model), every DP lives on its own physical card with
        # its own HCCL world, so all DPs share the same channel pool
        # (PREFILL_1..2, DECODE_1).  Only in the shared-model topology
        # (edge_npu_count == 1, dp_size > 1) do we slice by dp_rank.
        if not is_shared_model_edge:
            dp_rank = 0
        prefill_start = dp_rank * prefill_per_dp + 1

        self._free_prefills: deque[HiddenChannelType] = deque(
            HiddenChannelType.prefill(i)
            for i in range(prefill_start, prefill_start + prefill_per_dp)
        )
        # Decode uses a fixed channel per DP — no dynamic alloc/release.
        self._decode_channel: HiddenChannelType = (
            HiddenChannelType.decode(dp_rank + 1)
        )
        # Mapping from head_token to the allocated channel (prefill only).
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
        logger.info(
            "[PD] allocate_prefill: channel=%s head_token=%s free_left=%s",
            channel.value, head_token, list(self._free_prefills),
        )
        return channel

    def release_prefill(self, head_token: str) -> HiddenChannelType | None:
        """Release the prefill channel previously allocated for
        ``head_token``. Returns the freed channel (or None if not found)."""
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is None:
            return None
        self._free_prefills.append(channel)
        logger.info(
            "[PD] release_prefill: channel=%s head_token=%s free=%s",
            channel.value, head_token, list(self._free_prefills),
        )
        return channel

    def has_free_prefill(self) -> bool:
        return bool(self._free_prefills)

    # ------------------------------------------------------------------ #
    # Decode channel (fixed per DP)                                      #
    # ------------------------------------------------------------------ #
    def decode_channel(self) -> HiddenChannelType:
        """Return the fixed decode channel for this dp_rank."""
        return self._decode_channel

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def get_channel(self, head_token: str) -> HiddenChannelType | None:
        return self._head_token_to_channel.get(head_token)

    @property
    def in_use_prefills(self) -> list[HiddenChannelType]:
        return [
            ch for ch in self._head_token_to_channel.values()
            if ch.value.startswith("prefill_")
        ]

    @property
    def prefill_pool(self) -> frozenset[HiddenChannelType]:
        """The set of prefill channels managed by this instance."""
        return frozenset(self._free_prefills) | frozenset(self.in_use_prefills)

    @property
    def decode_pool(self) -> frozenset[HiddenChannelType]:
        """The set of decode channels managed by this instance."""
        return frozenset([self._decode_channel])

    @staticmethod
    def prefill_inflight_limit() -> int:
        return _PREFILL_CHANNELS_PER_DP

    @staticmethod
    def required_prefill_groups(dp_size: int) -> int:
        return dp_size * _PREFILL_CHANNELS_PER_DP

    @staticmethod
    def required_decode_groups(dp_size: int) -> int:
        return dp_size * _DECODE_CHANNELS_PER_DP


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
        self.decodes_first_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()
        self.drafts_first_ready: deque[SchedulerOutput] = deque()
        self.drafts_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches).
        self.prefill_inflight_limit: int = getattr(
            self.scheduler_config, "pd_prefill_inflight_limit",
            _PREFILL_CHANNELS_PER_DP,
        )
        self.prefill_inflight_count: int = 0
        self.decode_or_draft_inflight_limit: int = 1
        self.decode_or_draft_inflight_count: int = 0
        # DECODE_FIRST heads picked but not yet completed (update_from_output).
        # DRAFT_FIRST and DECODE_FIRST use different recv primitives but share
        # the DECODE stream, so they must not be in flight simultaneously
        # (the cloud's recv order could mismatch the edge's send order).
        # DRAFT_FIRST+DRAFT_FIRST is safe (same primitive, FIFO), so draft
        # pipelining only needs to gate on decode heads, not total heads.
        self.decode_head_inflight_count: int = 0
        self.draft_remote_pending_count: int = 0

        # Phase6 data-plane channel manager — per-dp_rank slice.
        dp_rank = getattr(self.vllm_config.parallel_config,
                          "data_parallel_rank", 0)
        is_shared = getattr(self.vllm_config.parallel_config,
                            "is_shared_model_edge", False)
        self.hidden_channel_manager = HiddenChannelManager(
            dp_rank=dp_rank,
            is_shared_model_edge=is_shared,
        )
        logger.info(
            "[PD] HiddenChannelManager(dp_rank=%d): prefill_pool=%s decode_pool=%s",
            dp_rank,
            self.hidden_channel_manager.prefill_pool,
            self.hidden_channel_manager.decode_pool,
        )

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
        self._force_draft_last: bool = False

        self._layer_slice_config_path: str | None = None
        self._layer_slice_config_mtime: float = 0.0
        self._load_layer_slice_config()
        # After scheduling a DECODE_LAST or DRAFT_LAST, briefly reserve the
        # next scheduling opportunity for DECODE_FIRST or DRAFT_FIRST only.
        self._decode_or_draft_first_only_start_ts: float | None = None
        self._decode_or_draft_first_only_window_ms: int = 10

        # [MTP] DRAFT_LAST delay scheduling (mirrors decode_last_delay).
        self._draft_last_delay_start_ts: float | None = None
        self._draft_last_delay_schedule_ms: int = 10

        # Async scheduled-MTP keeps real draft token IDs in the edge worker.
        # The scheduler only needs fixed-length placeholder SchedulerOutputs,
        # which can be generated and dispatched before the preceding worker
        # result is returned to EngineCore.  Cloud publication is finalized
        # separately once the target sampling scalars are available.
        self._draft_first_cloud_publish_pending: SchedulerOutput | None = None
        self._draft_first_scalars_patched: bool = False
        self._draft_first_dispatched: bool = False
        self._pregenerated_draft_task_ids: set[str] = set()
        self._pregenerated_draft_req_ids: dict[str, set[str]] = {}
        self._draft_remote_pending_limit: int = 2
        self._decode_first_placeholder_parent: SchedulerOutput | None = None

        # ------------------------------------------------------------------ #
        # Edge-cloud deferred-draft KV retention                             #
        # ------------------------------------------------------------------ #
        # Non-edge-cloud proposes draft tokens inside the same execute_model
        # call, i.e. strictly BEFORE update_from_output frees the finished
        # requests' KV blocks.  Edge-cloud defers the draft to a later
        # DRAFT_FIRST batch, so without retention the blocks could be freed
        # and reused before the cloud-side draft steps read/write them.
        # The EngineCore registers each deferred task locally from the parent
        # SchedulerOutput's head_token before update_from_output. _free_request
        # then delays _free_blocks for referenced requests until the draft task
        # completes or is dropped (release_draft_retained_blocks).
        # Everything is gated on `_edge_cloud_draft_retention_enabled` so
        # deployments without the scheduled edge-cloud draft behave exactly
        # as upstream.
        self._edge_cloud_draft_retention_enabled: bool = (
            self._check_scheduled_edge_cloud_draft()
        )
        self._edge_cloud_draft_req_tasks: dict[str, set[str]] = {}
        self._edge_cloud_draft_task_reqs: dict[str, set[str]] = {}
        self._draft_retained_requests: dict[str, dict[str, Request]] = {}
        # Draft task ids this scheduler dropped from its ready queues
        # while the runner still held the (enqueued) context.  Drained by
        # the EngineCore patch, which forwards them to the runner so the
        # orphaned context is reaped and the retention released.
        self._dropped_draft_task_ids_to_report: list[str] = []
        # Draft task ids whose cloud-side cached metadata will never be
        # (fully) consumed.  Stamped onto outgoing SchedulerOutputs so
        # the cloud model runner can purge the entries instead of
        # leaking them until the bounded cache evicts.
        self._pending_cloud_draft_invalidations: list[str] = []

    def invalidate_cloud_draft_tasks(self, task_ids: list[str]) -> None:
        """Queue cloud-side draft metadata invalidations (edge only)."""
        if not self._edge_cloud_draft_retention_enabled:
            return
        self._pending_cloud_draft_invalidations.extend(task_ids)

    def schedule(self) -> SchedulerOutput:
        scheduler_output = self._schedule_pd_separated()
        # Only FIRST-segment batches are published to the cloud over PRE_OUT
        # (the publish hook drops PL/DL/DRL tails), and only batches whose
        # cloud-side execution runs the purge hook can deliver the
        # invalidations.  EMPTY batches are not broadcast either.  Stamping
        # any other batch type would silently discard the pending list, so
        # keep the invalidations queued until a cloud-bound batch can carry
        # them.
        if self._pending_cloud_draft_invalidations and (
            scheduler_output.batch_type
            in (
                BatchType.PREFILL_FIRST,
                BatchType.DECODE_FIRST,
                BatchType.DRAFT_FIRST,
            )
        ):
            scheduler_output.cloud_draft_invalidate_task_ids = (
                self._pending_cloud_draft_invalidations
            )
            self._pending_cloud_draft_invalidations = []
        return scheduler_output

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

    def _decode_or_draft_first_only_active(self) -> bool:
        started_at = self._decode_or_draft_first_only_start_ts
        if started_at is None:
            return False
        elapsed_ms = (time.monotonic() - started_at) * 1000
        if elapsed_ms >= self._decode_or_draft_first_only_window_ms:
            self._decode_or_draft_first_only_start_ts = None
            return False
        return True

    def _start_decode_or_draft_first_only_window(self) -> None:
        self._decode_or_draft_first_only_start_ts = time.monotonic()

    def _clear_decode_or_draft_first_only_window(self) -> None:
        self._decode_or_draft_first_only_start_ts = None

    def _pick_decode_or_draft_first_only_or_empty(self) -> SchedulerOutput | None:
        if self._has_draft_work():
            # A pending draft chain must not be held back by the
            # decode-first-only window: returning EMPTY here breaks the
            # batch_queue fill loop right after a DECODE_LAST pick, so the
            # pre-generated DRF/DRL placeholder chain never reaches the
            # worker MQ in the same EngineCore turn.
            return None
        if not self._decode_or_draft_first_only_active():
            return None
        # Window: allow Draft首 (priority) or Decode首
        if self._can_schedule_draft_first():
            self._clear_decode_or_draft_first_only_window()
            return self._pick_draft_first_batch()
        if self._can_schedule_decode_first():
            self._clear_decode_or_draft_first_only_window()
            return self._pick_decode_first_batch()
        return self._make_empty_batch()

    def _pick_by_state(self, state: PrefillState) -> SchedulerOutput:
        if self._decode_first_placeholder_parent is not None:
            self._prepare_next_decode_first_placeholder(
                self._decode_first_placeholder_parent
            )
        # A placeholder DECODE_FIRST prepared when the final DRAFT_LAST was
        # dispatched must stay immediately behind that draft tail.  Its real
        # draft token IDs are filled from the worker-local _draft_token_ids
        # buffer when it executes, exactly like native async spec decode.
        if self.decodes_first_ready:
            return self.decodes_first_ready.popleft()

        first_only = self._pick_decode_or_draft_first_only_or_empty()
        if first_only is not None:
            return first_only

        if state == PrefillState.IDLE:
            # IDLE: P首/chunk0首 > Draft尾 > Draft首 > Decode尾 > Decode首 > Empty
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
            if self.drafts_last_ready and self._can_schedule_draft_last():
                return self._pick_draft_last_batch()
            if self._can_schedule_draft_first():
                return self._pick_draft_first_batch()
            if self.decodes_last_ready and self._can_schedule_decode_last():
                return self._pick_decode_last_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            return self._make_empty_batch()

        if state == PrefillState.LOW:
            # LOW: P首 > Draft尾 > Draft首 > D尾 > D首 > P尾 > Empty
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
            if self.drafts_last_ready and self._can_schedule_draft_last():
                return self._pick_draft_last_batch()
            if self._can_schedule_draft_first():
                return self._pick_draft_first_batch()
            if self.decodes_last_ready and self._can_schedule_decode_last():
                return self._pick_decode_last_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if self.prefills_last_ready:
                return self._pick_prefill_last_batch()
            return self._make_empty_batch()

        # HIGH: Draft尾 > Draft首 > D尾 > D首 > P尾 > Empty
        if self.drafts_last_ready and self._can_schedule_draft_last():
            return self._pick_draft_last_batch()
        if self._can_schedule_draft_first():
            return self._pick_draft_first_batch()
        if self.decodes_last_ready and self._can_schedule_decode_last():
            return self._pick_decode_last_batch()
        if self._can_schedule_decode_first():
            return self._pick_decode_first_batch()
        if self.prefills_last_ready:
            return self._pick_prefill_last_batch()
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
                or self.decode_or_draft_inflight_count > 0
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
            and self.decode_or_draft_inflight_count == 0
            and self.draft_remote_pending_count == 0
            and not self.drafts_first_ready
            and not self.drafts_last_ready
            and not self._force_decode_last
            and not self._force_draft_last
        )

    def _can_schedule_draft_first(self) -> bool:
        if not self.drafts_first_ready:
            return False
        next_output = self.drafts_first_ready[0]
        is_pregenerated = (
            next_output.draft_task_id in self._pregenerated_draft_task_ids
        )
        if is_pregenerated:
            # The edge and cloud workers consume the pre-generated chain in
            # strict FIFO order.  Do not wait for a DRAFT_FIRST result merely
            # to decrement draft_inflight_count; bound the amount of queued
            # work using the remote-pending credit instead.
            # `not self._force_draft_last` enforces the DRAFT_FIRST ->
            # DRAFT_LAST alternation invariant the same way the legacy
            # branch does: once a DRAFT_FIRST is picked, the next draft head
            # may not be picked until its DRAFT_LAST is picked (which clears
            # the flag).  Without it, a DRAFT_LAST dropped/drained without
            # clearing the flag would let a second DRAFT_FIRST through.
            # `decode_head_inflight_count == 0` (not total inflight == 0)
            # gates only on DECODE_FIRST heads: a DRAFT_FIRST is an edge->cloud
            # send while a DRAFT_LAST is a cloud->edge recv, so a second
            # DRAFT_FIRST may be dispatched while the previous DRAFT_LAST is
            # still in flight (draft pipelining).  But DRAFT_FIRST and
            # DECODE_FIRST use different recv primitives on the same stream,
            # so they must never be in flight together -- hence gate on decode
            # heads only, allowing draft+draft but not draft+decode.
            return bool(
                self.decode_head_inflight_count == 0
                and self.draft_remote_pending_count
                < self._draft_remote_pending_limit
                and not self.drafts_last_ready
                and not self._force_decode_last
                and not self._force_draft_last
            )

        # Scheduled draft head/tail payloads share the DECODE channel.
        # Do not start another head while an earlier head is still remote or
        # its tail is ready locally: otherwise edge and cloud can each wait
        # for the opposite-direction send before posting the matching recv.
        return bool(
            self.decode_or_draft_inflight_count == 0
            and self.draft_remote_pending_count == 0
            and not self.drafts_last_ready
            and not self._force_decode_last
            and not self._force_draft_last
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
                f"draft_remote_pending: {self.draft_remote_pending_count}, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}, "
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
                f"draft_remote_pending: {self.draft_remote_pending_count}, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
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
            _draft_key = "draft_last_delay_schedule_ms"
            if _draft_key in raw:
                try:
                    self._draft_last_delay_schedule_ms = int(raw[_draft_key])
                    logger.info(
                        "[PDSeparatedScheduler] %s set to %d from %s",
                        _draft_key, self._draft_last_delay_schedule_ms, yaml_path,
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid %s value %r in %s; keeping %d",
                        _draft_key, raw[_draft_key], yaml_path,
                        self._draft_last_delay_schedule_ms,
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

    # ------------------------------------------------------------------ #
    # DRAFT_LAST delay scheduling                                         #
    # ------------------------------------------------------------------ #
    def _start_draft_last_delay(self) -> None:
        """Draft首 pick 后启动，Draft尾 在延迟到期前不可被调度。"""
        self._draft_last_delay_start_ts = time.monotonic()

    def _can_schedule_draft_last(self) -> bool:
        """Return True if the delay since DRAFT_FIRST has elapsed."""
        if self._draft_last_delay_start_ts is None:
            return True
        elapsed_ms = (time.monotonic() - self._draft_last_delay_start_ts) * 1000
        if elapsed_ms >= self._draft_last_delay_schedule_ms:
            self._draft_last_delay_start_ts = None
            return True
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
                    # KV cache exhausted: super().schedule() scheduled nothing.
                    # _prepare_pf_running_state swapped self.running for the
                    # prefill candidate(s), so we MUST restore self.running =
                    # saved_running here too (the non-empty branch restores
                    # below).  Without this restore the decode requests that
                    # were in self.running are lost -- _can_schedule_decode_first
                    # never sees them, KV is never freed by finished decodes, and
                    # prefill keeps returning empty -> deadlock.  Also re-prepend
                    # the candidates that were not exposed to super() this round.
                    self.chunk_prefill_first = (
                        rest_candidates + self.chunk_prefill_first
                    )
                    self.running = saved_running
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
                        logger.info(
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
        # Mark whether this PL is the request's last prefill chunk.  Mid-chunk
        # PL still has to run the drafter so that the MTP layer populates its
        # KV cache for every prompt chunk; its sampled/draft tokens are only
        # discarded after the draft forward.  The flight is still in the map
        # here and is popped later in
        # _update_from_output_prefill_last_chunk_prior.
        flight = (
            self._prefill_flight_by_token.get(so.head_token)
            if so.head_token else None
        )
        batch_req_ids = tuple(so.num_scheduled_tokens)
        if flight is not None:
            draft_output_req_ids = (
                batch_req_ids if flight.is_last_chunk else ()
            )
        else:
            # Legacy mode may batch several prefill requests.  Preserve the
            # per-request distinction: every row warms draft KV, while only
            # requests whose prompt is complete may publish proposals.
            draft_output_req_ids = tuple(
                req_id
                for req_id in batch_req_ids
                if (request := self.requests.get(req_id)) is not None
                and not request.is_prefill_chunk
            )
        so.draft_output_req_ids = draft_output_req_ids
        so.is_last_prefill_chunk = bool(draft_output_req_ids)
        # Drop these reqs from chunk_prefill_first. Keep them in
        # prefill_last_pending until update_from_output() moves them to running.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        self._validate_prefill_tail_channel(so)
        # Every chunk needs a draft-prefill pass.  Only the last chunk's draft
        # tokens are published to the following target verify batch; the
        # worker uses draft_output_req_ids to discard mid-chunk outputs.
        self._pregenerate_draft_chain(so)
        return so

    def _validate_prefill_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        token = scheduler_output.head_token
        channel = scheduler_output.hidden_channel
        if not token:
            raise RuntimeError("PREFILL_LAST missing head_token")
        pool = self.hidden_channel_manager.prefill_pool
        if channel not in pool:
            raise RuntimeError(
                f"PREFILL_LAST expects a prefill hidden channel from "
                f"{pool}, got {channel}"
            )
        expected = self.hidden_channel_manager.get_channel(token)
        if expected != channel:
            raise RuntimeError(
                f"PREFILL_LAST hidden channel mismatch: expected {expected}, "
                f"got {channel}, head_token={token}"
            )

    def _validate_decode_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        pool = self.hidden_channel_manager.decode_pool
        if scheduler_output.hidden_channel not in pool:
            raise RuntimeError(
                f"DECODE_LAST expects a decode hidden channel from "
                f"{pool}, got {scheduler_output.hidden_channel}"
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
        while self.drafts_first_ready:
            scheduler_output = self.drafts_first_ready.popleft()
            if self._is_stale_draft_output(scheduler_output):
                if scheduler_output.draft_task_id:
                    self._pregenerated_draft_task_ids.discard(
                        scheduler_output.draft_task_id
                    )
                    self._pregenerated_draft_req_ids.pop(
                        scheduler_output.draft_task_id, None
                    )
                    # Report the cut chain so EngineCore can release the
                    # retained KV blocks and invalidate the cloud-side
                    # cached draft metadata (which will never be fully
                    # consumed now).
                    self._dropped_draft_task_ids_to_report.append(
                        scheduler_output.draft_task_id
                    )
                if scheduler_output is self._draft_first_cloud_publish_pending:
                    self._draft_first_cloud_publish_pending = None
                    self._draft_first_scalars_patched = False
                    self._draft_first_dispatched = False
                logger.info(
                    "[PD] drop stale DRAFT_FIRST task_id=%s step=%s",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
                continue
            break
        else:
            return self._make_empty_batch()

        if scheduler_output is self._draft_first_cloud_publish_pending:
            self._draft_first_dispatched = True
            if self._draft_first_scalars_patched:
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
        # is_last_prefill_chunk is a downstream dynamic SchedulerOutput
        # attribute, so dataclasses.replace() does not preserve it.
        draft_last.is_last_prefill_chunk = getattr(
            scheduler_output, "is_last_prefill_chunk", True
        )
        draft_last.draft_output_req_ids = getattr(
            scheduler_output,
            "draft_output_req_ids",
            tuple(scheduler_output.num_scheduled_tokens),
        )
        self._validate_draft_tail_channel(draft_last)
        self.drafts_last_ready.append(draft_last)
        self.decode_or_draft_inflight_count += 1
        self.draft_remote_pending_count += 1
        self._force_draft_last = True
        self._start_draft_last_delay()

        logger.info(
            "[MTP-DEBUG] scheduler picked DRAFT_FIRST: task_id=%s, "
            "parent_req_id=%s, draft_step_idx=%s, head_token=%s, "
            "remaining_ready=%d, decode_or_draft_inflight=%d, draft_remote_pending=%d",
            scheduler_output.draft_task_id,
            scheduler_output.parent_req_id,
            scheduler_output.draft_step_idx,
            scheduler_output.head_token,
            len(self.drafts_first_ready),
            self.decode_or_draft_inflight_count,
            self.draft_remote_pending_count,
        )
        return scheduler_output

    def _uses_async_scheduled_mtp_placeholders(self) -> bool:
        """Whether scheduled MTP can use native async placeholder semantics."""
        if not getattr(self.scheduler_config, "async_scheduling", False):
            return False
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or self.num_spec_tokens <= 0:
            return False
        method = getattr(speculative_config, "method", None)
        if method in ("qwen3_5_mtp", "qwen_mtp"):
            return True
        if method != "mtp":
            return False
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        return "qwen" in str(
            getattr(hf_config, "model_type", "")
        ).lower()

    def _pregenerate_draft_chain(
        self, target_tail: SchedulerOutput
    ) -> None:
        """Create fixed-length placeholder DRF tasks at target-tail pick time.

        The real token IDs remain in the edge worker.  FIFO ordering ensures
        every DRF executes after the DRL that produces its local inputs.  Only
        the step-0 accepted-token scalars are finalized later for the cloud;
        they are not consumed by the edge worker.
        """
        if not self._uses_async_scheduled_mtp_placeholders():
            return
        if (
            self.drafts_first_ready
            or self.drafts_last_ready
            or self._draft_first_cloud_publish_pending is not None
        ):
            return
        req_ids = list(target_tail.num_scheduled_tokens)
        if not req_ids or any(
            req_id not in self.requests for req_id in req_ids
        ):
            return
        task_id = target_tail.head_token
        if not task_id:
            return

        for step_idx in range(self.num_spec_tokens):
            draft_first = replace(
                target_tail,
                batch_type=BatchType.DRAFT_FIRST,
                head_token=None,
                hidden_channel=HiddenChannelType.DECODE,
                parent_req_id=req_ids[0],
                draft_task_id=task_id,
                draft_step_idx=step_idx,
                num_accepted_tokens=None,
                valid_sampled_token_count=None,
            )
            # Preserve the parent prefill phase across the independently
            # scheduled draft chain.  Mid-chunk chains warm the draft KV but
            # must not seed a target decode placeholder.
            draft_first.is_last_prefill_chunk = getattr(
                target_tail, "is_last_prefill_chunk", True
            )
            draft_first.draft_output_req_ids = getattr(
                target_tail,
                "draft_output_req_ids",
                tuple(target_tail.num_scheduled_tokens),
            )
            self.drafts_first_ready.append(draft_first)
            if step_idx == 0:
                self._draft_first_cloud_publish_pending = draft_first
                self._draft_first_scalars_patched = False
                self._draft_first_dispatched = False

        self._pregenerated_draft_task_ids.add(task_id)
        self._pregenerated_draft_req_ids[task_id] = set(req_ids)
        logger.info(
            "[PD] pre-generated async MTP placeholders task_id=%s steps=%d",
            task_id,
            self.num_spec_tokens,
        )

    def finalize_pre_generated_draft_first(
        self,
        *,
        draft_task_id: str,
        num_accepted_tokens: list[int] | None,
        valid_sampled_token_count: list[int] | None,
    ) -> SchedulerOutput | None:
        """Patch cloud-only sampling state into the queued step-0 control."""
        pending = self._draft_first_cloud_publish_pending
        if (
            pending is None
            or pending.draft_task_id != draft_task_id
        ):
            return None
        pending.num_accepted_tokens = num_accepted_tokens
        pending.valid_sampled_token_count = valid_sampled_token_count
        if not self._draft_first_dispatched:
            self._draft_first_scalars_patched = True
            return pending
        self._draft_first_cloud_publish_pending = None
        self._draft_first_scalars_patched = False
        return pending

    def is_pre_generated_draft(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        task_id = (
            scheduler_output.draft_task_id
            or scheduler_output.head_token
        )
        return bool(task_id and task_id in self._pregenerated_draft_task_ids)

    def active_pre_generated_draft_req_ids(self) -> set[str]:
        active: set[str] = set()
        for task_id in self._pregenerated_draft_task_ids:
            active.update(
                self._pregenerated_draft_req_ids.get(task_id, ())
            )
        return active

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
        # With KV retention, finished requests stay in self.requests until
        # their draft chain releases them, so this refusal only fires when
        # a request is genuinely gone (e.g. aborted before the parent
        # output was processed).  The chain then can never (fully) run:
        # report the task so EngineCore releases the retained KV blocks
        # and invalidates the cloud-side cached metadata.
        if not req_ids or any(
            req_id not in self.requests for req_id in req_ids
        ):
            if draft_task_id:
                self._dropped_draft_task_ids_to_report.append(draft_task_id)
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
        draft_first.is_last_prefill_chunk = getattr(
            source, "is_last_prefill_chunk", True
        )
        draft_first.draft_output_req_ids = getattr(
            source,
            "draft_output_req_ids",
            tuple(source.num_scheduled_tokens),
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
            if next_step_idx >= self.num_spec_tokens:
                self._pregenerated_draft_task_ids.discard(task_id)
                self._pregenerated_draft_req_ids.pop(task_id, None)
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
            # A DRAFT_LAST here always has its DRAFT_FIRST already dispatched
            # to the cloud (it was self-posted in _pick_draft_first_batch when
            # the head was picked).  The cloud does not track request
            # lifecycle, so it will isend a response even if the owning
            # request has since finished/aborted.  The edge MUST still execute
            # this tail (recv) to keep the DECODE hidden channel paired --
            # never drop it.  When the request is gone the worker drains the
            # recv and skips the tail-segment compute (see
            # _run_edge_cloud_draft_last_segment); we also must not spawn a
            # verify placeholder for a dead request.
            self._force_draft_last = False
            self._start_decode_or_draft_first_only_window()
            output_req_ids = getattr(
                scheduler_output,
                "draft_output_req_ids",
                tuple(scheduler_output.num_scheduled_tokens),
            )
            has_live_output_req = any(
                (request := self.requests.get(req_id)) is not None
                and not request.is_finished()
                for req_id in output_req_ids
            )
            if has_live_output_req:
                self._prepare_next_decode_first_placeholder(scheduler_output)
            elif not output_req_ids:
                logger.info(
                    "[PD] finish DRAFT_LAST task_id=%s step=%s "
                    "(mid-prefill KV warmup; no verify placeholder)",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
            else:
                logger.info(
                    "[PD] drain DRAFT_LAST task_id=%s step=%s "
                    "(request gone; worker will drain cloud response)",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
            return scheduler_output
        return self._make_empty_batch()

    def _prepare_next_decode_first_placeholder(
        self, draft_last: SchedulerOutput
    ) -> None:
        """Prepare the next target verify batch behind the final draft tail.

        Scheduler-side spec token values are placeholders.  The edge worker
        replaces them with its local ``_draft_token_ids`` after this DRL has
        executed, so no DraftTokenIds round-trip through EngineCore is needed.
        """
        if not self._uses_async_scheduled_mtp_placeholders():
            return
        if draft_last.draft_task_id not in self._pregenerated_draft_task_ids:
            self._decode_first_placeholder_parent = None
            return
        step_idx = int(draft_last.draft_step_idx or 0)
        if step_idx + 1 < self.num_spec_tokens:
            return
        self._decode_first_placeholder_parent = draft_last
        if self.decodes_first_ready:
            self._decode_first_placeholder_parent = None
            return
        if not self.running:
            # A chain pre-generated from the final PREFILL_LAST can reach its
            # final DRL before EngineCore has applied the prefill result. Retry
            # on the next schedule turn after that request moves to running.
            return

        next_decode = self._pick_decode_first_batch()
        if (
            next_decode is not None
            and next_decode.batch_type == BatchType.DECODE_FIRST
            and next_decode.total_num_scheduled_tokens > 0
        ):
            self.decodes_first_ready.append(next_decode)
            self._decode_first_placeholder_parent = None

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

    # ------------------------------------------------------------------ #
    # Edge-cloud deferred-draft KV retention                              #
    # ------------------------------------------------------------------ #
    def _check_scheduled_edge_cloud_draft(self) -> bool:
        """Mirror of the runner/engine-core scheduled edge-cloud draft
        predicate.  Gates the retention logic so schedulers without the
        edge-cloud draft methods behave exactly as upstream.
        """
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None:
            return False
        if not getattr(
            self.vllm_config.parallel_config, "enable_edge_cloud", False
        ):
            return False
        method = getattr(speculative_config, "method", None)
        if method == "eagle3":
            return True
        if method in ("qwen3_5_mtp", "qwen_mtp"):
            return True
        if method != "mtp":
            return False
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        return "qwen" in str(getattr(hf_config, "model_type", "")).lower()

    def register_edge_cloud_draft_task(
        self, task_id: str, req_ids: set[str]
    ) -> None:
        """Register a deferred draft before its parent output is applied."""
        if (
            not self._edge_cloud_draft_retention_enabled
            or not task_id
            or not req_ids
        ):
            return
        previous_req_ids = self._edge_cloud_draft_task_reqs.get(task_id)
        if previous_req_ids is not None:
            if previous_req_ids != req_ids:
                raise RuntimeError(
                    "Deferred draft task registration changed request set: "
                    f"task_id={task_id}, previous={previous_req_ids}, "
                    f"new={req_ids}"
                )
            return
        self._edge_cloud_draft_task_reqs[task_id] = set(req_ids)
        for req_id in req_ids:
            self._edge_cloud_draft_req_tasks.setdefault(req_id, set()).add(
                task_id
            )

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        task_ids = (
            self._edge_cloud_draft_req_tasks.get(request.request_id)
            if self._edge_cloud_draft_retention_enabled
            else None
        )
        if not task_ids:
            return super()._free_request(request, delay_free_blocks)
        # The request is referenced by a pending/in-flight deferred draft.
        # Finish it normally (outputs, finished_req_ids) but keep its KV
        # blocks until the draft chain completes on the cloud — same
        # ordering guarantee the non-edge-cloud path gets for free by
        # proposing drafts inside execute_model.
        for task_id in task_ids:
            self._draft_retained_requests.setdefault(task_id, {})[
                request.request_id
            ] = request
        return super()._free_request(request, delay_free_blocks=True)

    def release_draft_retained_blocks(self, task_id: str) -> None:
        """Free KV blocks retained for a completed/dropped draft task.

        Idempotent.  A request referenced by several in-flight draft
        tasks is only freed once its last referencing task releases.
        """
        task_req_ids = self._edge_cloud_draft_task_reqs.pop(task_id, set())
        retained = self._draft_retained_requests.pop(task_id, {})
        for req_id in task_req_ids:
            req_tasks = self._edge_cloud_draft_req_tasks.get(req_id)
            if req_tasks is not None:
                req_tasks.discard(task_id)
                if req_tasks:
                    # Still referenced by another in-flight draft task.
                    continue
                self._edge_cloud_draft_req_tasks.pop(req_id, None)
            request = retained.get(req_id)
            if request is None:
                continue
            current = self.requests.get(req_id)
            if current is request and request.is_finished():
                self._free_blocks(request)
            elif current is not None and current is not request:
                # req_id resubmitted after finishing: free the old
                # request's blocks without evicting the new entry.
                self.kv_cache_manager.free(request)

    def _scheduler_output_all_requests_finished(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        batch_req_ids = self._edge_cloud_draft_task_reqs.get(
            scheduler_output.draft_task_id or ""
        )
        if batch_req_ids is None:
            batch_req_ids = set(scheduler_output.num_scheduled_tokens)
        else:
            batch_req_ids = set(batch_req_ids)
        if scheduler_output.parent_req_id is not None:
            batch_req_ids.add(scheduler_output.parent_req_id)
        return bool(batch_req_ids) and all(
            (request := self.requests.get(req_id)) is None
            or request.is_finished()
            for req_id in batch_req_ids
        )

    def _drop_stale_drafts_for_req_ids(self, req_ids: set[str]) -> None:
        if not req_ids:
            return
        # Aligned with the model runner's deferred-draft policy: a draft
        # batch is dropped only when EVERY request it covers has finished.
        # Partial finishes keep the draft — the cloud-side cached
        # attention metadata is whole-batch and cannot be re-sliced.
        # Dropped task ids are reported to the runner (which may still
        # hold the enqueued context) via take_dropped_draft_task_ids().
        kept_first: deque[SchedulerOutput] = deque()
        for output in self.drafts_first_ready:
            if (
                self._scheduler_output_intersects_req_ids(output, req_ids)
                and self._scheduler_output_all_requests_finished(output)
            ):
                task_id = output.draft_task_id
                if task_id is not None:
                    self._pregenerated_draft_task_ids.discard(task_id)
                    self._pregenerated_draft_req_ids.pop(task_id, None)
                    self._dropped_draft_task_ids_to_report.append(task_id)
                if output is self._draft_first_cloud_publish_pending:
                    self._draft_first_cloud_publish_pending = None
                    self._draft_first_scalars_patched = False
                    self._draft_first_dispatched = False
            else:
                kept_first.append(output)
        self.drafts_first_ready = kept_first

        self.decodes_first_ready = deque(
            output
            for output in self.decodes_first_ready
            if not self._scheduler_output_intersects_req_ids(
                output, req_ids
            )
        )
        pending_decode = self._decode_first_placeholder_parent
        if (
            pending_decode is not None
            and self._scheduler_output_intersects_req_ids(
                pending_decode, req_ids
            )
        ):
            self._decode_first_placeholder_parent = None
        # Never drop a queued DRAFT_LAST.  Tails are self-posted in
        # _pick_draft_first_batch at the moment their head is picked, and
        # a picked DRAFT_FIRST is always dispatched to the cloud.  The
        # cloud does not track request lifecycle, so it will still isend
        # the response; the edge MUST execute (drain) the tail to keep
        # the DECODE hidden channel paired (see _pick_draft_last_batch
        # and the drain path in the worker's
        # _run_edge_cloud_draft_last_segment).  Dropping the tail also
        # strands _force_draft_last=True -- the flag is only cleared when
        # a DRAFT_LAST is picked -- which then blocks every future
        # DRAFT_FIRST/DECODE_FIRST and deadlocks the scheduler.
        for output in self.drafts_last_ready:
            if self._scheduler_output_intersects_req_ids(output, req_ids):
                gone = {
                    rid
                    for rid in output.num_scheduled_tokens
                    if rid in req_ids
                }
                if output.parent_req_id in req_ids:
                    gone.add(output.parent_req_id)
                logger.info(
                    "[PD] keep DRAFT_LAST task_id=%s step=%s for drain "
                    "(%d member request(s) gone; its DRAFT_FIRST was "
                    "already dispatched to the cloud)",
                    output.draft_task_id,
                    output.draft_step_idx,
                    len(gone),
                )

    def take_dropped_draft_task_ids(self) -> list[str]:
        """Drain draft task ids dropped from the ready queues since the
        last call (EngineCore patch forwards them to the runner)."""
        dropped = self._dropped_draft_task_ids_to_report
        self._dropped_draft_task_ids_to_report = []
        return dropped

    def _is_stale_draft_output(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        # A draft output is stale when EVERY backing request has finished.
        # This drives the DRAFT_FIRST skip in _pick_draft_first_batch:
        # once all owning requests are gone, future (not-yet-dispatched)
        # draft heads must not be picked -- the edge can no longer produce
        # their payload (the draft context was cleared on finish/abort) and
        # the cloud would be left waiting for data that never arrives.
        # Partial finishes keep the draft alive: the cloud-side cached
        # attention metadata is whole-batch and cannot be re-sliced, so the
        # chain runs to completion and the dead rows' draft tokens are
        # discarded by the worker (_run_edge_cloud_draft_last_segment).
        # Already-dispatched heads are handled separately: their matching
        # DRAFT_LAST is always executed (drained) in _pick_draft_last_batch
        # to pair the cloud's response, so this check intentionally does NOT
        # exempt pre-generated dispatched chains the way it used to.
        # NOTE: finished requests may still be present in self.requests
        # while their KV blocks are retained for an in-flight draft chain,
        # so liveness must go through is_finished() (via
        # _scheduler_output_all_requests_finished), not dict membership.
        return self._scheduler_output_all_requests_finished(scheduler_output)

    def _draft_output_reqs_live(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        """True if any request backing this draft output is still active.

        Used both to decide whether a DRAFT_LAST may spawn a verify
        placeholder and (inverted) whether a DRAFT_FIRST is stale.
        """
        return not self._scheduler_output_all_requests_finished(
            scheduler_output
        )

    def _pick_decode_last_batch(self) -> SchedulerOutput:
        if not self.decodes_last_ready:
            return self._make_empty_batch()
        so = self.decodes_last_ready.popleft()
        assert so.batch_type == BatchType.DECODE_LAST, (
            f"decodes_last_ready expects DECODE_LAST, got {so.batch_type}"
        )
        self._validate_decode_tail_channel(so)
        self._start_decode_or_draft_first_only_window()
        self._force_decode_last = False
        self._pregenerate_draft_chain(so)
        return so

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
            if req_id not in cached_reqs.all_token_ids:
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
                    logger.error(
                        "DECODE_FIRST returned empty batch (total_num_scheduled_tokens=0) "
                        "despite %d running requests. This indicates a severe KV cache "
                        "shortage or scheduler budget exhaustion.",
                        len(self.running),
                    )
                else:
                    scheduler_output.batch_type = BatchType.DECODE_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.hidden_channel = (
                        self.hidden_channel_manager.decode_channel()
                    )
                    self._ensure_cached_all_token_ids(scheduler_output)
                    self.decode_or_draft_inflight_count += 1
                    self.decode_head_inflight_count += 1
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
            # A preempted request lands in chunk_prefill_first (status
            # stays PREEMPTED) and may be re-added to running here.  If
            # super().schedule() later selects it for preemption again,
            # _preempt_request asserts status == RUNNING.  Restore the
            # status unconditionally: the request was fully prefilled and
            # is entering decode — it must be RUNNING.
            if req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        # In PD separation requests may re-enter ``running`` from
        # ``chunk_prefill_first`` with a stale ``PREEMPTED`` status when the
        # ``_migrate_prefill_to_running`` conditions are not met in time.
        # The upstream scheduler only picks ``RUNNING`` requests to preempt,
        # so the status is a lagging indicator; ensure it is correct.
        if request.status != RequestStatus.RUNNING:
            request.status = RequestStatus.RUNNING
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
            if self.decode_or_draft_inflight_count > 0:
                self.decode_or_draft_inflight_count -= 1
            if self.decode_head_inflight_count > 0:
                self.decode_head_inflight_count -= 1
            logger.info(
                f"[PD] update_from_output DECODE_FIRST done, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
            )
        if scheduler_output.batch_type == BatchType.DRAFT_FIRST:
            self.decode_or_draft_inflight_count = max(
                0, self.decode_or_draft_inflight_count - 1
            )
            logger.info(
                "[PD] update_from_output DRAFT_FIRST done, "
                "decode_or_draft_inflight: %d/%d",
                self.decode_or_draft_inflight_count,
                self.decode_or_draft_inflight_limit,
            )
        enqueue_next_draft = (
            scheduler_output.batch_type == BatchType.DRAFT_LAST
        )
        if enqueue_next_draft:
            self.draft_remote_pending_count = max(
                0, self.draft_remote_pending_count - 1
            )
        if scheduler_output.batch_type == BatchType.DECODE_LAST:
            # decode_or_draft_inflight_count 已在 DECODE_FIRST 的 update_from_output
            # 中释放，此处不再重复减 1。
            logger.info(
                f"[PD] update_from_output DECODE_LAST done, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if self.finished_req_ids:
            # Natural completion is handled inside the base update path and
            # does not pass through finish_requests().  Do not dispatch any
            # not-yet-enqueued placeholder tasks for those requests.
            self._drop_stale_drafts_for_req_ids(self.finished_req_ids)
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
        # Drain finished requests from running: update_from_output removes
        # completed requests from self.requests, but they may still be in
        # running.  That causes KeyError in super().schedule() /
        # _update_after_schedule when the base scheduler accesses
        # self.requests[req_id].  Clean them up here, once per step.
        self.running = [
            req for req in self.running
            if req.request_id in self.requests
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
            self.decodes_first_ready
            or self.drafts_first_ready
            or self.drafts_last_ready
            or self.decode_or_draft_inflight_count > 0
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
