#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Inject ascend PD-separation / edge-cloud / passive-PP hooks into the
upstream :class:`vllm.v1.engine.core.EngineCore` and
:class:`vllm.v1.engine.core.EngineCoreProc` without modifying upstream
sources.

This patch is the home of every line that the vllm-pdmix downstream fork
used to maintain inside ``vllm/v1/engine/core.py`` of vLLM:

* ``EngineCore.__init__`` — late-stage construction of the optional
  edge-cloud PD-separation channel (``self._pp_pd_channel``).
* ``EngineCore.step`` / ``EngineCore.step_with_batch_queue`` — drain
  cloud-returned batches into the local PD scheduler, publish
  head-segment batches on PRE_OUT, skip ``sample_tokens`` for head
  batches, and assign ``head_token`` ids.
* ``EngineCore._drain_pd_channel_inbox`` /
  ``EngineCore._maybe_publish_pre_out`` /
  ``EngineCore._needs_sample_tokens`` — three new helper methods used by
  the two ``step*`` paths above.
* ``EngineCore.shutdown`` — release the channel before the rest of the
  engine resources.
* ``EngineCoreProc._process_input_queue`` — force a blocking
  ``input_queue.get`` when the engine has nothing local to do, so the
  edge node never busy-spins while waiting for the next client request.

Design notes
------------
1. ``__init__`` and ``shutdown`` only append behavior at the end and at
   the start, respectively, so they are wrapped (call original + extra).
2. ``step`` / ``step_with_batch_queue`` / ``_process_input_queue`` insert
   logic in the middle of the original method body. They are rewritten
   in full here, bytewise-equivalent to upstream when no PD/edge-cloud
   feature flag is on.
3. Every flag read uses ``getattr(parallel_config, ..., default)`` so
   that even if the dest-only ``ParallelConfig`` extension fields are
   absent, this patch behaves identically to upstream.
4. The patch is installed at import time. A guard prevents double
   patching if this module is imported twice (e.g. from a child
   process).

Upstream sync
-------------
The reimplementations of ``step()``, ``step_with_batch_queue()`` and
``_process_input_queue()`` track upstream
``vllm-0.20.2_layerwise/vllm/v1/engine/core.py``. Whenever vLLM moves to
a new minor version, re-diff these methods against the new upstream
source and re-apply the dest-only inserts.
"""
from __future__ import annotations

import functools
from concurrent.futures import Future
from typing import cast
from uuid import uuid4

from vllm.config import ParallelConfig
from vllm.logger import init_logger, logger as vllm_logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_ascend.v1.engine.passive_core import PPSchedulerZmqChannel

logger = init_logger(__name__)


# Idempotency guard: re-importing this module (e.g. from a child process)
# must not double-wrap the original methods.
_INSTALLED_FLAG = "_vllm_ascend_engine_core_patched"


# -----------------------------------------------------------------------#
# Original method handles captured before any wrapping happens.           #
# -----------------------------------------------------------------------#
_ORIG_ENGINE_CORE_INIT = EngineCore.__init__
_ORIG_ENGINE_CORE_SHUTDOWN = EngineCore.shutdown
_ORIG_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core


# =======================================================================#
# EngineCore.__init__ — append PD/edge-cloud setup at the very end.       #
# =======================================================================#
@functools.wraps(_ORIG_ENGINE_CORE_INIT)
def _patched_engine_core_init(self, *args, **kwargs):
    _ORIG_ENGINE_CORE_INIT(self, *args, **kwargs)

    parallel_config: ParallelConfig = self.vllm_config.parallel_config

    # PD-separation is owned by the ascend plugin and lives under
    # ``additional_config.edge_cloud_config.pd_separation``. ``init_ascend_config``
    # is idempotent and returns the cached singleton if already initialized
    # in the main process; in a freshly-spawned subprocess it re-initializes
    # from the ``vllm_config`` we hold.
    from vllm_ascend.ascend_config import init_ascend_config
    ascend_config = init_ascend_config(self.vllm_config)
    edge_cloud = getattr(ascend_config, "edge_cloud_config", None)
    pd_enabled = bool(
        edge_cloud is not None
        and getattr(edge_cloud, "enabled", False)
        and getattr(edge_cloud, "pd_separation", None) is not None
        and edge_cloud.pd_separation.enabled
    )

    if getattr(parallel_config, "enable_edge_cloud", False):
        logger.info(
            "Edge-cloud mode enabled (pd_separation=%s)",
            pd_enabled,
        )

    # Load PD-separation configuration from environment variables
    from vllm_ascend.pd_separation_config import PDSeparationConfig
    pd_config = PDSeparationConfig.from_env()

    # Edge-cloud PD-separation bidirectional ZMQ channel (edge side).
    self._pp_pd_channel = None
    if pd_enabled and getattr(parallel_config, "is_edge_node", False):
        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)

        # Discover the cloud's IP via a one-shot TCPStore. The edge
        # acts as store master on ``master_port + 1 + dp_rank`` so
        # that each DP rank has its own store port (no EADDRINUSE).
        # The cloud connects once per edge DP rank and writes its
        # ``get_ip()`` result. See passive_core.py for the
        # symmetric writer side.
        import torch.distributed as dist
        from datetime import timedelta
        _addr_store = dist.TCPStore(
            host_name=parallel_config.master_addr,
            port=parallel_config.master_port + 1 + dp_rank,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=300),
        )
        cloud_addr = _addr_store.get("cloud_ip").decode()
        del _addr_store

        # Each DP rank needs its own ZMQ port pair to avoid bind
        # conflicts within the same edge process. Offset by 2 per
        # dp_rank: dp_rank 0 → {pre_out, post_out}, dp_rank 1 →
        # {pre_out+2, post_out+2}, etc. The cloud side must mirror
        # this offsetting in its own PPSchedulerZmqChannel setup.
        pre_out_port = pd_config.pre_out_port + dp_rank * 2
        post_out_port = pd_config.post_out_port + dp_rank * 2
        pre_out = f"tcp://*:{pre_out_port}"
        post_out = f"tcp://{cloud_addr}:{post_out_port}"
        self._pp_pd_channel = PPSchedulerZmqChannel(
            send_endpoint=pre_out,
            recv_endpoint=post_out,
            name=f"pd-edge-dp{dp_rank}",
        )
        logger.info(
            "PD-separation edge channel: PRE_OUT=%s, POST_OUT=%s "
            "(cloud_addr=%s auto-discovered)",
            pre_out, post_out, cloud_addr,
        )


# =======================================================================#
# Three helper methods bound on EngineCore. Mirror the dest fork.         #
# =======================================================================#
def _drain_pd_channel_inbox(self) -> None:
    """Move cloud-returned SchedulerOutputs into the local PDSeparated
    scheduler's ``prefills_last_ready`` / ``decodes_last_ready`` queues.
    """
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    if not (
        hasattr(self.scheduler, "prefills_last_ready")
        and hasattr(self.scheduler, "decodes_last_ready")
        and hasattr(self.scheduler, "drafts_last_ready")
    ):
        return
    new_outputs = self._pp_pd_channel.consume_new_outputs()
    for _seq, so in new_outputs:
        bt = so.batch_type
        logger.info(f"Received scheduler_output from cloud, batch_type: {bt}")
        if bt == BatchType.PREFILL_LAST:
            self.scheduler.prefills_last_ready.append(so)
        elif bt == BatchType.DECODE_LAST:
            self.scheduler.decodes_last_ready.append(so)
        elif bt == BatchType.DRAFT_LAST:
            self.scheduler.drafts_last_ready.append(so)
        else:
            logger.error(
                "PD-separation POST_OUT received unexpected batch_type=%s; "
                "expected PREFILL_LAST, DECODE_LAST, or DRAFT_LAST. "
                "Dropping.",
                bt.value if bt is not None else "<none>",
            )


def _maybe_publish_pre_out(
    self, scheduler_output: SchedulerOutput
) -> None:
    """Forward DECODE_FIRST / DRAFT_FIRST batches on the edge → cloud
    channel immediately.

    Both are published synchronously at schedule time because their
    cloud-side middle segment must start as soon as possible to keep the
    decode/draft pipeline full — and because the edge worker enters the
    hidden-channel send right after dequeue, so a late notification leaves
    the cloud side of the channel rendezvous (gloo metadata + HCCL comm
    init) too far behind the edge side.

    PREFILL_FIRST is handled by _publish_pre_out_when_ready instead, which
    delays the ZMQ notification until the prefill head segment becomes the
    next batch to execute, preventing the cloud from blocking on irecv while
    the edge prefill is still queued behind other batches.
    """
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    bt = scheduler_output.batch_type
    if bt in (BatchType.DECODE_FIRST, BatchType.DRAFT_FIRST):
        self._pp_pd_channel.publish(scheduler_output)
    elif bt in (
        BatchType.EMPTY,
        BatchType.PREFILL_FIRST,
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
        BatchType.DRAFT_LAST,
    ):
        return
    else:
        logger.debug(
            "PD-separation PRE_OUT skipping non-separated batch_type=%s",
            bt.value if bt is not None else "<none>",
        )


def _ensure_pd_head_token(self, scheduler_output: SchedulerOutput) -> None:
    if getattr(self, "_pp_pd_channel", None) is None:
        return
    if scheduler_output.batch_type not in (
        BatchType.PREFILL_FIRST,
        BatchType.DECODE_FIRST,
        BatchType.DRAFT_FIRST,
    ):
        return
    if not scheduler_output.head_token:
        scheduler_output.head_token = uuid4().hex


def _publish_pre_out_when_ready(self) -> None:
    """Publish the oldest PREFILL_FIRST batch in batch_queue only when it
    becomes the next batch to execute (rightmost in the deque).

    This delays the ZMQ PRE_OUT notification for prefill head segments until
    the edge worker is about to actually execute them, preventing the cloud
    from blocking on irecv while the edge prefill head segment is still
    queued behind other batches.

    DRAFT_FIRST is intentionally NOT handled here: like DECODE_FIRST it is
    published at schedule time (see _patched_step_with_batch_queue), because
    the edge draft head starts its channel send immediately upon execution
    and a delayed publish leaves the cloud side of the channel rendezvous
    tens of seconds behind, which tears down the half-open connection.
    """
    ch = getattr(self, "_pp_pd_channel", None)
    if ch is None:
        return

    batch_queue = self.batch_queue
    if not batch_queue:
        return

    _, oldest_so, _ = batch_queue[-1]
    if oldest_so.batch_type is not BatchType.PREFILL_FIRST:
        return

    head_token = getattr(oldest_so, "head_token", None)
    if not head_token:
        return

    published = getattr(self, "_published_pre_out_tokens", None)
    if published is None:
        published = set()
        self._published_pre_out_tokens = published
    if head_token in published:
        return

    ch.publish(oldest_so)
    published.add(head_token)
    logger.info(
        "[PRE_OUT] Published %s (head_token=%s) when it became next to execute, "
        "queue_len=%d",
        oldest_so.batch_type.value,
        head_token, len(batch_queue),
    )


def _clear_published_pre_out_token(self, scheduler_output: SchedulerOutput) -> None:
    """Remove the head_token from published set after the batch completes,
    preventing unbounded growth of the set."""
    head_token = getattr(scheduler_output, "head_token", None)
    if not head_token:
        return
    published = getattr(self, "_published_pre_out_tokens", None)
    if published is not None:
        published.discard(head_token)


def _needs_sample_tokens(self, scheduler_output: SchedulerOutput) -> bool:
    """Return True if sample_tokens should follow execute_model for this
    batch.

    In edge-cloud PD-separation mode, only tail-segment batches (PL/DL)
    produce logits and need sampling. Head-segment batches (PF/DF) output
    intermediate hidden states and must skip sampling.
    """
    if getattr(self, "_pp_pd_channel", None) is None:
        return True
    bt = scheduler_output.batch_type
    return bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST)


def _stash_empty_worker_cleanup(self, scheduler_output: SchedulerOutput) -> None:
    """Keep worker-side cleanup from EMPTY batches for the next real batch."""
    finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
    free_encoder_mm_hashes = getattr(scheduler_output, "free_encoder_mm_hashes", None)
    if not finished_req_ids and not free_encoder_mm_hashes:
        return

    pending_finished = getattr(self, "_pd_pending_finished_req_ids", None)
    if pending_finished is None:
        pending_finished = set()
        self._pd_pending_finished_req_ids = pending_finished
    pending_finished.update(finished_req_ids or ())

    pending_mm_hashes = getattr(self, "_pd_pending_free_encoder_mm_hashes", None)
    if pending_mm_hashes is None:
        pending_mm_hashes = set()
        self._pd_pending_free_encoder_mm_hashes = pending_mm_hashes
    pending_mm_hashes.update(free_encoder_mm_hashes or ())


def _merge_pending_worker_cleanup(self, scheduler_output: SchedulerOutput) -> None:
    """Attach cleanup skipped with EMPTY batches to the next worker batch."""
    pending_finished = getattr(self, "_pd_pending_finished_req_ids", None)
    if pending_finished:
        scheduler_output.finished_req_ids = set(
            scheduler_output.finished_req_ids
        ).union(pending_finished)
        pending_finished.clear()

    pending_mm_hashes = getattr(self, "_pd_pending_free_encoder_mm_hashes", None)
    if pending_mm_hashes:
        scheduler_output.free_encoder_mm_hashes = list(
            dict.fromkeys([
                *scheduler_output.free_encoder_mm_hashes,
                *pending_mm_hashes,
            ])
        )
        pending_mm_hashes.clear()


def _finish_empty_batch(self, scheduler_output: SchedulerOutput):
    """Complete an EMPTY SchedulerOutput without broadcasting to workers."""
    self._stash_empty_worker_cleanup(scheduler_output)
    self._process_aborts_queue()
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT
        )
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    return engine_core_outputs, False


def _defer_empty_batch(self, scheduler_output: SchedulerOutput) -> None:
    """Defer an EMPTY batch when queued model work must complete first."""
    deferred = getattr(self, "_pd_deferred_empty_batches", None)
    if deferred is None:
        deferred = []
        self._pd_deferred_empty_batches = deferred
    deferred.append(scheduler_output)


def _pop_deferred_empty_batch(self) -> SchedulerOutput | None:
    deferred = getattr(self, "_pd_deferred_empty_batches", None)
    if not deferred:
        return None
    return deferred.pop(0)


def _enqueue_pending_edge_cloud_draft_if_ready(self) -> None:
    """Move completed/new edge-cloud draft work across the executor boundary."""
    if not getattr(self, "use_spec_decode", False):
        return
    ready_queue = getattr(self.scheduler, "drafts_first_ready", None)
    if ready_queue is None:
        return

    take_completed = getattr(
        self.model_executor, "take_completed_edge_cloud_draft_result", None
    )
    if take_completed is not None:
        completed = take_completed()
        if completed is not None:
            draft_token_ids, _parent_scheduler_output = completed
            self.scheduler.update_draft_token_ids(draft_token_ids)

    take_pending = getattr(
        self.model_executor,
        "take_pending_edge_cloud_draft_scheduler_output",
        None,
    )
    if take_pending is None:
        return
    scheduler_output = take_pending()
    if scheduler_output is None:
        return
    if scheduler_output.batch_type != BatchType.DRAFT_FIRST:
        raise RuntimeError(
            "Pending edge-cloud draft must be DRAFT_FIRST, got "
            f"{scheduler_output.batch_type}"
        )
    queue_size_before = len(ready_queue)
    ready_queue.append(scheduler_output)
    logger.info(
        "[MTP-DEBUG] EngineCore enqueued DRAFT_FIRST: task_id=%s, "
        "parent_req_id=%s, draft_step_idx=%s, queue_size=%d->%d",
        scheduler_output.draft_task_id,
        scheduler_output.parent_req_id,
        scheduler_output.draft_step_idx,
        queue_size_before,
        len(ready_queue),
    )


def _clear_pending_edge_cloud_draft_for_finished_requests(self) -> None:
    if not getattr(self, "use_spec_decode", False):
        return
    finished_req_ids = set(
        getattr(self.scheduler, "finished_req_ids", set()) or ()
    )
    if not finished_req_ids:
        return
    clear_pending = getattr(
        self.model_executor,
        "clear_pending_edge_cloud_draft_for_req_ids",
        None,
    )
    if clear_pending is not None:
        clear_pending(finished_req_ids)


def _uses_scheduled_edge_cloud_draft(self) -> bool:
    speculative_config = self.vllm_config.speculative_config
    if (
        getattr(self, "_pp_pd_channel", None) is None
        or speculative_config is None
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


def _has_unresolved_edge_cloud_draft_parent(self) -> bool:
    """Keep async scheduling behind a tail that still has to create draft work."""
    if not self._uses_scheduled_edge_cloud_draft():
        return False
    batch_queue = getattr(self, "batch_queue", None)
    if not batch_queue:
        return False
    for _future, scheduler_output, _exec_future in batch_queue:
        if scheduler_output.batch_type == BatchType.DECODE_LAST:
            return True
        if (
            scheduler_output.batch_type == BatchType.PREFILL_LAST
            and getattr(scheduler_output, "is_last_prefill_chunk", True)
        ):
            return True
    return False


# =======================================================================#
# EngineCore.step — full replacement, mirrors upstream + dest inserts.    #
# =======================================================================#
def _patched_step(self):
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
    # Check for any requests remaining in the scheduler - unfinished,
    # or finished and not yet removed from the batch.
    if not self.scheduler.has_requests():
        return {}, False

    # [ascend insert] Drain POST_OUT (cloud → edge) into the
    # PDSeparatedScheduler's tail-segment ready queues before scheduling.
    self._drain_pd_channel_inbox()

    scheduler_output = self.scheduler.schedule()
    self._ensure_pd_head_token(scheduler_output)

    # [ascend insert] Forward head-segment batches on the PRE_OUT
    # (edge → cloud) channel.
    self._maybe_publish_pre_out(scheduler_output)

    if scheduler_output.batch_type == BatchType.EMPTY:
        return self._finish_empty_batch(scheduler_output)

    self._merge_pending_worker_cleanup(scheduler_output)

    future = self.model_executor.execute_model(
        scheduler_output, non_block=True
    )
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    self._process_aborts_queue()
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    self._enqueue_pending_edge_cloud_draft_if_ready()

    return (
        engine_core_outputs,
        scheduler_output.total_num_scheduled_tokens > 0,
    )


# =======================================================================#
# EngineCore.step_with_batch_queue — full replacement.                    #
# =======================================================================#
def _patched_step_with_batch_queue(self):
    """Schedule and execute batches with the batch queue."""
    batch_queue = self.batch_queue
    assert batch_queue is not None

    # Try to schedule a new batch if the batch queue is not full.
    assert len(batch_queue) < self.batch_queue_size

    model_executed = False
    deferred_scheduler_output = None
    if (
        self.scheduler.has_requests()
        and not self._has_unresolved_edge_cloud_draft_parent()
    ):
        # [ascend insert] Pull cloud-returned tail-segment batches into
        # the scheduler ready queues before picking the next batch.
        self._drain_pd_channel_inbox()

        scheduler_output = self.scheduler.schedule()

        # [ascend insert] Assign head-token for edge-cloud head-segment
        # batches so the tail-segment can be matched to the suspended
        # state.
        self._ensure_pd_head_token(scheduler_output)

        # [ascend insert] DECODE_FIRST and DRAFT_FIRST are published
        # immediately at schedule time to keep the decode/draft pipeline
        # full.  The edge worker starts the hidden-channel send as soon as
        # the batch is dequeued, so the cloud must learn about the batch
        # right away: on first use of the channel the two sides rendezvous
        # (gloo metadata exchange + HCCL comm init), and if the cloud side
        # only shows up tens of seconds later the half-open connection is
        # torn down before the peer joins, killing both workers.
        # PREFILL_FIRST is still delayed via _publish_pre_out_when_ready
        # until it becomes next to execute.
        if scheduler_output.batch_type in (
            BatchType.DECODE_FIRST,
            BatchType.DRAFT_FIRST,
        ):
            self._maybe_publish_pre_out(scheduler_output)

        if scheduler_output.batch_type == BatchType.EMPTY:
            if batch_queue:
                self._defer_empty_batch(scheduler_output)
                scheduler_output = None
            else:
                return self._finish_empty_batch(scheduler_output)

        if scheduler_output is not None:
            self._merge_pending_worker_cleanup(scheduler_output)

            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = (
                    scheduler_output.total_num_scheduled_tokens > 0
                )

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            elif not self._needs_sample_tokens(scheduler_output):
                # [ascend insert] Edge-cloud head segment (PF/DF): sampling is
                # done in the tail segment (PL/DL) after the cloud returns
                # intermediate tensors. Skip sample_tokens for the head
                # segment.
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                batch_queue.appendleft((future, scheduler_output, exec_future))
                # [ascend insert] Log batch_queue contents for debugging.
                queue_types = [
                    so.batch_type.value
                    for _, so, _ in batch_queue
                ]
                vllm_logger.info(
                    "[BATCH_QUEUE] Enqueued %s, queue_len=%d, types=%s",
                    scheduler_output.batch_type.value,
                    len(batch_queue),
                    queue_types,
                )
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    return None, True

    elif not batch_queue:
        return None, False

    # Block until the next result is available.
    # [ascend insert] Publish PRE_OUT for the head segment that is about
    # to execute (rightmost in deque).  FIFO guarantees every PREFILL_FIRST
    # eventually becomes batch_queue[-1] before pop().
    self._publish_pre_out_when_ready()
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    # [ascend insert] Clean up PRE_OUT tracking for completed batch.
    self._clear_published_pre_out_token(scheduler_output)
    bt = scheduler_output.batch_type
    vllm_logger.info(
        "[PD] EngineCore blocking on future.result(): "
        "batch_type=%s total_tokens=%d",
        bt.value if bt else "N/A",
        scheduler_output.total_num_scheduled_tokens)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    self._process_aborts_queue()
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    self._clear_pending_edge_cloud_draft_for_finished_requests()
    self._enqueue_pending_edge_cloud_draft_if_ready()

    if deferred_empty_batch := self._pop_deferred_empty_batch():
        empty_outputs, _ = self._finish_empty_batch(deferred_empty_batch)
        if empty_outputs:
            if engine_core_outputs:
                for client_index, output in empty_outputs.items():
                    existing = engine_core_outputs.get(client_index)
                    if existing is None:
                        engine_core_outputs[client_index] = output
                    elif output.finished_requests:
                        existing_finished = existing.finished_requests or set()
                        existing.finished_requests = existing_finished.union(
                            output.finished_requests
                        )
            else:
                engine_core_outputs = empty_outputs

    if deferred_scheduler_output:
        if (
            self.use_spec_decode
            and not self._uses_scheduled_edge_cloud_draft()
        ):
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_scheduler_output
        )
        future = self.model_executor.sample_tokens(
            grammar_output, non_block=True
        )
        batch_queue.appendleft(
            (future, deferred_scheduler_output, exec_future)
        )

    return engine_core_outputs, model_executed


# =======================================================================#
# EngineCore.shutdown — close PD/ZMQ resources before stopping the rest.  #
# =======================================================================#
@functools.wraps(_ORIG_ENGINE_CORE_SHUTDOWN)
def _patched_engine_core_shutdown(self):
    ch = getattr(self, "_pp_pd_channel", None)
    if ch is not None:
        try:
            ch.shutdown()
        except Exception:
            logger.exception(
                "Error while shutting down PD-separation ZMQ channel"
            )
        self._pp_pd_channel = None

    _ORIG_ENGINE_CORE_SHUTDOWN(self)


# =======================================================================#
# EngineCoreProc.run_engine_core — keep child-process patch import.       #
# =======================================================================#
def _patched_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0,
                             **kwargs):
    """Delegate to upstream while keeping this patch module as the process
    target so spawn-based child processes import and install the patches.
    """
    return _ORIG_RUN_ENGINE_CORE(
        *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs
    )


_patched_run_engine_core.__module__ = __name__
_patched_run_engine_core.__qualname__ = "_patched_run_engine_core"


# =======================================================================#
# EngineCoreProc._process_input_queue — full replacement to add the       #
# edge-cloud idle-block branch.                                            #
# =======================================================================#
# Imports kept inside the function-scope dict to mirror the upstream
# module-level imports (`queue`, `DEBUG`) without polluting our patch
# module's top-level namespace.
import queue as _queue_mod  # noqa: E402
from logging import DEBUG as _DEBUG  # noqa: E402


def _patched_process_input_queue(self):
    """Exits when an engine step needs to be performed."""
    waited = False
    while not self.has_work() and self.is_running():
        # Notify callbacks waiting for engine to become idle.
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
            with self.aborts_queue.mutex:
                self.aborts_queue.queue.clear()
            if logger.isEnabledFor(_DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block

        # [ascend insert] In edge-cloud mode the edge can be completely
        # idle for long periods while waiting for the next client
        # request. If no local work exists, force a blocking wait even
        # if an earlier mode (e.g. elastic scaling) left the input queue
        # in non-blocking polling mode; otherwise the outer busy loop
        # spins forever.
        if (
            not block
            and not self.scheduler.has_unfinished_requests()
            and not self.engines_running
            and not bool(self.batch_queue)
            and getattr(self, "eep_scaling_state", None) is None
        ):
            block = True

        try:
            if block and self.input_queue.empty():
                logger.info("input_queue is empty, EngineCore waiting for work.")
            req = self.input_queue.get(block=block)
            self._handle_client_request(*req)
        except _queue_mod.Empty:
            break
        if not block:
            break

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)


# =======================================================================#
# Install                                                                  #
# =======================================================================#
def install() -> None:
    if getattr(EngineCore, _INSTALLED_FLAG, False):
        return

    EngineCore.__init__ = _patched_engine_core_init
    EngineCore._drain_pd_channel_inbox = _drain_pd_channel_inbox
    EngineCore._maybe_publish_pre_out = _maybe_publish_pre_out
    EngineCore._ensure_pd_head_token = _ensure_pd_head_token
    EngineCore._publish_pre_out_when_ready = _publish_pre_out_when_ready
    EngineCore._clear_published_pre_out_token = _clear_published_pre_out_token
    EngineCore._needs_sample_tokens = _needs_sample_tokens
    EngineCore._stash_empty_worker_cleanup = _stash_empty_worker_cleanup
    EngineCore._merge_pending_worker_cleanup = _merge_pending_worker_cleanup
    EngineCore._finish_empty_batch = _finish_empty_batch
    EngineCore._defer_empty_batch = _defer_empty_batch
    EngineCore._pop_deferred_empty_batch = _pop_deferred_empty_batch
    EngineCore._enqueue_pending_edge_cloud_draft_if_ready = (
        _enqueue_pending_edge_cloud_draft_if_ready
    )
    EngineCore._clear_pending_edge_cloud_draft_for_finished_requests = (
        _clear_pending_edge_cloud_draft_for_finished_requests
    )
    EngineCore._uses_scheduled_edge_cloud_draft = (
        _uses_scheduled_edge_cloud_draft
    )
    EngineCore._has_unresolved_edge_cloud_draft_parent = (
        _has_unresolved_edge_cloud_draft_parent
    )
    EngineCore.step = _patched_step
    EngineCore.step_with_batch_queue = _patched_step_with_batch_queue
    EngineCore.shutdown = _patched_engine_core_shutdown

    EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
    EngineCoreProc._process_input_queue = _patched_process_input_queue

    setattr(EngineCore, _INSTALLED_FLAG, True)
    logger.info(
        "vllm-ascend EngineCore PD/edge-cloud patch installed."
    )


install()
