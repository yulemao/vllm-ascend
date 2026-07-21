from __future__ import annotations

import os
import signal
import threading
import traceback
import weakref
from collections import deque
from collections.abc import Callable
from functools import partial
from multiprocessing.synchronize import Lock as LockType
from threading import Thread

import cloudpickle
import vllm.v1.executor.multiproc_executor
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.tracing import maybe_init_worker_tracer
from vllm.utils import numa_utils
from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.utils.system_utils import get_mp_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)
from vllm.v1.outputs import DraftTokenIds
from vllm_ascend.passive_engine_core_state import (
    is_ascend_non_leader_passive_engine_core,
)

logger = init_logger(__name__)


class AscendMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tensor_parallel_size, pp_parallel_size, pcp_parallel_size = self._get_parallel_sizes()
        if not self.parallel_config.enable_edge_cloud:
            assert self.world_size == tensor_parallel_size * pp_parallel_size * pcp_parallel_size, (
                f"world_size ({self.world_size}) must be equal to the "
                f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
                f"_parallel_size ({pp_parallel_size}) x prefill_context"
                f"_parallel_size ({pcp_parallel_size}). "
            )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=self.parallel_config.master_addr,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        elif is_ascend_non_leader_passive_engine_core(self.vllm_config):
            # For non-leader PP rank running with a passive EngineCore,
            # create a local rpc_broadcast_mq to broadcast SchedulerOutput
            # to local workers. Workers will use this MQ instead of
            # inner_dp_world_group to receive scheduler_output.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.local_world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=get_loopback_ip(),
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            if self.parallel_config.enable_edge_cloud:
                global_start_rank = (
                    0
                    if self.parallel_config.is_edge_node
                    else self.parallel_config.edge_npu_count
                )
            else:
                global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp

            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = [] if context.get_start_method() == "fork" else None

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = AscendWorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = AscendWorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0 and (
                not self.parallel_config.enable_edge_cloud
                or self.parallel_config.is_edge_node
            ):
                for rank in range(self.world_size):
                    local_idx = rank - global_start_rank
                    if 0 <= local_idx < self.local_world_size:
                        local_message_queue = self.workers[local_idx].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[rank]
                        if remote_message_queue is None:
                            peer_mqs = self.workers[0].peer_worker_response_mqs
                            logger.error(
                                "remote response mq missing: rank=%s, "
                                "world_size=%s, local_world_size=%s, "
                                "global_start_rank=%s, node_rank_within_dp=%s, "
                                "enable_edge_cloud=%s, is_edge_node=%s, "
                                "is_cloud_node=%s, peer_mq_type=%s, "
                                "peer_mq_keys=%s, peer_mq_len=%s",
                                rank,
                                self.world_size,
                                self.local_world_size,
                                global_start_rank,
                                self.parallel_config.node_rank_within_dp,
                                self.parallel_config.enable_edge_cloud,
                                self.parallel_config.is_edge_node,
                                not self.parallel_config.is_edge_node,
                                type(peer_mqs),
                                (list(peer_mqs.keys()) if hasattr(peer_mqs, "keys")
                                 else None),
                                (len(peer_mqs) if hasattr(peer_mqs, "__len__")
                                 else None),
                            )
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)
            elif is_ascend_non_leader_passive_engine_core(self.vllm_config):
                # For non-leader PP rank with passive EngineCore,
                # collect local worker response mqs only.
                for rank in range(self.local_world_size):
                    local_message_queue = self.workers[rank].worker_response_mq
                    assert local_message_queue is not None
                    self.response_mqs.append(local_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()
            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()
            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        if not self.parallel_config.enable_edge_cloud:
            assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
                f"global world_size ({self.parallel_config.world_size}) must be "
                f"divisible by nnodes_within_dp "
                f"({self.parallel_config.nnodes_within_dp}). "
            )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        if self.parallel_config.enable_edge_cloud:
            return rank == (
                0
                if self.parallel_config.is_edge_node
                else self.parallel_config.edge_npu_count
            )
        return rank % self.parallel_config.tensor_parallel_size == 0

    def _get_output_rank(self) -> int:
        if self.parallel_config.enable_edge_cloud:
            return 0
        return super()._get_output_rank()

    def _edge_local_only(self) -> bool:
        # Mirror execute_model: on the edge node these control RPCs must stay
        # off the cross-node wire. The cloud receives work solely via ZMQ
        # (PassiveEngineCore -> cloud local rpc_broadcast_mq); remote readers
        # of rpc_broadcast_mq must not see them.
        pc = self.parallel_config
        return bool(
            getattr(pc, "enable_edge_cloud", False)
            and getattr(pc, "is_edge_node", False)
        )

    def take_pending_mtp_draft_scheduler_output(self) -> SchedulerOutput | None:
        # Edge-cloud Qwen-MTP: fetch the next pending draft task stashed by
        # the driver worker's model runner so EngineCore can hand it to the
        # scheduler's mtp_drafts_first_ready queue.
        return self.collective_rpc(
            "take_pending_mtp_draft_scheduler_output",
            unique_reply_rank=self.output_rank,
            local_only=self._edge_local_only(),
        )

    def take_completed_mtp_draft_result(
        self,
    ) -> tuple[DraftTokenIds, SchedulerOutput] | None:
        # Edge-cloud Qwen-MTP: fetch a finished draft chain (all draft steps)
        # plus its parent SchedulerOutput so EngineCore can write the draft
        # tokens back into request.spec_token_ids via update_draft_token_ids.
        return self.collective_rpc(
            "take_completed_mtp_draft_result",
            unique_reply_rank=self.output_rank,
            local_only=self._edge_local_only(),
        )

    def clear_pending_mtp_draft_for_req_ids(
        self, req_ids: set[str] | list[str]
    ) -> None:
        # Edge-cloud Qwen-MTP: drop pending draft contexts of finished
        # requests on every local worker.
        self.collective_rpc(
            "clear_pending_mtp_draft_for_req_ids",
            args=(req_ids,),
            local_only=self._edge_local_only(),
        )


class AscendWorkerProc(WorkerProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.local_rank = local_rank
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            input_shm_handle=input_shm_handle,
            shared_worker_lock=shared_worker_lock,
            is_driver_worker=is_driver_worker,
        )

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Single-node: use local MQ
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
            self.local_rpc_broadcast_mq = None
            self.local_worker_response_mq = None
        elif is_ascend_non_leader_passive_engine_core(vllm_config):
            # Non-leader PP rank with passive EngineCore:
            # Dual MQ — local MQ for passive enginecore handshake +
            # cross-node MQ for actual communication with pp rank0.
            from vllm.distributed.parallel_state import get_inner_dp_world_group
            # Local MQs (for passive enginecore handshake only)
            self.local_rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.local_rank
            )
            self.local_worker_response_mq = MessageQueue(1, 1)
            self.local_peer_response_handles: list = []
            # Cross-node MQs (for actual work with pp rank0)
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=None,
                blocking=False,
            )
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )
        else:
            # Delegate to parent class for the inner_dp_world_group path
            super()._init_message_queues(input_shm_handle, vllm_config)

    def shutdown(self):
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        if getattr(self, "local_rpc_broadcast_mq", None) is not None:
            self.local_rpc_broadcast_mq.shutdown()
            self.local_rpc_broadcast_mq = None
        if getattr(self, "local_worker_response_mq", None) is not None:
            self.local_worker_response_mq.shutdown()
            self.local_worker_response_mq = None
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        queues = [self.rpc_broadcast_mq, self.worker_response_mq]
        if getattr(self, "local_rpc_broadcast_mq", None) is not None:
            queues.append(self.local_rpc_broadcast_mq)
        if getattr(self, "local_worker_response_mq", None) is not None:
            queues.append(self.local_worker_response_mq)
        Thread(
            target=death_pipe_monitor,
            args=(queues,),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    @staticmethod
    def worker_main(*args, **kwargs):
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            worker = AscendWorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None
            if kwargs["vllm_config"].parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            if (
                is_ascend_non_leader_passive_engine_core(kwargs["vllm_config"])
                and worker.local_worker_response_mq is not None
            ):
                ready_writer.send(
                    {
                        "status": WorkerProc.READY_STR,
                        "handle": worker.local_worker_response_mq.export_handle(),
                        "peer_response_handles": worker.local_peer_response_handles,
                    }
                )
            else:
                ready_writer.send(
                    {
                        "status": WorkerProc.READY_STR,
                        "handle": worker.worker_response_mq.export_handle(),
                        "peer_response_handles": worker.peer_response_handles,
                    }
                )

            if getattr(worker, "local_rpc_broadcast_mq", None) is not None:
                worker.local_rpc_broadcast_mq.wait_until_ready()
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            if getattr(worker, "local_worker_response_mq", None) is not None:
                worker.local_worker_response_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")
            shutdown_requested.set()

        except SystemExit as e:
            logger.warning("WorkerProc was terminated")
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()

    def worker_busy_loop(self):
        assert self.rpc_broadcast_mq is not None
        run_rpc_broadcast_mq = True
        run_local_rpc_broadcast_mq = False
        while True:
            if getattr(self, "local_rpc_broadcast_mq", None) is not None and run_local_rpc_broadcast_mq:
                try:
                    method, args, kwargs, output_rank = (
                        self.local_rpc_broadcast_mq.dequeue(timeout=0)
                    )
                    if isinstance(method, bytes) and method == b"pp_scheduler_output":
                        scheduler_output = args[0]
                        slice_info = args[1] if len(args) > 1 else None
                        try:
                            func = getattr(self.worker, "execute_model")
                            output = func(
                                scheduler_output,
                                layer_slice_info=slice_info,
                            )
                        except Exception as e:
                            if hasattr(e, "add_note"):
                                e.add_note(traceback.format_exc())
                            logger.exception("PP worker execute_model failed.")
                            if output_rank is None or self.rank == output_rank:
                                run_rpc_broadcast_mq = True
                                run_local_rpc_broadcast_mq = False
                                self.handle_output(e)
                            continue
                        if slice_info is not None and not slice_info.is_last_slice:
                            continue
                        if output_rank is None or self.rank == output_rank:
                            run_rpc_broadcast_mq = True
                            run_local_rpc_broadcast_mq = False
                            self.handle_output(output)
                        continue
                except Exception:
                    pass

            if not run_rpc_broadcast_mq:
                continue

            try:
                method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                    timeout=0.1
                )
            except TimeoutError:
                continue

            if (
                getattr(self, "local_rpc_broadcast_mq", None) is not None
                and isinstance(method, str)
                and method == "execute_model"
            ):
                run_rpc_broadcast_mq = False
                run_local_rpc_broadcast_mq = True
                continue

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool = False,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=AscendWorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,
        )

        proc.start()
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)


vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
