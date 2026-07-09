# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from ..utils import weak_ref_tensors


# Set VLLM_ASCEND_ACLGRAPH_DEBUG=1 to enable detailed per-replay dumps.
# Useful for comparing inputs/graph-params between edge-cloud and non-edge-cloud.
_ACLGRAPH_DEBUG_ENABLED = os.environ.get("VLLM_ASCEND_ACLGRAPH_DEBUG", "") in (
    "1", "true", "True", "yes")


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self._runnable_str}"
            )
        raise AttributeError(f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for more details.")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            global _draft_graph_params
            global _draft_graph_prefill_params
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)
            weak_ref_workspaces(_draft_graph_prefill_params)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        # If we do not in main model and in full-graph mode when using merge-eagle-graph,
        # we do not need to synchronize.
        # When enable_enpu is on, model_runner orders update vs replay; skip here.
        # When FULL + EAGLE draft (merge path), replay does not need this barrier.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()
        _dump_aclgraph_state(
            "BEFORE_REPLAY", self, args, kwargs, entry=entry)
        entry.aclgraph.replay()
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None,
    layer_indices: list[int] | None = None,
    graph_params: GraphParams | None = None,
    draft_graph_params: GraphParams | None = None,
    unfiltered_attn_metadata: dict | None = None,
):
    """更新 attention 图参数，供下一次图回放使用。

    标准流程使用全局 GraphParams；边云流程为每个 segment 传入独立
    GraphParams，避免 segment_a / segment_e 的 task handle 相互错配。

    Args:
        unfiltered_attn_metadata: 真正未过滤的原始 attn_metadata（含 GDN key）。
            当上游代码（如 _update_full_graph_params_if_needed）为了 FIA update
            提前过滤掉了 skip_graph_params_update=True 的 key 时，需要传入此参数
            以保证 GDN 的 update_conv1d_graph_params 仍能按 layer_prefix 查找。
    """
    _dump_aclgraph_state(
        "BEFORE_UPDATE_GRAPH_PARAMS", None, (), {
            "attn_backend": attn_backend,
            "update_stream": update_stream,
            "forward_context": forward_context,
            "num_tokens": num_tokens,
            "vllm_config": vllm_config,
            "speculative_config": speculative_config,
            "num_dcp_pcp_tokens": num_dcp_pcp_tokens,
            "draft_attn_metadatas": draft_attn_metadatas,
            "layer_indices": layer_indices,
            "graph_params": graph_params,
            "draft_graph_params": draft_graph_params,
            "unfiltered_attn_metadata": unfiltered_attn_metadata,
        }, graph_params=graph_params, draft_graph_params=draft_graph_params)
    # Lazy import to avoid circular dependency:
    # acl_graph_edge_cloud.py imports ACLGraphWrapper / GraphParams from this module,
    # so we import graph_params_scope inside the function body.
    from vllm_ascend.compilation.acl_graph_edge_cloud import graph_params_scope

    with graph_params_scope(graph_params, draft_graph_params), set_current_vllm_config(vllm_config):
        impl_cls = attn_backend.get_impl_cls()

        # Use the caller-supplied unfiltered metadata if available;
        # otherwise fall back to forward_context.attn_metadata (non-edge-cloud path).
        unfiltered_metadata = unfiltered_attn_metadata or forward_context.attn_metadata
        filtered_metadata = None

        if layer_indices is not None:
            # 强制要求 layer_indices 为升序自然层号，与图捕获时 islice(self.layers)
            # 的遍历顺序严格一致，防止 zip(attn_keys, attn_params) 错位
            assert layer_indices == sorted(layer_indices), (
                "layer_indices must be in ascending natural order to align with "
                "graph_params.attn_params append order."
            )
            filtered_metadata = _filter_attn_metadata_for_layers(
                forward_context.attn_metadata, layer_indices
            )
            forward_context.attn_metadata = filtered_metadata

        try:
            impl_cls.update_graph_params(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )
            # For GDN Attention: AscendC operate(conv1d update) update graph params
            # _filter_attn_metadata_for_layers drops GDN keys (they do not contain
            # ".layers.{idx}.self_attn" and are absent from attn_params), but
            # update_conv1d_graph_params still needs the full metadata dict to look
            # up layer_prefix.  Temporarily restore the unfiltered metadata.
            from vllm_ascend.ops.gdn import update_conv1d_graph_params
            if unfiltered_metadata is not None and unfiltered_metadata is not forward_context.attn_metadata:
                old_metadata = forward_context.attn_metadata
                forward_context.attn_metadata = unfiltered_metadata
                try:
                    update_conv1d_graph_params(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        _EXTRA_CTX.is_draft_model,
                        draft_attn_metadatas,
                    )
                finally:
                    forward_context.attn_metadata = old_metadata
            else:
                update_conv1d_graph_params(
                    update_stream,
                    forward_context,
                    num_tokens,
                    vllm_config,
                    _EXTRA_CTX.is_draft_model,
                    draft_attn_metadatas,
                )
        finally:
            if filtered_metadata is not None:
                forward_context.attn_metadata = unfiltered_metadata

def _filter_attn_metadata_for_layers(
    attn_metadata: dict,
    layer_indices: list[int],
) -> dict:
    """返回仅包含指定层索引对应条目的 dict，key 顺序与 layer_indices 一致。

    attn_metadata 的 key 格式通常为 ``"model.layers.3.self_attn"``。
    通过匹配 ``.layers.{idx}.`` 子串来定位目标层。

    重要：边云流程中图捕获按自然层顺序遍历（islice(self.layers)），
    graph_params.attn_params 也按该顺序追加。因此过滤后必须保持
    layer_indices 的自然顺序，使 update_graph_params 的 zip 配对
    与图捕获顺序严格对齐，避免错位。
    """
    result: dict = {}
    skipped_no_key_layers: list[int] = []
    for idx in layer_indices:
        needle = f".layers.{idx}."
        matched_keys = [k for k in attn_metadata if needle in k]
        if not matched_keys:
            skipped_no_key_layers.append(idx)
            continue
        if len(matched_keys) == 1:
            # 保持原有单 key 路径不变，兼容 Qwen / MLA / FIA 等模型中
            # 可能带不同前缀的 ``*.layers.{idx}.self_attn`` key。
            result[matched_keys[0]] = attn_metadata[matched_keys[0]]
            continue

        base_keys = [k for k in matched_keys if k.endswith(f".layers.{idx}.self_attn")]
        if len(base_keys) == 1:
            result[base_keys[0]] = attn_metadata[base_keys[0]]
            continue

        # DeepSeekV4 DSA 会为同一层注册多个 KV-cache metadata key，
        # 如 ``self_attn.attn`` / ``self_attn.swa_cache`` / compressor / indexer。
        # 这些 key 供 DSA custom op 在 forward 时通过 prefix 过滤使用，
        # 不参与 full graph attention task update，也不会向
        # graph_params.attn_params 追加条目。只有确认全部都是 DSA 子 key 时
        # 才跳过，避免破坏原有多 key 防错逻辑。
        if _is_dsa_kv_metadata_keys(matched_keys, idx):
            skipped_no_key_layers.append(idx)
            continue

        # 边云流程要求每层恰好一个 attention graph-update metadata key，
        # 以确保 graph_params.attn_params 的追加顺序与过滤后顺序 1:1 对齐。
        # 未识别的多 key 仍然 fail-fast，避免静默错配导致挂死。
        raise ValueError(
            f"Layer {idx} has multiple attention metadata keys: {matched_keys}. "
            f"This breaks the 1:1 alignment between attn_metadata and attn_params."
        )

    return result


def _is_dsa_kv_metadata_keys(keys: list[str], layer_idx: int) -> bool:
    dsa_suffixes = {
        "attn",
        "swa_cache",
        "compressor.state_cache",
        "indexer.k_cache",
        "indexer.compressor.state_cache",
    }
    prefix = f".layers.{layer_idx}.self_attn."
    suffixes: set[str] = set()
    for key in keys:
        if prefix not in key:
            return False
        suffix = key.split(prefix, 1)[1]
        if suffix not in dsa_suffixes:
            return False
        suffixes.add(suffix)
    return bool(suffixes)


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    conv1d_params: dict[int, list[tuple]]  # for causal conv1d params
    conv1d_handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]  # for causal conv1d params handles
    conv1d_events: dict[int, list[torch.npu.ExternalEvent]]  # for causal conv1d params events


def _summarize_tensor(t: torch.Tensor, name: str = "") -> str:
    """Return a one-line summary of a tensor for aclgraph debugging."""
    if not isinstance(t, torch.Tensor):
        return f"{name}: not a tensor ({type(t).__name__})"
    info = [
        f"shape={list(t.shape)}",
        f"dtype={t.dtype}",
        f"device={t.device}",
        f"ptr={t.data_ptr()}",
        f"stride={list(t.stride())}",
        f"layout={t.layout}",
    ]
    try:
        if t.numel() > 0 and t.device.type == "npu":
            info.append(f"min={t.min().item():.6f}")
            info.append(f"max={t.max().item():.6f}")
            info.append(f"mean={t.float().mean().item():.6f}")
            info.append(f"has_nan={torch.isnan(t).any().item()}")
            info.append(f"has_inf={torch.isinf(t).any().item()}")
    except Exception as e:
        info.append(f"stats_error={e}")
    return f"{name}: " + " ".join(info)


def _summarize_obj(obj: Any, name: str = "", depth: int = 0,
                   max_depth: int = 3) -> list[str]:
    """Recursively summarize an object, expanding tensors/dataclasses/dicts."""
    if depth > max_depth:
        return [f"{'  ' * depth}{name}: ... (max depth)"]
    lines: list[str] = []
    if isinstance(obj, torch.Tensor):
        lines.append(f"{'  ' * depth}{_summarize_tensor(obj, name)}")
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{'  ' * depth}{name}: {type(obj).__name__}[len={len(obj)}]")
        for i, item in enumerate(obj):
            lines.extend(_summarize_obj(item, f"[{i}]", depth + 1, max_depth))
    elif isinstance(obj, dict):
        lines.append(f"{'  ' * depth}{name}: dict[len={len(obj)}]")
        for k, v in obj.items():
            lines.extend(_summarize_obj(v, str(k), depth + 1, max_depth))
    elif dataclasses.is_dataclass(obj):
        lines.append(f"{'  ' * depth}{name}: {type(obj).__name__}")
        try:
            for field in dataclasses.fields(obj):
                lines.extend(
                    _summarize_obj(getattr(obj, field.name), field.name,
                                   depth + 1, max_depth))
        except Exception as e:
            lines.append(f"{'  ' * depth}  <dataclass error {e}>")
    else:
        s = repr(obj)
        if len(s) > 200:
            s = s[:200] + "..."
        lines.append(f"{'  ' * depth}{name}: {s}")
    return lines


def _dump_aclgraph_state(
    phase: str,
    wrapper: "ACLGraphWrapper | None",
    args: tuple,
    kwargs: dict[str, Any],
    entry: "ACLGraphEntry | None" = None,
    graph_params: GraphParams | None = None,
    draft_graph_params: GraphParams | None = None,
) -> None:
    """Dump aclgraph inputs/params right before replay/update.

    Writes to stderr so it is independent of vLLM log configuration and is
    flushed immediately, which is important when debugging hangs.
    """
    if not _ACLGRAPH_DEBUG_ENABLED:
        return
    try:
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"ACLGRAPH DEBUG [{phase}]")
        if wrapper is not None:
            lines.append(f"  wrapper.runtime_mode={wrapper.runtime_mode}")
            lines.append(f"  wrapper.use_eagle={wrapper.use_eagle}")
            lines.append(f"  wrapper.enable_enpu={wrapper.enable_enpu}")
        if entry is not None:
            lines.append(f"  entry.batch_descriptor={entry.batch_descriptor}")
        forward_context = get_forward_context()
        if forward_context is not None:
            lines.append("  forward_context:")
            lines.append(
                "    cudagraph_runtime_mode="
                f"{getattr(forward_context, 'cudagraph_runtime_mode', None)}")
            lines.append(
                "    batch_descriptor="
                f"{getattr(forward_context, 'batch_descriptor', None)}")
            lines.append(
                f"    num_tokens={getattr(forward_context, 'num_tokens', None)}")
            lines.append(
                f"    capturing={getattr(forward_context, 'capturing', None)}")
            lines.append(
                "    is_draft_model="
                f"{getattr(forward_context, 'is_draft_model', None)}")
            lines.append(
                "    layer_idx="
                f"{getattr(forward_context, 'layer_idx', None)}")
            attn_metadata = getattr(forward_context, "attn_metadata", None)
            if attn_metadata is not None:
                lines.append(
                    f"    attn_metadata keys={list(attn_metadata.keys())}")
                for k, v in attn_metadata.items():
                    lines.extend(
                        _summarize_obj(v, f"    attn_metadata[{k}]", 0, 2))
        lines.append("  args:")
        for i, a in enumerate(args):
            lines.extend(_summarize_obj(a, f"    args[{i}]", 0, 2))
        lines.append("  kwargs:")
        for k, v in kwargs.items():
            lines.extend(_summarize_obj(v, f"    kwargs[{k}]", 0, 2))
        for label, params in (("graph_params", graph_params or _graph_params),
                              ("draft_graph_params",
                               draft_graph_params or _draft_graph_params)):
            if params is None:
                lines.append(f"  {label}=None")
                continue
            lines.append(f"  {label} id={id(params)}")
            for size in sorted(params.events.keys()):
                evs = params.events.get(size, [])
                ws = params.workspaces.get(size)
                hs = params.handles.get(size, [])
                ap = params.attn_params.get(size, [])
                cp = params.conv1d_params.get(size, [])
                ch = params.conv1d_handles.get(size, [])
                ce = params.conv1d_events.get(size, [])
                if ws is None:
                    ws_str = "None"
                else:
                    ws_str = (f"shape={list(ws.shape)} ptr={ws.data_ptr()} "
                              f"dtype={ws.dtype}")
                lines.append(
                    f"    size={size}: events={len(evs)}, handles={len(hs)}, "
                    f"attn_params={len(ap)}, conv1d_params={len(cp)}, "
                    f"conv1d_handles={len(ch)}, conv1d_events={len(ce)}, "
                    f"workspace={ws_str}")
        lines.append("=" * 80)
        print("\n".join(lines), file=sys.stderr, flush=True)
    except Exception as e:
        print(f"ACLGRAPH DEBUG dump failed: {e}", file=sys.stderr, flush=True)


_graph_params: GraphParams | None = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    graph_params = get_graph_params()
    if graph_params is not None:
        graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params


_draft_graph_prefill_params: GraphParams | None = None


def set_draft_graph_prefill_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph preill parameters have already been set!")
    _draft_graph_prefill_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_prefill_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        _draft_graph_prefill_params.workspaces[num_tokens] = workspace


def get_draft_graph_prefill_params():
    return _draft_graph_prefill_params
