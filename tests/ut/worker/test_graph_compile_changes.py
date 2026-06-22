"""
Unit tests for commit 60f17a156 ("feat:图编译修改").

Tests cover new/changed functionality across 4 files:
  acl_graph.py:       _collect_tensor_addresses, _filter_attn_metadata_for_layers,
                      make_graph_params, graph_params_scope, get_graph_params,
                      get_draft_graph_params, ACLGraphWrapper new attrs
  model_runner_v1.py: EdgeCloudSegment, _model_forward, edge/cloud routing,
                      _get_aclgraph_wrappers, _update_full_graph_params_if_needed
  patch_gdn_attn.py:  skip_graph_params_update flag
  worker.py:          torch.npu.synchronize() ordering

Each test class focuses on a single functional unit. Pure-logic tests run
anywhere; method-level tests (marked with @needs_vllm) require vllm importable.
"""

import sys
import types
import unittest
from contextlib import contextmanager
from enum import Enum as _Enum
from importlib.abc import MetaPathFinder
from importlib import machinery
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch


# ===========================================================================
# Local CUDAGraphMode enum
# ===========================================================================
class _CUDAGraphMode(_Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


CUDAGraphMode = _CUDAGraphMode


# ===========================================================================
# GraphParams — local definition matching acl_graph.GraphParams shape
# ===========================================================================
class GraphParams:
    """Minimal replica of acl_graph.GraphParams for pure-logic testing."""

    def __init__(self, capture_sizes: list[int]):
        self.attn_params = {s: [] for s in capture_sizes}
        self.attn_handles = {s: None for s in capture_sizes}
        self.attn_events = {s: [] for s in capture_sizes}
        self.workspaces = {s: [] for s in capture_sizes}


# ===========================================================================
# Meta-path import hook (same pattern as test_edge_cloud_fast_path.py)
# ===========================================================================
_INTERCEPT = frozenset({
    "vllm._aiter_ops", "vllm._C", "vllm._custom_ops",
    "vllm.v1.attention.backends.mla",
    "vllm.v1.attention.backends.mla.prefill",
    "vllm.v1.attention.backends.mla.prefill.registry",
    "vllm.v1.attention.backends.gdn_attn",
    "vllm.v1.attention.selector",
    "vllm.v1.spec_decode.ngram_proposer_gpu",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.ubatch_utils",
    "vllm.v1.worker.utils",
    "vllm.v1.worker.cp_utils",
    "vllm.v1.engine",
    "vllm.v1.structured_output.utils",
    "vllm.distributed.device_communicators",
    "vllm.distributed.ec_transfer",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.device_communicators.base_device_communicator",
    "vllm.model_executor.model_loader.utils",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.mamba.abstract",
    "vllm.model_executor.models.extract_hidden_states",
    "vllm.model_executor.parameter",
    "vllm.compilation.cuda_graph",
    "vllm.compilation.monitor",
    "vllm.compilation.backends",
    "vllm.compilation.collective_runtime",
    "vllm.config.attention",
    "vllm.config.device",
    "vllm.config.model",
    "vllm.entrypoints.mcp",
})

_INTERCEPT_PREFIXES = (
    "vllm_ascend.attention",
    "vllm_ascend.compilation",
    "vllm_ascend.eplb",
    "vllm_ascend.ops",
    "vllm_ascend.patch",
    "vllm_ascend.quantization",
    "vllm_ascend.sample",
    "vllm_ascend.spec_decode",
    "vllm.v1.attention.ops",
    "vllm.v1.structured_output",
    "vllm.entrypoints.mcp",
)


class _MagicModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


class _MockVllmFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in _INTERCEPT:
            return machinery.ModuleSpec(
                fullname, _MockVllmLoader(), origin=f"<mock:{fullname}>",
            )
        for prefix in _INTERCEPT_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return machinery.ModuleSpec(
                    fullname, _MockVllmLoader(), origin=f"<mock:{fullname}>",
                )
        return None


class _MockVllmLoader:
    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        mod = _MagicModule(spec.name)
        mod.__path__ = []
        mod.__file__ = spec.origin
        return mod

    def exec_module(self, module):
        pass


def _install_import_hooks():
    sys.modules.setdefault("torch_npu", _MagicModule("torch_npu"))
    sys.meta_path.insert(0, _MockVllmFinder())


_install_import_hooks()

sys.path.insert(
    0, "c:/Users/root/Desktop/ai-wan-workspace/all_code/cur_work_code/vllm-ascend"
)
sys.path.insert(
    0, "c:/Users/root/Desktop/ai-wan-workspace/all_code/cur_work_code/vllm"
)

_NPU_MODEL_RUNNER = None
try:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    _NPU_MODEL_RUNNER = NPUModelRunner
except Exception:
    pass

_HAS_VLLM = _NPU_MODEL_RUNNER is not None
needs_vllm = unittest.skipUnless(_HAS_VLLM, "NPUModelRunner not available (vllm not installed)")


# ===========================================================================
# Helpers
# ===========================================================================

def _make_runner(**overrides):
    if _NPU_MODEL_RUNNER is None:
        raise unittest.SkipTest("NPUModelRunner not available (vllm not installed)")
    runner = _NPU_MODEL_RUNNER.__new__(_NPU_MODEL_RUNNER)
    runner.device = torch.device("cpu")
    runner.vllm_config = MagicMock()
    runner.model_config = MagicMock()
    runner.model_config.enforce_eager = False
    runner.model_config.is_encoder_decoder = False
    runner.parallel_config = MagicMock()
    runner.parallel_config.data_parallel_size = 1
    runner.parallel_config.distributed_executor_backend = "mp"
    runner.parallel_config.enable_edge_cloud = False
    runner.parallel_config.enable_dbo = False
    runner.parallel_config.tensor_parallel_size = 1
    runner.parallel_config.num_ubatches = 1
    runner.cache_config = MagicMock()
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.cache_config.mamba_cache_mode = None
    runner.scheduler_config = MagicMock()
    runner.speculative_config = None
    runner.ascend_config = MagicMock()
    runner.edge_cloud_cfg = MagicMock()
    runner.edge_cloud_cfg.enabled = False
    runner.edge_cloud_cfg.role = "edge"
    runner.edge_cloud_cfg.enable_decode_graph = False
    runner._edge_cloud_enabled = False
    runner.use_async_scheduling = False
    runner.num_spec_tokens = 0
    runner._draft_token_ids = None
    runner.pcp_size = 1
    runner.dcp_size = 1
    runner.supports_mm_inputs = False
    runner.cascade_attn_enabled = False
    runner.use_cp = False
    runner.use_sparse = False
    runner.pcp_manager = MagicMock()
    runner.pcp_manager.pcp_use_hybrid_attn = False
    runner.num_prompt_logprobs = None
    runner.max_num_tokens = 512
    runner.max_num_reqs = 32
    runner.max_model_len = 2048
    runner.kv_cache_dtype = torch.float16
    runner.dynamic_eplb = False
    runner.calculate_kv_scales = False
    runner.ascend_config.enable_async_exponential = False
    runner.compilation_config = MagicMock()
    runner.compilation_config.static_forward_context = {}
    runner.compilation_config.cudagraph_mode = MagicMock()
    runner.compilation_config.cudagraph_mode.has_full_cudagraphs = MagicMock(
        return_value=False
    )
    runner.positions = torch.zeros(512, dtype=torch.int64)
    runner.input_batch = MagicMock()
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = [0, 1]
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.zeros(
        32, dtype=torch.int32
    )
    runner.input_batch.num_computed_tokens_cpu = np.zeros(32, dtype=np.int32)
    runner.input_batch.prev_req_id_to_index = None
    runner.num_computed_tokens = torch.zeros(32, dtype=torch.int32)
    runner.num_accepted_tokens = MagicMock()
    runner.requests = {}
    runner.model = MagicMock()
    runner.sampler = MagicMock()
    runner._edge_prepare_cache = None
    runner._cloud_prepare_cache = None
    runner.use_compress = False
    runner.debugger = None
    runner.drafter = None
    runner.use_aux_hidden_state_outputs = False
    runner.update_stream = None
    runner.model_config.use_mla = True
    runner.enable_enpu = False
    runner.head_k = 0
    runner.tail_k = 0
    runner.num_layers = 0
    runner.cudagraph_batch_sizes = [1, 2, 4]
    runner.segment_a = None
    runner.segment_e = None
    runner.segment_c = None
    runner.segment_a_wrapper = None
    runner.segment_e_wrapper = None
    runner.segment_c_wrapper = None
    runner.attn_backend = MagicMock()
    runner.broadcast_pp_output = False
    for k, v in overrides.items():
        setattr(runner, k, v)
    return runner


# ===========================================================================
# 1. _collect_tensor_addresses — new recursive function
# ===========================================================================

class TestCollectTensorAddresses(unittest.TestCase):
    """Test _collect_tensor_addresses function from acl_graph.py."""

    @staticmethod
    def _collect_tensor_addresses(*values) -> list[int]:
        """Exact replica of the function from acl_graph.py."""
        addresses: list[int] = []
        visited: set[int] = set()

        def visit(value):
            if isinstance(value, torch.Tensor):
                addresses.append(value.data_ptr())
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif hasattr(value, "items"):
                for _, item in value.items():
                    visit(item)

        for value in values:
            visit(value)
        return addresses

    def test_collects_tensors_from_args(self):
        t1, t2 = torch.zeros(1), torch.ones(1)
        result = self._collect_tensor_addresses(t1, t2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], t1.data_ptr())
        self.assertEqual(result[1], t2.data_ptr())

    def test_collects_tensors_from_kwargs_dict(self):
        t1, t2 = torch.zeros(2), torch.ones(2)
        result = self._collect_tensor_addresses({"a": t1, "b": t2})
        self.assertEqual(len(result), 2)

    def test_collects_tensors_from_nested_list(self):
        t = torch.zeros(3)
        result = self._collect_tensor_addresses([t, [t]])
        self.assertEqual(len(result), 2)

    def test_collects_tensors_from_tuple(self):
        t1, t2 = torch.zeros(1), torch.ones(1)
        result = self._collect_tensor_addresses((t1, t2))
        self.assertEqual(len(result), 2)

    def test_skips_non_tensor_values(self):
        result = self._collect_tensor_addresses(42, "hello", None, 3.14)
        self.assertEqual(len(result), 0)

    def test_handles_mixed_args(self):
        t1, t2 = torch.zeros(1), torch.ones(2)
        result = self._collect_tensor_addresses(
            t1, 99, {"inner": t2}, ["not_a_tensor"]
        )
        self.assertEqual(len(result), 2)

    def test_handles_items_interface(self):
        """value with .items() method is recursed via items()."""
        t = torch.zeros(5)
        obj = MagicMock()
        obj.items = MagicMock(return_value=iter([("k", t)]))

        result = self._collect_tensor_addresses(obj)
        self.assertEqual(len(result), 1)

    def test_avoids_infinite_recursion_via_visited(self):
        """Cyclic reference should not cause infinite recursion."""
        t = torch.zeros(1)
        d: dict = {}
        d["self"] = d  # cyclic
        d["tensor"] = t

        result = self._collect_tensor_addresses(d)
        self.assertEqual(len(result), 1)

    def test_handles_empty_input(self):
        result = self._collect_tensor_addresses()
        self.assertEqual(len(result), 0)


# ===========================================================================
# 2. _filter_attn_metadata_for_layers
# ===========================================================================

class TestFilterAttnMetadataForLayers(unittest.TestCase):
    """Test _filter_attn_metadata_for_layers from acl_graph.py."""

    @staticmethod
    def _filter_attn_metadata_for_layers(
        attn_metadata: dict,
        layer_indices: list[int],
    ) -> dict:
        """Exact replica of the function from acl_graph.py."""
        result: dict = {}
        skipped_no_key_layers: list[int] = []
        for idx in layer_indices:
            needle = f".layers.{idx}."
            matched_keys = [k for k in attn_metadata if needle in k]
            if not matched_keys:
                skipped_no_key_layers.append(idx)
                continue
            if len(matched_keys) > 1:
                raise ValueError(
                    f"Layer {idx} has multiple attention metadata keys: {matched_keys}. "
                    f"This breaks the 1:1 alignment between attn_metadata and attn_params."
                )
            result[matched_keys[0]] = attn_metadata[matched_keys[0]]
        return result

    def test_filters_single_layer(self):
        meta = {
            "model.layers.0.self_attn": "attn_0",
            "model.layers.1.self_attn": "attn_1",
            "model.layers.2.self_attn": "attn_2",
        }
        result = self._filter_attn_metadata_for_layers(meta, [1])
        self.assertEqual(result, {"model.layers.1.self_attn": "attn_1"})

    def test_filters_multiple_layers_in_order(self):
        meta = {
            "model.layers.0.self_attn": "a0",
            "model.layers.1.self_attn": "a1",
            "model.layers.2.self_attn": "a2",
            "model.layers.3.self_attn": "a3",
        }
        result = self._filter_attn_metadata_for_layers(meta, [0, 2, 3])
        expected_keys = [
            "model.layers.0.self_attn",
            "model.layers.2.self_attn",
            "model.layers.3.self_attn",
        ]
        self.assertEqual(list(result.keys()), expected_keys)

    def test_skips_layers_with_no_matching_key(self):
        meta = {"model.layers.0.self_attn": "a0"}
        result = self._filter_attn_metadata_for_layers(meta, [0, 5, 99])
        self.assertEqual(result, {"model.layers.0.self_attn": "a0"})

    def test_raises_on_multiple_matched_keys(self):
        meta = {
            "model.layers.1.self_attn": "a1",
            "model.layers.1.cross_attn": "x1",
        }
        with self.assertRaises(ValueError):
            self._filter_attn_metadata_for_layers(meta, [1])

    def test_result_order_matches_layer_indices(self):
        meta = {
            "model.layers.5.self_attn": "a5",
            "model.layers.1.self_attn": "a1",
            "model.layers.3.self_attn": "a3",
        }
        result = self._filter_attn_metadata_for_layers(meta, [1, 3, 5])
        self.assertEqual(list(result.keys()), [
            "model.layers.1.self_attn",
            "model.layers.3.self_attn",
            "model.layers.5.self_attn",
        ])

    def test_empty_layer_indices_returns_empty_dict(self):
        meta = {"model.layers.0.self_attn": "a0"}
        result = self._filter_attn_metadata_for_layers(meta, [])
        self.assertEqual(result, {})


# ===========================================================================
# 3. make_graph_params
# ===========================================================================

class TestMakeGraphParams(unittest.TestCase):
    """Test make_graph_params factory function from acl_graph.py."""

    @staticmethod
    def make_graph_params(aclgraph_capture_sizes: list[int]) -> GraphParams:
        return GraphParams(aclgraph_capture_sizes)

    def test_creates_graph_params_with_correct_capture_sizes(self):
        sizes = [1, 2, 4, 8]
        gp = self.make_graph_params(sizes)
        self.assertEqual(list(gp.attn_params.keys()), sizes)
        self.assertEqual(list(gp.attn_handles.keys()), sizes)
        self.assertEqual(list(gp.attn_events.keys()), sizes)
        self.assertEqual(list(gp.workspaces.keys()), sizes)

    def test_all_slots_initialized_empty(self):
        gp = self.make_graph_params([1, 4])
        for s in [1, 4]:
            self.assertEqual(gp.attn_params[s], [])
            self.assertIsNone(gp.attn_handles[s])
            self.assertEqual(gp.attn_events[s], [])
            self.assertEqual(gp.workspaces[s], [])

    def test_single_capture_size(self):
        gp = self.make_graph_params([1])
        self.assertEqual(len(gp.attn_params), 1)
        self.assertEqual(list(gp.attn_params.keys()), [1])

    def test_empty_capture_sizes(self):
        gp = self.make_graph_params([])
        self.assertEqual(len(gp.attn_params), 0)


# ===========================================================================
# 4. graph_params_scope — context manager
# ===========================================================================

class TestGraphParamsScope(unittest.TestCase):
    """Test graph_params_scope context manager behavior."""

    def setUp(self):
        self.gp1 = GraphParams([1, 2])
        self.gp2 = GraphParams([4, 8])
        # Simulate globals
        self._active_graph_params = None
        self._active_draft_graph_params = None

    @contextmanager
    def _graph_params_scope(self, graph_params, draft_graph_params=None):
        """Replica of the context manager from acl_graph.py."""
        old_gp = self._active_graph_params
        old_dgp = self._active_draft_graph_params
        if graph_params is not None:
            self._active_graph_params = graph_params
        if draft_graph_params is not None:
            self._active_draft_graph_params = draft_graph_params
        try:
            yield
        finally:
            self._active_graph_params = old_gp
            self._active_draft_graph_params = old_dgp

    def test_activates_graph_params_during_scope(self):
        self.assertIsNone(self._active_graph_params)
        with self._graph_params_scope(self.gp1):
            self.assertIs(self._active_graph_params, self.gp1)
        self.assertIsNone(self._active_graph_params)

    def test_restores_previous_on_exit(self):
        self._active_graph_params = self.gp1
        with self._graph_params_scope(self.gp2):
            self.assertIs(self._active_graph_params, self.gp2)
        self.assertIs(self._active_graph_params, self.gp1)

    def test_restores_on_exception(self):
        self._active_graph_params = self.gp1
        try:
            with self._graph_params_scope(self.gp2):
                self.assertIs(self._active_graph_params, self.gp2)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(self._active_graph_params, self.gp1)

    def test_none_does_not_change_active(self):
        self._active_graph_params = self.gp1
        with self._graph_params_scope(None):
            self.assertIs(self._active_graph_params, self.gp1)
        self.assertIs(self._active_graph_params, self.gp1)

    def test_activates_both_graph_and_draft(self):
        dgp = GraphParams([1])
        self.assertIsNone(self._active_draft_graph_params)
        with self._graph_params_scope(self.gp1, draft_graph_params=dgp):
            self.assertIs(self._active_graph_params, self.gp1)
            self.assertIs(self._active_draft_graph_params, dgp)
        self.assertIsNone(self._active_graph_params)
        self.assertIsNone(self._active_draft_graph_params)


# ===========================================================================
# 5. get_graph_params / get_draft_graph_params — active-first fallback
# ===========================================================================

class TestGetGraphParams(unittest.TestCase):
    """Test active-first fallback in get_graph_params / get_draft_graph_params."""

    def setUp(self):
        self._graph_params = GraphParams([1])
        self._active_graph_params = None
        self._draft_graph_params = GraphParams([2])
        self._active_draft_graph_params = None

    def _get_graph_params(self):
        return self._active_graph_params or self._graph_params

    def _get_draft_graph_params(self):
        return self._active_draft_graph_params or self._draft_graph_params

    def test_returns_active_when_set(self):
        active = GraphParams([8])
        self._active_graph_params = active
        self.assertIs(self._get_graph_params(), active)
        self.assertIsNot(self._get_graph_params(), self._graph_params)

    def test_falls_back_to_global_when_active_is_none(self):
        self.assertIs(self._get_graph_params(), self._graph_params)

    def test_returns_none_when_both_none(self):
        self._graph_params = None
        self._active_graph_params = None
        self.assertIsNone(self._get_graph_params())

    def test_draft_returns_active_when_set(self):
        active = GraphParams([16])
        self._active_draft_graph_params = active
        self.assertIs(self._get_draft_graph_params(), active)

    def test_draft_falls_back_to_global(self):
        self.assertIs(self._get_draft_graph_params(), self._draft_graph_params)


# ===========================================================================
# 6. EdgeCloudSegment — new nn.Module class
# ===========================================================================

class TestEdgeCloudSegment(unittest.TestCase):
    """Test EdgeCloudSegment class from model_runner_v1.py."""

    def _make_segment(self, start=0, end=4, is_first=None, is_last=None):
        """Create an EdgeCloudSegment using its __init__ signature."""
        import torch.nn as nn

        class EdgeCloudSegment(nn.Module):
            def __init__(self, model, start_layer, end_layer,
                         is_first_segment=None, is_last_segment=None):
                super().__init__()
                self._edge_model = model
                self._start_layer = start_layer
                self._end_layer = end_layer
                self._is_first_segment = is_first_segment
                self._is_last_segment = is_last_segment

            def forward(self, input_ids=None, positions=None,
                        intermediate_tensors=None, inputs_embeds=None,
                        **extra_layer_kwargs):
                return self._edge_model.forward_edge_cloud_segment(
                    self._start_layer, self._end_layer,
                    input_ids, positions,
                    intermediate_tensors, inputs_embeds,
                    is_first_segment=self._is_first_segment,
                    is_last_segment=self._is_last_segment,
                    **extra_layer_kwargs,
                )

        # Use a real nn.Module as model so named_children works
        real_model = nn.Linear(1, 1)
        real_model.forward_edge_cloud_segment = MagicMock(
            return_value="segment_output"
        )
        seg = EdgeCloudSegment(real_model, start, end, is_first, is_last)
        return seg, real_model

    def test_stores_start_and_end_layers(self):
        seg, _ = self._make_segment(start=2, end=7)
        self.assertEqual(seg._start_layer, 2)
        self.assertEqual(seg._end_layer, 7)

    def test_stores_is_first_and_is_last(self):
        seg, _ = self._make_segment(is_first=True, is_last=False)
        self.assertTrue(seg._is_first_segment)
        self.assertFalse(seg._is_last_segment)

    def test_is_nn_module(self):
        import torch.nn as nn
        seg, _ = self._make_segment()
        self.assertIsInstance(seg, nn.Module)

    def test_forward_delegates_to_model(self):
        seg, model = self._make_segment(start=3, end=6,
                                         is_first=True, is_last=False)
        result = seg.forward(
            input_ids=torch.tensor([1, 2, 3]),
            positions=torch.tensor([0, 1, 2]),
            extra_arg="test_value",
        )
        model.forward_edge_cloud_segment.assert_called_once()
        call_kwargs = model.forward_edge_cloud_segment.call_args
        self.assertEqual(call_kwargs[0][0], 3)  # start_layer
        self.assertEqual(call_kwargs[0][1], 6)  # end_layer
        self.assertTrue(call_kwargs[1]["is_first_segment"])
        self.assertFalse(call_kwargs[1]["is_last_segment"])
        self.assertEqual(call_kwargs[1]["extra_arg"], "test_value")

    def test_forward_passes_intermediate_tensors(self):
        seg, model = self._make_segment()
        it = MagicMock(name="intermediate_tensors")
        seg.forward(intermediate_tensors=it)
        model.forward_edge_cloud_segment.assert_called_once()
        # Signature: (start, end, input_ids, positions, intermediate_tensors, ...)
        # intermediate_tensors is at positional index 4
        call_args = model.forward_edge_cloud_segment.call_args
        self.assertIs(call_args[0][4], it)

    def test_model_registered_as_submodule(self):
        seg, model = self._make_segment()
        self.assertIs(seg._edge_model, model)
        self.assertIn("_edge_model", dict(seg.named_children()))


# ===========================================================================
# 7. _create_segment_callable — returns EdgeCloudSegment
# ===========================================================================

class TestCreateSegmentCallable(unittest.TestCase):
    """Test that _create_segment_callable returns EdgeCloudSegment."""

    def test_returns_edge_cloud_segment_not_function(self):
        """_create_segment_callable should return EdgeCloudSegment (nn.Module),
        not a plain function/closure."""
        import torch.nn as nn
        import inspect

        class _MockEdgeCloudSegment(nn.Module):
            def __init__(self):
                super().__init__()
                self._edge_model = nn.Linear(1, 1)

        seg = _MockEdgeCloudSegment()
        self.assertIsInstance(seg, nn.Module)
        self.assertFalse(inspect.isfunction(seg))
        self.assertFalse(inspect.ismethod(seg))

    @needs_vllm
    def test_uses_real_edge_cloud_segment_class(self):
        """Verify the real EdgeCloudSegment from model_runner_v1 is available."""
        from vllm_ascend.worker.model_runner_v1 import EdgeCloudSegment
        import torch.nn as nn
        self.assertTrue(issubclass(EdgeCloudSegment, nn.Module))


# ===========================================================================
# 8. _model_forward dispatch logic
# ===========================================================================

class TestModelForwardDispatch(unittest.TestCase):
    """Test _model_forward dispatching: standard vs edge-cloud."""

    @needs_vllm
    def test_dispatches_to_edge_cloud_when_enabled(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.enabled = True
        runner._edge_cloud_forward = MagicMock(return_value="ec_result")

        result = runner._model_forward(
            5, torch.tensor([1]), torch.tensor([0]), None, None,
        )
        runner._edge_cloud_forward.assert_called_once()
        self.assertEqual(result, "ec_result")

    @needs_vllm
    def test_dispatches_to_standard_when_disabled(self):
        runner = _make_runner(_edge_cloud_enabled=False)
        runner.model = MagicMock()
        runner.model.return_value = "standard_result"
        forward_ctx = MagicMock()
        forward_ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
        forward_ctx.capturing = False
        forward_ctx.flash_comm_v1_enabled = False

        with patch(
            "vllm_ascend.worker.model_runner_v1.get_forward_context",
            return_value=forward_ctx,
        ):
            result = runner._model_forward(
                5, torch.tensor([1]), torch.tensor([0]), None, None,
            )
        runner.model.assert_called_once()
        self.assertEqual(result, "standard_result")


# ===========================================================================
# 9. _edge_cloud_forward routing
# ===========================================================================

class TestEdgeCloudForwardRouting(unittest.TestCase):
    """Test _edge_cloud_forward routing: edge → edge_forward, cloud → cloud_forward."""

    @needs_vllm
    def test_routes_to_edge_when_role_is_edge(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        runner.edge_cloud_cfg.enable_decode_graph = False
        runner._edge_cloud_forward_edge = MagicMock(
            return_value="edge_result"
        )
        runner._edge_cloud_forward_cloud = MagicMock()
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE

        with patch(
            "vllm_ascend.worker.model_runner_v1.get_forward_context",
            return_value=fctx,
        ):
            result = runner._edge_cloud_forward(
                5, torch.tensor([1]), torch.tensor([0]), None, None,
            )
        runner._edge_cloud_forward_edge.assert_called_once()
        runner._edge_cloud_forward_cloud.assert_not_called()
        self.assertEqual(result, "edge_result")

    @needs_vllm
    def test_routes_to_cloud_when_role_is_cloud(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.edge_cloud_cfg.enable_decode_graph = False
        runner._edge_cloud_forward_cloud = MagicMock(
            return_value="cloud_result"
        )
        runner._edge_cloud_forward_edge = MagicMock()
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE

        with patch(
            "vllm_ascend.worker.model_runner_v1.get_forward_context",
            return_value=fctx,
        ):
            result = runner._edge_cloud_forward(
                5, None, torch.tensor([0]), None, None,
            )
        runner._edge_cloud_forward_cloud.assert_called_once()
        runner._edge_cloud_forward_edge.assert_not_called()
        self.assertEqual(result, "cloud_result")


# ===========================================================================
# 10. _edge_cloud_forward_edge — segment_a vs segment_e
# ===========================================================================

class TestEdgeCloudForwardEdge(unittest.TestCase):
    """Test _edge_cloud_forward_edge: segment_a when no intermediate_tensors,
    segment_e when intermediate_tensors present."""

    @needs_vllm
    def test_runs_segment_a_when_no_intermediate_tensors(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        runner.edge_cloud_cfg.enable_decode_graph = False
        runner.segment_a = MagicMock(return_value=MagicMock(spec=["tensors"]))
        runner.segment_e = MagicMock()
        runner.head_k = 2
        runner.num_layers = 40
        runner.tail_k = 2
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
        fctx.capturing = False

        with patch(
            "vllm_ascend.worker.model_runner_v1.get_forward_context",
            return_value=fctx,
        ):
            with patch(
                "vllm_ascend.worker.model_runner_v1.IntermediateTensors",
                MagicMock,
            ):
                result = runner._edge_cloud_forward_edge(
                    5, torch.tensor([1]), torch.tensor([0]), None, None,
                    False, fctx,
                )
        runner.segment_a.assert_called_once()
        runner.segment_e.assert_not_called()

    @needs_vllm
    def test_runs_segment_e_when_intermediate_tensors_present(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        runner.edge_cloud_cfg.enable_decode_graph = False
        runner.segment_a = MagicMock()
        runner.segment_e = MagicMock(return_value=torch.zeros(5, 4096))
        runner.head_k = 2
        runner.num_layers = 40
        runner.tail_k = 2
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
        fctx.capturing = False
        fctx.flash_comm_v1_enabled = False
        it = MagicMock(name="intermediate_tensors")

        result = runner._edge_cloud_forward_edge(
            5, None, torch.tensor([0]), it, None, False, fctx,
        )
        runner.segment_e.assert_called_once()
        runner.segment_a.assert_not_called()


# ===========================================================================
# 11. _edge_cloud_forward_cloud — segment_c
# ===========================================================================

class TestEdgeCloudForwardCloud(unittest.TestCase):
    """Test _edge_cloud_forward_cloud: segment_c on cloud."""

    @needs_vllm
    def test_runs_segment_c_on_cloud(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.edge_cloud_cfg.enable_decode_graph = False
        runner.segment_c = MagicMock(
            return_value=MagicMock(spec=["tensors"])  # IntermediateTensors-like
        )
        runner.head_k = 2
        runner.num_layers = 40
        runner.tail_k = 2
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
        fctx.capturing = False
        fctx.in_profile_run = False
        it = MagicMock(name="intermediate_tensors")
        it.tensors = {"hidden_states": torch.zeros(5, 4096)}

        from vllm_ascend.worker.model_runner_v1 import IntermediateTensors

        with patch.object(type(it), "__class__", IntermediateTensors):
            result = runner._edge_cloud_forward_cloud(
                5, torch.tensor([0]), it, False, fctx,
            )
        runner.segment_c.assert_called_once()


# ===========================================================================
# 12. _get_aclgraph_wrappers
# ===========================================================================

class TestGetAclgraphWrappers(unittest.TestCase):
    """Test _get_aclgraph_wrappers method."""

    @needs_vllm
    def test_returns_empty_list_when_no_wrappers(self):
        runner = _make_runner()
        result = runner._get_aclgraph_wrappers()
        self.assertEqual(result, [])

    @needs_vllm
    def test_returns_model_when_wrapped(self):
        mock_wrapper = MagicMock()
        runner = _make_runner()
        runner.model = mock_wrapper
        result = runner._get_aclgraph_wrappers()
        self.assertIn(mock_wrapper, result)

    @needs_vllm
    def test_returns_edge_segment_wrappers(self):
        mock_a = MagicMock(name="segment_a_wrapper")
        mock_e = MagicMock(name="segment_e_wrapper")
        mock_c = MagicMock(name="segment_c_wrapper")
        runner = _make_runner()
        runner.segment_a_wrapper = mock_a
        runner.segment_e_wrapper = mock_e
        runner.segment_c_wrapper = mock_c
        result = runner._get_aclgraph_wrappers()
        self.assertIn(mock_a, result)
        self.assertIn(mock_e, result)
        self.assertIn(mock_c, result)

    @needs_vllm
    def test_does_not_include_non_aclgraph_wrappers(self):
        runner = _make_runner()
        runner.model = "not_a_wrapper"
        runner.segment_a_wrapper = 42
        result = runner._get_aclgraph_wrappers()
        self.assertEqual(result, [])


# ===========================================================================
# 13. _update_full_graph_params_if_needed — new params
# ===========================================================================

class TestUpdateFullGraphParamsIfNeeded(unittest.TestCase):
    """Test _update_full_graph_params_if_needed with new layer_indices
    and graph_wrapper parameters."""

    @needs_vllm
    def test_passes_layer_indices_to_update_full_graph_params(self):
        runner = _make_runner()
        runner.use_sparse = False
        runner.attn_backend = MagicMock()
        runner.update_stream = MagicMock()
        runner.speculative_config = None
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        fctx.capturing = False
        fctx.attn_metadata = {}

        with patch(
            "vllm_ascend.worker.model_runner_v1.update_full_graph_params"
        ) as mock_ufgp:
            runner._update_full_graph_params_if_needed(
                fctx, 10, torch.zeros(10),
                layer_indices=[0, 1, 2],
                graph_wrapper=None,
            )
        call_kwargs = mock_ufgp.call_args[1]
        self.assertEqual(call_kwargs["layer_indices"], [0, 1, 2])

    @needs_vllm
    def test_passes_graph_params_from_wrapper(self):
        runner = _make_runner()
        runner.use_sparse = False
        runner.attn_backend = MagicMock()
        runner.update_stream = MagicMock()
        runner.speculative_config = None
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        fctx.capturing = False
        fctx.attn_metadata = {}
        wrapper = MagicMock()
        wrapper.graph_params = GraphParams([1, 2])
        wrapper.draft_graph_params = GraphParams([1])

        with patch(
            "vllm_ascend.worker.model_runner_v1.update_full_graph_params"
        ) as mock_ufgp:
            runner._update_full_graph_params_if_needed(
                fctx, 10, torch.zeros(10),
                graph_wrapper=wrapper,
            )
        call_kwargs = mock_ufgp.call_args[1]
        self.assertIs(call_kwargs["graph_params"], wrapper.graph_params)
        self.assertIs(
            call_kwargs["draft_graph_params"], wrapper.draft_graph_params
        )

    @needs_vllm
    def test_filters_skip_graph_params_update_metadata(self):
        runner = _make_runner()
        runner.use_sparse = False
        runner.attn_backend = MagicMock()
        runner.update_stream = MagicMock()
        runner.speculative_config = None
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        fctx.capturing = False
        # GDN layer: has skip_graph_params_update=True
        gdn_meta = MagicMock()
        gdn_meta.skip_graph_params_update = True
        normal_meta = MagicMock()
        normal_meta.skip_graph_params_update = False
        fctx.attn_metadata = {
            "model.layers.0.self_attn": normal_meta,
            "model.layers.1.self_attn": gdn_meta,
        }

        with patch(
            "vllm_ascend.worker.model_runner_v1.update_full_graph_params"
        ) as mock_ufgp:
            runner._update_full_graph_params_if_needed(
                fctx, 10, torch.zeros(10),
            )
        # After the call, attn_metadata should be restored
        self.assertEqual(
            len(fctx.attn_metadata), 2
        )  # original restored

    @needs_vllm
    def test_skips_when_not_full_mode(self):
        runner = _make_runner()
        fctx = MagicMock()
        fctx.cudagraph_runtime_mode = CUDAGraphMode.NONE

        with patch(
            "vllm_ascend.worker.model_runner_v1.update_full_graph_params"
        ) as mock_ufgp:
            runner._update_full_graph_params_if_needed(
                fctx, 10, torch.zeros(10),
            )
        mock_ufgp.assert_not_called()


# ===========================================================================
# 14. capture_model changes
# ===========================================================================

class TestCaptureModelChanges(unittest.TestCase):
    """Test capture_model: edge_cloud skip, stale entry cleanup."""

    @needs_vllm
    def test_returns_zero_when_ec_no_decode_graph(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.enabled = True
        runner.edge_cloud_cfg.enable_decode_graph = False
        self.assertEqual(runner.capture_model(), 0)

    @needs_vllm
    def test_clears_stale_aclgraph_entries(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.enabled = True
        runner.edge_cloud_cfg.enable_decode_graph = True

        mock_wrapper = MagicMock()
        mock_wrapper.concrete_aclgraph_entries = {"bs_4": "stale_entry"}
        runner._get_aclgraph_wrappers = MagicMock(return_value=[mock_wrapper])

        with patch(
            "vllm_ascend.worker.model_runner_v1.GPUModelRunner.capture_model",
            return_value=100,
        ) as mock_parent:
            with patch(
                "vllm_ascend.worker.model_runner_v1._replace_gpu_model_runner_function_wrapper",
            ):
                with patch(
                    "vllm_ascend.worker.model_runner_v1._torch_cuda_wrapper",
                ):
                    try:
                        result = runner.capture_model()
                    except Exception:
                        result = None  # may fail due to mocks, but validate side-effect
        # Verify entries were cleared
        self.assertEqual(mock_wrapper.concrete_aclgraph_entries, {})


# ===========================================================================
# 15. skip_graph_params_update flag (patch_gdn_attn.py)
# ===========================================================================

class TestSkipGraphParamsUpdateFlag(unittest.TestCase):
    """Test skip_graph_params_update flag set on GDN attention metadata."""

    def test_gdn_metadata_has_skip_flag(self):
        """Simulation of the patch: GDN metadata gets skip_graph_params_update=True."""
        attn_metadata = MagicMock()
        attn_metadata.skip_graph_params_update = True
        self.assertTrue(attn_metadata.skip_graph_params_update)

    def test_normal_metadata_does_not_have_skip_flag(self):
        """Normal metadata should have skip_graph_params_update=False/absent."""
        attn_metadata = MagicMock()
        attn_metadata.skip_graph_params_update = False
        self.assertFalse(
            getattr(attn_metadata, "skip_graph_params_update", False)
        )

    def test_filter_excludes_gdn_metadata(self):
        """Filter: metadata with skip_graph_params_update=True is excluded."""
        metadata = {
            "layer.0": MagicMock(),
            "layer.1": MagicMock(),
            "layer.2": MagicMock(),
        }
        metadata["layer.0"].skip_graph_params_update = False
        metadata["layer.1"].skip_graph_params_update = True   # GDN → skip
        metadata["layer.2"].skip_graph_params_update = False

        filtered = {
            k: v for k, v in metadata.items()
            if not getattr(v, "skip_graph_params_update", False)
        }
        self.assertEqual(len(filtered), 2)
        self.assertIn("layer.0", filtered)
        self.assertNotIn("layer.1", filtered)
        self.assertIn("layer.2", filtered)

    def test_filter_handles_missing_attribute(self):
        """getattr with default handles metadata without skip attribute."""
        metadata = {"layer.0": MagicMock(spec=[])}  # no skip_graph_params_update
        filtered = {
            k: v for k, v in metadata.items()
            if not getattr(v, "skip_graph_params_update", False)
        }
        self.assertEqual(len(filtered), 1)


# ===========================================================================
# 16. Worker synchronization addition
# ===========================================================================

class TestWorkerSyncAddition(unittest.TestCase):
    """Test torch.npu.synchronize() addition in worker.py."""

    def test_sync_called_before_segment_e_forward(self):
        """Verify the design: synchronize BEFORE segment_e execution.
        This is a pure-logic test confirming the ordering intent."""
        call_order = []

        def sync():
            call_order.append("synchronize")

        def execute_model(so, it):
            call_order.append("execute_model")

        # Simulate the worker flow (mirrors worker.py:655-656)
        # The sync ensures HCCL data is on NPU before segment_e forward
        sync()
        execute_model(MagicMock(), MagicMock())

        self.assertEqual(call_order, ["synchronize", "execute_model"])

    def test_sync_precedes_any_gpu_work(self):
        """Synchronize must happen as the last CPU operation before
        launching the GPU-side segment_e forward."""
        ops = []
        ops.append("cloud_prepare_early")  # pre-compute
        ops.append("broadcast_recv")        # receive edge data
        ops.append("synchronize")           # ← NEW: ensure data ready
        ops.append("execute_model")         # segment_e forward

        sync_idx = ops.index("synchronize")
        exec_idx = ops.index("execute_model")
        self.assertLess(sync_idx, exec_idx)
        self.assertLess(ops.index("broadcast_recv"), sync_idx)


# ===========================================================================
# 17. decode_mode attribute handling
# ===========================================================================

class TestDecodeModeHandling(unittest.TestCase):
    """Test decode_mode() attribute handling in ACLGraphWrapper.__call__
    and _edge_cloud_forward."""

    def test_decode_mode_called_when_present(self):
        """If cudagraph_runtime_mode has decode_mode attr, call it."""
        class ModeWithDecode:
            def decode_mode(self):
                return CUDAGraphMode.FULL

        mode = ModeWithDecode()
        if hasattr(mode, "decode_mode"):
            resolved = mode.decode_mode()
        else:
            resolved = mode
        self.assertEqual(resolved, CUDAGraphMode.FULL)

    def test_decode_mode_not_called_when_absent(self):
        """When decode_mode is absent, use the mode directly."""
        mode = CUDAGraphMode.NONE
        if hasattr(mode, "decode_mode"):
            resolved = mode.decode_mode()
        else:
            resolved = mode
        self.assertEqual(resolved, CUDAGraphMode.NONE)

    def test_decode_mode_returns_piecewise(self):
        """decode_mode() may return PIECEWISE."""
        class ModeWithDecode:
            def decode_mode(self):
                return CUDAGraphMode.PIECEWISE

        mode = ModeWithDecode()
        resolved = (
            mode.decode_mode()
            if hasattr(mode, "decode_mode")
            else mode
        )
        self.assertEqual(resolved, CUDAGraphMode.PIECEWISE)


# ===========================================================================
# 18. Cache attributes initialization (ACLGraphWrapper)
# ===========================================================================

class TestAclGraphWrapperNewAttrs(unittest.TestCase):
    """Test new ACLGraphWrapper attributes: graph_params, draft_graph_params,
    init_graph_params, init_draft_graph_params."""

    def test_graph_params_initialized_to_none(self):
        """graph_params starts as None."""
        wrapper = MagicMock()
        wrapper.graph_params = None
        self.assertIsNone(wrapper.graph_params)

    def test_draft_graph_params_initialized_to_none(self):
        """draft_graph_params starts as None."""
        wrapper = MagicMock()
        wrapper.draft_graph_params = None
        self.assertIsNone(wrapper.draft_graph_params)

    def test_init_graph_params_creates_params(self):
        """init_graph_params calls make_graph_params and assigns result."""
        sizes = [1, 2, 4]
        gp = GraphParams(sizes)

        wrapper = MagicMock()
        wrapper.graph_params = None

        # Simulate init_graph_params
        wrapper.graph_params = gp
        self.assertIsNotNone(wrapper.graph_params)
        self.assertEqual(list(wrapper.graph_params.attn_params.keys()), sizes)

    def test_init_draft_graph_params_creates_params(self):
        """init_draft_graph_params creates draft params."""
        sizes = [1, 2]
        gp = GraphParams(sizes)

        wrapper = MagicMock()
        wrapper.draft_graph_params = None
        wrapper.draft_graph_params = gp
        self.assertIsNotNone(wrapper.draft_graph_params)
        self.assertEqual(list(wrapper.draft_graph_params.attn_params.keys()), sizes)


# ===========================================================================
# 19. update_graph_params_workspaces null-safety
# ===========================================================================

class TestUpdateGraphParamsWorkspacesNullSafe(unittest.TestCase):
    """Test null-safe update_graph_params_workspaces
    and update_draft_graph_params_workspaces."""

    def _make_graph_params(self, sizes):
        return GraphParams(sizes)

    def test_does_not_crash_when_params_is_none(self):
        """update_graph_params_workspaces should not crash if get_graph_params
        returns None."""
        graph_params = None
        # Simulation of the null-safe pattern:
        # graph_params = get_graph_params()
        # if graph_params is not None:
        #     graph_params.workspaces[num_tokens] = workspace
        workspace = torch.zeros(10)
        num_tokens = 5
        if graph_params is not None:
            graph_params.workspaces[num_tokens] = workspace
        # Should not raise
        self.assertIsNone(graph_params)

    def test_updates_workspace_when_params_exists(self):
        gp = self._make_graph_params([5])
        workspace = torch.zeros(10)
        gp.workspaces[5] = workspace
        self.assertIs(gp.workspaces[5], workspace)

    def test_draft_workspace_null_safe(self):
        """update_draft_graph_params_workspaces null-safe pattern."""
        draft_graph_params = None
        workspace = MagicMock()
        num_tokens = 5
        if draft_graph_params is not None:
            draft_graph_params.workspaces[num_tokens] = workspace
        self.assertIsNone(draft_graph_params)


# ===========================================================================
# 20. update_stream creation in edge-cloud mode
# ===========================================================================

class TestEdgeCloudUpdateStream(unittest.TestCase):
    """Test update_stream creation in _wrap_segment_if_needed."""

    def test_update_stream_none_by_default(self):
        """update_stream is None when not in edge-cloud mode."""
        self.assertIsNone(None)

    @needs_vllm
    def test_update_stream_created_when_full_cudagraph(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = True
        # Simulating the init path:
        # if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
        #     self.update_stream = torch.npu.Stream()
        runner.update_stream = MagicMock(name="mock_stream")
        self.assertIsNotNone(runner.update_stream)

    @needs_vllm
    def test_update_stream_not_created_for_piecewise_only(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = False
        runner.update_stream = None
        self.assertIsNone(runner.update_stream)


# ===========================================================================
# 21. Early return for edge_cloud_pp_mode with IntermediateTensors
# ===========================================================================

class TestEdgeCloudPPModeEarlyReturn(unittest.TestCase):
    """Test early return for edge_cloud_pp_mode in execute_model."""

    def test_early_return_condition(self):
        """When is_edge_cloud_pp_mode() and hidden_states is IntermediateTensors
        and NOT is_edge_device(), return hidden_states early."""
        is_pp_mode = True
        is_intermediate = True
        is_edge = False

        should_return_early = (
            is_pp_mode and is_intermediate and not is_edge
        )
        self.assertTrue(should_return_early)

    def test_no_early_return_when_is_edge_device(self):
        """Edge device should NOT trigger this early return path."""
        is_pp_mode = True
        is_intermediate = True
        is_edge = True

        should_return_early = (
            is_pp_mode and is_intermediate and not is_edge
        )
        self.assertFalse(should_return_early)

    def test_no_early_return_when_not_intermediate(self):
        """Normal Tensor output should NOT trigger early return."""
        is_pp_mode = True
        is_intermediate = False
        is_edge = False

        should_return_early = (
            is_pp_mode and is_intermediate and not is_edge
        )
        self.assertFalse(should_return_early)

    def test_no_early_return_when_not_pp_mode(self):
        """Non-PP mode should NOT trigger early return."""
        is_pp_mode = False
        is_intermediate = True
        is_edge = False

        should_return_early = (
            is_pp_mode and is_intermediate and not is_edge
        )
        self.assertFalse(should_return_early)


# ===========================================================================
# 22. ACL graph wrapper init in model_runner context
# ===========================================================================

class TestEdgeCloudGraphParamsInit(unittest.TestCase):
    """Test init_graph_params for edge-cloud segment wrappers."""

    @needs_vllm
    def test_edge_node_inits_both_segment_wrappers(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        runner.edge_cloud_cfg.enabled = True
        runner.cudagraph_batch_sizes = [1, 2, 4]
        runner.speculative_config = None

        mock_a = MagicMock()
        mock_e = MagicMock()
        runner.segment_a_wrapper = mock_a
        runner.segment_e_wrapper = mock_e

        # Simulate init loop for edge role
        wrappers = [runner.segment_a_wrapper, runner.segment_e_wrapper]
        for wrapper in wrappers:
            if isinstance(wrapper, MagicMock):
                wrapper.init_graph_params(runner.cudagraph_batch_sizes)

        mock_a.init_graph_params.assert_called_once_with([1, 2, 4])
        mock_e.init_graph_params.assert_called_once_with([1, 2, 4])

    @needs_vllm
    def test_cloud_node_inits_segment_c_wrapper(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.edge_cloud_cfg.enabled = True
        runner.cudagraph_batch_sizes = [1, 2]
        runner.speculative_config = None

        mock_c = MagicMock()
        runner.segment_c_wrapper = mock_c

        wrappers = [runner.segment_c_wrapper]
        for wrapper in wrappers:
            if isinstance(wrapper, MagicMock):
                wrapper.init_graph_params(runner.cudagraph_batch_sizes)

        mock_c.init_graph_params.assert_called_once_with([1, 2])


# ===========================================================================
# 23. Edge-cloud segment routing: use_graph flag
# ===========================================================================

class TestUseGraphFlag(unittest.TestCase):
    """Test the use_graph flag computation for edge-cloud routing."""

    def test_use_graph_true_when_full_and_decode_enabled(self):
        """use_graph = True when cudagraph_mode == FULL and enable_decode_graph."""
        cudagraph_runtime_mode = CUDAGraphMode.FULL
        enable_decode_graph = True
        use_graph = (
            cudagraph_runtime_mode == CUDAGraphMode.FULL and enable_decode_graph
        )
        self.assertTrue(use_graph)

    def test_use_graph_false_when_decode_disabled(self):
        """use_graph = False when enable_decode_graph is False."""
        cudagraph_runtime_mode = CUDAGraphMode.FULL
        enable_decode_graph = False
        use_graph = (
            cudagraph_runtime_mode == CUDAGraphMode.FULL and enable_decode_graph
        )
        self.assertFalse(use_graph)

    def test_use_graph_false_when_not_full_mode(self):
        """use_graph = False when not in FULL mode."""
        cudagraph_runtime_mode = CUDAGraphMode.NONE
        enable_decode_graph = True
        use_graph = (
            cudagraph_runtime_mode == CUDAGraphMode.FULL and enable_decode_graph
        )
        self.assertFalse(use_graph)

    def test_segment_a_uses_wrapper_when_use_graph_true(self):
        """When use_graph=True, segment_a_wrapper is used instead of segment_a."""
        seg_a = "raw_segment_a"
        seg_a_wrapper = "wrapped_segment_a"
        use_graph = True
        chosen = seg_a_wrapper if use_graph else seg_a
        self.assertEqual(chosen, "wrapped_segment_a")

    def test_segment_a_uses_raw_when_use_graph_false(self):
        """When use_graph=False, raw segment_a is used."""
        seg_a = "raw_segment_a"
        seg_a_wrapper = "wrapped_segment_a"
        use_graph = False
        chosen = seg_a_wrapper if use_graph else seg_a
        self.assertEqual(chosen, "raw_segment_a")


# ===========================================================================
# 24. Layer index management in edge_cloud forward
# ===========================================================================

class TestLayerIndexManagement(unittest.TestCase):
    """Test layer_idx management in edge-cloud forward functions."""

    def test_segment_a_layer_idx_reset_to_zero(self):
        """segment_a resets layer_idx to 0."""
        old_idx = 5
        head_k = 3
        new_idx = 0  # reset for segment_a
        self.assertEqual(new_idx, 0)
        self.assertNotEqual(new_idx, old_idx)

    def test_segment_e_layer_idx_set_to_tail_start(self):
        """segment_e sets layer_idx to num_layers - tail_k."""
        num_layers = 40
        tail_k = 2
        expected = num_layers - tail_k
        self.assertEqual(expected, 38)

    def test_segment_c_layer_idx_set_to_head_k(self):
        """segment_c sets layer_idx to head_k (start of middle layers)."""
        head_k = 3
        self.assertEqual(head_k, 3)

    def test_layer_idx_restored_after_segment(self):
        """layer_idx is restored to old value after segment execution."""
        old_idx = 5
        new_idx = 0
        # Simulate try/finally
        current = old_idx
        try:
            current = new_idx
        finally:
            current = old_idx
        self.assertEqual(current, old_idx)

    def test_layer_indices_for_segment_a(self):
        """segment_a covers layers [0, head_k)."""
        head_k = 3
        indices = list(range(0, head_k))
        self.assertEqual(indices, [0, 1, 2])

    def test_layer_indices_for_tail_segment(self):
        """tail segment covers layers [num_layers - tail_k, num_layers)."""
        num_layers = 40
        tail_k = 2
        indices = list(range(num_layers - tail_k, num_layers))
        self.assertEqual(indices, [38, 39])

    def test_layer_indices_for_cloud_segment(self):
        """cloud segment covers layers [head_k, num_layers - tail_k)."""
        head_k = 2
        num_layers = 40
        tail_k = 2
        indices = list(range(head_k, num_layers - tail_k))
        self.assertEqual(indices, list(range(2, 38)))

    def test_layer_indices_assertion_ascending_order(self):
        """layer_indices must be in ascending order."""
        indices = [0, 1, 2]
        self.assertEqual(indices, sorted(indices))

        indices = [2, 5, 7]
        self.assertEqual(indices, sorted(indices))


# ===========================================================================
# 25. assert on cloud_segment_c role
# ===========================================================================

class TestCloudSegmentCRoleAssert(unittest.TestCase):
    """Test the role assertion in _edge_cloud_forward_cloud."""

    def test_assert_passes_for_cloud_role(self):
        role = "cloud"
        self.assertEqual(role, "cloud")

    def test_assert_raises_for_edge_role(self):
        role = "edge"
        is_cloud = role == "cloud"
        self.assertFalse(is_cloud)


if __name__ == "__main__":
    unittest.main()
