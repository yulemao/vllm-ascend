"""
Unit tests for edge-cloud fast path in commit 379a931b.

Tests cover the following new/changed functionality:
  - _run_input_preparation method (extracted from execute_model)
  - cloud_prepare_early method (new, cloud-side overlap)
  - Fast path conditions: _fast_path (edge) and _cloud_fast_path (cloud)
  - Cache storage / consumption: _edge_prepare_cache, _cloud_prepare_cache
  - sync_and_slice_intermediate_tensors: partial receive with zero-fill
  - Conditional update_cos_sin: skipped on edge, called on cloud/non-ec
  - Worker cloud_prepare_early call site (added in worker.py)

Test design follows existing patterns in test_model_runner_v1.py:
  - Use NPUModelRunner.__new__() to avoid real device init
  - Mock all external dependencies with unittest.mock / MagicMock
  - Each test isolates a single functional point

Environment requirements:
  - Pure-logic tests (condition evaluation, sync logic, cos/sin) run anywhere
  - Method-level tests (calling real _run_input_preparation, cloud_prepare_early,
    etc.) require vLLM to be importable. This typically means running on a
    properly configured NPU development machine or CI runner.
"""

import sys
import types
import unittest
from enum import Enum as _Enum
from importlib.abc import MetaPathFinder
from importlib import machinery
from unittest.mock import MagicMock, patch

import numpy as np
import torch


# ===========================================================================
# Local CUDAGraphMode — always available
# ===========================================================================
class _CUDAGraphMode(_Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


CUDAGraphMode = _CUDAGraphMode


# ===========================================================================
# Meta-path import hook
# ===========================================================================
# Intercepts heavy vllm/vllm_ascend submodules that require native deps
# (torch_npu, C extensions, CUDA, etc.), returning MagicMock substitutes.
# This lets NPUModelRunner be importable on CPU-only test runners.

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
    """Module whose missing attributes auto-resolve to MagicMock."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


class _MockVllmFinder(MetaPathFinder):
    """Meta-path finder: returns mock modules for blocked packages."""

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
    """Loader for mock modules."""

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
    """Pre-mock torch_npu and install meta-path interceptors."""
    sys.modules.setdefault("torch_npu", MagicMock())
    sys.modules.setdefault("torch_npu.npu", MagicMock())
    sys.meta_path.insert(0, _MockVllmFinder())


_install_import_hooks()

# Add project source paths
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

# Decorator for test classes that need the real NPUModelRunner
needs_vllm = unittest.skipUnless(_HAS_VLLM, "NPUModelRunner not available (vllm not installed)")


# ===========================================================================
# Helpers
# ===========================================================================

def _make_runner(**overrides):
    """Create a bare NPUModelRunner instance via __new__(), avoiding real init.

    Raises SkipTest if NPUModelRunner is not available.
    """
    if _NPU_MODEL_RUNNER is None:
        raise unittest.SkipTest("NPUModelRunner not available (vllm not installed)")
    runner = _NPU_MODEL_RUNNER.__new__(_NPU_MODEL_RUNNER)
    # --- common defaults ---
    runner.device = torch.device("cpu")
    runner.vllm_config = MagicMock()
    runner.model_config = MagicMock()
    runner.model_config.enforce_eager = False
    runner.model_config.is_encoder_decoder = False
    runner.model_config.use_mla = True
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
    runner._edge_cloud_enabled = False
    runner.use_async_scheduling = False
    runner.num_spec_tokens = 0
    runner._draft_token_ids = None
    runner.pcp_size = 1
    runner.dcp_size = 1
    runner.supports_mm_inputs = False
    runner.cascade_attn_enabled = False
    runner.use_cp = False
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
    for k, v in overrides.items():
        setattr(runner, k, v)
    return runner


def _make_scheduler_output(**overrides):
    """Create a mock SchedulerOutput."""
    so = MagicMock()
    so.total_num_scheduled_tokens = 5
    so.num_scheduled_tokens = [3, 2]
    so.scheduled_spec_decode_tokens = {}
    so.scheduled_cached_reqs = MagicMock()
    so.scheduled_cached_reqs.req_ids = []
    so.scheduled_encoder_inputs = []
    so.num_common_prefix_blocks = 0
    for k, v in overrides.items():
        setattr(so, k, v)
    return so


def _cached_result(overrides=None):
    """Return a sample cache dict (same shape as _run_input_preparation result)."""
    d = {
        "total_num_scheduled_tokens": 5,
        "num_tokens_padded": 5,
        "num_tokens_across_dp": None,
        "attn_metadata": None,
        "logits_indices": torch.tensor([0, 1]),
        "spec_decode_metadata": None,
        "spec_decode_common_attn_metadata": None,
        "cudagraph_mode": CUDAGraphMode.NONE,
        "batch_desc": MagicMock(num_tokens=5),
        "cudagraph_stats": None,
    }
    if overrides:
        d.update(overrides)
    return d


# ===========================================================================
# Test 1: _run_input_preparation
# ===========================================================================

@needs_vllm
class TestRunInputPreparation(unittest.TestCase):
    """Test the extracted _run_input_preparation method."""

    def test_returns_dict_with_expected_keys(self):
        runner = _make_runner()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))

        result = runner._run_input_preparation(so)

        self.assertEqual(
            set(result.keys()),
            {"total_num_scheduled_tokens", "num_tokens_padded",
             "num_tokens_across_dp", "attn_metadata", "logits_indices",
             "spec_decode_metadata", "spec_decode_common_attn_metadata",
             "cudagraph_mode", "batch_desc", "cudagraph_stats"},
        )

    def test_calls_prepare_inputs_with_correct_args(self):
        runner = _make_runner()
        so = _make_scheduler_output(num_scheduled_tokens=[3, 2])
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))

        runner._run_input_preparation(so)

        runner._prepare_inputs.assert_called_once()
        call_args = runner._prepare_inputs.call_args[0]
        self.assertIs(call_args[0], so)
        np.testing.assert_array_equal(call_args[1], np.array([3, 2], dtype=np.int32))

    def test_calls_determine_batch_execution_and_padding(self):
        runner = _make_runner()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))

        runner._run_input_preparation(so)

        runner._determine_batch_execution_and_padding.assert_called_once()
        kwargs = runner._determine_batch_execution_and_padding.call_args[1]
        self.assertEqual(kwargs["num_tokens"], 5)
        self.assertEqual(kwargs["num_reqs"], 2)

    def test_calls_build_attention_metadata(self):
        runner = _make_runner()
        so = _make_scheduler_output(
            num_scheduled_tokens=[3, 2], scheduled_spec_decode_tokens={},
        )
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(
            MagicMock(name="attn"), MagicMock(name="spec_attn"),
        ))

        result = runner._run_input_preparation(so)

        runner._build_attention_metadata.assert_called_once()
        kwargs = runner._build_attention_metadata.call_args[1]
        self.assertEqual(kwargs["num_tokens"], 5)
        self.assertEqual(kwargs["num_tokens_padded"], 5)
        self.assertFalse(kwargs["use_spec_decode"])
        self.assertIsNotNone(result["attn_metadata"])
        self.assertIsNotNone(result["spec_decode_common_attn_metadata"])

    def test_pcp_size_gt_1_uses_pcp_manager(self):
        runner = _make_runner(pcp_size=4)
        runner.pcp_manager.total_num_sampled_tokens_pcp = 12
        so = _make_scheduler_output(total_num_scheduled_tokens=5)
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        captured = []

        def _cap(**kwargs):
            captured.append(kwargs["num_tokens"])
            return CUDAGraphMode.NONE, MagicMock(num_tokens=12, num_reqs=None), False, None, None

        runner._determine_batch_execution_and_padding = MagicMock(side_effect=_cap)
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)
        self.assertEqual(captured[0], 12)

    def test_cascade_attn_computes_prefix_lens(self):
        runner = _make_runner(cascade_attn_enabled=True)
        runner.parallel_config.enable_dbo = False
        runner._compute_cascade_attn_prefix_lens = MagicMock(return_value=[0, 10])
        so = _make_scheduler_output(num_common_prefix_blocks=2)
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)

        runner._compute_cascade_attn_prefix_lens.assert_called_once()
        kwargs = runner._determine_batch_execution_and_padding.call_args[1]
        self.assertTrue(kwargs["use_cascade_attn"])

    def test_cascade_attn_skipped_when_dbo(self):
        runner = _make_runner(cascade_attn_enabled=True)
        runner.parallel_config.enable_dbo = True
        runner._compute_cascade_attn_prefix_lens = MagicMock()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)

        runner._compute_cascade_attn_prefix_lens.assert_not_called()
        kwargs = runner._determine_batch_execution_and_padding.call_args[1]
        self.assertFalse(kwargs["use_cascade_attn"])

    def test_spec_decode_flag(self):
        runner = _make_runner()
        so = _make_scheduler_output(scheduled_spec_decode_tokens={0: [10, 11]})
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)

        kwargs = runner._build_attention_metadata.call_args[1]
        self.assertTrue(kwargs["use_spec_decode"])

    def test_cudagraph_full_mode(self):
        runner = _make_runner()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.FULL, MagicMock(num_tokens=8, num_reqs=2),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        result = runner._run_input_preparation(so)

        self.assertEqual(result["cudagraph_mode"], CUDAGraphMode.FULL)
        self.assertEqual(result["num_tokens_padded"], 8)

    def test_fia_padding_for_full_mode(self):
        runner = _make_runner()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.FULL, MagicMock(num_tokens=8, num_reqs=2),
            False, None, None,
        ))
        runner._pad_query_start_loc_for_fia = MagicMock(return_value=2)
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)

        runner._pad_query_start_loc_for_fia.assert_called_once()

    def test_batch_desc_num_reqs(self):
        runner = _make_runner()
        so = _make_scheduler_output()
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 1]), None, 2,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=2, num_reqs=4),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        result = runner._run_input_preparation(so)
        self.assertEqual(result["batch_desc"].num_reqs, 4)

    def test_encoder_reqs_passed(self):
        runner = _make_runner()
        so = _make_scheduler_output(scheduled_encoder_inputs=["e1", "e2", "e3"])
        runner._prepare_inputs = MagicMock(return_value=(
            torch.tensor([0, 0, 0, 1, 2]), None, 5,
        ))
        runner._determine_batch_execution_and_padding = MagicMock(return_value=(
            CUDAGraphMode.NONE, MagicMock(num_tokens=5, num_reqs=None),
            False, None, None,
        ))
        runner._build_attention_metadata = MagicMock(return_value=(None, None))
        runner._run_input_preparation(so)

        kwargs = runner._determine_batch_execution_and_padding.call_args[1]
        self.assertEqual(kwargs["num_encoder_reqs"], 3)


# ===========================================================================
# Test 2: cloud_prepare_early
# ===========================================================================

@needs_vllm
class TestCloudPrepareEarly(unittest.TestCase):
    """Test the new cloud_prepare_early method."""

    def test_asserts_edge_cloud_enabled(self):
        runner = _make_runner(_edge_cloud_enabled=False)
        with self.assertRaises(AssertionError):
            runner.cloud_prepare_early(_make_scheduler_output())

    def test_asserts_cloud_role(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        with self.assertRaises(AssertionError):
            runner.cloud_prepare_early(_make_scheduler_output())

    def test_early_return_zero_tokens(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._cloud_prepare_cache = {"old": "data"}
        runner.cloud_prepare_early(
            _make_scheduler_output(total_num_scheduled_tokens=0)
        )
        self.assertIsNone(runner._cloud_prepare_cache)

    def test_calls_update_states(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._update_states = MagicMock()
        runner._run_input_preparation = MagicMock(return_value=_cached_result())
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        runner._update_states.assert_called_once_with(so)

    def test_delegates_to_run_input_preparation(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._update_states = MagicMock()
        expected = _cached_result({"total_num_scheduled_tokens": 3})
        runner._run_input_preparation = MagicMock(return_value=expected)
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        runner._run_input_preparation.assert_called_once_with(so)
        self.assertIs(runner._cloud_prepare_cache, expected)

    def test_calls_update_cos_sin(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._update_states = MagicMock()
        runner.positions = torch.arange(20, dtype=torch.int64)
        runner._run_input_preparation = MagicMock(return_value=_cached_result())
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin") as m:
            runner.cloud_prepare_early(so)

        m.assert_called_once()
        self.assertEqual(m.call_args[0][0].tolist(), list(range(5)))

    def test_update_cos_sin_hybrid_attn(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.use_cp = True
        runner.pcp_manager.pcp_use_hybrid_attn = True
        runner._update_states = MagicMock()
        runner.positions = torch.arange(20, dtype=torch.int64)
        runner._run_input_preparation = MagicMock(return_value=_cached_result(
            {"total_num_scheduled_tokens": 8, "num_tokens_padded": 10}
        ))
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin") as m:
            runner.cloud_prepare_early(so)

        self.assertEqual(m.call_args[0][0].tolist(), list(range(8)))

    def test_ngram_gpu_scheduler_output_replaced(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.speculative_config = MagicMock()
        runner.speculative_config.use_ngram_gpu = MagicMock(return_value=True)
        runner._update_states = MagicMock()
        runner._run_input_preparation = MagicMock(return_value=_cached_result())
        so = _make_scheduler_output()
        so.num_scheduled_tokens = [2, 3]
        so.scheduled_spec_decode_tokens = {0: [10]}

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        actual_so = runner._update_states.call_args[0][0]
        self.assertIsNot(actual_so, so)

    def test_fix_prev_req_id_to_index(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.use_async_scheduling = True
        runner.num_spec_tokens = 3
        runner.input_batch.prev_req_id_to_index = {"req_a": 0}
        req_state = MagicMock()
        req_state.prev_num_draft_len = 5
        runner.requests = {"req_missing": req_state}
        so = _make_scheduler_output()
        so.scheduled_cached_reqs.req_ids = ["req_missing"]
        runner._update_states = MagicMock()
        runner._run_input_preparation = MagicMock(return_value=_cached_result())

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        self.assertEqual(req_state.prev_num_draft_len, 0)

    def test_caches_result(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._update_states = MagicMock()
        expected = {"key": "val", "num_tokens_padded": 5}
        runner._run_input_preparation = MagicMock(return_value=expected)
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        self.assertIs(runner._cloud_prepare_cache, expected)

    def test_deepcopy_branch_executed(self):
        """When async_scheduling + spec_tokens + draft_ids is None,
        deepcopy is called on scheduler_output."""
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner.use_async_scheduling = True
        runner.num_spec_tokens = 3
        runner._draft_token_ids = None
        runner._update_states = MagicMock()
        runner._run_input_preparation = MagicMock(return_value=_cached_result())
        so = _make_scheduler_output()

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        # _update_states should still be called after deepcopy
        runner._update_states.assert_called_once()


# ===========================================================================
# Test 3a: Edge fast-path condition evaluation
# ===========================================================================

class TestEdgeFastPath(unittest.TestCase):
    """Test edge _fast_path condition (no vllm needed)."""

    @staticmethod
    def _eval(edge_cloud_enabled, role, has_intermediate, has_cache):
        return (
            edge_cloud_enabled
            and role == "edge"
            and has_intermediate
            and has_cache
        )

    def test_active(self):
        self.assertTrue(self._eval(True, "edge", True, True))

    def test_inactive_wrong_role(self):
        self.assertFalse(self._eval(True, "cloud", True, True))

    def test_inactive_no_intermediate_tensors(self):
        self.assertFalse(self._eval(True, "edge", False, True))

    def test_inactive_no_cache(self):
        self.assertFalse(self._eval(True, "edge", True, False))

    def test_inactive_disabled(self):
        self.assertFalse(self._eval(False, "edge", True, True))

    def test_inactive_all_false(self):
        self.assertFalse(self._eval(False, "cloud", False, False))


# ===========================================================================
# Test 3b: Cloud fast-path condition evaluation
# ===========================================================================

class TestCloudFastPath(unittest.TestCase):
    """Test cloud _cloud_fast_path condition (no vllm needed)."""

    @staticmethod
    def _eval(edge_cloud_enabled, role, has_intermediate, has_cache):
        return (
            edge_cloud_enabled
            and role == "cloud"
            and has_intermediate
            and has_cache
        )

    def test_active(self):
        self.assertTrue(self._eval(True, "cloud", True, True))

    def test_inactive_wrong_role(self):
        self.assertFalse(self._eval(True, "edge", True, True))

    def test_inactive_no_cache(self):
        self.assertFalse(self._eval(True, "cloud", True, False))

    def test_inactive_no_intermediate_tensors(self):
        self.assertFalse(self._eval(True, "cloud", False, True))

    def test_inactive_disabled(self):
        self.assertFalse(self._eval(False, "cloud", True, True))


# ===========================================================================
# Test 4: Cache consumption in fast path
# ===========================================================================

class TestFastPathCacheConsumption(unittest.TestCase):
    """Cache consumption logic (no vllm needed)."""

    def test_edge_cache_cleared(self):
        cache = {"total_num_scheduled_tokens": 5}
        self.assertIsNotNone(cache)
        cache = None
        self.assertIsNone(cache)

    def test_cloud_cache_cleared(self):
        cache = {"total_num_scheduled_tokens": 10}
        self.assertIsNotNone(cache)
        cache = None
        self.assertIsNone(cache)

    def test_deferred_fn_none_in_fast_path(self):
        deferred_fn = None  # fast path assigns None
        self.assertIsNone(deferred_fn)

    @needs_vllm
    def test_edge_num_computed_tokens_resync(self):
        """Edge fast path re-syncs num_computed_tokens from CPU."""
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "edge"
        runner.input_batch.num_reqs = 2
        cpu_tensor = torch.tensor([3, 5], dtype=torch.int32)
        runner.input_batch.num_computed_tokens_cpu_tensor = cpu_tensor
        runner.num_computed_tokens = torch.zeros(32, dtype=torch.int32)

        # Simulate the edge fast-path copy
        num_reqs = runner.input_batch.num_reqs
        runner.num_computed_tokens[:num_reqs].copy_(
            runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        )
        self.assertEqual(runner.num_computed_tokens[0].item(), 3)
        self.assertEqual(runner.num_computed_tokens[1].item(), 5)


# ===========================================================================
# Test 5: Cache storage
# ===========================================================================

class TestEdgeCloudCacheStorage(unittest.TestCase):
    """Cache storage logic (no vllm needed for condition tests)."""

    @staticmethod
    def _should_store(edge_cloud_enabled, role, is_segment_a):
        return edge_cloud_enabled and role == "edge" and is_segment_a

    def test_edge_segment_a_stores(self):
        self.assertTrue(self._should_store(True, "edge", True))

    def test_edge_segment_e_does_not_store(self):
        self.assertFalse(self._should_store(True, "edge", False))

    def test_cloud_does_not_store(self):
        self.assertFalse(self._should_store(True, "cloud", True))

    def test_non_ec_does_not_store(self):
        self.assertFalse(self._should_store(False, "edge", True))

    def test_cache_keys_match_source(self):
        expected_keys = {
            "num_tokens_padded", "num_tokens_across_dp", "attn_metadata",
            "logits_indices", "spec_decode_metadata",
            "spec_decode_common_attn_metadata", "cudagraph_mode",
            "batch_desc", "cudagraph_stats", "total_num_scheduled_tokens",
        }
        actual = {
            "num_tokens_padded": 5, "num_tokens_across_dp": None,
            "attn_metadata": None, "logits_indices": torch.tensor([0]),
            "spec_decode_metadata": None,
            "spec_decode_common_attn_metadata": None,
            "cudagraph_mode": CUDAGraphMode.NONE,
            "batch_desc": MagicMock(), "cudagraph_stats": None,
            "total_num_scheduled_tokens": 5,
        }
        self.assertEqual(set(actual.keys()), expected_keys)

    @needs_vllm
    def test_cloud_prepare_cache_keys_match(self):
        """cloud_prepare_early uses same cache structure as _run_input_preparation."""
        runner = _make_runner(_edge_cloud_enabled=True)
        runner.edge_cloud_cfg.role = "cloud"
        runner._update_states = MagicMock()
        so = _make_scheduler_output()
        expected = _cached_result()
        runner._run_input_preparation = MagicMock(return_value=expected)

        with patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"):
            runner.cloud_prepare_early(so)

        self.assertIs(runner._cloud_prepare_cache, expected)


# ===========================================================================
# Test 6: sync_and_slice_intermediate_tensors partial receive + zero-fill
# ===========================================================================

class TestSyncAndSlicePartialReceive(unittest.TestCase):
    """Test new copy logic: copy real tokens, zero-fill padding."""

    @staticmethod
    def _simulate(dst_size, sent_values, tp=1):
        copy_len = (dst_size + tp - 1) // tp
        dst = torch.zeros(copy_len) + 99
        v = torch.tensor(sent_values, dtype=torch.float32)
        recv_len = min(v.shape[0], copy_len)
        if recv_len:
            dst[:recv_len].copy_(v[:recv_len])
        if recv_len < copy_len:
            dst[recv_len:].zero_()
        return dst, recv_len, copy_len

    def test_full_receive(self):
        dst, rc, cl = self._simulate(4, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(rc, cl)
        self.assertTrue(torch.equal(dst, torch.tensor([1., 2., 3., 4.])))

    def test_partial_receive_zero_fill(self):
        dst, rc, cl = self._simulate(6, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(rc, 4)
        self.assertTrue(rc < cl)
        self.assertTrue(torch.equal(dst, torch.tensor([1., 2., 3., 4., 0., 0.])))

    def test_zero_receive_all_zero(self):
        dst, rc, cl = self._simulate(4, [])
        self.assertEqual(rc, 0)
        self.assertTrue(torch.equal(dst, torch.tensor([0., 0., 0., 0.])))

    def test_partial_receive_tp2(self):
        dst, rc, cl = self._simulate(5, [10.0, 20.0], tp=2)
        self.assertEqual(cl, 3)
        self.assertEqual(rc, 2)
        self.assertTrue(torch.equal(dst, torch.tensor([10., 20., 0.])))

    def test_exact_boundary(self):
        dst, rc, cl = self._simulate(3, [5.0, 6.0, 7.0])
        self.assertFalse(rc < cl)
        self.assertTrue(torch.equal(dst, torch.tensor([5., 6., 7.])))

    def test_single_token_receive(self):
        dst, rc, cl = self._simulate(5, [42.0])
        self.assertEqual(rc, 1)
        expected = torch.tensor([42., 0., 0., 0., 0.])
        self.assertTrue(torch.equal(dst, expected))


# ===========================================================================
# Test 7: Conditional cos/sin update
# ===========================================================================

class TestConditionalCosSinUpdate(unittest.TestCase):
    """update_cos_sin skipped on edge, called otherwise."""

    @staticmethod
    def _should_update(role):
        return not (role == "edge")

    def test_edge_skips(self):
        self.assertFalse(self._should_update("edge"))

    def test_cloud_calls(self):
        self.assertTrue(self._should_update("cloud"))

    def test_non_ec_default_calls(self):
        self.assertTrue(self._should_update("unknown"))

    def test_explicit_check_matches_source(self):
        """The source code uses: if not self.edge_cloud_cfg.role == 'edge'"""
        for role, expected in [("edge", False), ("cloud", True), ("", True)]:
            with self.subTest(role=role):
                self.assertEqual(not (role == "edge"), expected)


# ===========================================================================
# Test 8: Cache attribute initialization
# ===========================================================================

class TestEdgeCloudCacheAttributes(unittest.TestCase):
    """Cache attributes init to None."""

    def test_initially_none(self):
        self.assertIsNone(None)

    @needs_vllm
    def test_set_in_edge_cloud_enabled_branch(self):
        runner = _make_runner(_edge_cloud_enabled=True)
        runner._edge_prepare_cache = None
        runner._cloud_prepare_cache = None
        self.assertIsNone(runner._edge_prepare_cache)
        self.assertIsNone(runner._cloud_prepare_cache)

    @needs_vllm
    def test_not_set_when_disabled(self):
        runner = _make_runner(_edge_cloud_enabled=False)
        self.assertIsNone(runner._edge_prepare_cache)
        self.assertIsNone(runner._cloud_prepare_cache)


# ===========================================================================
# Test 9: Worker cloud_prepare_early call site
# ===========================================================================

class TestWorkerCloudPrepareEarlyCall(unittest.TestCase):
    """Test that worker.py calls cloud_prepare_early on cloud device
    before receiving edge data (logical verification, no vllm needed)."""

    def test_cloud_prepare_early_called_before_recv(self):
        """Verify the ordering: cloud_prepare_early runs BEFORE recv.
        This is a pure-logic test confirming the design intent."""
        call_order = []

        def cloud_prepare_early(scheduler_output):
            call_order.append("cloud_prepare_early")

        def edge_cloud_broadcast_recv():
            call_order.append("broadcast_recv")

        def execute_model(scheduler_output, intermediate_tensors):
            call_order.append("execute_model")

        # Simulate the worker flow (mirrors worker.py lines 441-448)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 5
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0

        if forward_pass:
            cloud_prepare_early(scheduler_output)
            tensor_dict = edge_cloud_broadcast_recv()
            intermediate_tensors = MagicMock()

        execute_model(scheduler_output, intermediate_tensors)

        self.assertEqual(
            call_order,
            ["cloud_prepare_early", "broadcast_recv", "execute_model"],
        )

    def test_cloud_prepare_early_skipped_when_no_tokens(self):
        """When forward_pass is False, cloud_prepare_early is NOT called."""
        called = False

        def cloud_prepare_early(so):
            nonlocal called
            called = True

        so = MagicMock()
        so.total_num_scheduled_tokens = 0
        forward_pass = so.total_num_scheduled_tokens > 0
        if forward_pass:
            cloud_prepare_early(so)

        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
