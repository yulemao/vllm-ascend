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
# This file is a part of the vllm-ascend project.
#

"""Unit tests for edge-cloud monkey-patch modules.

These patches inject ``forward_edge_cloud_segment`` into vLLM model classes
(Qwen3.5 / DeepseekV2 / Kimi-K2.5). The segment logic is tested directly by
binding the unbound patch function to a lightweight mock ``self`` that exposes
just ``layers`` / ``embed_input_ids`` / ``norm``.

Importing the patch modules pulls in real vLLM model classes, so the whole
suite is skipped when vLLM (or the model classes) are not importable.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

try:
    from vllm.sequence import IntermediateTensors

    from vllm_ascend.patch.models.deepseek_v2_edge_cloud import (
        _deepseek_v2_lm_forward_edge_cloud_segment,
        _forward_edge_cloud_segment_deepseek_v2,
    )
    from vllm_ascend.patch.models.kimi_k25_edge_cloud import (
        _kimi_k25_forward_edge_cloud_segment,
    )
    from vllm_ascend.patch.models.qwen3_5_edge_cloud import (
        _forward_edge_cloud_segment_qwen3_5,
        _qwen3_5_cond_forward_edge_cloud_segment,
        _qwen3_5_lm_forward_edge_cloud_segment,
        _qwen3_5_set_moe_parameters,
        _qwen3_5_update_physical_experts_metadata,
        _qwen_next_set_moe_parameters,
        _qwen_next_update_physical_experts_metadata,
    )

    _PATCHES_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    _PATCHES_AVAILABLE = False
    _IMPORT_ERROR = exc

needs_patches = unittest.skipUnless(
    _PATCHES_AVAILABLE,
    f"edge-cloud patch modules not available: {_IMPORT_ERROR}",
)


# Module paths patched inside the MoE helper functions (they import the
# decoder/moe classes lazily, so patching the module attribute is enough).
_QWEN_NEXT_MODULE = "vllm.model_executor.models.qwen3_next"
_QWEN35_MODULE = "vllm.model_executor.models.qwen3_5"


class _FakeDecoderLayer:
    """Stand-in for Qwen3NextDecoderLayer / Qwen3_5DecoderLayer."""

    def __init__(self, mlp=None):
        self.mlp = mlp


class _FakeSparseMoeBlock:
    """Stand-in for Qwen3NextSparseMoeBlock with the attrs the helpers read."""

    def __init__(self, **attrs):
        self.experts = MagicMock()
        self.experts.update_expert_map = MagicMock()
        self.n_logical_experts = attrs.get("n_logical_experts", 8)
        self.n_physical_experts = attrs.get("n_physical_experts", 16)
        self.n_local_physical_experts = attrs.get("n_local_physical_experts", 4)
        self.n_routed_experts = attrs.get("n_routed_experts", 8)
        self.n_redundant_experts = attrs.get("n_redundant_experts", 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor(seed=0, shape=(2, 4)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _make_layer(return_hs=None, return_res=None, capture=None):
    """A fake decoder layer that returns (hidden_states, residual)."""

    def _call(*args, **kwargs):
        if capture is not None:
            capture["args"] = args
            capture["kwargs"] = kwargs
        return return_hs, return_res

    return _call


def _make_model_self(num_layers, embed_hs=None, norm_hs=None):
    """Build a mock ``self`` for the segment forward functions."""
    fake = MagicMock()
    fake.layers = [
        _make_layer(return_hs=embed_hs, return_res=embed_hs)
        for _ in range(num_layers)
    ]
    fake.embed_input_ids = MagicMock(return_value=embed_hs)
    fake.norm = MagicMock(return_value=(norm_hs, None))
    return fake


# ---------------------------------------------------------------------------
# DeepseekV2 segment forward
# ---------------------------------------------------------------------------


@needs_patches
class TestDeepseekV2EdgeCloudSegment(TestBase):
    """Tests for _forward_edge_cloud_segment_deepseek_v2."""

    NUM_LAYERS = 6

    def _model(self):
        hs = _tensor(seed=1)
        return _make_model_self(
            self.NUM_LAYERS, embed_hs=hs, norm_hs=_tensor(seed=2)
        )

    def test_invalid_segment_range_asserts(self):
        fake = self._model()
        with self.assertRaises(AssertionError):
            _forward_edge_cloud_segment_deepseek_v2(
                fake,
                start_layer=3,
                end_layer=2,
                input_ids=torch.tensor([0]),
                positions=torch.tensor([0]),
            )

    def test_first_segment_embeds_and_returns_intermediate_tensors(self):
        fake = self._model()
        result = _forward_edge_cloud_segment_deepseek_v2(
            fake,
            start_layer=0,
            end_layer=2,
            input_ids=torch.tensor([0, 1]),
            positions=torch.tensor([0, 1]),
            is_first_segment=True,
            is_last_segment=False,
        )
        fake.embed_input_ids.assert_called_once()
        self.assertIsInstance(result, IntermediateTensors)
        self.assertIn("hidden_states", result)
        self.assertIn("residual", result)

    def test_first_segment_prefers_inputs_embeds(self):
        fake = self._model()
        embeds = _tensor(seed=9)
        _forward_edge_cloud_segment_deepseek_v2(
            fake,
            start_layer=0,
            end_layer=1,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            inputs_embeds=embeds,
            is_first_segment=True,
            is_last_segment=False,
        )
        fake.embed_input_ids.assert_not_called()

    def test_middle_segment_reads_intermediate_tensors(self):
        fake = self._model()
        hs = _tensor(seed=3)
        res = _tensor(seed=4)
        result = _forward_edge_cloud_segment_deepseek_v2(
            fake,
            start_layer=2,
            end_layer=4,
            input_ids=None,
            positions=torch.tensor([0, 1]),
            intermediate_tensors=IntermediateTensors(
                {"hidden_states": hs, "residual": res}
            ),
            is_first_segment=False,
            is_last_segment=False,
        )
        self.assertIsInstance(result, IntermediateTensors)

    def test_middle_segment_asserts_when_intermediate_none(self):
        fake = self._model()
        with self.assertRaises(AssertionError):
            _forward_edge_cloud_segment_deepseek_v2(
                fake,
                start_layer=2,
                end_layer=4,
                input_ids=None,
                positions=torch.tensor([0, 1]),
                intermediate_tensors=None,
                is_first_segment=False,
                is_last_segment=False,
            )

    def test_last_segment_runs_norm_and_returns_tensor(self):
        norm_hs = _tensor(seed=5)
        fake = _make_model_self(
            self.NUM_LAYERS, embed_hs=_tensor(seed=1), norm_hs=norm_hs
        )
        result = _forward_edge_cloud_segment_deepseek_v2(
            fake,
            start_layer=4,
            end_layer=self.NUM_LAYERS,
            input_ids=None,
            positions=torch.tensor([0, 1]),
            intermediate_tensors=IntermediateTensors(
                {
                    "hidden_states": _tensor(seed=3),
                    "residual": _tensor(seed=4),
                }
            ),
            is_first_segment=False,
            is_last_segment=True,
        )
        fake.norm.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)

    def test_first_segment_empty_range_defaults_residual_to_zeros(self):
        # Empty layer range (start == end) leaves residual None; the
        # non-last-segment branch then materializes zeros_like(hidden_states).
        fake = self._model()
        result = _forward_edge_cloud_segment_deepseek_v2(
            fake,
            start_layer=0,
            end_layer=0,
            input_ids=torch.tensor([0, 1]),
            positions=torch.tensor([0, 1]),
            is_first_segment=True,
            is_last_segment=False,
        )
        self.assertTrue(
            torch.equal(result["residual"], torch.zeros_like(_tensor(seed=1)))
        )

    def test_is_first_last_defaulted_from_pp_group(self):
        fake = self._model()
        with patch(
            "vllm_ascend.patch.models.deepseek_v2_edge_cloud.get_pp_group"
        ) as mock_pp:
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True
            _forward_edge_cloud_segment_deepseek_v2(
                fake,
                start_layer=0,
                end_layer=self.NUM_LAYERS,
                input_ids=torch.tensor([0]),
                positions=torch.tensor([0]),
            )
            mock_pp.assert_called()


@needs_patches
class TestDeepseekV2LMWrapper(TestBase):
    """Tests for _deepseek_v2_lm_forward_edge_cloud_segment delegation."""

    def test_delegates_to_model_forward(self):
        fake = MagicMock()
        fake.model.forward_edge_cloud_segment = MagicMock(return_value="ok")
        result = _deepseek_v2_lm_forward_edge_cloud_segment(
            fake,
            start_layer=0,
            end_layer=2,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
        )
        self.assertEqual(result, "ok")
        args = fake.model.forward_edge_cloud_segment.call_args[0]
        self.assertEqual(args[0], 0)  # start_layer
        self.assertEqual(args[1], 2)  # end_layer
        self.assertTrue(torch.equal(args[2], torch.tensor([0])))  # input_ids
        self.assertTrue(torch.equal(args[3], torch.tensor([0])))  # positions
        self.assertIsNone(args[4])  # intermediate_tensors


# ---------------------------------------------------------------------------
# Qwen3.5 segment forward
# ---------------------------------------------------------------------------


@needs_patches
class TestQwen35EdgeCloudSegment(TestBase):
    """Tests for _forward_edge_cloud_segment_qwen3_5."""

    NUM_LAYERS = 4

    def _model(self):
        hs = _tensor(seed=1)
        return _make_model_self(self.NUM_LAYERS, embed_hs=hs, norm_hs=_tensor(seed=2))

    def test_invalid_segment_range_asserts(self):
        fake = self._model()
        with self.assertRaises(AssertionError):
            _forward_edge_cloud_segment_qwen3_5(
                fake,
                start_layer=2,
                end_layer=1,
                input_ids=torch.tensor([0]),
                positions=torch.tensor([0]),
            )

    def test_first_segment_returns_intermediate_tensors(self):
        fake = self._model()
        result = _forward_edge_cloud_segment_qwen3_5(
            fake,
            start_layer=0,
            end_layer=2,
            input_ids=torch.tensor([0, 1]),
            positions=torch.tensor([0, 1]),
            is_first_segment=True,
            is_last_segment=False,
        )
        fake.embed_input_ids.assert_called_once()
        self.assertIsInstance(result, IntermediateTensors)

    def test_layer_called_with_keyword_arguments(self):
        capture = {}
        hs = _tensor(seed=1)
        fake = MagicMock()
        fake.layers = [_make_layer(return_hs=hs, return_res=hs, capture=capture)]
        fake.embed_input_ids = MagicMock(return_value=hs)
        fake.norm = MagicMock(return_value=(_tensor(seed=2), None))
        _forward_edge_cloud_segment_qwen3_5(
            fake,
            start_layer=0,
            end_layer=1,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            is_first_segment=True,
            is_last_segment=False,
        )
        # Qwen layer signature uses keyword args exclusively
        self.assertEqual(capture["args"], ())
        self.assertIn("hidden_states", capture["kwargs"])
        self.assertIn("residual", capture["kwargs"])
        self.assertIn("positions", capture["kwargs"])

    def test_last_segment_returns_normed_tensor(self):
        norm_hs = _tensor(seed=7)
        fake = _make_model_self(
            self.NUM_LAYERS, embed_hs=_tensor(seed=1), norm_hs=norm_hs
        )
        result = _forward_edge_cloud_segment_qwen3_5(
            fake,
            start_layer=2,
            end_layer=self.NUM_LAYERS,
            input_ids=None,
            positions=torch.tensor([0, 1]),
            intermediate_tensors=IntermediateTensors(
                {"hidden_states": _tensor(seed=3), "residual": _tensor(seed=4)}
            ),
            is_first_segment=False,
            is_last_segment=True,
        )
        fake.norm.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)

    def test_middle_segment_asserts_when_intermediate_none(self):
        fake = self._model()
        with self.assertRaises(AssertionError):
            _forward_edge_cloud_segment_qwen3_5(
                fake,
                start_layer=1,
                end_layer=2,
                input_ids=None,
                positions=torch.tensor([0]),
                intermediate_tensors=None,
                is_first_segment=False,
                is_last_segment=False,
            )


@needs_patches
class TestQwen35Wrappers(TestBase):
    """Tests for the Qwen3.5 LM and ConditionalGeneration wrappers."""

    def test_lm_wrapper_delegates_to_model(self):
        fake = MagicMock()
        fake.model.forward_edge_cloud_segment = MagicMock(return_value="lm_ok")
        result = _qwen3_5_lm_forward_edge_cloud_segment(
            fake,
            start_layer=0,
            end_layer=2,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
        )
        self.assertEqual(result, "lm_ok")
        fake.model.forward_edge_cloud_segment.assert_called_once()

    def test_cond_wrapper_delegates_to_language_model(self):
        fake = MagicMock()
        fake.language_model.forward_edge_cloud_segment = MagicMock(
            return_value="cond_ok"
        )
        result = _qwen3_5_cond_forward_edge_cloud_segment(
            fake,
            start_layer=0,
            end_layer=2,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
        )
        self.assertEqual(result, "cond_ok")
        fake.language_model.forward_edge_cloud_segment.assert_called_once()


# ---------------------------------------------------------------------------
# Kimi-K2.5 wrapper
# ---------------------------------------------------------------------------


@needs_patches
class TestKimiK25Wrapper(TestBase):
    """Tests for _kimi_k25_forward_edge_cloud_segment delegation."""

    def test_delegates_to_language_model(self):
        fake = MagicMock()
        fake.language_model.forward_edge_cloud_segment = MagicMock(
            return_value="kimi_ok"
        )
        result = _kimi_k25_forward_edge_cloud_segment(
            fake,
            start_layer=0,
            end_layer=3,
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        self.assertEqual(result, "kimi_ok")
        fake.language_model.forward_edge_cloud_segment.assert_called_once()


# ---------------------------------------------------------------------------
# Patch installation sanity checks (only when importable)
# ---------------------------------------------------------------------------


@needs_patches
class TestEdgeCloudPatchInstallation(TestBase):
    """Verify the monkey-patches are actually installed on the model classes."""

    def test_deepseek_v2_model_has_segment_forward(self):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

        self.assertTrue(hasattr(DeepseekV2Model, "forward_edge_cloud_segment"))

    def test_qwen3_5_model_has_segment_forward(self):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

        self.assertTrue(hasattr(Qwen3_5Model, "forward_edge_cloud_segment"))

    def test_kimi_k25_has_segment_forward(self):
        from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration

        self.assertTrue(
            hasattr(KimiK25ForConditionalGeneration, "forward_edge_cloud_segment")
        )


# ---------------------------------------------------------------------------
# MoE physical-expert metadata helpers (Qwen3-Next flavour)
# ---------------------------------------------------------------------------


@needs_patches
class TestQwenNextMoeUpdateMetadata(TestBase):
    """Tests for _qwen_next_update_physical_experts_metadata."""

    def _patch_classes(self):
        return (
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextDecoderLayer", _FakeDecoderLayer),
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextSparseMoeBlock", _FakeSparseMoeBlock),
        )

    def test_updates_self_attributes_and_redundant_count(self):
        moe = _FakeSparseMoeBlock()
        layer = _FakeDecoderLayer(mlp=moe)
        fake_self = SimpleNamespace(
            num_local_physical_experts=4,
            num_logical_experts=8,
            model=SimpleNamespace(layers=[object(), layer]),
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_update_physical_experts_metadata(
                fake_self, num_physical_experts=16, num_local_physical_experts=4
            )
        self.assertEqual(fake_self.num_physical_experts, 16)
        self.assertEqual(fake_self.num_local_physical_experts, 4)
        self.assertEqual(fake_self.num_redundant_experts, 8)  # 16 - 8

    def test_updates_moe_block_attrs_and_calls_update_expert_map(self):
        moe = _FakeSparseMoeBlock()
        layer = _FakeDecoderLayer(mlp=moe)
        fake_self = SimpleNamespace(
            num_local_physical_experts=4,
            num_logical_experts=8,
            model=SimpleNamespace(layers=[layer]),
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_update_physical_experts_metadata(
                fake_self, num_physical_experts=20, num_local_physical_experts=4
            )
        self.assertEqual(moe.n_physical_experts, 20)
        self.assertEqual(moe.n_local_physical_experts, 4)
        self.assertEqual(moe.n_redundant_experts, 12)  # 20 - 8
        moe.experts.update_expert_map.assert_called_once()

    def test_skips_non_decoder_and_non_moe_layers(self):
        moe = _FakeSparseMoeBlock()
        good = _FakeDecoderLayer(mlp=moe)
        decoder_no_moe = _FakeDecoderLayer(mlp=object())  # mlp not a moe block
        non_decoder = object()
        fake_self = SimpleNamespace(
            num_local_physical_experts=4,
            num_logical_experts=8,
            model=SimpleNamespace(
                layers=[non_decoder, decoder_no_moe, good]
            ),
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_update_physical_experts_metadata(
                fake_self, num_physical_experts=16, num_local_physical_experts=4
            )
        moe.experts.update_expert_map.assert_called_once()

    def test_asserts_when_local_physical_experts_mismatch(self):
        fake_self = SimpleNamespace(
            num_local_physical_experts=4,
            num_logical_experts=8,
            model=SimpleNamespace(layers=[]),
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            with self.assertRaises(AssertionError):
                _qwen_next_update_physical_experts_metadata(
                    fake_self,
                    num_physical_experts=16,
                    num_local_physical_experts=2,  # != self.num_local_physical_experts
                )


@needs_patches
class TestQwenNextMoeSetParameters(TestBase):
    """Tests for _qwen_next_set_moe_parameters."""

    def _patch_classes(self):
        return (
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextDecoderLayer", _FakeDecoderLayer),
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextSparseMoeBlock", _FakeSparseMoeBlock),
        )

    def test_collects_moe_layers_and_reads_example_attrs(self):
        moe = _FakeSparseMoeBlock(
            n_logical_experts=8, n_physical_experts=16,
            n_local_physical_experts=4, n_routed_experts=8,
            n_redundant_experts=8,
        )
        layer = _FakeDecoderLayer(mlp=moe)
        fake_self = SimpleNamespace(
            model=SimpleNamespace(layers=[object(), layer])
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_set_moe_parameters(fake_self)
        self.assertEqual(fake_self.expert_weights, [])
        self.assertEqual(fake_self.moe_layers, [moe.experts])
        self.assertEqual(fake_self.num_moe_layers, 1)
        self.assertEqual(fake_self.num_expert_groups, 1)
        self.assertEqual(fake_self.num_shared_experts, 0)
        self.assertEqual(fake_self.num_logical_experts, 8)
        self.assertEqual(fake_self.num_physical_experts, 16)
        self.assertEqual(fake_self.num_local_physical_experts, 4)
        self.assertEqual(fake_self.num_routed_experts, 8)
        self.assertEqual(fake_self.num_redundant_experts, 8)

    def test_counts_multiple_moe_layers(self):
        moe1 = _FakeSparseMoeBlock()
        moe2 = _FakeSparseMoeBlock()
        layers = [_FakeDecoderLayer(mlp=moe1), _FakeDecoderLayer(mlp=moe2)]
        fake_self = SimpleNamespace(model=SimpleNamespace(layers=layers))
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_set_moe_parameters(fake_self)
        self.assertEqual(fake_self.num_moe_layers, 2)
        self.assertEqual(len(fake_self.moe_layers), 2)

    def test_none_found_zeros_all_counters(self):
        fake_self = SimpleNamespace(
            model=SimpleNamespace(layers=[object(), _FakeDecoderLayer(mlp=object())])
        )
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen_next_set_moe_parameters(fake_self)
        self.assertEqual(fake_self.moe_layers, [])
        self.assertEqual(fake_self.num_moe_layers, 0)
        self.assertEqual(fake_self.num_expert_groups, 0)
        self.assertEqual(fake_self.num_logical_experts, 0)
        self.assertEqual(fake_self.num_physical_experts, 0)
        self.assertEqual(fake_self.num_local_physical_experts, 0)
        self.assertEqual(fake_self.num_routed_experts, 0)
        self.assertEqual(fake_self.num_redundant_experts, 0)


# ---------------------------------------------------------------------------
# MoE physical-expert metadata helpers (Qwen3.5 flavour)
# ---------------------------------------------------------------------------


@needs_patches
class TestQwen35MoeUpdateMetadata(TestBase):
    """Tests for _qwen3_5_update_physical_experts_metadata.

    Differs from Qwen3-Next by iterating ``language_model.model.layers`` and
    matching ``Qwen3_5DecoderLayer``.
    """

    def _patch_classes(self):
        return (
            patch(f"{_QWEN35_MODULE}.Qwen3_5DecoderLayer", _FakeDecoderLayer),
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextSparseMoeBlock", _FakeSparseMoeBlock),
        )

    def _self_with_layers(self, layers):
        return SimpleNamespace(
            num_local_physical_experts=4,
            num_logical_experts=8,
            language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        )

    def test_updates_self_and_moe_via_language_model_path(self):
        moe = _FakeSparseMoeBlock()
        layer = _FakeDecoderLayer(mlp=moe)
        fake_self = self._self_with_layers([layer])
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen3_5_update_physical_experts_metadata(
                fake_self, num_physical_experts=24, num_local_physical_experts=4
            )
        self.assertEqual(fake_self.num_physical_experts, 24)
        self.assertEqual(fake_self.num_redundant_experts, 16)  # 24 - 8
        self.assertEqual(moe.n_physical_experts, 24)
        moe.experts.update_expert_map.assert_called_once()

    def test_skips_non_qwen35_decoder_layers(self):
        moe = _FakeSparseMoeBlock()
        good = _FakeDecoderLayer(mlp=moe)
        # A bare object is not a Qwen3_5DecoderLayer instance -> skipped
        fake_self = self._self_with_layers([object(), good])
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen3_5_update_physical_experts_metadata(
                fake_self, num_physical_experts=16, num_local_physical_experts=4
            )
        moe.experts.update_expert_map.assert_called_once()

    def test_asserts_when_local_physical_experts_mismatch(self):
        fake_self = self._self_with_layers([])
        p1, p2 = self._patch_classes()
        with p1, p2:
            with self.assertRaises(AssertionError):
                _qwen3_5_update_physical_experts_metadata(
                    fake_self,
                    num_physical_experts=16,
                    num_local_physical_experts=1,
                )


@needs_patches
class TestQwen35MoeSetParameters(TestBase):
    """Tests for _qwen3_5_set_moe_parameters."""

    def _patch_classes(self):
        return (
            patch(f"{_QWEN35_MODULE}.Qwen3_5DecoderLayer", _FakeDecoderLayer),
            patch(f"{_QWEN_NEXT_MODULE}.Qwen3NextSparseMoeBlock", _FakeSparseMoeBlock),
        )

    def _self_with_layers(self, layers):
        return SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )

    def test_collects_via_language_model_path(self):
        moe = _FakeSparseMoeBlock(n_logical_experts=10, n_routed_experts=10)
        layer = _FakeDecoderLayer(mlp=moe)
        fake_self = self._self_with_layers([layer])
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen3_5_set_moe_parameters(fake_self)
        self.assertEqual(fake_self.moe_layers, [moe.experts])
        self.assertEqual(fake_self.num_moe_layers, 1)
        self.assertEqual(fake_self.num_logical_experts, 10)
        self.assertEqual(fake_self.num_expert_groups, 1)

    def test_none_found_zeros_counters(self):
        fake_self = self._self_with_layers([object(), _FakeDecoderLayer(mlp=object())])
        p1, p2 = self._patch_classes()
        with p1, p2:
            _qwen3_5_set_moe_parameters(fake_self)
        self.assertEqual(fake_self.num_moe_layers, 0)
        self.assertEqual(fake_self.num_logical_experts, 0)
        self.assertEqual(fake_self.num_physical_experts, 0)


if __name__ == "__main__":
    unittest.main()
