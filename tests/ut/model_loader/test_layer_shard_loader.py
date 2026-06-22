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

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.model_loader.layer_shard_loader import (
    EdgeCloudLayerPlan,
    LayerShardLoader,
)


# ---------------------------------------------------------------------------
# Minimal nn.Module stand-ins that expose the same surface the sharder relies
# on (model.layers / embed_tokens / norm / lm_head) without pulling in real
# vLLM model classes.
# ---------------------------------------------------------------------------


class _FakeLayer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        return args[0] if args else None


class _FakeTransformerModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(i) for i in range(num_layers)])
        self.embed_tokens = nn.Linear(1, 1)
        self.norm = nn.Linear(1, 1)


class _FakeLanguageModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.model = _FakeTransformerModel(num_layers)
        self.lm_head = nn.Linear(1, 1)


class _FakeCausalLM(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.model = _FakeTransformerModel(num_layers)
        self.lm_head = nn.Linear(1, 1)


class _FakeVLModel(nn.Module):
    """Model that exposes language_model.model.layers (multimodal shape)."""

    def __init__(self, num_layers):
        super().__init__()
        self.language_model = _FakeLanguageModel(num_layers)
        self.vision_tower = nn.Linear(1, 1)
        self.multi_modal_projector = nn.Linear(1, 1)


class TestEdgeCloudLayerPlan(TestBase):
    """Tests for EdgeCloudLayerPlan construction and layer-set computation."""

    # ---- construction: k parsing ----

    def test_int_k_is_symmetric(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=2)
        self.assertEqual(plan.head_k, 2)
        self.assertEqual(plan.tail_k, 2)

    def test_list_k_is_asymmetric(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=[1, 3])
        self.assertEqual(plan.head_k, 1)
        self.assertEqual(plan.tail_k, 3)

    def test_tuple_k_is_asymmetric(self):
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=10, k=(4, 2))
        self.assertEqual(plan.head_k, 4)
        self.assertEqual(plan.tail_k, 2)

    def test_default_mode_is_head_tail(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=1)
        self.assertEqual(plan.mode, "head_tail")

    # ---- get_local_layers ----

    def test_local_layers_edge_head_tail(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=[2, 3])
        self.assertEqual(plan.get_local_layers(), {0, 1, 7, 8, 9})

    def test_local_layers_cloud_head_tail(self):
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=10, k=[2, 3])
        self.assertEqual(plan.get_local_layers(), {2, 3, 4, 5, 6})

    def test_local_layers_embedding_only_edge_empty(self):
        plan = EdgeCloudLayerPlan(
            role="edge", total_layers=10, k=0, mode="embedding_only"
        )
        self.assertEqual(plan.get_local_layers(), set())

    def test_local_layers_embedding_only_cloud_all(self):
        plan = EdgeCloudLayerPlan(
            role="cloud", total_layers=10, k=0, mode="embedding_only"
        )
        self.assertEqual(plan.get_local_layers(), set(range(10)))

    def test_local_layers_edge_and_cloud_partition_all_layers(self):
        edge_plan = EdgeCloudLayerPlan(role="edge", total_layers=8, k=[1, 2])
        cloud_plan = EdgeCloudLayerPlan(role="cloud", total_layers=8, k=[1, 2])
        edge_local = edge_plan.get_local_layers()
        cloud_local = cloud_plan.get_local_layers()
        self.assertEqual(edge_local | cloud_local, set(range(8)))
        self.assertEqual(edge_local & cloud_local, set())

    # ---- get_released_layers ----

    def test_released_layers_edge(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=[2, 3])
        self.assertEqual(plan.get_released_layers(), {2, 3, 4, 5, 6})

    def test_released_layers_cloud(self):
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=10, k=[2, 3])
        self.assertEqual(plan.get_released_layers(), {0, 1, 7, 8, 9})

    def test_released_layers_embedding_only_edge(self):
        plan = EdgeCloudLayerPlan(
            role="edge", total_layers=6, k=0, mode="embedding_only"
        )
        self.assertEqual(plan.get_released_layers(), set(range(6)))

    # ---- validation failures ----

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="fog", total_layers=10, k=2)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="edge", total_layers=10, k=2, mode="split")

    def test_head_tail_requires_positive_head(self):
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="edge", total_layers=10, k=0)

    def test_head_tail_requires_positive_tail(self):
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="edge", total_layers=10, k=[2, 0])

    def test_head_tail_must_leave_cloud_layer(self):
        # head_k + tail_k >= total_layers leaves nothing for cloud
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="edge", total_layers=4, k=[2, 2])

    def test_head_tail_must_leave_cloud_layer_exact(self):
        with self.assertRaises(ValueError):
            EdgeCloudLayerPlan(role="edge", total_layers=4, k=2)

    def test_float_k_is_coerced_to_int(self):
        plan = EdgeCloudLayerPlan(role="edge", total_layers=10, k=2.0)
        self.assertEqual(plan.head_k, 2)
        self.assertIsInstance(plan.head_k, int)
        self.assertIsInstance(plan.tail_k, int)

    def test_embedding_only_accepts_any_k_value(self):
        # In embedding_only mode head_k/tail_k are forced to 0 regardless of k,
        # so a zero / "normally invalid" k must not raise here.
        plan = EdgeCloudLayerPlan(
            role="edge", total_layers=10, k=0, mode="embedding_only"
        )
        self.assertEqual(plan.head_k, 0)
        self.assertEqual(plan.tail_k, 0)

    def test_edge_and_cloud_released_sets_are_complementary(self):
        edge_plan = EdgeCloudLayerPlan(role="edge", total_layers=8, k=[1, 2])
        cloud_plan = EdgeCloudLayerPlan(role="cloud", total_layers=8, k=[1, 2])
        edge_released = edge_plan.get_released_layers()
        cloud_released = cloud_plan.get_released_layers()
        self.assertEqual(edge_released, cloud_plan.get_local_layers())
        self.assertEqual(cloud_released, edge_plan.get_local_layers())

    def test_cloud_embedding_only_releases_nothing(self):
        plan = EdgeCloudLayerPlan(
            role="cloud", total_layers=5, k=0, mode="embedding_only"
        )
        self.assertEqual(plan.get_released_layers(), set())


class TestLayerShardLoaderTraversal(unittest.TestCase):
    """Tests for _get_language_model / _get_transformer_model traversal."""

    def test_get_language_model_from_causal_lm(self):
        model = _FakeCausalLM(num_layers=4)
        lm = LayerShardLoader._get_language_model(model)
        self.assertIs(lm, model)

    def test_get_language_model_from_vl_model(self):
        model = _FakeVLModel(num_layers=4)
        lm = LayerShardLoader._get_language_model(model)
        self.assertIs(lm, model.language_model)

    def test_get_transformer_model_from_causal_lm(self):
        model = _FakeCausalLM(num_layers=4)
        transformer = LayerShardLoader._get_transformer_model(model)
        self.assertIs(transformer, model.model)

    def test_get_language_model_raises_on_unsupported_shape(self):
        bad = nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            LayerShardLoader._get_language_model(bad)


class TestLayerShardLoaderApplySharding(unittest.TestCase):
    """Tests for apply_sharding edge/cloud behavior and module replacement."""

    def test_edge_keeps_head_and_tail_replaces_middle(self):
        model = _FakeCausalLM(num_layers=6)
        plan = EdgeCloudLayerPlan(role="edge", total_layers=6, k=[1, 2])
        LayerShardLoader.apply_sharding(model, plan)

        layers = model.model.layers
        # local = {0, 4, 5}; replaced = {1, 2, 3}
        self.assertNotIsInstance(layers[0], PPMissingLayer)
        self.assertIsInstance(layers[1], PPMissingLayer)
        self.assertIsInstance(layers[2], PPMissingLayer)
        self.assertIsInstance(layers[3], PPMissingLayer)
        self.assertNotIsInstance(layers[4], PPMissingLayer)
        self.assertNotIsInstance(layers[5], PPMissingLayer)
        # edge keeps embed_tokens / norm / lm_head
        self.assertNotIsInstance(model.model.embed_tokens, PPMissingLayer)
        self.assertNotIsInstance(model.model.norm, PPMissingLayer)
        self.assertNotIsInstance(model.lm_head, PPMissingLayer)

    def test_cloud_keeps_middle_replaces_head_tail_and_heads(self):
        model = _FakeCausalLM(num_layers=6)
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=6, k=[1, 2])
        LayerShardLoader.apply_sharding(model, plan)

        layers = model.model.layers
        # local = {1, 2, 3}; replaced = {0, 4, 5}
        self.assertIsInstance(layers[0], PPMissingLayer)
        self.assertNotIsInstance(layers[1], PPMissingLayer)
        self.assertNotIsInstance(layers[2], PPMissingLayer)
        self.assertNotIsInstance(layers[3], PPMissingLayer)
        self.assertIsInstance(layers[4], PPMissingLayer)
        self.assertIsInstance(layers[5], PPMissingLayer)
        # cloud also replaces embed_tokens / norm / lm_head
        self.assertIsInstance(model.model.embed_tokens, PPMissingLayer)
        self.assertIsInstance(model.model.norm, PPMissingLayer)
        self.assertIsInstance(model.lm_head, PPMissingLayer)

    def test_cloud_replaces_vision_components_on_vl_model(self):
        model = _FakeVLModel(num_layers=6)
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=6, k=[1, 1])
        LayerShardLoader.apply_sharding(model, plan)
        self.assertIsInstance(model.vision_tower, PPMissingLayer)
        self.assertIsInstance(model.multi_modal_projector, PPMissingLayer)

    def test_edge_does_not_touch_vision_components(self):
        model = _FakeVLModel(num_layers=6)
        plan = EdgeCloudLayerPlan(role="edge", total_layers=6, k=[1, 1])
        LayerShardLoader.apply_sharding(model, plan)
        self.assertNotIsInstance(model.vision_tower, PPMissingLayer)
        self.assertNotIsInstance(model.multi_modal_projector, PPMissingLayer)

    def test_apply_sharding_skips_already_missing_layers(self):
        model = _FakeCausalLM(num_layers=4)
        # Pre-mark one middle layer as PPMissingLayer
        model.model.layers[1] = PPMissingLayer()
        plan = EdgeCloudLayerPlan(role="edge", total_layers=4, k=[1, 1])
        # Should not raise when re-encountering an already-missing layer
        LayerShardLoader.apply_sharding(model, plan)
        self.assertIsInstance(model.model.layers[1], PPMissingLayer)


class TestLayerShardLoaderValidateSharding(unittest.TestCase):
    """Tests for validate_sharding success and failure."""

    def test_validate_passes_when_consistent(self):
        model = _FakeCausalLM(num_layers=6)
        plan = EdgeCloudLayerPlan(role="edge", total_layers=6, k=[1, 2])
        LayerShardLoader.apply_sharding(model, plan)
        # Should not raise
        LayerShardLoader.validate_sharding(model, plan)

    def test_validate_raises_on_unexpected_real_layer(self):
        model = _FakeCausalLM(num_layers=6)
        plan = EdgeCloudLayerPlan(role="cloud", total_layers=6, k=[1, 2])
        # Cloud local = {1,2,3}; manually mark layer 1 as missing -> conflict
        model.model.layers[1] = PPMissingLayer()
        with self.assertRaises(RuntimeError):
            LayerShardLoader.validate_sharding(model, plan)

    def test_validate_raises_on_unexpected_missing_layer(self):
        model = _FakeCausalLM(num_layers=6)
        plan = EdgeCloudLayerPlan(role="edge", total_layers=6, k=[1, 2])
        # Edge local = {0,4,5}; manually mark layer 0 as missing -> conflict
        model.model.layers[0] = PPMissingLayer()
        with self.assertRaises(RuntimeError):
            LayerShardLoader.validate_sharding(model, plan)


class TestLayerShardLoaderEmbeddingOnly(unittest.TestCase):
    """apply_sharding behavior for embedding_only plans (edge releases all)."""

    def test_embedding_only_edge_replaces_all_layers(self):
        model = _FakeCausalLM(num_layers=5)
        plan = EdgeCloudLayerPlan(
            role="edge", total_layers=5, k=0, mode="embedding_only"
        )
        LayerShardLoader.apply_sharding(model, plan)
        for layer in model.model.layers:
            self.assertIsInstance(layer, PPMissingLayer)
        # edge keeps embed_tokens / norm / lm_head even in embedding_only
        self.assertNotIsInstance(model.model.embed_tokens, PPMissingLayer)
        self.assertNotIsInstance(model.model.norm, PPMissingLayer)
        self.assertNotIsInstance(model.lm_head, PPMissingLayer)

    def test_embedding_only_cloud_keeps_all_layers(self):
        model = _FakeCausalLM(num_layers=5)
        plan = EdgeCloudLayerPlan(
            role="cloud", total_layers=5, k=0, mode="embedding_only"
        )
        LayerShardLoader.apply_sharding(model, plan)
        for layer in model.model.layers:
            self.assertNotIsInstance(layer, PPMissingLayer)
        # cloud still replaces embed_tokens / norm / lm_head
        self.assertIsInstance(model.model.embed_tokens, PPMissingLayer)
        self.assertIsInstance(model.model.norm, PPMissingLayer)
        self.assertIsInstance(model.lm_head, PPMissingLayer)


class TestLayerShardLoaderApplyWithCompilationConfig(unittest.TestCase):
    """apply_sharding integration with compilation-config cleanup."""

    def _make_compilation_config(self, context, all_moe=None):
        cfg = MagicMock()
        cfg.static_forward_context = dict(context)
        if all_moe is not None:
            cfg.static_all_moe_layers = list(all_moe)
        return cfg

    def test_apply_sharding_cleans_stale_context_entries(self):
        model = _FakeCausalLM(num_layers=4)
        live = model.model.layers[0]
        stale = _FakeLayer(99)
        comp_cfg = self._make_compilation_config(
            {"live": live, "stale": stale}, all_moe=["live", "stale"]
        )
        plan = EdgeCloudLayerPlan(role="edge", total_layers=4, k=[1, 1])
        LayerShardLoader.apply_sharding(model, plan, comp_cfg)
        self.assertIn("live", comp_cfg.static_forward_context)
        self.assertNotIn("stale", comp_cfg.static_forward_context)

    def test_apply_sharding_without_compilation_config_skips_cleanup(self):
        model = _FakeCausalLM(num_layers=4)
        plan = EdgeCloudLayerPlan(role="edge", total_layers=4, k=[1, 1])
        # Passing compilation_config=None must not raise and still validate.
        LayerShardLoader.apply_sharding(model, plan, None)
        self.assertIsInstance(model.model.layers[1], PPMissingLayer)


class TestLayerShardLoaderCleanCompilationConfig(unittest.TestCase):
    """Tests for _clean_compilation_config stale-entry removal."""

    def _make_compilation_config(self, context, all_moe=None):
        cfg = MagicMock()
        cfg.static_forward_context = dict(context)
        if all_moe is not None:
            cfg.static_all_moe_layers = list(all_moe)
        return cfg

    def test_removes_entries_not_in_model(self):
        model = _FakeCausalLM(num_layers=4)
        keep_module = model.model.layers[0]
        stale_module = _FakeLayer(99)
        cfg = self._make_compilation_config(
            {"keep": keep_module, "stale": stale_module},
            all_moe=["keep", "stale"],
        )
        LayerShardLoader._clean_compilation_config(model, cfg)
        self.assertIn("keep", cfg.static_forward_context)
        self.assertNotIn("stale", cfg.static_forward_context)
        self.assertEqual(cfg.static_all_moe_layers, ["keep"])

    def test_keeps_entries_when_all_present(self):
        model = _FakeCausalLM(num_layers=4)
        live = model.model.layers[0]
        cfg = self._make_compilation_config({"live": live}, all_moe=["live"])
        LayerShardLoader._clean_compilation_config(model, cfg)
        self.assertEqual(set(cfg.static_forward_context.keys()), {"live"})
        self.assertEqual(cfg.static_all_moe_layers, ["live"])

    def test_skips_all_moe_cleanup_when_attr_missing(self):
        # A compilation config that does not expose static_all_moe_layers must
        # not raise (the hasattr branch in _clean_compilation_config guards it).
        class _CompCfg:
            def __init__(self, ctx):
                self.static_forward_context = dict(ctx)

        model = _FakeCausalLM(num_layers=4)
        live = model.model.layers[0]
        stale = _FakeLayer(99)
        cfg = _CompCfg({"live": live, "stale": stale})
        self.assertFalse(hasattr(cfg, "static_all_moe_layers"))
        LayerShardLoader._clean_compilation_config(model, cfg)
        self.assertNotIn("stale", cfg.static_forward_context)
        self.assertIn("live", cfg.static_forward_context)

    def test_no_stale_entries_leaves_context_untouched(self):
        model = _FakeCausalLM(num_layers=4)
        live = model.model.layers[0]
        cfg = self._make_compilation_config({"live": live}, all_moe=["live"])
        LayerShardLoader._clean_compilation_config(model, cfg)
        self.assertEqual(set(cfg.static_forward_context.keys()), {"live"})


class TestLayerShardLoaderReleaseWeights(unittest.TestCase):
    """Tests for release_layer_weights emptying params and buffers."""

    def test_releases_parameters_and_buffers(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.ones(4))
        layer.register_buffer("buf", torch.ones(4))
        LayerShardLoader.release_layer_weights(layer)
        self.assertEqual(layer.weight.data.numel(), 0)
        self.assertEqual(layer.buf.data.numel(), 0)

    def test_safe_on_parameterless_module(self):
        empty = nn.Module()
        # Should not raise
        LayerShardLoader.release_layer_weights(empty)


if __name__ == "__main__":
    unittest.main()
