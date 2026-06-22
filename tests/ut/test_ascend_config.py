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

import json
import os
from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (
    AscendConfig,
    EdgeCloudConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
)


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_with_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": False,
            },
            "multistream_overlap_shared_expert": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertFalse(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
            "refresh": True,
        }
        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_with_dump_config_materializes_fixed_file(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        dump_config = {"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}
        test_vllm_config.additional_config = {"dump_config": dump_config}

        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config.dump_config_path)
        assert ascend_config.dump_config_path is not None
        expected_path = os.path.join(os.getcwd(), ".vllm_ascend", "msprobe", "msprobe_dump_config.json")
        self.assertEqual(ascend_config.dump_config_path, expected_path)
        self.assertTrue(os.path.exists(ascend_config.dump_config_path))
        with open(ascend_config.dump_config_path, encoding="utf-8") as file:
            persisted = json.load(file)
        self.assertEqual(persisted, dump_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_dump_config_and_path_conflict(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config_path": "/tmp/config.json", "dump_config": {"task": "tensor"}}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_dump_config_type_validation(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config": "/tmp/config.json"}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)


class TestEdgeCloudConfig(TestBase):
    """Tests for EdgeCloudConfig parsing, validation and head_tail_k."""

    # ---- defaults & disabled behavior ----

    def test_defaults_when_not_provided(self):
        cfg = EdgeCloudConfig({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.role, "edge")
        self.assertEqual(cfg.mode, "head_tail")
        self.assertEqual(cfg.edge_head_tail_layers, 1)
        self.assertFalse(cfg.enable_decode_graph)
        self.assertEqual(cfg.decode_graph_min_tokens, 1)
        self.assertEqual(cfg.transfer_config, {})
        self.assertEqual(cfg.hidden_dtype, "bf16")

    def test_disabled_skips_validation(self):
        # role/mode are invalid but enabled=False so _validate is never called
        cfg = EdgeCloudConfig({"enabled": False, "role": "weird", "mode": "nope"})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.role, "weird")

    # ---- head_tail_k property ----

    def test_head_tail_k_symmetric_from_int(self):
        cfg = EdgeCloudConfig({"enabled": True, "edge_head_tail_layers": 3})
        self.assertEqual(cfg.head_tail_k, (3, 3))

    def test_head_tail_k_asymmetric_from_list(self):
        cfg = EdgeCloudConfig({"enabled": True, "edge_head_tail_layers": [2, 5]})
        self.assertEqual(cfg.head_tail_k, (2, 5))

    def test_head_tail_k_asymmetric_from_tuple(self):
        cfg = EdgeCloudConfig({"enabled": True, "edge_head_tail_layers": (4, 1)})
        self.assertEqual(cfg.head_tail_k, (4, 1))

    def test_head_tail_k_embedding_only_is_zero_zero(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "mode": "embedding_only", "edge_head_tail_layers": 3}
        )
        self.assertEqual(cfg.head_tail_k, (0, 0))

    # ---- happy-path validated configs ----

    def test_enabled_edge_head_tail(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "role": "edge", "mode": "head_tail", "edge_head_tail_layers": 2}
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.role, "edge")
        self.assertEqual(cfg.head_tail_k, (2, 2))

    def test_enabled_cloud_head_tail_asymmetric(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "role": "cloud", "edge_head_tail_layers": [1, 3]}
        )
        self.assertEqual(cfg.role, "cloud")
        self.assertEqual(cfg.head_tail_k, (1, 3))

    def test_enabled_embedding_only_forces_layers_to_zero(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "mode": "embedding_only", "edge_head_tail_layers": 7}
        )
        # embedding_only forces edge_head_tail_layers back to 0 with a warning
        self.assertEqual(cfg.edge_head_tail_layers, 0)
        self.assertEqual(cfg.head_tail_k, (0, 0))

    def test_custom_optional_fields(self):
        cfg = EdgeCloudConfig(
            {
                "enabled": True,
                "enable_decode_graph": True,
                "decode_graph_min_tokens": 64,
                "transfer_config": {"dtype": "fp16"},
                "hidden_dtype": "fp16",
            }
        )
        self.assertTrue(cfg.enable_decode_graph)
        self.assertEqual(cfg.decode_graph_min_tokens, 64)
        self.assertEqual(cfg.transfer_config, {"dtype": "fp16"})
        self.assertEqual(cfg.hidden_dtype, "fp16")

    # ---- validation failures ----

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudConfig({"enabled": True, "role": "fog"})

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudConfig({"enabled": True, "mode": "split"})

    def test_head_tail_mode_requires_positive_head_k(self):
        with self.assertRaises(ValueError):
            EdgeCloudConfig({"enabled": True, "mode": "head_tail", "edge_head_tail_layers": 0})

    def test_negative_head_k_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudConfig(
                {"enabled": True, "edge_head_tail_layers": [-1, 2]}
            )

    def test_negative_tail_k_raises(self):
        with self.assertRaises(ValueError):
            EdgeCloudConfig(
                {"enabled": True, "edge_head_tail_layers": [2, -3]}
            )

    # ---- repr ----

    def test_repr_contains_key_fields(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "role": "cloud", "edge_head_tail_layers": [1, 2],
             "enable_decode_graph": True}
        )
        text = repr(cfg)
        self.assertIn("EdgeCloudConfig", text)
        self.assertIn("enabled=True", text)
        self.assertIn("role=cloud", text)
        self.assertIn("mode=head_tail", text)
        self.assertIn("enable_decode_graph=True", text)

    def test_repr_default_config_disabled(self):
        cfg = EdgeCloudConfig({})
        text = repr(cfg)
        self.assertIn("enabled=False", text)
        self.assertIn("role=edge", text)
        self.assertIn("mode=head_tail", text)
        self.assertIn("enable_decode_graph=False", text)

    # ---- misc behavior ----

    def test_head_tail_k_computable_even_when_disabled(self):
        # head_tail_k is a pure property; it works regardless of `enabled`.
        cfg = EdgeCloudConfig({"enabled": False, "edge_head_tail_layers": 4})
        self.assertEqual(cfg.head_tail_k, (4, 4))

    def test_enabled_does_not_require_edge_head_tail_layers_field(self):
        # Defaults to 1 when omitted, still valid for head_tail mode.
        cfg = EdgeCloudConfig({"enabled": True})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.head_tail_k, (1, 1))

    def test_embedding_only_with_zero_layers_does_not_warn(self):
        # When mode==embedding_only and edge_head_tail_layers already 0,
        # the warning/force-to-zero path is a no-op (still valid config).
        cfg = EdgeCloudConfig(
            {"enabled": True, "mode": "embedding_only", "edge_head_tail_layers": 0}
        )
        self.assertEqual(cfg.edge_head_tail_layers, 0)
        self.assertEqual(cfg.head_tail_k, (0, 0))

    def test_role_normalization_edge_and_cloud_both_valid(self):
        for role in ("edge", "cloud"):
            cfg = EdgeCloudConfig({"enabled": True, "role": role})
            self.assertEqual(cfg.role, role)

    def test_mode_normalization_both_valid(self):
        for mode in ("head_tail", "embedding_only"):
            cfg = EdgeCloudConfig(
                {"enabled": True, "mode": mode, "edge_head_tail_layers": 0}
                if mode == "embedding_only"
                else {"enabled": True, "mode": mode}
            )
            self.assertEqual(cfg.mode, mode)

    def test_hidden_dtype_and_transfer_config_passthrough(self):
        cfg = EdgeCloudConfig(
            {"enabled": True, "hidden_dtype": "fp32",
             "transfer_config": {"bucket_size_mb": 8}}
        )
        self.assertEqual(cfg.hidden_dtype, "fp32")
        self.assertEqual(cfg.transfer_config, {"bucket_size_mb": 8})


class TestAscendConfigMixPlacement(TestBase):
    """Tests for AscendConfig._check_mix_placement edge-cloud guard."""

    @staticmethod
    def _make_bare_config(**attrs):
        cfg = AscendConfig.__new__(AscendConfig)
        cfg.mix_placement = attrs.get("mix_placement", False)
        cfg.enable_shared_expert_dp = attrs.get("enable_shared_expert_dp", False)
        cfg.multistream_overlap_shared_expert = attrs.get(
            "multistream_overlap_shared_expert", False
        )
        return cfg

    def test_no_raise_when_mix_placement_disabled(self):
        cfg = self._make_bare_config(mix_placement=False,
                                     enable_shared_expert_dp=True,
                                     multistream_overlap_shared_expert=True)
        cfg._check_mix_placement()

    def test_no_raise_when_mix_placement_enabled_no_conflicts(self):
        cfg = self._make_bare_config(mix_placement=True,
                                     enable_shared_expert_dp=False,
                                     multistream_overlap_shared_expert=False)
        cfg._check_mix_placement()

    def test_raises_with_shared_expert_dp(self):
        cfg = self._make_bare_config(mix_placement=True,
                                     enable_shared_expert_dp=True)
        with self.assertRaises(ValueError):
            cfg._check_mix_placement()

    def test_raises_with_multistream_overlap(self):
        cfg = self._make_bare_config(mix_placement=True,
                                     multistream_overlap_shared_expert=True)
        with self.assertRaises(ValueError):
            cfg._check_mix_placement()
