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

"""Unit tests for the EdgeCloudSegment nn.Module wrapper.

EdgeCloudSegment wraps a base model's ``forward_edge_cloud_segment`` into a
standard ``nn.Module`` so that ACLGraphWrapper can capture it the same way it
captures full models. These tests verify attribute storage, submodule
registration, and the exact delegation contract (positional layer range +
inputs, keyword is_first/is_last flags, and extra_layer_kwargs forwarding).
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tests.ut.base import TestBase

try:
    from vllm_ascend.worker.model_runner_v1 import EdgeCloudSegment

    _SEGMENT_AVAILABLE = True
    _SEGMENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    _SEGMENT_AVAILABLE = False
    _SEGMENT_IMPORT_ERROR = exc

needs_segment = unittest.skipUnless(
    _SEGMENT_AVAILABLE,
    f"EdgeCloudSegment not available: {_SEGMENT_IMPORT_ERROR}",
)


class _RecordingModel(nn.Module):
    """Real nn.Module whose forward_edge_cloud_segment records its call."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.call = None

    def forward_edge_cloud_segment(self, *args, **kwargs):
        self.call = (args, kwargs)
        return "segment_result"


@needs_segment
class TestEdgeCloudSegmentInit(TestBase):
    """Tests for EdgeCloudSegment construction and attribute storage."""

    def test_is_nn_module_subclass(self):
        self.assertTrue(issubclass(EdgeCloudSegment, nn.Module))

    def test_init_stores_layer_range(self):
        model = MagicMock()
        seg = EdgeCloudSegment(model, start_layer=3, end_layer=7)
        self.assertEqual(seg._start_layer, 3)
        self.assertEqual(seg._end_layer, 7)

    def test_init_defaults_is_first_last_to_none(self):
        seg = EdgeCloudSegment(MagicMock(), 0, 2)
        self.assertIsNone(seg._is_first_segment)
        self.assertIsNone(seg._is_last_segment)

    def test_init_stores_is_first_last_flags(self):
        seg = EdgeCloudSegment(
            MagicMock(), 0, 2, is_first_segment=True, is_last_segment=False
        )
        self.assertTrue(seg._is_first_segment)
        self.assertFalse(seg._is_last_segment)

    def test_model_registered_as_submodule_when_nn_module(self):
        # A real nn.Module model must be registered as a child module so that
        # torch.npu.graph can statically discover its parameters.
        model = _RecordingModel()
        seg = EdgeCloudSegment(model, 0, 2)
        children = dict(seg.named_children())
        self.assertIn("_edge_model", children)
        self.assertIs(children["_edge_model"], model)
        # Parameter discovery reaches the wrapped model's parameters.
        param_names = {n for n, _ in seg.named_parameters()}
        self.assertTrue(any("weight" in n for n in param_names))


@needs_segment
class TestEdgeCloudSegmentForward(TestBase):
    """Tests for EdgeCloudSegment.forward delegation contract."""

    def _make(self, **init_kwargs):
        model = _RecordingModel()
        seg = EdgeCloudSegment(model, 2, 5, **init_kwargs)
        return seg, model

    def test_returns_model_result_unchanged(self):
        seg, _ = self._make()
        self.assertEqual(seg.forward(), "segment_result")

    def test_delegates_to_forward_edge_cloud_segment(self):
        seg, model = self._make()
        seg.forward()
        self.assertIsNotNone(model.call)

    def test_passes_layer_range_as_first_positional_args(self):
        seg, model = self._make()
        seg.forward()
        args, _ = model.call
        self.assertEqual(args[0], 2)  # start_layer
        self.assertEqual(args[1], 5)  # end_layer

    def test_passes_inputs_positionally(self):
        seg, model = self._make()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        embeds = torch.zeros(2, 4)
        inter = {"hidden_states": torch.zeros(2, 4)}
        seg.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=inter,
            inputs_embeds=embeds,
        )
        args, _ = model.call
        self.assertIs(args[2], input_ids)
        self.assertIs(args[3], positions)
        self.assertIs(args[4], inter)
        self.assertIs(args[5], embeds)

    def test_defaults_optional_inputs_to_none(self):
        seg, model = self._make()
        seg.forward()
        args, _ = model.call
        self.assertIsNone(args[2])  # input_ids
        self.assertIsNone(args[3])  # positions
        self.assertIsNone(args[4])  # intermediate_tensors
        self.assertIsNone(args[5])  # inputs_embeds

    def test_is_first_last_forwarded_as_keywords(self):
        seg, model = self._make(is_first_segment=True, is_last_segment=False)
        seg.forward()
        _, kwargs = model.call
        self.assertTrue(kwargs["is_first_segment"])
        self.assertFalse(kwargs["is_last_segment"])

    def test_none_is_first_last_forwarded_when_unset(self):
        seg, model = self._make()
        seg.forward()
        _, kwargs = model.call
        self.assertIsNone(kwargs["is_first_segment"])
        self.assertIsNone(kwargs["is_last_segment"])

    def test_extra_layer_kwargs_forwarded(self):
        seg, model = self._make()
        seg.forward(extra_mask=torch.tensor([1]), custom_flag=True)
        _, kwargs = model.call
        self.assertIn("extra_mask", kwargs)
        self.assertIn("custom_flag", kwargs)
        self.assertTrue(kwargs["custom_flag"])

    def test_call_dunder_invokes_forward(self):
        # __call__ on nn.Module dispatches to forward().
        seg, model = self._make()
        input_ids = torch.tensor([0])
        result = seg(input_ids)
        self.assertEqual(result, "segment_result")
        args, _ = model.call
        # input_ids is the 3rd positional arg forwarded to the model
        self.assertIs(args[2], input_ids)


if __name__ == "__main__":
    unittest.main()
