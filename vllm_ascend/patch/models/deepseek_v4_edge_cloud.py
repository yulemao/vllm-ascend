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

"""Edge-cloud collaborative inference patch for DeepSeek-V4.

This module monkey-patches ``forward_edge_cloud_segment`` onto
``DeepseekV4Model`` and ``AscendDeepseekV4ForCausalLM`` so that the edge-cloud
ModelRunner can execute arbitrary layer ranges without modifying upstream vLLM.

Scheme: "head-3 / tail-1"
  - Edge side holds the first 3 layers (segment A) and the last 1 layer
    (segment E).
  - Cloud side holds the middle layers (segment C).
  - All Hash MoE layers (layer_idx < config.num_hash_layers) reside on the
    edge side (segment A), so the cloud side never encounters hash routing.

Key differences from standard Llama-style models:
  - DecoderLayer signature: ``layer(positions, hidden_states, residual,
    llama_4_scaling)`` returns ``(hidden_states, residual)``.
  - Residual is managed internally via ``hc_pre`` / ``hc_post`` and must be
    passed across segments through ``IntermediateTensors``.
  - Embedding needs ``unsqueeze(-2).repeat(1, hc_mult, 1)``.
  - Tail segment needs ``hc_head()`` + ``norm()``.
  - Only ``hidden_states`` and ``residual`` are transmitted across the
    edge-cloud network; ``input_ids`` is kept locally on the edge side.
"""

from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV4Model,
)


def _forward_edge_cloud_segment_v4(
    self: DeepseekV4Model,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Edge-cloud segment forward for DeepSeek-V4.

    Args:
        start_layer: First layer index to execute (inclusive).
        end_layer: Last layer index to execute (exclusive).
        input_ids: Token IDs for embedding (first segment only).
        positions: Position IDs.
        intermediate_tensors: Carries ``hidden_states`` and ``residual`` from
            the previous segment.
        inputs_embeds: Optional pre-computed embeddings.

    Returns:
        ``IntermediateTensors`` for non-last segments, or final
        ``hidden_states`` for the last segment.
    """
    num_layers = len(self.layers)
    assert 0 <= start_layer < end_layer <= num_layers, (
        f"Invalid segment range: [{start_layer}, {end_layer}) "
        f"for {num_layers} layers"
    )

    is_first_segment = (start_layer == 0 and get_pp_group().is_first_rank)
    is_last_segment = (end_layer == num_layers and get_pp_group().is_last_rank)

    # ----- Embedding or restore intermediate state -----
    if is_first_segment:
        # Ensure int64 before embedding lookup (align with standard pipeline)
        if input_ids is not None:
            input_ids = input_ids.to(torch.int64)

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        residual = None
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors required for non-first segment in V4"
        )
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # ----- Execute layers in [start_layer, end_layer) -----
    # llama_4_scaling is currently None because scaling config is not enabled.
    # When enabled, compute it from config (see DeepseekV4Model.forward).
    llama_4_scaling = None
    for idx, layer in enumerate(islice(self.layers, start_layer, end_layer)):
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )

    # ----- Return intermediate state or final hidden_states -----
    # In the "head-3 / tail-1" edge-cloud scheme, all Hash MoE layers reside
    # on the edge side (segment A).  The cloud side (segment C) and the edge
    # tail (segment E) do not need ``input_ids``.  We only pass
    # ``hidden_states`` and ``residual`` across the network.

    if not is_last_segment:
        # residual keeps its original (n, hc_mult, h) shape, aligned with
        # standard forward path and make_empty_intermediate_tensors buffer.
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        })

    # Last segment: hc_head + norm
    hidden_states = self.hc_head(
        hidden_states,
        self.hc_head_fn,
        self.hc_head_scale,
        self.hc_head_base,
    )


    hidden_states = self.norm(hidden_states)
    return hidden_states


def _deepseek_v4_forward_edge_cloud_segment_wrapper(
    self: AscendDeepseekV4ForCausalLM,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Wrapper that delegates to DeepseekV4Model.forward_edge_cloud_segment."""
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


# ── Monkey Patch: runtime binding ──
DeepseekV4Model.forward_edge_cloud_segment = _forward_edge_cloud_segment_v4
AscendDeepseekV4ForCausalLM.forward_edge_cloud_segment = (
    _deepseek_v4_forward_edge_cloud_segment_wrapper
)
