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

"""Edge-cloud collaborative inference patch for the DeepSeek-V4 MTP draft.

Split scheme (mirrors the Qwen3.5 MTP edge-cloud draft):
  - Edge side (first segment): ``embed_tokens`` + per-layer ``enorm`` /
    ``hnorm`` / ``e_proj`` / ``h_proj``, i.e. everything before the decoder
    block in ``DeepSeekMultiTokenPredictorLayer.forward``.
  - Cloud side (middle segment): the decoder block (``mtp_block``) only.
  - Edge side (last segment): identity passthrough — ``hc_head`` and the
    shared head are applied by ``DeepSeekV4MTP.compute_logits`` on the edge
    when sampling draft tokens.

Key differences from the Qwen3.5 MTP variant:
  - The projection/norm submodules live *inside* each MTP layer instead of
    at the predictor level, so ``_setup_edge_cloud_draft`` shards V4 MTP at
    submodule granularity (edge drops ``mtp_block``, cloud drops the rest).
  - Hidden states are hc-expanded: the tensors crossing the edge-cloud
    network have shape ``[num_tokens, hc_mult, hidden_size]``.
  - No residual crosses the network.  The decoder block manages its
    residual internally via ``hc_pre`` / ``hc_post`` and the non-edge-cloud
    forward discards the block's residual output, so the cloud segment
    calls the block with ``residual=None`` and only returns hidden_states.

Note: DeepSeek-V4 ships a single MTP layer (``num_nextn_predict_layers ==
1``), so ``spec_step_idx`` always selects layer "0" in practice.  The
modulo selection is kept to mirror ``DeepSeekMultiTokenPredictor.forward``.
"""

from typing import Any

import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV4MoE,
)
from vllm_ascend.models.deepseek_v4_mtp import (
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    DeepSeekV4MTP,
)


def _forward_edge_cloud_segment_deepseek_v4_mtp(
    self: DeepSeekMultiTokenPredictor,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Edge-cloud segment forward for the DeepSeek-V4 MTP predictor.

    ``start_layer``/``end_layer`` are kept in the signature for
    compatibility with ``EdgeCloudSegment`` but the actual split is driven
    by ``is_first_segment``/``is_last_segment``: all MTP decoder blocks run
    on the cloud, while the edge handles embed+projections (first segment)
    and the sampling head via ``compute_logits`` (last segment).
    """
    num_layers = len(self.layers)
    if is_first_segment is None:
        is_first_segment = start_layer == 0
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers

    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[str(current_step_idx)]

    if is_first_segment:
        # Edge: embed + norms + projections, i.e. the pre-block part of
        # DeepSeekMultiTokenPredictorLayer.forward.
        if inputs_embeds is None:
            assert input_ids is not None, (
                "input_ids is None in DeepSeek-V4 MTP edge-cloud first "
                "segment; either input_ids or inputs_embeds must be provided."
            )
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states is not None, (
            "hidden_states (target pre-hc_head residual stream) is required "
            "in the DeepSeek-V4 MTP edge-cloud first segment."
        )
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0, inputs_embeds
        )
        inputs_embeds = mtp_layer.enorm(inputs_embeds)
        previous_hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        previous_hidden_states = mtp_layer.hnorm(previous_hidden_states)
        hidden_states = mtp_layer.e_proj(inputs_embeds).unsqueeze(
            -2
        ) + mtp_layer.h_proj(previous_hidden_states)
        return IntermediateTensors({"hidden_states": hidden_states})

    assert intermediate_tensors is not None, (
        "intermediate_tensors is None in DeepSeek-V4 MTP edge-cloud segment; "
        "check that all TP ranks receive tensors correctly."
    )

    if is_last_segment:
        # Edge tail: hc_head + shared head are applied by compute_logits()
        # on the edge side, so the tail segment simply returns the cloud's
        # decoder output.
        return intermediate_tensors["hidden_states"]

    # Cloud segment: run only the decoder block selected by spec_step_idx.
    hidden_states = intermediate_tensors["hidden_states"]
    hidden_states, _ = mtp_layer.mtp_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )
    return IntermediateTensors({"hidden_states": hidden_states})


def _deepseek_v4_mtp_forward_edge_cloud_segment(
    self: DeepSeekV4MTP,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Wrapper that delegates to the predictor's forward_edge_cloud_segment."""
    hidden_states = extra_layer_kwargs.pop("hidden_states", None)
    spec_step_idx = extra_layer_kwargs.pop("spec_step_idx", 0)
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors,
        inputs_embeds,
        spec_step_idx,
        **extra_layer_kwargs,
    )


DeepSeekMultiTokenPredictor.forward_edge_cloud_segment = (
    _forward_edge_cloud_segment_deepseek_v4_mtp
)
DeepSeekV4MTP.forward_edge_cloud_segment = (
    _deepseek_v4_mtp_forward_edge_cloud_segment
)


def _deepseek_v4_mtp_make_empty_intermediate_tensors(
    self: DeepSeekV4MTP,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    """Allocate persistent intermediate buffers for edge-cloud draft sync.

    The SupportsPP default produces ``[batch_size, hidden_size]`` buffers
    for hidden_states/residual, but the V4 MTP draft transmits hc-expanded
    hidden states of shape ``[batch_size, hc_mult, hidden_size]`` (and no
    residual), so ``_sync_edge_cloud_draft_intermediate_tensors`` would
    fail on the shape mismatch.
    """
    config = self.config
    hc_mult = getattr(config, "hc_mult", 1)
    return IntermediateTensors(
        {
            "hidden_states": torch.zeros(
                (batch_size, hc_mult, config.hidden_size),
                dtype=dtype,
                device=device,
            ),
        }
    )


DeepSeekV4MTP.make_empty_intermediate_tensors = (
    _deepseek_v4_mtp_make_empty_intermediate_tensors
)


def _deepseek_v4_mtp_set_moe_parameters(self: DeepSeekV4MTP) -> None:
    """PPMissingLayer-tolerant variant of DeepSeekV4MTP.set_moe_parameters.

    In edge-cloud mode the draft is sharded after loading: the edge side
    replaces each layer's ``mtp_block`` with a PPMissingLayer, so the
    original implementation's ``assert isinstance(layer.mtp_block,
    DeepseekV2DecoderLayer)`` would fail there.  Layers without a real
    decoder block simply contribute no MoE parameters on that side.
    """
    self.expert_weights = []
    self.num_expert_groups = getattr(self.config, "n_group", 1)

    self.moe_layers = []
    self.moe_mlp_layers = []
    example_moe = None
    for layer in self.model.layers.values():
        if isinstance(layer, PPMissingLayer):
            continue
        assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
        if isinstance(layer.mtp_block, PPMissingLayer):
            # Edge side: the decoder block runs on the cloud.
            continue
        block = layer.mtp_block
        assert isinstance(block, DeepseekV2DecoderLayer)
        if isinstance(block.mlp, DeepseekV4MoE):
            # Pick last one layer since the first ones may be dense layers.
            example_moe = block.mlp
            self.moe_mlp_layers.append(block.mlp)
            self.moe_layers.append(block.mlp.experts)
    self.extract_moe_parameters(example_moe)


DeepSeekV4MTP.set_moe_parameters = _deepseek_v4_mtp_set_moe_parameters
