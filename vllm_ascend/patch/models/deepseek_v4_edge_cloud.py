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
  - Residual is managed internally via ``hc_pre`` / ``hc_post``.  In the
    edge-cloud variant we do **not** pass residual across segments: each
    segment recomputes its own residual locally from ``hidden_states``.  This
    avoids transmitting the large residual tensor between edge and cloud,
    which was causing the cloud-side KV-cache allocation to OOM.
  - Embedding needs ``unsqueeze(-2).repeat(1, hc_mult, 1)``.
  - Tail segment needs ``hc_head()`` + ``norm()``.
  - Only ``hidden_states`` is transmitted across the edge-cloud network;
    ``input_ids`` is kept locally on the edge side.
"""

from itertools import islice
from typing import Any

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_forward_context import get_forward_context
from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from vllm_ascend.models.deepseek_v4_mtp import (
    DeepSeekMultiTokenPredictor,
    DeepSeekV4MTP,
)


def _forward_edge_cloud_segment_v4(
    self: DeepseekV4Model,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Edge-cloud segment forward for DeepSeek-V4.

    Args:
        start_layer: First layer index to execute (inclusive).
        end_layer: Last layer index to execute (exclusive).
        input_ids: Token IDs for embedding (first segment only).
        positions: Position IDs.
        intermediate_tensors: Carries ``hidden_states`` from the previous
            segment.  Residual is intentionally omitted; each segment
            recomputes it locally.
        inputs_embeds: Optional pre-computed embeddings.

    Returns:
        ``IntermediateTensors`` for non-last segments, or final
        ``hidden_states`` for the last segment.
    """
    num_layers = len(self.layers)
    assert 0 <= start_layer <= end_layer <= num_layers, (
        f"Invalid segment range: [{start_layer}, {end_layer}) "
        f"for {num_layers} layers"
    )

    if is_first_segment is None:
        is_first_segment = start_layer == 0 and get_pp_group().is_first_rank
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers and get_pp_group().is_last_rank

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
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors required for non-first segment in V4"
        )
        hidden_states = intermediate_tensors["hidden_states"]

    # ----- Execute layers in [start_layer, end_layer) -----
    # llama_4_scaling is currently None because scaling config is not enabled.
    # When enabled, compute it from config (see DeepseekV4Model.forward).
    llama_4_scaling = None
    residual = None
    for layer in islice(self.layers, start_layer, end_layer):
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
    # ``hidden_states`` across the network; residual is recreated locally by
    # the next segment's first layer.

    if not is_last_segment:
        return IntermediateTensors({
            "hidden_states": hidden_states,
        })

    # DeepSeek V4 MTP consumes the flattened pre-hc_head stream rather than
    # the final 2-D target hidden states. Keep this aligned with the standard
    # DeepseekV4Model.forward path. With sequence parallelism enabled, gather
    # token shards before refreshing the target buffer used by the proposer.
    forward_ctx = get_forward_context()
    if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
        mtp_hidden_states = tensor_model_parallel_all_gather(
            hidden_states.flatten(1), dim=0
        )
        if forward_ctx.pad_size > 0:
            mtp_hidden_states = mtp_hidden_states[: -forward_ctx.pad_size]
    else:
        mtp_hidden_states = hidden_states.flatten(1)
    num_mtp_tokens = mtp_hidden_states.shape[0]
    self._mtp_hidden_buffer[:num_mtp_tokens].copy_(mtp_hidden_states)

    # Last segment: hc_head + norm
    hidden_states = self.hc_head(
        hidden_states,
        self.hc_head_fn,
        self.hc_head_scale,
        self.hc_head_base,
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states


def _forward_edge_cloud_segment_v4_mtp(
    self: DeepSeekMultiTokenPredictor,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Run one DeepSeek V4 MTP stage in the edge-cloud pipeline.

    V4 keeps its input projections and output head on each MTP layer, unlike
    Qwen MTP where they live on the predictor. The edge therefore prepares
    the 3-D HC stream, the cloud runs only ``mtp_block``, and the edge tail
    returns that stream for the normal ``compute_logits`` implementation.
    """
    del start_layer, end_layer, extra_layer_kwargs

    if is_first_segment is None:
        is_first_segment = False
    if is_last_segment is None:
        is_last_segment = False

    layer_idx = spec_step_idx % self.num_mtp_layers
    layer = self.layers[str(layer_idx)]

    if is_first_segment:
        assert hidden_states is not None, (
            "hidden_states are required by the DeepSeek V4 MTP edge segment"
        )
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)

        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0, inputs_embeds
        )
        inputs_embeds = layer.enorm(inputs_embeds)
        hidden_states = hidden_states.view(
            -1, layer.hc_mult, layer.config.hidden_size
        )
        hidden_states = layer.hnorm(hidden_states)
        hidden_states = layer.e_proj(inputs_embeds).unsqueeze(-2) + layer.h_proj(
            hidden_states
        )
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors are required by a non-first DeepSeek V4 "
            "MTP edge-cloud segment"
        )
        hidden_states = intermediate_tensors["hidden_states"]

    if not is_first_segment and not is_last_segment:
        hidden_states, _ = layer.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )

    if not is_last_segment:
        return IntermediateTensors({"hidden_states": hidden_states})
    return hidden_states


def _deepseek_v4_mtp_forward_edge_cloud_segment(
    self: DeepSeekV4MTP,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
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
        is_first_segment=is_first_segment,
        is_last_segment=is_last_segment,
        **extra_layer_kwargs,
    )


def _deepseek_v4_mtp_make_empty_intermediate_tensors(
    self: DeepSeekV4MTP,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    first_layer = next(iter(self.model.layers.values()))
    return IntermediateTensors(
        {
            "hidden_states": torch.zeros(
                (
                    batch_size,
                    first_layer.hc_mult,
                    first_layer.config.hidden_size,
                ),
                dtype=dtype,
                device=device,
            ),
        }
    )


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
DeepSeekMultiTokenPredictor.forward_edge_cloud_segment = (
    _forward_edge_cloud_segment_v4_mtp
)
DeepSeekV4MTP.forward_edge_cloud_segment = (
    _deepseek_v4_mtp_forward_edge_cloud_segment
)
DeepSeekV4MTP.make_empty_intermediate_tensors = (
    _deepseek_v4_mtp_make_empty_intermediate_tensors
)
