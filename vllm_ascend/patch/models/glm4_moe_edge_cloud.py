#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from itertools import islice
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MixtureOfExperts,
)
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
    Glm4LiteMixtureOfExperts,
)
from vllm.sequence import IntermediateTensors


# ---------------------------------------------------------------------------
# Layer-segment forward for edge-cloud
# ---------------------------------------------------------------------------

def _forward_edge_cloud_segment_glm4_moe(
    self: Glm4MoeModel,
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
    num_layers = len(self.layers)
    assert 0 <= start_layer <= end_layer <= num_layers, (
        f"Invalid segment range [{start_layer}, {end_layer}) for {num_layers} layers"
    )

    if is_first_segment is None:
        is_first_segment = start_layer == 0 and get_pp_group().is_first_rank
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers and get_pp_group().is_last_rank

    if is_first_segment:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors is None in edge-cloud segment; "
            "check that all TP ranks receive tensors correctly."
        )
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Glm4MoeDecoderLayer.forward uses positional args:
    #   forward(positions, hidden_states, residual)
    for layer in islice(self.layers, start_layer, end_layer):
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not is_last_segment:
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def _glm4_moe_lm_forward_edge_cloud_segment(
    self: Glm4MoeForCausalLM,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


def _glm4_moe_lite_lm_forward_edge_cloud_segment(
    self: Glm4MoeLiteForCausalLM,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


# Apply patches to Glm4Moe (standard MoE) classes
Glm4MoeModel.forward_edge_cloud_segment = _forward_edge_cloud_segment_glm4_moe
Glm4MoeForCausalLM.forward_edge_cloud_segment = (
    _glm4_moe_lm_forward_edge_cloud_segment
)

# Apply patches to Glm4MoeLite (Lite MoE) classes
Glm4MoeLiteModel.forward_edge_cloud_segment = _forward_edge_cloud_segment_glm4_moe
Glm4MoeLiteForCausalLM.forward_edge_cloud_segment = (
    _glm4_moe_lite_lm_forward_edge_cloud_segment
)


# ---------------------------------------------------------------------------
# Monkey-patch MoE methods so they tolerate PPMissingLayer in edge-cloud mode
# ---------------------------------------------------------------------------

def _glm4_mixture_update_physical_experts_metadata(
    self,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> None:
    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_local_physical_experts = num_local_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for moe in self.moe_mlp_layers:
        moe.n_local_physical_experts = num_local_physical_experts
        moe.n_physical_experts = num_physical_experts
        moe.n_redundant_experts = self.num_redundant_experts
        moe.experts.update_expert_map()


def _glm4_moe_set_moe_parameters(self) -> None:
    from vllm.model_executor.models.glm4_moe import Glm4MoE, Glm4MoeDecoderLayer
    from vllm.model_executor.models.utils import PPMissingLayer

    self.expert_weights = []
    self.moe_layers = []
    self.moe_mlp_layers = []
    example_moe = None
    for layer in self.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if isinstance(layer, Glm4MoeDecoderLayer) and isinstance(
            layer.mlp, Glm4MoE
        ):
            example_moe = layer.mlp
            self.moe_mlp_layers.append(layer.mlp)
            self.moe_layers.append(layer.mlp.experts)

    if example_moe is None:
        self.num_moe_layers = 0
        self.num_expert_groups = 0
        self.num_shared_experts = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        return

    self.num_moe_layers = len(self.moe_layers)
    self.num_expert_groups = 1
    self.num_shared_experts = example_moe.n_shared_experts
    self.num_logical_experts = example_moe.n_logical_experts
    self.num_physical_experts = example_moe.n_physical_experts
    self.num_local_physical_experts = example_moe.n_local_physical_experts
    self.num_routed_experts = example_moe.n_routed_experts
    self.num_redundant_experts = example_moe.n_redundant_experts


def _glm4_moe_lite_set_moe_parameters(self) -> None:
    from vllm.model_executor.models.glm4_moe_lite import (
        Glm4MoeLite,
        Glm4MoeLiteDecoderLayer,
    )
    from vllm.model_executor.models.utils import PPMissingLayer

    self.expert_weights = []
    self.moe_layers = []
    self.moe_mlp_layers = []
    example_moe = None
    for layer in self.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if isinstance(layer, Glm4MoeLiteDecoderLayer) and isinstance(
            layer.mlp, Glm4MoeLite
        ):
            example_moe = layer.mlp
            self.moe_mlp_layers.append(layer.mlp)
            self.moe_layers.append(layer.mlp.experts)

    if example_moe is None:
        self.num_moe_layers = 0
        self.num_expert_groups = 0
        self.num_shared_experts = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        return

    self.num_moe_layers = len(self.moe_layers)
    self.num_expert_groups = 1
    self.num_shared_experts = example_moe.n_shared_experts
    self.num_logical_experts = example_moe.n_logical_experts
    self.num_physical_experts = example_moe.n_physical_experts
    self.num_local_physical_experts = example_moe.n_local_physical_experts
    self.num_routed_experts = example_moe.n_routed_experts
    self.num_redundant_experts = example_moe.n_redundant_experts


Glm4MixtureOfExperts.update_physical_experts_metadata = (
    _glm4_mixture_update_physical_experts_metadata
)
Glm4MoeForCausalLM.set_moe_parameters = _glm4_moe_set_moe_parameters
Glm4LiteMixtureOfExperts.update_physical_experts_metadata = (
    _glm4_mixture_update_physical_experts_metadata
)
Glm4MoeLiteForCausalLM.set_moe_parameters = _glm4_moe_lite_set_moe_parameters
