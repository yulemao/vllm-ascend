#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from typing import Any

import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.sequence import IntermediateTensors


def _forward_edge_cloud_segment_eagle3(
    self: Eagle3LlamaForCausalLM,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    """Edge-cloud segmented forward for Eagle3LlamaForCausalLM (cloud fusion).

    Split:
      - First segment (edge): embed input_ids only.
      - Middle segment (cloud): combine target aux hidden states with input
        embeds, then run all decoder layers + final norm.
      - Last segment (edge): return post-norm hidden states for logits sampling.

    ``start_layer``/``end_layer`` are kept in the signature for compatibility
    with ``EdgeCloudSegment`` but the actual split is driven by
    ``is_first_segment``/``is_last_segment``. In this cloud-fusion variant the
    EAGLE3 fc projection runs on the cloud using the target model's aux hidden
    states cached by the main model runner.
    """
    num_layers = len(self.model.layers)
    if is_first_segment is None:
        is_first_segment = start_layer == 0
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers

    if is_first_segment:
        if inputs_embeds is None:
            assert input_ids is not None, (
                "input_ids is None in Eagle3 edge-cloud first segment; "
                "either input_ids or inputs_embeds must be provided."
            )
            inputs_embeds = self.model.embed_input_ids(input_ids)
        # Cloud-fusion mode: the edge side only sends input_embeds to the cloud.
        # The cloud segment will fuse target aux hidden states via
        # combine_hidden_states before running decoder layers.
        return IntermediateTensors(
            {
                "input_embeds": inputs_embeds,
                "hidden_states": torch.empty(
                    0,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                ),
                "residual": None,
            }
        )

    assert intermediate_tensors is not None, (
        "intermediate_tensors is None in Eagle3 edge-cloud segment; "
        "check that all TP ranks receive tensors correctly."
    )
    input_embeds = intermediate_tensors["input_embeds"]
    residual = intermediate_tensors.tensors.get("residual", None)

    if not is_last_segment:
        # Cloud segment: fuse target aux hidden states on the first draft step,
        # or consume the previous-step draft hidden states on later steps.
        spec_step_idx = extra_layer_kwargs.get("spec_step_idx", 0)
        if spec_step_idx == 0:
            aux_hidden_states = extra_layer_kwargs.get("aux_hidden_states", None)
            if aux_hidden_states is not None and self.model.use_aux_hidden_state:
                hidden_states = self.model.combine_hidden_states(aux_hidden_states)
            else:
                # Fallback for warmup / missing aux: use the placeholder hidden
                # states sent by the edge. This should not happen in normal runtime.
                hidden_states = intermediate_tensors["hidden_states"]
                if hidden_states.numel() == 0:
                    raise RuntimeError(
                        "EAGLE3 cloud segment received empty aux_hidden_states "
                        "and an empty placeholder hidden_states tensor."
                    )
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            if hidden_states.numel() == 0:
                raise RuntimeError(
                    "EAGLE3 cloud segment received empty hidden_states tensor "
                    f"for spec_step_idx={spec_step_idx}; the edge side must "
                    "send the previous draft step's hidden states."
                )
        for layer in self.model.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, hidden_prenorm = self.model.norm(hidden_states, residual)
        return IntermediateTensors(
            {
                "hidden_states": hidden_states,
                "residual": hidden_prenorm,
            }
        )

    # Last segment (edge): return post-norm hidden states and pre-norm residual
    # so that the proposer can sample logits and carry hidden_states to the
    # next draft step, matching the tuple return of Eagle3LlamaForCausalLM.forward.
    return intermediate_tensors["hidden_states"], intermediate_tensors["residual"]


def _eagle3_make_empty_intermediate_tensors(
    self: Eagle3LlamaForCausalLM,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IntermediateTensors:
    hidden_size = self.model.config.hidden_size
    return IntermediateTensors(
        {
            "input_embeds": torch.empty(
                batch_size, hidden_size, dtype=dtype, device=device
            ),
            "hidden_states": torch.empty(
                batch_size, hidden_size, dtype=dtype, device=device
            ),
            "residual": torch.empty(
                batch_size, hidden_size, dtype=dtype, device=device
            ),
        }
    )


# The upstream Eagle3LlamaForCausalLM.forward does not accept
# ``intermediate_tensors``, so vLLM's static ``supports_pp()`` inspection
# (which checks the forward signature) returns False even though we set
# ``supports_pp = True`` above. Wrap the original forward to accept
# ``intermediate_tensors``; the actual edge-cloud runtime path uses
# ``forward_edge_cloud_segment`` instead of this wrapper.
_original_eagle3_forward = Eagle3LlamaForCausalLM.forward


def _eagle3_forward_with_pp(
    self: Eagle3LlamaForCausalLM,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    *,
    intermediate_tensors: IntermediateTensors | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if intermediate_tensors is not None:
        hidden_states = intermediate_tensors["hidden_states"]
    if hidden_states is None:
        raise ValueError(
            "Eagle3LlamaForCausalLM.forward requires hidden_states or "
            "intermediate_tensors containing hidden_states."
        )
    return _original_eagle3_forward(
        self, input_ids, positions, hidden_states, inputs_embeds
    )


Eagle3LlamaForCausalLM.forward_edge_cloud_segment = (
    _forward_edge_cloud_segment_eagle3
)
Eagle3LlamaForCausalLM.supports_pp = True
Eagle3LlamaForCausalLM.make_empty_intermediate_tensors = (
    _eagle3_make_empty_intermediate_tensors
)
Eagle3LlamaForCausalLM.forward = _eagle3_forward_with_pp

# Clear stale _ModelInfo caches so that inspect_model_cls re-computes
# supports_pp with the patched class instead of loading the old cached value.
from pathlib import Path  # noqa: E402

from vllm.envs import VLLM_CACHE_ROOT  # noqa: E402
from vllm.model_executor.models.registry import _try_inspect_model_cls  # noqa: E402

# Clear in-memory lru_cache in case it was populated before the patch.
_try_inspect_model_cls.cache_clear()

# Clear on-disk cache files for eagle3 draft architectures so the next
# inspect runs _ModelInfo.from_model_cls on the patched class.
_cache_dir = Path(VLLM_CACHE_ROOT) / "modelinfos"
if _cache_dir.exists():
    for _cache_file in _cache_dir.glob("*eagle3*"):
        _cache_file.unlink()
    for _cache_file in _cache_dir.glob("*llama_eagle3*"):
        _cache_file.unlink()
