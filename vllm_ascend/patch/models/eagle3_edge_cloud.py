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
    aux_hidden_states: torch.Tensor | None = None,
    spec_step_idx: int = 0,
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

    if is_last_segment:
        # Last segment (edge): return post-norm hidden states and pre-norm residual
        # so that the proposer can sample logits and carry hidden_states to the
        # next draft step, matching the tuple return of Eagle3LlamaForCausalLM.forward.
        return intermediate_tensors["hidden_states"], intermediate_tensors["residual"]

    # Cloud segment: the caller prepares ``hidden_states`` before invoking this
    # segment.  On the first draft step it writes the result of
    # ``combine_hidden_states`` into the stable intermediate buffer; later
    # steps write the previous draft step's hidden states there.  Keep this
    # forward free of a ``spec_step_idx`` branch: ACL graph capture/replay (and
    # the no-guard torch.compile wrapper) would otherwise freeze whichever
    # Python branch happened to be captured first.
    input_embeds = intermediate_tensors["input_embeds"]
    hidden_states = intermediate_tensors["hidden_states"]
    residual = intermediate_tensors.tensors.get("residual", None)
    if hidden_states.numel() == 0:
        raise RuntimeError(
            "EAGLE3 cloud segment received an empty hidden_states tensor; "
            "the caller must prepare it before running the cloud segment."
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

# ----------------------------------------------------------------------------
# Fix: correct the EAGLE3 draft layer naming offset (start_layer_id) under
# edge-cloud mode, otherwise the first target-verification decode deadlocks
# inside ACL graph replay.
#
# Upstream Eagle3LlamaForCausalLM.__init__ derives the draft layer naming
# offset as
#     target_layer_num = model_config.get_num_layers(parallel_config)
# and builds ``self.model = LlamaModel(..., start_layer_id=target_layer_num)``,
# so the (single) draft attention layer is named ``model.layers.{N}`` and is
# expected to sit *after* every target layer (target layers are 0..N-1).
#
# ``get_num_layers`` is PP-aware: in edge-cloud the cloud process is a PP rank,
# so it returns the PP-divided count (e.g. 31) instead of the full target
# layer count (e.g. 61). The draft layer is then named ``model.layers.31``,
# which collides with the real target layer ``language_model.model.layers.31``
# living in the same ``forward_context.attn_metadata`` dict. That collision is
# silently swallowed by ``acl_graph._filter_attn_metadata_for_layers`` (the two
# keys are misclassified as DeepSeek-V4 DSA sub-keys via
# ``_is_dsa_kv_metadata_keys``), so target layer 31 is dropped from the
# filtered metadata. ``AscendAttentionBackendImpl.update_graph_params`` then
# ``zip``s 60 keys against the 61 captured ``attn_params``/``events``, leaving
# the last layer's event never ``record()``-ed; the captured graph's
# ``event.wait()`` deadlocks on replay -> hang on the first decode verify.
#
# Non-edge-cloud is unaffected because there ``get_num_layers`` returns the
# full count (PP=1) and the draft layer lands at ``model.layers.61`` (no
# collision).
#
# Fix: under edge-cloud mode, make the draft use the *full* target layer count
# as ``start_layer_id`` -- identical to the known-good non-edge (PP=1) value --
# so the draft layer is ``model.layers.{N}`` and never collides. We achieve
# this by temporarily shadowing ``model_config.get_num_layers`` for the
# duration of the draft ``__init__`` only, so nothing else is affected.
_original_eagle3_init = Eagle3LlamaForCausalLM.__init__


def _edge_cloud_draft_start_layer_id(vllm_config) -> int | None:
    """Full target layer count to use as the draft ``start_layer_id``.

    Equals the value non-edge-cloud (PP=1) computes via ``get_num_layers``.
    Returns ``None`` if it cannot be determined, in which case the caller
    falls back to the original behaviour unchanged.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        return None
    num_layers = getattr(hf_text_config, "num_hidden_layers", None)
    if isinstance(num_layers, int) and num_layers > 0:
        return num_layers
    return None


def _eagle3_init_with_edge_cloud_start_layer(
    self: Eagle3LlamaForCausalLM, *, vllm_config, prefix: str = ""
):
    # Only override the naming offset in edge-cloud mode; non-edge-cloud and
    # regular PP keep the upstream behaviour verbatim.
    is_edge_cloud = False
    try:
        from vllm.distributed.parallel_state import is_edge_cloud_pp_mode

        is_edge_cloud = bool(is_edge_cloud_pp_mode())
    except Exception:
        is_edge_cloud = False

    full_target_layers = (
        _edge_cloud_draft_start_layer_id(vllm_config) if is_edge_cloud else None
    )
    if full_target_layers is None:
        _original_eagle3_init(self, vllm_config=vllm_config, prefix=prefix)
        return

    # Shadow get_num_layers on this specific ModelConfig instance for the
    # duration of __init__ only, so the internally-computed target_layer_num /
    # start_layer_id / target_layer_count all pick up the full count. Restore
    # the original (class-level) lookup afterwards.
    model_config = vllm_config.model_config
    had_instance_attr = "get_num_layers" in model_config.__dict__
    prev_get_num_layers = model_config.__dict__.get("get_num_layers", None)
    model_config.get_num_layers = lambda _parallel_config: full_target_layers
    try:
        _original_eagle3_init(self, vllm_config=vllm_config, prefix=prefix)
    finally:
        if had_instance_attr:
            model_config.get_num_layers = prev_get_num_layers
        else:
            # Remove the instance attribute so attribute lookup falls back to
            # the unmodified class method.
            model_config.__dict__.pop("get_num_layers", None)


Eagle3LlamaForCausalLM.__init__ = _eagle3_init_with_edge_cloud_start_layer

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
