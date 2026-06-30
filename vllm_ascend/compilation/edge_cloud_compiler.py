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
#
import sys
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import logger

from vllm_ascend.ascend_config import AscendCompilationConfig
from vllm_ascend.utils import COMPILATION_PASS_KEY


class EdgeCloudCompiledSegment(nn.Module):
    """Wraps an edge-cloud segment with npugraph_ex compile-time optimization.

    In the standard (non-edge-cloud) flow, the full model's forward is compiled
    by vLLM's compilation infrastructure::

        Dynamo trace → AscendCompiler.compile() → npugraph_ex_compile()
            → optimized callable → ACLGraphWrapper captures torch.npu.NPUGraph

    In the original edge-cloud flow, each ``EdgeCloudSegment`` was wrapped
    directly by ``ACLGraphWrapper``, **skipping** the compile-time FX graph
    optimization stage entirely.  Only runtime graph capture/replay was used.

    This class bridges that gap by applying ``torch.compile`` with a backend
    that delegates to ``npugraph_ex_compile`` (or ``fusion_pass_compile`` as
    fallback), producing an optimized callable that ``ACLGraphWrapper`` can
    then capture — exactly mirroring the standard flow's two-stage pipeline.

    Usage::

        segment = EdgeCloudSegment(model, 0, head_k, ...)
        compiled = EdgeCloudCompiledSegment(segment, vllm_config, ascend_config)
        wrapper = ACLGraphWrapper(compiled, vllm_config, runtime_mode=FULL)

    During the first forward call, ``torch.compile`` triggers Dynamo tracing
    and backend compilation.  The backend calls ``npugraph_ex_compile()``,
    which applies the torchair FX graph optimization passes (operator fusion,
    in-place restoration, etc.).  The resulting optimized callable is then
    captured by ``ACLGraphWrapper`` via ``torch.npu.NPUGraph()``.
    """

    def __init__(
        self,
        segment: nn.Module,
        vllm_config: VllmConfig,
        ascend_compilation_config: AscendCompilationConfig,
    ):
        super().__init__()
        self._segment = segment
        self._vllm_config = vllm_config
        self._ascend_compilation_config = ascend_compilation_config

        # Skip all Dynamo guards so that the segment is compiled exactly once.
        # Without guard skipping, each call with a different batch size during
        # warmup triggers a fresh Dynamo compilation, quickly exhausting the
        # recompile limit (default 8) and raising FailOnRecompileLimitHit.
        # This matches TorchCompileWithNoGuardsWrapper in the standard flow:
        # the single compiled graph is shape-agnostic; shape-specific
        # capture / replay is handled independently by ACLGraphWrapper.
        options: dict[str, Any] = {}
        if hasattr(torch.compiler, "skip_all_guards_unsafe"):
            options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
        else:
            options["guard_filter_fn"] = lambda x: [False for _ in x]

        # dynamic=True is essential here so that Dynamo treats the batch
        # dimension symbolically (s0) rather than baking a concrete value
        # (e.g. 8192) into the FX graph.  Without it, npugraph_ex produces
        # a shape-specific binary that fails on the next warmup run with a
        # different batch size.  This mirrors the standard flow's behaviour
        # where _mark_dynamic_inputs() marks dim-0 dynamic before tracing.
        if ascend_compilation_config.enable_npugraph_ex:
            logger.info(
                "EdgeCloudCompiledSegment: enable_npugraph_ex=True, "
                "using npugraph_ex backend for segment compilation."
            )
            self._compiled = torch.compile(
                segment,
                fullgraph=True,
                dynamic=True,
                backend=self._npugraph_ex_backend,
                options=options,
            )
        else:
            logger.info(
                "EdgeCloudCompiledSegment: enable_npugraph_ex=False, "
                "using fusion_pass backend for segment compilation."
            )
            self._compiled = torch.compile(
                segment,
                fullgraph=True,
                dynamic=True,
                backend=self._fusion_pass_backend,
                options=options,
            )

    def _npugraph_ex_backend(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Any:
        """torch.compile backend: delegates to npugraph_ex_compile.

        Called by Dynamo after tracing the segment's forward pass into an
        FX graph.  Invokes ``npugraph_ex_compile()`` which:
        1. Sets up torchair compiler config (reduce-overhead mode)
        2. Optionally enables static shape kernel compilation
        3. Calls ``torchair.get_npu_backend()`` for FX graph optimization
        4. Returns the optimized callable
        """
        from vllm_ascend.compilation.compiler_interface import npugraph_ex_compile

        # Use sys.maxsize as the compile range upper bound so that all
        # compile-range-gated fusion passes apply to the segment.
        compile_range = Range(0, sys.maxsize)

        compiled_fn, _ = npugraph_ex_compile(
            graph=gm,
            example_inputs=list(example_inputs),
            compiler_config={},
            vllm_config=self._vllm_config,
            ascend_compilation_config=self._ascend_compilation_config,
            compile_range=compile_range,
            key=None,
        )
        return compiled_fn

    def _fusion_pass_backend(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Any:
        """torch.compile backend: delegates to fusion_pass_compile.

        Fallback path when ``enable_npugraph_ex`` is False (e.g. PIECEWISE
        cudagraph mode).  Applies only the basic ``GraphFusionPassManager``
        passes (norm_quant, qknorm_rope, allreduce_rms, muls_add, SP)
        without the torchair backend.

        Creates a fresh ``GraphFusionPassManager`` and configures it with
        the fusion pass flags from ``AscendCompilationConfig``, mirroring
        what the standard inductor pass system does.
        """
        from vllm_ascend.compilation.compiler_interface import fusion_pass_compile
        from vllm_ascend.compilation.graph_fusion_pass_manager import (
            GraphFusionPassManager,
        )

        # Build a fresh pass manager configured with the current
        # AscendCompilationConfig flags, consistent with how
        # NPUPlatform.get_pass_manager_cls() initialises it for the
        # standard flow.
        pass_manager = GraphFusionPassManager()
        pass_manager.configure(self._vllm_config)
        compiler_config = {COMPILATION_PASS_KEY: pass_manager}

        compile_range = Range(0, sys.maxsize)

        compiled_fn, _ = fusion_pass_compile(
            graph=gm,
            example_inputs=list(example_inputs),
            compiler_config=compiler_config,
            compile_range=compile_range,
            key=None,
        )
        return compiled_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the compiled segment forward.

        On first call this triggers Dynamo tracing + backend compilation
        (one-time cost).  Subsequent calls use the compiled callable directly.
        """
        return self._compiled(*args, **kwargs)
