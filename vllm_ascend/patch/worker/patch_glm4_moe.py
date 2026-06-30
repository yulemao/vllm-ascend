#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# GLM-4/5 MoE on Ascend: inject is_sequence_parallel into FusedMoE.
#

from vllm.model_executor.models.glm4_moe import Glm4MoE
from vllm_ascend.utils import enable_sp

_original_glm4_moe_init = Glm4MoE.__init__
_original_fusedmoe_init = None


def _patched_glm4_moe_init(self, config, quant_config=None, prefix="", enable_eplb=False):
    """Patch to inject is_sequence_parallel into FusedMoE without modifying upstream.

    Temporarily replaces FusedMoE.__init__ so that any FusedMoE created
    inside the original Glm4MoE.__init__ automatically receives the correct
    is_sequence_parallel value from enable_sp().
    """
    from vllm.model_executor.layers.fused_moe import FusedMoE

    global _original_fusedmoe_init
    if _original_fusedmoe_init is None:
        _original_fusedmoe_init = FusedMoE.__init__

    is_sequence_parallel = enable_sp()

    def _inject_sp_init(fm_self, *args, **kwargs):
        if "is_sequence_parallel" not in kwargs:
            kwargs["is_sequence_parallel"] = is_sequence_parallel
        return _original_fusedmoe_init(fm_self, *args, **kwargs)

    # Temporarily patch FusedMoE.__init__
    FusedMoE.__init__ = _inject_sp_init
    try:
        _original_glm4_moe_init(self, config, quant_config, prefix, enable_eplb)
    finally:
        FusedMoE.__init__ = _original_fusedmoe_init


Glm4MoE.__init__ = _patched_glm4_moe_init  # type: ignore[misc]
