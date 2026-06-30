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

import vllm.v1.executor.multiproc_executor as _mpe

# Resolve the class in the MRO that actually defines `max_concurrent_batches`
# and capture its (upstream) descriptor. Walking the MRO instead of reading the
# immediate `__dict__` keeps this correct no matter the import order relative to
# `patch_multiproc_executor` (which, under DYNAMIC_EPLB/EXPERT_MAP_RECORD, rebinds
# `_mpe.MultiprocExecutor` to a subclass). We patch the *owning* class, so the
# subclass inherits it and there is no double-patch / no conflict.
_OWNER_CLS = next(
    c
    for c in _mpe.MultiprocExecutor.__mro__
    if "max_concurrent_batches" in c.__dict__
)
_ORIG_DESC = _OWNER_CLS.__dict__["max_concurrent_batches"]
# Idempotency guard: never wrap our own property (would recurse on delegation).
_ALREADY_PATCHED = getattr(_ORIG_DESC, "_ec_batch_queue_patched", False)


def _max_concurrent_batches(self) -> int:
    # Edge-cloud runs as pipeline parallel (edge=PP0, cloud=PP1), so the base
    # implementation returns pp_size (>1). That makes EngineCore enable the
    # pipelined `step_with_batch_queue` path. With speculative decoding AND
    # synchronous scheduling (`--no-async-scheduling`) that path schedules the
    # next spec-verify batch before the previous batch's sampled token is
    # applied, and there is no `num_output_placeholders` accounting (only
    # AsyncScheduler maintains it) to compensate. The verify window then loses
    # the +1 base token, every draft is checked one position early and is
    # rejected, collapsing the draft hit rate.
    #
    # Force a single in-flight batch so EngineCore falls back to the sequential
    # `step()` path, which always applies the sampled token before scheduling
    # the next batch. Async edge-cloud keeps placeholders (correct) and the
    # pipeline overlap, so it is left untouched.
    if (
        getattr(self.parallel_config, "enable_edge_cloud", False)
        and self.speculative_config is not None
        and not self.scheduler_config.async_scheduling
    ):
        return 1
    # Delegate to the original implementation. `__get__` works for both
    # cached_property and property; for cached_property it caches onto the
    # instance __dict__, which our class-level data descriptor shadows.
    return _ORIG_DESC.__get__(self, type(self))


# `max_concurrent_batches` is a cached_property on the upstream class; replace it
# with a plain property on the owning class. It is read once at EngineCore init,
# so recomputation cost is irrelevant. Patching the owning class also covers
# AscendMultiprocExecutor (used under DYNAMIC_EPLB/EXPERT_MAP_RECORD), which
# inherits this attribute.
if not _ALREADY_PATCHED:
    _patched = property(_max_concurrent_batches)
    _patched.fget._ec_batch_queue_patched = True
    _OWNER_CLS.max_concurrent_batches = _patched


