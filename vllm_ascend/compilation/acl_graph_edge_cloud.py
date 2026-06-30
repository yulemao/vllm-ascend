# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation import acl_graph as _acl_graph
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    GraphParams,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


# ============================================================
#  GraphParams 作用域管理
#  —— 通过直接交换 acl_graph._graph_params / _draft_graph_params
#     来影响 get_graph_params() 的返回值。
#     attention 后端通过 from-import 获取的函数引用不受影响，
#     因为函数体内读取的是模块级变量。
# ============================================================

def make_graph_params(aclgraph_capture_sizes: list[int]) -> GraphParams:
    """创建 GraphParams 实例（供边云 segment wrapper 初始化独立参数）。

    与 acl_graph.set_graph_params 字段完全一致（7 字段），
    但不写入全局 _graph_params，而是返回独立实例供每个
    EdgeCloudACLGraphWrapper 持有，实现 segment 间参数隔离。
    """
    return GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


@contextmanager
def graph_params_scope(
    graph_params: GraphParams | None,
    draft_graph_params: GraphParams | None = None,
):
    """临时将 acl_graph 的 _graph_params / _draft_graph_params 替换为指定参数。

    所有通过 get_graph_params() / get_draft_graph_params() 或 global 语句
    访问这些模块级变量的代码（包括 attention 后端和 ACLGraphWrapper）
    都会自动感知到替换后的值。

    退出时同步 NPU 流并恢复原值，防止后续 segment 读到错误参数。
    """
    old_graph_params = _acl_graph._graph_params
    old_draft_graph_params = _acl_graph._draft_graph_params
    if graph_params is not None:
        _acl_graph._graph_params = graph_params
    if draft_graph_params is not None:
        _acl_graph._draft_graph_params = draft_graph_params
    try:
        yield
    finally:
        if graph_params is not None:
            torch.npu.current_stream().synchronize()
        _acl_graph._graph_params = old_graph_params
        _acl_graph._draft_graph_params = old_draft_graph_params


# ============================================================
#  边云分段 ACLGraphWrapper
#  —— 继承标准 ACLGraphWrapper，在 __call__ 中注入
#     graph_params_scope，使图捕获/回放期间 attention 后端
#     获取到本 segment 的独立 GraphParams。
# ============================================================

class EdgeCloudACLGraphWrapper(ACLGraphWrapper):
    """边云分段 ACL 图包装器。

    与标准 ACLGraphWrapper 的唯一区别：__call__ 中通过 graph_params_scope
    将 acl_graph 的 _graph_params 临时替换为本 segment 的独立 GraphParams，
    确保 attention 后端在 capture/replay 期间操作正确的参数集。
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        super().__init__(
            runnable, vllm_config, runtime_mode, cudagraph_options,
            use_eagle=use_eagle, enable_enpu=enable_enpu,
        )
        # 每个 segment wrapper 持有独立的 GraphParams，
        # 避免 segment_a / segment_e / segment_c 的
        # task handle / event / attn_params 混入同一全局列表
        self.graph_params: GraphParams | None = None
        self.draft_graph_params: GraphParams | None = None

    def __call__(self, *args, **kwargs):
        with graph_params_scope(self.graph_params, self.draft_graph_params):
            return super().__call__(*args, **kwargs)
