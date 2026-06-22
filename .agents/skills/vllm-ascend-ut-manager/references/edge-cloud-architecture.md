# Edge-Cloud 协同推理架构详解

> 本文档固化 vllm-ascend 中 Edge-Cloud（边云协同推理）特性的知识体系。
> 当 AI 需要理解、修改或测试边云相关代码时，应优先阅读本文档。

---

## 1. 核心概念

Edge-Cloud 协同推理将大模型推理负载拆分到**边缘设备（Edge）**和**云端设备（Cloud）**上：

| 维度 | Edge 侧 | Cloud 侧 |
|---|---|---|
| 保留层 | 首 `head_k` 层 + 尾 `tail_k` 层 | 中间层 `[head_k, N-tail_k)` |
| 职责 | Embedding → 首段推理 → 发送中间态 → 接收结果 → 尾段推理 → Norm → Logits | 接收中间态 → 中段推理 → 返回中间态 |
| 通信方向 | Edge → Cloud（首段结果） | Cloud → Edge（中段结果） |
| 最终输出 | Edge 负责采样和输出 token | Cloud 不直接输出 |

**设计目标**：降低 Edge 侧显存占用，利用 Cloud 算力执行大部分 Transformer 层。

---

## 2. 系统修改层面总览

边云特性在 vllm-ascend 中涉及 7 个层面的修改：

### 2.1 配置层
**文件**: `vllm_ascend/ascend_config.py`

- `EdgeCloudConfig` 类：解析 `additional_config.edge_cloud_config`
  - `enabled`: 是否启用边云模式
  - `role`: `"edge"` 或 `"cloud"`
  - `edge_head_tail_layers`: 对称（int）或非对称（list/tuple）
  - `enable_decode_graph`: Decode 阶段是否启用 ACL Graph
  - `decode_graph_min_tokens`: 图模式最小 token 数
  - `hidden_dtype`: 中间态数据类型（默认 bf16）
- `AscendConfig._check_mix_placement()`: `mix_placement` 与共享专家 DP 互斥检查

### 2.2 模型定义层
**文件**: `vllm_ascend/models/deepseek_v4.py`

- `DeepseekV4Model.__init__`: 边云模式下**跳过 PP 占位逻辑**，全量创建所有层（后续由 `LayerShardLoader` 裁剪）
- `DeepseekV4ForCausalLM.__init__`: 边云模式下始终创建 `lm_head`（Edge 需要计算 logits）

### 2.3 模型加载层
**文件**: `vllm_ascend/model_loader/layer_shard_loader.py`

- `EdgeCloudLayerPlan`: 分片策略对象
  - `get_local_layers()`: Edge 返回 `{0..head_k-1} ∪ {N-tail_k..N-1}`，Cloud 返回 `{head_k..N-tail_k-1}`
  - `get_released_layers()`: 补集
  - `validate()`: 校验 Edge + Cloud 覆盖全部层且无交集
- `LayerShardLoader.apply_sharding(model, layer_plan)`:
  - 将非本地层替换为 `PPMissingLayer`（AutoWeightsLoader 会自动跳过）
  - Cloud 侧额外替换 `embed_tokens` / `norm` / `lm_head`
  - Edge 侧保留这些模块用于 Embedding 和 logits 计算

### 2.4 模型运行时补丁层
**文件**: `vllm_ascend/patch/models/`

所有补丁采用 **Monkey Patch** 方式，在模型加载后动态注入 `forward_edge_cloud_segment` 方法：

| 文件 | 目标模型 | 关键适配 |
|---|---|---|
| `llama_edge_cloud.py` | `LlamaModel` / `LlamaForCausalLM` | 标准 `layer(positions, hidden_states, residual)` |
| `deepseek_v4_edge_cloud.py` | `DeepseekV4Model` | `layer(x, positions, input_ids)` + `hc_head` + `unsqueeze(-2).repeat(1, hc_mult, 1)` |
| `qwen3_5_edge_cloud.py` | `Qwen3_5Model` | `layer(hidden_states, residual, positions)`（参数顺序与 Llama 不同） |

**通用逻辑**（三个补丁文件结构相似）：
- 首段（`is_first_segment`）: 做 Embedding，返回 `IntermediateTensors`
- 中段: 接收 `intermediate_tensors`，执行指定层范围，返回 `IntermediateTensors`
- 末段（`is_last_segment`）: 执行层 + Norm，返回最终 `hidden_states` 张量
- 非末段**不携带 `input_ids`**（防止 token 信息通过网络泄漏）

### 2.5 Worker / ModelRunner 层（核心调度）
**文件**: `vllm_ascend/worker/model_runner_v1.py`

初始化：
- `__init__` 中读取 `ascend_config.edge_cloud_config`，设置 `head_k` / `tail_k`，检测模型类型（`_is_deepseek_v4`, `_is_qwen3_5`）

模型加载：
- `_load_model_edge_cloud()`: 替代标准 `load_model`
  1. `initialize_model()` 创建全量模型（CPU 上，避免 NPU OOM）
  2. `LayerShardLoader.apply_sharding()` 裁剪非本地层
  3. `process_weights_after_loading()` 量化/格式调整
  4. 导入对应 monkey patch 模块
  5. 创建 `segment_a` / `segment_e`（Edge）或 `segment_c`（Cloud）
  6. 按需 `_wrap_segment_if_needed()` 包装 ACLGraphWrapper

分段 callable：
- `_create_segment_callable(model, start_layer, end_layer)`: 返回闭包，调用 `model.forward_edge_cloud_segment(...)`

图编译包装：
- `_wrap_segment_if_needed(segment)`:
  - `enable_decode_graph=False` → 原样返回
  - 无 full cudagraphs → 原样返回
  - `_is_dummy_or_profile_run()` 为 True → 原样返回（防止 HCCL 与 NPUGraph 死锁）
  - 否则包装为 `ACLGraphWrapper`

前向执行：
- `_forward()`（原 `execute_model` 内调用）:
  - 非边云模式：走原逻辑
  - Edge 模式：
    - `intermediate_tensors is None` → 执行 `segment_a`（首段），返回 `IntermediateTensors`
    - `intermediate_tensors is not None` → 执行 `segment_e`（尾段），恢复 `_EXTRA_CTX.layer_idx`
  - Cloud 模式：执行 `segment_c`（中段），必须返回 `IntermediateTensors`

状态管理：
- `_is_dummy_or_profile_run()`: 检测是否处于图捕获阶段
- `execute_model` 中 Edge 执行 segment_e 时**跳过 `_update_states`**（避免重复更新 batch states）
- `capture_model()`: 边云模式下直接 `return`（跳过标准模块遍历 capture）

**文件**: `vllm_ascend/worker/worker.py`

- `execute_model`:
  - Cloud 侧：`edge_cloud_broadcast_recv()` 接收 Edge 发送的 intermediate_tensors
  - Edge 侧：首次执行后 `isend_tensor_dict` 发送给 Cloud，然后 `edge_cloud_broadcast_recv()` 接收 Cloud 结果，再执行 segment_e
  - Cloud 最后阶段：`isend_tensor_dict` 将结果发回 Edge

### 2.6 分布式通信层
**文件**: `vllm_ascend/distributed/parallel_state.py`

- `edge_cloud_broadcast_recv()`: 统一的广播接收函数，返回 `(tensor_dict, handles, postprocess)`
- 从 vllm 导入的辅助函数：
  - `is_edge_cloud_pp_mode()`: 是否处于边云 PP 模式
  - `is_edge_device()`: 当前进程是否为 Edge 侧
  - `is_cloud_device()`: 当前进程是否为 Cloud 侧

### 2.7 平台 / 调度补丁层
**文件**: `vllm_ascend/patch/platform/`

- `patch_core.py`: scheduler 配置日志中标识 EdgeCloud 模式
- `patch_multiproc_executor.py`: 边云模式下 `world_size` 为 `edge_npu_count + cloud_npu_count`

---

## 3. 数据流时序图

```
Edge (NPU)                              Cloud (NPU)
  │                                        │
  │  Step 1: segment_a [0, head_k)         │
  │  (embedding + first layers)            │
  │                                        │
  │  IntermediateTensors ────────────────> │
  │           (isend_tensor_dict)          │
  │                                        │
  │         Step 2: segment_c [head_k, N-tail_k)
  │                    (middle layers)     │
  │                                        │
  │  <──────────────────────── IntermediateTensors
  │           (irecv / broadcast)          │
  │                                        │
  │  Step 3: segment_e [N-tail_k, N)       │
  │  (tail layers + norm + logits)         │
  │                                        │
  │  Sampling / Output                     │
  ▼                                        ▼
```

---

## 4. 关键设计决策与约束

1. **加载时裁剪（V3）而非加载后释放（V2）**
   - `apply_sharding` 在 `load_weights()` 之前执行，非本地层替换为 `PPMissingLayer`
   - 权重根本不会加载到这些层，大幅节省显存

2. **PPMissingLayer 作为占位层**
   - `PPMissingLayer.forward()` 返回 `args[0]`，对大多数模型签名安全
   - `DeepSeekV4MissingLayer` 显式定义以适配 `layer(x, positions, input_ids)`

3. **图编译的独立性**
   - 每个 segment 独立包装 `ACLGraphWrapper`
   - 通信（send/recv）始终发生在图外 Eager 执行
   - 避免 HCCL 与 `torch.npu.NPUGraph` 死锁

4. **input_ids 不跨网络传输**
   - 中段 `IntermediateTensors` 仅携带 `hidden_states`（Llama/Qwen 还携带 `residual`）
   - DeepSeek-V4 中段不携带 `input_ids`，防止 Hash MoE 的 token 信息泄漏

5. **_update_states 的去重**
   - Edge 执行 segment_e 时跳过 `_update_states`，因为 segment_a 已更新过

---

## 5. 测试覆盖要点

当修改边云相关代码时，UT 应覆盖：

- **配置**: `EdgeCloudConfig` 的解析、校验、`head_tail_k` 计算
- **分片**: `EdgeCloudLayerPlan` 的 local/released 层计算、`validate`
- **加载**: `apply_sharding` 对 Edge/Cloud 的裁剪行为、embed/norm/lm_head 的处理
- **补丁**: 各模型的 `forward_edge_cloud_segment` 首段/中段/末段逻辑、wrapper 委托
- **Runner**: `_create_segment_callable`、`_wrap_segment_if_needed` 的各分支、`_is_dummy_or_profile_run`
- **Worker**: `edge_cloud_broadcast_recv` 的调用路径（E2E 测试覆盖）
