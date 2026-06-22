# vLLM 项目架构分析报告

> 分析日期：2026-05-16
> 分析对象：vLLM 推理引擎（当前为 V1 架构）
> 代码路径：vllm/vllm/

---

## 一、总体架构概述

vLLM 是一个高性能、高吞吐的大语言模型（LLM）推理和服务引擎，核心创新包括：
- **PagedAttention**：通过块化的 KV Cache 管理大幅减少显存浪费
- **Continuous Batching**：动态调度实现请求级流水线并行
- **Prefix Caching**：前缀缓存复用已计算的 KV Cache
- **Speculative Decoding**：通过草稿模型加速解码
- **多模态支持**：图像、音频、视频输入处理

### 1.1 架构演进

当前代码库处于 **V1 架构唯一实现** 状态：
- **V0（Legacy）**：已完全移除，仅保留兼容别名
- **V1（当前）**：完整模块化设计，位于 `vllm/vllm/v1/` 下

### 1.2 分层架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API / Entrypoints 层                               │
│  entrypoints/llm.py │ entrypoints/openai/ │ entrypoints/cli/                │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Engine 引擎层                                      │
│  v1/engine/llm_engine.py │ v1/engine/async_llm.py │ v1/engine/core.py       │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Core 调度层                                        │
│  v1/core/sched/scheduler.py │ v1/core/kv_cache_manager.py                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Executor 执行策略层                                │
│  v1/executor/uniproc_executor.py │ v1/executor/multiproc_executor.py        │
│  v1/executor/ray_executor.py                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Worker 工作节点层                                  │
│  v1/worker/gpu_worker.py │ v1/worker/gpu_model_runner.py                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Model Executor 计算内核层                          │
│  model_executor/models/ │ model_executor/layers/ │ model_executor/kernels/  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、各模块功能详细描述

### 2.1 engine/ — 引擎入口与协调层

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/engine/llm_engine.py` | `LLMEngine` | 同步引擎包装器（Legacy 兼容层），负责初始化配置、创建 `EngineCoreClient`、输入/输出处理器和统计日志 |
| `v1/engine/async_llm.py` | `AsyncLLM` | 异步引擎（原 `AsyncLLMEngine` 的 V1 实现），实现 `EngineClient` 协议，支持 asyncio 流式生成 |
| `v1/engine/core.py` | `EngineCore` | **引擎核心内循环**，负责实例化 `Executor`、初始化 KV Cache、创建 `Scheduler`，并运行调度-执行主循环 |
| `v1/engine/core_client.py` | `EngineCoreClient` | 抽象基类，封装与 `EngineCore` 的通信方式。子类：`InprocClient`（进程内）、`SyncMPClient`（同步多进程）、`AsyncMPClient`（异步多进程） |
| `v1/engine/input_processor.py` | `InputProcessor` | 输入预处理：tokenizer、多模态编码器预算、prompt 渲染 |
| `v1/engine/output_processor.py` | `OutputProcessor` | 输出后处理：detokenization、logprob 处理、流式输出聚合 |
| `v1/engine/detokenizer.py` | `Detokenizer` | 将模型输出的 token ID 流实时解码为人类可读文本 |
| `v1/engine/parallel_sampling.py` | `ParentRequest` | 并行采样（n > 1）的请求拆分与管理 |

**核心逻辑**：`EngineCore` 是引擎的心跳，每步执行 `scheduler.schedule()` → `executor.execute_model()` → `output_processor.process_outputs()` 的循环。

---

### 2.2 core/ — 调度与 KV Cache 管理层

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/core/sched/scheduler.py` | `Scheduler` | **核心调度器**，管理请求队列、chunked prefill、prefix caching、speculative decoding、KV Connector 交互 |
| `v1/core/sched/interface.py` | `SchedulerInterface` | 调度器抽象接口，定义 `schedule()`、`get_grammar_bitmask()` 等方法 |
| `v1/core/sched/output.py` | `SchedulerOutput` | 调度结果数据结构，包含 `NewRequestData`、`CachedRequestData`、`num_scheduled_tokens` |
| `v1/core/sched/request_queue.py` | `RequestQueue` | 请求队列实现，支持 FCFS 等调度策略 |
| `v1/core/sched/async_scheduler.py` | `AsyncScheduler` | 异步调度器变体 |
| `v1/core/kv_cache_manager.py` | `KVCacheManager` | KV Cache 的分配、回收、前缀缓存块管理（替代 V0 的 BlockManager） |
| `v1/core/block_pool.py` | `BlockPool` | 物理块池管理，维护 `FreeKVCacheBlockQueue`（双向链表）实现 LRU 风格回收 |
| `v1/core/kv_cache_utils.py` | 辅助函数 | 块 hash 生成、KV Cache 组管理 |
| `v1/core/encoder_cache_manager.py` | `EncoderCacheManager` | 多模态编码器输出的缓存管理 |
| `v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` | 跨组 KV Cache 协调 |
| `v1/request.py` | `Request` / `RequestStatus` | 请求实体和状态机（WAITING → RUNNING → FINISHED） |

**核心创新 — Prefix Caching**：
- `BlockPool` 维护 `BlockHashToBlockMap`，将块内容 hash 映射到物理块
- 调度时 `KVCacheManager.get_computed_blocks()` 查找最长缓存命中路径
- 新请求可直接复用已缓存的 KV Cache，跳过前缀的重复计算

---

### 2.3 executor/ — 分布式执行策略层

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/executor/abstract.py` | `Executor` | 执行器抽象基类，封装 `execute_model`（通过 `collective_rpc` 调用所有 worker） |
| `v1/executor/uniproc_executor.py` | `UniProcExecutor` | 单进程执行器（本地单卡），直接持有 `WorkerWrapperBase` |
| `v1/executor/multiproc_executor.py` | `MultiprocExecutor` | 多进程执行器（多卡 MP），使用 multiprocessing + cloudpickle |
| `v1/executor/ray_executor.py` | `RayDistributedExecutor` | Ray 分布式执行器，支持大规模集群部署 |

**设计模式**：Executor 是**策略模式**的典型应用，上层 `EngineCore` 只依赖 `Executor` 抽象，具体分布式策略（单卡/多卡/Ray）由子类实现。

---

### 2.4 worker/ — 工作节点与设备抽象层

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/worker/worker_base.py` | `WorkerBase` | Worker 抽象基类，定义 `init_device`、`load_model`、`execute_model`、`sample_tokens` 接口 |
| `v1/worker/worker_base.py` | `WorkerWrapperBase` | Worker 惰性包装器，负责跨进程生命周期管理 |
| `v1/worker/gpu_worker.py` | `GPUWorker` | GPU Worker 实现，管理设备初始化、模型加载、CUDA Graph、分布式通信 |
| `v1/worker/gpu_model_runner.py` | `GPUModelRunner` | GPU 模型运行核心，构建输入 batch、调用模型 forward、采样 |
| `v1/worker/gpu_input_batch.py` | `GPUInputBatch` | GPU 输入 batch 的数据结构和管理 |
| `v1/worker/block_table.py` | `BlockTable` | 请求的 KV Cache 块映射表 |
| `v1/worker/workspace.py` | `Workspace` | GPU 工作空间内存管理 |
| `v1/worker/ubatching.py` | 工具 | micro-batching 辅助 |
| `v1/worker/cpu_worker.py` | `CPUWorker` / `CPUModelRunner` | CPU 设备实现 |
| `v1/worker/xpu_worker.py` | `XPUWorker` / `XPUModelRunner` | Intel XPU 设备实现 |
| `v1/worker/kv_connector_model_runner_mixin.py` | `KVConnectorModelRunnerMixin` | KV Connector 混入，支持 P/D 分离和 KV 卸载 |

---

### 2.5 model_executor/ — 模型执行与计算层

#### 2.5.1 models/ — 模型实现

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `models/registry.py` | `ModelRegistry` | 模型注册表，维护 HuggingFace 架构名到 vLLM 模型类的映射 |
| `models/interfaces_base.py` | `VllmModel` (Protocol) | 所有模型必须实现的基础协议：embed_input_ids、forward |
| `models/interfaces.py` | `SupportsMultiModal` 等 | 模型能力标记接口 |
| `models/llama.py` | `LlamaForCausalLM` | Llama 系列模型实现 |
| `models/qwen2.py` | `Qwen2ForCausalLM` | Qwen 系列模型实现 |
| `models/deepseek_v3.py` | `DeepseekV3ForCausalLM` | DeepSeek-V3 模型实现 |

**注册表机制**：
- `_TEXT_GENERATION_MODELS`：文本生成模型映射
- `_EMBEDDING_MODELS`：Embedding 模型映射
- `_MULTIMODAL_MODELS`：多模态模型映射
- `_SPECULATIVE_DECODING_MODELS`：推测解码模型映射
- `_TRANSFORMERS_BACKEND_MODELS`：基于 Transformers 后端的模型映射

#### 2.5.2 layers/ — 通用层实现

| 子目录 | 核心类 | 职责 |
|--------|--------|------|
| `layers/attention/` | `Attention`、`MLAAttention` | 注意力机制实现，自动选择后端（FlashAttention、Triton、FlashInfer） |
| `layers/fused_moe/` | `FusedMoE` | MoE（混合专家）融合算子，支持大量后端 |
| `layers/quantization/` | `GPTQConfig`、`FP8Config` 等 | 量化方案（AWQ、GPTQ、FP8、MXFP4、GGUF） |
| `layers/rotary_embedding/` | 各种 RoPE | 旋转位置编码实现 |
| `layers/linear.py` | `QKVParallelLinear`、`MergedColumnParallelLinear` | 并行线性层 |
| `layers/mamba/` | Mamba/SSM 层 | 状态空间模型层 |
| `layers/pooler/` | Pooling 层 | Embedding/分类池化 |

#### 2.5.3 model_loader/ — 模型加载

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `model_loader/loader.py` | `DefaultModelLoader` | 默认 HF 格式加载 |
| `model_loader/gguf_loader.py` | `GGUFModelLoader` | GGUF 格式加载 |
| `model_loader/tensorizer.py` | `TensorizerLoader` | Tensorizer 格式加载 |
| `model_loader/bitsandbytes.py` | BnB 加载器 | BitsAndBytes 量化加载 |

---

### 2.6 attention/ — 注意力机制（V1 后端）

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/attention/backend.py` | `AttentionBackend` | 注意力后端抽象基类，定义接口和 KV Cache 内存布局 |
| `v1/attention/selector.py` | `get_attn_backend()` | 根据 head_size、dtype、MLA 等条件选择最优后端 |
| `v1/attention/backends/flash_attn.py` | `FlashAttentionBackend` | FlashAttention 后端 |
| `v1/attention/backends/flashinfer.py` | `FlashInferBackend` | FlashInfer 后端 |
| `v1/attention/backends/triton_attn.py` | `TritonAttentionBackend` | Triton 手写 kernel 后端 |
| `v1/attention/backends/rocm_attn.py` | `ROCmAttentionBackend` | AMD ROCm 后端 |
| `v1/attention/ops/paged_attn.py` | `paged_attention` | PagedAttention 核心算子 |
| `v1/attention/ops/prefix_prefill.py` | `prefix_prefill` | Prefix Prefill 算子 |

---

### 2.7 sample/ — 采样逻辑（V1）

| 文件 | 核心类 | 职责 |
|------|--------|------|
| `v1/sample/sampler.py` | `Sampler` | 核心采样器（nn.Module），从 logits 中采样下一个 token |
| `v1/sample/metadata.py` | `SamplingMetadata` | 采样参数张量（temperature、top_p、top_k、penalties） |
| `v1/sample/rejection_sampler.py` | `RejectionSampler` | 推测解码的拒绝采样器 |
| `v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | Top-K + Top-P 联合采样 |
| `v1/sample/ops/penalties.py` | 惩罚算子 | repetition、frequency、presence 惩罚 |
| `v1/sample/ops/bad_words.py` | BadWordsFilter | bad words 过滤 |
| `v1/sample/ops/logprobs.py` | Logprobs 工具 | logprob 统计 |
| `v1/sample/logits_processor/` | `LogitsProcessor` | logits 后处理器 |

**采样流程**：
1. 计算原始 logprobs
2. logits 转 float32
3. 应用 allowed token whitelist 和 bad words 屏蔽
4. 应用非 argmax-invariant logits 处理器（min tokens、logit bias）
5. 应用 penalties（repetition、frequency、presence）
6. 采样：greedy/random 判断 → 温度缩放 → argmax-invariant 处理器（min-p）→ Top-K/Top-P 截断 → 概率采样
7. 收集 top-k logprobs 和采样 token 的 logprob

---

### 2.8 distributed/ — 分布式通信

| 子目录 | 核心功能 | 职责 |
|--------|----------|------|
| `device_communicators/` | NCCL、pynccl、shm、Ray、CUDA 通信器 | 设备间通信抽象 |
| `kv_transfer/` | Mooncake、LMCache、p2p 等 connector | KV Cache 跨节点传输 |
| `ec_transfer/` | Elastic Connector | 弹性连接器传输 |
| `elastic_ep/` | Elastic EP | 弹性专家并行 |
| `eplb/` | Expert Parallel Load Balancing | 专家并行负载均衡 |
| `weight_transfer/` | 权重传输 | 模型权重热迁移 |

---

### 2.9 其他重要模块

| 模块 | 路径 | 职责 |
|------|------|------|
| multimodal/ | `vllm/multimodal/` | 多模态处理（图像、音频、视频预处理） |
| lora/ | `vllm/lora/` | LoRA 适配器支持（layers、ops、punica_wrapper） |
| tokenizers/ | `vllm/tokenizers/` | 分词器封装 |
| config/ | `vllm/config/` | 配置系统（model.py、attention.py、scheduler.py 等） |
| compilation/ | `vllm/compilation/` | PyTorch 编译优化（Inductor passes、CUDA graph） |
| inputs/ | `vllm/inputs/` | 输入处理（PromptType、preprocess、parse） |
| platforms/ | `vllm/platforms/` | 平台抽象（cuda、rocm、tpu、cpu、xpu） |
| spec_decode/ | `vllm/v1/spec_decode/` | 投机解码（V1） |
| structured_output/ | `vllm/v1/structured_output/` | 结构化输出（grammar、json、regex） |

---

## 三、模块间调用与依赖关系

### 3.1 顶层调用链（从用户 API 到模型执行）

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  entrypoints/llm.py (LLM 类)                                             │
│  entrypoints/openai/api_server.py (OpenAI 兼容服务)                      │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  v1/engine/llm_engine.py (LLMEngine)                                     │
│  v1/engine/async_llm.py (AsyncLLM)                                       │
│  - 初始化配置、创建 EngineCoreClient                                      │
│  - 输入/输出处理器、统计日志                                              │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  v1/engine/core_client.py (EngineCoreClient)                             │
│  - InprocClient / SyncMPClient / AsyncMPClient                           │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  v1/engine/core.py (EngineCore)                                          │
│  - 实例化 Executor、初始化 KV Cache、创建 Scheduler                      │
│  - 运行 调度-执行 主循环                                                  │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ├───► ┌────────────────────────────────────────────────────────────┐
    │     │  v1/core/sched/scheduler.py (Scheduler)                    │
    │     │  - 管理 RequestQueue                                       │
    │     │  - 每步决定 running batch                                  │
    │     │  - 输出 SchedulerOutput                                    │
    │     └────────────────────────────────────────────────────────────┘
    │                           │
    │                           ▼
    │     ┌────────────────────────────────────────────────────────────┐
    │     │  v1/core/kv_cache_manager.py (KVCacheManager)              │
    │     │  v1/core/block_pool.py (BlockPool)                         │
    │     │  - 分配/回收 KV Cache 块                                   │
    │     │  - 前缀缓存查找                                            │
    │     └────────────────────────────────────────────────────────────┘
    │
    └───► ┌────────────────────────────────────────────────────────────┐
          │  v1/executor/abstract.py (Executor)                        │
          │  - UniProcExecutor / MultiprocExecutor / RayDistributedExecutor │
          │  - collective_rpc("execute_model", SchedulerOutput)        │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/worker/worker_base.py (WorkerWrapperBase)              │
          │  - 跨进程生命周期管理                                       │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/worker/gpu_worker.py (GPUWorker)                       │
          │  - 设备初始化、模型加载、CUDA Graph                          │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/worker/gpu_model_runner.py (GPUModelRunner)            │
          │  - 构建 GPUInputBatch                                       │
          │  - 调用 model.forward()                                    │
          │  - LogitsProcessor 计算 logits                             │
          │  - Sampler 采样                                            │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  model_executor/models/*.py (具体模型，如 LlamaForCausalLM) │
          │  - Attention 层调用                                        │
          │  - MLP 层计算                                              │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  model_executor/layers/attention/attention.py              │
          │  - 通过 get_attn_backend() 选择后端                        │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/attention/backends/*.py (FlashAttention/FlashInfer/...)│
          │  v1/attention/ops/*.py (Triton kernel)                     │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/sample/sampler.py (Sampler)                            │
          │  - Top-K/Top-P 采样                                        │
          │  - 惩罚、logprobs                                          │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/outputs.py (ModelRunnerOutput)                         │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  v1/engine/output_processor.py (OutputProcessor)           │
          │  - detokenization                                          │
          │  - 流式输出聚合                                            │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                              返回给用户
```

### 3.2 核心 Import 与类继承关系

#### Engine 层依赖
```
LLMEngine
  ├── EngineCoreClient (v1/engine/core_client.py)
  ├── InputProcessor (v1/engine/input_processor.py)
  ├── OutputProcessor (v1/engine/output_processor.py)
  └── EngineArgs (engine/arg_utils.py)

AsyncLLM
  ├── EngineCoreClient
  ├── InputProcessor
  ├── OutputProcessor
  └── EngineClient (协议)
```

#### Core 层依赖
```
Scheduler (继承 SchedulerInterface)
  ├── KVCacheManager (v1/core/kv_cache_manager.py)
  ├── EncoderCacheManager (v1/core/encoder_cache_manager.py)
  ├── RequestQueue (v1/core/sched/request_queue.py)
  ├── StructuredOutputManager (v1/structured_output/)
  └── KVCacheCoordinator (v1/core/kv_cache_coordinator.py)

SchedulerOutput
  ├── NewRequestData
  ├── CachedRequestData
  └── GrammarOutput
```

#### Executor 层依赖
```
Executor (抽象基类)
  ├── WorkerBase (v1/worker/worker_base.py)
  ├── SchedulerOutput
  └── ModelRunnerOutput (v1/outputs.py)

UniProcExecutor (继承 Executor)
  └── WorkerWrapperBase

MultiprocExecutor (继承 Executor)
  └── WorkerWrapperBase + multiprocessing

RayDistributedExecutor (继承 Executor)
  └── Ray worker
```

#### Worker 层依赖
```
WorkerBase (抽象基类)
  ├── init_device()
  ├── load_model()
  ├── execute_model()
  └── sample_tokens()

GPUWorker (继承 WorkerBase)
  ├── GPUModelRunner (v1/worker/gpu_model_runner.py)
  ├── current_platform (vllm/platforms/)
  └── distributed 通信库 (vllm/distributed/)

GPUModelRunner
  ├── model_loader.get_model_loader() (model_executor/model_loader/)
  ├── model_executor.layers.* (attention/linear/rotary_embedding/...)
  ├── model_executor.models.interfaces.*
  ├── LoRAModelRunnerMixin (v1/worker/)
  ├── KVConnectorModelRunnerMixin (v1/worker/)
  └── ECConnectorModelRunnerMixin (v1/worker/)
```

#### Model Executor 层依赖
```
model_loader.get_model_loader()
  └── models.registry.ModelRegistry (解析架构名)
      └── 具体模型类 (如 LlamaForCausalLM)

具体模型 (如 LlamaForCausalLM)
  ├── layers.attention.Attention
  ├── layers.linear.QKVParallelLinear
  ├── layers.rotary_embedding.RotaryEmbedding
  └── layers.quantization.QuantizationConfig
```

### 3.3 完整数据流向

```
1. 请求流入
   LLMEngine.add_request() / AsyncLLM.generate()
   → InputProcessor 生成 EngineCoreRequest

2. 调度阶段
   EngineCore 调用 Scheduler.schedule()
   → Scheduler 与 KVCacheManager 交互分配 KV 块
   → Scheduler 处理前缀缓存命中
   → 产出 SchedulerOutput

3. 执行阶段
   Executor.execute_model(SchedulerOutput)
   → collective_rpc("execute_model", ...) 分发到所有 Worker

4. 模型前向
   Worker.execute_model()
   → GPUModelRunner.execute_model()
   → 构建 GPUInputBatch
   → model.forward() 生成 hidden states
   → LogitsProcessor 计算 logits

5. 采样阶段
   Worker.sample_tokens()
   → Sampler 采样产出 token
   → 产出 ModelRunnerOutput

6. 输出回流
   OutputProcessor 将 ModelRunnerOutput 转换为 RequestOutput
   → detokenization
   → 流式输出聚合
   → 返回给用户
```

---

## 四、关键架构设计模式

### 4.1 策略模式（Strategy Pattern）

**Executor 层**：`UniProcExecutor`、`MultiprocExecutor`、`RayDistributedExecutor` 都继承自 `Executor` 抽象基类，上层代码无感知切换分布式策略。

**Attention 后端**：通过 `get_attn_backend()` 根据硬件和能力自动选择最优注意力实现（FlashAttention、FlashInfer、Triton、ROCm）。

### 4.2 注册表模式（Registry Pattern）

**ModelRegistry**：在 `models/registry.py` 中维护架构名到模型类的映射，支持延迟加载避免子进程重复初始化 CUDA。

### 4.3 协议/接口模式（Protocol Pattern）

**VllmModel Protocol**：`models/interfaces_base.py` 定义所有模型必须实现的接口（embed_input_ids、forward），不强制继承特定基类。

### 4.4 混入模式（Mixin Pattern）

**GPUModelRunner** 通过混入扩展能力：
- `LoRAModelRunnerMixin`：LoRA 适配器支持
- `KVConnectorModelRunnerMixin`：KV Cache 传输支持
- `ECConnectorModelRunnerMixin`：弹性连接器支持

---

## 五、外部依赖与平台支持

### 5.1 构建依赖
- **CMake + Ninja**：C++/CUDA 内核编译
- **PyTorch 2.10.0**：深度学习框架
- **setuptools-scm**：版本管理

### 5.2 运行时依赖
- **CUDA / ROCm**：GPU 计算
- **NCCL / pynccl**：多卡通信
- **Ray**：分布式集群
- **Transformers / Tokenizers**：HuggingFace 生态集成
- **Triton**：自定义 GPU kernel
- **FlashAttention / FlashInfer**：高效注意力实现

### 5.3 平台支持
| 平台 | 路径 | 状态 |
|------|------|------|
| CUDA | `platforms/cuda.py` | 主要平台，功能最全 |
| ROCm | `platforms/rocm.py` | AMD GPU 支持 |
| CPU | `platforms/cpu.py` | CPU 推理支持 |
| TPU | `platforms/tpu.py` | Google TPU 支持 |
| XPU | `platforms/xpu.py` | Intel XPU 支持 |

---

## 六、V0 vs V1 架构对比

| 模块 | V0（已移除） | V1（当前） | 变化说明 |
|------|-------------|-----------|----------|
| Engine | `vllm/engine/llm_engine.py`（完整实现） | `vllm/v1/engine/llm_engine.py` | 完全重写，`engine/` 下仅存别名 |
| Async Engine | `vllm/engine/async_llm_engine.py` | `vllm/v1/engine/async_llm.py` | 重命名为 AsyncLLM |
| Scheduler | 无独立目录 | `vllm/v1/core/sched/` | 新增模块化调度层 |
| BlockManager | `vllm/core/block_manager.py` | `vllm/v1/core/kv_cache_manager.py` + `block_pool.py` | 重构为 KVCacheManager |
| Worker | `vllm/worker/` | `vllm/v1/worker/` | 完全重写，新增 GPUModelRunner |
| Executor | 无独立目录 | `vllm/v1/executor/` | 新增策略模式执行器层 |
| Attention | `vllm/model_executor/layers/attention/` | `vllm/v1/attention/` + `vllm/model_executor/layers/attention/` | 新增 V1 后端选择器 |
| Sampling | `vllm/sampling_params.py`（仅参数） | `vllm/v1/sample/`（完整采样器） | 新增模块化采样层 |

---

## 七、总结

vLLM V1 架构采用清晰的分层设计，各层职责明确：

| 层级 | 目录 | 核心职责 |
|------|------|----------|
| **API 层** | `entrypoints/` | 用户接口（Python API、OpenAI API、CLI） |
| **Engine 层** | `v1/engine/` | 引擎门面，处理同步/异步语义、I/O 处理 |
| **Core 层** | `v1/core/` | 调度中枢，请求调度、KV Cache 管理、前缀缓存 |
| **Executor 层** | `v1/executor/` | 分布式策略，屏蔽单卡/多卡/Ray 差异 |
| **Worker 层** | `v1/worker/` | 设备抽象，调度输出 → 张量计算 |
| **Model Executor 层** | `model_executor/` | 计算内核，模型定义、算子实现、加载逻辑 |
| **Attention 后端** | `v1/attention/` | 注意力高效实现（FlashAttention、Triton 等） |
| **采样层** | `v1/sample/` | 采样策略（Top-K/Top-P、惩罚、推测解码） |

**最核心的依赖路径**：
```
EngineCoreClient → EngineCore → Scheduler + Executor → Worker → ModelRunner → model_executor/models + layers
```

这种分层设计使得 vLLM 能够灵活支持多种分布式策略、多种硬件平台和多种模型架构，同时通过 PagedAttention 和 Prefix Caching 等创新技术实现高效的 KV Cache 管理，显著提升推理吞吐量。
