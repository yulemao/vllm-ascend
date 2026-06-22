# vLLM-Ascend 架构分析报告

> 分析日期：2026-05-16
> 分析对象：vLLM-Ascend（昇腾 NPU 后端插件）
> 代码路径：vllm-ascend/vllm_ascend/

---

## 一、总体架构概述

### 1.1 项目定位

**vllm-ascend** 是 **vLLM 的昇腾 NPU 后端插件**，采用 **Out-of-Tree (OOT)** 架构模式。它不是 vLLM 的分支（fork），而是以 vLLM 为上游依赖，通过 Python 的继承、运行时注册和 Monkey-patch 机制，在不修改 vLLM 核心代码的前提下，完成对华为昇腾 NPU 的全面适配。

### 1.2 架构模式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           vLLM 主框架（上游依赖）                            │
│  vllm/entrypoints/  vllm/v1/engine/  vllm/v1/core/  vllm/v1/executor/      │
│  vllm/model_executor/  vllm/platforms/  vllm/distributed/                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ entry_points 插件注册
                                    │ import 引用（1911+ 条）
                                    │ 基类继承
                                    │ Monkey-patch 运行时替换
┌─────────────────────────────────────────────────────────────────────────────┐
│                        vLLM-Ascend（NPU 后端插件）                           │
│  vllm_ascend/platform.py (NPUPlatform)                                     │
│  vllm_ascend/worker/ (NPUWorker/NPUModelRunner)                            │
│  vllm_ascend/attention/ (AscendAttentionBackend)                           │
│  vllm_ascend/ops/ (NPU 算子替换)                                            │
│  vllm_ascend/patch/ (Monkey-patch 系统)                                     │
│  vllm_ascend/distributed/ (HCCL 通信)                                       │
│  vllm_ascend/compilation/ (ACL Graph)                                       │
│  csrc/ (CANN 自定义算子，1410+ 文件)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           华为昇腾 NPU 硬件栈                                │
│  torch-npu │ CANN │ HCCL │ AscendC │ ATB │ ACL                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 插件注册机制

vllm-ascend 通过 `setup.py` 的 `entry_points` 向 vllm 注册多个插件：

```python
"vllm.platform_plugins": [
    "ascend = vllm_ascend:register"              # 注册 NPUPlatform
],
"vllm.general_plugins": [
    "ascend_kv_connector = vllm_ascend:register_connector",         # KV 传输连接器
    "ascend_model_loader = vllm_ascend:register_model_loader",      # 模型加载器
    "ascend_service_profiling = vllm_ascend:register_service_profiling",  # 性能分析
    "ascend_0day_model = vllm_ascend:register_model"                # 新模型注册
]
```

vllm 在启动时通过 `importlib.metadata.entry_points("vllm.platform_plugins")` 自动发现并加载 vllm-ascend。

### 1.4 核心依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| `vllm` | - | 上游核心框架（运行时必需） |
| `torch` | `==2.9.0` | PyTorch |
| `torch-npu` | `==2.9.0` | 华为昇腾 PyTorch 适配 |
| `torchvision` | `==0.24.0` | - |
| `triton-ascend` | `==3.2.0` | Ascend Triton |
| `transformers` | `>=4.57.4` | HuggingFace Transformers |
| `xgrammar` | `>=0.1.30` | 结构化解码 |
| `compressed_tensors` | `>=0.11.0` | 量化支持 |

---

## 二、vllm-ascend 对 vllm 的核心修改

### 2.1 硬件抽象层替换

| vllm (GPU) | vllm-ascend (NPU) | 说明 |
|-----------|-------------------|------|
| CUDA (`torch.cuda`) | NPU (`torch.npu`) | 设备切换 |
| CUDA Graph | ACL Graph (`compilation/acl_graph.py`) | 图捕获与重放 |
| NCCL | HCCL / FlashComm2 (`distributed/device_communicators/`) | 分布式通信 |
| Cutlass / Triton CUDA kernels | CANN Custom Ops (`csrc/`) | 自定义算子 |
| `device_name = "cuda"` | `device_name = "npu"` | 平台标识 |
| `dispatch_key = "CUDA"` | `dispatch_key = "PrivateUse1"` | PyTorch dispatch |

### 2.2 编译系统替换

| vllm | vllm-ascend |
|------|-------------|
| `torch.compile()` + Inductor | Eager 模式（NPU 不兼容 inductor） |
| CUDA Graph (`CUDAGraphWrapper`) | ACL Graph (`ACLGraphWrapper`) |
| `VllmInductorPass` 图融合 | NPU 专属图融合 pass（`compilation/passes/`） |
| `norm_quant_fusion_pass.py` | LayerNorm + Quant 融合 |
| `qknorm_rope_fusion_pass.py` | QKNorm + RoPE 融合 |
| `allreduce_rmsnorm_fusion_pass.py` | AllReduce + RMSNorm 融合 |
| `sequence_parallelism.py` | 序列并行优化 |

### 2.3 Attention 后端替换

| vllm Attention 后端 | vllm-ascend 对应后端 | 说明 |
|---------------------|----------------------|------|
| `FlashAttentionBackend` | `AscendAttentionBackend` | 标准 Paged Attention |
| `FlashInferBackend` | `AscendAttentionBackend` | - |
| `MLAAttentionImpl` (FlashAttn/FlashInfer) | `AscendMLAAttentionImpl` | Multi-Latent Attention |
| `index_topk` 稀疏注意力 | `AscendSFAImpl` | Sparse Flash Attention |
| DeepSeek 稀疏注意力 | `AscendDSAImpl` | DeepSeek Sparse Attention |
| 基础 PCP/DCP | `AscendAttentionCPImpl` | 扩展的上下文并行 |

### 2.4 Worker/Runner 替换

| vllm 类 | vllm-ascend 类 | 继承关系 |
|---------|---------------|----------|
| `WorkerBase` | `NPUWorker` | 直接继承 |
| `GPUModelRunner` (v1) | `NPUModelRunner` (v1) | 直接继承 |
| `GPUModelRunner` (v2) | `NPUModelRunner` (v2) | 直接继承 |
| `GPUWorker` | `NPUWorker310` | 继承 NPUWorker（310P 专用） |
| `GPUWorker` | `XliteWorker` | 继承 NPUWorker（Xlite 专用） |

### 2.5 分布式通信替换

| vllm (NCCL) | vllm-ascend (HCCL) |
|-------------|-------------------|
| `NcclCommunicator` | `NPUCommunicator` (`distributed/device_communicators/npu_communicator.py`) |
| `pyNccl` | `pyHCCL` (`distributed/device_communicators/pyhccl.py`) |
| `init_model_parallel` | `init_ascend_model_parallel` (`distributed/parallel_state.py`) |

---

## 三、模块对应与依赖关系

### 3.1 模块对应总表

| vllm-ascend 模块 | 对应 vllm 模块 | 关系类型 | 说明 |
|------------------|---------------|----------|------|
| `vllm_ascend/platform.py` | `vllm/platforms/interface.py` | 继承 + 注册 | NPUPlatform 继承 Platform，OOT 注册 |
| `vllm_ascend/worker/worker.py` | `vllm/v1/worker/gpu_worker.py` | 继承 | NPUWorker 继承 WorkerBase |
| `vllm_ascend/worker/model_runner_v1.py` | `vllm/v1/worker/gpu_model_runner.py` | 继承 | NPUModelRunner v1 继承 GPUModelRunner |
| `vllm_ascend/worker/v2/model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | 继承 | NPUModelRunner v2 继承 GPUModelRunner |
| `vllm_ascend/attention/attention_v1.py` | `vllm/v1/attention/backend.py` | 继承 + 注册 | AscendAttentionBackend 继承 AttentionBackend |
| `vllm_ascend/attention/mla_v1.py` | `vllm/model_executor/layers/attention/mla_attention.py` | 继承 | AscendMLAAttentionImpl 继承 MLAAttentionImpl |
| `vllm_ascend/attention/dsa_v1.py` | vllm DSA 概念 | 自定义抽象 | DSAAttentionImpl 自定义抽象类 |
| `vllm_ascend/attention/sfa_v1.py` | vllm 稀疏注意力 | 继承 + 扩展 | AscendSFABackend |
| `vllm_ascend/ops/` | `vllm/model_executor/layers/` | 替换 | NPU 算子替换 GPU 算子 |
| `vllm_ascend/distributed/` | `vllm/distributed/` | 扩展 | HCCL 替换 NCCL，新增 KV Transfer |
| `vllm_ascend/compilation/` | `vllm/compilation/` | 替换 | ACL Graph 替换 CUDA Graph |
| `vllm_ascend/quantization/` | `vllm/model_executor/layers/quantization/` | 扩展 | NPU 专属量化方案 |
| `vllm_ascend/sample/` | `vllm/v1/sample/` | 继承 + 扩展 | NPU 采样适配 |
| `vllm_ascend/spec_decode/` | `vllm/v1/spec_decode/` | 继承 + 扩展 | NPU 投机解码 |
| `vllm_ascend/models/` | `vllm/model_executor/models/` | 扩展 | DeepSeek V4 等模型覆盖 |
| `vllm_ascend/lora/` | `vllm/lora/` | 扩展 | PunicaWrapperNPU |
| `vllm_ascend/transformers_utils/` | `vllm/transformers_utils/` | 扩展 | 配置适配 |
| `vllm_ascend/device_allocator/` | `vllm/device_allocator/` | 扩展 | NPU 内存分配 |
| `vllm_ascend/core/` | `vllm/v1/core/` | 扩展 | 动态 batch 调度器 |
| `vllm_ascend/patch/` | - | 特有 | Monkey-patch 系统 |
| `vllm_ascend/eplb/` | - | 特有 | 专家并行负载均衡 |
| `vllm_ascend/kv_offload/` | - | 特有 | KV Cache 卸载 |
| `vllm_ascend/xlite/` | - | 特有 | Xlite 虚拟化支持 |
| `vllm_ascend/_310p/` | - | 特有 | Atlas 310P 芯片专用 |
| `vllm_ascend/model_loader/netloader/` | - | 特有 | 网络弹性模型加载 |
| `vllm_ascend/model_loader/rfork/` | - | 特有 | rfork 加载协议 |
| `csrc/` | `vllm/csrc/` | 替换 | CANN 算子替换 CUDA kernel |

### 3.2 vllm-ascend 未实现/复用的 vllm 模块

vllm-ascend **直接复用** vllm 的以下模块，不做修改：

| vllm 模块 | 复用方式 | 说明 |
|-----------|---------|------|
| `vllm/v1/engine/` | 直接 import | LLMEngine、AsyncLLM、EngineCore、Executor 等直接复用 |
| `vllm/v1/executor/` | 配置指向 | 通过 `worker_cls` 配置指向 NPUWorker，不实现独立 Executor |
| `vllm/entrypoints/` | 直接 import | API 入口（LLM 类、OpenAI API、CLI）直接复用 |
| `vllm/config/` | 直接 import | 配置系统直接复用，NPUPlatform 做适配修改 |
| `vllm/inputs/` | 直接 import | 输入处理直接复用 |
| `vllm/outputs.py` | 直接 import | 输出数据结构直接复用 |
| `vllm/sampling_params.py` | 直接 import | 采样参数直接复用 |
| `vllm/tokenizers/` | 直接 import | 分词器直接复用 |

---

## 四、核心类继承与调用关系

### 4.1 类继承关系图

```
# ==================== Platform 层 ====================
vllm.platforms.Platform (vllm)
└── vllm_ascend.platform.NPUPlatform
    - device_name = "npu"
    - dispatch_key = "PrivateUse1"
    - _enum = PlatformEnum.OOT

# ==================== Engine 层（复用 vllm）====================
vllm.v1.engine.llm_engine.LLMEngine (vllm)
vllm.v1.engine.async_llm.AsyncLLM (vllm)
vllm.v1.engine.core.EngineCore (vllm)
    └── 调用 Executor.execute_model()
    └── 调用 Scheduler.schedule()

# ==================== Executor 层（复用 vllm + patch）====================
vllm.v1.executor.abstract.Executor (vllm)
└── vllm.v1.executor.uniproc_executor.UniProcExecutor (vllm)
└── vllm.v1.executor.multiproc_executor.MultiprocExecutor (vllm)
    └── AscendMultiprocExecutor (通过 patch 覆盖)

# ==================== Worker 层 ====================
vllm.v1.worker.worker_base.WorkerBase (vllm)
└── vllm_ascend.worker.worker.NPUWorker
    ├── vllm_ascend._310p.worker_310p.NPUWorker310
    └── vllm_ascend.xlite.xlite_worker.XliteWorker

# ==================== ModelRunner 层 ====================
vllm.v1.worker.gpu_model_runner.GPUModelRunner (vllm)
└── vllm_ascend.worker.model_runner_v1.NPUModelRunner (v1)

vllm.v1.worker.gpu.model_runner.GPUModelRunner (vllm)
└── vllm_ascend.worker.v2.model_runner.NPUModelRunner (v2)
    ├── vllm_ascend.worker.v2.model_runner.NPUModelRunner310
    └── vllm_ascend.xlite.xlite_model_runner.XliteModelRunner

# ==================== Attention 层 ====================
vllm.v1.attention.backend.AttentionBackend (vllm)
├── vllm_ascend.attention.attention_v1.AscendAttentionBackend
├── vllm_ascend.attention.mla_v1.AscendMLABackend
├── vllm_ascend.attention.sfa_v1.AscendSFABackend
└── vllm_ascend.attention.dsa_v1.AscendDSABackend

vllm.v1.attention.backend.AttentionImpl (vllm)
├── vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl
│   ├── vllm_ascend.attention.attention_v1.AscendC8AttentionBackendImpl
│   └── vllm_ascend.attention.attention_v1.AscendAttentionCPImpl
└── vllm_ascend.attention.mla_v1.AscendMLAAttentionImpl

# ==================== 算子层 ====================
vllm.model_executor.layers.activation.SiluAndMul (vllm)
└── vllm_ascend.ops.activation.AscendSiluAndMul

vllm.model_executor.layers.layernorm.RMSNorm (vllm)
└── vllm_ascend.ops.layernorm.AscendRMSNorm

vllm.model_executor.layers.rotary_embedding.RotaryEmbedding (vllm)
└── vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding

vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod (vllm)
└── vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod

# ==================== 分布式通信层 ====================
vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase (vllm)
└── vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator

# ==================== 编译层 ====================
vllm.compilation.wrapper.CUDAGraphWrapper (vllm)
└── vllm_ascend.compilation.acl_graph.ACLGraphWrapper

vllm.compilation.backends.EagerAdaptor (vllm)
└── vllm_ascend.compilation.compiler_interface.AscendCompiler

# ==================== 量化层 ====================
vllm.model_executor.layers.quantization.base_config.QuantizationConfig (vllm)
└── vllm_ascend.quantization.methods.base.AscendLinearScheme
    ├── W8A8DynamicLinearScheme
    ├── W8A8StaticLinearScheme
    ├── W8A8MXFP8LinearScheme
    ├── W4A16LinearScheme
    ├── W4A8LinearScheme
    ├── W4A4FlatQuantLinearScheme
    └── KVC8LinearScheme
```

### 4.2 调用链（从用户 API 到 NPU 执行）

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  vllm.entrypoints.llm.LLM (直接复用 vllm)                                │
│  vllm.entrypoints.openai.api_server (直接复用 vllm)                      │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  vllm.v1.engine.llm_engine.LLMEngine (直接复用 vllm)                     │
│  vllm.v1.engine.async_llm.AsyncLLM (直接复用 vllm)                       │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  vllm.v1.engine.core.EngineCore (直接复用 vllm)                          │
│  - 通过 NPUPlatform.get_worker_cls() 获取 NPUWorker                      │
│  - 通过 NPUPlatform.get_attn_backend_cls() 获取 AscendAttentionBackend   │
│  - 通过 NPUPlatform.get_compile_backend() 获取 AscendCompiler            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ├───► ┌────────────────────────────────────────────────────────────┐
    │     │  vllm.v1.core.sched.scheduler.Scheduler (复用 vllm)        │
    │     │  - 调度逻辑与 vllm 一致                                    │
    │     │  - 可能通过 patch 注入自定义调度策略                       │
    │     └────────────────────────────────────────────────────────────┘
    │                           │
    │                           ▼
    │     ┌────────────────────────────────────────────────────────────┐
    │     │  vllm.v1.core.kv_cache_manager.KVCacheManager (复用 vllm)  │
    │     │  vllm.v1.core.block_pool.BlockPool (复用 vllm)             │
    │     │  - 可能通过 patch 扩展 DSA/Sparse C8 KV Cache             │
    │     └────────────────────────────────────────────────────────────┘
    │
    └───► ┌────────────────────────────────────────────────────────────┐
          │  vllm.v1.executor.multiproc_executor.MultiprocExecutor     │
          │  (复用 vllm，可能通过 patch 覆盖为 AscendMultiprocExecutor)│
          │  - collective_rpc("execute_model", SchedulerOutput)        │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm_ascend.worker.worker.NPUWorker                       │
          │  - init_device(): 初始化 torch.npu                         │
          │  - load_model(): 加载模型到 NPU                            │
          │  - adapt_patch(): 应用 Monkey-patch                        │
          │  - register_ascend_customop(): 注册自定义算子              │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm_ascend.worker.model_runner_v1.NPUModelRunner (v1)    │
          │  或 vllm_ascend.worker.v2.model_runner.NPUModelRunner (v2) │
          │  - 构建 NPUInputBatch / AscendInputBatch                   │
          │  - 调用 model.forward()                                    │
          │  - 使用 ACLGraph 替代 CUDA Graph                           │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm.model_executor.models.llama.LlamaForCausalLM (复用)  │
          │  或 vllm_ascend.models.layer.attention.DSAAttention (覆盖) │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm_ascend.ops.activation.AscendSiluAndMul               │
          │  vllm_ascend.ops.layernorm.AscendRMSNorm                   │
          │  vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding    │
          │  vllm_ascend.ops.fused_moe.*                               │
          │  vllm_ascend.ops.linear.*                                  │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm_ascend.attention.attention_v1.AscendAttentionBackend │
          │  或 vllm_ascend.attention.mla_v1.AscendMLABackend          │
          │  或 vllm_ascend.attention.sfa_v1.AscendSFABackend          │
          │  或 vllm_ascend.attention.dsa_v1.AscendDSABackend          │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  csrc/ CANN 自定义算子                                      │
          │  - attention/sparse_flash_attention                        │
          │  - mc2/dispatch_ffn_combine                                │
          │  - moe/moe_grouped_matmul                                  │
          │  - gmm/grouped_matmul_swiglu_quant                         │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌────────────────────────────────────────────────────────────┐
          │  vllm_ascend.sample.sampler.AscendSampler (v2)             │
          │  或 vllm.v1.sample.sampler.Sampler (复用 vllm，v1)         │
          └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                              返回 token + logprobs
```

---

## 五、vllm-ascend 特有模块详解

### 5.1 patch/ — Monkey-patch 系统

vllm-ascend 最独特的架构设计之一是 **Monkey-patch 系统**，用于在不动 vllm 源码的情况下覆盖其行为。

#### Platform Patches（Worker 启动前应用）

| Patch 文件 | 作用 |
|-----------|------|
| `patch/platform/patch_distributed.py` | 310P tensor alignment 适配 |
| `patch/platform/patch_balance_schedule.py` | 调度策略平衡 |
| `patch/platform/patch_multiproc_executor.py` | daemon=False 适配 EPLB |
| `patch/platform/patch_kv_cache_interface.py` | DSA/Sparse C8 KV Cache 扩展 |
| `patch/platform/patch_deepseek_v4_agentic.py` | DeepSeek V4 tokenizer/tool parser backport |

#### Worker Patches（Worker 启动时应用）

| Patch 文件 | 作用 |
|-----------|------|
| `patch/worker/patch_triton.py` | 替换 Triton ops 为 NPU 版本 |
| `patch/worker/patch_v2/patch_triton.py` | v2 runner Triton ops 替换 |
| `patch/worker/patch_gdn_attn.py` | GDN prefill metadata 预构建 |
| `patch/worker/patch_distributed.py` | GroupCoordinator all_to_all 适配 |
| `patch/worker/patch_cudagraph.py` | FULL graph mode 适配 ACL Graph |
| `patch/worker/patch_qwen3_next.py` | Qwen3 算子融合 |
| `patch/worker/patch_qwen3_5.py` | Qwen3.5 算子融合 |
| `patch/worker/patch_qwen3vl.py` | Qwen3VL 算子融合 |
| `patch/worker/patch_minimax_m2.py` | MiniMax-M2 MoE/Attention/加载适配 |
| `patch/worker/patch_v2/patch_block_table.py` | v2 block_table 替换 |
| `patch/worker/patch_v2/patch_input_batch.py` | v2 input_batch 替换 |
| `patch/worker/patch_v2/patch_model_state.py` | v2 model_state 替换 |

**Patch 应用机制**：`NPUWorker.adapt_patch()` 在初始化时动态加载并应用这些 patch，通过替换模块属性实现运行时覆盖。

### 5.2 eplb/ — 专家并行负载均衡

| 文件 | 功能 |
|------|------|
| `eplb/eplb_strategy.py` | EPLB 策略定义（动态专家权重重分布） |
| `eplb/device_transfer_loader.py` | 设备间专家权重传输加载器 |

**功能**：在 MoE 模型推理时，动态平衡各 NPU 上的专家负载，避免某些设备过载。

### 5.3 kv_offload/ — KV Cache 卸载

| 文件 | 功能 |
|------|------|
| `kv_offload/kv_offload_manager.py` | KV Cache 卸载管理器 |
| `kv_offload/npu_offload.py` | NPU-CPU 卸载策略 |

**功能**：在显存不足时将 KV Cache 卸载到 CPU 内存，支持大上下文窗口。

### 5.4 xlite/ — Xlite 虚拟化支持

| 文件 | 功能 |
|------|------|
| `xlite/xlite_worker.py` | XliteWorker，适配 openEuler GVirt |
| `xlite/xlite_model_runner.py` | XliteModelRunner |

**功能**：支持在 openEuler Xlite 虚拟化环境中运行 vLLM，用于云原生部署。

### 5.5 _310p/ — Atlas 310P 边缘芯片

| 子目录 | 功能 |
|--------|------|
| `_310p/attention/` | 310P 专用 attention 实现 |
| `_310p/ops/` | 310P 专用算子 |
| `_310p/quantization/` | 310P 专用量化 |
| `_310p/sample/` | 310P 专用采样 |
| `_310p/worker_310p.py` | NPUWorker310 |

**功能**：针对昇腾 310P 低功耗边缘芯片的完整适配子系统。

### 5.6 model_loader/netloader/ — 网络弹性模型加载

| 文件 | 功能 |
|------|------|
| `model_loader/netloader/elastic_loader.py` | 弹性模型加载器 |
| `model_loader/netloader/netloader_pg.py` | 网络加载进程组 |

**功能**：支持从网络分布式加载大模型权重，无需本地完整存储。

### 5.7 model_loader/rfork/ — rfork 加载协议

| 文件 | 功能 |
|------|------|
| `model_loader/rfork/seed_protocol.py` | Seed 协议实现 |
| `model_loader/rfork/transfer_backend.py` | 传输后端 |

**功能**：基于 rfork 的模型权重快速分发协议。

---

## 六、数据流向对比

### 6.1 vllm (GPU) 数据流

```
Request → LLMEngine → EngineCore → Scheduler → Executor → GPUWorker
→ GPUModelRunner → model.forward() → FlashAttention → CUDA kernel
→ Sampler → RequestOutput
```

### 6.2 vllm-ascend (NPU) 数据流

```
Request → LLMEngine (复用 vllm) → EngineCore (复用 vllm)
→ Scheduler (复用 vllm，可能 patch)
→ Executor (复用 vllm，可能 patch)
→ NPUWorker → NPUModelRunner → model.forward()
→ AscendAttentionBackend → CANN Custom Op (csrc/)
→ AscendSampler → RequestOutput
```

**关键差异点**：
- 绿色模块：直接复用 vllm，无修改
- 蓝色模块：继承 vllm 基类并重写
- 橙色模块：vllm-ascend 特有模块

---

## 七、csrc/ 自定义算子详解

vllm-ascend 的 `csrc/` 目录包含 **1410+ 文件**，是 CANN 自定义算子库，与 vllm 的 `csrc/`（CUDA/Cutlass kernel）一一对应。

### 7.1 算子目录结构

| 目录 | 功能 | 对应 vllm CUDA 模块 |
|------|------|---------------------|
| `csrc/attention/` | Attention NPU 算子 | `vllm/csrc/attention/flash_attn/` |
| `csrc/mc2/` | MoE 通信-计算融合 | `vllm/csrc/moe/` |
| `csrc/moe/` | MoE 基础算子 | `vllm/csrc/moe/` |
| `csrc/gmm/` | Grouped MatMul | `vllm/csrc/moe/` |
| `csrc/kernels/` | LoRA / Prefix 内核 | `vllm/csrc/lora/`、`vllm/csrc/quantization/` |
| `csrc/mla_preprocess/` | MLA 预处理 | `vllm/csrc/attention/` |
| `csrc/batch_matmul_transpose/` | BMM 转置优化 | `vllm/csrc/ops/` |
| `csrc/aclnn_torch_adapter/` | PyTorch-NPU 桥接 | - |

### 7.2 Ascend CANN 算子开发规范

每个算子遵循标准的三层结构：

```
op_host/        # Host 侧（CPU）
  ├── *_def.cpp       # 算子定义（输入输出、数据类型）
  ├── *_proto.cpp     # 原型注册（注册到 CANN）
  └── *_tiling.cpp    # Tiling 算法（数据切分策略）

op_kernel/      # Device 侧（NPU AscendC）
  ├── *.cpp           # 主核函数入口
  └── arch32/         # 按 NPU 架构分离的实现
      └── *.cpp
      arch35/
      └── *.cpp

op_api/         # ACLNN API 封装
  └── *.cpp           # 供 PyTorch 调用的 API 层
```

### 7.3 关键算子列表

| 算子 | 功能 | 对应场景 |
|------|------|----------|
| `sparse_flash_attention` | 稀疏 Flash Attention | 长序列推理 |
| `compressor` | KV Cache 压缩 | KV Cache 量化 |
| `lightning_indexer_quant` | 量化索引器 | 稀疏注意力索引 |
| `rms_norm_dynamic_quant` | 动态量化 RMSNorm | LayerNorm 量化 |
| `dispatch_ffn_combine` | MoE dispatch + FFN + combine | MoE 通信计算融合 |
| `matmul_allreduce_add_rmsnorm` | MatMul + AllReduce + Add + RMSNorm | 融合算子 |
| `moe_grouped_matmul` | MoE Grouped MatMul | MoE 计算 |
| `grouped_matmul_swiglu_quant` | Grouped MatMul + SwiGLU + 量化 | MoE 量化 |
| `mla_preprocess_kernel` | MLA 预处理 | DeepSeek MLA |
| `batch_matmul_transpose_kernel` | BMM 转置 | 注意力优化 |
| `bgmv_expand` / `bgmv_shrink` | LoRA BGMV | LoRA 适配 |
| `inplace_partial_rotary_mul` | 原地 RoPE | 位置编码优化 |

---

## 八、量化模块详解

### 8.1 架构设计

```python
# 抽象基类
AscendLinearScheme(ABC)
  ├── @abstractmethod apply_weights()
  └── @abstractmethod apply_activations()

# 注册表（按 (quant_type, layer_type) 索引）
_SCHEME_REGISTRY: dict[tuple[str, str], type]

# 具体实现（注册到 registry）
@register_scheme("W8A8_DYNAMIC", "linear")
class W8A8DynamicLinearScheme(AscendLinearScheme): ...
```

### 8.2 支持的量化方法

| 方法 | 文件 | 权重位宽 | 激活位宽 | KV Cache | 说明 |
|------|------|----------|----------|----------|------|
| W8A8_DYNAMIC | `methods/w8a8_dynamic.py` | 8 | 8 (动态) | - | 动态 per-token 量化 |
| W8A8_STATIC | `methods/w8a8_static.py` | 8 | 8 (静态) | - | 静态量化 |
| W8A8_MXFP8 | `methods/w8a8_mxfp8.py` | 8 | 8 (MXFP8) | - | MXFP8 格式 |
| W8A8_PDMIX | `methods/w8a8_pdmix.py` | 8 | 8 (混合) | - | PD 分离混合量化 |
| W4A16 | `methods/w4a16.py` | 4 | 16 | - | 4-bit 权重量化 |
| W4A8 | `methods/w4a8.py` | 4 | 8 | - | 4-bit 权重 + 8-bit 激活 |
| W4A4_FLATQUANT | `methods/w4a4_flatquant.py` | 4 | 4 | - | FlatQuant |
| W4A4_LAOS_DYNAMIC | `methods/w4a4_laos_dynamic.py` | 4 | 4 | - | LAOS 动态量化 |
| KV_C8 | `methods/kv_c8.py` | - | - | 8 | KV Cache C8 量化 |

### 8.3 与 vllm 量化模块的关系

| vllm | vllm-ascend |
|------|-------------|
| `QuantizationConfig` (基类) | `AscendLinearScheme` (基类) |
| `AWQConfig` / `GPTQConfig` / `FP8Config` | `W8A8DynamicLinearScheme` / `W4A16LinearScheme` / `KVC8LinearScheme` |
| 注册到 vllm 量化注册表 | 注册到 vllm-ascend 私有注册表 `_SCHEME_REGISTRY` |

---

## 九、测试架构

### 9.1 测试目录结构

```
tests/
├── ut/                      # 单元测试
│   ├── ut_attention.py
│   ├── ut_worker.py
│   ├── ut_ops.py
│   └── ...
└── e2e/                     # 端到端测试
    ├── singlecard/          # 单卡测试
    │   ├── test_basic_correctness.py
    │   ├── test_quantization.py
    │   └── ...
    ├── multicard/           # 多卡测试
    │   ├── test_tp.py       # Tensor Parallel
    │   ├── test_pp.py       # Pipeline Parallel
    │   └── ...
    ├── 310p/                # Atlas 310P 测试
    ├── weekly/              # 周级回归测试
    └── nightly/             # 夜间全量测试
```

### 9.2 与 vllm 测试的关系

大量测试直接改编自 vllm：
- `tests/e2e/singlecard/test_basic_correctness.py` → 改编自 `vllm/tests/basic_correctness/test_basic_correctness.py`
- `examples/offline_inference_npu.py` → 改编自 `vllm/examples/basic/offline_inference/basic.py`

---

## 十、总结

### 10.1 架构设计哲学

vllm-ascend 采用 **"继承 + 补丁 + 注册"** 的三重适配策略：

1. **显式继承**：核心类直接继承 vllm 基类，保持接口兼容（Worker、ModelRunner、AttentionBackend）
2. **运行时补丁**：`patch/` 目录下 30+ 个 patch 文件动态替换 vllm 内部函数和类
3. **后端注册**：通过 `@register_backend` 和 `entry_points` 将 NPU 实现注册到 vllm 的选择器

### 10.2 模块对应总览

```
vllm (GPU)                              vllm-ascend (NPU)
─────────────────────────────────────────────────────────────────
Platform (CUDA/ROCm)        ──────►    NPUPlatform
  - device_name="cuda"                 - device_name="npu"
  - dispatch_key="CUDA"                - dispatch_key="PrivateUse1"

Engine (LLMEngine/AsyncLLM) ──────►    直接复用（无修改）
Core (Scheduler/BlockPool)  ──────►    直接复用 + patch 扩展
Executor (UniProc/MultiProc)──────►    直接复用 + patch 覆盖

GPUWorker / WorkerBase      ──────►    NPUWorker
GPUModelRunner              ──────►    NPUModelRunner (v1/v2)
  - CUDA Graph                         - ACL Graph
  - CUDA kernel                        - CANN Custom Op

FlashAttentionBackend       ──────►    AscendAttentionBackend
FlashAttentionImpl          ──────►    AscendAttentionBackendImpl
MLAAttentionImpl            ──────►    AscendMLAAttentionImpl

NCCL                        ──────►    HCCL / FlashComm2
CUDAGraphWrapper            ──────►    ACLGraphWrapper
csrc/ (CUDA/Cutlass)        ──────►    csrc/ (CANN/AscendC)

新增模块：
  - eplb/           (专家并行负载均衡)
  - kv_offload/     (KV Cache 卸载)
  - xlite/          (Xlite 虚拟化)
  - _310p/          (Atlas 310P 支持)
  - patch/          (Monkey-patch 系统)
  - model_loader/netloader/ (网络弹性加载)
```

### 10.3 关键数据

| 指标 | vllm | vllm-ascend |
|------|------|-------------|
| Python 文件数 | ~1429 | ~672 |
| `__init__.py` 数 | - | 55 |
| csrc 文件数 | - | 1410+ |
| 对 vllm import 引用 | - | 1911+ 条 |
| 标注 "Adapted from vllm" 的文件 | - | 50+ |
| 含 vllm copyright 的文件 | - | 176+ |
| Monkey-patch 文件 | - | 30+ |
| 支持的量化方法 | - | 9 种 |
| Attention 后端 | 3-4 种 | 4 种 + CP |
| Worker 变体 | 1 种 | 3 种（NPU/310P/Xlite） |

### 10.4 核心依赖路径

```
vllm.entrypoints.LLM
  → vllm.v1.engine.LLMEngine
    → vllm.v1.engine.core.EngineCore
      → vllm.v1.core.sched.Scheduler (复用)
      → vllm.v1.executor.MultiprocExecutor (复用 + patch)
        → vllm_ascend.worker.NPUWorker
          → vllm_ascend.worker.NPUModelRunner
            → vllm.model_executor.models.* (复用)
              → vllm_ascend.ops.* (替换)
                → vllm_ascend.attention.* (替换)
                  → csrc/ CANN Custom Ops
                    → vllm_ascend.sample.* (替换/复用)
                      → RequestOutput
```

vllm-ascend 成功地在不修改 vllm 核心代码的前提下，通过 OOT 插件架构实现了对昇腾 NPU 的完整适配，是一个优秀的硬件后端插件设计范例。
