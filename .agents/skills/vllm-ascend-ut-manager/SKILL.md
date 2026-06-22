---
name: vllm-ascend-ut-manager
description: 当用户在 vllm-ascend 项目中修改代码后，需要自动检测、更新、新增或执行单元测试（UT）时触发。适用于：(1) 代码变更后查找并补充相关UT，(2) 为新增代码编写UT，(3) 执行UT验证，(4) 修复失败的UT，(5) 在无执行环境时对UT进行静态检查。
---

# vllm-ascend UT 管理助手

## 项目测试体系速览

- **测试框架**: pytest（主运行器）+ unittest（兼容）
- **UT 目录**: `tests/ut/`
- **源代码目录**: `vllm_ascend/`
- **映射规则**: 
  - `vllm_ascend/<模块>.py` → `tests/ut/test_<模块>.py`
  - `vllm_ascend/<子目录>/<模块>.py` → `tests/ut/<子目录>/test_<模块>.py`
- **基类**: `tests.ut.base.TestBase`（unittest 风格）、`PytestBase`（pytest 风格）
- **全局 Mock**: `tests/ut/conftest.py` 在导入阶段 mock `triton.runtime` 和 `torch_npu`
- **CI 执行命令参考**:
  ```bash
  pytest -sv --cov --cov-report=xml:unittests-coverage.xml tests/ut \
    --ignore tests/ut/model_loader/netloader/test_netloader_elastic.py \
    --ignore tests/ut/kv_connector/test_remote_prefill_lifecycle.py \
    --ignore tests/ut/kv_connector/test_remote_decode_lifecycle.py \
    --ignore tests/ut/core/test_scheduler_dynamic_batch.py \
    --ignore tests/ut/kv_connector/test_mooncake_connector.py \
    --ignore tests/ut/worker/test_worker_v1.py \
    --ignore tests/ut/worker/test_worker_multi_instance.py \
    --ignore tests/ut/kv_connector/test_mooncake_layerwise_connector.py
  ```
- **创建容器命令参考**:
  ```bash
  docker run -itd --privileged \
  --net=host \
  --name vllm-ascend \
  --shm-size=1g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  quay.io/ascend/vllm-ascend:deepseekv4
  ```
## 核心工作流程

### 步骤1: 创建docker，准备运行环境
1. 如果用户指定了要进入的容器名称，则跳过容器创建，直接进入用户指定的容器。否则按下列步骤创建并进入容器：检查当前可用的镜像，如果用户指定了使用哪个镜像则使用对应镜像，否则使用任意已有的vllm-ascend镜像，创建docker，默认名称为vllm-test。如果已经有这个名称的容器则在后面添加任意后缀避免重复。
2. 创建好docker后进入docker，进入/vllm-workspace路径。否则进行如下操作：
检测当前目录下是否有mycode目录，如果不存在则创建。进入这个目录。
3. 如果用户指定了无需更新代码，则跳过这一步。否则执行：检测mycode下是否存在vllm和vllm-ascend代码仓，如果不存在，则从 https://gitcode.com/qxxxw/vllm 和 https://gitcode.com/qxxxw/vllm-ascend 下载vllm和vllm-ascend代码。检查vllm和vllm-ascend的分支，需要vllm位于layerwise分支，vllm-ascend位于layerwise_ut_test分支，如果分支不对则切换到对应分支。使用git pull更新代码确保代码已是最新。
4. 进入vllm-ascend代码仓，设置环境变量：
```bash
export PYTHONPATH=/vllm-workspace/mycode/vllm-ascend:$PYTHONPATH
export PYTHONPATH=/vllm-workspace/mycode/vllm:$PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 步骤1: 检测代码变更与相关 UT
如果用户指定了这次运行的UT，则跳过这步。否则：
运行脚本获取当前代码变更（git diff）对应的测试文件：

```bash
python .agents/skills/vllm-ascend-ut-manager/scripts/find_related_tests.py --json
```

输出包含三类信息：
- `existing_tests`: 已存在的测试文件路径（需检查是否需要更新）
- `missing_tests`: 不存在的测试文件路径（需要新增 UT）
- `test_mappings`: 每个源文件到测试文件的映射关系

**处理策略**:
- 对于 `existing_tests`：阅读对应源文件的 diff，判断测试是否需要补充新用例。
- 对于 `missing_tests`：为新增源文件创建对应的 UT 文件。

### 步骤2: 检查 UT 执行环境
先下载需要的依赖：
```bash
pip install pytest pytest-cov pytest-mock
```
运行环境检查脚本：

```bash
python .agents/skills/vllm-ascend-ut-manager/scripts/check_ut_env.py --json
```

- 若返回 `can_run_ut: true` → 进入**步骤4: 执行 UT 验证**
- 若返回 `can_run_ut: false` → 进入**步骤3: 静态检查**

### 步骤3: 静态检查（无执行环境时）

对需要处理的 UT 文件运行静态检查：

```bash
# 检查单个测试文件
python .agents/skills/vllm-ascend-ut-manager/scripts/static_check.py \
  tests/ut/xxx/test_yyy.py --source vllm_ascend/xxx/yyy.py --json

# 检查整个目录
python .agents/skills/vllm-ascend-ut-manager/scripts/static_check.py \
  tests/ut/xxx/ --json
```

静态检查会分析：
1. **语法正确性**: AST 解析是否通过。
2. **导入完整性**: 测试文件是否导入了被测模块。
3. **语句/分支覆盖（启发式）**: 
   - 对比源文件和测试文件中定义的函数/类名，计算同名调用覆盖率。
   - 检查源文件中的 `if/for/while/try/with` 分支是否在测试中有对应结构。
   - 输出 `untested_items` 列表，必须为其补充测试用例。

**修复策略**:
- 语法错误：直接修正 Python 语法。
- 未覆盖项（`untested_items`）：针对每个未覆盖的函数/类，参照下面的 UT 编写规范新增测试用例。
- 分支覆盖不足：增加带有不同条件分支的测试用例（如 mock 不同返回值触发 if/else）。

静态检查完成后，向用户报告哪些测试已覆盖、哪些需要补充。

### 步骤4: 执行 UT 验证（有执行环境时）
执行前先阅读 `.github/workflows/_unit_test.yaml` 中定义的 `--ignore` 列表里的测试文件，这些是已知不稳定或有强环境依赖的，需要跳过这些测试用例。
对需要处理的 UT 执行测试：

```bash
# 执行单个测试文件
python .agents/skills/vllm-ascend-ut-manager/scripts/run_tests.py \
  tests/ut/xxx/test_yyy.py --json --cov

# 执行目录
python .agents/skills/vllm-ascend-ut-manager/scripts/run_tests.py \
  tests/ut/xxx/ --json --cov
```

**结果处理**:
- 若全部通过 → 任务完成。
- 若有失败 → 如果是环境问题导致的失败，不要改动现有环境，记录失败原因，跳过这个用例往下执行，如果是代码导致的失败，进入**步骤5: 修复失败 UT**。

### 步骤5: 修复失败 UT

根据 `run_tests.py` 的 JSON 输出中的 `failures` 和 `stdout_tail` 分析失败原因，常见类型与修复方法：

| 失败类型 | 可能原因 | 修复方法 |
|---|---|---|
| `ImportError` / `ModuleNotFoundError` | 测试导入了不存在的模块或路径错误 | 修正 import 路径 |
| `AttributeError` | mock 对象缺少被测代码访问的属性 | 补充 `MagicMock` 的属性和返回值 |
| `AssertionError` | 实际输出与预期不符 | 检查 mock 的 side_effect、源文件逻辑变更、修正断言 |
| `TypeError` | 函数签名变更 | 更新测试中的调用参数 |
| `RuntimeError` | torch_npu 相关调用未 mock | 对涉及的 NPU 算子添加 `@patch` |
| `FixtureLookupError` | pytest fixture 名称错误 | 检查 fixture 名称拼写或补充定义 |

**迭代修复**: 修改测试代码后，重新执行 `run_tests.py`，直到所有测试通过。最多迭代 5 轮，若仍失败则向用户汇报并请求协助。

## 新增 UT 编写规范

当需要为新增代码创建 UT 时，按以下步骤进行：

1. **确定测试文件路径**: 严格遵循映射规则。
2. **阅读源文件**: 提取需要测试的公共函数/类，分析其：
   - 正常输入输出
   - 边界条件（空输入、极大/极小值、异常形状）
   - 分支逻辑（if/else、try/except、循环）
   - 外部依赖（需要 mock 的 NPU 调用、分布式组、vllm_config）
3. **选择风格**: 
   - 类方法多 → `unittest` + `TestBase`
   - 纯函数/需参数化 → `pytest` + `PytestBase`/`fixture`
4. **编写测试**: 参照 `references/ut_templates.md` 中的模板和 Mock 最佳实践。
5. **验证**: 编写后先用 `static_check.py` 检查语法和覆盖率，再尽可能用 `run_tests.py` 执行。

### 覆盖率要求

- 新增代码的 PR 增量覆盖率目标为 **≥ 80%**。
- 必须覆盖：正常路径 + 至少一条异常/边界路径。
- 对源文件中的每个 `if` 分支，尽量设计 mock 条件使两侧都被执行。

## 注意事项

- 不要修改 `.github/workflows/_unit_test.yaml` 中定义的 `--ignore` 列表里的测试文件，这些是已知不稳定或有强环境依赖的。
- `tests/ut/conftest.py` 已全局 mock `triton.runtime` 和部分 `torch_npu`，但具体的算子调用（如 `torch_npu.npu_fast_gelu`）仍需在测试中按需 patch。
- 执行 UT 前设置环境变量（脚本已自动处理）：
  ```bash
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  ```
- 若测试需要真实 NPU（极少数情况），使用 `@pytest.mark.skipif(not torch.npu.is_available(), ...)`，但 UT 应尽量在无硬件环境下运行。
