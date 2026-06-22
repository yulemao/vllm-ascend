# vllm-ascend UT 模板与风格指南

## 文件头规范

所有测试文件必须包含以下 Apache License 头：

```python
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
# This file is a part of the vllm-ascend project.
#
```

## 模板 A: unittest 风格（推荐用于复杂类测试）

适用于测试类的方法、需要 setUp/tearDown、或需要大量 helper 方法时。

```python
import unittest
from unittest.mock import MagicMock, patch

import torch
from tests.ut.base import TestBase
from vllm_ascend.xxx.your_module import YourClass


class TestYourClass(TestBase):
    """Test suite for YourClass
    
    简要描述测试覆盖的场景。
    """

    def setUp(self):
        """初始化公共fixture"""
        self.device = torch.device("cpu")

    def test_method_normal_case(self):
        """测试正常场景"""
        with patch("vllm_ascend.xxx.your_module.some_npu_call") as mock_call:
            mock_call.return_value = torch.tensor([1.0])
            obj = YourClass()
            result = obj.method()
            self.assertEqual(result, expected)
            mock_call.assert_called_once()

    def test_method_edge_case(self):
        """测试边界/异常场景"""
        pass

    def _create_mock_config(self):
        """辅助方法：创建mock配置"""
        mock_config = MagicMock()
        mock_config.compilation_config.dispatch_forward_backend = "eager"
        return mock_config
```

## 模板 B: pytest 风格（推荐用于简单函数测试）

适用于测试纯函数、需要参数化、或需要使用 pytest fixture/mocker 时。

```python
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config

from vllm_ascend.xxx.your_module import your_function


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


@pytest.fixture
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.dispatch_forward_backend = "eager"
    with set_current_vllm_config(mock_config):
        yield mock_config


@patch("vllm_ascend.xxx.your_module.some_npu_call", side_effect=lambda x: x + 1)
def test_your_function_normal(mock_call, dummy_tensor, default_vllm_config):
    out = your_function(dummy_tensor)
    expected = dummy_tensor + 1
    assert torch.allclose(out, expected)
    mock_call.assert_called_once()


@pytest.mark.parametrize("input_shape", [(2, 4), (8, 16)])
def test_your_function_shapes(input_shape):
    x = torch.randn(*input_shape)
    out = your_function(x)
    assert out.shape == input_shape
```

## Mock 最佳实践

### 1. mock torch_npu 调用

UT 环境通常没有真实 NPU，`tests/ut/conftest.py` 已经全局 mock 了 `triton.runtime` 和部分 `torch_npu`，但对具体算子调用仍需在测试中按需 patch：

```python
@patch("torch_npu.npu_fast_gelu", side_effect=lambda x: x + 1)
def test_something(mock_gelu):
    ...
```

### 2. mock vllm_config

大量 Ascend 模块依赖 `vllm_config`，测试中应使用 `set_current_vllm_config` 上下文或 mock：

```python
from vllm.config import set_current_vllm_config

mock_config = MagicMock()
mock_config.compilation_config.dispatch_forward_backend = "eager"
with set_current_vllm_config(mock_config):
    # 执行被测代码
    ...
```

### 3. mock 分布式通信组

涉及 `parallel_state` 的代码需要 mock GroupCoordinator：

```python
from vllm.distributed.parallel_state import GroupCoordinator

with patch("vllm_ascend.xxx.get_dcp_group") as mock_get_group:
    mock_group = MagicMock(spec=GroupCoordinator)
    mock_group.world_size = 2
    mock_group.rank_in_group = 0
    mock_get_group.return_value = mock_group
    ...
```

## 命名规范

- 测试文件：`test_<模块名>.py`
- 测试类：`Test<被测类名>`
- 测试方法：`test_<被测方法>_<场景描述>`，如 `test_compute_slot_mapping_dcp_rank_0`

## 硬件相关跳过

如果测试只在特定硬件上执行：

```python
from vllm_ascend.utils import is_310p as is_310p_hw

@pytest.mark.skipif(is_310p_hw(), reason="non_310P device unittest case.")
def test_non_310p_case():
    ...

@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
def test_310p_case():
    ...
```

## 覆盖率目标

- PR 增量覆盖率目标：≥ 80%
- 尽量覆盖：正常路径、边界条件、异常分支、不同参数组合
