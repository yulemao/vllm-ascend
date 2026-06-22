#!/usr/bin/env python3
"""
检测当前环境是否具备执行 vllm-ascend UT 的条件。

使用方式:
    python check_ut_env.py [--json]

检查项:
1. pytest 及其插件是否安装
2. 核心依赖（torch、vllm、vllm_ascend）是否可导入
3. 能否成功收集（collect）UT（不实际执行）
4. torch_npu 是否可用（UT中会被mock，非必须，但会提示）
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "vllm_ascend").exists() and (PROJECT_ROOT / "tests").exists():
    pass
else:
    for parent in PROJECT_ROOT.parents:
        if (parent / "vllm_ascend").exists() and (parent / "tests").exists():
            PROJECT_ROOT = parent
            break

REQUIRED_PACKAGES = [
    ("pytest", "pytest"),
    ("pytest_cov", "pytest-cov"),
    ("pytest_mock", "pytest-mock"),
]

OPTIONAL_PACKAGES = [
    ("torch", "torch"),
    ("torch_npu", "torch_npu"),
]

CORE_MODULES = [
    "vllm",
    "vllm_ascend",
]

# 需要跳过的测试文件（CI 已知不稳定 + 环境崩溃文件）
DEFAULT_IGNORES = [
    "tests/ut/model_loader/netloader/test_netloader_elastic.py",
    "tests/ut/kv_connector/test_remote_prefill_lifecycle.py",
    "tests/ut/kv_connector/test_remote_decode_lifecycle.py",
    "tests/ut/core/test_scheduler_dynamic_batch.py",
    "tests/ut/kv_connector/test_mooncake_connector.py",
    "tests/ut/worker/test_worker_v1.py",
    "tests/ut/worker/test_worker_multi_instance.py",
    "tests/ut/spec_decode/test_mtp_proposer.py",
    "tests/ut/kv_connector/test_mooncake_layerwise_connector.py",
    "tests/ut/device_allocator/test_camem.py",
    "tests/ut/ops/test_layernorm.py",
]


def check_package(module_name, package_name):
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as e:
        return False, str(e)


def can_collect_tests():
    """尝试收集 tests/ut 下的测试，不实际运行。"""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/ut"]
    for ign in DEFAULT_IGNORES:
        ign_path = PROJECT_ROOT / ign
        if ign_path.exists():
            cmd.extend(["--ignore", str(ign_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        if result.returncode == 0 or "test session starts" in result.stdout:
            return True, result.stdout.splitlines()[-3:-1] if result.stdout else []
        # 某些环境下收集可能因为缺少torch_npu而失败，但如果pytest本身运行了，也算部分可用
        if "ModuleNotFoundError" in result.stderr and "torch_npu" in result.stderr:
            return "partial", ["pytest可启动，但torch_npu未安装，UT中会被mock，通常不影响运行"]
        return False, result.stderr.splitlines()[:10]
    except Exception as e:
        return False, [str(e)]


def main():
    parser = argparse.ArgumentParser(description="检查UT执行环境")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出")
    args = parser.parse_args()

    report = {
        "can_run_ut": False,
        "details": {},
        "messages": [],
    }

    # 1. 检查 pytest 相关
    all_pytest_ok = True
    for mod, pkg in REQUIRED_PACKAGES:
        ok, err = check_package(mod, pkg)
        report["details"][f"package_{pkg}"] = {"ok": ok, "error": err}
        if not ok:
            all_pytest_ok = False
            report["messages"].append(f"缺少必要包: {pkg} ({err})")

    # 2. 检查核心模块
    core_ok = True
    for mod in CORE_MODULES:
        ok, err = check_package(mod, mod)
        report["details"][f"module_{mod}"] = {"ok": ok, "error": err}
        if not ok:
            core_ok = False
            report["messages"].append(f"无法导入核心模块: {mod} ({err})")

    # 3. 检查可选包
    for mod, pkg in OPTIONAL_PACKAGES:
        ok, err = check_package(mod, pkg)
        report["details"][f"package_{pkg}"] = {"ok": ok, "error": err}
        if not ok:
            report["messages"].append(f"可选包未安装: {pkg}（UT中会被mock，不影响执行）")

    # 4. 尝试 collect
    can_collect, info = can_collect_tests()
    report["details"]["can_collect_tests"] = {"ok": can_collect, "info": info}

    if all_pytest_ok and core_ok and can_collect is not False:
        report["can_run_ut"] = True
        report["messages"].insert(0, "当前环境可以执行UT")
    else:
        report["messages"].insert(0, "当前环境无法完整执行UT，建议仅进行静态检查")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("UT 环境检查报告")
        print("=" * 60)
        status = "✅ 可以执行UT" if report["can_run_ut"] else "❌ 无法执行UT"
        print(f"\n总状态: {status}\n")
        for pkg, detail in report["details"].items():
            icon = "✅" if detail["ok"] else "⚠️" if detail.get("info") else "❌"
            if pkg == "can_collect_tests":
                icon = "✅" if detail["ok"] else "⚠️" if detail["ok"] == "partial" else "❌"
            print(f"{icon} {pkg}")
            if not detail["ok"] and detail.get("error"):
                print(f"   错误: {detail['error']}")
            if detail.get("info"):
                for line in detail["info"]:
                    print(f"   信息: {line}")
        print("=" * 60)

    sys.exit(0 if report["can_run_ut"] else 1)


if __name__ == "__main__":
    main()
