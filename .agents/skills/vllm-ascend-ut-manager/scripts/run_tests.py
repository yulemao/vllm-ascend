#!/usr/bin/env python3
"""
执行指定的UT，并输出详细的失败信息。

使用方式:
    python run_tests.py <test_path> [--json] [--timeout SECONDS]

test_path 可以是:
- 单个测试文件: tests/ut/ops/test_activation.py
- 测试目录: tests/ut/ops/
- 单个测试用例: tests/ut/ops/test_activation.py::test_QuickGELU_forward
"""

import argparse
import json
import os
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

# CI中跳过的测试列表（来自 .github/workflows/_unit_test.yaml）
# 以及环境中收集/执行会导致崩溃的测试
DEFAULT_IGNORES = [
    # CI 已知不稳定/有强环境依赖
    "tests/ut/model_loader/netloader/test_netloader_elastic.py",
    "tests/ut/kv_connector/test_remote_prefill_lifecycle.py",
    "tests/ut/kv_connector/test_remote_decode_lifecycle.py",
    "tests/ut/core/test_scheduler_dynamic_batch.py",
    "tests/ut/kv_connector/test_mooncake_connector.py",
    "tests/ut/worker/test_worker_v1.py",
    "tests/ut/worker/test_worker_multi_instance.py",
    "tests/ut/spec_decode/test_mtp_proposer.py",
    "tests/ut/kv_connector/test_mooncake_layerwise_connector.py",
    # 环境中收集/执行会触发 Python Aborted
    "tests/ut/device_allocator/test_camem.py",
    "tests/ut/ops/test_layernorm.py",
]


def run_pytest(test_path, timeout=300, extra_args=None):
    """执行pytest并返回结果。"""
    cmd = [
        sys.executable, "-m", "pytest",
        "-sv",
        "--tb=long",
        "--color=no",
    ]

    # 自动添加 ignore 参数
    for ign in DEFAULT_IGNORES:
        ign_path = PROJECT_ROOT / ign
        if ign_path.exists():
            cmd.extend(["--ignore", str(ign_path)])

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(test_path))

    env = dict(os.environ)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            env=env,
        )
        return result
    except subprocess.TimeoutExpired:
        class FakeResult:
            returncode = -1
            stdout = ""
            stderr = f"执行超时（>{timeout}秒）"
        return FakeResult()
    except Exception as e:
        class FakeResult:
            returncode = -1
            stdout = ""
            stderr = str(e)
        return FakeResult()


def parse_failures(stdout, stderr):
    """从pytest输出中解析失败的测试用例。"""
    combined = stdout + "\n" + stderr
    failures = []

    # pytest 的失败格式通常是:
    # FAILED tests/ut/...::test_name - AssertionError: ...
    import re
    pattern = re.compile(
        r"FAILED\s+([\w/._-]+::[\w_<>-]+)\s+-\s+(.*)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(combined):
        failures.append({
            "test_case": match.group(1),
            "short_error": match.group(2).strip(),
        })

    # 也尝试提取 ERROR 级别
    error_pattern = re.compile(
        r"ERROR\s+([\w/._-]+::[\w_<>-]+)\s+-\s+(.*)$",
        re.MULTILINE,
    )
    for match in error_pattern.finditer(combined):
        failures.append({
            "test_case": match.group(1),
            "short_error": "ERROR: " + match.group(2).strip(),
        })

    return failures


def main():
    parser = argparse.ArgumentParser(description="执行UT并收集结果")
    parser.add_argument("test_path", help="测试文件、目录或用例路径")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--cov", action="store_true", help="启用覆盖率收集")
    args = parser.parse_args()

    extra = []
    if args.cov:
        extra.extend(["--cov", "--cov-report=term-missing"])

    result = run_pytest(args.test_path, timeout=args.timeout, extra_args=extra)
    failures = parse_failures(result.stdout, result.stderr)

    out = {
        "test_path": args.test_path,
        "returncode": result.returncode,
        "success": result.returncode == 0,
        "failure_count": len(failures),
        "failures": failures,
        "stdout_tail": result.stdout.splitlines()[-100:] if result.stdout else [],
        "stderr_tail": result.stderr.splitlines()[-50:] if result.stderr else [],
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"UT 执行结果: {args.test_path}")
        print("=" * 70)
        if out["success"]:
            print("✅ 所有测试通过")
        else:
            print(f"❌ 失败用例数: {out['failure_count']}")
            for f in failures:
                print(f"   - {f['test_case']}: {f['short_error']}")
            print("\n--- stdout 尾部 ---")
            for line in out["stdout_tail"][-30:]:
                print(line)
            if out["stderr_tail"]:
                print("\n--- stderr 尾部 ---")
                for line in out["stderr_tail"][-20:]:
                    print(line)
        print("=" * 70)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
