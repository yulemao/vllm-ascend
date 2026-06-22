#!/usr/bin/env python3
"""
根据当前代码变更（git diff）自动检测涉及哪些UT。

使用方式:
    python find_related_tests.py [--diff-target TARGET] [--json]

输出格式（JSON）:
    {
        "source_files": ["vllm_ascend/ops/activation.py", ...],
        "test_mappings": [
            {
                "source": "vllm_ascend/ops/activation.py",
                "test_path": "tests/ut/ops/test_activation.py",
                "status": "exists" | "missing"
            }
        ],
        "existing_tests": [...],
        "missing_tests": [...]
    }
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "vllm_ascend").exists() and (PROJECT_ROOT / "tests").exists():
    pass
else:
    # 尝试向上寻找项目根目录
    for parent in PROJECT_ROOT.parents:
        if (parent / "vllm_ascend").exists() and (parent / "tests").exists():
            PROJECT_ROOT = parent
            break

SRC_DIR = PROJECT_ROOT / "vllm_ascend"
TEST_DIR = PROJECT_ROOT / "tests" / "ut"


def get_changed_files(diff_target=None):
    """获取变更的源文件列表，仅限 vllm_ascend 目录下的 .py 文件。"""
    target = diff_target or "HEAD"
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", target, "--", "vllm_ascend/"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError:
        # 可能没有 git 或者 diff target 不存在，尝试用 unstaged changes
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", "vllm_ascend/"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    # 同时检查 staged 和 unstaged
    result2 = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", "vllm_ascend/"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    files += [f.strip() for f in result2.stdout.strip().split("\n") if f.strip()]
    # 去重
    files = list(dict.fromkeys(files))
    return [f for f in files if f.endswith(".py")]


def map_source_to_test(source_file):
    """将源文件路径映射到对应的测试文件路径。"""
    rel = Path(source_file).relative_to("vllm_ascend")
    parts = rel.parts
    stem = rel.stem

    if len(parts) == 1:
        # vllm_ascend/foo.py -> tests/ut/test_foo.py
        test_path = TEST_DIR / f"test_{stem}.py"
    else:
        # vllm_ascend/a/b/c/foo.py -> tests/ut/a/b/c/test_foo.py
        test_path = TEST_DIR / Path(*parts[:-1]) / f"test_{stem}.py"

    return test_path.relative_to(PROJECT_ROOT)


def find_existing_tests_for_source(source_file):
    """如果标准映射不存在，尝试在同目录下查找其他可能相关的测试文件。"""
    rel = Path(source_file).relative_to("vllm_ascend")
    parts = rel.parts
    stem = rel.stem

    if len(parts) == 1:
        search_dir = TEST_DIR
    else:
        search_dir = TEST_DIR / Path(*parts[:-1])

    candidates = []
    if search_dir.exists():
        for f in search_dir.glob("test_*.py"):
            candidates.append(f.relative_to(PROJECT_ROOT))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="查找代码变更相关的UT")
    parser.add_argument("--diff-target", default=None, help="git diff 目标，如 HEAD~1")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出")
    parser.add_argument("--files", nargs="*", help="直接指定源文件，跳过git diff")
    args = parser.parse_args()

    changed_files = args.files if args.files else get_changed_files(args.diff_target)
    if not changed_files:
        out = {
            "source_files": [],
            "test_mappings": [],
            "existing_tests": [],
            "missing_tests": [],
            "message": "未检测到 vllm_ascend/ 目录下的代码变更",
        }
        print(json.dumps(out, indent=2, ensure_ascii=False) if args.json else "未检测到代码变更")
        sys.exit(0)

    mappings = []
    existing = []
    missing = []

    for src in changed_files:
        mapped = map_source_to_test(src)
        full_mapped = PROJECT_ROOT / mapped
        if full_mapped.exists():
            status = "exists"
            existing.append(str(mapped))
        else:
            status = "missing"
            missing.append(str(mapped))
            # 尝试查找其他候选测试
            candidates = find_existing_tests_for_source(src)
            for c in candidates:
                if str(c) not in existing:
                    existing.append(str(c))

        mappings.append({
            "source": src,
            "test_path": str(mapped),
            "status": status,
        })

    out = {
        "source_files": changed_files,
        "test_mappings": mappings,
        "existing_tests": existing,
        "missing_tests": missing,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("变更的源文件:")
        for f in changed_files:
            print(f"  {f}")
        print("\n测试映射:")
        for m in mappings:
            icon = "✓" if m["status"] == "exists" else "✗"
            print(f"  {icon} {m['source']} -> {m['test_path']}")
        if missing:
            print("\n缺少UT覆盖的源文件（需新增UT）:")
            for m in mappings:
                if m["status"] == "missing":
                    print(f"  - {m['source']} -> {m['test_path']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
