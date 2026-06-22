#!/usr/bin/env python3
"""
对指定的UT文件进行静态检查，分析语法正确性、导入完整性、
以及对源文件的语句/分支覆盖情况（基于AST的启发式分析）。

使用方式:
    python static_check.py <test_file_or_dir> [--source <source_file>] [--json]

示例:
    python static_check.py tests/ut/ops/test_activation.py --source vllm_ascend/ops/activation.py
    python static_check.py tests/ut/ops/
"""

import argparse
import ast
import json
import os
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


def parse_file(path):
    """解析Python文件为AST，返回(tree, error)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(path))
        return tree, None
    except SyntaxError as e:
        return None, f"语法错误: {e}"
    except Exception as e:
        return None, str(e)


def get_defined_functions_and_classes(tree):
    """获取模块中定义的顶层函数和类名。"""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def get_called_names(tree):
    """获取测试代码中调用的所有名称（函数调用、类实例化、属性访问的根对象）。"""
    called = set()
    accessed_attrs = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 直接调用 func()
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            # 方法调用 obj.method()
            elif isinstance(node.func, ast.Attribute):
                # 记录完整的属性链，如 mock_gelu.call_args
                root = node.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    called.add(root.id)
                accessed_attrs.add(node.func.attr)
        # 单独的属性访问（非调用）
        elif isinstance(node, ast.Attribute):
            accessed_attrs.add(node.attr)
    return called, accessed_attrs


def get_imported_modules(tree):
    """获取导入的模块名。"""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def check_syntax(path):
    """检查文件语法。"""
    _, error = parse_file(path)
    return error is None, error


def analyze_coverage(test_path, source_path):
    """
    基于AST启发式分析测试对源文件的覆盖情况。
    返回一个字典，包含：
    - source_functions: 源文件中定义的函数和类
    - tested_items: 测试代码中调用的、与源文件同名的函数/类
    - coverage_ratio: 粗略覆盖率
    - branch_hints: 分支覆盖提示（如 if/else、try/except 等）
    """
    source_tree, s_err = parse_file(source_path)
    test_tree, t_err = parse_file(test_path)

    if s_err:
        return {"error": f"源文件解析失败: {s_err}"}
    if t_err:
        return {"error": f"测试文件解析失败: {t_err}"}

    source_items = get_defined_functions_and_classes(source_tree)
    called_names, accessed_attrs = get_called_names(test_tree)

    # 测试代码中可能通过 mock.patch("vllm_ascend.xxx.yyy") 引用，也检查字符串
    patched_names = set()
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # 尝试提取 patch 路径中的最后一段
            if node.value.startswith("vllm_ascend"):
                patched_names.add(node.value.split(".")[-1])

    tested_items = source_items & (called_names | accessed_attrs | patched_names)
    untested_items = source_items - tested_items

    # 检查源文件中的分支结构
    branches = {"if": 0, "for": 0, "while": 0, "try": 0, "with": 0}
    for node in ast.walk(source_tree):
        if isinstance(node, ast.If):
            branches["if"] += 1
        elif isinstance(node, ast.For):
            branches["for"] += 1
        elif isinstance(node, ast.While):
            branches["while"] += 1
        elif isinstance(node, ast.Try):
            branches["try"] += 1
        elif isinstance(node, ast.With):
            branches["with"] += 1

    # 检查测试代码中是否有对应分支的测试提示
    test_branches = {"if": 0, "for": 0, "while": 0, "try": 0, "with": 0}
    for node in ast.walk(test_tree):
        if isinstance(node, ast.If):
            test_branches["if"] += 1
        elif isinstance(node, ast.For):
            test_branches["for"] += 1
        elif isinstance(node, ast.While):
            test_branches["while"] += 1
        elif isinstance(node, ast.Try):
            test_branches["try"] += 1
        elif isinstance(node, ast.With):
            test_branches["with"] += 1

    coverage_ratio = len(tested_items) / len(source_items) if source_items else 1.0

    return {
        "source_file": str(source_path),
        "test_file": str(test_path),
        "source_items": sorted(source_items),
        "tested_items": sorted(tested_items),
        "untested_items": sorted(untested_items),
        "coverage_ratio": round(coverage_ratio, 2),
        "source_branches": branches,
        "test_branches": test_branches,
        "notes": [
            "覆盖率为基于AST的启发式估计，不代表实际运行时覆盖率。",
            "建议为 untested_items 中的函数/类补充测试用例。",
        ],
    }


def infer_source_from_test(test_path):
    """根据测试文件路径推断对应的源文件路径。"""
    rel = Path(test_path).relative_to(PROJECT_ROOT / "tests" / "ut")
    parts = rel.parts
    stem = rel.stem

    if stem.startswith("test_"):
        src_stem = stem[5:]
    else:
        src_stem = stem

    if len(parts) == 1:
        src_path = PROJECT_ROOT / "vllm_ascend" / f"{src_stem}.py"
    else:
        src_path = PROJECT_ROOT / "vllm_ascend" / Path(*parts[:-1]) / f"{src_stem}.py"

    return src_path if src_path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="UT静态检查工具")
    parser.add_argument("target", help="测试文件或目录")
    parser.add_argument("--source", help="对应的源文件（可选，自动推断）")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出")
    args = parser.parse_args()

    target = PROJECT_ROOT / args.target
    if not target.exists():
        print(f"错误: 目标不存在 {target}", file=sys.stderr)
        sys.exit(1)

    results = []
    test_files = list(target.rglob("test_*.py")) if target.is_dir() else [target]

    for test_file in test_files:
        syntax_ok, syntax_err = check_syntax(test_file)
        item = {
            "test_file": str(test_file.relative_to(PROJECT_ROOT)),
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_err,
        }

        source_file = None
        if args.source:
            source_file = PROJECT_ROOT / args.source
        else:
            source_file = infer_source_from_test(test_file)

        if source_file and source_file.exists():
            cov = analyze_coverage(test_file, source_file)
            item["coverage_analysis"] = cov
        else:
            item["coverage_analysis"] = {"note": "无法自动推断对应的源文件，跳过覆盖分析"}

        results.append(item)

    summary = {
        "total": len(results),
        "syntax_pass": sum(1 for r in results if r["syntax_ok"]),
        "syntax_fail": sum(1 for r in results if not r["syntax_ok"]),
        "details": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"UT 静态检查报告 (共 {summary['total']} 个测试文件)")
        print("=" * 70)
        for r in results:
            print(f"\n📄 {r['test_file']}")
            if r['syntax_ok']:
                print("   语法: ✅ 通过")
            else:
                print(f"   语法: ❌ 失败 - {r['syntax_error']}")

            cov = r.get("coverage_analysis", {})
            if "error" in cov:
                print(f"   覆盖分析: ❌ {cov['error']}")
            elif "note" in cov:
                print(f"   覆盖分析: ⚠️ {cov['note']}")
            else:
                ratio = cov.get("coverage_ratio", 0)
                icon = "✅" if ratio >= 0.8 else "⚠️" if ratio >= 0.5 else "❌"
                print(f"   覆盖率(启发式): {icon} {ratio * 100:.0f}%")
                if cov.get("untested_items"):
                    print(f"   未覆盖项: {', '.join(cov['untested_items'][:5])}")
                src_br = cov.get("source_branches", {})
                if src_br.get("if", 0) > 0 and cov.get("test_branches", {}).get("if", 0) == 0:
                    print("   ⚠️ 源文件包含 if 分支，但测试代码中未检测到分支结构，建议补充分支用例")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
