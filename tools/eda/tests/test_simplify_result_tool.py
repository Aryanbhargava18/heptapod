#!/usr/bin/env python3
"""
# test_simplify_result_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for SimplifyResult tool -- sidecar loading, code generation, error handling.

Run with:
    python test_simplify_result_tool.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.wolfram import load_expression_from_sidecar
from tools.eda.simplify_result_tool import SimplifyResult


# ---------------------------------------------------------------------------
# Shared sidecar helper tests
# ---------------------------------------------------------------------------

def test_load_expression_from_sidecar():
    """Tests for the shared load_expression_from_sidecar utility."""
    print("=" * 60)
    print("Testing load_expression_from_sidecar")
    print("=" * 60)

    all_passed = True

    # loads symbolic result
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("Print[42]")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"width": "g^2*M/(48*Pi)", "ampSq": "2*g^2*M^2"},
            "numerical": {"width_GeV": 0.00234},
        }))

        expr, err = load_expression_from_sidecar(str(script), "width")
        ok = err is None and expr == "g^2*M/(48*Pi)"
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loads symbolic result")
    finally:
        shutil.rmtree(tmp_dir)

    # loads different result name
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"width": "expr1", "ampSq": "expr2"},
        }))

        expr, err = load_expression_from_sidecar(str(script), "ampSq")
        ok = err is None and expr == "expr2"
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loads different result name")
    finally:
        shutil.rmtree(tmp_dir)

    # missing sidecar
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "nonexistent.wl"
        expr, err = load_expression_from_sidecar(str(script), "width")
        ok = expr is None and "No results sidecar" in err
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing sidecar")
    finally:
        shutil.rmtree(tmp_dir)

    # missing result key
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"ampSq": "2*g^2"},
        }))

        expr, err = load_expression_from_sidecar(str(script), "width")
        ok = expr is None and "width" in err and "ampSq" in err
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing result key")
    finally:
        shutil.rmtree(tmp_dir)

    # strips InputForm wrapper
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"simplified": "InputForm[(g^2*M)/(48*Pi)]"},
        }))

        expr, err = load_expression_from_sidecar(str(script), "simplified")
        ok = err is None and expr == "(g^2*M)/(48*Pi)" and "InputForm" not in expr
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: strips InputForm wrapper")
    finally:
        shutil.rmtree(tmp_dir)

    # loads numerical category
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"width": "g^2*M/(48*Pi)"},
            "numerical": {"width_GeV": 0.00234},
        }))

        expr, err = load_expression_from_sidecar(
            str(script), "width_GeV", category="numerical"
        )
        ok = err is None and expr == "0.00234"
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loads numerical category")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ---------------------------------------------------------------------------
# SimplifyResult code generation tests
# ---------------------------------------------------------------------------

def test_simplify_codegen():
    """Tests for generated Mathematica code correctness."""
    print("=" * 60)
    print("Testing SimplifyResult code generation")
    print("=" * 60)

    all_passed = True
    tmp_dir = tempfile.mkdtemp()
    try:
        # basic simplify
        tool = SimplifyResult(base_directory=tmp_dir, expr="x^2 + 2*x + 1")
        code = tool._generate_mathematica("x^2 + 2*x + 1")
        ok = (
            "expr = x^2 + 2*x + 1;" in code
            and "Simplify[expr]" in code
            and 'SYMBOLIC_RESULT[simplified]' in code
            and 'LATEX_RESULT[simplified]' in code
            and 'STATUS: complete' in code
            and 'InputForm[expr]' not in code
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: basic simplify")

        # substitutions
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="mfbar^2 + mf^2",
            substitutions={"mfbar": "mf"},
        )
        code = tool._generate_mathematica("mfbar^2 + mf^2")
        ok = "expr /. {mfbar -> mf}" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: substitutions")

        # multiple substitutions
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="a + b + c",
            substitutions={"a": "1", "b": "2"},
        )
        code = tool._generate_mathematica("a + b + c")
        ok = "a -> 1" in code and "b -> 2" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: multiple substitutions")

        # limit
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="Sin[x]/x",
            limit={"var": "x", "point": "0"},
        )
        code = tool._generate_mathematica("Sin[x]/x")
        ok = "Limit[expr, x -> 0]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: limit")

        # limit with direction
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="1/x",
            limit={"var": "x", "point": "0", "direction": "1"},
        )
        code = tool._generate_mathematica("1/x")
        ok = "Direction -> 1" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: limit with direction")

        # series
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="Exp[eps]",
            series={"var": "eps", "point": "0", "order": 3},
        )
        code = tool._generate_mathematica("Exp[eps]")
        ok = "Normal[Series[expr, {eps, 0, 3}]]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: series")

        # assumptions
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="Sqrt[x^2]",
            assumptions=["x > 0"],
        )
        code = tool._generate_mathematica("Sqrt[x^2]")
        ok = "Assuming[{x > 0}, Simplify[expr]]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: assumptions")

        # FullSimplify
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="expr",
            simplify="FullSimplify",
        )
        code = tool._generate_mathematica("expr")
        ok = "FullSimplify[expr]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: FullSimplify")

        # Factor
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="x^2 - 1",
            simplify="Factor",
        )
        code = tool._generate_mathematica("x^2 - 1")
        ok = "Factor[expr]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: Factor")

        # no simplify
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="x + y",
            simplify="None",
        )
        code = tool._generate_mathematica("x + y")
        ok = (
            "Simplify[" not in code
            and "FullSimplify[" not in code
            and 'SYMBOLIC_RESULT[simplified]' in code
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: no simplify")

        # operation order: sub, limit, series, simplify
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="f[x, eps]",
            substitutions={"x": "1"},
            limit={"var": "eps", "point": "0"},
            simplify="Simplify",
        )
        code = tool._generate_mathematica("f[x, eps]")
        lines = code.split("\n")
        sub_idx = next(i for i, l in enumerate(lines) if "/." in l)
        lim_idx = next(i for i, l in enumerate(lines) if "Limit[" in l)
        simp_idx = next(i for i, l in enumerate(lines) if "Simplify[" in l)
        ok = sub_idx < lim_idx < simp_idx
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: operation order")

        # assumptions with FullSimplify
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="Sqrt[M^2]",
            assumptions=["M > 0", "m > 0"],
            simplify="FullSimplify",
        )
        code = tool._generate_mathematica("Sqrt[M^2]")
        ok = "Assuming[{M > 0, m > 0}, FullSimplify[expr]]" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: assumptions with FullSimplify")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ---------------------------------------------------------------------------
# SimplifyResult error handling
# ---------------------------------------------------------------------------

def test_simplify_errors():
    """Tests for error handling in SimplifyResult."""
    print("=" * 60)
    print("Testing SimplifyResult error handling")
    print("=" * 60)

    all_passed = True

    # missing both inputs
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = SimplifyResult(base_directory=tmp_dir)
        result_str = tool._run()
        ok = "error" in result_str.lower() or "Missing Parameter" in result_str
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing both inputs")
    finally:
        shutil.rmtree(tmp_dir)

    # missing sidecar
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = SimplifyResult(
            base_directory=tmp_dir,
            script_path=str(Path(tmp_dir) / "nonexistent.wl"),
            result_name="width",
        )
        result_str = tool._run()
        ok = "error" in result_str.lower() or "Not Found" in result_str
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing sidecar")
    finally:
        shutil.rmtree(tmp_dir)

    # sidecar missing key
    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"ampSq": "2*g^2"},
        }))

        tool = SimplifyResult(
            base_directory=tmp_dir,
            script_path=str(script),
            result_name="width",
        )
        result_str = tool._run()
        ok = "error" in result_str.lower() or "Not Found" in result_str
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: sidecar missing key")
    finally:
        shutil.rmtree(tmp_dir)

    # direct expr input accepted
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = SimplifyResult(
            base_directory=tmp_dir,
            expr="x^2 + 1",
        )
        code = tool._generate_mathematica("x^2 + 1")
        ok = "expr = x^2 + 1;" in code
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: direct expr input accepted")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ---------------------------------------------------------------------------
# SimplifyResult sidecar loading integration
# ---------------------------------------------------------------------------

def test_simplify_sidecar_loading():
    """Tests that SimplifyResult correctly loads from sidecar."""
    print("=" * 60)
    print("Testing SimplifyResult sidecar loading")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        script = Path(tmp_dir) / "test.wl"
        script.write_text("")
        sidecar = Path(tmp_dir) / "test_results.json"
        sidecar.write_text(json.dumps({
            "symbolic": {"width": "g^2*M/(48*Pi)"},
        }))

        tool = SimplifyResult(
            base_directory=tmp_dir,
            script_path=str(script),
            result_name="width",
            substitutions={"g": "0.1"},
            simplify="Simplify",
        )
        expr_str, err = load_expression_from_sidecar(str(script), "width")
        ok_load = err is None
        code = tool._generate_mathematica(expr_str)
        ok = ok_load and "g^2*M/(48*Pi)" in code and "g -> 0.1" in code
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loads from sidecar and generates code")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SimplifyResult Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("load_expression_from_sidecar", test_load_expression_from_sidecar),
        ("Code generation", test_simplify_codegen),
        ("Error handling", test_simplify_errors),
        ("Sidecar loading integration", test_simplify_sidecar_loading),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "[✓] PASS" if result else "[✗] FAIL"))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, "[✗] ERROR"))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {status}: {name}")

    passed = sum(1 for _, s in results if s == "[✓] PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} test groups passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
