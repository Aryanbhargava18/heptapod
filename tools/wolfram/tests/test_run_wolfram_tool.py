#!/usr/bin/env python3
"""
# test_run_wolfram_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for RunWolframScript BaseTool.

Run with:
    python test_run_wolfram_tool.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.wolfram import RunWolframScript, WolframRunner, resolve_wolframscript


def _check_wolfram():
    """Return skip reason or None."""
    runner = WolframRunner()
    avail, msg = runner.check_available()
    if not avail:
        return f"wolframscript not available: {msg}"
    return None


def _make_tool(tmp_dir):
    """Create a RunWolframScript tool instance.

    Resolves the executable the same way the runner does rather than reading
    config.wolframscript_path directly: the shipped config carries the
    placeholder "/path/to/wolframscript", which used to make every test here
    skip (and report PASS) on a machine that has Wolfram installed.
    """
    return RunWolframScript(
        base_directory=tmp_dir,
        wolframscript_path=resolve_wolframscript(),
    )


# ---------------------------------------------------------------------------
# Missing code (no wolframscript needed)
# ---------------------------------------------------------------------------

def test_missing_code():
    """Test that empty code returns an error."""
    print("=" * 60)
    print("Testing missing code")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = ""
        result = tool._run()
        ok = "error" in result.lower() or "Missing" in result
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: empty code returns error")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# Integration tests (require wolframscript)
# ---------------------------------------------------------------------------

def test_simple_execution():
    """Test simple wolframscript execution."""
    print("=" * 60)
    print("Testing simple execution")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = 'Print["hello from mathematica"]'
        result = tool._run()
        data = json.loads(result)
        ok = data["success"] and "hello from mathematica" in data["stdout"]
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: simple execution")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_script_saving():
    """Test that scripts are saved to disk."""
    print("=" * 60)
    print("Testing script saving")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = 'Print[42]'
        tool.script_name = "test_saving"
        result = tool._run()
        data = json.loads(result)
        ok = (
            data["success"]
            and data["script_path"].endswith(".wl")
            and Path(data["script_path"]).exists()
        )
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: script saving")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_symbolic_computation():
    """Symbolic result parsing, with no Wolfram package loaded at all.

    Deliberately uses only kernel builtins: this package is generic, so its
    own suite must not require any add-on package to be installed.
    """
    print("=" * 60)
    print("Testing symbolic computation (no packages)")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = (
            'res = Factor[x^3 - 6 x^2 + 11 x - 6];\n'
            'Print["SYMBOLIC_RESULT[factored]: ", res]\n'
            'Print["STATUS: complete"]\n'
        )
        result = tool._run()
        data = json.loads(result)
        factored = data.get("symbolic", {}).get("factored", "")
        ok = (
            data["success"]
            and "factored" in data.get("symbolic", {})
            and "stdout" not in data
            # (x-1)(x-2)(x-3) in whatever order Wolfram prints it
            and all(t in factored for t in ("-1", "-2", "-3"))
        )
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: symbolic computation")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_numerical_output():
    """Test numerical output parsing."""
    print("=" * 60)
    print("Testing numerical output")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = (
            'Print["NUMERICAL_RESULT[pi]: ", N[Pi]]\n'
            'Print["STATUS: complete"]\n'
        )
        result = tool._run()
        data = json.loads(result)
        pi_val = data["numerical"]["pi"]
        if isinstance(pi_val, str):
            pi_val = float(pi_val)
        ok = data["success"] and "numerical" in data and abs(pi_val - 3.14159265) < 0.001
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: numerical output (pi = {pi_val})")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_stdout_kept_without_parsed_results():
    """Test that stdout is included when no parsed results exist."""
    print("=" * 60)
    print("Testing stdout kept without parsed results")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = 'Print[1]'
        result = tool._run()
        data = json.loads(result)
        ok = (
            data["success"]
            and "stdout" in data
            and "execution_time_s" not in data
        )
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: stdout kept without parsed results")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_latex_output():
    """Test LaTeX output parsing."""
    print("=" * 60)
    print("Testing LaTeX output")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = (
            'expr = x^2/(4 Pi);\n'
            'Print["SYMBOLIC_RESULT[test]: ", expr]\n'
            'Print["LATEX_RESULT[test]: ", TeXForm[expr]]\n'
            'Print["STATUS: complete"]\n'
        )
        result = tool._run()
        data = json.loads(result)
        ok = (
            data["success"]
            and "latex" in data
            and "test" in data["latex"]
        )
        if ok:
            latex_str = data["latex"]["test"]
            ok = "frac" in latex_str or "pi" in latex_str.lower()
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: LaTeX output")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


def test_error_includes_hint():
    """Test that errors include error_hint."""
    print("=" * 60)
    print("Testing error includes hint")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = 'Quit[1]'
        result = tool._run()
        data = json.loads(result)
        ok = not data["success"] and "error_hint" in data
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: error includes hint")
        print()
        return ok
    finally:
        shutil.rmtree(tmp_dir)


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RunWolframScript Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Missing code", test_missing_code),
        ("Simple execution", test_simple_execution),
        ("Script saving", test_script_saving),
        ("Symbolic computation", test_symbolic_computation),
        ("Numerical output", test_numerical_output),
        ("Stdout kept without parsed results", test_stdout_kept_without_parsed_results),
        ("LaTeX output", test_latex_output),
        ("Error includes hint", test_error_includes_hint),
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
