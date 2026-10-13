#!/usr/bin/env python3
"""
# test_wolfram_runner.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for WolframRunner subprocess manager.

Run with:
    python test_wolfram_runner.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.wolfram.wolfram_runner import (
    WolframRunner,
    WolframResult,
    _parse_structured_output,
)


# ---------------------------------------------------------------------------
# Unit tests (no wolframscript needed)
# ---------------------------------------------------------------------------

def test_parse_structured_output():
    """Test _parse_structured_output parsing."""
    print("=" * 60)
    print("Testing _parse_structured_output")
    print("=" * 60)

    all_passed = True

    # empty
    result = _parse_structured_output("")
    ok = result == {"symbolic": {}, "numerical": {}, "latex": {}, "status": None}
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: empty input")

    # symbolic result
    stdout = 'SYMBOLIC_RESULT[ampSq]: 4 g^2 SP[p1, p2]\n'
    result = _parse_structured_output(stdout)
    ok = result["symbolic"]["ampSq"] == "4 g^2 SP[p1, p2]"
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: symbolic result")

    # numerical result
    stdout = 'NUMERICAL_RESULT[width_GeV]: 0.00234\n'
    result = _parse_structured_output(stdout)
    ok = abs(result["numerical"]["width_GeV"] - 0.00234) < 1e-10
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: numerical result")

    # status
    stdout = 'STATUS: complete\n'
    result = _parse_structured_output(stdout)
    ok = result["status"] == "complete"
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: status")

    # mixed output
    stdout = (
        "Loading SomePackage...\n"
        "Some banner text\n"
        "SYMBOLIC_RESULT[trace]: 4*Pair[Momentum[p, D], Momentum[q, D]]\n"
        "NUMERICAL_RESULT[value]: 42.5\n"
        "STATUS: complete\n"
    )
    result = _parse_structured_output(stdout)
    ok = (
        "trace" in result["symbolic"]
        and abs(result["numerical"]["value"] - 42.5) < 1e-10
        and result["status"] == "complete"
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: mixed output")

    # non-numeric numerical result
    stdout = 'NUMERICAL_RESULT[mass]: 125.0 + I*3.2\n'
    result = _parse_structured_output(stdout)
    ok = isinstance(result["numerical"]["mass"], str)
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: non-numeric numerical result stored as string")

    # latex result
    stdout = r'LATEX_RESULT[width]: \frac{3 y_b^2 M_H}{8 \pi}' + '\n'
    result = _parse_structured_output(stdout)
    ok = "width" in result["latex"] and r"\frac" in result["latex"]["width"]
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: latex result")

    # mixed with latex
    stdout = (
        "SYMBOLIC_RESULT[width]: 3*yb^2*MH/(8*Pi)\n"
        r"LATEX_RESULT[width]: \frac{3 y_b^2 M_H}{8 \pi}" + "\n"
        "NUMERICAL_RESULT[width_GeV]: 0.00234\n"
        "STATUS: complete\n"
    )
    result = _parse_structured_output(stdout)
    ok = (
        "width" in result["symbolic"]
        and "width" in result["latex"]
        and abs(result["numerical"]["width_GeV"] - 0.00234) < 1e-10
        and result["status"] == "complete"
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: mixed with latex")

    print()
    return all_passed


def test_wolfram_runner_init():
    """Test WolframRunner initialization."""
    print("=" * 60)
    print("Testing WolframRunner initialization")
    print("=" * 60)

    all_passed = True

    # default init
    runner = WolframRunner()
    ok = runner.wolframscript_path is not None and runner.timeout_sec == 120
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: default init")

    # custom timeout
    runner = WolframRunner(timeout_sec=60)
    ok = runner.timeout_sec == 60
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: custom timeout")

    # custom path
    runner = WolframRunner(wolframscript_path="/custom/path")
    ok = runner.wolframscript_path == "/custom/path"
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: custom path")

    print()
    return all_passed


def test_run_script_missing():
    """Test running a missing script."""
    print("=" * 60)
    print("Testing run_script with missing file")
    print("=" * 60)

    runner = WolframRunner()
    result = runner.run_script("/nonexistent/script.wl")
    ok = not result.success and "not found" in result.stderr
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing script returns error")

    print()
    return ok


# ---------------------------------------------------------------------------
# Integration tests (require wolframscript)
# ---------------------------------------------------------------------------

def _check_wolfram():
    """Return (runner, skip_reason). skip_reason is None if available."""
    r = WolframRunner()
    avail, msg = r.check_available()
    if not avail:
        return None, f"wolframscript not available: {msg}"
    return r, None


def test_check_available():
    """Test wolframscript availability check."""
    print("=" * 60)
    print("Testing wolframscript availability")
    print("=" * 60)

    runner, skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    avail, msg = runner.check_available()
    ok = avail and "WolframScript" in msg
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: check_available reports the kernel version")

    print()
    return ok


def test_run_code():
    """Test running code via WolframRunner."""
    print("=" * 60)
    print("Testing run_code")
    print("=" * 60)

    runner, skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    all_passed = True

    # simple print
    tmp_dir = tempfile.mkdtemp()
    try:
        result = runner.run_code('Print[2 + 3]', working_dir=tmp_dir)
        ok = result.success and "5" in result.stdout
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: simple print")
    finally:
        shutil.rmtree(tmp_dir)

    # symbolic result, kernel builtins only (no add-on package required)
    tmp_dir = tempfile.mkdtemp()
    try:
        code = (
            'res = Integrate[1/(1 + x^2), x];\n'
            'Print["SYMBOLIC_RESULT[arctan]: ", res]\n'
            'Print["STATUS: complete"]\n'
        )
        result = runner.run_code(code, working_dir=tmp_dir)
        ok = (
            result.success
            and result.parsed_results["status"] == "complete"
            and "arctan" in result.parsed_results["symbolic"]
            and "ArcTan" in result.parsed_results["symbolic"]["arctan"]
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: symbolic result")
    finally:
        shutil.rmtree(tmp_dir)

    return all_passed


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WolframRunner Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Parse structured output", test_parse_structured_output),
        ("WolframRunner init", test_wolfram_runner_init),
        ("Run script missing", test_run_script_missing),
        ("Check available", test_check_available),
        ("Run code", test_run_code),
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
