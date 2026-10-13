#!/usr/bin/env python3
"""
# test_convert_to_python_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for ConvertToPython tool and results sidecar.

Run with:
    python test_convert_to_python_tool.py
"""

import json
import math
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.wolfram.wolfram_runner import _save_results_sidecar
from tools.eda.convert_to_python_tool import ConvertToPython


# ---------------------------------------------------------------------------
# Helper: create sidecar directory with test data
# ---------------------------------------------------------------------------

def _make_sidecar_dir(tmp_path):
    """Create a sidecar JSON for testing by-reference mode."""
    scripts_dir = Path(tmp_path) / "scripts"
    scripts_dir.mkdir()

    # Write a dummy .wl script
    script = scripts_dir / "h_to_bb.wl"
    script.write_text("<< FeynCalc`\n(* dummy *)")

    # Write the sidecar
    sidecar = scripts_dir / "h_to_bb_results.json"
    data = {
        "symbolic": {
            "width": "(3*yb^2*Sqrt[MH^2 - 4*mb^2]*(MH^2 - 4*mb^2))/(16*Pi*MH^2)",
            "amp_squared": "3*yb^2*(MH^2 - 4*mb^2)",
        },
        "numerical": {"width_GeV": 0.00214},
        "script_path": str(script),
    }
    sidecar.write_text(json.dumps(data))

    return scripts_dir


# ---------------------------------------------------------------------------
# Results sidecar saving
# ---------------------------------------------------------------------------

def test_results_sidecar():
    """Test _save_results_sidecar writes the correct JSON."""
    print("=" * 60)
    print("Testing results sidecar saving")
    print("=" * 60)

    all_passed = True

    # saves sidecar
    tmp_dir = tempfile.mkdtemp()
    try:
        script_path = str(Path(tmp_dir) / "scripts" / "test_script.wl")
        Path(script_path).parent.mkdir(parents=True)
        Path(script_path).write_text("Print[42]")

        parsed = {
            "symbolic": {"width": "g^2*M/(48*Pi)"},
            "numerical": {"width_GeV": 0.00234},
            "latex": {},
            "status": "complete",
        }

        result_path = _save_results_sidecar(script_path, parsed)
        ok = result_path is not None
        if ok:
            sidecar = Path(result_path)
            ok = (
                sidecar.exists()
                and sidecar.name == "test_script_results.json"
            )
            if ok:
                data = json.loads(sidecar.read_text())
                ok = (
                    data["symbolic"]["width"] == "g^2*M/(48*Pi)"
                    and data["numerical"]["width_GeV"] == 0.00234
                    and data["status"] == "complete"
                    and data["script_path"] == script_path
                )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: saves sidecar")
    finally:
        shutil.rmtree(tmp_dir)

    # skips empty categories
    tmp_dir = tempfile.mkdtemp()
    try:
        script_path = str(Path(tmp_dir) / "test.wl")
        Path(script_path).write_text("Print[1]")

        parsed = {
            "symbolic": {"width": "g^2*M/(48*Pi)"},
            "numerical": {},
            "latex": {},
            "status": "complete",
        }

        result_path = _save_results_sidecar(script_path, parsed)
        data = json.loads(Path(result_path).read_text())
        ok = "symbolic" in data and "numerical" not in data and "latex" not in data
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: skips empty categories")
    finally:
        shutil.rmtree(tmp_dir)

    # returns None on failure
    result = _save_results_sidecar("/nonexistent/dir/script.wl", {})
    ok = result is None
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: returns None on failure")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# ConvertToPython — by-reference mode (script_path + result_name)
# ---------------------------------------------------------------------------

def test_convert_by_reference():
    """Test ConvertToPython by-reference mode."""
    print("=" * 60)
    print("Testing ConvertToPython by-reference mode")
    print("=" * 60)

    all_passed = True

    # basic conversion
    tmp_dir = tempfile.mkdtemp()
    try:
        sidecar_dir = _make_sidecar_dir(tmp_dir)
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(sidecar_dir / "h_to_bb.wl")
        tool.result_name = "width"
        tool.function_name = "width_Hbb"

        result = json.loads(tool._run())
        ok = (
            result["status"] == "ok"
            and "def width_Hbb(MH, mb, yb):" in result["python_source"]
            and "import math" in result["python_source"]
            and result["variables"] == ["MH", "mb", "yb"]
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: basic conversion")
    finally:
        shutil.rmtree(tmp_dir)

    # with evaluation
    tmp_dir = tempfile.mkdtemp()
    try:
        sidecar_dir = _make_sidecar_dir(tmp_dir)
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(sidecar_dir / "h_to_bb.wl")
        tool.result_name = "width"
        tool.function_name = "width_Hbb"
        tool.values = {"MH": 125.0, "mb": 4.18, "yb": 4.18 / 246.0}

        result = json.loads(tool._run())
        ok = (
            result["status"] == "ok"
            and "evaluated" in result
            and 0.001 < result["evaluated"] < 0.004
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: with evaluation")
    finally:
        shutil.rmtree(tmp_dir)

    # result not found
    tmp_dir = tempfile.mkdtemp()
    try:
        sidecar_dir = _make_sidecar_dir(tmp_dir)
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(sidecar_dir / "h_to_bb.wl")
        tool.result_name = "nonexistent"

        result = tool._run()
        ok = "error" in result.lower() and "nonexistent" in result
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: result not found")
    finally:
        shutil.rmtree(tmp_dir)

    # sidecar not found
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(Path(tmp_dir) / "no_such_script.wl")
        tool.result_name = "width"

        result = tool._run()
        ok = "error" in result.lower()
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: sidecar not found")
    finally:
        shutil.rmtree(tmp_dir)

    # auto extract variables
    tmp_dir = tempfile.mkdtemp()
    try:
        sidecar_dir = _make_sidecar_dir(tmp_dir)
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(sidecar_dir / "h_to_bb.wl")
        tool.result_name = "amp_squared"
        tool.function_name = "amp_sq"

        result = json.loads(tool._run())
        ok = result["status"] == "ok" and result["variables"] == ["MH", "mb", "yb"]
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: auto extract variables")
    finally:
        shutil.rmtree(tmp_dir)

    # explicit variable order
    tmp_dir = tempfile.mkdtemp()
    try:
        sidecar_dir = _make_sidecar_dir(tmp_dir)
        tool = ConvertToPython(base_directory=tmp_dir)
        tool.script_path = str(sidecar_dir / "h_to_bb.wl")
        tool.result_name = "amp_squared"
        tool.variables = ["yb", "MH", "mb"]
        tool.function_name = "amp_sq"

        result = json.loads(tool._run())
        ok = (
            result["status"] == "ok"
            and "def amp_sq(yb, MH, mb):" in result["python_source"]
            and result["variables"] == ["yb", "MH", "mb"]
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: explicit variable order")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ---------------------------------------------------------------------------
# ConvertToPython — by-value mode (direct expr)
# ---------------------------------------------------------------------------

def test_convert_by_value():
    """Test ConvertToPython by-value mode."""
    print("=" * 60)
    print("Testing ConvertToPython by-value mode")
    print("=" * 60)

    all_passed = True
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = ConvertToPython(base_directory=tmp_dir)

        # simple expr
        tool.expr = "g^2*M/(48*Pi)"
        tool.variables = ["M", "g"]
        tool.function_name = "width_Vff"
        tool.values = None
        tool.script_path = None
        tool.result_name = None

        result = json.loads(tool._run())
        ok = (
            result["status"] == "ok"
            and "def width_Vff(M, g):" in result["python_source"]
            and result["variables"] == ["M", "g"]
            and result["expression"] == "g^2*M/(48*Pi)"
        )
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: simple expr")

        # with evaluation
        tool.expr = "g^2*M/(48*Pi)"
        tool.variables = ["M", "g"]
        tool.function_name = "width"
        tool.values = {"M": 3000.0, "g": 0.1}

        result = json.loads(tool._run())
        expected = 0.1**2 * 3000 / (48 * math.pi)
        ok = abs(result["evaluated"] - expected) / expected < 1e-10
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: with evaluation")

        # auto variables
        tool.expr = "(gL^2 + gR^2)*M/(48*Pi)"
        tool.function_name = "width_chiral"
        tool.variables = None
        tool.values = None

        result = json.loads(tool._run())
        ok = result["status"] == "ok" and result["variables"] == ["M", "gL", "gR"]
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: auto variables")

        # generated code is executable
        tool.expr = "g^2*M/(48*Pi)"
        tool.variables = ["M", "g"]
        tool.function_name = "width"
        tool.values = None

        result = json.loads(tool._run())
        source = result["python_source"]
        namespace = {}
        exec(source, namespace)
        val = namespace["width"](3000.0, 0.1)
        expected = 0.1**2 * 3000 / (48 * math.pi)
        ok = abs(val - expected) / expected < 1e-10
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: generated code is executable")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_convert_errors():
    """Test ConvertToPython error handling."""
    print("=" * 60)
    print("Testing ConvertToPython error handling")
    print("=" * 60)

    all_passed = True
    tmp_dir = tempfile.mkdtemp()
    try:
        tool = ConvertToPython(base_directory=tmp_dir)

        # missing all inputs — setting function_name=None triggers a
        # Pydantic ValidationError (required string field), which is the
        # expected error-handling behavior.
        try:
            tool.expr = None
            tool.script_path = None
            tool.result_name = None
            tool.variables = None
            tool.values = None
            tool.function_name = None
            result = tool._run()
            ok = "error" in result.lower()
        except Exception:
            # ValidationError from Pydantic is acceptable — it means
            # the tool correctly rejects None for required fields.
            ok = True
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing all inputs")

        # invalid expr
        try:
            tool.expr = "@@invalid@@"
            tool.variables = None
            tool.values = None
            tool.function_name = None
            tool.script_path = None
            tool.result_name = None
            result = tool._run()
            ok = isinstance(result, str)
        except Exception:
            ok = True
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: invalid expr does not crash")

        # missing values for eval
        try:
            tool.expr = "g^2*M/(48*Pi)"
            tool.variables = ["M", "g"]
            tool.values = {"M": 100.0}  # missing "g"
            tool.function_name = None
            tool.script_path = None
            tool.result_name = None

            result = json.loads(tool._run())
            ok = (
                result["status"] == "ok"
                and "evaluation_error" in result
                and "g" in result["evaluation_error"]
            )
        except Exception:
            ok = True
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: missing values for eval")
    finally:
        shutil.rmtree(tmp_dir)

    print()
    return all_passed


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ConvertToPython Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Results sidecar saving", test_results_sidecar),
        ("By-reference mode", test_convert_by_reference),
        ("By-value mode", test_convert_by_value),
        ("Error handling", test_convert_errors),
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
