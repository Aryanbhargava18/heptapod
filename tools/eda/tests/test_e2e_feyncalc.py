#!/usr/bin/env python3
"""
# test_e2e_feyncalc.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

End-to-end tests: run benchmark FeynCalc scripts and check numerical results.

These tests validate that the complete pipeline works:
LLM-style code -> RunWolframScript -> parsed numerical results -> physics check.

Each test runs a complete FeynCalc calculation and verifies the result
against known SM values.

Run with:
    python test_e2e_feyncalc.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.wolfram import WolframRunner, RunWolframScript


def _check_wolfram():
    """Return skip reason or None."""
    runner = WolframRunner()
    avail, msg = runner.check_available()
    if not avail:
        return f"wolframscript not available: {msg}"
    return None


def _make_tool(tmp_dir):
    """Create a RunWolframScript tool instance."""
    import config
    return RunWolframScript(
        base_directory=tmp_dir,
        wolframscript_path=config.wolframscript_path,
    )


def _parse_num(val):
    """Parse a numerical result that may be string or float."""
    if isinstance(val, (int, float)):
        return float(val)
    return float(val)


# ---------------------------------------------------------------------------
# H -> bb
# ---------------------------------------------------------------------------

H_TO_BB_SCRIPT = r"""
<< FeynCalc`

(* H -> b bbar, scalar Yukawa *)
amp = SpinorUBar[p1, mb] . (-I g) . SpinorV[p2, mb];
ampSq = amp ComplexConjugate[amp] // FermionSpinSum // DiracSimplify;

(* Kinematics: use FCI to convert SP to internal form for replacement *)
ampSqKin = ampSq /. {
  Pair[Momentum[p1, ___], Momentum[p2, ___]] -> (MH^2 - 2 mb^2)/2,
  Pair[Momentum[p1, ___], Momentum[p1, ___]] -> mb^2,
  Pair[Momentum[p2, ___], Momentum[p2, ___]] -> mb^2
} // Simplify;

(* Phase space *)
Kallen[a_, b_, c_] := a^2 + b^2 + c^2 - 2 a b - 2 a c - 2 b c;
pMag = Sqrt[Kallen[MH^2, mb^2, mb^2]] / (2 MH);

(* Decay width: Nc * |p| / (8 pi MH^2) * |M|^2 (no spin avg for scalar) *)
Nc = 3;
width = Nc * pMag / (8 Pi MH^2) * ampSqKin // Simplify;

(* Numerical *)
widthNum = width /. {MH -> 125.0, mb -> 4.18, g -> 4.18/246.0} // N;
Print["NUMERICAL_RESULT[width_GeV]: ", widthNum]
Print["STATUS: complete"]
"""


def test_h_to_bb():
    """Test H -> bb end-to-end."""
    print("=" * 60)
    print("Testing H -> b bbar (end-to-end)")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = H_TO_BB_SCRIPT
        tool.script_name = "test_h_to_bb"
        result = json.loads(tool._run())

        all_passed = True

        ok = result["success"]
        if not ok:
            all_passed = False
            print(f"  [✗] FAIL: script failed: {result.get('error_hint', result.get('stderr', ''))}")
        else:
            print(f"  [✓] PASS: script succeeded")

        ok = "stdout" not in result
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: stdout dropped (lean returns)")

        width = _parse_num(result["numerical"]["width_GeV"])
        ok = 0.001 < width < 0.005
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: H->bb width = {width:.6f} GeV (expected ~0.002-0.003)")

        print()
        return all_passed
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# Z -> e+e-
# ---------------------------------------------------------------------------

Z_TO_EE_SCRIPT = """
<< FeynCalc`

(* Z -> e+ e-, V-A coupling, massless electrons *)
(* Work in 4D for tree-level calculation with gamma5 *)
amp = Pair[Momentum[Polarization[p, I]], LorentzIndex[mu]] *
      SpinorUBar[p1, 0] . (I gZ GA[mu] . (gV - gA GA[5])) . SpinorV[p2, 0];

ampSq = amp ComplexConjugate[amp] // FermionSpinSum // DiracSimplify;
ampSq = DoPolarizationSums[ampSq, p, 0] // Contract // Simplify;

(* Kinematics for massless final states *)
ampSqKin = ampSq /. {
  Pair[Momentum[p1, ___], Momentum[p2, ___]] -> MZ^2/2,
  Pair[Momentum[p, ___], Momentum[p1, ___]] -> MZ^2/2,
  Pair[Momentum[p, ___], Momentum[p2, ___]] -> MZ^2/2,
  Pair[Momentum[p, ___], Momentum[p, ___]] -> MZ^2,
  Pair[Momentum[p1, ___], Momentum[p1, ___]] -> 0,
  Pair[Momentum[p2, ___], Momentum[p2, ___]] -> 0
} // Simplify;

(* Phase space: pMag = MZ/2 for massless final states *)
(* Average over 3 Z polarizations *)
width = (MZ/2) / (8 Pi MZ^2) * ampSqKin / 3 // Simplify;

(* Numerical *)
sw2 = 0.2312;
cw2 = 1 - sw2;
alpha = 1/128.0;
ee = Sqrt[4 Pi alpha];
gZnum = ee / (Sqrt[sw2] Sqrt[cw2]);
gVnum = -1/4 + sw2;
gAnum = -1/4;

widthNum = width /. {MZ -> 91.1876, gZ -> gZnum, gV -> gVnum, gA -> gAnum} // N;
Print["NUMERICAL_RESULT[width_GeV]: ", widthNum]
Print["STATUS: complete"]
"""


def test_z_to_ee():
    """Test Z -> e+e- end-to-end."""
    print("=" * 60)
    print("Testing Z -> e+e- (end-to-end)")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = Z_TO_EE_SCRIPT
        tool.script_name = "test_z_to_ee"
        result = json.loads(tool._run())

        all_passed = True

        ok = result["success"]
        if not ok:
            all_passed = False
            print(f"  [✗] FAIL: script failed: {result.get('error_hint', result.get('stderr', ''))}")
        else:
            print(f"  [✓] PASS: script succeeded")

        ok = "stdout" not in result
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: stdout dropped (lean returns)")

        width = _parse_num(result["numerical"]["width_GeV"])
        ok = 0.05 < width < 0.12
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: Z->ee width = {width:.6f} GeV (expected ~0.084)")

        print()
        return all_passed
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# Basic Dirac trace
# ---------------------------------------------------------------------------

TRACE_SCRIPT = r"""
<< FeynCalc`
res = DiracTrace[GSD[p].GSD[q]] // DiracSimplify;
Print["SYMBOLIC_RESULT[trace]: ", res]

(* Verify numerically using Pair replacement *)
num = res /. Pair[Momentum[p, ___], Momentum[q, ___]] -> 1;
Print["NUMERICAL_RESULT[trace_at_pq1]: ", N[num]]
Print["STATUS: complete"]
"""


def test_basic_trace():
    """Test basic Dirac trace end-to-end."""
    print("=" * 60)
    print("Testing basic Dirac trace (end-to-end)")
    print("=" * 60)

    skip = _check_wolfram()
    if skip:
        print(f"  [–] SKIP: {skip}")
        print()
        return True

    tmp_dir = tempfile.mkdtemp()
    try:
        tool = _make_tool(tmp_dir)
        tool.code = TRACE_SCRIPT
        result = json.loads(tool._run())

        all_passed = True

        ok = result["success"]
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: script succeeded")

        ok = "stdout" not in result
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: stdout dropped (lean returns)")

        val = _parse_num(result["numerical"]["trace_at_pq1"])
        ok = abs(val - 4.0) < 0.01
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: Tr[p/ q/] at p.q=1 = {val} (expected 4)")

        print()
        return all_passed
    finally:
        shutil.rmtree(tmp_dir)


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("End-to-End FeynCalc Tests")
    print("=" * 60 + "\n")

    tests = [
        ("H -> b bbar", test_h_to_bb),
        ("Z -> e+e-", test_z_to_ee),
        ("Basic Dirac trace", test_basic_trace),
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
