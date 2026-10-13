#!/usr/bin/env python3
"""
# test_feyncalc_codegen.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for FeynCalcCodeGenerator — verifies generated Mathematica code
strings without requiring wolframscript (pure Python).

Run with:
    python test_feyncalc_codegen.py
"""

import json
import sys
from pathlib import Path

# Add repo root to path
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.eda.feyncalc_codegen import (
    FeynCalcCodeGenerator,
    GeneratedCode,
    ProcessType,
    Channel,
    _is_antiparticle,
    _safe_symbol,
)
from tools.nda.simple_diagram import (
    Diagram, Particle, Vertex, Propagator, parse_diagram,
)


# ---------------------------------------------------------------------------
# Diagram factories
# ---------------------------------------------------------------------------

def _make_h_to_bb() -> Diagram:
    """H -> b bbar: SFF scalar vertex, scalar parent, tree-level."""
    return parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "scalar", "coupling": "y_b"}],
        "couplings": {"y_b": 0.024},
        "color_factor": 3.0,
    })


def _make_z_to_ee() -> Diagram:
    """Z -> e+ e-: VFF left-handed vertex, massive vector parent."""
    return parse_diagram({
        "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
        ],
        "vertices": [{"type": "left-handed", "coupling": "g_Z"}],
        "couplings": {"g_Z": 0.074},
        "color_factor": 1.0,
    })


def _make_ee_to_mumu() -> Diagram:
    """e+ e- -> mu+ mu-: 2->2, s-channel photon."""
    return parse_diagram({
        "initial": [
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
        ],
        "final": [
            {"label": "mu+", "spin": "1/2", "mass": 0.1057},
            {"label": "mu-", "spin": "1/2", "mass": 0.1057},
        ],
        "vertices": [{"type": "vector", "coupling": "e_em"}],
        "couplings": {"e_em": 0.303},
        "propagators": [{"label": "gamma", "mass": 0, "spin": 1}],
        "color_factor": 1.0,
    })


def _make_phi_to_phiphi() -> Diagram:
    """phi -> phi phi: SSS pure scalar cubic."""
    return parse_diagram({
        "initial": [{"label": "phi", "spin": 0, "mass": 100.0}],
        "final": [
            {"label": "phi1", "spin": 0, "mass": 10.0},
            {"label": "phi2", "spin": 0, "mass": 10.0},
        ],
        "vertices": [{"type": "scalar-3pt", "coupling": "lam"}],
        "couplings": {"lam": 0.1},
        "color_factor": 1.0,
    })


def _make_s_to_vv() -> Diagram:
    """S -> V V: SVV vertex (like H -> WW*)."""
    return parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "W+", "spin": 1, "mass": 80.4},
            {"label": "W-", "spin": 1, "mass": 80.4},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g_HWW"}],
        "couplings": {"g_HWW": 0.3},
        "color_factor": 1.0,
    })


def _make_unsupported_loop() -> Diagram:
    """Loop diagram — should be classified UNSUPPORTED."""
    return parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "gamma", "spin": 1, "mass": 0},
            {"label": "gamma", "spin": 1, "mass": 0},
        ],
        "vertices": [
            {"type": "yukawa", "coupling": "y_t"},
            {"type": "gauge-vector", "coupling": "e"},
        ],
        "couplings": {"y_t": 1.0, "e": 0.303},
        "propagators": [
            {"label": "t", "mass": 173.0, "spin": 0.5, "is_loop_propagator": True},
        ],
        "color_factor": 1.0,
    })


def _make_three_body() -> Diagram:
    """1->3 body — should be classified UNSUPPORTED."""
    return parse_diagram({
        "initial": [{"label": "mu-", "spin": "1/2", "mass": 0.1057}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "nu_ebar", "spin": "1/2", "mass": 0},
            {"label": "nu_mu", "spin": "1/2", "mass": 0},
        ],
        "vertices": [{"type": "dim6-4fermion", "coupling": "GF"}],
        "couplings": {"GF": 1.166e-5},
        "color_factor": 1.0,
    })


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_helpers():
    """Test _is_antiparticle and _safe_symbol helper functions."""
    print("=" * 60)
    print("Testing helper functions")
    print("=" * 60)

    all_passed = True

    # _is_antiparticle
    checks = [
        (_is_antiparticle("bbar"), True, "bbar is antiparticle"),
        (_is_antiparticle("e+"), True, "e+ is antiparticle"),
        (_is_antiparticle("mu+"), True, "mu+ is antiparticle"),
        (_is_antiparticle("W~"), True, "W~ is antiparticle"),
        (_is_antiparticle("b"), False, "b is not antiparticle"),
        (_is_antiparticle("e-"), False, "e- is not antiparticle"),
        (_is_antiparticle("H"), False, "H is not antiparticle"),
        (_is_antiparticle(None), False, "None is not antiparticle"),
    ]
    for actual, expected, desc in checks:
        ok = actual == expected
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: _is_antiparticle: {desc}")

    # _safe_symbol
    sym_checks = [
        (_safe_symbol("e+"), "ep", "e+ -> ep"),
        (_safe_symbol("e-"), "em", "e- -> em"),
        (_safe_symbol("bbar"), "bbar", "bbar -> bbar"),
        (_safe_symbol(None), "X", "None -> X"),
    ]
    for actual, expected, desc in sym_checks:
        ok = actual == expected
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: _safe_symbol: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Process classification
# ---------------------------------------------------------------------------

def test_classification():
    """Test process type classification."""
    print("=" * 60)
    print("Testing process classification")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # DECAY_1TO2
    result = gen.generate(_make_h_to_bb())
    ok = result.process_type == ProcessType.DECAY_1TO2
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: H->bb classified as DECAY_1TO2")

    # DECAY_1TO2 (VFF variant)
    result = gen.generate(_make_z_to_ee())
    ok = result.process_type == ProcessType.DECAY_1TO2
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: Z->ee classified as DECAY_1TO2")

    # SCATTERING_2TO2
    result = gen.generate(_make_ee_to_mumu(), sqrt_s=91.2)
    ok = result.process_type == ProcessType.SCATTERING_2TO2
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: e+e- -> mu+mu- classified as SCATTERING_2TO2")

    # UNSUPPORTED — loop
    result = gen.generate(_make_unsupported_loop())
    ok = result.process_type == ProcessType.UNSUPPORTED and len(result.warnings) > 0
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loop diagram classified as UNSUPPORTED with warnings")

    # UNSUPPORTED — 3-body
    result = gen.generate(_make_three_body())
    ok = result.process_type == ProcessType.UNSUPPORTED
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: 3-body classified as UNSUPPORTED")

    # sqrt_s is OPTIONAL for 2->2: the cross section is built symbolically in
    # s, so a missing sqrt_s costs only the numeric evaluation, not the script.
    result = gen.generate(_make_ee_to_mumu())
    ok = (
        result.process_type == ProcessType.SCATTERING_2TO2
        and result.code != ""
        and "SYMBOLIC_RESULT[sigma]" in result.code
        and "NUMERICAL_RESULT[sigma_GeV2]" not in result.code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: 2->2 without sqrt_s still builds a symbolic sigma")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# H -> b bbar (SFF Yukawa)
# ---------------------------------------------------------------------------

def test_h_to_bb():
    """Test H -> b bbar code generation (SFF Yukawa vertex)."""
    print("=" * 60)
    print("Testing H -> b bbar (SFF Yukawa)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    result = gen.generate(_make_h_to_bb())
    code = result.code
    all_passed = True

    checks = [
        (code != "" and len(code) > 100, "generates non-trivial code"),
        ("<< FeynCalc`" in code, "loads FeynCalc"),
        ("amp =" in code, "has amplitude assignment"),
        ("SpinorUBar" in code and "SpinorV" in code, "has spinor structure"),
        ("0.024" in code, "has Yukawa coupling value"),
        ("mH" in code or "125" in code, "has parent mass"),
        ("mb" in code or "4.18" in code, "has daughter mass"),
        ("FermionSpinSum" in code, "has fermion spin sum"),
        ("FCClearScalarProducts[]" in code, "uses FCClearScalarProducts"),
        ("ScalarProduct[" in code, "uses ScalarProduct assignments"),
        ("width" in code.lower() and "pMag" in code, "has width formula"),
        ("colorFactor" in code and "3" in code, "has color factor"),
        ("SYMBOLIC_RESULT[width]" in code, "has symbolic result marker"),
        ("NUMERICAL_RESULT[width_GeV]" in code, "has numerical result marker"),
        ("STATUS: complete" in code, "has status marker"),
        ("DoPolarizationSums" not in code, "no polarization sums (scalar parent + fermion daughters)"),
        (result.momentum_map["initial_0"] == "p", "momentum map: parent -> p"),
        (result.momentum_map["final_0"] == "p1", "momentum map: daughter 0 -> p1"),
        (result.momentum_map["final_1"] == "p2", "momentum map: daughter 1 -> p2"),
        (len(result.warnings) == 0, "no warnings"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Z -> e+ e- (VFF gauge-axial)
# ---------------------------------------------------------------------------

def test_z_to_ee():
    """Test Z -> e+ e- code generation (VFF left-handed vertex)."""
    print("=" * 60)
    print("Testing Z -> e+ e- (VFF left-handed)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    result = gen.generate(_make_z_to_ee())
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("DoPolarizationSums" in code, "has polarization sum (massive vector parent)"),
        ("VirtualBoson" not in code, "physical massive vector: no VirtualBoson (uses -g + pp/m²)"),
        ("PolarizationVector[" in code, "has explicit PolarizationVector in amplitude"),
        ("GAD[" in code, "has gamma matrix"),
        ("GA[7]" in code, "has left-handed projector GA[7]"),
        ("FermionSpinSum" in code, "has fermion spin sum"),
        ("nInit = 3" in code, "spin averaging: Z has 3 spin states"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# phi -> phi phi (SSS)
# ---------------------------------------------------------------------------

def test_sss():
    """Test phi -> phi phi code generation (SSS pure scalar)."""
    print("=" * 60)
    print("Testing phi -> phi phi (SSS scalar cubic)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    result = gen.generate(_make_phi_to_phiphi())
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("SpinorU" not in code and "SpinorV" not in code and "SpinorUBar" not in code,
         "no spinors (pure scalar)"),
        ("FermionSpinSum" not in code, "no fermion spin sum"),
        ("DoPolarizationSums" not in code, "no polarization sum"),
        ("I (" in code or "I*" in code, "has I*g amplitude structure"),
        ("nInit = 1" in code, "spin averaging: scalar has 1 spin state"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# S -> V V (SVV)
# ---------------------------------------------------------------------------

def test_svv():
    """Test S -> V V code generation (SVV vertex, like H -> WW)."""
    print("=" * 60)
    print("Testing S -> V V (SVV vertex)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    result = gen.generate(_make_s_to_vv())
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("MTD[" in code, "has metric tensor"),
        ("DoPolarizationSums" in code, "has polarization sums (two vector daughters)"),
        ("VirtualBoson" not in code, "physical massive vectors: no VirtualBoson (uses -g + pp/m²)"),
        ("PolarizationVector[" in code, "has explicit PolarizationVector in amplitude"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# e+ e- -> mu+ mu- (2->2 scattering)
# ---------------------------------------------------------------------------

def test_scattering():
    """Test e+e- -> mu+mu- code generation (2->2, s-channel photon)."""
    print("=" * 60)
    print("Testing e+ e- -> mu+ mu- (2->2 scattering)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    result = gen.generate(_make_ee_to_mumu(), sqrt_s=91.2)
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        (result.process_type == ProcessType.SCATTERING_2TO2, "process type is SCATTERING_2TO2"),
        ("FCClearScalarProducts[]" in code, "uses FCClearScalarProducts"),
        # Kinematics are explicit ScalarProduct assignments in (s, t) with u
        # eliminated on shell -- NOT SetMandelstam, which left u alive as an
        # independent symbol and blocked the t-integration.
        ("ScalarProduct[p1, p3]" in code, "has explicit (s,t) scalar products"),
        ("uExpr" in code, "eliminates u on shell"),
        ("SetMandelstam" not in code, "does not use SetMandelstam"),
        ("sigma" in code, "has cross section variable"),
        ("SpinorU" in code or "SpinorVBar" in code, "has spinors"),
        # FAD is a LOOP denominator and drags in D-4 pieces; a tree propagator
        # must be a plain scalar 1/(q^2 - M^2).
        ("FAD[" not in code, "no FAD in a tree amplitude"),
        ("ChangeDimension[amp, 4]" in code, "amplitude normalised to 4 dimensions"),
        ("symmetryFactor" in code, "has identical-particle symmetry factor"),
        ("SYMBOLIC_RESULT[sigma]" in code, "has symbolic sigma marker"),
        ("NUMERICAL_RESULT[sigma_GeV2]" in code, "has numerical sigma marker"),
        ("kallen" in code, "has Kallen function"),
        ("91.2" in code, "has sqrt_s value"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Unsupported topologies — graceful failure
# ---------------------------------------------------------------------------

def test_unsupported():
    """Test graceful handling of unsupported topologies."""
    print("=" * 60)
    print("Testing unsupported topologies")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Loop
    result = gen.generate(_make_unsupported_loop())
    ok = result.code == "" and result.process_type == ProcessType.UNSUPPORTED
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: loop returns empty code + UNSUPPORTED")

    # 3-body
    result = gen.generate(_make_three_body())
    ok = result.code == "" and result.process_type == ProcessType.UNSUPPORTED
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: 3-body returns empty code + UNSUPPORTED")

    # Warnings present
    result = gen.generate(_make_unsupported_loop())
    ok = len(result.warnings) > 0 and (
        "loop" in result.warnings[0].lower() or "Unsupported" in result.warnings[0]
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: warnings mention loop/unsupported")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Pseudoscalar vertex: A0 -> t tbar
# ---------------------------------------------------------------------------

def test_pseudoscalar():
    """Test pseudoscalar Yukawa vertex (A0 -> t tbar)."""
    print("=" * 60)
    print("Testing A0 -> t tbar (pseudoscalar vertex)")
    print("=" * 60)

    diagram = parse_diagram({
        "initial": [{"label": "A0", "spin": 0, "mass": 300.0}],
        "final": [
            {"label": "t", "spin": "1/2", "mass": 173.0},
            {"label": "tbar", "spin": "1/2", "mass": 173.0},
        ],
        "vertices": [{"type": "pseudoscalar", "coupling": "y_t"}],
        "couplings": {"y_t": 1.0},
        "color_factor": 3.0,
    })

    gen = FeynCalcCodeGenerator()
    result = gen.generate(diagram)
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("GA[5]" in code, "has GA[5] for pseudoscalar coupling"),
        ("SpinorUBar" in code and "SpinorV" in code, "has spinor structure"),
        (result.process_type == ProcessType.DECAY_1TO2, "classified as DECAY_1TO2"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Chiral VFF vertex: Z -> e+ e- with gL/gR
# ---------------------------------------------------------------------------

def test_chiral_vff():
    """Test chiral VFF vertex with dict coupling (Z -> e+ e-)."""
    print("=" * 60)
    print("Testing Z -> e+ e- (chiral vertex, gL/gR)")
    print("=" * 60)

    diagram = parse_diagram({
        "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": 0.27, "gR": 0.23}}],
        "color_factor": 1.0,
    })

    gen = FeynCalcCodeGenerator()
    result = gen.generate(diagram)
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("GA[7]" in code, "has left-handed projector GA[7]"),
        ("GA[6]" in code, "has right-handed projector GA[6]"),
        ("0.27" in code, "has gL coupling value"),
        ("0.23" in code, "has gR coupling value"),
        ("PolarizationVector[" in code, "has polarization vector"),
        (result.process_type == ProcessType.DECAY_1TO2, "classified as DECAY_1TO2"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Axial-vector vertex: V -> f fbar
# ---------------------------------------------------------------------------

def test_axial_vector():
    """Test axial-vector VFF vertex (V -> f fbar)."""
    print("=" * 60)
    print("Testing V -> f fbar (axial-vector vertex)")
    print("=" * 60)

    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "axial-vector", "coupling": "gA"}],
        "couplings": {"gA": 0.5},
        "color_factor": 1.0,
    })

    gen = FeynCalcCodeGenerator()
    result = gen.generate(diagram)
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("GA[5]" in code, "has GA[5] for axial-vector coupling"),
        ("GAD[" in code, "has gamma matrix GAD"),
        # Axial-vector: γ^μ γ^5, NOT a chiral projector
        ("GA[7]" not in code and "GA[6]" not in code, "no chiral projectors (pure axial)"),
        (result.process_type == ProcessType.DECAY_1TO2, "classified as DECAY_1TO2"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Right-handed vertex: Z' -> f fbar
# ---------------------------------------------------------------------------

def test_right_handed():
    """Test right-handed VFF vertex (Z' -> f fbar)."""
    print("=" * 60)
    print("Testing Z' -> f fbar (right-handed vertex)")
    print("=" * 60)

    diagram = parse_diagram({
        "initial": [{"label": "Zp", "spin": 1, "mass": 1000.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "right-handed", "coupling": "gR"}],
        "couplings": {"gR": 0.3},
        "color_factor": 1.0,
    })

    gen = FeynCalcCodeGenerator()
    result = gen.generate(diagram)
    code = result.code
    all_passed = True

    checks = [
        (code != "", "generates code"),
        ("GA[6]" in code, "has right-handed projector GA[6]"),
        ("GA[7]" not in code, "no left-handed projector"),
        (result.process_type == ProcessType.DECAY_1TO2, "classified as DECAY_1TO2"),
    ]

    for ok, desc in checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# GA[5] convention regression test
# ---------------------------------------------------------------------------

def test_ga5_convention():
    """Regression test: bare GA5 and GAD[5] must never appear in generated code."""
    print("=" * 60)
    print("Testing GA[5] convention (no bare GA5 or GAD[5])")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Test several diagrams that involve gamma-5
    test_cases = [
        ("Z -> e+e- (left-handed)", _make_z_to_ee()),
        ("A0 -> t tbar (pseudoscalar)", parse_diagram({
            "initial": [{"label": "A0", "spin": 0, "mass": 300.0}],
            "final": [
                {"label": "t", "spin": "1/2", "mass": 173.0},
                {"label": "tbar", "spin": "1/2", "mass": 173.0},
            ],
            "vertices": [{"type": "pseudoscalar", "coupling": 1.0}],
            "color_factor": 3.0,
        })),
        ("chiral Z -> e+e-", parse_diagram({
            "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
            "final": [
                {"label": "e-", "spin": "1/2", "mass": 0.000511},
                {"label": "e+", "spin": "1/2", "mass": 0.000511},
            ],
            "vertices": [{"type": "chiral", "coupling": {"gL": 0.27, "gR": 0.23}}],
        })),
    ]

    import re
    # Pattern matches bare GA5 (not inside GA[5]) — i.e., GA5 not preceded by GA[
    bare_ga5_pattern = re.compile(r'(?<!GA\[)GA5(?!\])')
    gad5_pattern = re.compile(r'GAD\[5\]')

    for label, diagram in test_cases:
        result = gen.generate(diagram)
        code = result.code
        has_bare_ga5 = bool(bare_ga5_pattern.search(code))
        has_gad5 = bool(gad5_pattern.search(code))
        ok = not has_bare_ga5 and not has_gad5
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {label} — no bare GA5 or GAD[5]")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# LaTeX and intermediate symbolic extraction markers
# ---------------------------------------------------------------------------

def test_latex_markers():
    """Test that generated code emits TeXForm/LATEX_RESULT markers."""
    print("=" * 60)
    print("Testing LaTeX and intermediate symbolic markers")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Decay: H -> b bbar
    result = gen.generate(_make_h_to_bb())
    code = result.code

    decay_checks = [
        ('SYMBOLIC_RESULT[ampSq]' in code, "decay: has SYMBOLIC_RESULT[ampSq] marker"),
        ('LATEX_RESULT[ampSq]' in code, "decay: has LATEX_RESULT[ampSq] marker"),
        ('ToString[TeXForm[ampSqKin]]' in code, "decay: uses ToString[TeXForm[ampSqKin]]"),
        ('SYMBOLIC_RESULT[width]' in code, "decay: has SYMBOLIC_RESULT[width] marker"),
        ('LATEX_RESULT[width]' in code, "decay: has LATEX_RESULT[width] marker"),
        ('ToString[TeXForm[width]]' in code, "decay: uses ToString[TeXForm[width]]"),
        ('NUMERICAL_RESULT[width_GeV]' in code, "decay: has NUMERICAL_RESULT[width_GeV]"),
        ('NUMERICAL_RESULT[width_MeV]' in code, "decay: has NUMERICAL_RESULT[width_MeV]"),
    ]

    for ok, desc in decay_checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    # Scattering: e+e- -> mu+mu-
    result = gen.generate(_make_ee_to_mumu(), sqrt_s=91.2)
    code = result.code

    scat_checks = [
        ('SYMBOLIC_RESULT[ampSq]' in code, "scattering: has SYMBOLIC_RESULT[ampSq] marker"),
        ('LATEX_RESULT[ampSq]' in code, "scattering: has LATEX_RESULT[ampSq] marker"),
        ('SYMBOLIC_RESULT[sigma]' in code, "scattering: has SYMBOLIC_RESULT[sigma] marker"),
        ('LATEX_RESULT[sigma]' in code, "scattering: has LATEX_RESULT[sigma] marker"),
        ('ToString[TeXForm[sigma]]' in code, "scattering: uses ToString[TeXForm[sigma]]"),
        ('NUMERICAL_RESULT[sigma_GeV2]' in code, "scattering: has NUMERICAL_RESULT[sigma_GeV2]"),
    ]

    for ok, desc in scat_checks:
        if not ok:
            all_passed = False
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {desc}")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Complex coupling conjugation
# ---------------------------------------------------------------------------

def test_complex_coupling_conjugation():
    """Tests for coupling conjugation via /. replacement rules on ComplexConjugate."""
    print("=" * 60)
    print("Testing complex coupling conjugation")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Default: symbolic couplings get conjugation replacement rules
    diagram = parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "scalar", "coupling": "y_b"}],
        "color_factor": 3.0,
    })
    result = gen.generate(diagram)
    ok = "y_b -> Conjugate[y_b]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: symbolic coupling gets conjugation rule")

    # Numeric coupling values should not get conjugation rules
    diagram = parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "scalar", "coupling": "y_b"}],
        "couplings": {"y_b": 0.024},
        "color_factor": 3.0,
    })
    result = gen.generate(diagram)
    ok = "-> Conjugate[" not in result.code and "ComplexConjugate[amp]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: numeric coupling not in conjugate list")

    # assume_real_couplings=True
    diagram = parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "scalar", "coupling": "y_b"}],
        "color_factor": 3.0,
    })
    gen_real = FeynCalcCodeGenerator(assume_real_couplings=True)
    result = gen_real.generate(diagram)
    ok = "-> Conjugate[" not in result.code and "ComplexConjugate[amp]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: assume_real skips conjugate")

    # Chiral dict couplings: symbolic values get conjugation rules
    diagram = parse_diagram({
        "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = (
        "gL" in result.code and "gR" in result.code
        and "gL -> Conjugate[gL]" in result.code
        and "gR -> Conjugate[gR]" in result.code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: chiral symbolic couplings get conjugation")

    # Mixed numeric/symbolic chiral
    diagram = parse_diagram({
        "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": 0.27, "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "gR -> Conjugate[gR]" in result.code and "0.27 -> Conjugate" not in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: mixed numeric/symbolic chiral")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Chiral scalar vertex
# ---------------------------------------------------------------------------

def test_chiral_scalar():
    """Tests for chiral SFF vertex (scalar parent with dict coupling)."""
    print("=" * 60)
    print("Testing chiral scalar vertex")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # dict chiral coupling
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": "yL", "gR": "yR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = (
        result.process_type == ProcessType.DECAY_1TO2
        and "GA[7]" in code and "GA[6]" in code
        and "yL" in code and "yR" in code
        and "{'gL'" not in code
        and "GAD[" not in code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dict chiral coupling")

    # string chiral coupling
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "chiral", "coupling": "y"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "yL" in code and "yR" in code and "GA[7]" in code and "GA[6]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: string chiral coupling derives names")

    # pseudoscalar still uses GA[5]
    diagram = parse_diagram({
        "initial": [{"label": "A0", "spin": 0, "mass": 300.0}],
        "final": [
            {"label": "t", "spin": "1/2", "mass": 173.0},
            {"label": "tbar", "spin": "1/2", "mass": 173.0},
        ],
        "vertices": [{"type": "pseudoscalar", "coupling": "y_t"}],
        "couplings": {"y_t": 1.0},
        "color_factor": 3.0,
    })
    result = gen.generate(diagram)
    ok = "GA[5]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: pseudoscalar still uses GA[5]")

    # plain scalar Yukawa
    diagram = _make_h_to_bb()
    result = gen.generate(diagram)
    ok = (
        "SpinorUBar" in result.code and "SpinorV" in result.code
        and "GA[7]" not in result.code and "GA[6]" not in result.code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: plain scalar Yukawa (no chiral projectors)")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Polarization sum massive vs massless
# ---------------------------------------------------------------------------

def test_polarization_sum():
    """Tests for correct DoPolarizationSums calls."""
    print("=" * 60)
    print("Testing polarization sum (massive vs massless)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Massive vector: no gauge reference
    diagram = parse_diagram({
        "initial": [{"label": "Z", "spin": 1, "mass": 91.19}],
        "final": [
            {"label": "e-", "spin": "1/2", "mass": 0.000511},
            {"label": "e+", "spin": "1/2", "mass": 0.000511},
        ],
        "vertices": [{"type": "left-handed", "coupling": "g_Z"}],
        "couplings": {"g_Z": 0.074},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "DoPolarizationSums[ampSq, p];" in result.code and "DoPolarizationSums[ampSq, p, 0]" not in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: massive vector no gauge ref")

    # Massless vector: has gauge reference
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "gamma", "spin": 1, "mass": 0},
            {"label": "gamma", "spin": 1, "mass": 0},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
        "couplings": {"g": 0.1},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = (
        ", 0];" in result.code
        and "DoPolarizationSums[ampSq, p1, 0]" in result.code
        and "DoPolarizationSums[ampSq, p2, 0]" in result.code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: massless vector has gauge ref")

    # Massive daughter vectors: no gauge reference
    diagram = _make_s_to_vv()
    result = gen.generate(diagram)
    ok = (
        "DoPolarizationSums[ampSq, p1];" in result.code
        and "DoPolarizationSums[ampSq, p2];" in result.code
        and ", 0]" not in result.code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: massive daughter vectors no gauge ref")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Integer color factor
# ---------------------------------------------------------------------------

def test_integer_color_factor():
    """Tests for integer color factor emission."""
    print("=" * 60)
    print("Testing integer color factor")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Whole number
    diagram = _make_h_to_bb()
    result = gen.generate(diagram)
    ok = "colorFactor = 3;" in result.code and "colorFactor = 3.0" not in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: color_factor=3.0 -> 'colorFactor = 3'")

    # Non-integer
    diagram = parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "scalar", "coupling": "y_b"}],
        "couplings": {"y_b": 0.024},
        "color_factor": 1.5,
    })
    result = gen.generate(diagram)
    ok = "colorFactor = 1.5;" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: color_factor=1.5 kept as float")

    # Mass integer form
    diagram = _make_h_to_bb()
    result = gen.generate(diagram)
    ok = "mH = 125;" in result.code and "mb = 4.18;" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: mass integer form (125.0 -> 125)")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Vertex type aliases
# ---------------------------------------------------------------------------

def test_vertex_type_aliases():
    """Tests for vertex type alias resolution."""
    print("=" * 60)
    print("Testing vertex type aliases")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # yukawa-chiral
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "yukawa-chiral", "coupling": {"gL": "yL", "gR": "yR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "GA[7]" in code and "GA[6]" in code and "yL" in code and "yR" in code and "GAD[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: yukawa-chiral alias")

    # scalar-chiral
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "scalar-chiral", "coupling": {"gL": "yL", "gR": "yR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "GA[7]" in code and "GA[6]" in code and "GAD[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: scalar-chiral alias")

    # vector-chiral
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "vector-chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "GAD[" in code and "GA[7]" in code and "GA[6]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: vector-chiral alias")

    # gauge-vector (regression)
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
        "couplings": {"g": 0.3},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "GAD[" in result.code and result.process_type == ProcessType.DECAY_1TO2
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: gauge-vector alias")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# V-A vertex type
# ---------------------------------------------------------------------------

def test_va_vertex_type():
    """Tests for vector-axial (V-A) vertex type."""
    print("=" * 60)
    print("Testing V-A vertex type")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # string coupling
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "vector-axial", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "gV" in code and "gA" in code and "GA[5]" in code and "GAD[" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: vector-axial string coupling")

    # dict coupling
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "vector-axial", "coupling": {"gV": "gV_e", "gA": "gA_e"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "gV_e" in code and "gA_e" in code and "GA[5]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: vector-axial dict coupling")

    # 'va' alias
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "va", "coupling": {"gV": "gV", "gA": "gA"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "gV" in code and "gA" in code and "GA[5]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: 'va' alias")

    # scalar-va string
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "scalar-va", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "gS" in code and "gP" in code and "GA[5]" in code and "GAD[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: scalar-va string")

    # scalar-va dict
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "scalar-va", "coupling": {"gS": "y_S", "gP": "y_P"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "y_S" in code and "y_P" in code and "GA[5]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: scalar-va dict")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Tensor/dipole vertex types
# ---------------------------------------------------------------------------

def test_tensor_vertex_type():
    """Tests for tensor/dipole vertex types."""
    print("=" * 60)
    print("Testing tensor/dipole vertex types")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # tensor type
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "tensor", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "DiracSigma[" in code and "GA[" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tensor type")

    # dipole alias
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "dipole", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "DiracSigma[" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dipole alias")

    # tensor-chiral
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "tensor-chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "DiracSigma[" in code and "GA[7]" in code and "GA[6]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tensor-chiral")

    # dipole-chiral alias
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "dipole-chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "DiracSigma[" in code and "GA[7]" in code and "GA[6]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dipole-chiral alias")

    # tensor needs momentum
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "tensor", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "FV[" in code and "nuT" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tensor has FV momentum")

    # F -> F V tensor
    diagram = parse_diagram({
        "initial": [{"label": "f", "spin": "1/2", "mass": 10.0}],
        "final": [
            {"label": "f2", "spin": "1/2", "mass": 1.0},
            {"label": "gamma", "spin": 1, "mass": 0},
        ],
        "vertices": [{"type": "tensor", "coupling": "d"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "DiracSigma[" in code and "FV[" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: F -> F V tensor")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# SFF crossed channel (F -> f' S)
# ---------------------------------------------------------------------------

def test_sff_crossed_channel():
    """Tests for F -> f' S (fermion-initial SFF vertex)."""
    print("=" * 60)
    print("Testing SFF crossed channel (F -> f' S)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # scalar-va F -> f S
    diagram = parse_diagram({
        "initial": [{"label": "F", "spin": "1/2", "mass": 10.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "S", "spin": 0, "mass": 0.5},
        ],
        "vertices": [{"type": "scalar-va", "coupling": {"gS": "gS", "gP": "gP"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = (
        result.process_type == ProcessType.DECAY_1TO2
        and "SpinorUBar" in code and "SpinorU[" in code
        and "gS" in code and "gP" in code and "GA[5]" in code
        and "{'gS'" not in code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: scalar-va F -> f S")

    # Yukawa F -> f S
    diagram = parse_diagram({
        "initial": [{"label": "F", "spin": "1/2", "mass": 10.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "S", "spin": 0, "mass": 0.5},
        ],
        "vertices": [{"type": "yukawa", "coupling": "y"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "SpinorUBar" in result.code and "SpinorU[" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: Yukawa F -> f S")

    # chiral F -> f S
    diagram = parse_diagram({
        "initial": [{"label": "F", "spin": "1/2", "mass": 10.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "S", "spin": 0, "mass": 0.5},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": "yL", "gR": "yR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "GA[7]" in code and "GA[6]" in code and "yL" in code and "yR" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: chiral F -> f S")

    # scalar first in final state
    diagram = parse_diagram({
        "initial": [{"label": "F", "spin": "1/2", "mass": 10.0}],
        "final": [
            {"label": "S", "spin": 0, "mass": 0.5},
            {"label": "f", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "scalar-va", "coupling": {"gS": "gS", "gP": "gP"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "SpinorUBar[p2" in code and "SpinorU[p" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: F -> S f (scalar first)")

    # S -> f fbar regression
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "scalar-va", "coupling": {"gS": "gS", "gP": "gP"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "SpinorUBar" in code and "SpinorV" in code and "gS" in code and "GA[5]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: S -> f fbar regression")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# V -> S V vertex
# ---------------------------------------------------------------------------

def test_vsv_vertex():
    """Tests for V -> S V (vector parent, scalar + vector daughters)."""
    print("=" * 60)
    print("Testing V -> S V vertex")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # parent polarization
    diagram = parse_diagram({
        "initial": [{"label": "Zp", "spin": 1, "mass": 1000.0}],
        "final": [
            {"label": "H", "spin": 0, "mass": 125.0},
            {"label": "Z", "spin": 1, "mass": 91.19},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g_ZpHZ"}],
        "couplings": {"g_ZpHZ": 0.1},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = (
        result.process_type == ProcessType.DECAY_1TO2
        and "PolarizationVector[p, mu0]" in code
        and "PolarizationVector[p2, mu1]" in code
        and "MTD[mu0, mu1]" in code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: V -> S V parent polarization")

    # scalar first
    diagram = parse_diagram({
        "initial": [{"label": "Zp", "spin": 1, "mass": 1000.0}],
        "final": [
            {"label": "H", "spin": 0, "mass": 125.0},
            {"label": "Z", "spin": 1, "mass": 91.19},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
        "couplings": {"g": 0.1},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    ok = "PolarizationVector[p2, mu1]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: V -> S V scalar first")

    # vector first
    diagram = parse_diagram({
        "initial": [{"label": "Zp", "spin": 1, "mass": 1000.0}],
        "final": [
            {"label": "Z", "spin": 1, "mass": 91.19},
            {"label": "H", "spin": 0, "mass": 125.0},
        ],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
        "couplings": {"g": 0.1},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "PolarizationVector[p1, mu1]" in code and "PolarizationVector[p, mu0]" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: V -> V S vector first")

    # S -> V V regression
    diagram = _make_s_to_vv()
    result = gen.generate(diagram)
    code = result.code
    ok = (
        "PolarizationVector[p1, mu1]" in code
        and "PolarizationVector[p2, mu2]" in code
        and "MTD[mu1, mu2]" in code
        and "PolarizationVector[p, " not in code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: S -> V V regression")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Symmetry factor
# ---------------------------------------------------------------------------

def test_symmetry_factor():
    """Tests for identical-particle symmetry factor in width formula."""
    print("=" * 60)
    print("Testing symmetry factor")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # Identical labels
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 500.0}],
        "final": [
            {"label": "gamma", "spin": 1, "mass": 0.0},
            {"label": "gamma", "spin": 1, "mass": 0.0},
        ],
        "vertices": [{"type": "scalar-3pt", "coupling": 0.1}],
        "couplings": {},
    })
    result = gen.generate(diagram)
    ok = "symmetryFactor = 2" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: identical labels -> factor 2")

    # Distinct labels
    diagram = parse_diagram({
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
            {"label": "b", "spin": "1/2", "mass": 4.18},
            {"label": "bbar", "spin": "1/2", "mass": 4.18},
        ],
        "vertices": [{"type": "yukawa", "coupling": 0.024}],
        "couplings": {},
        "color_factor": 3.0,
    })
    result = gen.generate(diagram)
    ok = "symmetryFactor = 1" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: distinct labels -> factor 1")

    # Explicit override
    diagram = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 500.0}],
        "final": [
            {"label": "a", "spin": 0, "mass": 0.0},
            {"label": "b", "spin": 0, "mass": 0.0},
        ],
        "vertices": [{"type": "scalar-3pt", "coupling": 0.1}],
        "couplings": {},
        "symmetry_factor": 6,
    })
    result = gen.generate(diagram)
    ok = "symmetryFactor = 6" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: explicit override -> factor 6")

    # compute_symmetry_factor helper
    from tools.nda.simple_diagram import compute_symmetry_factor
    d = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 100.0}],
        "final": [{"label": "X", "spin": 0, "mass": 0.0}, {"label": "X", "spin": 0, "mass": 0.0}],
        "vertices": [{"type": "scalar-3pt", "coupling": 0.1}],
    })
    d2 = parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 100.0}],
        "final": [{"label": "X", "spin": 0, "mass": 0.0}, {"label": "Y", "spin": 0, "mass": 0.0}],
        "vertices": [{"type": "scalar-3pt", "coupling": 0.1}],
    })
    ok = compute_symmetry_factor(d) == 2 and compute_symmetry_factor(d2) == 1
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: compute_symmetry_factor helper")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Field strength vertices
# ---------------------------------------------------------------------------

def _make_svv_fs(vtype):
    return parse_diagram({
        "initial": [{"label": "S", "spin": 0, "mass": 500.0}],
        "final": [
            {"label": "V1", "spin": 1, "mass": 0.0},
            {"label": "V2", "spin": 1, "mass": 0.0},
        ],
        "vertices": [{"type": vtype, "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })


def test_field_strength_vertices():
    """Tests for field-strength and dual-field-strength dim-5 operator codegen."""
    print("=" * 60)
    print("Testing field strength vertices")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # field-strength
    result = gen.generate(_make_svv_fs("field-strength"))
    code = result.code
    ok = "SPD[p1, p2]" in code and "FVD[p1," in code and "FVD[p2," in code and "PolarizationVector" in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: field-strength has SPD, FVD")

    # dim5-FF alias
    result = gen.generate(_make_svv_fs("dim5-FF"))
    ok = "SPD[p1, p2]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dim5-FF alias")

    # dual-field-strength
    result = gen.generate(_make_svv_fs("dual-field-strength"))
    code = result.code
    ok = "Eps[" in code and "PolarizationVector" in code and "SPD[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dual-field-strength has Eps, no SPD")

    # dim5-FF-dual alias
    result = gen.generate(_make_svv_fs("dim5-FF-dual"))
    ok = "Eps[" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: dim5-FF-dual alias")

    # default SVV unchanged
    result = gen.generate(_make_svv_fs("scalar-3pt"))
    code = result.code
    ok = "MTD[mu1, mu2]" in code and "SPD[p1, p2]" not in code and "Eps[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: default SVV uses MTD (no SPD/Eps)")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# Tensor dimension fix (4D objects)
# ---------------------------------------------------------------------------

def test_tensor_dimension_fix():
    """Tests for tensor/dipole vertex using 4D objects (GA/FV, not GAD/FVD)."""
    print("=" * 60)
    print("Testing tensor dimension fix (4D objects)")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # tensor uses 4D
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "tensor", "coupling": "g"}],
        "couplings": {},
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = (
        "DiracSigma[GA[" in code
        and "FV[" in code
        and "DiracSigma[GAD[" not in code
        and "FVD[p, nuT]" not in code
    )
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tensor uses 4D objects")

    # tensor-chiral uses 4D
    diagram = parse_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": 200.0}],
        "final": [
            {"label": "f", "spin": "1/2", "mass": 1.0},
            {"label": "fbar", "spin": "1/2", "mass": 1.0},
        ],
        "vertices": [{"type": "tensor-chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
        "color_factor": 1.0,
    })
    result = gen.generate(diagram)
    code = result.code
    ok = "DiracSigma[GA[" in code and "FV[" in code and "DiracSigma[GAD[" not in code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tensor-chiral uses 4D objects")

    print()
    return all_passed


# ---------------------------------------------------------------------------
# TeXForm ToString wrapping
# ---------------------------------------------------------------------------

def test_texform_tostring():
    """Tests for ToString[TeXForm[...]] wrapping in generated code."""
    print("=" * 60)
    print("Testing TeXForm ToString wrapping")
    print("=" * 60)

    gen = FeynCalcCodeGenerator()
    all_passed = True

    # decay
    diagram = _make_h_to_bb()
    result = gen.generate(diagram)
    ok = "ToString[TeXForm[ampSqKin]]" in result.code and "ToString[TeXForm[width]]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: decay TeXForm wrapped")

    # scattering
    diagram = _make_ee_to_mumu()
    result = gen.generate(diagram, sqrt_s=91.2)
    ok = "ToString[TeXForm[ampSqKin]]" in result.code and "ToString[TeXForm[sigma]]" in result.code
    if not ok:
        all_passed = False
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: scattering TeXForm wrapped")

    print()
    return all_passed


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FeynCalc Code Generator Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Helpers", test_helpers),
        ("Process Classification", test_classification),
        ("H -> b bbar (SFF scalar)", test_h_to_bb),
        ("Z -> e+ e- (VFF left-handed)", test_z_to_ee),
        ("phi -> phi phi (SSS)", test_sss),
        ("S -> V V (SVV)", test_svv),
        ("e+e- -> mu+mu- (2->2)", test_scattering),
        ("Unsupported Topologies", test_unsupported),
        ("A0 -> t tbar (pseudoscalar)", test_pseudoscalar),
        ("Z -> e+e- (chiral gL/gR)", test_chiral_vff),
        ("V -> f fbar (axial-vector)", test_axial_vector),
        ("Z' -> f fbar (right-handed)", test_right_handed),
        ("GA[5] convention", test_ga5_convention),
        ("LaTeX & intermediate markers", test_latex_markers),
        ("Complex coupling conjugation", test_complex_coupling_conjugation),
        ("Chiral scalar vertex", test_chiral_scalar),
        ("Polarization sum", test_polarization_sum),
        ("Integer color factor", test_integer_color_factor),
        ("Vertex type aliases", test_vertex_type_aliases),
        ("V-A vertex type", test_va_vertex_type),
        ("Tensor/dipole vertex types", test_tensor_vertex_type),
        ("SFF crossed channel", test_sff_crossed_channel),
        ("V -> S V vertex", test_vsv_vertex),
        ("Symmetry factor", test_symmetry_factor),
        ("Field strength vertices", test_field_strength_vertices),
        ("Tensor dimension fix", test_tensor_dimension_fix),
        ("TeXForm ToString wrapping", test_texform_tostring),
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
