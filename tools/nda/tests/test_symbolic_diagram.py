#!/usr/bin/env python3
"""
# test_symbolic_diagram.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for SymbolicDiagram, parse_symbolic_diagram, and resolve_diagram.

Run with:
    python test_symbolic_diagram.py
"""

import json
import sys
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.nda.symbolic_diagram import (
    SymbolicDiagram,
    SymbolicParticle,
    SymbolicVertex,
    SymbolicPropagator,
    parse_symbolic_diagram,
    build_diagram_from_symbolic,
    _is_massive_particle,
)
from tools.nda.diagram_resolution import resolve_diagram
from tools.nda.simple_diagram import Diagram


# ============================================================================
# SymbolicDiagram parsing
# ============================================================================

def test_parse_minimal_h_to_bb():
    """Parse H -> bb with just labels and vertex types."""
    print("=" * 60)
    print("Testing parse minimal H -> bb")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    assert isinstance(sym, SymbolicDiagram)
    assert sym.initial[0].label == "H"
    assert sym.initial[0].mass is None  # Not specified
    assert sym.initial[0].spin is None
    assert len(sym.final) == 2
    assert sym.final[0].label == "b"
    assert sym.vertices[0].type == "yukawa"
    assert sym.vertices[0].coupling == "y_b"
    assert sym.topology == "tree_2body"  # inferred

    print("  PASS")
    return True


def test_parse_with_topology():
    """Explicit topology is preserved."""
    print("=" * 60)
    print("Testing parse with topology")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "topology": "tree_3body",
        "initial": [{"label": "mu-"}],
        "final": [
            {"label": "e-"},
            {"label": "nu_ebar"},
            {"label": "nu_mu"},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g_W"},
            {"type": "gauge-vector", "coupling": "g_W"},
        ],
        "propagators": [{"label": "W-"}],
    })
    assert sym.topology == "tree_3body"
    assert len(sym.vertices) == 2
    assert len(sym.propagators) == 1
    assert sym.propagators[0].label == "W-"

    print("  PASS")
    return True


def test_parse_with_explicit_spin():
    """Spins specified as strings are parsed."""
    print("=" * 60)
    print("Testing parse with explicit spin")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H", "spin": 0}],
        "final": [
            {"label": "b", "spin": "1/2"},
            {"label": "bbar", "spin": "fermion"},
        ],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    assert sym.initial[0].spin == 0.0
    assert sym.final[0].spin == 0.5
    assert sym.final[1].spin == 0.5

    print("  PASS")
    return True


def test_parse_with_explicit_masses():
    """Masses are optional but preserved when given."""
    print("=" * 60)
    print("Testing parse with explicit masses")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H", "mass": 125.0}],
        "final": [
            {"label": "b", "mass": 4.18},
            {"label": "bbar"},
        ],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    assert sym.initial[0].mass == 125.0
    assert sym.final[0].mass == 4.18
    assert sym.final[1].mass is None

    print("  PASS")
    return True


def test_missing_initial_raises():
    """Missing initial raises ValueError."""
    print("=" * 60)
    print("Testing missing initial raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "final": [{"label": "b"}],
            "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "initial" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but missing 'initial' in message: {e}")
        return False


def test_missing_final_raises():
    """Missing final raises ValueError."""
    print("=" * 60)
    print("Testing missing final raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "initial": [{"label": "H"}],
            "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "final" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but missing 'final' in message: {e}")
        return False


def test_missing_vertices_raises():
    """Missing vertices raises ValueError."""
    print("=" * 60)
    print("Testing missing vertices raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "initial": [{"label": "H"}],
            "final": [{"label": "b"}, {"label": "bbar"}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "vertices" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but missing 'vertices' in message: {e}")
        return False


def test_numerical_coupling_raises():
    """Coupling must be a string name, not a number."""
    print("=" * 60)
    print("Testing numerical coupling raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "initial": [{"label": "H"}],
            "final": [{"label": "b"}, {"label": "bbar"}],
            "vertices": [{"type": "yukawa", "coupling": 0.024}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "symbolic name" in str(e) and "not a number" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_chiral_dict_coupling():
    """Dict coupling for chiral vertices parses correctly."""
    print("=" * 60)
    print("Testing chiral dict coupling")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "V", "spin": 1}],
        "final": [
            {"label": "tau-", "spin": "1/2"},
            {"label": "tau+", "spin": "1/2"},
        ],
        "vertices": [{"type": "chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
    })
    assert isinstance(sym.vertices[0].coupling, dict)
    assert sym.vertices[0].coupling == {"gL": "gL", "gR": "gR"}

    print("  PASS")
    return True


def test_chiral_dict_numeric_value_raises():
    """Dict coupling with numeric values should be rejected."""
    print("=" * 60)
    print("Testing chiral dict numeric value raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "initial": [{"label": "V", "spin": 1}],
            "final": [
                {"label": "f", "spin": "1/2"},
                {"label": "fbar", "spin": "1/2"},
            ],
            "vertices": [{"type": "chiral", "coupling": {"gL": 0.27, "gR": "gR"}}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "string names" in str(e) and "not numeric" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_list_coupling_raises():
    """List coupling should be rejected with helpful suggestion showing the dict form."""
    print("=" * 60)
    print("Testing list coupling raises")
    print("=" * 60)

    try:
        parse_symbolic_diagram({
            "initial": [{"label": "V", "spin": 1}],
            "final": [
                {"label": "f", "spin": "1/2"},
                {"label": "fbar", "spin": "1/2"},
            ],
            "vertices": [{"type": "chiral", "coupling": ["gL", "gR"]}],
        })
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        msg = str(e)
        if "does not accept a list" in msg and '"gL": "gL"' in msg and '"gR": "gR"' in msg:
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_color_factor():
    """Color factor is parsed from diagram dict."""
    print("=" * 60)
    print("Testing color factor")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        "color_factor": 3.0,
    })
    assert sym.color_factor == 3.0

    print("  PASS")
    return True


# ============================================================================
# SymbolicDiagram validation
# ============================================================================

def test_valid_diagram():
    """Valid diagram passes validation."""
    print("=" * 60)
    print("Testing valid diagram validation")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    is_valid, warnings = sym.validate()
    assert is_valid
    assert len(warnings) == 0

    print("  PASS")
    return True


def test_empty_label_warning():
    """Empty label triggers validation warning."""
    print("=" * 60)
    print("Testing empty label warning")
    print("=" * 60)

    sym = SymbolicDiagram(
        topology="tree_2body",
        initial=[SymbolicParticle(label="")],
        final=[SymbolicParticle(label="b"), SymbolicParticle(label="bbar")],
        vertices=[SymbolicVertex(type="yukawa", coupling="y_b")],
    )
    is_valid, warnings = sym.validate()
    assert not is_valid
    assert any("label" in w for w in warnings)

    print("  PASS")
    return True


# ============================================================================
# SymbolicDiagram to_dict
# ============================================================================

def test_round_trip():
    """Round-trip through parse and to_dict preserves data."""
    print("=" * 60)
    print("Testing round trip")
    print("=" * 60)

    original = {
        "topology": "tree_2body",
        "initial": [{"label": "H", "spin": 0.0}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        "color_factor": 3.0,
    }
    sym = parse_symbolic_diagram(original)
    d = sym.to_dict()
    assert d["topology"] == "tree_2body"
    assert d["initial"][0]["label"] == "H"
    assert d["color_factor"] == 3.0
    assert len(d["vertices"]) == 1

    print("  PASS")
    return True


# ============================================================================
# resolve_diagram
# ============================================================================

def test_resolve_h_to_bb():
    """Resolve H -> bb: masses and couplings come from SM databases."""
    print("=" * 60)
    print("Testing resolve H -> bb")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    diagram = resolve_diagram(sym)
    assert isinstance(diagram, Diagram)
    # Masses should be filled from SM database
    assert diagram.initial[0].mass is not None
    assert abs(diagram.initial[0].mass - 125.0) < 1.0
    assert diagram.initial[0].spin == 0.0
    assert abs(diagram.final[0].mass - 4.18) < 0.1
    assert diagram.final[0].spin == 0.5
    # Coupling should be resolved
    assert "y_b" in diagram.couplings
    assert abs(diagram.couplings["y_b"] - 0.0242) < 0.01

    print("  PASS")
    return True


def test_resolve_muon_decay_propagator():
    """Resolve mu- -> e- nu_ebar nu_mu with W propagator."""
    print("=" * 60)
    print("Testing resolve muon decay propagator")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "topology": "tree_3body",
        "initial": [{"label": "mu-"}],
        "final": [
            {"label": "e-"},
            {"label": "nu_ebar"},
            {"label": "nu_mu"},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g_W"},
            {"type": "gauge-vector", "coupling": "g_W"},
        ],
        "propagators": [{"label": "W-"}],
    })
    diagram = resolve_diagram(sym)
    assert len(diagram.propagators) == 1
    # W mass should be resolved
    assert abs(diagram.propagators[0].mass - 80.4) < 0.5
    assert diagram.propagators[0].spin == 1.0
    assert "g_W" in diagram.couplings

    print("  PASS")
    return True


def test_resolve_mass_overrides():
    """User overrides take priority over database values."""
    print("=" * 60)
    print("Testing resolve mass overrides")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    diagram = resolve_diagram(sym, mass_overrides={"H": 126.0, "b": 5.0})
    assert diagram.initial[0].mass == 126.0
    assert diagram.final[0].mass == 5.0

    print("  PASS")
    return True


def test_resolve_coupling_overrides():
    """User coupling overrides take priority."""
    print("=" * 60)
    print("Testing resolve coupling overrides")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    diagram = resolve_diagram(sym, coupling_overrides={"y_b": 0.03})
    assert diagram.couplings["y_b"] == 0.03

    print("  PASS")
    return True


def test_resolve_unknown_coupling_raises():
    """Unknown coupling without override raises ValueError."""
    print("=" * 60)
    print("Testing resolve unknown coupling raises")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_unknown_bsm"}],
    })
    try:
        resolve_diagram(sym)
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "y_unknown_bsm" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but missing coupling name in message: {e}")
        return False


def test_resolve_color_factor_inferred():
    """Color factor is inferred from particle content if not given."""
    print("=" * 60)
    print("Testing resolve color factor inferred")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
    })
    diagram = resolve_diagram(sym)
    # b + bbar = triplet + antitriplet -> color factor 3
    assert diagram.color_factor == 3.0

    print("  PASS")
    return True


def test_resolve_color_factor_explicit():
    """Explicit color_factor is preserved."""
    print("=" * 60)
    print("Testing resolve color factor explicit")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H"}],
        "final": [{"label": "b"}, {"label": "bbar"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        "color_factor": 1.0,
    })
    diagram = resolve_diagram(sym)
    assert diagram.color_factor == 1.0

    print("  PASS")
    return True


# ============================================================================
# build_diagram_from_symbolic
# ============================================================================

def test_build_generic_scalar_to_fermions():
    """Build Diagram from generic (non-SM) particles with explicit spins."""
    print("=" * 60)
    print("Testing build generic scalar to fermions")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": "y"}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert isinstance(diag, Diagram)
    assert diag.initial[0].label == "S"
    assert diag.initial[0].spin == 0.0
    assert diag.final[0].spin == 0.5
    assert diag.couplings == {}
    assert diag.color_factor == 1.0

    print("  PASS")
    return True


def test_build_sm_particles_with_spins():
    """SM labels still work when spins are provided."""
    print("=" * 60)
    print("Testing build SM particles with spins")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "H", "spin": 0}],
        "final": [{"label": "b", "spin": "1/2"}, {"label": "bbar", "spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": "y_b"}],
        "color_factor": 3.0,
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.initial[0].label == "H"
    assert diag.initial[0].spin == 0.0
    assert diag.color_factor == 3.0

    print("  PASS")
    return True


def test_build_missing_spin_raises():
    """Missing spin on any particle raises ValueError."""
    print("=" * 60)
    print("Testing build missing spin raises")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S"}],  # no spin
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": "y"}],
    })
    try:
        build_diagram_from_symbolic(sym)
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "Missing spin" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_build_missing_spin_on_final_raises():
    """Missing spin on a final-state particle raises ValueError."""
    print("=" * 60)
    print("Testing build missing spin on final raises")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [{"label": "f"}, {"label": "fbar"}],  # no spins
        "vertices": [{"type": "yukawa", "coupling": "y"}],
    })
    try:
        build_diagram_from_symbolic(sym)
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "Missing spin" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_build_missing_spin_on_propagator_raises():
    """Missing spin on a propagator raises ValueError."""
    print("=" * 60)
    print("Testing build missing spin on propagator raises")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [
            {"label": "f", "spin": "1/2"},
            {"label": "s1", "spin": 0},
            {"label": "s2", "spin": 0},
        ],
        "vertices": [
            {"type": "yukawa", "coupling": "g1"},
            {"type": "yukawa", "coupling": "g2"},
        ],
        "propagators": [{"label": "X"}],  # no spin
    })
    try:
        build_diagram_from_symbolic(sym)
        print("  [✗] FAIL: Expected ValueError")
        return False
    except ValueError as e:
        if "Missing spin" in str(e) and "propagator" in str(e):
            print("  PASS")
            return True
        print(f"  [✗] FAIL: ValueError raised but wrong message: {e}")
        return False


def test_build_massive_propagator():
    """Massive propagator gets mass=1.0 sentinel for codegen checks."""
    print("=" * 60)
    print("Testing build massive propagator")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [
            {"label": "f", "spin": "1/2"},
            {"label": "s1", "spin": 0},
            {"label": "s2", "spin": 0},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g1"},
            {"type": "gauge-vector", "coupling": "g2"},
        ],
        "propagators": [{"label": "X", "spin": 1, "massive": True}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.propagators[0].mass == 1.0

    print("  PASS")
    return True


def test_build_massless_propagator():
    """Massless propagator gets mass=0 for codegen checks."""
    print("=" * 60)
    print("Testing build massless propagator")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [
            {"label": "f", "spin": "1/2"},
            {"label": "s1", "spin": 0},
            {"label": "s2", "spin": 0},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g1"},
            {"type": "gauge-vector", "coupling": "g2"},
        ],
        "propagators": [{"label": "gamma", "spin": 1, "massive": False}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.propagators[0].mass == 0

    print("  PASS")
    return True


def test_build_massive_inferred_from_mass_value():
    """When massive not set, infer from mass: mass>0 means massive."""
    print("=" * 60)
    print("Testing build massive inferred from mass value")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [
            {"label": "f", "spin": "1/2"},
            {"label": "s1", "spin": 0},
            {"label": "s2", "spin": 0},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g1"},
            {"type": "gauge-vector", "coupling": "g2"},
        ],
        "propagators": [{"label": "X", "spin": 1, "mass": 80.4}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.propagators[0].mass == 1.0  # sentinel for massive

    print("  PASS")
    return True


def test_build_massless_inferred_from_mass_zero():
    """When massive not set, mass=0 means massless."""
    print("=" * 60)
    print("Testing build massless inferred from mass zero")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [
            {"label": "f", "spin": "1/2"},
            {"label": "s1", "spin": 0},
            {"label": "s2", "spin": 0},
        ],
        "vertices": [
            {"type": "gauge-vector", "coupling": "g1"},
            {"type": "gauge-vector", "coupling": "g2"},
        ],
        "propagators": [{"label": "gamma", "spin": 1, "mass": 0}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.propagators[0].mass == 0

    print("  PASS")
    return True


def test_build_initial_state_implicitly_massive():
    """Initial-state particle in a decay is always massive."""
    print("=" * 60)
    print("Testing build initial state implicitly massive")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": "y"}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.initial[0].mass == 1.0  # massive sentinel

    print("  PASS")
    return True


def test_build_final_state_defaults_to_massless():
    """Final-state particles default to massless when no mass info given."""
    print("=" * 60)
    print("Testing build final state defaults to massless")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": "y"}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.final[0].mass == 0
    assert diag.final[1].mass == 0

    print("  PASS")
    return True


def test_build_final_state_massive_explicit():
    """Final-state particle with massive=True gets mass=1.0 sentinel."""
    print("=" * 60)
    print("Testing build final state massive explicit")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [
            {"label": "V1", "spin": 1, "massive": True},
            {"label": "V2", "spin": 1, "massive": True},
        ],
        "vertices": [{"type": "scalar-3pt", "coupling": "g"}],
    })
    diag = build_diagram_from_symbolic(sym)
    assert diag.final[0].mass == 1.0
    assert diag.final[1].mass == 1.0

    print("  PASS")
    return True


def test_build_string_mass_means_massive():
    """String mass like 'mV' signals massive, and is CARRIED as the symbol.

    This asserted the numeric sentinel (mass == 1.0). The intent in its own
    title -- a named mass means massive -- still holds: `_is_massive_particle`
    returns True and the generator emits the Proca polarisation sum. What
    changed is the ENCODING: the name is preserved now, because it names the
    symbol the formula is written in, and collapsing it to 1.0 here made the
    caller's `values` bind nothing (see `_mass_for_codegen`). The assertions
    move from the sentinel to what actually matters -- massive, and the name
    survives.
    """
    print("=" * 60)
    print("Testing build string mass means massive")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
    })
    assert _is_massive_particle(sym.initial[0], is_initial=True)
    diag = build_diagram_from_symbolic(sym)
    assert diag.initial[0].mass == "mV"

    # Massive, as the title says: the Proca sum, not the massless one (which
    # carries a trailing reference-vector argument of 0).
    from tools.eda.feyncalc_codegen import SymbolicFeynCalcCodeGenerator
    code = SymbolicFeynCalcCodeGenerator().generate(diag).code
    assert "DoPolarizationSums[ampSq, p];" in code, code

    print("  PASS")
    return True


def test_build_string_mass_on_final_state():
    """String mass on a final-state particle signals massive."""
    print("=" * 60)
    print("Testing build string mass on final state")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [
            {"label": "V1", "spin": 1, "mass": "mW"},
            {"label": "V2", "spin": 1, "mass": "mW"},
        ],
        "vertices": [{"type": "scalar-3pt", "coupling": "g"}],
    })
    diag = build_diagram_from_symbolic(sym)
    # Carried as the declared symbol (was the 1.0 sentinel). Both daughters
    # share ONE name, which is the point: a degenerate pair must not be handed
    # two different mass symbols.
    assert diag.final[0].mass == "mW"
    assert diag.final[1].mass == "mW"
    assert _is_massive_particle(sym.final[0])
    assert _is_massive_particle(sym.final[1])

    print("  PASS")
    return True


# ============================================================================
# SymbolicParticle massive field
# ============================================================================

def test_particle_parse_massive_true():
    """massive=True is parsed from particle dict."""
    print("=" * 60)
    print("Testing particle parse massive true")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "V", "spin": 1, "massive": True}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "fbar", "spin": "1/2"}],
        "vertices": [{"type": "gauge-vector", "coupling": "g"}],
    })
    assert sym.initial[0].massive is True

    print("  PASS")
    return True


def test_particle_parse_massive_false():
    """massive=False is parsed from particle dict."""
    print("=" * 60)
    print("Testing particle parse massive false")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "S", "spin": 0}],
        "final": [{"label": "gamma", "spin": 1, "massive": False}, {"label": "gamma2", "spin": 1, "massive": False}],
        "vertices": [{"type": "scalar-3pt", "coupling": "g"}],
    })
    assert sym.final[0].massive is False

    print("  PASS")
    return True


def test_particle_massive_in_to_dict():
    """massive field appears in SymbolicParticle.to_dict() when set."""
    print("=" * 60)
    print("Testing particle massive in to_dict")
    print("=" * 60)

    p = SymbolicParticle(label="V", spin=1.0, massive=True)
    d = p.to_dict()
    assert d["massive"] is True

    print("  PASS")
    return True


def test_particle_massive_absent_in_to_dict_when_none():
    """massive field is omitted from SymbolicParticle.to_dict() when None."""
    print("=" * 60)
    print("Testing particle massive absent in to_dict when None")
    print("=" * 60)

    p = SymbolicParticle(label="V", spin=1.0)
    d = p.to_dict()
    assert "massive" not in d

    print("  PASS")
    return True


# ============================================================================
# SymbolicPropagator massive field
# ============================================================================

def test_propagator_parse_massive_true():
    """massive=True is parsed from propagator dict."""
    print("=" * 60)
    print("Testing propagator parse massive true")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "s", "spin": 0}],
        "vertices": [{"type": "yukawa", "coupling": "g"}],
        "propagators": [{"label": "X", "spin": 1, "massive": True}],
    })
    assert sym.propagators[0].massive is True

    print("  PASS")
    return True


def test_propagator_parse_massive_false():
    """massive=False is parsed from propagator dict."""
    print("=" * 60)
    print("Testing propagator parse massive false")
    print("=" * 60)

    sym = parse_symbolic_diagram({
        "initial": [{"label": "F", "spin": "1/2"}],
        "final": [{"label": "f", "spin": "1/2"}, {"label": "s", "spin": 0}],
        "vertices": [{"type": "yukawa", "coupling": "g"}],
        "propagators": [{"label": "gamma", "spin": 1, "massive": False}],
    })
    assert sym.propagators[0].massive is False

    print("  PASS")
    return True


def test_propagator_massive_in_to_dict():
    """massive field appears in to_dict() when set."""
    print("=" * 60)
    print("Testing propagator massive in to_dict")
    print("=" * 60)

    prop = SymbolicPropagator(label="X", spin=1.0, massive=True)
    d = prop.to_dict()
    assert d["massive"] is True

    print("  PASS")
    return True


def test_propagator_massive_absent_in_to_dict_when_none():
    """massive field is omitted from to_dict() when None."""
    print("=" * 60)
    print("Testing propagator massive absent in to_dict when None")
    print("=" * 60)

    prop = SymbolicPropagator(label="X", spin=1.0)
    d = prop.to_dict()
    assert "massive" not in d

    print("  PASS")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Symbolic Diagram Tests")
    print("=" * 60 + "\n")

    tests = [
        # Parsing
        ("Parse minimal H -> bb", test_parse_minimal_h_to_bb),
        ("Parse with topology", test_parse_with_topology),
        ("Parse with explicit spin", test_parse_with_explicit_spin),
        ("Parse with explicit masses", test_parse_with_explicit_masses),
        ("Missing initial raises", test_missing_initial_raises),
        ("Missing final raises", test_missing_final_raises),
        ("Missing vertices raises", test_missing_vertices_raises),
        ("Numerical coupling raises", test_numerical_coupling_raises),
        ("Chiral dict coupling", test_chiral_dict_coupling),
        ("Chiral dict numeric raises", test_chiral_dict_numeric_value_raises),
        ("List coupling raises", test_list_coupling_raises),
        ("Color factor", test_color_factor),
        # Validation
        ("Valid diagram", test_valid_diagram),
        ("Empty label warning", test_empty_label_warning),
        # to_dict
        ("Round trip", test_round_trip),
        # resolve_diagram
        ("Resolve H -> bb", test_resolve_h_to_bb),
        ("Resolve muon decay propagator", test_resolve_muon_decay_propagator),
        ("Resolve mass overrides", test_resolve_mass_overrides),
        ("Resolve coupling overrides", test_resolve_coupling_overrides),
        ("Resolve unknown coupling raises", test_resolve_unknown_coupling_raises),
        ("Resolve color factor inferred", test_resolve_color_factor_inferred),
        ("Resolve color factor explicit", test_resolve_color_factor_explicit),
        # build_diagram_from_symbolic
        ("Build generic scalar to fermions", test_build_generic_scalar_to_fermions),
        ("Build SM particles with spins", test_build_sm_particles_with_spins),
        ("Build missing spin raises", test_build_missing_spin_raises),
        ("Build missing spin on final raises", test_build_missing_spin_on_final_raises),
        ("Build missing spin on propagator raises", test_build_missing_spin_on_propagator_raises),
        ("Build massive propagator", test_build_massive_propagator),
        ("Build massless propagator", test_build_massless_propagator),
        ("Build massive inferred from mass value", test_build_massive_inferred_from_mass_value),
        ("Build massless inferred from mass zero", test_build_massless_inferred_from_mass_zero),
        ("Build initial state implicitly massive", test_build_initial_state_implicitly_massive),
        ("Build final state defaults to massless", test_build_final_state_defaults_to_massless),
        ("Build final state massive explicit", test_build_final_state_massive_explicit),
        ("Build string mass means massive", test_build_string_mass_means_massive),
        ("Build string mass on final state", test_build_string_mass_on_final_state),
        # SymbolicParticle massive field
        ("Particle parse massive true", test_particle_parse_massive_true),
        ("Particle parse massive false", test_particle_parse_massive_false),
        ("Particle massive in to_dict", test_particle_massive_in_to_dict),
        ("Particle massive absent when None", test_particle_massive_absent_in_to_dict_when_none),
        # SymbolicPropagator massive field
        ("Propagator parse massive true", test_propagator_parse_massive_true),
        ("Propagator parse massive false", test_propagator_parse_massive_false),
        ("Propagator massive in to_dict", test_propagator_massive_in_to_dict),
        ("Propagator massive absent when None", test_propagator_massive_absent_in_to_dict_when_none),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "[✓] PASS" if result else "[✗] FAIL"))
        except AssertionError as e:
            print(f"  [✗] FAIL: {name}: {e}")
            results.append((name, "[✗] FAIL"))
        except Exception as e:
            print(f"  [✗] ERROR in {name}: {e}")
            results.append((name, "[✗] ERROR"))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {status}: {name}")

    passed = sum(1 for _, s in results if s == "[✓] PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
