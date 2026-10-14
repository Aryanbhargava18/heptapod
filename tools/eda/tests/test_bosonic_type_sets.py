"""
# test_bosonic_type_sets.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Every vertex type resolves against a CLOSED set for the spin configuration it
was given, so an unrecognised name is refused with the valid keys listed.

WHY. The fermionic bilinears have been closed since 2.9.1 -- an unknown VFF or
SFF type raises and lists its spellings. The four bosonic configurations never
got the same treatment: they dispatch on the spin multiset and accept any type
string. So a Dirac-fermion decay declared with `spin: 1` on both fermions and
`type: "left_vector"` selected the Yang-Mills triple-gauge branch and returned
a confident width for a different process -- 2 of 5 graded trials of hepbench
decay_F_to_FpV, both scoring 0.04 (2026-09-30, heptapod 2.16.3).

2.16.3 caught this only when the type was spelled from S/V/F letters, which is
how the agents wrote it on one run (`FFV`, `VFF`) and not the next
(`left_vector`). Closing the sets removes the dependence on notation: any
unrecognised name is refused, whatever it is.

THE ERROR HAS TO NAME BOTH BRANCHES. Listing the keys valid for the declared
spins, alone, steers the caller toward renaming -- which yields the other
vertex. So the message states that the available types are determined by the
declared spins, and gives the fermionic configuration's own keys for the case
where the spins are what is wrong.

COMPATIBILITY. The accepted set per configuration is deliberately generous:
the configuration's letter-string, a curated list, and the names already in
the fermionic tables -- `scalar`, `vector`, `gauge` and `left-handed` are in
circulation for bosonic vertices in this codebase via the `-3pt` spellings,
which normalize onto them. The structure is still decided by the spins; this
only governs which NAMES are refused outright.
"""

import pytest

from tools.eda.feyncalc_codegen import (
    SymbolicFeynCalcCodeGenerator, UnknownVertexType, BOSONIC_VERTEX_TYPES,
)
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)

CONFIGS = {
    "SSS": ([("phi", 0, "mS")], [("c1", 0, "m1"), ("c2", 0, "m2")]),
    "SSV": ([("V", 1, "mV")], [("S1", 0, "m1"), ("S2", 0, "m2")]),
    "SVV": ([("phi", 0, "mS")], [("V1", 1, "m1"), ("V2", 1, "m2")]),
    "VVV": ([("Zp", 1, "mZp")], [("Wp", 1, "mW"), ("Wm", 1, "mW")]),
}


def _spec(kind, vtype, coupling="g"):
    ini, fin = CONFIGS[kind]
    return {"topology": "decay_1to2",
            "initial": [{"label": l, "spin": s, "mass": m} for l, s, m in ini],
            "final": [{"label": l, "spin": s, "mass": m} for l, s, m in fin],
            "vertices": [{"type": vtype, "coupling": coupling}]}


def _gen(spec):
    d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
    return SymbolicFeynCalcCodeGenerator().generate(d)


# ---------------------------------------------------------------------------
# Every bosonic configuration declares a closed set.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(CONFIGS))
def test_each_configuration_has_a_declared_type_set(kind):
    assert kind in BOSONIC_VERTEX_TYPES
    spec = BOSONIC_VERTEX_TYPES[kind]
    assert spec["spellings"], f"{kind} declares no accepted spellings"


@pytest.mark.parametrize("kind", sorted(CONFIGS))
def test_the_letter_string_is_accepted_for_its_own_configuration(kind):
    assert _gen(_spec(kind, kind.lower())).code


# ---------------------------------------------------------------------------
# The regression: an unrecognised name is refused.
# ---------------------------------------------------------------------------

def test_the_left_vector_case_is_refused():
    """The exact name the agent sent, over three spin-1 legs. 2.16.3's letter
    rule let this through."""
    with pytest.raises(UnknownVertexType):
        _gen(_spec("VVV", "left_vector"))


@pytest.mark.parametrize("kind", sorted(CONFIGS))
def test_an_invented_name_is_refused(kind):
    with pytest.raises(UnknownVertexType):
        _gen(_spec(kind, "totally-made-up-xyzzy"))


def test_the_error_lists_the_keys_for_the_declared_spins():
    with pytest.raises(UnknownVertexType) as e:
        _gen(_spec("VVV", "left_vector"))
    msg = str(e.value)
    assert "vvv" in msg or "triple-gauge" in msg, msg


def test_the_error_states_the_set_follows_from_the_spins():
    """Without this the message reads as an invitation to rename, which
    returns the other vertex."""
    with pytest.raises(UnknownVertexType) as e:
        _gen(_spec("VVV", "left_vector"))
    msg = str(e.value).lower()
    assert "determined by" in msg and "spin" in msg, msg


def test_the_error_offers_the_fermionic_configuration_too():
    """The likely intent behind `left_vector` over three spin-1 legs is a
    fermion-vector vertex, so its keys and its spins are given."""
    with pytest.raises(UnknownVertexType) as e:
        _gen(_spec("VVV", "left_vector"))
    msg = str(e.value)
    assert "0.5" in msg, msg
    assert "axial-vector" in msg or "chiral" in msg, msg


# ---------------------------------------------------------------------------
# Compatibility: names in circulation must keep working.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,vtype", [
    ("SSS", "scalar"), ("SSS", "scalar-3pt"), ("SSS", "sss"),
    ("SSV", "ssv"), ("SSV", "gauge-3pt"),
    ("SVV", "svv"), ("SVV", "field-strength"), ("SVV", "dual-field-strength"),
    ("VVV", "vvv"), ("VVV", "triple-gauge"), ("VVV", "vector-3pt"),
])
def test_names_already_in_circulation_are_accepted(kind, vtype):
    assert _gen(_spec(kind, vtype)).code, f"{kind}/{vtype}"


def test_the_svv_dim5_variants_still_select_their_branches():
    """field-strength and its dual emit different structures; closing the set
    must not collapse them onto the plain contact vertex."""
    plain = _gen(_spec("SVV", "svv")).code
    fs = _gen(_spec("SVV", "field-strength")).code
    dual = _gen(_spec("SVV", "dual-field-strength")).code
    assert plain != fs and fs != dual


# ---------------------------------------------------------------------------
# The fermionic paths are untouched.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vtype", ["vector", "gauge-vector", "chiral",
                                   "vector-axial", "tensor"])
def test_vff_types_are_unaffected(vtype):
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
            "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                      {"label": "fbar", "spin": 0.5, "mass": "mf"}],
            "vertices": [{"type": vtype, "coupling": "g"}]}
    assert _gen(spec).code


def test_an_unknown_vff_type_still_raises_with_its_own_list():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
            "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                      {"label": "fbar", "spin": 0.5, "mass": "mf"}],
            "vertices": [{"type": "nonsense-xyzzy", "coupling": "g"}]}
    with pytest.raises(UnknownVertexType) as e:
        _gen(spec)
    assert "vector-axial" in str(e.value)
