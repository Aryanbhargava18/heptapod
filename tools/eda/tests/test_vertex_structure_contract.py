"""
# test_vertex_structure_contract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The vertex TYPE and the declared SPINS fix the same vertex, so they must agree.

THE BUG. The generator dispatches on the spin multiset and, on the bosonic
branches, never looks at the type at all. So this call -- from a graded trial
of hepbench decay_F_to_FpV, a Dirac fermion decaying to a lighter fermion plus
a massive vector -- succeeded:

    "initial": [{"label": "Psi", "spin": 1, "mass": "mF"}],
    "final":   [{"label": "psi", "spin": 1, "mass": "mf"},
                {"label": "V",   "spin": 1, "mass": "mV"}],
    "vertices":[{"particles": ["Psi","psi","V"], "coupling": "g",
                 "type": "FFV"}]

Spins (1,1,1) selected the Yang-Mills triple-gauge branch; `type: "FFV"` was
discarded unread. The tool returned `status: ok`, no warnings, and
width 423.67 GeV against a ground truth of 109.52 -- a different process,
reported as a result. Two trials of ten made this mistake and both scored
0.04; the eight that declared spins correctly scored 0.37 to 1.00.

WHY THE TYPE IS ENOUGH TO CATCH IT. `"FFV"` is not noise, it is the agent
stating the structure it wanted: F fermion, F fermion, V vector. The diagram
therefore contains two declarations of one fact, and they contradict. The tool
cannot know WHICH is wrong -- an agent that genuinely wanted triple-gauge and
typed a bad name is equally consistent with the input -- so it must not guess.
It states the contradiction and both readings, and the caller resolves it.

WHERE THE IMPLIED STRUCTURE COMES FROM. Two derived sources, neither keyed to
an individual vertex:

  letters   a type spelled only from S/V/F is self-describing: FFV -> spins
            (0.5, 0.5, 1). This also covers the convention the codebase
            already uses for bosonic vertices -- sss, ssv, svv, vvv ARE
            letter-strings -- so they are validated for free.
  tables    a named type carries its table's signature: `vector` is a VFF
            type, spins (0.5, 0.5, 1); `yukawa` is SFF, (0, 0.5, 0.5).

Checked: no existing spelling in either table parses as a letter-string, so
the letter rule cannot capture a name that means something else.

WHAT THIS DOES NOT DO. It does not make a wrong answer right. An agent told
its spins and type disagree may fix the type rather than the spins -- this
agent's response to an earlier error was to add the field the error named
without revisiting the physics. And spins and type that are wrong TOGETHER are
undetectable by anything: inside a diagram spec, spin and type are the only
witnesses to a particle being a fermion. The residue is capability, and the
benchmark measures it correctly.
"""

import pytest

from tools.eda.feyncalc_codegen import (
    SymbolicFeynCalcCodeGenerator, VertexStructureMismatch,
    implied_vertex_spins, VFF_VERTEX_TYPES, SFF_VERTEX_TYPES,
)
from tools.eda.fc_symbols import _structure_from_letters
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


def _gen(spec):
    d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
    return SymbolicFeynCalcCodeGenerator().generate(d)


def _ffv_spec(spins, vtype):
    """decay_F_to_FpV's shape, with the leg spins and type under test."""
    s_parent, s_f, s_v = spins
    return {"topology": "decay_1to2",
            "initial": [{"label": "Psi", "spin": s_parent, "mass": "mF"}],
            "final": [{"label": "psi", "spin": s_f, "mass": "mf"},
                      {"label": "V", "spin": s_v, "mass": "mV"}],
            "vertices": [{"type": vtype, "coupling": "g"}]}


# ---------------------------------------------------------------------------
# The letter rule. A type spelled from S/V/F declares its own spin content.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,spins", [
    ("ffv", (0.5, 0.5, 1.0)),
    ("vff", (0.5, 0.5, 1.0)),
    ("fvf", (0.5, 0.5, 1.0)),
    ("sff", (0.0, 0.5, 0.5)),
    ("ffs", (0.0, 0.5, 0.5)),
    ("sss", (0.0, 0.0, 0.0)),
    ("ssv", (0.0, 0.0, 1.0)),
    ("svv", (0.0, 1.0, 1.0)),
    ("vvv", (1.0, 1.0, 1.0)),
    ("ffff", (0.5, 0.5, 0.5, 0.5)),     # four-point contact
])
def test_letter_strings_declare_their_spins(name, spins):
    assert _structure_from_letters(name) == spins


@pytest.mark.parametrize("name", [
    "vector", "gaugevector", "axialvector", "chiral", "lefthanded",
    "righthanded", "tensor", "dipole", "tensorchiral", "yukawa", "scalar",
    "pseudoscalar", "scalarva", "fieldstrength", "anythingatall", "",
])
def test_real_type_names_are_not_read_as_letters(name):
    """The letter rule must not capture a name that means something else."""
    assert _structure_from_letters(name) is None


def test_no_declared_spelling_collides_with_the_letter_rule():
    """Guards the rule as the tables grow: adding a spelling that happens to
    be an S/V/F string would silently give it a spin signature."""
    from tools.eda.feyncalc_codegen import FeynCalcCodeGenerator
    norm = FeynCalcCodeGenerator()._normalize_vtype
    for table in (VFF_VERTEX_TYPES, SFF_VERTEX_TYPES):
        for base, spec in table.items():
            for sp in spec["spellings"]:
                assert _structure_from_letters(norm(sp)) is None, sp


def test_two_letters_is_not_a_structure():
    assert _structure_from_letters("sf") is None
    assert _structure_from_letters("v") is None


# ---------------------------------------------------------------------------
# A NAMED type declares NO structure -- deliberately.
#
# Deriving a signature from the table a name lives in looked attractive and is
# unsound: a table's spellings are not exclusive to its structure. `scalar` is
# registered as an alias of the SFF `yukawa` vertex AND is the ordinary name
# for an SSS vertex, L = -(mu/2) phi chi^2. Taking (0, 0.5, 0.5) from the name
# rejected 29 valid calls, including the decay_S_to_SS ground-truth check.
# `vector` has the same double life.
#
# So only a self-describing letter-string declares a structure. The cost is
# that `vector` over three spin-1 legs is not caught: a name that is merely
# unconventional for its spins stays legal, as it always was. The benefit is
# no false positives, which is what makes the check shippable.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vtype", [
    "vector", "gauge-vector", "vector-axial", "tensor", "yukawa",
    "pseudoscalar", "scalar-va", "scalar", "anything-at-all",
])
def test_a_named_type_declares_no_structure(vtype):
    assert implied_vertex_spins(vtype) is None


def test_scalar_stays_usable_for_both_structures():
    """The false positive that killed the table rule: `scalar` must work as an
    SSS vertex, while remaining an SFF alias."""
    sss = {"topology": "decay_1to2",
           "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
           "final": [{"label": "chi", "spin": 0, "mass": "mchi"},
                     {"label": "chi", "spin": 0, "mass": "mchi"}],
           "vertices": [{"type": "scalar", "coupling": "mu"}]}
    assert _gen(sss).code
    sff = {"topology": "decay_1to2",
           "initial": [{"label": "S", "spin": 0, "mass": "mS"}],
           "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                     {"label": "fbar", "spin": 0.5, "mass": "mf"}],
           "vertices": [{"type": "scalar", "coupling": "y"}]}
    assert _gen(sff).code


# ---------------------------------------------------------------------------
# The regression: the exact seed1001 contradiction.
# ---------------------------------------------------------------------------

def test_ffv_with_three_vector_spins_raises():
    with pytest.raises(VertexStructureMismatch):
        _gen(_ffv_spec((1, 1, 1), "FFV"))


def test_the_error_states_both_the_type_and_the_declared_spins():
    with pytest.raises(VertexStructureMismatch) as e:
        _gen(_ffv_spec((1, 1, 1), "FFV"))
    msg = str(e.value)
    assert "FFV" in msg
    assert "0.5" in msg, msg          # what the type implies
    assert "1, 1, 1" in msg or "1.0, 1.0, 1.0" in msg, msg


def test_the_error_offers_both_readings_without_choosing():
    """The tool cannot know which declaration is wrong, so it must not assert
    one. It names the fermion fix and the triple-gauge fix."""
    with pytest.raises(VertexStructureMismatch) as e:
        _gen(_ffv_spec((1, 1, 1), "FFV"))
    msg = str(e.value).lower()
    assert "spin" in msg and "0.5" in msg
    assert "vvv" in msg or "triple" in msg


def test_a_named_fermionic_type_with_bosonic_spins_is_NOT_caught():
    """The acknowledged gap. `vector` declares no structure (see above), so
    this resolves as a VVV vertex. Pinned so the limit is explicit rather
    than discovered later: only letter-spelled types are checked."""
    assert _gen(_ffv_spec((1, 1, 1), "vector")).code


def test_a_bosonic_letter_type_with_fermionic_spins_raises():
    """The mirror image: vvv declared over legs that include fermions."""
    with pytest.raises(VertexStructureMismatch):
        _gen(_ffv_spec((0.5, 0.5, 1), "vvv"))


# ---------------------------------------------------------------------------
# Correct specs are untouched. Eight of the ten trials took these paths.
# ---------------------------------------------------------------------------

def test_the_correct_ffv_spec_still_works():
    res = _gen(_ffv_spec((0.5, 0.5, 1), "vector"))
    assert res.code
    assert "GAD" in res.code          # a Dirac structure, i.e. the VFF branch


def test_the_shorthand_is_enough_to_check_but_not_to_dispatch():
    """`FFV` with spins (0.5,0.5,1) is CONSISTENT, so the structure check
    passes -- and the call still fails, correctly, because `FFV` names the
    structure and not the Dirac structure. A VFF vertex can be vector, axial,
    chiral or tensor, and choosing one for the caller would be the same
    invention this release exists to remove. The error should say that rather
    than only listing names."""
    from tools.eda.feyncalc_codegen import UnknownVertexType
    with pytest.raises(UnknownVertexType) as e:
        _gen(_ffv_spec((0.5, 0.5, 1), "FFV"))
    msg = str(e.value)
    assert "structure" in msg.lower(), msg
    assert "vector" in msg          # the choices are offered
    # and it must NOT be the spin-mismatch error: the spins were right
    assert "must agree" not in msg, msg


@pytest.mark.parametrize("vtype", ["svv", "S-V-V", "contact"])
def test_bosonic_dispatch_accepts_the_names_in_its_closed_set(vtype):
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "V1", "spin": 1, "mass": "m1"},
                      {"label": "V2", "spin": 1, "mass": "m2"}],
            "vertices": [{"type": vtype, "coupling": "g"}]}
    assert _gen(spec).code, vtype


def test_sss_shorthand_matches_its_spins():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "c1", "spin": 0, "mass": "m1"},
                      {"label": "c2", "spin": 0, "mass": "m2"}],
            "vertices": [{"type": "sss", "coupling": "mu"}]}
    assert _gen(spec).code


def test_a_scalar_decay_with_yukawa_is_unaffected():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "S", "spin": 0, "mass": "mS"}],
            "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                      {"label": "fbar", "spin": 0.5, "mass": "mf"}],
            "vertices": [{"type": "yukawa", "coupling": "y"}]}
    assert _gen(spec).code


# ---------------------------------------------------------------------------
# `vertex_structures`: name what was dispatched.
#
# The output already CONTAINS the evidence -- seed1001's
# `amp_squared_mathematica` has no Dirac structure anywhere, no GAD, no
# spinors -- it just is not labelled. This names it.
#
# DEDUPLICATED AND SORTED, because a per-vertex or per-leg listing does not
# scale: a 2->2 coherent sum over s and t channels is ~6 vertices and ~24 legs,
# and noise in an output field is what teaches a caller to stop reading it.
# There are only six structures, so this field has a hard size bound whatever
# the vertex count. Sorted rather than diagram order, so it does not invite a
# correspondence with individual vertices that it cannot keep.
# ---------------------------------------------------------------------------

def test_a_fermionic_decay_reports_its_structure():
    res = _gen(_ffv_spec((0.5, 0.5, 1), "vector"))
    assert res.vertex_structures == ["VFF"]


def test_the_misdispatched_case_reports_VVV():
    """What seed1001 would now see beside its 423.67: the structure it got."""
    spec = _ffv_spec((1, 1, 1), "vector")      # named type: not caught by the
    res = _gen(spec)                            # letter rule, so it resolves
    assert res.vertex_structures == ["VVV"]


def test_a_scalar_pair_reports_SFF():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "S", "spin": 0, "mass": "mS"}],
            "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                      {"label": "fbar", "spin": 0.5, "mass": "mf"}],
            "vertices": [{"type": "yukawa", "coupling": "y"}]}
    assert _gen(spec).vertex_structures == ["SFF"]


def test_a_three_scalar_vertex_reports_SSS():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "c1", "spin": 0, "mass": "m1"},
                      {"label": "c2", "spin": 0, "mass": "m2"}],
            "vertices": [{"type": "scalar", "coupling": "mu"}]}
    assert _gen(spec).vertex_structures == ["SSS"]


def test_the_field_is_deduplicated_and_sorted():
    """Two VFF vertices report one entry, not two -- the bound that makes this
    scale to a coherent sum."""
    from tools.eda.feyncalc_codegen import _structures_used
    assert _structures_used([(0.5, 0.5, 1.0), (0.5, 0.5, 1.0)]) == ["VFF"]
    assert _structures_used([(1.0, 1.0, 1.0), (0.0, 0.5, 0.5)]) == ["SFF", "VVV"]


def test_an_unrecognised_spin_set_is_not_invented():
    """A structure the generator has no name for must not be guessed at."""
    from tools.eda.feyncalc_codegen import _structures_used
    assert _structures_used([(0.5, 0.5, 0.5)]) == []
