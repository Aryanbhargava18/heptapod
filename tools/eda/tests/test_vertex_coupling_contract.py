"""
# test_vertex_coupling_contract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

A vertex type declares which couplings it takes, and a supplied coupling is
VALIDATED against that declaration rather than sniffed for its shape.

THE CLASS OF BUG THIS CLOSES. The fermionic dispatches inferred intent from the
input's shape instead of checking it against a contract, and did so in three
places:

  unknown vertex type   -> silently emitted the plain coupling      (fixed 2.9.1)
  unknown spin config   -> silently emitted `amp = I g`             (fixed 2.9.1)
  ANY dict coupling     -> silently emitted a CHIRAL vertex         (this change)

The third shadowed six legitimate types, because the dict check sat mid-chain,
ahead of them. Asking for a plain vector vertex while writing the coupling as
{"g": "g"} instead of "g" therefore returned a chiral rule: the declared physics
lost to a formatting detail. Measured cost, on a graded trial -- the tool
returned a formula in gL/gR, symbols the caller had no values for, so no number
came back, the caller abandoned the tool and hand-derived a width 4x too small.
Two runs of one identical config scored 1.00 and 0.04 purely on whether that
coupling was written as a string or a dict.

WHY THE SHAPE-BLIND FIX IS NOT ENOUGH, and hence this contract. Routing the six
branches through the existing `_extract_single_coupling` repairs
vector + {"g": "g"} and leaves its siblings just as silent:

    _extract_single_coupling({"gL": "gL", "gR": "gR"}) -> "gL"   gR discarded
    _extract_single_coupling({})                       -> "g"    name invented
    _extract_chiral_couplings({"gV": "gV", "gA": "gA"})-> ("gL", "gR")

that last being the original failure reached by another door. Every one of them
answers a different question without saying so, which is the only bug class an
agent cannot detect. So the rule is: the type says which couplings exist, the
caller must supply those, and any mismatch RAISES naming what was expected.
"""

import pytest

from tools.eda.feyncalc_codegen import (
    FeynCalcCodeGenerator, UnknownVertexType,
    VFF_VERTEX_TYPES, SFF_VERTEX_TYPES,
)


@pytest.fixture
def gen():
    return FeynCalcCodeGenerator()


def _vff(gen, vtype, coupling):
    return gen._vff_gamma_structure(vtype, coupling, "mu", vec_momentum="p")


def _sff(gen, vtype, coupling):
    return gen._sff_coupling_structure(vtype, coupling)


# ---------------------------------------------------------------------------
# The tables must declare the coupling contract, not just the spellings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", [VFF_VERTEX_TYPES, SFF_VERTEX_TYPES])
def test_every_type_declares_its_couplings(table):
    """Adding a vertex type must mean adding its contract, so validation comes
    free rather than depending on a branch remembering the right extractor."""
    for base, spec in table.items():
        assert "spellings" in spec, f"{base} declares no spellings"
        assert "couplings" in spec, f"{base} declares no coupling contract"
        assert isinstance(spec["couplings"], tuple) and spec["couplings"], \
            f"{base} coupling contract must be a non-empty tuple"


@pytest.mark.parametrize("base,names", [
    ("vector", ("g",)),
    ("axialvector", ("g",)),
    ("lefthanded", ("g",)),
    ("righthanded", ("g",)),
    ("tensor", ("g",)),
    ("vectoraxial", ("gV", "gA")),
    ("chiral", ("gL", "gR")),
    ("tensorchiral", ("gL", "gR")),
])
def test_vff_contracts_are_what_the_physics_requires(base, names):
    assert VFF_VERTEX_TYPES[base]["couplings"] == names


@pytest.mark.parametrize("base,names", [
    ("yukawa", ("g",)),
    ("pseudoscalar", ("g",)),
    ("chiral", ("gL", "gR")),
    ("scalarva", ("gS", "gP")),
])
def test_sff_contracts_are_what_the_physics_requires(base, names):
    assert SFF_VERTEX_TYPES[base]["couplings"] == names


# ---------------------------------------------------------------------------
# A declared type wins over the coupling's shape. This is the regression.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vtype", ["vector", "gauge-vector", "axial-vector",
                                   "left-handed", "right-handed", "tensor"])
def test_single_coupling_types_accept_a_one_entry_dict(gen, vtype):
    """The six types the mid-chain dict check used to shadow. A 1-entry dict is
    unambiguous, so it must give the SAME rule the bare string gives."""
    assert _vff(gen, vtype, {"g": "g"}) == _vff(gen, vtype, "g")


@pytest.mark.parametrize("vtype", ["yukawa", "scalar", "pseudoscalar"])
def test_sff_single_coupling_types_accept_a_one_entry_dict(gen, vtype):
    assert _sff(gen, vtype, {"g": "g"}) == _sff(gen, vtype, "g")


def test_plain_vector_with_a_dict_is_not_chiral(gen):
    """The exact call that failed the graded trial: it must emit the vector
    rule, with no chiral projectors anywhere in it."""
    out = _vff(gen, "vector", {"g": "g"})
    assert "GA[7]" not in out and "GA[6]" not in out
    assert "gL" not in out and "gR" not in out


# ---------------------------------------------------------------------------
# Mismatches raise, naming what was expected. Previously all silent.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vtype", ["vector", "axial-vector", "left-handed"])
def test_multi_entry_dict_on_a_single_coupling_type_raises(gen, vtype):
    """Ambiguous: which of the two is the coupling? It used to take the first
    and discard the rest."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, vtype, {"gL": "gL", "gR": "gR"})
    assert "g" in str(e.value)


def test_wrong_keys_on_a_two_coupling_type_raises(gen):
    """chiral + {gV, gA} used to return a rule in gL/gR -- symbols the caller
    never supplied, so the number silently went missing. This IS the original
    failure, reached by another door."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "chiral", {"gV": "gV", "gA": "gA"})
    msg = str(e.value)
    assert "gL" in msg and "gR" in msg


def test_wrong_keys_on_vector_axial_raises(gen):
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "vector-axial", {"gL": "gL", "gR": "gR"})
    msg = str(e.value)
    assert "gV" in msg and "gA" in msg


def test_empty_dict_raises_rather_than_inventing_a_name(gen):
    """It used to return the literal "g"."""
    with pytest.raises(UnknownVertexType):
        _vff(gen, "vector", {})


def test_sff_scalar_va_wrong_keys_raises(gen):
    with pytest.raises(UnknownVertexType) as e:
        _sff(gen, "scalar-va", {"gL": "gL", "gR": "gR"})
    msg = str(e.value)
    assert "gS" in msg and "gP" in msg


# ---------------------------------------------------------------------------
# Conveniences that must SURVIVE: they are documented and unambiguous.
# ---------------------------------------------------------------------------

def test_a_string_still_derives_suffixes_for_two_coupling_types(gen):
    """`chiral` with "y" means yL, yR -- a documented shorthand, and
    unambiguous because the type fixes the arity."""
    out = _vff(gen, "chiral", "y")
    assert "yL" in out and "yR" in out


def test_correct_dict_keys_are_used_verbatim(gen):
    out = _vff(gen, "vector-axial", {"gV": "gV_e", "gA": "gA_e"})
    assert "gV_e" in out and "gA_e" in out


def test_unknown_type_still_raises_with_the_spelling_list(gen):
    """The 2.9.1 behaviour must not regress while reworking the same chain."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "totally_made_up_xyzzy", "g")
    assert "vector-axial" in str(e.value)


# ---------------------------------------------------------------------------
# The bosonic path stays outside the scheme: it dispatches on the SPIN
# configuration, so its type string is genuinely free and must remain so.
# ---------------------------------------------------------------------------

def test_bosonic_type_strings_remain_unconstrained():
    """Free-form means a name that makes NO structural claim.

    This used to include "svv" over scalar-scalar-vector legs, and
    "anything-at-all". Neither survives: 2.16.3 made a letter-spelled type
    declare its own spin content, and 2.17.0 closed the bosonic name sets, so
    an unrecognised name is refused with the keys valid for the declared
    spins. What the test now protects is that the ACCEPTED names for a
    configuration all work, including the `-3pt` spellings in circulation.
    """
    from tools.nda.symbolic_diagram import (
        parse_symbolic_diagram, build_diagram_from_symbolic,
    )
    from tools.eda.feyncalc_codegen import SymbolicFeynCalcCodeGenerator
    # An empty type is rejected upstream by SymbolicVertex, which requires
    # the field; "free" here means any NAME, not the absence of one.
    for vt in ("ssv", "scalar-3pt", "derivative"):
        spec = {"topology": "decay_1to2",
                "initial": [{"label": "S", "spin": 0}],
                "final": [{"label": "Sp", "spin": 0}, {"label": "V", "spin": 1}],
                "vertices": [{"type": vt, "coupling": "g"}]}
        d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
        res = SymbolicFeynCalcCodeGenerator().generate(d)
        assert res.code, f"bosonic dispatch rejected type {vt!r}"


# ---------------------------------------------------------------------------
# The error text is the whole recovery path. An agent that trips this has no
# other channel: it cannot read this file, and the tool description it was
# given may not be in front of it any more. So the message has to carry the
# contract, a corrected call, and -- critically -- the fact that the fix is a
# physics decision and not a reshaping exercise. An agent told only "wrong
# keys" will satisfy the checker by renaming its keys and silently get the
# vertex it did not want, which is the failure this change exists to end.
# ---------------------------------------------------------------------------

def test_wrong_keys_error_names_the_type_whose_contract_matches(gen):
    """{gV, gA} on `chiral` IS the vector-axial contract. Say so: the likely
    intent was the other vertex, not a rename."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "chiral", {"gV": "gV", "gA": "gA"})
    msg = str(e.value)
    assert "vector-axial" in msg, msg


def test_wrong_keys_error_warns_against_merely_renaming(gen):
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "chiral", {"gV": "gV", "gA": "gA"})
    msg = str(e.value).lower()
    assert "rename" in msg or "different vertex" in msg, msg


def test_errors_carry_a_corrected_example(gen):
    """A copy-pasteable corrected call, so the fix needs no guessing."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "vector", {"a": "1", "b": "2"})
    msg = str(e.value)
    assert '"coupling"' in msg or "'coupling'" in msg, msg
    assert "vector" in msg


def test_ambiguity_error_states_which_names_were_given(gen):
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "vector", {"alpha": "1", "beta": "2"})
    msg = str(e.value)
    assert "alpha" in msg and "beta" in msg, msg


def test_empty_coupling_error_does_not_suggest_a_matching_type(gen):
    """No keys means no contract to match, so the message must not invent a
    suggestion -- just state what is required."""
    with pytest.raises(UnknownVertexType) as e:
        _vff(gen, "vector", {})
    msg = str(e.value)
    assert "takes exactly one coupling" in msg, msg
    # and no "those keys are the contract for ..." clause, since there are none
    assert "DIFFERENT VERTEX" not in msg, msg
