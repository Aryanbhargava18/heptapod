"""
# test_bosonic_coupling_contract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

A bosonic vertex takes exactly ONE coupling, and a dict coupling is resolved
against that contract rather than interpolated into the script.

THE BUG THIS CLOSES. 2.15.2 gave the FERMIONIC dispatches a coupling contract
(`resolve_coupling`, VFF_VERTEX_TYPES / SFF_VERTEX_TYPES). The bosonic branches
were left taking `_coupling_value`'s return value straight into an f-string --
and `_coupling_value` passes a dict THROUGH unchanged. So a dict coupling
reached Mathematica as a Python repr:

    amp = I ({'g': 'g'}) PolarizationVector[p1, mu1] ... MTD[mu1, mu2];
    ToExpression::sntx: Invalid syntax in or before "amp = I ({'g': 'g'}) ..."

The script then failed to PARSE, so no amplitude, no formula and no number came
back -- while the tool still reported `status: ok`, because it had a script and
a results file, just with nothing in them. All four bosonic spin
configurations leaked it:

    sss [0,0,0]   amp = I ({'mu': 'mu'});
    ssv [0,0,1]   amp = I ({'g': 'g'}) PolarizationVector[p, mu1] FVD[...]
    svv [0,1,1]   amp = I ({'g': 'g'}) PolarizationVector[p1, mu1] ...
    vvv [1,1,1]   amp = I ({'g': 'g'}) PolarizationVector[p, mu0] ...

MEASURED COST. `{"g": "g"}` is the spelling agents reach for -- every one of
the five bosonic tasks in the hepbench `symbolic` family wrote it that way, and
all five failed (2026-09-30 smoke, heptapod 2.16.0, claude-haiku-4.5):

    decay_S_to_SS 0.10   decay_S_to_VV 0.04   decay_V_to_SS   0.31
    decay_V_to_SVp 0.10  decay_V_to_VVp 0.04

The five FERMIONIC tasks, which route through the 2.15.2 contract, all scored
1.00. The split is exactly the one this test closes.

WHY REUSE `resolve_coupling`. A second validator would drift from the first,
and the fermionic error text is already the recovery path an agent needs. A
bosonic vertex is just a type whose contract is ("g",), so it is expressed as
one and validated by the same function -- which also means an ambiguous or
empty dict raises here for the same reasons it raises there, instead of taking
the first value or inventing a name.

WHAT MUST NOT CHANGE. The bosonic dispatch keys off the SPIN CONFIGURATION, so
the type STRING stays free-form: "svv", "ssv" and "anything-at-all" are all
accepted. This contract constrains the coupling, never the type name.
"""

import pytest

from tools.eda.feyncalc_codegen import (
    SymbolicFeynCalcCodeGenerator, UnknownVertexType,
)
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


# The four bosonic spin configurations, as (initial, final, type).
CONFIGS = {
    "sss": ([("phi", 0, "mS")], [("c1", 0, "m1"), ("c2", 0, "m2")], "scalar"),
    "ssv": ([("V", 1, "mV")], [("S1", 0, "m1"), ("S2", 0, "m2")], "ssv"),
    "svv": ([("phi", 0, "mS")], [("V1", 1, "m1"), ("V2", 1, "m2")], "svv"),
    "vvv": ([("Zp", 1, "mZp")], [("Wp", 1, "mW"), ("Wm", 1, "mW")], "vvv"),
}


def _spec(cfg, coupling, vtype=None):
    ini, fin, vt = CONFIGS[cfg]
    return {"topology": "decay_1to2",
            "initial": [{"label": l, "spin": s, "mass": m} for l, s, m in ini],
            "final": [{"label": l, "spin": s, "mass": m} for l, s, m in fin],
            "vertices": [{"type": vtype or vt, "coupling": coupling}]}


def _code(cfg, coupling, vtype=None):
    d = build_diagram_from_symbolic(parse_symbolic_diagram(
        _spec(cfg, coupling, vtype)))
    return SymbolicFeynCalcCodeGenerator().generate(d).code


def _amp_line(code):
    for line in code.splitlines():
        if line.startswith("amp ="):
            return line
    return ""


# ---------------------------------------------------------------------------
# The regression: no Python repr may reach the script.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_a_dict_coupling_does_not_leak_a_python_repr(cfg):
    code = _code(cfg, {"g": "g"})
    assert "{'" not in code and '{"' not in code, _amp_line(code)


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_a_one_entry_dict_gives_the_same_rule_as_the_bare_string(cfg):
    """A 1-entry dict is unambiguous, so it must mean what the string means."""
    assert _code(cfg, {"g": "g"}) == _code(cfg, "g")


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_the_dict_value_is_the_name_used(cfg):
    """{"g": "gEff"} means the coupling is called gEff, not g."""
    code = _code(cfg, {"g": "gEff"})
    assert "gEff" in _amp_line(code), _amp_line(code)


def test_the_scalar_coupling_name_is_honoured():
    """decay_S_to_SS wrote {"mu": "mu"}; mu must reach the amplitude."""
    assert "mu" in _amp_line(_code("sss", {"mu": "mu"}))


# ---------------------------------------------------------------------------
# Generated code must be syntactically valid Wolfram at the amplitude line --
# the specific thing that broke. A bare balance check is enough to catch a
# repr leak without re-implementing a parser.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_the_amplitude_line_has_no_stray_quotes_or_colons(cfg):
    amp = _amp_line(_code(cfg, {"g": "g"}))
    assert "'" not in amp and ":" not in amp, amp


# ---------------------------------------------------------------------------
# Ambiguity raises, naming what is expected. Previously silent.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_a_multi_entry_dict_raises(cfg):
    """Which of the two is the coupling? Taking the first and discarding the
    rest is what the fermionic path used to do, and it is what this refuses."""
    with pytest.raises(UnknownVertexType):
        _code(cfg, {"gL": "gL", "gR": "gR"})


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_an_empty_dict_raises_rather_than_inventing_a_name(cfg):
    with pytest.raises(UnknownVertexType):
        _code(cfg, {})


def test_the_ambiguity_error_states_the_names_given():
    with pytest.raises(UnknownVertexType) as e:
        _code("svv", {"alpha": "1", "beta": "2"})
    msg = str(e.value)
    assert "alpha" in msg and "beta" in msg, msg


def test_the_error_names_the_vertex_type_it_was_asked_for():
    with pytest.raises(UnknownVertexType) as e:
        _code("svv", {"alpha": "1", "beta": "2"})
    assert "svv" in str(e.value)


def test_the_error_carries_a_corrected_call():
    with pytest.raises(UnknownVertexType) as e:
        _code("vvv", {"alpha": "1", "beta": "2"})
    msg = str(e.value)
    assert '"coupling"' in msg or "'coupling'" in msg, msg


# ---------------------------------------------------------------------------
# The type string stays free: the bosonic dispatch reads the SPIN
# configuration, and 2.9.1's behaviour here must not regress.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vtype", ["svv", "S-V-V", "contact", "field-strength"])
def test_bosonic_type_names_are_a_closed_set(vtype):
    """Bosonic type names are a CLOSED SET per spin configuration.

    This asserted the opposite -- that any string was accepted -- on the
    reasoning that the bosonic branches dispatch on spins, so the name
    carries nothing. That reasoning is what let a fermion decay declared
    with spin 1 on both fermions name its vertex "left_vector" and receive
    the triple-gauge rule, scoring 0.04 on a confident width for a different
    process. An unrecognised name is now refused with the keys valid for the
    declared spins, and the fermionic configurations' keys alongside, since a
    caller refused here usually wanted one of those.

    The accepted sets stay generous -- the configuration's letter-string, a
    curated list, and every name already closed for VFF/SFF -- so names in
    circulation via the `-3pt` spellings keep working.
    """
    assert _code("svv", "g", vtype=vtype), f"rejected accepted type {vtype!r}"


@pytest.mark.parametrize("vtype", ["anything-at-all", "xyz", "left_vector"])
def test_an_unrecognised_bosonic_name_is_refused(vtype):
    with pytest.raises(UnknownVertexType):
        _code("svv", "g", vtype=vtype)


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_a_numeric_coupling_is_still_refused_upstream(cfg):
    """`SymbolicVertex` requires a symbolic NAME, because the formula is the
    deliverable and a number in the vertex would bake itself into it. That
    rejection predates this contract and must survive it -- resolving dict
    couplings must not accidentally open a numeric path."""
    with pytest.raises(ValueError, match="symbolic name"):
        _code(cfg, 0.65)


@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_a_bare_string_coupling_is_unchanged_by_this_feature(cfg):
    """The path that already worked. Must stay byte-identical."""
    assert "g" in _amp_line(_code(cfg, "g"))


# ---------------------------------------------------------------------------
# End to end: a dict coupling must now yield a NUMBER, which is the assertion
# each of the five bosonic task failures would have tripped.
# ---------------------------------------------------------------------------

import json
import math
import os
import shutil
from pathlib import Path


def _wolframscript():
    for cand in (os.environ.get("WOLFRAMSCRIPT_PATH"),
                 "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
                 shutil.which("wolframscript")):
        if cand and Path(cand).exists():
            return cand
    return None


needs_wolfram = pytest.mark.skipif(
    _wolframscript() is None, reason="wolframscript not available",
)


@needs_wolfram
def test_S_to_SS_distinct_scalars_yields_the_known_width(tmp_path):
    """phi -> chi1 chi2 via L = -mu phi chi1 chi2, distinct masses:

        Gamma = mu^2 sqrt((M^2-(m1+m2)^2)(M^2-(m1-m2)^2)) / (16 pi M^3)

    the ground truth hepbench grades decay_S_to_SS part (b) against.
    """
    from tools.eda.compute_rate_tools import ComputeDecayRate

    mu, M, m1, m2 = 315.0, 950.0, 350.0, 410.0
    expect = (mu ** 2 * math.sqrt((M ** 2 - (m1 + m2) ** 2)
                                  * (M ** 2 - (m1 - m2) ** 2))
              / (16 * math.pi * M ** 3))

    out = json.loads(ComputeDecayRate(
        diagram=_spec("sss", {"mu": "mu"}),
        values={"mu": mu, "mS": M, "m1": m1, "m2": m2},
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert out.get("width_GeV") == pytest.approx(expect, rel=1e-6), out


@needs_wolfram
def test_S_to_VV_yields_a_number(tmp_path):
    """The call from the smoke run that returned `ok` with nothing in it."""
    from tools.eda.compute_rate_tools import ComputeDecayRate

    out = json.loads(ComputeDecayRate(
        diagram=_spec("svv", {"g": "g"}),
        values={"g": 0.8, "mS": 950.0, "m1": 200.0, "m2": 300.0},
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert isinstance(out.get("width_GeV"), float), out
    assert out["width_GeV"] > 0, out
