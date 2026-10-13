"""
# test_mass_symbol_contract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

A particle's declared `mass` NAMES the symbol the formula is written in, and a
`values` key that cannot bind RAISES instead of silently yielding no number.

THE BUG THIS CLOSES. `_mass_symbol` synthesised `m<label>` and ignored
`particle.mass` entirely, while the numeric-substitution list was built from
the caller's `values` keys verbatim. The two namespaces never met:

    declared   {"label": "Zp", "mass": "MZP"},  values {"MZP": 2150}
    formula    width = pMag/(8 Pi mZp^2) * ...          <- mZp, not MZP
    substitute widthNum = N[width /. {MZP -> 2150}]     <- binds nothing

So the width came back as an expression, not a number, and the call still
reported `status: ok`. The caller's declared name was accepted, stored on the
parsed particle, and then never used.

MEASURED COST. Five of the ten 1->2 tasks in the hepbench `symbolic` family
failed this way on claude-haiku-4.5 (2026-09-30 smoke, heptapod 2.16.0):

    decay_S_to_SS    0.10   decay_V_to_SS     0.31
    decay_S_to_VV    0.04   decay_V_to_SVp    0.10
    decay_V_to_VVp   0.04

Every one followed the same trajectory: two ComputeDecayRate calls returning
`ok` with no number, then the agent abandoned the tool and hand-derived. On
decay_S_to_SS that hand derivation was out by exactly 2.000x on BOTH parts.
The tool did not fail loudly; it failed in a way the agent could not attribute,
which is the only bug class an agent cannot route around.

WHY HONOURING THE NAME, NOT JUST RAISING. Raising on a mismatch would have told
each of those five agents to rename its keys to `m<label>`. Honouring the
declared name instead makes the first call succeed, and it is the better
contract for this family besides: the rubric grades `*_width_formula` in the
caller's own symbols, so a tool that renames the caller's masses is answering a
different question from the one asked. The guard still exists for a mass string
that is not a usable symbol -- that cannot be honoured, so it raises.
"""

import pytest

from tools.eda.fc_symbols import _mass_symbol
from tools.eda.feyncalc_codegen import (
    SymbolicFeynCalcCodeGenerator, InvalidMassSymbol,
)
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


def _gen(spec, values=None):
    d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
    return SymbolicFeynCalcCodeGenerator(values=values).generate(d)


def _vff(initial_mass="MZP", final_mass="MQ", coupling="gz"):
    return {"topology": "decay_1to2",
            "initial": [{"label": "Zp", "spin": 1, "mass": initial_mass}],
            "final": [{"label": "q", "spin": 0.5, "mass": final_mass},
                      {"label": "qb", "spin": 0.5, "mass": final_mass}],
            "vertices": [{"type": "vector", "coupling": {"g": coupling}}]}


class _P:
    """Minimal stand-in for the generator's Particle protocol."""
    def __init__(self, label=None, mass=None):
        self.label, self.mass = label, mass


# ---------------------------------------------------------------------------
# The symbol contract, at the chokepoint every caller funnels through.
# ---------------------------------------------------------------------------

def test_a_declared_symbolic_mass_is_the_symbol():
    """The regression. It used to return 'mZp' and discard 'MZP'."""
    assert _mass_symbol(_P("Zp", "MZP"), 0) == "MZP"


def test_an_omitted_mass_still_defaults_to_m_label():
    """The documented default, and the behaviour every passing call relies on.
    Must not change."""
    assert _mass_symbol(_P("Zp", None), 0) == "mZp"


def test_an_unlabelled_particle_still_falls_back_to_the_index():
    assert _mass_symbol(_P(None, None), 3) == "m3"


def test_a_numeric_mass_keeps_the_label_derived_symbol():
    """A number is a VALUE, not a name: there is nothing to name the symbol
    after, and the on-shell section substitutes the number against
    `m<label>`. Honouring strings must not disturb this path."""
    assert _mass_symbol(_P("Zp", 2150.0), 0) == "mZp"
    assert _mass_symbol(_P("Zp", 0), 0) == "mZp"


def test_a_declared_mass_survives_label_sanitisation():
    """`_safe_symbol` rewrites a LABEL for Mathematica ('e+' -> 'ep'). A
    declared mass is already a symbol and is used as given, so a label that
    would be mangled does not mangle the mass."""
    assert _mass_symbol(_P("e+", "me"), 0) == "me"


def test_two_particles_may_declare_the_same_mass():
    """Identical final states are the common case (phi -> chi chi), and they
    must share one symbol rather than collide."""
    a, b = _P("chi", "mchi"), _P("chi", "mchi")
    assert _mass_symbol(a, 0) == _mass_symbol(b, 1) == "mchi"


def test_distinct_labels_may_still_declare_one_shared_mass():
    """Degenerate but distinct particles, e.g. a mass-degenerate doublet."""
    assert _mass_symbol(_P("chi1", "mchi"), 0) == "mchi"
    assert _mass_symbol(_P("chi2", "mchi"), 1) == "mchi"


# ---------------------------------------------------------------------------
# The guard: a mass string that cannot be a symbol must raise, not be
# silently rewritten into one. Explicit beats a quiet transform, because a
# quiet transform reintroduces exactly the namespace split above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["m 1", "2m", "m-1", "m+", "", "  ",
                                 "m/1", "m(1)", "m.1"])
def test_an_unusable_mass_string_raises(bad):
    with pytest.raises(InvalidMassSymbol):
        _mass_symbol(_P("Zp", bad), 0)


def test_the_error_names_the_offending_value_and_the_particle():
    with pytest.raises(InvalidMassSymbol) as e:
        _mass_symbol(_P("Zp", "2m"), 0)
    msg = str(e.value)
    assert "2m" in msg and "Zp" in msg


def test_the_error_states_the_default_so_omitting_is_a_known_option():
    """An agent that trips this needs to know it can just leave mass out."""
    with pytest.raises(InvalidMassSymbol) as e:
        _mass_symbol(_P("Zp", "m 1"), 0)
    assert "mZp" in str(e.value)


def test_underscores_and_digits_are_usable_symbols():
    """Physics names look like this; they must not trip the guard."""
    for ok in ("m_chi", "mChi1", "M2", "mZprime", "Lambda_QCD"):
        assert _mass_symbol(_P("x", ok), 0) == ok


# ---------------------------------------------------------------------------
# End to end: the formula and the substitution list must share ONE namespace.
# This is the invariant whose violation produced the five failures; asserting
# it on generated code is what keeps a future refactor from splitting them
# again somewhere other than `_mass_symbol`.
# ---------------------------------------------------------------------------

def test_the_formula_is_written_in_the_declared_masses():
    code = _gen(_vff()).code
    assert "MZP" in code and "MQ" in code
    assert "mZp" not in code, "label-derived symbol leaked into the formula"


def test_every_substitution_key_appears_in_the_generated_formula():
    """The bug in one assertion: `MZP -> 2150` used to bind nothing."""
    values = {"gz": 0.47, "MZP": 2150.0, "MQ": 640.0}
    code = _gen(_vff(), values=values).code
    body = code.split("widthNum")[0]          # everything before substitution
    for key in values:
        assert key in body, f"values key {key!r} is absent from the formula"


def test_reported_symbols_are_the_declared_names():
    res = _gen(_vff(), values={"gz": 0.47, "MZP": 2150.0, "MQ": 640.0})
    syms = set(getattr(res, "symbols", []) or [])
    if syms:                                   # generator exposes them
        assert {"MZP", "MQ"} <= syms
        assert "mZp" not in syms


# ---------------------------------------------------------------------------
# The five smoke failures, as named regressions. Each is the diagram shape the
# agent actually sent, with the `values` keys it actually passed.
# ---------------------------------------------------------------------------

def _assert_binds(spec, values):
    code = _gen(spec, values=values).code
    body = code.split("widthNum")[0]
    unbound = [k for k in values if k not in body]
    assert not unbound, f"{unbound} bind nothing in the generated formula"


def test_regression_decay_S_to_VV_distinct_vectors():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
         "final": [{"label": "V1", "spin": 1, "mass": "m1"},
                   {"label": "V2", "spin": 1, "mass": "m2"}],
         "vertices": [{"type": "svv", "coupling": {"g": "g"}}]},
        {"g": 0.8, "mS": 950.0, "m1": 200.0, "m2": 300.0})


def test_regression_decay_S_to_SS_identical_scalars():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
         "final": [{"label": "chi", "spin": 0, "mass": "mchi"},
                   {"label": "chi", "spin": 0, "mass": "mchi"}],
         "vertices": [{"type": "scalar", "coupling": {"mu": "mu"}}]},
        {"mu": 480.0, "mS": 950.0, "mchi": 270.0})


def test_regression_decay_S_to_SS_distinct_scalars():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
         "final": [{"label": "chi1", "spin": 0, "mass": "m1"},
                   {"label": "chi2", "spin": 0, "mass": "m2"}],
         "vertices": [{"type": "scalar", "coupling": {"mu": "mu"}}]},
        {"mu": 315.0, "mS": 950.0, "m1": 350.0, "m2": 410.0})


def test_regression_decay_V_to_SS():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
         "final": [{"label": "S1", "spin": 0, "mass": "m1"},
                   {"label": "S2", "spin": 0, "mass": "m2"}],
         "vertices": [{"type": "ssv", "coupling": {"g": "g"}}]},
        {"g": 0.6, "mV": 1200.0, "m1": 150.0, "m2": 220.0})


def test_regression_decay_V_to_SVp():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
         "final": [{"label": "S", "spin": 0, "mass": "mS"},
                   {"label": "Vp", "spin": 1, "mass": "mVp"}],
         "vertices": [{"type": "svv", "coupling": {"g": "g"}}]},
        {"g": 0.7, "mV": 1500.0, "mS": 300.0, "mVp": 400.0})


def test_regression_decay_V_to_VVp():
    _assert_binds(
        {"topology": "decay_1to2",
         "initial": [{"label": "Zp", "spin": 1, "mass": "mZp"}],
         "final": [{"label": "Wp", "spin": 1, "mass": "mW"},
                   {"label": "Wm", "spin": 1, "mass": "mW"}],
         "vertices": [{"type": "vvv", "coupling": {"g": "g"}}]},
        {"g": 0.65, "mZp": 2000.0, "mW": 80.4})


# ---------------------------------------------------------------------------
# Mixed and omitted declarations must keep working alongside declared ones,
# since an agent will do this.
# ---------------------------------------------------------------------------

def test_a_diagram_may_mix_declared_and_omitted_masses():
    code = _gen(
        {"topology": "decay_1to2",
         "initial": [{"label": "Zp", "spin": 1, "mass": "MZP"}],
         "final": [{"label": "q", "spin": 0.5},
                   {"label": "qb", "spin": 0.5}],
         "vertices": [{"type": "vector", "coupling": {"g": "g"}}]},
        values={"g": 0.47, "MZP": 2150.0}).code
    assert "MZP" in code
    assert "mq" in code, "omitted mass should still default to m<label>"


def test_omitting_every_mass_is_unchanged_by_this_feature():
    """The pre-existing happy path (labels chosen so m<label> matches the
    values keys) must be byte-for-byte unaffected."""
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1}],
            "final": [{"label": "f", "spin": 0.5},
                      {"label": "fbar", "spin": 0.5}],
            "vertices": [{"type": "vector", "coupling": {"g": "g"}}]}
    code = _gen(spec, values={"g": 0.47, "mV": 2150.0, "mf": 640.0}).code
    for sym in ("mV", "mf", "mfbar"):
        assert sym in code


# ---------------------------------------------------------------------------
# The warning an agent actually reads. When no number comes back, the message
# must NAME the formula's symbols and the keys that were passed, because that
# difference IS the fix.
#
# There were two branches and the unhelpful one fired in the worse case: if
# Mathematica emitted a width marker that would not parse as a float, the
# message named the missing symbols; if no width marker parsed at all, it said
# only "compare the `symbols` field against the keys you passed". That second
# branch is what the five failing hepbench trials hit -- the case where the
# agent has nothing bound and least to go on.
# ---------------------------------------------------------------------------

from tools.eda.compute_rate_tools import _unresolved_warning


def test_warning_names_the_symbols_and_the_keys_given():
    w = _unresolved_warning(["width_GeV"], ["g", "mphi", "mV1", "mV2"],
                            {"g": 1, "mS": 2, "m1": 3, "m2": 4})
    assert "mphi" in w and "mV1" in w and "mV2" in w
    assert "mS" in w and "m1" in w


def test_warning_with_no_parsed_marker_still_names_them():
    """The branch that used to be vague: nothing came back at all."""
    w = _unresolved_warning([], ["g", "mphi"], {"g": 1, "mS": 2})
    assert "mphi" in w, w
    assert "compare the `symbols` field" not in w, w


def test_warning_points_at_the_mass_declaration_when_masses_are_unbound():
    """An unbound MASS is the signature of this bug, so the message should say
    how to fix it at the source -- the diagram's `mass` field, not `values`."""
    w = _unresolved_warning(["width_GeV"], ["mphi", "mV1"],
                            {"mS": 2, "m1": 3},
                            mass_symbols=("mphi", "mV1"))
    assert "mass" in w.lower()
    assert "`mass` field" in w


def test_warning_is_quiet_about_masses_when_a_coupling_is_what_is_missing():
    w = _unresolved_warning(["width_GeV"], ["gz"], {})
    assert "gz" in w
    assert "names a mass" not in w


def test_a_coupling_called_mu_is_not_mistaken_for_a_mass():
    """`mu` is the coupling in decay_S_to_SS (L = -(mu/2) phi chi^2). A
    "starts with m" heuristic would call it a mass and point the caller at
    the wrong field."""
    w = _unresolved_warning(["width_GeV"], ["mu"], {}, mass_symbols=("mphi",))
    assert "names a mass" not in w, w


def test_the_mass_hint_fires_when_a_real_mass_symbol_is_unbound():
    w = _unresolved_warning(["width_GeV"], ["mphi", "mV1"], {"mS": 1},
                            mass_symbols=("mphi", "mV1"))
    assert "mass" in w.lower()


# ---------------------------------------------------------------------------
# End to end through Mathematica: the declared names must produce a NUMBER.
# This is the assertion the five failures would each have tripped.
# ---------------------------------------------------------------------------

import json
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
def test_declared_masses_yield_a_number_end_to_end(tmp_path):
    """decay_V_to_FFbar with caller-chosen mass names. Checked against the
    closed form Gamma = g^2 M/(12 pi) (1 + 2 x) sqrt(1 - 4 x), x = (m/M)^2."""
    import math
    from tools.eda.compute_rate_tools import ComputeDecayRate

    g, M, m = 0.47, 2150.0, 640.0
    x = (m / M) ** 2
    expect = g ** 2 * M / (12 * math.pi) * (1 + 2 * x) * math.sqrt(1 - 4 * x)

    out = json.loads(ComputeDecayRate(
        diagram=_vff(initial_mass="MZP", final_mass="MQ", coupling="gz"),
        values={"gz": g, "MZP": M, "MQ": m},
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert out.get("width_GeV") == pytest.approx(expect, rel=1e-9), out
    assert set(out["symbols"]) <= {"gz", "MZP", "MQ"}, out["symbols"]
    assert not out.get("warnings"), out.get("warnings")


@needs_wolfram
def test_the_S_to_VV_failure_now_produces_a_number(tmp_path):
    """The exact call from the 2026-09-30 smoke run that returned `ok` with no
    number and sent the agent off to hand-derive."""
    from tools.eda.compute_rate_tools import ComputeDecayRate

    out = json.loads(ComputeDecayRate(
        diagram={"topology": "decay_1to2",
                 "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
                 "final": [{"label": "V1", "spin": 1, "mass": "m1"},
                           {"label": "V2", "spin": 1, "mass": "m2"}],
                 "vertices": [{"type": "svv", "coupling": {"g": "g"}}]},
        values={"g": 0.8, "mS": 950.0, "m1": 200.0, "m2": 300.0},
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert isinstance(out.get("width_GeV"), float), out
    assert out["width_GeV"] > 0
