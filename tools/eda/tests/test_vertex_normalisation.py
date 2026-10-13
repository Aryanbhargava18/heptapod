"""
# test_vertex_normalisation.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

`coupling` is the FEYNMAN-RULE coefficient, not the Lagrangian coefficient.

WHY THIS NEEDED SAYING. The emitted SVV rule is i g g^{mu nu}, which is
L = g S V^mu W_mu for DISTINCT vectors. For identical vectors, L = g S V^mu
V_mu gives 2 i g g^{mu nu}, so the caller must pass 2*g. Nothing said so, and
an agent computing S -> V V passed g and got a width exactly 1/4 of ground
truth (hepbench decay_S_to_VV part (a), 0.04 reach, 2026-09-30). It knew the
factor of 2 belonged somewhere; it could not know whether the tool had already
applied it.

THE ASYMMETRY THAT MAKES IT CONFUSING. The scalar case looks like it needs no
such factor:

    L = -(mu/2) phi chi^2   identical chi   pass "mu"    -> correct
    L = g S V^mu V_mu       identical V     pass "2*g"   -> correct

Both follow one rule, and neither is a special case: the vertex factor is what
multiplies i in the Feynman rule, with the combinatorial factor from identical
fields already folded in by the caller. The scalar Lagrangian conventionally
carries an explicit 1/2 which cancels against the two contractions, so "mu"
is already the rule. The vector one does not, so the 2 survives.

These tests PIN both conventions against hepbench ground truth. They are not
testing FeynCalc; they are testing that the meaning of `coupling` does not
drift, because a drift here is a factor of 2 or 4 in every width the toolkit
produces and no caller would see it.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


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


def _width(diagram, values):
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram=diagram, values=values, assume_real_couplings=True,
        base_directory=tempfile.mkdtemp(),
        wolframscript_path=_wolframscript())._run())
    assert out["status"] == "ok", out
    return out


def _sss(coupling):
    return {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "chi", "spin": 0, "mass": "mchi"},
                      {"label": "chi", "spin": 0, "mass": "mchi"}],
            "vertices": [{"type": "sss", "coupling": coupling}]}


def _svv(coupling):
    return {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "V", "spin": 1, "mass": "mV"},
                      {"label": "V", "spin": 1, "mass": "mV"}],
            "vertices": [{"type": "svv", "coupling": coupling}]}


# hepbench ground truth, decay_S_to_SS part (a) and decay_S_to_VV part (a).
SSS_VALUES = {"mu": 480.0, "mS": 950.0, "mchi": 270.0}
SSS_TRUTH = 1.9848170213
SVV_VALUES = {"g": 130.0, "mS": 400.0, "mV": 90.0}
SVV_TRUTH = 121.29029103418922


@needs_wolfram
def test_scalar_identical_pair_takes_the_rule_coupling():
    """L = -(mu/2) phi chi^2 -> rule -i mu, so `mu` is what to pass: the 1/2
    is cancelled by the two contractions of the identical fields."""
    out = _width(_sss("mu"), SSS_VALUES)
    assert out["symmetry_factor"] == 2
    assert out["width_GeV"] == pytest.approx(SSS_TRUTH, rel=1e-8)


@needs_wolfram
def test_passing_the_lagrangian_coefficient_is_wrong_by_four():
    """The mistake this documents. Passing "mu/2" -- the Lagrangian's
    coefficient -- gives exactly 1/4."""
    out = _width(_sss("mu/2"), SSS_VALUES)
    assert out["width_GeV"] == pytest.approx(SSS_TRUTH / 4, rel=1e-8)


@needs_wolfram
def test_vector_identical_pair_needs_the_explicit_factor_two():
    """L = g S V^mu V_mu -> rule 2 i g, so pass `2*g`. This is the call that
    scored 0.04 when it passed `g`."""
    out = _width(_svv("2*g"), SVV_VALUES)
    assert out["symmetry_factor"] == 2
    assert out["width_GeV"] == pytest.approx(SVV_TRUTH, rel=1e-8)


@needs_wolfram
def test_the_bare_coupling_on_identical_vectors_is_wrong_by_four():
    out = _width(_svv("g"), SVV_VALUES)
    assert out["width_GeV"] == pytest.approx(SVV_TRUTH / 4, rel=1e-8)


# ---------------------------------------------------------------------------
# The docstring is the only channel a caller has. If the convention is not in
# it, the convention is not usable -- that is the whole finding.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name", ["ComputeDecayRate", "ComputeCrossSection"])
def test_the_docstring_states_the_coupling_convention(tool_name):
    import tools.eda.compute_rate_tools as m
    doc = getattr(m, tool_name).__doc__ or ""
    # Case-insensitive: the heading spells it FEYNMAN-RULE, the prose
    # "Feynman rule". Pinning the casing would test the formatting.
    assert "feynman" in doc.lower(), \
        f"{tool_name}: no statement of what `coupling` is"
    assert "2*g" in doc, \
        f"{tool_name}: the identical-vector factor of 2 is not shown"
    assert "identical" in doc.lower(), \
        f"{tool_name}: says nothing about repeated fields"
