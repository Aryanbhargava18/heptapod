"""
# test_identical_field_vertex.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

When two legs of a vertex share a label they are the same field, and the
vertex factor carries a combinatorial factor the caller must supply.

WHY A WARNING AND NOT A FIX. The emitted SVV rule is `i g`, the distinct-field
normalisation, so `L = g S V^mu V_mu` needs `2*g`. That has been documented
since 2.16.2 and missed in every run since: decay_S_to_VV part (a) has failed
on it each time, scoring 0 on all four of its stages while part (b) -- distinct
vectors, same tool path -- scores 1.00.

The tool cannot apply the factor itself, for two reasons. A caller following
the current documentation already passes `2*g`, so doubling silently would
hand them 4g. And the direction is convention-dependent: the scalar case runs
the other way, because `L = -(mu/2) phi chi^2` carries an explicit 1/2 that
the two contractions cancel, so its rule is `-i mu` and `"mu"` is correct.
Nothing in a diagram says which Lagrangian normalisation the caller wrote
down, so asserting either would be inventing intent.

So the warning fires where the question arises -- identical labels at a vertex
-- states both cases, and leaves the choice. It is deliberately a warning: it
will fire on calls that are already correct, and that is the acceptable
direction for a reminder whose absence has cost a stage every run.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tools.eda.compute_rate_tools import _identical_field_vertex_warning
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


def _built(final, spin, mass="m", vtype="svv"):
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": l, "spin": spin, "mass": mass} for l in final],
            "vertices": [{"type": vtype, "coupling": "g"}]}
    return build_diagram_from_symbolic(parse_symbolic_diagram(spec))


# ---------------------------------------------------------------------------
# It fires exactly where the question arises.
# ---------------------------------------------------------------------------

def test_identical_vector_legs_warn():
    assert _identical_field_vertex_warning(_built(["V", "V"], 1))


def test_identical_scalar_legs_warn():
    assert _identical_field_vertex_warning(
        _built(["chi", "chi"], 0, vtype="scalar"))


def test_distinct_labels_do_not_warn():
    assert not _identical_field_vertex_warning(_built(["V1", "V2"], 1))


def test_a_conjugate_pair_does_not_warn():
    """f and fbar are not the same field, so no combinatorial factor arises."""
    assert not _identical_field_vertex_warning(_built(["f", "fbar"], 0.5))


def test_distinct_masses_do_not_warn():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final": [{"label": "V", "spin": 1, "mass": "m1"},
                      {"label": "V", "spin": 1, "mass": "m2"}],
            "vertices": [{"type": "svv", "coupling": "g"}]}
    # Same label but different declared masses is contradictory input; the
    # identity question is not what to raise about it.
    d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
    assert not _identical_field_vertex_warning(d)


# ---------------------------------------------------------------------------
# It states both conventions and asserts neither.
# ---------------------------------------------------------------------------

def test_the_warning_gives_the_vector_case():
    w = _identical_field_vertex_warning(_built(["V", "V"], 1))
    assert "2*g" in w, w


def test_the_warning_gives_the_scalar_case_too():
    """Without this the message reads as 'always double', which is wrong for
    a Lagrangian that already carries the 1/2."""
    w = _identical_field_vertex_warning(_built(["V", "V"], 1))
    assert "mu/2" in w or "1/2" in w, w


def test_the_warning_names_the_legs():
    w = _identical_field_vertex_warning(_built(["V", "V"], 1))
    assert "V" in w


def test_the_warning_says_coupling_is_the_rule_coefficient():
    w = _identical_field_vertex_warning(_built(["V", "V"], 1))
    assert "Feynman" in w or "rule coefficient" in w.lower()


def test_the_warning_does_not_claim_the_tool_applied_it():
    """The symmetry factor IS applied automatically; the vertex factor is not.
    Conflating them is how a caller double-counts."""
    w = _identical_field_vertex_warning(_built(["V", "V"], 1))
    assert "symmetry" in w.lower()


# ---------------------------------------------------------------------------
# It reaches the caller.
# ---------------------------------------------------------------------------

def _wolframscript():
    for c in (os.environ.get("WOLFRAMSCRIPT_PATH"),
              "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
              shutil.which("wolframscript")):
        if c and Path(c).exists():
            return c
    return None


@pytest.mark.skipif(_wolframscript() is None, reason="wolframscript missing")
def test_the_warning_appears_in_the_tool_output():
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram={"topology": "decay_1to2",
                 "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
                 "final": [{"label": "V", "spin": 1, "mass": "mV"},
                           {"label": "V", "spin": 1, "mass": "mV"}],
                 "vertices": [{"type": "svv", "coupling": "g"}]},
        values={"g": 130.0, "mS": 400.0, "mV": 90.0},
        assume_real_couplings=True, base_directory=tempfile.mkdtemp(),
        wolframscript_path=_wolframscript())._run())
    assert any("2*g" in w for w in out.get("warnings", [])), out.get("warnings")
