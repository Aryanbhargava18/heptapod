"""
# test_chiral_sum_guidance.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Refusing a coherent sum for a decay must name the chiral vertex, because the
obvious fallback is wrong.

THE TRAP. `generate_sum` handles 2->2 only. An agent computing a chiral decay
width -- L = fbar gamma^mu (gL P_L + gR P_R) F V_mu -- reaches for
`diagrams: [L-diagram, R-diagram]`, is refused, and then adds the two pieces
separately. That is not the same calculation: the projectors act before the
spinor contraction, so

    |M_L + M_R|^2 = |M_L|^2 + |M_R|^2 + 2 Re(M_L M_R^*)

and the interference term is dropped. In its own words, from a graded trial of
hepbench decay_F_to_FprimeS: "User falls back to `yukawa` type with separate
diagrams, which *appears* to work but silently produces wrong physics (missing
interference)."

The capability already exists -- one diagram, `type: "chiral"`, coupling
{gL, gR} -- so this is a discoverability failure, and the refusal is the one
place the caller is guaranteed to be reading. Two of the stages the family
grades are part (b) chiral widths, and the b_* stages are where both arms fail
most, so the cost is systematic rather than incidental.
"""

import pytest

from tools.eda.feyncalc_codegen import SymbolicFeynCalcCodeGenerator, Channel
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


def _decay(coupling, vtype="chiral"):
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "Psi", "spin": 0.5, "mass": "mF"}],
            "final": [{"label": "psi", "spin": 0.5, "mass": "mf"},
                      {"label": "V", "spin": 1, "mass": "mV"}],
            "vertices": [{"type": vtype, "coupling": coupling}]}
    return build_diagram_from_symbolic(parse_symbolic_diagram(spec))


def _sum_two_decays():
    gen = SymbolicFeynCalcCodeGenerator()
    return gen.generate_sum([(_decay("gL", "left-handed"), Channel.CONTACT),
                             (_decay("gR", "right-handed"), Channel.CONTACT)])


def test_a_decay_handed_to_generate_sum_is_still_refused():
    res = _sum_two_decays()
    assert res.warnings
    assert not res.code


def test_the_refusal_names_the_chiral_vertex():
    res = _sum_two_decays()
    msg = " ".join(res.warnings)
    assert "chiral" in msg.lower(), msg


def test_the_refusal_shows_the_coupling_form():
    """A caller needs the shape, not just the name."""
    res = _sum_two_decays()
    msg = " ".join(res.warnings)
    assert "gL" in msg and "gR" in msg, msg


def test_the_refusal_says_why_adding_separately_is_wrong():
    """Without this the caller does the arithmetic by hand and loses the
    cross term -- which is what happened."""
    res = _sum_two_decays()
    msg = " ".join(res.warnings).lower()
    assert "interference" in msg, msg


def test_the_refusal_still_says_decays_go_one_at_a_time():
    res = _sum_two_decays()
    msg = " ".join(res.warnings)
    assert "generate()" in msg or "one at a time" in msg


def test_a_single_chiral_decay_is_the_supported_path():
    """The thing the refusal points at must actually work."""
    res = SymbolicFeynCalcCodeGenerator().generate(
        _decay({"gL": "gL", "gR": "gR"}, "chiral"))
    assert res.code
    assert "GA[7]" in res.code and "GA[6]" in res.code   # both projectors
