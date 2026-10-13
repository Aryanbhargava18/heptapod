"""
# test_symmetry_factor_visibility.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The identical-particle symmetry factor is DERIVED from the diagram, REPORTED in
the output, and WARNED about when the diagram is ambiguous.

WHAT WENT WRONG. `compute_symmetry_factor` counts identical LABELS in the final
state. An agent computing S -> chi chi labelled the two identical scalars
`chi1` and `chi2` and gave them the same mass symbol `mchi`, so the counter saw
two groups of one and the script got `symmetryFactor = 1` instead of 2. The
graded width came out at exactly 0.5000x ground truth (hepbench
decay_S_to_SS part (a), 2026-09-30). Nothing in the output said which factor
had been used, so neither the agent nor a reader of the trial could see the
assumption they had inherited.

WHY THE LABEL STAYS THE DECLARATION, rather than inferring identity from equal
spin and equal mass. Two same-spin, mass-degenerate particles are a legitimate
and common configuration -- a degenerate doublet -- and applying 1/2! to those
would be wrong by 2 in the other direction, with nothing to signal it. The
diagram already carries the caller's statement of identity in the labels;
adding a second channel that can contradict the first is the bug class this
release exists to remove (a dict coupling contradicting the declared vertex
type, a `mass` field contradicting the label). `Diagram.symmetry_factor`
remains as an explicit OVERRIDE for exotic cases, which is the right role for
it: an escape hatch, not the primary road.

So the rule is: labels declare identity, the factor is derived from them, the
derivation is reported, and a final state that looks identical but is labelled
as distinct produces a warning naming the particles.
"""

import json
import math
import os
import shutil
from pathlib import Path

import pytest

from tools.nda.simple_diagram import compute_symmetry_factor
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)
from tools.eda.compute_rate_tools import _ambiguous_identity_warning


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


def _spec(final_labels, final_mass="mchi", coupling="mu"):
    return {"topology": "decay_1to2",
            "initial": [{"label": "S", "spin": 0, "mass": "mS"}],
            "final": [{"label": l, "spin": 0, "mass": final_mass}
                      for l in final_labels],
            "vertices": [{"type": "scalar", "coupling": coupling}]}


def _built(spec):
    return build_diagram_from_symbolic(parse_symbolic_diagram(spec))


# ---------------------------------------------------------------------------
# The derivation itself is unchanged: labels declare identity.
# ---------------------------------------------------------------------------

def test_shared_labels_give_the_identical_particle_factor():
    assert compute_symmetry_factor(_built(_spec(["chi", "chi"]))) == 2


def test_distinct_labels_give_no_factor():
    assert compute_symmetry_factor(_built(_spec(["chi1", "chi2"]))) == 1


def test_degeneracy_alone_does_not_imply_identity():
    """Same spin, same mass symbol, different labels: a degenerate doublet is
    a real configuration, so this must NOT silently become 1/2!."""
    assert compute_symmetry_factor(_built(_spec(["chi1", "chi2"]))) == 1


def test_an_explicit_override_wins():
    d = _built(_spec(["chi1", "chi2"]))
    d.symmetry_factor = 2
    assert compute_symmetry_factor(d) == 2


# ---------------------------------------------------------------------------
# The ambiguity warning: identical-looking, labelled distinct.
# ---------------------------------------------------------------------------

def test_same_spin_and_mass_but_different_labels_warns():
    w = _ambiguous_identity_warning(_built(_spec(["chi1", "chi2"])))
    assert w, "a degenerate, distinctly-labelled final state must warn"
    assert "chi1" in w and "chi2" in w


def test_the_warning_states_the_factor_in_force():
    w = _ambiguous_identity_warning(_built(_spec(["chi1", "chi2"])))
    assert "symmetry factor 1" in w.lower(), w


def test_the_warning_says_how_to_fix_it_both_ways():
    """One label if they are identical; `symmetry_factor` to be explicit."""
    w = _ambiguous_identity_warning(_built(_spec(["chi1", "chi2"])))
    assert "label" in w.lower()
    assert "symmetry_factor" in w


def test_shared_labels_do_not_warn():
    """Unambiguous: the factor of 2 is already applied."""
    assert not _ambiguous_identity_warning(_built(_spec(["chi", "chi"])))


def test_different_masses_do_not_warn():
    """Genuinely distinct particles, nothing ambiguous about them."""
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "S", "spin": 0, "mass": "mS"}],
            "final": [{"label": "chi1", "spin": 0, "mass": "m1"},
                      {"label": "chi2", "spin": 0, "mass": "m2"}],
            "vertices": [{"type": "scalar", "coupling": "mu"}]}
    assert not _ambiguous_identity_warning(_built(spec))


def test_different_spins_do_not_warn():
    spec = {"topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
            "final": [{"label": "a", "spin": 0, "mass": "mx"},
                      {"label": "b", "spin": 1, "mass": "mx"}],
            "vertices": [{"type": "svv", "coupling": "g"}]}
    assert not _ambiguous_identity_warning(_built(spec))


def test_an_explicit_override_silences_the_warning():
    """The caller has stated the factor; there is nothing left to guess at."""
    d = _built(_spec(["chi1", "chi2"]))
    d.symmetry_factor = 1
    assert not _ambiguous_identity_warning(d)


# ---------------------------------------------------------------------------
# Visibility: the factor and the initial-spin average reach the output.
# ---------------------------------------------------------------------------

@needs_wolfram
def test_the_output_reports_the_symmetry_factor(tmp_path):
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram=_spec(["chi", "chi"]),
        values={"mu": 480.0, "mS": 950.0, "mchi": 270.0},
        assume_real_couplings=True, base_directory=str(tmp_path),
        wolframscript_path=_wolframscript())._run())
    assert out["status"] == "ok", out
    assert out["symmetry_factor"] == 2, out


@needs_wolfram
def test_the_output_reports_the_initial_spin_states(tmp_path):
    """A massive vector parent averages over 3 polarisations; getting that
    wrong is a factor of 3, and it is invisible unless reported."""
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram={"topology": "decay_1to2",
                 "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
                 "final": [{"label": "S1", "spin": 0, "mass": "m1"},
                           {"label": "S2", "spin": 0, "mass": "m2"}],
                 "vertices": [{"type": "ssv", "coupling": "g"}]},
        values={"g": 0.6, "mV": 1200.0, "m1": 150.0, "m2": 220.0},
        assume_real_couplings=True, base_directory=str(tmp_path),
        wolframscript_path=_wolframscript())._run())
    assert out["status"] == "ok", out
    assert out["n_initial_spin_states"] == 3, out


@needs_wolfram
def test_the_ambiguity_warning_reaches_the_caller(tmp_path):
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram=_spec(["chi1", "chi2"]),
        values={"mu": 480.0, "mS": 950.0, "mchi": 270.0},
        assume_real_couplings=True, base_directory=str(tmp_path),
        wolframscript_path=_wolframscript())._run())
    assert out["symmetry_factor"] == 1, out
    assert any("chi1" in w for w in out.get("warnings", [])), out


@needs_wolfram
def test_shared_labels_reproduce_the_known_identical_particle_width(tmp_path):
    """S -> chi chi with L = -(mu/2) phi chi^2. The vertex factor is mu (the
    1/2 is cancelled by the two Wick contractions) and the width carries the
    1/2! identical-particle factor:

        Gamma = mu^2 sqrt(1 - 4 mchi^2/mS^2) / (32 pi mS)

    the hepbench decay_S_to_SS part (a) ground truth. This is the number that
    came out 0.5000x when the factor was 1.
    """
    from tools.eda.compute_rate_tools import ComputeDecayRate
    mu, mS, mchi = 480.0, 950.0, 270.0
    expect = mu ** 2 * math.sqrt(1 - 4 * mchi ** 2 / mS ** 2) / (32 * math.pi * mS)

    out = json.loads(ComputeDecayRate(
        diagram=_spec(["chi", "chi"]),
        values={"mu": mu, "mS": mS, "mchi": mchi},
        assume_real_couplings=True, base_directory=str(tmp_path),
        wolframscript_path=_wolframscript())._run())
    assert out["status"] == "ok", out
    assert out["symmetry_factor"] == 2, out
    assert out["width_GeV"] == pytest.approx(expect, rel=1e-6), out


# ---------------------------------------------------------------------------
# A CONJUGATE PAIR IS NOT AMBIGUOUS. f and fbar have equal spin and equal
# mass and are never the same particle, so the warning must not fire -- and
# must certainly not advise giving them one label, which would apply a 1/2!
# that does not belong there.
#
# Caught by test_declared_masses_yield_a_number_end_to_end, which asserts a
# clean call emits no warnings: the first version of this warning fired on its
# q/qb pair. Two of the ten hepbench symbolic tasks (decay_V_to_FFbar,
# decay_S_to_FFbar) have an f fbar final state, so the advice would have gone
# to an agent on every one of those trials.
# ---------------------------------------------------------------------------

def _pair_spec(l1, l2, spin=0.5, mass="mf"):
    return {"topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
            "final": [{"label": l1, "spin": spin, "mass": mass},
                      {"label": l2, "spin": spin, "mass": mass}],
            "vertices": [{"type": "vector", "coupling": "g"}]}


@pytest.mark.parametrize("l1,l2", [
    ("f", "fbar"),
    ("q", "qb"),
    ("mu", "mubar"),
    ("e-", "e+"),
    ("chi", "chi~"),
])
def test_a_conjugate_pair_does_not_warn(l1, l2):
    assert not _ambiguous_identity_warning(_built(_pair_spec(l1, l2))), (l1, l2)


def test_two_same_sign_labels_still_warn():
    """Not a conjugate pair -- chi1/chi2 remains the ambiguous case."""
    assert _ambiguous_identity_warning(_built(_pair_spec("chi1", "chi2", spin=0)))


def test_a_self_conjugate_pair_written_twice_still_gets_the_factor():
    """A Majorana-like or real scalar pair written with one label is
    unambiguous and keeps its 1/2!."""
    d = _built(_pair_spec("chi", "chi", spin=0))
    assert compute_symmetry_factor(d) == 2
    assert not _ambiguous_identity_warning(d)
