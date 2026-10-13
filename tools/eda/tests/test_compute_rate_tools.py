"""
# test_compute_rate_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for ComputeDecayRate / ComputeCrossSection — the one-call observables.

Split in two. The codegen and parsing tests are pure Python and always run.
The end-to-end cases need wolframscript and are skipped without it; they are
the ones that matter, because they check the tools against decay widths and
cross sections whose values were established independently (explicit
polarization vectors and Dirac spinors, not FeynCalc), so a convention error
inside the generator cannot agree with them by construction.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from tools.eda.feyncalc_codegen import SymbolicFeynCalcCodeGenerator
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)
from tools.wolfram.wolfram_runner import _to_float


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


V_TO_FFBAR = {
    "topology": "decay_1to2",
    "initial": [{"label": "V", "spin": 1}],
    "final": [{"label": "f", "spin": 0.5}, {"label": "fbar", "spin": 0.5}],
}

PHI_CONTACT = {
    "topology": "scattering_2to2",
    "initial": [{"label": "phi", "spin": 0}, {"label": "phi", "spin": 0}],
    "final": [{"label": "phi", "spin": 0}, {"label": "phi", "spin": 0}],
    "vertices": [{"type": "contact", "coupling": "lam"}],
}


def _decay_spec(vertex):
    return {**V_TO_FFBAR, "vertices": [vertex]}


# ---------------------------------------------------------------------------
# Mathematica number parsing (the bug that made small results strings)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2.2839*^-7", 2.2839e-7),      # Mathematica scientific notation
    ("-1.5*^12", -1.5e12),
    ("2.5*^+4", 2.5e4),
    ("88.92958931828068", 88.92958931828068),
    ("3", 3.0),
    ("1.23`20.", 1.23),             # precision-marked real
])
def test_to_float_parses_mathematica_reals(raw, expected):
    got = _to_float(raw)
    assert isinstance(got, float)
    assert got == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("raw", ["Undefined", "Indeterminate", "mf + 3"])
def test_to_float_keeps_non_numbers_as_strings(raw):
    """A value Mathematica could not reduce must NOT masquerade as a number."""
    assert _to_float(raw) == raw


# ---------------------------------------------------------------------------
# Codegen: `values` adds numerics without disturbing the symbolic path
# ---------------------------------------------------------------------------

def _generate(spec, values=None):
    sym = parse_symbolic_diagram(spec)
    diagram = build_diagram_from_symbolic(sym)
    return SymbolicFeynCalcCodeGenerator(values=values).generate(diagram)


def test_no_values_emits_no_numerics():
    """The pre-existing ComputeSymbolicAmplitude behaviour is unchanged."""
    code = _generate(_decay_spec({"type": "vector", "coupling": "g"})).code
    assert "SYMBOLIC_RESULT[width]" in code
    assert "NUMERICAL_RESULT" not in code


def test_values_emit_both_formula_and_number():
    code = _generate(
        _decay_spec({"type": "vector", "coupling": "g"}),
        values={"g": 0.47, "mV": 2150, "mf": 640, "mfbar": 640},
    ).code
    assert "SYMBOLIC_RESULT[width]" in code       # closed form survives
    assert "NUMERICAL_RESULT[width_GeV]" in code
    # The substitution must come AFTER the symbolic print, or the formula
    # would be reported with the numbers already folded in.
    assert code.index("SYMBOLIC_RESULT[width]") < code.index("widthNum = N[")


# ---------------------------------------------------------------------------
# End to end, against independently established values
# ---------------------------------------------------------------------------

@needs_wolfram
@pytest.mark.parametrize("vertex,values,expected_gev", [
    # Gamma = g^2 mV/(12 pi) (1 + 2 mf^2/mV^2) sqrt(1 - 4 mf^2/mV^2)
    ({"type": "vector", "coupling": "g"},
     {"g": 0.47}, 11.915957872612209),
    # Gamma = mV/(12 pi) sqrt(1 - 4x) [gV^2 (1 + 2x) + gA^2 (1 - 4x)]
    ({"type": "vector-axial", "coupling": {"gV": "gV", "gA": "gA"}},
     {"gV": 0.52, "gA": 0.36}, 18.419815247250312),
])
def test_decay_rate_matches_reference(tmp_path, vertex, values, expected_gev):
    from tools.eda.compute_rate_tools import ComputeDecayRate

    out = json.loads(ComputeDecayRate(
        diagram=_decay_spec(vertex),
        values={"mV": 2150, "mf": 640, "mfbar": 640, **values},
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert out["width_GeV"] == pytest.approx(expected_gev, rel=1e-9)
    # One call is enough: the formula comes back too, ready for sympy.
    assert out["width_formula"]
    import sympy
    sympy.sympify(out["width_formula"])


@needs_wolfram
def test_cross_section_matches_reference(tmp_path):
    """phi phi -> phi phi through a quartic contact: sigma = lam^2/(32 pi s)."""
    from tools.eda.compute_rate_tools import ComputeCrossSection

    out = json.loads(ComputeCrossSection(
        diagram=PHI_CONTACT, channel="contact", sqrt_s=480.0,
        values={"lam": 2.3, "mphi": 75.0}, assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "ok", out
    assert out["sigma_pb"] == pytest.approx(88.92958931828066, rel=1e-9)
    # sigma_GeV2 is ~2.3e-7, i.e. exactly the magnitude Mathematica prints in
    # its own notation — it must still arrive as a float.
    assert isinstance(out["sigma_GeV2"], float)
    assert out["sigma_GeV2"] == pytest.approx(2.2838803411353252e-07, rel=1e-9)


@needs_wolfram
def test_missing_value_is_reported_not_silently_symbolic(tmp_path):
    """Omitting a symbol must not yield a 'number' that is really a string.

    This asserted `status: ok` plus a warning. 2.16.1 strengthens exactly that
    intent into an invariant: if `values` were supplied and no number came
    back, the CALL FAILED, whatever the cause -- because `ok` with no number
    is what let every instance of this bug class past the agent. It read `ok`,
    found no number, and hand-derived rather than retrying. So the assertions
    move from "warned" to "errored", and additionally pin that the closed form
    survives the failure, since the formula is a graded deliverable in its own
    right.
    """
    from tools.eda.compute_rate_tools import ComputeDecayRate

    out = json.loads(ComputeDecayRate(
        diagram=_decay_spec({"type": "vector", "coupling": "g"}),
        values={"g": 0.47, "mV": 2150, "mf": 640},      # no mfbar
        assume_real_couplings=True,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())

    assert out["status"] == "error", out
    assert out["error"] == "No Numeric Result", out
    assert "width_GeV" not in out            # absent, not a string
    assert "mfbar" in out["symbols"]
    assert "mfbar" in out["reason"], out["reason"]
    # The closed form is still there to be used and graded.
    assert out.get("width_formula"), out


@needs_wolfram
def test_wrong_process_type_is_refused(tmp_path):
    """A 2->2 handed to the decay tool should say so, not produce nonsense."""
    from tools.eda.compute_rate_tools import ComputeDecayRate

    out = ComputeDecayRate(
        diagram=PHI_CONTACT,
        base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run()
    assert "ComputeCrossSection" in out


# ---------------------------------------------------------------------------
# The archetype sweep: one case per vertex class, against values established
# independently of FeynCalc. These are the cases that would catch a convention
# regression in the generator, which unit tests on the facade cannot.
# ---------------------------------------------------------------------------

S = lambda l, **kw: {"label": l, "spin": 0, **kw}
Fm = lambda l, **kw: {"label": l, "spin": 0.5, **kw}
Vc = lambda l, **kw: {"label": l, "spin": 1, **kw}


@needs_wolfram
@pytest.mark.parametrize("name,initial,final,vertices,values,expected", [
    ("SFF scalar", [S("S")], [Fm("f"), Fm("fbar")],
     [{"type": "yukawa", "coupling": "y"}],
     {"y": 0.87, "mS": 1240., "mf": 320., "mfbar": 320.}, 23.464911722945107),
    ("SFF pseudoscalar", [S("S")], [Fm("f"), Fm("fbar")],
     [{"type": "pseudoscalar", "coupling": "y"}],
     {"y": 0.54, "mS": 1240., "mf": 320., "mfbar": 320.}, 12.322595987052027),
    ("SFF chiral", [S("S")], [Fm("f1"), Fm("f2bar")],
     [{"type": "chiral", "coupling": {"gL": "yL", "gR": "yR"}}],
     {"yL": 0.31, "yR": 0.78, "mS": 1240., "mf1": 410., "mf2bar": 95.},
     13.049667657321574),
    ("VFF vector", [Vc("V")], [Fm("f"), Fm("fbar")],
     [{"type": "vector", "coupling": "g"}],
     {"g": 0.47, "mV": 2150., "mf": 640., "mfbar": 640.}, 11.915957872612209),
    ("VFF vector-axial", [Vc("V")], [Fm("f"), Fm("fbar")],
     [{"type": "vector-axial", "coupling": {"gV": "gV", "gA": "gA"}}],
     {"gV": 0.52, "gA": 0.36, "mV": 2150., "mf": 640., "mfbar": 640.},
     18.419815247250312),
    ("SFF F->f'S", [Fm("F")], [Fm("f"), S("S")],
     [{"type": "yukawa", "coupling": "y"}],
     {"y": 0.66, "mF": 880., "mf": 130., "mS": 340.}, 7.312873036225345),
    ("VFF F->f'V chiral", [Fm("F")], [Fm("f"), Vc("V")],
     [{"type": "vector-chiral", "coupling": {"gL": "gL", "gR": "gR"}}],
     {"gL": 0.58, "gR": 0.29, "mF": 900., "mf": 180., "mV": 240.},
     42.847094120552605),
    ("SSS identical", [S("S")], [S("chi"), S("chi")],
     [{"type": "scalar", "coupling": "mu"}],
     {"mu": 480., "mS": 950., "mchi": 270.}, 1.9848170214975707),
    ("SSV S->S'V", [S("S")], [S("Sp"), Vc("V")],
     [{"type": "ssv", "coupling": "g"}],
     {"g": 0.9, "mS": 600., "mSp": 140., "mV": 250.}, 23.219067397452875),
    ("SSV V->SS", [Vc("V")], [S("a"), S("b")],
     [{"type": "ssv", "coupling": "g"}],
     {"g": 0.85, "mV": 500., "ma": 120., "mb": 210.}, 0.9668097975501307),
    ("SVV S->V1V2", [S("S")], [Vc("a"), Vc("b")],
     [{"type": "svv", "coupling": "g"}],
     {"g": 110., "mS": 400., "ma": 90., "mb": 150.}, 11.885133513029855),
    ("SVV V->SV'", [Vc("V")], [S("S"), Vc("Vp")],
     [{"type": "svv", "coupling": "g"}],
     {"g": 180., "mV": 700., "mS": 160., "mVp": 250.}, 1.0525716496014468),
])
def test_decay_archetype_matches_reference(tmp_path, name, initial, final,
                                           vertices, values, expected):
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram={"topology": "decay_1to2", "initial": initial,
                 "final": final, "vertices": vertices},
        values=values, assume_real_couplings=True,
        base_directory=str(tmp_path), wolframscript_path=_wolframscript(),
    )._run())
    assert out["status"] == "ok", out
    assert out["width_GeV"] == pytest.approx(expected, rel=1e-9)


@needs_wolfram
def test_vvv_matches_the_literature_zprime_width(tmp_path):
    """The triple-gauge vertex, against Gamma(Z' -> W+W-) at three mass ratios.

    All-incoming momenta is not a free convention -- the cyclic vertex formula
    is DERIVED under k1+k2+k3 = 0 -- and the as-drawn assignment agrees only
    asymptotically. mV/m = 3 is deep in the regime where they differ (as-drawn
    lands at 0.209 of the literature value), so this case discriminates.
    """
    import math
    from tools.eda.compute_rate_tools import ComputeDecayRate

    def literature(g, mV, m):
        x = m * m / (mV * mV)
        return (g * g / (192 * math.pi)) * mV * (mV / m) ** 4 \
            * (1 - 4 * x) ** 1.5 * (1 + 20 * x + 12 * x * x)

    for g, mV, m in ((0.8, 900., 200.), (0.65, 1200., 180.), (0.8, 900., 300.)):
        out = json.loads(ComputeDecayRate(
            diagram={"topology": "decay_1to2", "initial": [Vc("V")],
                     "final": [Vc("a"), Vc("b")],
                     "vertices": [{"type": "vvv", "coupling": "g"}]},
            values={"g": g, "mV": mV, "ma": m, "mb": m},
            assume_real_couplings=True, base_directory=str(tmp_path),
            wolframscript_path=_wolframscript(),
        )._run())
        assert out["status"] == "ok", out
        assert out["width_GeV"] == pytest.approx(literature(g, mV, m), rel=1e-9)


@needs_wolfram
def test_identical_fields_need_the_doubled_vertex(tmp_path):
    """L = g S V_mu V^mu has Feynman rule 2 i g: the tool implements i g.

    Pinned because the failure is silent and looks like physics: the width
    comes out a factor of 4 low, with no error. The final-state symmetry
    factor IS automatic; only the vertex doubling is the caller's.
    """
    from tools.eda.compute_rate_tools import ComputeDecayRate

    def width(coupling):
        out = json.loads(ComputeDecayRate(
            diagram={"topology": "decay_1to2", "initial": [S("S")],
                     "final": [Vc("V"), Vc("V")],
                     "vertices": [{"type": "svv", "coupling": coupling}]},
            values={"g": 130., "mS": 400., "mV": 90.},
            assume_real_couplings=True, base_directory=str(tmp_path),
            wolframscript_path=_wolframscript(),
        )._run())
        assert out["status"] == "ok", out
        return out["width_GeV"]

    doubled, bare = width("2*g"), width("g")
    assert doubled == pytest.approx(121.29029103418922, rel=1e-9)
    assert doubled / bare == pytest.approx(4.0, rel=1e-9)


@needs_wolfram
@pytest.mark.parametrize("propagator,vertices,values,expected", [
    # s-channel MASSLESS vector: the numerator is chosen at generation time
    # from the DECLARED mass, so "mass": 0 is load-bearing. Leave it symbolic
    # and -g + qq/m^2 divides by zero when mProp0 -> 0 is substituted later.
    ({"label": "V", "spin": 1, "mass": 0},
     [{"type": "vector", "coupling": "g"}, {"type": "vector", "coupling": "g"}],
     {"g": 0.55, "mf1": 0., "mf1bar": 0., "mf2": 0., "mf2bar": 0.},
     1.773563977991934),
    # s-channel massive scalar
    ({"label": "MS", "spin": 0},
     [{"type": "yukawa", "coupling": "y1"}, {"type": "yukawa", "coupling": "y2"}],
     {"y1": 0.92, "y2": 0.34, "mf1": 0., "mf1bar": 0., "mf2": 0., "mf2bar": 0.,
      "mMS": 1900., "mProp0": 1900.},
     0.04265786089078201),
])
def test_schannel_cross_section_matches_reference(tmp_path, propagator,
                                                  vertices, values, expected):
    from tools.eda.compute_rate_tools import ComputeCrossSection
    out = json.loads(ComputeCrossSection(
        diagram={"topology": "scattering_2to2",
                 "initial": [Fm("f1"), Fm("f1bar")],
                 "final": [Fm("f2"), Fm("f2bar")],
                 "vertices": vertices, "propagators": [propagator]},
        channel="s", sqrt_s=730., values=values, assume_real_couplings=True,
        base_directory=str(tmp_path), wolframscript_path=_wolframscript(),
        timeout=600,
    )._run())
    assert out["status"] == "ok", out
    assert out["sigma_pb"] == pytest.approx(expected, rel=1e-9)


@needs_wolfram
def test_three_body_decay_is_cleanly_unsupported(tmp_path):
    """1->3 has no ProcessType; it must say so rather than produce a number."""
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = json.loads(ComputeDecayRate(
        diagram={"topology": "decay_1to3", "initial": [Fm("F")],
                 "final": [Fm("a"), Fm("b"), Fm("c")],
                 "vertices": [{"type": "vector", "coupling": "g"},
                              {"type": "vector", "coupling": "g"}],
                 "propagators": [{"label": "V", "spin": 1}]},
        values={"g": 0.62, "mF": 17.3}, base_directory=str(tmp_path),
        wolframscript_path=_wolframscript(),
    )._run())
    assert out["status"] == "unsupported"
    assert "width_GeV" not in out


# ---------------------------------------------------------------------------
# Strict vertex types. Until 2.9.1 an unrecognized type silently produced the
# plain-coupling vertex, so a typo -- or a type belonging to the other
# bilinear -- returned a confident, well-formed, wrong width.
# ---------------------------------------------------------------------------

from tools.eda.feyncalc_codegen import (  # noqa: E402
    FeynCalcCodeGenerator, UnknownVertexType,
)

VFF_SPELLINGS = ["vector", "gauge-vector", "axial-vector", "vector-axial",
                 "va", "chiral", "vector-chiral", "left-handed",
                 "right-handed", "tensor", "dipole", "tensor-chiral",
                 "dipole-chiral"]
SFF_SPELLINGS = ["yukawa", "scalar", "pseudoscalar", "chiral",
                 "yukawa-chiral", "scalar-chiral", "scalar-va"]


@pytest.mark.parametrize("vtype", VFF_SPELLINGS)
def test_every_documented_vff_spelling_is_accepted(vtype):
    """Guards the hyphen trap: `base` is normalized before matching, so a
    check written against the display spelling rejects 'gauge-vector'."""
    FeynCalcCodeGenerator()._vff_gamma_structure(
        vtype, "g", "mu", vec_momentum="p")


@pytest.mark.parametrize("vtype", SFF_SPELLINGS)
def test_every_documented_sff_spelling_is_accepted(vtype):
    FeynCalcCodeGenerator()._sff_coupling_structure(vtype, "g")


@pytest.mark.parametrize("vtype", [
    "totally_made_up_xyzzy",
    "vector_axial_coupling",     # plausible-looking near-miss
    "pseudoscalar",              # real type, but SFF -- not valid on a VFF leg
])
def test_unknown_vff_type_raises_rather_than_defaulting(vtype):
    with pytest.raises(UnknownVertexType) as e:
        FeynCalcCodeGenerator()._vff_gamma_structure(
            vtype, "g", "mu", vec_momentum="p")
    # The message must teach the vocabulary: an agent that guessed has no
    # other way to discover it when ComputeSymbolicAmplitude is not served.
    assert "vector-axial" in str(e.value)
    assert "Valid types" in str(e.value)


def test_error_echoes_the_callers_literal_string():
    """Not the normalized form -- showing 'vectoraxialcoupling' for an input
    of 'vector_axial_coupling' reads like a second, unrelated bug."""
    spec = {"topology": "decay_1to2", "initial": [Vc("V")],
            "final": [Fm("f"), Fm("fbar")],
            "vertices": [{"type": "vector_axial_coupling", "coupling": "g"}]}
    sym = parse_symbolic_diagram(spec)
    with pytest.raises(UnknownVertexType) as e:
        SymbolicFeynCalcCodeGenerator().generate(build_diagram_from_symbolic(sym))
    assert "'vector_axial_coupling'" in str(e.value)


@needs_wolfram
def test_tool_surfaces_unknown_type_as_an_error(tmp_path):
    """End to end: the agent gets a usable error, not a plausible number."""
    from tools.eda.compute_rate_tools import ComputeDecayRate
    out = ComputeDecayRate(
        diagram={"topology": "decay_1to2", "initial": [Vc("V")],
                 "final": [Fm("f"), Fm("fbar")],
                 "vertices": [{"type": "made_up_vertex", "coupling": "g"}]},
        values={"g": 0.47, "mV": 2150., "mf": 640., "mfbar": 640.},
        base_directory=str(tmp_path), wolframscript_path=_wolframscript(),
    )._run()
    assert "Unknown Vertex Type" in out
    assert "vector-axial" in out
    assert "width_GeV" not in out


@needs_wolfram
def test_distinct_vertex_types_give_distinct_widths(tmp_path):
    """The regression this whole change exists to prevent: before 2.9.1 these
    three all returned 11.9159..., the plain-vector width."""
    from tools.eda.compute_rate_tools import ComputeDecayRate

    def width(vtype, coupling):
        out = json.loads(ComputeDecayRate(
            diagram={"topology": "decay_1to2", "initial": [Vc("V")],
                     "final": [Fm("f"), Fm("fbar")],
                     "vertices": [{"type": vtype, "coupling": coupling}]},
            values={"g": 0.47, "gV": 0.52, "gA": 0.36,
                    "mV": 2150., "mf": 640., "mfbar": 640.},
            assume_real_couplings=True, base_directory=str(tmp_path),
            wolframscript_path=_wolframscript(),
        )._run())
        assert out["status"] == "ok", out
        return out["width_GeV"]

    widths = {
        "vector": width("vector", "g"),
        "axial-vector": width("axial-vector", "g"),
        "left-handed": width("left-handed", "g"),
    }
    assert widths["vector"] == pytest.approx(11.915957872612209, rel=1e-9)
    assert len(set(round(w, 9) for w in widths.values())) == 3, widths


def test_unknown_spin_configuration_raises():
    """Previously emitted `amp = I (g)` behind a comment -- a bare constant
    amplitude that integrates to a plausible number and reports success."""
    gen = FeynCalcCodeGenerator()
    with pytest.raises(UnknownVertexType) as e:
        gen._amplitude_decay_no_prop(
            build_diagram_from_symbolic(parse_symbolic_diagram({
                "topology": "decay_1to2",
                "initial": [{"label": "X", "spin": 1.5}],
                "final": [Fm("a"), Fm("b")],
                "vertices": [{"type": "vector", "coupling": "g"}],
            })),
            {"initial_0": "p", "final_0": "p1", "final_1": "p2"},
        )
    assert "spin configuration" in str(e.value)
