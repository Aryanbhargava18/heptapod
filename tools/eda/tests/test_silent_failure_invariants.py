"""
# test_silent_failure_invariants.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Three CAUSE-INDEPENDENT invariants, so the next instance of this bug class is
loud even though nobody knows yet what it will be.

THE PATTERN THESE EXIST FOR. The same failure shape has now been fixed three
times, each time only in the branch where it happened to be found:

    2.9.1    unknown vertex type / unknown spin config  -> silent   (fermionic)
    2.15.2   any dict coupling read as CHIRAL           -> silent   (fermionic)
    2.16.1   dict coupling emitted as a Python repr     -> silent   (bosonic)
             a declared `mass` discarded for m<label>   -> silent   (all)

Every one is: input accepted, silently mishandled, `status: ok`, and an agent
that cannot attribute the failure. 2.15.2 fixed the fermionic half of the dict
problem and left the bosonic half broken for a month, because the fix was
local to the dispatch it was found in. 256 unit tests did not see it; one
benchmark run did -- five bosonic tasks scoring 0.04-0.31 while the five
fermionic ones scored 1.00.

So these three tests do not assert anything about vertices, couplings or
masses. They assert properties of the tool's OUTPUT and of the EMITTED SCRIPT,
which hold whatever the cause, and would each have caught the bugs above
without anyone having to suspect them first.

  1. A numberless success is impossible. If `values` were supplied and no
     number came back, that is an error status -- not `ok` with a warning.
  2. No Python repr may reach the emitted Wolfram, from any branch.
  3. A `values` key that binds to nothing is reported.

WHY 1 IS THE LOAD-BEARING ONE. It needs no knowledge of the cause at all. An
agent that gets an error retries with corrected input; an agent that gets `ok`
with no number abandons the tool and hand-derives -- which is exactly how a
tool bug became a physics answer 2.000x off ground truth on decay_S_to_SS.
"""

import json
import math
import os
import shutil
from pathlib import Path

import pytest

from tools.eda.feyncalc_codegen import (
    FeynCalcCodeGenerator, SymbolicFeynCalcCodeGenerator, EmittedCodeLeak,
)
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)


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

V_TO_FF = {"topology": "decay_1to2",
           "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
           "final": [{"label": "f", "spin": 0.5, "mass": "mf"},
                     {"label": "fbar", "spin": 0.5, "mass": "mf"}],
           "vertices": [{"type": "vector", "coupling": {"g": "g"}}]}


def _run(**kw):
    from tools.eda.compute_rate_tools import ComputeDecayRate
    kw.setdefault("diagram", V_TO_FF)
    kw.setdefault("assume_real_couplings", True)
    kw.setdefault("wolframscript_path", _wolframscript())
    return json.loads(ComputeDecayRate(**kw)._run())


# ===========================================================================
# INVARIANT 1 — a numberless success is impossible.
# ===========================================================================

@needs_wolfram
def test_values_supplied_but_no_number_is_an_error_not_ok(tmp_path):
    """`values` given, one symbol unbound. This used to be `status: ok` with a
    warning, which is how every instance of this bug class got past the
    agent."""
    out = _run(values={"g": 0.47, "mV": 2150},      # mf missing
               base_directory=str(tmp_path))
    assert out["status"] == "error", out
    assert "width_GeV" not in out
    assert "mf" in json.dumps(out)


@needs_wolfram
def test_the_error_keeps_the_closed_form(tmp_path):
    """The formula is still a real deliverable -- the rubric grades it -- so
    failing the numeric step must not throw it away. This is why the error is
    a JSON payload with `status: error` rather than a bare error string."""
    out = _run(values={"g": 0.47, "mV": 2150},
               base_directory=str(tmp_path))
    assert out["status"] == "error"
    assert out.get("width_formula"), out
    assert out.get("symbols"), out


@needs_wolfram
def test_formula_only_mode_is_still_a_success(tmp_path):
    """No `values` means the caller asked for the closed form alone. That is
    not a failure and must not be reported as one."""
    out = _run(base_directory=str(tmp_path))
    assert out["status"] == "ok", out
    assert "width_GeV" not in out
    assert out.get("width_formula")


@needs_wolfram
def test_a_fully_bound_call_is_still_ok(tmp_path):
    out = _run(values={"g": 0.47, "mV": 2150, "mf": 640},
               base_directory=str(tmp_path))
    assert out["status"] == "ok", out
    g, M, m = 0.47, 2150.0, 640.0
    x = (m / M) ** 2
    expect = g ** 2 * M / (12 * math.pi) * (1 + 2 * x) * math.sqrt(1 - 4 * x)
    assert out["width_GeV"] == pytest.approx(expect, rel=1e-9)


@needs_wolfram
def test_an_empty_values_dict_is_formula_only_not_an_error(tmp_path):
    """`values={}` is indistinguishable from omitting it: the caller supplied
    no numbers, so no number is expected."""
    out = _run(values={}, base_directory=str(tmp_path))
    assert out["status"] == "ok", out


# ===========================================================================
# INVARIANT 2 — no Python repr may reach the emitted Wolfram.
#
# `{'g': 'g'}` reached Mathematica as a literal dict repr and the script failed
# to PARSE, so nothing came back at all. One check at the single point every
# script is assembled catches that from any branch, including branches nobody
# has looked at.
# ===========================================================================

def test_assembling_a_script_with_a_dict_repr_raises():
    gen = FeynCalcCodeGenerator()
    with pytest.raises(EmittedCodeLeak):
        gen._assemble_script(["amp = I ({'g': 'g'}) MTD[mu1, mu2];"])


def test_the_leak_error_shows_the_offending_line():
    gen = FeynCalcCodeGenerator()
    with pytest.raises(EmittedCodeLeak) as e:
        gen._assemble_script(["(* fine *)", "amp = I ({'mu': 'mu'});"])
    assert "{'mu': 'mu'}" in str(e.value) or "mu" in str(e.value)


def test_a_list_repr_also_raises():
    """`['a', 'b']` is the same mistake with a different container."""
    gen = FeynCalcCodeGenerator()
    with pytest.raises(EmittedCodeLeak):
        gen._assemble_script(["amp = I (['gL', 'gR']);"])


def test_legitimate_wolfram_braces_are_not_flagged():
    """Wolfram uses braces for lists and rules constantly -- a substitution
    list is the whole numeric-evaluation step. The check must key off the
    Python-repr signature, not off braces."""
    gen = FeynCalcCodeGenerator()
    code = gen._assemble_script([
        "widthNum = N[width /. {g -> 0.47, mV -> 2150, mf -> 640}];",
        "ampCC = ampCC /. {g -> Conjugate[g]};",
        'Print["SYMBOLIC_RESULT[width]: ", width];',
        "Print[Style[\"x\", {Text, Italic}]];",
    ])
    assert "widthNum" in code


@pytest.mark.parametrize("spec", [
    V_TO_FF,
    {"topology": "decay_1to2",
     "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
     "final": [{"label": "V1", "spin": 1, "mass": "m1"},
               {"label": "V2", "spin": 1, "mass": "m2"}],
     "vertices": [{"type": "svv", "coupling": {"g": "g"}}]},
    {"topology": "decay_1to2",
     "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
     "final": [{"label": "c1", "spin": 0, "mass": "m1"},
               {"label": "c2", "spin": 0, "mass": "m2"}],
     "vertices": [{"type": "scalar", "coupling": {"mu": "mu"}}]},
    {"topology": "decay_1to2",
     "initial": [{"label": "Zp", "spin": 1, "mass": "mZp"}],
     "final": [{"label": "Wp", "spin": 1, "mass": "mW"},
               {"label": "Wm", "spin": 1, "mass": "mW"}],
     "vertices": [{"type": "vvv", "coupling": {"g": "g"}}]},
    {"topology": "decay_1to2",
     "initial": [{"label": "V", "spin": 1, "mass": "mV"}],
     "final": [{"label": "S1", "spin": 0, "mass": "m1"},
               {"label": "S2", "spin": 0, "mass": "m2"}],
     "vertices": [{"type": "ssv", "coupling": {"g": "g"}}]},
])
def test_no_archetype_emits_a_repr(spec):
    """Generation must SUCCEED for each -- the guard raising here would mean a
    branch is still leaking."""
    d = build_diagram_from_symbolic(parse_symbolic_diagram(spec))
    code = SymbolicFeynCalcCodeGenerator().generate(d).code
    assert code
    assert "{'" not in code and "['" not in code


# ===========================================================================
# INVARIANT 3 — a `values` key that binds to nothing is reported.
#
# Stated as a warning, not an error: a key binding nothing is suspicious but
# not provably wrong (a caller may reuse one dict across several calls), while
# the harmful case -- nothing bound AND no number -- is already invariant 1.
# ===========================================================================

from tools.eda.compute_rate_tools import _unbound_value_keys


def test_a_key_matching_no_symbol_is_reported():
    assert _unbound_value_keys({"g": 1, "mS": 2, "m1": 3},
                               ["g", "mphi", "mV1"]) == ["m1", "mS"]


def test_keys_that_all_bind_report_nothing():
    assert _unbound_value_keys({"g": 1, "mV": 2}, ["g", "mV"]) == []


def test_the_tool_injected_s_is_not_reported(tmp_path):
    """ComputeCrossSection adds `s = sqrt_s^2` to `values` itself. Flagging
    the tool's own injection as a caller mistake would be noise -- and the
    kind of noise that trains an agent to ignore the field."""
    assert _unbound_value_keys({"s": 4.0, "g": 1}, ["g"]) == []


def test_no_symbols_reported_means_nothing_to_compare():
    """Formula-only or a failed script: invariant 1 owns that case, and
    claiming every key is unbound here would be misleading."""
    assert _unbound_value_keys({"g": 1}, []) == []


def test_empty_values_is_not_an_unbound_report():
    assert _unbound_value_keys({}, ["g", "mV"]) == []
    assert _unbound_value_keys(None, ["g"]) == []


@needs_wolfram
def test_an_unbound_key_is_warned_about_but_still_succeeds(tmp_path):
    """A number came back, so the call worked; the stray key is still worth
    saying out loud."""
    out = _run(values={"g": 0.47, "mV": 2150, "mf": 640, "mTypo": 1.0},
               base_directory=str(tmp_path))
    assert out["status"] == "ok", out
    assert isinstance(out.get("width_GeV"), float)
    assert any("mTypo" in w for w in out.get("warnings", [])), out


# ===========================================================================
# A symbolic PROPAGATOR mass must not crash.
#
# `_is_massive_particle` has guarded string masses since 2.9.x
# ("a named mass means massive"); `_is_massive_propagator` never got the same
# guard, so `prop.mass > 0` raised
#
#     TypeError: '>' not supported between instances of 'str' and 'int'
#
# for any diagram declaring one. That predates this release -- found while
# checking whether the 2->2 path shared the bosonic coupling leak -- and it
# means the xsec_* tasks cannot name their propagator masses at all. Not the
# silent class (a TypeError is loud), but an opaque crash where the particle
# path has a documented default.
#
# NOTE the asymmetry that remains, deliberately out of scope here: a declared
# propagator mass makes the propagator MASSIVE but does not yet NAME its
# symbol, which stays mProp<i>. So `values` keyed to the declared name binds
# nothing -- and is now a loud error via invariant 1, with mProp<i> listed in
# `symbols`, rather than a silent one. Naming propagator masses is a separate
# change against the 2->2 benchmarks.
# ===========================================================================

SCATTER_2TO2 = {
    "topology": "scattering_2to2",
    "initial": [{"label": "S1", "spin": 0, "mass": "m1"},
                {"label": "S2", "spin": 0, "mass": "m2"}],
    "final": [{"label": "S3", "spin": 0, "mass": "m3"},
              {"label": "S4", "spin": 0, "mass": "m4"}],
    "propagators": [{"label": "phi", "spin": 0, "mass": "mphi"}],
    "vertices": [{"type": "scalar", "coupling": {"mu": "mu"}},
                 {"type": "scalar", "coupling": {"mu": "mu"}}],
}


def test_a_symbolic_propagator_mass_does_not_crash():
    d = build_diagram_from_symbolic(parse_symbolic_diagram(SCATTER_2TO2))
    code = SymbolicFeynCalcCodeGenerator().generate(d).code
    assert code


def test_a_symbolic_propagator_mass_means_massive():
    """Same rule the particle path has documented for releases: a NAME means
    massive. A massless propagator is declared with mass 0 or massive=False."""
    from tools.nda.symbolic_diagram import _is_massive_propagator
    sym = parse_symbolic_diagram(SCATTER_2TO2)
    assert _is_massive_propagator(sym.propagators[0])


def test_a_zero_propagator_mass_is_still_massless():
    from tools.nda.symbolic_diagram import _is_massive_propagator
    spec = dict(SCATTER_2TO2)
    spec["propagators"] = [{"label": "A", "spin": 1, "mass": 0}]
    sym = parse_symbolic_diagram(spec)
    assert not _is_massive_propagator(sym.propagators[0])


def test_an_explicit_massive_flag_still_wins_on_a_propagator():
    from tools.nda.symbolic_diagram import _is_massive_propagator
    spec = dict(SCATTER_2TO2)
    spec["propagators"] = [{"label": "A", "spin": 1, "mass": "mA",
                            "massive": False}]
    sym = parse_symbolic_diagram(spec)
    assert not _is_massive_propagator(sym.propagators[0])


def test_the_2to2_bosonic_path_does_not_leak_a_repr():
    """The reason this path was examined at all: it shares `_coupling_value`
    with the 1->2 bosonic branches that were emitting Python reprs."""
    d = build_diagram_from_symbolic(parse_symbolic_diagram(SCATTER_2TO2))
    code = SymbolicFeynCalcCodeGenerator().generate(d).code
    assert "{'" not in code and "['" not in code
