"""
# compute_rate_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

ComputeDecayRate / ComputeCrossSection — one call, one observable.

WHY THESE EXIST. ComputeSymbolicAmplitude writes a .wl script and stops;
getting an answer out of it means a second RunWolframScript call, then
usually a third step to turn Mathematica InputForm into something a
caller can evaluate. Every one of those hops is a place for an agent to
lose the thread, and the recovery path is reading raw Mathematica errors
-- which is most of what the FeynCalc reference corpus exists to explain.

These tools close the loop: hand them a diagram (and values, if you want
a number) and they generate, execute, and parse, returning the closed-form
observable AND its numeric value together. The generate-then-run pair is
still there and still the right tool when a script needs hand-editing
before it runs -- a color factor FeynCalc will not supply on its own, say.

They are a facade, not a reimplementation: the same
SymbolicFeynCalcCodeGenerator, the same WolframRunner, the same result
markers. The formula is derived symbolically and the substitution happens
last, so asking for a number never costs you the closed form.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .feyncalc_codegen import (
    SymbolicFeynCalcCodeGenerator, ProcessType, UnknownVertexType,
)
from .symbolic_to_python import mathematica_to_sympy, extract_variables
from tools.nda.symbolic_diagram import (
    parse_symbolic_diagram, build_diagram_from_symbolic,
)
from tools.wolfram.wolfram_runner import WolframRunner
from tools.logging.findings import append_finding


_DECAY_TYPES = (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP)


def _sympy_str(mma_expr: str) -> Optional[str]:
    """Mathematica InputForm -> a Python/sympy-parseable single-line string.

    Returns None (rather than raising) when the expression will not parse:
    a formula we cannot convert is worth reporting as absent, not worth
    failing an otherwise-good calculation over. The Mathematica form is
    always returned alongside, so nothing is lost.
    """
    try:
        return str(mathematica_to_sympy(mma_expr))
    except Exception:
        return None


class _ComputeRateBase(BaseTool):
    """Shared machinery for the one-call observable tools.

    Subclasses declare their own runtime fields and supply ``_observable``
    (the codegen's variable name) plus the process types they accept.
    """

    # Subclasses set these.
    _observable: str = "width"
    _accepted: Tuple[ProcessType, ...] = _DECAY_TYPES

    def _resolve_diagrams(self, specs: List[Dict[str, Any]], channel):
        """Parse + validate + build every spec. Returns (syms, built, channels)
        or a formatted error string."""
        syms, built, channels = [], [], []
        for idx, spec in enumerate(specs):
            try:
                sym = parse_symbolic_diagram(spec)
            except Exception as e:
                return self.format_error(
                    error="Diagram Parse Error", reason=f"diagram {idx}: {e}",
                )
            ok, warnings = sym.validate()
            if not ok:
                return self.format_error(
                    error="Diagram Validation Failed",
                    reason=f"diagram {idx}: " + "; ".join(warnings),
                )
            try:
                built.append(build_diagram_from_symbolic(sym))
            except Exception as e:
                return self.format_error(
                    error="Diagram Build Error",
                    reason=f"diagram {idx}: {e}",
                    suggestion=(
                        "Every particle and propagator needs a 'spin'. Masses "
                        "may be omitted — they stay symbolic."
                    ),
                )
            syms.append(sym)
            channels.append(spec.get("channel", channel))
        return syms, built, channels

    def _generate_run_parse(
        self,
        *,
        specs: List[Dict[str, Any]],
        channel,
        relative_signs,
        values: Optional[Dict[str, float]],
        assume_real_couplings: bool,
        simplifications,
        sqrt_s: Optional[float],
        script_name: Optional[str],
        timeout: int,
        prefix: str,
    ) -> str:
        resolved = self._resolve_diagrams(specs, channel)
        if isinstance(resolved, str):
            return resolved          # already a formatted error
        syms, built, channels = resolved
        sym = syms[0]

        # A cross section comes back as a function of s, so a numeric answer
        # needs s bound as well as the masses and couplings. sqrt_s is the
        # natural way for a caller to say that, so fold it in here rather than
        # making them pass s = sqrt_s^2 by hand. An explicit `s` in `values`
        # wins, on the principle that the more specific input should.
        if self._observable == "sigma" and sqrt_s is not None:
            values = dict(values or {})
            values.setdefault("s", float(sqrt_s) ** 2)

        generator = SymbolicFeynCalcCodeGenerator(
            assume_real_couplings=assume_real_couplings,
            simplifications=simplifications,
            channel=channel,
            values=values,
        )
        # An unrecognized vertex type is refused, not defaulted (see
        # UnknownVertexType). Surface it as a normal tool error carrying the
        # valid vocabulary, so a caller that guessed a type string learns the
        # right one from the response instead of receiving a plausible width
        # computed from a vertex they did not ask for.
        try:
            if len(specs) > 1:
                gen = generator.generate_sum(
                    [(d, ch or "s") for d, ch in zip(built, channels)],
                    sqrt_s=sqrt_s, relative_signs=relative_signs,
                )
            else:
                gen = generator.generate(built[0], sqrt_s=sqrt_s)
        except UnknownVertexType as e:
            return self.format_error(
                error="Unknown Vertex Type",
                reason=str(e),
                suggestion=(
                    "Set the vertex 'type' to one of the listed values. The "
                    "type carries the physics, so a near-miss is not a "
                    "spelling detail: 'pseudoscalar' and 'vector' are "
                    "different vertices, and a type valid for one bilinear "
                    "(SFF) is not valid for the other (VFF)."
                ),
            )

        if gen.process_type == ProcessType.UNSUPPORTED:
            return json.dumps({
                "status": "unsupported",
                "process_type": "UNSUPPORTED",
                "warnings": gen.warnings,
                "suggestion": (
                    "Automatic code generation does not cover this topology. "
                    "Write the FeynCalc by hand and run it with "
                    "RunWolframScript."
                ),
            }, indent=2)

        if gen.process_type not in self._accepted:
            wanted = " or ".join(p.name for p in self._accepted)
            other = ("ComputeCrossSection" if self._observable == "width"
                     else "ComputeDecayRate")
            return self.format_error(
                error="Wrong Process Type",
                reason=(
                    f"this diagram is {gen.process_type.name}, but this tool "
                    f"handles {wanted}"
                ),
                suggestion=f"Use {other} for this process.",
            )

        if not gen.code:
            return json.dumps({
                "status": "error",
                "process_type": gen.process_type.name,
                "warnings": gen.warnings,
            }, indent=2)

        # --- write the script (kept as the reproducibility artifact) ---
        if script_name:
            name = script_name if script_name.endswith(".wl") else script_name + ".wl"
        else:
            labels = [p.label for p in sym.initial] + [p.label for p in sym.final]
            safe = "_".join(l.replace("+", "p").replace("-", "m") for l in labels)
            name = f"{prefix}_{safe}.wl"

        scripts_dir = Path(self.base_directory) / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = str(scripts_dir / name)
        with open(script_path, "w") as f:
            f.write(gen.code)

        # --- execute ---
        runner = WolframRunner(
            wolframscript_path=self.wolframscript_path, timeout_sec=timeout,
        )
        result = runner.run_script(script_path, working_dir=self.base_directory)

        process_str = " → ".join([
            " ".join(p.label for p in sym.initial),
            " ".join(p.label for p in sym.final),
        ])

        if not result.success:
            # Hand back the script path regardless: a failed run that leaves a
            # readable .wl behind is debuggable, and the agent can re-run or
            # edit it with RunWolframScript rather than starting over.
            return json.dumps({
                "status": "error",
                "process": process_str,
                "process_type": gen.process_type.name,
                "script_path": script_path,
                "reason": "wolframscript did not complete successfully",
                "stderr": (result.stderr or "")[-2000:],
                "stdout": (result.stdout or "")[-2000:],
                "suggestion": (
                    "Inspect or edit the script and re-run it with "
                    "RunWolframScript."
                ),
            }, indent=2)

        parsed = result.parsed_results or {}
        symbolic = parsed.get("symbolic", {})
        numerical = parsed.get("numerical", {})
        latex = parsed.get("latex", {})

        obs = self._observable
        # Prefer the simplified variant when simplifications were requested:
        # it is the same expression, tidier.
        key = f"{obs}_simplified" if f"{obs}_simplified" in symbolic else obs
        mma = symbolic.get(key)

        output: Dict[str, Any] = {
            "status": "ok",
            "process": process_str,
            "process_type": gen.process_type.name,
            "script_path": script_path,
        }
        if result.results_path:
            output["results_path"] = result.results_path

        if mma:
            output[f"{obs}_mathematica"] = mma
            formula = _sympy_str(mma)
            if formula:
                output[f"{obs}_formula"] = formula
            else:
                output["warnings"] = output.get("warnings", []) + [
                    f"could not convert {key} to a sympy-parseable string; "
                    f"the Mathematica form is under {obs}_mathematica"
                ]
            # Which symbols a caller still has to supply to get a number.
            try:
                output["symbols"] = extract_variables(mma)
            except Exception:
                pass
        if latex.get(key):
            output[f"{obs}_latex"] = latex[key]
        if symbolic.get("ampSq"):
            output["amp_squared_mathematica"] = symbolic["ampSq"]

        # Numeric results, whichever units the script emitted. A marker whose
        # value did not parse as a float is an expression Mathematica could not
        # reduce -- almost always a symbol with no entry in `values` -- so it is
        # NOT a number and must not be returned as one. Reporting the leftover
        # symbols is the actionable part: `symbols` says what was needed,
        # `values` says what was given, and the difference is the fix.
        candidates = {k: v for k, v in numerical.items()
                      if k.startswith("sigma") or k.startswith("width")}
        numbers = {k: v for k, v in candidates.items()
                   if isinstance(v, (int, float))}
        unresolved = sorted(set(candidates) - set(numbers))
        output.update(numbers)

        # INVARIANT: a numberless success is impossible. If `values` were
        # supplied, the caller asked for a number; not producing one is a
        # FAILED call, whatever the cause. It used to be `status: ok` with a
        # warning, and that is how every instance of this bug class got past
        # the agent -- it read `ok`, saw no number, and hand-derived instead
        # of retrying. The closed form stays in the payload, because it is a
        # real deliverable and the numeric step failing does not invalidate
        # it.
        if unresolved or (values and not numbers):
            detail = _unresolved_warning(
                unresolved, output.get("symbols", []), values,
                mass_symbols=_diagram_mass_symbols(built))
            output["status"] = "error"
            output["error"] = "No Numeric Result"
            output["reason"] = detail

        # INVARIANT: a `values` key that binds to nothing is reported. A
        # warning rather than an error -- reusing one dict across several
        # calls is legitimate -- while the harmful case (nothing bound AND no
        # number) is the error above.
        stray = _unbound_value_keys(values, output.get("symbols", []))
        if stray:
            output.setdefault("warnings", []).append(
                f"`values` keys that match no symbol in the formula: "
                f"{', '.join(stray)}. They had no effect. The formula needs "
                f"{', '.join(output.get('symbols', [])) or '(none)'}."
            )

        for extra in ("warnings",):
            if gen.warnings:
                output.setdefault(extra, []).extend(gen.warnings)
        # The conventions the formula was built with. Reported because a
        # wrong identical-particle factor is a factor of 2 and a wrong
        # initial-spin average is a factor of 3, and both were previously
        # visible only inside the generated script.
        if gen.symmetry_factor is not None:
            output["symmetry_factor"] = gen.symmetry_factor
        if gen.n_initial_spin_states is not None:
            output["n_initial_spin_states"] = gen.n_initial_spin_states
        if gen.vertex_structures:
            output["vertex_structures"] = gen.vertex_structures
        for d in (built or []):
            amb = _ambiguous_identity_warning(d)
            if amb:
                output.setdefault("warnings", []).append(amb)
                break

        if gen.channel:
            output["channel"] = gen.channel.name
        if gen.channels:
            output["channels"] = [c.name for c in gen.channels]
            output["n_diagrams_summed"] = len(gen.channels)

        try:
            entries = [f"Process: {process_str} ({gen.process_type.name})"]
            if output.get(f"{obs}_formula"):
                entries.append(f"{obs}: {output[f'{obs}_formula']}")
            for k, v in numbers.items():
                entries.append(f"{k} = {v}")
            entries.append(f"Script: {script_path}")
            append_finding(self.base_directory, f"{obs.title()}: {process_str}",
                           entries)
        except Exception:
            pass

        return json.dumps(output, indent=2)





#: `values` entries the TOOL injects, which are never a caller mistake.
#: ComputeCrossSection sets `s = sqrt_s^2` itself so callers need not square
#: it by hand; flagging that as a stray key would be noise, and noise in this
#: field is what teaches an agent to stop reading it.
_TOOL_INJECTED_VALUE_KEYS = frozenset({"s"})




def _is_conjugate_set(labels) -> bool:
    """True when these labels are a particle/antiparticle set.

    f and fbar have equal spin and equal mass and are NEVER the same particle,
    so there is nothing ambiguous about them and no warning to give -- advising
    one label there would apply a 1/2! that does not belong. An f fbar final
    state is two of the ten hepbench symbolic tasks, so this is the common
    case, not the corner.

    Two detectors, both deliberately biased toward saying "conjugate":

    - `_is_antiparticle`, the suffix heuristic the spinor ordering already
      relies on -- a set holding both a barred and an unbarred label.
    - one label being another plus a short conjugation suffix, which catches
      the spellings the first misses (`q`/`qb`). This is NOT added to
      `_ANTIPARTICLE_SUFFIXES`, because plenty of real particles end in "b"
      and the spinor ordering must not start guessing.

    THE BIAS IS THE POINT. In this function a false "conjugate" only
    suppresses a warning, while a false "ambiguous" hands the caller advice
    that is actively wrong -- merge the labels and pick up a 1/2! that does
    not belong. So where the two readings are close, stay quiet.
    """
    from tools.eda.fc_symbols import _is_antiparticle
    if len({_is_antiparticle(l) for l in labels}) > 1:
        return True
    _CONJ_SUFFIXES = ("bar", "b", "~", "c", "B", "Bar")
    for a in labels:
        for b in labels:
            if a == b or not b.startswith(a):
                continue
            if b[len(a):] in _CONJ_SUFFIXES:
                return True
    return False


def _ambiguous_identity_warning(diagram) -> str:
    """Warn when a final state LOOKS identical but is labelled as distinct.

    Identity is declared by the LABEL (see `compute_symmetry_factor`), and
    that is deliberate: same spin and same mass is a degenerate doublet as
    often as it is one particle written twice, so inferring identity from
    degeneracy would apply 1/2! to configurations that must not have it --
    wrong by two, in the other direction, and just as silent.

    But the caller who writes `chi1`/`chi2` for one particle gets symmetry
    factor 1 and a width out by exactly 2, which is what happened on hepbench
    decay_S_to_SS part (a). So the ambiguous case is named instead of guessed:
    equal spin, equal mass symbol, different labels, and no explicit
    `symmetry_factor` to settle it.
    """
    if getattr(diagram, "symmetry_factor", None) is not None:
        return ""                      # the caller has stated it; nothing to guess
    final = list(getattr(diagram, "final", []) or [])
    if len(final) < 2:
        return ""
    groups = {}
    for p in final:
        groups.setdefault((p.spin, str(p.mass)), []).append(p.label)
    for (_spin, _mass), labels in groups.items():
        named = [l for l in labels if l]
        if len(named) >= 2 and len(set(named)) > 1 \
                and not _is_conjugate_set(named):
            return (
                f"final-state particles {', '.join(named)} have the same spin "
                f"and the same mass symbol but different labels, so they are "
                f"treated as DISTINCT: symmetry factor 1, no 1/n! applied. "
                f"If they are the same particle, give them one label "
                f"(\"{named[0]}\" twice) and the factor follows. If they are "
                f"genuinely distinct but degenerate, this is correct -- set "
                f"`symmetry_factor` on the diagram to say so explicitly and "
                f"silence this."
            )
    return ""


def _unbound_value_keys(values, symbols) -> list:
    """`values` keys that match no free symbol of the formula, sorted.

    A key that binds to nothing had no effect, which is either a typo or the
    namespace split that this release exists to fix -- so it is worth saying
    out loud even when a number did come back.

    Returns [] when there are no symbols to compare against: that means
    formula-only mode or a script that produced nothing, and announcing that
    every key is unbound would be actively misleading. That case belongs to
    the numberless-success invariant instead.
    """
    if not values or not symbols:
        return []
    known = set(symbols)
    return sorted(k for k in values
                  if k not in known and k not in _TOOL_INJECTED_VALUE_KEYS)


def _diagram_mass_symbols(built) -> tuple:
    """Every mass symbol the built diagrams are written in.

    Passed to `_unresolved_warning` so it can tell a mass from a coupling by
    fact rather than by spelling -- see that function on why `mu` makes a
    "starts with m" test actively misleading.
    """
    from tools.eda.fc_symbols import _mass_symbol
    out = []
    for diagram in (built or []):
        for group in (diagram.initial, diagram.final):
            for idx, particle in enumerate(group):
                try:
                    sym = _mass_symbol(particle, idx)
                except Exception:
                    continue          # a bad declaration is reported elsewhere
                if sym not in out:
                    out.append(sym)
    return tuple(out)


def _unresolved_warning(unresolved, symbols, values, mass_symbols=()):
    """The message an agent reads when no number came back.

    IT MUST NAME BOTH SIDES. `symbols` is what the formula needs, `values` is
    what was supplied, and the difference is the fix -- so the difference is
    what gets printed. There used to be a second, vaguer branch ("compare the
    `symbols` field against the keys you passed") for the case where no width
    marker parsed at all, which is precisely the case where the agent has
    nothing bound and the least to go on. Both cases now route here.

    WHEN AN UNBOUND SYMBOL IS ONE OF THE DIAGRAM'S MASSES, say so at the
    source: that is the signature of a caller who named its masses in `values`
    without declaring them on the particles, and the actionable fix is the
    diagram's `mass` field, not the `values` dict.

    `mass_symbols` is passed in rather than guessed. A "starts with m"
    heuristic would call the coupling `mu` a mass -- and `mu` is the coupling
    in the archetypal scalar decay, L = -(mu/2) phi chi^2 -- sending the
    caller to the wrong field with full confidence.
    """
    given = sorted(values or {})
    missing = [s for s in (symbols or []) if s not in (values or {})]
    head = (f"no numeric value for {', '.join(unresolved)}: "
            if unresolved else "no numeric result came back: ")
    msg = (head + "the expression still contains free symbols after "
           f"substitution. The formula needs {', '.join(symbols) or '(none)'}"
           + (f"; `values` supplied {', '.join(given)}" if given
              else "; `values` supplied nothing")
           + (f". Unbound: {', '.join(missing)}" if missing else "."))
    unbound_masses = [m for m in missing if m in set(mass_symbols or ())]
    if unbound_masses:
        msg += (f" {', '.join(unbound_masses)} "
                + ("names" if len(unbound_masses) == 1 else "name")
                + " a mass: a particle's mass symbol is its `mass` field when"
                  ' you declare one ("mass": "mS"), and otherwise defaults to'
                  " m<label>. Either declare the masses you keyed `values` to,"
                  " or key `values` to the defaults.")
    return msg


class ComputeDecayRate(_ComputeRateBase):
    """
    Compute a tree-level partial decay width from a diagram — in one call.

    Generates the FeynCalc, runs it, and returns the closed-form width
    together with its numeric value. No follow-up RunWolframScript call is
    needed. The full chain the script performs:

        amplitude -> |M|^2 -> spin/polarization sums -> Dirac traces
        -> 2-body kinematics -> Gamma = |p|/(8 pi M^2) (1/n_init) |M|^2
           * colorFactor / symmetryFactor

    Input:
        diagram:  SymbolicDiagram dict (same format ComputeSymbolicAmplitude
                  takes: 'initial', 'final', 'vertices', optional
                  'propagators'; every particle needs a 'spin'). A
                  particle's 'mass' NAMES its mass symbol -- "mass": "mS"
                  puts mS in the formula, so `values` must be keyed to mS.
                  Omit 'mass' and the symbol defaults to m<label>: a particle
                  labelled "V1" gets mV1. A number pins the mass instead.
                  KEY `values` TO WHATEVER THE SYMBOL IS -- a key that matches
                  no symbol binds nothing and no number comes back.
        diagrams: several specs to add coherently, instead of `diagram`.
        values:   {symbol: number} for a numeric width. Omit for formula only.

    Output (JSON):
        width_formula        closed form, sympy-parseable ('**', 'sqrt', 'pi')
        width_mathematica    the same expression in Mathematica InputForm
        width_latex          TeXForm
        width_GeV, width_MeV present when `values` was supplied
        symbols              free symbols in the formula — what `values` needs
        amp_squared_mathematica, script_path, results_path

    Example — a massive vector to a Dirac pair via a V-A vertex. No 'mass' is
    declared, so the symbols DEFAULT to m<label>: mV, mf, mfbar. That is why
    `values` is keyed mV / mf, and why mfbar (label "fbar") must be given
    too::

        {
          "diagram": {
            "topology": "decay_1to2",
            "initial": [{"label": "V", "spin": 1}],
            "final":   [{"label": "f",    "spin": 0.5},
                        {"label": "fbar", "spin": 0.5}],
            "vertices": [{"type": "vector-axial",
                          "coupling": {"gV": "gV", "gA": "gA"}}]
          },
          "values": {"gV": 0.52, "gA": 0.36,
                     "mV": 2150, "mf": 640, "mfbar": 640}
        }

    Example — the same thing with the masses NAMED, which is what to do when
    the labels are not what you want the formula written in. Labels "V1"/"V2"
    would default to mV1/mV2; declaring 'mass' overrides that, and `values`
    follows the declared names::

        {
          "diagram": {
            "topology": "decay_1to2",
            "initial": [{"label": "phi", "spin": 0, "mass": "mS"}],
            "final":   [{"label": "V1", "spin": 1, "mass": "m1"},
                        {"label": "V2", "spin": 1, "mass": "m2"}],
            "vertices": [{"type": "svv", "coupling": {"g": "g"}}]
          },
          "values": {"g": 0.8, "mS": 950, "m1": 200, "m2": 300}
        }

    IDENTICAL PARTICLES AND THE PREFACTORS. Two final-state particles are
    identical IF AND ONLY IF THEY SHARE A LABEL. That is the whole rule, and
    it decides a factor of 2:

        "final": [{"label": "chi", ...}, {"label": "chi", ...}]
            -> one particle written twice. symmetry factor 2, i.e. the width
               carries 1/2!.
        "final": [{"label": "chi1", ...}, {"label": "chi2", ...}]
            -> two DISTINCT particles, even with the same mass. Symmetry
               factor 1. Equal mass is NOT read as identity, because a
               mass-degenerate doublet is a real configuration and applying
               1/2! to it would be wrong.

    So labelling one particle two ways silently halves the width. Write
    identical particles with one label, or set `symmetry_factor` on the
    diagram to state the factor outright. A final state with equal spin, equal
    mass and different labels returns a WARNING saying which reading was used.

    The vertex factor is a separate question the caller still owns. For
    L = -(mu/2) phi chi^2 the Feynman rule is -i mu, NOT -i mu/2: the 1/2 is
    cancelled by the two ways of contracting the identical fields. Passing
    "mu/2" here and letting the tool apply 1/2! counts the same factor twice.

    Both prefactors are REPORTED in the output -- `symmetry_factor` and
    `n_initial_spin_states` (2J+1 for the parent: 1 scalar, 2 fermion, 3
    massive vector) -- so the conventions the number was built with are
    visible without reading the generated script.

    `vertex_structures` reports which vertex was actually built, e.g.
    ["VFF"] for a vector plus two fermions, ["SFF"] scalar plus two fermions,
    ["SSS"], ["SSV"], ["SVV"], ["VVV"]. THE LEG SPINS DECIDE THIS, not the
    type name, so check it against the process you meant: three legs declared
    `spin: 1` give ["VVV"] -- a triple-gauge vertex -- even if you called the
    type "FFV". The list is deduplicated, so a diagram with two VFF vertices
    reports one entry.

    VERTEX TYPES. The `type` string carries the physics and is AUTHORITATIVE:
    it decides the Feynman rule, and `coupling` must supply exactly the names
    that rule takes. A type that is not listed, or a coupling that cannot
    satisfy the listed one, is REFUSED rather than guessed at.

    The rightmost column is each type's coupling contract:

      SFF (scalar/pseudoscalar -> fermion pair)
        "yukawa" / "scalar"          I g                            takes g
        "pseudoscalar"               I g gamma5                     takes g
        "chiral"                     I (gL P_L + gR P_R)            takes gL, gR
        "scalar-va"                  I (gS + gP gamma5)             takes gS, gP

      VFF (vector -> fermion pair)
        "vector" / "gauge-vector"    I g gamma^mu                   takes g
        "axial-vector"               I g gamma^mu gamma5            takes g
        "vector-axial" / "va"        I gamma^mu (gV - gA gamma5)    takes gV, gA
        "chiral" / "vector-chiral"   I gamma^mu (gL P_L + gR P_R)   takes gL, gR
        "left-handed"                I g gamma^mu P_L               takes g
        "right-handed"               I g gamma^mu P_R               takes g
        "tensor" / "dipole"          I g sigma^{mu nu} k_nu         takes g
        "tensor-chiral"              I (gL P_L+gR P_R) sigma k      takes gL, gR

    HOW TO WRITE `coupling`. For a type taking ONE name, pass a bare string --
    `"coupling": "g"` -- or a single-entry dict, `{"g": "gEff"}`, whose value
    is the symbol used. For a type taking SEVERAL, pass a dict with exactly
    those keys: `{"gV": "gV_e", "gA": "gA_e"}`. A bare string also works there
    as shorthand and derives the suffixes, so `"y"` under a (gV, gA) contract
    means yV, yA.

    Anything else raises, naming what was expected. In particular a two-entry
    dict on a one-coupling type is ambiguous rather than "use the first", and
    the wrong keys on a two-coupling type is an error rather than a silent
    fall back to the contract's own names -- which would return a formula in
    symbols you never supplied, so `values` would match nothing and the
    number would simply go missing.

      Bosonic (SSS, SSV, SVV, VVV) dispatches on the SPIN configuration, so
      any type string is accepted there -- "ssv", "svv", "vvv" read best.
        [0,0,0] SSS    I g
        [0,0,1] SSV    I g eps.(p1-p2)
        [0,1,1] SVV    I g eps1.eps2
        [1,1,1] VVV    Yang-Mills triple gauge, ALL-INCOMING momenta

    TWO THINGS THE TOOL CANNOT INFER FOR YOU:

    1. IDENTICAL FIELDS IN THE LAGRANGIAN DOUBLE THE VERTEX. For
       ``L = g S V_mu V^mu`` the Feynman rule is ``2 i g eta_{mu nu}``, because
       both V fields can contract with either leg. The vertex table implements
       ``i g``, the rule for ``L = g S V1.V2`` with distinct fields, so pass
       ``coupling: "2*g"`` when the two legs are the same field. Note the
       final-state symmetry factor 1/2 IS applied automatically -- only the
       vertex factor is yours. Getting this wrong costs a factor of 4 in the
       width, which looks like a physics error and is not.

       The scalar case runs the other way and needs nothing extra:
       ``L = -(mu/2) phi chi^2`` already carries the 1/2 that the two
       contractions cancel, so its rule is ``-i mu`` and you pass ``"mu"``.
       Passing ``"mu/2"`` there costs the same factor of 4. In both cases
       `coupling` is the Feynman-rule coefficient, not the Lagrangian's.

    2. A MASSLESS MEDIATOR MUST BE DECLARED IN THE DIAGRAM, not substituted
       later. The spin-1 propagator numerator is chosen when the code is
       GENERATED, from the propagator's declared mass: leave it symbolic and
       you get the massive form ``-g_{mu nu} + q_mu q_nu / m^2``, which divides
       by zero if you then pass ``mProp0 -> 0`` in `values`. Write
       ``{"label": "V", "spin": 1, "mass": 0}`` and the Feynman-gauge
       ``-g_{mu nu}`` is emitted instead. (Same for ComputeCrossSection, where
       an s-channel massless vector is the common case.)
    """

    diagram: Optional[Dict[str, Any]] = RuntimeField(
        default=None,
        description=(
            "SymbolicDiagram spec: keys 'topology', 'initial', 'final', "
            "'vertices', optional 'propagators'. Every particle needs 'spin'; "
            "omit 'mass' to keep it symbolic. Provide this OR 'diagrams'."
        ),
    )
    diagrams: Optional[List[Dict[str, Any]]] = RuntimeField(
        default=None,
        description=(
            "Several diagram specs to add coherently before squaring. "
            "Provide this OR 'diagram'."
        ),
    )
    values: Optional[Dict[str, float]] = RuntimeField(
        default=None,
        description=(
            "Numeric values keyed by symbol name, e.g. "
            "{'gV': 0.52, 'mV': 2150, 'mf': 640}. Supply to get width_GeV "
            "alongside the formula; omit for the closed form alone. Check the "
            "returned 'symbols' list if a number does not come back."
        ),
    )
    relative_signs: Optional[List[int]] = RuntimeField(
        default=None,
        description="Relative signs (+1/-1) per diagram when summing 'diagrams'.",
    )
    assume_real_couplings: bool = RuntimeField(
        default=False,
        description="Treat couplings as real, dropping Conjugate[] in |M|^2.",
    )
    simplifications: Optional[Dict[str, Any]] = RuntimeField(
        default=None,
        description=(
            "Post-computation steps applied in the script: 'substitutions', "
            "'limit', 'series', 'assumptions', 'simplify'."
        ),
    )
    script_name: Optional[str] = RuntimeField(
        default=None, description="Name for the saved .wl script (no extension).",
    )
    timeout: int = RuntimeField(
        default=300,
        description="Seconds to allow wolframscript. Traces can be slow.",
    )

    base_directory: str = StateField(description="Working directory")
    wolframscript_path: str = StateField(description="Path to wolframscript")

    _observable = "width"
    _accepted = _DECAY_TYPES

    def _run(self) -> str:
        if not self.diagram and not self.diagrams:
            return self.format_error(
                error="Missing Parameter",
                reason="one of 'diagram' or 'diagrams' is required",
            )
        if self.diagram and self.diagrams:
            return self.format_error(
                error="Ambiguous Input",
                reason="'diagram' and 'diagrams' are mutually exclusive",
            )
        return self._generate_run_parse(
            specs=self.diagrams if self.diagrams else [self.diagram],
            channel=None,
            relative_signs=self.relative_signs,
            values=self.values,
            assume_real_couplings=self.assume_real_couplings,
            simplifications=self.simplifications,
            sqrt_s=None,
            script_name=self.script_name,
            timeout=self.timeout,
            prefix="width",
        )


class ComputeCrossSection(_ComputeRateBase):
    """
    Compute a tree-level 2->2 cross section from a diagram — in one call.

    Generates the FeynCalc, runs it, and returns sigma in closed form (in s)
    together with its numeric value at a chosen sqrt(s). No follow-up
    RunWolframScript call is needed.

    Input:
        diagram:  SymbolicDiagram dict, as ComputeDecayRate takes.
        diagrams: several specs to add coherently (s/t/u interference).
        channel:  's', 't', 'u' or 'contact' for an exchange diagram. If
                  omitted, s-channel is assumed and the assumption is
                  reported in 'warnings' rather than hidden.
        sqrt_s:   centre-of-mass energy in GeV, for numeric evaluation.
        values:   {symbol: number} for masses and couplings.

    Output (JSON):
        sigma_formula        closed form in s, sympy-parseable
        sigma_mathematica    Mathematica InputForm
        sigma_latex          TeXForm
        sigma_GeV2, sigma_pb, sigma_fb   present when sqrt_s (and the needed
                             `values`) were supplied
        symbols              free symbols in the formula
        channel / channels / n_diagrams_summed
        amp_squared_mathematica, script_path, results_path

    A MASSLESS s-CHANNEL MEDIATOR MUST BE DECLARED IN THE DIAGRAM:
    ``{"label": "V", "spin": 1, "mass": 0}``. The spin-1 propagator numerator
    is fixed when the code is generated, from the declared mass -- leaving it
    symbolic emits the massive form ``-g_{mu nu} + q_mu q_nu / m^2``, and
    substituting ``mProp0 -> 0`` afterwards divides by zero. The symbolic
    sigma still comes back looking fine; it is the NUMBER that goes missing.

    `coupling` IS THE FEYNMAN-RULE COEFFICIENT, not the Lagrangian's: what
    multiplies `i` in the vertex factor, with the factor from repeated fields
    already folded in. The emitted SVV vertex carries the distinct-field
    normalisation (`L = g S V^mu W_mu`), so identical vectors
    (`L = g S V^mu V_mu`) need "2*g", while `L = -(mu/2) phi chi^2` needs just
    "mu" -- its 1/2 already cancels the two contractions. The
    identical-particle symmetry factor is separate and applied for you.

    IDENTICAL PARTICLES AND THE PREFACTORS. Two final-state particles are
    identical iff they SHARE A LABEL; equal mass alone is not read as identity
    (a degenerate doublet is a real configuration). Sharing a label applies
    the 1/n! identical-particle factor, distinct labels do not, and a final
    state with equal spin, equal mass and different labels returns a warning
    naming which reading was used. `symmetry_factor` on the diagram overrides
    the derivation. `symmetry_factor` and `n_initial_spin_states` are reported
    in the output so the conventions behind the number are visible.

    """

    diagram: Optional[Dict[str, Any]] = RuntimeField(
        default=None,
        description=(
            "SymbolicDiagram spec. Every particle needs 'spin'; omit 'mass' "
            "to keep it symbolic. Provide this OR 'diagrams'."
        ),
    )
    diagrams: Optional[List[Dict[str, Any]]] = RuntimeField(
        default=None,
        description=(
            "Several diagram specs to add coherently before squaring — use "
            "for s/t/u interference. Provide this OR 'diagram'."
        ),
    )
    channel: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Exchange channel for a 2->2 diagram: 's', 't', 'u' or 'contact'. "
            "Omitted means s-channel, reported as an assumption in 'warnings'."
        ),
    )
    sqrt_s: Optional[float] = RuntimeField(
        default=None,
        description="Centre-of-mass energy in GeV, for numeric evaluation.",
    )
    values: Optional[Dict[str, float]] = RuntimeField(
        default=None,
        description=(
            "Numeric values keyed by symbol name, for masses and couplings. "
            "Check the returned 'symbols' list if no number comes back."
        ),
    )
    relative_signs: Optional[List[int]] = RuntimeField(
        default=None,
        description="Relative signs (+1/-1) per diagram when summing 'diagrams'.",
    )
    assume_real_couplings: bool = RuntimeField(
        default=False,
        description="Treat couplings as real, dropping Conjugate[] in |M|^2.",
    )
    simplifications: Optional[Dict[str, Any]] = RuntimeField(
        default=None,
        description=(
            "Post-computation steps applied in the script: 'substitutions', "
            "'limit', 'series', 'assumptions', 'simplify'."
        ),
    )
    script_name: Optional[str] = RuntimeField(
        default=None, description="Name for the saved .wl script (no extension).",
    )
    timeout: int = RuntimeField(
        default=300,
        description="Seconds to allow wolframscript. Traces can be slow.",
    )

    base_directory: str = StateField(description="Working directory")
    wolframscript_path: str = StateField(description="Path to wolframscript")

    _observable = "sigma"
    _accepted = (ProcessType.SCATTERING_2TO2,)

    def _run(self) -> str:
        if not self.diagram and not self.diagrams:
            return self.format_error(
                error="Missing Parameter",
                reason="one of 'diagram' or 'diagrams' is required",
            )
        if self.diagram and self.diagrams:
            return self.format_error(
                error="Ambiguous Input",
                reason="'diagram' and 'diagrams' are mutually exclusive",
            )
        return self._generate_run_parse(
            specs=self.diagrams if self.diagrams else [self.diagram],
            channel=self.channel,
            relative_signs=self.relative_signs,
            values=self.values,
            assume_real_couplings=self.assume_real_couplings,
            simplifications=self.simplifications,
            sqrt_s=self.sqrt_s,
            script_name=self.script_name,
            timeout=self.timeout,
            prefix="sigma",
        )
