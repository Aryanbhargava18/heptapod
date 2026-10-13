"""
# checks.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
FeynRules consistency-check verdicts.

``UFO_generator.wl`` runs the six model-level checks FeynRules exposes in
``ToolBox.m`` and prints one verdict line per check.  Those verdicts are read
from each function's RETURN VALUE rather than from its printed prose:

    CheckHermiticity                returns the Feynman rules of L - HC[L];
                                    ``{}`` means hermitian.
    CheckDiagonalKineticTerms       returns True / False.
    CheckDiagonalQuadraticTerms     returns True / False.
    CheckDiagonalMassTerms          returns True / False.
    CheckKineticTermNormalisation   returns True, or Null on failure.
    CheckMassSpectrum               returns a TableForm whose passing rows have
                                    their "OK" stripped, so only discrepant rows
                                    retain a StyleForm marker.

The prose those functions print is not parsed.  It is phrased for a human
reading a notebook ("The lagrangian appears not to be hermitian.", "Non
diagonal kinetic term found: ...") and classifying it by keyword is both
fragile and silent when it goes wrong.

Verdicts are three-valued.  Several of these functions return ``Null`` rather
than ``False``: ``CheckKineticTermNormalisation`` has no else-branch on its
final ``If``, and the three diagonality checks return ``Null`` when there are
no terms of the relevant kind to examine.  ``not True`` therefore does not mean
"failed", and this module never collapses the two -- an undecided check is
reported as INCONCLUSIVE so it cannot be mistaken for a passing one.

This module turns verdicts into messages aimed at an agent that has to go and
fix the ``.fr``; interpretation lives here rather than in Wolfram Language for
the same reason it does in ``tools/wolfram/run_wolfram_tool.py`` -- it is
unit-testable without a Wolfram kernel.
"""

import re
from typing import Dict, List, Optional

# Verdicts. A check is PASS only on an affirmative return value; anything that
# leaves the question open is INCONCLUSIVE, never PASS.
PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"

# One verdict line per check, emitted by UFO_generator.wl. Pipes and newlines
# are stripped from `detail` on the Wolfram side so the line stays parseable.
# `scope` is "model" or "baseline"; it is optional so that output from an older
# generator still parses, defaulting to the model scope.
_LINE_RE = re.compile(
    r"^HEPTAPOD-CHECK\|"
    r"(?:scope=(?P<scope>[^|]*)\|)?"
    r"name=(?P<name>[^|]*)\|"
    r"verdict=(?P<verdict>[^|]*)\|"
    r"detail=(?P<detail>.*)$",
    re.MULTILINE,
)

MODEL = "model"
BASELINE = "baseline"

_VERDICTS = {PASS, FAIL, INCONCLUSIVE}

_MASS_RE = re.compile(
    r"^HEPTAPOD-MASS\|particle=(?P<particle>[^|]*)\|derived=(?P<derived>[^|]*)"
    r"\|declared=(?P<declared>[^|]*)\|agrees=(?P<agrees>\w+)",
    re.MULTILINE,
)

_RESIDUAL_RE = re.compile(
    r"^HEPTAPOD-RESIDUAL\|count=(?P<count>\d+)\|fields=(?P<fields>.*)$",
    re.MULTILINE,
)

_GAUGE_RE = re.compile(
    r"^HEPTAPOD-GAUGE\|feynman=(?P<feynman>\w+)\|source=(?P<source>.*)$",
    re.MULTILINE,
)

# CheckMassSpectrum's own tolerance, from ToolBox.m: a state agrees when its
# derived value lies between 0.999 and 1.001 times the declared one. Named here
# because the number decides what a caller can get away with -- one model
# author, not knowing it, spent a large part of a build proving an O(eta^2)
# shift was structurally unavoidable and then chose a conservative mixing
# parameter to stay clear of a bound they could not see.
MASS_SPECTRUM_REL_TOL = 1e-3


def parse_mass_spectrum(stdout_text: Optional[str]) -> List[Dict[str, object]]:
    """Declared versus derived mass, per state.

    CheckMassSpectrum's verdict is a single pass/fail over the whole spectrum,
    which says neither which state disagrees nor by how much. The numbers do.
    A model can be hermitian, have diagonal kinetic and mass terms and exactly
    the right interaction content while declaring a mass its own Lagrangian
    does not produce -- a sign error in a mass-matrix inversion looks like
    that, and every other check passes it. "125.0 declared, 164.2 derived" is
    the whole diagnosis.

    Args:
        stdout_text: Captured stdout from the wolframscript run.

    Returns:
        ``{particle, declared, derived, agrees, rel_diff, rel_tol}`` per state,
        empty if the check did not run. ``rel_diff`` is None when either value
        is not a number FeynRules could evaluate; ``rel_tol`` is the tolerance
        ``agrees`` was decided against (MASS_SPECTRUM_REL_TOL, 0.1%).

        The numbers arrive at full precision. They did not always: the
        generator rendered them with Mathematica's default six significant
        figures, which reported ``rel_diff`` as exactly 0.0 for a state whose
        true residual was 1.8e-8.
    """
    out: List[Dict[str, object]] = []
    for m in _MASS_RE.finditer(stdout_text or ""):
        def _num(v: str) -> Optional[float]:
            try:
                return float(v)
            except ValueError:
                return None
        declared, derived = _num(m.group("declared")), _num(m.group("derived"))
        rel = None
        if declared not in (None, 0.0) and derived is not None:
            rel = abs(derived - declared) / abs(declared)
        out.append({
            "particle": m.group("particle").strip(),
            "declared": declared if declared is not None else m.group("declared"),
            "derived": derived if derived is not None else m.group("derived"),
            "agrees": m.group("agrees").strip().lower() == "true",
            "rel_diff": rel,
            "rel_tol": MASS_SPECTRUM_REL_TOL,
        })
    return out


def parse_gauge(stdout_text: Optional[str]) -> Optional[Dict[str, object]]:
    """Which gauge the model was compiled in.

    FeynmanGauge is a global set inside SM.fr, True by default there, so an
    add-on inherits Feynman gauge without deciding to and acquires a Goldstone
    and ghost sector with it. That is invisible in every other output and shows
    up only as a vertex count tens of entries larger than expected.

    Args:
        stdout_text: Captured stdout from the wolframscript run.

    Returns:
        ``{feynman_gauge, source}``, or None from a generator that predates the
        line.
    """
    m = _GAUGE_RE.search(stdout_text or "")
    if m is None:
        return None
    return {
        "feynman_gauge": m.group("feynman").strip().lower() == "true",
        "source": m.group("source").strip(),
    }

# Appended to the diagonality remedies. See pitfalls.md P13.
# The discriminator, not reassurance. An earlier version of this note said a
# cross term with a mixing-angle coefficient was "usually a false positive",
# named the published HAHM implementation as the example, and advised
# confirming numerically at your parameter point. An agent hit a cross term
# matching that description on every axis -- HAHM, coefficient `ch sh` -- and
# it was a real defect: a replacement rule had failed to match, so the entire
# |D Phi_SM|^2 term was absent, taking the W and Z masses and every SM Higgs
# coupling with it. The model compiled and passed hermiticity. It was caught
# only because that agent had independently derived that the term must cancel
# SYMBOLICALLY, so "vanishes at the physical angle" could not explain a
# residue. The "confirm numerically" advice was also wrong: that residue was
# non-zero at every parameter point, and the phrasing invites substituting the
# default angle and eyeballing one number.
#
# So: state the test, not the verdict. A benign cross term vanishes IDENTICALLY
# once the angle's defining relation is substituted. A residue that survives
# symbolically is a real defect, and so is a missing term that should have
# cancelled it.
MIXING_ANGLE_NOTE = (
    " DIAGNOSING THIS: eigenstates defined through an angle are diagonal only"
    " at the value that diagonalises the mass matrix, and that relation is not"
    " declared anywhere FeynRules can see -- so a cross term whose coefficient"
    " is a product of mixing parameters (ch sh, ca sa) MAY be benign. The test"
    " is symbolic, not numerical: substitute the angle's defining relation and"
    " check the term vanishes IDENTICALLY. If a residue survives symbolically,"
    " or if the term that should have cancelled it is absent altogether, this"
    " is a real defect -- most commonly a replacement rule that silently failed"
    " to match, which drops whole Lagrangian terms while leaving the model"
    " compilable and hermitian. Do not conclude 'false positive' from the"
    " coefficient's shape alone, and do not test by plugging in the default"
    " parameter point: a genuine defect is non-zero at every point, which one"
    " numerical spot-check does not distinguish from round-off. This applies to"
    " the diagonality checks only -- mass_spectrum compares numbers and a"
    " mixing angle cannot confuse it, so do not switch the whole check family"
    " off over this one."
)

# Human-facing label and remedy for each check. The remedy is what an agent
# editing the .fr should actually do, not a restatement of the failure.
_SPECS = {
    "hermiticity": {
        "label": "Hermiticity",
        "fail": (
            "The Lagrangian is not Hermitian: FeynRules found residual terms in "
            "L - HC[L]. Almost always a term is missing its conjugate -- add "
            "`+ HC[...]` around the offending interaction, or write the "
            "conjugate out explicitly."
        ),
        "inconclusive": (
            "Hermiticity could not be decided. Check that the Lagrangian "
            "expression passed to the tool is the complete one."
        ),
    },
    "kinetic_diagonal": {
        "label": "Diagonal kinetic terms",
        "fail": (
            "Kinetic terms are not diagonal: two fields mix at quadratic order, "
            "so the fields in the Lagrangian are not mass eigenstates. Rotate "
            "them to the mass basis, or declare the mixing so FeynRules can do "
            "it." + MIXING_ANGLE_NOTE
        ),
        "inconclusive": (
            "No kinetic terms were found, so diagonality could not be decided. "
            "Check that every propagating field has a kinetic term."
        ),
    },
    "quadratic_diagonal": {
        "label": "Diagonal quadratic terms",
        "fail": (
            "Quadratic terms are not diagonal: fields mix at quadratic order. "
            "Rotate to the mass basis, or declare the mixing."
        ),
        "inconclusive": (
            "No quadratic terms were found, so diagonality could not be decided."
        ),
    },
    "mass_diagonal": {
        "label": "Diagonal mass terms",
        "fail": (
            "Mass terms are not diagonal: the mass matrix mixes fields, so the "
            "declared particles are not mass eigenstates. Diagonalise the mass "
            "matrix, or declare the mixing." + MIXING_ANGLE_NOTE
        ),
        "inconclusive": (
            "No mass terms were found, so diagonality could not be decided."
        ),
    },
    "kinetic_norm": {
        "label": "Kinetic term normalisation",
        "fail": (
            "One or more kinetic terms are not canonically normalised. "
            "FeynRules expects a coefficient of -1/+1 for a complex field and "
            "-1/2 / +1/2 for a real one. Rescale the field or fix the "
            "coefficient."
        ),
        "inconclusive": (
            "Kinetic-term normalisation could not be decided. FeynRules returns "
            "no verdict both when normalisation fails and when the kinetic "
            "sector is not diagonal -- see the diagonal kinetic terms result, "
            "which identifies which case this is."
        ),
    },
    "mass_spectrum": {
        "label": "Mass spectrum",
        "fail": (
            "Masses computed from the Lagrangian disagree with the declared "
            "values by more than 0.1%. Update the `Mass` entry in the particle "
            "class, or fix the mass term it is meant to match."
        ),
        "inconclusive": (
            "The mass spectrum could not be computed. This usually means the "
            "mass terms are not diagonal -- see that result."
        ),
    },
}

# Appended when the SM-only baseline shows the same result, so the reader can
# tell an inherited quirk from something the model introduced.
_PRE_EXISTING_NOTE = (
    "The Standard Model alone gives the same result here, so this is inherited "
    "rather than introduced by this model -- CheckDiagonalQuadraticTerms reports "
    "the R_xi gauge-Goldstone mixing term, and CheckKineticTermNormalisation "
    "warns about the massless neutrinos. Worth fixing only if the model touches "
    "that sector."
)

# Order checks are reported in: cheapest and most fundamental first, so the
# first failure an agent reads is the one most likely to explain the rest.
CHECK_ORDER = (
    "hermiticity",
    "kinetic_diagonal",
    "quadratic_diagonal",
    "mass_diagonal",
    "kinetic_norm",
    "mass_spectrum",
)

# The checks worth running against an arbitrary model. Each of these passes for
# the Standard Model on its own, so a failure is about the model under test.
DEFAULT_CHECKS = (
    "hermiticity",
    "kinetic_diagonal",
    "mass_diagonal",
    "mass_spectrum",
)

# Excluded from the default set because they fail for the Standard Model itself,
# and would therefore fail for essentially every model built on it:
#
#   quadratic_diagonal  CheckDiagonalQuadraticTerms runs over GetQuadraticTerms
#                       -- kinetic + mass + mixing -- without the scalar-vector
#                       DeleteCases that CheckDiagonalKineticTerms applies, so
#                       it reports the R_xi gauge-Goldstone mixing term of any
#                       spontaneously broken gauge theory. What remains after
#                       that is already covered by the kinetic and mass checks.
#
#   kinetic_norm        CheckKineticTermNormalisation expects a coefficient of
#                       -1/+1 (or -1/2 / +1/2 for a real field), which does not
#                       apply to a Weyl fermion. It trips on the SM's massless
#                       neutrinos, and on any model with new chiral matter.
#
# Available on request; ``baseline_compare`` exists to make them interpretable
# when you do ask for them.
OPTIONAL_CHECKS = (
    "quadratic_diagonal",
    "kinetic_norm",
)


def _message(name: str, verdict: str, detail: str, pre_existing: bool = False) -> str:
    """Build the agent-facing message for one check result.

    Args:
        name: Check name as emitted by UFO_generator.wl.
        verdict: One of PASS, FAIL, INCONCLUSIVE.
        detail: Supporting text from FeynRules (may be empty).
        pre_existing: True when the SM-only baseline gave the same verdict.

    Returns:
        A one-or-two sentence message. For a passing check this is a bare
        confirmation; for the others it names the remedy, and says so when the
        result was inherited from the Standard Model rather than introduced.
    """
    spec = _SPECS.get(name)
    label = spec["label"] if spec else name

    if verdict == PASS:
        return f"{label}: passed."

    if spec is None:
        text = f"{label}: {verdict}."
    else:
        text = spec[FAIL] if verdict == FAIL else spec[INCONCLUSIVE]

    if pre_existing:
        text = f"{text} {_PRE_EXISTING_NOTE}"
    return f"{text} ({detail})" if detail else text


def _disambiguate_kinetic_norm(results: List[Dict[str, object]]) -> None:
    """Resolve the ambiguous ``Null`` from CheckKineticTermNormalisation.

    That function returns ``Null`` both when normalisation genuinely fails and
    when it never got that far because the kinetic sector is not diagonal --
    its outer ``If[isdiag, ...]`` has no else-branch. The two cases are
    distinguishable only by looking at the diagonality result alongside it, so
    the correlation is done here rather than in Wolfram Language.

    If the kinetic terms ARE diagonal, an undecided normalisation can only mean
    normalisation failed, and the verdict is promoted to FAIL. A result the SM
    baseline shares is left alone: the FeynRules SM has massless neutrinos whose
    kinetic terms trip this check, and promoting that to a failure would report
    a problem against every SM extension ever written.

    Args:
        results: Parsed model-scope check results, modified in place.
    """
    by_name = {r["name"]: r for r in results}
    norm = by_name.get("kinetic_norm")
    diag = by_name.get("kinetic_diagonal")

    if norm is None or diag is None:
        return
    if norm["verdict"] != INCONCLUSIVE or diag["verdict"] != PASS:
        return
    if norm.get("pre_existing"):
        return

    norm["verdict"] = FAIL
    norm["passed"] = False
    norm["message"] = _message("kinetic_norm", FAIL, str(norm.get("detail", "")))


def parse_check_verdicts(stdout_text: Optional[str]) -> List[Dict[str, object]]:
    """Extract check results from a UFO_generator.wl stdout log.

    The generator emits one verdict line per check for the model, and -- when
    BaselineCompare is on -- another set for the Standard Model on its own. Only
    model-scope results are returned; the baseline is used to mark which of them
    the model actually introduced.

    Args:
        stdout_text: Captured stdout from the wolframscript run. May be None.

    Returns:
        A list of ``{name, label, verdict, passed, pre_existing, detail,
        message}`` dicts in CHECK_ORDER, containing only the checks that ran.
        ``passed`` is True only for PASS, so an INCONCLUSIVE check is never
        counted as a success by a caller that looks at that field alone.
    """
    scopes: Dict[str, Dict[str, Dict[str, object]]] = {MODEL: {}, BASELINE: {}}

    for match in _LINE_RE.finditer(stdout_text or ""):
        scope = (match.group("scope") or MODEL).strip() or MODEL
        name = match.group("name").strip()
        verdict = match.group("verdict").strip().lower()
        detail = match.group("detail").strip()

        if verdict not in _VERDICTS:
            verdict = INCONCLUSIVE
        scopes.setdefault(scope, {})[name] = {
            "name": name,
            "verdict": verdict,
            "detail": detail,
        }

    model = scopes.get(MODEL, {})
    baseline = scopes.get(BASELINE, {})

    results: List[Dict[str, object]] = []
    names = [n for n in CHECK_ORDER if n in model]
    names.extend(n for n in model if n not in CHECK_ORDER)

    for name in names:
        raw = model[name]
        verdict = str(raw["verdict"])
        detail = str(raw["detail"])
        # Inherited only if the SM alone reached the same non-passing verdict.
        pre_existing = (
            verdict != PASS
            and name in baseline
            and baseline[name]["verdict"] == verdict
        )
        results.append({
            "name": name,
            "label": _SPECS.get(name, {}).get("label", name),
            "verdict": verdict,
            "passed": verdict == PASS,
            "pre_existing": pre_existing,
            "detail": detail,
            "message": _message(name, verdict, detail, pre_existing),
        })

    _disambiguate_kinetic_norm(results)
    return results


def failed_checks(
    results: List[Dict[str, object]],
    include_pre_existing: bool = False,
) -> List[Dict[str, object]]:
    """Return the results that are outright failures.

    INCONCLUSIVE results are excluded: they warrant a warning, not a hard stop,
    and a caller that wants them can filter on ``verdict``. Failures the SM
    baseline also exhibits are excluded by default for the same reason -- they
    say nothing about the model under test.

    Args:
        results: Output of :func:`parse_check_verdicts`.
        include_pre_existing: Also return failures inherited from the SM.

    Returns:
        The matching subset, in the order given.
    """
    return [
        r for r in results
        if r["verdict"] == FAIL
        and (include_pre_existing or not r.get("pre_existing"))
    ]


def summarise(results: List[Dict[str, object]]) -> str:
    """One-line summary of a check run, for logs and findings entries.

    Args:
        results: Output of :func:`parse_check_verdicts`.

    Returns:
        A string such as ``"4 passed, 1 failed, 1 inconclusive (1 inherited
        from the SM)"``, or a note that no checks ran.
    """
    if not results:
        return "no checks run"

    counts = {PASS: 0, FAIL: 0, INCONCLUSIVE: 0}
    for r in results:
        key = str(r["verdict"])
        counts[key] = counts.get(key, 0) + 1

    parts = [f"{counts[PASS]} passed"]
    if counts[FAIL]:
        parts.append(f"{counts[FAIL]} failed")
    if counts[INCONCLUSIVE]:
        parts.append(f"{counts[INCONCLUSIVE]} inconclusive")

    inherited = sum(1 for r in results if r.get("pre_existing"))
    suffix = f" ({inherited} inherited from the SM)" if inherited else ""
    return ", ".join(parts) + suffix


def parse_residual_fields(stdout_text: Optional[str]) -> Optional[Dict[str, object]]:
    """Unphysical fields still present in the expanded Lagrangian.

    An add-on that re-expresses an SM gauge eigenstate adds a rule to
    MR$Definitions. When the pattern does not match what FeynRules stored, the
    rule is never applied, the unphysical field survives expansion, and vertex
    extraction drops every term containing it -- silently. The model still
    compiles and still passes hermiticity; what it has lost is whole terms.

    Args:
        stdout_text: Captured stdout from the wolframscript run.

    Returns:
        ``{count, fields, ok}``, or None from a generator predating the line.
    """
    m = _RESIDUAL_RE.search(stdout_text or "")
    if m is None:
        return None
    fields = [f for f in m.group("fields").split(",") if f.strip()]
    return {"count": int(m.group("count")), "fields": fields, "ok": not fields}


def residual_fields_message(residual: Optional[Dict[str, object]]) -> Optional[str]:
    """An agent-facing diagnosis for surviving unphysical fields."""
    if not residual or residual.get("ok"):
        return None
    names = ", ".join(str(f) for f in (residual.get("fields") or []))
    return (
        f"Unphysical field(s) survived expansion: {names}. These are gauge "
        f"eigenstates that should have been replaced by mass eigenstates "
        f"before the Lagrangian was expanded. Vertex extraction DROPS every "
        f"term containing them, so the model will compile, pass hermiticity "
        f"and be missing whole sectors. The usual cause is a rule added to "
        f"MR$Definitions whose pattern never matched: FeynRules stores each "
        f"definition twice, bare (Phi[2], Wi[mu_,1]) and index-wrapped "
        f"(Phi[Index[SU2D,2]], Wi[Index[Lorentz,mu_],Index[SU2W,3]]). A "
        f"single Lorentz slot absorbs the wrapper, so B[mu_] matches both, "
        f"but Phi[2] and Wi[mu_,1] match only the bare form. Match both "
        f"(e.g. Phi[2|Index[SU2D,2]]), and REPLACE or PREPEND the rule rather "
        f"than appending it -- ReplaceAll takes the first matching rule and "
        f"SM.fr's own rules are already in the list."
    )
