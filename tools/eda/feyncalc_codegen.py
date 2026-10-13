"""
# feyncalc_codegen.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

FeynCalc code generator — converts Diagram dataclasses into runnable
Mathematica/FeynCalc scripts for exact tree-level calculations.

Supports:
  - 1 -> 2 decays (tree-level, with or without one propagator)
  - 2 -> 2 scattering (tree-level, s/t/u-channel + contact)

The generated scripts follow the standard workflow:
  amplitude -> square -> spin/pol sums -> traces -> kinematics -> observable
and emit SYMBOLIC_RESULT / NUMERICAL_RESULT markers compatible with
tools/wolfram/wolfram_runner.py parsing.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

from tools.nda.simple_diagram import Diagram, Particle, Vertex, Propagator
from .scattering import (  # noqa: F401
    Channel,
    UnsupportedScattering,
    UnsupportedTopology,
    build_amplitude as _build_scattering_amplitude,
    build_amplitude_sum as _build_scattering_amplitude_sum,
    kinematics_block as _scattering_kinematics,
    cross_section_block as _scattering_cross_section,
    DEFAULT_INTEGRATE_TIMEOUT_S as _DEFAULT_INTEGRATE_TIMEOUT_S,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class UnknownVertexType(ValueError):
    """A vertex type string that no dispatch branch recognizes.

    RAISED RATHER THAN DEFAULTED. Until 2.9.1 the fermionic dispatches ended
    in an unconditional fallback -- VFF to plain `I g gamma^mu`, SFF to plain
    `I g` -- so ANY unrecognized string silently produced a different vertex
    than the caller asked for, reported as a normal result. A typo, a type
    belonging to the other bilinear (`pseudoscalar` on a VFF vertex), or an
    invented name all returned the plain-coupling width with status ok.

    That is the worst failure mode a physics tool can have: confident, well
    formed, and wrong, with nothing downstream able to tell. An agent driving
    these tools has no way to notice, because the number looks reasonable.
    """


#: Accepted VFF vertex types, normalized base -> user-facing spellings.
#: Bosonic vertices are NOT listed: those dispatch on the spin configuration,
#: so the type string is genuinely free there and must stay that way.
VFF_VERTEX_TYPES = {
    "vector": {"spellings": ("vector", "gauge-vector"), "couplings": ("g",)},
    "axialvector": {"spellings": ("axial-vector",), "couplings": ("g",)},
    "vectoraxial": {"spellings": ("vector-axial", "va"),
                    "couplings": ("gV", "gA")},
    "chiral": {"spellings": ("chiral", "vector-chiral"),
               "couplings": ("gL", "gR")},
    "lefthanded": {"spellings": ("left-handed",), "couplings": ("g",)},
    "righthanded": {"spellings": ("right-handed",), "couplings": ("g",)},
    "tensor": {"spellings": ("tensor", "dipole"), "couplings": ("g",)},
    "tensorchiral": {"spellings": ("tensor-chiral", "dipole-chiral"),
                     "couplings": ("gL", "gR")},
}

#: Accepted SFF vertex types, normalized base -> spellings + coupling contract.
SFF_VERTEX_TYPES = {
    "yukawa": {"spellings": ("yukawa", "scalar"), "couplings": ("g",)},
    "pseudoscalar": {"spellings": ("pseudoscalar",), "couplings": ("g",)},
    "chiral": {"spellings": ("chiral", "yukawa-chiral", "scalar-chiral"),
               "couplings": ("gL", "gR")},
    "scalarva": {"spellings": ("scalar-va",), "couplings": ("gS", "gP")},
}


def _vertex_type_error(kind: str, vtype: str, table: dict) -> UnknownVertexType:
    """`vtype` should be the caller's LITERAL string where available.

    The dispatches normalize before matching (lowercase, hyphens and
    underscores stripped), so echoing the normalized form back would show a
    caller something they never typed -- "vectoraxialcoupling" for an input of
    "vector_axial_coupling" -- which reads like a second, unrelated bug.
    """
    spellings = sorted(sp for spec in table.values()
                       for sp in spec["spellings"])
    # A letter-spelled type whose structure matches this bilinear is a
    # specific, common mistake: the caller named the STRUCTURE (two fermions
    # and a vector) and stopped, but the structure does not fix the Feynman
    # rule -- vector, axial, chiral and tensor all live there. Say so, rather
    # than leaving them to infer it from a list.
    structural = _structure_from_letters(_normalize_vtype_static(vtype))
    hint = ""
    if structural is not None:
        hint = (f" {vtype!r} names the vertex STRUCTURE, which the leg spins "
                f"already give; it does not say WHICH {kind} vertex, and the "
                f"structure does not determine that. Pick the Dirac "
                f"structure you mean from the list.")
    return UnknownVertexType(
        f"unknown {kind} vertex type {vtype!r}. Valid types: "
        f"{', '.join(spellings)}.{hint} (Bosonic vertices dispatch on the spin "
        f"configuration instead, so any type string is accepted there.)"
    )


def _named(coupling: dict) -> str:
    """" (you gave a, b)" — echo the caller's own keys back."""
    keys = ", ".join(sorted(coupling))
    return f" (you gave {keys})" if keys else ""


def _suggest_type(kind: str, table: dict, coupling: dict) -> str:
    """Name the vertex type whose contract the supplied KEYS actually match.

    THE POINT IS TO STOP A RENAME. An agent told only "wrong keys" will
    satisfy the checker by relabelling its dict and silently receive the
    vertex it did not ask for -- the exact class of failure this validation
    exists to end, reintroduced by the caller in good faith. If the keys are
    some other type's contract, the likely mistake is the TYPE, and the
    message has to say which and say that the two are different physics.
    """
    keys = set(coupling)
    if not keys:
        return ""
    for other, spec in table.items():
        if set(spec["couplings"]) == keys:
            spellings = " / ".join(repr(sp) for sp in spec["spellings"])
            return (
                f" Those keys are the contract for {spellings}, which is a"
                f" DIFFERENT VERTEX, not a different spelling of this one --"
                f" if that is the physics you meant, change the type rather"
                f" than renaming the couplings."
            )
    return ""


def resolve_coupling(kind: str, base: str, table: dict, coupling):
    """Coupling symbol names for `base`, VALIDATED against its contract.

    THE TYPE IS AUTHORITATIVE. Each vertex type declares the couplings it
    takes, and the supplied value must be able to satisfy exactly those. The
    previous design inferred the vertex from the coupling's SHAPE instead --
    a dict meant chiral, wherever it appeared -- so the declared physics lost
    to a formatting detail, and six legitimate types silently emitted a chiral
    rule. Shape-sniffing is why that bug existed; validating against a
    declaration is why it cannot recur.

    Accepted, for a type declaring one name:
      "g"            -> ("g",)
      {"g": "gEff"}  -> ("gEff",)   one entry, unambiguous
    for a type declaring several (say gV, gA):
      {"gV": "a", "gA": "b"} -> ("a", "b")   exactly those keys
      "y"                    -> ("yV", "yA") the documented suffix shorthand,
                                unambiguous because the type fixes the arity

    Everything else raises. In particular a multi-entry dict on a
    single-coupling type used to take the first value and discard the rest, an
    empty dict used to invent the name "g", and wrong keys on a two-coupling
    type used to fall back to the contract's own names -- returning a formula
    in symbols the caller had never supplied, so the number silently went
    missing. All three answered a different question without saying so.
    """
    required = table[base]["couplings"]
    if isinstance(coupling, dict):
        if len(required) == 1:
            if len(coupling) != 1:
                raise UnknownVertexType(
                    f"{kind} vertex type {base!r} takes exactly one coupling "
                    f"({required[0]}), but {len(coupling)} were given"
                    f"{_named(coupling)}."
                    + _suggest_type(kind, table, coupling)
                    + f" To keep this vertex, pass one name: "
                      f'"coupling": "{required[0]}"'
                      f'  (or a single-entry dict, {{"{required[0]}": "<name>"}}).'
                )
            return (str(next(iter(coupling.values()))),)
        missing = [k for k in required if k not in coupling]
        if missing:
            raise UnknownVertexType(
                f"{kind} vertex type {base!r} requires the couplings "
                f"{', '.join(required)}, and {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} absent"
                f"{_named(coupling)}."
                + _suggest_type(kind, table, coupling)
                + f" To keep this vertex, supply its own names: "
                + '"coupling": {'
                + ", ".join(f'"{k}": "<name>"' for k in required)
                + "}."
            )
        return tuple(str(coupling[k]) for k in required)
    if coupling is None or str(coupling) == "":
        raise UnknownVertexType(
            f"{kind} vertex type {base!r} requires couplings "
            f"{', '.join(required)}; none were given."
        )
    if len(required) == 1:
        return (str(coupling),)
    # Suffix shorthand: "y" with a (gV, gA) contract means yV, yA. The suffix
    # is the contract's own name minus its leading "g", so the mapping stays
    # derived from the declaration rather than hardcoded per type.
    return tuple(f"{coupling}{name.lstrip('g') or name}" for name in required)



#: A bosonic vertex's coupling contract. The bosonic dispatches key off the
#: SPIN CONFIGURATION rather than a type table, so there is no per-type entry
#: to hang this on -- but the arity is fixed for all of them: one coupling.
#: Expressing that as a contract lets `resolve_coupling` validate it, so the
#: bosonic and fermionic paths share one implementation and one error voice.
BOSONIC_COUPLINGS = ("g",)


def resolve_bosonic_coupling(vtype, coupling) -> str:
    """The single coupling symbol for a bosonic vertex, VALIDATED.

    WHY THIS EXISTS. The bosonic branches took `_coupling_value`'s result
    straight into an f-string, and `_coupling_value` returns a dict UNCHANGED
    for dict input. So `{"g": "g"}` -- the spelling agents reach for -- was
    interpolated as a Python repr and reached Mathematica as

        amp = I ({'g': 'g'}) PolarizationVector[p1, mu1] MTD[mu1, mu2];

    which is not Wolfram syntax. The script failed to parse, so no amplitude,
    no formula and no number came back, and the tool still said `status: ok`
    because a script and a results file both existed. 2.15.2 fixed the
    fermionic half of exactly this; the bosonic half was missed, and all five
    bosonic tasks in the hepbench `symbolic` family failed on it.

    The type string stays free-form (the dispatch reads spins, not names), so
    this constrains only the coupling.
    """
    table = {vtype: {"spellings": (vtype,), "couplings": BOSONIC_COUPLINGS}}
    return resolve_coupling("bosonic", vtype, table, coupling)[0]


class EmittedCodeLeak(AssertionError):
    """A Python object repr reached the generated Wolfram script.

    An internal invariant, not a user error: some branch interpolated a dict
    or list straight into the script instead of resolving it to a symbol.
    AssertionError rather than ValueError because the caller cannot fix it and
    it must never be reported as though they could.
    """


#: The Python-repr signatures. Braces alone say nothing -- Wolfram uses them
#: for every list and rule, `{g -> 0.47, mV -> 2150}` -- and neither does a
#: bracket before a double quote, which is just `Print["..."]`. What no
#: Wolfram we emit ever contains is:
#:
#:   a QUOTED KEY before a colon      {'g': 'g'}   {"g": "g"}
#:   a SINGLE-QUOTED string           ['gL', 'gR']  ('a',)
#:
#: Wolfram strings are double-quoted; `'` is the derivative operator, which
#: this generator never emits. So a single quote just inside an opening
#: bracket is always a Python container that escaped resolution.
_REPR_LEAK = re.compile(r"""[\[{(]\s*'|[{]\s*["'][^"']*["']\s*:""")


def _guard_emitted_code(code: str) -> None:
    """Fail loudly if a Python repr reached the emitted script.

    WHY AT THIS CHOKEPOINT. The leak above came from four separate bosonic
    branches, and the result was not a wrong answer but an unparseable script:
    Mathematica raised ToExpression::sntx, printed no markers, and the tool
    reported `status: ok` with an empty formula. The per-branch fix is the
    coupling contract; this is the net under it, so a branch nobody has
    audited -- or one added later -- cannot reintroduce the same silence.
    Every generated script passes through `_assemble_script`.

    It has already earned this: with the 1->2 branches fixed, it caught the
    identical leak in the 2->2 scattering path.
    """
    if _REPR_LEAK.search(code) is None:
        return
    line = next((l for l in code.splitlines() if _REPR_LEAK.search(l)), "")
    raise EmittedCodeLeak(
        "a Python object repr reached the generated Wolfram script: "
        f"{line.strip()!r}. Some vertex branch interpolated a dict or list "
        "instead of resolving it to a coupling symbol (see "
        "`resolve_bosonic_coupling` / `resolve_coupling`). The script would "
        "not parse, and the tool would report success with an empty result."
    )



def _normalize_vtype_static(vtype: str) -> str:
    """Module-level twin of `FeynCalcCodeGenerator._normalize_vtype`.

    The structure checks run before any generator instance is in hand, and
    duplicating the rule would be a second source of truth -- so the method
    delegates here.
    """
    base = (vtype or "").lower().replace("-", "").replace("_", "")
    for suffix in ("3pt", "4pt", "5pt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    _ALIASES = {
        "yukawachiral": "chiral", "scalarchiral": "chiral",
        "vectorchiral": "chiral", "scalarva": "scalarva",
        "vectoraxial": "vectoraxial", "dipolechiral": "tensorchiral",
    }
    return _ALIASES.get(base, base)



#: The spin multiset each fermionic table serves. Declared once per TABLE, not
#: per type, so a spelling added to a table inherits its structure for free.
#: VFF is a vector plus two fermions; SFF a scalar plus two fermions.
VFF_SPINS = (0.5, 0.5, 1.0)
SFF_SPINS = (0.0, 0.5, 0.5)

_TABLE_STRUCTURES = (
    ("VFF", VFF_VERTEX_TYPES, VFF_SPINS),
    ("SFF", SFF_VERTEX_TYPES, SFF_SPINS),
)


def implied_vertex_spins(vtype: str):
    """The spin multiset the TYPE declares, or None if it declares none.

    Two derived sources, neither keyed to an individual vertex:

      letters  a type spelled from S/V/F is self-describing -- `ffv` is
               (0.5, 0.5, 1). This covers the agent shorthands (FFV, VFF,
               SFF) and, for free, the bosonic names the generator already
               uses, since `sss`/`ssv`/`svv`/`vvv` ARE letter-strings.
      tables   a named type carries its table's signature: `vector` is VFF,
               `yukawa` is SFF.

    None means the type declares no structure -- a free-form bosonic name like
    `field-strength`. There is then nothing to contradict, so nothing to
    check, and the long-standing promise that bosonic type names are free
    survives for exactly the names it was made for.
    """
    # LETTER RULE ONLY. A table's spellings are NOT exclusive to its
    # structure: `scalar` is registered as an alias of the SFF `yukawa`
    # vertex AND is the ordinary name for an SSS vertex
    # (L = -(mu/2) phi chi^2). Deriving (0, 0.5, 0.5) from the name therefore
    # rejected valid, long-standing calls -- 29 tests, including the
    # decay_S_to_SS ground-truth check. The same holds for `vector`, which
    # names both the VFF Dirac structure and, informally, bosonic vertices.
    #
    # So only a self-describing letter-string declares a structure. That is
    # the half with no collisions (asserted over both tables by a test), and
    # it is the half that catches the bug: `FFV` over three spin-1 legs.
    # The cost is that `vector` + (1,1,1) is NOT caught -- a name that is
    # merely unconventional for its spins stays legal, as it always was.
    return _structure_from_letters(_normalize_vtype_static(vtype))


def implied_vertex_kind(vtype: str):
    """The table kind a named type belongs to ("VFF"/"SFF"), else None."""
    base = _normalize_vtype_static(vtype)
    for kind, table, _spins in _TABLE_STRUCTURES:
        if base in table or base in {_normalize_vtype_static(sp)
                                     for spec in table.values()
                                     for sp in spec["spellings"]}:
            return kind
    return None


def _fmt_spins(spins) -> str:
    return ", ".join(("%g" % x) for x in spins)


def check_vertex_structure(vtype: str, leg_spins, raw_type=None) -> None:
    """Raise if the type and the declared leg spins describe different vertices.

    THE BUG THIS CLOSES. The generator dispatches on the spin multiset and the
    bosonic branches never read the type, so a Dirac-fermion decay declared
    with `spin: 1` on both fermions and `type: "FFV"` selected the Yang-Mills
    triple-gauge vertex and returned a confident width for a different
    process -- 423.67 GeV against a ground truth of 109.52 (hepbench
    decay_F_to_FpV, 2026-09-30, two trials of ten).

    It states the contradiction and BOTH readings without choosing between
    them, because the input cannot distinguish a mistyped spin from a mistyped
    name. Deciding for the caller would be inventing intent -- and an agent
    told only "wrong name" will satisfy the checker by renaming, which is the
    same silent failure reintroduced in good faith.
    """
    implied = implied_vertex_spins(vtype)
    if implied is None:
        return
    declared = tuple(sorted(float(s or 0) for s in leg_spins))
    if len(declared) != len(implied) or declared != implied:
        shown = raw_type or vtype
        kind = implied_vertex_kind(vtype)
        what = (f"is a {kind} type" if kind
                else "describes a vertex")
        raise VertexStructureMismatch(
            f"vertex type {shown!r} {what} -- spins ({_fmt_spins(implied)}) -- "
            f"but the legs you declared have spins ({_fmt_spins(declared)}). "
            f"The type and the spins fix the same vertex and must agree.\n"
            f"- If these legs are what {shown!r} says, correct the `spin` "
            f"fields to ({_fmt_spins(implied)}) -- a Dirac fermion is 0.5, a "
            f"massive vector is 1, a scalar is 0.\n"
            f"- If the spins are what you meant, use a type for "
            f"({_fmt_spins(declared)}) instead; for three spin-1 legs that is "
            f"a triple-gauge vertex (`vvv`).\n"
            f"Nothing else in the diagram says which you intended, so this is "
            f"not a renaming exercise -- changing the name to match the spins "
            f"gives you the other vertex."
        )



#: Spin multiset -> the name this generator dispatches it as. Six entries,
#: which is the whole space for a three- or four-leg vertex built from
#: scalars, fermions and vectors.
_STRUCTURE_NAMES = {
    (0.0, 0.0, 0.0): "SSS",
    (0.0, 0.0, 1.0): "SSV",
    (0.0, 1.0, 1.0): "SVV",
    (1.0, 1.0, 1.0): "VVV",
    (0.0, 0.5, 0.5): "SFF",
    (0.5, 0.5, 1.0): "VFF",
}


def _structures_used(spin_sets) -> list:
    """The distinct vertex structures a diagram dispatched, sorted.

    DEDUPLICATED, because a per-vertex listing does not scale: a 2->2 coherent
    sum over s and t channels runs to about six vertices, and a field that
    grows with the diagram is a field a caller learns to skip. There are only
    six structures, so this is bounded whatever the vertex count.

    Reported so the caller can see WHICH vertex the tool actually built. The
    evidence was already in the response -- an amplitude with no Dirac
    structure is not a fermion amplitude -- but it was not named, and a
    diagram whose spins say one thing while its author meant another produces
    a confident number either way. A spin set with no name is omitted rather
    than guessed at.
    """
    out = {_STRUCTURE_NAMES[k] for k in
           (tuple(sorted(float(x or 0) for x in s)) for s in spin_sets)
           if k in _STRUCTURE_NAMES}
    return sorted(out)


class ProcessType(Enum):
    """Classification of the scattering / decay process."""
    DECAY_1TO2 = auto()            # 1 initial, 2 final, 0 propagators
    DECAY_1TO2_1PROP = auto()      # 1 initial, 2 final, 1 propagator
    SCATTERING_2TO2 = auto()       # 2 initial, 2 final
    UNSUPPORTED = auto()


# Channel now lives in `scattering`, alongside the 2->2 builder that uses it.
# Re-exported here so existing importers of feyncalc_codegen.Channel keep working.


@dataclass
class GeneratedCode:
    """Container for the generated FeynCalc script."""
    code: str = ""
    process_type: ProcessType = ProcessType.UNSUPPORTED
    warnings: List[str] = field(default_factory=list)
    momentum_map: Dict[str, str] = field(default_factory=dict)
    channel: Optional[Channel] = None
    #: For a coherent sum, the channel of each summed diagram in order.
    #: `channel` stays None there, since a sum has no single channel.
    channels: List[Channel] = field(default_factory=list)
    #: Prefactors the width/cross-section formula was built with. Reported so
    #: the caller can SEE the conventions it inherited: a wrong
    #: identical-particle factor is a factor of 2 and a wrong initial-spin
    #: average is a factor of 3, and both used to be invisible -- they appeared
    #: only inside the generated script, which the caller has to go and read.
    symmetry_factor: Optional[int] = None
    n_initial_spin_states: Optional[int] = None
    #: Distinct vertex structures dispatched, e.g. ["VFF"]. Names what the
    #: tool built, which a diagram's spins decide and its author may not have
    #: intended -- see `_structures_used`.
    vertex_structures: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pure symbol/formatting helpers live in fc_symbols so that `scattering`
# can reuse them without importing this module (which imports it in turn).
# Re-exported here: this module's public surface is unchanged.
from .fc_symbols import (  # noqa: F401
    _ANTIPARTICLE_SUFFIXES,
    _is_numeric,
    _fmt_mma,
    _is_antiparticle,
    _safe_symbol,
    _mass_symbol,
    _coupling_value,
    InvalidMassSymbol,
    VertexStructureMismatch,
    _structure_from_letters,
)



#: (hbar c)^2 = 3.893793721e8 GeV^2 pb  (PDG physical constants).  The old
#: code hardcoded an nb factor only, while every cross-section benchmark
#: reports pb; emitting the constant by name keeps the script auditable.
GEV2_TO_BARN_COMMENT = "GeV2ToPb = 3.893793721*10^8;   (* (hbar c)^2 in GeV^2 pb *)"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class FeynCalcCodeGenerator:
    """
    Generate complete FeynCalc Mathematica scripts from Diagram objects.

    Usage::

        gen = FeynCalcCodeGenerator()
        result = gen.generate(diagram, sqrt_s=91.2)
        print(result.code)

    Args:
        assume_real_couplings: If True, skip coupling conjugation rules
            (treats all couplings as real). Default False: symbolic couplings get
            ``/. {g -> Conjugate[g], ...}`` rules so |M|^2 is correct for complex couplings.
    """

    def __init__(self, assume_real_couplings: bool = False, simplifications=None,
                 channel: Optional[str] = None,
                 integrate_timeout_s: float = _DEFAULT_INTEGRATE_TIMEOUT_S):
        self.assume_real_couplings = assume_real_couplings
        self.simplifications = simplifications
        # Explicit s/t/u/contact selection for 2->2. None means "infer",
        # which falls back to s-channel and records the assumption in
        # GeneratedCode.warnings rather than hiding it.
        self.channel = channel
        self._channel_assumed = False
        # Seconds the generated script may spend on the symbolic
        # t-integration before falling through to sigmaNIntegrate.
        self.integrate_timeout_s = integrate_timeout_s

    def _collect_coupling_symbols(self, diagram: Diagram) -> List[str]:
        """Extract symbolic (non-numeric) coupling names from diagram vertices."""
        symbols = []
        for v in diagram.vertices:
            g = _coupling_value(v, diagram.couplings)
            if isinstance(g, str):
                if not _is_numeric(g):
                    symbols.append(g)
            elif isinstance(g, dict):
                for val in g.values():
                    if not _is_numeric(val):
                        symbols.append(val)
        return list(dict.fromkeys(symbols))  # dedupe, preserve order

    def generate(self, diagram: Diagram, sqrt_s: Optional[float] = None) -> GeneratedCode:
        """
        Main entry point.

        Args:
            diagram: A Diagram dataclass (from tools.nda.simple_diagram).
            sqrt_s: Centre-of-mass energy in GeV. OPTIONAL for 2->2 now that
                the cross section is built symbolically in ``s``; supply it
                only to have the script also report a number.

        Returns:
            GeneratedCode with the Mathematica script and metadata.
        """
        result = GeneratedCode()
        self._channel_assumed = False

        # 1. Classify
        proc = self._classify_process(diagram)
        result.process_type = proc

        if proc == ProcessType.UNSUPPORTED:
            result.warnings.append(
                f"Unsupported topology: {len(diagram.initial)} initial, "
                f"{len(diagram.final)} final, "
                f"{len(diagram.propagators)} propagators, "
                f"{sum(1 for p in diagram.propagators if p.is_loop_propagator)} loops."
            )
            return result

        # 2. Assign momenta
        mom_map = self._assign_momenta(diagram, proc)
        result.momentum_map = mom_map

        is_decay = proc in (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP)

        # 3. Build sections
        sections: List[str] = []
        sections.append(self._header(diagram, proc))
        sections.append(self._mass_definitions(diagram))

        try:
            amp_section, amp_warnings = self._build_amplitude(diagram, proc, mom_map)
            sections.append(amp_section)
            result.warnings.extend(amp_warnings)

            if is_decay:
                # Decay ordering (unchanged, and validated): square and trace
                # first, then substitute rest-frame kinematics.
                coupling_syms = self._collect_coupling_symbols(diagram)
                sections.append(self._square_amplitude(coupling_syms))
                sections.append(self._spin_pol_sums(diagram, proc, mom_map))
                sections.append(self._trace_and_contract())
                sections.append(self._kinematics_decay(diagram, mom_map))
                sections.append(self._width_formula(diagram, proc, mom_map))
            else:
                # Scattering ordering: kinematics are fixed BEFORE the square,
                # so the traces are evaluated against on-shell scalar products
                # in (s, t) and no symbolic u survives into the observable.
                result.channel = self._infer_channel(diagram, proc)
                sections.extend(self._scattering_tail(
                    diagram, mom_map, sqrt_s,
                    self._collect_coupling_symbols(diagram)))
        except UnsupportedTopology as exc:
            result.process_type = ProcessType.UNSUPPORTED
            result.warnings.append(str(exc))
            if self._channel_assumed:
                # The failure may be nothing more than the assumed channel
                # being the one channel this process does not have.
                result.warnings.append(
                    "No channel was specified, so s-channel was assumed — and "
                    "that assumption is what failed. Pass channel='t' or "
                    "channel='u' if the diagram you meant is an exchange one."
                )
            return result

        if self._channel_assumed:
            result.warnings.append(
                "No channel was specified for this exchange diagram; assumed "
                "s-channel. Pass channel='s'|'t'|'u' to make it explicit."
            )

        sections.append(self._numerical_eval(diagram, proc, sqrt_s=sqrt_s))

        result.code = self._assemble_script(sections)
        result.symmetry_factor = getattr(self, "_last_symmetry_factor", None)
        result.vertex_structures = _structures_used(
            getattr(self, "_last_spin_sets", []) or [])
        result.n_initial_spin_states = getattr(
            self, "_last_n_initial_spin_states", None)
        return result

    def _scattering_tail(self, diagram: Diagram, mom_map: Dict[str, str],
                         sqrt_s: Optional[float],
                         coupling_syms: List[str]) -> List[str]:
        """Everything after the amplitude, shared by the single and summed paths.

        Kinematics come FIRST: the traces are then evaluated against on-shell
        scalar products in (s, t) and no symbolic u survives into the
        observable.
        """
        return [
            self._kinematics_scattering(diagram, mom_map, sqrt_s),
            self._square_amplitude(coupling_syms),
            self._spin_pol_sums(diagram, ProcessType.SCATTERING_2TO2, mom_map),
            self._trace_and_contract(),
            "(* Step 5: on-shell squared amplitude in (s, t) *)\n"
            "ampSqKin = ampSq // Simplify;\n",
            self._cross_section_formula(diagram, mom_map, sqrt_s),
        ]

    def generate_sum(self, diagrams: Sequence[Tuple[Diagram, "Channel | str"]],
                     sqrt_s: Optional[float] = None,
                     relative_signs: Optional[Sequence[int]] = None) -> GeneratedCode:
        """Build several 2->2 diagrams and add their amplitudes COHERENTLY.

        Most real 2->2 processes are a sum -- phi phi -> phi phi with a cubic
        coupling is s + t + u, identical-fermion scattering is s + t -- so a
        generator that only ever emits one diagram is silently wrong whenever
        more than one contributes.

        `diagrams` is a list of (Diagram, channel) pairs sharing external legs
        and masses; the channel may be a Channel or one of 's'/'t'/'u'/
        'contact'. `relative_signs` carries the (-1) between diagrams related
        by interchange of two external fermion lines.

        The caller supplies WHICH diagrams contribute: deciding that needs a
        Lagrangian, and the diagram spec carries topology rather than a model.
        """
        result = GeneratedCode()
        self._channel_assumed = False

        if not diagrams:
            result.process_type = ProcessType.UNSUPPORTED
            result.warnings.append("generate_sum needs at least one diagram.")
            return result

        pairs: List[Tuple[Diagram, Channel]] = []
        for d, ch in diagrams:
            if self._classify_process(d) != ProcessType.SCATTERING_2TO2:
                result.process_type = ProcessType.UNSUPPORTED
                result.warnings.append(
                    "generate_sum handles 2->2 scattering only; got a "
                    f"{self._classify_process(d).name} diagram. Decays are "
                    "generated one at a time with generate()."
                )
                return result
            if isinstance(ch, str):
                key = ch.strip().lower()
                try:
                    ch = {"s": Channel.S, "t": Channel.T, "u": Channel.U,
                          "contact": Channel.CONTACT}[key]
                except KeyError:
                    result.process_type = ProcessType.UNSUPPORTED
                    result.warnings.append(
                        f"Unknown scattering channel {ch!r}; expected one of "
                        "s, t, u, contact."
                    )
                    return result
            pairs.append((d, ch))

        proc = ProcessType.SCATTERING_2TO2
        result.process_type = proc
        result.channels = [c for _, c in pairs]

        head = pairs[0][0]
        mom_map = self._assign_momenta(head, proc)
        result.momentum_map = mom_map

        coupling_syms: List[str] = []
        for d, _ in pairs:
            for sym in self._collect_coupling_symbols(d):
                if sym not in coupling_syms:
                    coupling_syms.append(sym)

        sections: List[str] = [
            self._header(head, proc),
            self._mass_definitions(head),
        ]
        try:
            sections.append(
                _build_scattering_amplitude_sum(self, pairs,
                                                relative_signs=relative_signs)
            )
            sections.extend(self._scattering_tail(head, mom_map, sqrt_s,
                                                  coupling_syms))
        except UnsupportedTopology as exc:
            result.process_type = ProcessType.UNSUPPORTED
            result.warnings.append(str(exc))
            return result

        sections.append(self._numerical_eval(head, proc, sqrt_s=sqrt_s))
        result.code = self._assemble_script(sections)
        result.symmetry_factor = getattr(self, "_last_symmetry_factor", None)
        result.vertex_structures = _structures_used(
            getattr(self, "_last_spin_sets", []) or [])
        result.n_initial_spin_states = getattr(
            self, "_last_n_initial_spin_states", None)
        return result

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_process(self, diagram: Diagram) -> ProcessType:
        n_init = len(diagram.initial)
        n_final = len(diagram.final)
        n_prop = len(diagram.propagators)
        n_loop = sum(1 for p in diagram.propagators if p.is_loop_propagator)

        if n_loop > 0:
            return ProcessType.UNSUPPORTED

        if n_init == 1 and n_final == 2 and n_prop == 0:
            return ProcessType.DECAY_1TO2
        if n_init == 1 and n_final == 2 and n_prop == 1:
            return ProcessType.DECAY_1TO2_1PROP
        if n_init == 2 and n_final == 2:
            return ProcessType.SCATTERING_2TO2

        return ProcessType.UNSUPPORTED

    # ------------------------------------------------------------------
    # Momentum assignment
    # ------------------------------------------------------------------

    def _assign_momenta(self, diagram: Diagram, proc: ProcessType) -> Dict[str, str]:
        """Return mapping {role -> momentum label}."""
        if proc in (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP):
            mom = {"initial_0": "p", "final_0": "p1", "final_1": "p2"}
            if proc == ProcessType.DECAY_1TO2_1PROP:
                mom["prop_0"] = "q"
            return mom
        else:  # 2->2
            mom = {
                "initial_0": "p1", "initial_1": "p2",
                "final_0": "p3", "final_1": "p4",
            }
            if diagram.propagators:
                mom["prop_0"] = "q"
            return mom

    # ------------------------------------------------------------------
    # Spinor type assignment
    # ------------------------------------------------------------------

    def _spinor_expr(self, particle: Particle, momentum: str, role: str) -> str:
        """
        Return the FeynCalc spinor for an external fermion.

        role: 'incoming' or 'outgoing'
        """
        mass = _mass_symbol(particle, 0)
        anti = _is_antiparticle(particle.label)

        if role == "incoming":
            if anti:
                return f"SpinorVBar[{momentum}, {mass}]"
            else:
                return f"SpinorU[{momentum}, {mass}]"
        else:  # outgoing
            if anti:
                return f"SpinorV[{momentum}, {mass}]"
            else:
                return f"SpinorUBar[{momentum}, {mass}]"

    # ------------------------------------------------------------------
    # Code sections
    # ------------------------------------------------------------------

    def _header(self, diagram: Diagram, proc: ProcessType) -> str:
        labels_init = " ".join(p.label or "?" for p in diagram.initial)
        labels_final = " ".join(p.label or "?" for p in diagram.final)
        return (
            f'(* FeynCalc script generated by HEPTAPOD/Diagrammatica *)\n'
            f'(* Process: {labels_init} -> {labels_final} *)\n'
            f'(* Process type: {proc.name} *)\n\n'
            f'<< FeynCalc`\n'
        )

    def _mass_definitions(self, diagram: Diagram) -> str:
        lines = ["(* Mass definitions *)"]
        all_particles: List[Tuple[Particle, int]] = []
        for i, p in enumerate(diagram.initial):
            all_particles.append((p, i))
        for i, p in enumerate(diagram.final):
            all_particles.append((p, i))

        seen = set()
        for p, idx in all_particles:
            sym = _mass_symbol(p, idx)
            if sym not in seen:
                seen.add(sym)
                # A symbolic mass IS the symbol (see `_mass_symbol`), so there
                # is nothing to assign -- emitting `MZP = MZP;` would be a
                # self-assignment. Only a numeric mass pins a value here.
                if isinstance(p.mass, str):
                    continue
                val = p.mass if p.mass is not None else 0
                lines.append(f"{sym} = {_fmt_mma(val)};")

        # Propagator masses
        for i, prop in enumerate(diagram.propagators):
            sym = f"mProp{i}"
            val = prop.mass if prop.mass is not None else 0
            lines.append(f"{sym} = {_fmt_mma(val)};")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Vertex type normalization and coupling extraction
    # ------------------------------------------------------------------

    def _normalize_vtype(self, vtype: str) -> str:
        """Normalize vertex type: lowercase, strip hyphens/underscores, resolve aliases.

        Delegates to the module-level `_normalize_vtype_static` so the
        structure checks, which run without a generator instance, cannot drift
        from this.
        """
        return _normalize_vtype_static(vtype)

    def _normalize_vtype_unused(self, vtype: str) -> str:
        base = vtype.lower().replace("-", "").replace("_", "")
        # Strip valence suffixes
        for suffix in ("3pt", "4pt", "5pt"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        _ALIASES = {
            "yukawachiral": "chiral",
            "scalarchiral": "chiral",
            "vectorchiral": "chiral",
            "scalarva": "scalarva",
            "vectoraxial": "vectoraxial",
            "dipolechiral": "tensorchiral",
        }
        return _ALIASES.get(base, base)

    def _extract_chiral_couplings(self, coupling) -> Tuple[str, str]:
        """Extract gL, gR from coupling (string -> derive suffixes, dict -> use keys)."""
        if isinstance(coupling, dict):
            return coupling.get("gL", "gL"), coupling.get("gR", "gR")
        return f"{coupling}L", f"{coupling}R"

    def _extract_va_couplings(self, coupling) -> Tuple[str, str]:
        """Extract gV, gA from coupling (string -> derive suffixes, dict -> use keys)."""
        if isinstance(coupling, dict):
            return coupling.get("gV", "gV"), coupling.get("gA", "gA")
        return f"{coupling}V", f"{coupling}A"

    def _extract_va_sff_couplings(self, coupling) -> Tuple[str, str]:
        """Extract gS, gP from coupling (string -> derive suffixes, dict -> use keys)."""
        if isinstance(coupling, dict):
            return coupling.get("gS", "gS"), coupling.get("gP", "gP")
        return f"{coupling}S", f"{coupling}P"

    def _extract_single_coupling(self, coupling) -> str:
        """Extract a single coupling string (pass through str, take first dict value)."""
        if isinstance(coupling, dict):
            vals = list(coupling.values())
            return vals[0] if vals else "g"
        return str(coupling)

    # ------------------------------------------------------------------
    # VFF gamma structure dispatch
    # ------------------------------------------------------------------

    def _vff_gamma_structure(
        self, vtype: str, coupling, mu: str,
        vec_momentum: Optional[str] = None, raw_type: Optional[str] = None,
    ) -> str:
        """
        Build the VFF vertex factor string for FeynCalc.

        Args:
            vtype: Vertex type string (raw from diagram — normalized internally).
            coupling: Resolved coupling — a string for simple couplings, or a
                      dict for chiral / V-A vertices.
            mu: Lorentz index string (e.g., "mu0", "mu", "nu").
            vec_momentum: Momentum label for the vector boson (required for
                          tensor/dipole vertices; ignored for others).

        Returns:
            FeynCalc expression string for the vertex factor
            (without surrounding spinors).

        Notes:
            Uses FeynCalc's native chiral projectors:
              GA[7] = (1 - GA[5])/2 = P_L  (left-handed)
              GA[6] = (1 + GA[5])/2 = P_R  (right-handed)
        """
        base = self._normalize_vtype(vtype)

        # --- V-A: gV γ^μ - gA γ^μ γ^5 ---
        if base in ("vectoraxial", "va"):
            gV, gA = resolve_coupling("VFF", "vectoraxial",
                                     VFF_VERTEX_TYPES, coupling)
            return f"I GAD[{mu}] . (({gV}) - ({gA}) GA[5])"

        # --- Tensor / dipole: σ^{μν} k_ν ---
        if base in ("tensor", "dipole"):
            if vec_momentum is None:
                raise ValueError(
                    "Tensor/dipole vertex requires vec_momentum (the vector boson momentum)."
                )
            g = self._extract_single_coupling(coupling)
            return (
                f"I ({g}) DiracSigma[GA[{mu}], GA[nuT]] FV[{vec_momentum}, nuT]"
            )

        # --- Tensor-chiral / dipole-chiral: (gL P_L + gR P_R) σ^{μν} k_ν ---
        if base == "tensorchiral":
            if vec_momentum is None:
                raise ValueError(
                    "Tensor-chiral vertex requires vec_momentum (the vector boson momentum)."
                )
            gL, gR = resolve_coupling("VFF", "tensorchiral",
                                      VFF_VERTEX_TYPES, coupling)
            return (
                f"I (({gL}) GA[7] + ({gR}) GA[6]) . "
                f"DiracSigma[GA[{mu}], GA[nuT]] FV[{vec_momentum}, nuT]"
            )

        # --- Chiral VFF: γ^μ (gL P_L + gR P_R) ---
        if base == "chiral":
            gL, gR = resolve_coupling("VFF", "chiral",
                                      VFF_VERTEX_TYPES, coupling)
            return f"I GAD[{mu}] . (({gL}) GA[7] + ({gR}) GA[6])"

        # --- Single-projector types ---
        if base == "axialvector":
            (g,) = resolve_coupling("VFF", base, VFF_VERTEX_TYPES, coupling)
            return f"I ({g}) GAD[{mu}] . GA[5]"
        if base == "lefthanded":
            (g,) = resolve_coupling("VFF", base, VFF_VERTEX_TYPES, coupling)
            return f"I ({g}) GAD[{mu}] . GA[7]"
        if base == "righthanded":
            (g,) = resolve_coupling("VFF", base, VFF_VERTEX_TYPES, coupling)
            return f"I ({g}) GAD[{mu}] . GA[6]"

        # --- Pure vector γ^μ ---
        # Compare NORMALIZED spellings: `base` has already been lowercased with
        # hyphens and underscores stripped, so matching it against the display
        # forms would reject "gauge-vector" (normalizes to "gaugevector").
        if base in {self._normalize_vtype(v)
                    for v in VFF_VERTEX_TYPES["vector"]["spellings"]}:
            (g,) = resolve_coupling("VFF", "vector", VFF_VERTEX_TYPES, coupling)
            return f"I ({g}) GAD[{mu}]"

        raise _vertex_type_error("VFF", raw_type or vtype, VFF_VERTEX_TYPES)

    # ------------------------------------------------------------------
    # SFF coupling structure dispatch
    # ------------------------------------------------------------------

    def _sff_coupling_structure(self, vtype: str, coupling,
                                raw_type: Optional[str] = None) -> str:
        """
        Build the SFF vertex factor string for FeynCalc (no γ^μ — scalar parent).

        Args:
            vtype: Vertex type string (raw from diagram — normalized internally).
            coupling: Resolved coupling — a string for simple couplings, or a
                      dict for chiral / S-P scalar vertices.

        Returns:
            FeynCalc expression string for the vertex factor
            (without surrounding spinors).

        Notes:
            For chiral scalar couplings: ``yL GA[7] + yR GA[6]``
            (no γ^μ, unlike VFF where the vertex is γ^μ · projector).
        """
        base = self._normalize_vtype(vtype)

        # --- Scalar V-A: gS + gP γ^5 ---
        if base == "scalarva":
            gS, gP = resolve_coupling("SFF", "scalarva",
                                      SFF_VERTEX_TYPES, coupling)
            return f"I (({gS}) + ({gP}) GA[5])"

        # --- Chiral scalar: yL P_L + yR P_R ---
        if base == "chiral":
            gL, gR = resolve_coupling("SFF", "chiral",
                                      SFF_VERTEX_TYPES, coupling)
            return f"I (({gL}) GA[7] + ({gR}) GA[6])"

        # --- Pseudoscalar: i g γ^5 ---
        if base == "pseudoscalar":
            (g,) = resolve_coupling("SFF", base, SFF_VERTEX_TYPES, coupling)
            return f"I ({g}) GA[5]"

        # --- Scalar Yukawa  i y ---
        if base in {self._normalize_vtype(v)
                    for v in SFF_VERTEX_TYPES["yukawa"]["spellings"]}:
            (g,) = resolve_coupling("SFF", "yukawa", SFF_VERTEX_TYPES, coupling)
            return f"I ({g})"

        raise _vertex_type_error("SFF", raw_type or vtype, SFF_VERTEX_TYPES)

    # ------------------------------------------------------------------
    # Amplitude construction
    # ------------------------------------------------------------------

    def _build_amplitude(
        self, diagram: Diagram, proc: ProcessType, mom_map: Dict[str, str]
    ) -> Tuple[str, List[str]]:
        """Build the amplitude expression. Returns (code_section, warnings)."""
        warnings: List[str] = []

        if proc == ProcessType.DECAY_1TO2:
            return self._amplitude_decay_no_prop(diagram, mom_map), warnings
        elif proc == ProcessType.DECAY_1TO2_1PROP:
            return self._amplitude_decay_1prop(diagram, mom_map), warnings
        elif proc == ProcessType.SCATTERING_2TO2:
            return self._amplitude_scattering(diagram, mom_map), warnings

        warnings.append("Cannot build amplitude for unsupported process.")
        return "(* Amplitude: unsupported *)\namp = 0;\n", warnings

    def _amplitude_decay_no_prop(self, diagram: Diagram, mom_map: Dict[str, str]) -> str:
        """1->2 decay with a single vertex, no propagator."""
        parent = diagram.initial[0]
        d0 = diagram.final[0]
        d1 = diagram.final[1]
        vertex = diagram.vertices[0] if diagram.vertices else None
        g = _coupling_value(vertex, diagram.couplings) if vertex else "g"
        vtype = (vertex.type.lower() if vertex else "").replace("-", "").replace("_", "")

        # The BOSONIC branches below need a single resolved symbol; the
        # fermionic ones pass `g` on to their own contract
        # (`_vff_gamma_structure` / `_sff_coupling_structure`), which validates
        # against the type's declared couplings. So resolve lazily, per branch:
        # resolving eagerly would reject a legitimate fermionic {gL, gR}.
        def gb():
            return resolve_bosonic_coupling(
                (vertex.type if vertex else "bosonic"), g)

        p = mom_map["initial_0"]
        p1 = mom_map["final_0"]
        p2 = mom_map["final_1"]

        lines = [f"(* Step 1: Amplitude for {parent.label} -> {d0.label} {d1.label} *)"]

        # Determine vertex structure from spins -- but FIRST require that the
        # declared type agrees with them. Dispatching on spins alone let a
        # fermion decay typed "FFV" but declared with spin 1 on both fermions
        # select the triple-gauge branch and return a confident width for a
        # different process.
        spins = sorted([parent.spin or 0, d0.spin or 0, d1.spin or 0])
        if vertex is not None:
            check_vertex_structure(vertex.type, spins, raw_type=vertex.type)
        self._last_spin_sets = [tuple(spins)]

        if spins == [0, 0, 0]:
            # SSS
            lines.append(f"amp = I ({gb()});")

        elif spins == [0, 0.5, 0.5]:
            # SFF or VFF?  Check parent spin
            if (parent.spin or 0) == 0:
                # SFF: Scalar/Pseudoscalar -> F Fbar
                fbar, f_ = self._order_fermion_pair(d0, d1, p1, p2, "outgoing")
                sff_vertex = self._sff_coupling_structure(
                    vtype, g, raw_type=(vertex.type if vertex else None))
                lines.append(f"amp = {fbar} . ({sff_vertex}) . {f_};")
            else:
                # F -> f' S: parent is incoming fermion, one daughter is scalar
                parent_spinor = self._spinor_expr(parent, p, "incoming")
                sff_vertex = self._sff_coupling_structure(
                    vtype, g, raw_type=(vertex.type if vertex else None))
                if (d0.spin or 0) == 0.5:
                    # d0 is the fermion daughter, d1 is the scalar
                    out_fermion = self._spinor_expr(d0, p1, "outgoing")
                else:
                    # d1 is the fermion daughter, d0 is the scalar
                    out_fermion = self._spinor_expr(d1, p2, "outgoing")
                lines.append(
                    f"amp = {out_fermion} . ({sff_vertex}) . {parent_spinor};"
                )

        elif spins == [0, 0, 1]:
            # SSV derivative vertex: i g (ka - kb)^mu over the TWO SCALARS,
            # ALL-INCOMING (see scattering.VVV_CONVENTION_NOTE -- the same
            # convention governs every momentum-carrying vertex). An outgoing
            # leg enters as -p.
            mu = "mu1"
            if (parent.spin or 0) == 1:
                # V -> S1 S2: both scalars outgoing, so ka - kb = p2 - p1.
                lines.append(
                    f"amp = I ({gb()}) PolarizationVector[{p}, {mu}] "
                    f"FVD[{p2} - {p1}, {mu}];"
                )
            else:
                # S -> S' V: the scalars are the parent (incoming, +p) and the
                # scalar daughter (outgoing, -ps), so ka - kb = p + ps.
                #
                # This WAS `FVD[p - ps]`, using the momenta as drawn. By
                # momentum conservation p - ps is the vector's own momentum,
                # and eps(pv) . pv = 0, so the amplitude vanished identically:
                # heptapod returned ZERO for every S -> S' V decay.
                v_mom = p1 if (d0.spin or 0) == 1 else p2
                s_mom = p2 if (d0.spin or 0) == 1 else p1
                lines.append(
                    f"amp = I ({gb()}) PolarizationVector[{v_mom}, {mu}] "
                    f"FVD[{p} + {s_mom}, {mu}];"
                )

        elif spins == [0, 1, 1]:
            if (parent.spin or 0) == 1:
                # V -> S V: parent is vector, one daughter is scalar, one is vector
                mu0 = "mu0"
                if (d0.spin or 0) == 1:
                    v_mom, s_mom = p1, p2
                else:
                    v_mom, s_mom = p2, p1
                mu1 = "mu1"
                lines.append(
                    f"amp = I ({gb()}) PolarizationVector[{p}, {mu0}] "
                    f"PolarizationVector[{v_mom}, {mu1}] MTD[{mu0}, {mu1}];"
                )
            else:
                # S -> V V: scalar parent, two vector daughters
                mu1, mu2 = "mu1", "mu2"
                if vtype in ("fieldstrength", "dim5ff"):
                    # φFF: 2ig [(k1·k2)(ε1·ε2) - (k1·ε2)(k2·ε1)]
                    lines.append(
                        f"amp = 2 I ({gb()}) ("
                        f"SPD[{p1}, {p2}] MTD[{mu1}, {mu2}] - "
                        f"FVD[{p1}, {mu2}] FVD[{p2}, {mu1}]"
                        f") PolarizationVector[{p1}, {mu1}] PolarizationVector[{p2}, {mu2}];"
                    )
                elif vtype in ("dualfieldstrength", "dim5ffdual"):
                    # φFF̃: 2g ε^{μνρσ} k1_ρ k2_σ ε1_μ ε2_ν
                    lines.append(
                        f"amp = 2 ({gb()}) Eps[LorentzIndex[{mu1}], LorentzIndex[{mu2}], "
                        f"Momentum[{p1}], Momentum[{p2}]] "
                        f"PolarizationVector[{p1}, {mu1}] PolarizationVector[{p2}, {mu2}];"
                    )
                else:
                    # Default SVV: I g g^{μν}
                    lines.append(
                        f"amp = I ({gb()}) PolarizationVector[{p1}, {mu1}] "
                        f"PolarizationVector[{p2}, {mu2}] MTD[{mu1}, {mu2}];"
                    )

        elif spins == [0.5, 0.5, 1]:
            # VFF
            if (parent.spin or 0) == 1:
                # V -> F Fbar: parent is the vector boson
                mu = "mu0"  # parent polarization index
                fbar, f_ = self._order_fermion_pair(d0, d1, p1, p2, "outgoing")
                gamma_str = self._vff_gamma_structure(
                    vtype, g, mu, vec_momentum=p,
                    raw_type=(vertex.type if vertex else None))
                lines.append(
                    f"amp = PolarizationVector[{p}, {mu}] "
                    f"{fbar} . ({gamma_str}) . {f_};"
                )
            else:
                # F -> F V  (e.g., radiative fermion decay)
                mu = "mu1" if (d0.spin or 0) == 1 else "mu2"
                v_mom = p1 if (d0.spin or 0) == 1 else p2
                gamma_str = self._vff_gamma_structure(
                    vtype, g, mu, vec_momentum=v_mom,
                    raw_type=(vertex.type if vertex else None))
                # parent is incoming fermion
                parent_spinor = self._spinor_expr(parent, p, "incoming")
                if (d0.spin or 0) == 0.5:
                    out_fermion = self._spinor_expr(d0, p1, "outgoing")
                    lines.append(
                        f"amp = PolarizationVector[{v_mom}, {mu}] "
                        f"{out_fermion} . ({gamma_str}) . {parent_spinor};"
                    )
                else:
                    out_fermion = self._spinor_expr(d1, p2, "outgoing")
                    lines.append(
                        f"amp = PolarizationVector[{v_mom}, {mu}] "
                        f"{out_fermion} . ({gamma_str}) . {parent_spinor};"
                    )

        elif spins == [1, 1, 1]:
            # VVV: triple gauge, ALL-INCOMING momenta (see
            # scattering.VVV_CONVENTION_NOTE). For P -> q1 q2 the incoming
            # momenta are k0 = P, k1 = -q1, k2 = -q2, which satisfy the
            # k0 + k1 + k2 = 0 the Feynman rule is derived under.
            #
            # This CHANGED in the scattering-hardening work. The previous
            # code evaluated the same expression at (P, q1, q2), momenta
            # summing to 2P rather than 0, and was wrong at finite masses:
            # it reproduced the standard Z' -> W+W- width only
            # asymptotically (0.209 of it at mV/m = 3), while the form
            # below reproduces it exactly at every mass ratio.
            mu0, mu1, mu2 = "mu0", "mu1", "mu2"
            lines.append(
                f"amp = I ({gb()}) PolarizationVector[{p}, {mu0}] "
                f"PolarizationVector[{p1}, {mu1}] PolarizationVector[{p2}, {mu2}] ("
                f"MTD[{mu0}, {mu1}] FVD[{p} + {p1}, {mu2}] + "
                f"MTD[{mu1}, {mu2}] FVD[{p2} - {p1}, {mu0}] + "
                f"MTD[{mu2}, {mu0}] FVD[-{p2} - {p}, {mu1}]);"
            )

        else:
            # Previously emitted `amp = I (g)` behind a Mathematica comment --
            # a bare constant amplitude, which integrates to a plausible number
            # and reports success. Refuse instead.
            raise UnknownVertexType(
                f"no amplitude for vertex spin configuration {spins}. "
                "Supported: [0,0,0] SSS, [0,0,1] SSV, [0,1,1] SVV, "
                "[1,1,1] VVV, [0,0.5,0.5] SFF, [0.5,0.5,1] VFF."
            )

        return "\n".join(lines) + "\n"

    def _amplitude_decay_1prop(self, diagram: Diagram, mom_map: Dict[str, str]) -> str:
        """1 -> 2 with one internal line: refused, because it does not close.

        The previous implementation emitted ``amp = g0 g1 propNum`` with no
        spinor or Lorentz structure at all, under a comment reading "for
        production use, specialize vertex structures per topology" -- the same
        silent-nonsense failure mode the 2->2 fallback had. It returned a
        well-formed script and a confident, meaningless number.

        It cannot be fixed by supplying the missing structure, because the
        topology is not a tree amplitude. Count line-ends: three external legs
        plus one internal propagator give 3 + 2 = 5, while two three-point
        vertices need 6. The only assignment that closes is a two-point
        insertion (mass mixing, or a self-energy on a leg) beside one
        three-point vertex -- a propagator correction or a mixing chain, not a
        distinct amplitude. The diagram spec cannot say which is meant, so
        guessing would be inventing physics.
        """
        raise UnsupportedTopology(
            "A 1 -> 2 decay with one internal propagator is not a tree "
            "amplitude: three external legs plus one propagator give five "
            "line-ends, but two 3-point vertices need six. The topology only "
            "closes with a 2-point insertion (mass mixing or a self-energy), "
            "which is a propagator correction rather than a separate diagram. "
            "If you meant a mixing chain (e.g. A' -> gamma* -> f fbar), fold "
            "the mixing into an effective coupling and use the plain 1 -> 2 "
            "topology with no propagator. If you meant a cascade, the process "
            "has three or more final-state particles."
        )

    def _amplitude_scattering(self, diagram: Diagram, mom_map: Dict[str, str]) -> str:
        """2->2 tree amplitude, delegated to the compositional builder.

        The previous implementation covered exactly one exchange case
        (all-fermion externals with a vector mediator, s/t only -- u fell
        through to s) and emitted a structurally empty
        ``amp = I g^2 FAD[...]`` for everything else, which produced a
        confident wrong number rather than an error. `scattering` covers
        the full space and raises `UnsupportedScattering` outside it.
        """
        channel = self._infer_channel(diagram, ProcessType.SCATTERING_2TO2)
        return _build_scattering_amplitude(self, diagram, channel)

    # ------------------------------------------------------------------
    # Fermion ordering helpers
    # ------------------------------------------------------------------

    def _order_fermion_pair(
        self, p0: Particle, p1: Particle,
        mom0: str, mom1: str, role: str
    ) -> Tuple[str, str]:
        """
        For an outgoing fermion pair, return (UBar_expr, V_expr).
        For incoming, return (UBar_expr, V_expr) with incoming spinor types.

        Returns spinor expressions as (bar-spinor, spinor).
        """
        # Determine which is particle and which is antiparticle
        anti0 = _is_antiparticle(p0.label)
        anti1 = _is_antiparticle(p1.label)

        if role == "outgoing":
            if anti0 and not anti1:
                # p0 is antiparticle -> SpinorV, p1 is particle -> SpinorUBar
                return (
                    f"SpinorUBar[{mom1}, {_mass_symbol(p1, 1)}]",
                    f"SpinorV[{mom0}, {_mass_symbol(p0, 0)}]"
                )
            elif anti1 and not anti0:
                # p1 is antiparticle -> SpinorV, p0 is particle -> SpinorUBar
                return (
                    f"SpinorUBar[{mom0}, {_mass_symbol(p0, 0)}]",
                    f"SpinorV[{mom1}, {_mass_symbol(p1, 1)}]"
                )
            else:
                # Default: first is UBar, second is V
                return (
                    f"SpinorUBar[{mom0}, {_mass_symbol(p0, 0)}]",
                    f"SpinorV[{mom1}, {_mass_symbol(p1, 1)}]"
                )
        else:  # incoming
            if anti0 and not anti1:
                return (
                    f"SpinorU[{mom1}, {_mass_symbol(p1, 1)}]",
                    f"SpinorVBar[{mom0}, {_mass_symbol(p0, 0)}]"
                )
            elif anti1 and not anti0:
                return (
                    f"SpinorU[{mom0}, {_mass_symbol(p0, 0)}]",
                    f"SpinorVBar[{mom1}, {_mass_symbol(p1, 1)}]"
                )
            else:
                return (
                    f"SpinorU[{mom0}, {_mass_symbol(p0, 0)}]",
                    f"SpinorVBar[{mom1}, {_mass_symbol(p1, 1)}]"
                )

    def _order_fermion_pair_incoming(
        self, p0: Particle, p1: Particle, mom0: str, mom1: str
    ) -> Tuple[str, str]:
        """
        For an incoming fermion pair (e.g., e+ e-), return spinor bilinear parts.

        Convention for incoming:
          particle  -> SpinorU
          antiparticle -> SpinorVBar

        We return (bar_spinor, spinor) so the bilinear reads bar_spinor . Gamma . spinor.
        """
        anti0 = _is_antiparticle(p0.label)
        anti1 = _is_antiparticle(p1.label)

        if anti0 and not anti1:
            # p0 is antiparticle (VBar), p1 is particle (U)
            # bilinear: VBar[p0] . Gamma . U[p1]  but we want bar . G . spinor
            return (
                f"SpinorVBar[{mom0}, {_mass_symbol(p0, 0)}]",
                f"SpinorU[{mom1}, {_mass_symbol(p1, 1)}]"
            )
        elif anti1 and not anti0:
            # p1 is antiparticle, p0 is particle
            return (
                f"SpinorVBar[{mom1}, {_mass_symbol(p1, 1)}]",
                f"SpinorU[{mom0}, {_mass_symbol(p0, 0)}]"
            )
        else:
            # Default: assume p0 bar, p1 not
            return (
                f"SpinorVBar[{mom0}, {_mass_symbol(p0, 0)}]",
                f"SpinorU[{mom1}, {_mass_symbol(p1, 1)}]"
            )

    # ------------------------------------------------------------------
    # Square, spin sums, traces
    # ------------------------------------------------------------------

    def _square_amplitude(self, coupling_symbols: Optional[List[str]] = None) -> str:
        lines = "(* Step 2: Square the amplitude *)\n"
        lines += "ampCC = ComplexConjugate[amp];\n"
        if coupling_symbols and not self.assume_real_couplings:
            # Apply coupling conjugation as a separate replacement rule,
            # *after* ComplexConjugate has handled spinor chain reversal.
            # FeynCalc's ComplexConjugate[amp, Conjugate -> {...}] fails for
            # scalar chiral vertices (SFF with GA[7]/GA[6]) because the Dirac
            # structures interfere with the coupling-conjugation logic when
            # there are no Lorentz indices.  Separating the two operations
            # avoids this issue and works uniformly for all vertex types.
            rules = ", ".join(f"{s} -> Conjugate[{s}]" for s in coupling_symbols)
            lines += f"ampCC = ampCC /. {{{rules}}};\n"
        lines += "ampSq = amp ampCC;\n"
        return lines

    def _pol_sum_call(self, momentum: str, particle: Particle) -> str:
        """Generate DoPolarizationSums call for an external vector boson.

        Massive vectors: ``DoPolarizationSums[expr, p]`` — physical 3-state
        sum −g_μν + p_μ p_ν / M².

        Massless vectors: ``DoPolarizationSums[expr, p, 0]`` — covariant
        gauge reference vector n=0 (Ward identity ensures unphysical
        polarizations decouple).
        """
        mass = particle.mass
        is_massive = mass is not None and (isinstance(mass, str) or mass > 0)
        if is_massive:
            return f"ampSq = DoPolarizationSums[ampSq, {momentum}];"
        else:
            return f"ampSq = DoPolarizationSums[ampSq, {momentum}, 0];"

    def _spin_pol_sums(self, diagram: Diagram, proc: ProcessType, mom_map: Dict[str, str]) -> str:
        lines = ["(* Step 3: Spin and polarization sums *)"]

        # Check if there are any fermions
        has_fermions = any(
            (p.spin or 0) == 0.5
            for p in diagram.initial + diagram.final
        )
        if has_fermions:
            lines.append("ampSq = FermionSpinSum[ampSq];")

        # Polarization sums for external (on-shell) vector bosons.
        # Physical massive vectors: -g_{μν} + p_μ p_ν / p² (3-state sum).
        # Physical massless vectors: -g_{μν} (Ward identity ensures unphysical
        # polarizations decouple; equivalent to n=0 reference vector).
        # Note: VirtualBoson -> True gives -g_{μν} regardless of mass — only
        # correct for off-shell internal bosons, NOT for external particles.
        if proc in (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP):
            # Check parent
            if (diagram.initial[0].spin or 0) == 1:
                p_mom = mom_map["initial_0"]
                lines.append(self._pol_sum_call(p_mom, diagram.initial[0]))

            # Check daughters
            for i, fp in enumerate(diagram.final):
                if (fp.spin or 0) == 1:
                    fp_mom = mom_map[f"final_{i}"]
                    lines.append(self._pol_sum_call(fp_mom, fp))
        else:  # Scattering
            for i, ip in enumerate(diagram.initial):
                if (ip.spin or 0) == 1:
                    ip_mom = mom_map[f"initial_{i}"]
                    lines.append(self._pol_sum_call(ip_mom, ip))
            for i, fp in enumerate(diagram.final):
                if (fp.spin or 0) == 1:
                    fp_mom = mom_map[f"final_{i}"]
                    lines.append(self._pol_sum_call(fp_mom, fp))

        return "\n".join(lines) + "\n"

    def _trace_and_contract(self) -> str:
        return (
            "(* Step 4: Evaluate traces and contract *)\n"
            "ampSq = DiracSimplify[ampSq] // Contract // Simplify;\n"
        )

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _kinematics_decay(self, diagram: Diagram, mom_map: Dict[str, str]) -> str:
        """Rest-frame kinematic substitutions for 1->2 decay.

        Uses FCClearScalarProducts + ScalarProduct assignments (global)
        so FeynCalc resolves kinematics reliably — avoids late /. rules
        that can fail to substitute inside FeynCalc internal objects.
        """
        parent = diagram.initial[0]
        d0 = diagram.final[0]
        d1 = diagram.final[1]

        M = _mass_symbol(parent, 0)
        m1 = _mass_symbol(d0, 0)
        m2 = _mass_symbol(d1, 1)

        p = mom_map["initial_0"]
        p1 = mom_map["final_0"]
        p2 = mom_map["final_1"]

        lines = [
            "(* Step 5: Kinematics — rest frame of parent *)",
            f"(* p = ({M}, 0, 0, 0),  p1 + p2 = p *)",
            f"FCClearScalarProducts[];",
            f"ScalarProduct[{p}, {p}] = {M}^2;",
            f"ScalarProduct[{p1}, {p1}] = {m1}^2;",
            f"ScalarProduct[{p2}, {p2}] = {m2}^2;",
            f"ScalarProduct[{p}, {p1}] = ({M}^2 + {m1}^2 - {m2}^2)/2;",
            f"ScalarProduct[{p}, {p2}] = ({M}^2 - {m1}^2 + {m2}^2)/2;",
            f"ScalarProduct[{p1}, {p2}] = ({M}^2 - {m1}^2 - {m2}^2)/2;",
            f"ampSqKin = ampSq // Simplify;",
        ]

        return "\n".join(lines) + "\n"

    def _width_formula(self, diagram: Diagram, proc: ProcessType, mom_map: Dict[str, str]) -> str:
        """Compute partial decay width."""
        from tools.nda.simple_diagram import compute_symmetry_factor

        parent = diagram.initial[0]
        d0 = diagram.final[0]
        d1 = diagram.final[1]

        M = _mass_symbol(parent, 0)
        m1 = _mass_symbol(d0, 0)
        m2 = _mass_symbol(d1, 1)

        # Number of initial spin states
        spin_init = parent.spin if parent.spin is not None else 0
        n_spin_init = int(2 * spin_init + 1)

        # Identical-particle symmetry factor, DERIVED FROM THE LABELS: two
        # final-state particles are identical iff they share a label. See
        # `compute_symmetry_factor`; equal mass alone is deliberately not
        # taken as identity, since a degenerate doublet is a real thing.
        sym_factor = compute_symmetry_factor(diagram)
        self._last_symmetry_factor = sym_factor
        self._last_n_initial_spin_states = n_spin_init

        lines = [
            "(* Step 6: Partial decay width *)",
            f"(* Gamma = pMag / (8 pi M^2) * (1/nInit) * |M|^2 * colorFactor / symmetryFactor *)",
            f"pMag = Sqrt[({M}^2 - ({m1} + {m2})^2)({M}^2 - ({m1} - {m2})^2)] / (2 {M});",
            f"nInit = {n_spin_init};",
            f"colorFactor = {_fmt_mma(diagram.color_factor)};",
            f"symmetryFactor = {sym_factor};",
            f"width = pMag / (8 Pi {M}^2) * (1/nInit) * ampSqKin * colorFactor / symmetryFactor;",
            f"width = width // Simplify;",
        ]

        return "\n".join(lines) + "\n"

    def _kinematics_scattering(
        self, diagram: Diagram, mom_map: Dict[str, str],
        sqrt_s: Optional[float] = None
    ) -> str:
        """Delegated: explicit ScalarProducts in (s, t), u eliminated on shell.

        Replaces the old ``SetMandelstam`` block, which left ``u`` alive as
        an independent symbol and mixed a NUMERIC sqrt_s into the flux
        prefactor while |M|^2 still carried a SYMBOLIC s -- so the two never
        met and ``N[sigma]`` never resolved.
        """
        return _scattering_kinematics(diagram)

    def _cross_section_formula(
        self, diagram: Diagram, mom_map: Dict[str, str],
        sqrt_s: Optional[float] = None
    ) -> str:
        """Delegated: the 2->2 master formula, symbolic in s.

        Restores the identical-particle symmetry factor (absent before, which
        made phi phi -> phi phi come out exactly 2x too large) and the
        massless-vector polarisation count.
        """
        return _scattering_cross_section(
            diagram, integrate_timeout_s=self.integrate_timeout_s)

    # ------------------------------------------------------------------
    # Numerical evaluation + markers
    # ------------------------------------------------------------------

    def _numerical_eval(self, diagram: Diagram, proc: ProcessType,
                        sqrt_s: Optional[float] = None) -> str:
        """Emit numerical evaluation, symbolic extraction, and LaTeX markers.

        Emits three categories of structured output:
          - SYMBOLIC_RESULT: Mathematica InputForm expressions
          - NUMERICAL_RESULT: Floating-point numerical values
          - LATEX_RESULT: LaTeX strings via TeXForm[] for paper-ready formulas

        Intermediate quantities extracted:
          - ampSq: Spin/polarization-summed |M|^2 after kinematic substitution
          - width (decay) or sigma (scattering): Final observable
        """
        lines = ["(* Step 7: Symbolic extraction and numerical evaluation *)"]

        # -- Intermediate: |M|^2 after kinematics --
        lines.extend([
            '',
            '(* Squared amplitude after kinematics *)',
            'Print["SYMBOLIC_RESULT[ampSq]: ", ampSqKin];',
            'Print["LATEX_RESULT[ampSq]: ", ToString[TeXForm[ampSqKin]]];',
        ])

        if proc in (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP):
            lines.extend([
                '',
                '(* Decay width — symbolic and numerical *)',
                'Print["SYMBOLIC_RESULT[width]: ", width];',
                'Print["LATEX_RESULT[width]: ", ToString[TeXForm[width]]];',
                'widthNum = N[width];',
                'Print["NUMERICAL_RESULT[width_GeV]: ", widthNum];',
                '(* Convert to MeV for convenience *)',
                'Print["NUMERICAL_RESULT[width_MeV]: ", widthNum * 1000];',
            ])
        else:
            lines.extend([
                '',
                '(* Cross section — symbolic in s *)',
                'Print["SYMBOLIC_RESULT[sigma]: ", sigma];',
                'Print["LATEX_RESULT[sigma]: ", ToString[TeXForm[sigma]]];',
                'Print["SYMBOLIC_RESULT[dSigmaDt]: ", dSigmaDt];',
            ])
            if sqrt_s is not None:
                lines.extend([
                    '',
                    '(* Numerical evaluation at the requested sqrt(s) *)',
                    f'sigmaNum = N[sigma /. s -> ({_fmt_mma(sqrt_s)})^2];',
                    '(* A well-formed symbolic sigma can still evaluate to',
                    '   Undefined when the t-endpoints cancel; fall back to',
                    '   substituting first and integrating numerically. *)',
                    'If[!NumericQ[sigmaNum] || !FreeQ[sigmaNum, Undefined] ||',
                    '   !FreeQ[sigmaNum, Indeterminate] || !FreeQ[sigmaNum, DirectedInfinity],',
                    f'  sigmaNum = sigmaNIntegrate[{{s -> ({_fmt_mma(sqrt_s)})^2}}]];',
                    'Print["NUMERICAL_RESULT[sigma_GeV2]: ", sigmaNum];',
                    GEV2_TO_BARN_COMMENT,
                    'Print["NUMERICAL_RESULT[sigma_pb]: ", sigmaNum * GeV2ToPb];',
                    'Print["NUMERICAL_RESULT[sigma_fb]: ", sigmaNum * GeV2ToPb * 1000];',
                    'Print["NUMERICAL_RESULT[sigma_nb]: ", sigmaNum * GeV2ToPb / 1000];',
                ])

        lines.append('Print["STATUS: complete"];')

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Channel inference
    # ------------------------------------------------------------------

    def _infer_channel(self, diagram: Diagram, proc: ProcessType) -> Channel:
        """Resolve the s/t/u channel for a 2->2 diagram.

        The channel is a PROPERTY OF THE DIAGRAM, not something a topology
        spec determines: the same four external legs and the same mediator
        describe three different diagrams. The old heuristic returned
        ``Channel.S`` unconditionally, so every t- and u-channel request was
        silently computed as an s-channel one.

        Resolution order: an explicit ``channel`` on the generator, then a
        ``channel`` key carried by the diagram's topology string, then
        s-channel with a recorded assumption.
        """
        if proc != ProcessType.SCATTERING_2TO2:
            return Channel.S

        if not diagram.propagators:
            return Channel.CONTACT

        explicit = self.channel
        if explicit is None:
            topo = (diagram.topology or "").lower()
            for name in ("t_channel", "u_channel", "s_channel"):
                if name in topo:
                    explicit = name[0]
                    break

        if explicit is not None:
            key = str(explicit).strip().lower()
            try:
                return {"s": Channel.S, "t": Channel.T, "u": Channel.U,
                        "contact": Channel.CONTACT}[key]
            except KeyError:
                raise UnsupportedScattering(
                    f"Unknown scattering channel {explicit!r}; "
                    "expected one of s, t, u, contact."
                )

        self._channel_assumed = True
        return Channel.S

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_script(self, sections: List[str]) -> str:
        code = "\n".join(sections)
        _guard_emitted_code(code)
        return code


# ---------------------------------------------------------------------------
# Symbolic code generator (masses/couplings left as symbols)
# ---------------------------------------------------------------------------

class SymbolicFeynCalcCodeGenerator(FeynCalcCodeGenerator):
    """
    Generate FeynCalc scripts with symbolic (unresolved) masses and couplings.

    The amplitude construction methods already use mass *symbols* like mH, mb
    internally — this subclass simply avoids binding them to numerical values.

    ``values`` is the one exception, and it is applied LAST rather than at
    definition time: the script still derives the observable in closed form
    and prints it, then substitutes and prints the number. Binding masses up
    front (what the numeric base class does) would give a number but destroy
    the formula, and the formula is usually the point of asking symbolically.
    Passing ``values`` therefore yields BOTH from a single run, which is what
    ComputeDecayRate / ComputeCrossSection need to answer in one call.
    """

    def __init__(self, *args, values: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = values or None

    def _mass_definitions(self, diagram: Diagram) -> str:
        """Emit mass symbols without numerical assignments."""
        all_particles: List[Tuple[Particle, int]] = []
        for i, p in enumerate(diagram.initial):
            all_particles.append((p, i))
        for i, p in enumerate(diagram.final):
            all_particles.append((p, i))

        seen = set()
        symbols = []
        for p, idx in all_particles:
            sym = _mass_symbol(p, idx)
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)

        for i, prop in enumerate(diagram.propagators):
            symbols.append(f"mProp{i}")

        lines = [
            "(* Mass definitions — symbolic (no numerical values assigned) *)",
            f"(* Masses: {', '.join(symbols)} *)",
        ]
        return "\n".join(lines) + "\n"

    def _numerical_eval(self, diagram: Diagram, proc: ProcessType,
                        sqrt_s: Optional[float] = None) -> str:
        """Emit only symbolic and LaTeX results, skip numerical evaluation."""
        lines = ["(* Step 7: Symbolic extraction (no numerical evaluation) *)"]

        lines.extend([
            '',
            '(* Squared amplitude after kinematics *)',
            'Print["SYMBOLIC_RESULT[ampSq]: ", ampSqKin];',
            'Print["LATEX_RESULT[ampSq]: ", ToString[TeXForm[ampSqKin]]];',
        ])

        if proc in (ProcessType.DECAY_1TO2, ProcessType.DECAY_1TO2_1PROP):
            result_var = "width"
            lines.extend([
                '',
                '(* Decay width — symbolic only *)',
                'Print["SYMBOLIC_RESULT[width]: ", width];',
                'Print["LATEX_RESULT[width]: ", ToString[TeXForm[width]]];',
            ])
        else:
            result_var = "sigma"
            lines.extend([
                '',
                '(* Cross section — symbolic only *)',
                'Print["SYMBOLIC_RESULT[sigma]: ", sigma];',
                'Print["LATEX_RESULT[sigma]: ", ToString[TeXForm[sigma]]];',
                'Print["SYMBOLIC_RESULT[dSigmaDt]: ", dSigmaDt];',
            ])

        # In-script simplifications (optional)
        if self.simplifications:
            s = self.simplifications
            sv = f"{result_var}Simplified"
            lines.extend(['', f'(* Post-computation simplifications *)'])
            lines.append(f'{sv} = {result_var};')
            if s.get("substitutions"):
                rules = ", ".join(
                    f"{k} -> {v}" for k, v in s["substitutions"].items()
                )
                lines.append(f'{sv} = {sv} /. {{{rules}}};')
            if s.get("limit"):
                lim = s["limit"]
                var = lim.get("var", "x")
                point = lim.get("point", "0")
                direction = lim.get("direction")
                if direction:
                    lines.append(
                        f'{sv} = Limit[{sv}, {var} -> {point}, Direction -> {direction}];'
                    )
                else:
                    lines.append(f'{sv} = Limit[{sv}, {var} -> {point}];')
            if s.get("series"):
                ser = s["series"]
                var = ser.get("var", "eps")
                point = ser.get("point", "0")
                order = ser.get("order", 1)
                lines.append(f'{sv} = Normal[Series[{sv}, {{{var}, {point}, {order}}}]];')
            sfn = s.get("simplify", "Simplify")
            if sfn != "None":
                if s.get("assumptions"):
                    assumptions_str = ", ".join(s["assumptions"])
                    lines.append(
                        f'{sv} = Assuming[{{{assumptions_str}}}, {sfn}[{sv}]];'
                    )
                else:
                    lines.append(f'{sv} = {sfn}[{sv}];')
            lines.append(
                f'Print["SYMBOLIC_RESULT[{result_var}_simplified]: ", {sv}];'
            )
            lines.append(
                f'Print["LATEX_RESULT[{result_var}_simplified]: ", ToString[TeXForm[{sv}]]];'
            )

        # Numeric substitution LAST, so the closed form above is untouched.
        # `result_var` is whichever observable this process type produced, and
        # the simplified variant (if simplifications ran) is preferred as the
        # substitution target since it is the tidier equivalent expression.
        if self.values:
            target = f"{result_var}Simplified" if self.simplifications else result_var
            rules = ", ".join(
                f"{k} -> {_fmt_mma(v)}" for k, v in self.values.items()
            )
            lines.extend([
                '',
                '(* Numeric evaluation — substitute into the closed form *)',
                f'{result_var}Num = N[{target} /. {{{rules}}}];',
            ])
            if result_var == "width":
                lines.extend([
                    'Print["NUMERICAL_RESULT[width_GeV]: ", widthNum];',
                    'Print["NUMERICAL_RESULT[width_MeV]: ", widthNum * 1000];',
                ])
            else:
                # A well-formed symbolic sigma can still evaluate to Undefined
                # at the substitution point when the t-endpoints cancel -- the
                # massless-fermion s-channel case does exactly that. The base
                # class guards this; the symbolic path must too, or the caller
                # gets an unevaluated expression instead of a number and no
                # indication that a perfectly good fallback existed.
                # sigmaNIntegrate wants ALL the substitutions, not just s.
                lines.extend([
                    'If[!NumericQ[sigmaNum] || !FreeQ[sigmaNum, Undefined] ||',
                    '   !FreeQ[sigmaNum, Indeterminate] || !FreeQ[sigmaNum, DirectedInfinity],',
                    f'  sigmaNum = sigmaNIntegrate[{{{rules}}}]];',
                    'Print["NUMERICAL_RESULT[sigma_GeV2]: ", sigmaNum];',
                    GEV2_TO_BARN_COMMENT,
                    'Print["NUMERICAL_RESULT[sigma_pb]: ", sigmaNum * GeV2ToPb];',
                    'Print["NUMERICAL_RESULT[sigma_fb]: ", sigmaNum * GeV2ToPb * 1000];',
                ])

        lines.append('Print["STATUS: complete"];')
        return "\n".join(lines) + "\n"
