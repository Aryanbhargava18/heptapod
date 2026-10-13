"""
# scattering.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tree-level 2 -> 2 scattering: coverage, amplitude assembly, cross section.

WHY THIS MODULE EXISTS.  The 2->2 path used to consist of one hand-written
special case (all-fermion externals with a vector mediator, s- and
t-channel only) plus a fallback that emitted ``amp = I g^2 FAD[...]`` --
no spinors, no polarisation vectors, no vertex structure -- for every
other process.  That fallback did not raise; it produced a well-formed
script and a confident, meaningless number.  This module replaces it with
a compositional builder that covers the full space and refuses, loudly,
outside it.

THE SPACE IS CLOSED AND SMALL.  Enumerating over heptapod's own
``VALID_3PT_VERTICES`` / ``VALID_4PT_VERTICES`` gives 20 distinct external
spin classes for 2->2: 112 single-diagram (class, channel, mediator)
exchange structures plus 6 contact classes.  Those 118 topologies do NOT
need 118 code paths, because the case analysis factorises into small
independent tables:

    external wavefunction (spin 0 / 1/2 / 1, in|out, particle|anti)   3 cases
    vertex factor          reuse the existing library                 0 new
    propagator numerator   (1) / (q-slash + m) / (-g + q q / M^2)     3 cases
    denominator            1/(q^2 - M^2), q^2 in {s, t, u}            1 case
    index threading        bosonic mediator vs fermionic mediator     2 cases

CONVENTIONS (fixed here, asserted by the numeric harness):
  * Metric (+,-,-,-); FeynCalc default.
  * p1, p2 incoming; p3, p4 outgoing; all four on shell.
  * s = (p1+p2)^2, t = (p1-p3)^2, u = (p1-p4)^2, s + t + u = sum_i m_i^2.
    ``u`` is eliminated in favour of (s, t) EVERYWHERE, so the generated
    script has exactly two independent kinematic variables and the
    t-integration is well posed.
  * Vertex factors use the ALL-INCOMING momentum convention: a leg whose
    physical momentum flows out of the vertex enters as -p.
  * The assembled amplitude is normalised to 4 dimensions with
    ``ChangeDimension[amp, 4]`` before squaring.  Tree level never needs
    D dimensions, and mixing 4-dim spinors / polarisation vectors with
    D-dim Dirac algebra is precisely what FeynCalc rejects
    (``DiracTrace::mixmsg``).  Normalising once, centrally, lets the
    existing (D-dimensional) vertex library be reused verbatim.
  * Propagator denominators are PLAIN SCALARS ``1/(q^2 - M^2)``.  ``FAD``
    is a loop-integral denominator and carries D-4 pieces; it has no place
    in a tree amplitude.
  * sigma = (1/S) (1/N_init) C * Int |M|^2 / (16 pi lambda(s,m1^2,m2^2)) dt
    with S the identical-final-particle symmetry factor, N_init the
    initial-state spin/polarisation multiplicity, C the colour factor and
    lambda the Kallen function.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product as _iproduct
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tools.nda.simple_diagram import (
    Diagram,
    Particle,
    Propagator,
    Vertex,
    compute_symmetry_factor,
)
from tools.nda.lorentz_validation import VALID_3PT_VERTICES, VALID_4PT_VERTICES

from .fc_symbols import _coupling_value, _fmt_mma, _is_antiparticle, _mass_symbol


class Channel(Enum):
    """Scattering channel for 2->2 processes."""
    S = auto()
    T = auto()
    U = auto()
    CONTACT = auto()


#: Which external legs meet at each vertex, per channel.  Legs are indexed
#: 0..3 as (p1, p2, p3, p4) = (in, in, out, out).
CHANNEL_PAIRING: Dict[Channel, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    Channel.S: ((0, 1), (2, 3)),
    Channel.T: ((0, 2), (1, 3)),
    Channel.U: ((0, 3), (1, 2)),
}

#: Momentum symbol per leg index.
LEG_MOMENTA = ("p1", "p2", "p3", "p4")

#: True when the leg's physical momentum flows into the diagram.
LEG_INCOMING = (True, True, False, False)

#: The Mandelstam invariant carried by the propagator in each channel.
CHANNEL_INVARIANT = {Channel.S: "s", Channel.T: "t", Channel.U: "uExpr"}

PROP_MASS = "mProp0"

#: Seconds allowed for the symbolic t-integration before falling through to
#: the numeric route. Unbounded Integrate is not a theoretical concern: a
#: coherent sum with massless-mediator poles (Moller, Bhabha) grinds for
#: >10 minutes and would consume an agent's entire wall-clock budget.
DEFAULT_INTEGRATE_TIMEOUT_S = 120


class UnsupportedTopology(ValueError):
    """Raised when a requested structure is outside the covered space.

    Deliberately an exception rather than a warning-plus-nonsense: a
    silently wrong amplitude is worse than a refusal.
    """


#: Former name, kept so existing callers and tests keep working.
UnsupportedScattering = UnsupportedTopology


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _multiset(spins: Sequence[float]):
    return frozenset(Counter(float(s) for s in spins).items())


def is_valid_3pt(spins: Sequence[float]) -> bool:
    return _multiset(spins) in VALID_3PT_VERTICES


def is_valid_4pt(spins: Sequence[float]) -> bool:
    return _multiset(spins) in VALID_4PT_VERTICES


def allowed_exchange_structures() -> Set[Tuple[Tuple[float, ...], str, float]]:
    """Every (external spins, channel, mediator spin) with two valid vertices.

    External spins are ordered (p1, p2, p3, p4).  Returned for coverage
    reporting and to drive the parametrised tests.
    """
    out: Set[Tuple[Tuple[float, ...], str, float]] = set()
    spins = (0.0, 0.5, 1.0)
    for ext in _iproduct(spins, repeat=4):
        for chan, ((a, b), (c, d)) in CHANNEL_PAIRING.items():
            for med in spins:
                if is_valid_3pt([ext[a], ext[b], med]) and is_valid_3pt([ext[c], ext[d], med]):
                    out.add((ext, chan.name, med))
    return out


def allowed_contact_structures() -> Set[Tuple[float, ...]]:
    """Every external spin assignment admitting a valid 4-point vertex."""
    spins = (0.0, 0.5, 1.0)
    return {ext for ext in _iproduct(spins, repeat=4) if is_valid_4pt(ext)}


def external_spins(diagram: Diagram) -> Tuple[float, ...]:
    legs = list(diagram.initial) + list(diagram.final)
    return tuple(float(p.spin or 0.0) for p in legs)


def check_supported(diagram: Diagram, channel: Channel) -> None:
    """Raise :class:`UnsupportedScattering` if this structure is not covered."""
    ext = external_spins(diagram)
    if len(ext) != 4:
        raise UnsupportedScattering(
            f"2->2 needs exactly 4 external legs, got {len(ext)}."
        )

    if channel is Channel.CONTACT:
        if not is_valid_4pt(ext):
            raise UnsupportedScattering(
                f"External spins {ext} do not form a valid 4-point vertex. "
                f"Valid contact classes: SSSS, VVVV, FFFF, SSVV."
            )
        return

    med = diagram.propagators[0].spin if diagram.propagators else None
    if med is None:
        raise UnsupportedScattering(
            "An exchange diagram needs a propagator with a declared spin."
        )
    (a, b), (c, d) = CHANNEL_PAIRING[channel]
    if not is_valid_3pt([ext[a], ext[b], med]):
        raise UnsupportedScattering(
            f"Vertex ({_spin_name(ext[a])}, {_spin_name(ext[b])}, "
            f"{_spin_name(med)}) is not a valid 3-point vertex, so the "
            f"{channel.name.lower()}-channel diagram does not exist."
        )
    if not is_valid_3pt([ext[c], ext[d], med]):
        raise UnsupportedScattering(
            f"Vertex ({_spin_name(ext[c])}, {_spin_name(ext[d])}, "
            f"{_spin_name(med)}) is not a valid 3-point vertex, so the "
            f"{channel.name.lower()}-channel diagram does not exist."
        )


def _spin_name(s: float) -> str:
    return {0.0: "S", 0.5: "F", 1.0: "V"}.get(float(s), f"spin-{s}")


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

@dataclass
class VLeg:
    """One line attached to a vertex, with its all-incoming momentum."""
    particle: Optional[Particle]
    momentum: str            # bare physical momentum symbol, or "q"
    flows_in: bool           # True if the physical momentum flows into the vertex
    index: Optional[str]     # Lorentz index, for spin-1 lines
    spin: float
    is_mediator: bool = False
    leg_slot: Optional[int] = None   # 0..3 for external legs

    @property
    def p_in(self) -> str:
        """Momentum entering the vertex, as a Mathematica expression."""
        return self.momentum if self.flows_in else f"(-{self.momentum})"

    @property
    def mass(self) -> str:
        if self.is_mediator:
            return PROP_MASS
        return _mass_symbol(self.particle, self.leg_slot or 0)


def _external_legs(diagram: Diagram) -> List[VLeg]:
    parts = list(diagram.initial) + list(diagram.final)
    legs: List[VLeg] = []
    for i, p in enumerate(parts):
        spin = float(p.spin or 0.0)
        legs.append(
            VLeg(
                particle=p,
                momentum=LEG_MOMENTA[i],
                flows_in=LEG_INCOMING[i],
                index=f"mu{i + 1}" if spin == 1.0 else None,
                spin=spin,
                leg_slot=i,
            )
        )
    return legs


def _spinor_for(leg: VLeg) -> Tuple[str, bool]:
    """FeynCalc spinor for an external fermion leg, and whether it is barred.

    Incoming particle -> SpinorU, incoming antiparticle -> SpinorVBar,
    outgoing particle -> SpinorUBar, outgoing antiparticle -> SpinorV.
    Barred spinors sit at the LEFT end of a chain.
    """
    anti = _is_antiparticle(leg.particle.label if leg.particle else None)
    m = leg.mass
    if leg.flows_in:
        return (f"SpinorVBar[{leg.momentum}, {m}]", True) if anti else \
               (f"SpinorU[{leg.momentum}, {m}]", False)
    return (f"SpinorV[{leg.momentum}, {m}]", False) if anti else \
           (f"SpinorUBar[{leg.momentum}, {m}]", True)


# ---------------------------------------------------------------------------
# Vertex factors
# ---------------------------------------------------------------------------

@dataclass
class VertexExpr:
    """A vertex, split into its commuting part and its Dirac part.

    ``dirac`` is None for purely bosonic vertices; when present it must be
    placed inside a spinor chain rather than multiplied in.
    """
    prefactor: str = "1"
    dirac: Optional[str] = None


def _mom_sum(legs: Sequence[VLeg]) -> str:
    """All-incoming momentum sum over legs, e.g. 'p1 - p3'."""
    terms = [(leg.momentum if leg.flows_in else f"-{leg.momentum}") for leg in legs]
    out = " + ".join(terms).replace("+ -", "- ")
    return out


def _vtype(vertex: Optional[Vertex]) -> str:
    return (vertex.type.lower() if vertex else "").replace("-", "").replace("_", "")


def vertex_factor(gen, vertex: Optional[Vertex], legs: List[VLeg],
                  couplings: Dict[str, float]) -> VertexExpr:
    """Build the factor for one 3-point vertex from its three attached lines.

    Dispatch is on the multiset of spins, so a vertex is built the same way
    whether a given line is external or the mediator -- that uniformity is
    what makes the 118 topologies collapse onto six vertex types.
    """
    g = _coupling_value(vertex, couplings) if vertex else "g"
    vt = _vtype(vertex)
    spins = sorted(leg.spin for leg in legs)

    if spins == [0.0, 0.0, 0.0]:
        return VertexExpr(prefactor=f"I ({g})")

    if spins == [0.0, 0.0, 1.0]:
        # SSV: i g (pA - pB)^mu, both scalars incoming.  The A/B ordering
        # fixes an overall sign; it is taken in leg order (mediator last)
        # so repeated builds of the same diagram agree.
        scalars = [l for l in legs if l.spin == 0.0]
        vec = next(l for l in legs if l.spin == 1.0)
        a, b = scalars[0], scalars[1]
        return VertexExpr(
            prefactor=f"I ({g}) FVD[{a.p_in} - {b.p_in}, {vec.index}]"
        )

    if spins == [0.0, 1.0, 1.0]:
        # SVV: i g g^{mu nu} contracting the two vector lines.
        vecs = [l for l in legs if l.spin == 1.0]
        return VertexExpr(
            prefactor=f"I ({g}) MTD[{vecs[0].index}, {vecs[1].index}]"
        )

    if spins == [1.0, 1.0, 1.0]:
        # CONVENTION: ALL-INCOMING momenta, k1+k2+k3 = 0 -- the textbook
        # statement, and the one the decay path now shares. Verified against
        # the Z' -> W+W- width at every mass ratio. See
        # VVV_CONVENTION_NOTE at the bottom of this module.
        # VVV triple gauge:
        #   g [ g^{m1 m2}(k1-k2)^{m3} + g^{m2 m3}(k2-k3)^{m1}
        #                             + g^{m3 m1}(k3-k1)^{m2} ]
        v1, v2, v3 = legs
        m1, m2, m3 = v1.index, v2.index, v3.index
        k1, k2, k3 = v1.p_in, v2.p_in, v3.p_in
        return VertexExpr(
            prefactor=(
                f"I ({g}) ("
                f"MTD[{m1}, {m2}] FVD[{k1} - {k2}, {m3}] + "
                f"MTD[{m2}, {m3}] FVD[{k2} - {k3}, {m1}] + "
                f"MTD[{m3}, {m1}] FVD[{k3} - {k1}, {m2}])"
            )
        )

    if spins == [0.0, 0.5, 0.5]:
        # SFF -- reuse the validated decay-side vertex library.
        return VertexExpr(dirac=gen._sff_coupling_structure(vt, g))

    if spins == [0.5, 0.5, 1.0]:
        # VFF -- likewise; the vector line supplies the Lorentz index and,
        # for tensor/dipole structures, its momentum.
        vec = next(l for l in legs if l.spin == 1.0)
        return VertexExpr(
            dirac=gen._vff_gamma_structure(vt, g, vec.index, vec_momentum=vec.momentum)
        )

    raise UnsupportedScattering(
        f"No vertex factor for spin configuration {spins}."
    )


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

def _is_massless(mass) -> bool:
    return isinstance(mass, (int, float)) and float(mass) == 0.0


def propagator_numerator(prop: Propagator, q_expr: str,
                         mu: str, nu: str) -> str:
    """Numerator of the tree propagator for spin 0, 1/2 or 1.

    The 1/2 numerator is returned for completeness but is threaded into the
    spinor chain by the caller, not multiplied in.
    """
    spin = float(prop.spin if prop.spin is not None else 0.0)
    if spin == 0.0:
        return "1"
    if spin == 0.5:
        return f"(GSD[{q_expr}] + {PROP_MASS})"
    if spin == 1.0:
        if _is_massless(prop.mass):
            return f"(-MTD[{mu}, {nu}])"
        return (
            f"(-MTD[{mu}, {nu}] + FVD[{q_expr}, {mu}] FVD[{q_expr}, {nu}]"
            f"/{PROP_MASS}^2)"
        )
    raise UnsupportedScattering(
        f"No tree propagator for mediator spin {spin}."
    )


def propagator_denominator(prop: Propagator, channel: Channel) -> str:
    """i / (q^2 - M^2), with q^2 the channel invariant.

    The explicit ``I`` matters only once amplitudes are SUMMED: every
    propagator carries it, so the relative phase between two exchange
    diagrams survives either way, but between a contact diagram (no
    propagator) and an exchange diagram it does not. Carrying it here keeps
    the amplitude absolutely normalised, which is what a coherent sum needs.
    It has no effect on a single diagram, where |i| = 1.
    """
    inv = CHANNEL_INVARIANT[channel]
    if _is_massless(prop.mass):
        return f"I/({inv})"
    return f"I/({inv} - {PROP_MASS}^2)"


# ---------------------------------------------------------------------------
# Amplitude assembly
# ---------------------------------------------------------------------------

def _pol_factors(legs: Sequence[VLeg]) -> str:
    return " ".join(
        f"PolarizationVector[{l.momentum}, {l.index}]"
        for l in legs if l.spin == 1.0
    )


def _chain(bar: str, dirac: str, nonbar: str) -> str:
    return f"{bar} . ({dirac}) . {nonbar}"


def _split_fermion_pair(pair: Sequence[VLeg]) -> Tuple[str, str]:
    """Return (barred spinor, unbarred spinor) for a two-fermion vertex."""
    exprs = [(_spinor_for(l)) for l in pair]
    barred = [e for e, b in exprs if b]
    unbarred = [e for e, b in exprs if not b]
    if len(barred) != 1 or len(unbarred) != 1:
        labels = [l.particle.label if l.particle else "?" for l in pair]
        direction = ["incoming" if l.flows_in else "outgoing" for l in pair]
        raise UnsupportedTopology(
            f"No vertex joins {labels[0]} ({direction[0]}) and {labels[1]} "
            f"({direction[1]}): fermion number is not conserved there, so "
            "this channel's diagram does not exist. Two fermions meet at a "
            "vertex only as a particle-antiparticle pair (both incoming, or "
            "both outgoing) or as one line passing through (one incoming, "
            "one outgoing). e- e- -> e- e- for instance has t- and "
            "u-channel diagrams but no s-channel one. Check the channel, and "
            "the particle/antiparticle labelling (a 'bar'/'~'/'+' suffix "
            "marks an antiparticle)."
        )
    return barred[0], unbarred[0]


def _vertex_expr(gen, vertex, legs_at_vertex, couplings) -> str:
    """One vertex as a single multiplicative expression.

    A vertex carrying two external fermions becomes a closed spinor chain;
    everything else is a commuting factor.
    """
    vf = vertex_factor(gen, vertex, legs_at_vertex, couplings)
    if vf.dirac is None:
        return vf.prefactor
    fermions = [l for l in legs_at_vertex if l.spin == 0.5 and not l.is_mediator]
    if len(fermions) != 2:
        raise UnsupportedScattering(
            "Internal error: a Dirac vertex structure reached a vertex with "
            f"{len(fermions)} external fermions."
        )
    bar, nonbar = _split_fermion_pair(fermions)
    return _chain(bar, vf.dirac, nonbar)


def _exchange_bosonic(gen, diagram: Diagram, legs: List[VLeg],
                      channel: Channel, prop: Propagator,
                      amp_var: str = "amp", q_var: str = "qMom") -> List[str]:
    (a, b), (c, d) = CHANNEL_PAIRING[channel]
    A_ext, B_ext = [legs[a], legs[b]], [legs[c], legs[d]]
    med_spin = float(prop.spin if prop.spin is not None else 0.0)

    q_expr = _mom_sum(A_ext)
    # The mediator leaves vertex A and enters vertex B.
    medA = VLeg(None, q_var, flows_in=False,
                index="muP" if med_spin == 1.0 else None,
                spin=med_spin, is_mediator=True)
    medB = VLeg(None, q_var, flows_in=True,
                index="nuP" if med_spin == 1.0 else None,
                spin=med_spin, is_mediator=True)

    v0 = diagram.vertices[0] if diagram.vertices else None
    v1 = diagram.vertices[1] if len(diagram.vertices) >= 2 else v0

    exprA = _vertex_expr(gen, v0, A_ext + [medA], diagram.couplings)
    exprB = _vertex_expr(gen, v1, B_ext + [medB], diagram.couplings)
    num = propagator_numerator(prop, q_var, "muP", "nuP")
    den = propagator_denominator(prop, channel)
    pol = _pol_factors(legs)

    parts = [f"({exprA})", f"({num})", f"({exprB})", f"({den})"]
    if pol:
        parts.append(pol)
    return [f"{q_var} = {q_expr};",
            f"{amp_var} = " + " ".join(parts) + ";"]


def _exchange_fermionic(gen, diagram: Diagram, legs: List[VLeg],
                        channel: Channel, prop: Propagator,
                        amp_var: str = "amp", q_var: str = "qMom") -> List[str]:
    """Mediator is a fermion: the propagator sits INSIDE the spinor chain."""
    (a, b), (c, d) = CHANNEL_PAIRING[channel]
    A_ext, B_ext = [legs[a], legs[b]], [legs[c], legs[d]]

    def _split(pair):
        f = [l for l in pair if l.spin == 0.5]
        boson = [l for l in pair if l.spin != 0.5]
        if len(f) != 1 or len(boson) != 1:
            raise UnsupportedScattering(
                "A fermion-mediated vertex needs exactly one external fermion "
                "and one external boson."
            )
        return f[0], boson[0]

    fA, _ = _split(A_ext)
    fB, _ = _split(B_ext)
    sA, barA = _spinor_for(fA)
    sB, barB = _spinor_for(fB)
    if barA == barB:
        raise UnsupportedScattering(
            "The two external fermions must sit at opposite ends of the "
            "fermion line (one barred, one unbarred). Got "
            f"{fA.particle.label!r} and {fB.particle.label!r}; check the "
            "particle/antiparticle labelling."
        )

    v0 = diagram.vertices[0] if diagram.vertices else None
    v1 = diagram.vertices[1] if len(diagram.vertices) >= 2 else v0

    # The barred spinor sits at the LEFT end; momentum flows along the arrow
    # from the right vertex to the left one.
    if barA:
        left_ext, left_v, left_spinor = A_ext, v0, sA
        right_ext, right_v, right_spinor = B_ext, v1, sB
    else:
        left_ext, left_v, left_spinor = B_ext, v1, sB
        right_ext, right_v, right_spinor = A_ext, v0, sA

    q_expr = _mom_sum(right_ext)
    medR = VLeg(None, q_var, flows_in=False, index=None, spin=0.5, is_mediator=True)
    medL = VLeg(None, q_var, flows_in=True, index=None, spin=0.5, is_mediator=True)

    vfL = vertex_factor(gen, left_v, left_ext + [medL], diagram.couplings)
    vfR = vertex_factor(gen, right_v, right_ext + [medR], diagram.couplings)
    if vfL.dirac is None or vfR.dirac is None:
        raise UnsupportedScattering(
            "A fermion-mediated diagram needs a Dirac structure at both "
            "vertices."
        )

    num = f"(GSD[{q_var}] + {PROP_MASS})"
    den = propagator_denominator(prop, channel)
    pol = _pol_factors(legs)

    chain = (
        f"{left_spinor} . ({vfL.dirac}) . {num} . ({vfR.dirac}) . {right_spinor}"
    )
    parts = [f"({chain})", f"({den})"]
    if pol:
        parts.append(pol)
    return [f"{q_var} = {q_expr};",
            f"{amp_var} = " + " ".join(parts) + ";"]


def _contact_amplitude(gen, diagram: Diagram, legs: List[VLeg],
                       amp_var: str = "amp") -> List[str]:
    vertex = diagram.vertices[0] if diagram.vertices else None
    g = _coupling_value(vertex, diagram.couplings) if vertex else "g"
    vt = _vtype(vertex)
    ms = _multiset(external_spins(diagram))

    if ms == _multiset([0.0, 0.0, 0.0, 0.0]):
        return [f"{amp_var} = I ({g});"]

    if ms == _multiset([0.0, 0.0, 1.0, 1.0]):
        vecs = [l for l in legs if l.spin == 1.0]
        pol = _pol_factors(legs)
        return [
            f"{amp_var} = I ({g}) MTD[{vecs[0].index}, {vecs[1].index}] {pol};"
        ]

    if ms == _multiset([0.5, 0.5, 0.5, 0.5]):
        return _contact_ffff(diagram, legs, vertex, g, amp_var=amp_var)

    if ms == _multiset([1.0, 1.0, 1.0, 1.0]):
        return _contact_vvvv(diagram, legs, vertex, g)

    raise UnsupportedTopology(
        f"No contact amplitude for external spins {external_spins(diagram)}."
    )


#: The three independent Lorentz structures of a four-vector contact vertex.
#: Any such vertex built from metric tensors is a combination of these, so
#: taking them as the basis is fully general -- and it keeps the colour
#: algebra out of a diagram spec that has no way to express it.
VVVV_STRUCTURES = ("a", "b", "c")


def _contact_vvvv(diagram: Diagram, legs: List[VLeg],
                  vertex: Optional[Vertex], g) -> List[str]:
    """Four-vector contact vertex in the explicit three-structure basis.

    CONVENTION (chosen, not inferred):

        V^{m1 m2 m3 m4} = i [ a eta^{m1 m2} eta^{m3 m4}
                            + b eta^{m1 m3} eta^{m2 m4}
                            + c eta^{m1 m4} eta^{m2 m3} ]

    with legs numbered in the order (p1, p2, p3, p4) = (in, in, out, out).
    Those three products span every Lorentz structure a four-vector contact
    vertex can have, so nothing is lost by taking them as the basis.

    WHY THE CALLER SUPPLIES a, b, c. The Yang-Mills quartic is

        -i g^2 [ f^{abe}f^{cde}(eta^{m1 m3}eta^{m2 m4} - eta^{m1 m4}eta^{m2 m3})
               + f^{ace}f^{bde}(eta^{m1 m2}eta^{m3 m4} - eta^{m1 m4}eta^{m2 m3})
               + f^{ade}f^{bce}(eta^{m1 m2}eta^{m3 m4} - eta^{m1 m3}eta^{m2 m4}) ]

    -- three DIFFERENT colour contractions multiplying three different
    Lorentz structures. A diagram spec carrying one scalar `color_factor`
    cannot say which is which, so the tool asks for the already-contracted
    coefficients rather than guessing a gauge group. For pure SU(N) Yang-Mills
    contract the structure constants yourself and pass

        a = -i g^2 (f^{ace}f^{bde} + f^{ade}f^{bce})
        b = -i g^2 (f^{abe}f^{cde} - f^{ade}f^{bce})
        c = -i g^2 (-f^{abe}f^{cde} - f^{ace}f^{bde})

    For an effective operator such as (lambda/4)(V.V)^2 the fully symmetric
    choice a = b = c = lambda is what you want.
    """
    m1, m2, m3, m4 = (l.index for l in legs)
    pol = _pol_factors(legs)

    if isinstance(g, dict):
        missing = [k for k in VVVV_STRUCTURES if k not in g]
        if missing:
            raise UnsupportedTopology(
                f"A four-vector contact vertex needs coefficients "
                f"{list(VVVV_STRUCTURES)}; missing {missing}. See "
                "_contact_vvvv for the basis and the Yang-Mills mapping."
            )
        a, b, c = (g[k] for k in VVVV_STRUCTURES)
    else:
        vt = _vtype(vertex)
        if vt in ("symmetric", "contact", "", "quartic"):
            # (V.V)^2-type effective operator: fully symmetric.
            a = b = c = g
        else:
            raise UnsupportedTopology(
                "A four-vector contact vertex is not fixed by a single "
                f"coupling and type {vertex.type!r}: it has THREE independent "
                "Lorentz structures, and the Yang-Mills quartic multiplies "
                "each by a different colour contraction that a scalar "
                "color_factor cannot express. Pass coupling as "
                "{'a': ..., 'b': ..., 'c': ...} (see _contact_vvvv for the "
                "basis and the Yang-Mills mapping), or use type 'symmetric' "
                "for a (V.V)^2 operator where a = b = c."
            )

    return [
        f"amp = I (({a}) MTD[{m1}, {m2}] MTD[{m3}, {m4}] "
        f"+ ({b}) MTD[{m1}, {m3}] MTD[{m2}, {m4}] "
        f"+ ({c}) MTD[{m1}, {m4}] MTD[{m2}, {m3}]) {pol};"
    ]


#: Dirac structures for a four-fermion bilinear, as (expression, rank).
#: Rank-0 structures carry no Lorentz index; rank-1 ones share the index
#: muC between the two bilinears, which is what makes the operator a scalar.
#: GA[5] = gamma_5, GA[7] = P_L, GA[6] = P_R in FeynCalc.
FOUR_FERMION_STRUCTURES: Dict[str, Tuple[str, int]] = {
    "S": ("1", 0),                    # scalar        1 (x) 1
    "P": ("GA[5]", 0),                # pseudoscalar  g5 (x) g5
    "V": ("GAD[muC]", 1),             # vector        g^mu (x) g_mu
    "A": ("GAD[muC].GA[5]", 1),       # axial         g^mu g5 (x) g_mu g5
    "L": ("GAD[muC].GA[7]", 1),       # left-handed   g^mu P_L (x) g_mu P_L
    "R": ("GAD[muC].GA[6]", 1),       # right-handed  g^mu P_R (x) g_mu P_R
}


def _contact_ffff(diagram: Diagram, legs: List[VLeg],
                  vertex: Optional[Vertex], g,
                  amp_var: str = "amp") -> List[str]:
    """Four-fermion contact operator.

    A four-fermion operator is NOT determined by "four fermion legs plus a
    type name". Three things must come from the operator itself, and the
    previous implementation guessed all three:

    1. WHICH LEGS FORM WHICH BILINEAR. It hardcoded (p1 p2)(p3 p4), i.e.
       (in,in)(out,out) -- while its own comment said the pairing "is the
       operator's own definition, not something the tool can infer".
       (psibar_e G psi_e)(psibar_mu G psi_mu) driving e mu -> e mu pairs
       (in_e, out_e)(in_mu, out_mu) = legs (0,2)(1,3) instead. Worse, the
       hardcoded choice forces the incoming pair to be
       particle-antiparticle, so e- e- -> e- e- could not be expressed at
       all: it died in _split_fermion_pair on a fermion-number message.

    2. THE DIRAC STRUCTURE ON EACH BILINEAR, INDEPENDENTLY. It forced the
       same structure on both, so (V-A) (x) (V+A) was inexpressible, and it
       offered three crude options against the standard S, P, V, A, L, R
       basis.

    3. BOTH CONTRACTIONS, WHEN THE FERMIONS ARE IDENTICAL. Only one pairing
       was ever emitted. For identical fermions the second (exchange)
       contraction contributes too, with a relative minus -- the same class
       of omission as a missing exchange diagram, and just as silently
       wrong. That case is refused here rather than half-computed.

    The two structures must share a rank: a rank-1 bilinear contracts its
    index against the other, so pairing S with V is not a Lorentz scalar.
    """
    struct = list(vertex.structures) if vertex is not None and vertex.structures else None
    if struct is None:
        vt = _vtype(vertex)
        legacy = {"vector": ["V", "V"], "gaugevector": ["V", "V"],
                  "currentcurrent": ["V", "V"], "scalar": ["S", "S"],
                  "yukawa": ["S", "S"], "lefthanded": ["L", "L"],
                  "": ["V", "V"]}
        if vt not in legacy:
            raise UnsupportedTopology(
                "A four-fermion contact operator needs `structures`, e.g. "
                '"structures": ["V", "A"]. Available: '
                f"{sorted(FOUR_FERMION_STRUCTURES)} (S scalar, P pseudoscalar, "
                "V vector, A axial, L left-handed, R right-handed). The type "
                f"{vertex.type!r} does not determine one."
            )
        struct = legacy[vt]

    if len(struct) != 2:
        raise UnsupportedTopology(
            "`structures` needs exactly two entries, one per bilinear; got "
            f"{struct}."
        )
    for name in struct:
        if name not in FOUR_FERMION_STRUCTURES:
            raise UnsupportedTopology(
                f"Unknown four-fermion structure {name!r}. Available: "
                f"{sorted(FOUR_FERMION_STRUCTURES)}."
            )
    (e1, r1), (e2, r2) = (FOUR_FERMION_STRUCTURES[n] for n in struct)
    if r1 != r2:
        raise UnsupportedTopology(
            f"structures {struct} mix a rank-{r1} and a rank-{r2} bilinear, "
            "which is not a Lorentz scalar. Pair S/P with S/P, or V/A/L/R "
            "with V/A/L/R."
        )

    pairing = vertex.pairing if vertex is not None and vertex.pairing else None
    ident = identical_final_particles(diagram)
    if pairing is None:
        if ident:
            raise UnsupportedTopology(
                f"Final state has identical fermions {sorted(ident)}, so BOTH "
                "contractions of the four-fermion operator contribute, with a "
                "relative minus sign between them. Emitting one is silently "
                "wrong. Supply them as two diagrams with explicit \"pairing\" "
                "(e.g. [[0,1],[2,3]] and [[0,3],[2,1]]) and add them with "
                "relative_signs=[1, -1]."
            )
        pairing = [[0, 1], [2, 3]]

    flat = [i for pr in pairing for i in pr]
    if len(pairing) != 2 or any(len(pr) != 2 for pr in pairing) \
            or sorted(flat) != [0, 1, 2, 3]:
        raise UnsupportedTopology(
            "`pairing` must split legs 0..3 into two pairs, e.g. "
            f"[[0,1],[2,3]] or [[0,2],[1,3]]; got {pairing}."
        )

    bar1, non1 = _split_fermion_pair([legs[pairing[0][0]], legs[pairing[0][1]]])
    bar2, non2 = _split_fermion_pair([legs[pairing[1][0]], legs[pairing[1][1]]])
    return [
        f"(* four-fermion operator: ({struct[0]}) x ({struct[1]}), "
        f"pairing {pairing} *)",
        f"{amp_var} = I ({g}) ({bar1} . ({e1}) . {non1}) "
        f"({bar2} . ({e2}) . {non2});",
    ]

def build_amplitude(gen, diagram: Diagram, channel: Channel,
                    amp_var: str = "amp", q_var: str = "qMom",
                    normalise_dimension: bool = True) -> str:
    """Emit the ``amp = ...`` section for one 2->2 tree diagram.

    Raises :class:`UnsupportedScattering` rather than emitting a
    structurally empty amplitude.

    ``amp_var`` / ``q_var`` let a caller build several diagrams into
    distinct variables; :func:`build_amplitude_sum` uses that to add them
    coherently. ``normalise_dimension=False`` defers the
    ``ChangeDimension`` to the caller, which is what a sum wants: the
    normalisation is applied ONCE, to the total.
    """
    check_supported(diagram, channel)
    legs = _external_legs(diagram)

    labels_i = " ".join(p.label or "?" for p in diagram.initial)
    labels_f = " ".join(p.label or "?" for p in diagram.final)
    head = [
        f"(* Step 1: Amplitude for {labels_i} -> {labels_f} "
        f"({channel.name.lower()}-channel) *)"
    ]

    if channel is Channel.CONTACT:
        body = _contact_amplitude(gen, diagram, legs, amp_var=amp_var)
    else:
        prop = diagram.propagators[0]
        med_spin = float(prop.spin if prop.spin is not None else 0.0)
        if med_spin == 0.5:
            body = _exchange_fermionic(gen, diagram, legs, channel, prop,
                                       amp_var=amp_var, q_var=q_var)
        else:
            body = _exchange_bosonic(gen, diagram, legs, channel, prop,
                                     amp_var=amp_var, q_var=q_var)

    tail = []
    if normalise_dimension:
        tail = [
            "(* Tree level lives in 4 dimensions; normalising here lets the "
            "D-dimensional *)",
            "(* vertex library be reused without tripping DiracTrace::mixmsg. *)",
            f"{amp_var} = ChangeDimension[{amp_var}, 4];",
        ]
    return "\n".join(head + body + tail) + "\n"



def identical_final_particles(diagram: Diagram) -> Dict[str, int]:
    """Labels appearing more than once in the final state, with multiplicity."""
    counts = Counter(p.label for p in diagram.final if p.label)
    return {lab: n for lab, n in counts.items() if n > 1}


def _check_identical_particle_exchange(channels: Sequence[Channel],
                                       diagram: Diagram,
                                       relative_signs) -> None:
    """Guard the two places identical final-state particles bite.

    They are ONE problem, not two: the 1/S in the cross section is only
    correct alongside the exchange diagram, and for identical FERMIONS the
    exchange diagram enters with a relative minus sign. Getting 1/S right
    while omitting the exchange partner is a silently wrong answer, which
    is exactly what this refuses to emit.
    """
    ident = identical_final_particles(diagram)
    if not ident:
        return

    exch = {c for c in channels if c is not Channel.CONTACT}
    # t and u are the exchange pair; s does not have an identical-particle
    # partner of its own.
    if Channel.T in exch and Channel.U not in exch:
        raise UnsupportedScattering(
            f"Final state has identical particles {sorted(ident)} and the sum "
            "includes a t-channel diagram but not its u-channel exchange "
            "partner. The 1/S symmetry factor in the cross section assumes "
            "both are present; emitting one alone is silently wrong. Add the "
            "u-channel diagram, or relabel the final legs if they are not "
            "actually identical."
        )
    if Channel.U in exch and Channel.T not in exch:
        raise UnsupportedScattering(
            f"Final state has identical particles {sorted(ident)} and the sum "
            "includes a u-channel diagram but not its t-channel exchange "
            "partner. See the t-channel message."
        )

    fermionic = any(float(p.spin or 0.0) == 0.5
                    for p in diagram.final if p.label in ident)
    if fermionic and Channel.T in exch and Channel.U in exch:
        signs = list(relative_signs) if relative_signs is not None else [1] * len(channels)
        t_i = channels.index(Channel.T)
        u_i = channels.index(Channel.U)
        if signs[t_i] * signs[u_i] > 0:
            raise UnsupportedScattering(
                f"Final state has identical FERMIONS {sorted(ident)}, so the "
                "t- and u-channel diagrams differ by interchange of two "
                "external fermion lines and must enter with OPPOSITE signs. "
                "Pass relative_signs (e.g. [1, -1]) rather than letting them "
                "default to +1 -- the interference term, not just its "
                "magnitude, depends on it."
            )

def build_amplitude_sum(gen, diagrams: Sequence[Tuple[Diagram, Channel]],
                        relative_signs: Optional[Sequence[int]] = None) -> str:
    """Emit several diagrams and add their amplitudes COHERENTLY.

    Most real 2->2 processes are a sum: phi phi -> phi phi with a cubic
    coupling is s + t + u; identical-fermion scattering is s + t. A tool
    that only ever emits one diagram is silently wrong whenever more than
    one contributes, which is why this exists.

    WHAT THIS DOES NOT DO. It cannot work out WHICH diagrams contribute --
    that needs a Lagrangian, and heptapod's diagram spec deliberately
    carries topology rather than a model. The caller supplies the list.

    ``relative_signs`` carries the (-1) between diagrams related by
    interchange of two external fermion lines. It defaults to all +1, which
    is correct only when no such interchange relates them; pass it
    explicitly for identical-fermion final states.

    All diagrams must share external legs and masses, since they are added
    before squaring; that is checked here rather than left to produce a
    meaningless sum.
    """
    if not diagrams:
        raise UnsupportedScattering("build_amplitude_sum needs at least one diagram.")
    signs = list(relative_signs) if relative_signs is not None else [1] * len(diagrams)
    if len(signs) != len(diagrams):
        raise UnsupportedScattering(
            f"relative_signs has {len(signs)} entries for {len(diagrams)} diagrams."
        )

    _check_identical_particle_exchange([ch for _, ch in diagrams],
                                       diagrams[0][0], relative_signs)

    ref_spins = external_spins(diagrams[0][0])
    ref_masses = _leg_masses(diagrams[0][0])
    for d, _ in diagrams[1:]:
        if external_spins(d) != ref_spins or _leg_masses(d) != ref_masses:
            raise UnsupportedScattering(
                "All diagrams in a coherent sum must share the same external "
                "legs and masses; got "
                f"{ref_spins}/{ref_masses} and {external_spins(d)}/{_leg_masses(d)}."
            )

    sections: List[str] = []
    terms: List[str] = []
    for i, (d, ch) in enumerate(diagrams, start=1):
        sections.append(
            build_amplitude(gen, d, ch, amp_var=f"amp{i}", q_var=f"qMom{i}",
                            normalise_dimension=False)
        )
        terms.append(("- " if signs[i - 1] < 0 else "+ ") + f"amp{i}")
    total = " ".join(terms).lstrip("+ ").strip()
    sections.append(
        "(* Coherent sum of the diagrams above, normalised to 4 dimensions "
        "ONCE. *)\n"
        f"amp = ChangeDimension[{total}, 4];\n"
    )
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Kinematics and the cross section
# ---------------------------------------------------------------------------

def _leg_masses(diagram: Diagram) -> List[str]:
    parts = list(diagram.initial) + list(diagram.final)
    return [_mass_symbol(p, i) for i, p in enumerate(parts)]


def kinematics_block(diagram: Diagram) -> str:
    """Scalar products in terms of (s, t), with u eliminated on shell.

    Mirrors the decay path's proven idiom (explicit ``ScalarProduct``
    assignments) rather than ``SetMandelstam``: it leaves exactly two
    independent variables, which is what makes the t-integration well
    posed and stops a symbolic ``u`` surviving into the final answer.
    """
    m1, m2, m3, m4 = _leg_masses(diagram)
    lines = [
        "(* Step 2: Kinematics — two independent invariants (s, t). *)",
        "(* u is eliminated on shell: s + t + u = sum_i m_i^2. *)",
        f"uExpr = {m1}^2 + {m2}^2 + {m3}^2 + {m4}^2 - s - t;",
        "FCClearScalarProducts[];",
        f"ScalarProduct[p1, p1] = {m1}^2;",
        f"ScalarProduct[p2, p2] = {m2}^2;",
        f"ScalarProduct[p3, p3] = {m3}^2;",
        f"ScalarProduct[p4, p4] = {m4}^2;",
        f"ScalarProduct[p1, p2] = (s - {m1}^2 - {m2}^2)/2;",
        f"ScalarProduct[p3, p4] = (s - {m3}^2 - {m4}^2)/2;",
        f"ScalarProduct[p1, p3] = ({m1}^2 + {m3}^2 - t)/2;",
        f"ScalarProduct[p2, p4] = ({m2}^2 + {m4}^2 - t)/2;",
        f"ScalarProduct[p1, p4] = ({m1}^2 + {m4}^2 - uExpr)/2;",
        f"ScalarProduct[p2, p3] = ({m2}^2 + {m3}^2 - uExpr)/2;",
    ]
    return "\n".join(lines) + "\n"


def initial_state_multiplicity(diagram: Diagram) -> int:
    """N_init = product of (2s+1), with 2 for a massless vector.

    The old code used (2s+1) unconditionally, which over-counts a massless
    vector's polarisations by 3/2.
    """
    n = 1
    for p in diagram.initial:
        spin = float(p.spin or 0.0)
        if spin == 1.0 and _is_massless(p.mass):
            n *= 2
        else:
            n *= int(round(2 * spin + 1))
    return n


def cross_section_block(diagram: Diagram,
                       integrate_timeout_s: float = DEFAULT_INTEGRATE_TIMEOUT_S) -> str:
    """The 2->2 master formula, symbolic in s.

    sigma = (1/S)(1/N_init) C Int dt |M|^2 / (16 pi lambda(s, m1^2, m2^2))

    Every factor the old implementation dropped is explicit here: the
    identical-particle symmetry factor S (absent entirely, which made
    phi phi -> phi phi come out exactly a factor 2 high), the massless-vector
    polarisation count in N_init, and a genuinely symbolic s (the old code
    mixed a numeric sqrt_s into the prefactor while |M|^2 still carried a
    symbolic s, so the two never met).
    """
    m1, m2, m3, m4 = _leg_masses(diagram)
    n_init = initial_state_multiplicity(diagram)
    sym = compute_symmetry_factor(diagram)
    lines = [
        "(* Step 6: Total cross section *)",
        "kallen[a_, b_, c_] := a^2 + b^2 + c^2 - 2 a b - 2 a c - 2 b c;",
        f"lam12 = kallen[s, {m1}^2, {m2}^2];   (* initial-state Kallen *)",
        f"lam34 = kallen[s, {m3}^2, {m4}^2];   (* final-state Kallen *)",
        "",
        "(* t range at cos(theta) = -+1 *)",
        f"tMid = {m1}^2 + {m3}^2 - (s + {m1}^2 - {m2}^2) (s + {m3}^2 - {m4}^2)/(2 s);",
        "tHalf = Sqrt[lam12 lam34]/(2 s);",
        "tMin = tMid - tHalf;",
        "tMax = tMid + tHalf;",
        "",
        f"nInit = {n_init};            (* initial spin/polarisation average *)",
        f"symmetryFactor = {sym};      (* identical final-state particles *)",
        f"colorFactor = {_fmt_mma(diagram.color_factor)};",
        "",
        "(* dsigma/dt = |M|^2 / (16 pi lambda(s, m1^2, m2^2)) *)",
        "dSigmaDt = ampSqKin colorFactor/(16 Pi lam12 nInit symmetryFactor);",
        "(* Symbolic integration is bounded. It is not guaranteed to close --",
        "   a coherent sum gives a rational integrand whose antiderivative can",
        "   come back Undefined or unevaluated even when the integral is finite",
        "   -- and with massless-mediator poles it can grind indefinitely.",
        "   Neither should hang the caller, so cap it and fall through. *)",
        f"sigmaSym = TimeConstrained[Integrate[dSigmaDt, {{t, tMin, tMax}}], "
        f"{_fmt_mma(integrate_timeout_s)}, $Failed];",
        "If[sigmaSym === $Failed,",
        f"  Print[\"WARNING: the symbolic t-integration exceeded "
        f"{_fmt_mma(integrate_timeout_s)} s; \" <>",
        "        \"no closed form. Use sigmaNIntegrate[rules] for a number.\"];",
        "  sigma = $Failed,",
        "  sigma = Simplify[sigmaSym]];",
        "If[sigma =!= $Failed && !FreeQ[sigma, Integrate],",
        "  Print[\"WARNING: the symbolic t-integration did not close. \" <>",
        "        \"Use sigmaNIntegrate[rules] for a number.\"]];",
        "",
        "(* Numeric fallback. The symbolic antiderivative can be well formed and",
        "   STILL evaluate to Undefined at a parameter point, when the endpoints",
        "   cancel -- so this is the reliable numeric route, not just a rescue",
        "   for an unevaluated Integrate. Substitutes values FIRST, then",
        "   integrates:  sigmaNIntegrate[{mphi -> 60., mProp0 -> 300., g -> 55.,",
        "                                 s -> 700.^2}]  *)",
        "sigmaNIntegrate[rules_] := Module[{fInt, tLo, tHi},",
        "  fInt = N[dSigmaDt /. rules];",
        "  tLo = N[tMin /. rules]; tHi = N[tMax /. rules];",
        "  NIntegrate[fInt, {t, tLo, tHi}]];",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# VVV_CONVENTION_NOTE -- RESOLVED. The convention is ALL-INCOMING MOMENTA.
# ---------------------------------------------------------------------------
# The triple-gauge vertex is written everywhere in heptapod as
#
#     V^{m1 m2 m3}(k1,k2,k3) = g[ eta^{m1 m2}(k1-k2)^{m3}
#                               + eta^{m2 m3}(k2-k3)^{m1}
#                               + eta^{m3 m1}(k3-k1)^{m2} ]
#
# with ALL THREE MOMENTA FLOWING INTO THE VERTEX, so that k1+k2+k3 = 0. A leg
# whose physical momentum flows outward enters as -p. For a decay P -> q1 q2
# that means (k1,k2,k3) = (P, -q1, -q2); for a 2->2 vertex the mediator
# enters as -q at one end and +q at the other. This is the textbook statement
# (Peskin & Schroeder ch. 16, Schwartz ch. 25, Srednicki ch. 72) and it is
# what the `p_in` property on VLeg implements.
#
# WHY IT IS NOT A FREE CHOICE. The formula is DERIVED under k1+k2+k3 = 0 --
# the cyclic structure is what momentum conservation at the vertex buys you.
# heptapod's decay path used to evaluate the same expression at the physical
# momenta as drawn, (P, q1, q2), which sum to 2P rather than 0. That is the
# formula applied outside its domain, not an alternative convention, and it
# is wrong at finite masses.
#
# THE MEASUREMENT THAT SETTLED IT. Against the standard Z' -> W+W- width
#
#     Gamma = (g^2/192 pi) mV (mV/m)^4 (1-4x)^{3/2} [1 + 20x + 12x^2],
#     x = m^2/mV^2, equal daughter masses,
#
# the two assignments give (ratio to the literature value):
#
#     mV/m        all-incoming      as-drawn
#        3         1.000000000     0.208791209
#        5         1.000000000     0.472295515
#       10         1.000000000     0.800199800
#      300         1.000000000     0.999733393
#
# All-incoming is exact at every mass ratio. As-drawn converges only
# asymptotically, which is why both share the leading longitudinal
# coefficient 1/192 and why that limit could not discriminate. The generated
# decay script reproduces the literature number to 16 digits:
# 567.7435624334261 vs 567.7435624334262 at (g, mV, m) = (0.8, 900, 200).
#
# CONSEQUENCE FOR hepbench. Its decay_V_to_VVp ground truth is built from the
# as-drawn form (see that benchmark's derivation.py, which states the vertex
# with (P-q1), (q1-q2), (q2-P)) and validates it by two contractions of that
# same expression -- which checks the contraction algebra, not the vertex.
# Both of its parameter points sit at mV/m ~ 4, deep in the regime where the
# two disagree, so those committed widths are wrong by roughly 3x and need
# regenerating. heptapod no longer agrees with them, deliberately.
