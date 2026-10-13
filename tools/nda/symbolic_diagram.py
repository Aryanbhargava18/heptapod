"""
# symbolic_diagram.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Symbolic diagram dataclasses for NDA formula generation and symbolic
FeynCalc code generation.

Unlike Diagram (simple_diagram.py) which requires fully resolved numerical
values, SymbolicDiagram only requires structural information: particle labels,
vertex types, and topology. Masses, spins, and coupling values are optional
and left symbolic when omitted.

Use resolve_diagram() from diagram_resolution.py to convert a SymbolicDiagram
into a fully numerical Diagram using the SM particle database.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

try:
    from .simple_diagram import parse_spin, Particle, Vertex, Propagator, Diagram
except ImportError:
    from simple_diagram import parse_spin, Particle, Vertex, Propagator, Diagram


# ---------------------------------------------------------------------------
# Default coupling inference from vertex type
# ---------------------------------------------------------------------------

_CHIRAL_TYPES = {"chiral", "yukawa-chiral", "scalar-chiral", "vector-chiral", "tensor-chiral", "dipole-chiral"}
_VA_VFF_TYPES = {"vector-axial", "va"}
_VA_SFF_TYPES = {"scalar-va"}


def _default_coupling_for_type(vtype: str) -> Union[str, Dict[str, str]]:
    """Return the default coupling structure implied by a vertex type.

    When a user omits ``coupling`` from a vertex dict, this function
    generates the canonical default so the codegen always has something
    to work with.
    """
    normalized = vtype.lower().replace("_", "-")
    if normalized in _CHIRAL_TYPES:
        return {"gL": "gL", "gR": "gR"}
    if normalized in _VA_VFF_TYPES:
        return {"gV": "gV", "gA": "gA"}
    if normalized in _VA_SFF_TYPES:
        return {"gS": "gS", "gP": "gP"}
    if normalized == "axial-vector":
        return "gA"
    return "g"


@dataclass
class SymbolicParticle:
    """A particle with label required, mass/spin optional (left symbolic if omitted)."""
    label: str
    spin: Optional[float] = None
    mass: Optional[float] = None
    massive: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"label": self.label}
        if self.spin is not None:
            result["spin"] = self.spin
        if self.mass is not None:
            result["mass"] = self.mass
        if self.massive is not None:
            result["massive"] = self.massive
        return result


@dataclass
class SymbolicVertex:
    """A vertex with type and coupling name(s) — never numeric values.

    For simple vertices, coupling is a string (e.g., ``"y_b"``).
    For chiral vertices, coupling is a dict of string names
    (e.g., ``{"gL": "gL", "gR": "gR"}``).

    ``structures`` and ``pairing`` carry what a CONTACT operator needs
    beyond a type name: which Dirac structure sits on each bilinear, and
    which external legs form them. A four-fermion operator is not
    determined by "four fermion legs plus a type string" -- see
    tools/eda/scattering._contact_ffff.
    """
    type: str
    coupling: Union[str, Dict[str, str]]
    structures: Optional[List[str]] = None
    pairing: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type, "coupling": self.coupling}
        if self.structures:
            out["structures"] = self.structures
        if self.pairing:
            out["pairing"] = self.pairing
        return out


@dataclass
class SymbolicPropagator:
    """An internal propagator with label required, mass/spin optional."""
    label: str
    spin: Optional[float] = None
    mass: Optional[float] = None
    massive: Optional[bool] = None
    regime: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"label": self.label}
        if self.spin is not None:
            result["spin"] = self.spin
        if self.mass is not None:
            result["mass"] = self.mass
        if self.massive is not None:
            result["massive"] = self.massive
        if self.regime != "auto":
            result["regime"] = self.regime
        return result


@dataclass
class SymbolicDiagram:
    """
    Diagram specification using only structural information.

    Key difference from Diagram:
    - SymbolicVertex.coupling is always a string name (e.g. "y_b"), never a number
    - SymbolicParticle.mass is optional — omitted means "leave symbolic"
    - No couplings: Dict[str, float] resolution dict
    """
    topology: str
    initial: List[SymbolicParticle]
    final: List[SymbolicParticle]
    vertices: List[SymbolicVertex]
    propagators: List[SymbolicPropagator] = field(default_factory=list)
    color_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "topology": self.topology,
            "initial": [p.to_dict() for p in self.initial],
            "final": [p.to_dict() for p in self.final],
            "vertices": [v.to_dict() for v in self.vertices],
        }
        if self.propagators:
            result["propagators"] = [p.to_dict() for p in self.propagators]
        if self.color_factor is not None:
            result["color_factor"] = self.color_factor
        return result

    def validate(self) -> tuple[bool, List[str]]:
        """Validate structural consistency of the symbolic diagram."""
        warnings: List[str] = []

        if not self.initial:
            warnings.append("Diagram must have at least one initial-state particle")
        if not self.final:
            warnings.append("Diagram must have at least one final-state particle")
        if not self.vertices:
            warnings.append("Diagram must have at least one vertex")

        for p in self.initial:
            if not p.label:
                warnings.append("All initial-state particles must have a label")
        for p in self.final:
            if not p.label:
                warnings.append("All final-state particles must have a label")
        for v in self.vertices:
            if not v.type:
                warnings.append("All vertices must have a type")
        for prop in self.propagators:
            if not prop.label:
                warnings.append("All propagators must have a label")

        return (len(warnings) == 0, warnings)


def _parse_symbolic_particle(d: Dict[str, Any]) -> SymbolicParticle:
    """Parse a particle dict into SymbolicParticle."""
    label = d.get("label")
    if not label:
        raise ValueError("SymbolicParticle requires a 'label' field")

    spin = d.get("spin")
    if spin is not None:
        spin = parse_spin(spin)

    return SymbolicParticle(
        label=label,
        spin=spin,
        mass=d.get("mass"),
        massive=d.get("massive"),
    )


def _parse_symbolic_vertex(d: Dict[str, Any]) -> SymbolicVertex:
    """Parse a vertex dict into SymbolicVertex.

    Coupling can be:
      - A string name (e.g., ``"y_b"``)
      - A dict of string names for chiral vertices (e.g., ``{"gL": "gL", "gR": "gR"}``)
    """
    vtype = d.get("type")
    if not vtype:
        raise ValueError("SymbolicVertex requires a 'type' field")

    coupling = d.get("coupling")
    if coupling is None:
        coupling = _default_coupling_for_type(vtype)
    if isinstance(coupling, str):
        pass  # simple coupling name
    elif isinstance(coupling, dict):
        for key, val in coupling.items():
            if not isinstance(val, str):
                raise ValueError(
                    f"SymbolicVertex dict coupling values must be string names, "
                    f"got {type(val).__name__} for key '{key}': {val}. "
                    f'Use string names like {{"gL": "gL", "gR": "gR"}} '
                    f"(not numeric values)."
                )
    elif isinstance(coupling, list):
        raise ValueError(
            f"SymbolicVertex coupling does not accept a list: {coupling}. "
            f"For chiral couplings, use a dict with string names instead: "
            f'{{"gL": "{coupling[0] if coupling else "gL"}", '
            f'"gR": "{coupling[1] if len(coupling) > 1 else "gR"}"}}'
        )
    elif isinstance(coupling, (int, float)):
        raise ValueError(
            f"SymbolicVertex coupling must be a symbolic name, not a number ({coupling}). "
            f'Use a string name like "g" or "y_b". '
            f"For chiral vertices, use a dict: "
            f'{{"gL": "gL", "gR": "gR"}}.'
        )
    else:
        raise ValueError(
            f"SymbolicVertex coupling must be a string or dict of strings, "
            f"got {type(coupling).__name__}: {coupling}. "
            f'Examples: "y_b" or {{"gL": "gL", "gR": "gR"}}.'
        )

    return SymbolicVertex(type=vtype, coupling=coupling,
                          structures=d.get("structures"),
                          pairing=d.get("pairing"))


def _parse_symbolic_propagator(d: Dict[str, Any]) -> SymbolicPropagator:
    """Parse a propagator dict into SymbolicPropagator."""
    label = d.get("label")
    if not label:
        raise ValueError("SymbolicPropagator requires a 'label' field")

    spin = d.get("spin")
    if spin is not None:
        spin = parse_spin(spin)

    return SymbolicPropagator(
        label=label,
        spin=spin,
        mass=d.get("mass"),
        massive=d.get("massive"),
        regime=d.get("regime", "auto"),
    )


def parse_symbolic_diagram(diagram_dict: Dict[str, Any]) -> SymbolicDiagram:
    """
    Parse a dictionary into a SymbolicDiagram.

    Args:
        diagram_dict: Dictionary with symbolic diagram specification.
            Required keys: 'initial', 'final', 'vertices'
            Optional keys: 'topology', 'propagators', 'color_factor'

    Returns:
        SymbolicDiagram object

    Raises:
        ValueError: If required fields are missing or malformed
    """
    initial_list = diagram_dict.get("initial", [])
    if not initial_list:
        raise ValueError("SymbolicDiagram must have 'initial' field with at least one particle")
    initial = [_parse_symbolic_particle(p) for p in initial_list]

    final_list = diagram_dict.get("final", [])
    if not final_list:
        raise ValueError("SymbolicDiagram must have 'final' field with at least one particle")
    final = [_parse_symbolic_particle(p) for p in final_list]

    vertices_list = diagram_dict.get("vertices", [])
    if not vertices_list:
        raise ValueError("SymbolicDiagram must have 'vertices' field with at least one vertex")
    vertices = [_parse_symbolic_vertex(v) for v in vertices_list]

    propagators_list = diagram_dict.get("propagators", [])
    propagators = [_parse_symbolic_propagator(p) for p in propagators_list]

    # Infer topology if not given
    topology = diagram_dict.get("topology")
    if topology is None:
        n_final = len(final_list)
        n_props = len(propagators_list)
        has_loops = any(p.get("is_loop_propagator", False) for p in propagators_list)
        if has_loops:
            topology = f"loop_{n_final}body"
        else:
            topology = f"tree_{n_final}body"

    color_factor = diagram_dict.get("color_factor")

    return SymbolicDiagram(
        topology=topology,
        initial=initial,
        final=final,
        vertices=vertices,
        propagators=propagators,
        color_factor=color_factor,
    )


def _is_massive_particle(p: SymbolicParticle, is_initial: bool = False) -> bool:
    """Determine if an external particle is massive.

    Priority: explicit ``massive`` field > infer from ``mass`` value >
    default based on spin and role.

    Spin-1 particles default to massive (Proca) unless explicitly marked
    massless.  This is critical: a massless polarisation sum (-g_μν) vs
    the Proca sum (-g_μν + p_μ p_ν/m²) gives qualitatively different
    results.  Massless vectors (photon, gluon) are the exception and
    should be flagged with ``massive=False`` or ``mass=0``.
    """
    if p.massive is not None:
        return p.massive
    if p.mass is not None:
        if isinstance(p.mass, str):
            return True  # a named mass means massive
        return p.mass > 0
    # No explicit info: spin-1 particles default to massive (Proca);
    # initial-state (decaying) particles are always massive;
    # other final-state particles default to massless.
    if p.spin is not None and p.spin == 1:
        return True
    return is_initial


def _is_massive_propagator(prop: SymbolicPropagator) -> bool:
    """Determine if a propagator is massive.

    Priority: explicit `massive` field > infer from `mass` value > default True.
    """
    if prop.massive is not None:
        return prop.massive
    if prop.mass is not None:
        return prop.mass > 0
    # Default: assume massive (most interesting propagators are)
    return True


def _mass_sentinel(is_massive: bool) -> float:
    """Return the numeric sentinel the codegen expects.

    The codegen decides massive-vs-massless polarisation sums via
    ``if mass and mass > 0``, so we use 1.0 for massive and 0 for massless.
    """
    return 1.0 if is_massive else 0


def build_diagram_from_symbolic(sym: SymbolicDiagram) -> Diagram:
    """Build a Diagram directly from a SymbolicDiagram without any model database lookup.

    Requires spins on all external particles and propagators. Raises ValueError
    if any spin is missing. Particle masses are set to numeric sentinels
    (1.0 = massive, 0 = massless) so the codegen's polarisation-sum and
    propagator checks work correctly. The symbolic codegen derives mass
    *symbol names* from labels (e.g. ``mH``, ``mf``), not from these values.

    Initial-state particles in a decay are implicitly massive (a particle
    must have mass to decay). Final-state particles default to massless
    unless ``massive=True`` or a nonzero ``mass`` is provided.

    Args:
        sym: A SymbolicDiagram with spins specified on all particles and propagators.

    Returns:
        A Diagram suitable for FeynCalcCodeGenerator / SymbolicFeynCalcCodeGenerator.

    Raises:
        ValueError: If any particle or propagator is missing a spin value.
    """
    missing = []
    for i, p in enumerate(sym.initial):
        if p.spin is None:
            missing.append(f"initial particle {i} (label={p.label!r})")
    for i, p in enumerate(sym.final):
        if p.spin is None:
            missing.append(f"final particle {i} (label={p.label!r})")
    for i, prop in enumerate(sym.propagators):
        if prop.spin is None:
            missing.append(f"propagator {i} (label={prop.label!r})")
    if missing:
        raise ValueError(
            f"build_diagram_from_symbolic requires spins on all particles. "
            f"Missing spin for: {', '.join(missing)}"
        )

    initial = [
        Particle(
            label=p.label,
            spin=p.spin,
            mass=_mass_sentinel(_is_massive_particle(p, is_initial=True)),
        )
        for p in sym.initial
    ]
    final = [
        Particle(
            label=p.label,
            spin=p.spin,
            mass=_mass_sentinel(_is_massive_particle(p, is_initial=False)),
        )
        for p in sym.final
    ]
    vertices = [
        Vertex(type=v.type, coupling=v.coupling,
               structures=v.structures, pairing=v.pairing)
        for v in sym.vertices
    ]
    propagators = [
        Propagator(
            label=prop.label,
            mass=_mass_sentinel(_is_massive_propagator(prop)),
            spin=prop.spin,
            regime=prop.regime,
        )
        for prop in sym.propagators
    ]

    return Diagram(
        topology=sym.topology,
        initial=initial,
        final=final,
        vertices=vertices,
        couplings={},
        propagators=propagators,
        color_factor=sym.color_factor if sym.color_factor is not None else 1.0,
    )
