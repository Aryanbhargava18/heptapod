"""
# fc_symbols.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Pure symbol/formatting helpers shared by the FeynCalc code generators.

Extracted from ``feyncalc_codegen`` so that ``scattering`` can reuse them
without importing the generator module (which imports ``scattering`` in
turn).  ``feyncalc_codegen`` re-exports every name defined here, so its
own module-level API is unchanged.
"""

from typing import Dict

from tools.nda.simple_diagram import Particle, Vertex

_ANTIPARTICLE_SUFFIXES = ("bar", "~", "+")


def _is_numeric(s: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _fmt_mma(val) -> str:
    """Format a number for Mathematica — use integer form when possible.

    Avoids float contamination: ``3.0`` -> ``3``, ``1.0`` -> ``1``.
    Mathematica treats ``3.0`` as machine-precision, which spoils symbolic
    results.
    """
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _is_antiparticle(label: str) -> bool:
    """Heuristic: is this an antiparticle label?"""
    if label is None:
        return False
    label_lower = label.lower().strip()
    for suf in _ANTIPARTICLE_SUFFIXES:
        if label_lower.endswith(suf):
            return True
    if label_lower in ("e+", "mu+", "tau+", "positron"):
        return True
    return False


def _safe_symbol(label: str) -> str:
    """Turn a particle label into a safe Mathematica symbol fragment."""
    if label is None:
        return "X"
    s = label.replace("+", "p").replace("-", "m").replace("~", "bar").replace("/", "")
    s = s.replace("(", "").replace(")", "").replace(" ", "")
    if s and s[0].isdigit():
        s = "p" + s
    return s or "X"


def _mass_symbol(particle: Particle, idx: int) -> str:
    """Return a Mathematica symbol for the mass of a particle."""
    if particle.label:
        return f"m{_safe_symbol(particle.label)}"
    return f"m{idx}"


def _coupling_value(vertex: Vertex, couplings: Dict[str, float]) -> "str | Dict[str, str]":
    """Resolve coupling to Mathematica expression.

    Returns a string for simple couplings, or a dict of strings for
    chiral couplings (e.g., {"gL": "0.27", "gR": "0.23"}).
    """
    c = vertex.coupling
    if isinstance(c, (int, float)):
        return str(c)
    if isinstance(c, str):
        if c in couplings:
            return str(couplings[c])
        return c  # leave symbolic
    if isinstance(c, dict):
        resolved = {}
        for key, val in c.items():
            if isinstance(val, (int, float)):
                resolved[key] = str(val)
            elif isinstance(val, str):
                resolved[key] = str(couplings[val]) if val in couplings else val
            else:
                resolved[key] = str(val)
        return resolved
    return "g"
