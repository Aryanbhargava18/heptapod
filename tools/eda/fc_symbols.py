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

import re
from typing import Dict

from tools.nda.simple_diagram import Particle, Vertex

_ANTIPARTICLE_SUFFIXES = ("bar", "~", "+")

#: A mass declaration usable verbatim as a symbol in BOTH Mathematica and
#: sympy: leading letter or underscore, then letters/digits/underscores. The
#: intersection of the two languages' identifier rules, so the formula
#: round-trips from `width_mathematica` to `width_formula` without renaming.
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class InvalidMassSymbol(ValueError):
    """A particle's ``mass`` is a string that cannot serve as a symbol.

    Raised rather than sanitised, deliberately. Quietly rewriting the
    declaration -- as ``_safe_symbol`` does for LABELS, where it is correct
    because a label is prose -- would put the formula in a name the caller
    never wrote. That is the same namespace split that made ``values`` bind
    nothing, just moved one step later and made harder to see.
    """


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



#: Spin carried by each letter of a self-describing vertex type. S scalar,
#: F fermion, V vector -- the notation the codebase already uses for its
#: bosonic vertices (sss, ssv, svv, vvv) and the one agents reach for
#: unprompted (FFV, VFF, SFF).
_STRUCTURE_LETTERS = {"s": 0.0, "f": 0.5, "v": 1.0}


def _structure_from_letters(base: str):
    """The spin multiset a letter-spelled type declares, or None.

    `ffv` -> (0.5, 0.5, 1.0). A type spelled only from S/V/F states its own
    structure, which is what lets the generator check that structure against
    the spins the caller declared on the legs -- two statements of one fact
    that used to be read in different places and allowed to disagree.

    Returns None for anything else, including every spelling in the vertex
    tables (`vector`, `yukawa`, ...), so the rule cannot capture a name that
    means something other than its spin content. A test asserts that
    non-collision over both tables, so it keeps holding as they grow.

    Requires at least three letters: a two-point vertex is not a thing, and
    `sf` would otherwise read as a structure.
    """
    if not base or len(base) < 3:
        return None
    if any(c not in _STRUCTURE_LETTERS for c in base):
        return None
    return tuple(sorted(_STRUCTURE_LETTERS[c] for c in base))


class VertexStructureMismatch(ValueError):
    """The vertex type and the declared leg spins describe different vertices.

    Raised rather than resolved, because the tool cannot know which of the two
    is wrong. An agent that meant a fermion vertex and mistyped the spins, and
    an agent that meant a triple-gauge vertex and mistyped the name, send
    indistinguishable input. Picking one would be inventing intent; the message
    states the contradiction and both readings instead.
    """

def _mass_symbol(particle: Particle, idx: int) -> str:
    """The symbol this particle's mass is written as in the formula.

    A STRING ``mass`` NAMES THE SYMBOL and is used verbatim, so the generated
    expression is in the caller's own names and the ``values`` dict they pass
    alongside binds against it. This is the whole point: the substitution list
    is built from ``values`` keys, so any name the generator invents instead
    binds nothing and the width comes back as an expression rather than a
    number -- silently, because the script still runs.

    A NUMERIC ``mass`` is a value, not a name; there is nothing to name the
    symbol after, so it keeps the ``m<label>`` form and the number is
    substituted against that by ``_mass_definitions``.

    An OMITTED ``mass`` also gives ``m<label>`` (``m<idx>`` unlabelled) -- the
    documented default, and what every caller that passes ``{"mV": ...}`` for
    a particle labelled ``V`` already relies on.

    Raises ``InvalidMassSymbol`` for a string that is not a usable symbol.
    """
    mass = particle.mass
    if isinstance(mass, str):
        declared = mass.strip()
        if not _SYMBOL_RE.match(declared):
            label = particle.label or f"#{idx}"
            default = (f"m{_safe_symbol(particle.label)}" if particle.label
                       else f"m{idx}")
            raise InvalidMassSymbol(
                f"particle {label!r} declares mass={mass!r}, which cannot be "
                f"used as a symbol.\n"
                f"- A `mass` string NAMES the mass symbol, so it must be a "
                f"plain identifier: a letter or underscore followed by "
                f"letters, digits or underscores (e.g. \"mS\", \"m_chi\", "
                f"\"MZprime\"). No spaces, signs, dots or brackets, and it "
                f"may not start with a digit.\n"
                f"- A NUMBER is also fine if you want the mass fixed: "
                f"\"mass\": 950.\n"
                f"- Or omit `mass` entirely to get the default symbol "
                f"{default!r}, then key `values` to that."
            )
        return declared
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
