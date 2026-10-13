"""
# inspect_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Read a compiled UFO: particles, parameters, couplings, vertices, gauge.

The one tool in this package. The rest of tools/ufo is library code that other
tools call; this exposes it, because the agents that most need it are the ones
holding a freshly compiled UFO and no way to look inside it.
"""

import json
import os
from typing import Any, Dict, List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.ufo.evaluate import (
    evaluate_couplings,
    evaluate_parameters,
    is_ghost,
    is_goldstone,
    parse_parameters,
    ufo_gauge,
    vertex_couplings,
)
from tools.ufo.parser import parse_particles

SCHEMA_VERSION = "ufo-inspect-1.0"

_WHAT = ("summary", "particles", "parameters", "couplings", "vertices", "gauge")


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def _jsonable(v: Any) -> Any:
    """Render a value for JSON, keeping complex numbers legible.

    Parameters resolve through ``cmath.sqrt`` and come back complex even when
    the imaginary part is zero, which is noise in a report. A negligible
    imaginary part is dropped; a real one is kept as ``{re, im}`` rather than
    silently discarded, because a coupling with a genuine phase is physics.
    """
    if isinstance(v, complex):
        if abs(v.imag) <= 1e-12 * max(1.0, abs(v.real)):
            return v.real
        return {"re": v.real, "im": v.imag}
    try:
        return float(v) if v is not None and not isinstance(v, (str, bool)) else v
    except (TypeError, ValueError):
        return str(v)


class InspectUFOTool(BaseTool):
    """
    Read a compiled UFO model directory: particles, parameters, couplings,
    vertices and gauge, with numerical values evaluated on request.

    Answers "what does this model actually do, and with what strength". A UFO
    is Python 2 and executes arbitrary code on import, so it is parsed with
    `ast` and never imported -- which is also why writing this by hand is a
    trap. Three separate builds of one model each hand-rolled an equivalent
    parser; one silently returned nothing because the generated file closes
    `Parameter(` on its last argument line rather than on a line of its own,
    and another transposed the coupling dict key and made every right-handed
    coupling evaluate to zero, which reads as broken chirality rather than a
    swapped index.

    UFO coupling dictionaries are keyed `(colour, lorentz)`, NOT
    `(lorentz, colour)`. `what="vertices"` reports both indices explicitly so
    the convention never has to be guessed.

    Input:
        ufo_dir: The UFO directory, relative to base_directory.
        what: Which view to return (default "summary"):
            summary    — counts, gauge, new-physics states; the cheap first look
            particles  — declared states with quantum numbers, Goldstones and
                         ghosts marked rather than silently mixed in
            parameters — external and internal; values with evaluate=true
            couplings  — GC_* expressions; complex values with evaluate=true
            vertices   — particle content, Lorentz/colour structures, and the
                         (colour, lorentz) -> coupling map. Filter it:
                         names=["Zp"] for every Z' vertex, or
                         names=["e__plus__","e__minus__","Zp"] with
                         match="exact" for that one vertex
            gauge      — unitary or Feynman, with the artefact states listed
        evaluate: Resolve numbers (default true for parameters/couplings).
            Internal parameters are resolved in dependency order.
        overrides: JSON object of external-parameter overrides, e.g.
            {"MZp": 500.0, "chi": 1e-2}, to evaluate the model away from its
            card defaults. A parametrisation can be exact at the default point
            and degenerate elsewhere.
        names: Restrict the output. For particles/parameters/couplings, the
            names to keep. For vertices, PARTICLE names — the filter keeps
            vertices involving them, in the UFO's variable spelling
            (`e__plus__`, not `e+`).
        match: For vertices — "any" (default) keeps a vertex involving any
            named particle; "exact" keeps only the vertex whose particle set
            is exactly `names`.
        max_records: Cap on records returned (default 200). The count is always
            reported in full even when the list is truncated.

    State:
        base_directory.

    Returns JSON:
        {"status": "ok", "schema": "ufo-inspect-1.0", "what": "...",
         "counts": {...}, plus the requested view}
    On a genuine tool error (missing directory, unreadable UFO) returns a
    formatted error.
    """

    # ======================== Runtime fields ======================== #
    ufo_dir: str = RuntimeField(
        description="UFO directory to read, relative to base_directory")
    what: Optional[str] = RuntimeField(
        default="summary",
        description=(
            "View to return: 'summary' (default), 'particles', 'parameters', "
            "'couplings', 'vertices' or 'gauge'"))
    evaluate: Optional[bool] = RuntimeField(
        default=True,
        description=(
            "Resolve parameters and couplings to numbers, internals included "
            "(default true). Set false to see the expressions only"))
    overrides: Optional[str] = RuntimeField(
        default=None,
        description=(
            'JSON object of external-parameter overrides, e.g. {"MZp": 500.0}, '
            "to evaluate away from the card defaults"))
    names: Optional[List[str]] = RuntimeField(
        default=None,
        description=(
            "Restrict the output. For what='particles'/'parameters'/"
            "'couplings' these are the names to keep. For what='vertices' "
            "these are PARTICLE names and the filter keeps vertices that "
            "involve them -- e.g. names=['Zp'] returns only Z' vertices. Use "
            "the UFO's Python variable spelling (e__plus__, not e+)"))
    match: Optional[str] = RuntimeField(
        default="any",
        description=(
            "How names filters vertices: 'any' (default) keeps a vertex that "
            "involves ANY named particle; 'exact' keeps only vertices whose "
            "particle set is exactly the names given -- use it to pull one "
            "specific vertex, e.g. names=['e__plus__','e__minus__','Zp']"))
    max_records: Optional[int] = RuntimeField(
        default=200,
        description="Cap on records returned; counts are always reported in full")
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    # ================================================================ #

    def _run(self) -> str:
        what = (self.what or "summary").strip().lower()
        if what not in _WHAT:
            return self.format_error(
                error="Invalid Input",
                reason=f"what must be one of {', '.join(_WHAT)}",
                context=what,
            )

        d = _safe_join(self.base_directory, self.ufo_dir)
        if d is None:
            return self.format_error(
                error="Access Denied",
                reason="ufo_dir escapes base_directory",
                context=self.ufo_dir,
            )
        if not os.path.isdir(d):
            return self.format_error(
                error="File Not Found",
                reason="UFO directory does not exist",
                context=self.ufo_dir,
            )
        if not os.path.isfile(os.path.join(d, "particles.py")):
            return self.format_error(
                error="Invalid Input",
                reason="not a UFO directory: particles.py is missing",
                context=self.ufo_dir,
            )

        overrides: Dict[str, float] = {}
        if self.overrides:
            try:
                overrides = json.loads(self.overrides)
                if not isinstance(overrides, dict):
                    raise ValueError("not a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                return self.format_error(
                    error="Invalid Input",
                    reason=f"overrides is not a JSON object: {e}",
                    context=self.overrides,
                )

        wanted = {str(n) for n in (self.names or [])} or None
        mode = (self.match or "any").strip().lower()
        if mode not in ("any", "exact"):
            return self.format_error(
                error="Invalid Input",
                reason="match must be 'any' or 'exact'",
                context=mode,
            )
        cap = int(self.max_records or 200)
        gauge = ufo_gauge(d)
        particles = parse_particles(os.path.join(d, "particles.py"))

        result: Dict[str, Any] = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "what": what,
            "ufo_dir": self.ufo_dir,
        }

        if what == "gauge":
            result.update(gauge)
            result["note"] = (
                "FeynmanGauge is a global set inside SM.fr and an add-on "
                "inherits it silently. Feynman gauge adds Goldstone and ghost "
                "vertices that MadGraph ignores but that any vertex count "
                "includes; unitary gauge is conventional for a tree-level UFO."
            )
            return json.dumps(result, indent=2)

        if what in ("summary", "particles"):
            recs, phys, art = [], 0, 0
            for p in particles:
                gold, ghost = is_goldstone(p), is_ghost(p)
                if gold or ghost:
                    art += 1
                else:
                    phys += 1
                var = str(p.get("var") or p.get("name") or "")
                if wanted and var not in wanted and str(p.get("name")) not in wanted:
                    continue
                recs.append({
                    "var": var,
                    "name": p.get("name"),
                    "pdg_code": _jsonable(p.get("pdg_code")),
                    "spin": _jsonable(p.get("spin")),
                    "color": _jsonable(p.get("color")),
                    "charge": _jsonable(p.get("charge")),
                    "mass": p.get("mass"),
                    "role": "goldstone" if gold else ("ghost" if ghost else "physical"),
                })
            result["counts"] = {"particles": len(particles),
                                "physical": phys, "gauge_artifact": art}
            if what == "particles":
                result["truncated"] = len(recs) > cap
                result["particles"] = recs[:cap]
                return json.dumps(result, indent=2)

        if what == "parameters":
            declared = parse_parameters(d)
            values, unresolved = ({}, [])
            if self.evaluate is not False:
                values, unresolved = evaluate_parameters(d, overrides)
            recs = []
            for p in declared:
                if wanted and p["var"] not in wanted:
                    continue
                rec = {"var": p["var"], "nature": p.get("nature"),
                       "type": p.get("type"), "lhablock": p.get("lhablock")}
                if self.evaluate is not False and p["var"] in values:
                    rec["value"] = _jsonable(values[p["var"]])
                    if p.get("nature") == "internal":
                        rec["expression"] = p.get("value")
                else:
                    rec["value"] = p.get("value")
                recs.append(rec)
            result["counts"] = {
                "parameters": len(declared),
                "external": sum(1 for p in declared if p.get("nature") == "external"),
                "internal": sum(1 for p in declared if p.get("nature") == "internal"),
                "resolved": len(values),
            }
            if unresolved:
                result["unresolved"] = unresolved
            if overrides:
                result["overrides_applied"] = overrides
            result["truncated"] = len(recs) > cap
            result["parameters"] = recs[:cap]
            return json.dumps(result, indent=2)

        if what == "couplings":
            from tools.ufo.evaluate import parse_couplings
            exprs = parse_couplings(d)
            values, errs = ({}, {})
            if self.evaluate is not False:
                values, errs = evaluate_couplings(d, overrides)
            recs = []
            for name, expr in exprs.items():
                if wanted and name not in wanted:
                    continue
                rec: Dict[str, Any] = {"name": name, "expression": expr}
                if name in values:
                    rec["value"] = _jsonable(values[name])
                elif name in errs:
                    rec["error"] = errs[name]
                recs.append(rec)
            result["counts"] = {"couplings": len(exprs), "evaluated": len(values),
                                "failed": len(errs)}
            if overrides:
                result["overrides_applied"] = overrides
            result["truncated"] = len(recs) > cap
            result["couplings"] = recs[:cap]
            return json.dumps(result, indent=2)

        if what == "vertices":
            verts = vertex_couplings(d)
            role = {}
            for p in particles:
                var = str(p.get("var") or p.get("name") or "")
                role[var] = ("goldstone" if is_goldstone(p)
                             else "ghost" if is_ghost(p) else "physical")
            recs, phys, art = [], 0, 0
            for v in verts:
                artefact = any(role.get(n, "physical") != "physical"
                               for n in v["particles"])
                if artefact:
                    art += 1
                else:
                    phys += 1
                if wanted:
                    pset = set(v["particles"])
                    if mode == "exact":
                        if pset != wanted:
                            continue
                    elif not (wanted & pset):
                        continue
                recs.append({**v,
                             "kind": "gauge_artifact" if artefact else "physical"})
            result["counts"] = {"vertices": len(verts), "physical": phys,
                                "gauge_artifact": art,
                                "matched": len(recs),
                                "returned": min(len(recs), cap)}
            result["coupling_key_order"] = "(color_index, lorentz_index)"
            if wanted:
                result["filter"] = {"names": sorted(wanted), "match": mode}
            # Say what was cut, not just that something was. A filter whose
            # matches happen to sort after the cap returns a page containing
            # none of what was asked for, with truncated:true beside it -- which
            # reads as a filter that silently did nothing. One caller concluded
            # exactly that, went to vertices.py by hand, and got the
            # (colour, lorentz) key order backwards doing it.
            result["truncated"] = len(recs) > cap
            if len(recs) > cap:
                result["truncation_note"] = (
                    f"{len(recs)} vertices matched; showing {cap}. Raise "
                    f"max_records, narrow `names`, or set match='exact' -- the "
                    f"vertices you want may be among those not shown.")
            result["vertices"] = recs[:cap]
            return json.dumps(result, indent=2)

        # summary
        verts = vertex_couplings(d)
        vrole = {}
        for p in particles:
            var = str(p.get("var") or p.get("name") or "")
            vrole[var] = ("goldstone" if is_goldstone(p)
                          else "ghost" if is_ghost(p) else "physical")
        v_art = sum(1 for v in verts
                    if any(vrole.get(n, "physical") != "physical" for n in v["particles"]))
        declared = parse_parameters(d)
        values, unresolved = evaluate_parameters(d, overrides)
        result["gauge"] = gauge["gauge"]
        result["counts"].update({
            "parameters": len(declared),
            "parameters_unresolved": len(unresolved),
            "vertices": len(verts),
            "vertices_physical": len(verts) - v_art,
            "vertices_gauge_artifact": v_art,
        })
        result["files"] = sorted(
            f for f in os.listdir(d) if f.endswith(".py"))
        return json.dumps(result, indent=2)
