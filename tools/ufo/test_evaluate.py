#!/usr/bin/env python3
"""
# test_evaluate.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
UFO evaluation test suite.

Runs against a UFO written on the fly, so it needs neither Mathematica nor a
FeynRules install. The fixtures reproduce the shapes that actually broke
hand-rolled parsers: a Parameter( block closing on its last argument line, an
internal parameter declared before the one it depends on, a coupling dict keyed
(colour, lorentz) with two Lorentz structures, and a hidden-sector Goldstone
whose name carries no G0/G+ prefix.
"""

import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
HEPTAPOD_ROOT = SCRIPT_PATH.parent.parent.parent
sys.path.insert(0, str(HEPTAPOD_ROOT))

from tools.ufo.evaluate import (  # noqa: E402
    evaluate_couplings,
    evaluate_parameters,
    is_ghost,
    is_goldstone,
    parse_couplings,
    parse_parameters,
    safe_eval,
    ufo_gauge,
    vertex_couplings,
)
from tools.ufo.parser import parse_particles  # noqa: E402

# Note the formatting: `Parameter(` closes on its last argument line, not on a
# line of its own, and MZp is declared BEFORE the sw it depends on.
PARTICLES = '''
from object_library import all_particles, Particle
import parameters as Param

Zp = Particle(pdg_code = 9900032,
              name = 'Zp',
              antiname = 'Zp',
              spin = 3,
              color = 1,
              mass = Param.MZp,
              width = Param.WZp,
              texname = 'Zp',
              antitexname = 'Zp',
              charge = 0)

Hd = Particle(pdg_code = 9900025,
              name = 'Hd',
              antiname = 'Hd',
              spin = 1,
              color = 1,
              mass = Param.MHd,
              width = Param.WHd,
              texname = 'Hd',
              antitexname = 'Hd',
              charge = 0)

Gd = Particle(pdg_code = 9900250,
              name = 'Gd',
              antiname = 'Gd',
              spin = 1,
              color = 1,
              mass = Param.MZp,
              width = Param.WZp,
              texname = 'Gd',
              antitexname = 'Gd',
              goldstone = True,
              charge = 0)

ghZ = Particle(pdg_code = 9000003,
               name = 'ghZ',
               antiname = 'ghZ~',
               spin = -1,
               color = 1,
               mass = Param.MZ,
               width = Param.WZ,
               texname = 'ghZ',
               antitexname = 'ghZ~',
               charge = 0)

mu__minus__ = Particle(pdg_code = 13,
                       name = 'mu-',
                       antiname = 'mu+',
                       spin = 2,
                       color = 1,
                       mass = Param.ZERO,
                       width = Param.ZERO,
                       texname = 'mu-',
                       antitexname = 'mu+',
                       charge = -1)

mu__plus__ = mu__minus__.anti()
'''

PARAMETERS = '''
from object_library import all_parameters, Parameter

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 79.82435974619784,
               texname = '\\\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

MX = Parameter(name = 'MX',
               nature = 'external',
               type = 'real',
               value = 250.0,
               texname = '\\\\text{MX}',
               lhablock = 'HIDDEN',
               lhacode = [ 1 ])

MZp = Parameter(name = 'MZp',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(MX**2 + sw2*MZ**2)',
                texname = '\\\\text{MZp}')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\\\text{sw2}')

MHd = Parameter(name = 'MHd',
                nature = 'external',
                type = 'real',
                value = 200.0,
                texname = '\\\\text{MHd}')
'''

COUPLINGS = '''
from object_library import all_couplings, Coupling

GC_1 = Coupling(name = 'GC_1',
                value = '-(ee*complex(0,1))/3.',
                order = {'QED':1})

GC_2 = Coupling(name = 'GC_2',
                value = 'complex(0,1)*cmath.sqrt(sw2)*MZp',
                order = {'NP':1})
'''

VERTICES = '''
from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L

V_1 = Vertex(name = 'V_1',
             particles = [ P.mu__plus__, P.mu__minus__, P.Zp ],
             color = [ '1' ],
             lorentz = [ L.FFV1, L.FFV2 ],
             couplings = {(0,0):C.GC_1,(0,1):C.GC_2})

V_2 = Vertex(name = 'V_2',
             particles = [ P.Zp, P.Hd, P.Gd ],
             color = [ '1' ],
             lorentz = [ L.VSS1 ],
             couplings = {(0,0):C.GC_2})
'''


def _fixture(tmp: Path) -> str:
    (tmp / "particles.py").write_text(PARTICLES)
    (tmp / "parameters.py").write_text(PARAMETERS)
    (tmp / "couplings.py").write_text(COUPLINGS)
    (tmp / "vertices.py").write_text(VERTICES)
    return str(tmp)


def test_parameters_parse_multiline() -> bool:
    """A Parameter( block closing on its last argument line is still read."""
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        params = {p["var"]: p for p in parse_parameters(d)}
        ok = (len(params) == 6
              and params["MZ"]["nature"] == "external"
              and params["MZ"]["value"] == 91.1876
              and params["MZp"]["nature"] == "internal"
              and params["MZ"].get("lhablock") == "MASS"
              and params["MZ"].get("lhacode") == [23])
    print(f"[{'✓' if ok else '✗'}] parameters parse from multi-line blocks")
    return ok


def test_internals_resolve_out_of_order() -> bool:
    """MZp is declared before sw2, which it depends on, and still resolves."""
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        values, unresolved = evaluate_parameters(d)
        expected = (250.0 ** 2 + (1 - 79.82435974619784 ** 2 / 91.1876 ** 2)
                    * 91.1876 ** 2) ** 0.5
        got = values.get("MZp")
        got = got.real if isinstance(got, complex) else got
        ok = (not unresolved and got is not None
              and abs(got - expected) < 1e-9)
    print(f"[{'✓' if ok else '✗'}] internal parameters resolve regardless of "
          f"declaration order")
    return ok


def test_overrides_propagate() -> bool:
    """An override on an external parameter reaches the internals built on it."""
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        base, _ = evaluate_parameters(d)
        over, _ = evaluate_parameters(d, {"MX": 500.0})
        b = base["MZp"].real if isinstance(base["MZp"], complex) else base["MZp"]
        o = over["MZp"].real if isinstance(over["MZp"], complex) else over["MZp"]
        ok = o > b and abs(o - (500.0 ** 2 + (b ** 2 - 250.0 ** 2)) ** 0.5) < 1e-9
    print(f"[{'✓' if ok else '✗'}] overrides propagate into internal parameters")
    return ok


def test_coupling_key_order_preserved() -> bool:
    """The (colour, lorentz) key survives, distinguishing the two structures.

    The bug this guards: reading the dict as (lorentz, colour) returns the
    wrong coupling for a vertex with one colour structure and two Lorentz
    ones, which is exactly a fermion current split into chiral pieces.
    """
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        verts = {tuple(v["particles"]): v for v in vertex_couplings(d)}
        v = verts[("mu__plus__", "mu__minus__", "Zp")]
        by_key = {(c["color_index"], c["lorentz_index"]): c["coupling"]
                  for c in v["couplings"]}
        ok = (by_key == {(0, 0): "GC_1", (0, 1): "GC_2"}
              and v["lorentz"] == ["FFV1", "FFV2"])
    print(f"[{'✓' if ok else '✗'}] vertex coupling keys stay (colour, lorentz)")
    return ok


def test_couplings_evaluate() -> bool:
    """Couplings evaluate against the resolved namespace; failures are named."""
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        exprs = parse_couplings(d)
        values, errs = evaluate_couplings(d)
        # GC_1 references `ee`, which this fixture never declares, so it must
        # be REPORTED as unevaluated rather than silently dropped or zeroed.
        ok = (set(exprs) == {"GC_1", "GC_2"}
              and "GC_2" in values and "GC_1" in errs
              and abs(values["GC_2"].imag) > 0)
    print(f"[{'✓' if ok else '✗'}] couplings evaluate, and unresolved ones are "
          f"reported not zeroed")
    return ok


def test_goldstone_and_ghost_by_flag() -> bool:
    """Gd is a Goldstone by its flag, not by its name; ghZ is a ghost by spin.

    parse_particles folds numeric fields to Fraction, so a spin test written as
    isinstance(spin, (int, float)) sees nothing. Both predicates are checked
    here because both were wrong in code this replaces.
    """
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        parts = {str(p.get("var")): p for p in parse_particles(
            str(Path(d) / "particles.py"))}
        ok = (is_goldstone(parts["Gd"])
              and not str(parts["Gd"].get("name", "")).startswith(("G0", "G+"))
              and is_ghost(parts["ghZ"])
              and not is_goldstone(parts["Hd"])
              and not is_ghost(parts["Zp"]))
    print(f"[{'✓' if ok else '✗'}] Goldstones and ghosts identified by flag and "
          f"spin, not by name")
    return ok


def test_gauge_detection() -> bool:
    """A model carrying Goldstones or ghosts reports Feynman gauge."""
    with tempfile.TemporaryDirectory() as td:
        d = _fixture(Path(td))
        g = ufo_gauge(d)
        ok = (g["gauge"] == "feynman" and "Gd" in g["goldstones"]
              and "ghZ" in g["ghosts"])
    print(f"[{'✓' if ok else '✗'}] gauge read from the UFO's particle content")
    return ok


def test_expression_whitelist() -> bool:
    """Anything outside arithmetic over parameters and cmath is refused."""
    cases = ["__import__('os').system('true')", "open('/etc/passwd').read()",
             "(lambda: 1)()", "[x for x in range(3)]"]
    ok = True
    for src in cases:
        try:
            safe_eval(src, {})
            ok = False
        except (ValueError, SyntaxError):
            pass
    # ...and legitimate arithmetic still works.
    try:
        ok = ok and abs(safe_eval("cmath.sqrt(MZ**2)", {"MZ": 4.0}).real - 4.0) < 1e-12
    except Exception:
        ok = False
    print(f"[{'✓' if ok else '✗'}] expression whitelist refuses non-arithmetic")
    return ok


TESTS = [
    test_parameters_parse_multiline,
    test_internals_resolve_out_of_order,
    test_overrides_propagate,
    test_coupling_key_order_preserved,
    test_couplings_evaluate,
    test_goldstone_and_ghost_by_flag,
    test_gauge_detection,
    test_expression_whitelist,
]

if __name__ == "__main__":
    print("=" * 70)
    print("UFO evaluation test suite")
    print("=" * 70)
    results = [t() for t in TESTS]
    print("=" * 70)
    if all(results):
        print(f"All {len(results)} tests passed! [✓]")
    else:
        print(f"{results.count(False)} of {len(results)} tests failed! [✗]")
    print("=" * 70)
    sys.exit(0 if all(results) else 1)
