"""
# hepmc.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
HepMC3 as a first-class heptapod data format.

The pipeline's native event record is evtjsonl-1.0; HepMC3 ASCII is the
lingua franca of generator/detector-simulation interfaces (Delphes's
`DelphesHepMC3` reads it directly). These tools convert between the two
so the chain composes:

    MadGraph -> LHE -> evtjsonl -> Pythia -> evtjsonl
        -> JSONLToHepMCTool -> .hepmc -> DelphesFromHepMCTool -> .root

- `JSONLToHepMCTool`: evtjsonl-1.0 -> HepMC3 ASCII. Each event becomes
  one GenEvent with a single interaction vertex: two beam protons
  (status 4) in, the record's particles out (status 1 unless the record
  carries an explicit status). Units GEV/MM.
- `HepMCToJSONLTool`: HepMC3 ASCII -> evtjsonl-1.0. By default keeps
  only final-state (status==1) particles, mirroring LHEToJSONLTool's
  `finals_only` semantics.

Both use pyhepmc (scikit-hep's HepMC3 bindings), imported lazily so the
rest of the toolkit works without it.
"""

import json
import math
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "evtjsonl-1.0"

# Beam protons written into each GenEvent's interaction vertex. Delphes
# only consumes status==1 particles, but a structurally complete event
# (beams in, finals out) keeps the files valid for other HepMC readers.
_BEAM_PDGID = 2212
_BEAM_STATUS = 4
_FINAL_STATUS = 1


def _require_pyhepmc():
    try:
        import pyhepmc
        return pyhepmc
    except ImportError as e:
        raise ImportError(
            "HepMC tools need pyhepmc (`pip install pyhepmc`)."
        ) from e


class _SandboxedTool(BaseTool):
    """Shared base-directory plumbing (mirrors conversions.py)."""

    base_directory: str = StateField(default=".", description="Base sandbox root")

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist or is not a directory: {self.base_directory}"
            )

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel_or_abs:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        if full.startswith(self.base_directory + os.sep) or full == self.base_directory:
            return full
        return None


class JSONLToHepMCTool(_SandboxedTool):
    """
    Convert an evtjsonl-1.0 event file into HepMC3 ASCII.

    Input (runtime):
      - jsonl_path: events.jsonl (evtjsonl-1.0; one event per line)
      - hepmc_path: output .hepmc path (relative to base_directory)
      - beam_energy_gev: per-beam energy for the bookkeeping beam
        protons (default 6500.0, i.e. 13 TeV collisions)

    Output:
      JSON string: {"status": "ok", "hepmc": "<relative path>",
                    "n_events": <int>, "n_particles": <int>}

    Particles missing an explicit `status` are written as final state
    (status 1) — the evtjsonl `finals_only` convention.
    """

    jsonl_path: str = RuntimeField(description="Path to events.jsonl (evtjsonl-1.0)")
    hepmc_path: str = RuntimeField(description="Relative path to write the HepMC3 ASCII file")
    beam_energy_gev: float = RuntimeField(default=6500.0,
                                          description="Per-beam proton energy in GeV (bookkeeping)")

    def _run(self) -> str:
        try:
            pyhepmc = _require_pyhepmc()
        except ImportError as e:
            return self.format_error(error="Missing Dependency", reason=str(e),
                                     suggestion="pip install pyhepmc")
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        src = self._safe_path(self.jsonl_path)
        dst = self._safe_path(self.hepmc_path)
        if src is None:
            return self.format_error(error="Access Denied",
                                     reason="jsonl_path is outside allowed base_directory",
                                     context=self.jsonl_path)
        if dst is None:
            return self.format_error(error="Access Denied",
                                     reason="hepmc_path escapes base_directory",
                                     context=self.hepmc_path)
        if not os.path.exists(src):
            return self.format_error(error="File Not Found",
                                     reason="events.jsonl not found", context=self.jsonl_path)

        os.makedirs(os.path.dirname(dst) or self.base_directory, exist_ok=True)
        e_beam = float(self.beam_energy_gev)
        pz_beam = math.sqrt(max(e_beam * e_beam - 0.938272 ** 2, 0.0))

        n_events = 0
        n_particles = 0
        try:
            with pyhepmc.open(dst, "w") as out, open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    particles = (rec.get("data") or {}).get("particles") or []
                    ev = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
                    ev.event_number = int(rec.get("event_id", n_events))
                    vtx = pyhepmc.GenVertex()
                    for sign in (+1.0, -1.0):
                        beam = pyhepmc.GenParticle(
                            (0.0, 0.0, sign * pz_beam, e_beam),
                            _BEAM_PDGID, _BEAM_STATUS)
                        vtx.add_particle_in(beam)
                    for p in particles:
                        gp = pyhepmc.GenParticle(
                            (float(p["px"]), float(p["py"]),
                             float(p["pz"]), float(p["E"])),
                            int(p["id"]),
                            int(p.get("status", _FINAL_STATUS)) or _FINAL_STATUS)
                        if "m" in p:
                            gp.generated_mass = float(p["m"])
                        vtx.add_particle_out(gp)
                        n_particles += 1
                    ev.add_vertex(vtx)
                    out.write(ev)
                    n_events += 1
        except Exception as e:
            return self.format_error(error="Conversion Error",
                                     reason=f"{type(e).__name__}: {e}",
                                     context=self.jsonl_path,
                                     suggestion="Ensure the input matches schema evtjsonl-1.0")
        if n_events == 0:
            return self.format_error(error="Empty Input",
                                     reason="no events found in events.jsonl",
                                     context=self.jsonl_path)
        return json.dumps({
            "status": "ok",
            "hepmc": os.path.relpath(dst, self.base_directory),
            "n_events": n_events,
            "n_particles": n_particles,
        })


class HepMCToJSONLTool(_SandboxedTool):
    """
    Convert a HepMC3 ASCII file into the evtjsonl-1.0 schema.

    Input (runtime):
      - hepmc_path: input .hepmc file (relative to base_directory)
      - jsonl_path: output events.jsonl path
      - finals_only: keep only status==1 particles (default True)

    Output:
      JSON string: {"status": "ok", "events_jsonl": "<relative path>",
                    "n_events": <int>}

    Schema per line matches LHEToJSONLTool's output (evtjsonl-1.0):
    {"schema", "event_id", "finals_only", "data": {"n", "particles":
    [{"i", "id", "status", "px", "py", "pz", "E", "m"}]}}.
    """

    hepmc_path: str = RuntimeField(description="Path to the HepMC3 ASCII file")
    jsonl_path: str = RuntimeField(description="Relative path to write events.jsonl")
    finals_only: bool = RuntimeField(default=True,
                                     description="Keep only status==1 particles")

    def _run(self) -> str:
        try:
            pyhepmc = _require_pyhepmc()
        except ImportError as e:
            return self.format_error(error="Missing Dependency", reason=str(e),
                                     suggestion="pip install pyhepmc")
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        src = self._safe_path(self.hepmc_path)
        dst = self._safe_path(self.jsonl_path)
        if src is None:
            return self.format_error(error="Access Denied",
                                     reason="hepmc_path is outside allowed base_directory",
                                     context=self.hepmc_path)
        if dst is None:
            return self.format_error(error="Access Denied",
                                     reason="jsonl_path escapes base_directory",
                                     context=self.jsonl_path)
        if not os.path.exists(src):
            return self.format_error(error="File Not Found",
                                     reason="HepMC file not found", context=self.hepmc_path)

        os.makedirs(os.path.dirname(dst) or self.base_directory, exist_ok=True)
        n_events = 0
        try:
            with pyhepmc.open(src) as inp, open(dst, "w") as out:
                for ev in inp:
                    particles = []
                    for gp in ev.particles:
                        status = int(gp.status)
                        if self.finals_only and status != _FINAL_STATUS:
                            continue
                        mom = gp.momentum
                        particles.append({
                            "i": len(particles),
                            "id": int(gp.pid),
                            "status": status,
                            "px": float(mom.px),
                            "py": float(mom.py),
                            "pz": float(mom.pz),
                            "E": float(mom.e),
                            "m": float(gp.generated_mass),
                        })
                    out.write(json.dumps({
                        "schema": SCHEMA_VERSION,
                        "event_id": int(ev.event_number),
                        "finals_only": bool(self.finals_only),
                        "full_history": False,
                        "data": {"n": len(particles), "particles": particles},
                    }) + "\n")
                    n_events += 1
        except Exception as e:
            return self.format_error(error="Conversion Error",
                                     reason=f"{type(e).__name__}: {e}",
                                     context=self.hepmc_path,
                                     suggestion="Ensure the file is HepMC3 ASCII")
        if n_events == 0:
            return self.format_error(error="Empty Input",
                                     reason="no events found in HepMC file",
                                     context=self.hepmc_path)
        return json.dumps({
            "status": "ok",
            "events_jsonl": os.path.relpath(dst, self.base_directory),
            "n_events": n_events,
        })
