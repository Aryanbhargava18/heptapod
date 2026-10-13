#!/usr/bin/env python3
"""
# test_hepmc.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
HepMC conversion tools test suite.

Tests cover:
- evtjsonl -> HepMC3 -> evtjsonl round-trip (momenta, ids, counts)
- finals_only filtering on read (beam particles dropped)
- path traversal prevention (security)
- empty/missing input handling
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.hepmc import HepMCToJSONLTool, JSONLToHepMCTool  # noqa: E402

try:
    import pyhepmc  # noqa: F401
    HAVE_PYHEPMC = True
except ImportError:
    HAVE_PYHEPMC = False


def _event(event_id, particles):
    return {
        "schema": "evtjsonl-1.0",
        "event_id": event_id,
        "finals_only": True,
        "full_history": False,
        "data": {"n": len(particles), "particles": particles},
    }


def _particle(i, pid, px, py, pz, e, m=0.0):
    return {"i": i, "id": pid, "px": px, "py": py, "pz": pz, "E": e, "m": m}


@unittest.skipUnless(HAVE_PYHEPMC, "pyhepmc not installed")
class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = self._tmp.name
        self.events = [
            _event(0, [_particle(0, 13, 10.0, 0.0, 5.0, 11.2, 0.105),
                       _particle(1, -13, -10.0, 0.0, 3.0, 10.5, 0.105)]),
            _event(1, [_particle(0, 22, 0.0, 25.0, -4.0, 25.3)]),
        ]
        with open(os.path.join(self.base, "events.jsonl"), "w") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _to_hepmc(self):
        tool = JSONLToHepMCTool(jsonl_path="events.jsonl",
                                hepmc_path="events.hepmc")
        tool.base_directory = self.base
        out = tool._run()
        self.assertNotIn("Error", out.split("\n")[0])
        return json.loads(out)

    def test_write(self):
        res = self._to_hepmc()
        self.assertEqual(res["n_events"], 2)
        self.assertEqual(res["n_particles"], 3)
        text = Path(self.base, "events.hepmc").read_text()
        self.assertIn("HepMC::Version", text)

    def test_round_trip_preserves_momenta_and_ids(self):
        self._to_hepmc()
        back = HepMCToJSONLTool(hepmc_path="events.hepmc",
                                jsonl_path="events_back.jsonl")
        back.base_directory = self.base
        res = json.loads(back._run())
        self.assertEqual(res["n_events"], 2)
        lines = [json.loads(line) for line in
                 Path(self.base, "events_back.jsonl").read_text().splitlines()]
        for orig, rt in zip(self.events, lines):
            self.assertEqual(rt["schema"], "evtjsonl-1.0")
            self.assertEqual(rt["event_id"], orig["event_id"])
            # finals_only on read: the bookkeeping beams (status 4) are
            # dropped, leaving exactly the original particles.
            self.assertEqual(rt["data"]["n"], orig["data"]["n"])
            for p0, p1 in zip(orig["data"]["particles"],
                              rt["data"]["particles"]):
                self.assertEqual(p1["id"], p0["id"])
                for k in ("px", "py", "pz", "E"):
                    self.assertAlmostEqual(p1[k], p0[k], places=5)

    def test_beams_kept_without_finals_only(self):
        self._to_hepmc()
        back = HepMCToJSONLTool(hepmc_path="events.hepmc",
                                jsonl_path="all.jsonl", finals_only=False)
        back.base_directory = self.base
        json.loads(back._run())
        first = json.loads(
            Path(self.base, "all.jsonl").read_text().splitlines()[0])
        # 2 original finals + 2 beam protons
        self.assertEqual(first["data"]["n"], 4)
        self.assertEqual(
            sum(1 for p in first["data"]["particles"] if p["id"] == 2212), 2)


@unittest.skipUnless(HAVE_PYHEPMC, "pyhepmc not installed")
class TestSafety(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_path_escape_rejected(self):
        tool = JSONLToHepMCTool(jsonl_path="../outside.jsonl",
                                hepmc_path="out.hepmc")
        tool.base_directory = self.base
        self.assertIn("Access Denied", tool._run())

    def test_missing_input(self):
        tool = JSONLToHepMCTool(jsonl_path="nope.jsonl", hepmc_path="o.hepmc")
        tool.base_directory = self.base
        self.assertIn("File Not Found", tool._run())

    def test_empty_input(self):
        Path(self.base, "empty.jsonl").write_text("")
        tool = JSONLToHepMCTool(jsonl_path="empty.jsonl", hepmc_path="o.hepmc")
        tool.base_directory = self.base
        self.assertIn("Empty Input", tool._run())


if __name__ == "__main__":
    unittest.main()
