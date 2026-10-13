"""
# test_meson_decay_to_llp.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Unit tests for MesonDecayToLLPTool: decaying a harvested forward parent flux
to a weighted LLP flux with decay-in-flight production vertices. Mirrors the
conventions of test_llp_tools.py (in-test toy inputs; no ground-truth amplitude
in the tool).
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent                    # .../heptapod/tools/llp
REPO_ROOT = TOOL_DIR.parent.parent                      # .../heptapod
sys.path.insert(0, str(REPO_ROOT))

from tools.llp.meson_decay_to_llp import MesonDecayToLLPTool

base_directory = str(TOOL_DIR / "test_files_meson")

M_PARENT = 0.493677        # kaon
M_PHI = 0.25
M_MU = 0.1056584
B_HAT = 1.3e-5
CTAU_K = 3.712             # kaon proper decay length [m]
N_PARENTS = 200
W_PER = 3.6 / N_PARENTS    # per-collision weight per harvested parent


def _write_parent_flux(path):
    """A small mock forward-kaon flux: N parents with per-collision weights."""
    rng = np.random.default_rng(7)
    p = rng.gamma(2.0, 40.0, N_PARENTS)        # forward momenta ~ tens of GeV
    th = rng.gamma(2.0, 0.001, N_PARENTS)      # small forward angles
    az = rng.uniform(0.0, 2.0 * np.pi, N_PARENTS)
    px = p * np.sin(th) * np.cos(az)
    py = p * np.sin(th) * np.sin(az)
    pz = p * np.cos(th)
    E = np.sqrt(p ** 2 + M_PARENT ** 2)
    with open(path, "w") as fh:
        for i in range(N_PARENTS):
            fh.write(json.dumps({
                "parent": "K+", "parent_pid": 321, "E": float(E[i]),
                "px": float(px[i]), "py": float(py[i]), "pz": float(pz[i]),
                "weight_per_collision": float(W_PER)}) + "\n")


def _spectrum_table(path, m_phi=M_PHI, n=60):
    """Toy `table` spectrum (unnormalized quadratic bump); no amplitude."""
    x_lo = 2.0 * m_phi / M_PARENT
    x_hi = 1.0 + (m_phi ** 2 - M_MU ** 2) / M_PARENT ** 2
    xs = [x_lo + (x_hi - x_lo) * i / (n - 1) for i in range(n)]
    pdf = [(x - x_lo) * (x_hi - x) for x in xs]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# in-test analytic bump pdf (unnormalized)"])
        w.writerow(["x", "pdf"])
        for x, p in zip(xs, pdf):
            w.writerow([x, p])


def _setup():
    base = Path(base_directory)
    base.mkdir(parents=True, exist_ok=True)
    _write_parent_flux(base / "parents_K.jsonl")
    _spectrum_table(base / "spectrum.csv")


def _run(m_phi=M_PHI, ctau=CTAU_K, n_strata=8, seed=42, out="llp/phi_K.jsonl"):
    tool = MesonDecayToLLPTool(
        base_directory=base_directory,
        parent_flux_path="parents_K.jsonl",
        spectrum_spec="spectrum.csv",
        m_phi_gev=m_phi,
        parent_mass_gev=M_PARENT,
        B_hat=B_HAT,
        ctau_parent_m=ctau,
        n_strata=n_strata,
        seed=seed,
        output_path=out,
    )
    tool._setup()
    return json.loads(tool._run())


def _records(res):
    p = Path(base_directory) / res["output_path"]
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _recs(rel):
    p = Path(base_directory) / rel
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_grid_mode():
    print(">> grid mode: one call sweeps the mass grid, shared vertices ...")
    _setup()
    K = 8
    grid = [{"m_phi_gev": 0.22, "spectrum_spec": "spectrum.csv", "B_hat": 1e-5},
            {"m_phi_gev": 0.24, "spectrum_spec": "spectrum.csv", "B_hat": 2e-5},
            {"m_phi_gev": 0.25, "spectrum_spec": "spectrum.csv", "B_hat": 3e-5}]
    tool = MesonDecayToLLPTool(
        base_directory=base_directory, parent_flux_path="parents_K.jsonl",
        parent_mass_gev=M_PARENT, ctau_parent_m=CTAU_K, n_strata=K, seed=1,
        grid=grid, output_dir="gridout")
    tool._setup()
    res = json.loads(tool._run())
    assert res["status"] == "ok" and res["mode"] == "grid", res
    assert res["n_masses"] == 3
    assert res["n_samples_total"] == 3 * N_PARENTS * K
    per = {round(m["m_phi_gev"], 3): m for m in res["masses"]}
    # one file per mass, each N*K records, LLP on-shell AT THAT MASS
    for mval, m in per.items():
        recs = _recs(m["path"])
        assert len(recs) == N_PARENTS * K, (mval, len(recs))
        for r in recs[:30]:
            m2 = r["E"] ** 2 - r["px"] ** 2 - r["py"] ** 2 - r["pz"] ** 2
            assert abs(math.sqrt(max(m2, 0.0)) - mval) < 1e-6, (mval, r)
        # weight scales with that mass's B_hat
        assert abs(m["sum_weights"] - N_PARENTS * W_PER * m["B_hat"]) \
            / (N_PARENTS * W_PER * m["B_hat"]) < 1e-9
    # SHARED vertices: production points are mass-independent (sampled once),
    # so vz per event index is identical across masses
    r0 = _recs(per[0.22]["path"])
    r1 = _recs(per[0.24]["path"])
    assert all(abs(a["vx"] - b["vx"]) < 1e-12 and abs(a["vz"] - b["vz"]) < 1e-12
               for a, b in zip(r0, r1)), "vertices must be shared across masses"
    print("[OK] 3 masses in one call, on-shell per mass, vertices shared")


def test_path_traversal_prevention():
    print(">> traversal prevention ...")
    _setup()
    tool = MesonDecayToLLPTool(
        base_directory=base_directory,
        parent_flux_path="../../../etc/passwd",
        spectrum_spec="spectrum.csv", m_phi_gev=M_PHI,
        parent_mass_gev=M_PARENT, B_hat=B_HAT, seed=1,
        output_path="llp/o.jsonl")
    tool._setup()
    assert "denied" in tool._run().lower()
    tool = MesonDecayToLLPTool(
        base_directory=base_directory,
        parent_flux_path="parents_K.jsonl",
        spectrum_spec="spectrum.csv", m_phi_gev=M_PHI,
        parent_mass_gev=M_PARENT, B_hat=B_HAT, seed=1,
        output_path="../../../tmp/evil.jsonl")
    tool._setup()
    assert "denied" in tool._run().lower()
    print("[OK] input and output traversal rejected")


def test_decay_and_weights():
    print(">> decay + weights + on-shell LLP + vertex ...")
    _setup()
    K = 8
    res = _run(n_strata=K)
    assert res["status"] == "ok", res
    assert res["n_samples"] == N_PARENTS * K, res
    assert res["n_strata"] == K
    assert res["weight_convention"] == "g2_stripped_per_primary_interaction"
    assert not res["above_kinematic_cutoff"]
    # sum of LLP weights = sum(parent weights) * B_hat (the 1/K strata sum to 1)
    expected = N_PARENTS * W_PER * B_HAT
    assert abs(res["sum_weights"] - expected) / expected < 1e-9, \
        (res["sum_weights"], expected)
    recs = _records(res)
    assert len(recs) == N_PARENTS * K
    for r in recs[:100]:
        m2 = r["E"] ** 2 - r["px"] ** 2 - r["py"] ** 2 - r["pz"] ** 2
        assert abs(math.sqrt(max(m2, 0.0)) - M_PHI) < 1e-6, r
        assert all(k in r for k in ("vx", "vy", "vz"))
        assert r["parent_channel"] == "K+"
        assert abs(r["event_weight_g2_stripped"]
                   - W_PER * B_HAT / K) / (W_PER * B_HAT / K) < 1e-9
    print("[OK] N*K records, sum_weights exact, LLP on-shell, vertex present")


def test_decay_in_flight_vs_prompt():
    print(">> decay-in-flight vs prompt vertices ...")
    _setup()
    res_dif = _run(ctau=CTAU_K, n_strata=8, out="llp/dif.jsonl")
    vz = [r["vz"] for r in _records(res_dif)]
    assert max(vz) > 1.0, "kaon decay-in-flight must place vertices downstream"
    assert min(vz) >= 0.0
    res_prompt = _run(ctau=0.0, n_strata=1, out="llp/prompt.jsonl")
    assert all(abs(r["vz"]) < 1e-9 and abs(r["vx"]) < 1e-9
               for r in _records(res_prompt)), "prompt vertices at the IP"
    print("[OK] decay-in-flight spreads vertices in z; prompt sits at the IP")


def test_ctau_auto_from_pid():
    """Left at the -1 sentinel, the parent c*tau is auto-set from the flux PID;
    an explicit value overrides; an untabulated PID falls back to prompt + note."""
    print(">> parent c*tau auto-resolution from PID ...")
    _setup()

    # auto (default -1): the kaon flux carries PID 321 -> SM c*tau = 3.712 m
    res_auto = _run(ctau=-1.0, n_strata=8, out="llp/auto.jsonl")
    assert res_auto["ctau_source"] == "auto_pid"
    assert abs(res_auto["ctau_parent_m"] - 3.712) < 1e-6
    # and it actually spreads vertices downstream, i.e. the value was used
    assert max(r["vz"] for r in _records(res_auto)) > 1.0

    # explicit override wins over the table
    res_exp = _run(ctau=1.5, n_strata=8, out="llp/exp.jsonl")
    assert res_exp["ctau_source"] == "explicit"
    assert abs(res_exp["ctau_parent_m"] - 1.5) < 1e-12

    # an untabulated PID -> prompt fallback, flagged in a note
    flux = Path(base_directory) / "parents_unknown.jsonl"
    lines = (Path(base_directory) / "parents_K.jsonl").read_text().splitlines()
    with open(flux, "w") as fh:
        for ln in lines:
            r = json.loads(ln); r["parent_pid"] = 9999
            fh.write(json.dumps(r) + "\n")
    tool = MesonDecayToLLPTool(
        base_directory=base_directory, parent_flux_path="parents_unknown.jsonl",
        spectrum_spec="spectrum.csv", m_phi_gev=M_PHI, parent_mass_gev=M_PARENT,
        B_hat=B_HAT, ctau_parent_m=-1.0, n_strata=1, seed=1,
        output_path="llp/unknown.jsonl")
    tool._setup()
    res_unk = json.loads(tool._run())
    assert res_unk["ctau_source"] == "auto_fallback_prompt"
    assert res_unk["ctau_parent_m"] == 0.0
    assert "note" in res_unk and "9999" in res_unk["note"]
    print("[OK] auto from PID, explicit override, and untabulated fallback")


def test_scheme_prefix_and_optional_single_fields():
    print(">> table: prefix tolerated; single-mass fields not required ...")
    _setup()
    # a 'table:' scheme prefix on the spectrum PATH is stripped, not an error
    tool = MesonDecayToLLPTool(
        base_directory=base_directory, parent_flux_path="parents_K.jsonl",
        spectrum_spec="table:spectrum.csv", m_phi_gev=M_PHI,
        parent_mass_gev=M_PARENT, B_hat=B_HAT, ctau_parent_m=CTAU_K,
        n_strata=4, seed=1, output_path="llp/scheme.jsonl")
    tool._setup()
    res = json.loads(tool._run())
    assert res["status"] == "ok" and res["n_samples"] > 0, res
    # spectrum_spec / output_path now carry a default -> NOT MCP-required, so a
    # grid-mode call needs neither (the friction that cost Sonnet 3 retries)
    mf = MesonDecayToLLPTool.model_fields
    assert mf["spectrum_spec"].default == "" and mf["output_path"].default == ""
    # single mode without them gives a CLEAR error (not an opaque path error)
    bad = MesonDecayToLLPTool(
        base_directory=base_directory, parent_flux_path="parents_K.jsonl",
        m_phi_gev=M_PHI, parent_mass_gev=M_PARENT, B_hat=B_HAT, seed=1)
    bad._setup()
    out = bad._run()
    assert "single-mass mode needs" in out, out
    print("[OK] prefix stripped, single-mass fields optional, clear error")


def test_kinematic_cutoff():
    print(">> kinematic cutoff ...")
    _setup()
    # m_phi above the (toy table) rest-energy cutoff -> channel closed
    res = _run(m_phi=0.35)
    assert res["above_kinematic_cutoff"], res
    assert res["n_samples"] == 0
    assert res["sum_weights"] == 0.0
    print("[OK] closed channel -> ok with 0 samples")


def run_all():
    try:
        test_path_traversal_prevention()
        test_decay_and_weights()
        test_decay_in_flight_vs_prompt()
        test_ctau_auto_from_pid()
        test_scheme_prefix_and_optional_single_fields()
        test_kinematic_cutoff()
        test_grid_mode()
        print("\nAll MesonDecayToLLP tests passed.")
    finally:
        shutil.rmtree(base_directory, ignore_errors=True)


if __name__ == "__main__":
    run_all()


def test_grid_path_is_equivalent_to_an_inline_grid():
    """A long grid can come from a file instead of a hand-copied array.

    A 2026-08 trial reported hand-copying `grid`/`events_paths`/`g_grid` into
    tool calls 20+ times and typo'd one entry (a duplicated coupling) that was
    harmless only by luck. Generating the list with a script and pointing at it
    removes that whole class of transcription error, so the file route must
    produce byte-identical output to the inline route.
    """
    print(">> grid_path: file-supplied grid == inline grid ...")
    grid = [{"m_phi_gev": 0.22, "spectrum_spec": "spectrum.csv", "B_hat": 1e-5},
            {"m_phi_gev": 0.24, "spectrum_spec": "spectrum.csv", "B_hat": 2e-5}]

    def run(**kw):
        _setup()
        tool = MesonDecayToLLPTool(
            base_directory=base_directory, parent_flux_path="parents_K.jsonl",
            parent_mass_gev=M_PARENT, ctau_parent_m=CTAU_K, n_strata=4, seed=1,
            output_dir="gridout", **kw)
        tool._setup()
        res = json.loads(tool._run())
        assert res["status"] == "ok", res
        return res, [_recs(m["path"]) for m in res["masses"]]

    inline_res, inline_recs = run(grid=grid)
    Path(base_directory, "grid.json").write_text(json.dumps(grid))
    file_res, file_recs = run(grid_path="grid.json")

    assert file_res["mode"] == "grid" and file_res["n_masses"] == 2
    assert file_res["n_samples_total"] == inline_res["n_samples_total"]
    assert inline_recs == file_recs, "file-supplied grid must be identical"

    # the wrapped form is accepted too
    Path(base_directory, "grid2.json").write_text(json.dumps({"grid": grid}))
    wrapped, _ = run(grid_path="grid2.json")
    assert wrapped["n_masses"] == 2, wrapped
    print("[OK] grid_path reproduces the inline grid exactly; {'grid': [...]} ok")


def test_grid_path_failures_are_actionable():
    print(">> grid_path: missing / malformed files fail loudly ...")
    _setup()

    def err(path):
        """Tool errors come back as a formatted STRING, not JSON."""
        tool = MesonDecayToLLPTool(
            base_directory=base_directory, parent_flux_path="parents_K.jsonl",
            parent_mass_gev=M_PARENT, ctau_parent_m=CTAU_K, n_strata=4, seed=1,
            output_dir="gridout", grid_path=path)
        tool._setup()
        out = tool._run()
        assert out.lstrip().startswith("Error:"), out[:200]
        return out

    assert "grid_path" in err("nope.json")
    Path(base_directory, "bad.json").write_text("{not json")
    assert "JSON" in err("bad.json")
    Path(base_directory, "empty.json").write_text("[]")
    assert "non-empty" in err("empty.json")
    print("[OK] missing, malformed and empty grid files all refused")


def test_n_strata_advice_appears_only_when_coarse():
    """Which boundary K controls, said where the caller will see it."""
    print(">> n_strata advice surfaces in the result at coarse K ...")
    for K, expect in ((16, True), (64, False)):
        _setup()
        tool = MesonDecayToLLPTool(
            base_directory=base_directory, parent_flux_path="parents_K.jsonl",
            parent_mass_gev=M_PARENT, ctau_parent_m=CTAU_K, n_strata=K, seed=1,
            m_phi_gev=M_PHI, spectrum_spec="spectrum.csv", B_hat=B_HAT,
            output_path="llp/phi_K.jsonl")
        tool._setup()
        res = json.loads(tool._run())
        assert res["status"] == "ok", res
        has = "n_strata_advice" in res
        assert has is expect, (K, res.get("n_strata_advice"))
        if has:
            adv = res["n_strata_advice"]
            assert "STRONG" in adv and "64" in adv, adv
    print("[OK] advice at K=16, silent at K=64")
