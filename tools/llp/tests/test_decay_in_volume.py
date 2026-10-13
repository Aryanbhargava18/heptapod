"""
# test_decay_in_volume.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Unit tests for the DecayInVolume* yield tools: vertex-aware decay-in-volume
yields from g^2-stripped LLP records (MesonDecayToLLPTool output, schema
llpflux-2.0). Exercises the new physics the redesign added on top of the
prompt-at-IP tool: production-vertex propagation, the z_prod absorber, the
off-axis geometry, and the two_track/photon/none acceptance modes -- plus the
portal/ctau lifetime duality carried over from v0. Inputs are toy LLP records
written in-test (no ground-truth amplitude in the tool).
"""
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

from tools.llp.decay_in_volume import (
    DecayInVolumeVsCouplingTool,
    DecayInVolumeVsLifetimeTool,
)

base_directory = str(TOOL_DIR / "test_files_div")

M_MU = 0.1056584
M_PHI = 0.25                       # > 2 m_mu, so phi -> mu mu is open
N_INT = 2.7e18

# A far-forward decay volume with a production-region gap and detector plane.
GEOMETRY = """
geometry:
  z_min_m: 112.0
  z_max_m: 163.0
  r_volume_m: 0.92
  z_det_m: 165.0
  r_det_m: 0.92
  z_prod_m: 50.0
"""
GEOMETRY_OFFAXIS = """
geometry:
  z_min_m: 112.0
  z_max_m: 163.0
  r_volume_m: 0.92
  z_det_m: 165.0
  r_det_m: 0.92
  z_prod_m: 50.0
  x_off_m: 3.0
"""


def _width_ref(m_phi, m_mu):
    """Gamma(phi -> mu mu) at g = 1: (m_phi/8pi) beta^3, beta = sqrt(1-4mu^2/m^2)."""
    beta = math.sqrt(max(1.0 - 4.0 * m_mu ** 2 / m_phi ** 2, 0.0))
    return m_phi / (8.0 * math.pi) * beta ** 3


def _write_llp(path, n=500, vz=0.0, seed=3, w=1e-3, theta_max=0.003,
               m_phi=M_PHI):
    """Toy forward LLP records (schema llpflux-2.0) with a production vertex.

    vz may be a scalar (common z for all) or a length-n array. Directions are
    forward within theta_max so the rays enter the on-axis volume."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(80.0, 300.0, n)
    th = rng.uniform(0.0, theta_max, n)
    az = rng.uniform(0.0, 2.0 * np.pi, n)
    px = p * np.sin(th) * np.cos(az)
    py = p * np.sin(th) * np.sin(az)
    pz = p * np.cos(th)
    E = np.sqrt(p ** 2 + m_phi ** 2)
    vzs = np.full(n, float(vz)) if np.isscalar(vz) else np.asarray(vz, float)
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "schema": "llpflux-2.0", "event_id": i,
                "E": float(E[i]), "px": float(px[i]), "py": float(py[i]),
                "pz": float(pz[i]),
                "vx": 0.0, "vy": 0.0, "vz": float(vzs[i]),
                "parent_channel": "K+",
                "event_weight_g2_stripped": float(w)}) + "\n")


def _setup(geom=GEOMETRY):
    base = Path(base_directory)
    base.mkdir(parents=True, exist_ok=True)
    (base / "geometry.yaml").write_text(geom)
    _write_llp(base / "flux.jsonl")


def _tool(**kw):
    # Pick the concrete tool by lifetime axis: ctau_grid_m -> lifetime tool,
    # else the coupling (portal) tool. The two share the engine under test.
    kw.setdefault("base_directory", base_directory)
    kw.setdefault("events_path", "flux.jsonl")
    kw.setdefault("geometry_path", "geometry.yaml")
    kw.setdefault("m_phi_gev", M_PHI)
    kw.setdefault("n_int", N_INT)
    kw.setdefault("output_path", "yields/out.json")
    # These tests assert on yield VALUES, so ask for the full array in the
    # response; summary mode (the tool default) is covered separately below.
    kw.setdefault("response_verbosity", "full")
    cls = (DecayInVolumeVsLifetimeTool if "ctau_grid_m" in kw
           else DecayInVolumeVsCouplingTool)
    t = cls(**kw)
    t._setup()
    return t


def _audit(res):
    p = Path(base_directory) / res["audit_path"]
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_path_traversal_prevention():
    print(">> traversal prevention ...")
    _setup()
    t = _tool(events_path="../../../etc/passwd",
              width_ref_gev=1e-3, g_grid=[1e-7])
    assert "denied" in t._run().lower()
    t = _tool(output_path="../../../tmp/evil.json",
              width_ref_gev=1e-3, g_grid=[1e-7])
    assert "denied" in t._run().lower()
    print("[OK] input and output traversal rejected")


def test_portal_g4_reweighting():
    print(">> portal mode: g^4 rise from exact reweighting ...")
    _setup()
    wref = _width_ref(M_PHI, M_MU)
    # tiny couplings -> lam >> baseline, so P_dec is linear in 1/ctau ~ g^2 and
    # N_sig = g^2 * ... * P_dec ~ g^4 (one event set covers both g's exactly).
    res = json.loads(_tool(width_ref_gev=wref, g_grid=[1e-9, 2e-9],
                           acceptance="two_track")._run())
    assert res["status"] == "ok", res
    assert res["acceptance"] == "two_track"
    assert res["lifetime_mode"] == "portal"
    assert 0 < res["n_pass_geometry"] <= res["n_events"]
    assert 0 < res["n_pass_acceptance"] <= res["n_pass_geometry"]
    n1 = res["yields"][0]["n_sig"]
    n2 = res["yields"][1]["n_sig"]
    assert n1 > 0 and n2 > 0
    # doubling g quadruples-squared: (2)^4 = 16 in the g^4 regime
    assert abs(n2 / n1 - 16.0) / 16.0 < 0.02, (n1, n2, n2 / n1)
    print(f"[OK] N(2g)/N(g) = {n2 / n1:.3f} ~ 16 (g^4)")


def test_ctau_mode_and_axis_separation():
    print(">> ctau (lifetime) tool + per-tool axis separation ...")
    _setup()
    res = json.loads(_tool(ctau_grid_m=[0.05, 0.1, 0.5, 2.0])._run())
    assert res["status"] == "ok" and res["lifetime_mode"] == "ctau"
    assert len(res["yields"]) == 4
    assert all(y["n_sig"] >= 0.0 for y in res["yields"])
    assert any(y["n_sig"] > 0.0 for y in res["yields"])
    # The split removes mode ambiguity by construction: each tool exposes only
    # its own axis. A coupling tool with no g_grid errors on the missing axis;
    # a lifetime tool with no ctau_grid_m likewise.
    no_g = DecayInVolumeVsCouplingTool(
        base_directory=base_directory, events_path="flux.jsonl",
        geometry_path="geometry.yaml", m_phi_gev=M_PHI, n_int=N_INT,
        width_ref_gev=_width_ref(M_PHI, M_MU), output_path="yields/out.json")
    no_g._setup()
    assert "g_grid" in no_g._run().lower()
    no_ctau = DecayInVolumeVsLifetimeTool(
        base_directory=base_directory, events_path="flux.jsonl",
        geometry_path="geometry.yaml", m_phi_gev=M_PHI, n_int=N_INT,
        output_path="yields/out.json")
    no_ctau._setup()
    assert "ctau_grid_m" in no_ctau._run().lower()
    print("[OK] ctau yields computed; each tool guards its own axis")


def test_z_prod_absorber():
    print(">> z_prod production-region absorber ...")
    _setup()
    # half the LLPs are produced downstream of z_prod=50 (must be swept),
    # half at the IP (must survive to the volume).
    n = 400
    vz = np.where(np.arange(n) % 2 == 0, 0.0, 60.0)
    _write_llp(Path(base_directory) / "flux.jsonl", n=n, vz=vz, seed=5)
    res = json.loads(_tool(ctau_grid_m=[0.2])._run())
    audit = _audit(res)
    for a in audit:
        if a["vz_prod_m"] >= 50.0:
            assert not a["geom_pass"], a       # beyond z_prod -> swept
    # some upstream (vz=0) events do reach the volume
    assert any(a["geom_pass"] for a in audit if a["vz_prod_m"] < 50.0)
    print("[OK] vertices past z_prod are absorbed; upstream ones survive")


def test_off_axis_reduces_geometry():
    print(">> off-axis geometry sees fewer LLPs ...")
    _setup(GEOMETRY)
    on = json.loads(_tool(ctau_grid_m=[0.2])._run())
    # rebuild with the far off-axis geometry (detector 3 m off the beam line)
    (Path(base_directory) / "geometry.yaml").write_text(GEOMETRY_OFFAXIS)
    off = json.loads(_tool(ctau_grid_m=[0.2])._run())
    assert on["n_pass_geometry"] > 0
    assert off["n_pass_geometry"] < on["n_pass_geometry"], \
        (on["n_pass_geometry"], off["n_pass_geometry"])
    print(f"[OK] on-axis {on['n_pass_geometry']} -> off-axis "
          f"{off['n_pass_geometry']} geometry passes")


def test_photon_acceptance_and_closed_two_track():
    print(">> photon acceptance below 2 m_mu; two_track closed there ...")
    base = Path(base_directory)
    base.mkdir(parents=True, exist_ok=True)
    (base / "geometry.yaml").write_text(GEOMETRY)
    m_light = 0.05                                  # below 2 m_mu = 0.211
    _write_llp(base / "flux.jsonl", n=400, m_phi=m_light, seed=8)
    # two_track is kinematically closed for a sub-threshold scalar
    closed = _tool(m_phi_gev=m_light, ctau_grid_m=[0.2],
                   acceptance="two_track")._run()
    assert "closed" in closed.lower(), closed
    # photon acceptance is well-defined and contributes
    ph = json.loads(_tool(m_phi_gev=m_light, ctau_grid_m=[0.2],
                          acceptance="photon")._run())
    assert ph["status"] == "ok" and ph["acceptance"] == "photon"
    assert ph["n_pass_acceptance"] > 0
    audit = _audit(ph)
    assert all(a["acc_pass"] is not None for a in audit)
    print("[OK] photon mode works sub-threshold where two_track is closed")


def test_none_acceptance_and_br_visible():
    print(">> acceptance='none' equals geometry; br_visible scales yield ...")
    _setup()
    full = json.loads(_tool(ctau_grid_m=[0.2], acceptance="none",
                            br_visible=1.0)._run())
    assert full["n_pass_acceptance"] == full["n_pass_geometry"]
    assert all(a["acc_pass"] is None for a in _audit(full))
    half = json.loads(_tool(ctau_grid_m=[0.2], acceptance="none",
                            br_visible=0.5)._run())
    y_full = full["yields"][0]["n_sig"]
    y_half = half["yields"][0]["n_sig"]
    assert y_full > 0
    assert abs(y_half / y_full - 0.5) < 1e-9, (y_full, y_half)
    print("[OK] none == geometry; br_visible=0.5 halves the yield")


def test_empty_input_zero_yields():
    print(">> empty flux -> zero yields, not an error ...")
    _setup()
    (Path(base_directory) / "empty.jsonl").write_text("")
    res = json.loads(_tool(events_path="empty.jsonl",
                           ctau_grid_m=[0.2, 1.0])._run())
    assert res["status"] == "ok"
    assert res["n_events"] == 0 and res["n_pass_geometry"] == 0
    assert all(y["n_sig"] == 0.0 for y in res["yields"])
    print("[OK] kinematically-closed channel yields zeros")


def test_sum_weights_and_width_note():
    """The result echoes sum_weights of the input flux (completeness signal),
    and a width_ref_gev so large that Gamma = g^2*width_ref reaches m_phi is
    flagged as 'width exceeds mass'."""
    print(">> sum_weights surfaced + width-exceeds-mass guard ...")
    _setup()
    _write_llp(Path(base_directory) / "flux.jsonl", n=500, w=1e-3)
    # sum_weights = 500 * 1e-3 = 0.5, regardless of geometry/acceptance
    res = json.loads(_tool(width_ref_gev=_width_ref(M_PHI, M_MU),
                           g_grid=[1e-7, 3e-7])._run())
    assert abs(res["sum_weights"] - 0.5) < 1e-9, res.get("sum_weights")

    # A grossly oversized width_ref: g^2 * width_ref >= m_phi -> not a resonance.
    big = M_PHI / (1e-6 ** 2) * 10.0
    res2 = json.loads(_tool(width_ref_gev=big, g_grid=[1e-6])._run())
    assert "width exceeds mass" in (res2.get("note") or ""), res2.get("note")
    print("[OK] sum_weights reported; oversized width_ref flagged")


def test_grid_diagnostic_band_runs_off():
    """With n_target set, the diagnostic flags a reach band that runs off the
    g-grid even when the PEAK is interior (n001's failure), and stays quiet when
    the band is fully bracketed."""
    print(">> grid diagnostic: band-runs-off-grid vs bracketed ...")
    t = DecayInVolumeVsCouplingTool(base_directory=str(TOOL_DIR))
    t.n_target = 3.0
    # interior peak, but N_sig >> N* all the way to g_max -> truncated
    off = [{"g": 1e-7, "n_sig": 0.02}, {"g": 1e-5, "n_sig": 200},
           {"g": 7.7e-4, "n_sig": 2.1e6}, {"g": 1e-3, "n_sig": 1.5e6}]
    diag, note = t._grid_diagnostic(off, 3.0)
    assert diag["peak_at_boundary"] is False        # old note would miss it
    assert diag["band_runs_off_high"] is True
    assert note and "truncated" in note
    # band fully inside the grid -> no note
    ok = [{"g": 1e-7, "n_sig": 0.1}, {"g": 1e-5, "n_sig": 50},
          {"g": 1e-3, "n_sig": 0.5}]
    _, note2 = t._grid_diagnostic(ok, 3.0)
    assert note2 is None
    print("[OK] truncated band flagged (interior peak); bracketed band silent")


def test_response_verbosity_summary_default():
    print(">> response_verbosity: summary (default) omits yields, full inlines them ...")
    assert DecayInVolumeVsCouplingTool.model_fields[
        "response_verbosity"].default == "summary"
    wref = _width_ref(M_PHI, M_MU)
    g_grid = [1e-9, 2e-9, 1e-8]
    summ = json.loads(_tool(width_ref_gev=wref, g_grid=g_grid,
                            acceptance="two_track",
                            response_verbosity="summary")._run())
    assert "yields" not in summ, "summary must not inline the yields array"
    assert summ["n_yield_points"] == len(g_grid)
    assert summ.get("grid_diagnostic") is not None
    file_yields = json.loads(
        (Path(base_directory) / summ["output_path"]).read_text())["yields"]
    assert len(file_yields) == len(g_grid), "full table always written to output_path"
    full = json.loads(_tool(width_ref_gev=wref, g_grid=g_grid,
                            acceptance="two_track",
                            response_verbosity="full")._run())
    assert len(full["yields"]) == len(g_grid)
    print("[OK] summary omits array + writes full file; full inlines it")


def run_all():
    try:
        test_grid_diagnostic_band_runs_off()
        test_response_verbosity_summary_default()
        test_path_traversal_prevention()
        test_portal_g4_reweighting()
        test_ctau_mode_and_axis_separation()
        test_z_prod_absorber()
        test_off_axis_reduces_geometry()
        test_photon_acceptance_and_closed_two_track()
        test_none_acceptance_and_br_visible()
        test_empty_input_zero_yields()
        test_sum_weights_and_width_note()
        print("\nAll DecayInVolume tests passed.")
    finally:
        shutil.rmtree(base_directory, ignore_errors=True)


if __name__ == "__main__":
    run_all()


def test_partial_widths_sum_to_total_and_derive_br():
    """Per-channel widths reproduce the scalar path exactly, and derive br_visible.

    The scalar interface asks the caller to supply a TOTAL width and, separately,
    a branching ratio consistent with it -- two numbers that must agree by hand
    and that the tool cannot check. Supplying the channels instead makes the sum
    and the ratio the tool's arithmetic, so they cannot drift apart.
    """
    print(">> partial widths: sum == scalar total, br derived ...")
    _setup()
    wref = _width_ref(M_PHI, M_MU)
    # split the same total across a visible and an invisible channel
    vis, invis = 0.75 * wref, 0.25 * wref
    scalar = json.loads(_tool(width_ref_gev=wref, br_visible=0.75,
                              g_grid=[1e-8, 2e-8], acceptance="two_track")._run())
    perchan = json.loads(_tool(partial_widths_ref_gev={"mumu": vis, "dark": invis},
                               visible_channels=["mumu"],
                               g_grid=[1e-8, 2e-8], acceptance="two_track")._run())
    assert scalar["status"] == "ok" and perchan["status"] == "ok", (scalar, perchan)
    for a, b in zip(scalar["yields"], perchan["yields"]):
        assert abs(a["n_sig"] - b["n_sig"]) <= 1e-9 * max(1.0, abs(a["n_sig"])), (a, b)
    # the derived total drives the same lifetime as the scalar one
    assert abs(scalar["ctau_ref_g1_m"] - perchan["ctau_ref_g1_m"]) < 1e-12
    print("[OK] per-channel and scalar paths agree exactly")


def test_partial_widths_reject_inconsistent_input():
    print(">> partial widths: input validation ...")
    _setup()
    # channels given but no visible_channels -> cannot derive br
    r = _tool(partial_widths_ref_gev={"mumu": 1e-4}, g_grid=[1e-8])._run()
    assert "visible_channels" in r, r
    # visible channel not among the declared widths
    r = _tool(partial_widths_ref_gev={"mumu": 1e-4},
              visible_channels=["gammagamma"], g_grid=[1e-8])._run()
    assert "not keys of" in r, r
    # all channels closed -> no signal is possible, say so rather than divide by 0
    r = _tool(partial_widths_ref_gev={"mumu": 0.0},
              visible_channels=["mumu"], g_grid=[1e-8])._run()
    assert "sums to zero" in r, r
    print("[OK] missing/unknown/zero channel inputs rejected")


def test_offscale_lifetime_is_reported():
    """A scan whose lifetimes never reach the decay volume is flagged.

    N_sig alone looks like an ordinary falling curve in that regime, so the
    caller cannot tell a genuinely insensitive experiment from a wrong width or
    a mis-placed grid. The tool knows the geometry, so it can say.
    """
    print(">> off-scale lifetime diagnostic ...")
    _setup()
    wref = _width_ref(M_PHI, M_MU)
    # enormous couplings -> ctau ~ 1/g^2 far SHORT of the ~160 m volume
    short = json.loads(_tool(width_ref_gev=wref, g_grid=[1.0, 2.0],
                             acceptance="two_track")._run())
    assert short["lifetime_offscale"], short
    assert "SHORT" in short["lifetime_offscale"]
    # a sane grid straddling the volume is not flagged
    ok = json.loads(_tool(width_ref_gev=wref, g_grid=[1e-8, 1e-7],
                          acceptance="two_track")._run())
    assert not ok["lifetime_offscale"], ok["lifetime_offscale"]
    print("[OK] off-scale flagged, on-scale silent")


def test_multi_input_equals_manual_concatenation():
    """Several per-channel files == one hand-concatenated file, and the
    per-file record counts are echoed so a dropped channel is visible.

    The alternative the agent asked for -- a tool that APPENDS channels into a
    combined file -- would double-count on any re-run, silently inflating every
    downstream yield. Reading N files is idempotent, so it cannot.
    """
    print(">> multi-file input == manual concatenation ...")
    _setup()
    base = Path(base_directory)
    src = (base / "flux.jsonl").read_text().splitlines()
    half = len(src) // 2
    (base / "flux_a.jsonl").write_text("\n".join(src[:half]) + "\n")
    (base / "flux_b.jsonl").write_text("\n".join(src[half:]) + "\n")
    wref = _width_ref(M_PHI, M_MU)
    one = json.loads(_tool(width_ref_gev=wref, g_grid=[1e-8, 1e-7],
                           acceptance="two_track")._run())
    many = json.loads(_tool(events_path="", events_paths=["flux_a.jsonl", "flux_b.jsonl"],
                            width_ref_gev=wref, g_grid=[1e-8, 1e-7],
                            acceptance="two_track")._run())
    assert one["status"] == "ok" and many["status"] == "ok", (one, many)
    assert one["n_events"] == many["n_events"]
    for a, b in zip(one["yields"], many["yields"]):
        assert abs(a["n_sig"] - b["n_sig"]) <= 1e-9 * max(1.0, abs(a["n_sig"])), (a, b)
    assert [i["n_records"] for i in many["inputs"]] == [half, len(src) - half]
    # a missing member is a hard error, never a silent partial sum
    r = _tool(events_path="", events_paths=["flux_a.jsonl", "nope.jsonl"],
              width_ref_gev=wref, g_grid=[1e-8])._run()
    assert "not found" in r.lower(), r
    print("[OK] concatenation-free multi-input, counts echoed, missing file fatal")


def test_normalization_breakdown_is_reported():
    """The factors multiplying into N_sig are echoed so a discrepancy is
    locatable -- previously only their product was observable."""
    print(">> normalization breakdown ...")
    _setup()
    res = json.loads(_tool(width_ref_gev=_width_ref(M_PHI, M_MU),
                           g_grid=[1e-8], br_visible=0.5,
                           acceptance="two_track")._run())
    nz = res["normalization"]
    for key in ("n_int", "br_visible_effective", "width_ref_gev_effective",
                "sum_weights_per_collision", "mean_weight_per_event",
                "geometry_efficiency", "acceptance_efficiency",
                "overall_efficiency"):
        assert key in nz, key
    assert nz["br_visible_effective"] == 0.5
    assert 0.0 <= nz["geometry_efficiency"] <= 1.0
    assert 0.0 <= nz["acceptance_efficiency"] <= 1.0
    # the product identity the note describes must actually hold
    assert abs(nz["overall_efficiency"]
               - nz["geometry_efficiency"] * nz["acceptance_efficiency"]) < 1e-9
    print("[OK] normalization factors reported and self-consistent")


def test_g_grid_path_matches_an_inline_g_grid():
    """A dense reach grid can come from a file, not a hand-copied array."""
    print(">> g_grid_path: file-supplied couplings == inline ...")
    _setup()
    wref = _width_ref(M_PHI, M_MU)
    grid = [1e-9, 2e-9, 5e-9]
    inline = json.loads(_tool(width_ref_gev=wref, g_grid=grid,
                              acceptance="two_track")._run())
    Path(base_directory, "gs.json").write_text(json.dumps(grid))
    viafile = json.loads(_tool(width_ref_gev=wref, g_grid_path="gs.json",
                               acceptance="two_track")._run())
    assert viafile["status"] == "ok", viafile
    assert [y["n_sig"] for y in viafile["yields"]] == \
           [y["n_sig"] for y in inline["yields"]]
    # the wrapped form too
    Path(base_directory, "gs2.json").write_text(json.dumps({"g_grid": grid}))
    wrapped = json.loads(_tool(width_ref_gev=wref, g_grid_path="gs2.json",
                               acceptance="two_track")._run())
    assert len(wrapped["yields"]) == 3
    print("[OK] g_grid_path reproduces the inline grid exactly")


def test_events_paths_glob_collects_the_channel_files():
    """One glob replaces a hand-assembled per-parent list.

    Assembling that list by hand is where a channel gets dropped (silently
    lowering every yield) or repeated (silently inflating it), so the glob is
    sorted and an empty match is refused rather than run as an empty sample.
    """
    print(">> events_paths_glob: sorted match, empty refused ...")
    _setup()
    base = Path(base_directory)
    ch = base / "chan"
    ch.mkdir(exist_ok=True)
    for name in ("llp_K_m0.2500.jsonl", "llp_D_m0.2500.jsonl"):
        _write_llp(ch / name)
    wref = _width_ref(M_PHI, M_MU)

    explicit = json.loads(_tool(
        events_path="", events_paths=["chan/llp_D_m0.2500.jsonl",
                                      "chan/llp_K_m0.2500.jsonl"],
        width_ref_gev=wref, g_grid=[1e-9], acceptance="two_track")._run())
    globbed = json.loads(_tool(
        events_path="", events_paths_glob="chan/llp_*_m0.2500.jsonl",
        width_ref_gev=wref, g_grid=[1e-9], acceptance="two_track")._run())
    assert globbed["status"] == "ok", globbed
    assert globbed["n_events"] == explicit["n_events"]
    assert globbed["yields"][0]["n_sig"] == explicit["yields"][0]["n_sig"]
    assert len(globbed.get("inputs", [])) == 2, globbed.get("inputs")

    empty = _tool(events_path="", events_paths_glob="chan/nothing_*.jsonl",
                  width_ref_gev=wref, g_grid=[1e-9])._run()
    assert empty.lstrip().startswith("Error:") and "matched no files" in empty
    print("[OK] glob == explicit list; empty match refused")
