"""
# test_production_spectrum.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Unit tests for ProductionSpectrumTool and the spectra package.

The physics checks compare against things that are true independently of this
implementation -- a closed-form normalisation, an independent quadrature, and
convergence in the grid parameters -- rather than against stored golden
numbers, which would only lock in whatever the code did on the day.
"""
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.llp.production_spectrum import ProductionSpectrumTool   # noqa: E402
from tools.llp.spectra import kinematics as kin                    # noqa: E402
from tools.llp.spectra import parents as sm                        # noqa: E402
from tools.llp.spectra import vertices as vtx                      # noqa: E402
from tools.llp.llp_physics import LLPSpectrum                      # noqa: E402

base_directory = str(TOOL_DIR / "test_files_spectrum")


def _setup():
    if os.path.isdir(base_directory):
        shutil.rmtree(base_directory)
    os.makedirs(base_directory, exist_ok=True)


def _tool(**kw):
    kw.setdefault("base_directory", base_directory)
    return ProductionSpectrumTool(**kw)


def _run(**kw):
    return json.loads(_tool(**kw)._run())


# ---------------------------------------------------------------- physics --
def test_spectrum_is_normalised():
    """f_h(x) must integrate to 1 -- the contract the sampler relies on."""
    print(">> f(x) is normalised ...")
    msq = vtx.get("pseudoscalar", "scalar")
    for h in sm.PSEUDOSCALARS:
        p = sm.get(h)
        m_phi = 0.4 * sm.kinematic_limit(h)
        _, xs, fx = kin.width_and_spectrum(p.mass_gev, sm.M_MU_GEV, m_phi, msq)
        # integrate on the same GL nodes the spectrum was built on
        _, wx = kin._gl_nodes_x(p.mass_gev, sm.M_MU_GEV, m_phi, kin.N_X_DEFAULT)
        integral = float(np.dot(fx, wx))
        assert abs(integral - 1.0) < 1e-10, (h, integral)
    print("[OK] all hosted pseudoscalar spectra integrate to 1")


def test_quadrature_defaults_are_converged():
    """The shipped n_x/n_c must agree with a much finer grid.

    This is the guard on the defaults: if someone lowers them, or changes the
    substitution, this fails.
    """
    print(">> default quadrature is converged ...")
    msq = vtx.get("pseudoscalar", "scalar")
    worst = 0.0
    for h in sm.PSEUDOSCALARS:
        p = sm.get(h)
        lim = sm.kinematic_limit(h)
        for frac in (0.02, 0.5, 0.99):
            m_phi = frac * lim
            g_def, _, _ = kin.width_and_spectrum(
                p.mass_gev, sm.M_MU_GEV, m_phi, msq)
            g_fine, _, _ = kin.width_and_spectrum(
                p.mass_gev, sm.M_MU_GEV, m_phi, msq, n_x=192, n_c=192)
            worst = max(worst, abs(g_def / g_fine - 1.0))
    assert worst < 1e-4, f"defaults not converged: worst {worst:.2e}"
    print(f"[OK] worst deviation from a 192x192 grid: {worst:.2e}")


def test_substitution_beats_plain_gauss_legendre():
    """The sqrt substitution must actually be doing work.

    Without it the phase-space edge at x_min caps convergence; this pins the
    reason the integrator is written the way it is.
    """
    print(">> the x = x_min + t^2 substitution earns its place ...")
    msq = vtx.get("pseudoscalar", "scalar")
    p = sm.get("K")
    m_phi = 0.1 * sm.kinematic_limit("K")
    ref, _, _ = kin.width_and_spectrum(p.mass_gev, sm.M_MU_GEV, m_phi, msq,
                                       n_x=256, n_c=128)

    def plain(n_x):
        x_lo, x_hi = kin.x_domain(p.mass_gev, sm.M_MU_GEV, m_phi)
        t, w = np.polynomial.legendre.leggauss(n_x)
        xs = 0.5 * (x_hi - x_lo) * (t + 1.0) + x_lo
        wx = 0.5 * (x_hi - x_lo) * w
        cs, wc = np.polynomial.legendre.leggauss(64)
        P, pl, pn, pp, ok = kin.momenta(p.mass_gev, sm.M_MU_GEV, m_phi,
                                        xs[:, None], cs[None, :])
        val = np.where(ok, msq(P, pl, pn, pp, sm.M_MU_GEV, m_phi), 0.0)
        pref = kin.jacobian(p.mass_gev, sm.M_MU_GEV, m_phi, xs) / (
            (2 * np.pi) ** 3 * 32.0 * p.mass_gev ** 3)
        return float(np.dot(pref * (val @ wc), wx))

    n = kin.N_X_DEFAULT
    with_sub, _, _ = kin.width_and_spectrum(p.mass_gev, sm.M_MU_GEV, m_phi,
                                            msq, n_x=n, n_c=64)
    e_sub = abs(with_sub / ref - 1.0)
    e_plain = abs(plain(n) / ref - 1.0)
    assert e_sub < e_plain, (e_sub, e_plain)
    print(f"[OK] at n_x={n}: substituted {e_sub:.2e} vs plain {e_plain:.2e} "
          f"({e_plain/max(e_sub,1e-15):.0f}x better)")


def test_closed_channel_is_reported_not_an_error():
    """A mass grid must be allowed to span a threshold."""
    print(">> closed channels report, they do not fail ...")
    _setup()
    lim = sm.kinematic_limit("K")
    r = _run(parent="K", masses=[0.5 * lim, 1.5 * lim], output_dir="spec")
    assert r["status"] == "ok", r
    assert r["n_open"] == 1 and r["n_closed"] == 1, r
    closed = [m for m in r["masses"] if not m["open"]][0]
    assert "closed" in closed["reason"]
    print(f"[OK] limit {lim:.5f} GeV; open 1, closed 1, status ok")


# ------------------------------------------------------------- the chain --
def test_csv_feeds_the_downstream_sampler_unmodified():
    """The emitted CSV must load straight into LLPSpectrum.

    This is the seam between the two tools; if it needs conversion, the split
    has cost the agent something.
    """
    print(">> emitted CSV loads directly as a downstream spectrum ...")
    _setup()
    r = _run(parent="K", m_phi_gev=0.06, output_path="spec/K_m0.060.csv")
    assert r["status"] == "ok", r
    rec = r["masses"][0]
    path = os.path.join(base_directory, rec["spectrum_path"])
    spec = LLPSpectrum.from_path(path) if hasattr(LLPSpectrum, "from_path") \
        else LLPSpectrum.from_csv(path)
    rng = np.random.default_rng(3)
    x = spec.sample_x(sm.get("K").mass_gev, 0.06, 20000, rng)
    assert np.all(x >= rec["x_min"] - 1e-9)
    assert np.all(x <= rec["x_max"] + 1e-9)
    # the sampled mean must match the tabulated mean
    grid, pdf = spec.x, spec.pdf
    mean_tab = float(np.trapezoid(grid * pdf, grid) / np.trapezoid(pdf, grid))
    assert abs(x.mean() / mean_tab - 1.0) < 0.02, (x.mean(), mean_tab)
    print(f"[OK] sampled <x>={x.mean():.4f} vs tabulated {mean_tab:.4f}")


# --------------------------------------------------------------- the API --
def test_grid_mode_writes_manifest_and_creates_dirs():
    print(">> grid mode writes files + manifest, creating directories ...")
    _setup()
    r = _run(parent="D", masses=[0.2, 0.8, 1.4], output_dir="deep/nested/spec")
    assert r["status"] == "ok", r
    d = os.path.join(base_directory, "deep/nested/spec")
    assert os.path.isdir(d), "output_dir must be created"
    man = os.path.join(base_directory, r["manifest_path"])
    assert os.path.isfile(man)
    payload = json.load(open(man))
    assert payload["parent"] == "D" and "provenance" in payload
    for rec in r["masses"]:
        if rec["open"]:
            assert os.path.isfile(os.path.join(base_directory,
                                               rec["spectrum_path"]))
    print(f"[OK] {r['n_open']} spectra + manifest under {d}")


def test_provenance_cites_every_constant():
    print(">> every constant carries a source ...")
    _setup()
    r = _run(parent="K", m_phi_gev=0.06, output_path="s.csv")
    prov = r["provenance"]
    assert prov["constants"], prov
    for name, (value, source) in prov["constants"].items():
        assert source, f"{name} has no source"
        assert isinstance(value, float), name
    assert "Carlson" in prov["amplitude_reference"]
    assert "arXiv:2501.09071" in prov["application_reference"]
    assert prov["modelling_assumptions"]
    print(f"[OK] {len(prov['constants'])} constants cited, amplitude + "
          f"application references present")


def test_errors_are_actionable():
    print(">> bad input gives an actionable error, not a traceback ...")
    _setup()
    bad = _tool(parent="nosuch", m_phi_gev=0.1, output_path="s.csv")._run()
    assert "hosted parents are" in bad, bad
    bad = _tool(parent="K", m_phi_gev=0.1, interaction="tensor",
                output_path="s.csv")._run()
    assert "vertices.py" in bad, bad
    bad = _tool(parent="K", m_phi_gev=0.1,
                output_path="../../../tmp/evil.csv")._run()
    assert "denied" in bad.lower(), bad
    bad = _tool(parent="K", m_phi_gev=-1.0, output_path="s.csv")._run()
    assert "m_phi_gev" in bad, bad
    print("[OK] unknown parent, unsupported vertex, traversal, bad mass")


def test_kinematic_limits_match_the_physics():
    print(">> kinematic limits ...")
    for h in sm.PSEUDOSCALARS:
        p = sm.get(h)
        assert abs(sm.kinematic_limit(h)
                   - (p.mass_gev - sm.M_MU_GEV)) < 1e-12
    for h in sm.VECTORS:
        p = sm.get(h)
        assert abs(sm.kinematic_limit(h)
                   - (p.mass_gev - 2 * sm.M_MU_GEV)) < 1e-12
    assert sm.is_open("K", 0.3) and not sm.is_open("K", 0.4)
    # electron channels open much further than muon ones
    assert sm.kinematic_limit("K", "e") > sm.kinematic_limit("K", "mu")
    print("[OK] pseudoscalar m_h - m_l, vector m_h - 2 m_l, lepton-dependent")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items())
               if k.startswith("test_")]:
        fn()
    print("\nall production-spectrum tests passed")


# --------------------------------------------------- the two-tool seam --
def _write_parent_flux(path, n=400, w=1.0e-6):
    """A minimal forward kaon flux, in harvest_forward_flux's output format."""
    import random
    rng = random.Random(11)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for _ in range(n):
            p = rng.uniform(20.0, 400.0)
            th = rng.uniform(0.0, 2.0e-3)
            m = sm.get("K").mass_gev
            pz = p * math.cos(th)
            px = p * math.sin(th)
            fh.write(json.dumps({
                "parent": "K", "pdg": 321,
                "E": math.sqrt(p * p + m * m), "px": px, "py": 0.0, "pz": pz,
                "weight_per_collision": w}) + "\n")


def test_B_hat_flows_from_the_spectrum_into_the_sampler():
    """The whole point of the split: one path, no hand-carried number."""
    print(">> B_hat flows automatically into MesonDecayToLLPTool ...")
    from tools.llp.meson_decay_to_llp import MesonDecayToLLPTool
    _setup()
    _write_parent_flux(os.path.join(base_directory, "parents_K.jsonl"))
    spec = _run(parent="K", m_phi_gev=0.06, output_path="spec/K.csv")
    b_hat = spec["masses"][0]["B_hat"]

    # NOTE: no B_hat passed here at all
    dec = json.loads(MesonDecayToLLPTool(
        base_directory=base_directory, parent_flux_path="parents_K.jsonl",
        spectrum_spec="spec/K.csv", m_phi_gev=0.06,
        parent_mass_gev=sm.get("K").mass_gev, n_strata=4, seed=1,
        output_path="llp/K.jsonl")._run())
    assert dec["status"] == "ok", dec

    recs = [json.loads(l) for l in
            open(os.path.join(base_directory, "llp/K.jsonl")) if l.strip()]
    assert recs, "no LLP records written"
    # w = parent_weight * B_hat / n_strata -- so B_hat is recoverable
    got = recs[0]["event_weight_g2_stripped"] * 4 / 1.0e-6
    assert abs(got / b_hat - 1.0) < 1e-9, (got, b_hat)
    print(f"[OK] B_hat={b_hat:.6e} read from the spectrum and applied "
          f"({len(recs)} records)")


def test_mismatched_pairing_is_refused():
    """A spectrum for one mass must not be usable at another, and a
    hand-supplied B_hat must not silently override the file's own."""
    print(">> mismatched spectrum/normalisation pairings are refused ...")
    from tools.llp.meson_decay_to_llp import MesonDecayToLLPTool
    _setup()
    _write_parent_flux(os.path.join(base_directory, "parents_K.jsonl"))
    spec = _run(parent="K", m_phi_gev=0.06, output_path="spec/K.csv")
    b_hat = spec["masses"][0]["B_hat"]

    def call(**kw):
        base = dict(base_directory=base_directory,
                    parent_flux_path="parents_K.jsonl",
                    spectrum_spec="spec/K.csv",
                    parent_mass_gev=sm.get("K").mass_gev, n_strata=2, seed=1,
                    output_path="llp/x.jsonl")
        base.update(kw)
        return MesonDecayToLLPTool(**base)._run()

    wrong_mass = call(m_phi_gev=0.12)
    assert "Mismatch" in wrong_mass, wrong_mass
    wrong_norm = call(m_phi_gev=0.06, B_hat=b_hat * 2.0)
    assert "Mismatch" in wrong_norm, wrong_norm
    ok = json.loads(call(m_phi_gev=0.06, B_hat=b_hat))
    assert ok["status"] == "ok", ok
    print("[OK] wrong mass refused, wrong B_hat refused, matching pair ok")


# ── vector parents: V -> l+ l- phi ────────────────────────────────────────
#
# These channels carry the ENTIRE reach above the pseudoscalar kinematic wall
# (~1.76 GeV), where the charmonia are the only open production mode. They are
# also the easiest place to be silently wrong: two diagrams interfere and the
# parent's polarisations must be averaged, so an algebra slip yields a
# plausible spectrum with the wrong normalisation. Hence three independent
# checks -- gauge invariance, a closed-form limit, and quadrature convergence.

import csv

import numpy as np

from tools.llp.spectra import kinematics as kin
from tools.llp.spectra import parents as sm
from tools.llp.spectra import vertices as vtx

M_MU = sm.LEPTON_MASS_GEV["mu"]


def test_dirac_algebra_is_consistent():
    """The trace machinery is only as good as its gamma matrices."""
    print(">> Clifford algebra and slash^2 = p^2 ...")
    anti = (np.einsum('mij,njk->mnik', vtx._GAMMA, vtx._GAMMA)
            + np.einsum('nij,mjk->mnik', vtx._GAMMA, vtx._GAMMA))
    target = 2 * np.diag(vtx._METRIC)[:, :, None, None] * np.eye(4)
    assert np.max(np.abs(anti - target)) < 1e-12
    rng = np.random.default_rng(0)
    p = rng.normal(size=(6, 4))
    s = vtx.slash(p)
    ss = np.einsum('...ij,...jk->...ik', s, s)
    assert np.max(np.abs(ss - kin.mdot(p, p)[:, None, None] * np.eye(4))) < 1e-12
    print("[OK] {gamma^mu, gamma^nu} = 2 g^{mu nu} and p_slash^2 = p^2")


def test_vector_amplitude_satisfies_the_ward_identity():
    """P_mu T^{mu nu} = 0: the check a wrong relative sign cannot survive.

    Both diagrams are individually non-conserving; only their sum satisfies
    current conservation. A dropped diagram, or a flipped sign between them,
    still produces a smooth positive spectrum -- but fails here.
    """
    print(">> Ward identity on a physical Dalitz grid ...")
    for parent in ("Jpsi", "phi", "rho"):
        p = sm.get(parent)
        m_phi = 0.3 * (p.mass_gev - 2 * M_MU)
        xs, _ = kin._gl_nodes_x(p.mass_gev, M_MU, m_phi, 12, M_MU)
        cs = np.linspace(-0.9, 0.9, 7)
        P, pl, pn, pk, ok = kin.momenta(p.mass_gev, M_MU, m_phi,
                                        xs[:, None], cs[None, :], M_MU)
        w = vtx.ward_residual(P, pl, pn, pk, M_MU, m_phi)
        assert np.max(w[ok]) < 1e-10, (parent, float(np.max(w[ok])))
        msq = vtx.msq_scalar_vector_parent(P, pl, pn, pk, M_MU, m_phi)
        assert np.all(msq[ok] > 0.0), parent
    print("[OK] |P.T|/|T| < 1e-10 and |M|^2 > 0 for rho, phi, Jpsi")


def test_unit_two_body_width_matches_the_closed_form():
    """Fixes the polarisation average and the coupling normalisation.

    g_V is set by dividing the MEASURED V -> l+ l- width by this quantity, so
    an error here would rescale every vector spectrum by a constant -- exactly
    the kind of mistake that survives every shape check.
    """
    print(">> Gamma(V -> l+ l-) at unit coupling vs the closed form ...")
    for name in ("rho", "omega", "phi", "Jpsi", "psi2S"):
        M = sm.get(name).mass_gev
        r = 4 * M_MU * M_MU / (M * M)
        closed = M / (12 * np.pi) * (1 + r / 2) * np.sqrt(max(1 - r, 0.0))
        got = vtx.gamma_v_to_ll_unit(M, M_MU)
        assert abs(got / closed - 1) < 1e-12, (name, got, closed)
    assert vtx.gamma_v_to_ll_unit(0.1, M_MU) == 0.0     # closed channel
    print("[OK] matches M/(12 pi)(1+2m^2/M^2)sqrt(1-4m^2/M^2) to 1e-12")


def test_vector_spectrum_is_normalised_and_converged():
    """f(x) integrates to 1, and the default quadrature is already converged."""
    print(">> vector spectra: normalisation and n_x convergence ...")
    _setup()
    res = json.loads(_tool(parent="Jpsi", masses=[0.06, 0.7, 2.3],
                           output_dir="vec").  _run())
    assert res["status"] == "ok", res
    opened = [m for m in res["masses"] if m.get("open")]
    assert len(opened) == 3, res["masses"]
    for m in opened:
        assert m["B_hat"] > 0.0
        xs, pdf = [], []
        with open(os.path.join(base_directory, m["spectrum_path"])) as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#"):
                    continue
                try:
                    xs.append(float(row[0])); pdf.append(float(row[1]))
                except ValueError:
                    continue
        area = np.trapezoid(np.array(pdf), np.array(xs))
        assert abs(area - 1.0) < 5e-3, (m["m_phi_gev"], area)

    # the default n_x is converged: refining must not move B_hat
    ref = None
    for nx in (kin.N_X_DEFAULT, 4 * kin.N_X_DEFAULT):
        r = json.loads(_tool(parent="Jpsi", m_phi_gev=0.06, n_x=nx,
                             output_path=f"vconv_{nx}.csv")._run())
        rec = r["masses"][0] if "masses" in r else r
        if ref is None:
            ref = rec["B_hat"]
        else:
            assert abs(rec["B_hat"] / ref - 1) < 1e-4, (ref, rec["B_hat"])
    print("[OK] f(x) normalised; B_hat stable under 4x refinement")


def test_vector_parents_are_no_longer_rejected():
    """The gap a 2026-08 trial had to work around by hand."""
    print(">> every hosted vector parent produces a spectrum ...")
    _setup()
    for name in sm.VECTORS:
        limit = sm.kinematic_limit(name, "mu")
        r = json.loads(_tool(parent=name, m_phi_gev=0.5 * limit,
                             output_path=f"v_{name}.csv")._run())
        assert r.get("status") == "ok", (name, str(r)[:200])
        rec = r["masses"][0] if "masses" in r else r
        assert rec["open"] and rec["B_hat"] > 0.0, (name, rec)
    print(f"[OK] {list(sm.VECTORS)} all supported")
