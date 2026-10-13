"""
# test_llp_physics.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Unit tests for the LLP physics primitives, above all the UNBIASEDNESS of the
decay-vertex sampler under a hard downstream cut.

This is the regression test for the 2.4.0 bug. `sample_decay_vertices` used the
fixed stratum midpoint u_k = (k + 1/2)/K, which is a midpoint QUADRATURE rule,
not stratified sampling. A quadrature rule carries O(1) error whenever a
discontinuity falls inside a panel, and the downstream geometry supplies
exactly one: an LLP counts only if its parent decayed upstream of the absorber,
a window covering the first ~2% of the exponential for a forward kaon. With
K = 6 the estimator was low by FIVE ORDERS OF MAGNITUDE near the strong-coupling
edge, and -- because the nodes were deterministic -- re-running with a different
seed reproduced the same wrong answer exactly. Seed variation, the usual check,
is blind to it.

The tests below compare against the CLOSED FORM, which is the only check that
would have caught it.
"""
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent                    # .../heptapod/tools/llp
REPO_ROOT = TOOL_DIR.parent.parent                      # .../heptapod
sys.path.insert(0, str(REPO_ROOT))

from tools.llp import llp_physics as phys               # noqa: E402

M_PAR = 0.493677            # kaon mass [GeV]
CTAU = 3.711                # kaon c*tau [m]


def _forward_parents(n, p_gev=100.0):
    """n identical parents along +z, so lambda is a single known number."""
    pvec = np.zeros((n, 3))
    pvec[:, 2] = p_gev
    lam = (p_gev / M_PAR) * CTAU
    return pvec, lam


def test_sampler_is_unbiased_under_a_hard_cut():
    """The fraction decaying before z_cut must match 1 - exp(-z_cut/lambda).

    z_cut is chosen so the accepted window is ~2% of the exponential -- the
    regime the forward geometry actually imposes -- and K is small, as the
    campaign agents chose. The midpoint scheme returns exactly 0 here.
    """
    print(">> decay-vertex sampler is unbiased under a hard cut ...")
    n, K = 40000, 6
    pvec, lam = _forward_parents(n)
    z_cut = 0.02 * lam                                  # ~2% acceptance window
    exact = -math.expm1(-z_cut / lam)

    rng = np.random.default_rng(12345)
    rep, vertex = phys.sample_decay_vertices(pvec, CTAU, M_PAR, K, rng)
    assert vertex.shape == (n * K, 3)
    assert rep.shape == (n * K,)

    frac = float((vertex[:, 2] < z_cut).mean())         # each replica weighs 1/K

    # binomial error: only stratum 0 can contribute, with prob exact*K
    p0 = exact * K
    sd = math.sqrt(n * p0 * (1 - p0)) / (n * K)
    assert abs(frac - exact) < 5 * sd, (
        f"estimator biased: got {frac:.6e}, exact {exact:.6e}, "
        f"5 sd = {5*sd:.2e}")

    # and the midpoint rule -- what 2.4.0 shipped -- is catastrophically wrong
    u_mid = (np.arange(K) + 0.5) / K
    ell_mid = -lam * np.log1p(-u_mid)
    assert (ell_mid >= z_cut).all(), "fixture no longer exercises the bug"
    print(f"[OK] exact {exact:.5e}, sampled {frac:.5e} "
          f"({abs(frac-exact)/sd:.2f} sd); midpoint would give 0.0")


def test_sampler_converges_over_seeds():
    """Averaging over seeds must converge to the exact value.

    The old scheme could not: its error was a bias, identical for every seed.
    """
    print(">> sampler averages to the closed form over seeds ...")
    n, K = 8000, 4
    pvec, lam = _forward_parents(n)
    z_cut = 0.02 * lam
    exact = -math.expm1(-z_cut / lam)
    vals = []
    for s in range(12):
        _, vertex = phys.sample_decay_vertices(
            pvec, CTAU, M_PAR, K, np.random.default_rng(s))
        vals.append(float((vertex[:, 2] < z_cut).mean()))
    mean = sum(vals) / len(vals)
    assert abs(mean - exact) / exact < 0.05, (
        f"seed-averaged estimator off: {mean:.6e} vs exact {exact:.6e}")
    assert len(set(vals)) == len(vals), "seeds must give different draws"
    print(f"[OK] mean over 12 seeds {mean:.5e} vs exact {exact:.5e}")


def test_sampler_respects_geometry_and_weighting():
    """Vertices lie along the parent direction; rep indexes the source parent."""
    print(">> vertices lie along the parent direction ...")
    rng = np.random.default_rng(7)
    pvec = np.array([[0.0, 0.0, 50.0], [3.0, 4.0, 100.0]])
    rep, vertex = phys.sample_decay_vertices(pvec, CTAU, M_PAR, 3, rng)
    assert list(rep) == [0, 0, 0, 1, 1, 1]
    for i in range(len(vertex)):
        d = pvec[rep[i]] / np.linalg.norm(pvec[rep[i]])
        ell = np.linalg.norm(vertex[i])
        assert np.allclose(vertex[i], ell * d, atol=1e-9), i
        assert ell > 0.0
    print("[OK] vertices collinear with their parent, rep correct")


def test_larger_K_reduces_variance():
    """Stratification must still buy variance reduction, not just unbiasedness."""
    print(">> more strata -> lower variance ...")
    n = 4000
    pvec, lam = _forward_parents(n)
    z_cut = 0.05 * lam

    def spread(K):
        v = []
        for s in range(16):
            _, vert = phys.sample_decay_vertices(
                pvec, CTAU, M_PAR, K, np.random.default_rng(1000 + s))
            v.append(float((vert[:, 2] < z_cut).mean()))
        return float(np.std(v))

    s_lo, s_hi = spread(2), spread(32)
    assert s_hi < s_lo, f"variance did not fall with K: {s_lo:.3e} -> {s_hi:.3e}"
    print(f"[OK] sd {s_lo:.3e} (K=2) -> {s_hi:.3e} (K=32)")


def test_z_shield_and_deprecated_alias():
    """`z_shield_m` is canonical; `z_prod_m` still loads for old geometries."""
    print(">> z_shield naming + deprecated alias ...")
    base = {"z_min_m": 112.0, "z_max_m": 163.0, "r_volume_m": 0.92,
            "z_det_m": 165.0, "r_det_m": 0.92}
    g_new = phys.DecayVolume({**base, "z_shield_m": 50.0})
    g_old = phys.DecayVolume({**base, "z_prod_m": 50.0})
    g_def = phys.DecayVolume(dict(base))
    assert g_new.z_shield == 50.0
    assert g_old.z_shield == 50.0, "legacy z_prod_m must still be honoured"
    assert g_def.z_shield == g_def.z_min, "default is the fiducial face"
    # the canonical key wins if somebody supplies both
    g_both = phys.DecayVolume({**base, "z_shield_m": 40.0, "z_prod_m": 50.0})
    assert g_both.z_shield == 40.0
    print("[OK] z_shield_m canonical, z_prod_m accepted, canonical wins")


if __name__ == "__main__":
    test_sampler_is_unbiased_under_a_hard_cut()
    test_sampler_converges_over_seeds()
    test_sampler_respects_geometry_and_weighting()
    test_larger_K_reduces_variance()
    test_z_shield_and_deprecated_alias()
    print("\nall llp_physics tests passed")
