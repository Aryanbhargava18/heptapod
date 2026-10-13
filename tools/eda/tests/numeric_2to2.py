"""
# numeric_2to2.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Independent numeric 2 -> 2 cross sections, for validating the FeynCalc
scattering codegen.

WHY IT EXISTS.  ``scattering.py`` claims to cover the whole 2->2 space.
That claim is only worth something if the generated amplitudes are checked
against a computation that shares NO ALGEBRA with them: no FeynCalc, no
trace theorems, no Kallen-function phase-space shortcuts.  This module
builds amplitudes from explicit 4x4 Dirac matrices, explicit spinors and
explicit polarisation vectors, contracts Lorentz indices by summing them,
and integrates the angular distribution numerically.  The only thing it
shares with the generator is the physics CONVENTION -- which is precisely
what the comparison is meant to test agreement on.

CONVENTIONS.  Metric (+,-,-,-).  Dirac basis for the gamma matrices.
u-bar u = 2m, v-bar v = -2m, sum_s u u-bar = p-slash + m,
sum_s v v-bar = p-slash - m.  Massive vectors carry three polarisations
with sum_lam eps eps* = -g + p p / m^2; massless vectors carry two
transverse ones.  p1, p2 incoming along +-z; p3, p4 outgoing at angle
theta in the centre-of-mass frame.

The self-tests in :func:`check_conventions` assert every one of those
identities numerically, so a convention slip surfaces here rather than as
a mysterious factor downstream.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Dirac algebra
# ---------------------------------------------------------------------------

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])

_SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]
_I2 = np.eye(2, dtype=complex)
_Z2 = np.zeros((2, 2), dtype=complex)


def _block(a, b, c, d):
    return np.block([[a, b], [c, d]])


GAMMA = [
    _block(_I2, _Z2, _Z2, -_I2),                       # gamma^0
    *[_block(_Z2, s, -s, _Z2) for s in _SIGMA],        # gamma^i
]
GAMMA5 = _block(_Z2, _I2, _I2, _Z2)
P_LEFT = 0.5 * (np.eye(4, dtype=complex) - GAMMA5)     # (1 - g5)/2
P_RIGHT = 0.5 * (np.eye(4, dtype=complex) + GAMMA5)


def dot(a: np.ndarray, b: np.ndarray) -> complex:
    """Minkowski product with metric (+,-,-,-)."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def slash(p: np.ndarray) -> np.ndarray:
    """p-slash = p_mu gamma^mu."""
    return p[0] * GAMMA[0] - p[1] * GAMMA[1] - p[2] * GAMMA[2] - p[3] * GAMMA[3]


def bar(spinor: np.ndarray) -> np.ndarray:
    return spinor.conj() @ GAMMA[0]


# ---------------------------------------------------------------------------
# External wavefunctions
# ---------------------------------------------------------------------------

def spinor_u(p: np.ndarray, m: float, s: int) -> np.ndarray:
    """Positive-energy spinor, normalised to u-bar u = 2m."""
    chi = np.array([1, 0], dtype=complex) if s == 0 else np.array([0, 1], dtype=complex)
    E = p[0]
    norm = math.sqrt(max(E + m, 1e-300))
    sp = sum(_SIGMA[i] * p[i + 1] for i in range(3))
    lower = (sp @ chi) / (E + m) if (E + m) != 0 else np.zeros(2, dtype=complex)
    return norm * np.concatenate([chi, lower])


def spinor_v(p: np.ndarray, m: float, s: int) -> np.ndarray:
    """Negative-energy spinor, normalised to v-bar v = -2m."""
    eta = np.array([0, 1], dtype=complex) if s == 0 else np.array([-1, 0], dtype=complex)
    E = p[0]
    norm = math.sqrt(max(E + m, 1e-300))
    sp = sum(_SIGMA[i] * p[i + 1] for i in range(3))
    upper = (sp @ eta) / (E + m) if (E + m) != 0 else np.zeros(2, dtype=complex)
    return norm * np.concatenate([upper, eta])


def pol_vectors(p: np.ndarray, m: float) -> List[np.ndarray]:
    """Polarisation basis: 3 states if massive, 2 transverse if massless."""
    px, py, pz = p[1], p[2], p[3]
    pmag = math.sqrt(px * px + py * py + pz * pz)
    if pmag < 1e-12:
        e1 = np.array([0, 1, 0, 0], dtype=complex)
        e2 = np.array([0, 0, 1, 0], dtype=complex)
        e3 = np.array([0, 0, 0, 1], dtype=complex)
        return [e1, e2, e3] if m > 0 else [e1, e2]
    n = np.array([px, py, pz]) / pmag
    # any vector not parallel to n
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    eps1 = np.array([0.0, *t1], dtype=complex)
    eps2 = np.array([0.0, *t2], dtype=complex)
    if m <= 0:
        return [eps1, eps2]
    E = p[0]
    eps_L = np.array([pmag / m, *(E / m * n)], dtype=complex)
    return [eps1, eps2, eps_L]


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def kallen(a: float, b: float, c: float) -> float:
    return a * a + b * b + c * c - 2 * a * b - 2 * a * c - 2 * b * c


def cm_momenta(s: float, masses: Sequence[float], cos_theta: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """p1, p2 along +-z; p3, p4 at polar angle theta, phi = 0."""
    m1, m2, m3, m4 = masses
    rs = math.sqrt(s)
    pi = math.sqrt(max(kallen(s, m1 * m1, m2 * m2), 0.0)) / (2 * rs)
    pf = math.sqrt(max(kallen(s, m3 * m3, m4 * m4), 0.0)) / (2 * rs)
    E1 = math.sqrt(pi * pi + m1 * m1)
    E2 = math.sqrt(pi * pi + m2 * m2)
    E3 = math.sqrt(pf * pf + m3 * m3)
    E4 = math.sqrt(pf * pf + m4 * m4)
    st = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    p1 = np.array([E1, 0.0, 0.0, pi])
    p2 = np.array([E2, 0.0, 0.0, -pi])
    p3 = np.array([E3, pf * st, 0.0, pf * cos_theta])
    p4 = np.array([E4, -pf * st, 0.0, -pf * cos_theta])
    return p1, p2, p3, p4


# ---------------------------------------------------------------------------
# Cross section
# ---------------------------------------------------------------------------

def cross_section(m2_avg: Callable[[float], float], s: float,
                  masses: Sequence[float], symmetry_factor: float = 1.0,
                  n_points: int = 400) -> float:
    """sigma = (1/S) Int dOmega |M|^2_avg p_f / (64 pi^2 s p_i).

    Gauss-Legendre in cos(theta); phi is trivial for the processes here, so
    the azimuthal integral contributes 2 pi.
    """
    m1, m2, m3, m4 = masses
    rs = math.sqrt(s)
    pi_ = math.sqrt(max(kallen(s, m1 * m1, m2 * m2), 0.0)) / (2 * rs)
    pf = math.sqrt(max(kallen(s, m3 * m3, m4 * m4), 0.0)) / (2 * rs)
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    total = 0.0
    for x, w in zip(nodes, weights):
        total += w * m2_avg(float(x))
    # dsigma/dOmega = |M|^2 pf / (64 pi^2 s pi); Int dphi = 2 pi
    return symmetry_factor and (
        total * 2 * math.pi * pf / (64 * math.pi ** 2 * s * pi_) / symmetry_factor
    )


# ---------------------------------------------------------------------------
# Convention self-tests
# ---------------------------------------------------------------------------

def check_conventions(tol: float = 1e-9) -> List[str]:
    """Assert the spinor / polarisation identities this module relies on.

    Returns a list of failure messages; empty means everything holds.
    """
    fails: List[str] = []
    rng = np.random.default_rng(20260829)
    for m in (0.0, 1.7, 42.0):
        p3v = rng.normal(size=3) * 7.0
        E = math.sqrt(float(p3v @ p3v) + m * m)
        p = np.array([E, *p3v])

        if abs(dot(p, p) - m * m) > tol * max(1.0, m * m):
            fails.append(f"on-shell p^2 != m^2 for m={m}")

        # Dirac equation and normalisation
        for s in (0, 1):
            u = spinor_u(p, m, s)
            v = spinor_v(p, m, s)
            if np.max(np.abs((slash(p) - m * np.eye(4)) @ u)) > 1e-7 * max(1.0, E):
                fails.append(f"(p-slash - m) u != 0 for m={m}, s={s}")
            if np.max(np.abs((slash(p) + m * np.eye(4)) @ v)) > 1e-7 * max(1.0, E):
                fails.append(f"(p-slash + m) v != 0 for m={m}, s={s}")
            if abs(bar(u) @ u - 2 * m) > 1e-7 * max(1.0, E):
                fails.append(f"u-bar u != 2m for m={m}, s={s}")
            if abs(bar(v) @ v + 2 * m) > 1e-7 * max(1.0, E):
                fails.append(f"v-bar v != -2m for m={m}, s={s}")

        # Spin sums
        su = sum(np.outer(spinor_u(p, m, s), bar(spinor_u(p, m, s))) for s in (0, 1))
        sv = sum(np.outer(spinor_v(p, m, s), bar(spinor_v(p, m, s))) for s in (0, 1))
        if np.max(np.abs(su - (slash(p) + m * np.eye(4)))) > 1e-7 * max(1.0, E):
            fails.append(f"sum u u-bar != p-slash + m for m={m}")
        if np.max(np.abs(sv - (slash(p) - m * np.eye(4)))) > 1e-7 * max(1.0, E):
            fails.append(f"sum v v-bar != p-slash - m for m={m}")

        # Polarisations
        if m > 0:
            eps = pol_vectors(p, m)
            if len(eps) != 3:
                fails.append(f"massive vector should have 3 polarisations, got {len(eps)}")
            for e in eps:
                if abs(dot(e, p)) > 1e-7 * max(1.0, E):
                    fails.append(f"eps . p != 0 for m={m}")
                if abs(dot(e, e.conj()) + 1.0) > 1e-7:
                    fails.append(f"eps . eps* != -1 for m={m}")
            psum = np.zeros((4, 4), dtype=complex)
            for e in eps:
                psum += np.outer(e, e.conj())
            # raise both indices: sum eps^mu eps*^nu = -g^{mu nu} + p^mu p^nu/m^2
            target = -METRIC + np.outer(p, p) / (m * m)
            if np.max(np.abs(psum - target)) > 1e-6 * max(1.0, E * E / (m * m)):
                fails.append(f"massive polarisation sum wrong for m={m}")
        else:
            eps = pol_vectors(p, m)
            if len(eps) != 2:
                fails.append("massless vector should have 2 polarisations")
            for e in eps:
                if abs(dot(e, p)) > 1e-7 * max(1.0, E):
                    fails.append("transverse eps . p != 0")
    return fails


if __name__ == "__main__":
    problems = check_conventions()
    if problems:
        print("CONVENTION FAILURES:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("[OK] all spinor / polarisation conventions verified")


# ---------------------------------------------------------------------------
# Processes
#
# Each returns |M|^2 SUMMED over every external spin / polarisation, at the
# given s and cos(theta).  Vertex conventions mirror the generator's
# (SFF = i y, VFF = i g gamma^mu, SSS = i g, SSV = i g (ka - kb)^mu with
# ALL-INCOMING momenta, SVV = i g g^{mu nu}, VVV = the all-incoming triple
# gauge structure, propagator = i * numerator / (q^2 - M^2)) -- but nothing
# else is shared: indices are contracted by explicit summation and the
# Dirac algebra is explicit 4x4 matrix multiplication.
# ---------------------------------------------------------------------------

def _lower(v: np.ndarray) -> np.ndarray:
    return METRIC @ v


def _vec_prop_contract(A: np.ndarray, B: np.ndarray, q: np.ndarray,
                       mprop: float) -> complex:
    """A_mu (-g^{mu nu} + q^mu q^nu / M^2) B_nu, or -A.B when massless."""
    if mprop <= 0:
        return -dot(A, B)
    return -dot(A, B) + dot(A, q) * dot(B, q) / (mprop * mprop)


def m2_sf_to_sf_fermion(y1: float, y2: float, masses, mprop: float,
                        s: float, ct: float) -> float:
    """S F -> S F, s-channel FERMION exchange (SFF at both vertices).

    Legs (p1, p2, p3, p4) = (S in, F in, S out, F out).
    M = ubar(p4) (i y2) [i (q-slash + M)/(s - M^2)] (i y1) u(p2), q = p1 + p2.
    """
    mS_in, mF_in, mS_out, mF_out = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 + p2
    prop = 1j / (dot(q, q) - mprop * mprop)
    core = (1j * y2) * (slash(q) + mprop * np.eye(4)) * (1j * y1)
    tot = 0.0
    for s_in in (0, 1):
        u_in = spinor_u(p2, mF_in, s_in)
        for s_out in (0, 1):
            u_out = spinor_u(p4, mF_out, s_out)
            amp = prop * (bar(u_out) @ core @ u_in)
            tot += abs(amp) ** 2
    return tot


def m2_fv_to_fv_fermion(g1: float, g2: float, masses, mprop: float,
                        s: float, ct: float) -> float:
    """F V -> F V, s-channel FERMION exchange (VFF at both vertices).

    Legs (p1, p2, p3, p4) = (F in, V in, F out, V out).
    M = eps_mu(p2) eps*_nu(p4)
        ubar(p3) (i g2 gamma^nu) [i (q-slash + M)/(s - M^2)] (i g1 gamma^mu) u(p1)
    """
    mF_in, mV_in, mF_out, mV_out = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 + p2
    prop = 1j / (dot(q, q) - mprop * mprop)
    numer = slash(q) + mprop * np.eye(4)

    # gamma with a LOWER index, ready to contract with a contravariant eps
    def gamma_dot(e):
        return e[0] * GAMMA[0] - e[1] * GAMMA[1] - e[2] * GAMMA[2] - e[3] * GAMMA[3]

    tot = 0.0
    for e_in in pol_vectors(p2, mV_in):
        for e_out in pol_vectors(p4, mV_out):
            V_in = 1j * g1 * gamma_dot(e_in)
            V_out = 1j * g2 * gamma_dot(e_out.conj())
            core = V_out @ numer @ V_in
            for s_in in (0, 1):
                u_in = spinor_u(p1, mF_in, s_in)
                for s_out in (0, 1):
                    u_out = spinor_u(p3, mF_out, s_out)
                    amp = prop * (bar(u_out) @ core @ u_in)
                    tot += abs(amp) ** 2
    return tot


def m2_ss_to_vv_scalar(g1: float, g2: float, masses, mprop: float,
                       s: float, ct: float) -> float:
    """S S -> V V, s-channel SCALAR exchange (SSS then SVV).

    M = (i g1) [i/(s - M^2)] (i g2) (eps*(p3) . eps*(p4)).
    """
    m1, m2, m3, m4 = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 + p2
    prop = 1j / (dot(q, q) - mprop * mprop)
    tot = 0.0
    for e3 in pol_vectors(p3, m3):
        for e4 in pol_vectors(p4, m4):
            amp = (1j * g1) * prop * (1j * g2) * dot(e3.conj(), e4.conj())
            tot += abs(amp) ** 2
    return tot


def m2_ss_to_ss_t_vector(g1: float, g2: float, masses, mprop: float,
                         s: float, ct: float) -> float:
    """S S -> S S, t-channel VECTOR exchange (SSV at both vertices).

    Vertex A carries legs (p1 in, p3 out): all-incoming momenta p1 and -p3,
    so the derivative vertex is i g1 (p1 + p3)^mu.  Likewise B gives
    i g2 (p2 + p4)^nu.  q = p1 - p3.
    """
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 - p3
    A = (1j * g1) * (p1 + p3)
    B = (1j * g2) * (p2 + p4)
    prop = 1j / (dot(q, q) - mprop * mprop)
    amp = prop * _vec_prop_contract(A, B, q, mprop)
    return abs(amp) ** 2


def m2_vv_to_vv_s_vector(g1: float, g2: float, masses, mprop: float,
                         s: float, ct: float) -> float:
    """V V -> V V, s-channel VECTOR exchange (VVV at both vertices).

    All-incoming triple gauge vertex
      i g [ g^{ab}(k_a - k_b)^c + g^{bc}(k_b - k_c)^a + g^{ca}(k_c - k_a)^b ].
    At vertex A the lines are (p1, p2, mediator) with incoming momenta
    (p1, p2, -q); at vertex B they are (p3, p4, mediator) with (-p3, -p4, q).
    """
    m1, m2, m3, m4 = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 + p2
    prop = 1j / (dot(q, q) - mprop * mprop)

    def tri(e_a, e_b, k_a, k_b, k_c, g):
        """Contract the triple-gauge vertex with two polarisations, leaving
        the mediator index free (returned as a contravariant 4-vector)."""
        return (1j * g) * (
            dot(e_a, e_b) * (k_a - k_b)
            + dot(e_a, (k_b - k_c)) * e_b
            + dot(e_b, (k_c - k_a)) * e_a
        )

    tot = 0.0
    for e1 in pol_vectors(p1, m1):
        for e2 in pol_vectors(p2, m2):
            A = tri(e1, e2, p1, p2, -q, g1)
            for e3 in pol_vectors(p3, m3):
                for e4 in pol_vectors(p4, m4):
                    B = tri(e3.conj(), e4.conj(), -p3, -p4, q, g2)
                    amp = prop * _vec_prop_contract(A, B, q, mprop)
                    tot += abs(amp) ** 2
    return tot


def m2_ffbar_to_ffbar_vector(g1: float, g2: float, masses, mprop: float,
                             s: float, ct: float) -> float:
    """f fbar -> f' f'bar, s-channel VECTOR exchange.  Control case."""
    m1, m2, m3, m4 = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    q = p1 + p2
    prop = 1j / (dot(q, q) - mprop * mprop)
    tot = 0.0
    for s1 in (0, 1):
        u1 = spinor_u(p1, m1, s1)
        for s2 in (0, 1):
            v2 = spinor_v(p2, m2, s2)
            cur_in = np.array([bar(v2) @ (1j * g1 * GAMMA[mu]) @ u1
                               for mu in range(4)])
            for s3 in (0, 1):
                u3 = spinor_u(p3, m3, s3)
                for s4 in (0, 1):
                    v4 = spinor_v(p4, m4, s4)
                    cur_out = np.array([bar(u3) @ (1j * g2 * GAMMA[mu]) @ v4
                                        for mu in range(4)])
                    amp = prop * _vec_prop_contract(cur_in, cur_out, q, mprop)
                    tot += abs(amp) ** 2
    return tot


def m2_ss_to_ss_stu_scalar(g: float, masses, mprop: float,
                           s: float, ct: float) -> float:
    """S S -> S S summed COHERENTLY over s-, t- and u-channel scalar exchange.

    Exercises the coherent-sum path: each diagram is
    (i g)(i g) [i/(q^2 - M^2)], and the three are added before squaring, so
    the interference terms are what this actually tests.
    """
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    qs, qt, qu = p1 + p2, p1 - p3, p1 - p4
    amp = 0j
    for q in (qs, qt, qu):
        amp += (1j * g) * (1j * g) * (1j / (dot(q, q) - mprop * mprop))
    return abs(amp) ** 2


# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------

def m2_compton(e: float, m: float, s: float, ct: float) -> float:
    """gamma e -> gamma e, s + u diagrams summed coherently.

    Legs (p1, p2, p3, p4) = (e in, gamma in, e out, gamma out).  Summed over
    all spins and polarisations; divide by 4 for the average.
    """
    masses = (m, 0.0, m, 0.0)
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    qs, qu = p1 + p2, p1 - p4
    Ds, Du = dot(qs, qs) - m * m, dot(qu, qu) - m * m

    def gdot(v):
        return v[0] * GAMMA[0] - v[1] * GAMMA[1] - v[2] * GAMMA[2] - v[3] * GAMMA[3]

    I4 = np.eye(4)
    tot = 0.0
    for e2 in pol_vectors(p2, 0.0):
        for e4 in pol_vectors(p4, 0.0):
            g2, g4 = gdot(e2), gdot(e4.conj())
            # s: absorb the incoming photon, then emit the outgoing one
            Ms = (1j * e * g4) @ (1j * (slash(qs) + m * I4) / Ds) @ (1j * e * g2)
            # u: emit the outgoing photon first, then absorb the incoming one
            Mu = (1j * e * g2) @ (1j * (slash(qu) + m * I4) / Du) @ (1j * e * g4)
            core = Ms + Mu
            for s1 in (0, 1):
                u1 = spinor_u(p1, m, s1)
                for s3 in (0, 1):
                    u3 = spinor_u(p3, m, s3)
                    tot += abs(bar(u3) @ core @ u1) ** 2
    return tot


def m2_compton_peskin_5_87(e: float, m: float, s: float, ct: float) -> float:
    """Peskin & Schroeder eq. (5.87): the spin-averaged Compton |M|^2."""
    masses = (m, 0.0, m, 0.0)
    p1, p2, _, p4 = cm_momenta(s, masses, ct)
    pk = dot(p1, p2).real       # p . k
    pk2 = dot(p1, p4).real      # p . k'
    return float((2 * e ** 4 * (pk2 / pk + pk / pk2
                                + 2 * m * m * (1 / pk - 1 / pk2)
                                + m ** 4 * (1 / pk - 1 / pk2) ** 2)).real)


def sigma_klein_nishina(e: float, m: float, s: float) -> float:
    """Klein-Nishina total cross section in GeV^-2.

    x = omega/m with omega the photon energy in the electron rest frame;
    r_e = e^2/(4 pi m) = alpha/m.
    """
    r_e = e * e / (4 * math.pi * m)
    x = (s - m * m) / (2 * m * m)
    L = math.log(1 + 2 * x)
    return 2 * math.pi * r_e ** 2 * (
        (1 + x) / x ** 3 * (2 * x * (1 + x) / (1 + 2 * x) - L)
        + L / (2 * x) - (1 + 3 * x) / (1 + 2 * x) ** 2)


def sigma_ps_5_13(g: float, s: float) -> float:
    """P&S eq. (5.13) integrated: sigma = g^4 / (12 pi s), massless fermions."""
    return g ** 4 / (12 * math.pi * s)


def m2_moller(g: float, s: float, ct: float, relative_sign: int = -1) -> float:
    """e- e- -> e- e- (massless), t and u channels summed coherently.

    The two diagrams differ by interchange of the two identical external
    fermions, so they enter with OPPOSITE signs.  ``relative_sign=+1`` is
    exposed only so a test can show the wrong choice gives a visibly
    different answer -- the interference term, not just its size, depends
    on it.  Returned summed over all four spins.
    """
    masses = (0.0, 0.0, 0.0, 0.0)
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    qt, qu = p1 - p3, p1 - p4
    t, u = dot(qt, qt).real, dot(qu, qu).real
    tot = 0.0
    for s1 in (0, 1):
        u1 = spinor_u(p1, 0.0, s1)
        for s2 in (0, 1):
            u2 = spinor_u(p2, 0.0, s2)
            for s3 in (0, 1):
                u3 = spinor_u(p3, 0.0, s3)
                for s4 in (0, 1):
                    u4 = spinor_u(p4, 0.0, s4)
                    A = np.array([bar(u3) @ (1j * g * GAMMA[m]) @ u1 for m in range(4)])
                    B = np.array([bar(u4) @ (1j * g * GAMMA[m]) @ u2 for m in range(4)])
                    Mt = (1j / t) * _vec_prop_contract(A, B, qt, 0.0)
                    C = np.array([bar(u4) @ (1j * g * GAMMA[m]) @ u1 for m in range(4)])
                    D = np.array([bar(u3) @ (1j * g * GAMMA[m]) @ u2 for m in range(4)])
                    Mu = (1j / u) * _vec_prop_contract(C, D, qu, 0.0)
                    tot += abs(Mt + relative_sign * Mu) ** 2
    return tot


def m2_moller_literature(g: float, s: float, ct: float) -> float:
    """Moller scattering, spin-averaged, massless limit:

        <|M|^2> = 2 g^4 [ (s^2+u^2)/t^2 + (s^2+t^2)/u^2 + 2 s^2/(t u) ]

    The last term IS the interference; its coefficient is what the
    relative fermion sign controls.
    """
    masses = (0.0, 0.0, 0.0, 0.0)
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    t = dot(p1 - p3, p1 - p3).real
    u = dot(p1 - p4, p1 - p4).real
    return 2 * g ** 4 * ((s * s + u * u) / (t * t)
                         + (s * s + t * t) / (u * u)
                         + 2 * s * s / (t * u))


def m2_vv_to_vv_contact(a: float, b: float, c: float, masses,
                        s: float, ct: float) -> float:
    """V V -> V V through a four-vector CONTACT vertex.

        M = i [ a (e1.e2)(e3*.e4*) + b (e1.e3*)(e2.e4*) + c (e1.e4*)(e2.e3*) ]

    the three-structure basis the generator uses.  Summed over all
    polarisations.
    """
    m1, m2, m3, m4 = masses
    p1, p2, p3, p4 = cm_momenta(s, masses, ct)
    tot = 0.0
    for e1 in pol_vectors(p1, m1):
        for e2 in pol_vectors(p2, m2):
            for e3 in pol_vectors(p3, m3):
                for e4 in pol_vectors(p4, m4):
                    c3, c4 = e3.conj(), e4.conj()
                    amp = 1j * (a * dot(e1, e2) * dot(c3, c4)
                                + b * dot(e1, c3) * dot(e2, c4)
                                + c * dot(e1, c4) * dot(e2, c3))
                    tot += abs(amp) ** 2
    return tot
