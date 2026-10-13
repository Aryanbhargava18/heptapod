"""
# vertices.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

THE MODEL. Everything model-dependent about an LLP production spectrum lives in
this file: the spin-summed squared amplitude for radiating the LLP off a
charged-lepton leg, with the overall coupling factored out.

EXTENDING TO A NEW INTERACTION
------------------------------
Add one function with the signature

    msq(P, p_l, p_nu, p_phi, m_l, m_phi) -> array

returning the spin-summed |M|^2 at unit coupling, and register it in
`VERTICES`. Nothing else moves: `kinematics.py` (phase space, quadrature) and
`production_spectrum.py` (parents, I/O, provenance) are model-independent by
construction. The natural next entries are a pseudoscalar coupling
(gamma_5 at the emission vertex), a vector coupling (gamma^mu), and an
axial-vector coupling (gamma^mu gamma_5).

The SCALAR vertex is implemented and validated for BOTH parent families.

THE VECTOR CASE is the harder one and was added last. V -> l+ l- phi has two
diagrams (the LLP radiated off either lepton leg) which interfere, and the
polarisation sum must be averaged over the parent's three states. Neither
diagram conserves the leptonic current on its own, so an omitted diagram or a
flipped relative sign still yields a smooth, positive, entirely plausible
spectrum with the wrong normalisation -- which is why it is evaluated as an
explicit numerical trace and pinned by two checks that are physics rather than
self-consistency:

    Ward identity     P_mu T^{mu nu} = 0 numerically   (< 1e-10 on a
                                                        physical Dalitz grid)
    closed-form limit Gamma(V -> l+ l-) at unit coupling reproduces
                      M/(12 pi) (1 + 2 m^2/M^2) sqrt(1 - 4 m^2/M^2) to 1e-12

Both are in the test suite. The second doubles as the coupling normalisation:
there is no decay constant for a vector, so g_V is fixed by dividing the
MEASURED V -> l+ l- width by that same unit-coupling quantity, which cancels
this module's trace and polarisation conventions rather than assuming they
agree with a textbook formula's.

These channels are not a completeness exercise. Above the pseudoscalar
kinematic wall (m_phi ~ 1.76 GeV) the charmonia are the ONLY open production
mode, so they carry the entire high-mass reach on their own.

PHYSICS
-------
The LLP is radiated from an internal, off-shell charged lepton of momentum
q = p_l + p_phi, propagator

    S_F(q) = i (qslash + m_l) / (q^2 - m_l^2),
    q^2 - m_l^2 = m_phi^2 + 2 p_l . p_phi,

so the amplitude carries 1/D with D = 2 p_l.p_phi + m_phi^2. Squaring and
summing over spins gives a ratio of Dirac traces; the epsilon (gamma_5) terms
vanish for three independent momenta, so the closed form below is exact.

Amplitude structure follows Carlson & Rislow, Phys. Rev. D 86, 035013 (2012)
[arXiv:1206.3587]; its application to forward LLP production is set out in
Mammen Abraham & Fieg [arXiv:2501.09071]. The vector-parent amplitude, when
added, should follow Mitra & Sahoo, Phys. Rev. D 104, 015002 (2021)
[arXiv:2103.08284], including its Ward-identity check.
"""
from __future__ import annotations

import numpy as np

from .kinematics import mdot


#: Dirac representation, metric (+,-,-,-). Used by the VECTOR amplitude, which
#: is evaluated as an explicit numerical trace rather than a hand-derived
#: closed form. V -> l+ l- phi has two diagrams that interfere plus a
#: polarisation average, and an algebra slip there is SILENT: it yields a
#: plausible-looking spectrum with the wrong normalisation. The numerical route
#: is pinned by two physics checks a wrong trace cannot pass -- the Ward
#: identity, and reproducing the MEASURED V -> l+ l- width.
_ID4 = np.eye(4, dtype=complex)
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])
_GAMMA = np.array([
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
    [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]],
], dtype=complex)


def slash(p):
    """p_slash = gamma^mu p_mu, vectorised over any leading shape."""
    return np.einsum('...m,mij->...ij', np.asarray(p, dtype=float) * _METRIC,
                     _GAMMA)


def _bar(A):
    """gamma^0 A^dagger gamma^0, the Dirac adjoint of a 4x4 vertex block."""
    return np.einsum('ij,...kj,kl->...il', _GAMMA[0], A.conj(), _GAMMA[0])


def _tr4(ab, cd, ac, bd, ad, bc):
    """Tr[a/ b/ c/ d/] / 4 from the standard contraction identity."""
    return ab * cd - ac * bd + ad * bc


def msq_scalar_pseudoscalar_parent(P, p_l, p_nu, p_phi, m_l, m_phi):
    """Spin-summed |M|^2 for P -> l nu phi with a SCALAR L = -g phi lbar l.

    Unit coupling: the physical rate carries C_P^2 g^2 on top, with
    C_P = (G_F |V_P| f_P / sqrt(2)) supplied by the caller.

    Evaluated from the trace closed form

      T = 2 { Tr[l/_1 l/ P/ p/_2 P/ l/]
              + m^2 Tr[P/ p/_2 P/ l/]
              + m^2 Tr[l/_1 P/ p/_2 P/]
              + m^2 Tr[l/ P/ p/_2 P/] } / D^2,

    with l = p_l + p_phi the off-shell lepton momentum and
    D = 2 p_l.p_phi + m_phi^2 the propagator denominator. Vectorised over any
    leading grid shape.
    """
    ell = p_l + p_phi
    v = {"l1": p_l, "p2": p_nu, "P": P, "l": ell}
    keys = list(v)
    d = {(a, b): mdot(v[a], v[b]) for a in keys for b in keys}

    def t4(a, b, c, e):
        return _tr4(d[a, b], d[c, e], d[a, c], d[b, e], d[a, e], d[b, c])

    def t6(a, b, c, e, f, g):
        return (d[a, b] * t4(c, e, f, g)
                - d[a, c] * t4(b, e, f, g)
                + d[a, e] * t4(b, c, f, g)
                - d[a, f] * t4(b, c, e, g)
                + d[a, g] * t4(b, c, e, f))

    m2 = m_l * m_l
    T = 2.0 * 4.0 * (t6("l1", "l", "P", "p2", "P", "l")
                     + m2 * t4("P", "p2", "P", "l")
                     + m2 * t4("l1", "P", "p2", "P")
                     + m2 * t4("l", "P", "p2", "P"))
    D = 2.0 * mdot(p_l, p_phi) + m_phi * m_phi
    return T / (D * D)


def _vector_tensor(P, p_m, p_p, p_phi, m_l, m_phi):
    """T^{mu nu} = sum_spins M^mu (M^nu)* for V -> l-(p_m) l+(p_p) phi.

    The LLP is radiated from either lepton leg, so there are TWO diagrams and
    they interfere -- unlike the pseudoscalar case, where a single internal
    line carries the whole amplitude:

        Gamma^mu = (p_m/ + k/ + m) gamma^mu / D_m
                 + gamma^mu (-p_p/ - k/ + m) / D_p

    with D_m = 2 p_m.k + m_phi^2 and D_p = 2 p_p.k + m_phi^2 the two propagator
    denominators. The vector's coupling to the lepton current is factored out
    (unit coupling), exactly as C_P is for a pseudoscalar; `parents.py`
    restores it by normalising to the measured V -> l+ l- width.

    Returned with the Lorentz indices UNCONTRACTED so the caller can apply the
    polarisation sum and, separately, test the Ward identity.
    """
    k = p_phi
    D_m = 2.0 * mdot(p_m, k) + m_phi * m_phi
    D_p = 2.0 * mdot(p_p, k) + m_phi * m_phi

    sl_m, sl_p, sl_k = slash(p_m), slash(p_p), slash(k)
    num_m = sl_m + sl_k + m_l * _ID4          # (p_m + k)/ + m
    num_p = -sl_p - sl_k + m_l * _ID4         # -(p_p + k)/ + m

    # Gamma^mu, one 4x4 block per Lorentz index; gamma^mu carries an UPPER
    # index here, so no metric factor enters the vertex itself.
    G = (np.einsum('...ij,mjk->...mik', num_m, _GAMMA) / D_m[..., None, None, None]
         + np.einsum('mij,...jk->...mik', _GAMMA, num_p) / D_p[..., None, None, None])
    Gb = _bar(G)

    A = sl_m + m_l * _ID4                     # sum over l- spins
    B = sl_p - m_l * _ID4                     # sum over l+ spins
    # T^{mu nu} = Tr[A Gamma^mu B Gammabar^nu]
    left = np.einsum('...ij,...mjk->...mik', A, G)
    right = np.einsum('...ij,...njk->...nik', B, Gb)
    return np.einsum('...mij,...nji->...mn', left, right)


def _polarisation_average(T, P, m_h):
    """(1/3) (-g_{mu nu} + P_mu P_nu / m_h^2) T^{mu nu}.

    The average over the parent's three polarisation states. The P P term
    vanishes for a conserved current, but it is applied rather than dropped:
    keeping it means `msq_scalar_vector_parent` stays correct if the vertex is
    ever changed to something non-conserving, and its size is a live check on
    the amplitude (see the Ward-identity test).
    """
    g_term = -(T[..., 0, 0] - T[..., 1, 1] - T[..., 2, 2] - T[..., 3, 3])
    P_low = np.asarray(P, dtype=float) * _METRIC
    pp_term = np.einsum('...m,...n,...mn->...', P_low, P_low, T) / (m_h * m_h)
    return (g_term + pp_term).real / 3.0


def msq_scalar_vector_parent(P, p_l, p_nu, p_phi, m_l, m_phi):
    """Spin-summed, polarisation-AVERAGED |M|^2 for V -> l+ l- phi.

    Signature matches the pseudoscalar vertex so `kinematics.py` needs no
    special case: `p_l` is the l-, `p_nu` is the l+ (both massive here, which
    is why the recoil split in `kinematics.momenta` had to be generalised).

    Unit coupling: the physical rate carries g_V^2 g^2 on top, with g_V fixed
    by the measured V -> l+ l- width rather than by a decay constant.
    """
    m_h = np.sqrt(np.maximum(mdot(P, P), 0.0))
    T = _vector_tensor(P, p_l, p_nu, p_phi, m_l, m_phi)
    return _polarisation_average(T, P, m_h)


def gamma_v_to_ll_unit(m_h, m_l):
    """Gamma(V -> l+ l-) at UNIT coupling, from this module's own machinery.

    Used to convert a measured V -> l+ l- width into the effective coupling
    that multiplies the three-body amplitude. Computing it here, rather than
    substituting the textbook closed form, means the conversion cancels
    whatever trace and polarisation conventions this module uses instead of
    silently assuming they agree with a formula's. The test suite pins it
    against M/(12 pi) (1 + 2 m^2/M^2) sqrt(1 - 4 m^2/M^2).
    """
    if m_h <= 2.0 * m_l:
        return 0.0
    E = 0.5 * m_h
    p = np.sqrt(E * E - m_l * m_l)
    p_m = np.array([[E, 0.0, 0.0, p]])
    p_p = np.array([[E, 0.0, 0.0, -p]])
    P = np.array([[m_h, 0.0, 0.0, 0.0]])
    T = np.einsum('...ij,mjk,...kl,nli->...mn',
                  slash(p_m) + m_l * _ID4, _GAMMA,
                  slash(p_p) - m_l * _ID4, _GAMMA)
    msq = _polarisation_average(T, P, m_h)[0]
    return float(p / (8.0 * np.pi * m_h * m_h) * msq)


def ward_residual(P, p_l, p_nu, p_phi, m_l, m_phi):
    """|P_mu T^{mu nu}| relative to |T|, which must vanish for a conserved current.

    Exposed rather than kept in the test file because it is the cheapest real
    check on the vector amplitude: it is sensitive to a wrong relative sign or
    a missing diagram, both of which leave the spectrum's SHAPE plausible.
    """
    T = _vector_tensor(P, p_l, p_nu, p_phi, m_l, m_phi)
    P_low = np.asarray(P, dtype=float) * _METRIC
    contracted = np.einsum('...m,...mn->...n', P_low, T)
    scale = np.sqrt(np.einsum('...mn,...mn->...', abs(T) ** 2, np.ones_like(T.real)))
    scale = np.where(scale > 0, scale, 1.0)
    return np.max(np.abs(contracted), axis=-1) / scale


# Registry: (family, interaction) -> squared amplitude at unit coupling.
VERTICES = {
    ("pseudoscalar", "scalar"): msq_scalar_pseudoscalar_parent,
    ("vector", "scalar"): msq_scalar_vector_parent,
}

#: Interactions available per parent family, for error messages and discovery.
AVAILABLE = {
    "pseudoscalar": sorted(i for (f, i) in VERTICES if f == "pseudoscalar"),
    "vector": sorted(i for (f, i) in VERTICES if f == "vector"),
}


def get(family, interaction):
    """Look up a squared amplitude, with an actionable error if absent."""
    try:
        return VERTICES[(family, interaction)]
    except KeyError:
        have = AVAILABLE.get(family, [])
        raise KeyError(
            f"no {interaction!r} vertex for a {family} parent; "
            f"implemented: {have or 'none'}. Add one function to "
            f"spectra/vertices.py and register it in VERTICES -- the "
            f"kinematics and the tool shell are model-independent.")
