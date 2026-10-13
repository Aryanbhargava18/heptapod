"""
# kinematics.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Three-body kinematics and the quadrature that turns a matrix element into an
LLP energy spectrum. MODEL-INDEPENDENT: nothing here knows what the emitted
particle couples to. The interaction lives in `vertices.py`; swapping it is how
this module serves a different mediator.

VARIABLES (following the reference write-up)

    x = 2 E*_phi / m_h      LLP energy fraction in the parent rest frame
    c = cos(theta*)         helicity angle of the charged lepton in the
                            recoil (l nu) or (l+ l-) rest frame, measured
                            from that system's direction of flight

The Dalitz domain is a RECTANGLE in (x, c): the constraint m_12^2 > m_l^2
involves x only, so c always spans [-1, 1]. That is what lets the two
directions be integrated on independent grids.

QUADRATURE

Gauss-Legendre in both directions, with the substitution

    x = x_min + t^2,        dx = 2 t dt

in the energy direction. The Jacobian |d(m12^2, m23^2)/d(x, c)| carries a
factor |p*_phi| = sqrt(E*^2 - m_phi^2), which vanishes like sqrt(x - x_min) at
the lower endpoint. Left alone that square-root edge caps convergence at
O(h^1.5) and forces very fine grids; absorbing it into t makes the integrand
smooth and recovers Gauss-Legendre's fast convergence.

Measured over 28 (parent, m_phi) cases spanning 0.5%-99.5% of threshold:

    scheme                          evaluations   worst relative error
    midpoint  (n_x=512, n_c=128)         65536              1.3e-4
    GL + sqrt substitution (32, 64)       2048              3.8e-6

i.e. 32x cheaper and 34x more accurate. Defaults are set from that study.
"""
from __future__ import annotations

import numpy as np

# Defaults from the convergence study documented above. n_c is cheap and
# saturates; n_x is the binding direction because of the endpoint behaviour.
N_X_DEFAULT = 32
N_C_DEFAULT = 64


def mdot(p, q):
    """Minkowski product of contravariant 4-vectors, vectorised over axis 0."""
    return (p[..., 0] * q[..., 0] - p[..., 1] * q[..., 1]
            - p[..., 2] * q[..., 2] - p[..., 3] * q[..., 3])


def x_domain(m_h, m_l, m_phi, m_l2=0.0):
    """Physical range of x = 2 E*_phi / m_h.

    Lower edge: the LLP must be at least at rest, E* >= m_phi.
    Upper edge: the recoil system must be at least its constituents' mass,
    m_12 >= m_l + m_l2 -- m_l2 = 0 for P -> l nu phi (massless neutrino),
    m_l2 = m_l for V -> l+ l- phi.
    """
    m_rec = m_l + m_l2
    x_min = 2.0 * m_phi / m_h
    x_max = 1.0 + (m_phi * m_phi - m_rec * m_rec) / (m_h * m_h)
    return x_min, x_max


def _recoil_split(m12sq, m12, m_l, m_l2):
    """(E, p) of particle 1 in the recoil rest frame, for masses (m_l, m_l2).

    General two-body split, so the same kinematics serves P -> l nu phi
    (m_l2 = 0) and V -> l+ l- phi (m_l2 = m_l). Writing p as
    sqrt(E^2 - m_l^2) rather than the massless shortcut (m12^2 - m_l^2)/2m12
    is what makes the massive-massive case correct.
    """
    Es = (m12sq + m_l * m_l - m_l2 * m_l2) / (2.0 * m12)
    ps = np.sqrt(np.maximum(Es * Es - m_l * m_l, 0.0))
    return Es, ps


def momenta(m_h, m_l, m_phi, x, c, m_l2=0.0):
    """Parent-rest-frame four-momenta (P, p_l, p_nu, p_phi) on a grid.

    x, c are broadcast against each other; the return arrays carry a trailing
    axis of length 4. The LLP is placed along -z and the recoil system along
    +z, with the charged lepton decayed in the x-z plane -- the azimuth is
    physically irrelevant for an unpolarised parent, which is the modelling
    assumption stated in the reference write-up.

    Points outside the Dalitz domain are returned with a False `ok` mask
    rather than raising, so the caller can integrate over a rectangle.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    x, c = np.broadcast_arrays(x, c)

    m_rec = m_l + m_l2
    E3 = 0.5 * x * m_h
    p3sq = E3 * E3 - m_phi * m_phi
    m12sq = m_h * m_h + m_phi * m_phi - 2.0 * m_h * E3
    ok = (p3sq > 0.0) & (m12sq > m_rec * m_rec)

    p3mag = np.sqrt(np.maximum(p3sq, 0.0))
    m12 = np.sqrt(np.maximum(m12sq, 0.0))
    safe = np.where(ok, m12, 1.0)

    # charged lepton in the recoil rest frame
    Es, ps = _recoil_split(m12sq, safe, m_l, m_l2)
    # boost of the recoil system in the parent frame (along +z)
    E12 = m_h - E3
    gam = E12 / safe
    gb = p3mag / safe
    s = np.sqrt(np.maximum(1.0 - c * c, 0.0))

    zeros = np.zeros_like(x)
    p_l = np.stack([gam * Es + gb * ps * c, ps * s, zeros,
                    gam * ps * c + gb * Es], axis=-1)
    p_phi = np.stack([E3, zeros, zeros, -p3mag], axis=-1)
    P = np.stack([np.full_like(x, m_h), zeros, zeros, zeros], axis=-1)
    # The partner is fixed by momentum conservation, so it is on-shell at
    # m_l2 by construction rather than by a second boost.
    p_nu = P - p_l - p_phi
    return P, p_l, p_nu, p_phi, ok


def jacobian(m_h, m_l, m_phi, x, m_l2=0.0):
    """|d(m12^2, m23^2) / d(x, c)|, which is independent of c.

    m12^2 = m_h^2 + m_phi^2 - x m_h^2      -> |dm12^2/dx| = m_h^2
    m23^2 = m_h^2 + m_l^2 - 2 m_h E_l      -> |dm23^2/dc| = 2 m_h gb ps

    The gb factor is |p*_phi|/m12, and is the source of the sqrt edge at
    x_min that the substitution in `spectrum` removes.
    """
    x = np.asarray(x, dtype=float)
    E3 = 0.5 * x * m_h
    p3mag = np.sqrt(np.maximum(E3 * E3 - m_phi * m_phi, 0.0))
    m12sq = m_h * m_h + m_phi * m_phi - 2.0 * m_h * E3
    m12 = np.sqrt(np.maximum(m12sq, 0.0))
    good = m12 > (m_l + m_l2)
    safe = np.where(good, m12, 1.0)
    _, ps = _recoil_split(m12sq, safe, m_l, m_l2)
    gb = p3mag / safe
    return np.where(good, (m_h * m_h) * (2.0 * m_h * gb * ps), 0.0)


def _gl_nodes_x(m_h, m_l, m_phi, n_x, m_l2=0.0):
    """Gauss-Legendre nodes in x under x = x_min + t^2, with dx weights."""
    x_lo, x_hi = x_domain(m_h, m_l, m_phi, m_l2)
    if not (x_hi > x_lo):
        return np.zeros(0), np.zeros(0)
    t_node, t_w = np.polynomial.legendre.leggauss(int(n_x))
    T = np.sqrt(x_hi - x_lo)
    t = 0.5 * T * (t_node + 1.0)
    w = 0.5 * T * t_w
    return x_lo + t * t, w * 2.0 * t          # dx = 2t dt


def dgamma_dx(m_h, m_l, m_phi, msq, n_x=N_X_DEFAULT, n_c=N_C_DEFAULT,
              m_l2=0.0):
    """(x nodes, dGamma/dx at those nodes, dx quadrature weights).

    `msq` is a callable (P, p_l, p_nu, p_phi, m_l, m_phi) -> spin-summed
    |M|^2 with any overall coupling factored out; see `vertices.py`. The
    c-integral is done at each x node, so the return is already the
    single-differential rate.
    """
    xs, wx = _gl_nodes_x(m_h, m_l, m_phi, n_x, m_l2)
    if xs.size == 0:
        return xs, xs.copy(), xs.copy()
    cs, wc = np.polynomial.legendre.leggauss(int(n_c))

    X = xs[:, None]
    C = cs[None, :]
    P, p_l, p_nu, p_phi, ok = momenta(m_h, m_l, m_phi, X, C, m_l2)
    val = msq(P, p_l, p_nu, p_phi, m_l, m_phi)
    val = np.where(ok, val, 0.0)
    # PDG 50.22 three-body phase space, with the (x, c) Jacobian
    pref = jacobian(m_h, m_l, m_phi, xs, m_l2) / ((2.0 * np.pi) ** 3
                                                  * 32.0 * m_h ** 3)
    return xs, pref * (val @ wc), wx


def width_and_spectrum(m_h, m_l, m_phi, msq, n_x=N_X_DEFAULT,
                       n_c=N_C_DEFAULT, m_l2=0.0):
    """(Gamma, x nodes, normalised f(x)) with the coupling factored out.

    Gamma is the integral of dGamma/dx; f(x) = (1/Gamma) dGamma/dx integrates
    to one, which is the normalisation the downstream sampler expects.
    """
    xs, dg, wx = dgamma_dx(m_h, m_l, m_phi, msq, n_x=n_x, n_c=n_c, m_l2=m_l2)
    if xs.size == 0:
        return 0.0, xs, dg
    gamma = float(np.dot(dg, wx))
    if gamma <= 0.0:
        return 0.0, xs, np.zeros_like(dg)
    return gamma, xs, dg / gamma
