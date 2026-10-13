"""
# llp_physics.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Shared physics for the `llp` bundle: analytic parent-flux kernels, a
declared parent-rest-frame LLP energy spectrum (a pinned data product,
NOT a hard-coded amplitude), decay-volume geometry, and the boost /
decay-probability conventions used by LLPFluxFromMesonDecayTool and the
DecayInVolume* yield tools.

The bundle is model- and setting-agnostic by construction:
  - the collider-forward vs beam-dump choice lives entirely in the kernel
    YAML (parent mass, momentum and angular scales, per-interaction
    normalization) and the geometry YAML;
  - the PRODUCTION PHYSICS (which LLP energies the parent decay yields)
    lives entirely in the spectrum data product (`spectrum_spec`), so no
    benchmark's ground-truth amplitude is baked into tool code. A `table`
    spectrum (x, pdf) covers any declared 3-body radiation density; a
    `two_body` spectrum (delta function in x) covers dark-photon-like
    (pi0 -> gamma X) and dark-scalar-like (B -> K X) production.
Nothing in this module hard-codes a parent mass, a beam energy, a
detector location, or a matrix element.

Conventions (declared, and mirrored from the validated reference):
  - Kernels parametrize d^2 N_M / (dp dtheta) per primary interaction for
    *decaying* parents only (the decay-before-absorption fraction is folded
    into n_per_int by whoever pins the kernel).
  - The LLP energy in the parent rest frame is drawn from the declared
    spectrum in x = 2 E*/m_parent; the LLP direction is isotropic in the
    parent rest frame (spin-0 parent convention).
  - Event weights are g^2-stripped: w_i = n_per_int * B_hat_h / n_samples,
    so N_sig(g) = N_int * g^2 * sum_i w_i * P_dec,i(g) * acc_i.
  - The LLP travels in a straight line from the primary vertex at its lab
    angle; the parent flight length is absorbed into the kernel.
  - P_dec = exp(-L1/lam) - exp(-L2/lam), lam = beta*gamma*ctau. The
    lifetime is set either by portal scaling
    (ctau = hbar*c / (g^2 * width_ref)) or by a directly declared
    ctau_grid_m (model-agnostic).
  - Two-track acceptance is evaluated at the MIDPOINT of the in-volume
    segment (g-independent, preserving exact coupling reweighting), with
    the two daughter masses declared (default mu mu).

Attribution: the kernel and geometry loaders and the rest-frame -> lab
boost formula are adapted from the HEPbench reference ground-truth
implementation, benchmarks/llp_forward/_shared/{kernels, pipeline}.py
(HEPbench authors, GPL v3+). The production amplitude that used to live
here has been removed on purpose: it now lives in the pinned spectrum
data product, keeping the tools free of any benchmark's ground truth.
"""

import csv
import math
import os

import numpy as np
import yaml

# hbar*c in GeV*m — the single constant tying the g^2-stripped reference
# width to a lab-frame decay length.
HBARC_M_GEV = 1.973269804e-16
# speed of light in m/s, for the parent lab decay length beta*gamma*c*tau.
C_LIGHT_M_S = 2.99792458e8


# ---------------------------------------------------------------------------
# parent decay-in-flight (the LLP production vertex)
#
# The parent flux is the HARVESTED PYTHIA forward flux (per-parent lab momenta
# + per-collision weights from harvest_forward_flux), not an analytic kernel.
# What the physics core supplies is *where along the parent's flight the LLP is
# produced*: a parent of lab momentum p and proper decay length c*tau decays at
# lab length ell ~ Exp(lambda), lambda = beta*gamma c*tau, and the LLP is
# emitted there. This is g-independent (the parent is SM), so it is sampled
# once and preserves exact coupling reweighting downstream. It matters when
# c*tau is not small vs the baseline (e.g. kaons, c*tau = 3.7 m); prompt
# parents (charm, tau) collapse to ell ~ 0. Sampling is by DETERMINISTIC
# STRATIFICATION (a fixed decay-length quantile grid per parent): the upper
# (decay-length) contour branch is set by the sparse parents decaying just
# before the absorber, and a fixed grid removes the Monte-Carlo jitter there
# while keeping the exact g-reweighting.
# ---------------------------------------------------------------------------
def sample_decay_vertices(pvec_par, ctau_par_m, m_par, n_strata, rng):
    """Stratified decay-in-flight production vertices.

    pvec_par: (N, 3) parent lab 3-momenta; ctau_par_m: parent proper decay
    length [m] (an SM constant); m_par: parent mass [GeV]; n_strata: K strata
    per parent; rng: a numpy Generator (REQUIRED -- see below).

    Returns (rep, vertex): rep (N*K,) the source-parent index of each LLP
    replica, and vertex (N*K, 3) its production point. Each replica carries
    1/K of the parent weight (the caller applies the 1/K).

    The parent's lab decay-length distribution Exp(lambda) is split into K
    equal-probability strata and ONE point is drawn UNIFORMLY AT RANDOM inside
    each:

        u_k = (k + U_k)/K,   U_k ~ Uniform(0,1),   ell_k = -lambda ln(1 - u_k)

    WHY RANDOM AND NOT THE STRATUM MIDPOINT. Until 2.5.0 this used the fixed
    quantile u_k = (k + 1/2)/K. That is not stratified sampling, it is a
    midpoint QUADRATURE rule, and a quadrature rule has O(1) error whenever a
    discontinuity falls inside a panel. Downstream geometry supplies exactly
    such a discontinuity: an LLP counts only if its parent decayed upstream of
    the absorber, and for a forward kaon that acceptance window is the first
    ~2% of the exponential. A stratum straddling the cut was scored
    all-or-nothing, and because the nodes were DETERMINISTIC the error did not
    average out over seeds -- it was a silent bias, invisible to the usual
    seed-variation check, and it understated yields by orders of magnitude at
    strong coupling (five, at K=6).

    Drawing uniformly inside each stratum makes the estimator UNBIASED for any
    integrand, discontinuous ones included, while keeping the variance
    reduction that stratification buys. `rng` is required rather than optional
    precisely so this cannot be silently skipped; pass a seeded Generator to
    keep runs reproducible."""
    pvec_par = np.asarray(pvec_par, dtype=float)
    N = len(pvec_par)
    K = max(1, int(n_strata))
    p_par_mag = np.linalg.norm(pvec_par, axis=1)
    lam = (p_par_mag / m_par) * float(ctau_par_m)      # (N,) lab decay length
    rep = np.repeat(np.arange(N), K)
    strat = np.tile(np.arange(K), N)
    u = (strat + rng.uniform(0.0, 1.0, N * K)) / K     # random WITHIN stratum
    ell = -np.maximum(lam[rep], 1e-300) * np.log1p(-u)
    d_par = pvec_par[rep] / np.maximum(p_par_mag[rep], 1e-300)[:, None]
    vertex = ell[:, None] * d_par
    return rep, vertex


# ---------------------------------------------------------------------------
# parent-rest-frame LLP energy spectrum (a declared data product)
# ---------------------------------------------------------------------------
def read_spectrum_header(path):
    """Parse the `# key=value` metadata a ProductionSpectrumTool CSV carries.

    Returns a dict of strings (possibly empty). The spectrum data product is
    deliberately self-describing: the file records the parent, the LLP mass and
    the reduced branching fraction B_hat it was computed for, so a consumer can
    take B_hat from the same integral that produced f(x) instead of having the
    caller carry it separately -- and can refuse a spectrum that was built for
    a different mass.

    Tolerant by design: a hand-written table with no header is still a valid
    spectrum, and simply yields {}.
    """
    meta = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("#"):
                    break                       # header ends at the first data
                for tok in line.lstrip("#").split():
                    if "=" in tok:
                        k, _, v = tok.partition("=")
                        meta[k.strip()] = v.strip()
    except OSError:
        return {}
    return meta


class LLPSpectrum:
    """The parent-rest-frame LLP energy spectrum, declared as a pinned
    data product rather than computed from a hard-coded amplitude.

    The spectrum is expressed in the dimensionless energy fraction
    x = 2 E* / m_parent (E* the LLP energy in the parent rest frame).
    Two declared types are supported:

    `type: table`
        A tabulated density: columns `x` and `pdf` (the pdf may be
        unnormalized; it is normalized on load). x is sampled by
        inverse-CDF interpolation. This serves any declared 3-body
        radiation density (e.g. M -> lepton nu phi): the ground truth
        exports d(Br)/dx integrated over the other kinematics as the
        table, one file per (parent, m_phi). Loadable from a CSV file
        (header row `x,pdf`, optional `#` comment lines) or a YAML
        mapping carrying `type: table` and either inline `x:`/`pdf:`
        lists or a `csv:` path.

    `type: two_body`
        A delta-function spectrum from two-body kinematics
        M -> phi + X, with the other daughter mass `m_other_gev`. x is
        fixed at x0 = (m_parent^2 + m_phi^2 - m_other^2) / m_parent^2,
        so every event carries the same E*. This serves dark-photon-like
        (pi0 -> gamma X) and dark-scalar-like (B -> K X) production.
        Declared as a YAML mapping with `type: two_body` and
        `m_other_gev`.

    Kinematic openness and the reported cutoff are type-dependent:
      - two_body: open iff m_phi <= m_parent - m_other; cutoff is
        m_parent - m_other.
      - table: open iff the maximum tabulated x gives E* >= m_phi
        (E*_max = x_max * m_parent / 2 >= m_phi); cutoff is the
        E* implied by x_max, i.e. x_max * m_parent / 2.
    """

    def __init__(self, spec_type, *, x=None, pdf=None, m_other_gev=None,
                 source=None):
        self.type = spec_type
        self.source = source
        if spec_type == "table":
            x = np.asarray(x, dtype=float)
            pdf = np.asarray(pdf, dtype=float)
            if x.ndim != 1 or pdf.shape != x.shape or len(x) < 2:
                raise ValueError("table spectrum needs matching x and pdf "
                                 "columns of length >= 2")
            if np.any(pdf < 0.0):
                raise ValueError("table spectrum pdf must be non-negative")
            order = np.argsort(x)
            self.x = x[order]
            self.pdf = pdf[order]
            if not np.any(self.pdf > 0.0):
                raise ValueError("table spectrum pdf is identically zero")
            self.x_max = float(self.x[-1])
            self.x_min = float(self.x[0])
            self.m_other = None
            # inverse-CDF grid (trapezoidal cumulative integral)
            dcdf = 0.5 * (self.pdf[1:] + self.pdf[:-1]) * np.diff(self.x)
            cdf = np.concatenate([[0.0], np.cumsum(dcdf)])
            self._cdf = cdf / cdf[-1]
        elif spec_type == "two_body":
            if m_other_gev is None or float(m_other_gev) < 0.0:
                raise ValueError("two_body spectrum needs m_other_gev >= 0")
            self.m_other = float(m_other_gev)
            self.x = None
            self.pdf = None
            self.x_max = None
            self.x_min = None
            self._cdf = None
        else:
            raise ValueError(f"unknown spectrum type '{spec_type}' "
                             "(expected 'table' or 'two_body')")

    # ---- loaders ----
    @classmethod
    def from_path(cls, path):
        """Load a spectrum from a CSV (table) or YAML (table/two_body)."""
        lower = str(path).lower()
        if lower.endswith(".csv"):
            x, pdf = cls._read_csv(path)
            return cls("table", x=x, pdf=pdf, source=str(path))
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict):
            raise ValueError("spectrum YAML must be a mapping")
        spec = doc.get("spectrum", doc)
        stype = spec.get("type")
        if stype == "two_body":
            return cls("two_body", m_other_gev=spec.get("m_other_gev"),
                       source=str(path))
        if stype == "table":
            if "csv" in spec:
                csv_path = spec["csv"]
                if not os.path.isabs(csv_path):
                    csv_path = os.path.join(os.path.dirname(path), csv_path)
                x, pdf = cls._read_csv(csv_path)
            else:
                x, pdf = spec.get("x"), spec.get("pdf")
                if x is None or pdf is None:
                    raise ValueError("table spectrum YAML needs inline "
                                     "'x' and 'pdf' lists or a 'csv' path")
            return cls("table", x=x, pdf=pdf, source=str(path))
        raise ValueError("spectrum spec missing a valid 'type' "
                         "(expected 'table' or 'two_body')")

    @staticmethod
    def _read_csv(path):
        xs, ps = [], []
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header_seen = False
        for row in rows:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            if not header_seen and not _is_number(first):
                header_seen = True  # column header line (x,pdf)
                continue
            header_seen = True
            xs.append(float(row[0]))
            ps.append(float(row[1]))
        if len(xs) < 2:
            raise ValueError(f"CSV spectrum {path} has < 2 data rows")
        return np.asarray(xs, dtype=float), np.asarray(ps, dtype=float)

    # ---- kinematics ----
    def is_open(self, m_parent, m_phi):
        """Whether the channel is kinematically open at this mass."""
        if self.type == "two_body":
            return m_phi < m_parent - self.m_other
        # table: max tabulated x must reach the phi rest energy
        return 0.5 * self.x_max * m_parent > m_phi

    def cutoff_gev(self, m_parent):
        """The reported kinematic cutoff (max open m_phi)."""
        if self.type == "two_body":
            return m_parent - self.m_other
        return 0.5 * self.x_max * m_parent

    def sample_x(self, m_parent, m_phi, n, rng):
        """Draw n values of x = 2 E*/m_parent from the spectrum."""
        if self.type == "two_body":
            x0 = (m_parent * m_parent + m_phi * m_phi
                  - self.m_other * self.m_other) / (m_parent * m_parent)
            return np.full(n, x0, dtype=float)
        u = rng.uniform(0.0, 1.0, n)
        return np.interp(u, self._cdf, self.x)


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def boost_to_lab(estar, kstar, e_par, pvec_par, m_par):
    """Boost rest-frame momenta (estar, kstar) to the lab frame of a
    parent with lab energy e_par and 3-momentum pvec_par.

    Validated boost formula (reference pipeline):
        k_lab = k* + P [ (P.k*)/(M(E+M)) + E*/M ],
        E_lab = (E* E + P.k*) / M.

    estar: scalar or (n,); kstar: (n, 3); e_par: (n,); pvec_par: (n, 3).
    Returns (elab (n,), klab (n, 3))."""
    estar = np.broadcast_to(np.asarray(estar, dtype=float),
                            (len(kstar),))
    pdotk = np.einsum("ij,ij->i", pvec_par, kstar)
    elab = (estar * e_par + pdotk) / m_par
    coef = pdotk / (m_par * (e_par + m_par)) + estar / m_par
    klab = kstar + coef[:, None] * pvec_par
    return elab, klab


# ---------------------------------------------------------------------------
# geometry and decay-in-volume weighting
# (adapted from hepbench benchmarks/llp_forward/_shared/pipeline.py)
# ---------------------------------------------------------------------------
class DecayVolume:
    """Cylindrical decay volume + downstream detector plane.

    Loaded from a YAML spec carrying (optionally nested under a top-level
    `geometry:` key): z_min_m, z_max_m, r_volume_m, z_det_m, r_det_m, and
    optionally z_shield_m and x_off_m. All distances in meters from the primary
    interaction point.

    - z_shield_m (default z_min): the SHIELD FRONT. Operationally: an LLP
      counts only if its PRODUCTION VERTEX satisfies z < z_shield_m. The
      shielding BEGINS here and absorbs everything of the Standard Model
      downstream of it, so a parent decaying inside it yields nothing that can
      reach the detector; the region upstream, [0, z_shield_m], is the open
      parent decay region. (The earlier wording, "end of the parent production
      region", described the same plane from the other side and was read
      ambiguously in practice.) Parents decay or are swept in [0, z_shield]; the gap
      [z_shield, z_min] is empty baseline. An LLP counts only if its production
      vertex has z < z_shield. The gap sets a minimum survival distance, which
      bounds (well-conditions) the decay-length contour branch. Set
      z_shield = z_min for the absorber at the fiducial face (or the
      prompt-at-IP limit, where all vertices are 0).

      Accepts the legacy key `z_prod_m` as a deprecated alias. That name was
      ambiguous -- it reads as the LLP production point rather than the end of
      the shielding -- and `z_shield_m` is the notation used in the reference
      write-up.
    - x_off_m (default 0 = on-axis): transverse displacement of the detector
      from the beam line; the cylinder axis and the detector plane are centred
      on (x_off, 0) at all z. Off-axis sees a softer, lower-rate LLP flux."""

    def __init__(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("geometry YAML must be a mapping")
        g = spec["geometry"] if "geometry" in spec else spec
        try:
            self.z_min = float(g["z_min_m"])
            self.z_max = float(g["z_max_m"])
            self.r_vol = float(g["r_volume_m"])
            self.z_det = float(g["z_det_m"])
            self.r_det = float(g["r_det_m"])
        except KeyError as e:
            raise ValueError(f"geometry spec missing {e}")
        if "z_shield_m" in g:
            self.z_shield = float(g["z_shield_m"])
        else:                                   # deprecated alias
            self.z_shield = float(g.get("z_prod_m", self.z_min))
        self.x_off = float(g.get("x_off_m", 0.0))

    @classmethod
    def from_yaml(cls, path):
        with open(path) as fh:
            return cls(yaml.safe_load(fh))

    def segment(self, theta):
        """In-volume path segment [L1, L2] for on-axis rays FROM THE ORIGIN at
        angle theta (the prompt-at-IP special case of segment_from_vertex).

        Vectorized; returns (L1, L2, ok). The ray r(z) = z tan(theta) must
        satisfy r < r_vol, capping usable z at z_cap = r_vol / tan(theta).
        Valid for forward rays; callers mask backward rays out of `ok`."""
        theta = np.asarray(theta, dtype=float)
        tan_t = np.tan(theta)
        cos_t = np.cos(theta)
        with np.errstate(divide="ignore"):
            z_cap = np.where(tan_t > 0.0, self.r_vol / tan_t, np.inf)
        z_hi = np.minimum(self.z_max, z_cap)
        ok = (z_hi > self.z_min) & (theta < 0.5 * np.pi)
        L1 = self.z_min / cos_t
        L2 = np.where(ok, z_hi / cos_t, self.z_min / cos_t)
        return L1, L2, ok

    def segment_from_vertex(self, vtx, u):
        """In-volume path segment [L1, L2] for an LLP produced at `vtx` (N,3)
        travelling along unit direction `u` (N,3) -- the general
        (decay-in-flight, off-axis) case. The segment is where the ray
        {vtx + s u : s>0} is inside the radius about the (off-axis) axis AND in
        the z-window [z_min, z_max]; L1, L2 are path lengths from vtx (the LLP
        decay clock starts at production). A vertex at or beyond z_shield is
        absorbed/swept (ok=False); a vertex of 0 reduces to segment(theta)."""
        vtx = np.asarray(vtx, dtype=float)
        u = np.asarray(u, dtype=float)
        x0, y0, z0 = vtx[:, 0], vtx[:, 1], vtx[:, 2]
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        xr = x0 - self.x_off                       # radius about (x_off, 0)
        a = ux * ux + uy * uy
        b = 2.0 * (xr * ux + y0 * uy)
        c = xr * xr + y0 * y0 - self.r_vol ** 2
        disc = b * b - 4.0 * a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        a_safe = np.where(a > 1e-30, a, 1e-30)
        r_lo = np.minimum((-b - sq) / (2 * a_safe), (-b + sq) / (2 * a_safe))
        r_hi = np.maximum((-b - sq) / (2 * a_safe), (-b + sq) / (2 * a_safe))
        small = a <= 1e-30                          # ray parallel to axis
        r_lo = np.where(small, np.where(c < 0, -np.inf, np.inf), r_lo)
        r_hi = np.where(small, np.where(c < 0, np.inf, -np.inf), r_hi)
        noroot = (disc < 0) & (~small)
        r_lo = np.where(noroot, np.inf, r_lo)
        r_hi = np.where(noroot, -np.inf, r_hi)
        fwd = uz > 0
        uz_safe = np.where(fwd, uz, 1.0)
        z_lo = (self.z_min - z0) / uz_safe
        z_hi = (self.z_max - z0) / uz_safe
        s_lo = np.maximum(np.maximum(r_lo, z_lo), 0.0)
        s_hi = np.minimum(r_hi, z_hi)
        ok = fwd & (z0 < self.z_shield) & (s_hi > s_lo)
        L1 = np.where(ok, s_lo, 0.0)
        L2 = np.where(ok, s_hi, 0.0)
        return L1, L2, ok


def decay_probability(L1, L2, lam):
    """P(decay in [L1, L2]) for decay length lam (vectorized)."""
    return np.exp(-L1 / lam) - np.exp(-L2 / lam)


def two_track_pass(p4_phi, vertex, m_phi, m1, m2, geom, rng):
    """g-independent two-track acceptance at the midpoint convention.

    p4_phi: (n, 4) lab four-momenta [E, px, py, pz]; vertex: (n, 3)
    decay positions. Samples one isotropic two-body decay
    phi -> d1(m1) d2(m2) per event, boosts both daughters to the lab,
    propagates straight lines to z_det, and requires both to hit
    r < r_det moving forward. m1 and m2 need not be equal (default use
    is mu mu); the two daughters are back-to-back in the phi rest frame
    with a common momentum magnitude but individual energies."""
    n = len(p4_phi)
    if m_phi <= m1 + m2:
        raise ValueError(
            f"phi -> d1 d2 closed: m_phi <= m1 + m2 "
            f"(m_phi={m_phi}, m1={m1}, m2={m2})")
    # Kallen momentum: back-to-back |k*| for the two daughters.
    lam = (m_phi ** 2 - (m1 + m2) ** 2) * (m_phi ** 2 - (m1 - m2) ** 2)
    pstar = math.sqrt(lam) / (2.0 * m_phi)
    estar1 = (m_phi ** 2 + m1 ** 2 - m2 ** 2) / (2.0 * m_phi)
    estar2 = (m_phi ** 2 + m2 ** 2 - m1 ** 2) / (2.0 * m_phi)
    cth = rng.uniform(-1.0, 1.0, n)
    sth = np.sqrt(1.0 - cth ** 2)
    az = rng.uniform(0.0, 2.0 * np.pi, n)
    kstar = pstar * np.stack(
        [sth * np.cos(az), sth * np.sin(az), cth], axis=1)
    E, pvec = p4_phi[:, 0], p4_phi[:, 1:]
    ok = np.ones(n, dtype=bool)
    for sign, estar in ((+1.0, estar1), (-1.0, estar2)):
        k = sign * kstar
        _, klab = boost_to_lab(estar, k, E, pvec, m_phi)
        forward = klab[:, 2] > 0.0
        dz = geom.z_det - vertex[:, 2]
        t = np.where(forward, dz / np.where(forward, klab[:, 2], 1.0), np.inf)
        xy = vertex[:, :2] + t[:, None] * klab[:, :2]
        dx = xy[:, 0] - geom.x_off              # detector plane centred off-axis
        r = np.sqrt(dx * dx + xy[:, 1] ** 2)
        ok &= forward & (r < geom.r_det)
    return ok


def photon_pass(p4_phi, vertex, geom):
    """Acceptance for a two-photon (or otherwise collinear) LLP decay: for a
    light, highly boosted LLP the daughters are collinear with it, so require
    the LLP line-of-flight from the decay vertex to reach the detector face
    within r_det (about the off-axis axis). Lifetime-independent."""
    pmag = np.linalg.norm(p4_phi[:, 1:], axis=1)
    dirs = p4_phi[:, 1:] / np.maximum(pmag, 1e-300)[:, None]
    fwd = dirs[:, 2] > 0.0
    dz = geom.z_det - vertex[:, 2]
    t = np.where(fwd, dz / np.where(fwd, dirs[:, 2], 1.0), np.inf)
    xy = vertex[:, :2] + t[:, None] * dirs[:, :2]
    dx = xy[:, 0] - geom.x_off
    r = np.sqrt(dx * dx + xy[:, 1] ** 2)
    return fwd & (r < geom.r_det)
