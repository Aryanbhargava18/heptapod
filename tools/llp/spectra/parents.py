"""
# parents.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Hosted Standard-Model parent data for LLP production spectra.

Every number carries a source. The tool emits this provenance alongside its
results so a reader can audit which constants produced a given spectrum without
reading the source.

CONVENTIONS
-----------
Masses and widths in GeV, proper decay lengths c*tau in metres. A parent is
identified by a short name; charge conjugates are folded (``K`` covers K+ and
K-), matching the harvested forward flux.

Two production families are hosted, following the reference write-up:

    P -> mu nu_mu phi      P in {pi, K, D, Ds}          (pseudoscalar)
    V -> mu+ mu- phi       V in {rho, omega, phi(1020), (pseudoscalar)
                                 Jpsi, psi2S}            (vector)

For a pseudoscalar the weak amplitude is fixed by G_F, the CKM element and the
meson decay constant, giving the overall factor
C_P^2 = (G_F |V_P| f_P / sqrt(2))^2. For a vector the effective dimuon coupling
is instead normalised to the MEASURED V -> mu+ mu- partial width, so the hosted
datum is that width rather than a decay constant.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Universal constants
# --------------------------------------------------------------------------
# [PDG2024] R. L. Workman et al. (Particle Data Group),
#   Prog. Theor. Exp. Phys. 2024, 083C01 (2024).
G_F_GEV2 = 1.1663788e-5        # Fermi constant [GeV^-2]        [PDG2024]
ALPHA_EM = 7.2973525693e-3     # fine-structure constant        [PDG2024]
HBARC_M_GEV = 1.973269804e-16  # hbar c [GeV m]                 [PDG2024]
HBAR_GEV_S = 6.582119569e-25   # hbar [GeV s]                   [PDG2024]
C_LIGHT_M_S = 2.99792458e8     # speed of light [m/s]           exact, SI
M_MU_GEV = 0.1056583755        # muon mass [GeV]                [PDG2024]
M_E_GEV = 0.000510998950       # electron mass [GeV]            [PDG2024]
M_TAU_GEV = 1.77686            # tau mass [GeV]                 [PDG2024]

LEPTON_MASS_GEV = {"e": M_E_GEV, "mu": M_MU_GEV, "tau": M_TAU_GEV}

# --------------------------------------------------------------------------
# Citations, emitted verbatim in the provenance block
# --------------------------------------------------------------------------
REFERENCES = {
    "PDG2024": ("R. L. Workman et al. (Particle Data Group), "
                "Prog. Theor. Exp. Phys. 2024, 083C01 (2024)"),
    "CarlsonRislow2012": ("C. E. Carlson and B. C. Rislow, "
                          "Phys. Rev. D 86, 035013 (2012), arXiv:1206.3587 -- "
                          "pseudoscalar P -> l nu X amplitude"),
    "MitraSahoo2021": ("M. Mitra and D. Sahoo, Phys. Rev. D 104, 015002 "
                       "(2021), arXiv:2103.08284 -- vector V -> mu+ mu- X "
                       "amplitude"),
    "AbrahamFieg2025": ("R. Mammen Abraham and M. Fieg, Phys. Rev. D (2025), "
                        "arXiv:2501.09071 -- muonphilic scalars at forward "
                        "facilities"),
}


class Parent:
    """One hosted SM parent hadron.

    name         short key, charge conjugates folded
    pdg          |PDG id|
    family       'pseudoscalar' (P -> l nu phi) or 'vector' (V -> l+ l- phi)
    mass_gev     parent mass
    ctau_m       proper decay length [m]; 0.0 for strong/EM decays (prompt)
    width_gev    total width Gamma_{h,tot}
    ckm          |V_ij| governing the leptonic decay      (pseudoscalar only)
    f_gev        decay constant f_P                        (pseudoscalar only)
    gamma_ll_gev measured V -> l+ l- partial width          (vector only)
    sources      per-field citation keys
    """

    __slots__ = ("name", "pdg", "family", "mass_gev", "ctau_m", "width_gev",
                 "ckm", "f_gev", "gamma_ll_gev", "sources")

    def __init__(self, name, pdg, family, mass_gev, ctau_m, width_gev,
                 ckm=None, f_gev=None, gamma_ll_gev=None, sources=None):
        self.name = name
        self.pdg = pdg
        self.family = family
        self.mass_gev = mass_gev
        self.ctau_m = ctau_m
        self.width_gev = width_gev
        self.ckm = ckm
        self.f_gev = f_gev
        self.gamma_ll_gev = gamma_ll_gev
        self.sources = sources or {}

    def as_dict(self):
        d = {s: getattr(self, s) for s in self.__slots__ if s != "sources"}
        d["sources"] = dict(self.sources)
        return d


_PDG = "PDG2024"


def _tau(tau_s):
    """c*tau [m] from the PDG lifetime -- the PUBLISHED datum.

    Storing the lifetime and deriving c*tau, rather than storing a rounded
    c*tau, removes a class of silent drift: a c*tau rounded to four figures
    shifts B_hat by up to 4e-4, which is comparable to the reference's own
    Monte-Carlo precision and would show up as an unexplained tool-versus-
    reference offset.
    """
    return C_LIGHT_M_S * tau_s


def _width(tau_s):
    """Total width [GeV] from the same lifetime."""
    return HBAR_GEV_S / tau_s

# Widths are quoted directly where the PDG lists them, and otherwise derived
# from the lifetime as Gamma = hbar / tau = HBARC / c*tau. Both routes are
# marked in `sources` so the derivation is visible.
#
# PINNING. These constants must stay numerically identical to the ones the
# graded reference calculation uses. They are physically the same quantities,
# but two independently maintained tables drift: an 0.3% difference in |V_us|
# alone moves B_hat by 0.6% (the rate goes as |V|^2), which exceeds the
# reference's own Monte-Carlo precision and would appear as a systematic
# tool-versus-hand-written offset that looks like a physics result. If a value
# here changes, change it in the reference in the same commit.
PARENTS = {
    # ---- pseudoscalar: P -> mu nu_mu phi -------------------------------- #
    "pi": Parent(
        "pi", 211, "pseudoscalar", 0.13957039, _tau(2.6033e-8), _width(2.6033e-8),
        ckm=0.97435,          # |V_ud|
        f_gev=0.1302,         # f_pi
        sources={"mass_gev": _PDG, "ctau_m": _PDG + " (derived: c * tau)", "ckm": _PDG,
                 "f_gev": _PDG + " (FLAG N_f=2+1+1 average)",
                 "width_gev": "derived: hbar / tau"}),
    "K": Parent(
        "K", 321, "pseudoscalar", 0.493677, _tau(1.2380e-8), _width(1.2380e-8),
        ckm=0.2243,           # |V_us|
        f_gev=0.1557,         # f_K
        sources={"mass_gev": _PDG, "ctau_m": _PDG + " (derived: c * tau)", "ckm": _PDG,
                 "f_gev": _PDG + " (FLAG N_f=2+1+1 average)",
                 "width_gev": "derived: hbar / tau"}),
    "D": Parent(
        "D", 411, "pseudoscalar", 1.86966, _tau(1.033e-12), _width(1.033e-12),
        ckm=0.221,            # |V_cd|
        f_gev=0.212,          # f_D
        sources={"mass_gev": _PDG, "ctau_m": _PDG + " (derived: c * tau)", "ckm": _PDG,
                 "f_gev": _PDG + " (FLAG N_f=2+1+1 average)",
                 "width_gev": "derived: hbar / tau"}),
    "Ds": Parent(
        "Ds", 431, "pseudoscalar", 1.96835, _tau(5.04e-13), _width(5.04e-13),
        ckm=0.975,            # |V_cs|
        f_gev=0.2499,         # f_Ds
        sources={"mass_gev": _PDG, "ctau_m": _PDG + " (derived: c * tau)", "ckm": _PDG,
                 "f_gev": _PDG + " (FLAG N_f=2+1+1 average)",
                 "width_gev": "derived: hbar / tau"}),

    # ---- vector: V -> mu+ mu- phi --------------------------------------- #
    # Prompt (strong / electromagnetic), so c*tau = 0: the LLP is produced at
    # the parent's own production point.
    "rho": Parent(
        "rho", 113, "vector", 0.77526, 0.0, 1.474e-3,
        gamma_ll_gev=4.55e-5 * 1.474e-3,      # B(rho -> mu mu) * Gamma_tot
        sources={"mass_gev": _PDG, "width_gev": _PDG,
                 "gamma_ll_gev": _PDG + " (B(rho->mumu) x Gamma_tot)",
                 "ctau_m": "prompt (strong decay)"}),
    "omega": Parent(
        "omega", 223, "vector", 0.78266, 0.0, 8.68e-3,
        gamma_ll_gev=7.4e-5 * 8.68e-3,
        sources={"mass_gev": _PDG, "width_gev": _PDG,
                 "gamma_ll_gev": _PDG + " (B(omega->mumu) x Gamma_tot)",
                 "ctau_m": "prompt (strong decay)"}),
    "phi": Parent(
        "phi", 333, "vector", 1.019461, 0.0, 4.249e-3,
        gamma_ll_gev=2.86e-4 * 4.249e-3,
        sources={"mass_gev": _PDG, "width_gev": _PDG,
                 "gamma_ll_gev": _PDG + " (B(phi->mumu) x Gamma_tot)",
                 "ctau_m": "prompt (strong decay)"}),
    "Jpsi": Parent(
        "Jpsi", 443, "vector", 3.096900, 0.0, 9.26e-5,
        gamma_ll_gev=5.961e-2 * 9.26e-5,
        sources={"mass_gev": _PDG, "width_gev": _PDG,
                 "gamma_ll_gev": _PDG + " (B(Jpsi->mumu) x Gamma_tot)",
                 "ctau_m": "prompt (electromagnetic decay)"}),
    "psi2S": Parent(
        "psi2S", 100443, "vector", 3.68610, 0.0, 2.94e-4,
        gamma_ll_gev=8.0e-3 * 2.94e-4,
        sources={"mass_gev": _PDG, "width_gev": _PDG,
                 "gamma_ll_gev": _PDG + " (B(psi2S->mumu) x Gamma_tot)",
                 "ctau_m": "prompt (electromagnetic decay)"}),
}

PSEUDOSCALARS = tuple(k for k, p in PARENTS.items()
                      if p.family == "pseudoscalar")
VECTORS = tuple(k for k, p in PARENTS.items() if p.family == "vector")


def get(name):
    """Look up a hosted parent, with a helpful error listing the options."""
    try:
        return PARENTS[name]
    except KeyError:
        raise KeyError(
            f"unknown parent {name!r}; hosted parents are "
            f"{sorted(PARENTS)} (pseudoscalar {list(PSEUDOSCALARS)}, "
            f"vector {list(VECTORS)})")


def kinematic_limit(name, lepton="mu"):
    """Largest LLP mass this channel can produce [GeV].

    Pseudoscalar P -> l nu phi closes at m_P - m_l (the neutrino is massless).
    Vector V -> l+ l- phi closes at m_V - 2 m_l.
    """
    p = get(name)
    ml = LEPTON_MASS_GEV[lepton]
    return p.mass_gev - (ml if p.family == "pseudoscalar" else 2.0 * ml)


def is_open(name, m_phi_gev, lepton="mu"):
    """Is this channel kinematically open for the given LLP mass?"""
    return float(m_phi_gev) < kinematic_limit(name, lepton)
