"""
# production_spectrum.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .spectra import kinematics as kin
from .spectra import parents as sm
from .spectra import vertices as vtx

SCHEMA_VERSION = "llpspectrum-1.0"
MANIFEST_SCHEMA_VERSION = "llpspectrum-manifest-1.0"


class ProductionSpectrumTool(BaseTool):
    """
    Compute the parent-rest-frame LLP energy spectrum and the reduced
    branching fraction for a Standard-Model hadron decaying to a light new
    scalar. Step 0 of the reach chain
    `PRODUCTION-SPECTRUM -> pythia -> harvest -> meson-decay -> DecayInVolume`.

    Ask for physics, not implementation: name a hosted parent and an LLP mass
    and the tool returns everything the downstream chain needs for that
    channel. You do NOT supply a matrix element, decay constants, CKM
    elements, meson masses or lifetimes -- those are hosted and cited.

    WHAT COMES BACK, per (parent, mass):

      f_h(x)   the normalised energy spectrum, x = 2 E*_LLP / m_h, written as
               a two-column CSV (x, pdf) that feeds straight into
               MesonDecayToLLPTool's `spectrum_spec` with no conversion; and
      B_hat    the REDUCED branching fraction, i.e. Br(h -> l nu LLP) with the
               coupling factored out, so the physical branching at coupling g
               is B_h(g) = g^2 * B_hat_h. This is the `B_hat` that
               MesonDecayToLLPTool wants.

    Both come from the SAME integral, which is why they are returned together:
    supplying a spectrum and a branching fraction from different sources is a
    silent way to get a wrong normalisation. The written bundle records the
    parent and mass it was computed for, and MesonDecayToLLPTool reads B_hat
    from it and refuses a mismatch, so the pair cannot drift apart.

    HOSTED PARENTS
      pseudoscalar  h -> l nu LLP    pi, K, D, Ds
      vector        V -> l+ l- LLP   rho, omega, phi, Jpsi, psi2S
    Both families are implemented. A channel is open while
    m_LLP < m_h - m_l (pseudoscalar) or m_h - 2 m_l (vector); a closed channel
    returns ok with `open: false` and no file, never an error, so a mass grid
    can span the threshold.

    The vector channels matter more than their multiplicity suggests: above
    the pseudoscalar wall (m_LLP ~ 1.76 GeV) the charmonia are the ONLY open
    production channel, so they carry the entire high-mass reach on their own.

    ACCURACY. The Dalitz integral is done by Gauss-Legendre in both directions
    with the substitution x = x_min + t^2, which absorbs the square-root edge
    the phase-space Jacobian carries at x_min. Defaults (n_x=32, n_c=64) were
    fixed by a convergence study over 28 (parent, mass) cases spanning
    0.5%-99.5% of threshold: worst-case relative error 3.8e-6 at 2048
    integrand evaluations. You should not normally need to touch them; raise
    n_x first if you do, since the energy direction is the binding one.

    Grid mode (preferred): pass `masses` + `output_dir` to sweep a whole mass
    grid for one parent in a single call, one CSV per mass plus a manifest.

    Every constant used is emitted in the `provenance` block with its source,
    together with the amplitude reference, so a result can be audited without
    reading the source.
    """

    parent: str = RuntimeField(
        default="",
        description="SM parent hadron. Pseudoscalars pi, K, D, Ds "
                    "(h -> l nu LLP) and vectors rho, omega, phi, Jpsi, psi2S "
                    "(V -> l+ l- LLP) are all supported. Charge conjugates are "
                    "folded, matching the harvested forward flux. Above the "
                    "pseudoscalar kinematic wall the charmonia are the only "
                    "open channel, so Jpsi/psi2S carry the whole high-mass "
                    "reach.")
    m_phi_gev: float = RuntimeField(
        default=0.0,
        description="LLP mass in GeV (single-mass mode; ignored when `masses` "
                    "is given)")
    masses: List[float] = RuntimeField(
        default=[],
        description="GRID MODE (preferred): list of LLP masses in GeV to sweep "
                    "for this parent in one call. Writes one CSV per OPEN mass "
                    "into `output_dir` plus a manifest; kinematically closed "
                    "masses are reported as closed rather than failing.")
    lepton: str = RuntimeField(
        default="mu",
        description="Charged lepton the LLP is radiated from: mu, e or tau. "
                    "Sets the lepton mass and hence the kinematic threshold.")
    interaction: str = RuntimeField(
        default="scalar",
        description="LLP-lepton interaction. 'scalar' (L = -g phi lbar l) is "
                    "implemented and validated. The vertex registry is "
                    "extensible; an unimplemented choice reports which are "
                    "available rather than guessing.")
    n_x: int = RuntimeField(
        default=kin.N_X_DEFAULT,
        description=f"Gauss-Legendre nodes in the LLP energy fraction x "
                    f"(default {kin.N_X_DEFAULT}). The binding direction; "
                    f"raise this first if you need more accuracy.")
    n_c: int = RuntimeField(
        default=kin.N_C_DEFAULT,
        description=f"Gauss-Legendre nodes in the lepton helicity angle "
                    f"(default {kin.N_C_DEFAULT}). Saturates well before n_x.")
    n_table: int = RuntimeField(
        default=256,
        description="Rows written in the f(x) CSV. The spectrum is evaluated "
                    "on the quadrature nodes and resampled onto this uniform x "
                    "grid for the table; 256 is ample for inverse-CDF "
                    "sampling downstream.")
    output_dir: str = RuntimeField(
        default="",
        description="GRID MODE: relative directory for the per-mass CSVs and "
                    "the manifest. Created if it does not exist.")
    output_path: str = RuntimeField(
        default="",
        description="SINGLE-mass mode: relative path for the spectrum CSV. "
                    "Parent directories are created if needed.")
    base_directory: str = StateField(
        default=".", description="Base directory for safe paths")

    # ------------------------------------------------------------------ #
    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    # ------------------------------------------------------------------ #
    def _provenance(self, parent, lepton, interaction):
        p = sm.get(parent)
        used = {
            "m_h_gev": [p.mass_gev, p.sources.get("mass_gev")],
            "Gamma_h_tot_gev": [p.width_gev, p.sources.get("width_gev")],
            "ctau_h_m": [p.ctau_m, p.sources.get("ctau_m")],
            f"m_{lepton}_gev": [sm.LEPTON_MASS_GEV[lepton], "PDG2024"],
        }
        if p.family == "pseudoscalar":
            used["G_F_gev^-2"] = [sm.G_F_GEV2, "PDG2024"]
            used["|V_ij|"] = [p.ckm, p.sources.get("ckm")]
            used["f_h_gev"] = [p.f_gev, p.sources.get("f_gev")]
            amp = "CarlsonRislow2012"
        else:
            used["Gamma_V_to_ll_gev"] = [p.gamma_ll_gev,
                                         p.sources.get("gamma_ll_gev")]
            amp = "MitraSahoo2021"
        return {
            "constants": used,
            "amplitude_reference": sm.REFERENCES[amp],
            "application_reference": sm.REFERENCES["AbrahamFieg2025"],
            "data_source": sm.REFERENCES["PDG2024"],
            "modelling_assumptions": [
                "unpolarised parent ensemble; LLP emission isotropic in the "
                "parent rest frame after the recoil system is integrated out",
                "recoil leptons integrated out; only the LLP four-vector is "
                "retained",
                "charge conjugates folded into one parent label",
            ],
        }

    def _coupling_sq(self, p, ml):
        """Overall factor multiplying the unit-coupling squared amplitude.

        PSEUDOSCALAR: fixed by electroweak theory, C_P^2 = (G_F |V| f / sqrt2)^2.

        VECTOR: there is no decay constant to appeal to, so the effective
        coupling of V to the lepton current is fixed EMPIRICALLY, by requiring
        the same amplitude machinery to reproduce the MEASURED V -> l+ l-
        partial width:

            g_V^2 = Gamma_meas(V -> l+ l-) / Gamma_unit(V -> l+ l-)

        Both sides use this module's own trace and polarisation average, so the
        normalisation cancels any convention choice rather than depending on
        one. That is also what makes it testable: `Gamma_unit` is checked
        against the closed form M/(12 pi) (1 + 2m^2/M^2) sqrt(1 - 4m^2/M^2).
        """
        if p.family == "pseudoscalar":
            return (sm.G_F_GEV2 * p.ckm * p.f_gev / np.sqrt(2.0)) ** 2
        if p.family == "vector":
            if not p.gamma_ll_gev or p.gamma_ll_gev <= 0.0:
                raise ValueError(
                    f"vector parent {p.name!r} has no measured V -> l+ l- "
                    f"width; it is required to fix the coupling")
            unit = vtx.gamma_v_to_ll_unit(p.mass_gev, ml)
            if unit <= 0.0:
                raise ValueError(
                    f"V -> l+ l- is closed for {p.name!r} at m_l={ml}")
            return p.gamma_ll_gev / unit
        raise NotImplementedError(
            f"no normalisation for parent family {p.family!r}")

    def _one_mass(self, p, m_phi, msq, ml, out_file):
        """Compute and write one spectrum; returns the summary record."""
        if not sm.is_open(p.name, m_phi, self.lepton):
            return {"m_phi_gev": m_phi, "open": False,
                    "reason": f"closed: limit is "
                              f"{sm.kinematic_limit(p.name, self.lepton):.6f} GeV"}
        # The recoil partner is a massless neutrino for a pseudoscalar and the
        # second CHARGED lepton for a vector; that mass enters both the Dalitz
        # boundary and the recoil split.
        ml2 = ml if p.family == "vector" else 0.0
        gamma, xs, fx = kin.width_and_spectrum(
            p.mass_gev, ml, m_phi, msq, n_x=int(self.n_x), n_c=int(self.n_c),
            m_l2=ml2)
        if gamma <= 0.0 or xs.size == 0:
            return {"m_phi_gev": m_phi, "open": False,
                    "reason": "vanishing phase space"}
        gamma *= self._coupling_sq(p, ml)
        b_hat = gamma / p.width_gev

        # resample the spectrum onto a uniform grid for the CSV
        x_lo, x_hi = kin.x_domain(p.mass_gev, ml, m_phi, ml2)
        grid = np.linspace(x_lo, x_hi, int(self.n_table))
        pdf = np.interp(grid, xs, fx, left=0.0, right=0.0)

        # NOTE plain float() before formatting: numpy 2.x repr() of a scalar
        # is "np.float64(0.24)", which is not a number to any CSV reader --
        # including this toolkit's own downstream spectrum loader. %.17g
        # round-trips a double exactly.
        def _f(v):
            return f"{float(v):.17g}"

        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w") as fh:
            fh.write(f"# {SCHEMA_VERSION}\n")
            fh.write(f"# parent={p.name} lepton={self.lepton} "
                     f"interaction={self.interaction} "
                     f"m_phi_gev={_f(m_phi)}\n")
            # B_hat travels WITH the spectrum: they come from one integral, and
            # the downstream sampler reads it from here rather than having the
            # caller carry it by hand and risk pairing a spectrum with the
            # wrong normalisation.
            fh.write(f"# B_hat={_f(b_hat)}\n")
            fh.write(f"# f_h(x) normalised to 1 over [{_f(x_lo)}, "
                     f"{_f(x_hi)}]; x = 2 E*_LLP / m_h\n")
            fh.write("x,pdf\n")
            for xv, pv in zip(grid, pdf):
                fh.write(f"{_f(xv)},{_f(pv)}\n")
        return {"m_phi_gev": m_phi, "open": True,
                "B_hat": b_hat,
                "Gamma_h_to_llp_gev_at_g1": gamma,
                "x_min": float(x_lo), "x_max": float(x_hi),
                "spectrum_path": os.path.relpath(out_file,
                                                 self.base_directory),
                "n_table": int(self.n_table)}

    # ------------------------------------------------------------------ #
    def _run(self) -> str:
        name = (self.parent or "").strip()
        if not name:
            return self.format_error(
                error="Invalid Parameter", reason="no `parent` given",
                suggestion=f"Pick a hosted parent: {sorted(sm.PARENTS)}")
        try:
            p = sm.get(name)
        except KeyError as e:
            return self.format_error(
                error="Unknown Parent", reason=str(e),
                suggestion="Use one of the hosted parents; charge conjugates "
                           "are folded (K covers K+ and K-).")
        if self.lepton not in sm.LEPTON_MASS_GEV:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"unknown lepton {self.lepton!r}",
                suggestion=f"One of {sorted(sm.LEPTON_MASS_GEV)}")
        try:
            msq = vtx.get(p.family, self.interaction)
        except KeyError as e:
            return self.format_error(
                error="Unsupported Interaction", reason=str(e),
                suggestion="Use interaction='scalar', or add the vertex to "
                           "spectra/vertices.py.")

        ml = sm.LEPTON_MASS_GEV[self.lepton]
        grid_mode = bool(self.masses)
        if grid_mode:
            out_dir = self._safe_path(self.output_dir)
            if out_dir is None:
                return self.format_error(
                    error="Invalid Path",
                    reason="grid mode needs `output_dir` inside the base "
                           "directory (traversal denied)",
                    suggestion="Pass a relative output_dir, e.g. 'spectra'.")
            os.makedirs(out_dir, exist_ok=True)
            targets = [(float(m),
                        os.path.join(out_dir,
                                     f"spectrum_{p.name}_m{float(m):.4f}.csv"))
                       for m in self.masses]
        else:
            if float(self.m_phi_gev) <= 0.0:
                return self.format_error(
                    error="Invalid Parameter",
                    reason=f"need m_phi_gev > 0 (got {self.m_phi_gev})",
                    suggestion="Pass m_phi_gev, or `masses` for grid mode.")
            dst = self._safe_path(self.output_path)
            if dst is None:
                return self.format_error(
                    error="Invalid Path",
                    reason="single-mass mode needs `output_path` inside the "
                           "base directory (traversal denied)",
                    suggestion="Pass a relative output_path, e.g. "
                               "'spectra/spec_K_m0.060.csv'.")
            targets = [(float(self.m_phi_gev), dst)]

        try:
            rows = [self._one_mass(p, m, msq, ml, f) for m, f in targets]
        except Exception as e:                      # pragma: no cover
            return self.format_error(
                error="Computation Error", reason=f"{type(e).__name__}: {e}",
                suggestion="Check the parent/mass/lepton combination.")

        result: Dict[str, Any] = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "parent": p.name,
            "parent_pdg": p.pdg,
            "parent_family": p.family,
            "lepton": self.lepton,
            "interaction": self.interaction,
            "kinematic_limit_gev": sm.kinematic_limit(p.name, self.lepton),
            "quadrature": {"n_x": int(self.n_x), "n_c": int(self.n_c),
                           "scheme": "gauss-legendre; x = x_min + t^2"},
            "convention": ("B_h(g) = g^2 * B_hat ; f_h(x) normalised to 1 ; "
                           "x = 2 E*_LLP / m_h"),
            "masses": rows,
            "provenance": self._provenance(p.name, self.lepton,
                                           self.interaction),
        }
        if grid_mode:
            man = os.path.join(self._safe_path(self.output_dir),
                               "production_spectrum_manifest.json")
            payload = dict(result)
            payload["schema"] = MANIFEST_SCHEMA_VERSION
            with open(man, "w") as fh:
                json.dump(payload, fh, indent=2)
            result["manifest_path"] = os.path.relpath(man,
                                                      self.base_directory)
            result["n_open"] = sum(1 for r in rows if r.get("open"))
            result["n_closed"] = sum(1 for r in rows if not r.get("open"))
        return json.dumps(result, indent=2)
