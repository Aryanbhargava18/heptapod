"""
# decay_in_volume.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import glob
import json
import os
from typing import Dict, List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

YIELDS_SCHEMA_VERSION = "llpyields-1.0"
AUDIT_SCHEMA_VERSION = "llpaudit-1.0"
CONVENTIONS = {
    "acceptance": "midpoint",
    "weight": "g2_stripped",
    "propagation": "straight_line_from_production_vertex",
}
_ACCEPTANCE_MODES = ("two_track", "photon", "none")
# Below this many events passing geometry+acceptance, the yield sum is a noisy
# Monte Carlo estimate (~1/sqrt(N) relative error) and the reach is unreliable.
_MIN_EFFECTIVE_EVENTS = 50
# Per-event audit sample cap in the default 'summary' mode. The audit is for
# spot-checking a handful of events, not re-deriving the whole yield; a full
# per-event dump of a multi-million-event flux is ~GB and has exhausted disk
# mid-sweep. 'full' overrides the cap; 'none' writes an empty audit.
_AUDIT_SUMMARY_CAP = 1000

# numpy / yaml-backed physics is deferred to _run() so module import does
# not pull in numpy's cold load (heptapod toolkit startup convention).


class _DecayInVolumeBase(BaseTool):
    """Shared decay-in-volume engine for the two concrete yield tools.

    NOT a registered tool (has no lifetime axis of its own): the concrete tools
    `DecayInVolumeVsCouplingTool` and `DecayInVolumeVsLifetimeTool` subclass it
    and differ ONLY in how the lifetime axis is parameterized (via
    `_lifetime_spec`). All the convention-sensitive physics -- geometry
    intersection, decay probability, midpoint acceptance, audit, output -- lives
    here and is identical across both.
    """
    # ------------------- Runtime fields (shared, all modes) --------------- #
    events_path: str = RuntimeField(
        default="",
        description="EITHER this or `events_paths` (not both required). Path to "
                    "line-delimited JSON LLP records with E, px, py, "
                    "pz, the production vertex vx, vy, vz (absent => the IP) "
                    "and event_weight_g2_stripped (concatenation of multiple "
                    "parent channels is fine). The result echoes n_events and "
                    "sum_weights of the input, so check them against your full "
                    "production to confirm no parent channel was dropped")
    events_paths: List[str] = RuntimeField(
        default=[],
        description="Alternative to `events_path`: SEVERAL LLP record files to "
                    "treat as one sample, e.g. one per parent channel from "
                    "MesonDecayToLLP at the same mass "
                    "(['results/llp_K_m0.28.jsonl', 'results/llp_D_m0.28.jsonl', "
                    "...]). The tool reads them in order and concatenates in "
                    "memory, so a full-spectrum yield needs no hand-written "
                    "concatenation step and no combined file on disk. Reading "
                    "rather than appending keeps this idempotent: re-running a "
                    "channel cannot double-count its flux, which would silently "
                    "inflate every yield downstream. Takes precedence over "
                    "`events_path` when non-empty; the per-file record counts "
                    "are echoed as `inputs` so a dropped channel is visible.")
    events_paths_glob: str = RuntimeField(
        default="",
        description="Alternative to `events_paths`: ONE glob matching the "
                    "record files, e.g. 'llp/*/llp_*_m0.2300.jsonl'. Resolved "
                    "relative to base_directory and sorted, so a per-parent "
                    "sweep needs no hand-assembled list -- which is where a "
                    "channel gets dropped or duplicated. Matches are echoed as "
                    "`inputs` with their record counts; an empty match is an "
                    "error rather than a silently empty sample. Ignored when "
                    "`events_paths` is non-empty.")
    geometry_path: str = RuntimeField(
        description="Path to geometry YAML (z_min_m, z_max_m, r_volume_m, "
                    "z_det_m, r_det_m in meters; optional z_shield_m = SHIELD "
                    "FRONT, i.e. the LLP production vertex must satisfy "
                    "z < z_shield_m, "
                    "production region and x_off_m detector offset)")
    m_phi_gev: float = RuntimeField(
        description="LLP (phi) mass in GeV")
    daughter_masses_gev: List[float] = RuntimeField(
        default=[0.1056584, 0.1056584],
        description="The two visible-decay daughter masses in GeV "
                    "(length 2; default [0.1056584, 0.1056584] = mu mu). "
                    "Used by the two-track acceptance")
    n_int: float = RuntimeField(
        description="Number of primary interactions (sigma_inel * L_int "
                    "for a collider, N_POT for a beam dump)")
    br_visible: float = RuntimeField(
        default=1.0,
        description="Branching ratio into THIS search's detected final state (the "
                    "two daughters) at THIS mass, in [0, 1]. The yield is scaled by "
                    "it, while the lifetime comes from the TOTAL width "
                    "(width_ref_gev). Use BR(phi -> mu mu) for a two_track search "
                    "and BR(phi -> gamma gamma) for a photon search; below 2 m_mu "
                    "the mu mu channel is closed so BR(gamma gamma) ~ 1. Default "
                    "1.0 (this search's channel is the only visible mode).")
    n_target: float = RuntimeField(
        default=0.0,
        description="Optional sensitivity threshold N* (the reach criterion, "
                    "e.g. 3). If given, the tool checks whether the reach band "
                    "(N_sig >= N*) runs off either end of your scanned grid and "
                    "warns you to widen it; leave 0 to skip that check.")
    acceptance: str = RuntimeField(
        default="two_track",
        description="Detector acceptance model, evaluated ONCE at the in-volume "
                    "decay midpoint (so it is lifetime-independent and reused "
                    "across the whole grid): 'two_track' (isotropic phi -> d1 d2, "
                    "both daughters reach z_det within r_det; default), 'photon' "
                    "(the LLP line-of-flight reaches the detector face within "
                    "r_det; collinear light daughters), or 'none' (geometry only)")
    seed: int = RuntimeField(
        default=1,
        description="RNG seed for the isotropic phi -> d1 d2 decay "
                    "sampling (default 1)")
    audit_verbosity: str = RuntimeField(
        default="summary",
        description="Size of the per-event audit_path file: 'summary' (default, "
                    "a sample of up to ~1000 events -- enough to spot-check), "
                    "'full' (every event; a multi-million-event flux writes a "
                    "multi-GB file, so use only for deep debugging), or 'none' "
                    "(empty audit). The yields/counts in the result are the same "
                    "in all modes; only the per-event dump size changes.")
    response_verbosity: str = RuntimeField(
        default="summary",
        description="Size of the RETURNED result JSON: 'summary' (default) omits "
                    "the full per-point yields array from the response, returning "
                    "the peak, band edges and point count via grid_diagnostic -- "
                    "the complete table is always written to output_path; 'full' "
                    "also inlines the whole yields array. Use 'summary' for dense "
                    "scans to avoid a large, token-heavy payload; output_path is "
                    "identical in both modes.")
    output_path: str = RuntimeField(
        description="Relative path for the yields table JSON (e.g. "
                    "'yields/m0p25.json')")
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(default=".", description="Base directory for safe paths")
    # ---------------------------------------------------------------------- #

    def _resolve_br_visible(self):
        """Visible-channel branching ratio. Overridable: a subclass that is
        given the LLP's partial widths can DERIVE this instead of trusting a
        second, independently-supplied scalar that must be kept consistent with
        the total width by hand."""
        return float(self.br_visible)

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _lifetime_spec(self, phys):
        """Subclass hook: validate this tool's lifetime-axis fields and return
        `(points, wref, lifetime_mode)`, or a `format_error(...)` string on a
        bad axis. `points` is a list of
        `(label_key, label_value, ctau_m, prefactor)` -- one grid point each,
        with `prefactor` the g^2 production factor (portal) or 1.0 (direct)."""
        raise NotImplementedError

    def _run(self) -> str:
        """Validate inputs, then compute N_sig over the lifetime axis supplied
        by `_lifetime_spec` and write the yields + audit files."""
        import numpy as np

        from . import llp_physics as phys

        rel_srcs = [str(x) for x in (self.events_paths or []) if str(x).strip()]
        if not rel_srcs and str(self.events_paths_glob).strip():
            # Resolve inside base_directory and sort, so the sample is
            # reproducible and a per-parent sweep needs no hand-built list.
            pattern = os.path.join(self.base_directory,
                                   str(self.events_paths_glob).strip())
            hits = sorted(glob.glob(pattern))
            if not hits:
                return self.format_error(
                    error="File Not Found",
                    reason=f"events_paths_glob matched no files: "
                           f"{self.events_paths_glob}",
                    suggestion="Check the pattern; an empty match is refused "
                               "rather than run as an empty sample")
            rel_srcs = [os.path.relpath(h, self.base_directory) for h in hits]
        if not rel_srcs:
            rel_srcs = [self.events_path] if self.events_path else []
        if not rel_srcs:
            return self.format_error(
                error="Invalid Parameter",
                reason="no input given: events_path, events_paths and "
                       "events_paths_glob are all empty",
                suggestion="Set events_path, or events_paths / "
                           "events_paths_glob for several parent-channel "
                           "files at the same mass")
        srcs = [self._safe_path(r) for r in rel_srcs]
        geo = self._safe_path(self.geometry_path)
        dst = self._safe_path(self.output_path)
        if any(x is None for x in srcs) or not geo or not dst:
            return self.format_error(
                error="Access Denied",
                reason="events_path(s), geometry_path or output_path escapes "
                       "base_directory",
                suggestion="Use relative paths inside base_directory")
        for path, label in list(zip(srcs, rel_srcs)) + [(geo, "geometry_path")]:
            if not os.path.exists(path):
                return self.format_error(
                    error="File Not Found",
                    reason=f"{label} not found",
                    context=f"path={label}",
                    suggestion=f"Provide a valid path for {label}")
        src = srcs[0]

        # --------------------- validate parameters --------------------- #
        m_phi = float(self.m_phi_gev)
        dmasses = [float(m) for m in (self.daughter_masses_gev or [])]
        n_int = float(self.n_int)
        br_vis = self._resolve_br_visible()
        if isinstance(br_vis, str):      # a format_error(...) from the hook
            return br_vis
        if m_phi <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"m_phi must be positive (m_phi={m_phi})",
                suggestion="Provide m_phi_gev > 0")
        if not (0.0 <= br_vis <= 1.0):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"br_visible must be in [0, 1] (got {br_vis})",
                suggestion="br_visible is the visible-channel branching ratio")
        if len(dmasses) != 2 or any(m < 0.0 for m in dmasses):
            return self.format_error(
                error="Invalid Parameter",
                reason=f"daughter_masses_gev must be two non-negative "
                       f"masses (got {self.daughter_masses_gev})",
                suggestion="Provide e.g. [0.1056584, 0.1056584] for mu mu")
        m1, m2 = dmasses
        if n_int <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"n_int must be positive (got {n_int})",
                suggestion="n_int is the number of primary interactions")

        # ---------------- lifetime axis (mode-specific) ---------------- #
        spec = self._lifetime_spec(phys)
        if isinstance(spec, str):   # a format_error(...) from the subclass
            return spec
        points, wref, lifetime_mode = spec

        acc_mode = str(self.acceptance or "two_track").lower()
        if acc_mode not in _ACCEPTANCE_MODES:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"acceptance must be one of {_ACCEPTANCE_MODES} "
                       f"(got {self.acceptance!r})",
                suggestion="Use 'two_track', 'photon', or 'none'")
        if acc_mode == "two_track" and m_phi <= m1 + m2:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"two-track acceptance requires m_phi > m1 + m2 "
                       f"(m_phi={m_phi}, m1+m2={m1 + m2})",
                suggestion="phi -> d1 d2 is kinematically closed; check "
                           "m_phi_gev / daughter_masses_gev or use "
                           "acceptance='photon' / 'none'")
        # Fail loud on the silent-zero misconfig: a visible search with zero
        # visible branching returns N_sig = 0 everywhere, which reads as "no
        # sensitivity" when it is really a wrong br_visible (the classic error
        # is passing BR(phi->mu mu)=0 below the dimuon threshold to the photon
        # search, where the open channel is phi->gamma gamma).
        if acc_mode != "none" and br_vis == 0.0:
            return self.format_error(
                error="Zero visible branching",
                reason=f"br_visible = 0 with a visible '{acc_mode}' search makes "
                       f"every N_sig identically zero.",
                suggestion="br_visible is the branching into the DETECTED "
                           "channel at THIS mass: use BR(phi -> mu mu) for the "
                           "two-track search (it vanishes below 2 m_mu) and "
                           "BR(phi -> gamma gamma) for the photon search. Set "
                           "the channel that is open here; it must be > 0.")

        # -------------------------- load inputs ------------------------ #
        try:
            geom = phys.DecayVolume.from_yaml(geo)
        except Exception as e:
            return self.format_error(
                error="Geometry Error",
                reason=str(e),
                context=f"path={self.geometry_path}",
                suggestion="Geometry YAML must declare z_min_m, z_max_m, "
                           "r_volume_m, z_det_m, r_det_m")
        try:
            p4_list, vtx_list, w_list, channels = [], [], [], []
            input_counts = []
            for _src, _rel in zip(srcs, rel_srcs):
              n_before = len(p4_list)
              with open(_src) as fh:
                for ln, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p4_list.append([rec["E"], rec["px"], rec["py"],
                                    rec["pz"]])
                    # production vertex; absent => the IP (prompt-at-origin)
                    vtx_list.append([rec.get("vx", 0.0), rec.get("vy", 0.0),
                                     rec.get("vz", 0.0)])
                    w_list.append(rec["event_weight_g2_stripped"])
                    channels.append(rec.get("parent_channel"))
              input_counts.append({"path": _rel,
                                   "n_records": len(p4_list) - n_before})
        except KeyError as e:
            return self.format_error(
                error="Event Format Error",
                reason=f"event record at line {ln + 1} is missing field {e}",
                context=f"path={self.events_path}",
                suggestion="Records need E, px, py, pz and "
                           "event_weight_g2_stripped (the "
                           "MesonDecayToLLPTool output format)")
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={self.events_path}",
                suggestion="Verify the events file is line-delimited JSON")

        base, _ = os.path.splitext(dst)
        audit_path = base + ".audit.jsonl"
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        label_key = points[0][0]
        ctau_ref = points[0][2]
        pref_ref = points[0][3]
        n_events = len(p4_list)

        # ---------------- empty input: zero yields, not an error ------- #
        if n_events == 0:
            yields = [{label_key: val, "n_sig": 0.0}
                      for (_, val, _, _) in points]
            with open(audit_path, "w") as fh:
                pass
            return self._write_outputs(dst, audit_path, yields, 0, 0, 0,
                                       m_phi, dmasses, wref, n_int,
                                       lifetime_mode, acc_mode, 0.0,
                                       offscale_ctaus=[pt[2] for pt in points],
                                       offscale_vol_scale=getattr(geom, "z_max", None),
                                       br_vis=br_vis)

        # --------------------------- compute --------------------------- #
        try:
            p4 = np.asarray(p4_list, dtype=float)
            vtx = np.asarray(vtx_list, dtype=float)
            w = np.asarray(w_list, dtype=float)
            sum_w = float(np.sum(w))
            pmag = np.linalg.norm(p4[:, 1:], axis=1)
            dirs = p4[:, 1:] / np.maximum(pmag, 1e-300)[:, None]
            beta_gamma = pmag / m_phi
            # in-volume segment measured FROM the production vertex, along
            # the LLP direction (handles decay-in-flight + off-axis + z_shield)
            L1, L2, ok = geom.segment_from_vertex(vtx, dirs)
            # in-volume decay MIDPOINT for the lifetime-independent acceptance
            Lmid = 0.5 * (L1 + L2)
            decay_point = vtx + Lmid[:, None] * dirs
            if acc_mode == "two_track":
                rng = np.random.default_rng(int(self.seed))
                tt = phys.two_track_pass(p4, decay_point, m_phi, m1, m2,
                                         geom, rng)
            elif acc_mode == "photon":
                tt = phys.photon_pass(p4, decay_point, geom)
            else:  # "none": geometry only
                tt = None
            keep = ok & tt if tt is not None else ok
            # exact lifetime-grid reweighting (one event set covers all
            # points; the prefactor is g^2 in portal mode, 1.0 in ctau)
            yields = []
            for lkey, lval, ctau, pref in points:
                lam = beta_gamma[keep] * ctau
                pdec = phys.decay_probability(L1[keep], L2[keep], lam)
                n_sig = n_int * pref * br_vis * float(np.sum(w[keep] * pdec))
                yields.append({lkey: lval, "n_sig": n_sig})
            # per-event audit columns at the first grid point
            pdec_ref = phys.decay_probability(
                L1, L2, beta_gamma * ctau_ref)
            contrib_ref = n_int * pref_ref * w * pdec_ref \
                * keep.astype(float)
            # Audit is a spot-check sample, not a re-derivation: cap per-event
            # rows so a multi-million-event flux does not write a multi-GB file
            # (which has exhausted disk mid-sweep). 'full' dumps everything.
            _av = str(self.audit_verbosity or "summary").lower()
            n_audit = (0 if _av == "none"
                       else n_events if _av == "full"
                       else min(n_events, _AUDIT_SUMMARY_CAP))
            with open(audit_path, "w") as fh:
                for i in range(n_audit):
                    rec = {
                        "schema": AUDIT_SCHEMA_VERSION,
                        "event_id": i,
                        "parent_channel": channels[i],
                        "event_weight_g2_stripped": float(w[i]),
                        "beta_gamma": float(beta_gamma[i]),
                        "vz_prod_m": float(vtx[i, 2]),
                        "L1_m": float(L1[i]),
                        "L2_m": float(L2[i]),
                        "geom_pass": bool(ok[i]),
                        "acc_pass": (bool(tt[i]) if tt is not None
                                     else None),
                        "p_decay_at_ref": float(pdec_ref[i]),
                        "weighted_contribution": float(contrib_ref[i]),
                    }
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_geo = int(np.sum(ok))
            n_tt = int(np.sum(keep))
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e),
                context=f"events={self.events_path}, "
                        f"geometry={self.geometry_path}",
                suggestion="Check event four-momenta and geometry values")

        return self._write_outputs(dst, audit_path, yields, n_events,
                                   n_geo, n_tt, m_phi, dmasses, wref, n_int,
                                   lifetime_mode, acc_mode, sum_w,
                                   offscale_ctaus=[pt[2] for pt in points],
                                   offscale_vol_scale=getattr(geom, "z_max", None),
                                   br_vis=br_vis, inputs=input_counts)

    def _grid_diagnostic(self, yields, n_target):
        """Structured grid-boundary diagnostic + a note. Returns
        (diag_dict, note_or_None). The diagnostic reports where the peak sits
        and the N_sig at each grid end; the note fires when the reach band is
        TRUNCATED by the grid -- i.e. N_sig is still >= N* at an end (band runs
        off), or, when N* is unknown, when the peak sits at an end. This catches
        the classic wrong-width / too-narrow-grid case where the reported reach
        is silently clipped to the scanned range. When N* is known and the band
        sits inside the grid, `band_bracketed` is True and no note fires even if
        the peak is at an edge -- so an intentional zoom/refinement is not
        mistaken for a truncated scan."""
        if not yields:
            return None, None
        var = "g" if "g" in yields[0] else ("ctau_m" if "ctau_m" in yields[0]
                                            else None)
        if var is None:
            return None, None
        xs = [float(y[var]) for y in yields]
        ns = [float(y["n_sig"]) for y in yields]
        if len(ns) < 2 or max(ns) <= 0.0:
            return None, None
        i_lo = min(range(len(xs)), key=lambda i: xs[i])
        i_hi = max(range(len(xs)), key=lambda i: xs[i])
        imax = max(range(len(ns)), key=lambda i: ns[i])
        diag = {
            "var": var,
            f"{var}_min": xs[i_lo], f"{var}_max": xs[i_hi],
            "peak": {var: xs[imax], "n_sig": ns[imax]},
            "n_sig_at_min": ns[i_lo], "n_sig_at_max": ns[i_hi],
            "peak_at_boundary": imax in (i_lo, i_hi),
        }
        note = None
        nt = float(n_target or 0.0)
        if nt > 0.0:
            lo_off = ns[i_lo] >= nt
            hi_off = ns[i_hi] >= nt
            diag["band_runs_off_low"] = lo_off
            diag["band_runs_off_high"] = hi_off
            # When N* is known the band is fully captured iff N_sig < N* at BOTH
            # ends. In that case peak_at_boundary is harmless -- e.g. an
            # intentional zoom/refinement whose peak sits near a window edge --
            # so no warning fires; band_bracketed is the positive signal for it.
            diag["band_bracketed"] = not lo_off and not hi_off
            if lo_off or hi_off:
                ends = " and ".join(e for e, on in
                                    (("low", lo_off), ("high", hi_off)) if on)
                note = (f"reach band truncated: N_sig is still >= N* ({nt:g}) at "
                        f"the {ends} end of the {var} grid, so the true band "
                        f"extends beyond your scan -- widen the {var} grid until "
                        f"N_sig falls back below N* on both sides.")
        elif diag["peak_at_boundary"]:
            side = "low" if imax == i_lo else "high"
            note = (f"peak N_sig is at the {side} end of the scanned {var} grid "
                    f"({xs[imax]:g}). If this is a full scan, the reach may extend "
                    f"beyond it -- widen the {var} grid to bracket the band. If it "
                    f"is an intentional zoom into a known bracket, ignore this and "
                    f"pass n_target so the band edges are checked directly instead "
                    f"of the peak position.")
        if note and var == "g":
            note += (" If the yields also look too small, recheck width_ref_gev "
                     "(the physical g^2-stripped TOTAL width, not a placeholder).")
        return diag, note

    def _width_ref_note(self, yields, wref, m_phi, lifetime_mode) -> Optional[str]:
        """Non-fatal physical guard on width_ref_gev (portal mode only). A
        resonance must be narrower than its own mass, so the total width
        Gamma(g) = g^2 * width_ref_gev must stay below m_phi across the g grid.
        If it reaches m_phi anywhere, width_ref_gev is too large (or the top of
        the g grid too high) -- a magnitude error the docstring alone cannot
        catch. The opposite error (width_ref far too small) shows up instead as
        the peak pushed off the grid, flagged by _reach_quality_note."""
        if lifetime_mode != "portal" or wref <= 0.0 or m_phi <= 0.0:
            return None
        gmax = max((float(y["g"]) for y in yields if "g" in y), default=0.0)
        gamma_max = gmax * gmax * wref
        if gamma_max >= m_phi:
            return (f"width exceeds mass: Gamma = g^2*width_ref_gev reaches "
                    f"{gamma_max:.3g} GeV at g={gmax:g}, at or above m_phi="
                    f"{m_phi:g} GeV, so phi is not a narrow resonance there. "
                    f"Recheck width_ref_gev (too large?) or lower the top of "
                    f"the g grid.")
        return None

    @staticmethod
    def _offscale_note(ctaus, vol_scale) -> Optional[str]:
        """Flag a lifetime grid that cannot bracket the reach band.

        The band lives where ctau is within a few decades of the decay volume.
        More than ~3 decades clear of it on either side and the whole grid sits
        in a monotone regime: at short ctau everything decays before reaching
        the volume, at long ctau nothing decays inside it, and N_sig falls off
        smoothly either way. The caller cannot see that from N_sig alone -- it
        looks like an ordinary curve -- so the tool, which knows the geometry,
        says it. Model- and experiment-agnostic: it only compares the lifetimes
        actually scanned against the volume scale.
        """
        ctaus = [c for c in (ctaus or []) if c and c > 0.0]
        if not ctaus or not vol_scale or vol_scale <= 0.0:
            return None
        lo, hi = min(ctaus), max(ctaus)
        if hi < vol_scale * 1e-3:
            which = "far SHORT of"
        elif lo > vol_scale * 1e3:
            which = "far LONG of"
        else:
            return None
        return (f"every scanned lifetime is {which} the decay volume "
                f"(ctau spans {lo:.3g}-{hi:.3g} m, volume scale ~{vol_scale:.3g} m): "
                f"the scanned range cannot bracket the reach band, so any band "
                f"extracted from it is truncated rather than physical. Widen the "
                f"grid, or check the width/lifetime input -- an implied lifetime "
                f"this far off scale is far more often a wrong width than a "
                f"genuinely insensitive experiment.")

    @staticmethod
    def _normalization_breakdown(n_events, n_geo, n_tt, sum_weights, n_int,
                                 br_vis, wref) -> dict:
        """The factors that multiply into N_sig, echoed so a discrepancy can be
        localized.

        N_sig = n_int * [g^2] * br_visible * sum_i w_i * P_dec,i * acc_i. When a
        yield disagrees with an independent calculation by orders of magnitude
        the question is always WHICH factor -- the flux normalization, the
        branching ratio, or the geometric/kinematic acceptance -- and none of
        them were observable in the output, so the caller could see only the
        product. Each is cheap to report and turns a one-number disagreement
        into a locatable one.
        """
        return {
            "n_int": n_int,
            "br_visible_effective": br_vis,
            "width_ref_gev_effective": wref,
            "sum_weights_per_collision": sum_weights,
            "mean_weight_per_event": (sum_weights / n_events) if n_events else None,
            "n_events": n_events,
            "geometry_efficiency": (n_geo / n_events) if n_events else None,
            "acceptance_efficiency": (n_tt / n_geo) if n_geo else None,
            "overall_efficiency": (n_tt / n_events) if n_events else None,
            "note": ("N_sig = n_int * [g^2 in portal mode] * br_visible_effective "
                     "* sum_i w_i * P_dec,i * acc_i. If a yield is off by orders "
                     "of magnitude, compare these factors one at a time against "
                     "your own calculation: sum_weights_per_collision is the "
                     "upstream flux normalization, br_visible_effective the "
                     "branching into the detected state, and the efficiencies "
                     "the geometric and kinematic acceptance."),
        }

    def _write_outputs(self, dst, audit_path, yields, n_events, n_geo,
                       n_tt, m_phi, dmasses, wref, n_int,
                       lifetime_mode, acc_mode, sum_weights=0.0,
                       offscale_ctaus=None, offscale_vol_scale=None,
                       br_vis=None, inputs=None) -> str:
        """Write the yields table and format the tool result JSON."""
        notes = []
        grid_diag, rq = self._grid_diagnostic(yields, self.n_target)
        if rq:
            notes.append(rq)
        wn = self._width_ref_note(yields, wref, m_phi, lifetime_mode)
        if wn:
            notes.append(wn)
        # Low effective-statistics warning: N_sig is a Monte Carlo sum over the
        # events that pass geometry+acceptance; too few makes every yield noisy
        # and the reach unreliable. High-value signal that the upstream flux is
        # under-sampled (not that the physics is wrong).
        max_ns = max((y["n_sig"] for y in yields), default=0.0)
        if n_tt is not None and 0 < n_tt < _MIN_EFFECTIVE_EVENTS and max_ns > 0.0:
            notes.append(
                f"low statistics: only {n_tt} event(s) pass geometry+acceptance, "
                f"so every N_sig is a noisy Monte Carlo estimate (~1/sqrt(N)). "
                f"Generate a larger upstream flux (more Pythia events, or a "
                f"higher harvest/decay sample) so at least ~{_MIN_EFFECTIVE_EVENTS} "
                f"events reach the decay volume.")
        note = "; ".join(notes) if notes else None
        # Implied lab-frame proper decay length at g = 1 (portal mode): a direct
        # cross-check of width_ref_gev -- ctau(1) = hbar*c / width_ref_gev. If it
        # disagrees with the model's own lifetime, width_ref_gev is wrong.
        ctau_ref_g1 = None
        if lifetime_mode == "portal" and wref > 0.0:
            from . import llp_physics as phys
            ctau_ref_g1 = phys.HBARC_M_GEV / wref
        # Off-scale diagnostic. A scan whose implied proper decay length never
        # comes near the decay volume cannot resolve the reach band: at ctau far
        # below the volume everything decays before reaching it, far above it
        # nothing decays inside, and in both regimes N_sig is monotone across
        # the whole grid. The caller cannot see this from N_sig alone (the
        # numbers look like a normal falling curve), so say it explicitly.
        # Model- and experiment-agnostic: it compares the lifetimes the caller
        # actually scanned against the geometry this tool already knows.
        offscale = self._offscale_note(offscale_ctaus, offscale_vol_scale)
        if offscale:
            notes.append(offscale)
            note = "; ".join(notes)

        normalization = self._normalization_breakdown(
            n_events, n_geo, n_tt, sum_weights, n_int, br_vis, wref)

        table = {
            "schema": YIELDS_SCHEMA_VERSION,
            "conventions": dict(CONVENTIONS),
            "acceptance": acc_mode,
            "lifetime_mode": lifetime_mode,
            "m_phi_gev": m_phi,
            "daughter_masses_gev": list(dmasses),
            "width_ref_gev": wref,
            "ctau_ref_g1_m": ctau_ref_g1,
            "n_int": n_int,
            "n_events": n_events,
            "sum_weights": sum_weights,
            "n_pass_geometry": n_geo,
            "n_pass_acceptance": n_tt,
            "inputs": inputs,
            "normalization": normalization,
            "grid_diagnostic": grid_diag,
            "lifetime_offscale": offscale,
            "yields": yields,
        }
        if note:
            table["note"] = note
        with open(dst, "w") as fh:
            json.dump(table, fh, indent=2)
        result = {
            "status": "ok",
            "lifetime_mode": lifetime_mode,
            "acceptance": acc_mode,
            "n_events": n_events,
            "sum_weights": sum_weights,
            "ctau_ref_g1_m": ctau_ref_g1,
            "n_pass_geometry": n_geo,
            "n_pass_acceptance": n_tt,
            "inputs": inputs,
            "normalization": normalization,
            "grid_diagnostic": grid_diag,
            "lifetime_offscale": offscale,
            "output_path": os.path.relpath(dst, self.base_directory),
            "audit_path": os.path.relpath(audit_path, self.base_directory),
            "conventions": {"acceptance": CONVENTIONS["acceptance"],
                            "weight": CONVENTIONS["weight"]},
        }
        # The full per-point array is always on disk (output_path); only inline
        # it in the response when explicitly asked, so a dense scan does not
        # return a large token-heavy payload by default. grid_diagnostic already
        # carries the peak, band edges and end-of-grid N_sig needed to steer.
        if self.response_verbosity == "full":
            result["yields"] = yields
        else:
            result["n_yield_points"] = len(yields)
        if note:
            result["note"] = note
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


class DecayInVolumeVsCouplingTool(_DecayInVolumeBase):
    """Decay-in-volume LLP signal yield N_sig(g) over a coupling grid -- the
    tool for a g^2-portal reach scan. Step 3 of the reach chain, consuming
    g^2-stripped weighted LLP records with production vertices (MesonDecayToLLP
    output; parent channels may be concatenated).

    For a g^2 portal, production, total width, and lifetime scale with g while
    kinematics do not, so ONE event set covers the whole g axis by exact
    reweighting: N_sig(g) = N_int * g^2 * BR_vis * sum_i w_i * P_dec,i(g) *
    acc_i, with ctau(g) = hbar*c / (g^2 * width_ref_gev). Provide width_ref_gev
    + g_grid; the tool returns N_sig per g directly -- do NOT convert g to ctau
    or interpolate yourself. P_dec and acceptance come from each LLP's own
    production vertex (handling decay-in-flight parents and off-axis detectors)
    at the in-volume midpoint. Yields [{g, n_sig}] plus a per-event audit. For
    explicit lab-frame lifetimes instead, use DecayInVolumeVsLifetime.

    Prefer `partial_widths_ref_gev` (channel -> g^2-stripped partial width at
    g = 1) together with `visible_channels` over the scalar `width_ref_gev` +
    `br_visible` pair: the tool then sums the channels itself, so "include every
    channel open at this mass" becomes arithmetic the tool does rather than a
    convention the caller has to honour, and br_visible is derived from the same
    numbers instead of being supplied separately and kept in sync by hand. The
    scalar pair still works unchanged.
    """
    width_ref_gev: float = RuntimeField(
        default=0.0,
        description="EITHER this or `partial_widths_ref_gev` (not both "
                    "required); the latter takes precedence when given. "
                    "PHYSICAL g^2-stripped TOTAL width of phi at g = 1, in GeV: "
                    "Gamma_tot(g) / g^2, summed over EVERY decay channel open at "
                    "this mass. This sets the lifetime ctau(g) = "
                    "hbar*c/(g^2*width_ref_gev) and is a required physics input -- "
                    "it is NOT a free normalization or numerical-stability knob, so "
                    "do NOT pass 1.0 or any placeholder (a wrong value rescales "
                    "ctau and silently corrupts every yield). Compute it from the "
                    "width formula(s) with the g^2 stripped off, e.g. for the tree "
                    "phi -> mu mu channel width_ref = m_phi*(1-4*m_mu^2/m_phi^2)"
                    "^(3/2)/(8*pi); below 2 m_mu that channel is closed and the "
                    "total width is the loop-level gamma gamma width, which is far "
                    "smaller (~1e-10 to 1e-8 GeV). Must be > 0. The result echoes "
                    "the implied ctau at g=1 (hbar*c/width_ref_gev) -- sanity-check "
                    "it against your own lifetime.")
    g_grid: List[float] = RuntimeField(
        default=[],
        description="Couplings g at which to evaluate N_sig (non-empty, all "
                    "> 0). One event set covers the whole list by exact "
                    "reweighting, e.g. [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]. Supply "
                    "this OR `g_grid_path`.")
    g_grid_path: str = RuntimeField(
        default="",
        description="Alternative to `g_grid`: a path to a JSON file holding "
                    "the coupling list (a bare list, or {\"g_grid\": [...]}). "
                    "A reach scan wants a dense log-spaced grid, which is "
                    "natural to build with numpy and error-prone to hand-copy "
                    "into a call; point at the file instead. Ignored when "
                    "`g_grid` is non-empty.")
    partial_widths_ref_gev: Dict[str, float] = RuntimeField(
        default={},
        description="PREFERRED over width_ref_gev: the g^2-stripped PARTIAL "
                    "width of each decay channel open at this mass, in GeV at "
                    "g = 1, e.g. {'mumu': 7.5e-4, 'gammagamma': 2.2e-9}. The "
                    "tool sums them to get the total width that sets the "
                    "lifetime, so 'include every open channel' becomes "
                    "arithmetic the tool does rather than a convention you "
                    "have to remember. Combined with visible_channels it also "
                    "DERIVES br_visible, so the branching ratio and the total "
                    "width cannot drift out of sync -- the classic silent error "
                    "is supplying a total width that omits a channel the "
                    "branching ratio assumes, or vice versa. Channel names are "
                    "yours; only the split matters.")
    visible_channels: List[str] = RuntimeField(
        default=[],
        description="Which keys of partial_widths_ref_gev this search detects, "
                    "e.g. ['mumu'] for a two-track search or ['gammagamma'] "
                    "for a photon search. br_visible is then "
                    "sum(visible)/sum(all) and the separate br_visible field is "
                    "ignored. Required when partial_widths_ref_gev is given.")

    def _total_width_ref(self):
        """(width_ref, error_or_None) from partial widths if supplied."""
        pw = {str(k): float(v) for k, v in (self.partial_widths_ref_gev or {}).items()}
        if not pw:
            return None, None
        if any(v < 0.0 for v in pw.values()):
            return None, self.format_error(
                error="Invalid Parameter",
                reason=f"partial_widths_ref_gev has a negative width: {pw}",
                suggestion="Partial widths are non-negative; a channel closed "
                           "at this mass is 0 or simply absent")
        total = sum(pw.values())
        if total <= 0.0:
            return None, self.format_error(
                error="Invalid Parameter",
                reason="partial_widths_ref_gev sums to zero",
                suggestion="At least one channel must be open at this mass; "
                           "with no open channel the LLP is stable and there is "
                           "no decay-in-volume signal")
        return total, None

    def _resolve_br_visible(self):
        pw = {str(k): float(v) for k, v in (self.partial_widths_ref_gev or {}).items()}
        if not pw:
            return float(self.br_visible)
        vis = [str(c) for c in (self.visible_channels or [])]
        if not vis:
            return self.format_error(
                error="Invalid Parameter",
                reason="partial_widths_ref_gev was given without "
                       "visible_channels",
                suggestion="Name the detected channel(s), e.g. "
                           "visible_channels=['mumu'] for a two-track search; "
                           "br_visible is then derived as sum(visible)/sum(all)")
        unknown = [c for c in vis if c not in pw]
        if unknown:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"visible_channels {unknown} are not keys of "
                       f"partial_widths_ref_gev ({sorted(pw)})",
                suggestion="Use the same channel names in both fields")
        total, err = self._total_width_ref()
        if err:
            return err
        return sum(pw[c] for c in vis) / total

    def _lifetime_spec(self, phys):
        derived, err = self._total_width_ref()
        if err:
            return err
        wref = derived if derived is not None else float(self.width_ref_gev or 0.0)
        g_grid = [float(g) for g in (self.g_grid or [])]
        if not g_grid and str(self.g_grid_path).strip():
            gp = self._safe_path(self.g_grid_path)
            if not gp or not os.path.isfile(gp):
                return self.format_error(
                    error="File Not Found",
                    reason=f"g_grid_path does not resolve to a file inside "
                           f"base_directory: {self.g_grid_path}",
                    suggestion="Pass a relative path to a JSON file holding "
                               "the coupling list")
            try:
                with open(gp) as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    loaded = loaded.get("g_grid")
                g_grid = [float(g) for g in loaded]
            except Exception as exc:
                return self.format_error(
                    error="Invalid Input",
                    reason=f"could not read g_grid_path as a list of "
                           f"couplings: {exc}",
                    suggestion="The file must hold a JSON list of positive "
                               "numbers, or {'g_grid': [...]}")
        if wref <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"total g^2-stripped width must be positive (got {wref})",
                suggestion="Supply partial_widths_ref_gev (preferred) or "
                           "width_ref_gev. This is Gamma_tot at g = 1; for "
                           "phi -> mu mu it vanishes at m_phi <= 2 m_mu, "
                           "where no decay-in-volume signal exists")
        if not g_grid or any(g <= 0.0 for g in g_grid):
            return self.format_error(
                error="Invalid Parameter",
                reason="g_grid must be a non-empty list of positive couplings",
                suggestion="Provide g_grid like [1e-6, 3e-6, 1e-5]")
        # (label_key, label_value, ctau_m, prefactor)
        points = [("g", g, phys.HBARC_M_GEV / (g * g * wref), g * g)
                  for g in g_grid]
        return points, wref, "portal"


class DecayInVolumeVsLifetimeTool(_DecayInVolumeBase):
    """Decay-in-volume LLP signal yield N_sig(ctau) at explicit lab-frame
    lifetimes ctau [m] -- model-agnostic (no portal, no coupling, no g^2;
    weights used verbatim). Step 3 of the reach chain, consuming weighted LLP
    records with production vertices (MesonDecayToLLP output; parent channels
    may be concatenated).

    N_sig(ctau) = N_int * BR_vis * sum_i w_i * P_dec,i(ctau) * acc_i, with P_dec
    and acceptance from each LLP's own production vertex (handling
    decay-in-flight parents and off-axis detectors) at the in-volume midpoint.
    Provide ctau_grid_m; the tool returns N_sig per ctau. Yields [{ctau_m,
    n_sig}] plus a per-event audit. For a g^2-portal coupling scan (the usual
    reach study), use DecayInVolumeVsCoupling, which converts g to ctau and
    applies the g^2 production factor for you.
    """
    ctau_grid_m: List[float] = RuntimeField(
        description="Lab-frame ctau values in meters (non-empty, all > 0), "
                    "model-agnostic. e.g. [0.1, 1.0, 10.0]")

    def _lifetime_spec(self, phys):
        ctau_grid = [float(c) for c in (self.ctau_grid_m or [])]
        if not ctau_grid or any(c <= 0.0 for c in ctau_grid):
            return self.format_error(
                error="Invalid Parameter",
                reason="ctau_grid_m must be a non-empty list of positive "
                       "lab-frame ctau values in meters",
                suggestion="Provide ctau_grid_m like [0.1, 1.0, 10.0]")
        points = [("ctau_m", c, c, 1.0) for c in ctau_grid]
        return points, None, "ctau"
