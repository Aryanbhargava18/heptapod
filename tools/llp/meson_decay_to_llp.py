"""
# meson_decay_to_llp.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Any, Dict, List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "llpflux-2.0"
MANIFEST_SCHEMA_VERSION = "llpflux-manifest-2.0"
WEIGHT_CONVENTION = "g2_stripped_per_primary_interaction"

# Standard-Model parent proper decay length c*tau [m], keyed by |PDG id|. These
# are universal PDG constants, so they are auto-applied when `ctau_parent_m` is
# left at its -1 sentinel (the parent PID travels with each flux record). The
# agent can still override with an explicit `ctau_parent_m >= 0`. Weakly decaying
# parents carry their finite c*tau (decay-in-flight vertices); parents that decay
# strongly or electromagnetically are prompt -> 0.0 (LLP produced at the IP). The
# set spans the common forward-flux parents across light/charm/bottom sectors so
# the tool is not tuned to any single analysis.
MESON_CTAU_M = {
    # light, weakly decaying -- decay-in-flight is significant
    211: 7.8045,      # pi+/-
    321: 3.712,       # K+/-
    130: 15.34,       # K0_L
    310: 0.02684,     # K0_S
    # charm -- short but finite
    411: 3.098e-4,    # D+/-
    421: 1.229e-4,    # D0
    431: 1.511e-4,    # Ds+/-
    # bottom
    521: 4.911e-4,    # B+/-
    511: 4.554e-4,    # B0
    531: 4.535e-4,    # Bs0
    541: 1.529e-4,    # Bc+/-
    # prompt (strong / electromagnetic decays) -- LLP at the IP
    111: 0.0,         # pi0
    221: 0.0,         # eta
    331: 0.0,         # eta'
    113: 0.0,         # rho0
    213: 0.0,         # rho+/-
    223: 0.0,         # omega
    333: 0.0,         # phi(1020)
    441: 0.0,         # eta_c
    443: 0.0,         # J/psi
    100443: 0.0,      # psi(2S)
    553: 0.0,         # Upsilon(1S)
}


def _strip_spec_scheme(s: str) -> str:
    """Tolerate an accidental scheme prefix on a spectrum PATH (agents read
    'type: table' and sometimes write 'table:<path>'). The spectrum type is
    inferred from the file itself, so strip a leading table:/two_body: ."""
    for pre in ("table:", "two_body:"):
        if s.startswith(pre):
            return s[len(pre):]
    return s


#: Measured on a fixed kaon flux: reproducibility of the STRONG-coupling reach
#: boundary against the number of strata. The weak boundary and the yield are
#: already at the reference's own Monte-Carlo precision (~0.4%) by K=16, so
#: this table is what decides whether K needs raising.
_STRONG_EDGE_REPRODUCIBILITY = {16: "4.8%", 32: "2.1%", 64: "1.0%"}


def _n_strata_advice(K: int) -> str:
    """One line in the RESULT when K is coarse for reach work.

    The accuracy/cost trade-off is documented on the `n_strata` field, but a
    docstring paragraph is invisible at call time: a 2026-08 trial picked the
    default K=16 to bound file size, correctly noted in its own write-up that
    this is the coarse setting for the strong-coupling edge, and lost that
    grading stage to it. Which boundary K controls is the part worth saying
    where the caller will actually see it -- next to the number they chose.
    """
    if K >= 64:
        return ""
    got = _STRONG_EDGE_REPRODUCIBILITY.get(K)
    got = f"~{got}" if got else "coarser than ~1.0%"
    return (f"n_strata={K}: the STRONG-coupling (lifetime-limited) reach "
            f"boundary is reproducible to {got} here, vs ~1.0% at K=64. The "
            f"yield and the WEAK-coupling boundary are already converged at "
            f"this K. Raise n_strata to 64 if you are locating the upper edge "
            f"of a reach contour; record count scales linearly with K, so "
            f"raising it only for the small heavy-flavour fluxes is cheap.")


class MesonDecayToLLPTool(BaseTool):
    """
    Decay a harvested forward parent-meson flux (one parent channel) to a
    weighted LLP flux carrying the decay-in-flight production vertex, at one
    LLP mass or -- preferably -- across a whole mass GRID in one call. Step 2
    of the reach chain `pythia -> harvest -> MESON-DECAY -> DecayInVolume`: it
    consumes HarvestForwardFluxTool output and emits g^2-stripped LLP records
    with a production point. One call = one parent channel; run once per
    channel and concatenate across channels.

    Grid mode (preferred): pass `grid` (a list of {m_phi_gev, spectrum_spec,
    B_hat}) + `output_dir` to sweep the whole mass grid in one call, one LLP
    file per mass; the flux and the mass-independent decay-in-flight vertices
    are loaded/sampled once and reused across masses. Single-mass mode
    (`m_phi_gev` + `spectrum_spec` + `B_hat` + `output_path`, no `grid`) is
    kept for one-off use.

    Model-agnostic: the production amplitude is NOT in the tool. The LLP
    energy distribution is the declared `spectrum_spec` (a `table` for 3-body
    radiation M -> l nu X, a `two_body` delta for M -> X Y) and the
    g^2-stripped branching is `B_hat`; the tool supplies only the mechanics --
    rest-frame -> lab boost, isotropic LLP direction (unpolarised parent), and
    the decay-in-flight vertex. That vertex: a parent of lab momentum p decays
    at ell ~ Exp(lambda), lambda = beta*gamma * c*tau, drawn by STRATIFIED
    sampling (`n_strata` equal-probability strata, one point drawn uniformly
    inside each) -- unbiased even against the hard absorber cut applied
    downstream. The parent c*tau defaults to the SM value for
    the parent PID that travels with the flux (auto): finite for long-lived
    parents (kaons, pions, charm, bottom) so their decay-in-flight vertices are
    resolved, and 0 for prompt parents (vectors, charmonia) which collapse to
    the IP. Set `ctau_parent_m >= 0` to override. The absorber / production-
    region cut lives downstream in the DecayInVolume* yield tools.

    Output (one JSON line per LLP, schema "llpflux-2.0"): E, px, py, pz (GeV);
    vx, vy, vz (m, production vertex); theta_lab; parent_channel;
    event_weight_g2_stripped = parent_weight * B_hat / n_strata. Downstream:
    N_sig(g) = N_int * g^2 * BR_vis * sum_i w_i * P_dec,i(g) * acc_i. Grid mode
    writes `output_dir/llp_<channel>_m<mass>.jsonl` + `meson_decay_manifest.json`;
    single mode writes `output_path` + `.manifest.json`. A kinematically closed
    mass returns ok with n_samples = 0 (empty file). A spectrum file that does
    not exist is a HARD ERROR that aborts the whole call -- never a silent skip
    of that mass -- so an ok result means every requested mass was produced. The
    result reports each mass with its n_samples and sum_weights (grid mode: the
    `masses` list), so completeness is checkable directly: confirm no channel is
    missing or unexpectedly empty before feeding the flux downstream.
    """

    parent_flux_path: str = RuntimeField(
        description="Line-delimited JSON parent records for ONE parent channel "
                    "(harvest_forward_flux output): E,px,py,pz (GeV) + a "
                    "per-collision weight (weight_per_collision or weight)")
    spectrum_spec: str = RuntimeField(
        default="",
        description="SINGLE-mass mode only (ignored in grid mode). A plain "
                    "file PATH to the parent-rest-frame LLP energy spectrum "
                    "(x=2E*/m_parent): a .csv table (x,pdf) or a .yaml "
                    "('type: table'/'type: two_body'). Pass the path itself -- "
                    "do NOT add a scheme prefix like 'table:'.")
    m_phi_gev: float = RuntimeField(
        default=0.0,
        description="LLP mass in GeV (single-mass mode; ignored when `grid` "
                    "is given)")
    parent_mass_gev: float = RuntimeField(description="Parent mass in GeV")
    B_hat: float = RuntimeField(
        default=0.0,
        description="g^2-stripped branching Br(parent -> ... LLP) at g=1 "
                    "(single-mass mode; ignored when `grid` is given)")
    grid: List[Dict[str, Any]] = RuntimeField(
        default=[],
        description="GRID MODE (preferred): sweep the whole declared mass grid "
                    "in ONE call, reusing the loaded flux and the "
                    "(mass-independent) decay-in-flight vertices; one LLP file "
                    "is written per mass into `output_dir`. A list of entries, "
                    "each {m_phi_gev, spectrum_spec, B_hat} -- spectrum_spec is "
                    "a plain path (no scheme prefix). Example: "
                    "[{\"m_phi_gev\":0.23,\"spectrum_spec\":\"spectra/spec_K_"
                    "m0.230.csv\",\"B_hat\":1.2e-3}, ...]. In grid mode you do "
                    "NOT pass the single-mass fields (m_phi_gev/spectrum_spec/"
                    "B_hat/output_path); pass `grid` + `output_dir` only.")
    grid_path: str = RuntimeField(
        default="",
        description="Alternative to `grid`: a path to a JSON file holding the "
                    "same list. Use this when the grid is long -- building it "
                    "with a script and pointing at the file avoids hand-copying "
                    "a large array into the call, which is where transcription "
                    "errors happen. The file may hold either the bare list or "
                    "{\"grid\": [...]}. Ignored when `grid` is given.")
    output_dir: str = RuntimeField(
        default="",
        description="GRID MODE: relative directory for the per-mass LLP files "
                    "(required when `grid` is given)")
    ctau_parent_m: float = RuntimeField(
        default=-1.0,
        description="Parent proper decay length [m]. Leave at -1 to auto-apply "
                    "the SM value for the parent PID carried by the flux; set "
                    ">= 0 to override (0 = prompt, LLP at the IP)")
    n_strata: int = RuntimeField(
        default=16,
        description="Equal-probability strata of the parent decay-length "
                    "distribution, one point drawn at random inside each "
                    "(default 16). UNBIASED at any value -- K only trades "
                    "cost against variance, and the record count scales "
                    "linearly with it. Measured on a fixed kaon flux: at K=16 "
                    "the yield and the weak-coupling reach boundary are "
                    "reproducible to ~0.4%, which is the reference's own "
                    "Monte-Carlo precision. The STRONG-coupling boundary is "
                    "harder and keeps improving: ~4.8% at K=16, ~2.1% at "
                    "K=32, ~1.0% at K=64. Raise K if that boundary matters.")
    seed: int = RuntimeField(description="RNG seed for deterministic sampling")
    output_path: str = RuntimeField(
        default="",
        description="SINGLE-mass mode only: relative path for the LLP records "
                    "(line-delimited JSON). In grid mode use `output_dir`.")
    base_directory: str = StateField(
        default=".", description="Base directory for safe paths")

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        import numpy as np
        from . import llp_physics as phys

        mM = float(self.parent_mass_gev)
        if mM <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"parent mass must be > 0 (got {mM})",
                suggestion="Provide parent_mass_gev > 0")
        K = max(1, int(self.n_strata))

        # ---- resolve the work list: grid mode XOR single-mass mode ---- #
        # An inline `grid` wins; otherwise `grid_path` may supply the same
        # list from a file, so a long grid can be generated by a script rather
        # than hand-copied into the call.
        grid_entries = list(self.grid or [])
        if not grid_entries and str(self.grid_path).strip():
            gp = self._safe_path(self.grid_path)
            if not gp or not os.path.isfile(gp):
                return self.format_error(
                    error="File Not Found",
                    reason=f"grid_path does not resolve to a file inside "
                           f"base_directory: {self.grid_path}",
                    suggestion="Pass a relative path to a JSON file holding "
                               "the grid list")
            try:
                with open(gp) as fh:
                    loaded = json.load(fh)
            except Exception as exc:
                return self.format_error(
                    error="Invalid Input",
                    reason=f"could not parse grid_path as JSON: {exc}",
                    suggestion="The file must hold a JSON list, or an object "
                               "with a 'grid' key")
            if isinstance(loaded, dict):
                loaded = loaded.get("grid")
            if not isinstance(loaded, list) or not loaded:
                return self.format_error(
                    error="Invalid Input",
                    reason="grid_path must hold a non-empty JSON list of "
                           "{m_phi_gev, spectrum_spec, B_hat} entries",
                    suggestion="Write the same list you would pass inline")
            grid_entries = loaded

        grid_mode = bool(grid_entries)
        if grid_mode:
            out_dir = self._safe_path(self.output_dir)
            if not out_dir:
                return self.format_error(
                    error="Invalid Parameter",
                    reason="grid mode requires output_dir (a relative dir "
                           "inside base_directory)",
                    suggestion="Set output_dir when passing a grid")
            entries = []
            for j, e in enumerate(grid_entries):
                try:
                    # B_hat is OPTIONAL: a ProductionSpectrumTool spectrum
                    # carries it in its header, and reading it from there keeps
                    # the spectrum and its normalisation from being paired by
                    # hand (and mis-paired).
                    b = e.get("B_hat")
                    entries.append((float(e["m_phi_gev"]),
                                    _strip_spec_scheme(str(e["spectrum_spec"])),
                                    None if b is None else float(b)))
                except (KeyError, TypeError, ValueError) as ex:
                    return self.format_error(
                        error="Invalid Parameter",
                        reason=f"grid[{j}] must be "
                               f"{{m_phi_gev, spectrum_spec}} with optional "
                               f"B_hat ({ex})",
                        suggestion="Each grid entry needs m_phi_gev and "
                                   "spectrum_spec; B_hat is read from the "
                                   "spectrum header when omitted")
        else:
            if not self.spectrum_spec or not self.output_path:
                return self.format_error(
                    error="Invalid Parameter",
                    reason="single-mass mode needs spectrum_spec AND "
                           "output_path",
                    suggestion="Provide spectrum_spec + output_path, OR use "
                               "grid mode: pass `grid` + `output_dir` (and "
                               "none of the single-mass fields)")
            entries = [(float(self.m_phi_gev),
                        _strip_spec_scheme(str(self.spectrum_spec)),
                        float(self.B_hat) if self.B_hat else None)]
            dst = self._safe_path(self.output_path)
            if not dst:
                return self.format_error(
                    error="Access Denied",
                    reason="output_path escapes base_directory",
                    suggestion="Use a relative path inside base_directory")

        flux_src = self._safe_path(self.parent_flux_path)
        if not flux_src:
            return self.format_error(
                error="Access Denied",
                reason="parent_flux_path escapes base_directory",
                suggestion="Use a relative path inside base_directory")
        if not os.path.exists(flux_src):
            return self.format_error(
                error="File Not Found",
                reason=f"parent_flux_path not found: {self.parent_flux_path}",
                suggestion="Provide the harvest_forward_flux output")

        # validate + load every declared spectrum up front, resolving B_hat
        # from the spectrum header where the caller did not supply one
        specs = {}
        resolved = []
        for (m_phi, spec_rel, kap) in entries:
            if m_phi <= 0.0:
                return self.format_error(
                    error="Invalid Parameter",
                    reason=f"need m_phi > 0 (got {m_phi})",
                    suggestion="Pass the LLP mass in GeV")
            sp = self._safe_path(spec_rel)
            if not sp:
                return self.format_error(
                    error="Access Denied",
                    reason=f"spectrum_spec escapes base_directory: {spec_rel}",
                    suggestion="Use a relative path inside base_directory")
            if not os.path.exists(sp):
                return self.format_error(
                    error="File Not Found",
                    reason=f"spectrum_spec not found: {spec_rel}",
                    suggestion="Provide an existing spectrum file")
            if spec_rel not in specs:
                try:
                    specs[spec_rel] = phys.LLPSpectrum.from_path(sp)
                except Exception as e:
                    return self.format_error(
                        error="Spectrum Error", reason=str(e),
                        suggestion="Provide a valid table/two_body spectrum")

            # ---- reconcile B_hat and the mass against the spectrum ----- #
            meta = phys.read_spectrum_header(sp)
            hdr_b = meta.get("B_hat")
            hdr_m = meta.get("m_phi_gev")
            if hdr_m is not None:
                try:
                    if abs(float(hdr_m) - m_phi) > 1e-9 * max(1.0, m_phi):
                        return self.format_error(
                            error="Spectrum Mismatch",
                            reason=f"{spec_rel} was computed for m_phi_gev="
                                   f"{hdr_m}, but this call asks for {m_phi}",
                            suggestion="Spectra are mass-specific. Use the "
                                       "spectrum generated for THIS mass, or "
                                       "regenerate it with "
                                       "ProductionSpectrumTool.")
                except ValueError:
                    pass
            if kap is None:
                if hdr_b is None:
                    return self.format_error(
                        error="Invalid Parameter",
                        reason=f"no B_hat given and {spec_rel} carries none "
                               f"in its header",
                        suggestion="Either pass B_hat (the reduced branching "
                                   "fraction, Br(h -> ... LLP) with g^2 "
                                   "factored out), or generate the spectrum "
                                   "with ProductionSpectrumTool, which records "
                                   "it alongside f(x).")
                kap = float(hdr_b)
            elif hdr_b is not None:
                # both present: they must agree, or a wrong normalisation is
                # about to be applied to a correct spectrum
                hb = float(hdr_b)
                if hb > 0.0 and abs(kap / hb - 1.0) > 1e-6:
                    return self.format_error(
                        error="Spectrum Mismatch",
                        reason=f"B_hat={kap} was supplied but {spec_rel} was "
                               f"generated with B_hat={hb}",
                        suggestion="Drop the explicit B_hat and let it be read "
                                   "from the spectrum -- they come from one "
                                   "integral and must not be paired by hand.")
            if kap < 0.0:
                return self.format_error(
                    error="Invalid Parameter",
                    reason=f"need B_hat >= 0 (got {kap})",
                    suggestion="B_hat is a branching fraction with g^2 "
                               "factored out; it cannot be negative")
            resolved.append((m_phi, spec_rel, kap))
        entries = resolved

        # --------------------- load parent flux ONCE ------------------- #
        E, PX, PY, PZ, W, chan, chan_pid = [], [], [], [], [], None, None
        try:
            with open(flux_src) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    E.append(r["E"]); PX.append(r["px"]); PY.append(r["py"])
                    PZ.append(r["pz"])
                    W.append(r.get("weight_per_collision", r.get("weight")))
                    if chan is None:
                        chan = r.get("parent")
                    if chan_pid is None:
                        chan_pid = r.get("parent_pid")
        except Exception as e:
            return self.format_error(
                error="Parse Error", reason=f"bad parent flux record: {e}",
                suggestion="Records need E,px,py,pz and a per-collision weight")
        parent_channel = chan if chan is not None else "parent"

        # ---- resolve the parent decay length -------------------------- #
        # ctau_parent_m == -1 (default) means "auto": use the SM c*tau for the
        # parent PID that travels with the flux. Any value >= 0 is an explicit
        # override. If auto is requested but the PID is unknown/absent, fall back
        # to prompt and surface a note so the yield is not silently mis-set.
        ctau_note = None
        if float(self.ctau_parent_m) >= 0.0:
            ctau_used = float(self.ctau_parent_m)
            ctau_source = "explicit"
        else:
            pid_key = None if chan_pid is None else abs(int(chan_pid))
            if pid_key in MESON_CTAU_M:
                ctau_used = float(MESON_CTAU_M[pid_key])
                ctau_source = "auto_pid"
            else:
                ctau_used = 0.0
                ctau_source = "auto_fallback_prompt"
                ctau_note = (
                    f"parent PID {chan_pid!r} has no tabulated SM c*tau; "
                    "assumed prompt (ctau=0). Pass ctau_parent_m explicitly if "
                    "this parent is long-lived.")

        # --------- decay-in-flight vertices: sampled ONCE (mass-indep) --- #
        rng = np.random.default_rng(int(self.seed))
        if len(W) > 0:
            pvec = np.stack([np.asarray(PX, float), np.asarray(PY, float),
                             np.asarray(PZ, float)], axis=1)
            pw = np.asarray(W, float)
            rep, vertex = phys.sample_decay_vertices(
                pvec, ctau_used, mM, K, rng)
            pv = pvec[rep]
            e_par = np.sqrt(np.einsum("ij,ij->i", pv, pv) + mM * mM)
            pw_rep = pw[rep]
        else:
            rep = vertex = pv = e_par = pw_rep = None

        def _decay_one(m_phi, spectrum, kap, out_full):
            """Sample+write one mass, reusing the shared vertices. Returns
            (n_written, sum_weights, cutoff, above_cutoff)."""
            cutoff = float(spectrum.cutoff_gev(mM))
            above = not spectrum.is_open(mM, m_phi)
            os.makedirs(os.path.dirname(out_full) or ".", exist_ok=True)
            if above or rep is None:
                open(out_full, "w").close()
                return 0, 0.0, cutoff, above
            n = len(rep)
            x = spectrum.sample_x(mM, m_phi, n, rng)
            estar = 0.5 * x * mM
            pstar = np.sqrt(np.maximum(estar ** 2 - m_phi ** 2, 0.0))
            cth = rng.uniform(-1.0, 1.0, n)
            sth = np.sqrt(1.0 - cth ** 2)
            az = rng.uniform(0.0, 2.0 * np.pi, n)
            kstar = pstar[:, None] * np.stack(
                [sth * np.cos(az), sth * np.sin(az), cth], axis=1)
            elab, klab = phys.boost_to_lab(estar, kstar, e_par, pv, mM)
            theta_lab = np.arccos(np.clip(
                klab[:, 2] / np.maximum(np.linalg.norm(klab, axis=1),
                                        1e-300), -1.0, 1.0))
            w = pw_rep * kap / K
            with open(out_full, "w") as fh:
                for i in range(n):
                    fh.write(json.dumps({
                        "schema": SCHEMA_VERSION, "event_id": i,
                        "E": float(elab[i]), "px": float(klab[i, 0]),
                        "py": float(klab[i, 1]), "pz": float(klab[i, 2]),
                        "vx": float(vertex[i, 0]), "vy": float(vertex[i, 1]),
                        "vz": float(vertex[i, 2]),
                        "theta_lab": float(theta_lab[i]),
                        "parent_channel": parent_channel,
                        "event_weight_g2_stripped": float(w[i]),
                    }, separators=(",", ":")) + "\n")
            return n, float(w.sum()), cutoff, above

        # ----------------------------- run ----------------------------- #
        try:
            masses = []
            total_n, total_w = 0, 0.0
            for (m_phi, spec_rel, kap) in entries:
                if grid_mode:
                    out_full = os.path.join(
                        out_dir, f"llp_{parent_channel}_m{m_phi:.4f}.jsonl")
                else:
                    out_full = dst
                n_w, s_w, cutoff, above = _decay_one(
                    m_phi, specs[spec_rel], kap, out_full)
                total_n += n_w; total_w += s_w
                masses.append({
                    "m_phi_gev": m_phi, "B_hat": kap,
                    "spectrum_spec": os.path.relpath(
                        self._safe_path(spec_rel), self.base_directory),
                    "path": os.path.relpath(out_full, self.base_directory),
                    "n_samples": n_w, "sum_weights": s_w,
                    "kinematic_cutoff_gev": cutoff,
                    "above_kinematic_cutoff": above,
                })
        except Exception as e:
            return self.format_error(
                error="Decay Error", reason=str(e),
                context=f"parent={parent_channel}, grid={grid_mode}",
                suggestion="Check the spectrum, masses and parent flux")

        # ---------------------------- outputs -------------------------- #
        man_dir = out_dir if grid_mode else (os.path.dirname(dst) or ".")
        manifest_path = os.path.join(man_dir, "meson_decay_manifest.json") \
            if grid_mode else os.path.splitext(dst)[0] + ".manifest.json"
        manifest = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "weight_convention": WEIGHT_CONVENTION,
            "weight_definition": "event_weight_g2_stripped = parent_weight * "
                                 "B_hat / n_strata; N_sig(g) = N_int * g^2 * "
                                 "BR_vis * sum_i w_i * P_dec,i(g) * acc_i",
            "mode": "grid" if grid_mode else "single",
            "parent_flux_path": os.path.relpath(flux_src, self.base_directory),
            "parent": parent_channel,
            "parent_pid": chan_pid,
            "parent_mass_gev": mM,
            "ctau_parent_m": ctau_used,
            "ctau_source": ctau_source,
            "n_strata": K,
            "seed": int(self.seed),
            "n_samples_total": total_n,
            "sum_weights_total": total_w,
            "masses": masses,
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        result = {
            "status": "ok",
            "mode": "grid" if grid_mode else "single",
            "n_masses": len(masses),
            "n_samples_total": total_n,
            "sum_weights_total": total_w,
            "manifest_path": os.path.relpath(manifest_path, self.base_directory),
            "weight_convention": WEIGHT_CONVENTION,
            "parent": parent_channel,
            "ctau_parent_m": ctau_used,
            "ctau_source": ctau_source,
            "n_strata": K,
            "masses": masses,
        }
        if ctau_note is not None:
            result["note"] = ctau_note
        advice = _n_strata_advice(K)
        if advice:
            result["n_strata_advice"] = advice
        # single-mass back-compat fields
        if not grid_mode and masses:
            m0 = masses[0]
            result.update({
                "n_samples": m0["n_samples"],
                "sum_weights": m0["sum_weights"],
                "output_path": m0["path"],
                "kinematic_cutoff_gev": m0["kinematic_cutoff_gev"],
                "above_kinematic_cutoff": m0["above_kinematic_cutoff"],
            })
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
