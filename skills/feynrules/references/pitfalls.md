# Pitfalls catalog (feynrules)

Pitfalls that bite while writing the `.fr` and compiling the UFO.
For failures that appear once MG5 is running, see the `mg5` skill.

## P1. Duplicate mass/width parameter

**Symptom (MG5):** `name MX define multiple time. Please correct the UFO model!`

**Cause:** Particle class `Mass -> {MX, val}` auto-declares `MX`. Adding another entry in `M$Parameters` with the same name produces a duplicate.

**Fix:** Remove from `M$Parameters`. Only add to `M$Parameters` with `ParameterType -> Internal` if the mass is a derived quantity (e.g., `MZprime = gx * vx`).

## P2. Coupling with no interaction order

**Symptom (MG5):** `Some couplings have '1' order. This is not allowed in MG.`

**Cause:** A coupling expression contains only BSM parameters (no `ee`, `gs`, SM Yukawas) and FeynRules has no order to assign. Falls back to a placeholder `{'1':1}`.

**Fix:** Two changes in the `.fr`:

1. Declare an `NP` order block:
   ```mathematica
   M$InteractionOrderHierarchy = {{QCD,1},{QED,2},{NP,1}};
   M$InteractionOrderLimit     = {{QCD,99},{QED,99},{NP,99}};
   ```

2. Tag every BSM external parameter:
   ```mathematica
   gNew == { ParameterType -> External, ..., InteractionOrder -> {NP, 1} }
   ```

Mixed BSM×SM couplings (like `eps * ee`) inherit the right combined order automatically.

**Verification:** After regenerating UFO, `grep "'1':1" UFO/couplings.py` must return nothing.

## P5. Lorentz index `mu` collides with muon name

**Symptom:** Compilation OK but Feynman rules contain wrong propagator structures, or runtime errors deep in FeynRules.

**Cause:** FR's `Ga[mu]`, `del[psi, mu]`, and `FS[A, mu, nu]` all use `mu` as a Lorentz index symbol. The SM particle class also defines `mu` as the muon field. FR resolves by context, but you can construct ambiguous expressions.

**Defensive pattern:** Use distinct Lorentz index letters in BSM Lagrangian terms, e.g.
```mathematica
LphiAA := gphiAA phid Aprime[lor1] Aprime[lor1];
LFmunu := -1/4 FS[Aprime, lor1, lor2] FS[Aprime, lor1, lor2];
```
or just rebind the Lorentz indices inside `Block[{...}, ...]`.

## P6. Self-conjugate vector — Stuckelberg vs broken U(1)'

**Symptom:** Hidden U(1)' models compiled without the dark Higgs scalar give massless dark photon (or weird unitarity violations at high energy in cross sections).

**Cause:** A massive vector field needs a Higgs (or Stückelberg field) for its longitudinal mode. FR `Mass -> {MA, val}` simply gives the vector a Proca mass term, which is fine for tree-level cross sections but unphysical at high energies. Also the goldstone doesn't appear automatically.

**Practical guidance:** For LHC-scale processes with $\sqrt{s} \gg M_{A'}$, the longitudinal mode must be present. For low-energy ($\tau$ decays, $B$-meson decays) the Proca mass is fine. If in doubt, write the dark Higgs explicitly.

## P7. ChiralProjector convention

**Symptom:** Wrong sign on chiral coefficient or coupling appears as $g_L$ when you wanted $g_R$.

**Reference:**
- `ProjM = (1 - γ⁵)/2 = P_L`
- `ProjP = (1 + γ⁵)/2 = P_R`

For a scalar bilinear $\bar\psi_L \chi_R = \bar\psi\,P_R\,\chi$ → `psibar.ProjP.chi`.
For a current $\bar\psi_L\,\gamma^\mu\,\chi_L = \bar\psi\,\gamma^\mu\,P_L\,\chi$ → `psibar.Ga[mu].ProjM.chi`.

## P9. FR_PATH not propagating to MCP server

**Symptom:** Heptapod `FeynRulesToUFO` errors with `FeynRulesPath must be a directory. Got: /path/to/FeynRules_v2.X.X` despite the env var being set in your shell.

**Cause:** The MCP server runs in its own process and doesn't inherit shell env vars set after server startup.

**Fix:** Set `FR_PATH` in MCP server's launch environment (e.g., in `~/.claude/mcp.json` or a wrapper script). Alternatively, fall back to running wolframscript directly with absolute paths.

## P10. Lagrangian name assumed, or the SM added when it shouldn't be

**Symptom:** A UFO directory is produced and the tool reports success, but it holds only `__init__.py`, `function_library.py`, `object_library.py`, `propagators.py` and `write_param_card.py` — no `particles.py`, `vertices.py`, `couplings.py` or `parameters.py`. MG5 then fails to import it, or imports a model with no physics in it.

**Cause:** `LBSM` is a convention of one model, not of FeynRules. The generator used to hardcode `WriteUFO[LSM + LBSM, ...]`, so any model naming its total Lagrangian something else (`LNP`, `LTot`, `LLeptoQuark`, or only defining the pieces) left `LBSM` as a free symbol. `WriteUFO` writes its boilerplate regardless of whether vertex extraction found anything, and the script exited 0.

**Fix:** Pass the Lagrangian explicitly — `lagrangian="LSM + LNP"`, or `"LSM + LkinS1 + L1YukRR"` if the model never sums its pieces. Omit it and the tool reports the Lagrangian-like symbols the model actually defines rather than guessing. For a model that does not extend the Standard Model, also set `load_sm=false`; `LSM` is otherwise loaded and added unconditionally.

## P11. Two FeynRules checks that fail for the Standard Model

**Symptom:** `quadratic_diagonal` fails and `kinetic_norm` is inconclusive on a model that looks fine.

**Cause:** Neither is about your model. `CheckDiagonalQuadraticTerms` runs over `GetQuadraticTerms` — kinetic + mass + mixing — *without* the scalar–vector `DeleteCases` that `CheckDiagonalKineticTerms` applies, so it reports the R_ξ gauge–Goldstone mixing term (`∂_μ G^± W^∓μ`) present in any spontaneously broken gauge theory. `CheckKineticTermNormalisation` expects a ∓1 coefficient (∓1/2 for a real field), which does not apply to a Weyl fermion, so it trips on the SM's massless neutrinos `{ve, vm, vt}` and on any model with new chiral matter.

**Fix:** Nothing to fix, and nothing to see — both are off by default. The default set is `hermiticity`, `kinetic_diagonal`, `mass_diagonal`, `mass_spectrum`, each of which passes for the SM alone, so a failure from any of them is about your model. If you do want the other two, pass `checks="all"` and set `baseline_compare=true`: that runs the whole set against `LSM` alone as well and marks shared results as inherited, so `new_failures` still lists only what your model introduced. It costs a second pass of every check.

## P12. Wolfram 14+ needs FeynRules patched

**Symptom:** `SetDelayed::write ... is Protected`, then `Table::iterb: IndexRange[Index[Spin]] ...` and malformed `ISUMObject` names. The UFO export hangs or produces broken output.

**Cause:** Wolfram 14 made `Commutator` and `MatrixSymbol` Protected built-ins. FeynRules 2.3.49 (2021) redefines both.

**Fix:** Apply FeynRules upstream PR #6 (unprotect/reprotect around the redefinitions in `Core/ExtractVertexTools.m`), or run on Wolfram ≤ 13.x. The generator warns on ≥ 14 and defaults `add_decays` off there, since the 1→2 partial-width routine is also broken — let MadGraph compute widths from the UFO instead.

## P13. Diagonality checks fail for any model with a mixing angle

**Symptom:** `kinetic_diagonal` and `mass_diagonal` fail, and `mass_spectrum` comes back inconclusive, on a model that is correct and whose mass eigenstates are right. The detail names a cross term between two mass eigenstates proportional to a product of mixing-angle parameters:

```
Non diagonal kinetic term found: -(ch sh del[h1, mu] del[h2, mu])
Non diagonal mass term found:      ch h1 h2 sh (kappa xi^2 + ...)
```

**Cause:** The eigenstates are defined through an angle, `ch = Cos[thh]`, `sh = Sin[thh]`, with `thh` an external parameter. The cross terms vanish only when `thh` takes the value that diagonalises the mass matrix — a relation involving the potential parameters that is never written down anywhere FeynRules can see. Symbolically the terms are non-zero, so the check reports them; numerically, at the physical point, they cancel.

This is not a defect in the model. The published FeynRules implementation of the Hidden Abelian Higgs Model (`Hidden.fr`, C. Duhr) fails both checks for exactly this reason, and it is the reference implementation of a paper.

**Fix:** Nothing to fix in the model. Treat a diagonality failure whose detail is a product of mixing parameters as a false positive, and confirm it numerically instead: evaluate the cross term at the parameter point you care about and check it vanishes.

**Do not disable the whole check family over this.** The caveat covers `kinetic_diagonal` and `mass_diagonal`, which test symbolic diagonality. It does **not** cover `mass_spectrum`, which compares the declared mass of each state against the one the Lagrangian actually produces, numerically — a mixing angle cannot confuse a numeric comparison. An earlier version of this entry said to fall back to `checks="hermiticity"`, and an agent following that advice switched off `mass_spectrum` and shipped a model with a sign error in its scalar mass matrix: hermitian, cleanly compiled, right interactions, and a 125 GeV declaration against a 164 GeV Lagrangian pole. Hermiticity passes for plenty of wrong Lagrangians; `mass_spectrum` is the one check in the default set that catches a wrong mass.

The general shape, and P11 is the other instance: FeynRules' structural checks assume the Lagrangian is diagonal *symbolically*. Correct models routinely are not, because parameterising by an angle is how mixing is normally written.

## P14. Gauge is inherited silently from `SM.fr`

**Symptom:** The compiled UFO carries far more vertices than the model has interactions, and the extra ones involve fields nobody declared:

```
ghWp ghWp~ Zp        G0 G0 Hd        Zp G- G+        Gd Gd H
```

One model reviewed 97 "new physics" vertices this way, could not tell physics from bookkeeping because the inventory presented them intermixed, and discovered the gauge only after the fact. Its real split was 122 physical against 138 gauge artefacts.

**Cause:** `FeynmanGauge` is a plain global assigned inside `SM.fr` itself (`FeynmanGauge = True;`, the default there since FeynRules v1.4.2). An add-on loaded alongside it inherits Feynman gauge without deciding to, and inherits a Goldstone and ghost sector with it. Nothing in FeynRules announces this, and it is invisible in the `.fr` the author wrote.

Because `SM.fr`'s Lagrangians are all `SetDelayed`, they read `FeynmanGauge` when they are *used*, not when they are read — so the value in force is whatever it is at the moment the total Lagrangian is evaluated. `LHiggs` and `LYukawa` drop the Goldstones through their own `feynmangaugerules`, and `LGhost` collapses to the QCD ghost term, so one symbol governs the whole SM side.

**Fix:** Decide it explicitly. `FeynRulesToUFO` and `ValidateLagrangian` take `feynman_gauge`; setting it false gives unitary gauge, which is conventional for a tree-level UFO, and MadGraph ignores the Goldstone and ghost vertices either way. Measured on a bare `SM.fr` add-on, the total Lagrangian goes from 272 terms to 150.

Setting `FeynmanGauge = False` at the top of your own `.fr` does **not** work when `SM.fr` is loaded after it — `SM.fr` assigns the symbol itself and overwrites you. The tool applies the override after the load and before the Lagrangian is evaluated, which is the only window where it takes effect.

Whatever you choose, the gauge actually used now comes back in the result as `gauge`, and `vertex_inventory` splits its counts into `physical` and `gauge_artifact`, so this is visible rather than inferred.
