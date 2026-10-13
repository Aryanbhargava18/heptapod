# Lagrangian extraction: paper to a checked BSM model

Read a BSM paper, write a FeynRules model from it, prove the tool chain
accepts the model, then have an independent agent read the model back and
compare it against the paper.

```
 arxiv              the agent        frgen          feynrules          reverse
 ─────              ─────────        ─────          ─────────          ───────
 arXiv search  ->   paper text  ->   .fr file  ->   checks + UFO  ->   REVIEW.pdf
 LaTeX source       structured       FeynRules      MadGraph           physicist
                    spec             source         import             sign-off
```

Each tool stage is a separate bundle, so you install only what you need.
Only `feynrules` needs Mathematica. The calling agent writes the structured
spec itself, and the `FeynRulesModel` schema checks it.

## Run it

[`minimal_model.py`](minimal_model.py) runs the `frgen` stage. It needs no
Mathematica, no LLM and no network:

```bash
python examples/lagrangian_extraction/minimal_model.py
```

It builds a scalar leptoquark S1 and writes `sandbox/S1_LQ_minimal.fr`. S1 is
a colour triplet, weak singlet, charge −1/3 scalar with one new Yukawa
coupling.

The model is small on purpose. It has exactly one new field, one new
parameter and one new interaction, so you can read the generated `.fr` top to
bottom and match every line back to the spec in the script.

## The other stages

**Get the paper.** Prefer the LaTeX source over the PDF, because PDF text
extraction flattens fractions and roots:

```python
ArxivSearchTool(query="scalar leptoquark")
ArxivSourceTool(arxiv_id="1603.04993")     # LaTeX, not PDF
```

**Check the Lagrangian.** With `feynrules_path` and `wolframscript_path` set,
run the FeynRules consistency checks first. They do not write a UFO, so they
are the fast loop while you edit the `.fr`:

```python
ValidateLagrangianTool(model_path="S1_LQ_minimal.fr", lagrangian="LSM + LBSM")
```

The tools never guess the Lagrangian. Pass the full expression. `LSM` is
FeynRules' Standard Model Lagrangian, and `LBSM` is the total this model
defines. If the model's own total already contains `LSM`, do not add it
again.

A non-Hermitian Lagrangian is the defect the checks catch most often. That is
why the example writes the Yukawa term twice: once bare, then added to its own
`HC[...]`.

**Build and check the UFO.** Then compile, confirm every declared particle
reached the UFO, and import it into MadGraph:

```python
ValidateUFOModelTool(model_path="S1_LQ_minimal.fr", lagrangian="LSM + LBSM",
                     madgraph_check=True)
```

**Check the physics.** The checks prove the tool chain accepts the model. They
do not prove the model matches the paper. `ReverseLagrangianTool` tests that
separately: an agent that never saw the paper reconstructs the physics from
the `.fr` alone, and a second agent grades the reconstruction against the
paper term by term. See [`tools/reverse/README.md`](../../tools/reverse/README.md).

```python
ReverseLagrangianTool(model_path="S1_LQ_minimal.fr", action="full")
```

**Run the slow stages in the background.** UFO compiles and reverse checks
take minutes, which is longer than an agent should block for. The `jobs`
bundle detaches them:

```python
SubmitJobTool(
    tool_name="validateufomodel",
    tool_args='{"model_path": "S1_LQ_minimal.fr", "lagrangian": "LSM + LBSM"}',  # JSON string
)
JobStatusTool(job_id=...)
JobResultTool(job_id=...)
```

## What to install

| bundle | needs | gives you |
|---|---|---|
| `arxiv` | nothing beyond the base install | arXiv search, LaTeX source, PDF |
| `frgen` | jinja2 | spec to `.fr` |
| `feynrules` | FeynRules, wolframscript | consistency checks, `.fr` to UFO, MadGraph import |
| `reverse` | an agent CLI (`blank_agent_cmd`) | blank-slate review package |
| `jobs` | nothing | background execution |
| `logging` | nothing | `audit.json` provenance ledger |

```bash
tb install heptapod --bundle frgen --bundle arxiv
```

## A caution

Passing the checks means the tool chain accepts the model. It does not mean
the physics is right, and the tooling never claims otherwise. The reverse
check ends in a sign-off block for a physicist, not a verdict.
