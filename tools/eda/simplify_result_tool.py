"""
# simplify_result_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

SimplifyResult — post-processing tool for FeynCalc symbolic results.

After RunWolframScript produces a symbolic expression, this tool applies
algebraic manipulations (substitutions, limits, series expansion, simplification)
via a new wolframscript call. The result is saved to its own _results.json
sidecar so downstream tools (ConvertToPython) can chain from it.

Input modes (same dual pattern as ConvertToPython):
  - By reference: script_path + result_name
  - By value: expr (Mathematica InputForm string)

Transformation fields (all optional, applied in order):
  substitutions, limit, series, assumptions, simplify
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.wolfram import load_expression_from_sidecar
from tools.wolfram import WolframRunner
from .latex_symbols import clean_latex_symbols
from tools.logging.findings import append_finding


class SimplifyResult(BaseTool):
    """
    Simplify, substitute, or take limits of a Mathematica symbolic expression.

    Two input modes:

    1. **By reference** (preferred): provide ``script_path`` from a previous
       RunWolframScript call plus ``result_name`` (the key inside
       SYMBOLIC_RESULT[key]).

    2. **By value**: provide ``expr`` directly as a Mathematica InputForm string.

    Transformation fields (all optional, applied in order):

    - ``substitutions``: Dict[str, str] — replacement rules
      e.g. {"mfbar": "mf", "x": "0"} → expr /. {mfbar -> mf, x -> 0}
    - ``limit``: Dict[str, str] — take a limit
      e.g. {"var": "mf", "point": "0"} → Limit[expr, mf -> 0]
    - ``series``: Dict with "var", "point", "order" — series expansion
      e.g. {"var": "eps", "point": "0", "order": 2} → Normal[Series[...]]
    - ``assumptions``: List[str] — Mathematica assumptions
      e.g. ["MH > 0", "mb > 0"] → Assuming[{MH > 0, mb > 0}, ...]
    - ``simplify``: str — simplification function
      "Simplify" (default), "FullSimplify", "Factor", "Expand", "None"

    Returns:
        {
            "status": "ok",
            "script_path": "/path/to/simplify_*.wl",
            "results_path": "/path/to/simplify_*_results.json",
            "simplified": "<Mathematica InputForm>",
            "latex": "<LaTeX string>"
        }
    """

    # ======================== Runtime fields ======================== #
    script_path: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Path to a .wl script from RunWolframScript. "
            "The tool reads the _results.json sidecar to find the expression."
        ),
    )
    result_name: Optional[str] = RuntimeField(
        default=None,
        description="Key of the SYMBOLIC_RESULT to simplify (e.g., 'width').",
    )
    expr: Optional[str] = RuntimeField(
        default=None,
        description="Direct Mathematica InputForm string (alternative to script_path + result_name).",
    )
    substitutions: Optional[Dict[str, str]] = RuntimeField(
        default=None,
        description=(
            "Replacement rules as {old: new} pairs. "
            'e.g. {"mfbar": "mf", "x": "0"} → expr /. {mfbar -> mf, x -> 0}'
        ),
    )
    limit: Optional[Dict[str, str]] = RuntimeField(
        default=None,
        description=(
            'Limit specification: {"var": "mf", "point": "0"} → Limit[expr, mf -> 0]. '
            'Optional "direction" key for directional limits.'
        ),
    )
    series: Optional[Dict[str, Any]] = RuntimeField(
        default=None,
        description=(
            'Series expansion: {"var": "eps", "point": "0", "order": 2} '
            "→ Normal[Series[expr, {eps, 0, 2}]]"
        ),
    )
    assumptions: Optional[List[str]] = RuntimeField(
        default=None,
        description=(
            'Mathematica assumptions, e.g. ["MH > 0", "mb > 0"]. '
            "Applied via Assuming[{...}, ...]."
        ),
    )
    simplify: str = RuntimeField(
        default="Simplify",
        description=(
            'Simplification function: "Simplify" (default), "FullSimplify", '
            '"Factor", "Expand", or "None" to skip.'
        ),
    )
    script_name: Optional[str] = RuntimeField(
        default=None,
        description="Name for saved .wl script (without extension)",
    )
    timeout: int = RuntimeField(
        default=120,
        description="Timeout in seconds for wolframscript execution",
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Working directory for tool output"
    )
    # ================================================================ #

    def _run(self) -> str:
        # --- Resolve the expression ---
        has_ref = self.script_path and self.result_name
        has_expr = self.expr and self.expr.strip()

        if not has_ref and not has_expr:
            return self.format_error(
                error="Missing Parameter",
                reason=(
                    "Provide either (script_path + result_name) to reference a "
                    "RunWolframScript result, or expr as a direct Mathematica string."
                ),
            )

        if has_ref:
            expr_str, err = load_expression_from_sidecar(
                self.script_path, self.result_name, category="symbolic"
            )
            if err:
                return self.format_error(error="Results Not Found", reason=err)
        else:
            expr_str = self.expr.strip()

        # --- Generate Mathematica code ---
        code = self._generate_mathematica(expr_str)

        # --- Run via WolframRunner ---
        runner = WolframRunner(timeout_sec=self.timeout,
                               latex_postprocess=clean_latex_symbols)

        if self.script_name:
            name = self.script_name
            if not name.endswith(".wl"):
                name += ".wl"
        else:
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            name = f"simplify_{code_hash}.wl"

        scripts_dir = Path(self.base_directory) / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(scripts_dir / name)

        result = runner.run_code(
            code=code,
            save_path=save_path,
            working_dir=self.base_directory,
        )

        if not result.success:
            output = {
                "status": "error",
                "script_path": result.script_path,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            return json.dumps(output, indent=2)

        # --- Build output ---
        output: Dict[str, Any] = {
            "status": "ok",
            "script_path": result.script_path,
        }

        if result.results_path:
            output["results_path"] = result.results_path

        pr = result.parsed_results or {}
        if pr.get("symbolic", {}).get("simplified"):
            output["simplified"] = pr["symbolic"]["simplified"]
        if pr.get("latex", {}).get("simplified"):
            output["latex"] = pr["latex"]["simplified"]

        # Append findings (best-effort, lightweight index)
        try:
            ops = []
            if self.substitutions:
                ops.append(f"sub({','.join(self.substitutions.keys())})")
            if self.limit:
                ops.append(f"limit({self.limit.get('var','x')}→{self.limit.get('point','0')})")
            if self.series:
                ops.append(f"series({self.series.get('var','eps')},O({self.series.get('order',1)}))")
            if self.simplify and self.simplify != "None":
                ops.append(self.simplify)
            source = self.script_path.rsplit("/", 1)[-1] if self.script_path else "expr"
            entries = [f"Source: {source}"]
            if ops:
                entries.append(f"Ops: {' → '.join(ops)}")
            if output.get("results_path"):
                entries.append(f"Sidecar: {output['results_path']}")
            append_finding(self.base_directory, "SimplifyResult", entries)
        except Exception:
            pass

        return json.dumps(output, indent=2)

    def _generate_mathematica(self, expr_str: str) -> str:
        """Generate the Mathematica script for simplification."""
        lines = [
            "(* SimplifyResult — generated by HEPTAPOD/Diagrammatica *)",
            "",
            "(* Load the expression *)",
            f"expr = {expr_str};",
            "",
        ]

        # 1. Substitutions
        if self.substitutions:
            rules = ", ".join(
                f"{old} -> {new}" for old, new in self.substitutions.items()
            )
            lines.append(f"(* Apply substitutions *)")
            lines.append(f"expr = expr /. {{{rules}}};")
            lines.append("")

        # 2. Limit
        if self.limit:
            var = self.limit.get("var", "x")
            point = self.limit.get("point", "0")
            direction = self.limit.get("direction")
            if direction:
                lines.append(f"(* Take limit *)")
                lines.append(
                    f"expr = Limit[expr, {var} -> {point}, Direction -> {direction}];"
                )
            else:
                lines.append(f"(* Take limit *)")
                lines.append(f"expr = Limit[expr, {var} -> {point}];")
            lines.append("")

        # 3. Series expansion
        if self.series:
            var = self.series.get("var", "eps")
            point = self.series.get("point", "0")
            order = self.series.get("order", 1)
            lines.append(f"(* Series expansion *)")
            lines.append(
                f"expr = Normal[Series[expr, {{{var}, {point}, {order}}}]];"
            )
            lines.append("")

        # 4. Simplification (with optional assumptions)
        simplify_fn = self.simplify or "Simplify"
        if simplify_fn != "None":
            if self.assumptions:
                assumptions_str = ", ".join(self.assumptions)
                lines.append(f"(* Simplify with assumptions *)")
                lines.append(
                    f"expr = Assuming[{{{assumptions_str}}}, {simplify_fn}[expr]];"
                )
            else:
                lines.append(f"(* Simplify *)")
                lines.append(f"expr = {simplify_fn}[expr];")
            lines.append("")

        # 5. Output markers
        lines.extend([
            '(* Output *)',
            'Print["SYMBOLIC_RESULT[simplified]: ", expr];',
            'Print["LATEX_RESULT[simplified]: ", ToString[TeXForm[expr]]];',
            'Print["STATUS: complete"];',
        ])

        return "\n".join(lines)


class SimplifyResultBatch(BaseTool):
    """
    Simplify multiple expressions concurrently.

    Takes a list of simplification specs and runs them all in parallel via a
    thread pool, returning all results in one call. Use this instead of
    multiple sequential SimplifyResult calls for parameter scans and limits.

    Each spec in the list follows the same schema as SimplifyResult:
        {
            "script_path": "/path/to/script.wl",
            "result_name": "width",
            "substitutions": {"mfbar": "mf"},
            "limit": {"var": "mf", "point": "0"},
            "simplify": "Simplify",
            "script_name": "simplify_massless_limit"
        }

    Or by-value:
        {
            "expr": "mS^2/(8 Pi)",
            "simplify": "FullSimplify",
            "script_name": "simplify_scalar"
        }

    Returns:
        {
            "total": N, "succeeded": M, "failed": K,
            "total_time_s": T,
            "results": [...]
        }
    """

    # --- Runtime fields ---
    specs: List[Dict[str, Any]] = RuntimeField(
        description=(
            "List of simplification specs. Each spec is a dict with the same "
            "keys as SimplifyResult: script_path + result_name (or expr), "
            "plus optional substitutions, limit, series, assumptions, "
            "simplify, script_name, timeout."
        ),
    )
    max_concurrent: int = RuntimeField(
        default=4,
        description="Maximum number of concurrent wolframscript processes",
    )
    timeout: int = RuntimeField(
        default=120,
        description="Default timeout in seconds per script (overridable per spec)",
    )

    # --- State fields ---
    base_directory: str = StateField(
        description="Working directory for tool output"
    )

    def _run(self) -> str:
        if not self.specs:
            return self.format_error(
                error="Missing Parameter",
                reason="Provide 'specs' with at least one simplification spec",
            )

        t0 = time.monotonic()
        results_by_index: dict[int, dict] = {}

        def _run_one(idx: int, spec: dict) -> tuple[int, dict]:
            return idx, self._run_single_spec(spec)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(_run_one, i, spec): i
                for i, spec in enumerate(self.specs)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results_by_index[idx] = result

        total_time = time.monotonic() - t0

        # Build output in input order
        result_list = []
        succeeded = 0
        failed = 0
        for i in range(len(self.specs)):
            entry = results_by_index[i]
            result_list.append(entry)
            if entry.get("status") == "ok":
                succeeded += 1
            else:
                failed += 1

        output = {
            "total": len(self.specs),
            "succeeded": succeeded,
            "failed": failed,
            "total_time_s": round(total_time, 1),
            "results": result_list,
        }
        return json.dumps(output, indent=2)

    def _run_single_spec(self, spec: dict) -> dict:
        """Run a single simplification spec and return a result dict."""
        # Resolve the expression
        script_path = spec.get("script_path")
        result_name = spec.get("result_name")
        expr = spec.get("expr")

        has_ref = script_path and result_name
        has_expr = expr and str(expr).strip()

        if not has_ref and not has_expr:
            return {
                "status": "error",
                "error": "Provide (script_path + result_name) or expr",
            }

        if has_ref:
            expr_str, err = load_expression_from_sidecar(
                script_path, result_name, category="symbolic"
            )
            if err:
                return {"status": "error", "error": err}
        else:
            expr_str = str(expr).strip()

        # Build the Mathematica script using the same logic as SimplifyResult
        substitutions = spec.get("substitutions")
        limit = spec.get("limit")
        series = spec.get("series")
        assumptions = spec.get("assumptions")
        simplify_fn = spec.get("simplify", "Simplify")
        spec_timeout = spec.get("timeout", self.timeout)
        spec_script_name = spec.get("script_name")

        code = self._generate_mathematica(
            expr_str, substitutions, limit, series, assumptions, simplify_fn
        )

        # Determine script name
        if spec_script_name:
            name = spec_script_name
            if not name.endswith(".wl"):
                name += ".wl"
        else:
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            name = f"simplify_{code_hash}.wl"

        scripts_dir = Path(self.base_directory) / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(scripts_dir / name)

        runner = WolframRunner(timeout_sec=spec_timeout,
                               latex_postprocess=clean_latex_symbols)
        result = runner.run_code(
            code=code,
            save_path=save_path,
            working_dir=self.base_directory,
        )

        if not result.success:
            err_output: Dict[str, Any] = {
                "status": "error",
                "script_path": result.script_path,
            }
            if result.stderr and result.stderr.strip():
                err_output["stderr"] = result.stderr
            err_output["error_hint"] = (
                f"wolframscript exited with code {result.return_code}"
                if not result.stderr or not result.stderr.strip()
                else result.stderr.strip().splitlines()[-1]
            )
            return err_output

        output: Dict[str, Any] = {
            "status": "ok",
            "script_path": result.script_path,
        }
        if result.results_path:
            output["results_path"] = result.results_path

        pr = result.parsed_results or {}
        if pr.get("symbolic", {}).get("simplified"):
            output["simplified"] = pr["symbolic"]["simplified"]
        if pr.get("latex", {}).get("simplified"):
            output["latex"] = pr["latex"]["simplified"]

        # Append findings (best-effort)
        try:
            source = (script_path or "").rsplit("/", 1)[-1] if script_path else "expr"
            append_finding(
                self.base_directory,
                f"SimplifyBatch: {spec_script_name or source}",
                [f"Script: {result.script_path}"],
            )
        except Exception:
            pass

        return output

    @staticmethod
    def _generate_mathematica(
        expr_str: str,
        substitutions: Optional[Dict[str, str]] = None,
        limit: Optional[Dict[str, str]] = None,
        series: Optional[Dict[str, Any]] = None,
        assumptions: Optional[List[str]] = None,
        simplify_fn: str = "Simplify",
    ) -> str:
        """Generate the Mathematica script for a single simplification."""
        lines = [
            "(* SimplifyResultBatch — generated by HEPTAPOD/Diagrammatica *)",
            "",
            "(* Load the expression *)",
            f"expr = {expr_str};",
            "",
        ]

        if substitutions:
            rules = ", ".join(
                f"{old} -> {new}" for old, new in substitutions.items()
            )
            lines.append("(* Apply substitutions *)")
            lines.append(f"expr = expr /. {{{rules}}};")
            lines.append("")

        if limit:
            var = limit.get("var", "x")
            point = limit.get("point", "0")
            direction = limit.get("direction")
            lines.append("(* Take limit *)")
            if direction:
                lines.append(
                    f"expr = Limit[expr, {var} -> {point}, Direction -> {direction}];"
                )
            else:
                lines.append(f"expr = Limit[expr, {var} -> {point}];")
            lines.append("")

        if series:
            var = series.get("var", "eps")
            point = series.get("point", "0")
            order = series.get("order", 1)
            lines.append("(* Series expansion *)")
            lines.append(
                f"expr = Normal[Series[expr, {{{var}, {point}, {order}}}]];"
            )
            lines.append("")

        if simplify_fn and simplify_fn != "None":
            if assumptions:
                assumptions_str = ", ".join(assumptions)
                lines.append("(* Simplify with assumptions *)")
                lines.append(
                    f"expr = Assuming[{{{assumptions_str}}}, {simplify_fn}[expr]];"
                )
            else:
                lines.append("(* Simplify *)")
                lines.append(f"expr = {simplify_fn}[expr];")
            lines.append("")

        lines.extend([
            "(* Output *)",
            'Print["SYMBOLIC_RESULT[simplified]: ", expr];',
            'Print["LATEX_RESULT[simplified]: ", ToString[TeXForm[expr]]];',
            'Print["STATUS: complete"];',
        ])

        return "\n".join(lines)
