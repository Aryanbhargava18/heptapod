"""
# convert_to_python_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

ConvertToPython — convert a FeynCalc SYMBOLIC_RESULT to a Python function.

After RunWolframScript produces symbolic results, this tool converts them
into portable Python source code.  The agent can reference the result by
script_path + result_name (reads the _results.json sidecar) or pass the
Mathematica expression string directly.
"""

import json
from typing import Optional, Dict, Any, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .symbolic_to_python import (
    mathematica_to_python_source,
    mathematica_to_callable,
    extract_variables,
)
from tools.wolfram import load_expression_from_sidecar
from tools.logging.findings import append_finding


class ConvertToPython(BaseTool):
    """
    Convert a Mathematica symbolic expression to a Python function.

    Two input modes:

    1. **By reference** (preferred): provide ``script_path`` from a previous
       RunWolframScript call plus ``result_name`` (the key inside
       SYMBOLIC_RESULT[key]).  The tool reads the ``_results.json`` sidecar
       saved alongside the .wl script.

    2. **By value**: provide ``expr`` directly as a Mathematica InputForm
       string (e.g., ``"g^2*M/(48*Pi)"``).

    In both cases the tool returns Python source code defining a function.
    Optionally provide ``values`` to evaluate the function at a point.

    **Multi-result mode**: provide ``result_names`` (a list) instead of
    ``result_name`` to convert multiple results from the same sidecar into
    a single Python module with one function per result.

    Inputs (runtime):
        script_path:   Path to the .wl script (from RunWolframScript output).
                       Used to locate the _results.json sidecar.
        result_name:   Key of the SYMBOLIC_RESULT to convert (e.g., "width").
                       Required when using script_path in single-result mode.
        result_names:  List of SYMBOLIC_RESULT keys to convert (e.g.,
                       ["width", "ampSq"]). Each becomes a function in the
                       generated Python module. Requires script_path.
        expr:          Direct Mathematica InputForm string.  Use this OR
                       script_path+result_name, not both.
        variables:     Ordered list of variable names for the Python function
                       signature.  If omitted, auto-extracted and sorted
                       alphabetically.
        function_name: Name for the generated Python function (default "f").
        values:        Optional dict mapping variable names to numerical values
                       for immediate evaluation.

    Returns:
        {
            "status": "ok",
            "python_source": "import math\\n\\ndef width(MH, mb, yb):\\n    ...",
            "variables": ["MH", "mb", "yb"],
            "expression": "<original Mathematica expression>",
            "evaluated": 0.00214   // only if values provided
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
        description="Key of the SYMBOLIC_RESULT to convert (e.g., 'width').",
    )
    result_names: Optional[List[str]] = RuntimeField(
        default=None,
        description=(
            "List of SYMBOLIC_RESULT keys to convert (e.g., ['width', 'ampSq']). "
            "Requires script_path. Each result becomes a separate function in the "
            "generated Python module. Use this OR result_name, not both."
        ),
    )
    expr: Optional[str] = RuntimeField(
        default=None,
        description="Direct Mathematica InputForm string (alternative to script_path + result_name).",
    )
    variables: Optional[List[str]] = RuntimeField(
        default=None,
        description=(
            "Ordered list of variable names for the function signature. "
            "If omitted, auto-extracted from the expression."
        ),
    )
    function_name: str = RuntimeField(
        default="f",
        description="Name for the generated Python function.",
    )
    values: Optional[Dict[str, float]] = RuntimeField(
        default=None,
        description="Variable values for immediate evaluation (e.g., {\"MH\": 125.0, \"mb\": 4.18}).",
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Working directory for tool output"
    )
    # ================================================================ #

    def _run(self) -> str:
        # --- Multi-result mode ---
        if self.result_names:
            return self._run_multi()

        # --- Single-result mode ---
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
            expr_str, err = self._load_from_sidecar()
            if err:
                return err
        else:
            expr_str = self.expr.strip()

        # --- Determine variables ---
        if self.variables:
            var_list = list(self.variables)
        else:
            try:
                var_list = extract_variables(expr_str)
            except ValueError as e:
                return self.format_error(
                    error="Parse Error",
                    reason=str(e),
                )

        if not var_list:
            return self.format_error(
                error="No Variables",
                reason=(
                    "Expression contains no free variables. "
                    "It may be a pure number — no Python function needed."
                ),
            )

        # --- Generate Python source ---
        try:
            source = mathematica_to_python_source(
                expr_str, var_list, self.function_name,
                dimension_comment=True,
            )
        except ValueError as e:
            return self.format_error(
                error="Conversion Error",
                reason=str(e),
            )

        output: Dict[str, Any] = {
            "status": "ok",
            "python_source": source,
            "variables": var_list,
            "expression": expr_str,
        }

        # --- Dimensional info (structured) ---
        try:
            from .dim_analysis import infer_mass_dimension, compute_expression_dimension
            from .symbolic_to_python import mathematica_to_sympy, _preprocess_conjugate
            from fractions import Fraction

            expr_sympy = _preprocess_conjugate(mathematica_to_sympy(expr_str))
            var_dims = {v: infer_mass_dimension(v) for v in var_list}
            expr_dim = compute_expression_dimension(expr_sympy, var_dims)

            dim_info: Dict[str, Any] = {"variables": {}}
            for v in var_list:
                d = var_dims[v]
                dim_info["variables"][v] = d if d is not None else "unknown"
            if expr_dim is not None:
                dim_info["result"] = int(expr_dim) if expr_dim.denominator == 1 else str(expr_dim)
            else:
                dim_info["result"] = "unknown"
            output["dimensions"] = dim_info
        except Exception:
            pass  # best-effort

        # --- Optional evaluation ---
        if self.values:
            eval_result, err = self._evaluate(expr_str, var_list)
            if err:
                output["evaluation_error"] = err
            else:
                output["evaluated"] = eval_result

        # Append findings (best-effort, lightweight index)
        try:
            entries = [
                f"Function: {self.function_name}({', '.join(var_list)})",
                f"Variables: {', '.join(var_list)}",
            ]
            if self.values and "evaluated" in output:
                entries.append(f"Evaluated: {output['evaluated']}")
            append_finding(self.base_directory, f"ConvertToPython: {self.function_name}", entries)
        except Exception:
            pass

        return json.dumps(output, indent=2)

    def _run_multi(self) -> str:
        """Convert multiple results from the same sidecar into a Python module."""
        if not self.script_path:
            return self.format_error(
                error="Missing Parameter",
                reason="Multi-result mode (result_names) requires script_path.",
            )

        functions = []
        errors = []

        for rname in self.result_names:
            expr_str, err = load_expression_from_sidecar(
                self.script_path, rname, category="symbolic"
            )
            if err:
                errors.append({"result_name": rname, "error": err})
                continue

            # Determine variables
            try:
                var_list = extract_variables(expr_str)
            except ValueError as e:
                errors.append({"result_name": rname, "error": str(e)})
                continue

            if not var_list:
                errors.append({"result_name": rname, "error": "No free variables"})
                continue

            # Generate source for this function
            try:
                source = mathematica_to_python_source(
                    expr_str, var_list, rname,
                    dimension_comment=True,
                )
                functions.append({
                    "name": rname,
                    "source": source,
                    "variables": var_list,
                    "expression": expr_str,
                })
            except ValueError as e:
                errors.append({"result_name": rname, "error": str(e)})

        if not functions:
            return self.format_error(
                error="All Conversions Failed",
                reason=json.dumps(errors),
            )

        # Assemble into a single Python module
        module_source = self._assemble_module(functions)

        output: Dict[str, Any] = {
            "status": "ok",
            "python_source": module_source,
            "functions": [
                {
                    "name": f["name"],
                    "variables": f["variables"],
                    "expression": f["expression"],
                }
                for f in functions
            ],
        }
        if errors:
            output["errors"] = errors

        # Append findings (best-effort)
        try:
            func_names = [f["name"] for f in functions]
            append_finding(
                self.base_directory,
                f"ConvertToPython: module ({len(functions)} functions)",
                [f"Functions: {', '.join(func_names)}"],
            )
        except Exception:
            pass

        return json.dumps(output, indent=2)

    @staticmethod
    def _assemble_module(functions: list) -> str:
        """Combine individually generated function sources into one module."""
        header_lines = ['"""Auto-generated by HEPTAPOD/ConvertToPython."""', "", "import math"]

        # Check if any function needs the conjugate helper
        needs_conjugate = any("conjugate" in f["source"] for f in functions)
        if needs_conjugate:
            header_lines.append("")
            header_lines.append("conjugate = lambda x: complex(x).conjugate()")

        body_lines = []
        for f in functions:
            # Strip the import/conjugate header from each individual source
            func_lines = []
            in_header = True
            for line in f["source"].splitlines():
                if in_header:
                    if (line.startswith("import math")
                            or line.startswith("conjugate =")
                            or line.strip() == ""):
                        continue
                    in_header = False
                func_lines.append(line)
            body_lines.append("")  # blank line before each function
            body_lines.extend(func_lines)

        return "\n".join(header_lines + body_lines) + "\n"

    def _load_from_sidecar(self) -> tuple:
        """Load expression from the _results.json sidecar.

        Returns:
            (expr_str, error_json) — one of the two will be None.
        """
        expr_str, err = load_expression_from_sidecar(
            self.script_path, self.result_name, category="symbolic"
        )
        if err:
            return None, self.format_error(
                error="Results Not Found",
                reason=err,
            )
        return expr_str, None

    def _evaluate(self, expr_str: str, var_list: list) -> tuple:
        """Evaluate the expression at the given values.

        Returns:
            (result_float, error_string) — one will be None.
        """
        # Check all variables have values
        missing = [v for v in var_list if v not in self.values]
        if missing:
            return None, f"Missing values for variables: {missing}"

        try:
            fn = mathematica_to_callable(expr_str, var_list)
            args = [self.values[v] for v in var_list]
            result = fn(*args)
            return float(result), None
        except Exception as e:
            return None, str(e)
