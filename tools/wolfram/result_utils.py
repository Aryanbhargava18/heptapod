"""
# result_utils.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Loading expressions back out of RunWolframScript result sidecars.

The _results.json sidecar is saved alongside each .wl script by WolframRunner
and contains parsed SYMBOLIC_RESULT / NUMERICAL_RESULT / LATEX_RESULT values.
This is how a downstream tool refers to an earlier result by reference
(script_path + result_name) instead of having it pasted back in.
"""

import json
from pathlib import Path
from typing import Optional, Tuple


def load_expression_from_sidecar(
    script_path: str,
    result_name: str,
    category: str = "symbolic",
) -> Tuple[Optional[str], Optional[str]]:
    """Load an expression from a RunWolframScript _results.json sidecar.

    Args:
        script_path: Path to the .wl script file.
        result_name: Key of the result to load, i.e. the NAME in the
                  script's Print["SYMBOLIC_RESULT[NAME]: ", ...] marker.
        category: Which result category to search — "symbolic" (default),
                  "numerical", or "latex".

    Returns:
        (expression_string, error_message) — one of the two will be None.
    """
    sp = Path(script_path)
    sidecar = sp.parent / f"{sp.stem}_results.json"

    if not sidecar.exists():
        return None, (
            f"No results sidecar at {sidecar}. "
            "Run the script with RunWolframScript first, or "
            "provide the expression directly via the 'expr' parameter."
        )

    try:
        data = json.loads(sidecar.read_text())
    except Exception as e:
        return None, f"Failed to read {sidecar}: {e}"

    results = data.get(category, {})
    if result_name not in results:
        available = list(results.keys())
        return None, (
            f"No {category.upper()}_RESULT[{result_name}] in {sidecar}. "
            f"Available {category} results: {available}"
        )

    value = str(results[result_name])

    # Strip InputForm[...] wrapper if present — older SimplifyResult sidecars
    # stored expressions wrapped in InputForm which breaks downstream chaining.
    if value.startswith("InputForm[") and value.endswith("]"):
        value = value[len("InputForm["):-1]

    # Guard against $Failed — this is a Wolfram error symbol that propagates
    # silently through arithmetic (e.g. $Failed - $Failed simplifies to 0).
    if value.strip() == "$Failed" or value.strip() == "Null":
        return None, (
            f"{category.upper()}_RESULT[{result_name}] in {sidecar} is "
            f"'{value.strip()}' — the upstream computation failed. "
            f"Check the script that produced this sidecar for errors."
        )

    return value, None
