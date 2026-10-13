"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Generic Wolfram Language execution.

Runs arbitrary Wolfram Language / Mathematica code through a `wolframscript`
subprocess. This package is domain-neutral on purpose: it loads no Wolfram
packages, assumes nothing about what the code computes, and has no dependency
on any other tool bundle. A script that needs a package loads it itself.

Tools:
    RunWolframScript       — execute Wolfram Language code or a .wl file
    RunWolframScriptBatch  — execute several .wl files concurrently

Library:
    WolframRunner          — the subprocess manager behind both tools
    WolframResult          — the result record it returns
    resolve_wolframscript  — locate a usable wolframscript executable
    tidy_text_symbols      — default TeXForm \\text{...} -> subscript rewrite
    load_expression_from_sidecar
                           — read a result back out of a _results.json sidecar

Structured results are opt-in. A script that prints

    Print["SYMBOLIC_RESULT[name]: ", expr]
    Print["NUMERICAL_RESULT[name]: ", N[expr]]
    Print["LATEX_RESULT[name]: ", TeXForm[expr]]
    Print["STATUS: complete"]

gets those values parsed into the tool's JSON response and written to a
`<script>_results.json` sidecar. A script that prints nothing recognisable
still gets its raw stdout back.

Callers with their own symbol vocabulary can pass `latex_postprocess` to
WolframRunner to override the default typographic rewrite.

Requires only the standard library and the toolkit's BaseTool; notably NOT
sympy, so this bundle installs and imports on a bare environment.
"""

from .wolfram_runner import (
    WolframRunner,
    WolframResult,
    resolve_wolframscript,
    tidy_text_symbols,
)
from .run_wolfram_tool import RunWolframScript, RunWolframScriptBatch
from .result_utils import load_expression_from_sidecar

__all__ = [
    "RunWolframScript",
    "RunWolframScriptBatch",
    "WolframRunner",
    "WolframResult",
    "resolve_wolframscript",
    "tidy_text_symbols",
    "load_expression_from_sidecar",
]
