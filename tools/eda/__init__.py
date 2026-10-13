"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Agentic Diagrammatica — exact tree-level calculations via FeynCalc.

Tools:
    RunWolframScript          — re-exported from tools.wolfram (generic runner)
    RunWolframScriptBatch     — re-exported from tools.wolfram (generic runner)
    ComputeSymbolicAmplitude  — generate FeynCalc code from a SymbolicDiagram
    ConvertToPython           — convert FeynCalc SYMBOLIC_RESULT to Python function
    SimplifyResult            — simplify, substitute, or take limits of symbolic results
    SimplifyResultBatch       — simplify multiple expressions concurrently
    LookupTheory              — navigate the QFT skills graph (theory knowledge base)

Code generation:
    FeynCalcCodeGenerator — Diagram -> runnable FeynCalc script
    GeneratedCode         — container for generated code + metadata
    ProcessType           — enum of supported process topologies

Utilities:
    symbolic_to_python    — convert Mathematica SYMBOLIC_RESULT to Python callable
"""

# Wolfram execution is generic and lives in tools.wolfram; re-exported here
# so existing `from tools.eda import RunWolframScript` callers keep working.
from tools.wolfram import (
    RunWolframScript,
    RunWolframScriptBatch,
    load_expression_from_sidecar,
)
from .feyncalc_codegen import FeynCalcCodeGenerator, SymbolicFeynCalcCodeGenerator, GeneratedCode, ProcessType
from .compute_symbolic_amplitude_tool import ComputeSymbolicAmplitude
from .convert_to_python_tool import ConvertToPython
from .simplify_result_tool import SimplifyResult, SimplifyResultBatch
from .symbolic_to_python import (
    mathematica_to_sympy,
    mathematica_to_callable,
    mathematica_to_python_source,
    extract_variables,
)

__all__ = [
    "RunWolframScript",
    "RunWolframScriptBatch",
    "FeynCalcCodeGenerator",
    "SymbolicFeynCalcCodeGenerator",
    "GeneratedCode",
    "ProcessType",
    "ComputeSymbolicAmplitude",
    "ConvertToPython",
    "SimplifyResult",
    "SimplifyResultBatch",
    "load_expression_from_sidecar",
    "mathematica_to_sympy",
    "mathematica_to_callable",
    "mathematica_to_python_source",
    "extract_variables",
]

# Deferred import — LookupTheory requires the theory/ subpackage
try:
    from .theory.skills_graph_tool import LookupTheory
    __all__.append("LookupTheory")
except ImportError:
    pass
