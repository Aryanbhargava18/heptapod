"""
# wolfram_runner.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Subprocess manager for wolframscript execution.

Runs Wolfram Language code via wolframscript, captures output, saves each
script as a standalone .wl file for reproducibility, and parses the optional
structured-result markers a script may print.

Domain-neutral by construction: nothing here loads or assumes any Wolfram
package. A script that wants one loads it itself.
"""

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Callable


@dataclass
class WolframResult:
    """Result from a wolframscript execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time_s: float
    script_path: str
    parsed_results: Dict[str, Any] = field(default_factory=dict)
    results_path: Optional[str] = None


# Markers the LLM can use in Print[] statements for structured output
RESULT_MARKERS = {
    "SYMBOLIC_RESULT": re.compile(r"SYMBOLIC_RESULT\[(.+?)\]:\s*(.+)"),
    "NUMERICAL_RESULT": re.compile(r"NUMERICAL_RESULT\[(.+?)\]:\s*(.+)"),
    "LATEX_RESULT": re.compile(r"LATEX_RESULT\[(.+?)\]:\s*(.+)"),
    "STATUS": re.compile(r"STATUS:\s*(.+)"),
}


def tidy_text_symbols(tex: str) -> str:
    r"""Turn TeXForm's ``\text{...}`` wrappers into subscript notation.

    Mathematica's ``TeXForm`` renders a bare multi-letter symbol as
    ``\text{ab}``, which typesets as upright text rather than as a
    subscripted variable.  This rewrites ``\text{Xy}`` to ``X_y`` and
    ``\text{Xyz}`` to ``X_{yz}``, which is what a symbol of the form
    "letter + qualifier" almost always means.

    This is a purely typographic rule with no notion of what the symbols
    stand for.  A caller that knows its own symbol vocabulary should pass
    a ``latex_postprocess`` callable to :class:`WolframRunner` instead;
    it runs in place of this default.
    """
    return re.sub(
        r"\\text\{([a-zA-Z])([a-zA-Z0-9]+)\}",
        lambda m: (
            f"{m.group(1)}_{{{m.group(2)}}}"
            if len(m.group(2)) > 1
            else f"{m.group(1)}_{m.group(2)}"
        ),
        tex,
    )


def _parse_structured_output(
    stdout: str,
    latex_postprocess: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Extract structured results from wolframscript stdout.

    A script opts in to structured output by embedding markers in its
    Print[] statements:
        Print["SYMBOLIC_RESULT[antiderivative]: ", result]
        Print["NUMERICAL_RESULT[root]: ", N[root]]
        Print["LATEX_RESULT[result]: ", TeXForm[result]]
        Print["STATUS: complete"]

    Markers are optional; a script that prints nothing recognisable
    still returns its raw stdout to the caller.

    ``latex_postprocess`` rewrites each LATEX_RESULT value, replacing
    the default :func:`tidy_text_symbols`.
    """
    clean_latex = latex_postprocess or tidy_text_symbols
    parsed = {"symbolic": {}, "numerical": {}, "latex": {}, "status": None}

    for line in stdout.splitlines():
        line = line.strip()

        m = RESULT_MARKERS["SYMBOLIC_RESULT"].match(line)
        if m:
            parsed["symbolic"][m.group(1)] = m.group(2)
            continue

        m = RESULT_MARKERS["NUMERICAL_RESULT"].match(line)
        if m:
            try:
                parsed["numerical"][m.group(1)] = float(m.group(2))
            except ValueError:
                parsed["numerical"][m.group(1)] = m.group(2)
            continue

        m = RESULT_MARKERS["LATEX_RESULT"].match(line)
        if m:
            parsed["latex"][m.group(1)] = clean_latex(m.group(2))
            continue

        m = RESULT_MARKERS["STATUS"].match(line)
        if m:
            parsed["status"] = m.group(1)

    return parsed


def _save_results_sidecar(script_path: str, parsed: Dict[str, Any]) -> Optional[str]:
    """Save parsed results as a JSON sidecar next to the .wl script.

    File is named ``{stem}_results.json`` in the same directory as the script.
    Returns the sidecar path on success, None on failure.
    """
    try:
        sp = Path(script_path)
        sidecar = sp.parent / f"{sp.stem}_results.json"
        # Only include non-empty result categories
        data = {}
        for key in ("symbolic", "numerical", "latex"):
            if parsed.get(key):
                data[key] = parsed[key]
        if parsed.get("status"):
            data["status"] = parsed["status"]
        data["script_path"] = str(sp)
        sidecar.write_text(json.dumps(data, indent=2))
        return str(sidecar)
    except Exception:
        return None


# The path config.example.py ships with. It means "not configured yet", so
# honouring it verbatim would turn every call into a "not found" failure on a
# machine that does have Wolfram installed.
_PLACEHOLDER_PATHS = {
    "/path/to/wolframscript",
    "path/to/wolframscript",
}


def _is_placeholder(cand: Optional[str]) -> bool:
    """True for an unset-by-another-name path."""
    if not cand or not str(cand).strip():
        return True
    return str(cand).strip() in _PLACEHOLDER_PATHS


def resolve_wolframscript(explicit: Optional[str] = None) -> str:
    """Best-effort location of a ``wolframscript`` executable.

    Resolution order, first hit wins:

      1. ``explicit``, verbatim, when it is set and not the config placeholder.
      2. ``config.wolframscript_path``, on the same terms.
      3. ``wolframscript`` on PATH.
      4. The default install locations for macOS and Linux.
      5. ``"wolframscript"``, so the caller still gets a subprocess-level
         error naming what was looked for.

    A real path is returned verbatim rather than existence-tested: a caller
    that names an executable has stated its intent, and a wrong path should
    fail loudly rather than be silently swapped for another install. Only the
    placeholder is treated as "unset", which is what it means.
    """
    if not _is_placeholder(explicit):
        return str(explicit).strip()

    try:
        import config
        configured = getattr(config, "wolframscript_path", None)
        if not _is_placeholder(configured):
            return str(configured).strip()
    except ImportError:
        pass

    found = shutil.which("wolframscript")
    if found:
        return found

    for cand in (
        "/Applications/Mathematica.app/Contents/MacOS/wolframscript",
        "/usr/local/bin/wolframscript",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    return "wolframscript"


class WolframRunner:
    """Runs Wolfram Language code via a ``wolframscript`` subprocess.

    Knows nothing about what the scripts compute: it writes code to a
    ``.wl`` file, runs it, and parses any structured markers the script
    chose to print.  Domain-specific behaviour is supplied by the caller
    through ``latex_postprocess``.
    """

    def __init__(
        self,
        wolframscript_path: str = None,
        timeout_sec: int = 120,
        latex_postprocess: Optional[Callable[[str], str]] = None,
    ):
        self.wolframscript_path = resolve_wolframscript(wolframscript_path)
        self.timeout_sec = timeout_sec
        self.latex_postprocess = latex_postprocess

    def check_available(self) -> tuple:
        """Verify ``wolframscript`` runs and can evaluate an expression.

        Says nothing about which Wolfram packages are installed; use
        :meth:`check_package` for that.

        Returns:
            (available: bool, message: str)
        """
        try:
            proc = subprocess.run(
                [self.wolframscript_path, "-code",
                 'Print["WolframScript " <> ToString[$VersionNumber]]'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
                text=True,
            )
        except FileNotFoundError:
            return (False, f"wolframscript not found at: {self.wolframscript_path}")
        except subprocess.TimeoutExpired:
            return (False, "wolframscript timed out on basic test")

        if proc.returncode != 0 or "WolframScript" not in proc.stdout:
            return (False, f"wolframscript failed: {proc.stderr.strip()}")
        return (True, proc.stdout.strip())

    def check_package(self, package: str, timeout_sec: int = 60) -> tuple:
        """Verify a named Wolfram package loads.

        Loads ``package`` (a context name without the backtick, as in
        ``check_package("SomePackage")``) and echoes back whatever it
        printed, which for most packages includes a version banner.

        Callers that depend on a particular package own that dependency;
        this method only reports whether it is there.

        Returns:
            (available: bool, message: str)
        """
        code = f'<< {package}`; Print["{package} loaded"]'
        try:
            proc = subprocess.run(
                [self.wolframscript_path, "-code", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
                text=True,
            )
        except FileNotFoundError:
            return (False, f"wolframscript not found at: {self.wolframscript_path}")
        except subprocess.TimeoutExpired:
            return (False, f"{package} loading timed out ({timeout_sec}s)")

        if proc.returncode != 0 or f"{package} loaded" not in proc.stdout:
            detail = (proc.stderr or proc.stdout or "").strip()
            return (False, f"{package} failed to load: {detail}")
        return (True, proc.stdout.strip())

    def run_script(
        self,
        script_path: str,
        working_dir: str = None,
    ) -> WolframResult:
        """Run a .wl script file via wolframscript.

        Args:
            script_path: Path to the .wl file.
            working_dir: Working directory for the subprocess.

        Returns:
            WolframResult with execution details.
        """
        script_path = str(Path(script_path).resolve())
        if not Path(script_path).exists():
            return WolframResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Script not found: {script_path}",
                execution_time_s=0.0,
                script_path=script_path,
            )

        cmd = [self.wolframscript_path, "-f", script_path]
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                timeout=self.timeout_sec,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return WolframResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Timeout after {self.timeout_sec}s",
                execution_time_s=time.monotonic() - t0,
                script_path=script_path,
            )
        except FileNotFoundError:
            return WolframResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"wolframscript not found: {self.wolframscript_path}",
                execution_time_s=time.monotonic() - t0,
                script_path=script_path,
            )

        elapsed = time.monotonic() - t0
        parsed = _parse_structured_output(
            proc.stdout or "", latex_postprocess=self.latex_postprocess
        )

        # Save results sidecar JSON alongside the .wl script
        results_path = None
        has_parsed = bool(
            parsed.get("symbolic") or parsed.get("numerical") or parsed.get("latex")
        )
        if proc.returncode == 0 and has_parsed:
            results_path = _save_results_sidecar(script_path, parsed)

        return WolframResult(
            success=(proc.returncode == 0),
            return_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            execution_time_s=elapsed,
            script_path=script_path,
            parsed_results=parsed,
            results_path=results_path,
        )

    def run_code(
        self,
        code: str,
        save_path: str = None,
        working_dir: str = None,
    ) -> WolframResult:
        """Write code to a .wl file and run it.

        Args:
            code: Mathematica code to execute.
            save_path: Where to save the .wl file. If None, auto-generates
                       a path in working_dir/scripts/.
            working_dir: Base directory for script output.

        Returns:
            WolframResult with execution details.
        """
        if save_path is None:
            scripts_dir = Path(working_dir or ".") / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            # Generate unique name
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            ts = int(time.time())
            save_path = str(scripts_dir / f"wolfram_{ts}_{code_hash}.wl")

        save_path = str(Path(save_path).resolve())
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(code)

        return self.run_script(save_path, working_dir=working_dir)
