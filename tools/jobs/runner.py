"""
# runner.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Detached job runner: ``python -m tools.jobs.runner <job_dir>``.

Reads spec.json, runs the whitelisted tool, writes result.json and drives
status.json through running -> done | failed. Never trusts module/class
strings from the (user-writable) spec — the tool class is re-resolved from
the in-code WHITELIST. stdout/stderr already point at log.txt (the submitter
wired them), so plain prints and tracebacks land there.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from datetime import datetime, timezone

from tools.jobs.job_manager import (
    RESULT_SCHEMA,
    STATUS_SCHEMA,
    WHITELIST,
    read_json,
    resolve_tool,
    write_json_atomic,
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _set_status(jdir: str, **updates) -> None:
    path = os.path.join(jdir, "status.json")
    st = read_json(path) or {"schema": STATUS_SCHEMA}
    st.update(updates)
    write_json_atomic(path, st)


def _audit(spec: dict, status: str, summary: str, data: dict) -> None:
    """Best-effort audit event; never fails the job."""
    try:
        from tools.logging.audit import append_event

        append_event(
            spec["state"]["base_directory"],
            stage="job",
            status=status,
            summary=summary,
            data=data,
            ledger_name=spec.get("ledger_name") or "audit.json",
        )
    except Exception:  # noqa: BLE001
        pass


def main(jdir: str) -> int:
    spec = read_json(os.path.join(jdir, "spec.json"))
    if not spec or spec.get("tool") not in WHITELIST:
        _set_status(jdir, state="failed", finished=_now(),
                    error="invalid spec or non-whitelisted tool")
        return 2

    job_id, tool_name = spec["job_id"], spec["tool"]
    _set_status(jdir, state="running", pid=os.getpid(), started=_now())

    if spec.get("max_seconds"):
        def _kill() -> None:
            _set_status(jdir, state="failed", finished=_now(),
                        error=f"watchdog timeout after {spec['max_seconds']}s")
            os._exit(124)

        t = threading.Timer(float(spec["max_seconds"]), _kill)
        t.daemon = True
        t.start()

    try:
        cls = resolve_tool(tool_name)
        state = spec.get("state") or {}
        kwargs = {k: v for k, v in state.items() if k in cls.model_fields}
        # Instantiate with state + runtime args merged and call _run()
        # directly (the pattern the repo's own harnesses use): orchestral's
        # execute() re-checks "required" fields with default=None-means-
        # required semantics and would reject omitted optional args.
        # spec.json lives under the user-writable base_directory, so the
        # restriction submit() applied has to be applied again here rather than
        # trusted. Without it a spec carrying a StateField in "args" -- for
        # FeynRulesToUFOTool that includes wolframscript_path, which becomes
        # argv[0] of a subprocess -- would override the server-side config and
        # step around the whitelist this runner exists to enforce.
        spec_args = spec.get("args") or {}
        runtime = set(cls._get_runtime_fields())
        rejected = sorted(set(spec_args) - runtime)
        if rejected:
            _set_status(jdir, state="failed", finished=_now(),
                        error=f"spec args are not runtime fields: {', '.join(rejected)}")
            return 2
        kwargs.update(spec_args)
        tool = cls(**kwargs)
        output = tool._run()
        tool_error = isinstance(output, str) and output.startswith("Error:")
        write_json_atomic(
            os.path.join(jdir, "result.json"),
            {
                "schema": RESULT_SCHEMA,
                "job_id": job_id,
                "tool": tool_name,
                "tool_error": tool_error,
                "output": output,
            },
        )
        _set_status(jdir, state="done", exit_code=0, finished=_now())
        _audit(spec, "fail" if tool_error else "ok",
               f"{tool_name} job {job_id} finished",
               {"job_id": job_id, "tool": tool_name, "tool_error": tool_error})
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_status(jdir, state="failed", exit_code=1, finished=_now(),
                    error=f"{type(e).__name__}: {e}")
        _audit(spec, "error", f"{tool_name} job {job_id} crashed",
               {"job_id": job_id, "tool": tool_name, "error": str(e)[:300]})
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m tools.jobs.runner <job_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
