#!/usr/bin/env python3
"""
# test_jobs.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the background-job layer.

All tests are offline (no Wolfram / MG5 / network): validation failures are
exercised against the real ValidateUFOModelTool class (import only), and the full
submit -> detached runner -> result round-trip uses the dependency-free
AuditTrailTool. The detached subprocess is real.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools.jobs.job_manager as jm  # noqa: E402
from tools.jobs.jobs_tool import (  # noqa: E402
    JobResultTool,
    JobStatusTool,
    SubmitJobTool,
)

_TMPDIRS = []


def _base() -> str:
    d = tempfile.mkdtemp(prefix="jobs_test_")
    _TMPDIRS.append(d)
    return d


def cleanup_test_files() -> None:
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def _wait(base: str, jid: str, seconds: float = 15.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        st = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
        if st.get("state") in ("done", "failed"):
            return st
        time.sleep(0.2)
    return st


def test_job_id_format() -> bool:
    ids = {jm.new_job_id() for _ in range(50)}
    ok = len(ids) == 50 and all(jm.JOB_ID_RE.match(i) for i in ids)
    print(f"[{'✓' if ok else '✗'}] test_job_id_format")
    return ok


def test_validate_args_against_real_tool() -> bool:
    from tools.feynrules.validate_tool import ValidateUFOModelTool

    ok = True
    # unknown key rejected (state-field smuggling)
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "x", "base_directory": "/x"})
    ok &= err is not None and "unknown args" in err
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "x", "wolframscript_path": "/x"})
    ok &= err is not None and "unknown args" in err
    # valid passes (missing-with-default args are the tool's concern at run time)
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "m.fr", "madgraph_check": True,
                                                  "lagrangian": "LSM + LBSM"})
    ok &= err is None
    print(f"[{'✓' if ok else '✗'}] test_validate_args_against_real_tool")
    return ok


def test_whitelist_names_match_tools() -> bool:
    """Each whitelist key is the name the agent sees for that tool.

    An agent submits the name it was shown, which orchestral derives from the
    class name. A key that drifts from it makes the tool unreachable as a job.
    """
    import importlib

    ok = True
    for key, (module, cls_name) in jm.WHITELIST.items():
        cls = getattr(importlib.import_module(module), cls_name)
        if cls.get_name() != key:
            print(f"    whitelist key {key!r} != tool name {cls.get_name()!r}")
            ok = False
    print(f"[{'✓' if ok else '✗'}] test_whitelist_names_match_tools")
    return ok


def test_submit_rejections() -> bool:
    base = _base()
    ok = True
    r = SubmitJobTool(base_directory=base, tool_name="nope", tool_args="{}")._run()
    ok &= r.startswith("Error:") and "whitelist" in r
    r = SubmitJobTool(base_directory=base, tool_name="audittrail", tool_args="[1,2]")._run()
    ok &= r.startswith("Error:")
    r = SubmitJobTool(base_directory=base, tool_name="audittrail", tool_args="not json")._run()
    ok &= r.startswith("Error:")
    print(f"[{'✓' if ok else '✗'}] test_submit_rejections")
    return ok


def test_detached_round_trip() -> bool:
    base = _base()
    args = {"action": "record", "stage": "t", "status": "ok",
            "summary": "round trip", "data": "{}"}
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps(args))._run())
    st = _wait(base, out["job_id"])
    res_raw = JobResultTool(base_directory=base, job_id=out["job_id"])._run()
    ok = st.get("state") == "done" and not res_raw.startswith("Error:")
    if ok:
        res = json.loads(res_raw)
        ok &= res["tool_error"] is False and '"status": "ok"' in res["output"]
        # audit events (job_submit + job): the runner appends its event just
        # AFTER flipping status to done, so retry briefly.
        stages: list = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ledger = jm.read_json(os.path.join(base, "audit.json")) or {"events": []}
            stages = [e["stage"] for e in ledger["events"]]
            if "job" in stages:
                break
            time.sleep(0.1)
        ok &= "job_submit" in stages and "job" in stages
    print(f"[{'✓' if ok else '✗'}] test_detached_round_trip")
    return ok


def test_result_before_done_and_bad_ids() -> bool:
    base = _base()
    ok = True
    r = JobResultTool(base_directory=base, job_id="job_00000000_000000_abcdef")._run()
    ok &= "Job Not Found" in r
    r = JobStatusTool(base_directory=base, job_id="../../etc")._run()
    ok &= "Job Not Found" in r
    r = JobResultTool(base_directory=base, job_id="../../etc")._run()
    ok &= "Job Not Found" in r
    print(f"[{'✓' if ok else '✗'}] test_result_before_done_and_bad_ids")
    return ok


def test_stale_running_detection() -> bool:
    base = _base()
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps({"action": "record", "stage": "t",
                              "summary": "s", "data": "{}"}))._run())
    jid = out["job_id"]
    _wait(base, jid)
    jd = jm.job_dir(base, jid)
    st = jm.read_json(os.path.join(jd, "status.json"))
    st.update(state="running", pid=99999999)
    jm.write_json_atomic(os.path.join(jd, "status.json"), st)
    got = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
    ok = got["state"] == "failed" and "stale" in (got.get("error") or "")
    print(f"[{'✓' if ok else '✗'}] test_stale_running_detection")
    return ok


def test_listing() -> bool:
    base = _base()
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps({"action": "record", "stage": "t",
                              "summary": "s", "data": "{}"}))._run())
    _wait(base, out["job_id"])
    lst = json.loads(JobStatusTool(base_directory=base)._run())
    ok = lst["jobs"] and lst["jobs"][0]["job_id"] == out["job_id"]
    print(f"[{'✓' if ok else '✗'}] test_listing")
    return ok


def test_runner_rejects_state_fields_in_args() -> bool:
    """spec args must be runtime fields, re-checked when the runner reads them.

    submit() validates this, but spec.json sits under the user-writable
    base_directory. A StateField smuggled into "args" would override the
    server-side config: for FeynRulesToUFOTool, wolframscript_path becomes
    argv[0] of a subprocess, stepping around the whitelist entirely.
    """
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "audittrail",
        "args": {"stage": "x", "base_directory": "/somewhere/else"},
        "state": {"base_directory": base},
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "audittrail",
        "state": "queued", "pid": None, "exit_code": None,
        "submitted": None, "started": None, "finished": None, "error": None,
    })
    from tools.jobs.runner import main as runner_main
    rc = runner_main(jdir)
    st = jm.read_json(os.path.join(jdir, "status.json"))
    ok = rc == 2 and st["state"] == "failed" and "runtime fields" in (st["error"] or "")
    print(f"[{'✓' if ok else 'x'}] test_runner_rejects_state_fields_in_args")
    return ok


def test_runner_rejects_tampered_spec() -> bool:
    """A spec whose tool key isn't whitelisted must fail, not import."""
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "osystem",
        "args": {}, "state": {"base_directory": base},
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "osystem",
        "state": "queued", "pid": None, "exit_code": None,
        "submitted": None, "started": None, "finished": None, "error": None,
    })
    from tools.jobs.runner import main as runner_main
    rc = runner_main(jdir)
    st = jm.read_json(os.path.join(jdir, "status.json"))
    ok = rc == 2 and st["state"] == "failed" and "whitelisted" in st["error"]
    print(f"[{'✓' if ok else '✗'}] test_runner_rejects_tampered_spec")
    return ok


TESTS = [
    test_job_id_format,
    test_validate_args_against_real_tool,
    test_whitelist_names_match_tools,
    test_submit_rejections,
    test_detached_round_trip,
    test_result_before_done_and_bad_ids,
    test_stale_running_detection,
    test_listing,
    test_runner_rejects_state_fields_in_args,
    test_runner_rejects_tampered_spec,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the jobs toolkit")
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
