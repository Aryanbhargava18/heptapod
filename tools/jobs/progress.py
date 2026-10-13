"""
# progress.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Progress reporting from inside a tool's ``_run()``.

A tool calls ``report_progress(self, done, total, unit=..., message=...)``
wherever it knows how far along it is. What happens next depends on who ran
the tool:

  * the job runner installs a reporter that records the progress in the
    job's status.json, where jobstatus shows it as a progress bar;
  * orchestral's ``execute(stream_callback=...)`` installs a plain
    ``Callable[[str], None]``, which gets a one-line text rendering;
  * a direct call with no callback makes this a no-op.

So a tool reports once and stays unaware of how it is being run.
"""

from __future__ import annotations

from typing import Any, Optional


def format_progress(done: Optional[float], total: Optional[float],
                    unit: str = "", message: str = "") -> str:
    """One-line text form, e.g. ``survey · 5/6 channels``."""
    parts = []
    if message:
        parts.append(message)
    if done is not None:
        count = f"{done:g}/{total:g}" if total else f"{done:g}"
        parts.append(f"{count} {unit}".strip())
    return " · ".join(parts)


def report_progress(tool: Any, done: Optional[float] = None,
                    total: Optional[float] = None, unit: str = "",
                    message: str = "") -> None:
    """Report progress to whatever is running ``tool``; never raises."""
    cb = getattr(tool, "_stream_callback", None)
    if cb is None:
        return
    try:
        structured = getattr(cb, "progress", None)
        if callable(structured):
            structured(done=done, total=total, unit=unit, message=message)
        else:
            cb(format_progress(done, total, unit, message))
    except Exception:  # noqa: BLE001 - progress must never fail a run
        pass
