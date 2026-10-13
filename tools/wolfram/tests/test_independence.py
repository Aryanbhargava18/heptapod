"""
# test_independence.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The wolfram bundle stands alone.

Two properties are the whole reason this package is separate from its first
consumer, and both are the kind that regress silently the moment someone adds
a convenient import at the top of a module:

  1. It imports with no sympy installed. sympy is a heavy dependency of other
     bundles; a bundle that only shells out to wolframscript must not drag it
     in. Before the split, importing the runner ran tools/eda/__init__.py,
     which imports sympy eagerly -- so a sympy-free install could not use the
     Wolfram tools at all.

  2. It names no other bundle. No imports from, and no mentions of, the
     domain packages that consume it. A generic tool that quietly special-
     cases one caller's vocabulary is not generic.

Usage:
    python test_independence.py
"""

import builtins
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

PKG_DIR = REPO_ROOT / "tools" / "wolfram"

# Bundles that consume this one, plus the Wolfram add-on they happen to use.
# None of these may appear anywhere in tools/wolfram/.
FOREIGN_NAMES = ("feyncalc", "feynrules", "tools.eda", "tools/eda")


def test_imports_without_sympy():
    """The package imports on an environment with no sympy."""
    print("=" * 60)
    print("Testing import without sympy")
    print("=" * 60)

    # A subprocess, so a sympy already imported by another test cannot mask it.
    code = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'sympy' or name.startswith('sympy.'):\n"
        "        raise ImportError('sympy is blocked for this test')\n"
        "    return real_import(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import tools.wolfram as w\n"
        "assert w.RunWolframScript is not None\n"
        "assert w.WolframRunner is not None\n"
        "assert 'sympy' not in __import__('sys').modules\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    ok = proc.returncode == 0 and "OK" in proc.stdout
    if not ok:
        print(f"  stderr: {proc.stderr.strip()[-500:]}")
    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: tools.wolfram imports with sympy blocked")
    print()
    return ok


def test_no_foreign_imports():
    """No module in the package imports from a consuming bundle."""
    print("=" * 60)
    print("Testing for foreign imports")
    print("=" * 60)

    all_passed = True
    for path in sorted(PKG_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            low = stripped.lower()
            for name in FOREIGN_NAMES:
                if name in low:
                    rel = path.relative_to(REPO_ROOT)
                    print(f"  [✗] {rel}:{lineno} imports '{name}': {stripped}")
                    all_passed = False

    print(f"  {'[✓] PASS' if all_passed else '[✗] FAIL'}: no imports from consuming bundles")
    print()
    return all_passed


def test_no_foreign_mentions():
    """The package does not name a consuming bundle, even in prose.

    Tests are allowed to say what they are deliberately NOT requiring, so
    this checks the shipped modules only.
    """
    print("=" * 60)
    print("Testing for foreign mentions")
    print("=" * 60)

    all_passed = True
    for path in sorted(PKG_DIR.glob("*.py")):
        text = path.read_text().lower()
        for name in FOREIGN_NAMES:
            if name in text:
                rel = path.relative_to(REPO_ROOT)
                hits = [
                    i for i, l in enumerate(text.splitlines(), 1) if name in l
                ]
                print(f"  [✗] {rel} mentions '{name}' on line(s) {hits}")
                all_passed = False

    print(f"  {'[✓] PASS' if all_passed else '[✗] FAIL'}: no mention of a consuming bundle")
    print()
    return all_passed


def test_declares_no_pip_deps():
    """The bundle declares no pip dependencies in toolkit.yaml.

    The point of the split is that this bundle installs on a bare
    environment; a `deps:` list appearing here is the regression.
    """
    print("=" * 60)
    print("Testing bundle declares no deps")
    print("=" * 60)

    text = (REPO_ROOT / "toolkit.yaml").read_text()
    m = re.search(r"^  wolfram:\s*$(.*?)(?=^  \S|\Z)", text, re.M | re.S)
    if m is None:
        # single-line form, e.g. `wolfram: {}`
        m2 = re.search(r"^  wolfram:\s*(\{.*\}|.*)$", text, re.M)
        block = m2.group(1) if m2 else None
        ok = block is not None and "deps" not in block
    else:
        ok = "deps" not in m.group(1)

    print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: wolfram bundle declares no pip deps")
    print()
    return ok


def main():
    print()
    print("=" * 60)
    print("Wolfram Bundle Independence Tests")
    print("=" * 60)
    print()

    tests = [
        ("Import without sympy", test_imports_without_sympy),
        ("No foreign imports", test_no_foreign_imports),
        ("No foreign mentions", test_no_foreign_mentions),
        ("No pip deps declared", test_declares_no_pip_deps),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [✗] FAIL: {name} raised {type(e).__name__}: {e}")
            print()
            results.append((name, False))

    print("=" * 60)
    for name, ok in results:
        print(f"  {'[✓] PASS' if ok else '[✗] FAIL'}: {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  Total: {passed}/{len(results)} test groups passed")
    print("=" * 60)
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
