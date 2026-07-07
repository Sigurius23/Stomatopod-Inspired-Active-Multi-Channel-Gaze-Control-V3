"""
run_all.py — run every test file in tests/ as a subprocess
==========================================================

The project's test files are runnable scripts, not pytest discoveries,
so the simplest "run everything" entry point is to shell out to each
in turn and report a single pass/fail summary.

Usage:
    MUJOCO_GL=egl python tests/run_all.py            # quiet pass/fail summary
    MUJOCO_GL=egl python tests/run_all.py --verbose  # show full output
    MUJOCO_GL=egl python tests/run_all.py --filter scheduler  # subset

Exit code 0 if every file's exit status is 0, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable or "python"


def discover() -> list[Path]:
    """Return all test files in deterministic alphabetical order."""
    return sorted(THIS_DIR.glob("test_*.py"))


def run_one(path: Path, verbose: bool) -> tuple[bool, float, str]:
    """Run one test file, return (ok, elapsed_s, last_line)."""
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, str(path)],
        capture_output=not verbose,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        return proc.returncode == 0, elapsed, ""
    output = (proc.stdout or "") + (proc.stderr or "")
    last_line = (output.strip().splitlines() or [""])[-1]
    if proc.returncode != 0:
        # On failure, print the captured output so the user sees the traceback
        print(output)
    return proc.returncode == 0, elapsed, last_line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--verbose", action="store_true",
                   help="Stream each test's stdout instead of capturing it.")
    p.add_argument("--filter", type=str, default="",
                   help="Only run test files whose name contains this substring.")
    args = p.parse_args()

    files = discover()
    if args.filter:
        files = [f for f in files if args.filter in f.name]
    if not files:
        print(f"No test files matched filter {args.filter!r}.")
        sys.exit(2)

    print(f"Running {len(files)} test file(s) from {THIS_DIR}/")
    print("=" * 70)
    ok_count = 0
    fail_count = 0
    total_t0 = time.perf_counter()
    for f in files:
        ok, dt, last = run_one(f, args.verbose)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {f.name:<40s} {dt:5.1f}s   {last[:60]}")
        ok_count += ok
        fail_count += (not ok)
    total = time.perf_counter() - total_t0
    print("=" * 70)
    print(f"Summary: {ok_count} passed, {fail_count} failed   "
          f"(wall-clock {total:.1f}s)")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
