"""
run_all.py — orchestrate all three baselines + plot generation
==============================================================

Runs B1, B2, B3 back-to-back with matched ``--seed`` / ``--duration`` /
``--model`` / ``--results-dir`` flags, then invokes ``make_plots.py`` to
regenerate every figure under ``results/figures/``.

This is the single command used to reproduce the headline numbers and
figures in the report.

CLI
---
    # Default: 10 s simulation, seed 0, default scene
    python src/experiments/run_all.py

    # Reproduce a specific seed / duration combination
    python src/experiments/run_all.py --duration 20 --seed 7

    # Skip plotting (just regenerate JSON metrics)
    python src/experiments/run_all.py --no-plots

Outputs
-------
    results/data/B{1,2,3}_metrics.json
    results/data/B{1,2,3}_log.json
    results/figures/{headline,coverage_over_time,bandwidth_over_time,
                     polarization_timeline}.svg  (+ .png if --png)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import _REPO_ROOT, DEFAULT_MODEL, DEFAULT_RESULTS_DIR  # noqa: E402

# The three baseline scripts, in fixed order so console output is
# deterministic and the report's narrative (B1 → B2 → B3) is preserved.
_BASELINE_SCRIPTS = ("run_b1_fixed.py", "run_b2_preprocessed.py", "run_b3_active.py")

# Optional bonus baseline (opt-in via --dynamical) so the headline
# B1 → B2 → B3 sweep and its figures stay byte-stable by default.
_DYNAMICAL_SCRIPT = "run_b3_dynamical.py"


def _python_for_subprocess() -> str:
    """Use the same interpreter that's running us (keeps venvs honest)."""
    return sys.executable or "python"


def _run(script: str, common_args: list[str], extra: list[str] | None = None) -> float:
    """Run one script as a subprocess; return wall-clock seconds."""
    script_path = Path(__file__).resolve().parent / script
    cmd = [_python_for_subprocess(), str(script_path), *common_args, *(extra or [])]
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    # ``check=True`` so any failure aborts the whole sweep loudly.
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds (default: 10).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed shared by all three baselines (default: 0). "
                             "Ignored when --seeds is supplied.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Run each baseline once per seed in this list. "
                             "Produces per-seed JSON files plus a "
                             "<baseline>_summary.json containing mean/std across "
                             "seeds. When supplied, make_plots.py will draw "
                             "error bars / ribbons.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Path to the MuJoCo XML (default: "
                             f"{DEFAULT_MODEL.relative_to(_REPO_ROOT)}).")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Where to write JSON + figures (default: "
                             f"{DEFAULT_RESULTS_DIR.relative_to(_REPO_ROOT)}).")
    parser.add_argument("--quiet", action="store_true",
                        help="Forward --quiet to each baseline (suppress per-step output).")
    parser.add_argument("--dynamical", action="store_true",
                        help="Also run the bonus B3-Dynamical (Hopf limit-cycle) "
                             "scheduler after B3. Off by default so the headline "
                             "B1→B2→B3 numbers/figures stay byte-stable.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip make_plots.py at the end.")
    parser.add_argument("--png", action="store_true",
                        help="Forward --png to make_plots.py (also write 200-dpi PNGs).")
    args = parser.parse_args()

    # Build the common-args list. We forward EITHER --seed (single-seed
    # mode, original behaviour) OR --seeds N N N ... (multi-seed mode).
    common_args = [
        "--duration", str(args.duration),
        "--model", str(args.model),
        "--results-dir", str(args.results_dir),
    ]
    if args.seeds:
        common_args += ["--seeds", *[str(s) for s in args.seeds]]
        seed_desc = f"seeds={list(args.seeds)}"
    else:
        common_args += ["--seed", str(args.seed)]
        seed_desc = f"seed={args.seed}"
    if args.quiet:
        common_args.append("--quiet")

    print("=" * 60)
    print(f"run_all: duration={args.duration}s {seed_desc}")
    print(f"         model={args.model}")
    print(f"         results-dir={args.results_dir}")
    print("=" * 60)

    scripts = list(_BASELINE_SCRIPTS)
    if args.dynamical:
        scripts.append(_DYNAMICAL_SCRIPT)

    timings: dict[str, float] = {}
    for script in scripts:
        timings[script] = _run(script, common_args)

    if not args.no_plots:
        plot_args = ["--results-dir", str(args.results_dir)]
        if args.png:
            plot_args.append("--png")
        timings["make_plots.py"] = _run("make_plots.py", plot_args)

    print("\n" + "=" * 60)
    print("run_all summary (wall-clock seconds):")
    for name, secs in timings.items():
        print(f"  {name:<28s} {secs:6.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
