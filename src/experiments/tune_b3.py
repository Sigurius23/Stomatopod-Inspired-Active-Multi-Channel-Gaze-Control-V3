"""
tune_b3.py — coarse grid-search over SaliencyScheduler ScoringWeights
=====================================================================

Operates at a **deliberately short simulation duration** so that the
coverage metric is sensitive to weight choices (at 10 s with the hard
scene B3 saturates at 1.00 and weight differences are invisible).

For each candidate ``(w_novelty, w_salience, w_feasibility, w_pol)``,
runs B3 on the hard scene for ``--duration`` seconds across ``--seeds``
random seeds, then reports the mean ± std of coverage and polarization
accuracy. The best mean coverage (with bandwidth as tie-breaker) is
printed at the end.

This is a SCRIPT, not a Python-level grid-search library: it shells out
to ``run_b3_active.py`` for each cell so the production code path is
exactly what gets benchmarked.

CLI
---
    # Default coarse 3×3×3×3 grid, 5 seeds, 0.5s sim each
    MUJOCO_GL=egl python src/experiments/tune_b3.py

    # Tighter grid around the previous best
    MUJOCO_GL=egl python src/experiments/tune_b3.py \\
        --novelty 0.5 1.0 1.5 \\
        --salience 1.0 2.0 4.0 \\
        --feasibility 0.0 0.5 1.0 \\
        --pol 0.0 1.0 2.0 \\
        --seeds 0 1 2 3 4 5 6 7

Outputs
-------
    results/tuning/grid.csv — every (cell × seed) row
    results/tuning/best.json — the winning weights
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from _common import _REPO_ROOT, DEFAULT_RESULTS_DIR  # noqa: E402

HARD_MODEL = _REPO_ROOT / "models" / "stomatopod_eyes_hard.xml"


def _run_one(seed: int, duration: float, weights: tuple[float, float, float, float],
             results_dir: Path) -> dict:
    """Invoke run_b3_active.py once and return the parsed metrics dict."""
    wn, ws, wf, wp = weights
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_b3_active.py"),
        "--duration", str(duration),
        "--seed", str(seed),
        "--model", str(HARD_MODEL),
        "--results-dir", str(results_dir),
        "--w-novelty", str(wn),
        "--w-salience", str(ws),
        "--w-feasibility", str(wf),
        "--w-pol", str(wp),
        "--quiet",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads((results_dir / "data" / "B3_metrics.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--duration", type=float, default=0.5,
                        help="B3 simulation duration per cell (default: 0.5s — short "
                             "enough that weight choices visibly matter on the hard scene).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="RNG seeds to average over (default: 0..4).")
    parser.add_argument("--novelty", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="Grid values for w_novelty (default: 0.5 1.0 2.0).")
    parser.add_argument("--salience", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                        help="Grid values for w_salience (default: 1.0 2.0 4.0).")
    parser.add_argument("--feasibility", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="Grid values for w_feasibility (default: 0.0 0.5 1.0; note: the current ScoringWeights default is 0.0).")
    parser.add_argument("--pol", type=float, nargs="+", default=[0.0, 1.0, 2.0],
                        help="Grid values for w_polarization_info_gain (default: 0.0 1.0 2.0; note: the current ScoringWeights default is 0.0).")
    parser.add_argument("--out-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR / "tuning",
                        help="Where to write grid.csv and best.json.")
    parser.add_argument("--scratch-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR / "hard",
                        help="Scratch results dir reused for each cell "
                             "(B3_metrics.json is overwritten each time).")
    args = parser.parse_args()

    grid = list(itertools.product(args.novelty, args.salience,
                                  args.feasibility, args.pol))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "grid.csv"
    rows: list[dict] = []
    cell_summary: list[dict] = []

    print("=" * 76)
    print(f"tune_b3: scene = {HARD_MODEL.name}")
    print(f"         duration  = {args.duration}s per run")
    print(f"         seeds     = {args.seeds}")
    print(f"         grid size = {len(grid)} cells × {len(args.seeds)} seeds "
          f"= {len(grid) * len(args.seeds)} runs")
    print("=" * 76)

    t0 = time.perf_counter()
    for cell_i, w in enumerate(grid, start=1):
        cov_vals, pol_vals, bw_vals = [], [], []
        for seed in args.seeds:
            m = _run_one(seed, args.duration, w, args.scratch_dir)
            cov_vals.append(m["coverage"])
            pol_vals.append(m["polarization_accuracy"])
            bw_vals.append(m["bandwidth_bps"])
            rows.append({
                "w_novelty": w[0], "w_salience": w[1],
                "w_feasibility": w[2], "w_pol": w[3],
                "seed": seed,
                "coverage": m["coverage"],
                "polarization_accuracy": m["polarization_accuracy"],
                "bandwidth_bps": m["bandwidth_bps"],
                "median_latency_s": m["median_latency_s"],
            })
        cov_mean = statistics.mean(cov_vals)
        cov_std = statistics.pstdev(cov_vals) if len(cov_vals) > 1 else 0.0
        pol_mean = statistics.mean(pol_vals)
        bw_mean = statistics.mean(bw_vals)
        cell_summary.append({
            "w": w,
            "cov_mean": cov_mean, "cov_std": cov_std,
            "pol_mean": pol_mean, "bw_mean": bw_mean,
        })
        print(f"  [{cell_i:3d}/{len(grid)}] "
              f"nov={w[0]:.2f} sal={w[1]:.2f} feas={w[2]:.2f} pol={w[3]:.2f}  "
              f"cov={cov_mean:.3f}±{cov_std:.3f}  pol={pol_mean:.3f}  "
              f"bw={bw_mean:>7.0f}B/s")

    # Persist the full per-(cell, seed) grid
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path.relative_to(_REPO_ROOT) if csv_path.is_relative_to(_REPO_ROOT) else csv_path}")

    # Pick the winner: highest mean coverage; ties broken by LOWEST bandwidth
    cell_summary.sort(key=lambda c: (-c["cov_mean"], c["bw_mean"]))
    best = cell_summary[0]
    print("\n" + "=" * 76)
    print("BEST CELL (max mean coverage, ties broken by min bandwidth):")
    print(f"  w_novelty            = {best['w'][0]}")
    print(f"  w_salience           = {best['w'][1]}")
    print(f"  w_feasibility        = {best['w'][2]}")
    print(f"  w_polarization_info  = {best['w'][3]}")
    print(f"  → coverage           = {best['cov_mean']:.3f} ± {best['cov_std']:.3f}")
    print(f"  → polarization_acc.  = {best['pol_mean']:.3f}")
    print(f"  → bandwidth          = {best['bw_mean']:.0f} B/s")
    print(f"  (wall-clock so far: {time.perf_counter() - t0:.1f}s)")
    print("=" * 76)

    best_json = args.out_dir / "best.json"
    best_json.write_text(json.dumps({
        "duration_s": args.duration,
        "seeds": args.seeds,
        "weights": {
            "novelty":                  best["w"][0],
            "salience":                 best["w"][1],
            "feasibility":              best["w"][2],
            "polarization_info_gain":   best["w"][3],
        },
        "coverage_mean": best["cov_mean"],
        "coverage_std":  best["cov_std"],
        "polarization_accuracy_mean": best["pol_mean"],
        "bandwidth_bps_mean": best["bw_mean"],
    }, indent=2))
    print(f"Saved winning weights to {best_json.relative_to(_REPO_ROOT) if best_json.is_relative_to(_REPO_ROOT) else best_json}")


if __name__ == "__main__":
    main()
