"""
noise_ablation.py — how does coverage degrade under sensor noise?
=================================================================

Runs B3 on the hard scene at increasing levels of per-receptor Gaussian
noise (``receptor_noise_std``), 5 seeds each, and prints / saves a
small table of (noise, coverage, polarization-accuracy, bandwidth).

Why this exists
---------------
The headline numbers in the report all assume a noise-free sensor —
that's the deterministic case where coverage and polarization accuracy
saturate at 1.000 ± 0.000. Real biology has shot noise on
photoreceptors. This script sweeps noise from 0.0 to 0.20 (i.e. ~20 %
of the 1 m peak amplitude) and shows the graceful degradation curve.

CLI
---
    MUJOCO_GL=egl python src/experiments/noise_ablation.py
    MUJOCO_GL=egl python src/experiments/noise_ablation.py \\
        --noise-levels 0.0 0.02 0.05 0.10 0.15 0.20 --seeds 0 1 2 3 4

Outputs
-------
    results/noise_ablation/summary.json — the full sweep
    stdout                              — formatted table
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from _common import (  # noqa: E402
    _REPO_ROOT,
    DEFAULT_RESULTS_DIR,
    build_context,
    run_simulation,
)

from stomatopod_vision.metrics import compute_all  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import SaliencyScheduler  # noqa: E402

HARD_MODEL = _REPO_ROOT / "models" / "stomatopod_eyes_hard.xml"


def _run_one(noise_std: float, seed: int, duration: float) -> dict:
    """One B3 rollout on the hard scene at the given noise level."""
    ctx = build_context(HARD_MODEL, seed=seed, receptor_noise_std=noise_std)
    pipeline = PreprocessingPipeline()
    scheduler = SaliencyScheduler(seed=seed)
    log = run_simulation(
        ctx,
        setpoint_at=lambda t: scheduler.next_setpoint(t, scheduler._held_setpoint),
        pipeline=pipeline,
        duration_s=duration,
        quiet=True,
        on_events=lambda evs, t: scheduler.update_memory(evs, t),
        controller_rate_hz=10.0,
    )
    rep = compute_all(log, baseline="B3")
    return {
        "coverage":               rep.coverage,
        "polarization_accuracy":  rep.polarization_accuracy,
        "bandwidth_bps":          rep.bandwidth_bps,
        "median_latency_s":       rep.median_latency_s,
    }


def _agg(per_seed: list[dict], key: str) -> tuple[float, float]:
    vals = [s[key] for s in per_seed]
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--noise-levels", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
                   help="Receptor-noise std values to sweep "
                        "(default: 0 / 0.02 / 0.05 / 0.10 / 0.15 / 0.20).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="RNG seeds (default: 0..4).")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Simulated seconds per run (default: 10).")
    p.add_argument("--out-dir", type=Path,
                   default=DEFAULT_RESULTS_DIR / "noise_ablation",
                   help="Where to save the summary JSON.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"noise_ablation: hard scene, B3, T={args.duration}s × "
          f"{len(args.seeds)} seeds × {len(args.noise_levels)} noise levels")
    print("=" * 80)

    results = {}
    t0 = time.perf_counter()
    for noise in args.noise_levels:
        print(f"\n  noise_std = {noise:.2f}  …", end="", flush=True)
        ts = time.perf_counter()
        per_seed = [_run_one(noise, s, args.duration) for s in args.seeds]
        cov_m, cov_s = _agg(per_seed, "coverage")
        pol_m, pol_s = _agg(per_seed, "polarization_accuracy")
        bw_m,  bw_s  = _agg(per_seed, "bandwidth_bps")
        lat_m, lat_s = _agg(per_seed, "median_latency_s")
        results[f"{noise:.3f}"] = {
            "noise_std":         noise,
            "n_seeds":           len(args.seeds),
            "coverage":          {"mean": cov_m, "std": cov_s},
            "polarization_acc.": {"mean": pol_m, "std": pol_s},
            "bandwidth_bps":     {"mean": bw_m,  "std": bw_s},
            "median_latency_s":  {"mean": lat_m, "std": lat_s},
            "per_seed":          per_seed,
        }
        print(f" {time.perf_counter() - ts:5.1f}s  "
              f"cov={cov_m:.3f}±{cov_s:.3f}  pol={pol_m:.3f}±{pol_s:.3f}  "
              f"bw={bw_m:.0f}±{bw_s:.0f}")

    # Persist
    out_path = args.out_dir / "summary.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Pretty table
    print()
    print("=" * 80)
    print("  Noise ablation — hard scene, B3, T={:.0f}s, {} seeds".format(
        args.duration, len(args.seeds)))
    print("=" * 80)
    print(f"  {'noise':<6s} {'coverage':<14s} {'pol acc.':<14s} {'bandwidth (B/s)':<18s}")
    print("  " + "-" * 64)
    for noise_str, r in results.items():
        print(f"  {float(noise_str):<6.2f} "
              f"{r['coverage']['mean']:.3f}±{r['coverage']['std']:.3f}  "
              f"{r['polarization_acc.']['mean']:.3f}±{r['polarization_acc.']['std']:.3f}  "
              f"{r['bandwidth_bps']['mean']:>7.0f}±{r['bandwidth_bps']['std']:.0f}")
    print()
    print(f"  total wall clock: {time.perf_counter() - t0:.1f}s")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
