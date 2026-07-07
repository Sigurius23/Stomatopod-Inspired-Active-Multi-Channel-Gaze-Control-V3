"""
benchmark_dynamical.py — Hopf limit-cycle scanner vs. saliency scheduler
========================================================================

Compares the two active-sensing controllers head-to-head under identical
conditions:

    * **B3-Saliency** — :class:`SaliencyScheduler`, the mandatory
      hand-designed "score candidates → argmax" controller.
    * **B3-Hopf**     — :class:`HopfScanScheduler`, the bonus dynamical
      controller: a Hopf limit-cycle oscillator per eye that rhythmically
      scans and undergoes a Hopf bifurcation to fixate detected targets.

Both are driven through the *same* preprocessing pipeline and metrics, so
the numbers are directly comparable. This is the "benchmark it against the
current one" deliverable — the point is not that one strictly dominates,
but that a smooth continuous-dynamics controller is a viable alternative
to discrete replanning on this task.

CLI
---
    MUJOCO_GL=egl python src/experiments/benchmark_dynamical.py
    MUJOCO_GL=egl python src/experiments/benchmark_dynamical.py \\
        --duration 8 --seeds 0 1 2

Outputs
-------
    results/benchmark_dynamical/summary.json  — full per-scene/-seed table
    stdout                                     — formatted comparison table
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from _common import (  # noqa: E402  (sibling import resolved at runtime)
    _REPO_ROOT,
    DEFAULT_RESULTS_DIR,
    build_context,
    run_simulation,
)

from stomatopod_vision.metrics import compute_all  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    HopfScanScheduler,
    SaliencyScheduler,
)

SCENES: dict[str, Path] = {
    "default": _REPO_ROOT / "models" / "stomatopod_eyes.xml",
    "hard":    _REPO_ROOT / "models" / "stomatopod_eyes_hard.xml",
}

# The two controllers, each a zero-arg factory given a seed.
CONTROLLERS = {
    # (factory, setpoint-callback-rate-Hz). Saliency internally throttles to
    # 10 Hz; the Hopf oscillator is integrated continuously so we poll it
    # every physics step (rate=None) for the smoothest sweep.
    "B3-Saliency": (lambda seed: SaliencyScheduler(seed=seed), 10.0),
    "B3-Hopf":     (lambda seed: HopfScanScheduler(seed=seed), None),
}


def _run_one(model: Path, make_sched, rate, seed: int, duration: float) -> dict:
    """One controller × scene × seed → metrics dict."""
    ctx = build_context(model, seed=seed)
    pipeline = PreprocessingPipeline()
    sched = make_sched(seed)
    log = run_simulation(
        ctx,
        setpoint_at=lambda t: sched.next_setpoint(
            t, current_setpoint=sched._held_setpoint),
        pipeline=pipeline,
        duration_s=duration,
        quiet=True,
        on_events=lambda events, t: sched.update_memory(events, t),
        controller_rate_hz=rate,
    )
    rep = compute_all(log, baseline="B3")
    return {
        "coverage": rep.coverage,
        "bandwidth_bps": rep.bandwidth_bps,
        "polarization_accuracy": rep.polarization_accuracy,
        "circular_polarization_accuracy": rep.circular_polarization_accuracy,
        "median_latency_s": rep.median_latency_s,
    }


def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    return (statistics.mean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--duration", type=float, default=8.0,
                   help="Simulated seconds per run (default: 8).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds to average over (default: 0 1 2).")
    p.add_argument("--results-dir", type=Path,
                   default=DEFAULT_RESULTS_DIR / "benchmark_dynamical",
                   help="Where to write summary.json.")
    args = p.parse_args()

    results: dict = {}
    print(f"\nbenchmark_dynamical: {len(SCENES)} scenes × "
          f"{len(CONTROLLERS)} controllers × {len(args.seeds)} seeds, "
          f"T={args.duration}s\n")
    header = (f"{'scene':8} {'controller':12} "
              f"{'cover':>6} {'bw (B/s)':>10} {'pol':>5} {'circ':>5} {'lat(s)':>7}")
    print(header)
    print("-" * len(header))

    for scene_name, model in SCENES.items():
        results[scene_name] = {}
        for ctrl_name, (factory, rate) in CONTROLLERS.items():
            rows = [_run_one(model, factory, rate, s, args.duration)
                    for s in args.seeds]
            summary = {k: {"mean": _agg(rows, k)[0], "std": _agg(rows, k)[1]}
                       for k in rows[0]}
            results[scene_name][ctrl_name] = summary
            print(f"{scene_name:8} {ctrl_name:12} "
                  f"{summary['coverage']['mean']:6.3f} "
                  f"{summary['bandwidth_bps']['mean']:10,.0f} "
                  f"{summary['polarization_accuracy']['mean']:5.3f} "
                  f"{summary['circular_polarization_accuracy']['mean']:5.3f} "
                  f"{summary['median_latency_s']['mean']:7.2f}")
        print()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "summary.json"
    out.write_text(json.dumps(
        {"duration_s": args.duration, "seeds": args.seeds, "results": results},
        indent=2))
    print(f"Wrote {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
