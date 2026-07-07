"""
run_b3_dynamical.py — B3-Dynamical: Hopf limit-cycle active scanning
====================================================================

Same three-layer pipeline as :mod:`run_b3_active` (virtual eyes → Layer 2
preprocessing → active-sensing controller), but the controller is the
bonus :class:`~stomatopod_vision.scheduler.HopfScanScheduler` instead of
the hand-designed :class:`SaliencyScheduler`:

    * Each eye is a Hopf limit-cycle oscillator that rhythmically sweeps
      the narrow midband (scan).
    * Detecting a target triggers a Hopf bifurcation to a fixed-point
      attractor that foveates it (fixate), then bifurcates back.

Writes ``B3D_metrics.json`` / ``B3D_log.json`` so it can be compared
directly with B1/B2/B3. See :mod:`benchmark_dynamical` for a matched
head-to-head against B3.

CLI
---
    MUJOCO_GL=egl python src/experiments/run_b3_dynamical.py
    MUJOCO_GL=egl python src/experiments/run_b3_dynamical.py \\
        --model models/stomatopod_eyes_hard.xml --duration 8 --seeds 0 1 2
"""
from __future__ import annotations

import argparse

from _common import (  # noqa: E402  (sibling import resolved at runtime)
    add_common_args,
    build_context,
    resolve_seeds,
    run_simulation,
    save_and_report,
    write_summary_if_multi_seed,
)

from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import HopfScanScheduler  # noqa: E402


def add_b3d_args(p: argparse.ArgumentParser) -> None:
    """Hopf-scheduler tuning knobs on top of the common ones."""
    p.add_argument(
        "--omega", type=float, default=6.0,
        help="Angular speed of the scan limit cycle, rad/s (default: 6.0).")
    p.add_argument(
        "--fixation-dwell", type=float, default=0.40,
        help="Seconds the eye foveates a detected target (default: 0.40).")
    p.add_argument(
        "--controller-rate", type=float, default=None,
        help="Hz at which to poll the oscillator (default: every physics "
             "step). The ODE is sub-stepped, so lower rates stay accurate "
             "but produce a coarser commanded sweep.")


def _run_one(args, seed: int, seed_suffix_in_filenames: bool) -> None:
    """One end-to-end B3-Dynamical simulation for a single seed."""
    if not args.quiet:
        print("=" * 60)
        print(f"  Baseline B3D — Hopf limit-cycle scanning + Layer 2"
              f"  (seed={seed})")
        print("=" * 60)
        print(f"  duration         : {args.duration:.1f} s")
        print(f"  seed             : {seed}")
        print(f"  omega            : {args.omega:.2f} rad/s")
        print(f"  fixation_dwell   : {args.fixation_dwell:.3f} s")
        print(f"  model            : {args.model}")
        print(f"  results to       : {args.results_dir}")
        print()

    ctx = build_context(args.model, seed=seed)
    pipeline = PreprocessingPipeline()
    scheduler = HopfScanScheduler(
        omega=args.omega,
        fixation_dwell_s=args.fixation_dwell,
        seed=seed,
    )

    log = run_simulation(
        ctx,
        setpoint_at=lambda t: scheduler.next_setpoint(
            t, current_setpoint=scheduler._held_setpoint),
        pipeline=pipeline,
        duration_s=args.duration,
        quiet=args.quiet,
        on_events=lambda events, t: scheduler.update_memory(events, t),
        controller_rate_hz=args.controller_rate,
    )

    save_and_report(log, baseline="B3D",
                    results_dir=args.results_dir, quiet=args.quiet,
                    seed=seed if seed_suffix_in_filenames else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(parser)
    add_b3d_args(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args)
    multi = len(seeds) > 1
    for seed in seeds:
        _run_one(args, seed=seed, seed_suffix_in_filenames=multi)

    write_summary_if_multi_seed("B3D", args.results_dir, seeds, quiet=args.quiet)


if __name__ == "__main__":
    main()
