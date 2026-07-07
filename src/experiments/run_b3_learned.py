"""
run_b3_learned.py — Baseline B3-Learned: active scanning with a LEARNED scoring fn
==================================================================================

Identical to ``run_b3_active.py`` except the saliency scheduler's
hand-designed scoring sum is replaced by the trained MLP loaded from
``--mlp-path``. The simulation, sensor, preprocessing, gimbal control,
and candidate sampler are all unchanged.

This is the **bonus** baseline; it demonstrates that the action-
selection scoring function can be *learned* rather than hand-tuned
(Lecture 6, value-based action selection).

Workflow
--------
    # 1. Train the MLP (one-off, ~10 s on a CPU)
    MUJOCO_GL=egl python src/experiments/train_learned.py

    # 2. Run B3-Learned and report the four headline metrics
    MUJOCO_GL=egl python src/experiments/run_b3_learned.py \\
        --duration 10 --seed 0 \\
        --mlp-path results/learned/mlp.npz

The headline label is reported as ``B3L`` so it can be plotted alongside
``B1``/``B2``/``B3`` without colliding with the existing B3 results.

CLI
---
    python src/experiments/run_b3_learned.py --duration 10 --seed 0
    python src/experiments/run_b3_learned.py --duration 10 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (  # noqa: E402
    _REPO_ROOT,
    DEFAULT_RESULTS_DIR,
    add_common_args,
    build_context,
    resolve_seeds,
    run_simulation,
    save_and_report,
    write_summary_if_multi_seed,
)

from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import LearnedScheduler  # noqa: E402


def add_b3l_args(p: argparse.ArgumentParser) -> None:
    """B3-Learned-specific CLI flags."""
    p.add_argument(
        "--mlp-path", type=Path,
        default=DEFAULT_RESULTS_DIR / "learned" / "mlp.npz",
        help=f"Path to the trained MLP weights "
             f"(default: {(DEFAULT_RESULTS_DIR / 'learned' / 'mlp.npz').relative_to(_REPO_ROOT)}). "
             f"Train one with src/experiments/train_learned.py.")
    p.add_argument(
        "--n-candidates", type=int, default=30,
        help="Candidate directions per re-plan (default: 30).")
    p.add_argument(
        "--decision-period", type=float, default=0.10,
        help="Re-plan period in seconds (default: 0.10).")
    p.add_argument(
        "--baseline-label", type=str, default="B3L",
        help='String used in <label>_metrics.json / summary files (default: "B3L").')


def _run_one(args, seed: int, seed_suffix_in_filenames: bool) -> None:
    """One end-to-end B3-Learned simulation for a single seed."""
    if not args.mlp_path.exists():
        raise SystemExit(
            f"\nMLP weights not found at {args.mlp_path}.\n"
            f"Train them first:\n"
            f"    MUJOCO_GL=egl python src/experiments/train_learned.py\n"
        )

    if not args.quiet:
        print("=" * 60)
        print(f"  Baseline B3-Learned — active scanning + LEARNED scoring  (seed={seed})")
        print("=" * 60)
        print(f"  duration         : {args.duration:.1f} s")
        print(f"  seed             : {seed}")
        print(f"  n_candidates     : {args.n_candidates}")
        print(f"  decision_period  : {args.decision_period:.3f} s")
        print(f"  mlp              : {args.mlp_path}")
        print(f"  model            : {args.model}")
        print(f"  results to       : {args.results_dir}")
        print()

    ctx = build_context(args.model, seed=seed)
    pipeline = PreprocessingPipeline()
    scheduler = LearnedScheduler.from_file(
        args.mlp_path,
        n_candidates=args.n_candidates,
        decision_period_s=args.decision_period,
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
        # The scheduler internally throttles re-plans to decision_period_s
        # (default 0.10 s = 10 Hz), so we can safely query it at 10 Hz
        # instead of the 500 Hz physics rate. Identical setpoints, 50x fewer
        # callback invocations.
        controller_rate_hz=1.0 / args.decision_period,
    )

    save_and_report(log, baseline=args.baseline_label,
                    results_dir=args.results_dir, quiet=args.quiet,
                    seed=seed if seed_suffix_in_filenames else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(parser)
    add_b3l_args(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args)
    multi = len(seeds) > 1
    for seed in seeds:
        _run_one(args, seed=seed, seed_suffix_in_filenames=multi)

    write_summary_if_multi_seed(args.baseline_label, args.results_dir,
                                seeds, quiet=args.quiet)


if __name__ == "__main__":
    main()
