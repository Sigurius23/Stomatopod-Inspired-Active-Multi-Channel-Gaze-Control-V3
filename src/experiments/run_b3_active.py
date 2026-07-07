"""
run_b3_active.py — Baseline B3: active scanning + in-sensor preprocessing
=========================================================================

The full biomimetic system:
    - Both eye gimbals are driven by :class:`SaliencyScheduler`, which
      independently picks where each eye should look next based on a
      saliency map + visit-history memory.
    - Raw multi-channel sensor data runs through the full Layer 2
      preprocessing pipeline (channel reduction, polarization decoding,
      event encoding) on the way to the scheduler.
    - Bandwidth = the sparse event stream (low), but coverage should
      improve substantially vs. B1/B2 because the eyes actively scan
      and rotate to access polarization information.

This is the headline baseline the project is built around.

CLI
---
    python src/experiments/run_b3_active.py \\
        --duration 10 \\
        --seed 0 \\
        --n-candidates 30 \\
        --decision-period 0.10

Outputs
-------
    results/data/B3_metrics.json — four headline metrics for the run.
    results/data/B3_log.json     — full event log (for plotting).
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
from stomatopod_vision.scheduler import (  # noqa: E402
    SaliencyScheduler,
    ScoringWeights,
)


def add_b3_args(p: argparse.ArgumentParser) -> None:
    """B3-specific tuning knobs on top of the common ones.

    Default scoring-weight values are pulled from
    :class:`ScoringWeights` so there is one source of truth — change
    ``scheduler.py`` once and every CLI script picks it up.
    """
    _w = ScoringWeights()  # single source of truth for default weights
    p.add_argument(
        "--n-candidates", type=int, default=30,
        help="Candidate directions to score per eye per decision (default: 30).")
    p.add_argument(
        "--decision-period", type=float, default=0.10,
        help="Seconds between scheduler re-plans (default: 0.10).")
    p.add_argument(
        "--w-novelty", type=float, default=_w.novelty,
        help=f"Scoring weight: novelty (default: {_w.novelty}).")
    p.add_argument(
        "--w-salience", type=float, default=_w.salience,
        help=f"Scoring weight: salience (default: {_w.salience}).")
    p.add_argument(
        "--w-feasibility", type=float, default=_w.feasibility,
        help=f"Scoring weight: feasibility (default: {_w.feasibility}). "
             f"Set to 0.5 to recover the original hand-designed weights.")
    p.add_argument(
        "--w-pol", type=float, default=_w.polarization_info_gain,
        help=f"Scoring weight: polarization info gain (default: "
             f"{_w.polarization_info_gain}). Set to 1.0 to recover the "
             f"original hand-designed weights.")


def _run_one(args, seed: int, seed_suffix_in_filenames: bool) -> None:
    """One end-to-end B3 simulation for a single seed."""
    if not args.quiet:
        print("=" * 60)
        print(f"  Baseline B3 — active scanning + Layer 2 preprocessing"
              f"  (seed={seed})")
        print("=" * 60)
        print(f"  duration         : {args.duration:.1f} s")
        print(f"  seed             : {seed}")
        print(f"  n_candidates     : {args.n_candidates}")
        print(f"  decision_period  : {args.decision_period:.3f} s")
        print(f"  weights          : "
              f"nov={args.w_novelty} sal={args.w_salience} "
              f"feas={args.w_feasibility} pol={args.w_pol}")
        print(f"  model            : {args.model}")
        print(f"  results to       : {args.results_dir}")
        print()

    ctx = build_context(args.model, seed=seed)
    pipeline = PreprocessingPipeline()
    scheduler = SaliencyScheduler(
        n_candidates=args.n_candidates,
        decision_period_s=args.decision_period,
        weights=ScoringWeights(
            novelty=args.w_novelty,
            salience=args.w_salience,
            feasibility=args.w_feasibility,
            polarization_info_gain=args.w_pol,
        ),
        seed=seed,
    )

    # IMPORTANT: B3 needs to feed events back into the scheduler's
    # memory, which requires a small wrapper around run_simulation's
    # `setpoint_at` callback. We attach the scheduler to the pipeline
    # via a custom setpoint function and update_memory in the loop —
    # but run_simulation already calls pipeline.step() and exposes the
    # produced events. We thread the connection via a closure on the
    # held setpoint.
    #
    # The simplest implementation: subclass / re-use _common's loop but
    # extend it with a memory-update step. For clarity we *call*
    # run_simulation, then post-hoc fix up the scheduler memory from
    # the produced log. This is slightly less reactive than feeding
    # events live, but for static scenes it doesn't matter.
    #
    # For B3 to truly close the loop online, run_simulation needs a hook
    # for "after events are produced". We add that by passing a callable.
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

    save_and_report(log, baseline="B3",
                    results_dir=args.results_dir, quiet=args.quiet,
                    seed=seed if seed_suffix_in_filenames else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(parser)
    add_b3_args(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args)
    multi = len(seeds) > 1
    for seed in seeds:
        _run_one(args, seed=seed, seed_suffix_in_filenames=multi)

    write_summary_if_multi_seed("B3", args.results_dir, seeds, quiet=args.quiet)


if __name__ == "__main__":
    main()
