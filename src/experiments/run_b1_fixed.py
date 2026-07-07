"""
run_b1_fixed.py — Baseline B1: fixed cameras, no in-sensor preprocessing
========================================================================

The simplest "passive multi-channel camera" baseline.

What B1 represents
------------------
A traditional camera-like sensor: the raw multi-channel data stream is
transmitted in full from the sensor to a downstream processor. The
downstream processor *does* extract semantics (otherwise no controller
could ever use the data), but it does so AFTER paying the full
bandwidth cost.

In simulation terms:
    - Both eye gimbals are held at the rest pose (looking straight forward).
    - We still run Layer 2 internally so we have a fair comparison on
      coverage / polarization-accuracy / latency.
    - **Bandwidth is computed as if the raw stream were transmitted**
      (i.e. the full sample, per eye, every step) — this is the cost
      the system would have paid in a real camera-based pipeline.

This way the headline comparison vs B2 cleanly isolates ONE variable:
*where does preprocessing happen?* (At the sensor vs. downstream.)

CLI
---
    python src/experiments/run_b1_fixed.py \\
        --duration 10 \\
        --seed 0

Outputs
-------
    results/data/B1_metrics.json — four headline metrics for the run.
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

from stomatopod_vision.gimbal_control import GimbalSetpoint  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402


def _run_one(args, seed: int, seed_suffix_in_filenames: bool) -> None:
    """One end-to-end B1 simulation for a single seed."""
    if not args.quiet:
        print("=" * 60)
        print(f"  Baseline B1 — fixed cameras, no in-sensor preprocessing"
              f"  (seed={seed})")
        print("=" * 60)
        print(f"  duration   : {args.duration:.1f} s")
        print(f"  seed       : {seed}")
        print(f"  model      : {args.model}")
        print(f"  results to : {args.results_dir}")
        print()

    ctx = build_context(args.model, seed=seed)

    # B1 = eyes locked forward. Layer 2 still runs (otherwise coverage and
    # polarization metrics would be vacuously 0 — a passive camera plus
    # downstream processing CAN identify targets, it just pays full
    # bandwidth to do so). The B1-vs-B2 contrast lives in the bandwidth
    # metric, which is computed differently for "B1" baselines in
    # metrics.bandwidth_bytes_per_second.
    fixed_setpoint = GimbalSetpoint()
    pipeline = PreprocessingPipeline()
    log = run_simulation(
        ctx,
        setpoint_at=lambda _t: fixed_setpoint,
        pipeline=pipeline,
        duration_s=args.duration,
        quiet=args.quiet,
    )

    save_and_report(log, baseline="B1",
                    results_dir=args.results_dir, quiet=args.quiet,
                    seed=seed if seed_suffix_in_filenames else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_common_args(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args)
    multi = len(seeds) > 1
    for seed in seeds:
        _run_one(args, seed=seed, seed_suffix_in_filenames=multi)

    write_summary_if_multi_seed("B1", args.results_dir, seeds, quiet=args.quiet)


if __name__ == "__main__":
    main()
