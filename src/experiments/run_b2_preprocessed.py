"""
run_b2_preprocessed.py — Baseline B2: fixed cameras, with preprocessing
=======================================================================

Identical to B1 except that the full Layer 2 preprocessing pipeline is
attached:
    - Both eye gimbals are held at the rest pose (looking straight forward).
    - Raw multi-channel sensor data is run through:
        1. Mid-band channel reduction (pick dominant spectral row)
        2. Polarization decoding (vector-sum inversion)
        3. Event encoding (suppress unchanged sightings)
    - Only emitted :class:`PreprocessedEvent` instances are counted
      against bandwidth — so the comparison vs B1 isolates the value
      of moving computation closer to the sensor.

Coverage and polarization accuracy should be similar to B1 (the eyes
see the same scene). The headline difference is in bandwidth.

CLI
---
    python src/experiments/run_b2_preprocessed.py \\
        --duration 10 \\
        --seed 0

Outputs
-------
    results/data/B2_metrics.json — four headline metrics for the run.
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
    """One end-to-end B2 simulation for a single seed."""
    if not args.quiet:
        print("=" * 60)
        print(f"  Baseline B2 — fixed cameras + Layer 2 preprocessing"
              f"  (seed={seed})")
        print("=" * 60)
        print(f"  duration   : {args.duration:.1f} s")
        print(f"  seed       : {seed}")
        print(f"  model      : {args.model}")
        print(f"  results to : {args.results_dir}")
        print()

    ctx = build_context(args.model, seed=seed)

    # B2 = eyes locked forward + full Layer 2 pipeline
    fixed_setpoint = GimbalSetpoint()
    pipeline = PreprocessingPipeline()
    log = run_simulation(
        ctx,
        setpoint_at=lambda _t: fixed_setpoint,
        pipeline=pipeline,    # ← Layer 2 attached
        duration_s=args.duration,
        quiet=args.quiet,
    )

    save_and_report(log, baseline="B2",
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

    write_summary_if_multi_seed("B2", args.results_dir, seeds, quiet=args.quiet)


if __name__ == "__main__":
    main()
