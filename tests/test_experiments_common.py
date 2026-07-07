"""
Tests for src/experiments/_common.py — the multi-seed plumbing
==============================================================

Validates:
  1. resolve_seeds(args) returns [args.seed] when --seeds is absent.
  2. resolve_seeds(args) returns args.seeds when supplied.
  3. write_summary_if_multi_seed:
       a) returns None and writes nothing when given 0 or 1 seeds.
       b) given N>=2 seeds, reads <baseline>_seed{N}_metrics.json files
          and writes a <baseline>_summary.json whose mean/std fields
          match the per-seed records.
       c) raises FileNotFoundError if any per-seed file is missing.

These exercise the multi-seed contract in isolation (no MuJoCo).

Run from the repo root:
    python tests/test_experiments_common.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

# pylint: disable=wrong-import-position
import _common  # noqa: E402
from _common import resolve_seeds, write_summary_if_multi_seed  # noqa: E402


def _write_metrics(data_dir: Path, baseline: str, seed: int,
                   cov: float, bw: float, pol: float, lat: float) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / f"{baseline}_seed{seed}_metrics.json"
    p.write_text(json.dumps({
        "baseline": baseline,
        "coverage": cov,
        "bandwidth_bps": bw,
        "polarization_accuracy": pol,
        "median_latency_s": lat,
    }))


# ---------------------------------------------------------------------
# Test 1 — resolve_seeds with no --seeds
# ---------------------------------------------------------------------

def test_resolve_seeds_single():
    print("\nTest 1: resolve_seeds falls back to [args.seed] when --seeds missing …")
    args = Namespace(seed=42, seeds=None)
    out = resolve_seeds(args)
    assert out == [42], f"expected [42], got {out}"
    print(f"  ✓ resolve_seeds(seed=42, seeds=None) → {out}")

    args = Namespace(seed=0, seeds=[])      # explicit empty list
    out = resolve_seeds(args)
    assert out == [0], f"empty --seeds should still fall back to [args.seed], got {out}"
    print(f"  ✓ resolve_seeds(seed=0, seeds=[])    → {out}  (empty list ≡ unset)")


# ---------------------------------------------------------------------
# Test 2 — resolve_seeds honours --seeds when given
# ---------------------------------------------------------------------

def test_resolve_seeds_multi():
    print("\nTest 2: resolve_seeds honours --seeds when supplied …")
    args = Namespace(seed=0, seeds=[0, 1, 2, 3, 4])
    out = resolve_seeds(args)
    assert out == [0, 1, 2, 3, 4]
    print(f"  ✓ resolve_seeds(seed=0, seeds=[0..4]) → {out}")

    # Out-of-order seeds preserved
    args = Namespace(seed=99, seeds=[7, 3, 1])
    out = resolve_seeds(args)
    assert out == [7, 3, 1], f"order should be preserved, got {out}"
    print(f"  ✓ resolve_seeds preserves seed order: {out}")


# ---------------------------------------------------------------------
# Test 3 — write_summary_if_multi_seed no-ops on 0/1 seeds
# ---------------------------------------------------------------------

def test_summary_noop_for_single_seed():
    print("\nTest 3: write_summary_if_multi_seed is a no-op for ≤1 seeds …")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Single seed → no summary, returns None
        path = write_summary_if_multi_seed("B1", td, [0], quiet=True)
        assert path is None, f"expected None for single seed, got {path}"
        assert not (td / "data" / "B1_summary.json").exists(), \
            "should not have written a summary file"
        print("  ✓ single-seed → returns None, writes nothing")

        # Empty seed list → also no-op
        path = write_summary_if_multi_seed("B1", td, [], quiet=True)
        assert path is None
        print("  ✓ empty-seed list → returns None, writes nothing")


# ---------------------------------------------------------------------
# Test 4 — write_summary_if_multi_seed aggregates per-seed metrics
# ---------------------------------------------------------------------

def test_summary_aggregates():
    print("\nTest 4: write_summary_if_multi_seed aggregates per-seed metrics …")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_dir = td / "data"
        # Three seeds with KNOWN values whose mean/std are easy to verify
        per_seed = [
            (0, 1.00, 100.0, 1.0, 0.0),
            (1, 0.50, 200.0, 0.5, 0.5),
            (2, 0.00, 300.0, 0.0, 1.0),
        ]
        for seed, cov, bw, pol, lat in per_seed:
            _write_metrics(data_dir, "B3", seed, cov, bw, pol, lat)

        out_path = write_summary_if_multi_seed("B3", td, [0, 1, 2], quiet=True)
        assert out_path is not None and out_path.exists()
        print(f"  ✓ wrote summary at {out_path.name}")

        summary = json.loads(out_path.read_text())
        assert summary["baseline"] == "B3"
        assert summary["n_seeds"] == 3
        assert summary["seeds"] == [0, 1, 2]
        assert len(summary["per_seed"]) == 3
        print("  ✓ summary metadata correct")

        # Known means: cov = 0.5, bw = 200, pol = 0.5, lat = 0.5
        assert abs(summary["coverage"]["mean"] - 0.5) < 1e-9
        assert abs(summary["bandwidth_bps"]["mean"] - 200.0) < 1e-9
        assert abs(summary["polarization_accuracy"]["mean"] - 0.5) < 1e-9
        assert abs(summary["median_latency_s"]["mean"] - 0.5) < 1e-9
        print("  ✓ all four means correct (cov=0.5, bw=200, pol=0.5, lat=0.5)")

        # Population std of [0.0, 0.5, 1.0] = sqrt(((0.5)^2 + 0 + (0.5)^2)/3)
        #   = sqrt(0.5/3) ≈ 0.40825
        import math
        expected_std_pol = math.sqrt(((0.5)**2 + 0 + (0.5)**2) / 3)
        assert abs(summary["polarization_accuracy"]["std"] - expected_std_pol) < 1e-9, \
            f"std mismatch: {summary['polarization_accuracy']['std']} vs {expected_std_pol}"
        print(f"  ✓ population std correct (e.g. pol std ≈ {expected_std_pol:.5f})")

        # min/max/values bookkeeping
        assert summary["coverage"]["min"] == 0.0
        assert summary["coverage"]["max"] == 1.0
        assert summary["coverage"]["values"] == [1.0, 0.5, 0.0]
        print("  ✓ min/max/values arrays populated correctly")


# ---------------------------------------------------------------------
# Test 5 — missing per-seed file raises a clear error
# ---------------------------------------------------------------------

def test_summary_missing_seed_raises():
    print("\nTest 5: write_summary_if_multi_seed raises on missing per-seed file …")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_dir = td / "data"
        _write_metrics(data_dir, "B3", 0, 1.0, 100.0, 1.0, 0.0)
        _write_metrics(data_dir, "B3", 1, 0.5, 100.0, 0.5, 0.5)
        # missing seed=2

        try:
            write_summary_if_multi_seed("B3", td, [0, 1, 2], quiet=True)
            raise AssertionError("expected FileNotFoundError for missing seed 2")
        except FileNotFoundError as e:
            assert "seed2" in str(e), f"error message should reference the missing seed: {e}"
            print("  ✓ FileNotFoundError raised, mentions seed2")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Tests for src/experiments/_common.py multi-seed plumbing")
    print("=" * 60)
    test_resolve_seeds_single()
    test_resolve_seeds_multi()
    test_summary_noop_for_single_seed()
    test_summary_aggregates()
    test_summary_missing_seed_raises()
    print("\nAll _common multi-seed tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
