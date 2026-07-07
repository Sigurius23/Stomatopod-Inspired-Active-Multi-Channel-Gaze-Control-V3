"""
Tests for src/experiments/train_learned_rl.py — REINFORCE training of
the bonus LearnedScheduler MLP.

Validates:
  1. _REINFORCEScheduler subclasses LearnedScheduler and produces a
     valid GimbalSetpoint within joint limits.
  2. _REINFORCEScheduler.trace populates with one (features, log_prob)
     entry per (eye, re-plan) call.
  3. _reinforce_update changes the MLP weights when given a non-zero
     advantage and leaves them unchanged given zero advantage.
  4. A miniature end-to-end training loop (3 episodes × 1 s) runs
     without errors and writes the trained MLP + curve to disk.

These don't assert the *quality* of the learned policy (that would be
flaky given REINFORCE's high variance with small N). They assert the
mechanics are correct.

Run from the repo root:
    python tests/test_learned_rl.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

from train_learned_rl import (  # noqa: E402
    _reinforce_update,
    _REINFORCEScheduler,
    _rollout_episode,
)

from stomatopod_vision._mlp import TinyMLP  # noqa: E402
from stomatopod_vision.gimbal_control import GimbalSetpoint  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    _PITCH_LIMIT_RAD,
    _ROLL_LIMIT_RAD,
    _YAW_LIMIT_RAD,
    SaliencyScheduler,
)

N_FEATURES = len(SaliencyScheduler.FEATURE_NAMES)


# ---------------------------------------------------------------------
# Test 1 — _REINFORCEScheduler produces a valid setpoint
# ---------------------------------------------------------------------

def test_reinforce_scheduler_setpoint():
    print("\nTest 1: _REINFORCEScheduler.next_setpoint produces a valid GimbalSetpoint …")
    mlp = TinyMLP(n_in=N_FEATURES, n_hidden=8, n_out=1, seed=0)
    s = _REINFORCEScheduler(mlp=mlp, seed=0, temperature=1.0)
    sp = s.next_setpoint(0.0, GimbalSetpoint())
    assert isinstance(sp, GimbalSetpoint)
    for v, lim in [
        (sp.yaw_L, _YAW_LIMIT_RAD), (sp.yaw_R, _YAW_LIMIT_RAD),
        (sp.pitch_L, _PITCH_LIMIT_RAD), (sp.pitch_R, _PITCH_LIMIT_RAD),
        (sp.roll_L, _ROLL_LIMIT_RAD), (sp.roll_R, _ROLL_LIMIT_RAD),
    ]:
        assert abs(v) <= lim + 1e-9, f"{v} outside [-{lim}, {lim}]"
    print("  ✓ produces a valid GimbalSetpoint inside joint limits")


# ---------------------------------------------------------------------
# Test 2 — trace populates correctly
# ---------------------------------------------------------------------

def test_reinforce_trace_populates():
    print("\nTest 2: trace accumulates one (features, log_prob) per (eye, re-plan) …")
    mlp = TinyMLP(n_in=N_FEATURES, n_hidden=8, n_out=1, seed=0)
    s = _REINFORCEScheduler(mlp=mlp, seed=0, decision_period_s=0.1)
    assert len(s.trace) == 0

    # First call → 2 entries (one per eye)
    s.next_setpoint(0.0, GimbalSetpoint())
    assert len(s.trace) == 2, f"expected 2 entries, got {len(s.trace)}"

    # Second call within decision period → no new entries
    s.next_setpoint(0.05, GimbalSetpoint())
    assert len(s.trace) == 2, "should not log within decision period"

    # Third call past decision period → 2 more entries
    s.next_setpoint(0.15, GimbalSetpoint())
    assert len(s.trace) == 4, f"expected 4 entries, got {len(s.trace)}"

    # Each trace entry has (12-D features, scalar log_prob)
    for feats, log_p in s.trace:
        assert feats.shape == (N_FEATURES,), f"bad feature shape {feats.shape}"
        assert log_p <= 0.0, f"log_prob must be ≤ 0, got {log_p}"
    print(f"  ✓ trace has {len(s.trace)} entries with correct shapes")


# ---------------------------------------------------------------------
# Test 3 — _reinforce_update changes weights only when advantage ≠ 0
# ---------------------------------------------------------------------

def test_reinforce_update_changes_weights():
    print("\nTest 3: _reinforce_update changes MLP weights ↔ non-zero advantage …")
    mlp = TinyMLP(n_in=N_FEATURES, n_hidden=8, n_out=1, seed=0)
    rng = np.random.default_rng(0)
    trace = [(rng.standard_normal(N_FEATURES), -1.0) for _ in range(20)]

    W1_before = mlp.W1.copy()
    _reinforce_update(mlp, trace, advantage=0.0, lr=1e-2)
    assert np.allclose(mlp.W1, W1_before), \
        "zero advantage should leave weights unchanged (gradient is 0)"
    print("  ✓ advantage=0 leaves weights unchanged")

    _reinforce_update(mlp, trace, advantage=+2.0, lr=1e-2)
    assert not np.allclose(mlp.W1, W1_before), \
        "non-zero advantage should perturb weights"
    print("  ✓ advantage>0 changes weights")


# ---------------------------------------------------------------------
# Test 4 — miniature end-to-end training loop
# ---------------------------------------------------------------------

def test_mini_training_loop_runs():
    """3-episode × 1-s training loop runs without errors and saves outputs."""
    print("\nTest 4: a 3-episode training loop runs end-to-end …")
    mlp = TinyMLP(n_in=N_FEATURES, n_hidden=8, n_out=1, seed=0)
    returns = []
    for ep in range(3):
        R, trace = _rollout_episode(mlp, duration=1.0, seed=ep, temperature=1.0)
        assert isinstance(R, float), f"return must be a float, got {type(R)}"
        assert 0.0 <= R <= 10.0, f"return out of range [0, 10]: {R}"
        # trace will be empty for very short episodes if scheduler hasn't replanned
        # at all, but for 1 s it should have at least one re-plan per eye.
        assert isinstance(trace, list)
        _reinforce_update(mlp, trace, advantage=R - 5.0, lr=1e-2)
        returns.append(R)
    print(f"  ✓ 3 episodes ran; returns = {returns}")

    # Save + reload round-trips
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mlp.npz"
        mlp.save(out)
        m2 = TinyMLP.load(out)
        x = np.random.default_rng(0).standard_normal((10, N_FEATURES))
        assert np.allclose(mlp.forward(x), m2.forward(x)), \
            "save/load roundtrip must preserve MLP outputs"
    print("  ✓ save/load roundtrip OK")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Tests for src/experiments/train_learned_rl.py")
    print("=" * 60)
    test_reinforce_scheduler_setpoint()
    test_reinforce_trace_populates()
    test_reinforce_update_changes_weights()
    test_mini_training_loop_runs()
    print("\nAll REINFORCE training tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
