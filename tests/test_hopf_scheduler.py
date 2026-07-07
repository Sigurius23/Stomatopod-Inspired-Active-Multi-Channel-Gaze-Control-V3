"""
Tests for the dynamical (limit-cycle) scheduler — HopfScanScheduler
===================================================================

These tests validate the *dynamical-systems* properties the scheduler
claims, not just that it produces output:

  1. Interface conformance — it IS a BaseScheduler and is a drop-in for
     the experiment harness' ``next_setpoint`` / ``update_memory`` calls.
  2. Stable limit cycle — with mu>0 the oscillator converges to radius
     sqrt(mu) from any nonzero start, and stays there.
  3. Hopf bifurcation — detecting a target drops mu below zero so the
     oscillator spirals to the origin and the eye FIXATES the target;
     after the dwell it bifurcates back to a scanning limit cycle.
  4. Setpoints always respect the gimbal joint limits.
  5. Determinism — same seed ⇒ identical trajectory.
  6. End-to-end — a real MuJoCo B3 run with this scheduler reaches full
     coverage on the canonical scene.

Run:
    python tests/test_hopf_scheduler.py
    MUJOCO_GL=glfw pytest tests/test_hopf_scheduler.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))

from stomatopod_vision.gimbal_control import GimbalSetpoint  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessedEvent  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    BaseScheduler,
    HopfScanScheduler,
)

_LIMITS = (1.57, 1.05, 3.14)   # yaw, pitch, roll — matches the XML


def _event(eye: str, az_deg: float, el_deg: float,
           name: str = "prey", t: float = 0.0) -> PreprocessedEvent:
    return PreprocessedEvent(
        time=t, eye=eye, target_name=name,
        azimuth=np.deg2rad(az_deg), elevation=np.deg2rad(el_deg),
        distance=1.0, spectral_pattern=(0,) * 12,
        polarization_angle=None, circular_handedness=None,
    )


def _drive(sched: HopfScanScheduler, t0: float, t1: float, dt: float) -> GimbalSetpoint:
    """Poll next_setpoint from t0 to t1; return the last setpoint."""
    sp = GimbalSetpoint()
    t = t0
    while t < t1:
        t += dt
        sp = sched.next_setpoint(t, current_setpoint=sched._held_setpoint)
    return sp


# ---------------------------------------------------------------------
# Test 1 — interface conformance
# ---------------------------------------------------------------------

def test_is_base_scheduler():
    print("\nTest 1: HopfScanScheduler conforms to BaseScheduler …")
    s = HopfScanScheduler()
    assert isinstance(s, BaseScheduler)
    # Drop-in for the harness pattern.
    sp = s.next_setpoint(0.01, current_setpoint=s._held_setpoint)
    assert isinstance(sp, GimbalSetpoint)
    s.update_memory([], 0.01)   # empty batch is a no-op
    print("  ✓ is a BaseScheduler and accepts the harness call pattern")


# ---------------------------------------------------------------------
# Test 2 — stable limit cycle at radius sqrt(mu)
# ---------------------------------------------------------------------

def test_limit_cycle_is_stable():
    print("\nTest 2: mu>0 → stable limit cycle at radius sqrt(mu) …")
    for mu in (1.0, 4.0):
        s = HopfScanScheduler(mu_scan=mu)
        # Start far off the cycle (both inside and outside).
        for start in (np.array([0.02, 0.0]), np.array([3.0, 0.0])):
            s.reset()
            s._osc["L"] = start.copy()
            _drive(s, 0.0, 12.0, 0.01)
            r = s.limit_cycle_radius("L")
            assert abs(r - np.sqrt(mu)) < 0.02, (
                f"mu={mu}: radius {r:.4f} ≠ sqrt(mu)={np.sqrt(mu):.4f} "
                f"from start {start.tolist()}")
        print(f"  ✓ mu={mu}: converges to radius {np.sqrt(mu):.3f} from in/outside")


def test_limit_cycle_is_persistent():
    print("\nTest 3: the limit cycle is a sustained oscillation …")
    s = HopfScanScheduler()
    _drive(s, 0.0, 3.0, 0.01)          # settle onto the cycle
    xs = []
    t = 3.0
    for _ in range(400):
        t += 0.01
        s.next_setpoint(t, current_setpoint=s._held_setpoint)
        xs.append(s._osc["L"][0])
    xs = np.array(xs)
    # A genuine oscillation swings through both signs with O(1) amplitude.
    assert xs.max() > 0.8 and xs.min() < -0.8, (
        f"expected a full-amplitude oscillation, got range "
        f"[{xs.min():.2f}, {xs.max():.2f}]")
    print(f"  ✓ sustained oscillation, x ∈ [{xs.min():.2f}, {xs.max():.2f}]")


# ---------------------------------------------------------------------
# Test 4 — Hopf bifurcation: detect → fixate → resume scan
# ---------------------------------------------------------------------

def test_bifurcation_to_fixation_and_back():
    print("\nTest 4: detection triggers a Hopf bifurcation to fixation …")
    s = HopfScanScheduler(fixation_dwell_s=0.4)
    # Settle onto the scan cycle.
    _drive(s, 0.0, 3.0, 0.01)
    assert s._mode["L"] == "scan"
    assert abs(s.limit_cycle_radius("L") - 1.0) < 0.05

    # Detect a target ~20° right, 8° up of the eye's current gaze.
    t = 3.0
    s.next_setpoint(t, current_setpoint=s._held_setpoint)
    s.update_memory([_event("L", az_deg=20.0, el_deg=-8.0, t=t)], t)
    assert s._mode["L"] == "fixate", "detection should bifurcate to fixation"
    target = s._fixate_target["L"]

    # Sample partway through the dwell (0.35 s of a 0.4 s window, so we are
    # firmly still fixating): the oscillator should have spiralled toward the
    # origin and the gaze should sit on the target direction.
    sp = _drive(s, t, t + 0.35, 0.005)
    assert s._mode["L"] == "fixate", "should still be fixating mid-dwell"
    assert s.limit_cycle_radius("L") < 0.2, (
        f"oscillator should collapse during fixation, r="
        f"{s.limit_cycle_radius('L'):.3f}")
    err = np.hypot(sp.yaw_L - target[0], sp.pitch_L - target[1])
    assert err < 0.1, f"eye should foveate the target, gaze err={err:.3f} rad"
    print(f"  ✓ fixation: radius→{s.limit_cycle_radius('L'):.3f}, "
          f"gaze within {np.rad2deg(err):.1f}° of target")

    # After the dwell + recovery it bifurcates back to a scanning cycle.
    _drive(s, t + 0.4, t + 3.0, 0.01)
    assert s._mode["L"] == "scan", "should return to scanning after dwell"
    assert abs(s.limit_cycle_radius("L") - 1.0) < 0.05, (
        f"scan cycle should recover, r={s.limit_cycle_radius('L'):.3f}")
    print(f"  ✓ bifurcates back to scanning, radius {s.limit_cycle_radius('L'):.3f}")


def test_refixate_cooldown():
    print("\nTest 5: the same target is not re-fixated during its cooldown …")
    s = HopfScanScheduler(fixation_dwell_s=0.2, refixate_cooldown_s=1.0)
    t = 1.0
    s.next_setpoint(t, current_setpoint=s._held_setpoint)
    s.update_memory([_event("L", 10, -3, name="crab", t=t)], t)
    assert s._mode["L"] == "fixate"
    # Let the fixation lapse back to scanning.
    _drive(s, t, t + 0.5, 0.005)
    assert s._mode["L"] == "scan"
    # Same target again, still inside the cooldown window → no re-fixation.
    t2 = t + 0.6
    s.update_memory([_event("L", 10, -3, name="crab", t=t2)], t2)
    assert s._mode["L"] == "scan", "re-fixation should be suppressed by cooldown"
    print("  ✓ cooldown suppresses immediate re-fixation of the same target")


# ---------------------------------------------------------------------
# Test 6 — setpoints respect joint limits
# ---------------------------------------------------------------------

def test_setpoints_within_joint_limits():
    print("\nTest 6: every setpoint stays inside the gimbal joint limits …")
    s = HopfScanScheduler(joint_limits=_LIMITS)
    yl, pl, rl = _LIMITS
    t = 0.0
    for _ in range(4000):
        t += 0.004
        # Occasionally inject detections to exercise fixation excursions.
        if int(t * 10) % 7 == 0:
            s.update_memory([_event("L", 80, 40, name=f"t{int(t)}", t=t),
                             _event("R", -80, -40, name=f"u{int(t)}", t=t)], t)
        sp = s.next_setpoint(t, current_setpoint=s._held_setpoint)
        for v, lim in ((sp.yaw_L, yl), (sp.pitch_L, pl), (sp.roll_L, rl),
                       (sp.yaw_R, yl), (sp.pitch_R, pl), (sp.roll_R, rl)):
            assert abs(v) <= lim + 1e-9, f"setpoint {v} exceeds limit {lim}"
    print("  ✓ all setpoints within limits across a 16 s driven run")


# ---------------------------------------------------------------------
# Test 7 — determinism
# ---------------------------------------------------------------------

def test_determinism():
    print("\nTest 7: identical seed ⇒ identical trajectory …")

    def traj(seed):
        s = HopfScanScheduler(seed=seed)
        out = []
        t = 0.0
        for _ in range(500):
            t += 0.01
            sp = s.next_setpoint(t, current_setpoint=s._held_setpoint)
            out.append((sp.yaw_L, sp.pitch_L, sp.roll_R))
        return np.array(out)

    assert np.allclose(traj(0), traj(0))
    print("  ✓ deterministic given a fixed seed")


# ---------------------------------------------------------------------
# Test 8 — end-to-end coverage on a real MuJoCo run
# ---------------------------------------------------------------------

def test_end_to_end_coverage():
    print("\nTest 8: full B3 run with HopfScanScheduler reaches coverage 1.0 …")
    from _common import build_context, run_simulation

    from stomatopod_vision.metrics import compute_all
    from stomatopod_vision.preprocessing import PreprocessingPipeline

    ctx = build_context(str(REPO_ROOT / "models" / "stomatopod_eyes.xml"), seed=0)
    pipe = PreprocessingPipeline()
    sch = HopfScanScheduler(seed=0)
    log = run_simulation(
        ctx,
        setpoint_at=lambda t: sch.next_setpoint(t, current_setpoint=sch._held_setpoint),
        pipeline=pipe, duration_s=8.0, quiet=True,
        on_events=lambda ev, t: sch.update_memory(ev, t),
    )
    rep = compute_all(log, baseline="B3")
    assert rep.coverage == 1.0, f"expected full coverage, got {rep.coverage}"
    assert rep.polarization_accuracy == 1.0
    assert rep.circular_polarization_accuracy == 1.0
    print(f"  ✓ coverage={rep.coverage:.3f} pol={rep.polarization_accuracy:.3f} "
          f"circ={rep.circular_polarization_accuracy:.3f} bw={rep.bandwidth_bps:,.0f} B/s")


def main() -> None:
    print("=" * 60)
    print("  HopfScanScheduler — dynamical-systems scheduler tests")
    print("=" * 60)
    test_is_base_scheduler()
    test_limit_cycle_is_stable()
    test_limit_cycle_is_persistent()
    test_bifurcation_to_fixation_and_back()
    test_refixate_cooldown()
    test_setpoints_within_joint_limits()
    test_determinism()
    test_end_to_end_coverage()
    print("\nAll HopfScanScheduler tests passed. ✓")


if __name__ == "__main__":
    main()
