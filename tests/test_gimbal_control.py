"""
Tests for stomatopod_vision.gimbal_control

Validates:
  1. GimbalSetpoint round-trips correctly through as_vector / from_vector.
  2. GimbalPD finds all joints and actuators in the XML.
  3. The bias term is well-defined (≈ 0 with zero gravity).
  4. PD control drives all 6 DOFs to a non-trivial setpoint within
     reasonable tolerance.
  5. Cross-axis isolation: commanding one joint doesn't move the others.
  6. Scalar vs length-6 gain broadcasting both work.

Run from the repo root:
    python tests/test_gimbal_control.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

# Make `src/` importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision.gimbal_control import (  # noqa: E402
    ACTUATOR_ORDER,
    JOINT_ORDER,
    GimbalPD,
    GimbalSetpoint,
)

XML_PATH = REPO_ROOT / "models" / "stomatopod_eyes.xml"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_sim():
    """Load the model and return (model, data) with state reset."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data


def settle(model, data, pd, setpoint, duration_s=0.5):
    """Run the PD controller for `duration_s` of simulation time."""
    n_steps = int(duration_s / model.opt.timestep)
    for _ in range(n_steps):
        pd.step(data, setpoint)
        mujoco.mj_step(model, data)


# ---------------------------------------------------------------------
# Test 1 — GimbalSetpoint round-trip
# ---------------------------------------------------------------------

def test_setpoint_round_trip():
    print("Test 1: GimbalSetpoint round-trip …")
    sp = GimbalSetpoint(yaw_L=0.1, pitch_L=-0.2, roll_L=0.3,
                        yaw_R=-0.4, pitch_R=0.5, roll_R=-0.6)
    v = sp.as_vector()
    assert v.shape == (6,)
    assert np.allclose(v, [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    sp2 = GimbalSetpoint.from_vector(v)
    assert sp == sp2
    print("  ✓ as_vector / from_vector / __eq__ all consistent")

    # Wrong-shape input should raise
    try:
        GimbalSetpoint.from_vector(np.array([1, 2, 3]))
    except ValueError:
        print("  ✓ from_vector raises on wrong length")
    else:
        raise AssertionError("from_vector should have raised on length-3 input")


# ---------------------------------------------------------------------
# Test 2 — index lookups succeed
# ---------------------------------------------------------------------

def test_indices_resolve():
    print("\nTest 2: PD controller resolves all joint/actuator names …")
    model, _ = make_sim()
    pd = GimbalPD(model)
    assert pd._qpos_idx.shape == (6,)
    assert pd._qvel_idx.shape == (6,)
    assert pd._ctrl_idx.shape == (6,)
    # All indices should be non-negative
    assert (pd._qpos_idx >= 0).all()
    assert (pd._qvel_idx >= 0).all()
    assert (pd._ctrl_idx >= 0).all()
    print(f"  ✓ qpos indices: {pd._qpos_idx.tolist()}")
    print(f"  ✓ qvel indices: {pd._qvel_idx.tolist()}")
    print(f"  ✓ ctrl indices: {pd._ctrl_idx.tolist()}")


# ---------------------------------------------------------------------
# Test 3 — bias term is ~0 (gravity disabled in this model)
# ---------------------------------------------------------------------

def test_bias_near_zero():
    print("\nTest 3: bias term is ~0 (gravity disabled) …")
    model, data = make_sim()
    pd = GimbalPD(model)
    mujoco.mj_forward(model, data)
    bias = pd.bias(data)
    assert bias.shape == (6,)
    assert np.allclose(bias, 0.0, atol=1e-9), f"unexpected bias: {bias}"
    print(f"  ✓ bias = {bias} (within 1e-9 of zero)")


# ---------------------------------------------------------------------
# Test 4 — PD drives the gimbal to a non-trivial setpoint
# ---------------------------------------------------------------------

def test_pd_tracks_setpoint():
    print("\nTest 4: PD tracks a non-trivial setpoint …")
    model, data = make_sim()
    pd = GimbalPD(model, kp=50.0, kd=1.0)

    sp = GimbalSetpoint(yaw_L=0.4, pitch_L=0.3, roll_L=0.5,
                        yaw_R=-0.4, pitch_R=-0.3, roll_R=-0.5)
    settle(model, data, pd, sp, duration_s=1.0)

    err = pd.error(data, sp)
    err_norm = float(np.linalg.norm(err))
    print(f"  final error: {err.round(3).tolist()}")
    print(f"  ‖error‖ = {err_norm:.4f} rad")
    assert err_norm < 0.05, f"PD did not converge: ‖err‖={err_norm:.3f}"
    print("  ✓ PD tracking error below 0.05 rad (~3°)")


# ---------------------------------------------------------------------
# Test 5 — cross-axis isolation
# ---------------------------------------------------------------------

def test_cross_axis_isolation():
    print("\nTest 5: cross-axis isolation (one joint at a time) …")
    model, _ = make_sim()
    pd = GimbalPD(model, kp=50.0, kd=1.0)

    target = 0.5
    for i, joint_name in enumerate(JOINT_ORDER):
        _, data = make_sim()  # fresh data for each subtest
        q_des = np.zeros(6)
        q_des[i] = target
        sp = GimbalSetpoint.from_vector(q_des)
        settle(model, data, pd, sp, duration_s=1.0)

        q_actual = pd.current_q(data)
        # The commanded joint should be near target
        on_target = abs(q_actual[i] - target)
        # All other joints should remain near 0
        off_others = np.delete(q_actual, i)
        max_other = float(np.max(np.abs(off_others)))

        ok = on_target < 0.05 and max_other < 0.02
        marker = "✓" if ok else "✗"
        print(f"  {marker} {joint_name:14s}  "
              f"on-target err={on_target:.3f}  "
              f"max off-axis={max_other:.3f}")
        assert ok, f"cross-axis isolation failed for {joint_name}"


# ---------------------------------------------------------------------
# Test 6 — scalar vs length-6 gain broadcasting
# ---------------------------------------------------------------------

def test_gain_broadcasting():
    print("\nTest 6: gain broadcasting (scalar and length-6) …")
    model, _ = make_sim()

    # Scalar
    pd_scalar = GimbalPD(model, kp=10.0, kd=1.0)
    assert pd_scalar.kp.shape == (6,)
    assert np.allclose(pd_scalar.kp, 10.0)
    assert np.allclose(pd_scalar.kd, 1.0)
    print("  ✓ scalar gain broadcasts to length-6")

    # Length-6
    kp_vec = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    pd_vec = GimbalPD(model, kp=kp_vec, kd=1.0)
    assert np.allclose(pd_vec.kp, kp_vec)
    print("  ✓ length-6 gain passes through")

    # Wrong length
    try:
        GimbalPD(model, kp=np.array([1.0, 2.0, 3.0]))
    except ValueError:
        print("  ✓ length-3 gain raises ValueError")
    else:
        raise AssertionError("GimbalPD should reject length-3 gains")


# ---------------------------------------------------------------------
# Test 7 — ctrl is clipped to actuator range
# ---------------------------------------------------------------------

def test_ctrl_clipped():
    print("\nTest 7: control output is clipped to ±ctrl_clip …")
    model, data = make_sim()
    # Set absurdly high gain so the raw PD output saturates
    pd = GimbalPD(model, kp=10_000.0, kd=0.0, ctrl_clip=1.0)
    sp = GimbalSetpoint(yaw_L=1.0)  # large error → would produce u ≈ 10000
    u = pd.step(data, sp)
    assert np.all(np.abs(u) <= 1.0 + 1e-9), f"unclipped output: {u}"
    print(f"  ✓ all 6 commanded torques within ±1.0: max|u| = {np.max(np.abs(u)):.3f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.gimbal_control")
    print("=" * 60)
    test_setpoint_round_trip()
    test_indices_resolve()
    test_bias_near_zero()
    test_pd_tracks_setpoint()
    test_cross_axis_isolation()
    test_gain_broadcasting()
    test_ctrl_clipped()
    print("\nAll gimbal-control tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
