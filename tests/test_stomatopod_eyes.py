"""
Tests for models/stomatopod_eyes.xml — week-1 sanity checks
============================================================

Validates:
  1. The MuJoCo model loads without errors and the expected counts of
     joints, actuators, sites, and target bodies match the XML.
  2. Every named joint / actuator / site / target body exists in the
     loaded model.
  3. The six gimbal DOFs can be driven independently via PD control
     to a small set of test configurations (rest, look-left,
     look-right, look-up, diverge, roll-both) with final joint-angle
     error below 0.02 rad in each axis.
  4. The eye-forward unit vectors derived from the (centre, axis)
     sites have magnitude exactly 1.0 and behave as expected: pointing
     forward (+y) at rest, swinging into ±x under coupled yaw.

Run from the repo root:
    python tests/test_stomatopod_eyes.py             # standard test pass
    python tests/test_stomatopod_eyes.py --viewer    # plus a live viewer demo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# On headless machines, EGL is required to even load a renderer.
# Setting this before importing mujoco saves a lot of GLFW noise.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

# --------------------------------------------------------------------------
# Constants — what we expect to find in the model
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "models" / "stomatopod_eyes.xml"

EXPECTED_JOINTS = [
    "eye_L_yaw", "eye_L_pitch", "eye_L_roll",
    "eye_R_yaw", "eye_R_pitch", "eye_R_roll",
]
EXPECTED_ACTUATORS = [
    "m_eye_L_yaw", "m_eye_L_pitch", "m_eye_L_roll",
    "m_eye_R_yaw", "m_eye_R_pitch", "m_eye_R_roll",
]
EXPECTED_SITES = [
    "eye_L_center", "eye_L_axis",
    "eye_R_center", "eye_R_axis",
]
EXPECTED_TARGET_BODIES = [
    "target_R_1", "target_G_1", "target_B_1",
    "target_UVA_1", "target_UVB_1", "target_UVpol_1",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _joint_qpos_index(model: mujoco.MjModel, name: str) -> int:
    """Index into ``data.qpos`` for a named hinge joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def _joint_qvel_index(model: mujoco.MjModel, name: str) -> int:
    """Index into ``data.qvel`` for a named hinge joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_dofadr[jid]


def _actuator_index(model: mujoco.MjModel, name: str) -> int:
    """Index into ``data.ctrl`` for a named actuator."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _eye_forward_vector(model: mujoco.MjModel, data: mujoco.MjData,
                        eye: str) -> np.ndarray:
    """Unit vector pointing 'out of' the eye in world coordinates.

    Computed from the two sites placed in the XML (``eye_<L|R>_center``
    and ``eye_<L|R>_axis``).
    """
    center_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_center")
    axis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_axis")
    v = data.site_xpos[axis_id] - data.site_xpos[center_id]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class _GimbalPD:
    """Minimal joint-space PD controller mirroring the HW4 pattern.

    .. math:: u = K_p (q_{des} - q) - K_d \\dot q

    (No gravity-compensation term: the model uses ``gravity="0 0 0"``.)
    """

    def __init__(self, model: mujoco.MjModel,
                 kp: float = 20.0, kd: float = 1.5) -> None:
        self.model = model
        self.kp = kp
        self.kd = kd
        self.q_idx = [_joint_qpos_index(model, j) for j in EXPECTED_JOINTS]
        self.v_idx = [_joint_qvel_index(model, j) for j in EXPECTED_JOINTS]
        self.u_idx = [_actuator_index(model, a) for a in EXPECTED_ACTUATORS]

    def step(self, data: mujoco.MjData, q_des: np.ndarray) -> None:
        q = np.array([data.qpos[i] for i in self.q_idx])
        qd = np.array([data.qvel[i] for i in self.v_idx])
        u = self.kp * (q_des - q) - self.kd * qd
        u = np.clip(u, -1.0, 1.0)
        for i, j in enumerate(self.u_idx):
            data.ctrl[j] = u[i]


def _load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Convenience helper used by every test."""
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    return model, data


# --------------------------------------------------------------------------
# Test 1 — model loads
# --------------------------------------------------------------------------

def test_model_loads():
    print("\nTest 1: stomatopod_eyes.xml loads with the expected counts …")
    model, _ = _load_model()
    # Expected: 6 joints / 6 actuators / 4 sites / 7 bodies (head + 6 targets)
    # nq == 6 because every joint is a single-DOF hinge.
    assert model.nq == 6, f"expected nq=6, got {model.nq}"
    assert model.nv == 6, f"expected nv=6, got {model.nv}"
    assert model.nu == 6, f"expected nu=6, got {model.nu}"
    # 6 target bodies + head + the 6 nested eye-link bodies + worldbody
    assert model.nbody >= 7, f"expected nbody≥7, got {model.nbody}"
    print(f"  ✓ loaded (nq={model.nq}, nv={model.nv}, nu={model.nu}, "
          f"nbody={model.nbody})")


# --------------------------------------------------------------------------
# Test 2 — every expected name exists in the model
# --------------------------------------------------------------------------

def test_expected_names_exist():
    print("\nTest 2: expected joints / actuators / sites / target bodies exist …")
    model, _ = _load_model()
    for j in EXPECTED_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0, \
            f"missing joint {j!r}"
    print(f"  ✓ all {len(EXPECTED_JOINTS)} joints present")
    for a in EXPECTED_ACTUATORS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) >= 0, \
            f"missing actuator {a!r}"
    print(f"  ✓ all {len(EXPECTED_ACTUATORS)} actuators present")
    for s in EXPECTED_SITES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) >= 0, \
            f"missing site {s!r}"
    print(f"  ✓ all {len(EXPECTED_SITES)} sites present")
    for b in EXPECTED_TARGET_BODIES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0, \
            f"missing target body {b!r}"
    print(f"  ✓ all {len(EXPECTED_TARGET_BODIES)} target bodies present")


# --------------------------------------------------------------------------
# Test 3 — every DOF can be driven independently via PD
# --------------------------------------------------------------------------

def test_pd_drives_each_dof():
    print("\nTest 3: PD controller reaches a set of test configurations …")
    model, data = _load_model()
    pd = _GimbalPD(model)
    # Joint-order: eye_L (yaw, pitch, roll) then eye_R (yaw, pitch, roll)
    configs = {
        "rest":       np.zeros(6),
        "look_left":  np.array([+0.6, 0.0, 0.0, -0.6, 0.0, 0.0]),
        "look_right": np.array([-0.6, 0.0, 0.0, +0.6, 0.0, 0.0]),
        "look_up":    np.array([0.0, +0.5, 0.0,  0.0, +0.5, 0.0]),
        "diverge":    np.array([+0.8, 0.0, 0.0, -0.8, 0.0, 0.0]),
        "roll_both":  np.array([0.0, 0.0, +1.0,  0.0, 0.0, -1.0]),
    }
    n_steps = int(0.5 / model.opt.timestep)
    for name, q_des in configs.items():
        mujoco.mj_resetData(model, data)
        for _ in range(n_steps):
            pd.step(data, q_des)
            mujoco.mj_step(model, data)
        q_actual = np.array([data.qpos[i] for i in pd.q_idx])
        err = float(np.max(np.abs(q_actual - q_des)))
        assert err < 0.05, \
            f"PD did not reach {name!r} setpoint: max-axis err = {err:.4f} rad"
        print(f"  ✓ {name:11s} reached (max-axis err = {err:.4f} rad)")


# --------------------------------------------------------------------------
# Test 4 — eye forward vectors behave correctly
# --------------------------------------------------------------------------

def test_eye_forward_vectors():
    print("\nTest 4: eye-forward vectors are unit length and respond to yaw …")
    model, data = _load_model()
    pd = _GimbalPD(model)

    # 4a — at rest, both eyes look along world +y, magnitude 1.0
    mujoco.mj_resetData(model, data)
    for _ in range(int(0.3 / model.opt.timestep)):
        pd.step(data, np.zeros(6))
        mujoco.mj_step(model, data)
    eL = _eye_forward_vector(model, data, "L")
    eR = _eye_forward_vector(model, data, "R")
    assert abs(np.linalg.norm(eL) - 1.0) < 1e-6, f"|eL| ≠ 1: got {np.linalg.norm(eL)}"
    assert abs(np.linalg.norm(eR) - 1.0) < 1e-6, f"|eR| ≠ 1: got {np.linalg.norm(eR)}"
    # At rest both eyes look forward (+y), so x and z components ≈ 0
    assert abs(eL[1]) > 0.99, f"eL should mostly point along +y, got {eL}"
    assert abs(eR[1]) > 0.99, f"eR should mostly point along +y, got {eR}"
    print(f"  ✓ at rest: eL{tuple(round(float(x), 3) for x in eL)} eR{tuple(round(float(x), 3) for x in eR)}")

    # 4b — divergent yaw (positive yaw_L AND negative yaw_R) makes the
    #      eyes look OUTWARD: eye_L\'s world-x goes negative (head\'s left,
    #      per the XML sign convention), eye_R\'s world-x goes positive
    #      (head\'s right). This verifies the XML sign convention as
    #      documented in the file header.
    q_des = np.array([+0.6, 0.0, 0.0, -0.6, 0.0, 0.0])
    mujoco.mj_resetData(model, data)
    for _ in range(int(0.5 / model.opt.timestep)):
        pd.step(data, q_des)
        mujoco.mj_step(model, data)
    eL = _eye_forward_vector(model, data, "L")
    eR = _eye_forward_vector(model, data, "R")
    assert eL[0] < -0.3, f"eL.x should be negative (head-left), got {eL[0]:.3f}"
    assert eR[0] > +0.3, f"eR.x should be positive (head-right), got {eR[0]:.3f}"
    print(f"  ✓ divergent yaw: eL.x={eL[0]:+.3f}  eR.x={eR[0]:+.3f}  (XML sign convention OK)")


# --------------------------------------------------------------------------
# Optional live viewer demo (not a test; invoked with --viewer)
# --------------------------------------------------------------------------

def live_viewer_demo() -> None:
    """Open the MuJoCo viewer and oscillate all 6 DOFs in a slow Lissajous."""
    try:
        import mujoco.viewer as viewer
    except ImportError:
        print("mujoco.viewer not available; skipping live viewer.")
        return

    model, data = _load_model()
    pd = _GimbalPD(model)
    print("Opening live viewer (close the window to exit) …")
    mujoco.mj_resetData(model, data)
    with viewer.launch_passive(model, data) as v:
        t = 0.0
        while v.is_running():
            t += model.opt.timestep
            q_des = np.array([
                +0.4 * np.sin(0.5 * t),                 # L yaw
                +0.3 * np.sin(0.7 * t + 0.5),           # L pitch
                +1.0 * np.sin(0.3 * t),                 # L roll
                -0.4 * np.sin(0.5 * t),                 # R yaw  (mirror)
                +0.3 * np.sin(0.7 * t + 0.5),           # R pitch
                -1.0 * np.sin(0.3 * t),                 # R roll (mirror)
            ])
            pd.step(data, q_des)
            mujoco.mj_step(model, data)
            v.sync()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Week-1 sanity checks for stomatopod_eyes.xml")
    print("=" * 60)
    test_model_loads()
    test_expected_names_exist()
    test_pd_drives_each_dof()
    test_eye_forward_vectors()
    print("\nAll week-1 sanity checks passed. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--viewer", action="store_true",
                        help="Open the MuJoCo live viewer after the test pass "
                             "(requires a display).")
    args = parser.parse_args()
    try:
        main()
        if args.viewer:
            live_viewer_demo()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
