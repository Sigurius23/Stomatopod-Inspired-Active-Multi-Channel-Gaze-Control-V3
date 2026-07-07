"""
gimbal_control.py — low-level gimbal PD control (Layer 1)
=========================================================

Implements the joint-space PD controller that drives the 6 gimbal DOFs
(yaw, pitch, roll × 2 eyes) to a desired configuration. This is the same
template as Homework 4 and Mock Exam Q12:

    .. math::

        u = K_p (q^* - q) - K_d \\dot q + \\text{qfrc\\_bias}

Because gravity is disabled in ``models/stomatopod_eyes.xml`` (the gimbals
are small and the head is fixed), ``qfrc_bias`` is included for correctness
but is essentially zero in this model. We keep it in the formula so the
controller still works if a future revision turns gravity back on.

This module is intentionally simple — all the interesting choices live
in the scheduler (:mod:`scheduler`) that picks ``q*`` at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# ---------------------------------------------------------------------
# Joint and actuator naming (must match stomatopod_eyes.xml)
# ---------------------------------------------------------------------

JOINT_ORDER: tuple[str, ...] = (
    "eye_L_yaw", "eye_L_pitch", "eye_L_roll",
    "eye_R_yaw", "eye_R_pitch", "eye_R_roll",
)
ACTUATOR_ORDER: tuple[str, ...] = (
    "m_eye_L_yaw", "m_eye_L_pitch", "m_eye_L_roll",
    "m_eye_R_yaw", "m_eye_R_pitch", "m_eye_R_roll",
)


# ---------------------------------------------------------------------
# Gimbal setpoint
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GimbalSetpoint:
    """
    Desired (yaw, pitch, roll) for both eyes in radians.

    Note on sign conventions (see also ``models/stomatopod_eyes.xml`` header):
    - Positive yaw on eye_L turns it to the head's LEFT (world -X).
    - Positive yaw on eye_R turns it to the head's RIGHT (world +X).
      → For coupled motion the caller usually sets
        ``yaw_L = -yaw_R``.
    - Positive pitch tilts the eye DOWN. To expose "elevation = up
      positive" to the scheduler, negate the value before storing.
    - Roll is unconstrained; the polarization decoder uses it directly.
    """
    yaw_L: float = 0.0
    pitch_L: float = 0.0
    roll_L: float = 0.0
    yaw_R: float = 0.0
    pitch_R: float = 0.0
    roll_R: float = 0.0

    def as_vector(self) -> np.ndarray:
        """Return the 6-vector in :data:`JOINT_ORDER`."""
        return np.array(
            [self.yaw_L, self.pitch_L, self.roll_L,
             self.yaw_R, self.pitch_R, self.roll_R],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, q: np.ndarray) -> "GimbalSetpoint":
        """Inverse of :meth:`as_vector`. Useful for tests and logging."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 6:
            raise ValueError(f"Expected length-6 vector, got shape {q.shape}")
        return cls(
            yaw_L=q[0],   pitch_L=q[1], roll_L=q[2],
            yaw_R=q[3],   pitch_R=q[4], roll_R=q[5],
        )


# ---------------------------------------------------------------------
# PD controller
# ---------------------------------------------------------------------

class GimbalPD:
    """
    Drives the 6 gimbal joints to a desired :class:`GimbalSetpoint` using
    joint-space PD plus bias compensation.

    Parameters
    ----------
    model :
        Loaded MuJoCo model.
    kp :
        Proportional gain. Scalar (applied to all 6) or length-6 array.
    kd :
        Derivative gain. Scalar or length-6 array.
    ctrl_clip :
        Saturation bound matching the XML's actuator ``ctrlrange``.

    Notes
    -----
    The control law applied each step is::

        u = kp * (q* - q) - kd * qdot + qfrc_bias[joint_dofs]
        u = clip(u, -ctrl_clip, +ctrl_clip)

    All six DOFs share the same gains by default; pass length-6 arrays
    if you need per-axis tuning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        kp: float | np.ndarray = 50.0,
        kd: float | np.ndarray = 1.0,
        ctrl_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.kp = self._broadcast_gain(kp, "kp")
        self.kd = self._broadcast_gain(kd, "kd")
        self.ctrl_clip = float(ctrl_clip)

        # Index look-ups (cached at construction)
        self._qpos_idx: np.ndarray  # shape (6,)
        self._qvel_idx: np.ndarray  # shape (6,)
        self._ctrl_idx: np.ndarray  # shape (6,)
        self._cache_indices()

    # ----- internal: validation & caching --------------------------------
    @staticmethod
    def _broadcast_gain(gain: float | np.ndarray, name: str) -> np.ndarray:
        """Convert a scalar or length-6 array into a length-6 array."""
        g = np.asarray(gain, dtype=float).reshape(-1)
        if g.size == 1:
            return np.full(6, float(g[0]))
        if g.size == 6:
            return g.copy()
        raise ValueError(f"{name} must be a scalar or length-6 array, "
                         f"got shape {np.asarray(gain).shape}")

    def _cache_indices(self) -> None:
        """Populate qpos/qvel/ctrl index lookups from joint/actuator names."""
        qpos_idx = np.empty(6, dtype=int)
        qvel_idx = np.empty(6, dtype=int)
        ctrl_idx = np.empty(6, dtype=int)

        for i, jname in enumerate(JOINT_ORDER):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(
                    f"Joint '{jname}' not found in model. Did the XML rename?"
                )
            qpos_idx[i] = self.model.jnt_qposadr[jid]
            qvel_idx[i] = self.model.jnt_dofadr[jid]

        for i, aname in enumerate(ACTUATOR_ORDER):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(
                    f"Actuator '{aname}' not found in model."
                )
            ctrl_idx[i] = aid

        self._qpos_idx = qpos_idx
        self._qvel_idx = qvel_idx
        self._ctrl_idx = ctrl_idx

    # ----- queries -------------------------------------------------------
    def current_q(self, data: mujoco.MjData) -> np.ndarray:
        """Return the current 6-vector of joint angles, in :data:`JOINT_ORDER`."""
        return data.qpos[self._qpos_idx].copy()

    def current_qdot(self, data: mujoco.MjData) -> np.ndarray:
        """Return the current 6-vector of joint velocities."""
        return data.qvel[self._qvel_idx].copy()

    def bias(self, data: mujoco.MjData) -> np.ndarray:
        """
        Return the gravity + Coriolis bias forces projected onto our 6 DOFs.

        This is the analog of ``data.qfrc_bias`` from HW4 / Mock Q12. In
        this model gravity is zero so it's near-zero, but we keep the
        machinery in place for future variants that might turn gravity on.
        """
        return data.qfrc_bias[self._qvel_idx].copy()

    # ----- the actual control --------------------------------------------
    def step(self, data: mujoco.MjData, setpoint: GimbalSetpoint) -> np.ndarray:
        """
        Compute the PD control, write it to ``data.ctrl``, and return
        the 6-vector of commanded torques (post-clip).

        Side effect: modifies ``data.ctrl`` in place at the actuator
        indices in :data:`ACTUATOR_ORDER`. Other entries of
        ``data.ctrl`` are left untouched.
        """
        q_des = setpoint.as_vector()
        q     = self.current_q(data)
        qdot  = self.current_qdot(data)
        bias  = self.bias(data)

        # PD + feedback linearization (mirrors HW4 / Mock Q12)
        u = self.kp * (q_des - q) - self.kd * qdot + bias
        u = np.clip(u, -self.ctrl_clip, +self.ctrl_clip)

        # Write back into MuJoCo's control buffer
        data.ctrl[self._ctrl_idx] = u
        return u

    # ----- diagnostic ----------------------------------------------------
    def error(
        self,
        data: mujoco.MjData,
        setpoint: GimbalSetpoint,
    ) -> np.ndarray:
        """Return ``q* − q`` for the 6 joints (handy for logging)."""
        return setpoint.as_vector() - self.current_q(data)
