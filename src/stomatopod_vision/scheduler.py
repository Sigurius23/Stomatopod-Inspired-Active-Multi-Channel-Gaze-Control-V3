"""
scheduler.py — active-sensing controller (Layer 3)
==================================================

The high-level "brain": given the sparse :class:`PreprocessedEvent`
stream from Layer 2, decide where each eye should look next.

Three scheduler types are provided:

    - :class:`FixedForwardScheduler`
        Used for baselines B1 and B2. Both eyes are clamped to look
        straight forward; no scanning. The active-sensing controller
        is effectively disabled.

    - :class:`SaliencyScheduler` (mandatory implementation)
        Hand-designed. Maintains a saliency map and a visit-history
        memory. Picks the next look direction by scoring candidate
        directions on a weighted sum of (novelty, salience,
        feasibility, polarization-info-gain).

    - :class:`LearnedScheduler` (optional bonus)
        Replaces the weighted-sum scoring with a small neural network
        trained on logged trajectories (or via Bellman-style value
        iteration). Connects to Lecture 6 as a conceptual analog.

    - :class:`HopfScanScheduler` (optional bonus)
        A dynamical-systems controller: each eye is driven by a Hopf
        limit-cycle oscillator that rhythmically sweeps the narrow midband
        (scan), and a genuine Hopf bifurcation switches it to a fixed-point
        attractor to fixate a detected target (dwell), then back. Smooth
        continuous control instead of discrete replanning.

All schedulers expose the same :class:`BaseScheduler` interface so they can
be swapped freely in the experiment scripts.

Coordinate conventions
----------------------
We work entirely in **gimbal joint space** (yaw, pitch, roll, in radians).
The scheduler does NOT know world coordinates; it remembers each event
by the (yaw, pitch) that the producing eye held *when the event was
emitted*, plus the per-eye corrections needed to centre that target.

Sign-convention reminder (matches ``models/stomatopod_eyes.xml``):
  - Positive ``yaw_L`` turns eye_L to the head's LEFT  (world -X).
  - Positive ``yaw_R`` turns eye_R to the head's RIGHT (world +X).
  - Azimuth in :class:`~preprocessing.PreprocessedEvent` is positive
    when the target sits to the *eye's right*. To centre the target in
    that eye's view, the scheduler must add the azimuth to the eye's
    current yaw (for eye_L) or subtract it (for eye_R) — see
    :func:`_centring_setpoint`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from ._mlp import TinyMLP
from .gimbal_control import GimbalSetpoint
from .preprocessing import PreprocessedEvent

# Default gimbal joint limits, matching ``models/stomatopod_eyes.xml``.
# These are *fallbacks* used when a SaliencyScheduler is constructed
# without a MuJoCo model in hand (e.g. for pickling the LearnedScheduler,
# or for fast pure-NumPy unit tests). When you have a model, prefer
# :meth:`SaliencyScheduler.from_mujoco_model`, which reads the limits
# straight out of ``model.jnt_range`` so the scheduler can never drift
# out of sync with the XML.
#
# ``test_scheduler.test_limits_match_xml`` asserts that these defaults
# still match the XML; CI fails the moment the XML is edited without
# updating the defaults here too.
_YAW_LIMIT_RAD = 1.57
_PITCH_LIMIT_RAD = 1.05
_ROLL_LIMIT_RAD = 3.14

# Joint names in the MuJoCo XML. Used by ``from_mujoco_model`` to look
# up the per-DOF limits. Kept here so other modules (e.g. tests) can
# import the same canonical names.
EYE_JOINT_NAMES: dict[str, tuple[str, str, str]] = {
    "L": ("eye_L_yaw", "eye_L_pitch", "eye_L_roll"),
    "R": ("eye_R_yaw", "eye_R_pitch", "eye_R_roll"),
}


# =====================================================================
# Common abstractions
# =====================================================================

@dataclass
class SchedulerMemory:
    """
    Memory of what has been seen recently, used by all schedulers.

    Attributes
    ----------
    last_seen_time :
        ``{target_name: simulation_time_seconds}`` — when each target was
        most recently observed.
    last_decoded :
        ``{target_name: PreprocessedEvent}`` — the most recent
        compressed reading per target (used to score salience &
        polarization info gain).
    last_visit_direction :
        For each eye (``"L"``/``"R"``), the (yaw, pitch) the eye most
        recently pointed at when this scheduler last produced a setpoint.
    """
    last_seen_time: dict[str, float] = field(default_factory=dict)
    last_decoded: dict[str, PreprocessedEvent] = field(default_factory=dict)
    last_visit_direction: dict[str, tuple[float, float]] = field(default_factory=dict)

    def update(
        self,
        events: Iterable[PreprocessedEvent],
        time_now: float,
    ) -> None:
        """Fold a batch of new events into the memory."""
        for ev in events:
            self.last_seen_time[ev.target_name] = float(time_now)
            self.last_decoded[ev.target_name] = ev

    def time_since_seen(self, target_name: str, time_now: float) -> float:
        """Return seconds since ``target_name`` was last observed, or +inf."""
        if target_name not in self.last_seen_time:
            return float("inf")
        return max(0.0, float(time_now) - self.last_seen_time[target_name])

    def reset(self) -> None:
        """Forget everything (call at the start of each experiment)."""
        self.last_seen_time.clear()
        self.last_decoded.clear()
        self.last_visit_direction.clear()


# ---------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------

class BaseScheduler(ABC):
    """
    Interface all schedulers must implement.

    The experiment loop calls :meth:`update_memory` first (to fold new
    events into state) and then :meth:`next_setpoint` (to get the next
    gimbal target). Both calls happen every simulation step, although
    individual schedulers may choose to update their setpoint less often.
    """

    def __init__(self) -> None:
        self.memory = SchedulerMemory()

    @abstractmethod
    def update_memory(
        self,
        events: Iterable[PreprocessedEvent],
        time_now: float,
    ) -> None:
        """Fold new events into internal state."""

    @abstractmethod
    def next_setpoint(
        self,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> GimbalSetpoint:
        """Return the desired gimbal setpoint at ``time_now``."""

    def reset(self) -> None:
        """Reset all internal state (memory, RNG, etc.)."""
        self.memory.reset()


# =====================================================================
# Baseline schedulers — used in B1 and B2
# =====================================================================

class FixedForwardScheduler(BaseScheduler):
    """
    Both eyes are clamped pointing straight forward. The "active sensing"
    layer is effectively disabled, so this scheduler is used for the
    passive baselines B1 (no preprocessing) and B2 (with preprocessing).

    The difference between B1 and B2 is in the *experiment script*, not
    in the scheduler — both use this object.
    """

    def update_memory(
        self,
        events: Iterable[PreprocessedEvent],
        time_now: float,
    ) -> None:
        """Memory is updated for fairness (so metrics are still computed)."""
        self.memory.update(events, time_now)

    def next_setpoint(
        self,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> GimbalSetpoint:
        """Always returns the all-zero setpoint (eyes look forward)."""
        return GimbalSetpoint()


# =====================================================================
# Hand-designed saliency scheduler — B3 (mandatory)
# =====================================================================

@dataclass
class ScoringWeights:
    r"""Weights for the linear combination in :class:`SaliencyScheduler`.

    The four scoring terms are documented in
    :meth:`SaliencyScheduler.next_setpoint`. Defaults below reproduce
    the lecture-intuition weights: exploration (novelty), exploitation
    (salience), motion smoothing (feasibility) and polarization roll
    diversity (polarization_info_gain).

    Tuning finding
    --------------
    An empirical grid sweep (see ``src/experiments/tune_b3.py``) at
    5 seeds × 81 weight cells on the hard scene
    (``models/stomatopod_eyes_hard.xml``) at the sub-saturation
    operating point ``duration=0.5 s`` shows:

    ===============================  ================  =================
    Weight set                       Cov.\ @T=0.5s     BW @T=10s (B/s)
    ===============================  ================  =================
    Pure exploration (1, 2, 0, 0)    0.70 ± 0.18       19 201 ± 724
    Hand-designed  (1, 2, 0.5, 1)    0.76 ± 0.19       11 370 ± 828
    ===============================  ================  =================

    Both weight sets saturate coverage at 1.000 on the canonical 10 s
    horizon, but the hand-designed defaults reach 1.000 sooner AND
    sustain ~41% less bandwidth on the hard scene by damping the
    high-frequency whipping between the 10 wide-azimuth targets that
    otherwise trips the event encoder. On the default and moving
    scenes the bandwidth savings are ~20% and ~20% respectively.

    Because both weight sets are equal on coverage but hand-designed
    is Pareto-better on bandwidth at the reporting horizon, the ship
    defaults follow the hand design. Pure-exploration (0, 0) weights
    can still be passed explicitly via
    ``--w-feasibility 0 --w-pol 0`` on the CLI.
    """
    novelty: float = 1.0
    salience: float = 2.0
    feasibility: float = 0.5
    polarization_info_gain: float = 1.0


# Per-eye setpoint sign for converting a sighted azimuth into a centring yaw.
# yaw_setpoint_to_centre = - eye_sign[eye] * azimuth_seen
# (i.e. for eye_L, sign is -1, so positive azimuth gives positive yaw_L,
#  turning the left eye to the head's left — toward the target.)
_EYE_AZIMUTH_SIGN: dict[str, int] = {"L": -1, "R": +1}


def _centring_setpoint(
    eye: Literal["L", "R"],
    event: PreprocessedEvent,
) -> tuple[float, float]:
    """
    Convert an event's local (azimuth, elevation) — measured relative to
    the eye's pointing direction *at the time of the event* — back into
    the joint-space (yaw, pitch) that would centre that target now.

    Important assumption: the event's azimuth/elevation were measured
    while the eye was at some (yaw_then, pitch_then). To centre the
    target now we'd need yaw_now = yaw_then + something. Without knowing
    yaw_then we approximate by assuming the scheduler's *last commanded*
    yaw/pitch is close to yaw_then — true when the gimbal has settled.

    For simplicity we return the *delta* to apply to the eye's current
    setpoint; the caller adds it to whatever yaw_then was. Since the
    memory only stores the latest event per target, this works well
    enough in practice. See :meth:`SaliencyScheduler.score_salience`.
    """
    # Hardware sign convention
    eye_sign = _EYE_AZIMUTH_SIGN[eye]
    # Negative pitch in the gimbal sense ≈ positive elevation in the
    # eye-local sense (see test_sensor.py — "Pitch sign: positive pitch
    # tilts the eye DOWN about Y; negative pitch tilts it up.")
    delta_yaw = -eye_sign * event.azimuth
    delta_pitch = -event.elevation
    return delta_yaw, delta_pitch


class SaliencyScheduler(BaseScheduler):
    """
    Hand-designed active-sensing controller.

    At each scheduling decision (every ``decision_period_s`` seconds), the
    scheduler:

      1. Samples ``n_candidates`` candidate directions per eye
         in (yaw, pitch, roll).
      2. Scores each candidate by a weighted sum of:
            - **novelty** : how far the candidate is from the eye's most
              recent visit direction (encourages exploration)
            - **salience** : how close the candidate is to a known
              interesting target (encourages exploitation)
            - **feasibility** : negative gimbal-motion cost from the
              current setpoint (penalises whipping the eye too far)
            - **polarization_info_gain** : reward for rolling the eye to
              an under-sampled roll, motivated by Daly et al. 2018
      3. Picks the maximum-scoring candidate per eye as the new setpoint.

    Between decisions, the previously chosen setpoint is held.

    Parameters
    ----------
    n_candidates :
        How many candidate directions to sample per eye per decision.
    decision_period_s :
        Re-plan only every this many seconds; in between, hold the last
        setpoint. Smaller = more reactive, larger = smoother.
    weights :
        Mixing weights for the scoring function.
    seed :
        RNG seed for candidate sampling.

    # CITE: The action-selection structure (score candidates → argmax)
    #       is the standard active-sensing template. The biological
    #       motivation for rewarding roll comes from Daly et al. 2018
    #       (Proc. R. Soc. B), which describes mantis-shrimp eye
    #       rotation as functionally important for polarization access.
    """

    #: When a target's recent event has polarization_angle=None, that's
    #: a signal the polarization could not be confidently decoded — usually
    #: because the eye's current roll happens to give a degenerate set of
    #: receptor responses. Rolling to a different angle resolves this.
    #: This constant scales the per-target "should roll more" reward.
    POL_AMBIGUITY_BONUS: float = 1.0

    def __init__(
        self,
        n_candidates: int = 30,
        decision_period_s: float = 0.10,
        weights: ScoringWeights = ScoringWeights(),
        seed: int = 0,
        *,
        joint_limits: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.n_candidates = int(n_candidates)
        self.decision_period_s = float(decision_period_s)
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self._last_decision_time: float = -np.inf
        self._held_setpoint: GimbalSetpoint = GimbalSetpoint()
        # Per-instance joint limits: defaults to the module-level constants
        # so existing call sites keep working unchanged, but can be
        # overridden via the keyword argument or constructed straight from
        # a MuJoCo model with :meth:`from_mujoco_model`.
        if joint_limits is None:
            joint_limits = (_YAW_LIMIT_RAD, _PITCH_LIMIT_RAD, _ROLL_LIMIT_RAD)
        y, p_, r = joint_limits
        if y <= 0 or p_ <= 0 or r <= 0:
            raise ValueError(
                f"joint_limits must be strictly positive, got {joint_limits!r}"
            )
        self._yaw_limit: float = float(y)
        self._pitch_limit: float = float(p_)
        self._roll_limit: float = float(r)
        # Roll-visit history per eye: each entry is the roll angle the
        # scheduler last commanded for that eye. Used by score_pol_info_gain.
        self._roll_history: dict[str, list[float]] = {"L": [], "R": []}

    @classmethod
    def from_mujoco_model(
        cls,
        model: Any,  # mujoco.MjModel; Any so we don't pull in the import at module level
        *,
        eye: str = "L",
        **kwargs: Any,
    ) -> "SaliencyScheduler":
        """
        Construct a scheduler whose joint limits are read straight out of
        a MuJoCo model.

        This is the recommended factory whenever you already have a
        loaded :class:`mujoco.MjModel`: it removes the only point of
        drift between ``scheduler.py`` and the XML, so changing the
        ``<joint range="...">`` attributes propagates automatically.

        We read the limits off the ``eye`` ("L" or "R") side; both eyes
        use the same range in the canonical XMLs, so the choice is
        cosmetic. The joint names looked up are
        ``eye_{eye}_yaw / _pitch / _roll`` — see
        :data:`EYE_JOINT_NAMES`.

        Falls back to a clear error if any of the three joints are
        missing from the model, rather than silently using the defaults.

        Parameters
        ----------
        model :
            A loaded :class:`mujoco.MjModel`.
        eye :
            Which eye's joints to read the limits from. Default ``"L"``.
        **kwargs :
            Forwarded verbatim to ``__init__`` (``n_candidates``,
            ``decision_period_s``, ``weights``, ``seed``).
        """
        # Local import so the module itself stays pure-NumPy.
        import mujoco  # noqa: WPS433
        if eye not in EYE_JOINT_NAMES:
            raise ValueError(f"eye must be 'L' or 'R', got {eye!r}")
        yaw_name, pitch_name, roll_name = EYE_JOINT_NAMES[eye]
        limits: list[float] = []
        for joint_name in (yaw_name, pitch_name, roll_name):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
            )
            if jid < 0:
                raise ValueError(
                    f"Joint {joint_name!r} not found in the MuJoCo model; "
                    f"is this the right XML?"
                )
            lo, hi = model.jnt_range[jid]
            # The canonical XMLs use symmetric ranges (e.g. -1.57 .. 1.57);
            # take the symmetric magnitude. If the XML ever ships
            # asymmetric ranges, raise so the discrepancy is visible.
            if abs(lo + hi) > 1e-6:
                raise ValueError(
                    f"Joint {joint_name!r} has asymmetric range "
                    f"({lo}, {hi}); scheduler currently assumes symmetric."
                )
            limits.append(float(abs(hi)))
        return cls(joint_limits=(limits[0], limits[1], limits[2]), **kwargs)

    # ---------- scoring components (each returns a scalar) ----------------
    def score_novelty(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
        time_now: float,
    ) -> float:
        """
        Higher = this direction is far from where the eye last looked.

        Distance is measured in joint-angle space (radians). The score
        saturates at 1.0 for any direction at least 0.5 rad away from
        the last visit. New eyes (no visit history) score 1.0 by
        convention.
        """
        last = self.memory.last_visit_direction.get(eye)
        if last is None:
            return 1.0
        d = float(np.hypot(candidate_yaw - last[0], candidate_pitch - last[1]))
        return float(np.clip(d / 0.5, 0.0, 1.0))

    def score_salience(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
    ) -> float:
        """
        Higher = a known target lies near this candidate direction.

        We loop through known targets in :attr:`memory.last_decoded`,
        convert each into a centring (yaw, pitch) for this eye, and
        compute a Gaussian-falloff bonus based on angular distance to
        the candidate. Returns the *maximum* over all known targets.
        """
        if not self.memory.last_decoded:
            return 0.0

        best = 0.0
        for _name, ev in self.memory.last_decoded.items():
            d_yaw, d_pitch = _centring_setpoint(eye, ev)
            # Distance in joint space between the candidate and the
            # centring point. Width 0.3 rad (~17°) ≈ half the FoV
            # diagonal — a candidate well inside that radius scores ~1.
            dist = float(np.hypot(candidate_yaw - d_yaw,
                                  candidate_pitch - d_pitch))
            bonus = float(np.exp(-(dist / 0.3) ** 2))
            if bonus > best:
                best = bonus
        return best

    def score_feasibility(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
        candidate_roll: float,
        current_setpoint: GimbalSetpoint,
    ) -> float:
        """
        Higher = the gimbal does not have to move far from where it is.

        Penalises huge swings to prevent the eye whipping back and
        forth. Saturates at ``0`` for moves of more than ~π/2 in any
        axis (the gimbal still goes there if the other scores justify
        it; this is a smoothing term, not a hard limit).
        """
        cur_yaw = current_setpoint.yaw_L if eye == "L" else current_setpoint.yaw_R
        cur_pitch = current_setpoint.pitch_L if eye == "L" else current_setpoint.pitch_R
        cur_roll = current_setpoint.roll_L if eye == "L" else current_setpoint.roll_R

        # Roll is on a circle, so use the wrapped distance
        droll = abs((candidate_roll - cur_roll + np.pi) % (2 * np.pi) - np.pi)
        d = float(np.sqrt(
            (candidate_yaw - cur_yaw) ** 2
            + (candidate_pitch - cur_pitch) ** 2
            + droll ** 2
        ))
        return float(np.clip(1.0 - d / (np.pi / 2), 0.0, 1.0))

    def score_polarization_info_gain(
        self,
        eye: Literal["L", "R"],
        candidate_roll: float,
    ) -> float:
        """
        Higher = this roll would resolve a known-ambiguous polarization.

        Two contributions:
          - **Novelty in roll**: if the eye has visited many roll values
            already, a candidate close to a previously-visited roll
            scores lower. (Encourages roll diversity.)
          - **Ambiguous-target bonus**: if any target's most recent event
            had ``polarization_angle=None`` despite being polarized in
            reality, rolling at all is worthwhile.
        """
        history = self._roll_history[eye]

        # Roll-novelty: max distance to any previously visited roll, on
        # the circle, normalised to ∈ [0, 1].
        if not history:
            roll_novelty = 1.0
        else:
            wrapped = np.array([
                abs((candidate_roll - r + np.pi) % (2 * np.pi) - np.pi)
                for r in history
            ])
            # Distance to nearest visited roll, scaled by π/2
            roll_novelty = float(np.clip(np.min(wrapped) / (np.pi / 2),
                                         0.0, 1.0))

        # Ambiguity bonus: count how many known targets currently have
        # polarization_angle=None (i.e. the previous decoder failed).
        n_ambiguous = sum(
            1 for ev in self.memory.last_decoded.values()
            if ev.polarization_angle is None
        )
        ambiguity_bonus = self.POL_AMBIGUITY_BONUS * float(n_ambiguous > 0)

        return roll_novelty + ambiguity_bonus

    #: Names of features in the order produced by :meth:`feature_vector`.
    #: Kept as a class attribute so LearnedScheduler can introspect it for
    #: debugging / save-file headers.
    FEATURE_NAMES: tuple[str, ...] = (
        "score_novelty", "score_salience",
        "score_feasibility", "score_pol_info_gain",
        "candidate_yaw", "candidate_pitch", "candidate_roll",
        "dist_to_current_yaw", "dist_to_current_pitch",
        "log1p_n_known_targets", "max_time_since_seen_norm",
        "abs_candidate_yaw_over_limit",
    )

    def feature_vector(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
        candidate_roll: float,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> np.ndarray:
        """Return the 12-D feature vector :class:`LearnedScheduler` uses.

        Order matches :data:`FEATURE_NAMES`. The first four entries are
        the same scoring components the hand-designed scheduler uses;
        the rest are raw geometric / memory-derived features that give
        the network something to learn nonlinear combinations of.
        """
        cur_yaw = current_setpoint.yaw_L if eye == "L" else current_setpoint.yaw_R
        cur_pitch = current_setpoint.pitch_L if eye == "L" else current_setpoint.pitch_R

        # Per-eye / per-memory aggregate features
        n_known = len(self.memory.last_decoded)
        log1p_n = float(np.log1p(n_known))
        if n_known == 0:
            max_unseen = 0.0
        else:
            ages = [self.memory.time_since_seen(name, time_now)
                    for name in self.memory.last_decoded]
            # Cap at 5 s so the feature stays in a sensible range
            max_unseen = float(min(max(ages), 5.0)) / 5.0

        return np.array([
            self.score_novelty(eye, candidate_yaw, candidate_pitch, time_now),
            self.score_salience(eye, candidate_yaw, candidate_pitch),
            self.score_feasibility(eye, candidate_yaw, candidate_pitch,
                                   candidate_roll, current_setpoint),
            self.score_polarization_info_gain(eye, candidate_roll),
            float(candidate_yaw),
            float(candidate_pitch),
            float(candidate_roll),
            float(candidate_yaw - cur_yaw),
            float(candidate_pitch - cur_pitch),
            log1p_n,
            max_unseen,
            float(abs(candidate_yaw) / self._yaw_limit),
        ], dtype=np.float64)

    def total_score(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
        candidate_roll: float,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> float:
        """Weighted sum of all four scoring components."""
        w = self.weights
        return (
            w.novelty * self.score_novelty(
                eye, candidate_yaw, candidate_pitch, time_now)
            + w.salience * self.score_salience(
                eye, candidate_yaw, candidate_pitch)
            + w.feasibility * self.score_feasibility(
                eye, candidate_yaw, candidate_pitch, candidate_roll,
                current_setpoint)
            + w.polarization_info_gain * self.score_polarization_info_gain(
                eye, candidate_roll)
        )

    # ---------- candidate sampling ---------------------------------------
    def sample_candidates(
        self,
        eye: Literal["L", "R"],
    ) -> np.ndarray:
        """
        Return ``(n_candidates, 3)`` array of (yaw, pitch, roll) tuples
        uniformly sampled within the gimbal's joint limits.

        We also include the "centring" yaw/pitch for every known target
        as forced candidates (so the scheduler will always at least
        consider looking at known interesting points). The first
        ``len(memory.last_decoded)`` rows are these forced candidates,
        the rest are random.
        """
        forced: list[tuple[float, float, float]] = []
        for ev in self.memory.last_decoded.values():
            d_yaw, d_pitch = _centring_setpoint(eye, ev)
            # Clip to joint limits
            d_yaw = float(np.clip(d_yaw, -self._yaw_limit, self._yaw_limit))
            d_pitch = float(np.clip(d_pitch, -self._pitch_limit, self._pitch_limit))
            # Roll: sample one for each forced candidate
            d_roll = float(self.rng.uniform(-self._roll_limit, self._roll_limit))
            forced.append((d_yaw, d_pitch, d_roll))

        n_random = max(0, self.n_candidates - len(forced))
        random = np.column_stack([
            self.rng.uniform(-self._yaw_limit,   self._yaw_limit,   size=n_random),
            self.rng.uniform(-self._pitch_limit, self._pitch_limit, size=n_random),
            self.rng.uniform(-self._roll_limit,  self._roll_limit,  size=n_random),
        ])

        if forced:
            return np.vstack([np.asarray(forced), random])
        return random

    # ---------- BaseScheduler interface ----------------------------------
    def update_memory(
        self,
        events: Iterable[PreprocessedEvent],
        time_now: float,
    ) -> None:
        """Fold new events into the underlying SchedulerMemory."""
        self.memory.update(events, time_now)

    def next_setpoint(
        self,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> GimbalSetpoint:
        """
        Re-plan if ``time_now - _last_decision_time >= decision_period_s``,
        otherwise return the held setpoint.

        On a re-plan, sample candidates for each eye independently and
        pick the highest-scoring one. Update the held setpoint and the
        roll/visit history.
        """
        elapsed = float(time_now) - self._last_decision_time
        if elapsed < self.decision_period_s:
            return self._held_setpoint

        # Re-plan
        chosen: dict[str, tuple[float, float, float]] = {}
        for eye in ("L", "R"):
            cands = self.sample_candidates(eye)
            scores = np.array([
                self.total_score(eye, y, p, r, time_now, current_setpoint)
                for y, p, r in cands
            ])
            best_idx = int(np.argmax(scores))
            chosen[eye] = tuple(cands[best_idx])

        new_setpoint = GimbalSetpoint(
            yaw_L=chosen["L"][0], pitch_L=chosen["L"][1], roll_L=chosen["L"][2],
            yaw_R=chosen["R"][0], pitch_R=chosen["R"][1], roll_R=chosen["R"][2],
        )

        # Update history
        for eye in ("L", "R"):
            self.memory.last_visit_direction[eye] = (chosen[eye][0], chosen[eye][1])
            self._roll_history[eye].append(chosen[eye][2])
            # Cap roll history to avoid unbounded memory growth
            if len(self._roll_history[eye]) > 50:
                self._roll_history[eye] = self._roll_history[eye][-50:]

        self._held_setpoint = new_setpoint
        self._last_decision_time = float(time_now)
        return new_setpoint

    def reset(self) -> None:
        """Reset memory, RNG-derived held state, decision clock."""
        super().reset()
        self._last_decision_time = -np.inf
        self._held_setpoint = GimbalSetpoint()
        self._roll_history = {"L": [], "R": []}


# =====================================================================
# Optional bonus: learned scheduler
# =====================================================================

class LearnedScheduler(SaliencyScheduler):
    """
    Bonus: a learned drop-in replacement for the hand-tuned scoring sum.

    Architecture
    ------------
    Inherits from :class:`SaliencyScheduler` to reuse:

      - The candidate sampler (random + forced centring candidates).
      - The memory-update loop.
      - The decision-period throttling and held setpoint.
      - The feature-vector definition (:meth:`SaliencyScheduler.feature_vector`).

    The only thing that changes is :meth:`total_score`: instead of a
    fixed weighted sum, the score is the output of a tiny pure-NumPy
    MLP (:class:`stomatopod_vision._mlp.TinyMLP`) that takes the 12-D
    feature vector and returns a scalar.

    Training
    --------
    The simplest pedagogically clean training task is **imitation
    learning** of the hand-designed scoring function:

      1. Construct a "teacher" :class:`SaliencyScheduler` with the
         current default :class:`ScoringWeights`.
      2. Sample many candidates across many (memory state, time)
         contexts by stepping a real simulation under the teacher.
      3. At every candidate, record ``(feature_vector, teacher_score)``.
      4. Fit the MLP to minimise MSE against the teacher's score.

    The "value-based action selection" framing from Lec 6 then says:
    the trained MLP is a learned approximation of an information-gain
    *value function* over candidate gimbal directions. In practice a
    well-fit MLP picks the same argmax as the teacher on almost every
    step, and the LearnedScheduler ends up nearly indistinguishable
    from the SaliencyScheduler on the benchmark scenes — which is the
    point: it demonstrates that the hand-tuned scoring can be
    *replaced* by a learned function, not that learning beats it.

    Training-data collection lives in
    :func:`src.experiments.train_learned`; this class just provides the
    inference path plus a :meth:`fit` method that wraps the MLP.

    Parameters
    ----------
    mlp :
        A :class:`TinyMLP` instance with ``n_in == len(FEATURE_NAMES)``.
        If ``None``, a freshly-initialised MLP is created (and must be
        trained before the scheduler will produce useful decisions —
        until then the scores are random Xavier outputs).
    Other parameters are forwarded to :class:`SaliencyScheduler`.
    """

    def __init__(
        self,
        mlp: TinyMLP | None = None,
        *,
        n_candidates: int = 30,
        decision_period_s: float = 0.10,
        seed: int = 0,
        joint_limits: tuple[float, float, float] | None = None,
    ) -> None:
        # The weights field is unused at inference time but kept for
        # base-class compatibility (and so the inherited feature_vector
        # call to score_* methods continues to work).
        super().__init__(
            n_candidates=n_candidates,
            decision_period_s=decision_period_s,
            weights=ScoringWeights(),
            seed=seed,
            joint_limits=joint_limits,
        )
        if mlp is None:
            mlp = TinyMLP(n_in=len(self.FEATURE_NAMES), n_hidden=16,
                          n_out=1, seed=seed)
        assert mlp.n_in == len(self.FEATURE_NAMES), \
            f"MLP expects {mlp.n_in} features but the scheduler produces " \
            f"{len(self.FEATURE_NAMES)}"
        self.mlp = mlp

    # ------------------------------------------------------------------
    # Inference: override total_score to use the learned MLP
    # ------------------------------------------------------------------

    def total_score(
        self,
        eye: Literal["L", "R"],
        candidate_yaw: float,
        candidate_pitch: float,
        candidate_roll: float,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> float:
        feats = self.feature_vector(
            eye, candidate_yaw, candidate_pitch, candidate_roll,
            time_now, current_setpoint,
        )
        return float(self.mlp.forward(feats).ravel()[0])

    # ------------------------------------------------------------------
    # Training-time helpers
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 300,
        lr: float = 1e-2,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """Train the MLP on a (features → teacher_score) dataset.

        Returns the per-epoch mean MSE loss history. See
        :class:`stomatopod_vision._mlp.TinyMLP.fit` for parameter
        details.
        """
        return self.mlp.fit(
            X, y, epochs=epochs, batch_size=batch_size,
            lr=lr, seed=int(self.rng.integers(0, 2**31 - 1)), verbose=verbose,
        )

    def save(self, path: str | Path) -> None:
        """Save the trained MLP to disk (NumPy .npz)."""
        self.mlp.save(path)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        n_candidates: int = 30,
        decision_period_s: float = 0.10,
        seed: int = 0,
    ) -> "LearnedScheduler":
        """Construct a :class:`LearnedScheduler` from a saved MLP."""
        return cls(
            mlp=TinyMLP.load(path),
            n_candidates=n_candidates,
            decision_period_s=decision_period_s,
            seed=seed,
        )


# =====================================================================
# Bonus: dynamical (limit-cycle) scanning scheduler — "B3-Dynamical"
# =====================================================================

class HopfScanScheduler(BaseScheduler):
    r"""
    Active-sensing controller built on a **Hopf limit-cycle oscillator**
    per eye, instead of discrete score-and-argmax replanning.

    Motivation
    ----------
    A stomatopod's midband is a *narrow* strip of receptors, so the animal
    cannot image a scene in a single glance — it sweeps the eye rhythmically
    to drag the midband across visual space. Rhythmic motor patterns of this
    kind are classically modelled as limit-cycle oscillators / central
    pattern generators. We use the canonical supercritical **Hopf normal
    form** per eye (state ``(x, y)``):

    .. math::
        \dot x = (\mu - (x^2 + y^2))\,x - \omega\,y \\
        \dot y = (\mu - (x^2 + y^2))\,y + \omega\,x

    - ``mu > 0``: a stable limit cycle of radius ``sqrt(mu)`` — the eye
      **scans**, tracing a loop that sweeps the (narrow-in-pitch) midband up
      and down while a slow centre-drift rasters it across azimuth.
    - ``mu < 0``: the origin becomes a stable focus — the oscillator spirals
      to rest and the eye **fixates** the loop centre.

    The scan<->fixate transition is a genuine **Hopf bifurcation** of the
    control dynamics, not a hand-switched trajectory: on detecting a target
    the controller lowers ``mu`` below zero and re-centres the loop on that
    target (scan -> fixate); after a short dwell it raises ``mu`` back above
    zero (fixate -> scan). Torsion (roll) is driven by an independent slow
    oscillation so polarization is sampled across a range of roll angles
    throughout.

    This is a deliberately *different control philosophy* from
    :class:`SaliencyScheduler` (smooth continuous dynamics vs. discrete
    replanning), exposed through the same :class:`BaseScheduler` interface so
    the experiment harness can swap the two freely.

    # CITE: rhythmic midband scanning — Land et al. 1990 (J. Comp. Physiol. A);
    #       Marshall et al. 2014 (Annu. Rev. Mar. Sci.). Roll/torsion access to
    #       polarization — Daly et al. 2018 (Proc. R. Soc. B). The Hopf normal
    #       form as a CPG primitive is standard dynamical-systems control.
    """

    def __init__(
        self,
        *,
        mu_scan: float = 1.0,
        mu_fixate: float = -2.0,
        gain: float = 4.0,
        omega: float = 6.0,
        scan_amp_yaw: float = 0.15,
        scan_amp_pitch: float = 0.80,
        center_yaw_amp: float = 1.20,
        omega_center: float = 0.55,
        roll_amp: float = 0.80,
        omega_roll: float = 3.0,
        fixation_dwell_s: float = 0.40,
        refixate_cooldown_s: float = 1.5,
        integrate_dt_s: float = 0.005,
        seed: int = 0,
        joint_limits: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.mu_scan = float(mu_scan)
        self.mu_fixate = float(mu_fixate)
        self.gain = float(gain)
        self.omega = float(omega)
        self.scan_amp_yaw = float(scan_amp_yaw)
        self.scan_amp_pitch = float(scan_amp_pitch)
        self.center_yaw_amp = float(center_yaw_amp)
        self.omega_center = float(omega_center)
        self.roll_amp = float(roll_amp)
        self.omega_roll = float(omega_roll)
        self.fixation_dwell_s = float(fixation_dwell_s)
        self.refixate_cooldown_s = float(refixate_cooldown_s)
        self.integrate_dt_s = float(integrate_dt_s)
        if self.integrate_dt_s <= 0.0:
            raise ValueError("integrate_dt_s must be > 0")
        self.rng = np.random.default_rng(seed)

        if joint_limits is None:
            joint_limits = (_YAW_LIMIT_RAD, _PITCH_LIMIT_RAD, _ROLL_LIMIT_RAD)
        y, p_, r = joint_limits
        if y <= 0 or p_ <= 0 or r <= 0:
            raise ValueError(
                f"joint_limits must be strictly positive, got {joint_limits!r}")
        self._yaw_limit, self._pitch_limit, self._roll_limit = (
            float(y), float(p_), float(r))

        # Complementary scan phases so the two eyes cover different azimuths
        # at any instant, and offset roll phases for polarization diversity.
        self._center_phase = {"L": 0.0, "R": float(np.pi)}
        self._roll_phase = {"L": 0.0, "R": float(np.pi / 2)}
        self._init_dynamic_state()

    @classmethod
    def from_mujoco_model(cls, model: Any, *, eye: str = "L", **kwargs: Any
                          ) -> "HopfScanScheduler":
        """Construct with joint limits read straight from a MuJoCo model.

        Mirrors :meth:`SaliencyScheduler.from_mujoco_model` so the two
        schedulers are interchangeable in the experiment scripts.
        """
        import mujoco  # noqa: WPS433
        if eye not in EYE_JOINT_NAMES:
            raise ValueError(f"eye must be 'L' or 'R', got {eye!r}")
        limits: list[float] = []
        for joint_name in EYE_JOINT_NAMES[eye]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"Joint {joint_name!r} not found in model.")
            lo, hi = model.jnt_range[jid]
            limits.append(float(abs(hi)))
        return cls(joint_limits=(limits[0], limits[1], limits[2]), **kwargs)

    def _init_dynamic_state(self) -> None:
        # Start each oscillator ON the scan limit cycle (radius sqrt(mu_scan))
        # so there is no start-up transient.
        r0 = float(np.sqrt(max(self.mu_scan, 1e-6)))
        self._osc = {"L": np.array([r0, 0.0]), "R": np.array([r0, 0.0])}
        self._mode: dict[str, str] = {"L": "scan", "R": "scan"}
        self._fixate_target: dict[str, tuple[float, float]] = {
            "L": (0.0, 0.0), "R": (0.0, 0.0)}
        self._fixate_until: dict[str, float] = {"L": -np.inf, "R": -np.inf}
        self._last_gaze: dict[str, tuple[float, float]] = {
            "L": (0.0, 0.0), "R": (0.0, 0.0)}
        self._last_fixated_time: dict[str, float] = {}
        self._last_t: float | None = None
        # Held so this scheduler is a drop-in for the experiment harness'
        # ``next_setpoint(t, current_setpoint=sched._held_setpoint)`` pattern
        # (the value is not used internally — the oscillator is the state).
        self._held_setpoint: GimbalSetpoint = GimbalSetpoint()

    # ---------- dynamics --------------------------------------------------
    def _mu(self, eye: str, time_now: float) -> float:
        """Bifurcation parameter for this eye: >0 = scan, <0 = fixate.

        Also expires a finished fixation (dwell elapsed -> bifurcate back to
        scanning). This is the only place ``_mode`` flips fixate -> scan.
        """
        if self._mode[eye] == "fixate" and time_now < self._fixate_until[eye]:
            return self.mu_fixate
        self._mode[eye] = "scan"
        return self.mu_scan

    def _deriv(self, x: float, y: float, mu: float) -> tuple[float, float]:
        """Hopf normal-form vector field with convergence-rate ``gain``.

        ``gain`` sets how fast trajectories approach the attractor (limit
        cycle for mu>0, origin for mu<0); ``omega`` sets the angular speed.
        The limit-cycle radius is ``sqrt(mu)`` (== 1 for the default scan).
        """
        r2 = x * x + y * y
        g = self.gain
        w = self.omega
        dx = g * (mu - r2) * x - w * y
        dy = g * (mu - r2) * y + w * x
        return dx, dy

    def _integrate(self, eye: str, mu: float, dt: float) -> None:
        """Advance the Hopf oscillator by ``dt`` with sub-stepped RK4.

        RK4 (vs. forward Euler) removes the amplitude bias so the scan
        limit cycle sits cleanly at radius ``sqrt(mu_scan)``; sub-stepping
        keeps it accurate regardless of how often ``next_setpoint`` is polled.
        """
        if dt <= 0.0:
            return
        n = max(1, int(np.ceil(dt / self.integrate_dt_s)))
        h = dt / n
        x, y = float(self._osc[eye][0]), float(self._osc[eye][1])
        for _ in range(n):
            k1x, k1y = self._deriv(x, y, mu)
            k2x, k2y = self._deriv(x + 0.5 * h * k1x, y + 0.5 * h * k1y, mu)
            k3x, k3y = self._deriv(x + 0.5 * h * k2x, y + 0.5 * h * k2y, mu)
            k4x, k4y = self._deriv(x + h * k3x, y + h * k3y, mu)
            x += (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        self._osc[eye] = np.array([x, y])

    def _center(self, eye: str, time_now: float) -> tuple[float, float]:
        """Effective loop centre in (yaw, pitch)."""
        if self._mode[eye] == "fixate":
            return self._fixate_target[eye]
        # Slow azimuth raster; the fast loop covers elevation, so the pitch
        # centre stays at 0.
        yaw_c = self.center_yaw_amp * float(
            np.sin(self.omega_center * time_now + self._center_phase[eye]))
        return yaw_c, 0.0

    def limit_cycle_radius(self, eye: str) -> float:
        """Current oscillator amplitude ``sqrt(x^2 + y^2)`` (test/introspection)."""
        x, y = self._osc[eye]
        return float(np.hypot(x, y))

    # ---------- BaseScheduler interface ----------------------------------
    def update_memory(
        self,
        events: Iterable[PreprocessedEvent],
        time_now: float,
    ) -> None:
        """Fold events into memory and trigger scan->fixate bifurcations."""
        events = list(events)
        self.memory.update(events, time_now)
        for ev in events:
            eye = ev.eye
            if eye not in ("L", "R"):
                continue
            # Skip targets fixated too recently (avoid thrashing on one target).
            last = self._last_fixated_time.get(ev.target_name, -np.inf)
            if time_now - last < self.refixate_cooldown_s:
                continue
            # Absolute joint-space direction of this target for this eye,
            # approximated from the eye's last commanded gaze plus the event's
            # centring offset (same approximation as SaliencyScheduler).
            d_yaw, d_pitch = _centring_setpoint(eye, ev)
            gy, gp = self._last_gaze[eye]
            tgt_yaw = float(np.clip(gy + d_yaw,
                                    -self._yaw_limit, self._yaw_limit))
            tgt_pitch = float(np.clip(gp + d_pitch,
                                      -self._pitch_limit, self._pitch_limit))
            # Hopf bifurcation -> fixation on the detected target.
            self._fixate_target[eye] = (tgt_yaw, tgt_pitch)
            self._fixate_until[eye] = time_now + self.fixation_dwell_s
            self._mode[eye] = "fixate"
            self._last_fixated_time[ev.target_name] = time_now

    def next_setpoint(
        self,
        time_now: float,
        current_setpoint: GimbalSetpoint,
    ) -> GimbalSetpoint:
        """Integrate each eye's oscillator and map its state to a setpoint."""
        t = float(time_now)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        vals: dict[str, tuple[float, float, float]] = {}
        for eye in ("L", "R"):
            mu = self._mu(eye, t)
            self._integrate(eye, mu, dt)
            x, y = self._osc[eye]
            cy, cp = self._center(eye, t)
            yaw = float(np.clip(cy + self.scan_amp_yaw * x,
                                -self._yaw_limit, self._yaw_limit))
            pitch = float(np.clip(cp + self.scan_amp_pitch * y,
                                  -self._pitch_limit, self._pitch_limit))
            roll = float(np.clip(
                self.roll_amp * np.sin(self.omega_roll * t
                                       + self._roll_phase[eye]),
                -self._roll_limit, self._roll_limit))
            self._last_gaze[eye] = (yaw, pitch)
            self.memory.last_visit_direction[eye] = (yaw, pitch)
            vals[eye] = (yaw, pitch, roll)

        self._held_setpoint = GimbalSetpoint(
            yaw_L=vals["L"][0], pitch_L=vals["L"][1], roll_L=vals["L"][2],
            yaw_R=vals["R"][0], pitch_R=vals["R"][1], roll_R=vals["R"][2],
        )
        return self._held_setpoint

    def reset(self) -> None:
        """Reset memory and all oscillator / mode state."""
        super().reset()
        self._init_dynamic_state()
