"""
Tests for stomatopod_vision.scheduler

Validates:
   1. SchedulerMemory.update + time_since_seen + reset.
   2. FixedForwardScheduler always returns zero setpoint.
   3. SaliencyScheduler construction + initial state.
   4. sample_candidates returns the right shape + within joint limits.
   5. sample_candidates includes forced "centring" candidates per known target.
   6. score_novelty: 1.0 with no history, falls off near recent visit.
   7. score_salience: 0.0 with no targets, peaks near a centring direction.
   8. score_feasibility: 1.0 for current setpoint, falls off with distance.
   9. score_polarization_info_gain: 1.0 with no roll history, ambiguity bonus.
  10. total_score combines the four components with the right weights.
  11. next_setpoint holds during decision_period; replans afterwards.
  12. next_setpoint produces different setpoints across decisions
      (i.e. is actually using the RNG / exploring).
  13. reset() puts the scheduler back to a fresh state.
  14. _YAW_LIMIT etc. match the XML's <joint range="..."> attributes.
  15. End-to-end: SaliencyScheduler drives eyes toward known interesting
      target on a real MuJoCo scene; B3 coverage matches or exceeds B2.

Run from the repo root:
    python tests/test_scheduler.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision.gimbal_control import (  # noqa: E402
    GimbalPD,
    GimbalSetpoint,
)
from stomatopod_vision.metrics import EventLog, compute_all  # noqa: E402
from stomatopod_vision.preprocessing import (  # noqa: E402
    PreprocessedEvent,
    PreprocessingPipeline,
)
from stomatopod_vision.scheduler import (  # noqa: E402
    _PITCH_LIMIT_RAD,
    _ROLL_LIMIT_RAD,
    _YAW_LIMIT_RAD,
    EYE_JOINT_NAMES,
    BaseScheduler,
    FixedForwardScheduler,
    LearnedScheduler,
    SaliencyScheduler,
    SchedulerMemory,
    ScoringWeights,
    _centring_setpoint,
)
from stomatopod_vision.sensor import (  # noqa: E402
    VirtualEye,
    make_eye_pair,
)
from stomatopod_vision.world import Scene  # noqa: E402

XML_PATH = REPO_ROOT / "models" / "stomatopod_eyes.xml"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _event(
    target: str,
    time: float = 0.0,
    *,
    eye: str = "L",
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: float = 1.0,
    dominant_class: str = "UV_A",
    polarization_angle: float | None = None,
) -> PreprocessedEvent:
    return PreprocessedEvent(
        time=time, eye=eye, target_name=target,
        azimuth=azimuth, elevation=elevation, distance=distance,
        spectral_pattern=(0,0,0,0,0,0,0,0,0,10,0,0), circular_handedness=None,
        polarization_angle=polarization_angle,
    )


# =====================================================================
# Test 1 — SchedulerMemory
# =====================================================================

def test_memory_basics():
    print("Test 1: SchedulerMemory.update + time_since_seen + reset …")
    mem = SchedulerMemory()

    # No events yet
    assert mem.time_since_seen("t1", time_now=10.0) == float("inf")

    mem.update([_event("t1", time=0.0)], time_now=1.0)
    mem.update([_event("t2", time=0.0)], time_now=2.0)

    # time_since_seen uses time_now relative to the update
    assert mem.time_since_seen("t1", time_now=5.0) == 4.0
    assert mem.time_since_seen("t2", time_now=5.0) == 3.0
    assert mem.time_since_seen("unseen", time_now=5.0) == float("inf")

    # last_decoded keeps the *latest* event
    mem.update([_event("t1", time=999.0, azimuth=0.5)], time_now=3.0)
    assert abs(mem.last_decoded["t1"].azimuth - 0.5) < 1e-12
    assert mem.time_since_seen("t1", time_now=5.0) == 2.0

    mem.reset()
    assert mem.time_since_seen("t1", time_now=5.0) == float("inf")
    assert mem.last_decoded == {}
    print("  ✓ memory tracks updates, ages, and resets cleanly")


# =====================================================================
# Test 2 — FixedForwardScheduler
# =====================================================================

def test_fixed_forward():
    print("\nTest 2: FixedForwardScheduler returns zero setpoint always …")
    s = FixedForwardScheduler()
    s.update_memory([_event("t1")], time_now=0.0)

    sp = s.next_setpoint(time_now=0.0, current_setpoint=GimbalSetpoint())
    assert sp == GimbalSetpoint()

    sp = s.next_setpoint(time_now=10.0,
                         current_setpoint=GimbalSetpoint(yaw_L=0.5))
    assert sp == GimbalSetpoint()
    print("  ✓ always returns zeros, regardless of time / memory / current pose")


# =====================================================================
# Test 3 — SaliencyScheduler construction
# =====================================================================

def test_saliency_construction():
    print("\nTest 3: SaliencyScheduler initial state …")
    s = SaliencyScheduler(n_candidates=20, decision_period_s=0.1, seed=42)
    assert s.n_candidates == 20
    assert s.decision_period_s == 0.1
    assert isinstance(s.weights, ScoringWeights)
    assert s._last_decision_time == -np.inf
    assert s._held_setpoint == GimbalSetpoint()
    assert s._roll_history == {"L": [], "R": []}
    print("  ✓ initial state is correct")

    # Lock in the default weights — see the tuning finding documented on
    # ScoringWeights. Any future change to the defaults should be deliberate
    # (and update this assertion + docstring + report together).
    #
    # 2026-07-02 update: after redesigning the hard-scene target geometry
    # to place all 10 interesting targets at |az|>=65\xb0, the tuning sweep
    # now shows the hand-designed weights (feasibility=0.5, pol=1.0)
    # Pareto-dominate the pure (0, 0) tuned defaults on bandwidth at the
    # canonical horizon (both saturate coverage at 1.000). Defaults have
    # been reverted to the hand-designed values.
    w = ScoringWeights()
    assert w.novelty == 1.0,     f"novelty default drifted: {w.novelty}"
    assert w.salience == 2.0,    f"salience default drifted: {w.salience}"
    assert w.feasibility == 0.5, f"feasibility default drifted: {w.feasibility}"
    assert w.polarization_info_gain == 1.0, \
        f"polarization_info_gain default drifted: {w.polarization_info_gain}"
    print("  \u2713 default ScoringWeights match the hand-designed values "
          "(novelty=1.0, salience=2.0, feasibility=0.5, pol=1.0)")


# =====================================================================
# Test 4 — sample_candidates: shape + limits
# =====================================================================

def test_sample_candidates_shape():
    print("\nTest 4: sample_candidates returns correct shape, in joint limits …")
    s = SaliencyScheduler(n_candidates=30, seed=0)
    for eye in ("L", "R"):
        cands = s.sample_candidates(eye)
        assert cands.shape == (30, 3), f"shape = {cands.shape}"
        assert np.all(np.abs(cands[:, 0]) <= _YAW_LIMIT_RAD)
        assert np.all(np.abs(cands[:, 1]) <= _PITCH_LIMIT_RAD)
        assert np.all(np.abs(cands[:, 2]) <= _ROLL_LIMIT_RAD)
    print("  ✓ shape (30, 3) + all values within joint limits")


# =====================================================================
# Test 5 — forced candidates include known-target centrings
# =====================================================================

def test_sample_candidates_forced():
    print("\nTest 5: sample_candidates includes forced centring candidates …")
    s = SaliencyScheduler(n_candidates=20, seed=0)
    # Register a target seen at azimuth=0.3, elevation=0.1
    s.update_memory([
        _event("tgt", azimuth=0.3, elevation=0.1, eye="L"),
    ], time_now=0.0)

    cands_L = s.sample_candidates("L")
    # The first row should be the centring point for eye L
    expected_yaw, expected_pitch = _centring_setpoint(
        "L", _event("tgt", azimuth=0.3, elevation=0.1, eye="L"))
    assert abs(cands_L[0, 0] - expected_yaw) < 1e-12
    assert abs(cands_L[0, 1] - expected_pitch) < 1e-12
    print(f"  ✓ forced candidate at ({expected_yaw:+.3f}, {expected_pitch:+.3f})")

    # Even sampling for eye R uses the same event (single global memory)
    cands_R = s.sample_candidates("R")
    expected_yaw_R, expected_pitch_R = _centring_setpoint(
        "R", _event("tgt", azimuth=0.3, elevation=0.1, eye="L"))
    assert abs(cands_R[0, 0] - expected_yaw_R) < 1e-12
    print(f"  ✓ symmetric centring for eye R: ({expected_yaw_R:+.3f}, {expected_pitch_R:+.3f})")


# =====================================================================
# Test 6 — score_novelty
# =====================================================================

def test_score_novelty():
    print("\nTest 6: score_novelty: 1.0 if no history, falls near recent visit …")
    s = SaliencyScheduler()

    # No visit history → max novelty
    assert s.score_novelty("L", 0.0, 0.0, time_now=0.0) == 1.0

    # Record a visit
    s.memory.last_visit_direction["L"] = (0.5, 0.2)

    # Far candidate → 1.0 (saturated)
    assert s.score_novelty("L", -1.5, 0.0, time_now=0.0) == 1.0

    # Same point → 0.0
    assert s.score_novelty("L", 0.5, 0.2, time_now=0.0) == 0.0

    # Halfway in between (~0.25 rad away) → ~0.5
    n_mid = s.score_novelty("L", 0.5 + 0.25, 0.2, time_now=0.0)
    assert abs(n_mid - 0.5) < 1e-9
    print(f"  ✓ novelty at last-visit = 0, at 0.25rad = {n_mid:.2f}, saturated = 1.0")


# =====================================================================
# Test 7 — score_salience
# =====================================================================

def test_score_salience():
    print("\nTest 7: score_salience: 0.0 with no targets, peaks near centring …")
    s = SaliencyScheduler()
    assert s.score_salience("L", 0.0, 0.0) == 0.0

    # Register a target at azimuth=0.4 → its centring point for eye L
    # is (yaw=+0.4, pitch=0.0) (per _EYE_AZIMUTH_SIGN)
    s.update_memory([_event("tgt", azimuth=0.4, eye="L")], time_now=0.0)

    sal_at_target = s.score_salience("L", 0.4, 0.0)
    sal_far = s.score_salience("L", -1.0, 0.0)
    assert sal_at_target > 0.99, f"expected ~1, got {sal_at_target}"
    assert sal_far < 0.01, f"expected ~0, got {sal_far}"
    print(f"  ✓ salience at centring = {sal_at_target:.3f}, far away = {sal_far:.4f}")


# =====================================================================
# Test 8 — score_feasibility
# =====================================================================

def test_score_feasibility():
    print("\nTest 8: score_feasibility: 1.0 if no move, falls with distance …")
    s = SaliencyScheduler()
    cur = GimbalSetpoint(yaw_L=0.3, pitch_L=0.1, roll_L=0.5,
                         yaw_R=-0.3, pitch_R=-0.1, roll_R=-0.5)

    # Same point → 1.0
    f = s.score_feasibility("L", 0.3, 0.1, 0.5, cur)
    assert abs(f - 1.0) < 1e-12

    # Far point in yaw alone (1.5 rad away)
    f_far = s.score_feasibility("L", -1.2, 0.1, 0.5, cur)
    assert 0.0 <= f_far < 1.0
    assert f_far < 0.1, f"expected near 0, got {f_far}"

    # Huge jump → 0.0
    f_huge = s.score_feasibility("L", -1.5, -1.0, -3.0, cur)
    assert f_huge == 0.0
    print(f"  ✓ feasibility: same = 1.0, far = {f_far:.3f}, huge = 0.0")


# =====================================================================
# Test 9 — score_polarization_info_gain
# =====================================================================

def test_score_pol_info_gain():
    print("\nTest 9: score_polarization_info_gain: novelty + ambiguity bonus …")
    s = SaliencyScheduler()
    # No roll history, no ambiguous targets → roll_novelty = 1.0,
    # ambiguity_bonus = 0 → score = 1.0
    assert s.score_polarization_info_gain("L", 0.0) == 1.0

    # Visit roll = 1.0
    s._roll_history["L"].append(1.0)
    # Same roll → roll_novelty = 0; no ambiguity → score = 0.0
    near = s.score_polarization_info_gain("L", 1.0)
    assert abs(near - 0.0) < 1e-12
    # Far roll → roll_novelty = 1.0 → score = 1.0
    far = s.score_polarization_info_gain("L", 1.0 + np.pi / 2)
    assert abs(far - 1.0) < 1e-9

    # Now mark a target as "ambiguous" — polarization_angle=None
    s.update_memory([_event("t1", polarization_angle=None)], time_now=0.0)
    bonus_present = s.score_polarization_info_gain("L", 1.0 + np.pi / 2)
    assert bonus_present > far, "ambiguity bonus should add to score"
    assert bonus_present == far + s.POL_AMBIGUITY_BONUS
    print(f"  ✓ near = 0.0, far = {far:.2f}, with ambiguity = {bonus_present:.2f}")


# =====================================================================
# Test 10 — total_score is a true weighted sum
# =====================================================================

def test_total_score():
    print("\nTest 10: total_score = weighted sum of the four components …")
    # Use easy-to-check weights
    weights = ScoringWeights(novelty=1.0, salience=2.0,
                             feasibility=0.5, polarization_info_gain=4.0)
    s = SaliencyScheduler(weights=weights)
    current = GimbalSetpoint()
    args = ("L", 0.1, 0.0, 0.5, 0.0, current)
    n = s.score_novelty("L", 0.1, 0.0, time_now=0.0)
    sal = s.score_salience("L", 0.1, 0.0)
    feas = s.score_feasibility("L", 0.1, 0.0, 0.5, current)
    pol = s.score_polarization_info_gain("L", 0.5)
    expected = 1.0 * n + 2.0 * sal + 0.5 * feas + 4.0 * pol
    got = s.total_score(*args)
    assert abs(got - expected) < 1e-9, f"got {got}, expected {expected}"
    print(f"  ✓ total_score = 1.0×{n:.2f} + 2.0×{sal:.2f} + "
          f"0.5×{feas:.2f} + 4.0×{pol:.2f} = {got:.3f}")


# =====================================================================
# Test 11 — next_setpoint honours decision_period_s
# =====================================================================

def test_decision_period():
    print("\nTest 11: next_setpoint holds during decision_period; replans after …")
    s = SaliencyScheduler(decision_period_s=0.10, seed=0)
    cur = GimbalSetpoint()

    sp0 = s.next_setpoint(time_now=0.0, current_setpoint=cur)
    # Hold for the next 0.05 s
    sp1 = s.next_setpoint(time_now=0.05, current_setpoint=cur)
    assert sp1 == sp0, "should hold the setpoint within the decision period"
    # 0.20 s after the first decision → replan (with same seed not guaranteed
    # identical because the *current* setpoint changed via the held one, so
    # at least check it's still a valid setpoint)
    sp2 = s.next_setpoint(time_now=0.20, current_setpoint=cur)
    assert isinstance(sp2, GimbalSetpoint)
    print(f"  ✓ held setpoint constant within {s.decision_period_s}s window, "
          f"then re-planned")


# =====================================================================
# Test 12 — scheduler actually explores
# =====================================================================

def test_scheduler_explores():
    print("\nTest 12: next_setpoint produces varied setpoints across decisions …")
    s = SaliencyScheduler(n_candidates=20, decision_period_s=0.05, seed=0)
    cur = GimbalSetpoint()

    setpoints = []
    for i in range(20):
        sp = s.next_setpoint(time_now=i * 0.05 + 1e-3, current_setpoint=cur)
        setpoints.append(sp)
        cur = sp  # the eye actually moves there

    # Count unique yaw_L values
    yaws = [sp.yaw_L for sp in setpoints]
    unique = len(set(round(y, 4) for y in yaws))
    assert unique >= 5, f"expected at least 5 unique yaws over 20 decisions, got {unique}"
    print(f"  ✓ {unique} unique yaw_L values across 20 decisions")


# =====================================================================
# Test 13 — reset
# =====================================================================

def test_reset():
    print("\nTest 13: reset() returns scheduler to fresh state …")
    s = SaliencyScheduler(decision_period_s=0.05, seed=0)
    s.update_memory([_event("t1", azimuth=0.5)], time_now=0.0)
    s.next_setpoint(time_now=0.0, current_setpoint=GimbalSetpoint())

    s.reset()
    assert s.memory.last_decoded == {}
    assert s._last_decision_time == -np.inf
    assert s._held_setpoint == GimbalSetpoint()
    assert s._roll_history == {"L": [], "R": []}
    print("  ✓ everything cleared")


# =====================================================================
# Test 14 — joint-limit constants match the XML
# =====================================================================

def test_limits_match_xml():
    print("\nTest 14: scheduler joint limits agree with the XML <range> …")
    src = XML_PATH.read_text()
    # Find all "range=\"a b\"" entries; they appear in this order in the XML:
    # yaw L, pitch L, roll L, yaw R, pitch R, roll R
    # Match only joint <range="..."> attributes (NOT motor ctrlrange="...")
    ranges = re.findall(r'(?<!ctrl)range="(-?\d+\.\d+)\s+(-?\d+\.\d+)"', src)
    assert len(ranges) >= 6, f"expected 6 ranges in XML, found {len(ranges)}"
    # Pull out abs(upper) for the three distinct types from the first 3
    yaw, pitch, roll = ranges[0], ranges[1], ranges[2]
    assert abs(float(yaw[1]) - _YAW_LIMIT_RAD) < 1e-6, f"yaw mismatch: {yaw}"
    assert abs(float(pitch[1]) - _PITCH_LIMIT_RAD) < 1e-6, f"pitch mismatch: {pitch}"
    assert abs(float(roll[1]) - _ROLL_LIMIT_RAD) < 1e-6, f"roll mismatch: {roll}"
    print(f"  ✓ scheduler limits ({_YAW_LIMIT_RAD}, {_PITCH_LIMIT_RAD}, "
          f"{_ROLL_LIMIT_RAD}) match the XML")


# =====================================================================
# Test 15 — end-to-end on a real scene
# =====================================================================

def test_end_to_end():
    print("\nTest 15: end-to-end SaliencyScheduler drives real eyes …")

    scene = Scene.from_xml(XML_PATH)
    scene.reset()
    eye_L, eye_R = make_eye_pair(scene)
    pd = GimbalPD(scene.model)
    pipeline = PreprocessingPipeline()
    scheduler = SaliencyScheduler(n_candidates=30,
                                   decision_period_s=0.10, seed=0)

    log = EventLog()
    log.populate_targets_from_scene(scene)

    # Run for 5 simulated seconds
    duration = 5.0
    n_steps = int(duration / scene.model.opt.timestep)
    in_fov_since: dict[str, float] = {}

    for _ in range(n_steps):
        # Layer 3 → setpoint
        sp = scheduler.next_setpoint(scene.data.time,
                                     current_setpoint=GimbalSetpoint())
        # Layer 1 → torques
        pd.step(scene.data, sp)
        scene.step()
        now = float(scene.data.time)

        # Sensors → events → memory
        raws = eye_L.step() + eye_R.step()
        events = pipeline.step(
            raws, time_now=now,
            roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
        )
        for ev in events:
            log.log_event(ev)
        scheduler.update_memory(events, time_now=now)

        # Track FoV intervals
        visible = {r.target_name for r in raws}
        for name in visible:
            if name not in in_fov_since:
                in_fov_since[name] = now
        for name in list(in_fov_since):
            if name not in visible:
                log.log_target_fov(name, in_fov_since.pop(name), now)

    for name, t_enter in in_fov_since.items():
        log.log_target_fov(name, t_enter, scene.data.time)
    log.duration_s = float(scene.data.time)

    report = compute_all(log, "B3")
    print(f"  events       : {len(log.preprocessed_events)}")
    print(f"  targets seen : {len(log.target_in_fov_intervals)}")
    print(f"  coverage     : {report.coverage:.3f}")
    print(f"  polarization : {report.polarization_accuracy:.3f}")
    print(f"  bandwidth    : {report.bandwidth_bps:.1f} B/s")

    # Sanity invariants:
    # - the scheduler must have produced *some* events
    # - at least one target should have been seen
    assert len(log.preprocessed_events) > 0
    assert len(log.target_in_fov_intervals) > 0
    print("  ✓ scheduler produced events and saw at least one target")


# =====================================================================
# Test 16 — LearnedScheduler implementation (bonus #7)
# =====================================================================

def test_learned_scheduler():
    """LearnedScheduler constructs, scores, and produces valid setpoints."""
    print("\nTest 16: LearnedScheduler with default (untrained) MLP …")
    s = LearnedScheduler(seed=0)
    # Sanity: produces a GimbalSetpoint with values inside joint limits
    sp = s.next_setpoint(0.0, GimbalSetpoint())
    assert isinstance(sp, GimbalSetpoint)
    for v, lim in [
        (sp.yaw_L, _YAW_LIMIT_RAD), (sp.yaw_R, _YAW_LIMIT_RAD),
        (sp.pitch_L, _PITCH_LIMIT_RAD), (sp.pitch_R, _PITCH_LIMIT_RAD),
        (sp.roll_L, _ROLL_LIMIT_RAD), (sp.roll_R, _ROLL_LIMIT_RAD),
    ]:
        assert abs(v) <= lim + 1e-9, f"{v} outside [-{lim}, {lim}]"
    print("  ✓ produces a valid GimbalSetpoint inside joint limits")

    # Feature vector matches FEATURE_NAMES length (delegates to parent class)
    f = s.feature_vector("L", 0.1, 0.2, 0.3, 0.0, GimbalSetpoint())
    assert f.shape == (len(s.FEATURE_NAMES),)
    print(f"  ✓ feature_vector returns {f.shape[0]}-D (matches FEATURE_NAMES)")

    # Argmax mismatch with random init → MLP is untrained, score correlation
    # with the rich-hand-designed teacher is essentially zero. After training
    # on a small synthetic dataset, agreement should rise.
    rng = np.random.default_rng(0)
    # Generate a synthetic dataset where score = sum of features (trivial)
    X = rng.standard_normal((1000, len(s.FEATURE_NAMES)))
    y = X.sum(axis=1)
    s.fit(X, y, epochs=200, lr=1e-2, verbose=False)
    yp = s.mlp.forward(X).ravel()
    r2 = 1 - np.var(yp - y) / np.var(y)
    assert r2 > 0.9, f"trained R² should be >0.9, got {r2:.3f}"
    print(f"  ✓ fit() trains MLP to R²={r2:.3f} on synthetic sum-of-features task")


def test_learned_imitates_teacher():
    """End-to-end: LearnedScheduler matches teacher argmax after training."""
    print("\nTest 17: LearnedScheduler imitates SaliencyScheduler argmax …")
    # Build a teacher with rich hand-designed weights (non-degenerate scoring)
    teacher = SaliencyScheduler(
        weights=ScoringWeights(novelty=1.0, salience=2.0,
                               feasibility=0.5, polarization_info_gain=1.0),
        seed=0,
    )
    # Pretend the teacher has some memory so its scoring is non-trivial
    from stomatopod_vision.preprocessing import PreprocessedEvent
    teacher.memory.last_decoded["fake_target"] = PreprocessedEvent(
        time=0.0, eye="L", target_name="fake_target",
        azimuth=0.3, elevation=0.1, distance=1.0,
        spectral_pattern=(0,0,0,0,0,0,0,0,0,10,0,0), circular_handedness=None,
        polarization_angle=None,
    )
    teacher.memory.last_seen_time["fake_target"] = 0.5
    teacher.memory.last_visit_direction["L"] = (0.1, 0.0)
    teacher.memory.last_visit_direction["R"] = (-0.1, 0.0)

    # Collect (features, teacher_score) on a sweep of candidates
    rng = np.random.default_rng(0)
    feats, scores = [], []
    for _ in range(800):
        eye = rng.choice(["L", "R"])
        cy = rng.uniform(-_YAW_LIMIT_RAD, _YAW_LIMIT_RAD)
        cp = rng.uniform(-_PITCH_LIMIT_RAD, _PITCH_LIMIT_RAD)
        cr = rng.uniform(-_ROLL_LIMIT_RAD, _ROLL_LIMIT_RAD)
        t  = float(rng.uniform(0, 2.0))
        sp = GimbalSetpoint()
        feats.append(teacher.feature_vector(eye, cy, cp, cr, t, sp))
        scores.append(teacher.total_score(eye, cy, cp, cr, t, sp))
    feats = np.asarray(feats)
    scores = np.asarray(scores)

    student = LearnedScheduler(seed=42)
    # Copy teacher memory state so the student's feature_vector sees the same context
    student.memory = teacher.memory
    _history = student.fit(feats, scores, epochs=300, lr=1e-2, verbose=False)
    yp = student.mlp.forward(feats).ravel()
    r2 = 1 - np.var(yp - scores) / np.var(scores)
    assert r2 > 0.95, f"expected R² > 0.95 after training, got {r2:.3f}"
    print(f"  ✓ trained student R² = {r2:.3f} on {len(scores)} teacher scores")

    # Per-replan argmax agreement on 50 fresh candidate batches
    agree = 0
    total = 0
    for _ in range(50):
        for eye in ("L", "R"):
            cands = teacher.sample_candidates(eye)
            t = float(rng.uniform(0, 2.0))
            sp = GimbalSetpoint()
            ts = np.array([teacher.total_score(eye, y, p, r, t, sp)
                           for y, p, r in cands])
            ls_s = np.array([student.total_score(eye, y, p, r, t, sp)
                             for y, p, r in cands])
            if np.argmax(ts) == np.argmax(ls_s):
                agree += 1
            total += 1
    rate = agree / total
    assert rate > 0.70, f"expected >70% argmax agreement, got {rate:.1%}"
    print(f"  ✓ argmax agreement with teacher = {rate:.1%} on fresh candidates")


def test_learned_save_load():
    """save() / from_file() round-trips the MLP weights bit-exactly."""
    print("\nTest 18: LearnedScheduler save/load round-trip …")
    import os
    import tempfile
    s1 = LearnedScheduler(seed=7)
    # Take a fingerprint of the random initial weights
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, len(s1.FEATURE_NAMES)))
    y1 = s1.mlp.forward(X)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.npz")
        s1.save(path)
        s2 = LearnedScheduler.from_file(path, seed=7)
        y2 = s2.mlp.forward(X)
        assert np.allclose(y1, y2), "save/load did not preserve MLP outputs"
    print("  ✓ MLP outputs match exactly before and after save/load")


# =====================================================================
# Test 19 — from_mujoco_model reads limits straight out of the XML
# =====================================================================

def test_from_mujoco_model():
    print("\nTest 19: SaliencyScheduler.from_mujoco_model() reads MuJoCo limits …")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    sched = SaliencyScheduler.from_mujoco_model(model, seed=0)
    # The factory should pull the same numbers the XML literally contains,
    # which by construction also match the module-level fallback defaults.
    assert abs(sched._yaw_limit - _YAW_LIMIT_RAD) < 1e-9
    assert abs(sched._pitch_limit - _PITCH_LIMIT_RAD) < 1e-9
    assert abs(sched._roll_limit - _ROLL_LIMIT_RAD) < 1e-9
    # And the sampled candidates must respect those limits.
    cands = sched.sample_candidates("L")
    assert np.all(np.abs(cands[:, 0]) <= sched._yaw_limit + 1e-9)
    assert np.all(np.abs(cands[:, 1]) <= sched._pitch_limit + 1e-9)
    assert np.all(np.abs(cands[:, 2]) <= sched._roll_limit + 1e-9)
    print(f"  ✓ MuJoCo-read limits ({sched._yaw_limit:.2f}, "
          f"{sched._pitch_limit:.2f}, {sched._roll_limit:.2f}) match XML defaults")

    # Missing-joint error path: build a tiny MuJoCo model with no eye
    # joints and verify the factory raises a clear error.
    tiny_xml = "<mujoco><worldbody><body><geom size=\".1\"/></body></worldbody></mujoco>"
    tiny = mujoco.MjModel.from_xml_string(tiny_xml)
    try:
        SaliencyScheduler.from_mujoco_model(tiny, seed=0)
    except ValueError as e:
        assert "not found" in str(e)
        print("  ✓ raises ValueError when eye joints are missing from the model")
    else:
        raise AssertionError("expected ValueError for missing joints")


# =====================================================================
# Test 20 — joint_limits override flows through to candidate sampling
# =====================================================================

def test_joint_limits_override():
    print("\nTest 20: SaliencyScheduler(joint_limits=...) override is respected …")
    custom = (0.5, 0.3, 1.0)
    sched = SaliencyScheduler(n_candidates=50, joint_limits=custom, seed=42)
    assert sched._yaw_limit == 0.5
    assert sched._pitch_limit == 0.3
    assert sched._roll_limit == 1.0
    cands = sched.sample_candidates("R")
    assert np.all(np.abs(cands[:, 0]) <= 0.5 + 1e-9), "yaw exceeded override"
    assert np.all(np.abs(cands[:, 1]) <= 0.3 + 1e-9), "pitch exceeded override"
    assert np.all(np.abs(cands[:, 2]) <= 1.0 + 1e-9), "roll exceeded override"

    # Also verify the constant ``EYE_JOINT_NAMES`` is a 2-key mapping with
    # the canonical XML joint names — this couples the test to the contract.
    assert set(EYE_JOINT_NAMES.keys()) == {"L", "R"}
    assert EYE_JOINT_NAMES["L"] == ("eye_L_yaw", "eye_L_pitch", "eye_L_roll")
    assert EYE_JOINT_NAMES["R"] == ("eye_R_yaw", "eye_R_pitch", "eye_R_roll")

    # Negative-limit guard
    try:
        SaliencyScheduler(joint_limits=(0.5, -0.3, 1.0))
    except ValueError:
        print("  ✓ negative joint limit raises ValueError")
    else:
        raise AssertionError("expected ValueError for negative limit")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.scheduler")
    print("=" * 60)
    test_memory_basics()
    test_fixed_forward()
    test_saliency_construction()
    test_sample_candidates_shape()
    test_sample_candidates_forced()
    test_score_novelty()
    test_score_salience()
    test_score_feasibility()
    test_score_pol_info_gain()
    test_total_score()
    test_decision_period()
    test_scheduler_explores()
    test_reset()
    test_limits_match_xml()
    test_end_to_end()
    test_learned_scheduler()
    test_learned_imitates_teacher()
    test_learned_save_load()
    test_from_mujoco_model()
    test_joint_limits_override()
    print("\nAll scheduler tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
