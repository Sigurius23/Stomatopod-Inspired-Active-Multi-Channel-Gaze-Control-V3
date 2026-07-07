"""
Tests for stomatopod_vision.world

Validates:
  1. TargetMeta constructs correctly, rejects invalid spectral_class,
     and wraps polarization_angle to [0, π).
  2. Scene.from_xml loads the canonical model and finds all default targets.
  3. target_world_position returns sensible coordinates that match the XML.
  4. interesting_target_names and polarized_targets filter correctly.
  5. reset and step advance the simulation cleanly.
  6. Mismatched targets (a name not in the XML) raise a clear error.
  7. random_targets generates the requested number of targets with the
     correct number flagged as interesting.

Run from the repo root:
    python tests/test_world.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision.world import (  # noqa: E402
    SPECTRAL_CLASSES,
    Scene,
    TargetMeta,
    random_targets,
)

XML_PATH = REPO_ROOT / "models" / "stomatopod_eyes.xml"


# ---------------------------------------------------------------------
# Test 1 — TargetMeta validation
# ---------------------------------------------------------------------

def test_target_meta():
    print("Test 1: TargetMeta validation …")

    # Happy path
    t = TargetMeta("a", "C10", np.pi / 4, None, True)
    assert t.name == "a"
    assert t.spectral_class == "C10"
    assert abs(t.polarization_angle - np.pi / 4) < 1e-9
    assert t.is_interesting is True
    print("  ✓ basic construction")

    # Defaults
    t2 = TargetMeta("b", "C1")
    assert t2.polarization_angle is None
    assert t2.is_interesting is False
    print("  ✓ defaults applied")

    # Bad spectral_class
    try:
        TargetMeta("c", "UV_Z")
    except ValueError:
        print("  ✓ rejects unknown spectral_class")
    else:
        raise AssertionError("should have rejected UV_Z")

    # Polarization wrapping
    t3 = TargetMeta("d", "C1", polarization_angle=np.pi + 0.1)
    assert 0 <= t3.polarization_angle < np.pi
    assert abs(t3.polarization_angle - 0.1) < 1e-9
    print(f"  ✓ wraps polarization_angle into [0, π): "
          f"π+0.1 → {t3.polarization_angle:.4f}")


# ---------------------------------------------------------------------
# Test 2 — Scene loads from XML
# ---------------------------------------------------------------------

def test_scene_loads():
    print("\nTest 2: Scene.from_xml loads model and resolves all targets …")
    scene = Scene.from_xml(XML_PATH)
    assert scene.model is not None
    assert scene.data is not None
    assert len(scene.targets) == len(Scene.DEFAULT_TARGETS)
    print(f"  ✓ scene constructed with {len(scene.targets)} targets")
    for t in scene.targets:
        assert t.name in scene._target_body_ids
        print(f"  ✓ resolved body id for {t.name!r} → {scene._target_body_ids[t.name]}")


# ---------------------------------------------------------------------
# Test 3 — target positions match the XML
# ---------------------------------------------------------------------

def test_target_positions():
    print("\nTest 3: target world positions match the XML defaults …")
    scene = Scene.from_xml(XML_PATH)
    # These coordinates come from the XML <body name="target_X_Y" pos="..."/>
    expected = {
        "target_R_1":     np.array([-0.40, 0.80, 1.15]),
        "target_G_1":     np.array([ 0.30, 0.95, 1.20]),
        "target_B_1":     np.array([ 0.50, 0.60, 0.85]),
        "target_UVA_1":   np.array([-0.50, 0.75, 0.85]),
        "target_UVB_1":   np.array([ 0.00, 1.00, 1.45]),
        "target_UVpol_1": np.array([-0.35, 1.10, 1.05]),
    }
    for name, want in expected.items():
        got = scene.target_world_position(name)
        assert np.allclose(got, want, atol=1e-6), \
            f"{name}: got {got}, expected {want}"
        print(f"  ✓ {name:18s} pos = {got.round(2).tolist()}")


# ---------------------------------------------------------------------
# Test 4 — filters
# ---------------------------------------------------------------------

def test_filters():
    print("\nTest 4: interesting + polarized filters …")
    scene = Scene.from_xml(XML_PATH)

    interesting = scene.interesting_target_names()
    assert interesting == ("target_UVpol_1",), \
        f"expected only target_UVpol_1 to be interesting, got {interesting}"
    print(f"  ✓ interesting_target_names = {interesting}")

    polarized = list(scene.polarized_targets().keys()) + list(scene.circularly_polarized_targets().keys())
    assert polarized == ["target_UVpol_1"]
    print(f"  ✓ polarized_targets = {polarized}")


# ---------------------------------------------------------------------
# Test 5 — reset and step
# ---------------------------------------------------------------------

def test_reset_and_step():
    print("\nTest 5: reset + step advance the simulation …")
    scene = Scene.from_xml(XML_PATH)
    scene.reset()
    t0 = float(scene.data.time)
    scene.step(dt_steps=100)
    t1 = float(scene.data.time)
    elapsed = t1 - t0
    expected = 100 * scene.model.opt.timestep
    assert abs(elapsed - expected) < 1e-9, \
        f"expected {expected}s elapsed, got {elapsed}s"
    print(f"  ✓ 100 steps advanced time by {elapsed*1000:.1f} ms "
          f"(expected {expected*1000:.1f} ms)")


# ---------------------------------------------------------------------
# Test 6 — wrong target name raises
# ---------------------------------------------------------------------

def test_missing_target_raises():
    print("\nTest 6: a bogus target name produces a clear error …")
    bad_target = TargetMeta("does_not_exist", "C1")
    try:
        Scene.from_xml(XML_PATH, targets=(bad_target,))
    except ValueError as e:
        print(f"  ✓ rejected with: {e}")
    else:
        raise AssertionError("should have rejected unknown target body name")

    # Also test runtime queries
    scene = Scene.from_xml(XML_PATH)
    try:
        scene.target_world_position("not_a_real_target")
    except KeyError as e:
        print(f"  ✓ target_world_position raised KeyError: {e}")
    else:
        raise AssertionError("should have raised KeyError")


# ---------------------------------------------------------------------
# Test 7 — random_targets generates correct structure
# ---------------------------------------------------------------------

def test_random_targets():
    print("\nTest 7: random_targets generates correct structure …")

    targets = random_targets(n=10, seed=42, n_interesting=3)
    assert len(targets) == 10
    print(f"  ✓ generated {len(targets)} targets")

    n_interesting = sum(t.is_interesting for t in targets)
    assert n_interesting == 3, \
        f"expected 3 interesting, got {n_interesting}"
    print(f"  ✓ exactly {n_interesting} flagged as interesting")

    # All spectral classes valid
    for t in targets:
        assert t.spectral_class in SPECTRAL_CLASSES
    print("  ✓ all spectral_class values are valid")

    # Determinism: same seed → same output
    targets2 = random_targets(n=10, seed=42, n_interesting=3)
    assert all(a == b for a, b in zip(targets, targets2, strict=True))
    print("  ✓ seed produces deterministic output")

    # Different seed → different output
    targets3 = random_targets(n=10, seed=43, n_interesting=3)
    assert any(a != b for a, b in zip(targets, targets3, strict=True))
    print("  ✓ different seed produces different output")

    # Validation: bad inputs raise
    try:
        random_targets(n=-1)
    except ValueError:
        print("  ✓ rejects negative n")
    else:
        raise AssertionError("should have rejected n=-1")

    try:
        random_targets(n=5, n_interesting=10)
    except ValueError:
        print("  ✓ rejects n_interesting > n")
    else:
        raise AssertionError("should have rejected n_interesting=10 > n=5")

    # All-interesting works
    all_interesting = random_targets(n=4, seed=0, n_interesting=4)
    assert all(t.is_interesting for t in all_interesting)
    print("  ✓ all-interesting case works")




def test_hard_scene():
    """models/stomatopod_eyes_hard.xml + Scene.HARD_TARGETS — the B3-wins variant."""
    import math
    print()
    print("Test 8: hard scene loads with the right interesting targets")
    hard_xml = REPO_ROOT / "models" / "stomatopod_eyes_hard.xml"
    assert hard_xml.exists(), f"missing {hard_xml}"
    scene = Scene.from_xml(hard_xml)
    # Auto-detection by filename should pick HARD_TARGETS
    assert scene.targets == Scene.HARD_TARGETS, \
        "Scene.from_xml(models/stomatopod_eyes_hard.xml) should auto-pick HARD_TARGETS"
    print(f"  ✓ auto-picked HARD_TARGETS ({len(scene.targets)} entries)")

    # Exactly ten interesting targets, all polarized
    interesting = scene.interesting_target_names()
    assert len(interesting) == 10, f"expected 10 interesting, got {len(interesting)}: {interesting}"
    print("  ✓ 10 interesting targets")

    polarized_keys = set(scene.polarized_targets().keys()) | set(scene.circularly_polarized_targets().keys())
    assert polarized_keys == set(interesting), \
        "every interesting target in HARD_TARGETS should be polarized"
    print("  ✓ all 10 interesting targets carry a polarization angle")

    # Polarization angles span ~[0, π) — well distributed
    angles = sorted(scene.polarized_targets().values())
    assert (angles[-1] - angles[0]) > np.pi * 0.8, \
        f"angles too clustered: {angles}"
    print(f"  ✓ polarization angles span [{angles[0]:.2f}, {angles[-1]:.2f}] rad")

    # Verify ALL 10 interesting targets sit outside the rest FoV
    # (rest FoV is ±60° azimuth × ±5° elevation per eye, eyes at x=±0.12, z≈1.04).
    # Each target must be outside the rest FoV of BOTH eyes.
    eye_L = np.array([-0.12, 0.0, 1.04])
    eye_R = np.array([ 0.12, 0.0, 1.04])
    outside_count = 0
    for name in interesting:
        pos = scene.target_world_position(name)
        outside_both = True
        for ep in (eye_L, eye_R):
            v = pos - ep
            az = math.degrees(math.atan2(v[0], v[1]))
            el = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
            if abs(az) <= 60.0 and abs(el) <= 5.0:
                outside_both = False
                break
        assert outside_both, \
            f"{name} is inside the rest FoV of at least one eye — B1/B2 would see it for free"
        outside_count += 1
    print(f"  ✓ all {outside_count} interesting targets lie outside the rest FoV "
          f"(B1/B2 see 0 of them)")

    # UV decoys must be PRESENT but non-interesting
    decoy_names = ("target_UVA_dL", "target_UVA_dR",
                   "target_UVB_dL", "target_UVB_dR")
    for n in decoy_names:
        m = scene.target_meta(n)
        assert m.is_interesting is False, f"decoy {n} should NOT be interesting"
        assert m.polarization_angle is None, f"decoy {n} should NOT be polarized"
    print("  ✓ 4 UV decoys present, non-interesting, non-polarized")



def test_target_motion_kinds():
    """TargetMotion.displacement_at() obeys the documented contract."""
    print()
    print("Test 9: TargetMotion kinds (static/circular/linear) …")
    from stomatopod_vision.world import TargetMotion

    # static → always zero, regardless of time
    m_static = TargetMotion()  # default kind="static"
    assert np.allclose(m_static.displacement_at(0.0), 0.0)
    assert np.allclose(m_static.displacement_at(123.4), 0.0)
    print("  ✓ static returns zero displacement at any time")

    # circular: starts at "axis-perpendicular" basis * amplitude, period 4s
    # → traces a circle of radius 0.25 in the xy plane (axis = z).
    m_circ = TargetMotion(kind="circular", period_s=4.0,
                          amplitude_m=0.25, axis=(0, 0, 1))
    p0 = m_circ.displacement_at(0.0)
    p1 = m_circ.displacement_at(1.0)   # quarter cycle
    p2 = m_circ.displacement_at(2.0)   # half cycle
    p4 = m_circ.displacement_at(4.0)   # full cycle
    # |p| == amplitude at every t
    for t in (0.0, 0.5, 1.0, 1.7, 2.0, 3.3):
        assert abs(np.linalg.norm(m_circ.displacement_at(t)) - 0.25) < 1e-9, \
            f"|disp(t={t})| ≠ amplitude"
    # axis component (z) stays 0
    assert abs(p1[2]) < 1e-9 and abs(p2[2]) < 1e-9
    # quarter cycle is orthogonal to start; full cycle returns to start
    assert abs(np.dot(p0, p1)) < 1e-9
    assert np.allclose(p4, p0, atol=1e-9)
    print("  ✓ circular orbit radius=amplitude, axis-perp, periodic")

    # linear: sin oscillation along axis, amplitude is the peak
    m_lin = TargetMotion(kind="linear", period_s=2.0, amplitude_m=0.5,
                         axis=(0, 0, 1))
    assert np.allclose(m_lin.displacement_at(0.0), 0.0)
    assert np.allclose(m_lin.displacement_at(0.5), [0, 0,  0.5])
    assert np.allclose(m_lin.displacement_at(1.5), [0, 0, -0.5])
    print("  ✓ linear oscillation peaks at ±amplitude along axis")

    # validation: bad kind / non-positive period
    try:
        TargetMotion(kind="bogus")
    except ValueError as e:
        print(f"  ✓ rejected bogus kind ({e})")
    try:
        TargetMotion(kind="linear", period_s=0.0)
    except ValueError as e:
        print(f"  ✓ rejected period_s=0 ({e})")


def test_moving_scene_loads_and_animates():
    """Loading the moving scene gives MOVING_TARGETS + mocap bodies."""
    print()
    print("Test 10: moving scene XML loads with mocap targets …")
    from stomatopod_vision.world import MovingTargetController
    xml = REPO_ROOT / "models" / "stomatopod_eyes_moving.xml"
    assert xml.exists(), f"missing {xml}"

    scene = Scene.from_xml(xml)
    # Auto-detect by filename stem assigns MOVING_TARGETS (= DEFAULT_TARGETS alias)
    assert scene.targets == Scene.MOVING_TARGETS, \
        "from_xml should auto-pick MOVING_TARGETS for the moving XML"
    print(f"  ✓ auto-picked MOVING_TARGETS ({len(scene.targets)} entries)")

    # All 6 target bodies must be mocap=true in the XML
    n_mocap = int(scene.model.nmocap)
    assert n_mocap == 6, f"expected 6 mocap bodies, got {n_mocap}"
    print(f"  ✓ XML declares {n_mocap} mocap targets")

    # Build a controller and step it forward; check target_R_1 moved
    mc = MovingTargetController(scene, Scene.MOVING_MOTIONS)
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "target_R_1")
    pos_t0 = scene.data.xpos[bid].copy()
    # Step the controller forward to t=1.0 s (quarter of target_R_1\'s 4s period)
    # At t=0 the displacement is `amplitude * u` (a basis vector); at t=1
    # (quarter period) it\'s `amplitude * v` (the orthogonal basis vector).
    # The displacement from REST is therefore the amplitude itself (0.25 m).
    mc.step(1.0)
    mujoco.mj_forward(scene.model, scene.data)
    pos_t1 = scene.data.xpos[bid].copy()
    delta = float(np.linalg.norm(pos_t1 - pos_t0))
    expected = 0.25  # amplitude of the orbit
    assert abs(delta - expected) < 0.01, \
        f"target_R_1 moved {delta:.3f} m at t=1s, expected ≈ {expected:.3f} m"
    print(f"  ✓ target_R_1 moved {delta:.3f} m (≈amplitude) at t=1.0s")

    # Verify the rest position is always on the orbit\'s circle:
    # at t=0 the controller hasn\'t stepped yet, so xpos == rest. Step
    # the controller to t=4.0 (full period) and check return to rest.
    mc.step(4.0)
    mujoco.mj_forward(scene.model, scene.data)
    pos_t4 = scene.data.xpos[bid].copy()
    # Full period: should be (rest + amplitude*u) since u is the t=0 disp basis
    # so distance from rest = amplitude (NOT zero — t=0 disp is not zero).
    assert abs(float(np.linalg.norm(pos_t4 - pos_t0)) - 0.25) < 1e-6
    print("  ✓ target_R_1 at t=4.0s (one period) returns to t=0 phase point")

    # Static target (target_UVA_1) should NOT have moved
    bid2 = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "target_UVA_1")
    rest = np.array([-0.50, 0.75, 0.85])
    assert np.allclose(scene.data.xpos[bid2], rest, atol=1e-6), \
        f"target_UVA_1 should be static; got {scene.data.xpos[bid2]}"
    print("  ✓ unmotioned target_UVA_1 stayed at rest")

    # reset() snaps everything back to rest
    mc.reset()
    mujoco.mj_forward(scene.model, scene.data)
    assert np.allclose(scene.data.xpos[bid], pos_t0, atol=1e-9)
    print("  ✓ MovingTargetController.reset() restores rest positions")


def test_moving_controller_ignores_non_mocap():
    """MovingTargetController silently skips non-mocap targets."""
    print()
    print("Test 11: MovingTargetController skips non-mocap targets …")
    from stomatopod_vision.world import MovingTargetController, TargetMotion
    # Default scene's targets are NOT mocap
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    motions = {"target_R_1": TargetMotion(kind="circular", period_s=1.0,
                                          amplitude_m=0.1)}
    mc = MovingTargetController(scene, motions)
    # No animated targets in the plan since target_R_1 isn\'t mocap
    assert len(mc._plan) == 0, \
        f"expected 0 animated targets on the static scene, got {len(mc._plan)}"
    # step() should be a no-op (no error)
    mc.step(0.5)
    print("  ✓ silently skips non-mocap targets without raising")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.world")
    print("=" * 60)
    test_target_meta()
    test_scene_loads()
    test_target_positions()
    test_filters()
    test_reset_and_step()
    test_missing_target_raises()
    test_random_targets()
    test_hard_scene()
    test_target_motion_kinds()
    test_moving_scene_loads_and_animates()
    test_moving_controller_ignores_non_mocap()
    print("\nAll world tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
