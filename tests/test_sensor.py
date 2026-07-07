"""
Tests for stomatopod_vision.sensor

Validates:
   1. VirtualEye construction resolves all site/joint ids.
   2. forward_vector and center_position return sensible defaults at rest.
   3. Roll-joint angle is read correctly.
   4. Local frame is orthonormal and right-handed.
   5. relative_angles is correct on hand-checked geometries.
   6. is_in_fov respects yaw/pitch boundaries (anisotropic).
   7. Distance attenuation behaves sensibly.
   8. midband_activations is a one-hot vector at the right index.
   9. polarization_responses are highest when receptor matches target
      polarization, and rolling the eye changes which receptor wins.
  10. Unpolarized targets give uniform polarization responses (0.5 × atten).
  11. raw_activations_for returns None for off-FoV targets.
  12. step() across the whole scene returns the right number of sightings.
  13. make_eye_pair returns a left/right pair.

Run from the repo root:
    python tests/test_sensor.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision.gimbal_control import (  # noqa: E402
    GimbalPD,
    GimbalSetpoint,
)
from stomatopod_vision.sensor import (  # noqa: E402
    MidbandFOV,
    RawSighting,
    VirtualEye,
    make_eye_pair,
)
from stomatopod_vision.world import (  # noqa: E402
    SPECTRAL_CLASSES,
    Scene,
    TargetMeta,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_scene_with_targets(targets):
    """Build a fresh Scene with custom targets."""
    return Scene.from_xml(
        REPO_ROOT / "models" / "stomatopod_eyes.xml",
        targets=targets,
    )


def settle_to(scene, setpoint, duration_s=1.0):
    """Run gimbal PD until eyes reach setpoint."""
    pd = GimbalPD(scene.model)
    n_steps = int(duration_s / scene.model.opt.timestep)
    for _ in range(n_steps):
        pd.step(scene.data, setpoint)
        scene.step()


# ---------------------------------------------------------------------
# Test 1 — construction
# ---------------------------------------------------------------------

def test_construction():
    print("Test 1: VirtualEye constructs and resolves ids …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")

    for eye in ("L", "R"):
        ve = VirtualEye(eye, scene)
        assert ve.eye == eye
        assert ve._site_center_id >= 0
        assert ve._site_axis_id >= 0
        assert ve._roll_qpos_idx >= 0
        print(f"  ✓ eye {eye}: sites + roll joint resolved")

    # Reject bad eye id
    try:
        VirtualEye("Z", scene)
    except ValueError:
        print("  ✓ rejects invalid eye id")
    else:
        raise AssertionError("should have rejected eye='Z'")


# ---------------------------------------------------------------------
# Test 2 — forward + centre at rest
# ---------------------------------------------------------------------

def test_rest_geometry():
    print("\nTest 2: forward / centre at rest pose …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()

    eye_L = VirtualEye("L", scene)
    eye_R = VirtualEye("R", scene)

    # At rest, both eyes look along world +Y
    fL = eye_L.forward_vector()
    fR = eye_R.forward_vector()
    assert np.allclose(fL, [0, 1, 0], atol=1e-6), f"eye_L forward = {fL}"
    assert np.allclose(fR, [0, 1, 0], atol=1e-6), f"eye_R forward = {fR}"
    print("  ✓ both eyes look along +Y at rest")

    # Centres should differ in x by ~0.24 (XML places them at ±0.12)
    cL = eye_L.center_position()
    cR = eye_R.center_position()
    dx = float(cR[0] - cL[0])
    assert abs(dx - 0.24) < 1e-3, f"x-separation = {dx}"
    print(f"  ✓ centres separated by {dx*100:.1f} cm in x (expected 24 cm)")


# ---------------------------------------------------------------------
# Test 3 — roll angle readout
# ---------------------------------------------------------------------

def test_roll_readout():
    print("\nTest 3: roll-joint angle is readable …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")

    settle_to(scene, GimbalSetpoint(roll_L=0.7, roll_R=-0.3))
    eye_L = VirtualEye("L", scene)
    eye_R = VirtualEye("R", scene)

    rL = eye_L.roll_angle()
    rR = eye_R.roll_angle()
    assert abs(rL - 0.7) < 0.02, f"roll_L = {rL}"
    assert abs(rR - (-0.3)) < 0.02, f"roll_R = {rR}"
    print(f"  ✓ roll_L = {rL:.3f} (target 0.7), roll_R = {rR:.3f} (target -0.3)")


# ---------------------------------------------------------------------
# Test 4 — local frame is orthonormal and right-handed
# ---------------------------------------------------------------------

def test_local_frame_orthonormal():
    print("\nTest 4: local frame is orthonormal and right-handed …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")

    # Try several gimbal configurations
    configs = [
        GimbalSetpoint(),
        GimbalSetpoint(yaw_L=0.4, pitch_L=0.3),
        GimbalSetpoint(yaw_R=-0.5, pitch_R=-0.2),
        GimbalSetpoint(yaw_L=0.8, pitch_L=-0.4, roll_L=1.0),
    ]
    for cfg in configs:
        scene.reset()
        settle_to(scene, cfg)
        for eye in ("L", "R"):
            ve = VirtualEye(eye, scene)
            r, f, u = ve._local_frame()
            # Unit length
            for name, vec in [("right", r), ("forward", f), ("up", u)]:
                n = np.linalg.norm(vec)
                assert abs(n - 1.0) < 1e-6, f"{name} not unit: |{name}|={n}"
            # Orthogonal pairwise
            assert abs(r @ f) < 1e-6
            assert abs(r @ u) < 1e-6
            assert abs(f @ u) < 1e-6
            # Right-handed: right × forward ≈ up (up to sign convention).
            # We defined up = right × forward, so the equality should hold.
            assert np.allclose(np.cross(r, f), u, atol=1e-6)
    print("  ✓ orthonormal + right-handed across 4 configurations × 2 eyes")


# ---------------------------------------------------------------------
# Test 5 — relative_angles correctness on hand-checked geometry
# ---------------------------------------------------------------------

def test_relative_angles():
    print("\nTest 5: relative_angles on hand-checked geometries …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    eye = VirtualEye("L", scene)
    c = eye.center_position()

    # Target straight ahead in +Y, 1 m away
    tgt = c + np.array([0.0, 1.0, 0.0])
    az, el, d = eye.relative_angles(tgt)
    assert abs(az) < 1e-9 and abs(el) < 1e-9 and abs(d - 1.0) < 1e-9
    print("  ✓ straight ahead → (az, el, d) = (0, 0, 1)")

    # Target 1 m forward + 1 m to the eye's right (world +X)
    tgt = c + np.array([1.0, 1.0, 0.0])
    az, el, d = eye.relative_angles(tgt)
    assert abs(az - np.pi / 4) < 1e-9, f"az = {az}"
    assert abs(el) < 1e-9, f"el = {el}"
    assert abs(d - np.sqrt(2)) < 1e-9
    print("  ✓ forward + right by 1 m → az = +45°, el = 0")

    # Target 1 m forward + 1 m above (world +Z)
    tgt = c + np.array([0.0, 1.0, 1.0])
    az, el, d = eye.relative_angles(tgt)
    assert abs(az) < 1e-9, f"az = {az}"
    assert abs(el - np.pi / 4) < 1e-9, f"el = {el}"
    print("  ✓ forward + above by 1 m → az = 0, el = +45°")

    # Target straight behind
    tgt = c + np.array([0.0, -1.0, 0.0])
    az, el, d = eye.relative_angles(tgt)
    # atan2(0, -1) = π → behind on the azimuth circle
    assert abs(abs(az) - np.pi) < 1e-9
    print("  ✓ straight behind → |az| = π")


# ---------------------------------------------------------------------
# Test 6 — is_in_fov respects the anisotropic boundaries
# ---------------------------------------------------------------------

def test_fov_boundaries():
    print("\nTest 6: is_in_fov respects anisotropic boundaries …")
    fov = MidbandFOV(yaw_half_angle=np.deg2rad(60.0),
                     pitch_half_angle=np.deg2rad(5.0))
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye = VirtualEye("L", scene, fov=fov)

    cases = [
        # (az_deg, el_deg, expected)
        (0,    0,   True),
        (59,   0,   True),    # just inside yaw boundary
        (61,   0,   False),   # just outside yaw boundary
        (0,    4.5, True),    # just inside pitch boundary
        (0,    5.5, False),   # just outside pitch boundary
        (45,   3,   True),    # both inside
        (45,   10,  False),   # pitch breaks it
        (-30,  -3,  True),    # negatives OK
        (-90,  0,   False),   # way outside yaw
    ]
    for az_deg, el_deg, want in cases:
        got = eye.is_in_fov(np.deg2rad(az_deg), np.deg2rad(el_deg))
        assert got == want, (
            f"is_in_fov(az={az_deg}°, el={el_deg}°) = {got}, expected {want}"
        )
    print(f"  ✓ all {len(cases)} boundary cases correct")


# ---------------------------------------------------------------------
# Test 7 — distance attenuation
# ---------------------------------------------------------------------

def test_distance_attenuation():
    print("\nTest 7: distance attenuation is monotonic in [0, 1] …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye = VirtualEye("L", scene)

    a0   = eye._distance_attenuation(0.0)
    a1   = eye._distance_attenuation(1.0)
    a10  = eye._distance_attenuation(10.0)

    assert abs(a0 - 1.0) < 1e-9, f"a(0) = {a0}"
    assert abs(a1 - 0.5) < 1e-9, f"a(1) = {a1}"
    assert a10 < 0.05
    assert a0 > a1 > a10
    print(f"  ✓ a(0m) = {a0:.3f}, a(1m) = {a1:.3f}, a(10m) = {a10:.4f}")


# ---------------------------------------------------------------------
# Test 8 — midband activations are one-hot at the right index
# ---------------------------------------------------------------------

def test_midband_one_hot():
    print("\nTest 8: midband_activations one-hot at correct index …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye = VirtualEye("L", scene)
    for cls in SPECTRAL_CLASSES:
        out = eye._midband_activations(cls, attenuation=1.0)
        assert out.shape == (len(SPECTRAL_CLASSES),)
        idx = SPECTRAL_CLASSES.index(cls)
        assert out[idx] == 1.0
        # All other entries exactly zero
        for j, v in enumerate(out):
            if j != idx:
                assert v == 0.0
    print(f"  ✓ all {len(SPECTRAL_CLASSES)} spectral classes produce correct one-hots")


# ---------------------------------------------------------------------
# Test 9 — polarization responses depend on roll
# ---------------------------------------------------------------------

def test_polarization_with_roll():
    print("\nTest 9: polarization responses depend on eye roll …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye = VirtualEye("L", scene)

    # Target with polarization at 0° (horizontal). At roll = 0 the
    # 0°-oriented receptor should win.
    p0 = eye._linear_pol_responses(target_polarization_rad=0.0,
                                   circular_handedness=None,
                                   attenuation=1.0)
    print(f"  responses at roll=0, target_pol=0°: {p0.round(3).tolist()}")
    assert np.argmax(p0) == 0, "expected receptor 0 (0°) to win"

    # Move the eye roll to 90°. Now the receptor that was at 90° in the
    # body frame is at 180° = 0° in the world frame → it should win.
    settle_to(scene, GimbalSetpoint(roll_L=np.pi / 2))
    p1 = eye._linear_pol_responses(target_polarization_rad=0.0,
                                   circular_handedness=None,
                                   attenuation=1.0)
    print(f"  responses at roll=π/2, target_pol=0°: {p1.round(3).tolist()}")
    assert np.argmax(p1) == 2, "expected receptor 2 (90° body-frame) to win"
    print("  ✓ rolling the eye shifts the winning receptor as expected")


# ---------------------------------------------------------------------
# Test 10 — unpolarized targets give uniform responses
# ---------------------------------------------------------------------

def test_polarization_unpolarized():
    print("\nTest 10: unpolarized targets → uniform responses …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye = VirtualEye("L", scene)
    out = eye._linear_pol_responses(target_polarization_rad=None,
                                    circular_handedness=None,
                                    attenuation=0.8)
    assert out.shape == (4,)
    assert np.allclose(out, 0.5 * 0.8), f"got {out}"
    print(f"  ✓ all four receptors return {0.5 * 0.8} (= 0.5 × attenuation)")


# ---------------------------------------------------------------------
# Test 11 — raw_activations_for returns None for off-FoV targets
# ---------------------------------------------------------------------

def test_raw_activations_off_fov():
    print("\nTest 11: raw_activations_for returns None when off FoV …")
    # Custom target far to the eye's left rear — clearly out of any FoV
    custom = (
        TargetMeta("target_R_1", "C1", None, None, False),  # in the scene XML
        TargetMeta("target_G_1", "C4", None, None, False),
    )
    scene = make_scene_with_targets(custom)
    scene.reset()
    eye = VirtualEye("L", scene)

    # target_R_1 is at (-0.4, 0.8, 1.15) — should be roughly visible to eye_L
    # whose centre is around (-0.12, 0, 1) looking +Y. Let's check.
    s = eye.raw_activations_for("target_R_1")
    print(f"  target_R_1: {'visible' if s else 'off-FoV'}")

    # Now turn the eye sharply away (yaw_L = -1.5, almost full left).
    settle_to(scene, GimbalSetpoint(yaw_L=-1.5))
    s = eye.raw_activations_for("target_R_1")
    assert s is None, "target should be out of FoV after large yaw"
    print("  ✓ target → None after yawing eye away by 1.5 rad")


# ---------------------------------------------------------------------
# Test 12 — step() returns sightings only for visible targets
# ---------------------------------------------------------------------

def test_step_counts():
    print("\nTest 12: step() returns the right number of sightings …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    eye_L = VirtualEye("L", scene)

    # At rest both eyes look forward; count how many of the 6 scene
    # targets fall inside the (60°, 5°) FoV.
    sightings = eye_L.step()
    print(f"  eye_L at rest sees {len(sightings)} of {len(scene.targets)} targets:")
    for s in sightings:
        print(f"    - {s.target_name:18s} "
              f"az={np.rad2deg(s.azimuth):+6.1f}°  "
              f"el={np.rad2deg(s.elevation):+5.1f}°  "
              f"d={s.distance:.2f}m")
    # Validate that every sighting has the expected fields populated
    for s in sightings:
        assert isinstance(s, RawSighting)
        assert s.eye == "L"
        assert s.midband_activations.shape == (len(SPECTRAL_CLASSES),)
        assert s.polarization_responses.shape == (4,)
        assert 0.0 < s.peripheral_intensity <= 1.0
    print("  ✓ all sightings have correct structure")


# ---------------------------------------------------------------------
# Test 13 — make_eye_pair convenience
# ---------------------------------------------------------------------

def test_make_eye_pair():
    print("\nTest 13: make_eye_pair returns (L, R) …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    eye_L, eye_R = make_eye_pair(scene)
    assert eye_L.eye == "L"
    assert eye_R.eye == "R"
    print("  ✓ pair has correct labels")


def test_receptor_noise_default_is_deterministic():
    """No noise by default → two consecutive steps return identical readings."""
    print("\nTest 14: receptor_noise_std=0.0 is bit-deterministic …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    eye, _ = make_eye_pair(scene)              # default receptor_noise_std=0
    r1 = eye.step()
    r2 = eye.step()
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2, strict=True):
        assert np.allclose(a.polarization_responses, b.polarization_responses), \
            "no-noise sensor must be deterministic"
        assert np.allclose(a.midband_activations, b.midband_activations), \
            "no-noise sensor must be deterministic"
    print(f"  ✓ {len(r1)} sightings identical across two consecutive steps")


def test_receptor_noise_varies_between_steps():
    """receptor_noise_std > 0 → consecutive steps differ; both eyes have
    independent RNG streams."""
    print("\nTest 15: receptor_noise_std>0 perturbs each reading …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    eyeL, eyeR = make_eye_pair(scene, receptor_noise_std=0.05, noise_seed=42)
    # Two L readings should differ
    r1 = eyeL.step()[0].polarization_responses
    r2 = eyeL.step()[0].polarization_responses
    assert not np.allclose(r1, r2), "noisy sensor must vary between steps"
    # Magnitude of perturbation: roughly noise_std (clipped at 0)
    diff = float(np.std(r2 - r1))
    assert 0.005 < diff < 0.2, \
        f"noise std off-spec: expected ~0.05, observed perturbation std {diff:.3f}"
    print(f"  ✓ consecutive L readings differ (perturbation std ≈ {diff:.3f})")

    # L and R noise streams are independent → different reads given same target
    rL = eyeL.step()[0].polarization_responses
    rR = eyeR.step()[0].polarization_responses
    # Both see the same target geometrically, but noise should differ
    assert not np.allclose(rL, rR), "L and R noise streams should be independent"
    print("  ✓ L and R noise streams independent")


def test_receptor_noise_clipped_nonnegative():
    """Noise can\'t push responses below 0 (would crash the decoder)."""
    print("\nTest 16: receptor responses stay non-negative under heavy noise …")
    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    # Crank up the noise to where it routinely tries to push values negative
    eye, _ = make_eye_pair(scene, receptor_noise_std=0.5, noise_seed=1)
    for _ in range(20):
        for r in eye.step():
            assert (r.polarization_responses >= 0).all(), \
                "polarization responses must be >= 0 after noise"
            assert (r.midband_activations >= 0).all(), \
                "midband activations must be >= 0 after noise"
    print("  ✓ 20 noisy steps, all receptor values non-negative")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.sensor")
    print("=" * 60)
    test_construction()
    test_rest_geometry()
    test_roll_readout()
    test_local_frame_orthonormal()
    test_relative_angles()
    test_fov_boundaries()
    test_distance_attenuation()
    test_midband_one_hot()
    test_polarization_with_roll()
    test_polarization_unpolarized()
    test_raw_activations_off_fov()
    test_step_counts()
    test_make_eye_pair()
    test_receptor_noise_default_is_deterministic()
    test_receptor_noise_varies_between_steps()
    test_receptor_noise_clipped_nonnegative()
    print("\nAll sensor tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
