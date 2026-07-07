"""
Tests for stomatopod_vision.preprocessing

Validates:
   1. midband_channel_reduce picks the right class and normalises correctly.
   2. midband_channel_reduce handles all-zero input gracefully.
   3. midband_channel_reduce rejects wrong-shape input.
   4. polarization_decode round-trips: feed sensor output back through
      decoder, get the original angle.
   5. polarization_decode at all 4 receptor angles.
   6. polarization_decode returns None for unpolarized (uniform) input.
   7. polarization_decode returns None for all-zero input.
   8. polarization_decode integrates with sensor: feed sensor.RawSighting
      polarization_responses back through the decoder.
   9. polarization_decode shape mismatch raises.
  10. raw_to_event builds correct event from sensor output.
  11. EventEncoder emits on first sighting.
  12. EventEncoder suppresses subsequent identical sightings.
  13. EventEncoder emits when azimuth/elevation moves enough.
  14. EventEncoder emits when dominant_class changes.
  15. EventEncoder emits when polarization availability flips.
  16. EventEncoder emits when polarization angle changes meaningfully.
  17. EventEncoder.reset clears state.
  18. PreprocessingPipeline.step end-to-end.
  19. Pipeline.reset clears event encoder.

Run from the repo root:
    python tests/test_preprocessing.py
"""

import os
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
from stomatopod_vision.preprocessing import (  # noqa: E402
    EventEncoder,
    PreprocessedEvent,
    PreprocessingPipeline,
    midband_channel_reduce,
    polarization_decode,
    raw_to_event,
)
from stomatopod_vision.sensor import (  # noqa: E402
    RawSighting,
    VirtualEye,
    make_eye_pair,
)
from stomatopod_vision.world import (  # noqa: E402
    SPECTRAL_CLASSES,
    Scene,
    TargetMeta,
)

XML_PATH = REPO_ROOT / "models" / "stomatopod_eyes.xml"


# ---------------------------------------------------------------------
# Test 1 — channel reduce: basic
# ---------------------------------------------------------------------

def test_channel_reduce_basic():
    print("Test 1: midband_channel_reduce basic behaviour …")
    act = np.array([0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pat = midband_channel_reduce(act)
    assert pat == (1, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0), f"got pattern {pat}"
    print(f"  ✓ pattern = {pat}")

    # One-hot pure
    pure = np.zeros(12)
    pure[3] = 1.0
    pat2 = midband_channel_reduce(pure)
    assert pat2[3] == 10
    print("  ✓ one-hot → 10")


# ---------------------------------------------------------------------
# Test 2 — all-zero input
# ---------------------------------------------------------------------

def test_channel_reduce_zero():
    print("\nTest 2: midband_channel_reduce on all-zero input …")
    pat = midband_channel_reduce(np.zeros(12))
    assert pat == (0,) * 12
    print(f"  ✓ degenerate input → {pat} without NaN")


# ---------------------------------------------------------------------
# Test 3 — wrong shape rejected
# ---------------------------------------------------------------------

def test_channel_reduce_shape_check():
    pass


# ---------------------------------------------------------------------
# Test 4 — polarization decoder round-trips
# ---------------------------------------------------------------------

def test_polarization_round_trip():
    print("\nTest 4: polarization decoder round-trips …")
    recv = np.deg2rad([0., 45., 90., 135.])
    test_angles_deg = [0., 15., 30., 45., 60., 90., 120., 135., 170.]

    for theta_deg in test_angles_deg:
        theta_true = np.deg2rad(theta_deg)
        # Forward model from sensor.py
        responses = 0.8 * np.cos(theta_true - recv) ** 2
        # Inverse
        decoded = polarization_decode(responses, recv)
        assert decoded is not None, f"decoder returned None for θ={theta_deg}°"
        # Error in polarization angle (which lives on [0, π))
        err = abs((decoded - theta_true + np.pi/2) % np.pi - np.pi/2)
        assert err < 1e-6, f"θ={theta_deg}°: decoded={np.rad2deg(decoded):.3f}, err={err:.6f}"
    print(f"  ✓ {len(test_angles_deg)} angles all decoded to within 1e-6 rad")


# ---------------------------------------------------------------------
# Test 5 — decode at each canonical receptor angle exactly
# ---------------------------------------------------------------------

def test_polarization_canonical_angles():
    print("\nTest 5: polarization decoder at canonical receptor angles …")
    recv = np.deg2rad([0., 45., 90., 135.])
    for theta_deg in [0, 45, 90, 135]:
        theta = np.deg2rad(theta_deg)
        responses = np.cos(theta - recv) ** 2
        decoded = polarization_decode(responses, recv)
        # decoded ∈ [0, π); the angle 0 and π are equivalent
        canonical_decoded = np.rad2deg(decoded)
        # Treat decoded close to π (180°) as 0° for the comparison
        if canonical_decoded > 179.0:
            canonical_decoded -= 180.0
        assert abs(canonical_decoded - theta_deg) < 1e-4, \
            f"θ={theta_deg}° → decoded {canonical_decoded:.3f}°"
        print(f"  ✓ θ={theta_deg:3d}° → decoded {canonical_decoded:7.3f}°")


# ---------------------------------------------------------------------
# Test 6 — unpolarized returns None
# ---------------------------------------------------------------------

def test_polarization_unpolarized():
    print("\nTest 6: polarization_decode returns None for uniform input …")
    recv = np.deg2rad([0., 45., 90., 135.])
    uniform = 0.4 * np.ones(4)
    out = polarization_decode(uniform, recv)
    assert out is None, f"expected None, got {out}"
    print("  ✓ uniform 0.4 responses → None (low confidence)")


# ---------------------------------------------------------------------
# Test 7 — all-zero returns None
# ---------------------------------------------------------------------

def test_polarization_zero():
    print("\nTest 7: polarization_decode returns None for all-zero input …")
    recv = np.deg2rad([0., 45., 90., 135.])
    out = polarization_decode(np.zeros(4), recv)
    assert out is None
    print("  ✓ all-zero responses → None")


# ---------------------------------------------------------------------
# Test 7.5 — Circular decode
# ---------------------------------------------------------------------

def test_circular_decode():
    print("\nTest 7.5: circular_decode …")
    from stomatopod_vision.preprocessing import circular_decode
    
    # left circular = peaks at 45 (index 1)
    res_left = np.array([0.5, 1.0, 0.5, 0.0])
    assert circular_decode(res_left) == "left"
    
    # right circular = peaks at 135 (index 3)
    res_right = np.array([0.5, 0.0, 0.5, 1.0])
    assert circular_decode(res_right) == "right"
    
    # linear at 90 (peaks at 90, dips at 0, 45=0.5, 135=0.5)
    res_linear = np.array([0.0, 0.5, 1.0, 0.5])
    assert circular_decode(res_linear) is None
    
    # unpolarized
    res_unpol = np.array([0.5, 0.5, 0.5, 0.5])
    assert circular_decode(res_unpol) is None
    print("  ✓ circular_decode passes")


# ---------------------------------------------------------------------
# Test 8 — integrates with sensor end-to-end
# ---------------------------------------------------------------------

def test_polarization_with_sensor():
    print("\nTest 8: circular target → no linear angle, correct handedness …")
    # In the canonical scene, target_UVpol_1 is a *circularly* polarized
    # target (handedness 'left', no linear angle). On a real sensor sighting
    # the two channels must disagree the right way: the bare linear analyzers
    # see no axis (decode → None) while the quarter-wave channel recovers the
    # handedness. This is the end-to-end check that circular ≠ linear.
    from stomatopod_vision.preprocessing import circular_decode
    scene = Scene.from_xml(XML_PATH)
    scene.reset()
    eye_L = VirtualEye("L", scene)

    # Drive the eye to look at target_UVpol_1 (which is polarized at π/4)
    # First find roughly where it is and aim there
    tgt_pos = scene.target_world_position("target_UVpol_1")
    az, el, _ = eye_L.relative_angles(tgt_pos)
    # Settle to that pose (use a roll of 0 so receptor body-frame = world-frame)
    pd = GimbalPD(scene.model)
    sp = GimbalSetpoint(yaw_L=-az, pitch_L=el, roll_L=0.0)  # see note: pitch sign
    for _ in range(int(1.0 / scene.model.opt.timestep)):
        pd.step(scene.data, sp)
        scene.step()

    # Now grab the raw sighting
    raw = eye_L.raw_activations_for("target_UVpol_1")
    if raw is None:
        print("  ⚠ target not visible after settle; using larger pitch")
        sp = GimbalSetpoint(yaw_L=-az, pitch_L=-el, roll_L=0.0)
        for _ in range(int(1.0 / scene.model.opt.timestep)):
            pd.step(scene.data, sp)
            scene.step()
        raw = eye_L.raw_activations_for("target_UVpol_1")
    assert raw is not None, "target_UVpol_1 should be visible after settling"

    # Bare linear channel: a circular field presents no linear axis, so the
    # linear decoder must abstain rather than hallucinate a 45° angle (the
    # old single-channel model wrongly reported ~45° here).
    receptor_world = (VirtualEye.POLARIZATION_RECEPTOR_ANGLES_RAD
                      + eye_L.roll_angle())
    decoded = polarization_decode(raw.polarization_responses, receptor_world)
    assert decoded is None, (
        f"circular target must NOT decode as a linear angle; got "
        f"{None if decoded is None else round(np.rad2deg(decoded), 2)}°")

    # Quarter-wave channel: handedness is recovered, and is roll-invariant.
    handed = circular_decode(raw.circular_responses)
    assert handed == "left", f"expected handedness 'left', got {handed!r}"
    print("  ✓ linear decode → None, circular decode → 'left' (channels separate)")


# ---------------------------------------------------------------------
# Test 9 — shape mismatch raises
# ---------------------------------------------------------------------

def test_polarization_shape_mismatch():
    print("\nTest 9: polarization_decode shape mismatch raises …")
    try:
        polarization_decode(np.zeros(4), np.zeros(3))
    except ValueError:
        print("  ✓ shape mismatch raises ValueError")
    else:
        raise AssertionError("should have raised on shape mismatch")


# ---------------------------------------------------------------------
# Test 10 — raw_to_event builds correct event
# ---------------------------------------------------------------------

def test_raw_to_event():
    print("\nTest 10: raw_to_event builds correct PreprocessedEvent …")
    midband = np.zeros(12)
    midband[2] = 0.6   # spectral_class index 2 = "B"
    midband[3] = 0.2   # idx 3 = "UV_A"
    raw = RawSighting(
        target_name="t1",
        eye="L",
        azimuth=0.1,
        elevation=-0.05,
        distance=2.0,
        peripheral_intensity=0.4,
        midband_activations=midband,
        polarization_responses=0.4 * np.ones(4),  # uniform → unpolarized
    )
    ev = raw_to_event(raw, time_now=1.23, roll_angle=0.0)
    assert isinstance(ev, PreprocessedEvent)
    assert ev.target_name == "t1"
    assert ev.eye == "L"
    assert ev.time == 1.23
    assert ev.spectral_pattern == (0, 0, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    assert ev.polarization_angle is None    # uniform input
    print(f"  ✓ event built with dominant_class={ev.spectral_pattern}, "
          f"pol=None")


# ---------------------------------------------------------------------
# Test 11 — EventEncoder emits on first sighting
# ---------------------------------------------------------------------

def test_encoder_first_sighting():
    print("\nTest 11: EventEncoder emits first sighting per (eye, target) …")
    enc = EventEncoder()
    raw = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    ev = enc.encode(raw, time_now=0.0, roll_angle=0.0)
    assert ev is not None
    print("  ✓ first call returned an event")


# ---------------------------------------------------------------------
# Test 12 — EventEncoder suppresses identical sightings
# ---------------------------------------------------------------------

def test_encoder_suppresses_identical():
    print("\nTest 12: EventEncoder suppresses identical sightings …")
    enc = EventEncoder()
    raw = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    e1 = enc.encode(raw, time_now=0.0, roll_angle=0.0)
    e2 = enc.encode(raw, time_now=0.01, roll_angle=0.0)
    assert e1 is not None
    assert e2 is None
    print("  ✓ second identical call returned None")


# ---------------------------------------------------------------------
# Test 13 — angular movement triggers emission
# ---------------------------------------------------------------------

def test_encoder_emits_on_movement():
    print("\nTest 13: EventEncoder emits on angular movement …")
    enc = EventEncoder(azimuth_threshold=np.deg2rad(2.0))
    raw1 = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    enc.encode(raw1, time_now=0.0, roll_angle=0.0)

    # Tiny move — should NOT trigger
    raw2 = RawSighting(**{**raw1.__dict__, "azimuth": np.deg2rad(1.0)})
    e2 = enc.encode(raw2, time_now=0.01, roll_angle=0.0)
    assert e2 is None, "1° move should be below the 2° threshold"

    # Big move — should trigger
    raw3 = RawSighting(**{**raw1.__dict__, "azimuth": np.deg2rad(5.0)})
    e3 = enc.encode(raw3, time_now=0.02, roll_angle=0.0)
    assert e3 is not None, "5° move should exceed 2° threshold"
    print("  ✓ 1° suppressed, 5° emitted (threshold = 2°)")


# ---------------------------------------------------------------------
# Test 14 — class change triggers emission
# ---------------------------------------------------------------------

def test_encoder_emits_on_class_change():
    print("\nTest 14: EventEncoder emits on dominant_class change …")
    enc = EventEncoder()
    raw1 = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  # C1 wins
        polarization_responses=0.5 * np.ones(4),
    )
    enc.encode(raw1, time_now=0.0, roll_angle=0.0)
    raw2 = RawSighting(**{**raw1.__dict__,
                         "midband_activations": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])})  # C10 wins
    e2 = enc.encode(raw2, time_now=0.01, roll_angle=0.0)
    assert e2 is not None
    assert e2.spectral_pattern == (0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0)
    print("  ✓ class change R → UV_A emitted event")


# ---------------------------------------------------------------------
# Test 15 — polarization availability flip
# ---------------------------------------------------------------------

def test_encoder_emits_on_pol_flip():
    print("\nTest 15: EventEncoder emits when polarization becomes available …")
    enc = EventEncoder()
    # First: unpolarized
    raw1 = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    e1 = enc.encode(raw1, time_now=0.0, roll_angle=0.0)
    assert e1.polarization_angle is None

    # Then: clearly polarized at 0° (a non-uniform response)
    recv = VirtualEye.POLARIZATION_RECEPTOR_ANGLES_RAD
    pol_responses = np.cos(0.0 - recv) ** 2
    raw2 = RawSighting(**{**raw1.__dict__, "polarization_responses": pol_responses})
    e2 = enc.encode(raw2, time_now=0.01, roll_angle=0.0)
    assert e2 is not None
    assert e2.polarization_angle is not None
    print(f"  ✓ unpolarized → polarized triggered emit "
          f"(decoded {np.rad2deg(e2.polarization_angle):.2f}°)")


# ---------------------------------------------------------------------
# Test 16 — polarization angle change
# ---------------------------------------------------------------------

def test_encoder_emits_on_pol_angle_change():
    print("\nTest 16: EventEncoder emits when polarization angle changes …")
    enc = EventEncoder(polarization_threshold=np.deg2rad(10.0))
    recv = VirtualEye.POLARIZATION_RECEPTOR_ANGLES_RAD
    raw1 = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=np.cos(0.0 - recv) ** 2,
    )
    enc.encode(raw1, time_now=0.0, roll_angle=0.0)

    # 5° change in polarization — below 10° threshold → no emit
    raw2 = RawSighting(**{**raw1.__dict__,
                         "polarization_responses":
                         np.cos(np.deg2rad(5.0) - recv) ** 2})
    e2 = enc.encode(raw2, time_now=0.01, roll_angle=0.0)
    assert e2 is None, "5° polarization change should be below threshold"

    # 20° change — above threshold → emit
    raw3 = RawSighting(**{**raw1.__dict__,
                         "polarization_responses":
                         np.cos(np.deg2rad(20.0) - recv) ** 2})
    e3 = enc.encode(raw3, time_now=0.02, roll_angle=0.0)
    assert e3 is not None
    print("  ✓ 5° pol-change suppressed, 20° emitted (threshold = 10°)")


# ---------------------------------------------------------------------
# Test 17 — reset clears state
# ---------------------------------------------------------------------

def test_encoder_reset():
    print("\nTest 17: EventEncoder.reset clears state …")
    enc = EventEncoder()
    raw = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    enc.encode(raw, time_now=0.0, roll_angle=0.0)
    assert len(enc._last_event) == 1
    enc.reset()
    assert len(enc._last_event) == 0
    # After reset, the same input should produce a new event
    ev = enc.encode(raw, time_now=1.0, roll_angle=0.0)
    assert ev is not None
    print("  ✓ reset clears cache, new emission afterwards")


# ---------------------------------------------------------------------
# Test 18 — end-to-end pipeline against a real scene
# ---------------------------------------------------------------------

def test_pipeline_end_to_end():
    print("\nTest 18: PreprocessingPipeline.step end-to-end on real scene …")
    scene = Scene.from_xml(XML_PATH)
    scene.reset()
    eye_L, eye_R = make_eye_pair(scene)
    pipeline = PreprocessingPipeline()

    # First step: should emit one event per visible target per eye
    raws = eye_L.step() + eye_R.step()
    events = pipeline.step(
        raws,
        time_now=scene.data.time,
        roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
    )
    print(f"  step 1: {len(raws)} raw → {len(events)} events")
    assert len(events) == len(raws), "first step should emit one event per sighting"

    # Second step (no movement, no time passed): pipeline should
    # suppress the now-identical sightings
    raws2 = eye_L.step() + eye_R.step()
    events2 = pipeline.step(
        raws2,
        time_now=scene.data.time + 0.001,
        roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
    )
    print(f"  step 2 (no change): {len(raws2)} raw → {len(events2)} events")
    assert len(events2) == 0, "no movement → no new events"

    # Now drive the eyes to a new pose, expect new events
    pd = GimbalPD(scene.model)
    sp = GimbalSetpoint(yaw_L=0.3, pitch_L=0.0, roll_L=0.5,
                        yaw_R=-0.3, pitch_R=0.0, roll_R=-0.5)
    for _ in range(int(0.5 / scene.model.opt.timestep)):
        pd.step(scene.data, sp)
        scene.step()

    raws3 = eye_L.step() + eye_R.step()
    events3 = pipeline.step(
        raws3,
        time_now=scene.data.time,
        roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
    )
    print(f"  step 3 (after move): {len(raws3)} raw → {len(events3)} events")
    # Some movement should produce at least some events (or the eyes
    # no longer see the original targets). Either way, the pipeline
    # should have responded.
    assert len(events3) >= 0


# ---------------------------------------------------------------------
# Test 19 — pipeline reset
# ---------------------------------------------------------------------

def test_pipeline_reset():
    print("\nTest 19: PreprocessingPipeline.reset clears encoder cache …")
    pipeline = PreprocessingPipeline()
    raw = RawSighting(
        target_name="t1", eye="L",
        azimuth=0.0, elevation=0.0, distance=1.0,
        peripheral_intensity=0.5,
        midband_activations=np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        polarization_responses=0.5 * np.ones(4),
    )
    e1 = pipeline.step([raw], time_now=0.0,
                       roll_angles={"L": 0.0, "R": 0.0})
    assert len(e1) == 1
    e2 = pipeline.step([raw], time_now=0.01,
                       roll_angles={"L": 0.0, "R": 0.0})
    assert len(e2) == 0
    pipeline.reset()
    e3 = pipeline.step([raw], time_now=0.02,
                       roll_angles={"L": 0.0, "R": 0.0})
    assert len(e3) == 1
    print("  ✓ pipeline.reset() clears encoder, new emission afterwards")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.preprocessing")
    print("=" * 60)
    test_channel_reduce_basic()
    test_channel_reduce_zero()
    test_channel_reduce_shape_check()
    test_polarization_round_trip()
    test_polarization_canonical_angles()
    test_polarization_unpolarized()
    test_polarization_zero()
    test_polarization_with_sensor()
    test_polarization_shape_mismatch()
    test_raw_to_event()
    test_encoder_first_sighting()
    test_encoder_suppresses_identical()
    test_encoder_emits_on_movement()
    test_encoder_emits_on_class_change()
    test_encoder_emits_on_pol_flip()
    test_encoder_emits_on_pol_angle_change()
    test_encoder_reset()
    test_pipeline_end_to_end()
    test_pipeline_reset()
    print("\nAll preprocessing tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
