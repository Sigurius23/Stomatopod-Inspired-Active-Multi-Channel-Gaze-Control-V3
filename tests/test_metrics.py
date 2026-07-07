"""
Tests for stomatopod_vision.metrics

Validates:
   1. EventLog construction + log_* helpers + reset.
   2. populate_targets_from_scene copies the right fields.
   3. first_fov_entry returns min entry time or None.
   4. coverage: perfect identification → 1.0.
   5. coverage: missed targets → < 1.0.
   6. coverage: wrong class doesn't count.
   7. coverage: outside identification window doesn't count.
   8. coverage: empty interesting set → 1.0 (vacuous).
   9. bandwidth: B1 = fixed-rate stream regardless of events.
  10. bandwidth: B2/B3 = event volume / duration.
  11. bandwidth: unknown baseline raises.
  12. bandwidth: zero-duration B2/B3 returns 0.
  13. polarization_accuracy: perfect → 1.0.
  14. polarization_accuracy: doubled-angle wrap (179° ≈ 1°).
  15. polarization_accuracy: None events don't count as correct.
  16. polarization_accuracy: empty polarized set → 1.0.
  17. median_response_latency_s: hand-computed median.
  18. median_response_latency_s: censoring at duration_s.
  19. MetricsReport.to_dict / save_json / load_json round-trip.
  20. compute_all bundles everything correctly.
  21. End-to-end: real scene → log → metrics.

Run from the repo root:
    python tests/test_metrics.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision.gimbal_control import GimbalPD, GimbalSetpoint  # noqa: E402
from stomatopod_vision.metrics import (  # noqa: E402
    EventLog,
    MetricsReport,
    bandwidth_bytes_per_second,
    circular_polarization_accuracy,
    compute_all,
    coverage,
    median_response_latency_s,
    polarization_accuracy,
)
from stomatopod_vision.preprocessing import (  # noqa: E402
    PreprocessedEvent,
    PreprocessingPipeline,
)
from stomatopod_vision.sensor import (  # noqa: E402
    RawSighting,
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
    time: float,
    *,
    dominant_class: str = "C10",
    polarization_angle: float | None = None,
    circular_handedness: str | None = None,
    eye: str = "L",
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: float = 1.0,
) -> PreprocessedEvent:
    """Convenience constructor for tests."""
    from stomatopod_vision.world import SPECTRAL_CLASSES
    pat = [0] * 12
    if dominant_class in SPECTRAL_CLASSES:
        pat[SPECTRAL_CLASSES.index(dominant_class)] = 10
        
    return PreprocessedEvent(
        time=time,
        eye=eye,
        target_name=target,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        spectral_pattern=tuple(pat),
        polarization_angle=polarization_angle,
        circular_handedness=circular_handedness,
    )


# ---------------------------------------------------------------------
# Test 1 — EventLog basics
# ---------------------------------------------------------------------

def test_eventlog_basics():
    print("Test 1: EventLog construction + log_* helpers + reset …")
    log = EventLog()
    assert log.raw_sightings == []
    assert log.preprocessed_events == []
    assert log.duration_s == 0.0

    log.log_event(_event("t1", time=0.0))
    log.log_event(_event("t2", time=1.0))
    assert len(log.preprocessed_events) == 2

    log.log_target_fov("t1", t_enter=0.0, t_exit=0.5)
    log.log_target_fov("t1", t_enter=1.0, t_exit=2.0)
    assert len(log.target_in_fov_intervals["t1"]) == 2

    log.duration_s = 5.0
    log.reset()
    assert log.preprocessed_events == []
    assert log.target_in_fov_intervals == {}
    assert log.duration_s == 0.0
    print("  ✓ log helpers + reset work")


# ---------------------------------------------------------------------
# Test 2 — populate_targets_from_scene
# ---------------------------------------------------------------------

def test_populate_from_scene():
    print("\nTest 2: populate_targets_from_scene copies metadata …")
    scene = Scene.from_xml(XML_PATH)
    log = EventLog()
    log.populate_targets_from_scene(scene)

    # All 6 default targets should have their true class recorded
    assert len(log.target_true_class) == 6
    assert log.target_true_class["target_R_1"] == "C1"
    assert log.target_true_class["target_UVpol_1"] == "C10"

    # Only target_UVpol_1 is interesting and polarized
    assert log.interesting_targets == {"target_UVpol_1"}
    # It is circularly polarized, so log.polarized_targets should be empty
    assert list(log.polarized_targets.keys()) == []
    print("  ✓ all 6 classes recorded; interesting/polarized filtered correctly")


# ---------------------------------------------------------------------
# Test 3 — first_fov_entry
# ---------------------------------------------------------------------

def test_first_fov_entry():
    print("\nTest 3: first_fov_entry returns min entry time …")
    log = EventLog()
    log.log_target_fov("t1", 3.0, 4.0)
    log.log_target_fov("t1", 1.0, 2.0)
    log.log_target_fov("t1", 5.0, 6.0)
    assert log.first_fov_entry("t1") == 1.0
    assert log.first_fov_entry("nonexistent") is None
    print("  ✓ returns min entry / None correctly")


# ---------------------------------------------------------------------
# Test 4 — coverage perfect
# ---------------------------------------------------------------------

def test_coverage_perfect():
    print("\nTest 4: coverage = 1.0 when all interesting targets identified …")
    log = EventLog()
    log.interesting_targets = {"t1", "t2"}
    log.target_true_class = {"t1": "C10", "t2": "C1"}
    log.log_target_fov("t1", 0.0, 1.0)
    log.log_target_fov("t2", 0.0, 1.0)
    log.log_event(_event("t1", 0.1, dominant_class="C10"))
    log.log_event(_event("t2", 0.2, dominant_class="C1"))

    c = coverage(log)
    assert c == 1.0, f"expected 1.0, got {c}"
    print(f"  ✓ coverage = {c}")


# ---------------------------------------------------------------------
# Test 5 — coverage missing
# ---------------------------------------------------------------------

def test_coverage_partial():
    print("\nTest 5: coverage < 1.0 when some targets missed …")
    log = EventLog()
    log.interesting_targets = {"t1", "t2", "t3", "t4"}
    log.target_true_class = {"t1": "C1", "t2": "C2", "t3": "C3", "t4": "C4"}
    for t in ["t1", "t2", "t3", "t4"]:
        log.log_target_fov(t, 0.0, 1.0)
    # Only t1 and t2 produce identifying events
    log.log_event(_event("t1", 0.1, dominant_class="C1"))
    log.log_event(_event("t2", 0.2, dominant_class="C2"))

    c = coverage(log)
    assert abs(c - 0.5) < 1e-9, f"expected 0.5, got {c}"
    print(f"  ✓ coverage = {c} (2 of 4)")


# ---------------------------------------------------------------------
# Test 6 — coverage with wrong class
# ---------------------------------------------------------------------

def test_coverage_wrong_class():
    print("\nTest 6: coverage doesn't count events with wrong dominant_class …")
    log = EventLog()
    log.interesting_targets = {"t1"}
    log.target_true_class = {"t1": "C1"}
    log.log_target_fov("t1", 0.0, 10.0)
    log.log_event(_event("t1", time=1.0, dominant_class="C12"))  # wrong class!
    assert coverage(log) == 0.0
    print("  ✓ wrong class → 0.0")


# ---------------------------------------------------------------------
# Test 7 — coverage outside window
# ---------------------------------------------------------------------

def test_coverage_outside_window():
    print("\nTest 7: coverage doesn't count events outside identification window …")
    log = EventLog()
    log.interesting_targets = {"t1"}
    log.target_true_class = {"t1": "C10"}
    log.log_target_fov("t1", 0.0, 100.0)
    # Event arrives 2 seconds after FoV entry (window default = 0.5s)
    log.log_event(_event("t1", 2.0, dominant_class="C10"))
    assert coverage(log, identification_window_s=0.5) == 0.0
    # But with a wider window, it counts
    assert coverage(log, identification_window_s=5.0) == 1.0
    print("  ✓ window enforced strictly")


# ---------------------------------------------------------------------
# Test 8 — coverage vacuous
# ---------------------------------------------------------------------

def test_coverage_vacuous():
    print("\nTest 8: coverage = 1.0 when no interesting targets exist …")
    log = EventLog()  # empty interesting_targets
    assert coverage(log) == 1.0
    print("  ✓ vacuous coverage = 1.0")


# ---------------------------------------------------------------------
# Test 9 — bandwidth B1
# ---------------------------------------------------------------------

def test_bandwidth_b1():
    print("\nTest 9: bandwidth B1 = fixed-rate stream …")
    log = EventLog()
    log.duration_s = 10.0
    # Even with NO events, B1 streams every step:
    bps = bandwidth_bytes_per_second(log, "B1", raw_sample_rate_hz=500.0)
    # 2 eyes × RAW_BYTES_PER_SAMPLE × 500 Hz (default values; introspected from struct)
    expected = 2 * log.raw_stream_bytes_per_sample * 500.0
    assert abs(bps - expected) < 1e-9, f"expected {expected}, got {bps}"
    print(f"  ✓ B1 = {bps:.0f} bytes/s "
          f"(2 eyes × {log.raw_stream_bytes_per_sample} bytes × 500 Hz)")


# ---------------------------------------------------------------------
# Test 10 — bandwidth B2/B3
# ---------------------------------------------------------------------

def test_bandwidth_b2_b3():
    print("\nTest 10: bandwidth B2/B3 = event volume / duration …")
    log = EventLog()
    log.duration_s = 10.0
    for t in np.linspace(0, 10, 50, endpoint=False):
        log.log_event(_event("t1", time=t))

    bps_b2 = bandwidth_bytes_per_second(log, "B2")
    bps_b3 = bandwidth_bytes_per_second(log, "B3")
    expected = log.event_bytes_per_event * 50 / 10.0
    assert abs(bps_b2 - expected) < 1e-9
    assert abs(bps_b3 - expected) < 1e-9
    print(f"  ✓ B2 = B3 = {bps_b2:.0f} bytes/s "
          f"({log.event_bytes_per_event} × 50 / 10)")

    # And B2 << B1 (the whole point)
    bps_b1 = bandwidth_bytes_per_second(log, "B1")
    assert bps_b2 < 0.1 * bps_b1
    print(f"  ✓ B2 ({bps_b2:.0f}) << B1 ({bps_b1:.0f}) — sparsity benefit visible")


# ---------------------------------------------------------------------
# Test 11 — bandwidth unknown baseline raises
# ---------------------------------------------------------------------

def test_bandwidth_unknown():
    print("\nTest 11: bandwidth_bytes_per_second rejects unknown baseline …")
    log = EventLog()
    log.duration_s = 1.0
    try:
        bandwidth_bytes_per_second(log, "B7")
    except ValueError:
        print("  ✓ 'B7' raises ValueError")
    else:
        raise AssertionError("should have rejected baseline='B7'")


# ---------------------------------------------------------------------
# Test 12 — bandwidth zero-duration
# ---------------------------------------------------------------------

def test_bandwidth_zero_duration():
    print("\nTest 12: B2/B3 with duration_s=0 returns 0 (no /0) …")
    log = EventLog()
    log.duration_s = 0.0
    log.log_event(_event("t1", 0.0))
    assert bandwidth_bytes_per_second(log, "B2") == 0.0
    assert bandwidth_bytes_per_second(log, "B3") == 0.0
    # But B1 is still fine (independent of duration)
    assert bandwidth_bytes_per_second(log, "B1") > 0
    print("  ✓ zero-duration safe for all baselines")


# ---------------------------------------------------------------------
# Test 13 — polarization accuracy perfect
# ---------------------------------------------------------------------

def test_polarization_perfect():
    print("\nTest 13: polarization_accuracy = 1.0 when all decoded correctly …")
    log = EventLog()
    log.polarized_targets = {"t1": np.deg2rad(45.0), "t2": np.deg2rad(90.0)}
    log.log_event(_event("t1", 0.0,
                         polarization_angle=np.deg2rad(45.0 + 1.0)))  # within 15°
    log.log_event(_event("t2", 0.0,
                         polarization_angle=np.deg2rad(90.0 - 1.0)))
    assert polarization_accuracy(log) == 1.0
    print("  ✓ perfect polarization = 1.0")


# ---------------------------------------------------------------------
# Test 14 — polarization doubled-angle wrap
# ---------------------------------------------------------------------

def test_polarization_wrap():
    print("\nTest 14: polarization doubled-angle wrap (179° ≈ 1°) …")
    log = EventLog()
    log.polarized_targets = {"t1": np.deg2rad(1.0)}
    # Decoder reported 179° — physically equivalent to 1° because pol is mod π
    log.log_event(_event("t1", 0.0, polarization_angle=np.deg2rad(179.0)))
    assert polarization_accuracy(log, tolerance_rad=np.deg2rad(5.0)) == 1.0
    print("  ✓ 179° vs 1° treated as 2° apart, well within 5° tolerance")


# ---------------------------------------------------------------------
# Test 15 — None events don't count
# ---------------------------------------------------------------------

def test_polarization_none_events():
    print("\nTest 15: events with polarization_angle=None don't count …")
    log = EventLog()
    log.polarized_targets = {"t1": np.deg2rad(45.0)}
    log.log_event(_event("t1", 0.0, polarization_angle=None))
    log.log_event(_event("t1", 0.1, polarization_angle=None))
    assert polarization_accuracy(log) == 0.0
    print("  ✓ all-None for a target → 0.0")


# ---------------------------------------------------------------------
# Test 16 — vacuous polarization
# ---------------------------------------------------------------------

def test_polarization_vacuous():
    print("\nTest 16: polarization_accuracy = 1.0 when no polarized targets …")
    log = EventLog()  # empty polarized_targets
    assert polarization_accuracy(log) == 1.0
    print("  ✓ vacuous polarization = 1.0")


# ---------------------------------------------------------------------
# Test 16.5 — Circular polarization
# ---------------------------------------------------------------------

def test_circular_polarization():
    print("\nTest 16.5: circular polarization accuracy …")
    log = EventLog()
    
    log.circular_targets = {"crab": "left", "fish": "right"}
    log.target_in_fov_intervals = {
        "crab": [(0.0, 10.0)],
        "fish": [(1.0, 10.0)],
    }
    
    # Correctly ID crab at 0.1, incorrectly ID fish at 1.2
    log.log_event(_event("crab", 0.1, circular_handedness="left"))
    log.log_event(_event("fish", 1.2, circular_handedness="left"))  # wrong
    
    assert circular_polarization_accuracy(log) == 0.5
    
    # Now fix fish
    log.preprocessed_events.pop()
    log.log_event(_event("fish", 1.2, circular_handedness="right"))
    assert circular_polarization_accuracy(log) == 1.0
    print("  ✓ circular polarization metric works")


# ---------------------------------------------------------------------
# Test 17 — latency is median of first-identifications
# ---------------------------------------------------------------------

def test_latency_hand_computed():
    print("\nTest 17: latency = median over targets …")
    log = EventLog()
    log.duration_s = 100.0
    log.interesting_targets = {"t1", "t2", "t3"}
    log.target_true_class = {"t1": "C10", "t2": "C1", "t3": "C4"}
    # t1 enters at 0.0, identified at 0.5  → latency 0.5
    log.log_target_fov("t1", 0.0, 10.0)
    log.log_event(_event("t1", 0.5, dominant_class="C10"))
    # t2 enters at 1.0, identified at 1.3  → latency 0.3
    log.log_target_fov("t2", 1.0, 10.0)
    log.log_event(_event("t2", 1.3, dominant_class="C1"))
    # t3 enters at 2.0, identified at 2.2  → latency 0.2
    log.log_target_fov("t3", 2.0, 10.0)
    log.log_event(_event("t3", 2.2, dominant_class="C4"))

    latencies = [0.5, 0.3, 0.2]
    expected_median = float(np.median(latencies))
    got = median_response_latency_s(log)
    assert abs(got - expected_median) < 1e-9, f"expected {expected_median}, got {got}"
    print(f"  ✓ median latency = {got:.3f}s (expected {expected_median:.3f}s)")


# ---------------------------------------------------------------------
# Test 18 — latency censoring
# ---------------------------------------------------------------------

def test_latency_censoring():
    print("\nTest 18: never-identified targets censored at duration_s …")
    log = EventLog()
    log.duration_s = 10.0
    log.interesting_targets = {"t1", "t2"}
    log.target_true_class = {"t1": "C10", "t2": "C1"}
    log.log_target_fov("t1", 0.0, 1.0)
    log.log_target_fov("t2", 0.0, 1.0)
    # Only t1 identified
    log.log_event(_event("t1", 0.5, dominant_class="C10"))
    # t2 censored at 10.0
    # Latencies = [0.5, 10.0], median = 5.25
    got = median_response_latency_s(log)
    assert abs(got - 5.25) < 1e-9
    print(f"  ✓ median = {got:.3f}s (mix of identified + censored)")

    # Targets never even in FoV are also censored
    log2 = EventLog()
    log2.duration_s = 7.0
    log2.interesting_targets = {"t1"}
    # No fov log → first_fov_entry returns None → censored
    got2 = median_response_latency_s(log2)
    assert got2 == 7.0
    print(f"  ✓ never-in-FoV → censored at duration ({got2:.1f}s)")


# ---------------------------------------------------------------------
# Test 19 — MetricsReport JSON round-trip
# ---------------------------------------------------------------------

def test_report_json_round_trip():
    print("\nTest 19: MetricsReport JSON round-trip …")
    r = MetricsReport(
        baseline="B3",
        coverage=0.83,
        bandwidth_bps=512.5,
        polarization_accuracy=0.66,
        circular_polarization_accuracy=1.0,
        median_latency_s=1.25,
    )
    d = r.to_dict()
    assert d["baseline"] == "B3"
    assert d["coverage"] == 0.83

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        r.save_json(path)
        r2 = MetricsReport.load_json(path)
        assert r == r2
    print("  ✓ save + load round-trip preserves all fields")


# ---------------------------------------------------------------------
# Test 20 — compute_all
# ---------------------------------------------------------------------

def test_compute_all():
    print("\nTest 20: compute_all bundles all four metrics …")
    log = EventLog()
    log.duration_s = 1.0
    log.interesting_targets = {"t1"}
    log.target_true_class = {"t1": "C10"}
    log.polarized_targets = {"t1": 0.0}
    log.log_target_fov("t1", 0.0, 1.0)
    log.log_event(_event("t1", 0.1, dominant_class="C10", polarization_angle=0.0))

    rep = compute_all(log, baseline="B3")
    assert rep.baseline == "B3"
    assert rep.coverage == 1.0
    assert rep.polarization_accuracy == 1.0
    assert rep.bandwidth_bps > 0
    assert rep.median_latency_s == 0.1
    print(f"  ✓ {rep}")


# ---------------------------------------------------------------------
# Test 21 — end-to-end with real scene
# ---------------------------------------------------------------------

def test_end_to_end():
    print("\nTest 21: end-to-end real scene → log → metrics …")
    scene = Scene.from_xml(XML_PATH)
    scene.reset()
    eye_L, eye_R = make_eye_pair(scene)
    pipeline = PreprocessingPipeline()
    log = EventLog()
    log.populate_targets_from_scene(scene)

    # Helper to track FoV entry/exit per target
    in_fov_since: dict[str, float] = {}

    def update_fov_intervals(now: float):
        currently_visible = set()
        for raw in eye_L.step() + eye_R.step():
            currently_visible.add(raw.target_name)
            if raw.target_name not in in_fov_since:
                in_fov_since[raw.target_name] = now
        # Anyone who *left* the FoV this step
        for name in list(in_fov_since.keys()):
            if name not in currently_visible:
                log.log_target_fov(name, in_fov_since.pop(name), now)

    # Run for 2 seconds with both eyes static (B2-style baseline)
    pd = GimbalPD(scene.model)
    sp = GimbalSetpoint()  # eyes forward
    duration = 2.0
    n_steps = int(duration / scene.model.opt.timestep)
    for _ in range(n_steps):
        pd.step(scene.data, sp)
        scene.step()

        # Collect raw sightings + run preprocessing
        raws = eye_L.step() + eye_R.step()
        events = pipeline.step(
            raws,
            time_now=scene.data.time,
            roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
        )
        for ev in events:
            log.log_event(ev)
        update_fov_intervals(scene.data.time)

    # Close any still-open FoV intervals
    for name, t_enter in in_fov_since.items():
        log.log_target_fov(name, t_enter, scene.data.time)
    log.duration_s = float(scene.data.time)

    print(f"  duration {log.duration_s:.2f}s, "
          f"{len(log.preprocessed_events)} events emitted, "
          f"{len(log.target_in_fov_intervals)} targets seen")

    # Compute all metrics — should produce valid scalars
    rep_b1 = compute_all(log, "B1")
    rep_b2 = compute_all(log, "B2")
    rep_b3 = compute_all(log, "B3")
    print(f"  B1: coverage={rep_b1.coverage:.2f}  "
          f"bw={rep_b1.bandwidth_bps:.0f}  "
          f"pol={rep_b1.polarization_accuracy:.2f}  "
          f"lat={rep_b1.median_latency_s:.2f}s")
    print(f"  B2: coverage={rep_b2.coverage:.2f}  "
          f"bw={rep_b2.bandwidth_bps:.0f}  "
          f"pol={rep_b2.polarization_accuracy:.2f}  "
          f"lat={rep_b2.median_latency_s:.2f}s")
    print(f"  B3: coverage={rep_b3.coverage:.2f}  "
          f"bw={rep_b3.bandwidth_bps:.0f}  "
          f"pol={rep_b3.polarization_accuracy:.2f}  "
          f"lat={rep_b3.median_latency_s:.2f}s")

    # Sanity invariants
    assert 0.0 <= rep_b1.coverage <= 1.0
    assert 0.0 <= rep_b2.coverage <= 1.0
    assert rep_b1.bandwidth_bps > rep_b2.bandwidth_bps   # B1 should be wasteful
    print("  ✓ all reports valid; B1 bandwidth >> B2 (the headline story)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.metrics")
    print("=" * 60)
    test_eventlog_basics()
    test_populate_from_scene()
    test_first_fov_entry()
    test_coverage_perfect()
    test_coverage_partial()
    test_coverage_wrong_class()
    test_coverage_outside_window()
    test_coverage_vacuous()
    test_bandwidth_b1()
    test_bandwidth_b2_b3()
    test_bandwidth_unknown()
    test_bandwidth_zero_duration()
    test_polarization_perfect()
    test_polarization_wrap()
    test_polarization_none_events()
    test_polarization_vacuous()
    test_circular_polarization()
    test_latency_hand_computed()
    test_latency_censoring()
    test_report_json_round_trip()
    test_compute_all()
    test_end_to_end()
    print("\nAll metrics tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
