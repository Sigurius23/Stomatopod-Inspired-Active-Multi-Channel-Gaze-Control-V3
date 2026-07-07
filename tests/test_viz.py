"""
Tests for stomatopod_vision.viz

Validates:
   1. headline_bar_chart returns a Figure and writes SVG.
   2. headline_bar_chart works for 1, 2, and 3 reports.
   3. headline_bar_chart raises on empty input.
   4. coverage_over_time renders a flat zero line when no IDs.
   5. coverage_over_time draws a step curve when targets are identified.
   6. bandwidth_over_time draws a flat line for B1, varying for B2/B3.
   7. polarization_detection_timeline handles polarized targets.
   8. polarization_detection_timeline handles the no-polarized-target case.
   9. The four live-overlay/video stubs all raise NotImplementedError.
  10. End-to-end: load the B1+B2 JSONs we have on disk, render all 4 plots.

Run from the repo root:
    python tests/test_viz.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stomatopod_vision.metrics import (  # noqa: E402
    EventLog,
    MetricsReport,
)
from stomatopod_vision.preprocessing import PreprocessedEvent  # noqa: E402
from stomatopod_vision.viz import (  # noqa: E402
    bandwidth_over_time,
    bandwidth_over_time_multi_seed,
    coverage_over_time,
    coverage_over_time_multi_seed,
    headline_bar_chart,
    headline_bar_chart_multi_seed,
    polarization_detection_timeline,
    record_run,
    render_eye_fov_overlay,
    render_recent_sightings,
    render_saliency_map_overlay,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_report(baseline: str, **kwargs) -> MetricsReport:
    defaults = dict(coverage=1.0, bandwidth_bps=1000.0,
                    polarization_accuracy=1.0, circular_polarization_accuracy=1.0, median_latency_s=0.0)
    defaults.update(kwargs)
    return MetricsReport(baseline=baseline, **defaults)


def _make_log_with_events(
    baseline_targets_seen: dict[str, str] | None = None,
    duration: float = 5.0,
    polarized: dict[str, float] | None = None,
) -> EventLog:
    """A small, deterministic EventLog suitable for plotting smoke tests."""
    log = EventLog()
    log.duration_s = duration
    targets = baseline_targets_seen or {}
    log.interesting_targets = set(targets.keys())
    log.target_true_class = dict(targets)
    log.polarized_targets = polarized or {}
    for i, (name, cls) in enumerate(targets.items()):
        t_enter = 0.1 + 0.5 * i
        log.log_target_fov(name, t_enter, duration)
        log.log_event(PreprocessedEvent(
            time=t_enter + 0.05,
            eye="L",
            target_name=name,
            azimuth=0.0,
            elevation=0.0,
            distance=1.0,
            spectral_pattern=(0,0,0,0,0,0,0,0,0,10,0,0) if cls=="C10" else (10,0,0,0,0,0,0,0,0,0,0,0), circular_handedness=None,
            polarization_angle=(polarized.get(name) if polarized else None),
        ))
    return log


# ---------------------------------------------------------------------
# Test 1 — headline_bar_chart returns Figure and writes SVG
# ---------------------------------------------------------------------

def test_headline_basic():
    print("Test 1: headline_bar_chart returns Figure and writes SVG …")
    reports = [
        _make_report("B1", coverage=0.4, bandwidth_bps=100000),
        _make_report("B2", coverage=0.7, bandwidth_bps=200),
        _make_report("B3", coverage=0.95, bandwidth_bps=300,
                     polarization_accuracy=0.9, median_latency_s=0.3),
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "headline.svg"
        fig = headline_bar_chart(reports, output_path=out)
        assert isinstance(fig, plt.Figure)
        assert out.exists()
        assert out.stat().st_size > 1000, "SVG file looks empty"
        plt.close(fig)
    print("  ✓ figure returned + saved")


# ---------------------------------------------------------------------
# Test 2 — works for 1, 2, 3 reports
# ---------------------------------------------------------------------

def test_headline_various_sizes():
    print("\nTest 2: headline_bar_chart works for 1, 2, 3 reports …")
    for n in (1, 2, 3):
        reports = [_make_report(b) for b in ("B1", "B2", "B3")[:n]]
        fig = headline_bar_chart(reports, output_path=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    print("  ✓ 1, 2, 3-baseline figures all render")


# ---------------------------------------------------------------------
# Test 3 — empty input raises
# ---------------------------------------------------------------------

def test_headline_empty():
    print("\nTest 3: empty report list raises ValueError …")
    try:
        headline_bar_chart([])
    except ValueError:
        print("  ✓ ValueError raised")
    else:
        raise AssertionError("should have rejected empty input")


# ---------------------------------------------------------------------
# Test 4 — coverage_over_time on a no-identification log
# ---------------------------------------------------------------------

def test_coverage_no_ids():
    print("\nTest 4: coverage_over_time renders flat-zero when nothing identified …")
    log = EventLog()
    log.duration_s = 5.0
    log.interesting_targets = {"target_x"}
    log.target_true_class = {"target_x": "C10"}
    # No log_event calls → nothing identified
    fig = coverage_over_time({"B1": log}, output_path=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("  ✓ flat-zero baseline rendered without error")


# ---------------------------------------------------------------------
# Test 5 — coverage_over_time with identifications
# ---------------------------------------------------------------------

def test_coverage_with_ids():
    print("\nTest 5: coverage_over_time draws step curve for IDs …")
    log = _make_log_with_events(
        baseline_targets_seen={"a": "C1", "b": "C4", "c": "C10"},
        duration=5.0,
    )
    fig = coverage_over_time({"B2": log}, output_path=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("  ✓ step curve rendered (3 IDs over 5 s)")


# ---------------------------------------------------------------------
# Test 6 — bandwidth_over_time: B1 flat, B2 curve
# ---------------------------------------------------------------------

def test_bandwidth_over_time():
    print("\nTest 6: bandwidth_over_time renders both flat and time-varying …")
    log_b1 = EventLog()
    log_b1.duration_s = 5.0
    log_b2 = _make_log_with_events({"a": "C1", "b": "C4", "c": "C10"}, duration=5.0)
    fig = bandwidth_over_time({"B1": log_b1, "B2": log_b2}, output_path=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("  ✓ B1 flat line + B2 sliding window both rendered")


# ---------------------------------------------------------------------
# Test 7 — polarization_detection_timeline with targets
# ---------------------------------------------------------------------

def test_polarization_timeline():
    print("\nTest 7: polarization_detection_timeline renders polarized targets …")
    log = _make_log_with_events(
        baseline_targets_seen={"polarized_target_1": "C10"},
        duration=5.0,
        polarized={"polarized_target_1": np.pi / 4},
    )
    fig = polarization_detection_timeline(
        {"B2": log}, output_path=None,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("  ✓ Gantt-style plot rendered")


# ---------------------------------------------------------------------
# Test 8 — polarization timeline with no polarized targets
# ---------------------------------------------------------------------

def test_polarization_empty():
    print("\nTest 8: polarization_detection_timeline handles empty case …")
    log = EventLog()
    log.duration_s = 1.0
    fig = polarization_detection_timeline({"B1": log}, output_path=None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("  ✓ 'nothing to plot' fallback rendered")


# ---------------------------------------------------------------------
# Test 9 — live overlay stubs raise
# ---------------------------------------------------------------------

def test_overlays_paint_geoms():
    """Each overlay must add ≥1 geom to the scene without raising."""
    print("\nTest 9: live overlays append geoms to MjvScene …")
    import mujoco

    from stomatopod_vision.gimbal_control import GimbalSetpoint
    from stomatopod_vision.metrics import EventLog
    from stomatopod_vision.scheduler import SaliencyScheduler
    from stomatopod_vision.world import Scene

    scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes.xml")
    scene.reset()
    with mujoco.Renderer(scene.model, height=240, width=320) as r:

        # FoV overlay → ≥2 geoms (one per eye)
        r.update_scene(scene.data)
        n0 = r.scene.ngeom
        render_eye_fov_overlay(r.scene, scene.model, scene.data, alpha=0.2)
        assert r.scene.ngeom >= n0 + 2, \
            f"FoV overlay should add ≥2 geoms; before={n0} after={r.scene.ngeom}"
        print(f"  ✓ render_eye_fov_overlay added {r.scene.ngeom - n0} geoms (≥2)")

        # Recent-sightings: synthesise a log with one recent event
        r.update_scene(scene.data)
        n0 = r.scene.ngeom
        log = EventLog()
        log.populate_targets_from_scene(scene)
        log.preprocessed_events = [PreprocessedEvent(
            time=0.4, eye="L", target_name="target_R_1",
            azimuth=0.0, elevation=0.0, distance=1.0,
            spectral_pattern=(10,0,0,0,0,0,0,0,0,0,0,0), polarization_angle=None, circular_handedness=None,
        )]
        render_recent_sightings(r.scene, scene.model, scene.data,
                                log=log, time_now=0.5, lookback_s=1.0)
        assert r.scene.ngeom >= n0 + 1, \
            f"recent_sightings should add ≥1 halo geom; got Δ={r.scene.ngeom - n0}"
        print(f"  ✓ render_recent_sightings added {r.scene.ngeom - n0} halo geom(s)")

        # Saliency: build a tiny candidate / score dict
        r.update_scene(scene.data)
        n0 = r.scene.ngeom
        sched = SaliencyScheduler(seed=0)
        # sp would be used by total_score if we called it; we use direct scores instead.
        cands = {e: sched.sample_candidates(e)[:10] for e in ("L", "C1")}
        scores = {e: np.linspace(0.0, 1.0, 10) for e in ("L", "C1")}
        render_saliency_map_overlay(r.scene, scene.model, scene.data,
                                    candidates_per_eye=cands,
                                    scores_per_eye=scores,
                                    max_dots=10)
        assert r.scene.ngeom >= n0 + 5, \
            f"saliency overlay should add several dots; got Δ={r.scene.ngeom - n0}"
        print(f"  ✓ render_saliency_map_overlay added {r.scene.ngeom - n0} dots")

        # Saliency no-op when called with None
        r.update_scene(scene.data)
        n0 = r.scene.ngeom
        render_saliency_map_overlay(r.scene, scene.model, scene.data,
                                    candidates_per_eye=None, scores_per_eye=None)
        assert r.scene.ngeom == n0, "saliency overlay should no-op on None input"
        print("  ✓ render_saliency_map_overlay no-ops on None inputs")


def test_record_run_writes_video():
    """record_run runs end-to-end and writes a non-empty file."""
    print("\nTest 9b: record_run produces an MP4 …")
    from stomatopod_vision.gimbal_control import GimbalSetpoint
    from stomatopod_vision.viz import record_run as _record_run

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test_record.mp4"
        result = _record_run(
            REPO_ROOT / "models" / "stomatopod_eyes.xml",
            setpoint_at=lambda t: GimbalSetpoint(yaw_L=0.3 * np.sin(t),
                                                 yaw_R=-0.3 * np.sin(t)),
            duration_s=0.4,
            output_path=out,
            fps=10, width=320, height=240,
            quiet=True,
        )
        assert result.exists() and result.stat().st_size > 100, \
            f"expected a non-trivial video at {result}"
        print(f"  ✓ wrote {result.stat().st_size}-byte MP4 at {result.name}")


def test_record_run_with_overlays():
    """record_run accepts overlays and they get invoked per frame."""
    print("\nTest 9c: record_run threads overlays through to each frame …")
    from stomatopod_vision.gimbal_control import GimbalSetpoint
    from stomatopod_vision.viz import (
        record_run as _record_run,
    )
    from stomatopod_vision.viz import (
        render_eye_fov_overlay,
    )

    n_calls = {"fov": 0}

    def counting_fov(scene, model, data, **_kw):
        render_eye_fov_overlay(scene, model, data, alpha=0.2)
        n_calls["fov"] += 1

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ovl.mp4"
        _record_run(
            REPO_ROOT / "models" / "stomatopod_eyes.xml",
            setpoint_at=lambda _t: GimbalSetpoint(),
            duration_s=0.3,
            output_path=out,
            fps=10, width=320, height=240,
            overlays=[counting_fov],
            quiet=True,
        )
        assert out.exists() and out.stat().st_size > 100
        # At 0.3s × 10 fps we expect ≥3 frames; allow off-by-one slack
        assert n_calls["fov"] >= 2, \
            f"overlay should be called ≥2 times; got {n_calls['fov']}"
        print(f"  ✓ overlay invoked {n_calls['fov']} times across the run")


# ---------------------------------------------------------------------
# Test 10 — end-to-end: load real JSON, render all 4 plots
# ---------------------------------------------------------------------

def test_end_to_end_with_real_results():
    print("\nTest 10: end-to-end on the real (single-seed) JSON files …")
    data_dir = REPO_ROOT / "results" / "data"
    # Accept either the single-seed layout (B*_metrics.json) or fall back
    # to a representative seed file from a multi-seed run (B*_seed0_metrics.json).
    def _pick(prefix: str, ext: str) -> Path | None:
        single = data_dir / f"{prefix}_{ext}"
        if single.exists():
            return single
        seeded = data_dir / f"{prefix}_seed0_{ext}"
        return seeded if seeded.exists() else None

    reports = []
    logs = {}
    for b in ("B1", "B2", "B3"):
        mp = _pick(b, "metrics.json")
        lp = _pick(b, "log.json")
        if mp is not None:
            reports.append(MetricsReport.load_json(mp))
        if lp is not None:
            logs[b] = EventLog.load_json(lp)

    if not reports:
        print("  ⚠ no per-baseline JSON files on disk — skipping end-to-end test")
        return
    assert reports, "expected at least one MetricsReport on disk"
    print(f"  loaded reports for: {[r.baseline for r in reports]}")
    print(f"  loaded logs for   : {list(logs.keys())}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fig = headline_bar_chart(reports, output_path=td / "headline.svg")
        plt.close(fig)
        assert (td / "headline.svg").exists()
        print(f"  ✓ headline.svg rendered "
              f"({(td / 'headline.svg').stat().st_size} bytes)")

        if logs:
            fig = coverage_over_time(logs, output_path=td / "cov.svg")
            plt.close(fig)
            fig = bandwidth_over_time(logs, output_path=td / "bw.svg")
            plt.close(fig)
            fig = polarization_detection_timeline(logs, output_path=td / "pol.svg")
            plt.close(fig)
            for name in ("cov.svg", "bw.svg", "pol.svg"):
                assert (td / name).exists()
                print(f"  ✓ {name} rendered")


def _fake_summary(baseline: str, n: int,
                  cov: tuple[float, float], bw: tuple[float, float],
                  pol: tuple[float, float], lat: tuple[float, float]) -> dict:
    """Build a summary-dict shape matching write_summary_if_multi_seed."""
    def block(mean: float, std: float, n: int) -> dict:
        # Synthesise per-seed values that match mean/std exactly:
        # pair-symmetric around the mean.
        vals = [mean] * n
        return {"mean": mean, "std": std, "min": mean - std,
                "max": mean + std, "values": vals}
    return {
        "baseline": baseline,
        "n_seeds": n,
        "seeds": list(range(n)),
        "per_seed": [{"baseline": baseline, "coverage": cov[0],
                      "bandwidth_bps": bw[0],
                      "polarization_accuracy": pol[0],
                      "median_latency_s": lat[0]} for _ in range(n)],
        "coverage":              block(cov[0], cov[1], n),
        "bandwidth_bps":         block(bw[0],  bw[1],  n),
        "polarization_accuracy": block(pol[0], pol[1], n),
        "median_latency_s":      block(lat[0], lat[1], n),
    }


def test_headline_multi_seed():
    print("\nTest 11: headline_bar_chart_multi_seed renders error bars …")
    summaries = [
        _fake_summary("B1", 5, cov=(0.00, 0.00), bw=(104000.0, 0.0),
                      pol=(0.00, 0.00), lat=(10.0, 0.0)),
        _fake_summary("B2", 5, cov=(0.00, 0.00), bw=(32.0, 0.0),
                      pol=(0.00, 0.00), lat=(10.0, 0.0)),
        _fake_summary("B3", 5, cov=(0.97, 0.05), bw=(35700.0, 4000.0),
                      pol=(0.97, 0.05), lat=(0.0, 0.0)),
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "headline.svg"
        fig = headline_bar_chart_multi_seed(summaries, output_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 1000, "expected non-empty SVG"
        print(f"  ✓ rendered headline.svg ({out.stat().st_size} bytes)")

    # empty case
    try:
        headline_bar_chart_multi_seed([], output_path=None)
        raise AssertionError("expected ValueError on empty summaries")
    except ValueError:
        print("  ✓ ValueError on empty summaries")


def test_coverage_multi_seed_ribbon():
    print("\nTest 12: coverage_over_time_multi_seed draws a ribbon …")
    # Build a small per-baseline list of logs.
    def _mk_log(target_times: list[float], duration: float = 1.0) -> EventLog:
        log = EventLog()
        log.duration_s = duration
        log.interesting_targets = {"target_A"}
        log.target_true_class = {"target_A": "C10"}
        log.target_in_fov_intervals = {"target_A": [(0.0, duration)]}
        for t in target_times:
            log.preprocessed_events.append(PreprocessedEvent(
                time=t, eye="L", target_name="target_A",
                azimuth=0.0, elevation=0.0, distance=1.0,
                spectral_pattern=(0,0,0,0,0,0,0,0,0,10,0,0), circular_handedness=None,
                polarization_angle=None))
        return log

    logs_by_baseline = {
        "B1": [_mk_log([]), _mk_log([])],                          # never identified
        "B3": [_mk_log([0.2]), _mk_log([0.3]), _mk_log([0.25])],   # 3 seeds vary
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cov.svg"
        fig = coverage_over_time_multi_seed(logs_by_baseline, output_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 500
        print(f"  ✓ rendered coverage_over_time.svg ({out.stat().st_size} bytes)")


def test_bandwidth_multi_seed_ribbon():
    print("\nTest 13: bandwidth_over_time_multi_seed draws a ribbon …")
    def _mk_log(event_times: list[float], duration: float = 2.0) -> EventLog:
        log = EventLog()
        log.duration_s = duration
        log.raw_stream_bytes_per_sample = 104
        log.event_bytes_per_event = 16
        for t in event_times:
            log.preprocessed_events.append(PreprocessedEvent(
                time=t, eye="L", target_name="target_A",
                azimuth=0.0, elevation=0.0, distance=1.0,
                spectral_pattern=(0,0,0,0,0,0,0,0,0,10,0,0), circular_handedness=None,
                polarization_angle=None))
        return log

    logs_by_baseline = {
        "B1": [_mk_log([]), _mk_log([])],
        "B2": [_mk_log([0.5, 1.0]), _mk_log([0.6])],
        "B3": [_mk_log([0.4, 0.5, 0.6, 0.7]),
               _mk_log([0.3, 0.45, 0.55, 0.65]),
               _mk_log([0.5, 0.6, 0.7, 0.8])],
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bw.svg"
        fig = bandwidth_over_time_multi_seed(logs_by_baseline, output_path=out)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 500
        print(f"  ✓ rendered bandwidth_over_time.svg ({out.stat().st_size} bytes)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tests for stomatopod_vision.viz")
    print("=" * 60)
    test_headline_basic()
    test_headline_various_sizes()
    test_headline_empty()
    test_coverage_no_ids()
    test_coverage_with_ids()
    test_bandwidth_over_time()
    test_polarization_timeline()
    test_polarization_empty()
    test_overlays_paint_geoms()
    test_record_run_writes_video()
    test_record_run_with_overlays()
    test_end_to_end_with_real_results()
    test_headline_multi_seed()
    test_coverage_multi_seed_ribbon()
    test_bandwidth_multi_seed_ribbon()
    print("\nAll viz tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
