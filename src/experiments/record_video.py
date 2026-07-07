"""
record_video.py — record an MP4 of a simulation with live overlays
===================================================================

The bonus video-presentation tool. Runs one of the project's three
baselines (or B3-Learned) under :func:`stomatopod_vision.viz.record_run`
and writes an MP4 with optional overlays:

    * **fov**   — translucent ellipsoids showing each eye's mid-band FoV
                  (red = left, blue = right).
    * **rings** — green halos around any target that produced an event
                  in the last ``--lookback`` seconds.
    * **saliency** — golden spheres at each candidate direction the
                  scheduler is currently scoring, sized by score.
                  Only meaningful for ``--baseline B3`` / ``B3L``.

CLI
---
    # B3 on the hard scene with all three overlays
    MUJOCO_GL=egl python src/experiments/record_video.py \\
        --baseline B3 \\
        --model models/stomatopod_eyes_hard.xml \\
        --duration 8 --fps 30 \\
        --output results/videos/b3_hard.mp4 \\
        --overlays fov rings saliency

    # B2 on the default scene, no overlays
    MUJOCO_GL=egl python src/experiments/record_video.py \\
        --baseline B2 --duration 5 --output results/videos/b2_default.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _common import (  # noqa: E402
    _REPO_ROOT,
    DEFAULT_MODEL,
    DEFAULT_RESULTS_DIR,
)

from stomatopod_vision.gimbal_control import GimbalSetpoint  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    LearnedScheduler,
    SaliencyScheduler,
    ScoringWeights,
)
from stomatopod_vision.viz import (  # noqa: E402
    record_run,
    render_eye_fov_overlay,
    render_recent_sightings,
    render_saliency_map_overlay,
)

_VALID_OVERLAYS = ("fov", "rings", "saliency")


def _build_callbacks(args):
    """Return ``(setpoint_at, pipeline, on_events, scheduler)`` for this run."""
    if args.baseline == "B1":
        # B1 = eyes locked forward, raw stream → no preprocessing, no events.
        sp_const = GimbalSetpoint()
        return (lambda _t: sp_const, None, None, None)

    if args.baseline == "B2":
        # B2 = eyes locked forward, full preprocessing pipeline.
        sp_const = GimbalSetpoint()
        return (lambda _t: sp_const, PreprocessingPipeline(), None, None)

    if args.baseline == "B3":
        scheduler = SaliencyScheduler(
            n_candidates=args.n_candidates,
            decision_period_s=args.decision_period,
            weights=ScoringWeights(),
            seed=args.seed,
        )
        return (
            lambda t: scheduler.next_setpoint(t, scheduler._held_setpoint),
            PreprocessingPipeline(),
            lambda events, t: scheduler.update_memory(events, t),
            scheduler,
        )

    if args.baseline == "B3L":
        if not args.mlp_path.exists():
            raise SystemExit(
                f"MLP weights not found at {args.mlp_path}. Train first:\n"
                f"  MUJOCO_GL=egl python src/experiments/train_learned.py"
            )
        scheduler = LearnedScheduler.from_file(
            args.mlp_path,
            n_candidates=args.n_candidates,
            decision_period_s=args.decision_period,
            seed=args.seed,
        )
        return (
            lambda t: scheduler.next_setpoint(t, scheduler._held_setpoint),
            PreprocessingPipeline(),
            lambda events, t: scheduler.update_memory(events, t),
            scheduler,
        )

    raise SystemExit(f"Unknown baseline: {args.baseline}")


def _build_overlays(names: list[str], args):
    """Translate the list of overlay names from the CLI into callables."""
    overlays = []

    if "fov" in names:
        def fov(scene, model, data, **_kw):
            render_eye_fov_overlay(scene, model, data, alpha=args.fov_alpha,
                                   cone_length=args.fov_length)
        overlays.append(fov)

    if "rings" in names:
        def rings(scene, model, data, log=None, time_now=0.0, **_kw):
            render_recent_sightings(scene, model, data, log, time_now,
                                    lookback_s=args.lookback)
        overlays.append(rings)

    if "saliency" in names:
        def saliency(scene, model, data, scheduler=None, time_now=0.0, **_kw):
            if scheduler is None or not hasattr(scheduler, "sample_candidates"):
                return
            sp = GimbalSetpoint()
            cands = {e: scheduler.sample_candidates(e) for e in ("L", "R")}
            scores = {
                e: np.array([scheduler.total_score(e, y, p, r, time_now, sp)
                             for y, p, r in cands[e]])
                for e in ("L", "R")
            }
            render_saliency_map_overlay(
                scene, model, data,
                candidates_per_eye=cands, scores_per_eye=scores,
                max_dots=args.saliency_max_dots,
            )
        overlays.append(saliency)

    return overlays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--baseline", choices=["B1", "B2", "B3", "B3L"],
                        default="B3",
                        help="Which baseline to record (default: B3).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"MuJoCo XML to render "
                             f"(default: {DEFAULT_MODEL.relative_to(_REPO_ROOT)}).")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulated seconds to record (default: 5.0).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default: 0).")
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_RESULTS_DIR / "videos" / "out.mp4",
                        help=f"Output MP4 path "
                             f"(default: {(DEFAULT_RESULTS_DIR / 'videos' / 'out.mp4').relative_to(_REPO_ROOT)}).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frames per second (default: 30).")
    parser.add_argument("--width", type=int, default=1024,
                        help="Frame width (default: 1024; capped at the XML's "
                             "<global offwidth=...>).")
    parser.add_argument("--height", type=int, default=720,
                        help="Frame height (default: 720).")
    parser.add_argument("--camera", type=str, default="cinematic",
                        help='Named camera (default: "cinematic"). Options for the '
                             'project XMLs: overview, cinematic, close_left, close_right. '
                             'Pass an empty string to use the free MuJoCo camera.')
    parser.add_argument("--camera-pan", type=float, default=0.0,
                        help='Total camera pan sweep in degrees over the full clip '
                             'duration (default: 0.0). A gentle 8–15° pan is useful '
                             'for B1/B2 clips where the scene is otherwise static.')

    parser.add_argument("--overlays", nargs="*",
                        choices=_VALID_OVERLAYS, default=[],
                        help="Live overlays to paint each frame (default: none).")
    parser.add_argument("--fov-alpha", type=float, default=0.10,
                        help="FoV ellipsoid alpha (default: 0.10; keep low so the eye "
                             "towers underneath the cones stay visible).")
    parser.add_argument("--fov-length", type=float, default=1.2,
                        help="FoV cone length in metres (default: 1.2).")
    parser.add_argument("--lookback", type=float, default=0.5,
                        help='"rings" overlay lookback window (s, default: 0.5).')
    parser.add_argument("--saliency-max-dots", type=int, default=30,
                        help="Max saliency dots per eye (default: 30).")

    # Scheduler knobs (B3 / B3L only)
    parser.add_argument("--n-candidates", type=int, default=30,
                        help="Scheduler candidates per re-plan (default: 30).")
    parser.add_argument("--decision-period", type=float, default=0.10,
                        help="Scheduler re-plan period s (default: 0.10).")
    parser.add_argument("--mlp-path", type=Path,
                        default=DEFAULT_RESULTS_DIR / "learned" / "mlp.npz",
                        help="Trained MLP weights (B3L only).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-second progress output.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"record_video:  {args.baseline}  on {args.model.name}")
    print(f"  duration   : {args.duration}s @ {args.fps} fps  ({args.width}×{args.height})")
    print(f"  overlays   : {args.overlays or '(none)'}")
    print(f"  output     : {args.output}")
    print("=" * 70)

    setpoint_at, pipeline, on_events, scheduler = _build_callbacks(args)
    overlays = _build_overlays(args.overlays, args)

    out = record_run(
        args.model,
        setpoint_at=setpoint_at,
        duration_s=args.duration,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera=args.camera if args.camera else None,
        pipeline=pipeline,
        on_events=on_events,
        overlays=overlays,
        scheduler_for_overlays=scheduler,
        camera_pan_deg=args.camera_pan,
        quiet=args.quiet,
    )
    print(f"\n✓ Done. Wrote {out}.")


if __name__ == "__main__":
    main()
