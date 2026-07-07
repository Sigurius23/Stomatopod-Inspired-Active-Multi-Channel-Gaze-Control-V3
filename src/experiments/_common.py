"""
_common.py — shared experiment scaffolding
==========================================

Used by ``run_b1_fixed.py`` and ``run_b2_preprocessed.py`` (and later by
``run_b3_active.py``). Anything more than 5 lines that all three baselines
need lives here.

Provides:
  - :func:`add_common_args` — argparse helper for the standard CLI flags.
  - :class:`SimContext`   — bundle of the MuJoCo objects + eyes + scheduler
                            for a single run.
  - :class:`FovTracker`   — internal helper that turns per-step "what's
                            visible" snapshots into the (t_enter, t_exit)
                            intervals that :class:`EventLog` expects.
  - :func:`run_simulation` — the actual time-stepped loop. Returns a
                            populated :class:`EventLog`.

The B1/B2 entry-point files are thin wrappers around these. By keeping
all the shared machinery here we (a) avoid drift between baselines and
(b) make it trivial to add B3 later by passing a different scheduler.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure we can find the package whether the script is run as
# `python src/experiments/run_b1_fixed.py` or `python -m experiments.run_b1_fixed`.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Default to headless GPU rendering on Colab/servers. Override with
# MUJOCO_GL=glfw if you want a window.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from stomatopod_vision.gimbal_control import GimbalPD, GimbalSetpoint  # noqa: E402
from stomatopod_vision.metrics import (  # noqa: E402
    EventLog,
    MetricsReport,
    compute_all,
)
from stomatopod_vision.preprocessing import (  # noqa: E402
    PreprocessingPipeline,
)
from stomatopod_vision.sensor import (  # noqa: E402
    RawSighting,
    VirtualEye,
    make_eye_pair,
)
from stomatopod_vision.world import Scene  # noqa: E402

# Resolve canonical paths once so the scripts don't all hardcode them.
DEFAULT_MODEL = _REPO_ROOT / "models" / "stomatopod_eyes.xml"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"


# =====================================================================
# CLI argument helper
# =====================================================================

def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the CLI flags every baseline accepts. Mutates ``p``."""
    p.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulation duration in seconds (default: 10).")
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility (default: 0). Ignored when "
             "--seeds is supplied.")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run the simulation once per seed in this list, writing "
             "per-seed JSON files plus a <baseline>_summary.json with "
             "mean/std across seeds. When omitted, runs a single seed "
             "from --seed.")
    p.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"Path to the MuJoCo XML (default: {DEFAULT_MODEL.relative_to(_REPO_ROOT)}).")
    p.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help=f"Where to write JSON metrics (default: {DEFAULT_RESULTS_DIR.relative_to(_REPO_ROOT)}).")
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step progress output.")


# =====================================================================
# FoV interval tracker
# =====================================================================

class FovTracker:
    """
    Converts per-step "set of currently visible targets" snapshots into
    the ``(t_enter, t_exit)`` intervals that :class:`EventLog` expects.

    The simulation loop calls :meth:`update` every step with the targets
    visible in that step; at the end, :meth:`finalize` closes any still-
    open intervals at the supplied end-time.
    """

    def __init__(self) -> None:
        self._open: dict[str, float] = {}  # target → t_enter

    def update(
        self,
        log: EventLog,
        currently_visible: Iterable[str],
        time_now: float,
    ) -> None:
        """Fold one step's visibility snapshot into the log."""
        visible_set = set(currently_visible)
        # Newcomers
        for name in visible_set:
            if name not in self._open:
                self._open[name] = time_now
        # Leavers
        for name in list(self._open):
            if name not in visible_set:
                t_enter = self._open.pop(name)
                log.log_target_fov(name, t_enter, time_now)

    def finalize(self, log: EventLog, time_end: float) -> None:
        """Close any still-open intervals at ``time_end``."""
        for name, t_enter in self._open.items():
            log.log_target_fov(name, t_enter, time_end)
        self._open.clear()


# =====================================================================
# Simulation context (held by the entry-point scripts)
# =====================================================================

@dataclass
class SimContext:
    """All MuJoCo + project objects needed to step the simulation."""
    model: mujoco.MjModel
    data: mujoco.MjData
    scene: Scene
    eye_L: VirtualEye
    eye_R: VirtualEye
    pd: GimbalPD
    # Auto-populated by build_context() when the scene XML's file stem
    # is "stomatopod_eyes_moving"; None otherwise. Wired through to
    # run_simulation() if the caller doesn't pass an explicit
    # `motion_controller`.
    motion_controller: "object | None" = None


def build_context(
    model_path: Path,
    seed: int = 0,
    *,
    receptor_noise_std: float = 0.0,
) -> SimContext:
    """Load the model, build the Scene, eyes, and gimbal PD controller.

    For the moving-targets scene
    (``models/stomatopod_eyes_moving.xml``) the returned
    :class:`SimContext` also carries a pre-wired
    :class:`stomatopod_vision.world.MovingTargetController` so callers
    don\'t have to know that detail.

    Parameters
    ----------
    receptor_noise_std :
        Per-receptor additive Gaussian noise standard deviation,
        forwarded to :func:`make_eye_pair`. Default 0.0 reproduces
        the deterministic behaviour every test + figure assumes;
        set to e.g. 0.05 for the noise-ablation experiment.
    """
    np.random.seed(seed)
    scene = Scene.from_xml(model_path)
    scene.reset()
    # Each call to build_context gets a noise seed derived from the
    # general seed so multi-seed sweeps see independent noise streams.
    eye_L, eye_R = make_eye_pair(
        scene,
        receptor_noise_std=receptor_noise_std,
        noise_seed=2 * int(seed),
    )
    pd = GimbalPD(scene.model)

    # Auto-attach the moving-targets controller iff the model XML file
    # is the moving variant. We key on the filename stem because
    # MOVING_TARGETS is currently aliased to DEFAULT_TARGETS so identity
    # comparison on the targets tuple wouldn't distinguish them.
    motion_controller = None
    if Path(str(model_path)).stem.lower() == "stomatopod_eyes_moving":
        from stomatopod_vision.world import MovingTargetController
        motion_controller = MovingTargetController(scene, Scene.MOVING_MOTIONS)

    return SimContext(
        model=scene.model,
        data=scene.data,
        scene=scene,
        eye_L=eye_L,
        eye_R=eye_R,
        pd=pd,
        motion_controller=motion_controller,
    )


# =====================================================================
# The simulation loop (B1 and B2 share this; B3 will too)
# =====================================================================

def run_simulation(
    ctx: SimContext,
    *,
    setpoint_at: callable,
    pipeline: PreprocessingPipeline | None,
    duration_s: float,
    quiet: bool = False,
    progress_every_s: float = 1.0,
    on_events: callable | None = None,
    motion_controller: "object | None" = None,
    controller_rate_hz: float | None = None,
) -> EventLog:
    """
    Run a fixed-duration simulation and return a populated :class:`EventLog`.

    Parameters
    ----------
    ctx :
        The :class:`SimContext` from :func:`build_context`.
    setpoint_at :
        Callable ``(time_now: float) -> GimbalSetpoint``. For B1/B2 this
        will be a constant lambda returning ``GimbalSetpoint()`` (eyes
        forward). For B3 it will query the scheduler.
    pipeline :
        Either a :class:`PreprocessingPipeline` (B2/B3) or ``None`` (B1).
        When ``None``, raw sightings are still logged but no
        :class:`PreprocessedEvent` instances are produced.
    duration_s :
        How long (in simulated seconds) to run.
    quiet :
        Suppress periodic progress output.
    progress_every_s :
        How often to print progress (in simulated seconds).
    on_events :
        Optional callback ``(events: list[PreprocessedEvent], time: float) -> None``
        invoked after each step\'s pipeline output is produced. B3 uses
        this to feed events into the scheduler\'s memory online.
    motion_controller :
        Optional object with a ``step(time_s)`` method, called every
        physics step *before* ``mj_step``. Used by Bonus #9 to drive
        ``data.mocap_pos`` so target bodies move during the simulation.
    controller_rate_hz :
        Optional rate (Hz) at which to query ``setpoint_at``. When
        ``None`` (the default), the callback is invoked at every
        physics step — preserving the original behaviour. When set
        (e.g. ``10.0``), the callback is invoked at this rate and the
        most recent setpoint is held between calls. This is purely an
        efficiency optimisation: B3\'s scheduler already throttles
        replans internally, so calling it at 500 Hz wastes ~50 calls
        per useful re-plan. Setting ``controller_rate_hz=10.0`` for B3
        gives a ~50x speedup of the setpoint callback path with
        identical produced setpoints.

    Returns
    -------
    log :
        Fully populated event log, ready for :func:`compute_all`.
    """
    # The simulation loop is split into 5 single-responsibility helpers
    # below. The orchestrator here just wires them together and tracks
    # the held setpoint + progress timer. See:
    #
    #   _compute_setpoint_stride  — how often setpoint_at is queried
    #   _drive_gimbals            — setpoint → PD → (optional) target motion
    #   _read_sensors_and_fov     — virtual eye step + FoV bookkeeping
    #   _run_preprocessing        — Layer 2 pipeline + on_events callback
    #   _maybe_print_progress     — periodic stdout summary

    log = EventLog()
    log.populate_targets_from_scene(ctx.scene)
    fov = FovTracker()
    timestep = ctx.model.opt.timestep
    n_steps = int(duration_s / timestep)
    steps_per_setpoint = _compute_setpoint_stride(controller_rate_hz, timestep)

    sp: "GimbalSetpoint | None" = None
    progress_state = {"next_at": progress_every_s, "visible_now": 0}

    for step_idx in range(n_steps):
        sp = _drive_gimbals(
            ctx, sp, setpoint_at, step_idx, steps_per_setpoint,
            motion_controller=motion_controller,
        )
        mujoco.mj_step(ctx.model, ctx.data)
        now = float(ctx.data.time)

        raws = _read_sensors_and_fov(ctx, log, fov, now)
        progress_state["visible_now"] = len({r.target_name for r in raws})

        if pipeline is not None:
            _run_preprocessing(ctx, log, raws, pipeline, now, on_events)

        _maybe_print_progress(log, now, progress_every_s,
                              progress_state, quiet=quiet)

    fov.finalize(log, float(ctx.data.time))
    log.duration_s = float(ctx.data.time)
    return log


# ---------------------------------------------------------------------
# Inner helpers (each does one thing; tested via the existing
# test_metrics / test_viz suites which already drive the full loop).
# ---------------------------------------------------------------------

def _compute_setpoint_stride(controller_rate_hz: float | None,
                             timestep: float) -> int:
    """Number of physics steps between setpoint queries.

    When ``controller_rate_hz`` is None or ≤ 0, returns 1 (query every
    physics step — the original behaviour). Otherwise returns
    ``max(1, round(1 / (rate × timestep)))``, so the callback fires
    approximately at the requested rate.
    """
    if controller_rate_hz is None or controller_rate_hz <= 0:
        return 1
    return max(1, int(round(1.0 / (float(controller_rate_hz) * timestep))))


def _drive_gimbals(
    ctx: SimContext,
    held_setpoint: "GimbalSetpoint | None",
    setpoint_at: callable,
    step_idx: int,
    steps_per_setpoint: int,
    *,
    motion_controller: "object | None",
) -> "GimbalSetpoint":
    """Query the setpoint (if it\'s time), drive the PD, animate targets.

    Returns the (possibly newly queried, possibly held) setpoint so the
    outer loop can pass it back in for the next iteration.
    """
    if held_setpoint is None or step_idx % steps_per_setpoint == 0:
        held_setpoint = setpoint_at(ctx.data.time)
    ctx.pd.step(ctx.data, held_setpoint)

    # Explicit kwarg wins; otherwise fall back to the controller auto-
    # attached by build_context (set for the moving scene only).
    mc = motion_controller if motion_controller is not None else ctx.motion_controller
    if mc is not None:
        mc.step(float(ctx.data.time))
    return held_setpoint


def _read_sensors_and_fov(
    ctx: SimContext,
    log: EventLog,
    fov: FovTracker,
    now: float,
) -> list[RawSighting]:
    """Step both virtual eyes, log raw sightings, update the FoV tracker."""
    raws: list[RawSighting] = ctx.eye_L.step() + ctx.eye_R.step()
    for r in raws:
        log.log_raw_sighting(now, r)
    fov.update(log, {r.target_name for r in raws}, now)
    return raws


def _run_preprocessing(
    ctx: SimContext,
    log: EventLog,
    raws: list[RawSighting],
    pipeline: PreprocessingPipeline,
    now: float,
    on_events: callable | None,
) -> None:
    """Run Layer 2, log the events, fire the on_events callback if any."""
    events = pipeline.step(
        raws,
        time_now=now,
        roll_angles={
            "L": ctx.eye_L.roll_angle(),
            "R": ctx.eye_R.roll_angle(),
        },
    )
    for ev in events:
        log.log_event(ev)
    if on_events is not None:
        on_events(events, now)


def _maybe_print_progress(
    log: EventLog,
    now: float,
    progress_every_s: float,
    state: dict,
    *,
    quiet: bool,
) -> None:
    """Periodic per-second stdout summary; mutates ``state["next_at"]``."""
    if quiet or now < state["next_at"]:
        return
    print(f"  t = {now:5.2f}s  "
          f"raws = {len(log.raw_sightings):6d}  "
          f"events = {len(log.preprocessed_events):5d}  "
          f"in-fov now = {state['visible_now']}")
    state["next_at"] += progress_every_s


# =====================================================================
# Pretty-print the report
# =====================================================================

def print_report(report: MetricsReport, *, header: str | None = None) -> None:
    """Print a one-line summary of the report to stdout."""
    if header:
        print(f"\n{header}")
        print("-" * len(header))
    print(f"  Baseline               : {report.baseline}")
    print(f"  Coverage               : {report.coverage:.3f}  (1.0 = perfect)")
    print(f"  Bandwidth              : {report.bandwidth_bps:,.0f} bytes/s")
    print(f"  Polarization accuracy  : {report.polarization_accuracy:.3f}  (1.0 = perfect)")
    print(f"  Median latency         : {report.median_latency_s:.3f} s")


# =====================================================================
# Standard end-of-script work (save JSON, optionally print)
# =====================================================================

def save_and_report(
    log: EventLog,
    baseline: str,
    results_dir: Path,
    *,
    quiet: bool = False,
    seed: int | None = None,
) -> MetricsReport:
    """
    Compute the four metrics, save metrics + log as JSON, optionally print.
    Returns the :class:`MetricsReport` for further use.

    Two files are written under ``results/data/``:
      - ``<baseline>_metrics.json`` — the four headline scalars
      - ``<baseline>_log.json``     — the full preprocessed-event log,
                                      reloadable by :class:`EventLog.load_json`
                                      for the time-series plots in viz.py

    When ``seed`` is supplied, the filenames are suffixed with ``_seedN``
    (e.g. ``B1_seed0_metrics.json``). This lets the multi-seed sweep
    in ``run_all.py`` write per-seed artefacts without clobbering each
    other. Pass ``seed=None`` (the default) to keep the original
    single-seed filenames for backward compatibility.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    report = compute_all(log, baseline)
    suffix = "" if seed is None else f"_seed{seed}"
    metrics_path = data_dir / f"{baseline}{suffix}_metrics.json"
    log_path = data_dir / f"{baseline}{suffix}_log.json"
    report.save_json(metrics_path)
    log.save_json(log_path)

    if not quiet:
        print_report(report, header=f"=== {baseline} results"
                                    f"{' (seed=' + str(seed) + ')' if seed is not None else ''} ===")
        print(f"\n  Saved metrics : {_rel(metrics_path)}")
        print(f"  Saved log     : {_rel(log_path)}")
    return report


def _rel(p: Path) -> Path:
    """Best-effort relative-path display for log messages."""
    try:
        return p.resolve().relative_to(_REPO_ROOT.resolve())
    except ValueError:
        return p


def resolve_seeds(args) -> list[int]:
    """Pick the seed list from parsed CLI args.

    Returns ``args.seeds`` when supplied (a non-empty list of ints) and
    falls back to ``[args.seed]`` otherwise. Always returns a list with
    at least one element so callers can write ``for seed in resolve_seeds(args):``.
    """
    if getattr(args, "seeds", None):
        return list(args.seeds)
    return [int(args.seed)]


def write_summary_if_multi_seed(
    baseline: str,
    results_dir: Path,
    seeds: list[int],
    *,
    quiet: bool = False,
) -> Path | None:
    """Aggregate per-seed metrics into ``<baseline>_summary.json``.

    Reads every ``<baseline>_seed{N}_metrics.json`` for ``N in seeds``
    from ``results_dir/data/`` and writes a single summary file with
    per-metric mean and population std. Returns the summary path, or
    ``None`` if only one seed was used (back-compat with single-seed
    workflows that should keep the original two-file layout).
    """
    if len(seeds) <= 1:
        return None

    import json
    import statistics

    data_dir = results_dir / "data"
    per_seed: list[dict] = []
    for seed in seeds:
        mp = data_dir / f"{baseline}_seed{seed}_metrics.json"
        if not mp.exists():
            raise FileNotFoundError(
                f"Missing per-seed metrics file {mp}; cannot build summary."
            )
        per_seed.append(json.loads(mp.read_text()))

    # Numeric scalar metrics we aggregate. ``baseline`` is a string and
    # is taken from the first record.
    metric_keys = [
        "coverage", "bandwidth_bps",
        "polarization_accuracy", "median_latency_s",
    ]
    summary: dict = {
        "baseline": baseline,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "per_seed": per_seed,
    }
    for k in metric_keys:
        vals = [float(rec[k]) for rec in per_seed]
        summary[k] = {
            "mean":  statistics.mean(vals),
            "std":   statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min":   min(vals),
            "max":   max(vals),
            "values": vals,
        }

    out_path = data_dir / f"{baseline}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not quiet:
        print(f"\n=== {baseline} multi-seed summary (n={len(seeds)}) ===")
        for k in metric_keys:
            m, sd = summary[k]["mean"], summary[k]["std"]
            print(f"  {k:<24s}  mean={m:>12.4f}  std={sd:>10.4f}")
        print(f"  Saved summary : {_rel(out_path)}")
    return out_path
