"""
metrics.py — the four headline metrics
======================================

All four metrics are tracked per-baseline and dumped to
``results/data/<baseline>_metrics.json``. The plotting script reads
those files to produce the headline comparison figure.

The four metrics
----------------
1. **Coverage** — fraction of *interesting* targets correctly identified
   within the time horizon. "Identified" means at least one
   :class:`PreprocessedEvent` was received whose ``dominant_class``
   matches the target's true spectral class, within
   ``identification_window_s`` of the target first appearing in some
   eye's FoV.

2. **Bandwidth** — bytes transmitted from sensor to controller per
   second. For B1 (raw stream): the full multi-channel sample is sent
   every step **per eye** regardless of whether anything is visible.
   For B2/B3 (event stream): only :class:`PreprocessedEvent` instances
   are sent, and only when emitted.

3. **Polarization accuracy** — for *polarized* targets, the fraction
   correctly classified within tolerance. This isolates the value of
   the roll DOF: only B3 should score meaningfully above zero.

4. **Response latency** — median delay between a target entering an
   eye's FoV and the controller receiving the first identifying event
   about it. Targets that are never identified contribute the full
   simulation duration as a *right-censored* value. Smaller = more
   reactive.

Each metric returns a single scalar suitable for plotting.

Conventions
-----------
- ``baseline`` is the string ``"B1"``, ``"B2"`` or ``"B3"``. Each
  bandwidth calculation behaves differently based on this value.
- ``EventLog`` is the central record. Experiment scripts populate it
  via ``log_raw_sighting``, ``log_event``, and ``log_target_fov``.
- All time values are in seconds (matching ``mujoco.MjData.time``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .preprocessing import PreprocessedEvent
from .sensor import RawSighting
from .world import Scene

# Number of eyes the simulation runs — matches sensor.make_eye_pair.
_N_EYES = 2

# Recognised baseline labels — anything else raises in bandwidth().
# B3L is the bonus "learned-scoring" variant of B3 and B3D is the bonus
# Hopf limit-cycle scanner; both use the same sparse-event bandwidth
# accounting as B3.
_BASELINES = ("B1", "B2", "B3", "B3L", "B3D")


# ---------------------------------------------------------------------
# Byte-size accounting for the bandwidth metric
# ---------------------------------------------------------------------
#
# Both numbers below are *derived* from the actual struct layouts in
# `stomatopod_vision.sensor.RawSighting` and
# `stomatopod_vision.preprocessing.PreprocessedEvent`. Keeping the
# derivation here (instead of as magic constants on EventLog) means
# the numbers stay correct if those dataclasses grow new fields. The
# derivation assumes:
#   - Floats are 8 bytes (NumPy float64 / IEEE 754 double).
#   - The body name is transmitted as a short integer ID, not the
#     string, so it costs 2 bytes per event (16-bit ID supports 65 535
#     distinct targets, which is far more than the few we have).
#   - The eye field ("L"/"R") is a single byte.
#   - Categorical fields (spectral class) are a single byte enum.

_BYTES_PER_FLOAT  = 8     # np.float64 / IEEE 754 double
_BYTES_PER_ENUM   = 1     # category index (≤256 categories)
_BYTES_PER_ID     = 2     # body-id (16-bit; 65 535 targets max)


def _raw_sighting_bytes(n_spectral_classes: int = 12,
                        n_pol_receptors: int = 4) -> int:
    """Bytes the raw stream carries per (sample, visible target).

    Fields contributed by :class:`stomatopod_vision.sensor.RawSighting`:

        target_id              (id)       2 bytes
        eye                    (enum)     1 byte
        azimuth                (float)    8 bytes
        elevation              (float)    8 bytes
        distance               (float)    8 bytes
        peripheral_intensity   (float)    8 bytes
        midband_activations    (12 floats) 96 bytes
        polarization_responses (4 floats) 32 bytes   # bare linear
        circular_responses     (4 floats) 32 bytes   # quarter-wave
        -----------------------------------
        TOTAL                             195 bytes

    The two polarization banks (bare linear + quarter-wave retarder) are
    counted separately because the raw stream physically carries both —
    that is exactly what lets the downstream decoder tell linear from
    circular polarization. Handedness itself is NOT in the raw stream: it
    is decoded downstream, so no ground-truth enum is counted here.
    """
    return (
        _BYTES_PER_ID            # target_id
        + _BYTES_PER_ENUM        # eye
        + 4 * _BYTES_PER_FLOAT   # azimuth + elevation + distance + peripheral_intensity
        + n_spectral_classes * _BYTES_PER_FLOAT
        + n_pol_receptors * _BYTES_PER_FLOAT   # polarization_responses (bare)
        + n_pol_receptors * _BYTES_PER_FLOAT   # circular_responses (quarter-wave)
    )


def _preprocessed_event_bytes() -> int:
    """Bytes the sparse event stream carries per emitted event.

    Fields contributed by
    :class:`stomatopod_vision.preprocessing.PreprocessedEvent`:

        time                   (float)    8 bytes
        eye                    (enum)     1 byte
        target_id              (id)       2 bytes
        azimuth                (float)    8 bytes
        elevation              (float)    8 bytes
        distance               (float)    8 bytes
        spectral_pattern       (12 enum)  12 bytes
        polarization_angle     (opt float) 8 bytes
        circular_handedness    (enum)     1 byte
        ------------------------------------
        TOTAL                             56 bytes
    """
    return (
        _BYTES_PER_FLOAT         # time
        + _BYTES_PER_ENUM        # eye
        + _BYTES_PER_ID          # target_id
        + 3 * _BYTES_PER_FLOAT   # azimuth + elevation + distance
        + 12 * _BYTES_PER_ENUM   # spectral_pattern
        + _BYTES_PER_FLOAT       # polarization_angle
        + _BYTES_PER_ENUM        # circular_handedness
    )


# Module-level canonical sizes (derived once at import). The dataclass
# below uses these as defaults so anyone subclassing EventLog can
# override them while still being type-safe.
RAW_BYTES_PER_SAMPLE  = _raw_sighting_bytes()
EVENT_BYTES_PER_EVENT = _preprocessed_event_bytes()


# ---------------------------------------------------------------------
# Per-baseline event log (what the experiment writes during a run)
# ---------------------------------------------------------------------

@dataclass
class EventLog:
    """
    Time-ordered log of everything that happened during one run.

    Used by all four metric calculators. The experiment script
    populates it via the ``log_*`` helpers below, then passes it to
    :func:`compute_all` at the end.

    Attributes
    ----------
    raw_sightings :
        List of ``(time, sighting)`` for every raw sighting (used for
        diagnostics; bandwidth uses the per-eye stream rate instead).
    preprocessed_events :
        Every :class:`PreprocessedEvent` actually emitted (B2, B3 only).
    target_in_fov_intervals :
        ``{target_name: [(t_enter, t_exit), ...]}`` — when each target
        was inside *some* eye's FoV. Used for latency and coverage.
    target_true_class :
        ``{target_name: true_spectral_class}`` — used by coverage to
        check whether the controller decoded the right class.
    interesting_targets :
        Names of targets flagged as ``is_interesting=True``.
    polarized_targets :
        ``{target_name: true_polarization_angle_rad}``.
    duration_s :
        Total simulation time (seconds).
    raw_stream_bytes_per_sample :
        Width of one raw sample in bytes (for bandwidth calc). Derived
        by :func:`_raw_sighting_bytes` from the actual
        :class:`~stomatopod_vision.sensor.RawSighting` struct layout.
        See module-level constant :data:`RAW_BYTES_PER_SAMPLE`.
    event_bytes_per_event :
        Width of one preprocessed event in bytes (for bandwidth calc).
        Derived by :func:`_preprocessed_event_bytes` from
        :class:`~stomatopod_vision.preprocessing.PreprocessedEvent`.
        See module-level constant :data:`EVENT_BYTES_PER_EVENT`.
    """
    raw_sightings: list[tuple[float, RawSighting]] = field(default_factory=list)
    preprocessed_events: list[PreprocessedEvent] = field(default_factory=list)
    target_in_fov_intervals: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict)
    target_true_class: dict[str, str] = field(default_factory=dict)
    interesting_targets: set[str] = field(default_factory=set)
    polarized_targets: dict[str, float] = field(default_factory=dict)
    circular_targets: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    raw_stream_bytes_per_sample: int = RAW_BYTES_PER_SAMPLE
    event_bytes_per_event: int = EVENT_BYTES_PER_EVENT

    # ----- mutation helpers ---------------------------------------------
    def log_raw_sighting(self, time_now: float, sighting: RawSighting) -> None:
        """Append a raw sighting (call from the experiment loop)."""
        self.raw_sightings.append((float(time_now), sighting))

    def log_event(self, event: PreprocessedEvent) -> None:
        """Append a preprocessed event (call from the experiment loop)."""
        self.preprocessed_events.append(event)

    def log_target_fov(
        self,
        target_name: str,
        t_enter: float,
        t_exit: float,
    ) -> None:
        """
        Record a (t_enter, t_exit) interval during which `target_name`
        was inside some eye's FoV. Multiple intervals per target are
        allowed (the target may leave and re-enter the FoV).
        """
        self.target_in_fov_intervals.setdefault(target_name, []).append(
            (float(t_enter), float(t_exit))
        )

    def populate_targets_from_scene(self, scene: Scene) -> None:
        """
        Copy the scene's target metadata into the log so the metric
        calculators have ground truth without needing the scene later.
        """
        self.target_true_class = {t.name: t.spectral_class for t in scene.targets}
        self.interesting_targets = set(scene.interesting_target_names())
        self.polarized_targets = dict(scene.polarized_targets())
        self.circular_targets = dict(scene.circularly_polarized_targets())

    def reset(self) -> None:
        """Clear all logged data."""
        self.raw_sightings.clear()
        self.preprocessed_events.clear()
        self.target_in_fov_intervals.clear()
        self.target_true_class.clear()
        self.interesting_targets.clear()
        self.polarized_targets.clear()
        self.circular_targets.clear()
        self.duration_s = 0.0

    # ----- queries ------------------------------------------------------
    def first_fov_entry(self, target_name: str) -> float | None:
        """Earliest time the target entered any eye's FoV, or ``None``."""
        intervals = self.target_in_fov_intervals.get(target_name, [])
        if not intervals:
            return None
        return min(t_enter for t_enter, _ in intervals)

    # ----- JSON serialisation (for offline plotting) --------------------
    def save_json(self, path: "str | Path") -> None:
        """
        Save this log to ``path`` as JSON.

        Raw sightings are NOT serialised (they're bulky and only used
        for diagnostics during the run). Everything that the metric
        calculators and the plotting helpers need is preserved:
        preprocessed events, FoV intervals, target metadata, duration.
        """
        def _event_to_dict(ev):
            return {
                "time": ev.time,
                "eye": ev.eye,
                "target_name": ev.target_name,
                "azimuth": ev.azimuth,
                "elevation": ev.elevation,
                "distance": ev.distance,
                "spectral_pattern": list(ev.spectral_pattern),
                "polarization_angle": ev.polarization_angle,
                "circular_handedness": ev.circular_handedness,
            }
        payload = {
            "duration_s": self.duration_s,
            "raw_stream_bytes_per_sample": self.raw_stream_bytes_per_sample,
            "event_bytes_per_event": self.event_bytes_per_event,
            "n_raw_sightings": len(self.raw_sightings),
            "preprocessed_events": [_event_to_dict(e)
                                    for e in self.preprocessed_events],
            "target_in_fov_intervals": {
                name: list(map(list, ivals))
                for name, ivals in self.target_in_fov_intervals.items()
            },
            "target_true_class": dict(self.target_true_class),
            "interesting_targets": sorted(self.interesting_targets),
            "polarized_targets": {n: float(a)
                                  for n, a in self.polarized_targets.items()},
            "circular_targets": dict(self.circular_targets),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_json(cls, path: "str | Path") -> "EventLog":
        """
        Load an :class:`EventLog` previously saved by :meth:`save_json`.

        Note that ``raw_sightings`` will be empty (we don't serialise
        them); the bandwidth metric for B1 is recomputed from the
        per-eye stream rate, so this is fine.
        """
        d = json.loads(Path(path).read_text())
        log = cls()
        log.duration_s = float(d["duration_s"])
        log.raw_stream_bytes_per_sample = int(d["raw_stream_bytes_per_sample"])
        log.event_bytes_per_event = int(d["event_bytes_per_event"])
        log.target_in_fov_intervals = {
            name: [tuple(ival) for ival in ivals]
            for name, ivals in d["target_in_fov_intervals"].items()
        }
        log.target_true_class = dict(d["target_true_class"])
        log.interesting_targets = set(d["interesting_targets"])
        log.polarized_targets = {n: float(a)
                                 for n, a in d["polarized_targets"].items()}
        log.circular_targets = dict(d.get("circular_targets", {}))
        for evd in d["preprocessed_events"]:
            if "spectral_pattern" in evd:
                evd["spectral_pattern"] = tuple(evd["spectral_pattern"])
            elif "dominant_class" in evd:
                from .world import SPECTRAL_CLASSES
                dom = evd.pop("dominant_class")
                idx = SPECTRAL_CLASSES.index(dom) if dom in SPECTRAL_CLASSES else 0
                pat = [0] * len(SPECTRAL_CLASSES)
                pat[idx] = 10
                evd["spectral_pattern"] = tuple(pat)
            evd.pop("spectral_strength", None)
            if "circular_handedness" not in evd:
                evd["circular_handedness"] = None
            log.preprocessed_events.append(PreprocessedEvent(**evd))
        return log


# ---------------------------------------------------------------------
# Metric 1 — Coverage
# ---------------------------------------------------------------------

def coverage(
    log: EventLog,
    identification_window_s: float = 0.5,
) -> float:
    """
    Fraction of *interesting* targets correctly identified.

    A target is "identified" iff at least one preprocessed event with
    a matching ``dominant_class`` was received within
    ``identification_window_s`` of the target first appearing in any
    eye's FoV.

    If the log has no interesting targets at all (e.g. a scene of pure
    clutter), returns ``1.0`` by convention.

    Returns a value in ``[0, 1]``. Higher is better.
    """
    if not log.interesting_targets:
        return 1.0

    correctly_identified = 0
    for name in log.interesting_targets:
        first_seen = log.first_fov_entry(name)
        if first_seen is None:
            # Target was never even in any eye's FoV — fundamentally
            # un-identifiable for this run.
            continue
        true_class = log.target_true_class.get(name)
        if true_class is None:
            continue
            
        from .world import SPECTRAL_CLASSES
        idx = SPECTRAL_CLASSES.index(true_class)
        
        deadline = first_seen + identification_window_s
        for ev in log.preprocessed_events:
            if ev.target_name != name:
                continue
            if ev.time > deadline:
                break  # events are time-ordered; nothing later will help
            
            pattern_arr = np.array(ev.spectral_pattern)
            if np.max(pattern_arr) > 0 and np.argmax(pattern_arr) == idx:
                correctly_identified += 1
                break

    return float(correctly_identified) / float(len(log.interesting_targets))


# ---------------------------------------------------------------------
# Metric 2 — Bandwidth
# ---------------------------------------------------------------------

def bandwidth_bytes_per_second(
    log: EventLog,
    baseline: str,
    raw_sample_rate_hz: float = 500.0,
    n_eyes: int = _N_EYES,
) -> float:
    """
    Average bandwidth from sensor to controller (bytes/s).

    For ``baseline="B1"``:
        A camera-like raw stream — the full multi-channel sample is
        sent every step **per eye**, regardless of whether anything is
        visible. Bandwidth is:

            n_eyes × raw_stream_bytes_per_sample × raw_sample_rate_hz

    For ``baseline="B2"`` or ``"B3"``:
        Sparse event stream. Bandwidth is the total event volume
        divided by simulation duration:

            event_bytes_per_event × n_events / duration_s

    Lower is better.

    Raises
    ------
    ValueError
        If ``baseline`` is not one of ``"B1"``, ``"B2"``, ``"B3"``.
    """
    if baseline not in _BASELINES:
        raise ValueError(
            f"baseline must be one of {_BASELINES}, got {baseline!r}"
        )

    if baseline == "B1":
        return float(n_eyes * log.raw_stream_bytes_per_sample
                     * raw_sample_rate_hz)

    # B2 / B3 / B3L — all use sparse event bandwidth
    if log.duration_s <= 0.0:
        return 0.0
    return float(log.event_bytes_per_event * len(log.preprocessed_events)
                 / log.duration_s)


# ---------------------------------------------------------------------
# Metric 3 — Polarization accuracy
# ---------------------------------------------------------------------

def _angular_distance_pi(a: float, b: float) -> float:
    """
    Smallest distance between two polarization angles ∈ [0, π).

    Polarization is a "doubled-angle" feature, so 0 and π are equivalent.
    The minimum distance is:

        d = min(|a - b|, π - |a - b|)
    """
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)


def polarization_accuracy(
    log: EventLog,
    tolerance_rad: float = np.deg2rad(15.0),
) -> float:
    """
    Fraction of polarized targets whose decoded polarization angle was
    within ``tolerance_rad`` of the true value.

    A target counts as correctly classified iff *any* event received
    about it had a non-None ``polarization_angle`` within tolerance
    (using the doubled-angle distance: 0° ≡ 180°). Targets that never
    produced a polarization-bearing event count as incorrect.

    If the log has no polarized targets at all, returns ``1.0`` by
    convention (no opportunity to fail).

    Returns a value in ``[0, 1]``. Higher is better. Only B3 with its
    active roll DOF should score meaningfully above zero here.
    """
    if not log.polarized_targets:
        return 1.0

    correct = 0
    for name, true_pol in log.polarized_targets.items():
        # Look at every event about this target; succeed on the first
        # within-tolerance decoded angle.
        for ev in log.preprocessed_events:
            if ev.target_name != name:
                continue
            if ev.polarization_angle is None:
                continue
            if _angular_distance_pi(ev.polarization_angle, true_pol) <= tolerance_rad:
                correct += 1
                break

    return float(correct) / float(len(log.polarized_targets))


# ---------------------------------------------------------------------
# Metric 3.5 — Circular Polarization
# ---------------------------------------------------------------------

def circular_polarization_accuracy(
    log: EventLog,
    identification_window_s: float = 0.5,
) -> float:
    """
    Fraction of *circularly polarized* targets whose handedness was
    correctly identified.
    """
    if not log.circular_targets:
        return 1.0

    correct = 0
    for name, true_handedness in log.circular_targets.items():
        first_seen = log.first_fov_entry(name)
        if first_seen is None:
            continue
            
        deadline = first_seen + identification_window_s
        for ev in log.preprocessed_events:
            if ev.target_name != name:
                continue
            if ev.time > deadline:
                break
            if ev.circular_handedness == true_handedness:
                correct += 1
                break

    return float(correct) / float(len(log.circular_targets))


# ---------------------------------------------------------------------
# Metric 4 — Response latency
# ---------------------------------------------------------------------

def median_response_latency_s(log: EventLog) -> float:
    """
    Median delay between a target first entering any eye's FoV and the
    controller receiving the first identifying event about it.

    "Identifying" here uses the same definition as :func:`coverage`:
    a preprocessed event with ``dominant_class == true_class``.

    Targets that were never identified contribute the full simulation
    duration (``log.duration_s``) as a *right-censored* value. This is
    the same convention survival analysis uses for "still alive at
    end-of-study".

    Returns the median latency in seconds. Lower is better. If there
    are no interesting targets at all, returns ``0.0``.
    """
    if not log.interesting_targets:
        return 0.0

    latencies: list[float] = []
    for name in log.interesting_targets:
        first_seen = log.first_fov_entry(name)
        if first_seen is None:
            # Never in FoV → censored at duration.
            latencies.append(log.duration_s)
            continue
        true_class = log.target_true_class.get(name)
        
        expected_pattern = None
        if true_class is not None:
            from .world import SPECTRAL_CLASSES
            idx = SPECTRAL_CLASSES.index(true_class)
            pat_list = [0] * len(SPECTRAL_CLASSES)
            pat_list[idx] = 10
            expected_pattern = tuple(pat_list)
            
        first_id_time: float | None = None
        for ev in log.preprocessed_events:
            if ev.target_name != name:
                continue
            if ev.time < first_seen:
                continue
            
            if expected_pattern is None:
                first_id_time = ev.time
                break
                
            pattern_arr = np.array(ev.spectral_pattern)
            if np.max(pattern_arr) > 0 and np.argmax(pattern_arr) == idx:
                first_id_time = ev.time
                break
        if first_id_time is None:
            latencies.append(log.duration_s)
        else:
            latencies.append(max(0.0, first_id_time - first_seen))

    return float(np.median(latencies))


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------

@dataclass
class MetricsReport:
    """Bundle of all four metrics for a single baseline run."""
    baseline: str                    # "B1", "B2", or "B3"
    coverage: float                  # ∈ [0, 1]
    bandwidth_bps: float             # bytes per second
    polarization_accuracy: float     # ∈ [0, 1]
    circular_polarization_accuracy: float # ∈ [0, 1]
    median_latency_s: float          # seconds

    def to_dict(self) -> dict:
        """JSON-serialisable summary, used by ``make_plots.py``."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write this report as JSON to ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "MetricsReport":
        """Load a previously-saved report from JSON."""
        d = json.loads(Path(path).read_text())
        if "circular_polarization_accuracy" not in d:
            d["circular_polarization_accuracy"] = 0.0
        return cls(**d)


def compute_all(log: EventLog, baseline: str) -> MetricsReport:
    """
    Run all four metric functions on ``log`` and return a single
    :class:`MetricsReport`. The standard one-call entry point used by
    the experiment scripts at the end of each run.
    """
    return MetricsReport(
        baseline=baseline,
        coverage=coverage(log),
        bandwidth_bps=bandwidth_bytes_per_second(log, baseline),
        polarization_accuracy=polarization_accuracy(log),
        circular_polarization_accuracy=circular_polarization_accuracy(log),
        median_latency_s=median_response_latency_s(log),
    )
