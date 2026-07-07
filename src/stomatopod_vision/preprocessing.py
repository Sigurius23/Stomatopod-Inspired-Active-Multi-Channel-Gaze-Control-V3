"""
preprocessing.py — in-sensor preprocessing layer (Layer 2)
==========================================================

Compresses the raw multi-channel stream from :mod:`sensor` into a sparse
event stream that the active-sensing scheduler (:mod:`scheduler`) consumes.

Three operations, applied per :class:`~sensor.RawSighting`:

    1. **Mid-band channel reduction** — replace the per-spectral-class
       activation vector with ``(dominant_class, normalised_strength)``.
       Inspired by, NOT modelling, the rapid colour identification via
       temporal scanning described in Thoen et al. 2014.

    2. **Polarization decoding** — recover a single polarization angle
       from the 4 polarization-sensitive receptors using the standard
       vector-summation shortcut. Inspired by, NOT modelling, the
       receptor-orientation + active-roll mechanism of Daly et al. 2018.

    3. **Event encoding** — only emit a :class:`PreprocessedEvent` when
       the compressed reading changes meaningfully from the last one
       (per-target, per-eye). Inspired by neuromorphic event-based
       sensors (Gallego et al. 2020), NOT a literal model of retinal
       spiking.

Biological framing
------------------
This whole module is an *engineering abstraction* of "early channel
separation and bandwidth reduction at the sensor". See
``docs/biological_disclaimer.md`` for the inspires-vs-models breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .sensor import RawSighting, VirtualEye
from .world import SPECTRAL_CLASSES

# Maximum possible value of |Σ r_i exp(2i α_i)| / Σ r_i for the canonical
# 4 receptors at {0°, 45°, 90°, 135°}. Derived in test_preprocessing.py.
# We use it to normalise the confidence into [0, 1].
_MAX_POL_VECTOR_RATIO = 0.5


# ---------------------------------------------------------------------
# Output event type
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessedEvent:
    """
    Sparse, semantically meaningful sighting after Layer 2 has compressed
    a raw multi-channel reading.

    Attributes
    ----------
    time :
        MuJoCo time (seconds) at which the event was emitted.
    eye :
        Which eye fired the event.
    target_name :
        Identifier of the target this event is about.
    azimuth, elevation :
        Position of the target in the eye's local frame (radians).
    distance :
        Distance from eye centre to target centre (metres).
    dominant_class :
        The spectral class with the highest activation (e.g. ``"UV_A"``).
    spectral_strength :
        Normalised strength of ``dominant_class`` ∈ [0, 1].
    polarization_angle :
        Decoded polarization angle (radians), or ``None`` if unpolarized
        or below the polarization-confidence threshold.
    """
    time: float
    eye: Literal["L", "R"]
    target_name: str
    azimuth: float
    elevation: float
    distance: float
    spectral_pattern: tuple[int, ...]
    polarization_angle: float | None
    circular_handedness: Literal["left", "right"] | None


# ---------------------------------------------------------------------
# 4.1 — Mid-band channel reduction
# ---------------------------------------------------------------------

def midband_channel_reduce(
    midband_activations: np.ndarray,
) -> tuple[int, ...]:
    """
    Convert the mid-band response vector into a discretized pattern 
    (look-up table key).

    Parameters
    ----------
    midband_activations :
        Length-N vector of per-class activations.

    Returns
    -------
    spectral_pattern :
        A tuple of integers representing the discretized activation pattern, 
        binned into 10 levels.
    """
    a = np.asarray(midband_activations, dtype=float)
    if len(a) != len(SPECTRAL_CLASSES):
        raise ValueError(f"Expected {len(SPECTRAL_CLASSES)} midband activations, got {len(a)}")
    a_clipped = np.clip(a, 0.0, None)
    total = float(np.sum(a_clipped))
    if total <= 0.0:
        return tuple([0] * len(a))
    
    # Normalize and bin into 10 discrete levels
    normalized = a_clipped / total
    binned = np.round(normalized * 10).astype(int)
    return tuple(binned.tolist())


# ---------------------------------------------------------------------
# 4.2 — Polarization decoding
# ---------------------------------------------------------------------

def polarization_decode(
    receptor_responses: np.ndarray,
    receptor_angles_world: np.ndarray,
    confidence_threshold: float = 0.1,
) -> float | None:
    """
    Decode the polarization angle from 4 oriented receptors using the
    standard vector-summation shortcut.

    Parameters
    ----------
    receptor_responses :
        Length-4 array of receptor activations (one per oriented receptor).
    receptor_angles_world :
        Length-4 array of receptor orientations in WORLD coordinates
        (i.e. already offset by the eye's roll angle).
    confidence_threshold :
        Normalised confidence threshold ∈ [0, 1]. The raw confidence is
        ``|Σ r_i exp(2i α_i)| / Σ r_i``, which has a theoretical maximum
        of ``0.5`` for the canonical 4-receptor setup. We normalise by
        that max so the threshold is on a friendly [0, 1] scale.
        If confidence is below threshold, returns ``None``.

    Returns
    -------
    polarization_angle :
        Decoded angle in radians ∈ [0, π), or ``None`` if the readings
        do not support a confident decoding.

    Notes
    -----
    The vector-sum estimator is:

        .. math::

            \\theta_\\text{pol}
                = \\frac{1}{2}\\,\\angle
                  \\sum_i r_i \\, \\exp(2 i \\alpha_i)

    where ``r_i`` are receptor responses and ``α_i`` are their world
    orientations. The factor of 2 reflects polarization being a
    "doubled-angle" feature (180° ≡ 0°).

    This inverts the forward model in :class:`~sensor.VirtualEye`:

        .. math::

            r_i = a \\cos^2(\\theta_\\text{target} - \\alpha_i)

    by using the identity :math:`\\cos^2 x = (1 + \\cos 2x)/2`. With
    equally spaced receptors, the DC term cancels and only the
    polarization phase survives.

    # CITE: Real stomatopod decoding depends on receptor microvilli
    #       geometry and *active* eye roll (Daly et al. 2018). The
    #       vector sum captures the information flow ("rolling reveals
    #       polarization"), NOT the optical mechanism.
    """
    r = np.asarray(receptor_responses, dtype=float)
    a = np.asarray(receptor_angles_world, dtype=float)
    if r.shape != a.shape:
        raise ValueError(
            f"receptor_responses {r.shape} and receptor_angles_world "
            f"{a.shape} must have the same shape"
        )
    if r.size == 0:
        return None

    total = float(np.sum(np.clip(r, 0.0, None)))
    if total < 1e-12:
        # All receptors silent → no information at all.
        return None

    z = np.sum(r * np.exp(2j * a))
    raw_confidence = float(np.abs(z) / total)
    normalised_confidence = raw_confidence / _MAX_POL_VECTOR_RATIO
    if normalised_confidence < confidence_threshold:
        return None

    angle = float(np.angle(z) / 2.0)
    # Wrap into [0, π) — polarization is a "doubled-angle" feature.
    return angle % np.pi


def circular_decode(
    receptor_responses: np.ndarray,
    confidence_threshold: float = 0.1,
) -> Literal["left", "right"] | None:
    """
    Decode circular-polarization handedness from the 4 *quarter-wave*
    receptors (``RawSighting.circular_responses``, the midband rows-5/6
    analog) — NOT the bare linear receptors.

    Behind the retarder, circular light produces a strong split between the
    45° (index 1) and 135° (index 3) receptors whose sign encodes
    handedness (left → 45° peak, right → 135° peak), while *any* linear
    angle produces ``r[1] == r[3]`` (zero split). So the ``r[1] - r[3]``
    contrast cleanly separates circular from linear: linear light decodes
    to ``None`` regardless of its angle.
    """
    r = np.asarray(receptor_responses, dtype=float)
    if r.size != 4:
        return None
        
    total = float(np.sum(np.clip(r, 0.0, None)))
    if total < 1e-12:
        return None
        
    # r[1] is 45 deg, r[3] is 135 deg
    diff = (r[1] - r[3]) / total
    
    # We use _MAX_POL_VECTOR_RATIO to normalise the difference just like in polarization_decode
    normalised_diff = diff / _MAX_POL_VECTOR_RATIO
    
    if normalised_diff > confidence_threshold:
        return "left"
    elif normalised_diff < -confidence_threshold:
        return "right"
    return None


# ---------------------------------------------------------------------
# Convenience: build an event directly from a raw sighting
# ---------------------------------------------------------------------

def raw_to_event(
    raw: RawSighting,
    time_now: float,
    roll_angle: float,
    pol_confidence_threshold: float = 0.1,
) -> PreprocessedEvent:
    """
    Run channel reduction + polarization decoding on a single raw
    sighting and pack the result into a :class:`PreprocessedEvent`.

    This is the per-sighting half of the pipeline; :class:`EventEncoder`
    decides whether to *emit* the event.

    Parameters
    ----------
    raw :
        Raw sighting from :class:`VirtualEye`.
    time_now :
        Current simulation time, recorded in the event.
    roll_angle :
        Current roll of the eye that produced ``raw``. Used to recover
        the world-frame receptor orientations.
    pol_confidence_threshold :
        Forwarded to :func:`polarization_decode`.
    """
    spectral_pattern = midband_channel_reduce(raw.midband_activations)
    receptor_world_angles = (
        VirtualEye.POLARIZATION_RECEPTOR_ANGLES_RAD + roll_angle
    )
    polarization_angle = polarization_decode(
        raw.polarization_responses,
        receptor_world_angles,
        confidence_threshold=pol_confidence_threshold,
    )
    circular_handedness = circular_decode(
        raw.circular_responses,
        confidence_threshold=pol_confidence_threshold,
    )
    return PreprocessedEvent(
        time=float(time_now),
        eye=raw.eye,
        target_name=raw.target_name,
        azimuth=raw.azimuth,
        elevation=raw.elevation,
        distance=raw.distance,
        spectral_pattern=spectral_pattern,
        polarization_angle=polarization_angle,
        circular_handedness=circular_handedness,
    )


# ---------------------------------------------------------------------
# 4.3 — Event encoder (per-target, per-eye)
# ---------------------------------------------------------------------

class EventEncoder:
    """
    Emits a :class:`PreprocessedEvent` only when the compressed reading
    for a (eye, target) pair changes by more than some threshold from
    the last emission. Each (eye, target) gets its own internal state.

    Parameters
    ----------
    azimuth_threshold :
        Angular change (radians) in either azimuth or elevation that
        triggers a new emission.
    strength_threshold :
        Change in ``spectral_strength`` (∈[0,1]) that triggers a new
        emission, even if angular change is small.
    polarization_threshold :
        Change in decoded polarization angle (radians) that triggers
        a new emission.

    # CITE: Engineering analogy for sparse, event-driven sensing
    #       (Gallego et al. 2020, IEEE PAMI). We do NOT claim that
    #       real stomatopod retinae emit threshold-crossing spikes
    #       in this form.
    """

    def __init__(
        self,
        azimuth_threshold: float = np.deg2rad(2.0),
        strength_threshold: float = 0.1,
        polarization_threshold: float = np.deg2rad(10.0),
    ) -> None:
        self.azimuth_threshold = float(azimuth_threshold)
        self.strength_threshold = float(strength_threshold)
        self.polarization_threshold = float(polarization_threshold)
        self._last_event: dict[tuple[str, str], PreprocessedEvent] = {}

    # ----- change-detection logic ----------------------------------------
    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Wrap an angular difference into [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _significant_change(
        self,
        last: PreprocessedEvent,
        candidate: PreprocessedEvent,
    ) -> bool:
        """True if ``candidate`` differs enough from ``last`` to emit.

        We emit on ANY of the following:
        - azimuth or elevation moved by more than azimuth_threshold,
        - dominant_class changed,
        - spectral_strength changed by more than strength_threshold,
        - polarization availability changed (None ↔ value),
        - polarization angle moved by more than polarization_threshold
          (when both events have a decoded angle).
        """
        if abs(candidate.azimuth - last.azimuth) > self.azimuth_threshold:
            return True
        if abs(candidate.elevation - last.elevation) > self.azimuth_threshold:
            return True
        if candidate.spectral_pattern != last.spectral_pattern:
            return True
        if candidate.circular_handedness != last.circular_handedness:
            return True
        # Polarization comparison
        if (last.polarization_angle is None) != (candidate.polarization_angle is None):
            return True  # availability flipped
        if last.polarization_angle is not None and candidate.polarization_angle is not None:
            # Polarization wraps at π, so the meaningful distance is the
            # minimum of |Δ| and π - |Δ|. Equivalently, wrap (2 Δ) to [-π, π]
            # then divide by 2.
            d = self._wrap_pi(2.0 * (candidate.polarization_angle - last.polarization_angle)) / 2.0
            if abs(d) > self.polarization_threshold:
                return True
        return False

    def encode(
        self,
        raw: RawSighting,
        time_now: float,
        roll_angle: float,
        pol_confidence_threshold: float = 0.1,
    ) -> PreprocessedEvent | None:
        """
        Process a single raw sighting. Returns either a new event or
        ``None`` if nothing significant has changed since the last
        emission for this (eye, target) pair.

        Notes
        -----
        The first sighting for any (eye, target) pair *always* produces
        an event — that's the "this target just entered the FoV" signal
        the scheduler needs.
        """
        candidate = raw_to_event(
            raw,
            time_now=time_now,
            roll_angle=roll_angle,
            pol_confidence_threshold=pol_confidence_threshold,
        )
        key = (raw.eye, raw.target_name)
        last = self._last_event.get(key)
        if last is None or self._significant_change(last, candidate):
            self._last_event[key] = candidate
            return candidate
        return None

    def reset(self) -> None:
        """Clear all internal state. Call between experiments."""
        self._last_event.clear()


# ---------------------------------------------------------------------
# Convenience: full Layer-2 pipeline
# ---------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Bundles channel reduction + polarization decoding + event encoding
    into a single callable. This is the object an experiment will use
    in B2 and B3 baselines.

    Parameters
    ----------
    event_encoder :
        Optional pre-configured :class:`EventEncoder`. If ``None``, a
        default-configured one is created.
    pol_confidence_threshold :
        Forwarded to :func:`polarization_decode`.
    """

    def __init__(
        self,
        event_encoder: EventEncoder | None = None,
        pol_confidence_threshold: float = 0.1,
    ) -> None:
        self.event_encoder = event_encoder or EventEncoder()
        self.pol_confidence_threshold = float(pol_confidence_threshold)

    def step(
        self,
        raw_sightings: Iterable[RawSighting],
        time_now: float,
        roll_angles: dict[Literal["L", "R"], float],
    ) -> list[PreprocessedEvent]:
        """
        Apply the full Layer-2 pipeline to a list of raw sightings.

        Parameters
        ----------
        raw_sightings :
            All raw sightings produced by both eyes this step.
        time_now :
            Current simulation time.
        roll_angles :
            ``{"L": roll_L, "R": roll_R}`` — current roll angles of each
            eye, needed to decode polarization in world coordinates.

        Returns
        -------
        events :
            List (possibly empty) of new events emitted at ``time_now``.
        """
        out: list[PreprocessedEvent] = []
        for raw in raw_sightings:
            if raw.eye not in roll_angles:
                raise KeyError(
                    f"roll_angles missing eye {raw.eye!r}; got {list(roll_angles)}"
                )
            ev = self.event_encoder.encode(
                raw,
                time_now=time_now,
                roll_angle=roll_angles[raw.eye],
                pol_confidence_threshold=self.pol_confidence_threshold,
            )
            if ev is not None:
                out.append(ev)
        return out

    def reset(self) -> None:
        """Reset all internal state (encoder caches, etc.)."""
        self.event_encoder.reset()
