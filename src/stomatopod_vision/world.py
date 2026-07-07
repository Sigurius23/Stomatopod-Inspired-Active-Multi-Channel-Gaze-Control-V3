"""
world.py — scene management and target metadata
================================================

Loads the MuJoCo model and maintains a parallel Python-side dictionary of
*target metadata* (spectral class, polarization angle, etc.) that the
virtual sensor (``sensor.py``) consults when computing what each eye
"sees".

The metadata lives in Python rather than in the XML because:
    - XML colours are visual-only and would force us to encode information
      into RGB, which is awkward.
    - We will later want to randomise target attributes per experimental
      run, which is cleaner in Python.
    - Polarization angle has no natural visual representation.

Biological framing
------------------
The "spectral_class" attribute is an *engineering placeholder* for the
information a mantis-shrimp mid-band row would commit to early. It does
NOT correspond to species-specific spectral tuning curves (see
``docs/biological_disclaimer.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import mujoco
import numpy as np

# ---------------------------------------------------------------------
# Spectral and polarization placeholders
# ---------------------------------------------------------------------

#: Engineering placeholder for the eye's parallel sensing channels.
#: Each value is a *category label*, NOT a wavelength. The names happen
#: to evoke real photoreceptor classes but should not be read as a model
#: of the actual stomatopod spectral tuning.
SPECTRAL_CLASSES: tuple[str, ...] = (
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"
)


# ---------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TargetMeta:
    """
    All non-geometric metadata of a single target.

    Parameters
    ----------
    name :
        Body name in the MuJoCo XML (e.g. ``"target_R_1"``).
    spectral_class :
        Which of :data:`SPECTRAL_CLASSES` this target's energy is in.
    polarization_angle :
        Linear polarization angle in radians ∈ [0, π), or ``None`` for
        unpolarized targets.
    is_interesting :
        Whether this target should count toward the "coverage" metric.
        Typically ``True`` for UV + polarized targets, ``False`` for
        background clutter.
    """
    name: str
    spectral_class: str
    polarization_angle: float | None = None
    circular_handedness: Literal["left", "right"] | None = None
    is_interesting: bool = False

    def __post_init__(self) -> None:
        if self.spectral_class not in SPECTRAL_CLASSES:
            raise ValueError(
                f"spectral_class={self.spectral_class!r} is not one of "
                f"{SPECTRAL_CLASSES}"
            )
        if self.polarization_angle is not None:
            # Wrap into [0, π) to match the polarization-decoder convention
            wrapped = float(self.polarization_angle) % np.pi
            if wrapped != self.polarization_angle:
                # __setattr__ is blocked by frozen=True, use object.__setattr__
                object.__setattr__(self, "polarization_angle", wrapped)
        if self.circular_handedness not in (None, "left", "right"):
            raise ValueError("circular_handedness must be 'left', 'right', or None")


# ---------------------------------------------------------------------
# Target motion (Bonus #9)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TargetMotion:
    """Describe how a target body moves over time.

    Parameters
    ----------
    kind :
        ``"static"`` (no motion, default), ``"circular"`` (target orbits
        its rest position in a plane), or ``"linear"`` (target oscillates
        along ``axis`` like a sine wave).
    period_s :
        Time for one full circle / oscillation cycle (seconds).
        Ignored when ``kind == "static"``.
    amplitude_m :
        Orbit radius (for ``"circular"``) or peak displacement (for
        ``"linear"``), in metres. Ignored when ``kind == "static"``.
    axis :
        Three-vector. For ``"circular"`` motion this is the normal of
        the orbit plane (defaults to ``(0, 0, 1)`` — orbit in the xy
        plane). For ``"linear"`` motion this is the direction of
        oscillation (defaults to ``(1, 0, 0)``).
    phase_s :
        Phase offset (seconds) so multiple targets can move out of
        sync.
    """
    kind: str = "static"
    period_s: float = 1.0
    amplitude_m: float = 0.0
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    phase_s: float = 0.0

    _ALLOWED_KINDS: tuple[str, ...] = field(
        default=("static", "circular", "linear"), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in self._ALLOWED_KINDS:
            raise ValueError(
                f"motion.kind={self.kind!r} not in {self._ALLOWED_KINDS}"
            )
        if self.period_s <= 0:
            raise ValueError(f"motion.period_s must be > 0, got {self.period_s}")

    def displacement_at(self, time_s: float) -> np.ndarray:
        """Return the (3,) offset of this target from its rest position at ``time_s``.

        ``"static"`` returns the zero vector. ``"circular"`` returns a
        circle in the plane normal to ``axis``. ``"linear"`` returns a
        sinusoid along ``axis``.
        """
        if self.kind == "static" or self.amplitude_m == 0.0:
            return np.zeros(3, dtype=np.float64)

        omega = 2.0 * np.pi / float(self.period_s)
        t = float(time_s) - float(self.phase_s)

        if self.kind == "linear":
            ax = np.asarray(self.axis, dtype=np.float64)
            n = np.linalg.norm(ax)
            if n < 1e-12:
                return np.zeros(3, dtype=np.float64)
            return (self.amplitude_m * np.sin(omega * t)) * (ax / n)

        # kind == "circular": pick two basis vectors orthogonal to `axis`
        ax = np.asarray(self.axis, dtype=np.float64)
        n = np.linalg.norm(ax)
        if n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axn = ax / n
        # Build an orthonormal basis (u, v) spanning the plane normal to axn
        ref = np.array([1.0, 0.0, 0.0]) if abs(axn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axn, ref)
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(axn, u)
        v /= max(np.linalg.norm(v), 1e-12)
        return self.amplitude_m * (np.cos(omega * t) * u + np.sin(omega * t) * v)


class MovingTargetController:
    """Drive ``data.mocap_pos`` to animate target bodies over time.

    Usage
    -----
    Construct once after the model is loaded; call :meth:`step` from
    inside the simulation loop *before* ``mj_step``. Mocap bodies are
    kinematically driven (not physics-integrated), so this is the
    correct way to programmatically move them.

    Parameters
    ----------
    scene :
        A :class:`Scene` that was loaded from an XML containing
        ``mocap="true"`` target bodies. Non-mocap targets are silently
        ignored (so this controller is safe to wire into the default
        scene; it just has nothing to do there).
    motions :
        Mapping ``{target_name: TargetMotion}``. Targets not listed
        keep their rest position.
    """

    def __init__(
        self,
        scene: "Scene",
        motions: "dict[str, TargetMotion]",
    ) -> None:
        self.scene = scene
        self.motions = dict(motions)
        # Cache: (mocap_id, rest_pos, TargetMotion) for each animated target
        self._plan: list[tuple[int, np.ndarray, TargetMotion]] = []
        for name, motion in self.motions.items():
            bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"No body named {name!r} in this scene.")
            mocap_id = int(scene.model.body_mocapid[bid])
            if mocap_id < 0:
                # Target body is not declared mocap=true in the XML; can't move it.
                # We don't raise — this lets a single Python-side motion
                # spec work across both the static default scene and the
                # mocap-enabled moving scene.
                continue
            rest_pos = np.asarray(
                scene.data.mocap_pos[mocap_id], dtype=np.float64).copy()
            self._plan.append((mocap_id, rest_pos, motion))

    def step(self, time_s: float) -> None:
        """Update ``data.mocap_pos`` for every animated target."""
        for mocap_id, rest_pos, motion in self._plan:
            self.scene.data.mocap_pos[mocap_id] = rest_pos + motion.displacement_at(time_s)

    def reset(self) -> None:
        """Snap every animated target back to its rest position."""
        for mocap_id, rest_pos, _ in self._plan:
            self.scene.data.mocap_pos[mocap_id] = rest_pos


# ---------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------

class Scene:
    """
    Wraps a MuJoCo model+data plus the Python-side target metadata.

    Typical usage::

        scene = Scene.from_xml("models/stomatopod_eyes.xml")
        scene.reset()
        while scene.data.time < duration:
            mujoco.mj_step(scene.model, scene.data)
            for target in scene.targets:
                pos = scene.target_world_position(target.name)
                ...
    """

    #: The targets present in ``models/stomatopod_eyes.xml`` by default.
    #: Mirrors the body names + colours in the XML scene.
    DEFAULT_TARGETS: tuple[TargetMeta, ...] = (
        TargetMeta("target_R_1",     "C1",   None,       None,    False),
        TargetMeta("target_G_1",     "C4",   None,       None,    False),
        TargetMeta("target_B_1",     "C7",   None,       None,    False),
        TargetMeta("target_UVA_1",   "C10",  None,       None,    False),
        TargetMeta("target_UVB_1",   "C12",  None,       None,    False),
        TargetMeta("target_UVpol_1", "C10",  None,       "left",  True),  # Circular!
    )

    #: The targets present in models/stomatopod_eyes_hard.xml — the
    #: "B3-wins" variant. TEN interesting + polarized targets are
    #: scattered around the head, ALL outside the rest-pose FoV
    #: (±60° azimuth × ±5° elevation), so a fixed-gaze controller
    #: (B1/B2) sees zero of them. Decor + UV decoys are sprinkled
    #: inside / near the rest FoV to give the scheduler clutter to
    #: ignore.
    #:
    #: Polarization angles for the 10 interesting targets are spread
    #: evenly across [0, π) so the decoder cannot cheat by guessing
    #: one dominant angle.
    HARD_TARGETS: tuple[TargetMeta, ...] = (
        # ---- Decor (non-interesting; same four as default scene) ----
        TargetMeta("target_R_1",       "C1",   None,             None,    False),
        TargetMeta("target_G_1",       "C4",   None,             None,    False),
        TargetMeta("target_B_1",       "C7",   None,             None,    False),
        TargetMeta("target_UVA_1",     "C10",  None,             None,    False),
        # ---- UV decoys (non-interesting, no polarization) ----
        TargetMeta("target_UVA_dL",    "C10",  None,             None,    False),
        TargetMeta("target_UVA_dR",    "C10",  None,             None,    False),
        TargetMeta("target_UVB_dL",    "C12",  None,             None,    False),
        TargetMeta("target_UVB_dR",    "C12",  None,             None,    False),
        # ---- 10 interesting + polarized targets (all outside rest FoV) ----
        # Some are linearly polarized, some are circularly polarized
        TargetMeta("target_UVpol_FU",  "C10",  np.pi *  0 / 10,  None,    True),
        TargetMeta("target_UVpol_FD",  "C12",  np.pi *  1 / 10,  None,    True),
        TargetMeta("target_UVpol_LU",  "C10",  np.pi *  2 / 10,  None,    True),
        TargetMeta("target_UVpol_RU",  "C12",  None,             "right", True),
        TargetMeta("target_UVpol_LD",  "C10",  np.pi *  4 / 10,  None,    True),
        TargetMeta("target_UVpol_RD",  "C12",  np.pi *  5 / 10,  None,    True),
        TargetMeta("target_UVpol_FL",  "C10",  np.pi *  6 / 10,  None,    True),
        TargetMeta("target_UVpol_FR",  "C12",  None,             "left",  True),
        TargetMeta("target_UVpol_BL",  "C10",  np.pi *  8 / 10,  None,    True),
        TargetMeta("target_UVpol_BR",  "C12",  np.pi *  9 / 10,  None,    True),
    )

    #: Targets present in models/stomatopod_eyes_moving.xml — the
    #: moving-targets variant (bonus #9). Same metadata as the default
    #: scene (so coverage / polarization metrics are directly
    #: comparable); the per-target motion descriptors live in
    #: :attr:`MOVING_MOTIONS` and are applied by
    #: :class:`MovingTargetController`.
    MOVING_TARGETS: tuple[TargetMeta, ...] = DEFAULT_TARGETS

    #: Per-target choreography for the moving-targets scene. Four of
    #: the six targets orbit / oscillate; the other two stay put as a
    #: control. The amplitudes are deliberately moderate (≤ 0.3 m) so
    #: targets stay reachable within the gimbal joint limits and
    #: visible in the rest-pose FoV for at least part of each cycle.
    MOVING_MOTIONS: "dict[str, TargetMotion]" = {
        "target_R_1":     TargetMotion(kind="circular", period_s=4.0,
                                       amplitude_m=0.25,
                                       axis=(0.0, 0.0, 1.0)),
        "target_G_1":     TargetMotion(kind="circular", period_s=3.0,
                                       amplitude_m=0.20,
                                       axis=(0.0, 0.0, 1.0),
                                       phase_s=1.0),
        "target_B_1":     TargetMotion(kind="linear",   period_s=2.5,
                                       amplitude_m=0.25,
                                       axis=(1.0, 0.0, 0.0)),
        # The interesting one moves vertically so the eye must track
        # pitch as well as yaw.
        "target_UVpol_1": TargetMotion(kind="linear",   period_s=3.5,
                                       amplitude_m=0.20,
                                       axis=(0.0, 0.0, 1.0),
                                       phase_s=0.5),
        # target_UVA_1 and target_UVB_1 stay static (control).
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        targets: Iterable[TargetMeta] = DEFAULT_TARGETS,
    ) -> None:
        self.model = model
        self.data = data
        self.targets: tuple[TargetMeta, ...] = tuple(targets)

        # Cache body ids for every target so we can look up positions
        # without repeating a name search on every call.
        self._target_body_ids: dict[str, int] = {}
        for t in self.targets:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t.name)
            if bid < 0:
                raise ValueError(
                    f"Target body '{t.name}' not found in model. "
                    f"Either add it to the XML or remove it from the targets list."
                )
            self._target_body_ids[t.name] = bid

        # Make sure data.xpos is up-to-date before anyone queries it
        mujoco.mj_forward(self.model, self.data)

    # ----- construction --------------------------------------------------
    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path = "models/stomatopod_eyes.xml",
        targets: Iterable[TargetMeta] | None = None,
    ) -> "Scene":
        """Load a model from XML and return a :class:`Scene`.

        Parameters
        ----------
        xml_path :
            Path to the MuJoCo XML file. Default points at the canonical
            project model.
        targets :
            Iterable of :class:`TargetMeta` to attach to the scene. If
            ``None``, use :data:`DEFAULT_TARGETS`.
        """
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data  = mujoco.MjData(model)
        if targets is None:
            # Auto-pick the matching metadata tuple based on the XML file
            # name (MuJoCo doesn't expose the <mujoco model="..."> attribute
            # cleanly in Python). Falls back to DEFAULT_TARGETS for unknown
            # models, which is correct for the canonical scene.
            stem = Path(str(xml_path)).stem.lower()
            if stem == "stomatopod_eyes_hard":
                targets = cls.HARD_TARGETS
            elif stem == "stomatopod_eyes_moving":
                targets = cls.MOVING_TARGETS
            else:
                targets = cls.DEFAULT_TARGETS
        return cls(model, data, targets=targets)

    # ----- queries -------------------------------------------------------
    def target_world_position(self, name: str) -> np.ndarray:
        """Return the (3,) world position of the named target body."""
        if name not in self._target_body_ids:
            raise KeyError(f"No target named {name!r} in this scene.")
        return self.data.xpos[self._target_body_ids[name]].copy()

    def target_meta(self, name: str) -> TargetMeta:
        """Look up the :class:`TargetMeta` for a given target name."""
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(f"No target named {name!r} in this scene.")

    def interesting_target_names(self) -> tuple[str, ...]:
        """Names of targets with ``is_interesting=True``. Used by metrics."""
        return tuple(t.name for t in self.targets if t.is_interesting)

    def polarized_targets(self) -> dict[str, float]:
        """``{name: true_polarization_angle_rad}`` for polarized targets only."""
        return {
            t.name: t.polarization_angle
            for t in self.targets
            if t.polarization_angle is not None
        }

    def circularly_polarized_targets(self) -> dict[str, str]:
        """``{name: 'left'|'right'}`` for circularly polarized targets."""
        return {
            t.name: t.circular_handedness
            for t in self.targets
            if t.circular_handedness is not None
        }

    # ----- mutation ------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """
        Reset MuJoCo state to the model's keyframe (or zeros) and
        re-run a forward pass so positions are valid.

        ``seed`` is accepted for API symmetry with :func:`random_targets`
        but currently does nothing — the scene is deterministic once the
        XML is loaded.
        """
        del seed  # placeholder for future stochastic scene variations
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, dt_steps: int = 1) -> None:
        """Advance the simulation by ``dt_steps`` physics steps."""
        for _ in range(int(dt_steps)):
            mujoco.mj_step(self.model, self.data)


# ---------------------------------------------------------------------
# Random-scene generation (used by experiments for fresh seeds)
# ---------------------------------------------------------------------

def random_targets(
    n: int = 6,
    seed: int = 0,
    n_interesting: int = 1,
    spectral_classes: Iterable[str] = SPECTRAL_CLASSES,
    name_prefix: str = "target",
) -> tuple[TargetMeta, ...]:
    """
    Generate ``n`` targets with random spectral classes and polarization
    angles. ``n_interesting`` of them are flagged as ``is_interesting=True``.

    The returned :class:`TargetMeta` instances **do not** create matching
    MuJoCo bodies — the caller is responsible for ensuring the XML
    contains bodies with matching names. By default the names are
    ``target_001`` … ``target_<n>``; pass ``name_prefix`` to override.

    Examples
    --------
    >>> targets = random_targets(n=6, seed=42, n_interesting=2)
    >>> [t.name for t in targets]
    ['target_001', 'target_002', 'target_003',
     'target_004', 'target_005', 'target_006']
    >>> sum(t.is_interesting for t in targets)
    2

    Notes
    -----
    For now ``random_targets`` is mainly useful for experiments that
    permute target attributes between runs while keeping the body
    positions fixed. A future revision may also write/regenerate the
    XML with random positions.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= n_interesting <= n:
        raise ValueError(
            f"n_interesting must be in [0, n]={n}, got {n_interesting}"
        )

    spectral_list = tuple(spectral_classes)
    if not spectral_list:
        raise ValueError("spectral_classes must contain at least one entry")

    rng = np.random.default_rng(seed)

    # Decide which indices are "interesting" without replacement
    interesting_idx = set(rng.choice(n, size=n_interesting, replace=False)
                          .tolist()) if n_interesting > 0 else set()

    targets: list[TargetMeta] = []
    for i in range(n):
        cls = str(rng.choice(spectral_list))
        # ~half of all targets get a polarization angle (independent of "interesting")
        pol: float | None = None
        circ: Literal["left", "right"] | None = None
        if rng.random() < 0.5 or i in interesting_idx:
            if rng.random() < 0.2:
                circ = "left" if rng.random() < 0.5 else "right"
            else:
                pol = float(rng.uniform(0.0, np.pi))

        targets.append(TargetMeta(
            name=f"{name_prefix}_{i + 1:03d}",
            spectral_class=cls,
            polarization_angle=pol,
            circular_handedness=circ,
            is_interesting=(i in interesting_idx),
        ))

    return tuple(targets)
