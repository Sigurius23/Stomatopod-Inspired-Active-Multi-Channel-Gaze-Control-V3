"""
sensor.py — virtual receptor model (Layer 1)
============================================

Computes "what each eye sees" without any actual image rendering. For each
eye and each scene target, we compute:

    1. Whether the target falls inside the eye's narrow mid-band field of
       view (anisotropic: e.g. ±5° in pitch, ±60° in yaw).
    2. If so, the raw multi-channel receptor activations the eye would
       produce for that target — including both peripheral (broad-band)
       and mid-band (per-spectral-class + polarization) channels.

The output is a list of :class:`RawSighting` dataclasses, one per target
currently visible to the eye. Layer 2 (``preprocessing.py``) consumes
this list and compresses it into sparse events.

Geometry conventions
--------------------
The eye-local frame is defined by:
    - forward (+Y_eye)   : unit vector from `eye_X_center` to `eye_X_axis`
    - right   (+X_eye)   : normalize(forward × world_up), with world_up = +Z
    - up      (+Z_eye)   : right × forward

For a world point p, with eye centre at c:
    v = p − c
    x_local = v · right
    y_local = v · forward
    z_local = v · up
    azimuth   = atan2(x_local, y_local)        # +ve = target is to the right
    elevation = atan2(z_local, sqrt(x_local² + y_local²))   # +ve = above

This frame deliberately *ignores the eye's roll* — the FoV is rotationally
symmetric about the forward axis. The roll angle is read separately and
used only by the polarization-receptor responses.

Singularity note: when the eye points straight up (forward ≈ world_up),
the right-vector becomes ill-defined. Our XML constrains pitch to ±60°
so this is never reached in practice, but we guard against it with an
epsilon check.

Biological framing
------------------
This module is the engineering analog of the photoreceptor + earliest
neural readout stage of a real stomatopod eye. We *do not* model:
    - actual photon collection or ommatidial optics,
    - species-specific receptor counts or tuning curves,
    - the temporal scanning that mantis shrimp use to identify colour
      (Thoen et al. 2014) — see ``docs/biological_disclaimer.md``.

The sensor here is a *geometry-only* abstraction of the *information geometry*
of a real compound eye: a narrow horizontal stripe of categorical channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from .world import SPECTRAL_CLASSES, Scene

# Numerical floor for the cross-product when computing the eye's right vector
_RIGHT_VECTOR_EPS = 1e-6

# Reference distance for the inverse-square distance attenuation (metres).
# Targets at exactly this distance see attenuation = 0.5; closer = brighter.
_DISTANCE_REFERENCE_M = 1.0


# ---------------------------------------------------------------------
# Field-of-view geometry
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MidbandFOV:
    """
    Anisotropic field of view for one eye, mimicking the elongated shape
    of the real stomatopod mid-band (much wider in azimuth than in pitch).

    Parameters
    ----------
    yaw_half_angle :
        Half-width of the FoV in the yaw direction (radians).
    pitch_half_angle :
        Half-width of the FoV in the pitch direction (radians).
    """
    yaw_half_angle: float = np.deg2rad(60.0)
    pitch_half_angle: float = np.deg2rad(5.0)


# ---------------------------------------------------------------------
# A single "raw" sensor reading
# ---------------------------------------------------------------------

@dataclass
class RawSighting:
    """
    Raw multi-channel data for one target currently inside one eye's FoV.

    Attributes
    ----------
    target_name :
        Body name of the target (from :class:`TargetMeta`).
    eye :
        Which eye produced the sighting (``"L"`` or ``"R"``).
    azimuth :
        Yaw offset from the eye's forward axis (radians).
    elevation :
        Pitch offset from the eye's forward axis (radians).
    distance :
        Distance from eye centre to target centre (metres).
    peripheral_intensity :
        Single broad-band "intensity" scalar from the peripheral row.
    midband_activations :
        Per-spectral-class activations (vector of length
        ``len(SPECTRAL_CLASSES)``). Used by Layer 2's channel reducer.
    polarization_responses :
        Activations of the 4 *bare* linear-polarization receptors at
        orientations {0°, 45°, 90°, 135°} *in the eye's current rolled
        frame* (the analog of midband rows without a retarder). Used by
        Layer 2's linear-polarization decoder.
    circular_responses :
        Activations of the 4 polarization receptors that sit *behind a
        quarter-wave retarder* (the analog of midband rows 5-6). The
        retarder converts circular light into a strong linear signal at
        ±45° (roll-invariant, so handedness survives eye torsion) while
        converting linear light into a flat 45°/135° response — this is
        what lets Layer 2 tell circular from linear polarization.
        # CITE: Chiou et al. 2008 — rows 5/6 quarter-wave-plate circular
        #       polarization vision in stomatopods.

    Note
    ----
    This dataclass holds only *measurements*. Ground-truth handedness is
    NOT carried here — it is decoded downstream from ``circular_responses``
    by :func:`preprocessing.circular_decode`.
    """
    target_name: str
    eye: Literal["L", "R"]
    azimuth: float
    elevation: float
    distance: float
    peripheral_intensity: float
    midband_activations: np.ndarray              # shape (len(SPECTRAL_CLASSES),)
    polarization_responses: np.ndarray            # shape (4,) — bare linear
    #: shape (4,) — behind a quarter-wave retarder. Defaults to the
    #: unpolarized response so hand-built sightings need not specify it.
    circular_responses: np.ndarray = field(
        default_factory=lambda: 0.5 * np.ones(4, dtype=float))


# ---------------------------------------------------------------------
# The virtual eye
# ---------------------------------------------------------------------

class VirtualEye:
    """
    Per-eye sensor that reads MuJoCo state and produces :class:`RawSighting`
    objects for visible targets.

    Parameters
    ----------
    eye :
        ``"L"`` or ``"R"`` — must match the site names in the XML
        (``eye_L_center`` / ``eye_L_axis`` etc.) and the joint name
        ``eye_X_roll``.
    scene :
        The :class:`Scene` containing the model + targets.
    fov :
        Field-of-view geometry. Defaults to the elongated mid-band FoV.
    world_up :
        Reference "up" vector used to disambiguate the eye-local frame.
        Default is world +Z, which is correct for the project's XML.
    """

    #: Receptor orientations in the eye's body-local frame, before roll.
    #: After applying roll, world-frame angles become
    #: ``POLARIZATION_RECEPTOR_ANGLES_RAD + roll_angle``.
    POLARIZATION_RECEPTOR_ANGLES_RAD: np.ndarray = np.deg2rad(
        np.array([0.0, 45.0, 90.0, 135.0])
    )

    def __init__(
        self,
        eye: Literal["L", "R"],
        scene: Scene,
        fov: MidbandFOV = MidbandFOV(),
        world_up: np.ndarray | None = None,
        *,
        receptor_noise_std: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        """Construct a virtual eye.

        Parameters (in addition to those above):
            receptor_noise_std :
                Per-receptor additive Gaussian noise standard deviation,
                applied to BOTH the midband activations AND the
                polarization responses. ``0.0`` (the default) reproduces
                the deterministic behaviour of every test + figure in
                the project. The noise-free amplitude at a 1 m distance
                is 1.0 and falls as 1/d², so a noise std of 0.02 is
                "2 % at 1 m".
            noise_seed :
                Seed for the noise RNG. Each eye gets an independent
                stream so left/right noise is uncorrelated.
        """
        if eye not in ("L", "R"):
            raise ValueError(f"eye must be 'L' or 'R', got {eye!r}")
        self.eye = eye
        self.receptor_noise_std = float(receptor_noise_std)
        self._noise_rng = np.random.default_rng(noise_seed)
        self.scene = scene
        self.fov = fov
        self.world_up = (np.array([0.0, 0.0, 1.0])
                         if world_up is None else np.asarray(world_up, float))
        if self.world_up.shape != (3,):
            raise ValueError(f"world_up must be shape (3,), got {self.world_up.shape}")
        self.world_up = self.world_up / np.linalg.norm(self.world_up)

        # Cache MuJoCo ids for this eye
        m = scene.model
        self._site_center_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_center")
        self._site_axis_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, f"eye_{eye}_axis")
        if self._site_center_id < 0 or self._site_axis_id < 0:
            raise ValueError(
                f"Sites 'eye_{eye}_center' and/or 'eye_{eye}_axis' not "
                f"found in model — did the XML rename them?")

        roll_joint = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"eye_{eye}_roll")
        if roll_joint < 0:
            raise ValueError(f"Joint 'eye_{eye}_roll' not found in model.")
        self._roll_qpos_idx = int(m.jnt_qposadr[roll_joint])

    # ----- geometry queries ---------------------------------------------
    def center_position(self) -> np.ndarray:
        """World position of the eye centre (the gimbal pivot)."""
        return self.scene.data.site_xpos[self._site_center_id].copy()

    def forward_vector(self) -> np.ndarray:
        """
        Unit vector pointing along the eye's current forward axis,
        in world coordinates, derived from the two XML sites.
        """
        v = (self.scene.data.site_xpos[self._site_axis_id]
             - self.scene.data.site_xpos[self._site_center_id])
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            # Should never happen — center and axis sites are at fixed
            # offset in the eye body. Defensive return.
            return np.array([0.0, 1.0, 0.0])
        return v / n

    def roll_angle(self) -> float:
        """Current value of the eye's roll joint (radians)."""
        return float(self.scene.data.qpos[self._roll_qpos_idx])

    def _local_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build an orthonormal eye-local frame in world coordinates.

        Returns
        -------
        (right, forward, up) :
            World-frame unit vectors defining the eye-local axes
            (+X, +Y, +Z respectively, before roll is applied).
        """
        forward = self.forward_vector()
        # right = normalize(forward × world_up); falls back when degenerate
        right = np.cross(forward, self.world_up)
        n = float(np.linalg.norm(right))
        if n < _RIGHT_VECTOR_EPS:
            # forward is (nearly) parallel to world_up — pick an arbitrary
            # perpendicular direction. Should not happen with the XML's
            # ±60° pitch limit, but we guard against it.
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(forward @ arb)) > 0.999:
                arb = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, arb)
            n = float(np.linalg.norm(right))
        right = right / n
        up = np.cross(right, forward)   # already unit length
        return right, forward, up

    # ----- per-target computation --------------------------------------
    def relative_angles(self, target_pos: np.ndarray) -> tuple[float, float, float]:
        """
        Return (azimuth, elevation, distance) of ``target_pos`` relative
        to the eye, in the eye's local frame.

        - ``azimuth``  : positive = target is to the eye's right.
        - ``elevation``: positive = target is above the eye's forward axis.
        - ``distance`` : Euclidean distance from eye centre to target.
        """
        right, forward, up = self._local_frame()
        v = np.asarray(target_pos, float) - self.center_position()
        distance = float(np.linalg.norm(v))
        if distance < 1e-12:
            return 0.0, 0.0, 0.0
        x_local = float(v @ right)
        y_local = float(v @ forward)
        z_local = float(v @ up)
        azimuth = float(np.arctan2(x_local, y_local))
        elevation = float(np.arctan2(
            z_local, float(np.hypot(x_local, y_local))))
        return azimuth, elevation, distance

    def is_in_fov(self, azimuth: float, elevation: float) -> bool:
        """True if (azimuth, elevation) lies inside the mid-band FoV."""
        # Also require the target to be in front of the eye (|azimuth| < π/2).
        if abs(azimuth) > self.fov.yaw_half_angle:
            return False
        if abs(elevation) > self.fov.pitch_half_angle:
            return False
        return True

    def _distance_attenuation(self, distance: float) -> float:
        """Inverse-square attenuation factor in (0, 1]."""
        return 1.0 / (1.0 + (distance / _DISTANCE_REFERENCE_M) ** 2)

    def _midband_activations(
        self,
        spectral_class: str,
        attenuation: float,
    ) -> np.ndarray:
        """One-hot vector concentrated at ``spectral_class``, scaled."""
        idx = SPECTRAL_CLASSES.index(spectral_class)
        out = np.zeros(len(SPECTRAL_CLASSES), dtype=float)
        out[idx] = attenuation
        return out

    def _linear_pol_responses(
        self,
        target_polarization_rad: float | None,
        circular_handedness: Literal["left", "right"] | None,
        attenuation: float,
    ) -> np.ndarray:
        """
        Response of the 4 *bare* linear-polarization receptors (no retarder).

        - Linear light: response = cos²(θ_target − θ_receptor_world).
        - Circular light: response = 0.5 (a circular field has no linear
          axis, so a bare analyzer reads half in every orientation).
        - Unpolarized light: response = 0.5 (Malus averaged over all angles).
        """
        if target_polarization_rad is None:
            # Covers BOTH unpolarized and circularly polarized targets:
            # neither presents a linear axis to a bare analyzer.
            return 0.5 * attenuation * np.ones(4, dtype=float)
        receptor_world_angles = (self.POLARIZATION_RECEPTOR_ANGLES_RAD
                                 + self.roll_angle())
        return attenuation * np.cos(target_polarization_rad
                                    - receptor_world_angles) ** 2

    def _quarterwave_pol_responses(
        self,
        target_polarization_rad: float | None,
        circular_handedness: Literal["left", "right"] | None,
        attenuation: float,
    ) -> np.ndarray:
        """
        Response of the 4 receptors sitting *behind a quarter-wave retarder*
        (the midband rows-5/6 analog), at {0°, 45°, 90°, 135°}.

        A quarter-wave plate (fast axis fixed in the eye frame) swaps the
        roles of linear and circular light:

        - **Circular** light becomes linear at ±45°, so the array shows a
          strong, roll-invariant split between the 45° and 135° receptors —
          and the *sign* of that split encodes handedness. We adopt the
          convention: left → peak at 45° (index 1), right → peak at 135°
          (index 3).
        - **Linear** light at eye-local angle θ' becomes
          ``[cos²θ', 0.5, sin²θ', 0.5]`` — crucially the 45° and 135°
          receptors are *always equal* (0.5) for any linear angle, so the
          handedness split is zero. That is what makes circular reliably
          separable from linear (see :func:`preprocessing.circular_decode`).
        """
        if circular_handedness == "left":
            return attenuation * np.array([0.5, 1.0, 0.5, 0.0])
        if circular_handedness == "right":
            return attenuation * np.array([0.5, 0.0, 0.5, 1.0])
        if target_polarization_rad is None:
            return 0.5 * attenuation * np.ones(4, dtype=float)
        # Linear light through the retarder: 45°/135° receptors read 0.5.
        theta_local = target_polarization_rad - self.roll_angle()
        c2 = float(np.cos(theta_local) ** 2)
        s2 = float(np.sin(theta_local) ** 2)
        return attenuation * np.array([c2, 0.5, s2, 0.5])

    def raw_activations_for(self, target_name: str) -> RawSighting | None:
        """
        Compute the full :class:`RawSighting` for a single target, or
        ``None`` if it is outside the FoV.

        - ``midband_activations`` is a one-hot-ish vector concentrated on
          the target's spectral_class, attenuated by distance.
        - ``polarization_responses`` (bare linear analyzers) is computed
          from the target's linear polarization angle, modulated by the
          eye's current roll, as ``cos²(θ_target − receptor_angle_world)``;
          circular and unpolarized targets read a flat 0.5.
        - ``circular_responses`` (behind a quarter-wave retarder) carries a
          strong, roll-invariant handedness signature for circular targets
          and a handedness-free response for linear ones.
        """
        meta = self.scene.target_meta(target_name)
        target_pos = self.scene.target_world_position(target_name)
        azimuth, elevation, distance = self.relative_angles(target_pos)
        if not self.is_in_fov(azimuth, elevation):
            return None

        attenuation = self._distance_attenuation(distance)
        midband = self._midband_activations(meta.spectral_class, attenuation)

        # Two physically-distinct polarization channels (see the methods and
        # the RawSighting docstring): a bare linear analyzer bank and a
        # quarter-wave-retarder bank. Together they separate linear from
        # circular polarization; neither can on its own.
        polarization = self._linear_pol_responses(
            meta.polarization_angle, meta.circular_handedness, attenuation)
        circular = self._quarterwave_pol_responses(
            meta.polarization_angle, meta.circular_handedness, attenuation)

        # Optional: additive Gaussian noise per receptor, clipped at 0
        # to keep responses non-negative (negative cos² is unphysical
        # and would crash the vector-sum decoder's confidence ratio).
        if self.receptor_noise_std > 0.0:
            midband = midband + self._noise_rng.normal(
                0.0, self.receptor_noise_std, size=midband.shape)
            midband = np.clip(midband, 0.0, None)
            polarization = polarization + self._noise_rng.normal(
                0.0, self.receptor_noise_std, size=polarization.shape)
            polarization = np.clip(polarization, 0.0, None)
            circular = circular + self._noise_rng.normal(
                0.0, self.receptor_noise_std, size=circular.shape)
            circular = np.clip(circular, 0.0, None)

        return RawSighting(
            target_name=target_name,
            eye=self.eye,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            peripheral_intensity=attenuation,
            midband_activations=midband,
            polarization_responses=polarization,
            circular_responses=circular,
        )

    # ----- batched ------------------------------------------------------
    def step(self) -> list[RawSighting]:
        """
        Return the list of :class:`RawSighting` for every visible target,
        as of the current MuJoCo state.
        """
        out: list[RawSighting] = []
        for t in self.scene.targets:
            s = self.raw_activations_for(t.name)
            if s is not None:
                out.append(s)
        return out


# ---------------------------------------------------------------------
# Convenience: build both eyes
# ---------------------------------------------------------------------

def make_eye_pair(
    scene: Scene,
    fov: MidbandFOV = MidbandFOV(),
    *,
    receptor_noise_std: float = 0.0,
    noise_seed: int = 0,
) -> tuple[VirtualEye, VirtualEye]:
    """Construct ``(eye_L, eye_R)`` virtual eyes attached to ``scene``.

    Both eyes get the same ``receptor_noise_std`` but **independent**
    RNG streams seeded from ``noise_seed`` and ``noise_seed + 1`` so
    the left- and right-eye noise are uncorrelated.
    """
    return (
        VirtualEye("L", scene, fov=fov,
                   receptor_noise_std=receptor_noise_std,
                   noise_seed=noise_seed),
        VirtualEye("R", scene, fov=fov,
                   receptor_noise_std=receptor_noise_std,
                   noise_seed=noise_seed + 1),
    )
