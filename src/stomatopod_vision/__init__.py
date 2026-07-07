"""
Stomatopod-Inspired Active Multi-Channel Gaze Control
=====================================================

A biomimetic three-layer architecture in MuJoCo for active multi-channel
visual scanning, inspired by the mantis shrimp (Stomatopoda) visual system.

This is the top-level package. The three architectural layers are:

    Layer 1 — sensors + gimbal control:
        :mod:`stomatopod_vision.sensor`           virtual receptor model
        :mod:`stomatopod_vision.gimbal_control`   PD on the 6 gimbal DOFs

    Layer 2 — in-sensor preprocessing (biomimetic abstraction):
        :mod:`stomatopod_vision.preprocessing`    channel reduce, pol decode,
                                                  event encoding

    Layer 3 — active-sensing controller:
        :mod:`stomatopod_vision.scheduler`        decides where to look next

    Support modules:
        :mod:`stomatopod_vision.world`            scene / target management
        :mod:`stomatopod_vision.metrics`          the 4 headline metrics
        :mod:`stomatopod_vision.viz`              visualisation helpers

Biological-fidelity disclaimer
------------------------------
This package implements a *biomimetic* architecture inspired by mantis
shrimp vision; it is not a faithful simulation of stomatopod retinal
biophysics. See ``docs/biological_disclaimer.md`` for the detailed
inspires-vs-models breakdown.
"""

__version__ = "0.1.0"
__all__ = [
    "sensor",
    "preprocessing",
    "gimbal_control",
    "scheduler",
    "world",
    "metrics",
    "viz",
]
