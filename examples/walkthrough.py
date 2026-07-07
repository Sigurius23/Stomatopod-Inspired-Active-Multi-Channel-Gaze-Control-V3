# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stomatopod Active Vision — interactive walkthrough
#
# This notebook walks through the **sensor → preprocessing → scheduler**
# pipeline cell-by-cell so you can poke at each layer's outputs without
# running a full simulation. It complements the report + slides on the
# course platform (the narrative) and `tests/test_*.py` (the formal invariants).
#
# **What you'll see, in order:**
#
# 1. Load the hard scene and inspect its target metadata.
# 2. Step the virtual eye once and look at the raw multi-channel sighting.
# 3. Run the in-sensor preprocessing pipeline and watch sparse events come out.
# 4. Construct a `SaliencyScheduler` and inspect each of its 4 scoring
#    components for a single (eye, candidate) pair.
# 5. Run a short closed-loop simulation under B3 and plot coverage over time.
# 6. Compare B3 against B3-Learned on the same scene (if the MLP is trained).
#
# Built from `tests/` and `src/experiments/_common.py`. The cells are
# kept small and printable so you can re-run any one of them in isolation.

# %%
# Headless rendering must be set before mujoco is imported.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

# %%
# Project-package imports. From a clone, `pip install -e .` makes these
# work out of the box; in the workspace we add src/ to sys.path.
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "src" / "stomatopod_vision").exists():
    REPO_ROOT = REPO_ROOT.parent
    if REPO_ROOT == REPO_ROOT.parent:
        raise FileNotFoundError("could not find src/stomatopod_vision/")
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use('agg')   # works headlessly; switch to 'inline' in a real Jupyter session
import matplotlib.pyplot as plt
import mujoco
plt.rcParams['figure.dpi'] = 100

print(f"Repo root: {REPO_ROOT}")
print(f"MuJoCo version: {mujoco.__version__}")

# %% [markdown]
# ## 1. Load the scene
#
# `Scene.from_xml(path)` loads a MuJoCo XML and attaches the Python-side
# `TargetMeta` list (spectral class, polarization angle, is-interesting
# flag). For the **hard** scene there are 18 targets, 10 of them
# interesting + polarized + all outside the rest field of view.

# %%
from stomatopod_vision.world import Scene

scene = Scene.from_xml(REPO_ROOT / "models" / "stomatopod_eyes_hard.xml")
scene.reset()

print(f"Loaded {len(scene.targets)} target metadata records.")
print(f"Interesting (= coverage targets): "
      f"{len(scene.interesting_target_names())}")
print(f"Polarized:                       {len(scene.polarized_targets())}")
print()
print("First few targets:")
for t in scene.targets[:6]:
    pol = f"  pol={t.polarization_angle:+.2f}rad" if t.polarization_angle else "  unpolarized"
    interesting = "★" if t.is_interesting else " "
    print(f"  {interesting} {t.name:20s} class={t.spectral_class:5s}{pol}")

# %% [markdown]
# ## 2. Step the virtual eye
#
# A `VirtualEye` reads MuJoCo state and emits one `RawSighting` per
# in-FoV target. At rest, the hard scene has *no* interesting targets in
# either eye's FoV — that's the whole point of the scene. Let's verify.

# %%
from stomatopod_vision.sensor import make_eye_pair

eye_L, eye_R = make_eye_pair(scene)
raws = eye_L.step() + eye_R.step()

print(f"At rest, the eyes report {len(raws)} sightings in total.")
for r in raws[:5]:
    print(f"  eye={r.eye}  target={r.target_name:20s}  "
          f"az={np.degrees(r.azimuth):+6.1f}°  el={np.degrees(r.elevation):+5.1f}°  "
          f"d={r.distance:.2f}m")

if not any(t.startswith("target_UVpol_") for t in (r.target_name for r in raws)):
    print()
    print("→ Confirmed: no interesting (UV-polarized) targets in the rest FoV.")
    print("  This is what makes B1/B2 categorically fail on the hard scene.")

# %% [markdown]
# ## 3. In-sensor preprocessing — what an event looks like
#
# Drive the left eye to look at one of the interesting targets, take
# one sighting, run it through Layer 2, and see what a sparse
# `PreprocessedEvent` actually contains.

# %%
from stomatopod_vision.preprocessing import PreprocessingPipeline

# Manually point eye_L at target_UVpol_LU using a hand-computed setpoint.
import mujoco

def jid(name): return mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)

# Set both yaws; resets all velocities/accelerations.
scene.data.qpos[scene.model.jnt_qposadr[jid("eye_L_yaw")]]   = -0.85   # head-left
scene.data.qpos[scene.model.jnt_qposadr[jid("eye_L_pitch")]] = +0.50   # up
scene.data.qpos[scene.model.jnt_qposadr[jid("eye_L_roll")]]  = +0.30
mujoco.mj_forward(scene.model, scene.data)

raws_aimed = eye_L.step()
print(f"After re-aiming, eye_L sees {len(raws_aimed)} target(s):")
for r in raws_aimed:
    print(f"  {r.target_name:20s}  midband_argmax={np.argmax(r.midband_activations)}  "
          f"pol_responses={np.round(r.polarization_responses, 3).tolist()}")

# %%
# Run the pipeline on those raw sightings.
pipeline = PreprocessingPipeline()
events = pipeline.step(
    raws_aimed,
    time_now=0.0,
    roll_angles={"L": eye_L.roll_angle(), "R": eye_R.roll_angle()},
)
print(f"Pipeline emitted {len(events)} event(s):")
for ev in events:
    pol_str = f"{ev.polarization_angle:+.3f}" if ev.polarization_angle is not None else "None"
    print(f"  t={ev.time:.3f}  eye={ev.eye}  target={ev.target_name:20s}  "
          f"pattern={ev.spectral_pattern}  "
          f"θ_decoded={pol_str}  circ={ev.circular_handedness}")

# %% [markdown]
# ## 4. Inside the scheduler — scoring a candidate
#
# `SaliencyScheduler` samples `n_candidates` direction tuples per eye
# per re-plan and scores each via a weighted sum of four components.
# Let's manually score a single (eye, candidate) and look at the
# components separately.

# %%
from stomatopod_vision.scheduler import SaliencyScheduler, ScoringWeights
from stomatopod_vision.gimbal_control import GimbalSetpoint

scheduler = SaliencyScheduler(seed=0)
# Feed the scheduler the events we just generated so its memory isn't empty.
scheduler.update_memory(events, time_now=0.0)

# Score one candidate
cand_yaw, cand_pitch, cand_roll = -0.85, +0.50, +0.30   # the same direction we aimed
sp_current = GimbalSetpoint()                            # rest
t_now = 0.5

print("Scoring components for one candidate:")
print(f"  novelty   = {scheduler.score_novelty('L', cand_yaw, cand_pitch, t_now):.3f}")
print(f"  salience  = {scheduler.score_salience('L', cand_yaw, cand_pitch):.3f}")
print(f"  feasibility = {scheduler.score_feasibility('L', cand_yaw, cand_pitch, cand_roll, sp_current):.3f}")
print(f"  pol_info_gain = {scheduler.score_polarization_info_gain('L', cand_roll):.3f}")
print()
print(f"  weighted sum (current defaults) = "
      f"{scheduler.total_score('L', cand_yaw, cand_pitch, cand_roll, t_now, sp_current):.3f}")

print()
print(f"Current default ScoringWeights: {ScoringWeights()}")
print("  (see ScoringWeights docstring for why feasibility and pol_info_gain are 0)")

# %% [markdown]
# ## 5. Closed-loop simulation under B3
#
# Step the simulation for 5 seconds with the scheduler driving the
# gimbals, then plot the cumulative coverage curve.

# %%
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))
from _common import build_context, run_simulation
from stomatopod_vision.scheduler import SaliencyScheduler

ctx = build_context(REPO_ROOT / "models" / "stomatopod_eyes_hard.xml", seed=0)
sched = SaliencyScheduler(seed=0)

log = run_simulation(
    ctx,
    setpoint_at=lambda t: sched.next_setpoint(t, sched._held_setpoint),
    pipeline=PreprocessingPipeline(),
    duration_s=5.0,
    quiet=True,
    on_events=lambda evs, t: sched.update_memory(evs, t),
    controller_rate_hz=10.0,   # same throttling as run_b3_active.py
)

# Cumulative interesting-target identifications over time
from stomatopod_vision.viz import _cumulative_identifications
t_steps, cum = _cumulative_identifications(log, identification_window_s=0.5)
print(f"B3 identified {int(cum[-1])} of {len(log.interesting_targets)} "
      f"interesting targets in {log.duration_s:.1f}s.")

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.step(t_steps, cum, where="post", color="#d62728", linewidth=2)
ax.axhline(len(log.interesting_targets), color="black", ls="--", lw=0.8, alpha=0.5,
           label=f"total interesting = {len(log.interesting_targets)}")
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Cumulative identifications")
ax.set_title("B3 coverage over time (hard scene, seed 0, T=5s)")
ax.set_ylim(0, len(log.interesting_targets) + 1)
ax.grid(linestyle=":", alpha=0.4)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig('/tmp/walkthrough_coverage.png', dpi=100)
plt.show()  # in Jupyter this displays inline

# %% [markdown]
# ## 6. Compare B3 against B3-Learned
#
# If `results/learned/mlp.npz` exists (i.e. you've run
# `make learned` or `train_learned.py`), this cell loads the trained
# MLP and runs B3-Learned for the same 5 seconds. The two should hit
# the same coverage; B3L tends to use less bandwidth as a side benefit.

# %%
from stomatopod_vision.scheduler import LearnedScheduler

mlp_path = REPO_ROOT / "results" / "learned" / "mlp.npz"
if mlp_path.exists():
    ctx2 = build_context(REPO_ROOT / "models" / "stomatopod_eyes_hard.xml", seed=0)
    ls = LearnedScheduler.from_file(mlp_path, seed=0)
    log_l = run_simulation(
        ctx2,
        setpoint_at=lambda t: ls.next_setpoint(t, ls._held_setpoint),
        pipeline=PreprocessingPipeline(),
        duration_s=5.0,
        quiet=True,
        on_events=lambda evs, t: ls.update_memory(evs, t),
        controller_rate_hz=10.0,
    )
    n_events_b3   = len(log.preprocessed_events)
    n_events_b3l  = len(log_l.preprocessed_events)
    print(f"B3  identified {int(cum[-1])} targets, emitted {n_events_b3} events")
    print(f"B3L identified "
          f"{int(_cumulative_identifications(log_l, 0.5)[1][-1])} targets, "
          f"emitted {n_events_b3l} events  ({n_events_b3l / n_events_b3:.1%} of B3's stream)")
else:
    print("MLP not found at", mlp_path)
    print("Train it first with `make learned` or `train_learned.py`.")

# %% [markdown]
# ## Where to go next
#
# - **Full empirical sweep**: `make results` (or `run_all.py`) regenerates every
#   JSON / figure under `results/` for both scenes × 5 seeds.
# - **Scheduler tuning**: `make tune` to redo the 405-cell grid sweep.
# - **Tests**: `make test` runs all 119 tests in ~14 s.
#
# See [`docs/setup.md`](../docs/setup.md) for the full command reference.
