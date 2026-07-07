# Setup, install, run

Full installation and reproducibility guide. The top-level
[`README.md`](../README.md) has the project pitch + empirical headline;
this file is for everything you actually need to *do* to reproduce
the results.

> **Quick reference.** `make help` lists every Make target. `make all`
> rebuilds the multi-seed results sweep. `make test` runs the
> 128-test suite in ~18 s.

---

## Prerequisites

- Python ≥ 3.10
- MuJoCo ≥ 3.0 (installed as a Python package — no separate binary needed)
- NumPy, matplotlib, mediapy, imageio-ffmpeg (the last ships its own ffmpeg)
- A TeX distribution if you want to build the report or slides

On headless machines (Colab, servers) set `MUJOCO_GL=egl` for off-screen rendering;
the scripts default to this already, but you can override with `MUJOCO_GL=glfw` for a live window.

---

## Install

```bash
git clone https://github.com/Sigurius23/Stomatopod-Inspired-Active-Multi-Channel-Gaze-Control-V2.git
cd Stomatopod-Inspired-Active-Multi-Channel-Gaze-Control-V2
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional extras:
pip install -e .[dev]                # adds ruff + mypy for `make lint` and `make typecheck`
pip install -e .[tests]              # adds pytest if you prefer it over scripts
```

For the report and slide builds:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-fonts-recommended texlive-fonts-extra \
                 texlive-science lmodern poppler-utils
```

The Metropolis Beamer theme ships in `texlive-latex-extra`; `poppler-utils` is only needed if you want to extract preview PNGs from the PDFs.

---

## Verify the install — run the test suite

```bash
# Friendliest one-shot runner (138 named tests across 12 files):
MUJOCO_GL=egl python tests/run_all.py

# Or, from the top-level Makefile:
make test

# Or the original shell loop:
for f in tests/test_*.py; do MUJOCO_GL=egl python "$f"; done
```

You should see "✓ PASS" on every file and "Summary: 10 passed, 0 failed".

---

## Reproduce the full comparison

The orchestrator script `run_all.py` runs B1 → B2 → B3 with matched flags and then generates every figure:

```bash
# Default scene, single seed (fast: ~15 s wall-clock)
MUJOCO_GL=egl python src/experiments/run_all.py --duration 10 --seed 0 --png

# Hard scene (the "B3-wins" variant) — 10 interesting targets outside the rest FoV
MUJOCO_GL=egl python src/experiments/run_all.py \
    --model models/stomatopod_eyes_hard.xml \
    --results-dir results/hard \
    --duration 10 --seed 0 --png

# Multi-seed with error bars on the headline figure (5 seeds × 3 baselines × 10 s ≈ 4 min)
MUJOCO_GL=egl python src/experiments/run_all.py \
    --duration 10 --seeds 0 1 2 3 4 --png
```

Each invocation writes `<baseline>_metrics.json` + `<baseline>_log.json` to `results/data/` and four figures (headline, coverage-over-time, bandwidth-over-time, polarization-timeline) to `results/figures/`. In multi-seed mode a `<baseline>_summary.json` is also written with per-metric mean/std/min/max, and `make_plots.py` automatically switches to error-barred / ribbon variants.

---

## Quality gates (lint, type-check, test)

```bash
make lint         # ruff — line length 100, target Py 3.10
make lint-fix     # ruff --fix (auto-apply safe fixes)
make typecheck    # mypy — Python 3.12 syntax, 9 source files
make test         # 138 named tests across 12 files (~19 s)
```

All three are wired into `.github/workflows/test.yml` and run on every push.

---

## Optional experiments

A handful of experiments are deliberately *not* part of `make all` because
they extend wall-clock time substantially. Run them individually when you
want the corresponding tables / figures:

```bash
make tune             # 405-run scheduler-weight grid sweep (~7 min)
make learned          # imitation-trained MLP (B3-Learned)
make learned-rl       # REINFORCE-trained MLP — 100 episodes × 2 s (~3 min)
make noise-ablation   # sensor-noise sweep for §6.6 (~2 min)
```

Outputs land under `results/tuning/`, `results/learned/`, `results/learned_rl/`,
and `results/noise_ablation/` respectively.

---

## Tune the scheduler weights

```bash
# Coarse grid sweep: 3⁴ = 81 weight cells × 5 seeds = 405 runs (~7 min on a CPU)
MUJOCO_GL=egl python src/experiments/tune_b3.py \
    --duration 0.5 --seeds 0 1 2 3 4 \
    --novelty 0.5 1.0 2.0 \
    --salience 1.0 2.0 4.0 \
    --feasibility 0.0 0.5 1.0 \
    --pol 0.0 1.0 2.0
```

Writes `results/tuning/grid.csv` (one row per cell × seed) and `results/tuning/best.json` (the winning weights). See `docs/project_spec.md` §6.5.3 for the empirical finding that drove the current `ScoringWeights` defaults.

---

## Repository structure

```
stomatopod-active-vision/
├── README.md                ← project pitch + empirical headline + FAQ
├── LICENSE                  ← MIT
├── requirements.txt
├── pyproject.toml           ← installable package config
├── Makefile                 ← top-level convenience targets
├── CHANGELOG.md
├── CITATION.cff
├── BEFORE_SUBMITTING.md     ← placeholder-replacement checklist
├── OPEN_ISSUES.md           ← honest list of remaining issues
├── AGENT_TODO.md            ← the subset the agent can tackle
│
├── docs/
│   ├── setup.md             ← this file
│   ├── project_spec.md      ← full ~600-line capstone spec
│   ├── biological_disclaimer.md
│   └── lecture_mapping.md
│
├── models/
│   ├── stomatopod_eyes.xml         ← default scene (6 targets, 1 interesting)
│   ├── stomatopod_eyes_hard.xml    ← hard scene (18 targets, 10 interesting,
│   │                                  all outside the rest FoV)
│   └── stomatopod_eyes_moving.xml  ← mocap target bodies for bonus #3
│
├── src/
│   ├── stomatopod_vision/   ← installable Python package
│   │   ├── world.py         ← Scene + TargetMeta (DEFAULT_TARGETS, HARD_TARGETS)
│   │   ├── sensor.py        ← VirtualEye + MidbandFOV + RawSighting
│   │   ├── gimbal_control.py ← GimbalSetpoint + GimbalPD (Layer 1)
│   │   ├── preprocessing.py ← PreprocessingPipeline + event encoder (Layer 2)
│   │   ├── scheduler.py     ← SaliencyScheduler + LearnedScheduler (Layer 3)
│   │   ├── metrics.py       ← coverage / bandwidth / pol-accuracy / latency
│   │   ├── viz.py           ← single-seed + multi-seed plotting + record_run
│   │   └── _mlp.py          ← pure-NumPy MLP for the bonus LearnedScheduler
│   │
│   └── experiments/         ← CLI scripts (run from repo root)
│       ├── _common.py       ← shared scaffolding (split into single-purpose helpers)
│       ├── run_b{1,2,3}_*.py / run_b3_learned.py
│       ├── run_all.py       ← orchestrator with --seeds support
│       ├── tune_b3.py       ← scheduler-weight grid sweep
│       ├── train_learned.py ← train the bonus MLP
│       ├── record_video.py  ← MP4 with overlays
│       └── make_plots.py
│
├── tests/                   ← 12 test files (138 named tests; runnable as scripts)
│   ├── run_all.py           ← friendly test runner
│   └── test_*.py
│
├── examples/                ← Jupyter walkthrough notebooks
│   ├── walkthrough.{py,ipynb}            ← sensor → preprocessing → scheduler tour
│   └── 02_results_explorer.{py,ipynb}    ← tuning grid + REINFORCE plots
│
└── .github/                 ← CI workflow
```
