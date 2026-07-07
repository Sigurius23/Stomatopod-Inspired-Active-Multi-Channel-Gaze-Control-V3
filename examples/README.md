# Examples

Interactive walkthroughs of the project's internals.

| File | Purpose |
|---|---|
| `walkthrough.py` / `walkthrough.ipynb` | **Pipeline walkthrough.** Step through sensor → preprocessing → scheduler → closed-loop simulation cell-by-cell. |
| `02_results_explorer.py` / `02_results_explorer.ipynb` | **Empirical results.** Plot the scheduler-weight tuning grid + the REINFORCE learning curve from the committed JSON/CSV traces. |
| `README.md` | This file. |

Each notebook ships in two formats:

- The `.py` file is the **source of truth**, a
  [`jupytext`](https://jupytext.readthedocs.io/)-formatted percent script.
  Edit this when you change a notebook.
- The `.ipynb` file is the same content executed once with outputs
  saved, so you can read it on GitHub or open it in JupyterLab without
  running anything.

## Run interactively

```bash
pip install -e .[dev]      # if you haven't already
pip install jupyter jupytext
jupyter notebook examples/walkthrough.ipynb
```

## Re-generate notebooks after editing the .py source

```bash
jupytext --sync examples/walkthrough.py
jupytext --sync examples/02_results_explorer.py

# Optionally re-execute every cell (needs MUJOCO_GL=egl for walkthrough):
MUJOCO_GL=egl jupyter nbconvert --to notebook --execute --inplace \
    examples/walkthrough.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --to notebook --execute --inplace \
    examples/02_results_explorer.ipynb
```

## What each notebook covers

### `walkthrough.ipynb` (18 cells)

1. Load the hard scene and inspect its target metadata.
2. Step the virtual eye once and look at the raw multi-channel sighting.
3. Run the in-sensor preprocessing pipeline and watch sparse events come out.
4. Construct a `SaliencyScheduler` and inspect each of its 4 scoring components.
5. Run a short closed-loop simulation under B3 and plot coverage over time.
6. Compare B3 against B3-Learned on the same scene.

### `02_results_explorer.ipynb` (12 cells)

1. Load `results/tuning/grid.csv` and plot each weight's marginal effect on
   short-horizon coverage (the empirical finding driving the `(1, 2, 0, 0)`
   defaults).
2. Load `results/learned_rl/training_curve.json` and plot the REINFORCE
   return trace with a 20-episode trailing mean.

These are *teaching* artefacts, not tests of correctness — the formal
test suite lives under `tests/`.
