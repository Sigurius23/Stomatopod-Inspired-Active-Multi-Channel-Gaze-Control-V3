# =====================================================================
# Top-level Makefile — one entry point per deliverable
# =====================================================================
#
# Common usage:
#     make            → build results (multi-seed sweeps on both scenes)
#     make test       → run the 11-file test suite (128 named tests)
#     make results    → run the full multi-seed sweep on both scenes
#     make tune       → run the scheduler-weight grid search
#     make learned    → train the bonus LearnedScheduler MLP
#     make clean      → remove build artefacts (keeps committed outputs)
#     make distclean  → also remove the per-run results/ contents
#
# The report PDF, presentation slides, and video are submitted through the
# course platform separately — they are not built by this Makefile.
#
# Requirements:
#     pip install -r requirements.txt
#     sudo apt install texlive-latex-recommended texlive-latex-extra \
#                      texlive-fonts-recommended texlive-fonts-extra \
#                      texlive-science lmodern
# =====================================================================

SHELL    := /bin/bash
PYTHON   ?= python
MUJOCO   ?= MUJOCO_GL=egl
SEEDS    ?= 0 1 2 3 4

# Default scene (default + hard) results directories
RES_DEFAULT  := results
RES_HARD     := results/hard
RES_MOVING   := results/moving

.PHONY: all help lint lint-fix typecheck test results results-default \
        results-hard results-moving noise-ablation learned-rl \
        tune learned clean distclean

# ---------------------------------------------------------------------
# Top-level convenience targets
# ---------------------------------------------------------------------

all: results
	@echo
	@echo "✓ Built: results/ (JSONs + figures)."

help:
	@echo "Stomatopod Active Vision — top-level targets"
	@echo
	@echo "Build:"
	@echo "  all              Build results (multi-seed sweeps on default + hard)  (~5 min)"
	@echo "  results          Run multi-seed sweeps (default + hard)"
	@echo "  results-default  Only the default scene"
	@echo "  results-hard     Only the hard scene"
	@echo "  results-moving   Only the moving-targets scene  (Bonus #3)"
	@echo
	@echo "Quality gates:"
	@echo "  test             Run all 128 tests in ~19 s"
	@echo "  lint             Ruff check  (line length 100, target Py 3.10)"
	@echo "  lint-fix         Ruff --fix (auto-apply safe fixes)"
	@echo "  typecheck        Mypy on src/stomatopod_vision/"
	@echo
	@echo "Optional experiments (not part of 'make all'):"
	@echo "  tune             405-run scheduler-weight grid search  (~7 min)"
	@echo "  learned          Train + evaluate the imitation MLP scheduler"
	@echo "  learned-rl       Train the REINFORCE MLP scheduler  (~3 min)"
	@echo "  noise-ablation   Sensor-noise sweep for report §6.6  (~2 min)"
	@echo
	@echo "Cleanup:"
	@echo "  clean            Remove build artefacts (keep committed outputs)"
	@echo "  distclean        Also remove per-run results/ contents"


# ---------------------------------------------------------------------
# Tests — run every test file (each is a runnable script)
# ---------------------------------------------------------------------

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/stomatopod_vision/

test:
	@set -e; total=0; failed=0; \
	for f in tests/test_*.py; do \
	  printf "  %-45s " "$$f"; \
	  if out=$$($(MUJOCO) $(PYTHON) "$$f" 2>&1); then \
	    last=$$(echo "$$out" | tail -1); echo "$$last"; total=$$((total+1)); \
	  else \
	    echo "FAIL"; echo "$$out" | tail -10; failed=$$((failed+1)); \
	  fi; \
	done; \
	if [ $$failed -gt 0 ]; then echo; echo "$$failed FILE(S) FAILED"; exit 1; fi

# ---------------------------------------------------------------------
# Empirical results — multi-seed sweeps on both scenes
# ---------------------------------------------------------------------

results: results-default results-hard

results-default:
	$(MUJOCO) $(PYTHON) src/experiments/run_all.py \
	    --duration 10 --seeds $(SEEDS) --png

results-hard:
	$(MUJOCO) $(PYTHON) src/experiments/run_all.py \
	    --model models/stomatopod_eyes_hard.xml \
	    --results-dir $(RES_HARD) \
	    --duration 10 --seeds $(SEEDS) --png

# Moving-targets scene — used by Bonus #3 narrative
results-moving:
	$(MUJOCO) $(PYTHON) src/experiments/run_all.py \
	    --model models/stomatopod_eyes_moving.xml \
	    --results-dir $(RES_MOVING) \
	    --duration 5 --seeds $(SEEDS) --png

# ---------------------------------------------------------------------
# Bonus experiments
# ---------------------------------------------------------------------

tune:
	$(MUJOCO) $(PYTHON) src/experiments/tune_b3.py \
	    --duration 0.5 --seeds $(SEEDS)

noise-ablation:
	$(MUJOCO) $(PYTHON) src/experiments/noise_ablation.py \
	    --duration 10 --seeds $(SEEDS)

learned-rl:
	$(MUJOCO) $(PYTHON) src/experiments/train_learned_rl.py \
	    --episodes 200 --duration 3

learned:
	$(MUJOCO) $(PYTHON) src/experiments/train_learned.py \
	    --duration 5 --seeds $(SEEDS)
	$(MUJOCO) $(PYTHON) src/experiments/run_b3_learned.py \
	    --model models/stomatopod_eyes_hard.xml \
	    --results-dir $(RES_HARD) \
	    --duration 10 --seeds $(SEEDS)

# ---------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true

distclean: clean
	rm -rf results/
	@echo "  (Run 'make results' to regenerate.)"
