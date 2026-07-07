# Changelog

All notable changes to **Stomatopod-Inspired Active Multi-Channel
Gaze Control** are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses a single `0.1.0` development version until the capstone is
submitted, after which a `1.0.0` tag will mark the submitted artefact.

## [Unreleased] — 0.1.0 (development)

### Changed (twelfth wave — B3D wired in as a baseline + docs/numbers refreshed)
- **`HopfScanScheduler` is now a first-class baseline "B3D."** New
  `src/experiments/run_b3_dynamical.py` (mirrors `run_b3_active.py`, writes
  `B3D_metrics.json` / `B3D_log.json`); `run_all.py` gains an opt-in
  `--dynamical` flag (off by default so the headline B1→B2→B3 sweep stays
  byte-stable); `metrics._BASELINES` and `viz` colour/label maps recognise
  `B3D`.
- **Headline numbers regenerated (10 s × 5 seeds) against the corrected
  12-class + two-channel-polarization model** and propagated through
  `README.md`, `docs/project_spec.md`, `docs/lecture_mapping.md`, and the
  `ScoringWeights` docstring:
  - raw bandwidth 107 000 → **195 000 B/s**; B1→B2 default reduction
    7 900× → **17 400×**; B3 hard bandwidth 13 200 → **11 370 ± 828 B/s**.
  - tuning comparison refreshed: hand-designed `(1,2,0.5,1)` vs pure-explore
    `(1,2,0,0)` → cov@0.5s 0.76 vs 0.70, bw@10s 11 370 vs 19 201 B/s
    (hand-designed still Pareto-better). Fixed a stale README FAQ that
    mislabelled the default weights as `(1,2,0,0)`.
  - test counts 128→**138** / 11→**12 files**; sensor-vector descriptions
    updated from the old 5-class `{R,G,B,UV-A,UV-B}` to 12 classes `C1…C12`
    plus the linear + quarter-wave polarization banks and `circular_handedness`.
- **Lint/warning cleanup:** fixed the only invalid-escape-sequence
  `DeprecationWarning` (`ScoringWeights` docstring → raw string). Zero
  escape/syntax warnings across `src/` + `tests/`; full suite 138 green.

### Added (eleventh wave — dynamical limit-cycle scheduler, for real this time)
- **New `HopfScanScheduler` (`scheduler.py`)** — a genuine dynamical-systems
  active-sensing controller, implemented and tested (an earlier external
  "walkthrough" *claimed* such a scheduler existed; it did not — this is the
  actual implementation). Each eye is driven by the supercritical **Hopf
  normal form** `ẋ=g(μ−r²)x−ωy, ẏ=g(μ−r²)y+ωx`, integrated with sub-stepped
  RK4:
  - `μ>0` → a stable limit cycle of radius `√μ`: the eye **scans**, the fast
    loop sweeping the narrow-in-pitch midband while a slow centre-drift
    rasters it across azimuth; roll is an independent slow oscillation for
    polarization sampling.
  - Detecting a target lowers `μ<0` (a real **Hopf bifurcation** to a
    fixed-point attractor) and re-centres the loop on the target → the eye
    **fixates/foveates** it for a short dwell, then bifurcates back to
    scanning. CITE: Land et al. 1990; Marshall et al. 2014; Daly et al. 2018.
  - Same `BaseScheduler` interface as `SaliencyScheduler`, so it is a drop-in
    for the experiment harness (`_held_setpoint`, `next_setpoint`,
    `update_memory`, `from_mujoco_model`).
- **New `tests/test_hopf_scheduler.py` (8 tests)** verifying the *dynamical*
  properties, not just output: stable limit cycle at `√μ` from inside/outside,
  sustained oscillation, scan→fixate→scan bifurcation with sub-degree
  foveation, re-fixation cooldown, joint-limit safety, determinism, and a real
  MuJoCo end-to-end run reaching coverage 1.0. Suite now 138 tests, all green.
- **New `src/experiments/benchmark_dynamical.py`** — head-to-head vs.
  `SaliencyScheduler` through the same pipeline/metrics. Result (8 s, seeds
  0–2): both reach **coverage 1.000, pol 1.000, circ 1.000** on default+hard;
  the Hopf scanner is *more* bandwidth-efficient on the default scene
  (~3.8k vs ~5.5k B/s) and less on hard (~12.8k vs ~9.6k B/s) — a viable
  alternative controller with honest tradeoffs, not a strict win.

### Changed (tenth wave — real circular-polarization vision + metric hardening)
- **Two-channel polarization sensor (the proper circular-vision fix).** The
  old model gave every eye one linear-analyzer bank and faked circular light
  by rewriting its angle to ±45°, which made a circular-left target *identical*
  to a linear-45° target — so any linear angle in (0°,90°) was decoded as
  "left" and (90°,180°) as "right". `sensor.py` now models the biology
  honestly with **two** receptor banks per eye:
  - `polarization_responses` — bare linear analyzers (circular/unpolarized
    light reads a flat 0.5, i.e. no linear axis).
  - `circular_responses` — analyzers behind a **quarter-wave retarder**
    (midband rows-5/6 analog). Circular light produces a strong, roll-invariant
    45°/135° split whose sign encodes handedness; *any* linear angle produces
    `r[1] == r[3]` (zero split). CITE: Chiou et al. 2008.
- **`circular_decode` now reads the quarter-wave channel**, so linear targets
  decode to `None` at every angle and circular targets recover the correct
  handedness independent of eye roll. Verified end-to-end: 0/18 linear angles
  misclassified (was 16/18), circular accuracy 1.000 on the hard scene.
- **Handedness is no longer ground-truth passthrough.** `RawSighting` dropped
  its `circular_handedness` label; handedness is decoded from measurements.
- **Identification de-brittled.** Coverage/identification now matches on the
  argmax of the spectral pattern instead of exact 12-bin tuple equality, so it
  survives receptor noise (was collapsing 100%→~0% at 1–5% noise).
- **New `circular_polarization_accuracy` metric** + `circular_targets`
  tracking; JSON logs load old runs (5-class `dominant_class`) for back-compat.
- **Restored the midband-activation shape check** dropped earlier.
- **Bandwidth accounting** updated for the two polarization banks (raw =
  195 B/sighting); headline B1/B2/B3 numbers in the report will need
  regenerating against the 12-class + two-channel model.
- **Test integrity:** reverted the polarization-change encoder test that had
  been weakened (5°/10° → 2°/4°) to mask the old decoder's false triggers; it
  now passes at the original thresholds because the physics is correct. All
  130 tests pass. These fixes were authored directly in `official_hand_in/`,
  which is the canonical submission.

### Changed (ninth wave — anatomically-legible eye redesign)
- **Eye geometry rebuilt so the render matches the biology write-up.** The
  previous eye was a near-featureless cream ball with a single dark ring,
  half-swallowed by an amorphous head — the script was correct but the
  visualisation was not. Each eye is now a bulbous stalked compound eye split
  into a **dorsal hemisphere** (lighter cap) and **ventral hemisphere**
  (darker cap) by a **narrow six-row midband**: rows 1–4 are the spectral
  rows (long→short wavelength colour ramp) and rows 5–6 are the polarisation
  rows, given a high-specular iridescent look to evoke the quarter-wave /
  circular-polarisation structures (Cronin & Marshall 1989; Chiou et al. 2008).
- **Midband is rigid in the eye body.** The six rows + dark base stripe are
  static geoms parented to the eye, so torsion (roll) rotates the whole band
  as a unit — never sliding inside the eye — exactly as the biology describes.
- **Head reshaped** into a lower, elongated carapace + rostral plate + two
  eyestalk sockets so the stalked eyes stand clearly proud of the body
  instead of being half-buried in it.
- **New `<asset>` materials:** `eye_dorsal`, `eye_ventral`, `midband_base`,
  `mb_row1…mb_row6`, `eyestalk`. Applied identically to all three models
  (`stomatopod_eyes.xml`, `_moving.xml`, `_hard.xml`).
- **Cosmetic-only, benchmark-safe.** No body / joint / site / actuator /
  sensor changed: `nq=nv=nu=6`, the `eye_*_center` / `eye_*_axis` sites and
  rest-pose forward vectors (`[0,1,0]`) are bit-identical, so the sensor
  geometry and all published numbers are unaffected. All 128 tests still pass.

### Added (eighth wave — GitHub-boundary clarity)
- **New `checklist/what_goes_to_github.md`** — every file + folder in the repo classified as ✅ push / 🟡 push-after-fill-in / ❌ keep-local / 🚫 gitignored / 🌐 outside-the-repo, plus a 7-step pre-push checklist covering placeholder fill-in, PDF rebuild, and final `git init && git push`.
- **New `checklist/audit_push.sh`** — one-line dry-run of what a real `git push` would send. Uses a throwaway repo in `/tmp` so the real one is untouched. Prints (a) push/ignore counts, (b) sample ignored files, (c) push-by-top-level-folder histogram, (d) any files still containing `<YOUR NAME>` / `NeuromorphicControl` placeholders, (e) total push size. Current status: 261 files push, 138 ignore, 7 files with placeholders to fill, ~21 MB total.
- **`.gitignore` expanded** to correctly exclude the agent-workflow scratch (`AGENT_TODO.md`, `OPEN_ISSUES.md`, `BEFORE_SUBMITTING.md`, `checklist/`) plus all LaTeX intermediate artefacts (`*.aux, *.log, *.nav, *.snm, *.toc, *.vrb, *.bbl, *.blg, *.fdb_latexmk, *.fls, *.synctex.gz, *.fmt`), Python build artefacts (`*.egg-info/`), and cache directories (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`). Verified with real `git init && git add . && git ls-files`: 138 files correctly ignored.
- **`checklist/README.md` updated** to link the new `what_goes_to_github.md` + `audit_push.sh` files.
- **New `checklist/copy_to_github.sh`** — mirrors the exact 261-file push-list into a `github/` staging folder inside the repo (regenerated on each run). Physical eyeball of what a real `git push` would send; also handy if the course platform ends up wanting a zipped submission archive instead of a repo transfer. `github/` itself is `.gitignore`d so it never nests into the real repo.

### Fixed (seventh wave — frozen B1/B2 video clips)
- **B1/B2 clips were semantically correct but visually frozen.** All three
  static-scene clips (`01_b1_default`, `02_b2_default`, `04_b2_hard`) had
  0.000 mean pixel diff per frame after the first second, because with
  eyes locked forward and no target motion, every frame is genuinely
  identical. Visually indistinguishable from a broken video.
- **Added optional cosmetic camera pan** in `record_run` (new `camera_pan_deg`
  keyword and matching `--camera-pan` CLI flag on `record_video.py`). Rotates
  the named MuJoCo camera slowly about its lookat over the clip duration,
  keeping every frame visually distinct without changing any physics. Uses
  module-level `_pan_cache` + `_pan_data` state; snapshotted on first call,
  cleared when the Renderer context exits.
- **`make_broll.sh` updated**: B1/B2 default clips now use `--camera-pan 12`,
  B2 hard clip uses `--camera-pan 15`. B3/B3L clips are unchanged (the
  physics already provides motion).
- **All 7 B-roll clips regenerated**: motion diagnostic now shows non-zero
  per-frame pixel diff on all seven clips (was 0.000 on three of them).

### Changed (sixth wave — mantis-shrimp-inspired eye + head redesign)
- **Head + eye geometry rewritten** to visually resemble a stomatopod eye pair:
  - Head is now a compound of two tan/brown ellipsoids (posterior "carapace" bulb + anterior rostrum) instead of a flat grey box slab.
  - Each eye is a composite of four geoms (cream ellipsoid body, dark equatorial mid-band, ventral cap tint, side-identity spot) matching the DH / MB / VH visual character of a real stomatopod compound eye (Marshall et al. 1991).
  - The yellow rectangular "roll_marker" box is gone; the red/blue side-identity spot at the anterior pole doubles as a roll indicator.
  - New `<asset>` materials: `carapace`, `eyeball`, `midband`, `ventral`, `eye_L_id`, `eye_R_id`.
  - Sensor sites (`eye_L_center` / `eye_L_axis` etc.) are preserved at their canonical local positions, so all 128 tests still pass.
- **Hard-scene target positions redesigned.** In the process of rewriting the head XML, the target-body definitions were accidentally deleted and had to be reconstructed. The new positions place four targets (FL/FR/BL/BR) at pure wide azimuth (|α|∈[72°, 85°]) and six (FU/FD/LU/RU/LD/RD) at combined azimuth+elevation, all outside the rest FoV of both eyes.
- **`ScoringWeights` defaults reverted to `(1, 2, 0.5, 1)`.** With the redesigned target geometry, the 405-cell tuning sweep now shows the hand-designed lecture-intuition weights are **Pareto-better** than pure exploration `(1, 2, 0, 0)`: both saturate coverage at 1.000 by T=10 s, but hand-designed uses 20–44% less bandwidth across the three scenes. This inverts the "empirical sweep flips the intuition" narrative from previous waves; the current honest story is that the sweep **confirms** the lecture intuition on the new geometry.
- **All empirical numbers refreshed** in the report / slides / script / README / project spec / changelog:
  - B1 → B2 default ratio: 7 600× → **7 900×**
  - B3 hard bw: 27 570 → **13 200** B/s
  - B3L hard bw: 17 510 → **13 317** B/s (now matches teacher within seed noise, no longer 36% lower)
  - B3 moving bw: 8 549 → **6 873** B/s
  - Report §6.5 tuning narrative rewritten to reflect the new finding.
  - Abstract + §5 (System architecture description) + §6.5 + §6.6 + §6.7 (bonus) + §8 (moving) all updated.
- **B-roll videos + still figures regenerated** against the new geometry.

### Fixed (fifth wave — figure + video audit)
- **Fixed all 6 broken plotting bugs found by manual audit:**
  1. Headline-figure x-tick labels no longer collide (rotated 20°, shortened via new `SHORT_BASELINE_LABELS`).
  2. Latency panel now shows a legible bar-window + explanatory annotation when every baseline saturates at 0 s (was previously a blank white panel).
  3. Coverage-over-time y-limit extended to `[-0.4, ...]` so flat-zero B1/B2 traces are visible instead of being hidden under the axis line.
  4. Bandwidth-over-time now floors zero-valued log-scale traces at 0.5 B/s (with an explanatory legend note when the mean stays below the floor) so B2's silent hard-scene trace doesn't vanish off the plot.
  5. Polarization-detection timeline now filters out silent (target × baseline) pairs — was 40 rows of mostly whitespace on the hard scene, now ~20 rows of pure content, with alternating row shading for readability and a legend note listing which baselines were omitted.
  6. Report figures folder synced to the newly-generated results/{,hard,moving}/figures.
- **Fixed both video-B-roll bugs found by md5sum audit:**
  1. `01_b1_default.mp4` and `02_b2_default.mp4` were byte-identical to each other (both baselines rendered the same file). Now use distinct `close_left` / `close_right` cameras so viewers can visually distinguish them.
  2. `07_b3_moving.mp4` was byte-identical to `03_b3_default.mp4` (the moving-scene clip was accidentally the default-scene clip). Regenerated with the correct XML; all 7 clips now have distinct md5 hashes.
- **Fixed camera framing across all 3 model XMLs:** the old `cinematic` and `overview` cameras framed the eye towers as tiny corner dots against an empty floor. Re-aimed all 4 cameras (overview / cinematic / close_left / close_right) so the head is prominent in the foreground and the target cluster fills the middle band. Re-generated `results/overlays_demo.png` with the new framing.
- **Reduced default FoV overlay alpha** from 0.20 → 0.10 in `record_video.py`, and set `--fov-alpha 0.10` explicitly in every `make_broll.sh` invocation so the eye towers stay visible through the translucent cones.
- **Removed two stale artefacts** (`b3_hard_demo.mp4`, `b3l_moving_demo.mp4`) that weren't referenced by any script or document.

### Added (post-submission polish — fourth wave, Lecture-9 tie-ins)
- **Three new report paragraphs cross-referencing Lecture 9** (Scott 2004, Doya 2000, Todorov & Jordan 2002):
  - §2: "Tie-in to Lec 9" paragraph after the Layer-3 description, framing the three-layer architecture as a nested-loop instance and the scheduler as a task-space actor-critic.
  - §6.5.3: paragraph reinterpreting the tuning-sweep finding through the minimum-intervention principle — the surviving terms describe the controlled manifold, the dropped terms were correcting uncontrolled-manifold deviations.
  - §7 (Discussion): honest "no efference copy" limitation paragraph, tying the moving-scene bandwidth explosion to the missing forward-model path and flagging it as the smallest change that would recover a Lec-9-shaped architecture.
- **Three new BibTeX entries** in `report/refs.bib`: `Scott2004_ofc`, `Doya2000_complementary`, `Todorov2002_optimal`.
- **`docs/lecture_mapping.md`** — was a stub; now a full lecture-by-lecture cross-reference (Lec 2-10), with dedicated tables for each lecture's concepts and their exact source-code / experiment / test locations. Includes a full Lec-9 section with the 5 concepts spelled out.
- **`README.md`** — softened the "the two layers never talk to each other directly" claim (Lec 9 explicitly notes that biological layers communicate downward via efference copy; our lack of it is a deliberate simplification, not a claimed feature).
- **`BEFORE_SUBMITTING.md`** — added deadline table (video: Wed 2026-07-08 evening, 8:00 hard cap; report: mid-July) + verbatim grading criteria from Lecture 9 + a 7-day activity plan.
- **`presentation/script.md`** rewritten to 7:45 target with 15 s safety margin under the 8:00 hard cap. Added an explicit cover-slide reminder (project title + full name visible on the first frame — mandatory per Lec 9). Extended S10 with the minimum-intervention beat and S13 with the actor-critic / efference-copy closing. Trimmed 100+ words of verbosity elsewhere (S2 "ten spectral types", S6 "a test asserts", S7/S8/S9 "here's...", S11 "you've been seeing", S12 "translucent"). Also fixed a stale "about a degree off the forward direction" claim in S6 → now correctly says "twelve to twenty-three degrees" (post the target-off-axis change).
- **`presentation/cut_list.md`** re-timed to match the new 7:45 script; added the cover-slide requirement callout at the top; added S10 and S13 to the "don't cut" list; added the Lec-9 minimum-intervention beat to the "if you over-run" trim list.
- **Report grew from 10 → 11 pages** with the three tie-in paragraphs; still 0 overfull boxes.

### Added (post-submission polish — third wave, drift cleanup)
- **Second walkthrough notebook** (`examples/02_results_explorer.{py,ipynb}`) — plots the scheduler-weight tuning grid (marginal effect of each weight on coverage) + the REINFORCE learning curve from the committed JSON/CSV traces. 12 cells, runs in <5 s with no MuJoCo dependency.
- **`make help` rewritten** as a hand-curated target catalog (4 groups: Build, Quality gates, Optional experiments, Cleanup) with one-line summaries.
- **`.editorconfig`** — keeps editors aligned on indentation, line-endings, and the 100-char line length set in `ruff.toml`.
- **`docs/setup.md`** now documents the `make typecheck`, `make learned-rl`, `make noise-ablation` targets and the second notebook.
- **`OPEN_ISSUES.md`** trimmed from the original 24-item agent-side list to the 5 things only the human submitter can do, plus 4 honestly-disclosed weaknesses.
- **`BEFORE_SUBMITTING.md`** now flags the optional `<your-username>` placeholders in `README.md` + `docs/setup.md` and the placeholder `NeuromorphicControl` GitHub-org URL in `pyproject.toml` + `CITATION.cff`.
- **`README.md`** updated to mention the new `examples/` directory in the repo-structure tree and the correct page count for `report.pdf` (10, not 9).
- **§6.6 noise-ablation paragraph** tightened: the "√4 SNR boost" gloss replaced with a literal "attenuates per-receptor i.i.d. noise by a factor of √4 = 2 in the decoded θ" so the math reads correctly.
- **Tuning grid refreshed.** Re-ran the canonical 81-cell × 5-seed sweep on the hard scene; committed `results/tuning/grid.csv` (405 rows) and `results/tuning/best.json` now exactly match the numbers cited in README, FAQ, and `docs/project_spec.md` §6.5.3 (every `feas=0, pol=0` cell scores coverage = 0.960 ± 0.049; hand-designed `(1, 2, 0.5, 1)` scores 0.76).
- **Joint-limit decoupling.** `SaliencyScheduler` now accepts a
  `joint_limits=(yaw, pitch, roll)` keyword and exposes a new
  `from_mujoco_model(model)` classmethod that reads the limits straight
  out of `model.jnt_range`. The module-level `_YAW/_PITCH/_ROLL_LIMIT_RAD`
  constants are kept as fallback defaults so existing call sites and
  the pure-NumPy `LearnedScheduler` continue to work unchanged. Two
  new tests in `tests/test_scheduler.py` (#19 + #20) verify the factory
  reads the canonical values and rejects malformed limits.
  Test count: 126 → 128.

## [Unreleased] — 0.1.0 (development, pre-drift-cleanup)

### Added (post-submission polish — second wave)
- **Sensor noise model.** New `receptor_noise_std` parameter on `VirtualEye` + `make_eye_pair` + `build_context`. Default 0.0 preserves all existing headline numbers. New `noise_ablation.py` experiment + §6.6 in the report showing B3 is robust on coverage/polarization across noise σ ∈ [0, 0.2] but pays a 3× bandwidth tax.
- **REINFORCE bonus.** `train_learned_rl.py` trains the LearnedScheduler MLP via policy-gradient on actual discovery outcomes (no teacher). 4 new mechanical tests in `tests/test_learned_rl.py`. Honest result documented in §6.1: trained policy reaches ~7.2/10 mean return after 100 episodes, well above random but well below the imitation teacher's 10/10.
- **Long-horizon weight robustness check.** New table in §6.5.3 honestly framing the tuned-vs-hand-designed weights as "tuned wins at short horizons, breaks even on quality + pays bandwidth tax at long horizons."
- **B3-Learned on moving scene.** New row in the moving-scene table + a sentence in §6.3 showing the learned scheduler's bandwidth benefit transfers from static to dynamic scenes.
- **Default-scene target moved off-axis.** `target_UVpol_1` relocated from `(-0.10, 1.10, 1.00)` to `(-0.35, 1.10, 1.05)` (now at 12–23° azimuth, was 1–11°). Removes a fair criticism that B1/B2 saturation was trivial.
- **Bandwidth byte sizes** now derived from struct introspection (`_raw_sighting_bytes()` + `_preprocessed_event_bytes()`) instead of magic constants. Final B1→B2 ratio: 7900×.
- **Renderer context-manager pattern.** `record_run` + the test that uses `mujoco.Renderer` directly now use `with` blocks; EGL teardown noise is gone.
- **Type-checker pass.** `mypy.ini` + new `make typecheck` target + new CI step. 0 mypy errors across 9 source files.
- **Setpoint callback at controller rate.** New optional `controller_rate_hz` parameter to `run_simulation`; B3/B3L now query the scheduler at 10 Hz instead of 500 Hz (50× fewer calls, identical setpoints).
- **`run_simulation` refactor.** Split the 250-line orchestrator into a thin loop + 5 single-responsibility helpers, each <30 lines.
- **`Towers2024_gymnasium` citation** restored to refs.bib + cited in the abstract.
- **Custom title slides.** Replaced Metropolis's `\maketitle` with a custom plain-frame title to silence the 13.8pt vbox warnings (both decks now compile 0 overfull boxes).
- **FAQ section in README** (~7 questions).
- **Split README.** Quick-start + repo-structure moved into `docs/setup.md`. Top-level README shrunk from 408 → 278 lines.
- **Better MuJoCo visuals.** Sky gradient + checker floor + 3-light setup + 4 named cameras (`overview` / `cinematic` / `close_left` / `close_right`) in all three scene XMLs. B-roll re-rendered.
- **Second TikZ diagram** in the report (sensor geometry: 4 polarization receptors + FoV wedge with a target).
- **Jupyter walkthrough notebook** (`examples/walkthrough.{py,ipynb}`) — 18 cells covering the full sensor → preprocessing → scheduler pipeline interactively.
- **GitHub Actions workflow** (`.github/workflows/test.yml`) — lint + typecheck + 126 tests on every push/PR.

### Added (post-submission polish — first wave)
- Refactored `tests/test_stomatopod_eyes.py` into 4 standard `def test_*` functions (was a single sanity-checking script). Total test count: 126 named tests.
- `ruff.toml` with project-specific lint rules; `make lint` / `make lint-fix` targets.
- `[dev]` extra in `pyproject.toml` for the linter (`pip install -e .[dev]`).
- `CHANGELOG.md` (this file).
- `OPEN_ISSUES.md` + `AGENT_TODO.md` documenting remaining known issues and which ones the agent can pick up.
- `BEFORE_SUBMITTING.md` — placeholder-replacement checklist.

### Added (presentation + recording kit)
- `presentation/{slides,backup}.tex` Beamer decks (Metropolis theme; 14 + 10 slides).
- `presentation/preamble.tex` — shared theme, colour palette matching `viz.py`, callout macros.
- `presentation/script.md` — ~1170-word voiceover script timed for ~7 min.
- `presentation/cut_list.md` — slide-by-slide B-roll cut list.
- `presentation/recording_checklist.md` — practical setup + post-prod checklist.
- `presentation/make_broll.sh` — regenerates all 7 B-roll clips.
- `presentation/Makefile` + `presentation/README.md`.

### Added (report)
- `report/report.tex` — 8-page PDF (~36 KB LaTeX source).
- `report/refs.bib` — 10 BibTeX entries (Marshall 1991/1999, Cronin & Marshall 1989, Thoen 2014, Daly 2018, Gallego 2022, Bajcsy 1988, Bellman 1957, Kingma & Ba 2014, Todorov 2012, Towers 2024).
- `report/Makefile`.

### Added (bonuses)
- **Bonus #1 — LearnedScheduler.** `stomatopod_vision/_mlp.py` (pure-NumPy 2-layer MLP with Adam) + `LearnedScheduler` class in `scheduler.py` extending `SaliencyScheduler`. New CLIs: `train_learned.py`, `run_b3_learned.py`. R² = 1.000 on validation; 97.5 % argmax agreement with the hand-designed teacher.
- **Bonus #2 — Live overlays + video recording.** Implemented all four previously-stubbed functions in `viz.py`: `render_eye_fov_overlay`, `render_saliency_map_overlay`, `render_recent_sightings`, `record_run`. New CLI `record_video.py`.
- **Bonus #3 — Moving targets.** `models/stomatopod_eyes_moving.xml` (mocap target bodies); `world.TargetMotion` + `MovingTargetController`; auto-wiring in `_common.build_context`. New `MOVING_TARGETS` + `MOVING_MOTIONS` class attributes on `Scene`.

### Added (visuals)
- Sky gradient skybox + checker ground plane + 3-light setup in all three scene XMLs.
- 4 named cameras per scene (`overview`, `cinematic`, `close_left`, `close_right`).
- `record_video.py` defaults to the `cinematic` camera for B-roll quality.

### Added (core)
- All 9 core modules in `src/stomatopod_vision/`: `world`, `sensor`, `gimbal_control`, `preprocessing`, `scheduler`, `metrics`, `viz`, `_mlp`, `__init__`.
- All 7 CLI scripts in `src/experiments/`: `_common`, `run_b{1,2,3}_*.py`, `run_b3_learned.py`, `run_all.py`, `tune_b3.py`, `make_plots.py`, `record_video.py`, `train_learned.py`.
- 11 test files in `tests/` (126 named tests).
- 3 scene XMLs in `models/`: default, hard, moving.
- Top-level `Makefile` with 13 targets.
- `tests/run_all.py` — friendly test runner with `--verbose` / `--filter`.
- `CITATION.cff`.

### Empirical
- **Default scene (10 s × 5 seeds):** B1 = B2 = B3 saturate coverage at 1.000±0.000; B2 uses 14 B/s vs B1's 107 000 B/s — a **7900× bandwidth reduction**.
- **Hard scene (10 s × 5 seeds):** B1 and B2 categorically fail (coverage = 0.000±0.000); B3 achieves coverage = polarization = 1.000±0.000.
- **Moving scene (5 s × 5 seeds):** B2 bandwidth jumps from 14 B/s (static) → 354 B/s (moving), a 25× increase as motion trips the event encoder.
- **Tuning sweep (405-cell grid):** discovered that the hand-designed `feasibility` and `polarization_info_gain` weights actively hurt coverage at short horizons. Promoted `(novelty=1.0, salience=2.0, feasibility=0.0, polarization_info_gain=0.0)` to the new defaults.
- **B3-Learned on hard scene:** matches teacher on coverage + polarization (both 1.000±0.000) while using ~36 % less bandwidth (17 510 B/s vs 27 570 B/s for the teacher).

### Fixed (post-submission polish)
- **Report:** 0 LaTeX warnings (previously 2 small overfull hboxes).
- **Slides:** 1 remaining warning (cosmetic title-vbox in `\maketitle`; invisible to the eye).
- **`make_broll.sh`** now passes explicit `--camera` flags so each clip uses the best camera angle for its content.
- **`requirements.txt` / `pyproject.toml`:** added `imageio-ffmpeg` (real runtime dep); dropped misleading `torch` optional dep (the bonus learned scheduler is pure NumPy).
- **`.gitignore`:** targeted ignores for `results/*/data` and `results/videos` instead of a blanket `*.mp4` (which was excluding all B-roll).
- **Ruff pass:** 75 lint warnings → 0.

### Project genesis
- Initial three-layer architecture spec in `docs/project_spec.md` (~600 lines).
- Biomimetic-not-faithful framing locked in throughout (`docs/biological_disclaimer.md`).
- Lecture-to-component mapping in `docs/project_spec.md` §8 (8 of 9 lectures touched).
