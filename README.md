# Stomatopod-Inspired Active Multi-Channel Gaze Control

*Short form: **Stomatopod Active Vision** · Repo: `stomatopod-active-vision`*

> A biomimetic three-layer architecture in MuJoCo for scanning compound-eye-style sensing,
> inspired by the active scanning, multi-channel sensing, and early-stage information compression
> observed in the mantis shrimp (Stomatopoda) visual system.

**Capstone project · Neuromorphic Control (SS26, UOS) · Dr. Johannes Leugering**

---

## TL;DR

Two simulated "eye towers" with three rotational degrees of freedom each (yaw, pitch, roll) sample a 3-D scene through narrow mid-band fields of view. A computational preprocessing layer compresses each eye's raw multi-channel sensor stream into sparse events before sending anything to a high-level scheduler, which decides where each eye should look next to maximise information gathered from the scene.

The project compares three architectures to isolate the contributions of (a) moving sensors and (b) moving computation closer to the sensor:

| | **Active scanning?** | **In-sensor preprocessing?** |
|---|:---:|:---:|
| **B1** Fixed cameras                       | ❌ | ❌ |
| **B2** Fixed cameras + preprocessing       | ❌ | ✅ |
| **B3** Active scanning + preprocessing     | ✅ | ✅ |

The headline result is a four-metric bar chart (coverage, bandwidth, polarization detection accuracy, response latency) showing how each architectural step contributes.

---

## Why this is interesting

This project is a worked example of three big themes from the course:

1. **Hierarchical control** (Lec 8–9). A low-level PD controller drives the gimbals; a high-level scheduler chooses targets for them; the two layers communicate only through the scheduler's setpoints and the sensor's event stream — a deliberate simplification of the biological loop (Scott 2004), which additionally passes efference-copy signals downward for active disturbance rejection. See `docs/lecture_mapping.md` for the full Lec-9 cross-reference and the report §7 for why we treat this omission as an honest limitation rather than a claimed feature.
2. **Neuromorphic sensing** (Lec 7). Computation is pushed *close to the sensor* so that the controller receives sparse, semantically meaningful events rather than raw measurements.
3. **Active sensing as control** (Lec 1, 4, 6). Where to look next is itself an action with a cost and a value — the scheduler optimises over it.

The mantis shrimp visual system is one of the most striking biological examples of all three principles working together.

---

## ⚠️ Biological fidelity disclaimer

> This project is **biomimetic, not biologically faithful**.
> We use the mantis shrimp visual system as *inspiration* for a control architecture; we do **not** claim
> to simulate its actual optics, retinal circuitry, photoreceptor chemistry, or species-specific anatomy.
> Throughout the code and the report we use language like *"inspires," "motivates," "abstracts"*
> rather than *"models" or "reproduces."*

What is well-supported by the literature, and motivates our design choices:

- Compound eye with a specialised mid-band of receptor rows. *(Marshall et al. 1991, 1999)*
- Eyes that rotate independently in pitch, yaw, and roll. *(Daly et al. 2018)*
- Eye rotation is functionally important for accessing polarization information. *(Daly et al. 2018)*
- Rapid colour identification via temporal scanning across multiple receptor classes — *not* primate-style opponency. *(Thoen et al. 2014)*

What is **engineering abstraction**, not biological model:

- The specific number of "channels" per eye (real receptor counts are species-specific).
- The threshold-based event encoder (a neuromorphic-hardware analogy; mantis shrimp retinae do not literally emit threshold-crossing spikes).
- The four-receptor polarization decoder (a standard engineering shortcut, not the actual neural mechanism).
- The "pick the dominant row" preprocessing (an abstraction of the *outcome* of temporal scanning, not the mechanism).

See `docs/project_spec.md` §1 for the full biological-fidelity discussion.

---

## Architecture

```
   ┌────────────────────────────────────────────────────────────────┐
   │  LAYER 3 — ACTIVE-SENSING CONTROLLER  (high-level "brain")    │
   │  Receives event stream → plans next gimbal target              │
   │  src/stomatopod_vision/scheduler.py                            │
   └────────────────────────────────────────────────────────────────┘
                             ▲  sparse event stream
                             │
   ┌────────────────────────────────────────────────────────────────┐
   │  LAYER 2 — IN-SENSOR PREPROCESSING  (computational, biomim.)   │
   │  Channel reduction · polarization decode · event encoding      │
   │  src/stomatopod_vision/preprocessing.py                        │
   └────────────────────────────────────────────────────────────────┘
                             ▲  raw multi-channel sensor stream
                             │
   ┌────────────────────────────────────────────────────────────────┐
   │  LAYER 1 — GIMBAL CONTROL  (low-level "body")                  │
   │  PD control of 6 DOFs (yaw, pitch, roll × 2 eyes)              │
   │  src/stomatopod_vision/gimbal_control.py                       │
   │  models/stomatopod_eyes.xml                                    │
   └────────────────────────────────────────────────────────────────┘
```

---

## Quick start

See [`docs/setup.md`](docs/setup.md) for full install instructions, the
test-suite invocation, the orchestrator commands for every scene, and
the scheduler-weight tuning sweep. Quick taste:

```bash
pip install -r requirements.txt
make test         # 138 tests in ~19 s
make results      # multi-seed sweeps on default + hard scenes (~5 min)
```

> **Note.** This repository contains only the **runnable code** (source,
> tests, docs, examples, MuJoCo XMLs, configs, CI). The compiled
> **report PDF**, **presentation slides**, and **video presentation**
> are submitted through the course platform separately, not committed
> here. `results/` (per-seed JSONs, figures, B-roll videos) is
> regenerable from a single `make all` and is likewise not committed.

---

## Empirical headline (10 s × 5 seeds, current defaults)

| Scene | Baseline | Coverage | Polarization acc. | Bandwidth (B/s) |
|---|---|---:|---:|---:|
| **default** | B1 | 1.00 ± 0.00 | 1.00 ± 0.00 | 195 000 ± 0 |
| **default** | B2 | 1.00 ± 0.00 | 1.00 ± 0.00 | **11 ± 0** *(17 400× reduction)* |
| **default** | B3 | 1.00 ± 0.00 | 1.00 ± 0.00 | 5 601 ± 786 |
| **default** | B3D | 1.00 ± 0.00 | 1.00 ± 0.00 | 3 517 ± 0 |
| **hard** | B1 | **0.00 ± 0.00** | **0.00 ± 0.00** | 195 000 ± 0 |
| **hard** | B2 | **0.00 ± 0.00** | **0.00 ± 0.00** | 45 ± 0 |
| **hard** | B3 | **1.00 ± 0.00** | **1.00 ± 0.00** | 11 370 ± 828 |
| **hard** | B3D | **1.00 ± 0.00** | **1.00 ± 0.00** | 11 917 ± 0 |

→ The default scene is the headline **B1 → B2 bandwidth story** (move computation closer to the sensor, 17 400× reduction at zero task cost). The hard scene is the headline **B2 → B3 capability story** (active scanning is the *only* way to identify the wide-angle interesting targets). **B3D** is the bonus dynamical controller (`HopfScanScheduler`): a Hopf limit-cycle scanner that matches B3 on coverage and all detection metrics — cheaper than B3 on the default scene, comparable on hard. Full discussion in `docs/project_spec.md` §6.5.

Raw bandwidth is now 195 000 B/s under the current sensor model — per eye, 12 spectral channels plus two 4-receptor polarization banks (bare-linear and quarter-wave) streamed at 500 Hz; the headline reductions are the sparse-event stream measured against that.

---

## Repository structure

See [`docs/setup.md`](docs/setup.md#repository-structure) for the full tree.
Top level:

```
stomatopod-active-vision/
├── docs/           ← project spec + setup guide + supporting docs
├── models/         ← 3 MuJoCo scene XMLs (default / hard / moving)
├── src/            ← installable Python package + CLI experiment scripts
├── tests/          ← 12 test files, 138 named tests
├── examples/       ← 2 Jupyter walkthrough notebooks (pipeline + results)
└── .github/        ← CI workflow (lint + typecheck + tests on every push)
```

*(Report PDF, presentation slides, and video are submitted through the
course platform separately. `results/` is regenerable via `make all`
so it isn't committed.)*

---

## The three baselines in detail

### B1 — Fixed cameras (no scanning, raw stream)

Both eyes locked at the rest pose. The full Layer 2 pipeline still runs internally (otherwise nothing could ever be identified), but **bandwidth is reported as if the raw multi-channel stream were transmitted**: this is the cost a passive multi-channel-camera architecture would pay in practice. Coverage and polarization accuracy are therefore identical to B2 on a given scene — the B1-vs-B2 contrast lives in the bandwidth metric alone.

### B2 — Fixed cameras + in-sensor preprocessing

Eyes still locked forward, but bandwidth is now reported as the sparse event stream emitted by the encoder. Isolates the "move computation closer to the sensor" benefit: ~17 400× reduction with zero loss of downstream task performance on the default scene.

### B3 — Active scanning + in-sensor preprocessing (the full biomimetic system)

The six gimbal DOFs are driven by `SaliencyScheduler`, which independently picks where each eye should look next based on a saliency + visit-history scoring function. This is the only baseline that can identify targets sitting outside the ±60° × ±5° rest-pose FoV. Default `ScoringWeights` come from a 405-run grid sweep on the hard scene (see `tune_b3.py`); the empirical justification is documented in the `ScoringWeights` docstring and in `docs/project_spec.md` §6.5.3.

---

## Metrics

| Metric | What it measures | Default scene | Hard scene |
|---|---|---|---|
| **Coverage** | fraction of interesting targets correctly identified | B1=B2=B3=1.0 | B1=B2=0, **B3=1.0** |
| **Bandwidth** | average bytes/s sensor → controller | **B1≫B2≪B3** (B1 195 000, B2 11, B3 5 601) | B1≫B3≫B2 |
| **Polarization accuracy** | fraction of polarized targets correctly classified | all 1.0 | B1=B2=0, **B3=1.0** |
| **Median response latency** | time from first FoV entry to first correct identification | 0.0 s for all | 10.0 s for B1/B2 (right-censored), 0.0 s for B3 |

Implementation: `src/stomatopod_vision/metrics.py`. Tested by `tests/test_metrics.py` (21 unit tests).

---

## Lecture coverage

This project touches the following lectures explicitly:

| Lecture | Concept | Where in this repo |
|---|---|---|
| 1 | Saccades, smooth pursuit, gain learning | `scheduler.py` (inspiration for the look-next decision) |
| 2 | State-space models | `gimbal_control.py` (each gimbal is a 3-DOF SS system) |
| 3 | PID / closed-loop control | `gimbal_control.py` (low-level layer) |
| 4 | Observers, internal models, IMP | `scheduler.py` (internal saliency + memory map) |
| 5 | Multi-joint control, computed torque | `gimbal_control.py` (reuse of HW4 patterns) |
| 6 | Bellman, value-based action selection | `scheduler.py` (`SaliencyScheduler` and the bonus `LearnedScheduler`) |
| 7 | Neuromorphic hardware, event-driven sensing | `preprocessing.py` (the whole Layer 2) |
| 8 | Hierarchical motor control | The three-layer architecture itself |
| 9 | Hybrid / hierarchical control | The three-layer architecture itself |
| 10 | Learning in the brain, eye-movement control loop | `scheduler.py` (the full actor-critic loop and the MLP distillation bonus) |

---

## FAQ

**Why a pure-NumPy MLP for the bonus learned scheduler? Why not PyTorch?**
Three reasons. (1) The MLP is tiny (12 → 16 → 1 = 209 parameters); the overhead of a tensor framework is bigger than the model. (2) Keeping the runtime dependency surface to `mujoco + numpy + matplotlib + mediapy + imageio-ffmpeg` makes the install one `pip install -r requirements.txt` away on any machine. (3) The point of the bonus is "the scoring sum is replaceable by a learned approximator", not "we deployed a deep learning framework." A 150-line `_mlp.py` with hand-rolled forward / backward / Adam makes the imitation-learning step inspectable end-to-end.

**Why these specific `ScoringWeights` defaults `(1, 2, 0.5, 1)`?**
Empirical. A grid sweep (`tune_b3.py`) on the hard scene at the sub-saturated operating point `T=0.5 s` found that the lecture-intuition weights `(1, 2, 0.5, 1)` scored coverage 0.76±0.19 while the pure-exploration ablation `(1, 2, 0, 0)` scored only 0.70±0.18 — the motion-smoothing (feasibility) and roll-diversity (polarization_info_gain) terms help by preventing the eye from whipping across the head between successive wide-azimuth targets. Both weight sets saturate at coverage=1.000 by T=10 s, but the hand-designed set uses ~41% less bandwidth at that horizon (11 370 vs 19 201 B/s). See `docs/project_spec.md` §6.5 + `ScoringWeights` docstring for the full story.

**Why two scene XMLs (default + hard)? Why not one harder scene?**
The two scenes isolate different effects. The **default scene** keeps every interesting target inside the rest field of view, so all three baselines can in principle identify them — meaning the only metric that moves between B1 and B2 is bandwidth (the in-sensor preprocessing story). The **hard scene** moves every interesting target outside the rest field of view, so B1/B2 categorically fail — meaning the only meaningful contrast is B2 → B3 (the active-scanning story). A single "harder" scene would conflate the two stories and weaken both.

**Why is B3's bandwidth higher than B2's, often by several hundred ×?**
B3 moves the eyes. Every time an eye rotates, the relative azimuth/elevation of every visible target changes, which trips the event encoder's re-emission threshold, which produces a fresh event. B2's eyes are frozen, so only target-level changes (new targets appearing, polarization-decode confidence flipping above/below threshold) emit events. The bandwidth cost of motion is not a bug — it is the price of getting coverage from a sensor that is necessarily narrow at any given instant.

**Why 5 RNG seeds, not more?**
A 405-run tuning grid established that B3's coverage variance is ~0.05 at the sub-saturated horizon and 0.00 (exact saturation) at the canonical T=10 s horizon. Once a metric has σ=0.00 across the seeds you've already sampled, sampling more seeds adds no information. We used 5 seeds for all headline numbers and 8 for the tuning grid as a sensitivity check; both agreed.

**Why no PyTest? Why scripts?**
Each test file is runnable directly (`python tests/test_world.py`) and self-reports its own progress with `print` and one `Test N:` heading per checked invariant. This is more debuggable in a sandbox than pytest's quiet-by-default discovery, and the test suite is small enough (12 files, 138 tests, 19 s total) that the loss of pytest's parameterise/fixture machinery doesn't bite. `make test` (or `tests/run_all.py`) runs them all in a loop with a pass/fail summary. If you want pytest, `pip install -e .[tests]` enables it.

---

## Citations

Order them as they appear in the report. Key references:

```bibtex
@article{Marshall1999_colourful,
  author  = {Marshall, J. N. and Cronin, T. W. and Kleinlogel, S.},
  title   = {The colourful world of the mantis shrimp},
  journal = {Nature},
  volume  = {401},
  pages   = {873--874},
  year    = {1999},
  doi     = {10.1038/44751},
}

@article{Marshall1991_compound,
  author  = {Marshall, J. N. and Land, M. F. and King, C. A. and Cronin, T. W.},
  title   = {The compound eyes of mantis shrimps ({Crustacea}, {Hoplocarida}, {Stomatopoda})},
  journal = {Philosophical Transactions of the Royal Society B},
  volume  = {334},
  pages   = {33--56},
  year    = {1991},
}

@article{Cronin1989_ten,
  author  = {Cronin, T. W. and Marshall, N. J.},
  title   = {A retina with at least ten spectral types of photoreceptors in a mantis shrimp},
  journal = {Nature},
  volume  = {339},
  pages   = {137--140},
  year    = {1989},
  doi     = {10.1038/339137a0},
}

@article{Thoen2014_different,
  author  = {Thoen, H. H. and How, M. J. and Chiou, T.-H. and Marshall, J.},
  title   = {A different form of color vision in mantis shrimp},
  journal = {Science},
  volume  = {343},
  number  = {6169},
  pages   = {411--413},
  year    = {2014},
  doi     = {10.1126/science.1245824},
}

@article{Daly2018_complex,
  author  = {Daly, I. M. and How, M. J. and Partridge, J. C. and Roberts, N. W.},
  title   = {Complex gaze stabilization in mantis shrimp},
  journal = {Proceedings of the Royal Society B},
  volume  = {285},
  number  = {1878},
  pages   = {20180594},
  year    = {2018},
  doi     = {10.1098/rspb.2018.0594},
}

@article{Gallego2020_eventbased,
  author  = {Gallego, G. and Delbr\"{u}ck, T. and Orchard, G. and Bartolozzi, C. and
             Taba, B. and Censi, A. and Leutenegger, S. and Davison, A. J. and
             Conradt, J. and Daniilidis, K. and Scaramuzza, D.},
  title   = {Event-Based Vision: A Survey},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {44},
  number  = {1},
  pages   = {154--180},
  year    = {2022},
  doi     = {10.1109/TPAMI.2020.3008413},
}

@inproceedings{Todorov2012_mujoco,
  author    = {Todorov, E. and Erez, T. and Tassa, Y.},
  title     = {{MuJoCo}: A physics engine for model-based control},
  booktitle = {Proceedings of IROS},
  pages     = {5026--5033},
  year      = {2012},
  doi       = {10.1109/IROS.2012.6386109},
}

@article{Kingma2014_adam,
  author  = {Kingma, D. P. and Ba, J.},
  title   = {{Adam}: A Method for Stochastic Optimization},
  journal = {arXiv:1412.6980},
  year    = {2014},
}

@article{Bellman1957_dp,
  author  = {Bellman, R.},
  title   = {Dynamic Programming},
  journal = {Princeton University Press},
  year    = {1957},
}

@article{Bajcsy1988_active,
  author  = {Bajcsy, R.},
  title   = {Active perception},
  journal = {Proceedings of the IEEE},
  volume  = {76},
  number  = {8},
  pages   = {966--1005},
  year    = {1988},
  doi     = {10.1109/5.5968},
}

@misc{Towers2024_gymnasium,
  author       = {Towers, M. and Kwiatkowski, A. and Terry, J. and Balis, J. U. and
                  De~Cola, G. and Deleu, T. and Goul{\~a}o, M. and Kallinteris, A. and
                  Krimmel, M. and KG, A. and Perez-Vicente, R. and Pierr{\'e}, A. and
                  Schulhoff, S. and Tai, J. J. and Tan, H. and Younis, O. G.},
  title        = {Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  year         = {2024},
  eprint       = {2407.17032},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2407.17032},
}

@article{Scott2004_ofc,
  author  = {Scott, Stephen H.},
  title   = {Optimal feedback control and the neural basis of volitional motor control},
  journal = {Nature Reviews Neuroscience},
  volume  = {5},
  number  = {7},
  pages   = {532--546},
  year    = {2004},
  doi     = {10.1038/nrn1427},
  url     = {https://www.nature.com/articles/nrn1427},
}

@article{Doya2000_complementary,
  author  = {Doya, Kenji},
  title   = {Complementary roles of basal ganglia and cerebellum in learning and motor control},
  journal = {Current Opinion in Neurobiology},
  volume  = {10},
  number  = {6},
  pages   = {732--739},
  year    = {2000},
  doi     = {10.1016/S0959-4388(00)00153-7},
}

@article{Todorov2002_optimal,
  author  = {Todorov, Emanuel and Jordan, Michael I.},
  title   = {Optimal feedback control as a theory of motor coordination},
  journal = {Nature Neuroscience},
  volume  = {5},
  number  = {11},
  pages   = {1226--1235},
  year    = {2002},
  doi     = {10.1038/nn963},
}

@article{Chiou2008_circular,
  author  = {Chiou, Tsyr-Huei and Kleinlogel, Sonja and Cronin, Tom and
             Caldwell, Roy and Loeffler, Birte and Siddiqi, Afsheen and
             Goldizen, Alan and Marshall, Justin},
  title   = {Circular polarization vision in a stomatopod crustacean},
  journal = {Current Biology},
  volume  = {18},
  number  = {6},
  pages   = {429--434},
  year    = {2008},
  doi     = {10.1016/j.cub.2008.02.066},
}

@article{Ijspeert2008_cpg,
  author  = {Ijspeert, Auke Jan},
  title   = {Central pattern generators for locomotion control in animals
             and robots: A review},
  journal = {Neural Networks},
  volume  = {21},
  number  = {4},
  pages   = {642--653},
  year    = {2008},
  doi     = {10.1016/j.neunet.2008.03.014},
}
```

---

## Acknowledgements

- Course concepts, simulation infrastructure, and project assignment by **Dr. Johannes Leugering** (Neuromorphic Control, SS26, UOS).
- HW4 code patterns (PD control, `qfrc_bias` compensation, `mj_jacSite`) are reused for the gimbal layer.
- MuJoCo 3.x by Google DeepMind.

---

## License

This project is submitted for academic evaluation. Code is released under the MIT license; the report and figures are © the author.
