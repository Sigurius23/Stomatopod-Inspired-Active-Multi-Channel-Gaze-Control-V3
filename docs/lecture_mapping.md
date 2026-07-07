# Lecture-to-code mapping

This file expands `docs/project_spec.md` §8's short lecture table into a
concrete, module-level cross-reference. It exists so a lecturer (or a
reader from the course) can jump from a lecture concept to the exact
Python file and function that instantiates it.

The mapping is **exhaustive for the eight concepts we cite** in the
report and slides; there is plenty more the code touches that we
don't try to catalog here.

---

## Lecture 2 — State-space representation

| Concept | Where it lives in the repo |
|---|---|
| Discrete-time state update | `stomatopod_vision/gimbal_control.py::GimbalPD.step` — reads `data.qpos, data.qvel`, writes `data.ctrl` |
| Persistent internal state (memory) | `stomatopod_vision/scheduler.py::SchedulerMemory` — `last_seen_time`, `last_decoded`, `_roll_history` |
| Predictive state (event encoder retains last-emitted class) | `stomatopod_vision/preprocessing.py::EventEncoder._last_class` |

## Lecture 3 — Feedback (PID / PD) control

| Concept | Where it lives in the repo |
|---|---|
| PD control law with joint-limit clamping | `stomatopod_vision/gimbal_control.py::GimbalPD.step` — `τ = Kp(θ*−θ) − Kd·θ̇` |
| Gain tuning by empirical search | `tests/test_gimbal_control.py` locks the tuned `Kp=50, Kd=1.0` values; the search story is in the report §3.1 |
| Setpoint interface decoupled from controller | `stomatopod_vision/gimbal_control.py::GimbalSetpoint` dataclass |

## Lecture 4 — Observers / forward models / internal model principle

| Concept | Where it lives in the repo |
|---|---|
| Belief over recently-visited targets | `stomatopod_vision/scheduler.py::SchedulerMemory` |
| Forward model of "which candidate direction covers which known target" | `stomatopod_vision/scheduler.py::_centring_setpoint` — used inside `sample_candidates` to inject forced target-centring candidates |
| **Not implemented** — a *cerebellum-style* forward model of "given my last motor command, what will the next event stream look like" | Deliberately absent; disclosed in report §7 (Discussion, efference-copy paragraph) — see Lec 9 below |

## Lecture 5 — Computed torque / operational-space control

| Concept | Where it lives in the repo |
|---|---|
| HW4 `qfrc_bias` compensation pattern | Reused verbatim inside `GimbalPD.step` |
| PD gains as the diagonal case of LQR feedback | Noted in report §3.1; a full LQR-optimal derivation is out of scope |

## Lecture 6 — Value-based action selection / Bellman

| Concept | Where it lives in the repo |
|---|---|
| Value function over candidate actions | `stomatopod_vision/scheduler.py::SaliencyScheduler.total_score` — the weighted sum of four `score_*` methods |
| Argmax action selection | `stomatopod_vision/scheduler.py::SaliencyScheduler.next_setpoint` — `np.argmax` over candidate scores per eye |
| Empirical weight tuning | `src/experiments/tune_b3.py` — the 405-run grid sweep that confirmed the `(1,2,0.5,1)` defaults are Pareto-better than pure exploration; result in report §6.5 |

## Lecture 7 — Neuromorphic hardware / event-driven sensing

| Concept | Where it lives in the repo |
|---|---|
| Move computation close to the sensor | Layer 2 of the three-layer architecture — `stomatopod_vision/preprocessing.py` runs before anything reaches the controller |
| Threshold-crossing event encoding | `stomatopod_vision/preprocessing.py::EventEncoder.encode` |
| Bandwidth as an evaluation metric | `stomatopod_vision/metrics.py::bandwidth_bps` — event-count encoding vs. raw-stream reference (see §6.5 for the ~17 400× B1→B2 result) |
| In-sensor linear-polarization decoder (vector-sum inversion) | `stomatopod_vision/preprocessing.py::polarization_decode` |
| In-sensor **circular**-polarization decoder (quarter-wave channel) | `stomatopod_vision/preprocessing.py::circular_decode` + the second (quarter-wave-retarder) receptor bank in `sensor.py` — the channel unique to stomatopods |

## Lecture 8 — Subsumption / hierarchical control

| Concept | Where it lives in the repo |
|---|---|
| Three-layer hierarchy (body / preprocessing / brain) | The whole architecture; see report §2 and Fig. 1 |
| Higher layer sets the *goal* for the lower one, not the raw torque | `SaliencyScheduler.next_setpoint()` outputs a `GimbalSetpoint`; the PD layer decides how to reach it |
| Layer 3 never talks to Layer 1 directly | Scheduler emits setpoints only — see the softened README wording and the efference-copy discussion (below) for why this is a *deliberate simplification*, not a claim of biological fidelity |

## Lecture 9 — Hierarchical optimal feedback control

Added after Lecture 9 was delivered (2026-06-30). Five distinct
concepts from Scott (2004) and Doya (2000) that map onto the project:

| Lec 9 concept | Slide (Leugering) | Where it lives / where it is disclosed |
|---|---|---|
| **Nested-loop structure** — spinal reflex ⊂ joint-space loop ⊂ task-space loop | Slides 25, 29, 30 | Our three-layer architecture (`gimbal_control` ⊂ `preprocessing` ⊂ `scheduler`) is a direct instance; report §2 Tie-in-to-Lec-9 paragraph |
| **Actor-critic in task-space** — premotor cortex as actor, basal ganglia as critic | Slide 27 | `SaliencyScheduler.score_*` methods = critic (value estimate over candidate actions); `argmax` in `next_setpoint` = actor. Cited in report §2 |
| **Minimum intervention principle** — correct only the errors that matter for the task; ignore errors along the uncontrolled manifold | Slides 8, 11 | Instantiated by the hand-designed `feasibility` + `polarization_info_gain` terms: they suppress gaze motion that does *not* improve coverage (the uncontrolled manifold), which is why the `(1,2,0.5,1)` weights tie pure exploration on coverage while using ~41% less bandwidth. Cited in report §6.5.3 |
| **Efference copy + extended state observer** — the intermediate cerebellum receives copies of motor commands to predict future sensor state, enabling active disturbance rejection | Slide 24 | **Not implemented.** Disclosed in report §7 as an honest limitation — the moving-scene bandwidth explosion is partly a consequence. Cited as a "smallest change that recovers a Lec-9-shaped architecture" |
| **LQR / operational-space split** — `τ = τ_ff(θ̂) − M(θ̂)θ` with feedforward and feedback gains | Slides 14–17 | Our `GimbalPD` is the diagonal-LQR degenerate case with zero feedforward; noted in report §3.1 |

### Key Lec 9 references (added to `report/refs.bib`)

- Scott, S. H. (2004). Optimal feedback control and the neural basis of volitional motor control. *Nature Reviews Neuroscience* **5**, 532–546. `[Scott2004_ofc]`
- Doya, K. (2000). Complementary roles of basal ganglia and cerebellum in learning and motor control. *Curr. Opin. Neurobiol.* **10**, 732–739. `[Doya2000_complementary]`
- Todorov, E. & Jordan, M. I. (2002). Optimal feedback control as a theory of motor coordination. *Nat. Neurosci.* **5**, 1226–1235. `[Todorov2002_optimal]` — cited for the minimum intervention principle in §6.5.3

## Lecture 10 — Learning in the brain / physical AI

Lecture 10 (the capstone lecture) is one of the project's *strongest*
tie-ins, not an afterthought — it is literally about the eye-movement
control loop and about shallow-network few-shot learning:

| Lec 10 concept (slide) | How the project instantiates it |
|---|---|
| **The eye-movement control loop** (slide 10, *"Similar story: eye movements!"*) — Superior Colliculus / Frontal Eye Field / Basal Ganglia as a task-space actor–critic that selects *where to look*, over a brainstem + vestibulo-cerebellum inner loop | The whole project *is* this loop: `SaliencyScheduler` selects the next gaze target (the FEF/SC/BG actor–critic) and `GimbalPD` is the brainstem-nuclei inner loop. We build a computational instance of exactly that slide |
| **Actor–critic action selection** (slide 7) — premotor cortex = actor/policy, basal ganglia = critic; $x^*=\arg\max_{x\in\mathcal{X}} V(x,c)$, plus an *attention map* $a$ of where interesting things are | `SaliencyScheduler.next_setpoint` = `np.argmax` over candidate directions (actor) of `total_score`, the weighted value estimate (critic); the `score_salience` term is the "attention map" of where known interesting targets sit |
| **Cerebellum = Marr–Albus–Ito shallow perceptron** (slides 21–24) — single hidden layer $y_j=f_j(\sum_i w_{ij}g_i(x))$, no backprop, few-shot supervised learner | `LearnedScheduler`'s MLP is exactly a $12\to16\to1$ single-hidden-layer network trained by fast supervised **imitation** of the hand-designed teacher — the cerebellar-tutor / **knowledge-distillation** story (slide 19, Hinton/Doya) |
| **Tripartite learning: BG (RL) → cerebellum (supervised) → neocortex (self-supervised)** (slide 19) | Our two learned variants span the first two: the REINFORCE bonus (`train_learned_rl.py`) is the BG/RL layer; the imitation MLP (`train_learned.py`) is the cerebellar fast-supervised layer distilling the hand-designed critic |
| **Lesson 3 — "the hardware matters", specialise per function** (slide 32): CPG for rhythmic tasks · cerebellum for fast learning · cortex for feedback control · BG for action selection | The project ships one of each: `HopfScanScheduler` (CPG), `LearnedScheduler` (fast-learning shallow net), `GimbalPD` (feedback control), REINFORCE / argmax scheduler (RL / action selection) |
| **Lesson 1 — distributed nested control hierarchies, not monolithic black-boxes** (slide 30) | The three-layer architecture (`gimbal_control` ⊂ `preprocessing` ⊂ `scheduler`) is exactly this |
| **Lesson 2 — every loop needs a predictive observer / forward model** (slide 31) | Partial: `SchedulerMemory` retains state, but the project deliberately *omits* the efference-copy forward model — disclosed as the honest limitation (report §7). This is the one Lec-10 lesson we consciously leave on the table |

Code: `scheduler.py::{SaliencyScheduler, LearnedScheduler, HopfScanScheduler}`,
`gimbal_control.py::GimbalPD`, `experiments/train_learned{,_rl}.py`.

## Bonus — Dynamical scheduler (`HopfScanScheduler`, B3-Dynamical)

The bonus limit-cycle scheduler is a *cross-cutting* instance of the
dynamical-systems thread that runs through the course, not a
single-lecture tie-in. This is the strongest "applied lecture concepts"
argument in the project — it uses four lectures at once:

| Lecture concept | How the Hopf scheduler instantiates it |
|---|---|
| **Lec 2 — dynamical systems, fixed points, Lyapunov convergence** | Each eye is a nonlinear dynamical system (`scheduler.py::HopfScanScheduler._deriv`); in *fixate* mode ($\mu<0$) the origin is a stable fixed point the state asymptotically converges to — Lec 2's "the dynamical system will asymptotically converge to $x^*$" |
| **Lec 3 — eigenvalue stability & the oscillation criterion** | The Hopf normal form linearised at the origin has eigenvalues $g\mu \pm i\omega$. Lec 3 (slides 36, 41): *"it oscillates when an eigenvalue has nonzero imaginary part"* ($\omega\neq0$) and *"the fixed point is stable when all eigenvalues have negative real part"* ($g\mu<0$). The **scan↔fixate switch is exactly the real part $g\mu$ crossing zero** |
| **Lec 8 — bifurcations in multi-stable / hybrid systems; CPGs** | Lec 8 (slide 33) lists *"bifurcations in multi-stable systems"* and *"discrete decisions: fight or flight"*; slide 34 is *"Central Pattern Generators and Dynamical Movement Primitives."* The detection-driven scan→fixate transition is a supercritical **Hopf bifurcation** used as exactly such a discrete decision |
| **Lec 9 — CPGs modulated by a feedforward drive** | Lec 9 (slide on CPGs): *"$u_{ff}$ drives frequency / amplitude of the oscillator."* Target detection modulates the oscillator's bifurcation parameter $\mu$ — the CPG analog of a feedforward drive |

Code: `stomatopod_vision/scheduler.py::HopfScanScheduler`; benchmark in
`src/experiments/benchmark_dynamical.py`; report §sec:bonus\_hopf.

---

*Last reviewed: 2026-07-04.*
