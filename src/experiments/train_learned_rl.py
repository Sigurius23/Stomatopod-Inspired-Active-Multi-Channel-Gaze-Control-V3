"""
train_learned_rl.py — REINFORCE training of the LearnedScheduler MLP
=====================================================================

The original `train_learned.py` trains the MLP by **imitation** of the
hand-designed scoring function. This script trains the same MLP from
**scratch** by REINFORCE on actual discovery outcomes — no teacher.

Method
------
At each scheduler re-plan during a hard-scene rollout:

  1. Compute features for all `n_candidates` sampled directions per eye.
  2. Score them with the MLP → softmax → sample one candidate per eye
     (instead of argmax).
  3. Drive to those setpoints for `decision_period_s`.
  4. Record `(feature_vector, log_softmax_chosen)` for the chosen
     candidates.

At episode end (after `duration_s` of simulation):

  - Compute episode return = `(coverage * 10)` (range 0–10 on the hard
    scene; one point per interesting target identified).
  - Subtract a running mean baseline to reduce variance.
  - Update the MLP by REINFORCE: maximise
        E[log π(a|s) · (R - b)]
    via gradient ascent on the policy log-prob, weighted by advantage.

Honesty disclaimer
------------------
REINFORCE is notoriously high-variance on episodic problems. After a
modest number of episodes (~100–500) the policy should reliably reach
coverage ≥ 0.8 on the hard scene, but it will usually **NOT** exceed
the hand-designed teacher's 1.00 — it just demonstrates that the
scoring function *can* be learned end-to-end from rewards, not just by
imitation. The script saves a learning curve so the (modest) improvement
over a random-init MLP is visible.

CLI
---
    MUJOCO_GL=egl python src/experiments/train_learned_rl.py
    MUJOCO_GL=egl python src/experiments/train_learned_rl.py \\
        --episodes 200 --duration 5 --lr 5e-3

Outputs
-------
    results/learned_rl/mlp.npz             — trained MLP
    results/learned_rl/training_curve.json — episode-by-episode return
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from _common import (  # noqa: E402
    _REPO_ROOT,
    DEFAULT_RESULTS_DIR,
    build_context,
    run_simulation,
)

from stomatopod_vision._mlp import TinyMLP  # noqa: E402
from stomatopod_vision.gimbal_control import GimbalSetpoint  # noqa: E402
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    LearnedScheduler,
    SaliencyScheduler,
)

HARD_MODEL = _REPO_ROOT / "models" / "stomatopod_eyes_hard.xml"


class _REINFORCEScheduler(LearnedScheduler):
    """LearnedScheduler with softmax sampling + per-replan trace logging.

    The base class returns `argmax(scores)`. For REINFORCE we need to
    *sample* from the policy and remember which candidate we picked so
    we can credit-assign the reward later. We use a temperature-scaled
    softmax to control exploration.
    """

    def __init__(self, mlp: TinyMLP, *, temperature: float = 1.0,
                 n_candidates: int = 30, decision_period_s: float = 0.10,
                 seed: int = 0) -> None:
        super().__init__(mlp=mlp, n_candidates=n_candidates,
                         decision_period_s=decision_period_s, seed=seed)
        self.temperature = float(temperature)
        # trace = list of (features_chosen, log_prob_chosen) per re-plan
        self.trace: list[tuple[np.ndarray, float]] = []

    def next_setpoint(self, time_now: float,
                      current_setpoint: GimbalSetpoint) -> GimbalSetpoint:
        elapsed = float(time_now) - self._last_decision_time
        if elapsed < self.decision_period_s:
            return self._held_setpoint

        chosen: dict[str, tuple[float, float, float]] = {}
        for eye in ("L", "R"):
            cands = self.sample_candidates(eye)
            feats = np.stack([
                self.feature_vector(eye, y, p, r, time_now, current_setpoint)
                for y, p, r in cands
            ])  # (n, 12)
            scores = self.mlp.forward(feats).ravel()  # (n,)
            # Softmax over scores
            z = scores / max(self.temperature, 1e-6)
            z = z - np.max(z)
            probs = np.exp(z)
            probs /= probs.sum()
            idx = int(self.rng.choice(len(cands), p=probs))
            chosen[eye] = tuple(cands[idx])
            log_p = float(np.log(probs[idx] + 1e-12))
            self.trace.append((feats[idx], log_p))

        new_setpoint = GimbalSetpoint(
            yaw_L=chosen["L"][0], pitch_L=chosen["L"][1], roll_L=chosen["L"][2],
            yaw_R=chosen["R"][0], pitch_R=chosen["R"][1], roll_R=chosen["R"][2],
        )
        for eye in ("L", "R"):
            self.memory.last_visit_direction[eye] = (chosen[eye][0], chosen[eye][1])
            self._roll_history[eye].append(chosen[eye][2])
            if len(self._roll_history[eye]) > 50:
                self._roll_history[eye] = self._roll_history[eye][-50:]

        self._held_setpoint = new_setpoint
        self._last_decision_time = float(time_now)
        return new_setpoint


def _rollout_episode(mlp: TinyMLP, *, duration: float, seed: int,
                     temperature: float) -> tuple[float, list]:
    """Run one episode, return (episode_return, scheduler.trace)."""
    ctx = build_context(HARD_MODEL, seed=seed)
    pipeline = PreprocessingPipeline()
    sched = _REINFORCEScheduler(mlp=mlp, temperature=temperature, seed=seed)

    log = run_simulation(
        ctx,
        setpoint_at=lambda t: sched.next_setpoint(t, sched._held_setpoint),
        pipeline=pipeline,
        duration_s=duration,
        quiet=True,
        on_events=lambda evs, t: sched.update_memory(evs, t),
        controller_rate_hz=10.0,
    )
    # Episode return = number of distinct interesting targets identified
    from stomatopod_vision.metrics import coverage
    cov = coverage(log)                      # ∈ [0, 1]
    episode_return = cov * len(log.interesting_targets)
    return float(episode_return), sched.trace


def _reinforce_update(mlp: TinyMLP, trace: list,
                      advantage: float, *, lr: float) -> None:
    """One REINFORCE policy-gradient step on the accumulated trace.

    For each chosen candidate the policy-gradient says:
        ∇_θ J = E[advantage · ∇_θ log π(a|s)]
    Since π is softmax over MLP outputs, the gradient of log π wrt the
    MLP output at the chosen candidate is approximately 1 (for the chosen
    index) and 0 elsewhere when the softmax is sharp. We use that
    approximation: feed each chosen feature through the MLP and step
    its output up by `advantage` (or down if negative).
    """
    if not trace:
        return
    X = np.stack([f for f, _lp in trace])    # (T, 12)
    # Treat each step independently. Target is "MLP output, but bigger
    # by `advantage`" so the gradient nudges in the right direction.
    yp = mlp.forward(X)                      # (T, 1)
    target = yp + advantage                  # (T, 1)
    diff = (yp - target)                     # (T, 1)
    dy = 2.0 * diff / X.shape[0]
    grads = mlp.backward(dy)
    mlp.step(grads, lr=lr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of REINFORCE episodes (default: 200).")
    p.add_argument("--duration", type=float, default=3.0,
                   help="Simulated seconds per episode (default: 3.0).")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Adam learning rate (default: 5e-3).")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature (default: 0.5).")
    p.add_argument("--baseline-window", type=int, default=20,
                   help="Running-mean baseline window (default: 20).")
    p.add_argument("--out", type=Path,
                   default=DEFAULT_RESULTS_DIR / "learned_rl" / "mlp.npz",
                   help="Where to save the trained MLP.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for episode seeds.")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_features = len(SaliencyScheduler.FEATURE_NAMES)
    mlp = TinyMLP(n_in=n_features, n_hidden=16, n_out=1, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("train_learned_rl: REINFORCE on the hard scene")
    print("=" * 70)
    print(f"  episodes        = {args.episodes}")
    print(f"  duration/ep     = {args.duration}s")
    print(f"  lr              = {args.lr}")
    print(f"  temperature     = {args.temperature}")
    print(f"  baseline window = {args.baseline_window}")
    print("=" * 70)

    returns: list[float] = []
    baseline = 0.0
    t0 = time.perf_counter()
    for ep in range(int(args.episodes)):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        R, trace = _rollout_episode(
            mlp, duration=args.duration, seed=ep_seed,
            temperature=args.temperature,
        )
        # Running-mean baseline for variance reduction
        if returns:
            window = returns[-args.baseline_window:]
            baseline = float(np.mean(window))
        advantage = R - baseline
        _reinforce_update(mlp, trace, advantage=advantage, lr=args.lr)
        returns.append(R)

        if ep % 10 == 0 or ep == args.episodes - 1:
            recent = returns[-args.baseline_window:]
            print(f"  ep {ep:4d}  R={R:5.2f}  "
                  f"baseline={baseline:5.2f}  "
                  f"recent_mean={np.mean(recent):5.2f}±{np.std(recent):4.2f}  "
                  f"({time.perf_counter() - t0:5.1f}s)")

    mlp.save(args.out)
    log = args.out.with_name("training_curve.json")
    log.write_text(json.dumps({
        "args": vars(args) | {"out": str(args.out)},
        "returns": returns,
        "final_recent_mean": float(np.mean(returns[-args.baseline_window:])),
        "max_return_achieved": float(max(returns)),
        "wall_clock_s": time.perf_counter() - t0,
    }, indent=2, default=str))
    print()
    print(f"saved MLP → {args.out}")
    print(f"saved curve → {log}")
    print(f"final {args.baseline_window}-episode mean return: "
          f"{float(np.mean(returns[-args.baseline_window:])):.2f}")
    print(f"max single-episode return: {max(returns):.2f}")


if __name__ == "__main__":
    main()
