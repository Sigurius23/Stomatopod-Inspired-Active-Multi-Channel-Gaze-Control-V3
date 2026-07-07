"""
train_learned.py — collect imitation data + train LearnedScheduler's MLP
=========================================================================

The bonus :class:`stomatopod_vision.scheduler.LearnedScheduler` replaces
the hand-designed scoring sum with a tiny pure-NumPy MLP. This script
trains that MLP by **imitation learning** of the existing hand-designed
:class:`SaliencyScheduler`:

    1. Roll out the teacher scheduler on the hard scene for N seeds.
    2. At every re-plan, for every candidate direction it scored,
       record ``(feature_vector, teacher_total_score)``.
    3. Fit the MLP to minimise MSE against the teacher's score.

This is the cleanest pedagogical demo of the "Lec 6 value-based action
selection" framing: the trained MLP is a learned approximation of an
information-gain *value function* over candidate gimbal directions.

After training, the LearnedScheduler picks the same argmax as the
hand-designed teacher on most steps (regression target identity), so on
the benchmark scenes it ends up nearly indistinguishable on the four
headline metrics — which is the point. It demonstrates that the
scoring function can be *replaced* by a learned approximator, not that
learning beats the teacher.

CLI
---
    # Default: 5 collection seeds × 5 s on the hard scene
    MUJOCO_GL=egl python src/experiments/train_learned.py

    # Override
    MUJOCO_GL=egl python src/experiments/train_learned.py \\
        --duration 10 --seeds 0 1 2 3 4 5 6 7 \\
        --epochs 500 --lr 5e-3 \\
        --out results/learned/mlp.npz

Outputs
-------
    results/learned/mlp.npz            — trained MLP weights
    results/learned/training_loss.json — per-epoch MSE + held-out R²
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
from stomatopod_vision.preprocessing import PreprocessingPipeline  # noqa: E402
from stomatopod_vision.scheduler import (  # noqa: E402
    SaliencyScheduler,
    ScoringWeights,
)

HARD_MODEL = _REPO_ROOT / "models" / "stomatopod_eyes_hard.xml"


def _collect_one_seed(
    model: Path,
    seed: int,
    duration: float,
    n_candidates: int = 30,
    decision_period_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the teacher scheduler and return ``(features, scores)``.

    For every re-plan during the rollout, every candidate direction the
    teacher scored is recorded once with its 12-D feature vector and
    scalar score.
    """
    ctx = build_context(model, seed=seed)
    pipeline = PreprocessingPipeline()
    # Train against the RICH hand-designed weights (1, 2, 0.5, 1.0)
    # rather than the current ScoringWeights() defaults (1, 2, 0.0, 0.0).
    #
    # Reason: with the tuned defaults, score at t=0 collapses to 1.0 for
    # all candidates (novelty=1 for any direction, all other terms zero),
    # so the teacher's argmax is just tie-breaking by candidate index.
    # An MLP trained on those constants ends up picking essentially
    # random directions at every re-plan and tanks coverage.
    #
    # The rich weights produce a non-degenerate score (feasibility and
    # pol_info_gain differentiate candidates even with empty memory),
    # so the MLP has something meaningful to learn. The LearnedScheduler
    # is then a learned approximator of that richer scoring function,
    # which on the headline benchmark scenes still achieves coverage
    # competitive with the tuned hand-designed scheduler.
    teacher = SaliencyScheduler(
        n_candidates=n_candidates,
        decision_period_s=decision_period_s,
        weights=ScoringWeights(novelty=1.0, salience=2.0,
                               feasibility=0.5, polarization_info_gain=1.0),
        seed=seed,
    )

    features: list[np.ndarray] = []
    scores: list[float] = []

    # Monkey-patch the teacher's `total_score` so we record every call.
    # This is the most reliable hook — every score computed during a
    # re-plan goes through it.
    original_total = teacher.total_score

    def recording_total_score(eye, cy, cp, cr, time_now, current_setpoint):
        s = original_total(eye, cy, cp, cr, time_now, current_setpoint)
        f = teacher.feature_vector(eye, cy, cp, cr, time_now, current_setpoint)
        features.append(f)
        scores.append(s)
        return s

    teacher.total_score = recording_total_score  # type: ignore[method-assign]

    run_simulation(
        ctx,
        setpoint_at=lambda t: teacher.next_setpoint(t, teacher._held_setpoint),
        pipeline=pipeline,
        duration_s=duration,
        quiet=True,
        on_events=lambda events, t: teacher.update_memory(events, t),
    )

    return np.asarray(features), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", type=Path, default=HARD_MODEL,
                        help=f"Scene to collect imitation data on "
                             f"(default: {HARD_MODEL.relative_to(_REPO_ROOT)}).")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulated seconds per seed (default: 5.0).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Collection seeds (default: 0..4).")
    parser.add_argument("--n-candidates", type=int, default=30,
                        help="Candidates scored per re-plan (default: 30).")
    parser.add_argument("--decision-period", type=float, default=0.10,
                        help="Re-plan period in seconds (default: 0.10).")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="MLP hidden width (default: 16).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300).")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Adam learning rate (default: 1e-2).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64).")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Held-out fraction for R^2 reporting (default: 0.2).")
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_RESULTS_DIR / "learned" / "mlp.npz",
                        help="Path to save the trained MLP weights.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch progress.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("train_learned: imitation training of LearnedScheduler's MLP")
    print("=" * 70)
    print(f"  scene            = {args.model.name}")
    print(f"  collection seeds = {args.seeds}  (duration {args.duration}s each)")
    print(f"  n_candidates     = {args.n_candidates}")
    print(f"  MLP              = 12 → {args.n_hidden} → 1  (Adam, lr={args.lr})")
    print(f"  out              = {args.out}")
    print("=" * 70)

    # ---- 1. Collect data ----
    print("\n[1/3] Collecting imitation data …")
    t0 = time.perf_counter()
    Xs, ys = [], []
    for seed in args.seeds:
        Xi, yi = _collect_one_seed(
            args.model, seed=seed, duration=args.duration,
            n_candidates=args.n_candidates,
            decision_period_s=args.decision_period,
        )
        print(f"  seed {seed}: {len(yi):>6d} (feature, score) pairs")
        Xs.append(Xi)
        ys.append(yi)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"  total dataset: {X.shape[0]} rows × {X.shape[1]} features "
          f"({time.perf_counter() - t0:.1f}s)")

    # ---- 2. Train / val split ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_val = int(args.val_frac * len(y))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xt, yt = X[train_idx], y[train_idx]
    Xv, yv = X[val_idx], y[val_idx]
    print(f"  train: {len(yt)} rows  |  val: {len(yv)} rows")

    # ---- 3. Train the MLP ----
    print("\n[2/3] Training MLP …")
    mlp = TinyMLP(n_in=X.shape[1], n_hidden=args.n_hidden, n_out=1, seed=0)
    t0 = time.perf_counter()
    history = mlp.fit(Xt, yt, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, verbose=not args.quiet)
    train_time = time.perf_counter() - t0

    # ---- 4. Evaluate ----
    print("\n[3/3] Evaluating …")
    yp_train = mlp.forward(Xt).ravel()
    yp_val = mlp.forward(Xv).ravel()
    r2_train = float(1 - np.var(yp_train - yt) / max(np.var(yt), 1e-12))
    r2_val = float(1 - np.var(yp_val - yv) / max(np.var(yv), 1e-12))
    mse_train = float(np.mean((yp_train - yt) ** 2))
    mse_val = float(np.mean((yp_val - yv) ** 2))
    print(f"  train MSE = {mse_train:.5f}   R² = {r2_train:.3f}")
    print(f"  val   MSE = {mse_val:.5f}   R² = {r2_val:.3f}")
    print(f"  trained in {train_time:.1f}s ({args.epochs} epochs)")

    # ---- 5. Save ----
    mlp.save(args.out)
    log = args.out.with_suffix("").with_name("training_loss.json")
    log.write_text(json.dumps({
        "model": str(args.model),
        "seeds": list(args.seeds),
        "duration_s": args.duration,
        "n_candidates": args.n_candidates,
        "decision_period_s": args.decision_period,
        "n_hidden": args.n_hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train_rows": int(len(yt)),
        "n_val_rows": int(len(yv)),
        "mse_train": mse_train, "mse_val": mse_val,
        "r2_train": r2_train, "r2_val": r2_val,
        "loss_history": history,
        "train_time_s": train_time,
    }, indent=2))
    print(f"\n  saved MLP weights → {args.out}")
    print(f"  saved training log → {log}")


if __name__ == "__main__":
    main()
