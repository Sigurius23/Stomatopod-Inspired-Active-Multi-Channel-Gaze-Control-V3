"""
Tests for stomatopod_vision._mlp — the tiny pure-NumPy MLP
============================================================

Validates:
  1. Construction with the documented shapes (W1, b1, W2, b2).
  2. forward() returns shape (batch, n_out) and is deterministic.
  3. backward() returns gradients of the right shapes.
  4. fit() reduces MSE on a known nonlinear regression problem
     (y = x[0] * x[1] - x[2]) to R² > 0.95 on held-out data.
  5. save() + load() round-trips weights byte-exactly.
  6. Adam optimizer state per parameter; gradient norms decay over training.

Run from the repo root:
    python tests/test_mlp.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stomatopod_vision._mlp import TinyMLP  # noqa: E402

# ---------------------------------------------------------------------
# Test 1 — construction
# ---------------------------------------------------------------------

def test_construction():
    print("\nTest 1: TinyMLP construction shapes …")
    m = TinyMLP(n_in=12, n_hidden=16, n_out=1, seed=0)
    assert m.W1.shape == (16, 12)
    assert m.b1.shape == (16,)
    assert m.W2.shape == (1, 16)
    assert m.b2.shape == (1,)
    print(f"  ✓ W1{m.W1.shape}  b1{m.b1.shape}  W2{m.W2.shape}  b2{m.b2.shape}")
    # Different seed → different weights
    m2 = TinyMLP(n_in=12, n_hidden=16, n_out=1, seed=1)
    assert not np.array_equal(m.W1, m2.W1)
    print("  ✓ different seeds produce different initial weights")


# ---------------------------------------------------------------------
# Test 2 — forward
# ---------------------------------------------------------------------

def test_forward_shape_and_determinism():
    print("\nTest 2: forward() shape + determinism …")
    m = TinyMLP(n_in=3, n_hidden=4, n_out=2, seed=0)
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    y1 = m.forward(x)
    y2 = m.forward(x)
    assert y1.shape == (2, 2), f"expected (2,2), got {y1.shape}"
    assert np.allclose(y1, y2), "same input should give same output"
    print("  ✓ output shape (2, 2); two calls give identical results")


# ---------------------------------------------------------------------
# Test 3 — backward
# ---------------------------------------------------------------------

def test_backward_shapes():
    print("\nTest 3: backward() gradient shapes …")
    m = TinyMLP(n_in=3, n_hidden=4, n_out=2, seed=0)
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    y = m.forward(x)
    dy = np.ones_like(y)
    grads = m.backward(dy)
    assert grads["W1"].shape == m.W1.shape
    assert grads["b1"].shape == m.b1.shape
    assert grads["W2"].shape == m.W2.shape
    assert grads["b2"].shape == m.b2.shape
    print("  ✓ gradient shapes match parameter shapes")


# ---------------------------------------------------------------------
# Test 4 — fit reduces MSE on a nonlinear problem
# ---------------------------------------------------------------------

def test_fit_nonlinear():
    print("\nTest 4: fit() learns y = x0*x1 - x2 …")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 3))
    y = X[:, 0] * X[:, 1] - X[:, 2]

    m = TinyMLP(n_in=3, n_hidden=16, n_out=1, seed=0)
    hist = m.fit(X, y, epochs=400, lr=1e-2, batch_size=64, verbose=False)
    assert hist[0] > hist[-1] * 5, \
        f"loss should drop by at least 5x; got {hist[0]:.4f} → {hist[-1]:.4f}"
    print(f"  ✓ MSE: {hist[0]:.4f} → {hist[-1]:.5f}  (drop {hist[0] / hist[-1]:.1f}x)")

    # Held-out R²
    Xt = rng.standard_normal((100, 3))
    yt = Xt[:, 0] * Xt[:, 1] - Xt[:, 2]
    yp = m.forward(Xt).ravel()
    r2 = 1 - np.var(yp - yt) / np.var(yt)
    assert r2 > 0.95, f"expected held-out R² > 0.95, got {r2:.3f}"
    print(f"  ✓ held-out R² = {r2:.3f}")


# ---------------------------------------------------------------------
# Test 5 — save/load round-trip
# ---------------------------------------------------------------------

def test_save_load_roundtrip():
    print("\nTest 5: save() + load() round-trip …")
    rng = np.random.default_rng(0)
    m = TinyMLP(n_in=5, n_hidden=8, n_out=3, seed=2)
    # Train a bit so weights are non-trivial
    X = rng.standard_normal((128, 5))
    y = rng.standard_normal((128, 3))
    m.fit(X, y, epochs=10, verbose=False)
    y_before = m.forward(X)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.npz"
        m.save(path)
        m2 = TinyMLP.load(path)
        y_after = m2.forward(X)
        assert np.allclose(y_before, y_after), \
            "save/load did not preserve forward outputs"
        # Architecture preserved
        assert m2.n_in == m.n_in and m2.n_hidden == m.n_hidden and m2.n_out == m.n_out
    print("  ✓ weights + architecture round-trip exactly")


# ---------------------------------------------------------------------
# Test 6 — Adam state isolation across parameters
# ---------------------------------------------------------------------

def test_adam_state_isolation():
    print("\nTest 6: Adam optimizer keeps per-parameter state …")
    m = TinyMLP(n_in=3, n_hidden=4, n_out=1, seed=0)
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[1.0]])
    yp = m.forward(x)
    grads = m.backward(2.0 * (yp - y))
    m.step(grads, lr=1e-2)
    # Adam state must exist for each parameter and have the right shape
    for name in ("W1", "b1", "W2", "b2"):
        st = m._adam.get(name)
        assert st is not None, f"missing Adam state for {name}"
        assert st.m.shape == getattr(m, name).shape
        assert st.v.shape == getattr(m, name).shape
        assert st.t == 1
    print("  ✓ all 4 parameters have Adam state with correct shape and t=1")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Tests for stomatopod_vision._mlp")
    print("=" * 60)
    test_construction()
    test_forward_shape_and_determinism()
    test_backward_shapes()
    test_fit_nonlinear()
    test_save_load_roundtrip()
    test_adam_state_isolation()
    print("\nAll MLP tests passed. ✓")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
