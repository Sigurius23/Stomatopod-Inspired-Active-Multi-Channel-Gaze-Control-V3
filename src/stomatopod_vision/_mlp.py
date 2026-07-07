"""
_mlp.py — tiny pure-NumPy multilayer perceptron
================================================

A 2-layer feed-forward network with tanh hidden activation and a linear
output, plus forward, backward, and Adam-style update. ~150 LOC, no
external dependencies beyond NumPy.

Used by :class:`stomatopod_vision.scheduler.LearnedScheduler` so the
bonus learned scheduler can ship without forcing PyTorch or JAX as a
hard dependency.

This is NOT a general-purpose NN library — it has just enough to fit a
small scalar-regression problem (12 features → 1 score). For anything
non-trivial, swap in PyTorch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["TinyMLP", "AdamState"]


@dataclass
class AdamState:
    """Per-parameter Adam optimizer state."""
    m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t: int = 0


class TinyMLP:
    """
    Two-layer MLP with tanh hidden activation and linear output.

    Architecture::

        x  → W1 @ x + b1  → tanh  → W2 @ h + b2  →  y

    Parameters
    ----------
    n_in :
        Input feature dimension.
    n_hidden :
        Hidden layer width (default 16). Small enough to be cheap to
        train on a few thousand examples and still expressive enough
        to learn a non-linear scoring function over our 12 features.
    n_out :
        Output dimension. Defaults to 1 (scalar score).
    seed :
        RNG seed for weight initialisation.

    Notes
    -----
    Weights are initialised with Xavier/Glorot scaling. Biases start at zero.
    """

    def __init__(self, n_in: int, n_hidden: int = 16, n_out: int = 1,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # Xavier: stddev = sqrt(1/n_in)
        self.W1 = rng.standard_normal((n_hidden, n_in)) * np.sqrt(1.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_out, n_hidden)) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)

        # Cached intermediates from the last forward pass (used by backward).
        self._x: np.ndarray | None = None
        self._h: np.ndarray | None = None    # post-tanh hidden activations

        # Per-parameter Adam state, created lazily on first .step() call.
        self._adam: dict[str, AdamState] = {}

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute outputs for inputs ``x`` of shape ``(batch, n_in)``.

        Caches the input and hidden activations so :meth:`backward` can
        compute gradients without recomputation.
        """
        x = np.atleast_2d(x).astype(np.float64)
        z1 = x @ self.W1.T + self.b1
        h = np.tanh(z1)
        y = h @ self.W2.T + self.b2
        self._x = x
        self._h = h
        return y

    def backward(self, dy: np.ndarray) -> dict[str, np.ndarray]:
        """Backprop the gradient of the loss wrt the network output.

        ``dy`` has the same shape as the output of :meth:`forward`.
        Returns a dict of gradients keyed by parameter name.
        """
        assert self._x is not None and self._h is not None, \
            "Call forward() before backward()."
        x, h = self._x, self._h
        # Output layer
        dW2 = dy.T @ h
        db2 = dy.sum(axis=0)
        # Through W2
        dh = dy @ self.W2
        # Through tanh
        dz1 = dh * (1.0 - h * h)
        # Through W1
        dW1 = dz1.T @ x
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ------------------------------------------------------------------
    # Adam step
    # ------------------------------------------------------------------

    def step(self, grads: dict[str, np.ndarray], *,
             lr: float = 1e-2, beta1: float = 0.9, beta2: float = 0.999,
             eps: float = 1e-8) -> None:
        """Apply one Adam update using gradients from :meth:`backward`."""
        for name, g in grads.items():
            p = getattr(self, name)
            st = self._adam.get(name)
            if st is None or st.m.shape != p.shape:
                st = AdamState(m=np.zeros_like(p), v=np.zeros_like(p), t=0)
                self._adam[name] = st
            st.t += 1
            st.m = beta1 * st.m + (1 - beta1) * g
            st.v = beta2 * st.v + (1 - beta2) * (g * g)
            m_hat = st.m / (1 - beta1 ** st.t)
            v_hat = st.v / (1 - beta2 ** st.t)
            setattr(self, name, p - lr * m_hat / (np.sqrt(v_hat) + eps))

    # ------------------------------------------------------------------
    # Convenience: fit a batch dataset with MSE loss
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            epochs: int = 200, batch_size: int = 64,
            lr: float = 1e-2, seed: int = 0,
            verbose: bool = False) -> list[float]:
        """Fit the MLP to ``X → y`` with mean-squared-error loss.

        Returns the per-epoch mean loss history.

        Parameters
        ----------
        X, y :
            ``X`` has shape ``(N, n_in)``; ``y`` has shape ``(N, n_out)``
            or ``(N,)`` (will be reshaped to ``(N, 1)``).
        """
        X = np.atleast_2d(X).astype(np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        assert X.shape[1] == self.n_in, f"X has {X.shape[1]} features, expected {self.n_in}"
        assert y.shape[1] == self.n_out, f"y has {y.shape[1]} cols, expected {self.n_out}"

        n = X.shape[0]
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for epoch in range(int(epochs)):
            perm = rng.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                yp = self.forward(xb)
                diff = (yp - yb)               # shape (batch, n_out)
                loss = float(np.mean(diff ** 2))
                dy = 2.0 * diff / xb.shape[0]  # d(MSE)/dy
                grads = self.backward(dy)
                self.step(grads, lr=lr)
                losses.append(loss)
            mean_loss = float(np.mean(losses))
            history.append(mean_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}/{epochs}  mse={mean_loss:.5f}")
        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write weights + architecture metadata to a single .npz file."""
        np.savez(
            str(path),
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            meta=np.array(json.dumps({
                "n_in": self.n_in, "n_hidden": self.n_hidden, "n_out": self.n_out,
            })),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyMLP":
        """Reconstruct a :class:`TinyMLP` from a file written by :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            mlp = cls(n_in=meta["n_in"], n_hidden=meta["n_hidden"], n_out=meta["n_out"])
            mlp.W1 = data["W1"]
            mlp.b1 = data["b1"]
            mlp.W2 = data["W2"]
            mlp.b2 = data["b2"]
        return mlp
