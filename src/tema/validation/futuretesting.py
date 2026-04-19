from __future__ import annotations

from typing import Sequence

import numpy as np


def _default_block_size(n_obs: int) -> int:
    return max(1, int(round(float(max(n_obs, 1)) ** (1.0 / 3.0))))


def _sample_stationary_bootstrap(
    clean_returns: np.ndarray,
    *,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_obs = int(clean_returns.size)
    p_restart = 1.0 / float(max(int(block_size), 1))
    idx = np.empty(int(horizon), dtype=int)
    idx[0] = int(rng.integers(0, n_obs))
    for t in range(1, int(horizon)):
        if float(rng.random()) < p_restart:
            idx[t] = int(rng.integers(0, n_obs))
        else:
            idx[t] = int((idx[t - 1] + 1) % n_obs)
    return clean_returns[idx]


def run_futuretesting(
    returns: Sequence[float],
    *,
    n_paths: int = 200,
    horizon: int = 126,
    method: str = "stationary_bootstrap",
    block_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Generate synthetic forward paths from historical returns (experimental)."""
    arr = np.asarray(returns, dtype=float).reshape(-1)
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "empty_returns",
            "n_paths": int(max(int(n_paths), 1)),
            "horizon": int(max(int(horizon), 1)),
            "method": str(method),
        }

    paths = max(int(n_paths), 1)
    hz = max(int(horizon), 1)
    rng = np.random.default_rng(seed)
    method_norm = str(method or "stationary_bootstrap").strip().lower()
    if method_norm not in {"iid_bootstrap", "stationary_bootstrap"}:
        raise ValueError("method must be one of: iid_bootstrap, stationary_bootstrap")
    block = int(block_size) if block_size is not None else _default_block_size(int(clean.size))
    block = max(block, 1)

    if method_norm == "iid_bootstrap":
        idx = rng.integers(0, clean.size, size=(paths, hz))
        sampled = clean[idx]
    else:
        sampled = np.vstack(
            [
                _sample_stationary_bootstrap(clean, horizon=hz, block_size=block, rng=rng)
                for _ in range(paths)
            ]
        )
    equity = np.cumprod(1.0 + sampled, axis=1)
    terminal = equity[:, -1] - 1.0

    drawdowns = []
    for i in range(paths):
        curve = equity[i]
        running_max = np.maximum.accumulate(curve)
        dd = curve / np.where(running_max == 0.0, 1.0, running_max) - 1.0
        drawdowns.append(float(np.min(dd)))

    return {
        "enabled": True,
        "skipped": False,
        "method": method_norm,
        "n_paths": int(paths),
        "horizon": int(hz),
        "block_size": (int(block) if method_norm == "stationary_bootstrap" else None),
        "restart_probability": (float(1.0 / float(block)) if method_norm == "stationary_bootstrap" else None),
        "terminal_return_mean": float(np.mean(terminal)),
        "terminal_return_p05": float(np.quantile(terminal, 0.05)),
        "terminal_return_p50": float(np.quantile(terminal, 0.50)),
        "terminal_return_p95": float(np.quantile(terminal, 0.95)),
        "max_drawdown_mean": float(np.mean(drawdowns)),
        "max_drawdown_p05": float(np.quantile(drawdowns, 0.05)),
    }
