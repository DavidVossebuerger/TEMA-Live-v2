from __future__ import annotations

from typing import Sequence

import numpy as np


def run_futuretesting(
    returns: Sequence[float],
    *,
    n_paths: int = 200,
    horizon: int = 126,
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
        }

    paths = max(int(n_paths), 1)
    hz = max(int(horizon), 1)
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, clean.size, size=(paths, hz))
    sampled = clean[idx]
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
        "method": "iid_bootstrap",
        "n_paths": int(paths),
        "horizon": int(hz),
        "terminal_return_mean": float(np.mean(terminal)),
        "terminal_return_p05": float(np.quantile(terminal, 0.05)),
        "terminal_return_p50": float(np.quantile(terminal, 0.50)),
        "terminal_return_p95": float(np.quantile(terminal, 0.95)),
        "max_drawdown_mean": float(np.mean(drawdowns)),
        "max_drawdown_p05": float(np.quantile(drawdowns, 0.05)),
    }
