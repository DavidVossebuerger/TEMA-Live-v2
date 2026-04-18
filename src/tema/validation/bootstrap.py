from __future__ import annotations

from typing import Literal

import numpy as np


BootstrapMethod = Literal["iid", "block"]
MetricName = Literal["sharpe", "annual_return", "annual_vol", "max_drawdown"]


def _to_returns_array(returns: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("returns must be finite")
    return arr


def _sample_bootstrap_indices(
    n_obs: int,
    n_samples: int,
    rng: np.random.Generator,
    method: BootstrapMethod,
    block_size: int,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be > 0")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if method == "iid":
        return rng.integers(0, n_obs, size=(n_samples, n_obs))
    if method != "block":
        raise ValueError(f"unknown bootstrap method: {method}")
    if block_size <= 0:
        raise ValueError("block_size must be > 0 when method='block'")

    indices = np.empty((n_samples, n_obs), dtype=int)
    for i in range(n_samples):
        pos = 0
        while pos < n_obs:
            start = int(rng.integers(0, n_obs))
            block = (start + np.arange(block_size, dtype=int)) % n_obs
            take = min(block_size, n_obs - pos)
            indices[i, pos : pos + take] = block[:take]
            pos += take
    return indices


def sample_bootstrap_paths(
    returns: list[float] | np.ndarray,
    n_samples: int = 2000,
    seed: int = 42,
    method: BootstrapMethod = "block",
    block_size: int = 20,
) -> np.ndarray:
    arr = _to_returns_array(returns)
    rng = np.random.default_rng(seed)
    idx = _sample_bootstrap_indices(
        n_obs=arr.size,
        n_samples=int(n_samples),
        rng=rng,
        method=method,
        block_size=int(block_size),
    )
    return arr[idx]


def _metric_samples(paths: np.ndarray, annualization_factor: float) -> dict[str, np.ndarray]:
    if paths.ndim != 2:
        raise ValueError("paths must be 2D [n_samples, n_obs]")
    n_obs = int(paths.shape[1])
    if n_obs <= 0:
        raise ValueError("paths must have at least one observation per sample")

    gross = np.prod(1.0 + paths, axis=1)
    annual_return = np.where(gross > 0.0, np.power(gross, float(annualization_factor) / float(n_obs)) - 1.0, -1.0)
    annual_vol = np.std(paths, axis=1, ddof=0) * np.sqrt(float(annualization_factor))
    sharpe = np.divide(annual_return, annual_vol, out=np.zeros_like(annual_return), where=annual_vol > 1e-12)

    equity = np.cumprod(1.0 + paths, axis=1)
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdown = equity / np.where(running_peak == 0.0, 1.0, running_peak) - 1.0
    max_drawdown = np.min(drawdown, axis=1)

    return {
        "sharpe": sharpe.astype(float),
        "annual_return": annual_return.astype(float),
        "annual_vol": annual_vol.astype(float),
        "max_drawdown": max_drawdown.astype(float),
    }


def _summarize_samples(samples: np.ndarray, ci_level: float) -> dict[str, float]:
    alpha = (1.0 - float(ci_level)) / 2.0
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples, ddof=0)),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
    }


def bootstrap_metric_confidence_intervals(
    returns: list[float] | np.ndarray,
    n_samples: int = 2000,
    seed: int = 42,
    method: BootstrapMethod = "block",
    block_size: int = 20,
    annualization_factor: float = 252.0,
    ci_level: float = 0.95,
) -> dict:
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError("ci_level must be in (0,1)")

    paths = sample_bootstrap_paths(
        returns=returns,
        n_samples=n_samples,
        seed=seed,
        method=method,
        block_size=block_size,
    )
    metrics = _metric_samples(paths, annualization_factor=float(annualization_factor))
    return {
        "n_obs": int(paths.shape[1]),
        "n_samples": int(paths.shape[0]),
        "method": method,
        "block_size": int(block_size),
        "seed": int(seed),
        "annualization_factor": float(annualization_factor),
        "ci_level": float(ci_level),
        "metrics": {name: _summarize_samples(vals, ci_level=float(ci_level)) for name, vals in metrics.items()},
    }


def bootstrap_compare_returns(
    baseline_returns: list[float] | np.ndarray,
    candidate_returns: list[float] | np.ndarray,
    metric: MetricName = "sharpe",
    n_samples: int = 2000,
    seed: int = 42,
    method: BootstrapMethod = "block",
    block_size: int = 20,
    annualization_factor: float = 252.0,
    ci_level: float = 0.95,
    higher_is_better: bool | None = None,
) -> dict:
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError("ci_level must be in (0,1)")
    if metric not in {"sharpe", "annual_return", "annual_vol", "max_drawdown"}:
        raise ValueError(f"unsupported metric: {metric}")

    base = _to_returns_array(baseline_returns)
    cand = _to_returns_array(candidate_returns)
    n_obs = int(min(base.size, cand.size))
    if n_obs <= 0:
        raise ValueError("aligned return length must be > 0")
    base = base[-n_obs:]
    cand = cand[-n_obs:]

    rng = np.random.default_rng(seed)
    idx = _sample_bootstrap_indices(
        n_obs=n_obs,
        n_samples=int(n_samples),
        rng=rng,
        method=method,
        block_size=int(block_size),
    )
    base_paths = base[idx]
    cand_paths = cand[idx]
    base_metric_samples = _metric_samples(base_paths, annualization_factor=float(annualization_factor))[metric]
    cand_metric_samples = _metric_samples(cand_paths, annualization_factor=float(annualization_factor))[metric]

    if higher_is_better is None:
        higher_is_better = metric in {"sharpe", "annual_return", "max_drawdown"}
    improvement = cand_metric_samples - base_metric_samples if higher_is_better else base_metric_samples - cand_metric_samples

    alpha = (1.0 - float(ci_level)) / 2.0
    ci_lower = float(np.quantile(improvement, alpha))
    ci_upper = float(np.quantile(improvement, 1.0 - alpha))
    p_non_positive = float(np.mean(improvement <= 0.0))
    p_non_negative = float(np.mean(improvement >= 0.0))
    p_value_two_sided = float(min(1.0, 2.0 * min(p_non_positive, p_non_negative)))

    return {
        "metric": metric,
        "n_obs_aligned": int(n_obs),
        "n_samples": int(n_samples),
        "method": method,
        "block_size": int(block_size),
        "seed": int(seed),
        "annualization_factor": float(annualization_factor),
        "ci_level": float(ci_level),
        "higher_is_better": bool(higher_is_better),
        "baseline_mean": float(np.mean(base_metric_samples)),
        "candidate_mean": float(np.mean(cand_metric_samples)),
        "mean_improvement": float(np.mean(improvement)),
        "median_improvement": float(np.median(improvement)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "improvement_probability": float(np.mean(improvement > 0.0)),
        "p_value_two_sided": p_value_two_sided,
        "passed_ci_positive": bool(ci_lower > 0.0),
    }
