from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


_NORM = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def _to_returns_array(returns: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("returns must be finite")
    return arr


def compute_sample_moments(returns: list[float] | np.ndarray) -> dict[str, float]:
    arr = _to_returns_array(returns)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std <= 1e-12:
        skew = 0.0
        kurtosis = 3.0
    else:
        z = (arr - mean) / std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
    return {
        "n_obs": int(arr.size),
        "mean": mean,
        "std": std,
        "skewness": skew,
        "kurtosis": kurtosis,
    }


def compute_observed_sharpe(
    returns: list[float] | np.ndarray,
    annualization_factor: float = 252.0,
    mode: str = "mean_std",
) -> float:
    arr = _to_returns_array(returns)
    if mode != "mean_std":
        raise ValueError(f"unsupported Sharpe mode: {mode}")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((mean / std) * math.sqrt(float(annualization_factor)))


def probabilistic_sharpe_ratio(
    returns: list[float] | np.ndarray,
    sr_benchmark: float = 0.0,
    annualization_factor: float = 252.0,
    mode: str = "mean_std",
) -> dict[str, float | int | bool | str]:
    moments = compute_sample_moments(returns)
    n_obs = int(moments["n_obs"])
    if n_obs < 3:
        return {
            "ok": False,
            "reason": "insufficient_observations",
            "n_obs": n_obs,
            "psr": 0.0,
            "z_score": 0.0,
            "observed_sharpe": 0.0,
            "benchmark_sharpe": float(sr_benchmark),
            "annualization_factor": float(annualization_factor),
        }

    observed_sharpe = float(
        compute_observed_sharpe(
            returns=returns,
            annualization_factor=annualization_factor,
            mode=mode,
        )
    )
    skewness = float(moments["skewness"])
    kurtosis = float(moments["kurtosis"])

    denom_sq = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * (observed_sharpe**2)
    )
    if not np.isfinite(denom_sq) or denom_sq <= 1e-12:
        denom_sq = 1e-12

    z_score = float(
        ((observed_sharpe - float(sr_benchmark)) * math.sqrt(max(n_obs - 1, 1)))
        / math.sqrt(denom_sq)
    )
    psr = float(_NORM.cdf(z_score))
    return {
        "ok": True,
        "n_obs": n_obs,
        "observed_sharpe": observed_sharpe,
        "benchmark_sharpe": float(sr_benchmark),
        "z_score": z_score,
        "psr": psr,
        "annualization_factor": float(annualization_factor),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "denominator": float(math.sqrt(denom_sq)),
    }


def expected_max_sharpe_benchmark(
    n_trials: int,
    sr_mean: float = 0.0,
    sr_std: float = 1.0,
) -> float:
    n = int(n_trials)
    if n < 2:
        raise ValueError("n_trials must be >= 2")
    sigma = max(float(sr_std), 1e-12)
    t1 = _NORM.inv_cdf(1.0 - (1.0 / float(n)))
    t2 = _NORM.inv_cdf(1.0 - (1.0 / (float(n) * math.e)))
    return float(float(sr_mean) + sigma * ((1.0 - _EULER_MASCHERONI) * t1 + _EULER_MASCHERONI * t2))


def deflated_sharpe_ratio(
    returns: list[float] | np.ndarray,
    n_trials: int,
    sr_mean: float = 0.0,
    sr_std: float | None = None,
    annualization_factor: float = 252.0,
    mode: str = "mean_std",
) -> dict[str, float | int | bool | str]:
    moments = compute_sample_moments(returns)
    n_obs = int(moments["n_obs"])
    observed_sharpe = float(
        compute_observed_sharpe(
            returns=returns,
            annualization_factor=annualization_factor,
            mode=mode,
        )
    )
    skewness = float(moments["skewness"])
    kurtosis = float(moments["kurtosis"])
    denom_sq = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * (observed_sharpe**2)
    )
    if not np.isfinite(denom_sq) or denom_sq <= 1e-12:
        denom_sq = 1e-12

    est_sr_std = (
        float(sr_std)
        if sr_std is not None
        else float(math.sqrt(denom_sq / max(n_obs - 1, 1)))
    )
    bench = expected_max_sharpe_benchmark(
        n_trials=n_trials,
        sr_mean=float(sr_mean),
        sr_std=est_sr_std,
    )
    psr_payload = probabilistic_sharpe_ratio(
        returns=returns,
        sr_benchmark=bench,
        annualization_factor=annualization_factor,
        mode=mode,
    )
    return {
        "ok": bool(psr_payload.get("ok", False)),
        "n_obs": int(psr_payload.get("n_obs", n_obs)),
        "observed_sharpe": float(psr_payload.get("observed_sharpe", observed_sharpe)),
        "benchmark_sharpe_deflated": float(bench),
        "n_trials": int(n_trials),
        "sr_mean": float(sr_mean),
        "sr_std": float(est_sr_std),
        "dsr": float(psr_payload.get("psr", 0.0)),
        "z_score": float(psr_payload.get("z_score", 0.0)),
        "annualization_factor": float(annualization_factor),
    }


def pbo_from_rank_logits(rank_logits: list[float] | np.ndarray) -> dict[str, float | int]:
    logits = np.asarray(rank_logits, dtype=float).reshape(-1)
    if logits.size == 0:
        return {"pbo": 0.0, "n_logits": 0, "pbo_count": 0}
    finite = logits[np.isfinite(logits)]
    if finite.size == 0:
        return {"pbo": 0.0, "n_logits": 0, "pbo_count": 0}
    pbo_count = int(np.sum(finite <= 0.0))
    return {
        "pbo": float(pbo_count / finite.size),
        "n_logits": int(finite.size),
        "pbo_count": pbo_count,
    }

