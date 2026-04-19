from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PortfolioAllocationResult:
    weights: list[float]
    method: str
    used_fallback: bool
    diagnostics: dict


def _normalize_long_only(weights: np.ndarray, w_min: float = 0.0, w_max: float = 1.0) -> np.ndarray:
    x = np.asarray(weights, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, max(0.0, w_min), w_max)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full_like(x, 1.0 / float(x.size))
    return x / total


def _to_returns_array(returns_window: np.ndarray | None, n_assets: int) -> np.ndarray | None:
    if returns_window is None:
        return None
    x = np.asarray(returns_window, dtype=float)
    if x.ndim != 2 or x.shape[1] != n_assets or x.size == 0:
        return None
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _gerber_correlation(returns_window: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    x = np.asarray(returns_window, dtype=float)
    n_assets = x.shape[1]
    if x.shape[0] == 0:
        return np.eye(n_assets, dtype=float)
    std = np.std(x, axis=0, ddof=0)
    threshold = max(float(threshold), 0.0)
    thr = threshold * np.where(std > 1e-12, std, 1.0)
    corr = np.eye(n_assets, dtype=float)
    for i in range(n_assets):
        xi = x[:, i]
        for j in range(i + 1, n_assets):
            xj = x[:, j]
            uu = ((xi >= thr[i]) & (xj >= thr[j])) | ((xi <= -thr[i]) & (xj <= -thr[j]))
            ud = ((xi >= thr[i]) & (xj <= -thr[j])) | ((xi <= -thr[i]) & (xj >= thr[j]))
            concord = float(np.sum(uu))
            discord = float(np.sum(ud))
            denom = concord + discord
            val = 0.0 if denom <= 1e-12 else (concord - discord) / denom
            corr[i, j] = val
            corr[j, i] = val
    return np.clip(corr, -1.0, 1.0)


def _correlation_matrix(
    returns_window: np.ndarray | None,
    n_assets: int,
    corr_backend: str,
    gerber_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = _to_returns_array(returns_window=returns_window, n_assets=n_assets)
    if x is None:
        return np.eye(n_assets, dtype=float), np.ones(n_assets, dtype=float)
    if x.shape[0] < 2:
        return np.eye(n_assets, dtype=float), np.ones(n_assets, dtype=float)
    std = np.std(x, axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    backend = str(corr_backend or "pearson").lower()
    if backend == "gerber":
        corr = _gerber_correlation(x, threshold=gerber_threshold)
    else:
        corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
    return corr, std


def _covariance_with_shrinkage(
    returns_window: np.ndarray | None,
    n_assets: int,
    shrinkage: float,
    cov_backend: str = "sample",
    corr_backend: str = "pearson",
    gerber_threshold: float = 0.5,
) -> np.ndarray:
    x = _to_returns_array(returns_window=returns_window, n_assets=n_assets)
    if x is None:
        return np.eye(n_assets, dtype=float)
    if x.shape[0] < 2:
        return np.eye(n_assets, dtype=float)
    backend = str(cov_backend or "sample").lower()
    if backend == "gerber":
        backend = "correlation"
        corr_backend = "gerber"
    if backend in {"corr", "correlation"}:
        corr, std = _correlation_matrix(
            returns_window=x,
            n_assets=n_assets,
            corr_backend=corr_backend,
            gerber_threshold=gerber_threshold,
        )
        cov = corr * np.outer(std, std)
    else:
        cov = np.cov(x, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
    diag = np.diag(np.diag(cov))
    shrink = min(max(float(shrinkage), 0.0), 1.0)
    return (1.0 - shrink) * cov + shrink * diag + 1e-8 * np.eye(n_assets)


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    variances = np.diag(cov).astype(float)
    inv = 1.0 / np.clip(variances, 1e-12, None)
    return inv / float(np.sum(inv))


def _cluster_variance(cov: np.ndarray, indices: Sequence[int]) -> float:
    idx = list(indices)
    sub = cov[np.ix_(idx, idx)]
    w = _ivp_weights(sub)
    return float(w.T @ sub @ w)


def _hrp_order_from_correlation(corr: np.ndarray) -> list[int]:
    try:
        eigvals, eigvecs = np.linalg.eigh(corr)
        pc1 = eigvecs[:, int(np.argmax(eigvals))]
        order = np.argsort(pc1).tolist()
        if len(order) != corr.shape[0]:
            return list(range(corr.shape[0]))
        return [int(i) for i in order]
    except Exception:
        return list(range(corr.shape[0]))


def _hrp_allocate_from_covariance(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    n_assets = int(cov.shape[0])
    order = _hrp_order_from_correlation(corr)
    weights = np.ones(n_assets, dtype=float)
    clusters: list[list[int]] = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        if not left or not right:
            continue
        left_var = _cluster_variance(cov, left)
        right_var = _cluster_variance(cov, right)
        denom = left_var + right_var
        alloc_left = 0.5 if denom <= 1e-12 else 1.0 - left_var / denom
        alloc_left = float(np.clip(alloc_left, 0.0, 1.0))
        alloc_right = 1.0 - alloc_left
        weights[left] *= alloc_left
        weights[right] *= alloc_right
        clusters.append(left)
        clusters.append(right)
    return _normalize_long_only(weights)


def _black_litterman_like(
    expected_alphas: np.ndarray,
    cov: np.ndarray,
    signals: np.ndarray | None,
    tau: float,
    risk_aversion: float,
    view_confidence: float,
) -> np.ndarray:
    n_assets = expected_alphas.size
    pi = np.zeros(n_assets, dtype=float)
    if signals is not None and signals.size == n_assets:
        pi = 0.5 * np.asarray(signals, dtype=float)
    q = np.asarray(expected_alphas, dtype=float)
    p = np.eye(n_assets, dtype=float)

    tau_cov = max(float(tau), 1e-6) * cov
    confidence = min(max(float(view_confidence), 1e-3), 0.999)
    omega_scale = (1.0 - confidence) / confidence
    omega = omega_scale * np.diag(np.diag(cov) + 1e-8)

    inv_tau_cov = np.linalg.pinv(tau_cov)
    inv_omega = np.linalg.pinv(omega)
    lhs = inv_tau_cov + p.T @ inv_omega @ p
    rhs = inv_tau_cov @ pi + p.T @ inv_omega @ q
    posterior = np.linalg.pinv(lhs) @ rhs
    w = np.linalg.pinv(risk_aversion * cov) @ posterior
    return w.reshape(-1)


def _mean_variance_fallback(
    expected_alphas: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    target = expected_alphas.reshape(-1)
    w = np.linalg.pinv(risk_aversion * cov) @ target
    return w.reshape(-1)


def hrp_allocation_hook(
    expected_alphas: Sequence[float],
    returns_window: np.ndarray | None = None,
    cov_shrinkage: float = 0.15,
    cov_backend: str = "sample",
    corr_backend: str = "pearson",
    gerber_threshold: float = 0.5,
) -> PortfolioAllocationResult:
    n_assets = len(expected_alphas)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method="hrp-hook", used_fallback=True, diagnostics={"hook": "empty"})
    cov = _covariance_with_shrinkage(
        returns_window=returns_window,
        n_assets=n_assets,
        shrinkage=cov_shrinkage,
        cov_backend=cov_backend,
        corr_backend=corr_backend,
        gerber_threshold=gerber_threshold,
    )
    corr, _ = _correlation_matrix(
        returns_window=returns_window,
        n_assets=n_assets,
        corr_backend=corr_backend,
        gerber_threshold=gerber_threshold,
    )
    w = _hrp_allocate_from_covariance(cov, corr).tolist()
    return PortfolioAllocationResult(
        weights=w,
        method="hrp",
        used_fallback=False,
        diagnostics={
            "hook": "hrp",
            "cov_backend": str(cov_backend or "sample").lower(),
            "corr_backend": str(corr_backend or "pearson").lower(),
            "gerber_threshold": float(gerber_threshold),
        },
    )


def herc_cdar_allocation_hook(
    expected_alphas: Sequence[float],
    returns_window: np.ndarray | None = None,
    cov_shrinkage: float = 0.15,
    cov_backend: str = "sample",
    corr_backend: str = "pearson",
    gerber_threshold: float = 0.5,
) -> PortfolioAllocationResult:
    hrp = hrp_allocation_hook(
        expected_alphas=expected_alphas,
        returns_window=returns_window,
        cov_shrinkage=cov_shrinkage,
        cov_backend=cov_backend,
        corr_backend=corr_backend,
        gerber_threshold=gerber_threshold,
    )
    diag = dict(hrp.diagnostics or {})
    diag["extension_point"] = "herc_cdar_todo"
    return PortfolioAllocationResult(
        weights=hrp.weights,
        method="herc-cdar-hook",
        used_fallback=True,
        diagnostics=diag,
    )


def nco_allocation_hook(expected_alphas: Sequence[float]) -> PortfolioAllocationResult:
    n_assets = len(expected_alphas)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method="nco-hook", used_fallback=True, diagnostics={"hook": "empty"})
    abs_alpha = np.abs(np.asarray(expected_alphas, dtype=float))
    total = float(abs_alpha.sum())
    if total <= 1e-12:
        w = [1.0 / n_assets for _ in range(n_assets)]
    else:
        w = (abs_alpha / total).tolist()
    return PortfolioAllocationResult(
        weights=w,
        method="nco-hook",
        used_fallback=True,
        diagnostics={"hook": "placeholder-deterministic-abs-alpha"},
    )


def allocate_portfolio_weights(
    expected_alphas: Sequence[float],
    returns_window: np.ndarray | None,
    signals: Sequence[float] | None = None,
    method: str = "bl",
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_confidence: float = 0.65,
    cov_shrinkage: float = 0.15,
    cov_backend: str = "sample",
    corr_backend: str = "pearson",
    gerber_threshold: float = 0.5,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioAllocationResult:
    alpha = np.asarray(expected_alphas, dtype=float).reshape(-1)
    n_assets = int(alpha.size)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method=method, used_fallback=True, diagnostics={"reason": "no-assets"})

    signal_vec = None if signals is None else np.asarray(signals, dtype=float).reshape(-1)
    cov = _covariance_with_shrinkage(
        returns_window=returns_window,
        n_assets=n_assets,
        shrinkage=cov_shrinkage,
        cov_backend=cov_backend,
        corr_backend=corr_backend,
        gerber_threshold=gerber_threshold,
    )

    selected_method = (method or "bl").lower()
    used_fallback = False
    diagnostics: dict = {"n_assets": n_assets}
    try:
        if selected_method == "hrp":
            return hrp_allocation_hook(
                alpha.tolist(),
                returns_window=returns_window,
                cov_shrinkage=cov_shrinkage,
                cov_backend=cov_backend,
                corr_backend=corr_backend,
                gerber_threshold=gerber_threshold,
            )
        if selected_method in ("herc", "herc_cdar", "herc-cdar"):
            return herc_cdar_allocation_hook(
                alpha.tolist(),
                returns_window=returns_window,
                cov_shrinkage=cov_shrinkage,
                cov_backend=cov_backend,
                corr_backend=corr_backend,
                gerber_threshold=gerber_threshold,
            )
        if selected_method == "nco":
            return nco_allocation_hook(alpha.tolist())
        if selected_method in ("mv", "mean_variance"):
            raw = _mean_variance_fallback(alpha, cov=cov, risk_aversion=max(risk_aversion, 1e-6))
            selected_method = "mean_variance"
        else:
            raw = _black_litterman_like(
                expected_alphas=alpha,
                cov=cov,
                signals=signal_vec,
                tau=tau,
                risk_aversion=max(risk_aversion, 1e-6),
                view_confidence=view_confidence,
            )
            selected_method = "black_litterman_like"
    except Exception as exc:
        used_fallback = True
        diagnostics["fallback_reason"] = str(exc)
        raw = _mean_variance_fallback(alpha, cov=np.eye(n_assets, dtype=float), risk_aversion=max(risk_aversion, 1e-6))
        selected_method = "mean_variance_fallback"

    weights = _normalize_long_only(raw, w_min=min_weight, w_max=max_weight)
    diagnostics["sum_weights"] = float(weights.sum())
    diagnostics["min_weight"] = float(weights.min()) if weights.size else 0.0
    diagnostics["max_weight"] = float(weights.max()) if weights.size else 0.0
    return PortfolioAllocationResult(
        weights=weights.tolist(),
        method=selected_method,
        used_fallback=used_fallback,
        diagnostics=diagnostics,
    )
