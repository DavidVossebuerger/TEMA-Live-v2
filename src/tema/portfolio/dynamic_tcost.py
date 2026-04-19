from __future__ import annotations

from typing import Sequence

import numpy as np


def _normalize_weights_like_target(weights: np.ndarray, reference: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if w.size == 0:
        return w
    ref_is_long_only = bool(np.all(ref >= -1e-12))
    if ref_is_long_only:
        w = np.clip(w, 0.0, None)
        total = float(np.sum(w))
        if total > 1e-12:
            return w / total
    abs_sum = float(np.sum(np.abs(w)))
    if abs_sum > 1e-12:
        return w / abs_sum
    if ref_is_long_only:
        return np.full(w.size, 1.0 / float(w.size), dtype=float)
    return np.zeros_like(w)


def apply_dynamic_trading_path(
    target_weights: Sequence[Sequence[float]],
    *,
    lambda_cost: float = 0.0,
    aim_multiplier: float = 0.0,
    min_trade_rate: float = 0.10,
    max_trade_rate: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Apply a GP-inspired partial adjustment toward an "aim" portfolio.

    This is a heuristic scheduler, not the Gârleanu-Pedersen closed form.
    The schedule remains deterministic and conservative:
    - aim_t = target_t + aim_multiplier * (target_t - target_{t-1})
    - executed_t = executed_{t-1} + trade_rate_t * (aim_t - executed_{t-1})
    - trade_rate_t decreases with lambda_cost and desired trade size
    """
    target = np.asarray(target_weights, dtype=float)
    if target.ndim != 2:
        raise ValueError("target_weights must be 2D [time, assets]")
    if target.shape[0] == 0:
        return target.copy(), {
            "enabled": False,
            "reason": "empty_schedule",
            "trade_rate_mean": 0.0,
            "turnover_target_annualized": 0.0,
            "turnover_dynamic_annualized": 0.0,
        }

    out = np.zeros_like(target)
    out[0] = _normalize_weights_like_target(target[0], target[0])

    lam = max(float(lambda_cost), 0.0)
    aim_mult = float(aim_multiplier)
    rate_floor = float(np.clip(min_trade_rate, 0.0, 1.0))
    rate_cap = float(np.clip(max_trade_rate, rate_floor, 1.0))
    trade_rates: list[float] = [1.0]

    for t in range(1, target.shape[0]):
        prev_exec = out[t - 1]
        prev_target = target[t - 1]
        cur_target = target[t]
        aim = cur_target + aim_mult * (cur_target - prev_target)
        delta = aim - prev_exec
        desired_turnover = float(np.sum(np.abs(delta)))
        raw_rate = 1.0 / (1.0 + lam * desired_turnover)
        trade_rate = float(np.clip(raw_rate, rate_floor, rate_cap))
        out[t] = _normalize_weights_like_target(prev_exec + trade_rate * delta, cur_target)
        trade_rates.append(trade_rate)

    target_turnover = np.sum(np.abs(np.diff(target, axis=0))) if target.shape[0] > 1 else 0.0
    dynamic_turnover = np.sum(np.abs(np.diff(out, axis=0))) if out.shape[0] > 1 else 0.0
    periods = max(int(out.shape[0]), 1)

    diagnostics = {
        "enabled": True,
        "model_name": "gp_inspired_partial_adjustment",
        "is_closed_form": False,
        "lambda_cost": lam,
        "aim_multiplier": aim_mult,
        "min_trade_rate": rate_floor,
        "max_trade_rate": rate_cap,
        "trade_rate_mean": float(np.mean(trade_rates)),
        "trade_rate_min": float(np.min(trade_rates)),
        "trade_rate_max": float(np.max(trade_rates)),
        "turnover_target_annualized": float(target_turnover / periods * 252.0),
        "turnover_dynamic_annualized": float(dynamic_turnover / periods * 252.0),
    }
    return out, diagnostics
