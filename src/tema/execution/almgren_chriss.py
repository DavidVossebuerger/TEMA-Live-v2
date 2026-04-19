from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _effective_fill_rate(
    *,
    turnover: float,
    volatility: float,
    n_slices: int,
    risk_aversion: float,
    temporary_impact: float,
    permanent_impact: float,
) -> float:
    slices = max(int(n_slices), 1)
    sigma = max(float(volatility), 0.0)
    lam = max(float(risk_aversion), 0.0)
    eta = max(float(temporary_impact), 0.0)
    gamma = max(float(permanent_impact), 0.0)

    urgency = lam * sigma * math.sqrt(float(slices))
    impact_drag = eta * max(float(turnover), 0.0) + gamma
    raw = (1.0 / float(slices)) * (1.0 + urgency) / (1.0 + impact_drag)
    return float(np.clip(raw, 0.05, 1.0))


def apply_almgren_chriss_step(
    prev_weights: Sequence[float],
    target_weights: Sequence[float],
    *,
    volatility: float,
    n_slices: int = 4,
    risk_aversion: float = 1.0,
    temporary_impact: float = 0.10,
    permanent_impact: float = 0.01,
) -> tuple[np.ndarray, dict]:
    prev = np.asarray(prev_weights, dtype=float).reshape(-1)
    target = np.asarray(target_weights, dtype=float).reshape(-1)
    if prev.shape != target.shape:
        raise ValueError("prev_weights and target_weights must have same shape")

    desired = target - prev
    desired_turnover = float(np.sum(np.abs(desired)))
    fill_rate = _effective_fill_rate(
        turnover=desired_turnover,
        volatility=volatility,
        n_slices=n_slices,
        risk_aversion=risk_aversion,
        temporary_impact=temporary_impact,
        permanent_impact=permanent_impact,
    )
    executed = prev + fill_rate * desired
    return executed, {
        "fill_rate": float(fill_rate),
        "desired_turnover": desired_turnover,
        "volatility": float(volatility),
        "n_slices": int(max(int(n_slices), 1)),
        "risk_aversion": float(max(float(risk_aversion), 0.0)),
        "temporary_impact": float(max(float(temporary_impact), 0.0)),
        "permanent_impact": float(max(float(permanent_impact), 0.0)),
    }

