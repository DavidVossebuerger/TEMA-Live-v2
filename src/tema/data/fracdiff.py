from __future__ import annotations

import math

import numpy as np
import pandas as pd


def get_fracdiff_weights(
    order: float,
    *,
    threshold: float = 1e-5,
    max_terms: int = 256,
) -> np.ndarray:
    """Return deterministic fractional-differencing weights."""
    d = float(order)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("order must be a finite float >= 0")
    if not np.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError("threshold must be a finite float > 0")
    if int(max_terms) <= 0:
        raise ValueError("max_terms must be > 0")

    weights = [1.0]
    for k in range(1, int(max_terms)):
        next_w = -weights[-1] * (d - k + 1.0) / float(k)
        weights.append(float(next_w))
        if abs(next_w) < float(threshold):
            break
    return np.asarray(weights, dtype=float)


def fractionally_differentiate(
    series: pd.Series,
    *,
    order: float = 0.4,
    threshold: float = 1e-5,
    max_terms: int = 256,
    fillna_value: float = 0.0,
) -> pd.Series:
    """Apply fixed-width fractional differencing with safe defaults."""
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series")
    if not np.isfinite(float(fillna_value)):
        raise ValueError("fillna_value must be finite")

    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.empty:
        return s.copy()

    values = s.fillna(float(fillna_value)).to_numpy(dtype=float)
    weights = get_fracdiff_weights(
        float(order),
        threshold=float(threshold),
        max_terms=min(int(max_terms), len(values)),
    )

    out = np.zeros_like(values, dtype=float)
    n = len(values)
    n_w = len(weights)
    for i in range(n):
        width = min(i + 1, n_w)
        window = values[i - width + 1 : i + 1]
        out[i] = float(np.dot(weights[:width], window[::-1]))

    result = pd.Series(out, index=s.index, name=s.name)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        result = result.replace([math.inf, -math.inf], float(fillna_value)).fillna(float(fillna_value))
    return result
