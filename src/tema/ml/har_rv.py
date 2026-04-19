from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_windows(windows: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not windows:
        raise ValueError("windows must not be empty")
    out = tuple(int(w) for w in windows)
    if any(w <= 0 for w in out):
        raise ValueError("all windows must be > 0")
    return out


def build_har_rv_features(
    returns: pd.Series,
    *,
    windows: tuple[int, ...] = (1, 5, 22),
    shift_by: int = 1,
    use_log: bool = True,
    prefix: str = "har_rv",
) -> pd.DataFrame:
    """Build HAR-RV style features from squared returns (lookahead-safe)."""
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    if int(shift_by) < 0:
        raise ValueError("shift_by must be >= 0")
    win = _normalize_windows(windows)

    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    rv = r.pow(2)
    out = pd.DataFrame(index=r.index)

    for w in win:
        col = rv.rolling(w, min_periods=1).mean()
        if use_log:
            col = np.log1p(col.clip(lower=0.0))
        if int(shift_by) > 0:
            col = col.shift(int(shift_by))
        out[f"{prefix}_{w}"] = col.fillna(0.0).astype(float)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
