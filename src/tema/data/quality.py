from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityConfig:
    max_nan_frac: float = 0.05
    max_gap_days: float = 7.0
    min_price: float = 1e-12


class DataQualityFailed(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        summary = report.get("summary", "") if isinstance(report, dict) else ""
        super().__init__(f"data quality failed: {summary}")


def compute_data_quality_report(
    price_df: pd.DataFrame,
    *,
    cfg: DataQualityConfig | None = None,
) -> dict:
    """Compute lightweight, deterministic data-quality diagnostics.

    This is intentionally conservative and non-invasive: it only reports.

    Returns a dict with:
      - passed: bool
      - summary: short string
      - index: basic index diagnostics
      - per_asset: list of per-asset rows
      - thresholds: thresholds used
    """

    eff = cfg if cfg is not None else DataQualityConfig()

    if price_df is None or price_df.empty:
        return {
            "passed": False,
            "summary": "empty_price_df",
            "index": {"rows": 0},
            "per_asset": [],
            "thresholds": {
                "max_nan_frac": float(eff.max_nan_frac),
                "max_gap_days": float(eff.max_gap_days),
                "min_price": float(eff.min_price),
            },
        }

    panel = price_df.copy()
    idx = pd.DatetimeIndex(panel.index)
    idx = idx[~idx.isna()]

    # Basic index checks
    index_duplicates = int(pd.Index(idx).duplicated().sum())
    is_monotonic = bool(pd.Index(idx).is_monotonic_increasing)

    # Gap diagnostics
    max_gap_days = 0.0
    n_large_gaps = 0
    if len(idx) >= 2:
        deltas = pd.Series(idx).diff().dt.total_seconds().to_numpy(dtype=float) / 86400.0
        deltas = deltas[~np.isnan(deltas)]
        if deltas.size:
            max_gap_days = float(np.max(deltas))
            n_large_gaps = int(np.sum(deltas > float(eff.max_gap_days)))

    per_asset: list[dict] = []
    failed_assets: list[str] = []
    for col in panel.columns:
        s = pd.to_numeric(panel[col], errors="coerce")
        n = int(len(s))
        n_nan = int(s.isna().sum())
        nan_frac = float(n_nan / n) if n > 0 else 1.0
        nonpos = int((s.dropna() <= float(eff.min_price)).sum())
        start = str(s.dropna().index.min()) if s.dropna().size else None
        end = str(s.dropna().index.max()) if s.dropna().size else None

        row = {
            "asset": str(col),
            "rows": n,
            "nan": n_nan,
            "nan_frac": nan_frac,
            "non_positive": nonpos,
            "start": start,
            "end": end,
        }
        per_asset.append(row)

        if nan_frac > float(eff.max_nan_frac) or nonpos > 0:
            failed_assets.append(str(col))

    passed = True
    reasons: list[str] = []
    if index_duplicates > 0:
        passed = False
        reasons.append(f"index_duplicates={index_duplicates}")
    if not is_monotonic:
        passed = False
        reasons.append("index_not_monotonic")
    if n_large_gaps > 0:
        passed = False
        reasons.append(f"gaps_over_{eff.max_gap_days}_days={n_large_gaps}")
    if failed_assets:
        passed = False
        reasons.append(f"failed_assets={len(failed_assets)}")

    summary = "ok" if passed else ";".join(reasons)

    return {
        "passed": bool(passed),
        "summary": str(summary),
        "index": {
            "rows": int(len(panel)),
            "assets": int(panel.shape[1]),
            "start": str(pd.DatetimeIndex(panel.index).min()),
            "end": str(pd.DatetimeIndex(panel.index).max()),
            "monotonic": bool(is_monotonic),
            "duplicates": int(index_duplicates),
            "max_gap_days": float(max_gap_days),
            "n_gaps_over_threshold": int(n_large_gaps),
        },
        "per_asset": per_asset,
        "failed_assets": failed_assets,
        "thresholds": {
            "max_nan_frac": float(eff.max_nan_frac),
            "max_gap_days": float(eff.max_gap_days),
            "min_price": float(eff.min_price),
        },
    }
