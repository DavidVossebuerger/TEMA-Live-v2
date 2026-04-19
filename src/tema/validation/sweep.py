from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def parse_int_list(spec: str) -> list[int]:
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_float_list(spec: str) -> list[float]:
    if spec is None:
        return []
    s = str(spec).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def summarize_sweep(
    per_run_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    sharpe_threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate per-run sweep results.

    Expected columns in per_run_df:
    - group columns (e.g. vol_target_annual, ml_position_scalar_max)
    - base_test_sharpe, ml_test_sharpe
    - base_train_sharpe, ml_train_sharpe

    The function is tolerant: missing numeric cols are ignored.
    """

    if per_run_df is None or per_run_df.empty:
        return pd.DataFrame()

    df = per_run_df.copy()

    if "ml_test_sharpe" in df.columns and "base_test_sharpe" in df.columns:
        df["ml_win_vs_base"] = (df["ml_test_sharpe"].astype(float) > df["base_test_sharpe"].astype(float)).astype(int)
    if "ml_test_sharpe" in df.columns:
        df["ml_pass_threshold"] = (df["ml_test_sharpe"].astype(float) >= float(sharpe_threshold)).astype(int)

    metric_cols = [
        c
        for c in [
            "base_train_sharpe",
            "base_test_sharpe",
            "ml_train_sharpe",
            "ml_test_sharpe",
            "base_overfit_gap",
            "ml_overfit_gap",
            "ml_win_vs_base",
            "ml_pass_threshold",
            "hard_gate_passed",
            "runtime_s",
        ]
        if c in df.columns
    ]

    if not metric_cols:
        return pd.DataFrame()

    grouped = df.groupby(list(group_cols), dropna=False)

    rows: list[dict] = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        out: dict = {k: v for k, v in zip(group_cols, keys)}
        out["n_runs"] = int(len(sub))

        for col in metric_cols:
            s = pd.to_numeric(sub[col], errors="coerce")
            if s.isna().all():
                continue
            out[f"{col}_mean"] = float(s.mean())
            out[f"{col}_median"] = float(s.median())
            out[f"{col}_std"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
            out[f"{col}_min"] = float(s.min())
            out[f"{col}_max"] = float(s.max())

        rows.append(out)

    summary = pd.DataFrame(rows)
    # stable sort for readability
    return summary.sort_values(list(group_cols)).reset_index(drop=True)


def build_heatmap(
    summary_df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    if not all(c in summary_df.columns for c in (index, columns, values)):
        return pd.DataFrame()

    pivot = summary_df.pivot(index=index, columns=columns, values=values)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    # Values should be numeric, but keep this tolerant to unexpected types.
    try:
        return pivot.astype(float)
    except Exception:
        return pivot


def pick_best_config(
    summary_df: pd.DataFrame,
    *,
    score_col: str = "ml_test_sharpe_mean",
) -> dict:
    if summary_df is None or summary_df.empty:
        return {}
    if score_col not in summary_df.columns:
        return {}
    s = pd.to_numeric(summary_df[score_col], errors="coerce")
    if s.isna().all():
        return {}
    idx = int(s.idxmax())
    row = summary_df.loc[idx].to_dict()
    # Convert numpy scalars to native
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
    return row


def pick_best_config_with_hard_gate(
    summary_df: pd.DataFrame,
    *,
    score_col: str = "ml_test_sharpe_mean",
    hard_gate_col: str = "hard_gate_passed_mean",
) -> dict:
    if summary_df is None or summary_df.empty:
        return {}
    if score_col not in summary_df.columns:
        return {}

    df = summary_df.copy()
    score = pd.to_numeric(df[score_col], errors="coerce")
    if score.isna().all():
        return {}
    df["_score"] = score
    df = df[df["_score"].notna()]
    if df.empty:
        return {}

    if hard_gate_col in df.columns:
        gate_raw = pd.to_numeric(df[hard_gate_col], errors="coerce").fillna(0.0)
        df["_hard_gate"] = (gate_raw >= 0.5).astype(int)
    else:
        df["_hard_gate"] = 0

    ranked = df.sort_values(["_hard_gate", "_score"], ascending=[False, False], kind="mergesort")
    row = ranked.iloc[0].to_dict()
    row.pop("_score", None)
    row.pop("_hard_gate", None)
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
    return row
