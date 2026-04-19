from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .probabilistic import pbo_from_rank_logits


def _metric_from_returns(returns: np.ndarray, metric: str, annualization_factor: float) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if metric == "annual_return":
        gross = float(np.prod(1.0 + arr))
        if gross <= 0.0:
            return -1.0
        return float(gross ** (float(annualization_factor) / float(arr.size)) - 1.0)
    if metric == "annual_vol":
        return float(np.std(arr, ddof=0) * np.sqrt(float(annualization_factor)))
    if metric == "max_drawdown":
        equity = np.cumprod(1.0 + arr)
        running_peak = np.maximum.accumulate(equity)
        drawdown = equity / np.where(running_peak == 0.0, 1.0, running_peak) - 1.0
        return float(np.min(drawdown))
    if metric != "sharpe":
        raise ValueError(f"unsupported metric: {metric}")
    vol = float(np.std(arr, ddof=0))
    if vol <= 1e-12:
        return 0.0
    return float((np.mean(arr) / vol) * np.sqrt(float(annualization_factor)))


def build_cpcv_groups(index: pd.Index, n_groups: int) -> list[np.ndarray]:
    n = int(len(index))
    if n <= 0:
        return []
    groups = max(int(n_groups), 2)
    return [chunk.astype(int) for chunk in np.array_split(np.arange(n, dtype=int), groups) if chunk.size > 0]


def generate_cpcv_splits(
    index: pd.Index,
    n_groups: int = 10,
    n_test_groups: int = 2,
    purge_groups: int = 1,
    embargo_groups: int = 1,
    max_splits: int | None = None,
    seed: int = 42,
) -> list[dict]:
    groups = build_cpcv_groups(index, n_groups=n_groups)
    n = len(groups)
    k = int(n_test_groups)
    if n < 2 or k <= 0 or k >= n:
        return []
    all_combos = list(combinations(range(n), k))
    if max_splits is not None and len(all_combos) > int(max_splits):
        rng = np.random.default_rng(int(seed))
        selected_idx = np.sort(rng.choice(len(all_combos), size=int(max_splits), replace=False))
        combos = [all_combos[int(i)] for i in selected_idx]
    else:
        combos = all_combos

    guard = max(int(purge_groups), int(embargo_groups), 0)
    splits: list[dict] = []
    for split_id, test_groups in enumerate(combos):
        excluded: set[int] = set()
        for g in test_groups:
            excluded.add(int(g))
            for delta in range(1, guard + 1):
                if g - delta >= 0:
                    excluded.add(int(g - delta))
                if g + delta < n:
                    excluded.add(int(g + delta))

        test_idx = np.concatenate([groups[g] for g in test_groups]).astype(int)
        train_groups = [g for g in range(n) if g not in excluded]
        if not train_groups:
            continue
        train_idx = np.concatenate([groups[g] for g in train_groups]).astype(int)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append(
            {
                "split_id": int(split_id),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "test_groups": [int(g) for g in test_groups],
                "excluded_groups": sorted(int(g) for g in excluded),
            }
        )
    return splits


def _rank_percentile(values: dict[str, float], selected_key: str) -> float:
    finite_items = [(k, float(v)) for k, v in values.items() if np.isfinite(float(v))]
    if not finite_items:
        return 0.5
    finite_items = sorted(finite_items, key=lambda item: item[1], reverse=True)
    order = [k for k, _ in finite_items]
    if selected_key not in order:
        return 0.5
    if len(order) == 1:
        return 1.0
    rank = int(order.index(selected_key))
    pct = 1.0 - (float(rank) / float(len(order) - 1))
    return float(np.clip(pct, 1e-6, 1.0 - 1e-6))


def evaluate_cpcv_strategies(
    returns_df: pd.DataFrame,
    splits: list[dict],
    annualization_factor: float = 252.0,
    metric: str = "sharpe",
) -> dict:
    if returns_df is None or returns_df.empty:
        return {"skipped": True, "reason": "empty_returns_df"}
    if not isinstance(returns_df.index, pd.Index):
        return {"skipped": True, "reason": "invalid_index"}

    clean = returns_df.copy()
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(how="all")
    if clean.empty or len(clean.columns) < 2:
        return {"skipped": True, "reason": "need_at_least_two_strategies"}

    split_results: list[dict] = []
    rank_logits: list[float] = []
    selected_oos_rank_pcts: list[float] = []

    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        if np.max(train_idx) >= len(clean.index) or np.max(test_idx) >= len(clean.index):
            continue

        train_slice = clean.iloc[train_idx]
        test_slice = clean.iloc[test_idx]
        is_metrics: dict[str, float] = {}
        oos_metrics: dict[str, float] = {}
        for col in clean.columns:
            is_metrics[col] = _metric_from_returns(
                train_slice[col].to_numpy(dtype=float),
                metric=metric,
                annualization_factor=annualization_factor,
            )
            oos_metrics[col] = _metric_from_returns(
                test_slice[col].to_numpy(dtype=float),
                metric=metric,
                annualization_factor=annualization_factor,
            )
        finite_is = {k: v for k, v in is_metrics.items() if np.isfinite(v)}
        if not finite_is:
            continue
        selected = max(finite_is.items(), key=lambda item: item[1])[0]
        rank_pct = _rank_percentile(oos_metrics, selected)
        logit = float(np.log(rank_pct / (1.0 - rank_pct)))
        selected_oos_rank_pcts.append(rank_pct)
        rank_logits.append(logit)
        split_results.append(
            {
                "split_id": int(split.get("split_id", len(split_results))),
                "selected_strategy": str(selected),
                "selected_oos_rank_pct": rank_pct,
                "selected_oos_rank_logit": logit,
                "is_metrics": {k: float(v) for k, v in is_metrics.items()},
                "oos_metrics": {k: float(v) for k, v in oos_metrics.items()},
                "test_groups": list(split.get("test_groups", [])),
            }
        )

    pbo = pbo_from_rank_logits(np.asarray(rank_logits, dtype=float))
    return {
        "skipped": False,
        "metric": str(metric),
        "n_rows": int(len(clean)),
        "n_strategies": int(len(clean.columns)),
        "n_splits": int(len(split_results)),
        "strategies": [str(c) for c in clean.columns],
        "split_results": split_results,
        "rank_logits": [float(x) for x in rank_logits],
        "pbo": pbo,
        "summary": {
            "median_selected_oos_rank_pct": (
                float(np.median(np.asarray(selected_oos_rank_pcts, dtype=float)))
                if selected_oos_rank_pcts
                else None
            ),
        },
    }

