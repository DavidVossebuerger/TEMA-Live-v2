from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tema.config import BacktestConfig


def _make_features(baseline: pd.Series, ml: pd.Series, *, lags: int, roll: int) -> pd.DataFrame:
    df = pd.DataFrame({"baseline": baseline, "ml": ml}).copy()
    df_shift = df.shift(1)

    feat = pd.DataFrame(index=df.index)
    for lag in range(1, int(lags) + 1):
        feat[f"ml_lag_{lag}"] = df["ml"].shift(lag)
        feat[f"base_lag_{lag}"] = df["baseline"].shift(lag)

    feat["ml_roll_mean"] = df_shift["ml"].rolling(int(roll), min_periods=1).mean()
    feat["ml_roll_vol"] = df_shift["ml"].rolling(int(roll), min_periods=1).std().fillna(0.0)
    feat["base_roll_mean"] = df_shift["baseline"].rolling(int(roll), min_periods=1).mean()
    feat["base_roll_vol"] = df_shift["baseline"].rolling(int(roll), min_periods=1).std().fillna(0.0)

    return feat.fillna(0.0)


def _fit_linear_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    xb = np.hstack([X, ones])
    return np.linalg.pinv(xb) @ y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    xb = np.hstack([X, ones])
    return xb @ w


def _exposure_from_signed_proba(proba: np.ndarray, k: float, floor: float) -> np.ndarray:
    signed = 2.0 * (proba - 0.5)
    raw = np.clip(k * signed, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def _exposure_from_pred(pred: np.ndarray, k: float, floor: float) -> np.ndarray:
    raw = np.clip(k * pred, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def _compute_metrics_arithmetic(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = returns.fillna(0.0)
    cumulative = (1.0 + r).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0) if len(cumulative) else 0.0
    mean_ret = float(r.mean()) if len(r) else 0.0
    vol = float(r.std()) if len(r) else 0.0
    ann_ret = mean_ret * float(freq)
    ann_vol = vol * math.sqrt(float(freq))
    sharpe = (mean_ret / vol) * math.sqrt(float(freq)) if vol > 0 else 0.0
    running_max = cumulative.cummax() if len(cumulative) else cumulative
    drawdown = (cumulative - running_max) / running_max if len(cumulative) else cumulative
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def _align_features_targets(feat: pd.DataFrame, ml: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    df = feat.copy()
    df["ml"] = ml
    df["y"] = (df["ml"] > 0.0).astype(float)
    df = df.dropna(subset=["ml"])
    y = df["y"].values
    X = df.drop(columns=["ml", "y"]).values
    return X, y, df.index


def compute_ml_meta_overlay_series(
    *,
    baseline_train: pd.Series,
    baseline_test: pd.Series,
    ml_train: pd.Series,
    ml_test: pd.Series,
    cfg: BacktestConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    """Reproduce Template/phase1_meta_overlay.py.

    Learns an exposure time-series on train, then applies to test:
      ml_meta = exposure(t) * ml(t)

    Returns:
      (expo_train, expo_test, ml_meta_train, ml_meta_test, diagnostics)
    """

    lags = int(getattr(cfg, "ml_meta_lags", 5))
    roll = int(getattr(cfg, "ml_meta_roll", 5))

    feat_train = _make_features(baseline_train, ml_train, lags=lags, roll=roll)
    feat_test = _make_features(baseline_test, ml_test, lags=lags, roll=roll)

    X_train, y_train, idx_train = _align_features_targets(feat_train, ml_train)
    X_test, _y_dummy, idx_test = _align_features_targets(feat_test, ml_test)

    if X_train.size == 0 or X_test.size == 0:
        zero_train = pd.Series(0.0, index=idx_train, name="exposure")
        zero_test = pd.Series(0.0, index=idx_test, name="exposure")
        train_meta = pd.Series(0.0, index=idx_train, name="portfolio_return_ml_meta")
        test_meta = pd.Series(0.0, index=idx_test, name="portfolio_return_ml_meta")
        return zero_train, zero_test, train_meta, test_meta, {"enabled": False, "reason": "empty_features"}

    w_prob = _fit_linear_pinv(X_train, y_train)
    proba_train = _sigmoid(_predict(X_train, w_prob))
    proba_test = _sigmoid(_predict(X_test, w_prob))

    y_cont = ml_train.loc[idx_train].values
    w_ret = _fit_linear_pinv(X_train, y_cont)
    pred_train = _predict(X_train, w_ret)
    pred_test = _predict(X_test, w_ret)

    ks = np.linspace(
        float(getattr(cfg, "ml_meta_k_min", 0.5)),
        float(getattr(cfg, "ml_meta_k_max", 8.0)),
        int(getattr(cfg, "ml_meta_k_steps", 32)),
    )
    floors = list(getattr(cfg, "ml_meta_floors", (0.2, 0.4, 0.6, 0.8, 0.9)))

    min_vol_ratio = float(getattr(cfg, "ml_meta_min_vol_ratio", 0.5))
    min_mean_abs_exposure = float(getattr(cfg, "ml_meta_min_mean_abs_exposure", 0.7))
    min_turnover_per_year = float(getattr(cfg, "ml_meta_min_turnover_per_year", 5.0))
    target_mean_abs_exposure = float(getattr(cfg, "ml_meta_target_mean_abs_exposure", 0.9))
    target_turnover_per_year = float(getattr(cfg, "ml_meta_target_turnover_per_year", 20.0))

    train_ml_metrics = _compute_metrics_arithmetic(ml_train.loc[idx_train])
    train_ml_ann_vol = float(train_ml_metrics["annualized_vol"])

    def evaluate_candidate(expo: np.ndarray, series_vals: np.ndarray) -> Dict[str, float]:
        r = pd.Series(series_vals * expo)
        n = len(r)
        mean = float(r.mean()) if n else 0.0
        std = float(r.std()) if n else 0.0
        ann_vol = std * math.sqrt(252.0) if std > 0 else 0.0
        ann_ret = mean * 252.0
        sharpe = (mean / std) * math.sqrt(252.0) if std > 0 else 0.0
        turnover_per_year = float(np.mean(np.abs(np.diff(expo)))) * 252.0 if n > 1 else 0.0
        mean_abs_exposure = float(np.mean(np.abs(expo))) if n else 0.0
        total_return = float((1.0 + r).cumprod().iloc[-1] - 1.0) if n else 0.0
        return {
            "mean": float(mean),
            "std": float(std),
            "annualized_vol": float(ann_vol),
            "annualized_return": float(ann_ret),
            "sharpe": float(sharpe),
            "turnover_per_year": float(turnover_per_year),
            "mean_abs_exposure": float(mean_abs_exposure),
            "total_return": float(total_return),
        }

    best: Dict[str, float | str] = {"utility": -1e9}

    series_vals = ml_train.loc[idx_train].values
    for k in ks:
        for f in floors:
            expo = _exposure_from_signed_proba(proba_train, float(k), float(f))
            cand = evaluate_candidate(expo, series_vals)
            if cand["annualized_vol"] < min_vol_ratio * train_ml_ann_vol:
                continue
            if cand["total_return"] <= 0.0:
                continue
            if cand["mean_abs_exposure"] < min_mean_abs_exposure:
                continue
            if cand["turnover_per_year"] < min_turnover_per_year:
                continue

            exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
            turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
            util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
            if util > float(best["utility"]):
                best = {"utility": float(util), "method": "signed_proba", "k": float(k), "f": float(f), **cand}

    for k in ks:
        for f in floors:
            expo = _exposure_from_pred(pred_train, float(k), float(f))
            cand = evaluate_candidate(expo, series_vals)
            if cand["annualized_vol"] < min_vol_ratio * train_ml_ann_vol:
                continue
            if cand["total_return"] <= 0.0:
                continue
            if cand["mean_abs_exposure"] < min_mean_abs_exposure:
                continue
            if cand["turnover_per_year"] < min_turnover_per_year:
                continue

            exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
            turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
            util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
            if util > float(best["utility"]):
                best = {"utility": float(util), "method": "pred", "k": float(k), "f": float(f), **cand}

    if float(best["utility"]) < 0:
        candidate_best_util = -1e9
        candidate_best = None
        preferred_floors = {0.6, 0.8, 0.9}
        for k in ks:
            for f in floors:
                for method in ("pred", "signed_proba"):
                    expo = _exposure_from_pred(pred_train, float(k), float(f)) if method == "pred" else _exposure_from_signed_proba(proba_train, float(k), float(f))
                    cand = evaluate_candidate(expo, series_vals)
                    exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
                    turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
                    util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
                    score = float(util) + (1e-3 if float(f) in preferred_floors else 0.0)
                    if score > candidate_best_util:
                        candidate_best_util = score
                        candidate_best = {"utility": float(util), "method": method, "k": float(k), "f": float(f), **cand}
        if candidate_best is not None:
            best = candidate_best

    method = str(best.get("method", "pred"))
    k_best = float(best.get("k", 1.0))
    f_best = float(best.get("f", 0.0))

    if method == "signed_proba":
        expo_train = _exposure_from_signed_proba(proba_train, k_best, f_best)
        expo_test = _exposure_from_signed_proba(proba_test, k_best, f_best)
    else:
        expo_train = _exposure_from_pred(pred_train, k_best, f_best)
        expo_test = _exposure_from_pred(pred_test, k_best, f_best)

    expo_train_s = pd.Series(expo_train, index=idx_train, name="exposure")
    expo_test_s = pd.Series(expo_test, index=idx_test, name="exposure")

    ml_meta_train = pd.Series(expo_train * ml_train.loc[idx_train].values, index=idx_train, name="portfolio_return_ml_meta")
    ml_meta_test = pd.Series(expo_test * ml_test.loc[idx_test].values, index=idx_test, name="portfolio_return_ml_meta")

    diag = {
        "enabled": True,
        "chosen_method": method,
        "chosen_k": float(k_best),
        "chosen_floor": float(f_best),
        "train_ml_ann_vol": float(train_ml_ann_vol),
        "selection": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best.items()},
    }

    return expo_train_s, expo_test_s, ml_meta_train, ml_meta_test, diag
