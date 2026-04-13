import ctypes
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    data_dir: str = "../merged_d1"
    train_ratio: float = 0.60
    init_cash: float = 100_000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    freq: str = "D"
    batch_size: int = 1000
    max_workers: Optional[int] = None
    min_trades_per_year: float = 2.0
    min_history_rows: int = 400
    max_assets: Optional[int] = None
    verify_cpp_parity: bool = False
    parity_sample_size: int = 100

    # Black-Litterman parameters
    bl_tau: float = 0.05
    bl_delta: float = 2.5
    bl_omega_scale: float = 0.25
    bl_max_weight: float = 0.05

    # HMM regime filter (C++)
    hmm_enabled: bool = True
    hmm_n_states: int = 3
    hmm_n_iter: int = 30
    hmm_var_floor: float = 1e-8
    hmm_trans_sticky: float = 0.92
    hmm_sweep_enabled: bool = True
    hmm_sweep_min_states: int = 2
    hmm_sweep_max_states: int = 6
    hmm_validation_ratio: float = 0.25
    hmm_cv_folds: int = 3
    hmm_min_risk_on_share: float = 0.30
    hmm_max_risk_on_share: float = 0.80
    hmm_min_risk_on_states: int = 2


def resolve_max_workers(cfg: BacktestConfig) -> int:
    cpu_total = os.cpu_count() or 4
    if cfg.max_workers is not None and cfg.max_workers > 0:
        return int(cfg.max_workers)
    # Standard: fast volle CPU, aber einen Kern für OS/UI frei lassen.
    return max(1, cpu_total - 1)


@dataclass
class GridConfig:
    ema1_periods: Sequence[int]
    ema2_periods: Sequence[int]
    ema3_periods: Sequence[int]


class CppSignalEngine:
    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = self.lib.build_signals_batch
        self.fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.fn.restype = None

    def build_signals_batch(
        self,
        ema_values: np.ndarray,
        idx1: np.ndarray,
        idx2: np.ndarray,
        idx3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_periods, n_rows = ema_values.shape
        n_combos = idx1.shape[0]

        entries = np.zeros((n_rows, n_combos), dtype=np.uint8)
        exits = np.zeros((n_rows, n_combos), dtype=np.uint8)

        self.fn(
            ema_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_periods,
            n_rows,
            idx1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            idx2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            idx3.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_combos,
            entries.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            exits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return entries.astype(bool), exits.astype(bool)


class CppHmmEngine:
    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = self.lib.fit_predict_hmm_1d
        self.fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.fn.restype = ctypes.c_int

    def fit_predict(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        n_states: int,
        n_iter: int,
        var_floor: float,
        trans_sticky: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_arr = np.ascontiguousarray(train_returns, dtype=np.float64)
        test_arr = np.ascontiguousarray(test_returns, dtype=np.float64)

        train_states = np.zeros(train_arr.shape[0], dtype=np.int32)
        test_states = np.zeros(test_arr.shape[0], dtype=np.int32)
        means = np.zeros(n_states, dtype=np.float64)
        variances = np.zeros(n_states, dtype=np.float64)

        rc = self.fn(
            train_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(train_arr.shape[0]),
            test_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(test_arr.shape[0]),
            int(n_states),
            int(n_iter),
            float(var_floor),
            float(trans_sticky),
            train_states.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            test_states.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            variances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if rc != 0:
            raise RuntimeError(f"C++ HMM fit failed with code {rc}")

        return train_states, test_states, means, variances


def compile_cpp_signal_library(cpp_path: Path) -> Path:
    so_path = cpp_path.with_suffix(".so")
    needs_build = (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime)
    if not needs_build:
        return so_path

    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-fopenmp",
        str(cpp_path),
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return so_path


def compile_cpp_hmm_library(cpp_path: Path) -> Path:
    so_path = cpp_path.with_suffix(".so")
    needs_build = (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime)
    if not needs_build:
        return so_path

    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(cpp_path),
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return so_path


def generate_ema_combinations(grid_cfg: GridConfig) -> List[Tuple[int, int, int]]:
    combos: List[Tuple[int, int, int]] = []
    for e1 in grid_cfg.ema1_periods:
        for e2 in grid_cfg.ema2_periods:
            for e3 in grid_cfg.ema3_periods:
                if e1 < e2 and e1 < e3:
                    combos.append((e1, e2, e3))
    return combos


def ema_series(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def periods_per_year(freq: str) -> float:
    f = (freq or "D").upper()
    if f.startswith("D"):
        return 252.0
    if f.startswith("H"):
        return 252.0 * 24.0
    if f.startswith("W"):
        return 52.0
    if f.startswith("M"):
        return 12.0
    return 252.0


def simulate_batch_returns_and_stats(
    close: pd.Series,
    entries_np: np.ndarray,
    exits_np: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
    freq: str,
) -> Dict[str, np.ndarray]:
    close_values = close.to_numpy(dtype=np.float64)
    n_rows, n_combos = entries_np.shape
    if n_rows != close_values.shape[0]:
        raise ValueError("Signal rows do not match close length")

    ret_px = np.zeros(n_rows, dtype=np.float64)
    if n_rows > 1:
        ret_px[1:] = close_values[1:] / close_values[:-1] - 1.0

    cost_rate = float(fee_rate + slippage_rate)

    pos = np.zeros(n_combos, dtype=np.int8)
    strat_rets = np.zeros((n_rows, n_combos), dtype=np.float64)
    entry_count = np.zeros(n_combos, dtype=np.int32)
    exit_count = np.zeros(n_combos, dtype=np.int32)

    for t in range(1, n_rows):
        strat_rets[t, :] = pos * ret_px[t]

        new_pos = pos.copy()
        ex = exits_np[t, :]
        en = entries_np[t, :]
        new_pos[ex] = 0
        new_pos[en] = 1

        turnover = np.abs(new_pos - pos)
        strat_rets[t, :] -= turnover * cost_rate

        entry_count += (new_pos > pos).astype(np.int32)
        exit_count += (new_pos < pos).astype(np.int32)
        pos = new_pos

    periods = periods_per_year(freq)
    years = max(n_rows / periods, 1e-9)

    total_return = np.prod(1.0 + strat_rets, axis=0) - 1.0
    with np.errstate(invalid="ignore"):
        annualized_return = np.where(
            (1.0 + total_return) > 0,
            (1.0 + total_return) ** (1.0 / years) - 1.0,
            np.nan,
        )

    volatility = np.std(strat_rets, axis=0) * np.sqrt(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_ratio = np.where(volatility > 0, annualized_return / volatility, np.nan)

    downside = np.where(strat_rets < 0.0, strat_rets, 0.0)
    downside_vol = np.std(downside, axis=0) * np.sqrt(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino_ratio = np.where(downside_vol > 0, annualized_return / downside_vol, np.nan)

    cum = np.cumprod(1.0 + strat_rets, axis=0)
    peak = np.maximum.accumulate(cum, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (cum - peak) / peak
    max_drawdown = np.nanmin(dd, axis=0)
    ulcer_index = np.sqrt(np.nanmean(np.square(dd), axis=0))

    pos_daily = strat_rets[strat_rets > 0.0]
    neg_daily = strat_rets[strat_rets < 0.0]
    gains = np.sum(np.where(strat_rets > 0.0, strat_rets, 0.0), axis=0)
    losses = np.abs(np.sum(np.where(strat_rets < 0.0, strat_rets, 0.0), axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        profit_factor = np.where(losses > 0.0, gains / losses, np.inf)

    expectancy = np.mean(strat_rets, axis=0)
    win_count = np.sum(strat_rets > 0.0, axis=0)
    nonzero_count = np.sum(strat_rets != 0.0, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = np.where(nonzero_count > 0, (win_count / nonzero_count) * 100.0, np.nan)

    avg_win_amount = np.zeros(n_combos, dtype=np.float64)
    avg_loss_amount = np.zeros(n_combos, dtype=np.float64)
    for j in range(n_combos):
        pj = strat_rets[:, j]
        pj_pos = pj[pj > 0.0]
        pj_neg = pj[pj < 0.0]
        avg_win_amount[j] = float(pj_pos.mean()) if pj_pos.size else 0.0
        avg_loss_amount[j] = float(abs(pj_neg.mean())) if pj_neg.size else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        payoff_ratio = np.where(avg_loss_amount > 0.0, avg_win_amount / avg_loss_amount, np.inf)

    total_trades = np.minimum(entry_count, exit_count)
    trades_per_year = total_trades / years

    return {
        "strat_rets": strat_rets,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "ulcer_index": ulcer_index,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win_amount": avg_win_amount,
        "avg_loss_amount": avg_loss_amount,
        "payoff_ratio": payoff_ratio,
        "trades_per_year": trades_per_year,
    }


def precompute_ema_matrix(train_close: pd.Series, combos: Sequence[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    all_periods = np.array(sorted({p for combo in combos for p in combo}), dtype=np.int32)
    ema_matrix = np.column_stack([ema_series(train_close, int(p)).to_numpy(dtype=np.float64) for p in all_periods])
    ema_values = np.ascontiguousarray(ema_matrix.T, dtype=np.float64)
    return all_periods, ema_values


def verify_cpp_matches_python_signals(
    train_close: pd.Series,
    all_periods: np.ndarray,
    ema_values: np.ndarray,
    combos: Sequence[Tuple[int, int, int]],
    engine: CppSignalEngine,
    sample_size: int,
) -> None:
    sample = np.array(list(combos[:sample_size]), dtype=np.int32)
    if sample.size == 0:
        return

    idx1 = np.searchsorted(all_periods, sample[:, 0]).astype(np.int32)
    idx2 = np.searchsorted(all_periods, sample[:, 1]).astype(np.int32)
    idx3 = np.searchsorted(all_periods, sample[:, 2]).astype(np.int32)
    entries_cpp, exits_cpp = engine.build_signals_batch(ema_values, idx1, idx2, idx3)

    entries_py = np.zeros_like(entries_cpp, dtype=bool)
    exits_py = np.zeros_like(exits_cpp, dtype=bool)

    for j, (e1, e2, e3) in enumerate(sample.tolist()):
        s1 = pd.Series(ema_values[np.searchsorted(all_periods, e1)], index=train_close.index)
        s2 = pd.Series(ema_values[np.searchsorted(all_periods, e2)], index=train_close.index)
        s3 = pd.Series(ema_values[np.searchsorted(all_periods, e3)], index=train_close.index)

        entries_raw = crossed_above(s1, s2) | crossed_above(s1, s3) | crossed_above(s2, s3)
        exits_raw = crossed_below(s1, s2) | crossed_below(s1, s3) | crossed_below(s2, s3)

        entries_py[:, j] = entries_raw.shift(1, fill_value=False).to_numpy(dtype=bool)
        exits_py[:, j] = exits_raw.shift(1, fill_value=False).to_numpy(dtype=bool)

    if not np.array_equal(entries_cpp, entries_py) or not np.array_equal(exits_cpp, exits_py):
        entry_diff = int(np.count_nonzero(entries_cpp != entries_py))
        exit_diff = int(np.count_nonzero(exits_cpp != exits_py))
        raise ValueError(
            f"C++ signal parity check failed (entry_diff={entry_diff}, exit_diff={exit_diff})."
        )

    print(f"C++ parity check passed for {len(sample)} combinations.")


G_CLOSE = None
G_ALL_PERIODS = None
G_EMA_VALUES = None
G_ENGINE = None
G_CFG = None


def _init_worker(
    close_values: np.ndarray,
    close_index: np.ndarray,
    all_periods: np.ndarray,
    ema_values: np.ndarray,
    so_path: str,
    cfg_dict: Dict,
) -> None:
    global G_CLOSE, G_ALL_PERIODS, G_EMA_VALUES, G_ENGINE, G_CFG
    G_CLOSE = pd.Series(close_values, index=pd.to_datetime(close_index))
    G_ALL_PERIODS = all_periods
    G_EMA_VALUES = ema_values
    G_ENGINE = CppSignalEngine(Path(so_path))
    G_CFG = BacktestConfig(**cfg_dict)


def _extract_rows_from_stats(
    stats: Dict[str, np.ndarray],
    batch_combos: np.ndarray,
    min_trades_per_year: float,
) -> List[Dict]:
    rows: List[Dict] = []

    for idx, (e1, e2, e3) in enumerate(batch_combos.tolist()):
        total_return = float(stats["total_return"][idx])
        annualized_return = float(stats["annualized_return"][idx])
        max_drawdown = float(stats["max_drawdown"][idx])
        volatility = float(stats["volatility"][idx])
        sharpe_ratio = float(stats["sharpe_ratio"][idx])
        sortino_ratio = float(stats["sortino_ratio"][idx])
        total_trades = int(stats["total_trades"][idx])
        trades_per_year = float(stats["trades_per_year"][idx])
        if trades_per_year < min_trades_per_year:
            continue

        rows.append(
            {
                "ema1_period": int(e1),
                "ema2_period": int(e2),
                "ema3_period": int(e3),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "information_ratio": np.nan,
                "tail_ratio": np.nan,
                "deflated_sharpe_ratio": np.nan,
                "ulcer_index": float(stats["ulcer_index"][idx]),
                "total_trades": total_trades,
                "win_rate": float(stats["win_rate"][idx]),
                "profit_factor": float(stats["profit_factor"][idx]),
                "expectancy": float(stats["expectancy"][idx]),
                "avg_win_amount": float(stats["avg_win_amount"][idx]),
                "avg_loss_amount": float(stats["avg_loss_amount"][idx]),
                "payoff_ratio": float(stats["payoff_ratio"][idx]),
                "trades_per_year": trades_per_year,
            }
        )
    return rows


def _process_batch(batch_combos: np.ndarray) -> List[Dict]:
    idx1 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 0]).astype(np.int32)
    idx2 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 1]).astype(np.int32)
    idx3 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 2]).astype(np.int32)

    entries_np, exits_np = G_ENGINE.build_signals_batch(G_EMA_VALUES, idx1, idx2, idx3)
    stats = simulate_batch_returns_and_stats(
        close=G_CLOSE,
        entries_np=entries_np,
        exits_np=exits_np,
        fee_rate=G_CFG.fee_rate,
        slippage_rate=G_CFG.slippage_rate,
        freq=G_CFG.freq,
    )

    return _extract_rows_from_stats(
        stats=stats,
        batch_combos=batch_combos,
        min_trades_per_year=G_CFG.min_trades_per_year,
    )


def run_parallel_grid_search_cpp(
    train_close: pd.Series,
    combos: Sequence[Tuple[int, int, int]],
    cfg: BacktestConfig,
    so_path: Path,
) -> pd.DataFrame:
    all_periods, ema_values = precompute_ema_matrix(train_close, combos)
    max_workers = resolve_max_workers(cfg)

    combo_array = np.array(combos, dtype=np.int32)
    batches = [combo_array[i : i + cfg.batch_size] for i in range(0, len(combo_array), cfg.batch_size)]

    cfg_dict = {
        "data_dir": cfg.data_dir,
        "train_ratio": cfg.train_ratio,
        "init_cash": cfg.init_cash,
        "fee_rate": cfg.fee_rate,
        "slippage_rate": cfg.slippage_rate,
        "freq": cfg.freq,
        "batch_size": cfg.batch_size,
        "max_workers": max_workers,
        "min_trades_per_year": cfg.min_trades_per_year,
        "min_history_rows": cfg.min_history_rows,
        "max_assets": cfg.max_assets,
        "verify_cpp_parity": cfg.verify_cpp_parity,
        "parity_sample_size": cfg.parity_sample_size,
        "bl_tau": cfg.bl_tau,
        "bl_delta": cfg.bl_delta,
        "bl_omega_scale": cfg.bl_omega_scale,
    }

    all_rows: List[Dict] = []
    start_ts = time.perf_counter()

    def _fmt_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(
            train_close.values.astype(np.float64),
            train_close.index.values,
            all_periods,
            ema_values,
            str(so_path),
            cfg_dict,
        ),
    ) as ex:
        futures = {ex.submit(_process_batch, batch): i + 1 for i, batch in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += 1
            elapsed = time.perf_counter() - start_ts
            speed = done / max(elapsed, 1e-9)
            remaining = len(batches) - done
            eta = remaining / max(speed, 1e-9)
            progress = (done / len(batches)) * 100.0
            if done % 10 == 0 or done == len(batches):
                print(
                    f"Grid progress: {done}/{len(batches)} ({progress:.1f}%) | "
                    f"elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
                )

    if not all_rows:
        raise ValueError("No valid grid search results produced.")

    return pd.DataFrame(all_rows)


def list_asset_files(data_dir: Path, max_assets: Optional[int]) -> List[Path]:
    files = sorted(data_dir.glob("*_d1_merged.csv"))
    if max_assets is not None:
        files = files[:max_assets]
    return files


def count_csv_rows_fast(file_path: Path) -> int:
    # Schneller Vorab-Check: zählt Zeilen ohne vollständiges CSV-Parsing.
    with file_path.open("rb") as fh:
        line_count = sum(1 for _ in fh)
    return max(0, line_count - 1)  # Header abziehen


def prefilter_assets_by_min_rows(asset_files: Sequence[Path], min_rows: int) -> Tuple[List[Path], List[Tuple[str, int]]]:
    keep: List[Path] = []
    dropped: List[Tuple[str, int]] = []

    for fp in asset_files:
        try:
            rows = count_csv_rows_fast(fp)
        except Exception:
            rows = 0

        if rows >= min_rows:
            keep.append(fp)
        else:
            dropped.append((parse_asset_name(fp), rows))

    return keep, dropped


def parse_asset_name(file_path: Path) -> str:
    return file_path.name.replace("_d1_merged.csv", "")


def load_close_series_from_csv(file_path: Path) -> pd.Series:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("CSV is empty")

    close_candidates = ["close_mid", "close", "Close", "close_bid", "close_ask"]
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if close_col is None:
        raise KeyError("No close-like column found")

    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    else:
        raise KeyError("Neither datetime nor timestamp column found")

    s = pd.Series(df[close_col].astype(float).values, index=idx, name=parse_asset_name(file_path))
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    if s.empty:
        raise ValueError("No valid close values after cleaning")
    return s


def split_train_test(close: pd.Series, train_ratio: float) -> Tuple[pd.Series, pd.Series]:
    split_idx = int(len(close) * train_ratio)
    split_idx = max(2, min(split_idx, len(close) - 2))
    train_close = close.iloc[:split_idx].copy()
    test_close = close.iloc[split_idx:].copy()
    return train_close, test_close


def build_strategy_returns_for_combo(
    close: pd.Series,
    combo: Tuple[int, int, int],
    cfg: BacktestConfig,
) -> pd.Series:
    e1, e2, e3 = combo
    s1 = ema_series(close, e1)
    s2 = ema_series(close, e2)
    s3 = ema_series(close, e3)

    entries = (crossed_above(s1, s2) | crossed_above(s1, s3) | crossed_above(s2, s3)).shift(
        1,
        fill_value=False,
    )
    exits = (crossed_below(s1, s2) | crossed_below(s1, s3) | crossed_below(s2, s3)).shift(
        1,
        fill_value=False,
    )

    stats = simulate_batch_returns_and_stats(
        close=close,
        entries_np=entries.to_numpy(dtype=bool).reshape(-1, 1),
        exits_np=exits.to_numpy(dtype=bool).reshape(-1, 1),
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        freq=cfg.freq,
    )
    return pd.Series(stats["strat_rets"][:, 0], index=close.index, dtype=float).fillna(0.0)


def annualize_return_from_series(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    cum = float((1.0 + returns).prod())
    n = len(returns)
    if n <= 1 or cum <= 0:
        return np.nan
    return float(cum ** (252.0 / n) - 1.0)


def build_black_litterman_weights(
    train_returns_df: pd.DataFrame,
    view_returns: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    assets = [a for a in train_returns_df.columns if a in view_returns.index]
    if not assets:
        raise ValueError("No overlapping assets between train return matrix and views")

    rets = train_returns_df[assets].fillna(0.0)
    q = view_returns[assets].astype(float).to_numpy()

    sigma = rets.cov().to_numpy() * 252.0
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    jitter = 1e-8 * np.eye(len(assets))
    sigma = sigma + jitter

    w_mkt = np.repeat(1.0 / len(assets), len(assets))
    pi = cfg.bl_delta * (sigma @ w_mkt)

    p = np.eye(len(assets))
    tau_sigma = cfg.bl_tau * sigma

    omega_diag = np.diag(p @ tau_sigma @ p.T).copy()
    omega_diag = np.where(omega_diag <= 1e-12, 1e-12, omega_diag)
    omega = np.diag(omega_diag * cfg.bl_omega_scale)

    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)

    middle = np.linalg.pinv(inv_tau_sigma + p.T @ inv_omega @ p)
    mu_bl = middle @ (inv_tau_sigma @ pi + p.T @ inv_omega @ q)

    raw_w = np.linalg.pinv(cfg.bl_delta * sigma) @ mu_bl

    # Long-only + Weight-Cap Projektion
    raw_w = np.maximum(raw_w, 0.0)

    cap = float(cfg.bl_max_weight)
    cap = min(max(cap, 1e-6), 1.0)
    n = len(assets)
    if cap * n < 1.0:
        cap = 1.0 / n

    w_sum = float(raw_w.sum())
    if w_sum <= 0:
        raw_w = np.repeat(1.0 / n, n)
    else:
        raw_w = raw_w / w_sum

    for _ in range(20):
        clipped = np.minimum(raw_w, cap)
        deficit = 1.0 - float(clipped.sum())
        if deficit <= 1e-12:
            raw_w = clipped
            break
        free = clipped < (cap - 1e-12)
        free_sum = float(clipped[free].sum())
        if not np.any(free):
            raw_w = clipped
            break
        if free_sum <= 1e-12:
            clipped[free] += deficit / float(np.count_nonzero(free))
            raw_w = np.minimum(clipped, cap)
            continue
        clipped[free] += deficit * (clipped[free] / free_sum)
        raw_w = clipped

    raw_w = np.maximum(raw_w, 0.0)
    raw_w = raw_w / max(float(raw_w.sum()), 1e-12)

    return pd.Series(raw_w, index=assets, name="bl_weight")


def evaluate_weighted_portfolio(
    returns_df: pd.DataFrame,
    weights: pd.Series,
) -> Tuple[pd.Series, Dict[str, float]]:
    aligned = returns_df.reindex(columns=weights.index).fillna(0.0)
    portfolio_returns = aligned.mul(weights, axis=1).sum(axis=1)

    if portfolio_returns.empty:
        metrics = {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }
        return portfolio_returns, metrics

    cum = (1.0 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    total_return = float(cum.iloc[-1] - 1.0)
    ann_return = annualize_return_from_series(portfolio_returns)
    ann_vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 and np.isfinite(ann_return) else np.nan
    max_dd = float(dd.min()) if not dd.empty else np.nan

    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
    return portfolio_returns, metrics


def apply_hmm_regime_filter_oos(
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    cfg: BacktestConfig,
    hmm_engine: CppHmmEngine,
    n_states: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Dict[str, float]]:
    tr = train_port_rets.fillna(0.0).to_numpy(dtype=np.float64)
    te = test_port_rets.fillna(0.0).to_numpy(dtype=np.float64)

    n_states_eff = int(n_states if n_states is not None else cfg.hmm_n_states)

    train_states, test_states, means, variances = hmm_engine.fit_predict(
        train_returns=tr,
        test_returns=te,
        n_states=n_states_eff,
        n_iter=cfg.hmm_n_iter,
        var_floor=cfg.hmm_var_floor,
        trans_sticky=cfg.hmm_trans_sticky,
    )

    risk_on_states = np.where(means > 0.0)[0]
    if risk_on_states.size == 0:
        risk_on_states = np.array([int(np.argmax(means))], dtype=np.int32)

    train_mask = np.isin(train_states, risk_on_states)
    test_mask = np.isin(test_states, risk_on_states)

    filtered_train = pd.Series(np.where(train_mask, tr, 0.0), index=train_port_rets.index, name="portfolio_return_hmm")
    filtered_test = pd.Series(np.where(test_mask, te, 0.0), index=test_port_rets.index, name="portfolio_return_hmm")

    state_df = pd.DataFrame(
        {
            "state": np.arange(n_states_eff, dtype=int),
            "mean_return": means,
            "variance": variances,
            "is_risk_on": np.isin(np.arange(n_states_eff), risk_on_states),
        }
    )

    diag = {
        "n_states": float(n_states_eff),
        "train_risk_on_share": float(train_mask.mean()) if train_mask.size else np.nan,
        "test_risk_on_share": float(test_mask.mean()) if test_mask.size else np.nan,
        "n_risk_on_states": float(risk_on_states.size),
    }
    return filtered_train, filtered_test, state_df, diag


def run_hmm_regime_sweep(
    train_rets: pd.Series,
    cfg: BacktestConfig,
    hmm_engine: CppHmmEngine,
) -> Tuple[int, pd.DataFrame]:
    rows: List[Dict] = []
    best_n = cfg.hmm_n_states
    best_score = -np.inf
    best_dd = -np.inf

    low = int(min(cfg.hmm_sweep_min_states, cfg.hmm_sweep_max_states))
    high = int(max(cfg.hmm_sweep_min_states, cfg.hmm_sweep_max_states))

    n_total = len(train_rets)
    n_folds = max(1, int(cfg.hmm_cv_folds))
    fold_len = max(50, n_total // (n_folds + 1))

    fold_slices: List[Tuple[slice, slice]] = []
    for k in range(n_folds):
        tr_end = n_total - (n_folds - k) * fold_len
        va_end = tr_end + fold_len
        if tr_end < 80:
            continue
        if va_end > n_total:
            break
        fold_slices.append((slice(0, tr_end), slice(tr_end, va_end)))

    if not fold_slices:
        split_idx = int(n_total * (1.0 - cfg.hmm_validation_ratio))
        split_idx = max(80, min(split_idx, n_total - 50))
        fold_slices = [(slice(0, split_idx), slice(split_idx, n_total))]

    for n_states in range(low, high + 1):
        try:
            fold_rows: List[Dict] = []
            for tr_slice, va_slice in fold_slices:
                subtrain_rets = train_rets.iloc[tr_slice].copy()
                val_rets = train_rets.iloc[va_slice].copy()

                hmm_subtrain, hmm_val, _, diag = apply_hmm_regime_filter_oos(
                    train_port_rets=subtrain_rets,
                    test_port_rets=val_rets,
                    cfg=cfg,
                    hmm_engine=hmm_engine,
                    n_states=n_states,
                )

                _, subtrain_metrics = evaluate_weighted_portfolio(
                    pd.DataFrame({"p": hmm_subtrain}),
                    pd.Series({"p": 1.0}),
                )
                _, val_metrics = evaluate_weighted_portfolio(
                    pd.DataFrame({"p": hmm_val}),
                    pd.Series({"p": 1.0}),
                )

                fold_rows.append(
                    {
                        "subtrain_sharpe": subtrain_metrics["sharpe_ratio"],
                        "val_sharpe": val_metrics["sharpe_ratio"],
                        "subtrain_total_return": subtrain_metrics["total_return"],
                        "val_total_return": val_metrics["total_return"],
                        "val_max_drawdown": val_metrics["max_drawdown"],
                        "subtrain_risk_on_share": diag["train_risk_on_share"],
                        "val_risk_on_share": diag["test_risk_on_share"],
                        "n_risk_on_states": int(diag["n_risk_on_states"]),
                    }
                )

            folds_df = pd.DataFrame(fold_rows)
            agg = folds_df.mean(numeric_only=True).to_dict()

            eligible = (
                np.isfinite(agg.get("val_sharpe", np.nan))
                and (cfg.hmm_min_risk_on_share <= agg.get("val_risk_on_share", np.nan) <= cfg.hmm_max_risk_on_share)
                and (agg.get("n_risk_on_states", 0.0) >= float(cfg.hmm_min_risk_on_states))
            )

            row = {
                "n_states": n_states,
                "cv_folds": len(fold_slices),
                "subtrain_sharpe": agg.get("subtrain_sharpe", np.nan),
                "val_sharpe": agg.get("val_sharpe", np.nan),
                "subtrain_total_return": agg.get("subtrain_total_return", np.nan),
                "val_total_return": agg.get("val_total_return", np.nan),
                "val_max_drawdown": agg.get("val_max_drawdown", np.nan),
                "subtrain_risk_on_share": agg.get("subtrain_risk_on_share", np.nan),
                "val_risk_on_share": agg.get("val_risk_on_share", np.nan),
                "n_risk_on_states": int(round(agg.get("n_risk_on_states", 0.0))),
                "eligible": bool(eligible),
            }
            rows.append(row)

            score = row["val_sharpe"]
            dd = row["val_max_drawdown"]
            if eligible and np.isfinite(score):
                if (score > best_score) or (np.isclose(score, best_score, equal_nan=False) and dd > best_dd):
                    best_score = score
                    best_dd = dd
                    best_n = n_states
        except Exception as exc:
            rows.append({
                "n_states": n_states,
                "error": str(exc),
            })

    sweep_df = pd.DataFrame(rows)
    return best_n, sweep_df


def run_asset_pipeline(
    asset_file: Path,
    combos: Sequence[Tuple[int, int, int]],
    cfg: BacktestConfig,
    so_path: Path,
    parity_checked: bool,
) -> Tuple[Dict, pd.Series, pd.Series, bool]:
    asset = parse_asset_name(asset_file)
    close = load_close_series_from_csv(asset_file)

    if len(close) < cfg.min_history_rows:
        raise ValueError(f"History too short ({len(close)} rows)")

    train_close, test_close = split_train_test(close, cfg.train_ratio)

    if cfg.verify_cpp_parity and not parity_checked:
        all_periods, ema_values = precompute_ema_matrix(train_close, combos)
        parity_engine = CppSignalEngine(so_path)
        verify_cpp_matches_python_signals(
            train_close=train_close,
            all_periods=all_periods,
            ema_values=ema_values,
            combos=combos,
            engine=parity_engine,
            sample_size=cfg.parity_sample_size,
        )
        parity_checked = True

    results_df = run_parallel_grid_search_cpp(train_close, combos, cfg, so_path)
    best = results_df.loc[results_df["sharpe_ratio"].idxmax()]

    best_combo = (int(best["ema1_period"]), int(best["ema2_period"]), int(best["ema3_period"]))

    train_returns = build_strategy_returns_for_combo(train_close, best_combo, cfg)
    test_returns = build_strategy_returns_for_combo(test_close, best_combo, cfg)

    summary = {
        "asset": asset,
        "rows_total": int(len(close)),
        "rows_train": int(len(train_close)),
        "rows_test": int(len(test_close)),
        "ema1_period": best_combo[0],
        "ema2_period": best_combo[1],
        "ema3_period": best_combo[2],
        "train_sharpe": float(best["sharpe_ratio"]),
        "train_total_return": float(best["total_return"]),
        "train_annualized_return": float(best["annualized_return"]),
        "train_max_drawdown": float(best["max_drawdown"]),
        "train_trades": int(best["total_trades"]),
        "view_q": annualize_return_from_series(train_returns),
    }

    train_returns.name = asset
    test_returns.name = asset
    return summary, train_returns, test_returns, parity_checked


def main() -> None:
    cfg = BacktestConfig()
    max_workers = resolve_max_workers(cfg)
    grid_cfg = GridConfig(
        ema1_periods=list(range(4, 40, 3)),
        ema2_periods=list(range(80, 200, 3)),
        ema3_periods=list(range(100, 200, 3)),
    )

    here = Path(__file__).resolve().parent
    data_dir = (here / cfg.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cpp_path = here / "grid_signals.cpp"
    so_path = compile_cpp_signal_library(cpp_path)
    hmm_cpp_path = here / "hmm_regime.cpp"
    hmm_so_path = compile_cpp_hmm_library(hmm_cpp_path)
    hmm_engine = CppHmmEngine(hmm_so_path)

    combos = generate_ema_combinations(grid_cfg)
    print(f"Total EMA combinations: {len(combos)}")
    print(f"Python workers per asset-grid: {max_workers}")

    asset_files = list_asset_files(data_dir, cfg.max_assets)
    if not asset_files:
        raise ValueError(f"No *_d1_merged.csv files found in {data_dir}")

    print(f"Assets selected (raw): {len(asset_files)}")
    asset_files, dropped_assets = prefilter_assets_by_min_rows(asset_files, cfg.min_history_rows)
    print(
        f"Assets after min-row prefilter (>= {cfg.min_history_rows}): {len(asset_files)} | "
        f"dropped: {len(dropped_assets)}"
    )
    if dropped_assets:
        preview = ", ".join(f"{a}({r})" for a, r in dropped_assets[:10])
        print(f"Dropped preview: {preview}")

    if not asset_files:
        raise ValueError("No assets left after min-row prefilter.")

    summaries: List[Dict] = []
    train_ret_list: List[pd.Series] = []
    test_ret_list: List[pd.Series] = []

    parity_checked = False
    assets_start_ts = time.perf_counter()

    def _fmt_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    for i, asset_file in enumerate(asset_files, 1):
        asset = parse_asset_name(asset_file)
        try:
            summary, train_ret, test_ret, parity_checked = run_asset_pipeline(
                asset_file=asset_file,
                combos=combos,
                cfg=cfg,
                so_path=so_path,
                parity_checked=parity_checked,
            )
            summaries.append(summary)
            train_ret_list.append(train_ret)
            test_ret_list.append(test_ret)
            status = "OK"
        except Exception as exc:
            status = f"SKIP ({exc})"

        elapsed = time.perf_counter() - assets_start_ts
        speed = i / max(elapsed, 1e-9)
        remaining = len(asset_files) - i
        eta = remaining / max(speed, 1e-9)
        progress = (i / len(asset_files)) * 100.0

        if status == "OK":
            print(
                f"Asset progress: {i}/{len(asset_files)} ({progress:.1f}%) | "
                f"{asset} {status} | best EMA=({summary['ema1_period']},{summary['ema2_period']},{summary['ema3_period']}) | "
                f"elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
            )
        else:
            print(
                f"Asset progress: {i}/{len(asset_files)} ({progress:.1f}%) | "
                f"{asset} {status} | elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
            )

    if not summaries:
        raise ValueError("No assets produced valid strategy results.")

    summary_df = pd.DataFrame(summaries).sort_values("train_sharpe", ascending=False).reset_index(drop=True)

    train_returns_df = pd.concat(train_ret_list, axis=1).sort_index().fillna(0.0)
    test_returns_df = pd.concat(test_ret_list, axis=1).sort_index().fillna(0.0)

    view_q = summary_df.set_index("asset")["view_q"]
    bl_weights = build_black_litterman_weights(train_returns_df, view_q, cfg).sort_values(ascending=False)

    train_port_rets, train_metrics = evaluate_weighted_portfolio(train_returns_df, bl_weights)
    test_port_rets, test_metrics = evaluate_weighted_portfolio(test_returns_df, bl_weights)

    hmm_train_rets = train_port_rets.copy()
    hmm_test_rets = test_port_rets.copy()
    hmm_state_df = pd.DataFrame()
    hmm_diag: Dict[str, float] = {}
    hmm_sweep_df = pd.DataFrame()
    hmm_train_metrics = train_metrics.copy()
    hmm_test_metrics = test_metrics.copy()
    if cfg.hmm_enabled:
        selected_hmm_states = int(cfg.hmm_n_states)
        if cfg.hmm_sweep_enabled:
            selected_hmm_states, hmm_sweep_df = run_hmm_regime_sweep(
                train_rets=train_port_rets,
                cfg=cfg,
                hmm_engine=hmm_engine,
            )

        hmm_train_rets, hmm_test_rets, hmm_state_df, hmm_diag = apply_hmm_regime_filter_oos(
            train_port_rets=train_port_rets,
            test_port_rets=test_port_rets,
            cfg=cfg,
            hmm_engine=hmm_engine,
            n_states=selected_hmm_states,
        )
        _, hmm_train_metrics = evaluate_weighted_portfolio(
            pd.DataFrame({"p": hmm_train_rets}),
            pd.Series({"p": 1.0}),
        )
        _, hmm_test_metrics = evaluate_weighted_portfolio(
            pd.DataFrame({"p": hmm_test_rets}),
            pd.Series({"p": 1.0}),
        )

    out_summary = here / "asset_strategy_summary.csv"
    out_weights = here / "black_litterman_weights.csv"
    out_train_rets = here / "portfolio_train_returns.csv"
    out_test_rets = here / "portfolio_test_returns.csv"
    out_train_hmm_rets = here / "portfolio_train_returns_hmm.csv"
    out_test_hmm_rets = here / "portfolio_test_returns_hmm.csv"
    out_hmm_states = here / "hmm_state_params.csv"
    out_hmm_diag = here / "hmm_diagnostics.csv"
    out_hmm_sweep = here / "hmm_regime_sweep.csv"
    out_metrics = here / "bl_portfolio_metrics.csv"

    summary_df.to_csv(out_summary, index=False)
    bl_weights.rename("weight").to_csv(out_weights, header=True)
    train_port_rets.rename("portfolio_return").to_csv(out_train_rets, header=True)
    test_port_rets.rename("portfolio_return").to_csv(out_test_rets, header=True)
    hmm_train_rets.rename("portfolio_return_hmm").to_csv(out_train_hmm_rets, header=True)
    hmm_test_rets.rename("portfolio_return_hmm").to_csv(out_test_hmm_rets, header=True)
    if not hmm_state_df.empty:
        hmm_state_df.to_csv(out_hmm_states, index=False)
    if hmm_diag:
        pd.DataFrame([hmm_diag]).to_csv(out_hmm_diag, index=False)
    if not hmm_sweep_df.empty:
        hmm_sweep_df.to_csv(out_hmm_sweep, index=False)

    metrics_df = pd.DataFrame(
        [
            {"dataset": "train", **train_metrics},
            {"dataset": "test", **test_metrics},
            {"dataset": "train_hmm", **hmm_train_metrics},
            {"dataset": "test_hmm", **hmm_test_metrics},
        ]
    )
    metrics_df.to_csv(out_metrics, index=False)

    print("=" * 70)
    print("MULTI-ASSET PIPELINE DONE (NO PLOTS)")
    print("=" * 70)
    print(f"Valid assets: {len(summary_df)}")
    print("Top 10 BL weights:")
    for asset, w in bl_weights.head(10).items():
        print(f"  {asset:<20} {w:>8.2%}")

    print("Train metrics:")
    print(train_metrics)
    print("Test metrics:")
    print(test_metrics)
    if cfg.hmm_enabled:
        print("Train metrics (HMM filtered):")
        print(hmm_train_metrics)
        print("Test metrics (HMM filtered):")
        print(hmm_test_metrics)
        if hmm_diag:
            print("HMM diagnostics:")
            print(hmm_diag)
        if not hmm_sweep_df.empty:
            valid = hmm_sweep_df[hmm_sweep_df.get("eligible", pd.Series(dtype=bool)) == True] if "eligible" in hmm_sweep_df.columns else pd.DataFrame()
            if not valid.empty:
                best_row = valid.sort_values(["val_sharpe", "val_max_drawdown"], ascending=[False, False]).iloc[0]
                print(
                    "HMM sweep best (selected on validation only): "
                    f"n_states={int(best_row['n_states'])} | "
                    f"val_sharpe={best_row['val_sharpe']:.3f} | "
                    f"val_max_drawdown={best_row['val_max_drawdown']:.3%} | "
                    f"val_risk_on_share={best_row['val_risk_on_share']:.1%}"
                )

    print("Saved:")
    print(f"  {out_summary}")
    print(f"  {out_weights}")
    print(f"  {out_train_rets}")
    print(f"  {out_test_rets}")
    print(f"  {out_train_hmm_rets}")
    print(f"  {out_test_hmm_rets}")
    if cfg.hmm_enabled:
        print(f"  {out_hmm_states}")
        print(f"  {out_hmm_diag}")
        if not hmm_sweep_df.empty:
            print(f"  {out_hmm_sweep}")
    print(f"  {out_metrics}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
