#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tema.config import BacktestConfig
from tema.pipeline import run_pipeline
from tema.validation.ablation import collect_run_summary
from tema.validation.oos import validate_oos_gates
from tema.validation.probabilistic import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from tema.validation.sweep import (
    parse_float_list,
    parse_int_list,
    pick_best_config_with_hard_gate,
    summarize_sweep,
)


def _float_token(x: float) -> str:
    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")


def _extract_overlay_metrics(overlay: dict) -> dict:
    def _get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    base_train = _get(overlay, "base_train_metrics", "sharpe")
    base_test = _get(overlay, "base_test_metrics", "sharpe")
    ml_train = _get(overlay, "ml_train_metrics", "sharpe")
    ml_test = _get(overlay, "ml_test_metrics", "sharpe")

    out = {
        "base_train_sharpe": float(base_train) if base_train is not None else None,
        "base_test_sharpe": float(base_test) if base_test is not None else None,
        "ml_train_sharpe": float(ml_train) if ml_train is not None else None,
        "ml_test_sharpe": float(ml_test) if ml_test is not None else None,
    }
    if out["base_train_sharpe"] is not None and out["base_test_sharpe"] is not None:
        out["base_overfit_gap"] = float(out["base_train_sharpe"] - out["base_test_sharpe"])
    if out["ml_train_sharpe"] is not None and out["ml_test_sharpe"] is not None:
        out["ml_overfit_gap"] = float(out["ml_train_sharpe"] - out["ml_test_sharpe"])
    return out


def _load_returns_series(csv_path: Path, preferred_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    col = preferred_col if preferred_col in df.columns else None
    if col is None:
        candidates = [c for c in df.columns if c != "datetime"]
        if not candidates:
            raise ValueError(f"no return column found in {csv_path}")
        col = candidates[0]
    values = pd.to_numeric(df[col], errors="coerce")
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        valid = values.notna() & idx.notna()
        series = pd.Series(values[valid].to_numpy(dtype=float), index=idx[valid], name=csv_path.stem)
    else:
        series = pd.Series(values.dropna().to_numpy(dtype=float), name=csv_path.stem)
    if series.empty:
        raise ValueError(f"no finite returns found in {csv_path}")
    return series.sort_index()


def _gate_value(gates: dict, name: str, field: str):
    payload = gates.get(name, {}) if isinstance(gates, dict) else {}
    if isinstance(payload, dict):
        return payload.get(field)
    return None


def run(
    *,
    run_id: str,
    out_root: str = "outputs",
    seeds: list[int],
    hmm_n_states: list[int],
    hmm_n_iter: list[int],
    hmm_var_floor: list[float],
    hmm_trans_sticky: list[float],
    rf_n_estimators: list[int],
    rf_max_depth: list[int],
    fixtures: bool = True,
    oos_min_sharpe: float | None = 0.5,
    oos_max_drawdown: float | None = 0.25,
    oos_max_turnover: float | None = 5.0,
    oos_min_calmar: float | None = None,
    psr_threshold: float | None = 0.95,
    dsr_threshold: float | None = 0.80,
    require_hard_gates: bool = False,
    min_hard_gate_pass_rate: float = 1.0,
) -> int:
    sweep_dir = Path(out_root) / run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if fixtures:
        os.environ["TEMA_IGNORE_TEMPLATE_DIR"] = "1"

    rows: list[dict] = []
    failures: list[dict] = []
    n_trials = max(
        2,
        int(
            len(seeds)
            * len(hmm_n_states)
            * len(hmm_n_iter)
            * len(hmm_var_floor)
            * len(hmm_trans_sticky)
            * len(rf_n_estimators)
            * len(rf_max_depth)
        ),
    )

    for seed in seeds:
        for n_states in hmm_n_states:
            for n_iter in hmm_n_iter:
                for var_floor in hmm_var_floor:
                    for trans_sticky in hmm_trans_sticky:
                        for n_est in rf_n_estimators:
                            for depth in rf_max_depth:
                                sub_id = (
                                    f"{run_id}__s{seed}__ns{n_states}__ni{n_iter}"
                                    f"__vf{_float_token(var_floor)}__ts{_float_token(trans_sticky)}"
                                    f"__rf{n_est}d{depth}"
                                )
                                t0 = time.perf_counter()
                                try:
                                    cfg = BacktestConfig(
                                        template_default_universe=True,
                                        template_use_precomputed_artifacts=False,
                                        strict_independent_mode=True,
                                        ml_meta_comparator_use_benchmark_csv=False,
                                        ml_template_overlay_enabled=True,
                                        ml_meta_overlay_enabled=False,
                                        rf_random_state=int(seed),
                                        hmm_n_states=int(n_states),
                                        hmm_n_iter=int(n_iter),
                                        hmm_var_floor=float(var_floor),
                                        hmm_trans_sticky=float(trans_sticky),
                                        rf_n_estimators=int(n_est),
                                        rf_max_depth=int(depth),
                                    )
                                    res = run_pipeline(run_id=sub_id, cfg=cfg, out_root=out_root)
                                    out_dir = Path(res["out_dir"])
                                    overlay = json.loads((out_dir / "template_ml_overlay.json").read_text(encoding="utf-8"))
                                    overlay_metrics = _extract_overlay_metrics(overlay)

                                    run_summary = collect_run_summary(
                                        out_dir,
                                        run_id=sub_id,
                                        config_summary={"seed": int(seed)},
                                    )
                                    gates = run_summary.get("validation_hard_gates", {})
                                    benchmark_injection = bool(run_summary.get("benchmark_injection_detected", False))

                                    oos_report = validate_oos_gates(
                                        str(out_dir / "manifest.json"),
                                        min_sharpe=oos_min_sharpe,
                                        max_drawdown=oos_max_drawdown,
                                        max_turnover=oos_max_turnover,
                                        min_calmar=oos_min_calmar,
                                    )
                                    ml_psr = None
                                    ml_dsr = None
                                    ml_returns_path = out_dir / "portfolio_test_returns_ml.csv"
                                    if ml_returns_path.exists():
                                        ml_series = _load_returns_series(ml_returns_path, preferred_col="portfolio_return_ml")
                                        psr_payload = probabilistic_sharpe_ratio(
                                            returns=ml_series.to_numpy(dtype=float),
                                            sr_benchmark=0.0,
                                            annualization_factor=252.0,
                                        )
                                        dsr_payload = deflated_sharpe_ratio(
                                            returns=ml_series.to_numpy(dtype=float),
                                            n_trials=n_trials,
                                            sr_mean=0.0,
                                            annualization_factor=252.0,
                                        )
                                        if isinstance(psr_payload.get("psr"), (int, float)):
                                            ml_psr = float(psr_payload["psr"])
                                        if isinstance(dsr_payload.get("dsr"), (int, float)):
                                            ml_dsr = float(dsr_payload["dsr"])

                                    gate_checks = {
                                        "oos": bool(oos_report.get("passed", False)),
                                        "psr": True if psr_threshold is None or ml_psr is None else bool(ml_psr >= float(psr_threshold)),
                                        "dsr": True if dsr_threshold is None or ml_dsr is None else bool(ml_dsr >= float(dsr_threshold)),
                                        "benchmark_injection": (not benchmark_injection),
                                    }
                                    hard_gate_passed = bool(all(gate_checks.values()))
                                    hard_gate_reasons = [k for k, ok in gate_checks.items() if not ok]

                                    rows.append(
                                        {
                                            "seed": int(seed),
                                            "hmm_n_states": int(n_states),
                                            "hmm_n_iter": int(n_iter),
                                            "hmm_var_floor": float(var_floor),
                                            "hmm_trans_sticky": float(trans_sticky),
                                            "rf_n_estimators": int(n_est),
                                            "rf_max_depth": int(depth),
                                            "run_id": str(sub_id),
                                            "out_dir": str(out_dir),
                                            "runtime_s": float(time.perf_counter() - t0),
                                            "oos_passed": bool(oos_report.get("passed", False)),
                                            "ml_psr": ml_psr,
                                            "ml_dsr": ml_dsr,
                                            "hard_gate_passed": hard_gate_passed,
                                            "hard_gate_reasons": ",".join(hard_gate_reasons),
                                            "benchmark_injection_detected": benchmark_injection,
                                            "validation_oos_available": _gate_value(gates, "oos", "available"),
                                            "validation_oos_passed": _gate_value(gates, "oos", "passed"),
                                            "validation_psr_available": _gate_value(gates, "psr", "available"),
                                            "validation_psr_passed": _gate_value(gates, "psr", "passed"),
                                            "validation_dsr_available": _gate_value(gates, "dsr", "available"),
                                            "validation_dsr_passed": _gate_value(gates, "dsr", "passed"),
                                            "validation_pbo_available": _gate_value(gates, "pbo", "available"),
                                            "validation_pbo_passed": _gate_value(gates, "pbo", "passed"),
                                            "sharpe": run_summary.get("metrics", {}).get("sharpe"),
                                            "annual_return": run_summary.get("metrics", {}).get("annual_return"),
                                            "annual_volatility": run_summary.get("metrics", {}).get("annual_volatility"),
                                            "max_drawdown": run_summary.get("metrics", {}).get("max_drawdown"),
                                            "turnover": run_summary.get("metrics", {}).get("turnover"),
                                            **overlay_metrics,
                                        }
                                    )
                                except Exception as exc:
                                    failures.append(
                                        {
                                            "seed": int(seed),
                                            "hmm_n_states": int(n_states),
                                            "hmm_n_iter": int(n_iter),
                                            "hmm_var_floor": float(var_floor),
                                            "hmm_trans_sticky": float(trans_sticky),
                                            "rf_n_estimators": int(n_est),
                                            "rf_max_depth": int(depth),
                                            "run_id": str(sub_id),
                                            "error": str(exc),
                                            "runtime_s": float(time.perf_counter() - t0),
                                        }
                                    )

    per_run_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    per_run_path = sweep_dir / "cpp_hmm_sweep_runs.csv"
    per_run_df.to_csv(per_run_path, index=False)
    if not failures_df.empty:
        failures_df.to_csv(sweep_dir / "cpp_hmm_sweep_failures.csv", index=False)

    group_cols = (
        "hmm_n_states",
        "hmm_n_iter",
        "hmm_var_floor",
        "hmm_trans_sticky",
        "rf_n_estimators",
        "rf_max_depth",
    )
    summary_df = summarize_sweep(per_run_df, group_cols=group_cols, sharpe_threshold=0.0)
    summary_path = sweep_dir / "cpp_hmm_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    best = pick_best_config_with_hard_gate(
        summary_df,
        score_col="ml_test_sharpe_mean",
        hard_gate_col="hard_gate_passed_mean",
    )

    per_run_pass_rate = (
        float(pd.to_numeric(per_run_df["hard_gate_passed"], errors="coerce").fillna(0.0).mean())
        if (not per_run_df.empty and "hard_gate_passed" in per_run_df.columns)
        else 0.0
    )
    hard_gates_passed = bool(per_run_pass_rate >= float(min_hard_gate_pass_rate))

    report = {
        "run_id": str(run_id),
        "out_root": str(out_root),
        "honest_defaults": {
            "template_default_universe": True,
            "template_use_precomputed_artifacts": False,
            "strict_independent_mode": True,
            "ml_meta_comparator_use_benchmark_csv": False,
            "parity_metrics_bridge": False,
        },
        "inputs": {
            "seeds": [int(s) for s in seeds],
            "hmm_n_states": [int(x) for x in hmm_n_states],
            "hmm_n_iter": [int(x) for x in hmm_n_iter],
            "hmm_var_floor": [float(x) for x in hmm_var_floor],
            "hmm_trans_sticky": [float(x) for x in hmm_trans_sticky],
            "rf_n_estimators": [int(x) for x in rf_n_estimators],
            "rf_max_depth": [int(x) for x in rf_max_depth],
        },
        "n_runs": int(len(per_run_df)),
        "n_failures": int(len(failures_df)),
        "best": best,
        "hard_gates": {
            "oos_min_sharpe": oos_min_sharpe,
            "oos_max_drawdown": oos_max_drawdown,
            "oos_max_turnover": oos_max_turnover,
            "oos_min_calmar": oos_min_calmar,
            "psr_threshold": psr_threshold,
            "dsr_threshold": dsr_threshold,
            "per_run_hard_gate_pass_rate": per_run_pass_rate,
            "min_hard_gate_pass_rate": float(min_hard_gate_pass_rate),
            "passed": hard_gates_passed,
        },
        "artifacts": {
            "per_run_csv": str(per_run_path),
            "summary_csv": str(summary_path),
            "failures_csv": str(sweep_dir / "cpp_hmm_sweep_failures.csv") if not failures_df.empty else None,
        },
    }
    report_path = sweep_dir / "cpp_hmm_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sweep_dir": str(sweep_dir), "report": str(report_path), "n_runs": report["n_runs"], "n_failures": report["n_failures"]}))

    if failures:
        return 1
    if require_hard_gates and not hard_gates_passed:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("run_cpp_hmm_sweep")
    p.add_argument("--run-id", default="cpp-hmm-sweep")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--seeds", default="42")
    p.add_argument("--hmm-n-states", default="2,3")
    p.add_argument("--hmm-n-iter", default="30,50")
    p.add_argument("--hmm-var-floor", default="1e-8,1e-6")
    p.add_argument("--hmm-trans-sticky", default="0.90,0.92,0.95")
    p.add_argument("--rf-n-estimators", default="400")
    p.add_argument("--rf-max-depth", default="4")
    p.add_argument("--no-fixtures", action="store_true", help="Do not set TEMA_IGNORE_TEMPLATE_DIR=1")
    p.add_argument("--oos-min-sharpe", type=float, default=0.5)
    p.add_argument("--oos-max-drawdown", type=float, default=0.25)
    p.add_argument("--oos-max-turnover", type=float, default=5.0)
    p.add_argument("--oos-min-calmar", type=float, default=None)
    p.add_argument("--psr-threshold", type=float, default=0.95)
    p.add_argument("--dsr-threshold", type=float, default=0.80)
    p.add_argument("--require-hard-gates", action="store_true")
    p.add_argument("--min-hard-gate-pass-rate", type=float, default=1.0)
    args = p.parse_args(argv)

    seeds = parse_int_list(args.seeds)
    n_states = parse_int_list(args.hmm_n_states)
    n_iter = parse_int_list(args.hmm_n_iter)
    var_floor = parse_float_list(args.hmm_var_floor)
    trans_sticky = parse_float_list(args.hmm_trans_sticky)
    rf_n_estimators = parse_int_list(args.rf_n_estimators)
    rf_max_depth = parse_int_list(args.rf_max_depth)

    if not seeds:
        raise SystemExit("--seeds must not be empty")
    if not n_states:
        raise SystemExit("--hmm-n-states must not be empty")
    if not n_iter:
        raise SystemExit("--hmm-n-iter must not be empty")
    if not var_floor:
        raise SystemExit("--hmm-var-floor must not be empty")
    if not trans_sticky:
        raise SystemExit("--hmm-trans-sticky must not be empty")
    if not rf_n_estimators:
        raise SystemExit("--rf-n-estimators must not be empty")
    if not rf_max_depth:
        raise SystemExit("--rf-max-depth must not be empty")

    return run(
        run_id=str(args.run_id),
        out_root=str(args.out_root),
        seeds=seeds,
        hmm_n_states=n_states,
        hmm_n_iter=n_iter,
        hmm_var_floor=var_floor,
        hmm_trans_sticky=trans_sticky,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        fixtures=not bool(args.no_fixtures),
        oos_min_sharpe=(float(args.oos_min_sharpe) if args.oos_min_sharpe is not None else None),
        oos_max_drawdown=(float(args.oos_max_drawdown) if args.oos_max_drawdown is not None else None),
        oos_max_turnover=(float(args.oos_max_turnover) if args.oos_max_turnover is not None else None),
        oos_min_calmar=(float(args.oos_min_calmar) if args.oos_min_calmar is not None else None),
        psr_threshold=(float(args.psr_threshold) if args.psr_threshold is not None else None),
        dsr_threshold=(float(args.dsr_threshold) if args.dsr_threshold is not None else None),
        require_hard_gates=bool(args.require_hard_gates),
        min_hard_gate_pass_rate=float(args.min_hard_gate_pass_rate),
    )


if __name__ == "__main__":
    raise SystemExit(main())
