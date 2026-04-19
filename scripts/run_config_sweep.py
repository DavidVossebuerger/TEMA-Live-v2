#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure repo src is on sys.path (like other scripts)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tema.config import BacktestConfig
from tema.pipeline import run_pipeline
from tema.validation.cpcv import evaluate_cpcv_strategies, generate_cpcv_splits
from tema.validation.oos import validate_oos_gates
from tema.validation.probabilistic import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from tema.validation.sweep import (
    build_heatmap,
    parse_float_list,
    parse_int_list,
    pick_best_config,
    summarize_sweep,
)


def _float_token(x: float) -> str:
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


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


def run(
    *,
    run_id: str,
    out_root: str = "outputs",
    seeds: list[int],
    vol_targets: list[float],
    max_scalars: list[float],
    fixtures: bool = True,
    sharpe_threshold: float = 0.5,
    ml_meta: bool = False,
    oos_min_sharpe: float | None = 0.5,
    oos_max_drawdown: float | None = 0.25,
    oos_max_turnover: float | None = 5.0,
    oos_min_calmar: float | None = None,
    psr_threshold: float | None = 0.95,
    dsr_threshold: float | None = 0.80,
    pbo_max: float | None = 0.50,
    cpcv_n_groups: int = 10,
    cpcv_n_test_groups: int = 2,
    cpcv_purge_groups: int = 1,
    cpcv_embargo_groups: int = 1,
    cpcv_max_splits: int | None = 128,
    require_hard_gates: bool = False,
    min_hard_gate_pass_rate: float = 1.0,
) -> int:
    sweep_dir = Path(out_root) / run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if fixtures:
        os.environ["TEMA_IGNORE_TEMPLATE_DIR"] = "1"

    rows: list[dict] = []
    failures: list[dict] = []
    ml_return_series_by_run: dict[str, pd.Series] = {}
    n_trials = max(2, int(len(seeds) * len(vol_targets) * len(max_scalars)))

    for seed in seeds:
        for vol in vol_targets:
            for mx in max_scalars:
                sub_id = f"{run_id}__s{seed}__v{_float_token(vol)}__m{_float_token(mx)}"
                t0 = time.perf_counter()
                try:
                    cfg = BacktestConfig(
                        template_default_universe=True,
                        template_use_precomputed_artifacts=True,
                        ml_template_overlay_enabled=True,
                        ml_meta_overlay_enabled=bool(ml_meta),
                        rf_random_state=int(seed),
                        vol_target_annual=float(vol),
                        ml_position_scalar_max=float(mx),
                    )
                    res = run_pipeline(run_id=sub_id, cfg=cfg, out_root=out_root)
                    out_dir = Path(res["out_dir"])

                    overlay_path = out_dir / "template_ml_overlay.json"
                    if not overlay_path.exists():
                        raise FileNotFoundError(f"missing template_ml_overlay.json in {out_dir}")
                    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
                    metrics = _extract_overlay_metrics(overlay)

                    manifest_path = out_dir / "manifest.json"
                    if not manifest_path.exists():
                        raise FileNotFoundError(f"missing manifest.json in {out_dir}")
                    oos_report = validate_oos_gates(
                        str(manifest_path),
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
                        ml_return_series_by_run[sub_id] = ml_series
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
                    }
                    hard_gate_passed = bool(all(gate_checks.values()))
                    hard_gate_reasons = [k for k, ok in gate_checks.items() if not ok]

                    rows.append(
                        {
                            "seed": int(seed),
                            "vol_target_annual": float(vol),
                            "ml_position_scalar_max": float(mx),
                            "run_id": str(sub_id),
                            "out_dir": str(out_dir),
                            "runtime_s": float(time.perf_counter() - t0),
                            "oos_passed": bool(oos_report.get("passed", False)),
                            "ml_psr": ml_psr,
                            "ml_dsr": ml_dsr,
                            "hard_gate_passed": hard_gate_passed,
                            "hard_gate_reasons": ",".join(hard_gate_reasons),
                            **metrics,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "seed": int(seed),
                            "vol_target_annual": float(vol),
                            "ml_position_scalar_max": float(mx),
                            "run_id": str(sub_id),
                            "error": str(exc),
                            "runtime_s": float(time.perf_counter() - t0),
                        }
                    )

    per_run_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)

    per_run_path = sweep_dir / "config_sweep_runs.csv"
    per_run_df.to_csv(per_run_path, index=False)
    if not failures_df.empty:
        failures_path = sweep_dir / "config_sweep_failures.csv"
        failures_df.to_csv(failures_path, index=False)

    summary_df = summarize_sweep(
        per_run_df,
        group_cols=("vol_target_annual", "ml_position_scalar_max"),
        sharpe_threshold=sharpe_threshold,
    )
    summary_path = sweep_dir / "config_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    heatmap = build_heatmap(
        summary_df,
        index="ml_position_scalar_max",
        columns="vol_target_annual",
        values="ml_test_sharpe_mean",
    )
    heatmap_path = sweep_dir / "config_sweep_heatmap_ml_test_sharpe_mean.csv"
    heatmap.to_csv(heatmap_path)

    best = pick_best_config(summary_df, score_col="ml_test_sharpe_mean")

    cpcv_report: dict = {"skipped": True, "reason": "insufficient_sweep_runs_for_cpcv"}
    pbo_value = None
    pbo_passed = True
    if len(ml_return_series_by_run) >= 2:
        cpcv_df = pd.concat(ml_return_series_by_run, axis=1).dropna(how="all")
        cpcv_df.columns = [str(c) for c in cpcv_df.columns]
        if cpcv_df.shape[0] >= max(20, int(cpcv_n_groups)):
            splits = generate_cpcv_splits(
                cpcv_df.index,
                n_groups=int(cpcv_n_groups),
                n_test_groups=int(cpcv_n_test_groups),
                purge_groups=int(cpcv_purge_groups),
                embargo_groups=int(cpcv_embargo_groups),
                max_splits=cpcv_max_splits,
                seed=42,
            )
            cpcv_report = evaluate_cpcv_strategies(
                cpcv_df,
                splits=splits,
                annualization_factor=252.0,
                metric="sharpe",
            )
            cpcv_report["split_generation"] = {
                "n_groups": int(cpcv_n_groups),
                "n_test_groups": int(cpcv_n_test_groups),
                "purge_groups": int(cpcv_purge_groups),
                "embargo_groups": int(cpcv_embargo_groups),
                "n_splits_generated": int(len(splits)),
            }
            if (
                isinstance(cpcv_report.get("pbo"), dict)
                and isinstance(cpcv_report["pbo"].get("pbo"), (int, float))
            ):
                pbo_value = float(cpcv_report["pbo"]["pbo"])
    if pbo_max is not None and pbo_value is not None:
        pbo_passed = bool(pbo_value <= float(pbo_max))
    cpcv_report_path = sweep_dir / "config_sweep_cpcv_report.json"
    cpcv_report_path.write_text(json.dumps(cpcv_report, indent=2), encoding="utf-8")

    per_run_pass_rate = (
        float(pd.to_numeric(per_run_df["hard_gate_passed"], errors="coerce").fillna(0.0).mean())
        if (not per_run_df.empty and "hard_gate_passed" in per_run_df.columns)
        else 0.0
    )
    hard_gates_passed = bool(per_run_pass_rate >= float(min_hard_gate_pass_rate) and pbo_passed)

    report = {
        "run_id": str(run_id),
        "out_root": str(out_root),
        "fixtures": bool(fixtures),
        "seeds": [int(s) for s in seeds],
        "vol_targets": [float(v) for v in vol_targets],
        "max_scalars": [float(m) for m in max_scalars],
        "sharpe_threshold": float(sharpe_threshold),
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
            "pbo_max": pbo_max,
            "per_run_hard_gate_pass_rate": per_run_pass_rate,
            "min_hard_gate_pass_rate": float(min_hard_gate_pass_rate),
            "pbo": pbo_value,
            "pbo_passed": bool(pbo_passed),
            "passed": hard_gates_passed,
        },
        "artifacts": {
            "per_run_csv": str(per_run_path),
            "summary_csv": str(summary_path),
            "heatmap_csv": str(heatmap_path),
            "failures_csv": str(sweep_dir / "config_sweep_failures.csv") if not failures_df.empty else None,
            "cpcv_report_json": str(cpcv_report_path),
        },
    }
    report_path = sweep_dir / "config_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "sweep_dir": str(sweep_dir),
                "report": str(report_path),
                "n_runs": report["n_runs"],
                "n_failures": report["n_failures"],
                "hard_gates_passed": hard_gates_passed,
            }
        )
    )
    if failures:
        return 1
    if require_hard_gates and not hard_gates_passed:
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser("run_config_sweep")
    p.add_argument("--run-id", default="config-sweep")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    p.add_argument("--vol-targets", default="0.08,0.10,0.12")
    p.add_argument("--max-scalars", default="10,25,50")
    p.add_argument("--sharpe-threshold", type=float, default=0.5)
    p.add_argument("--no-fixtures", action="store_true", help="Do not set TEMA_IGNORE_TEMPLATE_DIR=1")
    p.add_argument("--ml-meta", action="store_true", help="Also enable ML_META overlay")
    p.add_argument("--oos-min-sharpe", type=float, default=0.5)
    p.add_argument("--oos-max-drawdown", type=float, default=0.25)
    p.add_argument("--oos-max-turnover", type=float, default=5.0)
    p.add_argument("--oos-min-calmar", type=float, default=None)
    p.add_argument("--psr-threshold", type=float, default=0.95)
    p.add_argument("--dsr-threshold", type=float, default=0.80)
    p.add_argument("--pbo-max", type=float, default=0.50)
    p.add_argument("--cpcv-n-groups", type=int, default=10)
    p.add_argument("--cpcv-n-test-groups", type=int, default=2)
    p.add_argument("--cpcv-purge-groups", type=int, default=1)
    p.add_argument("--cpcv-embargo-groups", type=int, default=1)
    p.add_argument("--cpcv-max-splits", type=int, default=128)
    p.add_argument("--require-hard-gates", action="store_true")
    p.add_argument("--min-hard-gate-pass-rate", type=float, default=1.0)
    args = p.parse_args(argv)

    seeds = parse_int_list(args.seeds)
    vol_targets = parse_float_list(args.vol_targets)
    max_scalars = parse_float_list(args.max_scalars)

    if not seeds:
        raise SystemExit("--seeds must not be empty")
    if not vol_targets:
        raise SystemExit("--vol-targets must not be empty")
    if not max_scalars:
        raise SystemExit("--max-scalars must not be empty")

    return run(
        run_id=str(args.run_id),
        out_root=str(args.out_root),
        seeds=seeds,
        vol_targets=vol_targets,
        max_scalars=max_scalars,
        fixtures=not bool(args.no_fixtures),
        sharpe_threshold=float(args.sharpe_threshold),
        ml_meta=bool(args.ml_meta),
        oos_min_sharpe=(float(args.oos_min_sharpe) if args.oos_min_sharpe is not None else None),
        oos_max_drawdown=(float(args.oos_max_drawdown) if args.oos_max_drawdown is not None else None),
        oos_max_turnover=(float(args.oos_max_turnover) if args.oos_max_turnover is not None else None),
        oos_min_calmar=(float(args.oos_min_calmar) if args.oos_min_calmar is not None else None),
        psr_threshold=(float(args.psr_threshold) if args.psr_threshold is not None else None),
        dsr_threshold=(float(args.dsr_threshold) if args.dsr_threshold is not None else None),
        pbo_max=(float(args.pbo_max) if args.pbo_max is not None else None),
        cpcv_n_groups=int(args.cpcv_n_groups),
        cpcv_n_test_groups=int(args.cpcv_n_test_groups),
        cpcv_purge_groups=int(args.cpcv_purge_groups),
        cpcv_embargo_groups=int(args.cpcv_embargo_groups),
        cpcv_max_splits=(int(args.cpcv_max_splits) if args.cpcv_max_splits is not None else None),
        require_hard_gates=bool(args.require_hard_gates),
        min_hard_gate_pass_rate=float(args.min_hard_gate_pass_rate),
    )


if __name__ == "__main__":
    raise SystemExit(main())
