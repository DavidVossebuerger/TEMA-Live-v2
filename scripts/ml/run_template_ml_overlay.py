from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tema.backtest import compute_backtest_metrics


def _metrics_from_csv(path: str, value_col: str) -> dict:
    df = pd.read_csv(path)
    r = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    eq = np.cumprod(1.0 + r)
    m = compute_backtest_metrics(r, eq, np.zeros_like(r), 252.0)
    m["equity_final"] = float(eq[-1]) if eq.size else 1.0
    return m


def main() -> int:
    p = argparse.ArgumentParser("run_template_ml_overlay")
    p.add_argument("--run-id", default="ml-overlay")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--ml-meta-overlay", action="store_true", help="Also compute ML_META (phase1 meta overlay) and validate parity")
    args = p.parse_args()

    # Run modular pipeline with template-default-universe + template ML overlay.
    import subprocess

    cmd = [
        "python",
        str(ROOT / "run_pipeline.py"),
        "--run-id",
        str(args.run_id),
        "--out-root",
        str(args.out_root),
        "--template-default-universe",
        "--ml-template-overlay",
    ]
    if bool(args.ml_meta_overlay):
        cmd.append("--ml-meta-overlay")

    subprocess.run(cmd, check=True, cwd=str(ROOT))

    out_dir = os.path.join(args.out_root, args.run_id)
    base_csv = os.path.join(out_dir, "portfolio_test_returns.csv")
    ml_csv = os.path.join(out_dir, "portfolio_test_returns_ml.csv")

    if not (os.path.exists(base_csv) and os.path.exists(ml_csv)):
        raise SystemExit(f"Missing output CSVs in {out_dir}. Expected portfolio_test_returns.csv + portfolio_test_returns_ml.csv")

    base_m = _metrics_from_csv(base_csv, "portfolio_return")
    ml_m = _metrics_from_csv(ml_csv, "portfolio_return_ml")

    print("SRC baseline:", base_m)
    print("SRC ML      :", ml_m)

    meta_csv = os.path.join(out_dir, "portfolio_test_returns_ml_meta.csv")
    if bool(args.ml_meta_overlay):
        if not os.path.exists(meta_csv):
            raise SystemExit(f"Missing output CSV in {out_dir}. Expected portfolio_test_returns_ml_meta.csv")
        meta_m = _metrics_from_csv(meta_csv, "portfolio_return_ml_meta")
        print("SRC ML_META :", meta_m)

    # Optional: compare against ground truth (Template/ if present; else src fixture).
    candidates = [
        {
            "base": os.path.join("Template", "portfolio_test_returns.csv"),
            "ml": os.path.join("Template", "portfolio_test_returns_ml.csv"),
            "meta": os.path.join("Template", "portfolio_test_returns_ml_meta.csv"),
        },
        {
            "base": os.path.join(ROOT, "src", "tema", "benchmarks", "template_default_universe", "portfolio_test_returns.csv"),
            "ml": os.path.join(ROOT, "src", "tema", "benchmarks", "template_default_universe", "portfolio_test_returns_ml.csv"),
            "meta": os.path.join(ROOT, "src", "tema", "benchmarks", "template_default_universe", "portfolio_test_returns_ml_meta.csv"),
        },
    ]

    for cand in candidates:
        tpl_base = cand["base"]
        tpl_ml = cand["ml"]
        tpl_meta = cand["meta"]
        if not (os.path.exists(tpl_base) and os.path.exists(tpl_ml)):
            continue
        if bool(args.ml_meta_overlay) and (not os.path.exists(tpl_meta)):
            continue

        a = pd.read_csv(tpl_base)["portfolio_return"].to_numpy(dtype=float)
        b = pd.read_csv(base_csv)["portfolio_return"].to_numpy(dtype=float)
        c = pd.read_csv(tpl_ml)["portfolio_return_ml"].to_numpy(dtype=float)
        d = pd.read_csv(ml_csv)["portfolio_return_ml"].to_numpy(dtype=float)

        print("ground_truth:", tpl_base)
        print("baseline max_abs_diff:", float(np.max(np.abs(a - b))) if len(a) == len(b) else "len_mismatch")
        print("ml       max_abs_diff:", float(np.max(np.abs(c - d))) if len(c) == len(d) else "len_mismatch")

        if bool(args.ml_meta_overlay):
            e = pd.read_csv(tpl_meta)["portfolio_return_ml_meta"].to_numpy(dtype=float)
            f = pd.read_csv(meta_csv)["portfolio_return_ml_meta"].to_numpy(dtype=float)
            print("ml_meta  max_abs_diff:", float(np.max(np.abs(e - f))) if len(e) == len(f) else "len_mismatch")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
