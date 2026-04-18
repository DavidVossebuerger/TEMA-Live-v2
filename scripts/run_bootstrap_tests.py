#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure repo src is on sys.path (same pattern as other scripts)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tema.validation.bootstrap import bootstrap_compare_returns, bootstrap_metric_confidence_intervals


def _find_returns_csv(run_path: Path, selector: str) -> Path:
    mapping = {
        "baseline": "portfolio_test_returns.csv",
        "ml": "portfolio_test_returns_ml.csv",
        "ml_meta": "portfolio_test_returns_ml_meta.csv",
    }
    name = mapping.get(selector)
    if name is None:
        raise ValueError(f"unknown selector: {selector}")
    p = run_path / name
    if p.exists():
        return p

    # Fallback scan
    for f in run_path.iterdir():
        if not (f.is_file() and f.name.startswith("portfolio_test_returns") and f.suffix == ".csv"):
            continue
        if selector == "baseline" and f.name == "portfolio_test_returns.csv":
            return f
        if selector == "ml" and "ml" in f.name and "meta" not in f.name:
            return f
        if selector == "ml_meta" and "ml_meta" in f.name:
            return f
    raise FileNotFoundError(f"returns CSV not found for selector '{selector}' in {run_path}")


def _load_returns(csv_path: Path, selector: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    preferred_cols = {
        "baseline": "portfolio_return",
        "ml": "portfolio_return_ml",
        "ml_meta": "portfolio_return_ml_meta",
    }
    col = preferred_cols.get(selector)
    if col not in df.columns:
        candidates = [c for c in df.columns if c != "datetime"]
        if not candidates:
            raise ValueError(f"no return column found in {csv_path}")
        col = candidates[0]
    return df[col].astype(float)


def run(
    path: str,
    baseline_selector: str = "baseline",
    candidate_selector: str = "ml",
    metric: str = "sharpe",
    n_samples: int = 2000,
    method: str = "block",
    block_size: int = 20,
    seed: int = 42,
    annualization_factor: float = 252.0,
    ci_level: float = 0.95,
) -> dict:
    p = Path(path)
    if p.is_dir():
        run_dir = p
    elif p.exists():
        run_dir = p.parent
    else:
        raise FileNotFoundError(f"path not found: {path}")

    base_csv = _find_returns_csv(run_dir, baseline_selector)
    cand_csv = _find_returns_csv(run_dir, candidate_selector)
    base_returns = _load_returns(base_csv, baseline_selector).to_numpy(dtype=float)
    cand_returns = _load_returns(cand_csv, candidate_selector).to_numpy(dtype=float)

    baseline_ci = bootstrap_metric_confidence_intervals(
        returns=base_returns,
        n_samples=n_samples,
        seed=seed,
        method=method,
        block_size=block_size,
        annualization_factor=annualization_factor,
        ci_level=ci_level,
    )
    candidate_ci = bootstrap_metric_confidence_intervals(
        returns=cand_returns,
        n_samples=n_samples,
        seed=seed + 101,
        method=method,
        block_size=block_size,
        annualization_factor=annualization_factor,
        ci_level=ci_level,
    )
    comparison = bootstrap_compare_returns(
        baseline_returns=base_returns,
        candidate_returns=cand_returns,
        metric=metric,
        n_samples=n_samples,
        seed=seed + 202,
        method=method,
        block_size=block_size,
        annualization_factor=annualization_factor,
        ci_level=ci_level,
    )

    baseline_path = run_dir / f"bootstrap_{baseline_selector}.json"
    candidate_path = run_dir / f"bootstrap_{candidate_selector}.json"
    comparison_path = run_dir / f"bootstrap_comparison_{baseline_selector}_vs_{candidate_selector}.json"

    baseline_path.write_text(json.dumps(baseline_ci, indent=2), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_ci, indent=2), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    payload = {
        "run_dir": str(run_dir),
        "baseline_selector": baseline_selector,
        "candidate_selector": candidate_selector,
        "metric": metric,
        "artifacts": {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "comparison": str(comparison_path),
        },
        "comparison": comparison,
    }
    print(json.dumps(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run generic bootstrap tests on run returns CSVs")
    parser.add_argument("path", help="Run directory containing returns CSVs or path to manifest.json")
    parser.add_argument("--baseline-selector", default="baseline", choices=["baseline", "ml", "ml_meta"])
    parser.add_argument("--candidate-selector", default="ml", choices=["baseline", "ml", "ml_meta"])
    parser.add_argument("--metric", default="sharpe", choices=["sharpe", "annual_return", "annual_vol", "max_drawdown"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--method", default="block", choices=["iid", "block"])
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annualization-factor", type=float, default=252.0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--require-positive-ci",
        action="store_true",
        help="Exit with code 1 unless comparison CI lower bound is > 0",
    )
    args = parser.parse_args(argv)

    try:
        payload = run(
            path=args.path,
            baseline_selector=args.baseline_selector,
            candidate_selector=args.candidate_selector,
            metric=args.metric,
            n_samples=args.n_samples,
            method=args.method,
            block_size=args.block_size,
            seed=args.seed,
            annualization_factor=args.annualization_factor,
            ci_level=args.ci_level,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    if args.require_positive_ci and not payload.get("comparison", {}).get("passed_ci_positive", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
