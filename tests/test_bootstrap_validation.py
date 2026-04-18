import json
from pathlib import Path

import numpy as np
import pandas as pd

from tema.validation.bootstrap import (
    bootstrap_compare_returns,
    bootstrap_metric_confidence_intervals,
    sample_bootstrap_paths,
)


def test_sample_bootstrap_paths_block_deterministic():
    returns = np.array([0.01, -0.02, 0.005, 0.015, -0.01], dtype=float)
    a = sample_bootstrap_paths(returns, n_samples=8, seed=7, method="block", block_size=2)
    b = sample_bootstrap_paths(returns, n_samples=8, seed=7, method="block", block_size=2)
    assert a.shape == (8, 5)
    assert np.allclose(a, b)


def test_bootstrap_ci_contains_mean_for_sharpe():
    rng = np.random.default_rng(5)
    returns = rng.normal(loc=0.0003, scale=0.01, size=300)
    result = bootstrap_metric_confidence_intervals(
        returns=returns,
        n_samples=400,
        seed=11,
        method="iid",
        ci_level=0.95,
    )
    sharpe = result["metrics"]["sharpe"]
    assert sharpe["ci_lower"] <= sharpe["mean"] <= sharpe["ci_upper"]
    assert result["n_samples"] == 400
    assert result["n_obs"] == 300


def test_bootstrap_compare_detects_candidate_improvement():
    rng = np.random.default_rng(9)
    baseline = rng.normal(loc=0.0001, scale=0.01, size=420)
    candidate = baseline + 0.0006
    comparison = bootstrap_compare_returns(
        baseline_returns=baseline,
        candidate_returns=candidate,
        metric="annual_return",
        n_samples=500,
        seed=17,
        method="block",
        block_size=10,
        ci_level=0.95,
    )
    assert comparison["improvement_probability"] > 0.8
    assert comparison["passed_ci_positive"] is True


def test_bootstrap_script_smoke(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2021-01-01", periods=280, freq="B", tz="UTC")
    rng = np.random.default_rng(21)
    baseline = rng.normal(loc=0.0002, scale=0.009, size=len(dates))
    ml = baseline + 0.00025

    pd.DataFrame({"datetime": dates.astype(str), "portfolio_return": baseline}).to_csv(
        run_dir / "portfolio_test_returns.csv", index=False
    )
    pd.DataFrame({"datetime": dates.astype(str), "portfolio_return_ml": ml}).to_csv(
        run_dir / "portfolio_test_returns_ml.csv", index=False
    )

    from scripts import run_bootstrap_tests

    payload = run_bootstrap_tests.run(
        path=str(run_dir),
        baseline_selector="baseline",
        candidate_selector="ml",
        metric="sharpe",
        n_samples=300,
        method="block",
        block_size=8,
        seed=33,
    )

    artifacts = payload["artifacts"]
    for key in ("baseline", "candidate", "comparison"):
        p = Path(artifacts[key])
        assert p.exists()
        # ensure valid JSON payload
        json.loads(p.read_text(encoding="utf-8"))
