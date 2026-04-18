import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline


def test_main_enables_default_validation_suite_by_default(monkeypatch):
    captured = {}

    def _fake_run_modular(run_id, out_root="outputs", **kwargs):
        captured["run_id"] = run_id
        captured["out_root"] = out_root
        captured["kwargs"] = kwargs
        return {"run_id": run_id, "out_dir": "x", "manifest_path": "y"}

    monkeypatch.setattr(run_pipeline, "run_modular", _fake_run_modular)
    run_pipeline.main(["--run-id", "suite-default", "--out-root", "o"])

    assert captured["run_id"] == "suite-default"
    assert captured["out_root"] == "o"
    assert captured["kwargs"]["default_validation_suite_enabled"] is True
    assert captured["kwargs"]["validation_mc_n_paths"] == 10000


def test_run_modular_validation_suite_writes_artifacts(tmp_path, monkeypatch):
    def _fake_pipeline(run_id, cfg, out_root):
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range("2020-01-01", periods=320, freq="B", tz="UTC")
        rng = np.random.default_rng(123)
        baseline = rng.normal(loc=0.0003, scale=0.01, size=len(dates))
        ml = baseline + 0.0002

        pd.DataFrame({"datetime": dates.astype(str), "portfolio_return": baseline}).to_csv(
            out_dir / "portfolio_test_returns.csv",
            index=False,
        )
        pd.DataFrame({"datetime": dates.astype(str), "portfolio_return_ml": ml}).to_csv(
            out_dir / "portfolio_test_returns_ml.csv",
            index=False,
        )

        perf = {
            "sharpe": 1.0,
            "annual_return": 0.1,
            "annual_volatility": 0.1,
            "annualized_turnover": 0.5,
            "max_drawdown": -0.1,
        }
        (out_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")
        manifest = {"run_id": run_id, "artifacts": ["performance"]}
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "manifest_path": str(out_dir / "manifest.json"),
            "out_dir": str(out_dir),
            "manifest": manifest,
        }

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)

    res = run_pipeline.run_modular(
        run_id="suite-artifacts",
        out_root=str(tmp_path / "outputs"),
        default_validation_suite_enabled=True,
        validation_mc_n_paths=500,
        validation_mc_horizon=80,
        validation_bootstrap_n_samples=300,
    )

    out_dir = Path(res["out_dir"])
    assert (out_dir / "walkforward_report.json").exists()
    assert (out_dir / "oos_report.json").exists()
    assert (out_dir / "bootstrap_baseline.json").exists()
    assert (out_dir / "bootstrap_ml.json").exists()
    assert (out_dir / "bootstrap_comparison_baseline_vs_ml.json").exists()
    assert (out_dir / "mc_paths_summary.json").exists()
    assert (out_dir / "validation_summary.json").exists()
    assert (out_dir / "validation_charts" / "wf_sharpe_comparison.png").exists()
    assert (out_dir / "validation_charts" / "wf_drawdown_comparison.png").exists()
    assert (out_dir / "validation_charts" / "bootstrap_sharpe_ci.png").exists()
    assert (out_dir / "validation_charts" / "mc_terminal_distribution.png").exists()

    summary = json.loads((out_dir / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["mc"]["n_paths"] == 500
    assert "default_validation_suite" in res
