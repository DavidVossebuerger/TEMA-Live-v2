import json
from pathlib import Path

import pandas as pd
import pytest


def test_pipeline_writes_data_quality_artifact_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    from tema.config import BacktestConfig
    from tema.pipeline import run_pipeline

    out_root = str(tmp_path / "outputs")
    cfg = BacktestConfig(
        template_default_universe=True,
        ml_template_overlay_enabled=True,
        data_quality_enabled=True,
        # Don't require pass here; some datasets may legitimately have large gaps.
        data_quality_fail_fast=False,
    )

    res = run_pipeline(run_id="dq-test", cfg=cfg, out_root=out_root)
    out_dir = Path(res["out_dir"])
    dq_path = out_dir / "data_quality.json"
    assert dq_path.exists()

    payload = json.loads(dq_path.read_text(encoding="utf-8"))
    assert "passed" in payload
    assert "per_asset" in payload


def test_pipeline_data_quality_fail_fast_raises_and_writes_artifact(tmp_path):
    from tema.config import BacktestConfig
    from tema.data.quality import DataQualityFailed
    from tema.pipeline import run_pipeline

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "datetime": [
                "2020-01-01T00:00:00Z",
                "2020-01-02T00:00:00Z",
                "2020-01-03T00:00:00Z",
                "2020-01-04T00:00:00Z",
                "2020-01-05T00:00:00Z",
            ],
            "Close": [1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    df.to_csv(data_dir / "AAA_merged.csv", index=False)

    out_root = str(tmp_path / "outputs")
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        data_path=str(data_dir),
        data_max_assets=1,
        data_min_rows=3,
        data_quality_enabled=True,
        data_quality_fail_fast=True,
        data_quality_min_price=1e-6,
    )

    with pytest.raises(DataQualityFailed):
        run_pipeline(run_id="dq-failfast", cfg=cfg, out_root=out_root)

    dq_path = Path(out_root) / "dq-failfast" / "data_quality.json"
    assert dq_path.exists()
