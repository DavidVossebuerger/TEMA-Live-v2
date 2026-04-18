import pandas as pd

from tema.data.quality import DataQualityConfig, compute_data_quality_report


def test_data_quality_report_flags_nans_and_nonpositive():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "A": [1.0, 1.1, None, 1.2, 1.3],
            "B": [1.0, 0.0, 1.1, 1.2, 1.3],
        },
        index=idx,
    )

    cfg = DataQualityConfig(max_nan_frac=0.10, max_gap_days=10.0, min_price=1e-12)
    rep = compute_data_quality_report(df, cfg=cfg)
    assert rep["passed"] is False
    assert "failed_assets" in rep
    assert set(rep["failed_assets"]) == {"A", "B"}


def test_data_quality_report_ok():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"A": [1.0, 1.1, 1.05, 1.2, 1.3]}, index=idx)
    cfg = DataQualityConfig(max_nan_frac=0.20, max_gap_days=10.0, min_price=1e-12)
    rep = compute_data_quality_report(df, cfg=cfg)
    assert rep["passed"] is True
    assert rep["summary"] == "ok"
