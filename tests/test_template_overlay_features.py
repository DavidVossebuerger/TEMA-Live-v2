import numpy as np
import pandas as pd

from tema.config import BacktestConfig
from tema.ml.template_overlay import build_rf_feature_matrix


def test_build_rf_feature_matrix_defaults_unchanged():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    r = pd.Series(np.linspace(-0.01, 0.02, len(idx)), index=idx)
    hmm = np.column_stack([np.linspace(0.2, 0.8, len(idx)), np.linspace(0.8, 0.2, len(idx))])

    out = build_rf_feature_matrix(r, hmm)
    assert "ret_fracdiff" not in out.columns
    assert "har_rv_1" not in out.columns
    assert "hmm_p_0" in out.columns
    assert "hmm_p_1" in out.columns


def test_build_rf_feature_matrix_includes_optional_advanced_features():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    r = pd.Series(np.sin(np.arange(len(idx)) / 3.0) * 0.01, index=idx)
    hmm = np.column_stack([np.linspace(0.3, 0.7, len(idx)), np.linspace(0.7, 0.3, len(idx))])
    cfg = BacktestConfig(
        ml_feature_fracdiff_enabled=True,
        ml_feature_fracdiff_order=0.4,
        ml_feature_har_rv_enabled=True,
        ml_feature_har_rv_windows=(1, 5, 10),
    )

    out = build_rf_feature_matrix(r, hmm, cfg=cfg)
    assert "ret_fracdiff" in out.columns
    assert "har_rv_1" in out.columns
    assert "har_rv_5" in out.columns
    assert "har_rv_10" in out.columns
