import numpy as np
import pandas as pd

from tema.ml.har_rv import build_har_rv_features


def test_har_rv_features_are_shifted_and_deterministic():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.01], index=idx)

    f1 = build_har_rv_features(r, windows=(1, 2, 3), shift_by=1, use_log=False)
    f2 = build_har_rv_features(r, windows=(1, 2, 3), shift_by=1, use_log=False)
    assert list(f1.columns) == ["har_rv_1", "har_rv_2", "har_rv_3"]
    assert f1.equals(f2)
    assert float(f1.iloc[0]["har_rv_1"]) == 0.0

    expected_day_2 = float(r.iloc[1] ** 2)
    assert np.isclose(float(f1.iloc[2]["har_rv_1"]), expected_day_2)


def test_har_rv_log_transform_non_negative():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    r = pd.Series([0.01, 0.0, -0.02, 0.01], index=idx)
    f = build_har_rv_features(r, windows=(1, 2), shift_by=1, use_log=True)
    assert (f.to_numpy(dtype=float) >= 0.0).all()
