import numpy as np
import pandas as pd

from tema.validation.cpcv import (
    build_cpcv_groups,
    evaluate_cpcv_strategies,
    generate_cpcv_splits,
)


def test_generate_cpcv_splits_with_purge_embargo():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    groups = build_cpcv_groups(idx, n_groups=6)
    assert len(groups) == 6

    splits = generate_cpcv_splits(
        idx,
        n_groups=6,
        n_test_groups=2,
        purge_groups=1,
        embargo_groups=1,
        max_splits=None,
        seed=42,
    )
    assert len(splits) > 0
    first = splits[0]
    assert len(first["test_groups"]) == 2
    assert len(first["train_idx"]) > 0
    assert len(first["test_idx"]) > 0


def test_evaluate_cpcv_returns_pbo_block():
    rng = np.random.default_rng(99)
    idx = pd.date_range("2021-01-01", periods=260, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "s1": rng.normal(0.0004, 0.01, len(idx)),
            "s2": rng.normal(0.0001, 0.01, len(idx)),
            "s3": rng.normal(-0.0001, 0.01, len(idx)),
        },
        index=idx,
    )
    splits = generate_cpcv_splits(
        df.index,
        n_groups=10,
        n_test_groups=2,
        purge_groups=1,
        embargo_groups=1,
        max_splits=32,
        seed=7,
    )
    report = evaluate_cpcv_strategies(df, splits=splits, metric="sharpe")
    assert report["skipped"] is False
    assert report["n_splits"] > 0
    assert "pbo" in report
    assert 0.0 <= float(report["pbo"]["pbo"]) <= 1.0

