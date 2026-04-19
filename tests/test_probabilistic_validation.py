import numpy as np

from tema.validation.probabilistic import (
    deflated_sharpe_ratio,
    pbo_from_rank_logits,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_sharpe_high_for_positive_drift():
    rng = np.random.default_rng(123)
    r = rng.normal(loc=0.0008, scale=0.01, size=600)
    payload = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert payload["ok"] is True
    assert float(payload["psr"]) > 0.80


def test_deflated_sharpe_not_above_psr_same_series():
    rng = np.random.default_rng(321)
    r = rng.normal(loc=0.0005, scale=0.01, size=600)
    psr_payload = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    dsr_payload = deflated_sharpe_ratio(r, n_trials=50)
    assert dsr_payload["ok"] is True
    assert float(dsr_payload["dsr"]) <= float(psr_payload["psr"])


def test_pbo_from_logits():
    payload = pbo_from_rank_logits(np.array([1.2, 0.2, -0.1, -2.0], dtype=float))
    assert payload["n_logits"] == 4
    assert payload["pbo_count"] == 2
    assert payload["pbo"] == 0.5

