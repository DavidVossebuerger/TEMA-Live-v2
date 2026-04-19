import numpy as np

from tema.validation.futuretesting import run_futuretesting


def test_run_futuretesting_reports_summary():
    returns = np.array([0.01, -0.005, 0.002, 0.004, -0.003], dtype=float)
    out = run_futuretesting(returns, n_paths=50, horizon=20, seed=7)
    assert out["enabled"] is True
    assert out["skipped"] is False
    assert out["n_paths"] == 50
    assert out["horizon"] == 20
    assert out["method"] == "stationary_bootstrap"
    assert isinstance(out["block_size"], int)
    assert out["block_size"] >= 1
    assert "terminal_return_mean" in out
    assert "max_drawdown_mean" in out


def test_run_futuretesting_handles_empty_input():
    out = run_futuretesting([], n_paths=10, horizon=5)
    assert out["enabled"] is True
    assert out["skipped"] is True
    assert out["reason"] == "empty_returns"


def test_run_futuretesting_iid_method_supported():
    returns = np.array([0.01, -0.005, 0.002, 0.004, -0.003], dtype=float)
    out = run_futuretesting(returns, n_paths=10, horizon=8, method="iid_bootstrap", seed=11)
    assert out["enabled"] is True
    assert out["skipped"] is False
    assert out["method"] == "iid_bootstrap"
    assert out["block_size"] is None
