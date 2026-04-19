import numpy as np

from tema.execution.almgren_chriss import apply_almgren_chriss_step
from tema.portfolio.dynamic_tcost import apply_dynamic_trading_path


def test_apply_dynamic_trading_path_is_shape_stable():
    target = np.array([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]], dtype=float)
    out, diag = apply_dynamic_trading_path(
        target,
        lambda_cost=0.5,
        aim_multiplier=0.2,
        min_trade_rate=0.1,
        max_trade_rate=0.9,
    )
    assert out.shape == target.shape
    assert diag["enabled"] is True
    assert np.all(np.isfinite(out))
    assert np.allclose(np.sum(out, axis=1), 1.0)


def test_apply_almgren_chriss_step_partially_moves_to_target():
    prev = np.array([0.6, 0.4], dtype=float)
    target = np.array([0.2, 0.8], dtype=float)
    executed, diag = apply_almgren_chriss_step(
        prev,
        target,
        volatility=0.02,
        n_slices=5,
        risk_aversion=1.0,
        temporary_impact=0.1,
        permanent_impact=0.02,
    )
    assert np.all(np.isfinite(executed))
    assert diag["fill_rate"] > 0.0
    assert not np.allclose(executed, prev)
