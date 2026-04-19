import numpy as np
import pandas as pd
import pytest

from tema.data.fracdiff import fractionally_differentiate, get_fracdiff_weights


def test_get_fracdiff_weights_deterministic():
    w1 = get_fracdiff_weights(0.4, threshold=1e-5, max_terms=64)
    w2 = get_fracdiff_weights(0.4, threshold=1e-5, max_terms=64)
    assert np.allclose(w1, w2)
    assert len(w1) <= 64


def test_fractional_diff_order_zero_is_identity():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = fractionally_differentiate(s, order=0.0, threshold=1e-8, max_terms=16)
    assert np.allclose(out.to_numpy(dtype=float), s.to_numpy(dtype=float))


def test_fractional_diff_rejects_invalid_order():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="order must be"):
        fractionally_differentiate(s, order=-0.1)
