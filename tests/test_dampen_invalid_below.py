"""
Unit tests for optional invalid_below threshold in dampen_turnover (R6-3).
Validates that:
(a) When fewer than top_k_in stocks are valid:
    - invalid_below=None admits filtered stocks (legacy behavior).
    - invalid_below=-900.0 excludes filtered stocks, allocating only to valid stocks and holding cash.
(b) When all stocks are filtered (<= invalid_below), weights remain 0 (100% cash).
(c) invalid_below=None preserves legacy behavior exactly.
"""
import numpy as np
import pytest
from engine import alpha_ops as ops


def test_dampen_invalid_below_fewer_than_top_k():
    """
    (a) Synthetic data with T=5, N=4, top_k_in=2, top_k_out=3.
    Only stock 0 is valid (> -900.0); stocks 1, 2, 3 have -999.0 (trend-filtered).
    """
    T, N = 5, 4
    alpha = np.full((T, N), -999.0)
    alpha[:, 0] = 1.5  # Only stock 0 is valid
    
    # Legacy behavior (invalid_below=None):
    # -999.0 is treated as valid (> -1e30), so dampen_turnover forces 2 stocks into portfolio.
    w_legacy = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=2, max_daily_turnover=1.0, invalid_below=None)
    # At t=0, legacy must hold 2 stocks (stock 0 and one of the filtered stocks)
    assert np.sum(w_legacy[0] > 0.0) == 2
    assert w_legacy[0, 0] > 0.0
    assert np.any(w_legacy[0, 1:] > 0.0)  # Filtered stock was admitted!
    
    # Sanitized behavior (invalid_below=-900.0):
    # Stocks 1, 2, 3 are disqualified; only stock 0 is admitted.
    w_clean = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=2, max_daily_turnover=1.0, invalid_below=-900.0)
    # Only stock 0 is held, remaining capital is cash (sum of weights <= 1.0 / top_k_in = 0.5)
    assert w_clean[0, 0] == 0.5
    assert np.all(w_clean[0, 1:] == 0.0)
    assert np.sum(w_clean[0]) == pytest.approx(0.5)
    
    # Over time, only stock 0 continues to be held
    for t in range(T):
        assert np.all(w_clean[t, 1:] == 0.0)
        assert w_clean[t, 0] <= 0.5 + 1e-9


def test_dampen_invalid_below_all_filtered():
    """
    (b) When all stocks are filtered (<= invalid_below), weights must be all zero (100% cash).
    """
    T, N = 4, 5
    alpha = np.full((T, N), -999.0)  # All stocks below threshold
    
    w_clean = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=4, ema_span=5, max_daily_turnover=0.5, invalid_below=-900.0)
    assert np.all(w_clean == 0.0)
    assert np.max(np.abs(w_clean)) == 0.0


def test_dampen_invalid_below_none_preserves_legacy():
    """
    (c) invalid_below=None preserves exact legacy numerical results.
    """
    rng = np.random.RandomState(42)
    T, N = 10, 6
    alpha = rng.randn(T, N)
    key = rng.randn(T, N)
    
    w_default = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=4, ema_span=10, max_daily_turnover=0.05, tie_break_key=key)
    w_explicit_none = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=4, ema_span=10, max_daily_turnover=0.05, tie_break_key=key, invalid_below=None)
    
    assert np.array_equal(w_default, w_explicit_none)
