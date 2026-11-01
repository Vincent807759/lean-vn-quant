import numpy as np
import pytest
from engine import alpha_ops as ops


def test_tiebreak_key_none_preserves_legacy_behavior():
    """
    (a) Verify tie_break_key=None produces exact legacy output matching hand-calculated values.
    T=3, N=4.
    At t=0, alpha=[1.0, 1.0, 0.5, 0.2]. Top-2 are tied at 1.0. Stable argsort selects indices 0 and 1.
    """
    T, N = 3, 4
    alpha = np.array([
        [1.0, 1.0, 0.5, 0.2],
        [0.8, 0.8, 0.3, 0.1],
        [0.5, 0.5, 0.5, 0.5]
    ], dtype=np.float64)
    
    w_legacy = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=1.0, tie_break_key=None)
    
    # Hand-calculation:
    # t=0: stocks 0 and 1 enter (weights 0.5, 0.5)
    assert np.isclose(w_legacy[0, 0], 0.5)
    assert np.isclose(w_legacy[0, 1], 0.5)
    assert np.isclose(w_legacy[0, 2], 0.0)
    assert np.isclose(w_legacy[0, 3], 0.0)
    
    # At t=1, stocks 0 and 1 still rank 1 and 2, held remains {0, 1}
    assert np.isclose(w_legacy[1, 0], 0.5)
    assert np.isclose(w_legacy[1, 1], 0.5)


def test_tiebreak_key_permutation_invariance_and_legacy_failure():
    """
    (b) When scores are tied at boundary (all 0.5), ADV20 liquidity tie_break_key ensures
    permutation invariance w_perm == w_orig[:, pi].
    Conversely, tie_break_key=None FAILS permutation invariance because argsort naively
    picks the first columns (indices 0, 1).
    """
    T, N = 3, 4
    # All alpha scores are completely tied at 0.5
    alpha = np.full((T, N), 0.5, dtype=np.float64)
    
    # Unique ADV liquidity key: stock 1 has 400M, stock 3 has 300M, stock 2 has 200M, stock 0 has 100M
    key = np.array([
        [100.0, 400.0, 200.0, 300.0],
        [100.0, 400.0, 200.0, 300.0],
        [100.0, 400.0, 200.0, 300.0]
    ], dtype=np.float64)
    
    # Permutation pi reverses the columns: [3, 2, 1, 0]
    pi = np.array([3, 2, 1, 0])
    
    # 1. With tie_break_key: must be strictly permutation-invariant
    w_orig = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=1.0, tie_break_key=key)
    w_perm = ops.dampen_turnover(alpha[:, pi], top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=1.0, tie_break_key=key[:, pi])
    
    # The selected stocks in w_orig must be the highest ADV (stock 1 and stock 3)
    assert np.isclose(w_orig[0, 1], 0.5)
    assert np.isclose(w_orig[0, 3], 0.5)
    assert np.isclose(w_orig[0, 0], 0.0)
    assert np.isclose(w_orig[0, 2], 0.0)
    
    # Permutation invariance: w_orig[:, pi] == w_perm
    assert np.allclose(w_orig[:, pi], w_perm), "Permutation invariance violated when tie_break_key is provided!"
    
    # 2. Without tie_break_key (None): proves that legacy argsort fails permutation invariance
    w_none_orig = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=1.0, tie_break_key=None)
    w_none_perm = ops.dampen_turnover(alpha[:, pi], top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=1.0, tie_break_key=None)
    
    # Legacy argsort picks first columns in array order:
    # In original order, it picks columns 0 and 1.
    # In permuted order [3, 2, 1, 0], it picks permuted columns 0 and 1 (original columns 3 and 2).
    assert not np.allclose(w_none_orig[:, pi], w_none_perm), "Legacy argsort unexpectedly invariant; expected order dependence!"


def test_tiebreak_key_identical_is_deterministic():
    """
    (c) When tie_break_key is identical for all stocks (flat liquidity), tie-breaking falls back
    to column index, ensuring bitwise deterministic output across calls.
    """
    T, N = 4, 5
    alpha = np.full((T, N), 0.5, dtype=np.float64)
    key_flat = np.full((T, N), 100.0, dtype=np.float64)
    
    w1 = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, tie_break_key=key_flat)
    w2 = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, tie_break_key=key_flat)
    
    assert np.array_equal(w1, w2)
