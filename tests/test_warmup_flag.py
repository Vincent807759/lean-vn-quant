import numpy as np
import pytest
from engine import vn_simulator as sim
from engine import data_pipeline as dp


def test_synthetic_warmup_zeroes_early_positions():
    """
    Test that when warmup N > 0, the strategy holds zero positions
    and maintains exact initial cash for sessions 0..N-1.
    """
    df = dp.generate_fallback_data(["ACB", "FPT"], start_date="2023-01-01", end_date="2023-03-01", seed=42)
    raw_close, _, _ = dp.pivot_to_matrix(df, "raw_close")
    raw_open, _, _ = dp.pivot_to_matrix(df, "raw_open")
    T, N = raw_close.shape
    
    # Target weights with warmup = 5
    WARMUP = 5
    w_target = np.full((T, N), 0.5)
    w_target[:WARMUP] = 0.0
    
    # Execution weights (lagged by 1 session)
    w_exec = np.zeros_like(w_target)
    w_exec[1:] = w_target[:-1]
    
    init_cash = 1.0e9
    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=np.full((T, N), 1.0e6),
        floor=np.zeros((T, N)),
        adv20_shares=np.full((T, N), 1.0e6),
        target_weights=w_exec,
        initial_cash=init_cash,
        broker_fee=0.0015,
        sell_tax=0.0010
    )
    
    equity = res["equity"]
    # During sessions 0..WARMUP, w_exec is 0.0, so equity must be exactly initial cash
    for t in range(WARMUP + 1):
        assert np.isclose(equity[t], init_cash, atol=1e-6), (
            f"At session t={t}, expected equity {init_cash}, got {equity[t]}"
        )
    
    # Starting from t=WARMUP+1, positions are entered, so equity begins fluctuating
    assert not np.isclose(equity[WARMUP + 2], init_cash, atol=1e-6), (
        "Positions should be active after warmup period"
    )


def test_synthetic_warmup_zero_preserves_immediate_trading():
    """
    Test that when warmup = 0, trading initiates immediately on the first active execution session.
    """
    df = dp.generate_fallback_data(["ACB", "FPT"], start_date="2023-01-01", end_date="2023-03-01", seed=42)
    raw_close, _, _ = dp.pivot_to_matrix(df, "raw_close")
    raw_open, _, _ = dp.pivot_to_matrix(df, "raw_open")
    T, N = raw_close.shape
    
    w_target = np.full((T, N), 0.5)
    w_exec = np.zeros_like(w_target)
    w_exec[1:] = w_target[:-1]
    
    init_cash = 1.0e9
    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=np.full((T, N), 1.0e6),
        floor=np.zeros((T, N)),
        adv20_shares=np.full((T, N), 1.0e6),
        target_weights=w_exec,
        initial_cash=init_cash,
        broker_fee=0.0015,
        sell_tax=0.0010
    )
    
    equity = res["equity"]
    # Trading occurs on t=1 (w_exec[1] = 0.5)
    assert not np.isclose(equity[1], init_cash, atol=1e-4), (
        "Trading should occur immediately at session t=1 when warmup = 0"
    )
