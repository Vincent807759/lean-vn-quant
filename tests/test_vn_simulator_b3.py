"""
tests/test_vn_simulator_b3.py - Unit tests for B3, S4, T5, T13
"""
import numpy as np
import pytest
from engine.vn_simulator import (
    simulate_vietnam,
    sanitize_market_matrices
)

def create_synthetic_sim_data(T=30, N=3):
    np.random.seed(42)
    raw_close = np.full((T, N), 25000.0)
    for t in range(1, T):
        raw_close[t] = raw_close[t - 1] * (1.0 + np.random.normal(0, 0.01, N))
    raw_open = raw_close * (1.0 + np.random.normal(0, 0.005, (T, N)))
    ceiling = raw_close * 1.07
    floor = raw_close * 0.93
    adv20 = np.full((T, N), 1_000_000.0)
    target_weights = np.full((T, N), 1.0 / N)
    return raw_open, raw_close, ceiling, floor, adv20, target_weights

def test_b3_nan_in_zero_weight_asset_does_not_infect_portfolio():
    raw_open, raw_close, ceiling, floor, adv20, target_weights = create_synthetic_sim_data(T=20, N=3)
    
    # Asset 2 has 0 weight throughout
    target_weights[:, 2] = 0.0
    target_weights[:, 0] = 0.5
    target_weights[:, 1] = 0.5
    
    # Introduce NaNs into asset 2
    raw_close[5:10, 2] = np.nan
    raw_open[5:10, 2] = np.nan
    
    # Run simulation with sanitize_inputs=True
    res = simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20,
        target_weights=target_weights,
        initial_cash=1.0e9,
        sanitize_inputs=True
    )
    
    # Portfolio equity must be 100% finite
    equity = res["equity"]
    assert np.all(np.isfinite(equity))
    assert equity[-1] > 0.0

def test_b3_floating_point_error_raised_when_non_finite():
    raw_open, raw_close, ceiling, floor, adv20, target_weights = create_synthetic_sim_data(T=20, N=2)
    
    # Intentionally bypass sanitization and introduce NaN into held asset
    raw_close[5:, 0] = np.nan
    raw_open[5:, 0] = np.nan
    
    with pytest.raises(FloatingPointError) as excinfo:
        simulate_vietnam(
            raw_open=raw_open,
            raw_close=raw_close,
            ceiling=ceiling,
            floor=floor,
            adv20_shares=adv20,
            target_weights=target_weights,
            initial_cash=1.0e9,
            sanitize_inputs=False
        )
    assert "Simulator Invariant B3 Violated" in str(excinfo.value)

def test_s4_and_t13_halt_haircut_applied_starting_from_halt_session():
    T = 20
    N = 2
    raw_open, raw_close, ceiling, floor, adv20, target_weights = create_synthetic_sim_data(T=T, N=N)
    
    # Asset 1 trades until day 5, then is missing/halted from day 6 to day 19 (> 5 sessions)
    raw_close[6:, 1] = np.nan
    raw_open[6:, 1] = np.nan
    
    # Sanitize with 30% halt haircut
    h = 0.30
    c_open, c_close, c_ceil, c_floor, c_adv, c_w, c_pit = sanitize_market_matrices(
        raw_open, raw_close, ceiling, floor, adv20, target_weights,
        max_missing_sessions=5,
        halt_haircut=h
    )
    
    # Day 6 to 10 is missing <= 5: forward filled at last price (no haircut yet)
    last_price = raw_close[5, 1]
    assert np.isclose(c_close[6, 1], last_price)
    assert np.isclose(c_close[10, 1], last_price)
    
    # Day 11 is missing session 6 (> 5): haircut applied!
    discounted_price = last_price * (1.0 - h)
    assert np.isclose(c_close[11, 1], discounted_price)
    assert np.isclose(c_close[19, 1], discounted_price)
    assert c_pit[11, 1] == False
    assert c_w[11, 1] == 0.0
