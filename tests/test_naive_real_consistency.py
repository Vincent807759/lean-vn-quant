import numpy as np
import pytest
from engine import vn_simulator as sim


def test_naive_real_deterministic_8day():
    """
    Test A (Deterministic): 8-day trace verifying that simulate_naive(..., raw_open=raw_open)
    matches simulate_vietnam under zero friction (0 fees, 0 tax, 0 slippage, 0 half_spread_floor)
    within rtol=1e-5.
    """
    T = 8
    N = 1
    init_cash = 1e12

    raw_open = np.array([[100.0], [100.0], [104.0], [108.0], [110.0], [108.0], [114.0], [117.0]], dtype=np.float64)
    raw_close = np.array([[100.0], [105.0], [108.0], [112.0], [107.0], [115.0], [118.0], [120.0]], dtype=np.float64)

    ceiling = raw_close * 1.50
    floor = raw_close * 0.50
    adv20_shares = np.full((T, N), 1e18, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    weights = np.zeros((T, N), dtype=np.float64)
    # Positions held for >= 3 sessions to satisfy T+2.5 settlement
    weights[1] = 1.0
    weights[2] = 1.0
    weights[3] = 1.0
    weights[4] = 0.0
    weights[5] = 0.0
    weights[6] = 1.0
    weights[7] = 1.0

    res_real = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=weights,
        initial_cash=init_cash,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        advance_fee_rate=0.0,
        min_advance_fee=0.0,
        half_spread_floor=0.0,
        daily_vol=daily_vol,
        max_adv_participation=1.0,
        penalty_rate_per_day=0.0
    )

    naive_new = sim.simulate_naive(
        raw_close=raw_close,
        target_weights=weights,
        initial_cash=init_cash,
        raw_open=raw_open
    )

    max_rel_diff = np.max(np.abs(res_real["equity"] - naive_new) / naive_new)
    assert max_rel_diff < 1e-5, f"Real and Naive diverged! Max rel diff: {max_rel_diff:.2e}"


def test_naive_real_random_seeds():
    """
    Test B (Randomized): 20 random seeds across N in {1, 3}, holding positions >= 3 sessions
    under zero friction with daily_vol=0, verifying Real and Naive match within rtol=1e-5.
    """
    init_cash = 1e12
    T = 16

    for N in [1, 3]:
        for seed in range(20):
            rng = np.random.RandomState(seed)

            base_price = rng.uniform(20000.0, 50000.0, size=(1, N))
            daily_ret = rng.normal(0.001, 0.0005, size=(T, 1))
            raw_close = np.cumprod(1.0 + daily_ret, axis=0) * base_price
            raw_open = np.zeros_like(raw_close)
            raw_open[0] = raw_close[0]
            raw_open[1:] = raw_close[:-1] * (1.0 + rng.normal(0.0, 0.0005, size=(T - 1, 1)))

            ceiling = raw_close * 1.50
            floor = raw_close * 0.50
            adv20_shares = np.full((T, N), 1e18, dtype=np.float64)
            daily_vol = np.zeros((T, N), dtype=np.float64)

            # Generate weights with blocks of >= 3 sessions
            weights = np.zeros((T, N), dtype=np.float64)
            for block_start in range(1, T, 4):
                block_end = min(T, block_start + 4)
                w_raw = rng.uniform(0.1, 1.0, size=N)
                w_block = w_raw / np.sum(w_raw)
                weights[block_start:block_end] = w_block

            res_real = sim.simulate_vietnam(
                raw_open=raw_open,
                raw_close=raw_close,
                ceiling=ceiling,
                floor=floor,
                adv20_shares=adv20_shares,
                target_weights=weights,
                initial_cash=init_cash,
                broker_fee=0.0,
                slippage=0.0,
                sell_tax=0.0,
                advance_fee_rate=0.0,
                min_advance_fee=0.0,
                half_spread_floor=0.0,
                daily_vol=daily_vol,
                max_adv_participation=1.0,
                penalty_rate_per_day=0.0
            )

            naive_eq = sim.simulate_naive(
                raw_close=raw_close,
                target_weights=weights,
                initial_cash=init_cash,
                raw_open=raw_open
            )

            max_rel_diff = np.max(np.abs(res_real["equity"] - naive_eq) / naive_eq)
            assert max_rel_diff < 1e-5, (
                f"Seed {seed} (N={N}) failed with max relative difference {max_rel_diff:.2e}"
            )


def test_naive_real_t0_weights():
    """
    Test C (Edge Case): Non-zero weights at t=0 are ignored by both Real (initialization session)
    and Naive (first rebalance occurs at t=1 Open).
    """
    T = 8
    N = 1
    init_cash = 1e12

    raw_open = np.array([[100.0], [100.0], [104.0], [108.0], [110.0], [108.0], [114.0], [117.0]], dtype=np.float64)
    raw_close = np.array([[100.0], [105.0], [108.0], [112.0], [107.0], [115.0], [118.0], [120.0]], dtype=np.float64)

    ceiling = raw_close * 1.50
    floor = raw_close * 0.50
    adv20_shares = np.full((T, N), 1e18, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    # Non-zero weights at t=0
    weights = np.zeros((T, N), dtype=np.float64)
    weights[0] = 0.5  # Non-zero t=0 weight must be ignored
    weights[1:4] = 1.0
    weights[4:6] = 0.0
    weights[6:8] = 1.0

    res_real = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=weights,
        initial_cash=init_cash,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        advance_fee_rate=0.0,
        min_advance_fee=0.0,
        half_spread_floor=0.0,
        daily_vol=daily_vol,
        max_adv_participation=1.0,
        penalty_rate_per_day=0.0
    )

    naive_eq = sim.simulate_naive(
        raw_close=raw_close,
        target_weights=weights,
        initial_cash=init_cash,
        raw_open=raw_open
    )

    # Real and Naive equity at t=0 must be exactly initial_cash
    assert res_real["equity"][0] == init_cash
    assert naive_eq[0] == init_cash

    max_rel_diff = np.max(np.abs(res_real["equity"] - naive_eq) / naive_eq)
    assert max_rel_diff < 1e-5, (
        f"Real and Naive diverged on non-zero t=0 weights! Max rel diff: {max_rel_diff:.2e}"
    )


def test_t25_lock_diverges_from_naive_on_t1_flip():
    """
    Verifies that T+2.5 settlement lock prevents flipping positions at T+1 (t=2) in Real simulation,
    causing Real and Naive to diverge when weights flip at T+1.
    """
    T = 6
    N = 2
    init_cash = 1e9

    raw_open = np.array([
        [100.0, 100.0],
        [100.0, 100.0],
        [100.0, 120.0],
        [70.0, 150.0],
        [70.0, 150.0],
        [70.0, 150.0],
    ], dtype=np.float64)

    raw_close = np.array([
        [100.0, 100.0],
        [100.0, 100.0],
        [100.0, 130.0],
        [60.0, 160.0],
        [60.0, 160.0],
        [60.0, 160.0],
    ], dtype=np.float64)

    ceiling = raw_close * 1.50
    floor = raw_close * 0.50
    adv20_shares = np.full((T, N), 1e18, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    weights = np.zeros((T, N), dtype=np.float64)
    weights[1] = [1.0, 0.0]
    weights[2] = [0.0, 1.0]
    weights[3] = [0.0, 1.0]
    weights[4] = [0.0, 1.0]
    weights[5] = [0.0, 1.0]

    res_real = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=weights,
        initial_cash=init_cash,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        advance_fee_rate=0.0,
        min_advance_fee=0.0,
        half_spread_floor=0.0,
        daily_vol=daily_vol,
        max_adv_participation=1.0,
        penalty_rate_per_day=0.0
    )

    naive_eq = sim.simulate_naive(
        raw_close=raw_close,
        target_weights=weights,
        initial_cash=init_cash,
        raw_open=raw_open
    )

    assert not np.allclose(res_real["equity"], naive_eq), (
        "Real and Naive should diverge when flipping positions at T+1 due to T+2.5 lock!"
    )
