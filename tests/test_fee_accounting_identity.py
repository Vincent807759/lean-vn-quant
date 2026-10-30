import numpy as np
import pytest
from engine import vn_simulator as sim


def test_fee_accounting_identity_zero_notional():
    """
    Conservation of Value Invariant under zero price volatility and zero slippage:
    initial_cash - equity[-1] == broker_fees[-1] + sell_tax[-1] + advance_fees[-1]
    with min_advance_notional = 0.0.
    """
    T = 15
    N = 3
    p = 100_000.0
    raw_open = np.full((T, N), p, dtype=np.float64)
    raw_close = np.full((T, N), p, dtype=np.float64)
    ceiling = np.full((T, N), p * 1.5, dtype=np.float64)
    floor = np.full((T, N), p * 0.5, dtype=np.float64)
    adv20_shares = np.full((T, N), 1e9, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    for seed in range(10):
        rng = np.random.RandomState(seed)
        init_cash = rng.uniform(1e8, 1e9)
        weights = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            w = rng.uniform(0.0, 1.0, size=N)
            weights[t] = (w / np.sum(w)) * rng.uniform(0.5, 1.5)

        res = sim.simulate_vietnam(
            raw_open=raw_open,
            raw_close=raw_close,
            ceiling=ceiling,
            floor=floor,
            adv20_shares=adv20_shares,
            target_weights=weights,
            initial_cash=init_cash,
            broker_fee=0.0015,
            slippage=0.0,
            sell_tax=0.0010,
            advance_fee_rate=0.00035,
            min_advance_fee=50000.0,
            half_spread_floor=0.0,
            daily_vol=daily_vol,
            max_adv_participation=1.0,
            penalty_rate_per_day=0.0,
            min_advance_notional=0.0
        )

        eq_loss = init_cash - res["equity"][-1]
        fees_sum = res["broker_fees"][-1] + res["sell_tax"][-1] + res["advance_fees"][-1]
        diff = abs(eq_loss - fees_sum)
        assert diff < 1.0, f"Seed {seed} failed accounting identity: eq_loss={eq_loss}, fees_sum={fees_sum}, diff={diff}"


def test_fee_accounting_identity_smart_sweep_leakage():
    """
    Verifies that under min_advance_notional = 71,428,571.0, broker fees and turnover
    are NOT charged when an advance order is aborted because shortfall < min_advance_notional.
    On unpatched code, seed 1 leaks 270,000 VND of broker fees on unexecuted orders.
    """
    T = 15
    N = 3
    p = 100_000.0
    raw_open = np.full((T, N), p, dtype=np.float64)
    raw_close = np.full((T, N), p, dtype=np.float64)
    ceiling = np.full((T, N), p * 1.5, dtype=np.float64)
    floor = np.full((T, N), p * 0.5, dtype=np.float64)
    adv20_shares = np.full((T, N), 1e9, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    for seed in range(10):
        rng = np.random.RandomState(seed)
        init_cash = rng.uniform(1e8, 1e9)
        weights = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            w = rng.uniform(0.0, 1.0, size=N)
            weights[t] = (w / np.sum(w)) * rng.uniform(0.5, 1.5)

        res = sim.simulate_vietnam(
            raw_open=raw_open,
            raw_close=raw_close,
            ceiling=ceiling,
            floor=floor,
            adv20_shares=adv20_shares,
            target_weights=weights,
            initial_cash=init_cash,
            broker_fee=0.0015,
            slippage=0.0,
            sell_tax=0.0010,
            advance_fee_rate=0.00035,
            min_advance_fee=50000.0,
            half_spread_floor=0.0,
            daily_vol=daily_vol,
            max_adv_participation=1.0,
            penalty_rate_per_day=0.0,
            min_advance_notional=71428571.0
        )

        eq_loss = init_cash - res["equity"][-1]
        fees_sum = res["broker_fees"][-1] + res["sell_tax"][-1] + res["advance_fees"][-1]
        diff = abs(eq_loss - fees_sum)
        assert diff < 1.0, (
            f"Seed {seed} leaked fees on unexecuted orders! "
            f"eq_loss={eq_loss:,.2f}, fees_sum={fees_sum:,.2f}, diff={diff:,.2f}"
        )


def test_cash_advance_no_ghost_fee_when_t1_is_zero():
    """
    Verifies that when only pending_cash_t0 (receivable from today's sale) is available
    and pending_cash_t1 == 0, advance fee is charged only once for t0, no ghost 50,000 VND
    fee is charged to t1, and cash balances never become negative.
    """
    T = 5
    N = 2
    init_cash = 100_000_000.0
    p = 50_000.0
    raw_open = np.full((T, N), p, dtype=np.float64)
    raw_close = np.full((T, N), p, dtype=np.float64)
    ceiling = np.full((T, N), p * 1.5, dtype=np.float64)
    floor = np.full((T, N), p * 0.5, dtype=np.float64)
    adv20_shares = np.full((T, N), 1e9, dtype=np.float64)
    daily_vol = np.zeros((T, N), dtype=np.float64)

    weights = np.zeros((T, N), dtype=np.float64)
    weights[1] = [1.0, 0.0]
    weights[2] = [1.0, 0.0]
    weights[3] = [0.0, 1.0]

    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=weights,
        initial_cash=init_cash,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        advance_fee_rate=0.00035,
        min_advance_fee=50000.0,
        half_spread_floor=0.0,
        daily_vol=daily_vol,
        max_adv_participation=1.0,
        penalty_rate_per_day=0.0,
        min_advance_notional=0.0
    )

    adv_fees_t3 = res["advance_fees"][3] - res["advance_fees"][2]
    assert 0.0 < adv_fees_t3 < 90000.0, (
        f"Expected single advance fee (~50k), got {adv_fees_t3:,.0f} VND"
    )
    assert np.all(res["cash_history"] >= -1e-6), "Cash history should not be negative"
