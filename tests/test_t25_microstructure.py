"""
tests/test_t25_microstructure.py - Hermetic unit tests for Vietnam Microstructure & Simulator
Migrated from run_verification.py (Test 3, 4, 5) into standalone pytest suite.
All tests use purely in-memory synthetic arrays with zero external network or parquet dependencies.
"""
import datetime
import numpy as np
import pytest
from engine import alpha_ops as ops
from engine import vn_simulator as sim


def test_turnover_dampener_reduces_turnover_under_5pct():
    """
    Test 3: Vietnam Turnover Dampener (Rank Hysteresis + EMA).
    Demonstrates that high-turnover raw alpha (> 25%/day) is suppressed to <= 5%/day.
    """
    np.random.seed(42)
    T, N = 250, 30

    raw_alpha = np.zeros((T, N))
    for t in range(1, T):
        raw_alpha[t] = 0.50 * raw_alpha[t - 1] + np.random.randn(N)

    # Naive Top-10 portfolio without hysteresis
    naive_w = np.zeros((T, N))
    for t in range(T):
        top10 = np.argsort(-raw_alpha[t])[:10]
        naive_w[t, top10] = 0.10

    naive_to = float(np.mean(0.5 * np.sum(np.abs(naive_w[1:] - naive_w[:-1]), axis=1)))
    assert naive_to > 0.25, f"Naive turnover {naive_to * 100:.2f}% should be > 25%/day"

    # Apply dampen_turnover
    dampened_w = ops.dampen_turnover(
        alpha=raw_alpha,
        top_k_in=10,
        top_k_out=18,
        ema_span=10,
        max_daily_turnover=0.045
    )

    dampened_to = float(np.mean(0.5 * np.sum(np.abs(dampened_w[1:] - dampened_w[:-1]), axis=1)))
    assert dampened_to <= 0.05, f"Dampened turnover {dampened_to * 100:.2f}% must be <= 5%/day"


def test_simulation_microstructure_costs_positive():
    """
    Subtest 5.1 & 5.2: Independent Brokerage Fee (0.15%), Sell Tax (0.10%),
    and Calendar-Day Advance Interest with 50k floor fee.
    """
    T, N = 6, 2
    raw_open = np.full((T, N), 50000.0)
    raw_close = np.full((T, N), 50000.0)
    ceiling = np.full((T, N), 53500.0)
    floor = np.full((T, N), 46500.0)
    adv20_shares = np.full((T, N), 1000000.0)

    # Day 0: Baseline 100% cash, zero trades (C2 Invariant)
    # Day 1-2: Buy & hold asset 0
    # Day 3: Sell asset 0 (settled at T+2)
    target_weights = np.zeros((T, N), dtype=np.float64)
    target_weights[1:3, 0] = 0.50  # Buy day 1-2
    target_weights[3:, 0] = 0.0    # Sell day 3

    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=target_weights,
        initial_cash=1.0e9,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010
    )

    b_fees = res["broker_fees"]
    s_tax = res["sell_tax"]

    # Day 0: No trades executed (strict 0 look-ahead)
    assert b_fees[0] == 0.0
    assert s_tax[0] == 0.0

    # Day 1: Buy only -> broker fee > 0, sell tax == 0
    assert b_fees[1] > 0.0
    assert s_tax[1] == 0.0

    # Day 3: Sell -> sell fee / sell tax = 0.15 / 0.10 = 1.50
    sell_fee_val = b_fees[3] - b_fees[2]
    sell_tax_val = s_tax[3]
    ratio_sell = sell_fee_val / max(sell_tax_val, 1e-6)
    assert abs(ratio_sell - 1.50) < 1e-2

    # Advance interest floor check on actual simulator output (asserting res["advance_fees"])
    target_weights_adv = np.zeros((T, N), dtype=np.float64)
    target_weights_adv[1:3, 0] = 1.0  # Buy 1B of asset 0
    target_weights_adv[3:, 1] = 1.0   # Sell asset 0, buy 1B of asset 1 using pending sales cash
    res_adv = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=target_weights_adv,
        initial_cash=1.0e9,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        advance_fee_rate=0.00035,
        min_advance_fee=50000.0
    )
    assert res_adv["advance_fees"][-1] >= 50000.0, f"Expected advance fee >= 50k, got {res_adv['advance_fees'][-1]}"


def test_microstructure_floor_trapped_queue():
    """
    Subtest 5.4: Floor-Trapped Order Priority Queue.
    Verifies that sell orders cannot execute during floor locks and execute when unlocked.
    """
    T, N = 7, 1
    raw_open = np.full((T, N), 50000.0)
    raw_close = np.full((T, N), 50000.0)
    ceiling = np.full((T, N), 53500.0)
    floor = np.full((T, N), 46500.0)
    adv_shares = np.full((T, N), 10000000.0)

    # Days 3, 4: locked at floor
    raw_open[3, 0] = 46500.0
    raw_open[4, 0] = 46500.0
    # Days 5, 6: unlocked
    raw_open[5, 0] = 48000.0
    raw_open[6, 0] = 48000.0

    w_flr = np.zeros((T, N), dtype=np.float64)
    w_flr[1:3, 0] = 1.0  # Buy at t=1, settles at t=3
    w_flr[3:, 0] = 0.0   # Attempt sell at t=3 (floor), t=4 (floor), t=5 (unlocked)

    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv_shares,
        target_weights=w_flr,
        initial_cash=1.0e9,
        broker_fee=0.0015,
        slippage=0.0010,
        sell_tax=0.0010
    )

    assert res["sell_tax"][1] == 0.0  # No sell yet
    assert res["sell_tax"][2] == 0.0  # Holding
    assert res["sell_tax"][3] == 0.0  # Cannot sell at floor lock t=3
    assert res["sell_tax"][4] == 0.0  # Cannot sell at floor lock t=4
    assert res["sell_tax"][5] > 0.0   # Sell executed at t=5 when unlocked


def test_corporate_action_nav_preservation():
    """
    Subtest 5.3: Dual Corporate Actions Ledger (No artificial NAV drop at ex-date).
    """
    T, N = 10, 1
    ro = np.full((T, N), 50000.0)
    rc = np.full((T, N), 50000.0)
    # Ex-date at t=3: price drops from 50k to 40k (-10k cash dividend)
    rc[3:] = 40000.0
    ro[3:] = 40000.0
    ceil = np.full((T, N), 60000.0)
    flr = np.full((T, N), 30000.0)
    adv = np.full((T, N), 10000000.0)

    w = np.ones((T, N), dtype=np.float64)

    div_cash = np.zeros((T, N), dtype=np.float64)
    div_pay = np.zeros((T, N), dtype=np.float64)
    div_cash[3, 0] = 10000.0
    div_pay[3, 0] = 7.0

    res = sim.simulate_vietnam(
        raw_open=ro,
        raw_close=rc,
        ceiling=ceil,
        floor=flr,
        adv20_shares=adv,
        target_weights=w,
        initial_cash=1.0e9,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        corp_action_div_cash=div_cash,
        corp_action_div_pay_t=div_pay
    )

    eq = res["equity"]
    nav_pre = eq[2]
    nav_ex = eq[3]
    nav_delta_pct = abs(nav_ex - nav_pre) / nav_pre
    assert nav_delta_pct < 0.01, f"Artificial NAV drop on Ex-Date: {nav_delta_pct*100:.2f}%"


def test_full_microstructure_simulation_synthetic():
    """
    Hermetic version of Test 4: Full Microstructure Backtest on synthetic price paths.
    Verifies that Real Microstructure CAGR is strictly less than Naive CAGR due to friction,
    and implementation shortfall is strictly positive.
    """
    np.random.seed(123)
    T, N = 120, 10
    prices = np.zeros((T, N))
    prices[0] = 30000.0 + np.random.uniform(-5000, 5000, size=N)
    for t in range(1, T):
        # Moderate positive drift so strategy is profitable and shortfall ratio is healthy
        ret = np.random.normal(0.0015, 0.010, size=N)
        prices[t] = prices[t - 1] * (1.0 + ret)

    raw_open = prices * (1.0 + np.random.uniform(-0.002, 0.002, size=(T, N)))
    raw_close = prices
    ceiling = raw_open * 1.07
    floor = raw_open * 0.93
    adv20_shares = np.full((T, N), 500000.0)

    # Strategy: Top-3 momentum with dampening
    raw_alpha = np.zeros((T, N))
    for t in range(20, T):
        mom = raw_close[t] / raw_close[t - 20] - 1.0
        raw_alpha[t] = mom

    w_target = ops.dampen_turnover(
        alpha=raw_alpha,
        top_k_in=2,
        top_k_out=4,
        ema_span=20,
        max_daily_turnover=0.015
    )

    # Causal lag: trade on next day open
    w_exec = np.zeros_like(w_target)
    w_exec[1:] = w_target[:-1]

    naive_eq = sim.simulate_naive(
        raw_close=raw_close,
        target_weights=w_exec,
        initial_cash=1.0e9,
        raw_open=raw_open
    )

    vn_res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=w_exec,
        initial_cash=1.0e9,
        broker_fee=0.0015,
        slippage=0.0010,
        sell_tax=0.0010,
        advance_fee_rate=0.00035,
        max_adv_participation=0.05
    )
    real_eq = vn_res["equity"]

    metrics_naive = sim.evaluate_strategy(
        equity_curve=naive_eq,
        daily_weights=w_exec,
        initial_capital=1.0e9
    )
    metrics_real = sim.evaluate_strategy(
        equity_curve=real_eq,
        daily_weights=w_exec,
        naive_equity_curve=naive_eq,
        initial_capital=1.0e9,
        realized_daily_turnover=vn_res["realized_daily_turnover"]
    )

    # Assertions
    assert metrics_real["cagr"] < metrics_naive["cagr"], "Real CAGR must be less than Naive CAGR due to friction"
    assert metrics_real["implementation_shortfall_pct"] > 0.0, "Implementation shortfall must be strictly positive"
    assert metrics_real["daily_turnover"] > 0.0
    assert not metrics_real["kill_switch_triggered"], f"Kill switch triggered: {metrics_real['kill_switch_reasons']}"
