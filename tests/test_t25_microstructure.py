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
        corp_action_div_pay_t=div_pay,
        dividend_tax_rate=0.0,
        warn_unlagged=False
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


def test_cash_dividend_5pct_pit_tax():
    """
    Verifies that when dividend_tax_rate=0.05 (default), personal income tax (5%)
    is deducted, and the investor receives exactly 95% of the gross cash dividend.
    """
    T, N = 10, 1
    ro = np.full((T, N), 50000.0)
    rc = np.full((T, N), 50000.0)
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
        corp_action_div_pay_t=div_pay,
        daily_vol=np.zeros((T, N)),
        dividend_tax_rate=0.05,
        half_spread_floor=0.0,
        warn_unlagged=False
    )

    # 1.0e9 / 50k = 20,000 shares * 10,000 * 0.95 = 190,000,000 VND
    expected_net_div = 20000.0 * 10000.0 * 0.95
    assert np.isclose(res["dividend_cash_receivable"][3], expected_net_div), (
        f"Expected net dividend {expected_net_div:,.0f}, got {res['dividend_cash_receivable'][3]:,.0f}"
    )



def test_strict_t25_ato_lock_earliest_sell_day():
    """
    Cover earliest-sell day under default T+2.5 vs strict T+2.5 ATO lock:
    Scenario:
    - 1 ticker, T=9
    - Open/Close price increases 10%/day from t=2 onwards.
    - Buy at t=2 (w[2, 0] = 1.0), then target weight drops to 0 for all t >= 3.
    - Zero friction: broker_fee=0, slippage=0, sell_tax=0, half_spread_floor=0.
    
    Daily Trace Table (equity[t] / equity[t-1]):
    | t | Price | Target W | Default Lock (1.1 = held, 1.0 = sold) | Strict Lock (1.1 = held, 1.0 = sold) |
    |---|-------|----------|----------------------------------------|--------------------------------------|
    | 0 | 50,000| 0.0      | 1.0000 (Cash 100%)                     | 1.0000 (Cash 100%)                   |
    | 1 | 50,000| 0.0      | 1.0000 (Cash 100%)                     | 1.0000 (Cash 100%)                   |
    | 2 | 55,000| 1.0      | 1.1000 (Buy matched at Open, holds)   | 1.1000 (Buy matched at Open, holds)  |
    | 3 | 60,500| 0.0      | 1.1000 (T+1 locked, cannot sell)       | 1.1000 (T+1 locked, cannot sell)     |
    | 4 | 66,550| 0.0      | 1.0000 (T+2 ATO unlocked, SOLD at Open)| 1.1000 (T+2 ATO locked, still holds) |
    | 5 | 73,205| 0.0      | 1.0000 (Cash 100%, already sold)       | 1.0000 (T+3 ATO unlocked, SOLD Open) |
    | 6 | 80,525| 0.0      | 1.0000 (Cash 100%)                     | 1.0000 (Cash 100%)                   |
    | 7 | 88,578| 0.0      | 1.0000 (Cash 100%)                     | 1.0000 (Cash 100%)                   |
    | 8 | 97,436| 0.0      | 1.0000 (Cash 100%)                     | 1.0000 (Cash 100%)                   |
    
    Assertion:
    - Default lock (strict_t25_ato_lock=False): Earliest day ratio equals 1.0 is t=4.
    - Strict lock (strict_t25_ato_lock=True): Earliest day ratio equals 1.0 is t=5.
    """
    T, N = 9, 1
    op = np.zeros(T)
    cl = np.zeros(T)
    p = 50000.0
    for t in range(T):
        op[t] = p
        cl[t] = op[t] * (1.1 if t >= 2 else 1.0)
        p = cl[t]
        
    w = np.zeros((T, 1))
    w[2, 0] = 1.0
    big = 1e15
    
    # 1. Default lock: sellable at T+2 ATO (t=4)
    r_def = sim.simulate_vietnam(
        raw_open=op.reshape(T, 1),
        raw_close=cl.reshape(T, 1),
        ceiling=np.full((T, 1), big),
        floor=np.zeros((T, 1)),
        adv20_shares=np.full((T, 1), big),
        target_weights=w,
        initial_cash=1e12,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        min_advance_fee=0.0,
        max_adv_participation=1.0,
        daily_vol=np.zeros((T, 1)),
        half_spread_floor=0.0,
        strict_t25_ato_lock=False
    )
    eq_def = r_def["equity"]
    ratios_def = {t: round(float(eq_def[t] / eq_def[t-1]), 4) for t in range(1, T)}
    earliest_1_def = min(t for t in range(1, T) if ratios_def[t] == 1.0 and t >= 3)
    assert earliest_1_def == 4, f"Expected earliest sell day t=4 for default, got t={earliest_1_def}"
    assert ratios_def[2] == 1.1
    assert ratios_def[3] == 1.1
    assert ratios_def[4] == 1.0
    
    # 2. Strict lock: sellable at T+3 ATO (t=5)
    r_str = sim.simulate_vietnam(
        raw_open=op.reshape(T, 1),
        raw_close=cl.reshape(T, 1),
        ceiling=np.full((T, 1), big),
        floor=np.zeros((T, 1)),
        adv20_shares=np.full((T, 1), big),
        target_weights=w,
        initial_cash=1e12,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        min_advance_fee=0.0,
        max_adv_participation=1.0,
        daily_vol=np.zeros((T, 1)),
        half_spread_floor=0.0,
        strict_t25_ato_lock=True
    )
    eq_str = r_str["equity"]
    ratios_str = {t: round(float(eq_str[t] / eq_str[t-1]), 4) for t in range(1, T)}
    earliest_1_str = min(t for t in range(1, T) if ratios_str[t] == 1.0 and t >= 3)
    assert earliest_1_str == 5, f"Expected earliest sell day t=5 for strict, got t={earliest_1_str}"
    assert ratios_str[2] == 1.1
    assert ratios_str[3] == 1.1
    assert ratios_str[4] == 1.1
    assert ratios_str[5] == 1.0


def test_exec_price_clamp_and_hose_tick_rounding():
    # 1. Floor trapped exit 5 sessions with 35% penalty
    T_syn, N_syn = 10, 1
    ro_syn = np.full((T_syn, N_syn), 50000.0)
    rc_syn = np.full((T_syn, N_syn), 50000.0)
    ceil_syn, flr_syn = sim.compute_bands_and_ticks(ro_syn, "HOSE")
    for t_trap in range(3, 8):
        ro_syn[t_trap, 0] = flr_syn[t_trap, 0]
        rc_syn[t_trap, 0] = flr_syn[t_trap, 0]
    adv_syn = np.full((T_syn, N_syn), 10_000_000.0)
    w_syn = np.zeros((T_syn, N_syn))
    w_syn[1:3, 0] = 0.5
    dates_syn = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(T_syn)]

    # Raw execution (clamp_and_round_exec_price=False) breaches floor on exit
    r_raw = sim.simulate_vietnam(
        ro_syn, rc_syn, ceil_syn, flr_syn, adv_syn, w_syn,
        dates=dates_syn, penalty_rate_per_day=0.07, clamp_and_round_exec_price=False,
        warn_unlagged=False
    )
    # Single exit of 10,000 sh at ~32,420 (50,000 x (1 - 0.35 penalty - slippage)), below the 46,500 floor.
    # Pre-fix value (~818M) included a phantom second sale + buy-back from the trapped-sell double count.
    assert r_raw["equity"][-1] < 830_000_000.0

    # Clamped execution preserves floor price (46,500 VND)
    r_clamped = sim.simulate_vietnam(
        ro_syn, rc_syn, ceil_syn, flr_syn, adv_syn, w_syn,
        dates=dates_syn, penalty_rate_per_day=0.07, clamp_and_round_exec_price=True,
        round_exec_adverse=False, warn_unlagged=False
    )
    # 1e9 - 501M buy (10,000 x 50,100) - 751.5k fee + 465M sell at floor - 697.5k fee - 465k tax
    assert r_clamped["equity"][-1] == 962_086_000.0
    assert np.all(r_clamped["cash_history"] >= -1e-6)

    # Adverse rounding mode
    r_adv = sim.simulate_vietnam(
        ro_syn, rc_syn, ceil_syn, flr_syn, adv_syn, w_syn,
        dates=dates_syn, penalty_rate_per_day=0.07, clamp_and_round_exec_price=True,
        round_exec_adverse=True, warn_unlagged=False
    )
    assert r_adv["equity"][-1] == 962_086_000.0
    assert np.all(r_adv["cash_history"] >= -1e-6)

    # 2. Test off-tick boundaries clamping and rounding
    ro_off = np.array([[30000.0], [30870.0], [26840.0], [30000.0]], dtype=np.float64)
    rc_off = ro_off.copy()
    ceil_off = np.full((4, 1), 30874.85)  # non-tick ceiling
    flr_off = np.full((4, 1), 26835.15)   # non-tick floor
    adv_off = np.full((4, 1), 1000.0)     # small ADV triggers dynamic slippage
    w_off = np.zeros((4, 1))
    w_off[1, 0] = 0.8
    w_off[2, 0] = 0.0

    r_off = sim.simulate_vietnam(
        ro_off, rc_off, ceil_off, flr_off, adv_off, w_off,
        clamp_and_round_exec_price=True, round_exec_adverse=False,
        warn_unlagged=False
    )
    assert np.all(r_off["cash_history"] >= -1e-6)

    r_off_adv = sim.simulate_vietnam(
        ro_off, rc_off, ceil_off, flr_off, adv_off, w_off,
        clamp_and_round_exec_price=True, round_exec_adverse=True,
        warn_unlagged=False
    )
    assert np.all(r_off_adv["cash_history"] >= -1e-6)

    # 3. Test Step D max_budget and cash_avail scaling branches with while-loop backoff
    # Scenario A: Off-tick slippage rounding in scaling branch (raw 50,025 -> 50,000 in M2 vs 50,100 in M3)
    ro_a = np.full((3, 1), 50000.0)
    rc_a = np.full((3, 1), 50000.0)
    c_a, f_a = sim.compute_bands_and_ticks(ro_a, "HOSE")
    adv_a = np.full((3, 1), 10_000_000.0)
    w_a = np.zeros((3, 1))
    w_a[1, 0] = 1.0

    r_scen_a_m2 = sim.simulate_vietnam(
        ro_a, rc_a, c_a, f_a, adv_a, w_a,
        initial_cash=60_000_000.0, broker_fee=0.0015, slippage=0.0005,
        half_spread_floor=0.0, clamp_and_round_exec_price=True,
        round_exec_adverse=False, warn_unlagged=False
    )
    r_scen_a_m3 = sim.simulate_vietnam(
        ro_a, rc_a, c_a, f_a, adv_a, w_a,
        initial_cash=60_000_000.0, broker_fee=0.0015, slippage=0.0005,
        half_spread_floor=0.0, clamp_and_round_exec_price=True,
        round_exec_adverse=True, warn_unlagged=False
    )
    exec_p_m2 = r_scen_a_m2["broker_fees"][1] / (0.0015 * 1100.0)
    exec_p_m3 = r_scen_a_m3["broker_fees"][1] / (0.0015 * 1100.0)
    assert np.isclose(exec_p_m2, 50000.0)
    assert np.isclose(exec_p_m3, 50100.0)
    assert round(exec_p_m2) % 100 == 0
    assert round(exec_p_m3) % 100 == 0
    assert np.all(r_scen_a_m2["cash_history"] >= -1e-6)
    assert np.all(r_scen_a_m3["cash_history"] >= -1e-6)

    # Scenario B: Multiple-lot backoff in while-loop under max_budget (Step D branch 3)
    # p_open is below floor, so exec_price is clamped upwards to flr_p
    ro_b = np.full((5, 2), 50000.0)
    rc_b = np.full((5, 2), 50000.0)
    c_b, f_b = sim.compute_bands_and_ticks(ro_b, "HOSE")
    adv_b = np.full((5, 2), 10_000_000.0)
    w_b = np.zeros((5, 2))
    w_b[1, 0] = 0.5
    w_b[3, 0] = 0.0
    w_b[3, 1] = 1.0
    ro_b[3, 1] = 45000.0
    rc_b[3, 1] = 45000.0
    f_b[3, 1] = 46500.0

    r_scen_b = sim.simulate_vietnam(
        ro_b, rc_b, c_b, f_b, adv_b, w_b,
        initial_cash=100_000_000.0, broker_fee=0.0015, slippage=0.0,
        sell_tax=0.001, advance_fee_rate=0.00035,
        clamp_and_round_exec_price=True, round_exec_adverse=False,
        warn_unlagged=False
    )
    assert np.all(r_scen_b["cash_history"] >= -1e-6)


def test_sanitize_market_matrices_ffill_respects_hose_ticks():
    ro_na = np.array([[28855.0], [np.nan]], dtype=np.float64)
    rc_na = np.array([[28855.0], [np.nan]], dtype=np.float64)
    c_na, f_na = sim.compute_bands_and_ticks(np.array([[28855.0], [28855.0]]), "HOSE")
    _, _, c_clean, f_clean, _, _, _ = sim.sanitize_market_matrices(
        ro_na, rc_na, c_na, f_na, np.full((2, 1), 1e6), np.zeros((2, 1))
    )
    assert c_clean[1, 0] == 30850.0
    assert f_clean[1, 0] == 26850.0
