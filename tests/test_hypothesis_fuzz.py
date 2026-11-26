"""
tests/test_hypothesis_fuzz.py - Property-Based Fuzzing Suite (Hypothesis) for Lean VN-Quant Engine (P1)

Targets:
- engine/vn_simulator.py (HOSE/HNX/UPCOM ±7%/±10%/±15% bands, T+2.5 & T+3 settlement,
  liquidity traps, floor-trapped queues, odd fees/taxes, advance cash borrowing, DSR, Newey-West)
- engine/alpha_ops.py (Cross-sectional & rolling time-series operators, turnover dampener,
  QR / Ridge / Löwdin orthogonalization, Hawkes/OU decay, Van der Waerden, LOB market impact)
- engine/ml_alpha_combiner.py (Purged Walk-Forward CV with T+3 embargo, factor orthogonalization)
- engine/data_pipeline.py (HOSE tick rounding, price limits, PIT fundamental alignment)
"""

import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import warnings

from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np
import polars as pl
import pytest

from engine import alpha_ops as ops
from engine import data_pipeline as dp
from engine import ml_alpha_combiner as mlc
from engine import vn_simulator as sim


# ==============================================================================
# 1. HYPOTHESIS COMPOSITE STRATEGIES FOR VIETNAMESE MARKET MICROSTRUCTURE
# ==============================================================================

@st.composite
def hose_market_scenario(
    draw,
    min_t: int = 8,
    max_t: int = 30,
    min_n: int = 1,
    max_n: int = 5,
    allow_limit_locks: bool = True,
    allow_zero_adv: bool = True,
):
    """
    Generates realistic and adversarial HOSE market scenarios:
    - Price trajectories across all 3 HOSE tick tiers (<10k, 10k-50k, >=50k VND)
    - Frequent ceiling locks (+7%) and floor traps (-7%)
    - Liquidity squeezes (tiny or zero ADV20)
    - Random target weights, T+2.5 / T+3 regimes, weekend calendar days, and odd fee/tax rates
    """
    T = draw(st.integers(min_value=min_t, max_value=max_t))
    N = draw(st.integers(min_value=min_n, max_value=max_n))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # Initial reference prices spanning <10k, 10k..50k, >50k boundaries
    init_prices = rng.uniform(5_000.0, 120_000.0, size=N)
    raw_open = np.zeros((T, N), dtype=np.float64)
    raw_close = np.zeros((T, N), dtype=np.float64)
    ceiling = np.zeros((T, N), dtype=np.float64)
    floor = np.zeros((T, N), dtype=np.float64)

    cur_ref = np.array([dp.round_tick_hose(float(p)) for p in init_prices], dtype=np.float64)

    limit_lock_prob = draw(st.floats(min_value=0.05, max_value=0.45)) if allow_limit_locks else 0.0

    for t in range(T):
        c_row, f_row = sim.compute_bands_and_ticks(cur_ref, exchange="HOSE")
        ceiling[t] = c_row
        floor[t] = f_row

        for j in range(N):
            u = rng.random()
            if allow_limit_locks and u < limit_lock_prob * 0.5:
                # Ceiling locked (+7%)
                o_p = c_row[j]
                c_p = c_row[j]
            elif allow_limit_locks and u < limit_lock_prob:
                # Floor trapped (-7%)
                o_p = f_row[j]
                c_p = f_row[j]
            elif u < limit_lock_prob + 0.10:
                # Flat session (zero volatility)
                o_p = cur_ref[j]
                c_p = cur_ref[j]
            else:
                # Normal intraday move clamped to [floor, ceiling] and rounded to HOSE tick
                o_raw = cur_ref[j] * (1.0 + rng.uniform(-0.065, 0.065))
                o_p = dp.round_tick_hose(float(np.clip(o_raw, f_row[j], c_row[j])))
                o_p = float(np.clip(o_p, f_row[j], c_row[j]))

                c_raw = o_p * (1.0 + rng.uniform(-0.04, 0.04))
                c_p = dp.round_tick_hose(float(np.clip(c_raw, f_row[j], c_row[j])))
                c_p = float(np.clip(c_p, f_row[j], c_row[j]))

            raw_open[t, j] = o_p
            raw_close[t, j] = c_p
        cur_ref = raw_close[t].copy()

    # ADV20 shares: mix of liquid, illiquid (100-500 shares), and zero liquidity
    adv20_shares = rng.uniform(500.0, 2_000_000.0, size=(T, N))
    if allow_zero_adv:
        zero_mask = rng.random((T, N)) < 0.10
        adv20_shares[zero_mask] = 0.0
        tight_mask = rng.random((T, N)) < 0.25
        adv20_shares[tight_mask] = rng.uniform(100.0, 3_500.0, size=int(np.sum(tight_mask)))

    # Target weights (strictly causal: row 0 is 0.0)
    target_weights = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        if rng.random() < 0.15:
            # All cash session
            target_weights[t] = 0.0
        else:
            w = rng.uniform(0.0, 1.0, size=N)
            w[rng.random(N) < 0.3] = 0.0
            s = float(np.sum(w))
            if s > 1e-9:
                target_weights[t] = (w / s) * rng.uniform(0.2, 1.0)

    # Odd fees, taxes, and microstructure parameters
    initial_cash = draw(st.floats(min_value=10_000_000.0, max_value=2_000_000_000.0))
    broker_fee = draw(st.floats(min_value=0.0, max_value=0.005))
    slippage = draw(st.floats(min_value=0.0, max_value=0.01))
    sell_tax = draw(st.floats(min_value=0.0, max_value=0.005))
    advance_fee_rate = draw(st.floats(min_value=0.0001, max_value=0.001))
    min_advance_fee = draw(st.floats(min_value=0.0, max_value=100_000.0))
    max_adv_participation = draw(st.floats(min_value=0.01, max_value=0.20))
    penalty_rate_per_day = draw(st.floats(min_value=0.0, max_value=0.05))
    strict_t25 = draw(st.booleans())
    clamp_exec = draw(st.booleans())
    round_adverse = draw(st.booleans())

    cal_t1 = rng.choice([1.0, 1.0, 1.0, 3.0, 4.0], size=T).astype(np.float64)
    cal_t2 = (cal_t1 + rng.choice([1.0, 2.0, 3.0], size=T)).astype(np.float64)

    return {
        "T": T,
        "N": N,
        "raw_open": raw_open,
        "raw_close": raw_close,
        "ceiling": ceiling,
        "floor": floor,
        "adv20_shares": adv20_shares,
        "target_weights": target_weights,
        "initial_cash": initial_cash,
        "broker_fee": broker_fee,
        "slippage": slippage,
        "sell_tax": sell_tax,
        "advance_fee_rate": advance_fee_rate,
        "min_advance_fee": min_advance_fee,
        "max_adv_participation": max_adv_participation,
        "penalty_rate_per_day": penalty_rate_per_day,
        "strict_t25_ato_lock": strict_t25,
        "clamp_and_round_exec_price": clamp_exec,
        "round_exec_adverse": round_adverse,
        "calendar_days_t1": cal_t1,
        "calendar_days_t2": cal_t2,
    }


# ==============================================================================
# 2. PROPERTY-BASED TESTS: PRICE BANDS (±7%), TICKS & EXECUTION CLAMPING
# ==============================================================================

@settings(max_examples=300, deadline=None)
@given(
    ref_prices=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=40),
        elements=st.integers(min_value=100, max_value=50_000).map(lambda k: float(k * 10)),
    )
)
def test_fuzz_hose_bands_and_ticks_invariants(ref_prices: np.ndarray):
    """
    Property: For any tick-aligned reference price >= 1,000 VND on HOSE:
    1. floor <= ref_price <= ceiling
    2. ceiling <= ref_price * 1.07 + 1e-7 and floor >= ref_price * 0.93 - 1e-7
    3. Both ceiling and floor are exact multiples of their HOSE tick tier (10, 50, 100 VND)
    4. Vectorized sim.compute_bands_and_ticks matches scalar dp.compute_hose_limits
    """
    ceil, flr = sim.compute_bands_and_ticks(ref_prices, exchange="HOSE")

    assert np.all(np.isfinite(ceil)) and np.all(np.isfinite(flr))
    assert np.all(flr <= ref_prices + 1e-9)
    assert np.all(ref_prices <= ceil + 1e-9)
    assert np.all(ceil <= ref_prices * 1.07 + 1e-7)
    assert np.all(flr >= ref_prices * 0.93 - 1e-7)

    for r_p, c_p, f_p in zip(ref_prices, ceil, flr):
        c_tk = sim._get_hose_tick_size(float(c_p))
        f_tk = sim._get_hose_tick_size(float(f_p))
        assert float(c_p) % c_tk == pytest.approx(0.0, abs=1e-6)
        assert float(f_p) % f_tk == pytest.approx(0.0, abs=1e-6)

        dp_c, dp_f = dp.compute_hose_limits(float(r_p))
        assert float(c_p) == pytest.approx(dp_c, abs=1e-9)
        assert float(f_p) == pytest.approx(dp_f, abs=1e-9)


@settings(max_examples=300, deadline=None)
@given(
    ref_price=st.integers(min_value=100, max_value=25_000).map(lambda k: float(k * 10)),
    slip_mult=st.floats(min_value=0.80, max_value=1.20, allow_nan=False, allow_infinity=False),
    side=st.sampled_from([-1, 1]),
    round_adverse=st.booleans(),
)
def test_fuzz_finalize_exec_price_clamping_and_tick_alignment(
    ref_price: float,
    slip_mult: float,
    side: int,
    round_adverse: bool,
):
    """
    Property: _finalize_exec_price with clamp_and_round=True must always:
    1. Clamp execution price strictly inside [floor, ceiling]
    2. Align execution price to a valid HOSE tick multiple
    3. Make round_adverse=True at least as costly as round_adverse=False (BUY >= nearest, SELL <= nearest)
    4. Match pure Python .py_func bit-for-bit
    """
    ceil_arr, flr_arr = sim.compute_bands_and_ticks(np.array([ref_price]), exchange="HOSE")
    ceil_p = float(ceil_arr[0])
    flr_p = float(flr_arr[0])
    raw_exec = ref_price * slip_mult

    p_near = sim._finalize_exec_price(raw_exec, side, flr_p, ceil_p, True, False)
    p_adv = sim._finalize_exec_price(raw_exec, side, flr_p, ceil_p, True, True)
    p_py = sim._finalize_exec_price.py_func(raw_exec, side, flr_p, ceil_p, True, round_adverse)
    p_jit = sim._finalize_exec_price(raw_exec, side, flr_p, ceil_p, True, round_adverse)

    assert p_jit == p_py
    assert flr_p - 1e-9 <= p_jit <= ceil_p + 1e-9
    tk = sim._get_hose_tick_size(p_jit)
    assert p_jit % tk == pytest.approx(0.0, abs=1e-6)

    if side > 0:
        assert p_adv >= p_near - 1e-9
    else:
        assert p_adv <= p_near + 1e-9


@settings(max_examples=300, deadline=None)
@given(
    pending=st.floats(min_value=0.0, max_value=5_000_000_000.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.0001, max_value=0.005, allow_nan=False, allow_infinity=False),
    days=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_fee=st.floats(min_value=0.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
)
def test_fuzz_borrow_cap_never_exceeds_pending_receivable(
    pending: float,
    rate: float,
    days: float,
    min_fee: float,
):
    """
    Property: Any advance cash amount `take` in (0, _calc_borrow_cap] plus its advance fee
    max(min_fee, take * rate * days) must never exceed `pending` receivable cash.
    """
    cap = sim._calc_borrow_cap(pending, rate, days, min_fee)
    assert np.isfinite(cap)
    assert cap >= 0.0
    if pending <= min_fee:
        assert cap == 0.0
    elif cap > 0.0:
        fee = max(min_fee, cap * rate * days)
        assert cap + fee <= pending + 1e-5


# ==============================================================================
# 3. PROPERTY-BASED TESTS: FULL SIMULATOR MICROSTRUCTURE & ACCOUNTING INVARIANTS
# ==============================================================================

@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=hose_market_scenario())
def test_fuzz_simulate_vietnam_global_invariants(scenario):
    """
    Property: Across random ±7% limit locks, floor traps, illiquid ADV, T+2.5/T+3 regimes,
    and odd fee/tax schedules:
    1. All 9 simulator output arrays are 100% finite (zero NaN/Inf).
    2. Session t=0 is strictly 100% initial_cash with zero trades and zero fees.
    3. Cumulative broker_fees, sell_tax, advance_fees, and trading_fees are monotonically non-decreasing.
    4. Fee decomposition holds at every bar: trading_fees == broker_fees + sell_tax.
    5. Realized daily turnover is non-negative and finite at every bar.
    """
    res = sim.simulate_vietnam(
        raw_open=scenario["raw_open"],
        raw_close=scenario["raw_close"],
        ceiling=scenario["ceiling"],
        floor=scenario["floor"],
        adv20_shares=scenario["adv20_shares"],
        target_weights=scenario["target_weights"],
        initial_cash=scenario["initial_cash"],
        broker_fee=scenario["broker_fee"],
        slippage=scenario["slippage"],
        sell_tax=scenario["sell_tax"],
        advance_fee_rate=scenario["advance_fee_rate"],
        min_advance_fee=scenario["min_advance_fee"],
        max_adv_participation=scenario["max_adv_participation"],
        penalty_rate_per_day=scenario["penalty_rate_per_day"],
        strict_t25_ato_lock=scenario["strict_t25_ato_lock"],
        clamp_and_round_exec_price=True,
        round_exec_adverse=scenario["round_exec_adverse"],
        calendar_days_t1=scenario["calendar_days_t1"],
        calendar_days_t2=scenario["calendar_days_t2"],
        warn_unlagged=False,
    )

    for key in (
        "equity",
        "cash_history",
        "trading_fees",
        "broker_fees",
        "sell_tax",
        "advance_fees",
        "realized_daily_turnover",
        "dividend_cash_receivable",
        "locked_corp_action_shares_val",
    ):
        arr = res[key]
        assert np.all(np.isfinite(arr)), f"Non-finite values in {key}"

    # Session t=0 baseline invariants
    assert res["equity"][0] == pytest.approx(scenario["initial_cash"])
    assert res["cash_history"][0] == pytest.approx(scenario["initial_cash"])
    assert res["trading_fees"][0] == 0.0
    assert res["broker_fees"][0] == 0.0
    assert res["sell_tax"][0] == 0.0
    assert res["advance_fees"][0] == 0.0
    assert res["realized_daily_turnover"][0] == 0.0

    # Fee monotonicity and exact additive identity
    assert np.all(np.diff(res["broker_fees"]) >= -1e-9)
    assert np.all(np.diff(res["sell_tax"]) >= -1e-9)
    assert np.all(np.diff(res["advance_fees"]) >= -1e-9)
    assert np.all(np.diff(res["trading_fees"]) >= -1e-9)
    assert np.allclose(res["trading_fees"], res["broker_fees"] + res["sell_tax"], atol=1e-9)
    assert res["total_friction"] == pytest.approx(res["trading_fees"][-1] + res["advance_fees"][-1], abs=1e-9)

    # Equity is always at least available cash (since holdings >= 0 and prices > 0)
    assert np.all(res["equity"] >= res["cash_history"] - 1e-6)
    assert np.all(res["realized_daily_turnover"] >= 0.0)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    T=st.integers(min_value=5, max_value=20),
    N=st.integers(min_value=1, max_value=4),
    price=st.integers(min_value=100, max_value=10_000).map(lambda k: float(k * 10)),
    initial_cash=st.integers(min_value=50, max_value=2_000).map(lambda k: float(k * 1_000_000)),
    broker_fee=st.floats(min_value=0.0, max_value=0.005, allow_nan=False, allow_infinity=False),
    sell_tax=st.floats(min_value=0.0, max_value=0.005, allow_nan=False, allow_infinity=False),
    advance_fee_rate=st.floats(min_value=0.0001, max_value=0.001, allow_nan=False, allow_infinity=False),
    min_advance_fee=st.floats(min_value=0.0, max_value=80_000.0, allow_nan=False, allow_infinity=False),
    strict_t25=st.booleans(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fuzz_conservation_of_value_flat_prices_and_odd_fees(
    T: int,
    N: int,
    price: float,
    initial_cash: float,
    broker_fee: float,
    sell_tax: float,
    advance_fee_rate: float,
    min_advance_fee: float,
    strict_t25: bool,
    seed: int,
):
    """
    Property (Accounting Conservation Identity):
    When prices are constant and execution slippage / half-spread floor are zero,
    at every session t in 0..T-1:
        initial_cash - equity[t] == broker_fees[t] + sell_tax[t] + advance_fees[t]
    """
    rng = np.random.default_rng(seed)
    ro = np.full((T, N), price, dtype=np.float64)
    rc = np.full((T, N), price, dtype=np.float64)
    ceil, flr = sim.compute_bands_and_ticks(ro, exchange="HOSE")
    adv = np.full((T, N), 10_000_000.0, dtype=np.float64)
    vol = np.zeros((T, N), dtype=np.float64)

    w = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        raw_w = rng.uniform(0.0, 1.0, size=N)
        s = float(np.sum(raw_w))
        if s > 1e-9:
            w[t] = (raw_w / s) * rng.uniform(0.0, 0.95)

    res = sim.simulate_vietnam(
        raw_open=ro,
        raw_close=rc,
        ceiling=ceil,
        floor=flr,
        adv20_shares=adv,
        target_weights=w,
        initial_cash=initial_cash,
        broker_fee=broker_fee,
        slippage=0.0,
        sell_tax=sell_tax,
        advance_fee_rate=advance_fee_rate,
        min_advance_fee=min_advance_fee,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=0.20,
        penalty_rate_per_day=0.0,
        strict_t25_ato_lock=strict_t25,
        warn_unlagged=False,
    )

    eq_loss = initial_cash - res["equity"]
    total_fees = res["broker_fees"] + res["sell_tax"] + res["advance_fees"]
    assert np.allclose(eq_loss, total_fees, atol=1.0)
    assert np.all(res["cash_history"] >= -1e-6)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=hose_market_scenario(min_t=8, max_t=20, min_n=1, max_n=4))
def test_fuzz_numba_jit_vs_python_core_equivalence(scenario):
    """
    Property: Numba JIT _simulate_vn_microstructure_core and pure Python .py_func
    must produce bit-for-bit identical outputs across all 9 return arrays.
    """
    T, N = scenario["T"], scenario["N"]
    corp_div_ex = np.zeros((T, N), dtype=np.float64)
    corp_div_pay_t = np.full((T, N), T + 5, dtype=np.float64)
    corp_stock_ratio = np.zeros((T, N), dtype=np.float64)
    corp_stock_list_t = np.full((T, N), T + 5, dtype=np.float64)
    in_pit_universe = np.ones((T, N), dtype=np.bool_)
    daily_vol = np.full((T, N), 0.015, dtype=np.float64)
    is_t3_regime = np.zeros(T, dtype=np.bool_)

    args = (
        scenario["raw_open"],
        scenario["raw_close"],
        scenario["ceiling"],
        scenario["floor"],
        scenario["adv20_shares"],
        scenario["target_weights"].copy(),
        scenario["initial_cash"],
        scenario["broker_fee"],
        scenario["slippage"],
        scenario["sell_tax"],
        scenario["advance_fee_rate"],
        scenario["max_adv_participation"],
        scenario["calendar_days_t1"],
        scenario["calendar_days_t2"],
        scenario["min_advance_fee"],
        corp_div_ex,
        corp_div_pay_t,
        corp_stock_ratio,
        corp_stock_list_t,
        in_pit_universe,
        daily_vol,
        scenario["penalty_rate_per_day"],
        is_t3_regime,
        0.0,
        0.0012,
        0.05,
        scenario["strict_t25_ato_lock"],
        scenario["clamp_and_round_exec_price"],
        scenario["round_exec_adverse"],
    )

    jit_out = sim._simulate_vn_microstructure_core(*args)
    args_py = list(args)
    args_py[5] = scenario["target_weights"].copy()
    py_out = sim._simulate_vn_microstructure_core.py_func(*args_py)

    for idx in range(len(jit_out)):
        assert np.allclose(jit_out[idx], py_out[idx], atol=1e-10, equal_nan=True), f"Mismatch in output {idx}"


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    T=st.integers(min_value=8, max_value=18),
    t_buy=st.integers(min_value=1, max_value=4),
    strict_t25=st.booleans(),
    is_t3=st.booleans(),
    price=st.floats(min_value=10_000.0, max_value=80_000.0, allow_nan=False, allow_infinity=False),
)
def test_fuzz_t25_and_t3_settlement_lock_prevents_early_resale(
    T: int,
    t_buy: int,
    strict_t25: bool,
    is_t3: bool,
    price: float,
):
    """
    Property: Shares purchased at ATO of session `t_buy` (with 0 initial holdings):
    - Under strict_t25_ato_lock=True or T+3 regime (pre-2022-08-29): cannot be sold at
      t_buy, t_buy+1, or t_buy+2; earliest legal ATO sale is t_buy+3.
    - Under legacy T+2 approximation (strict_t25_ato_lock=False, post-2022-08-29):
      cannot be sold at t_buy or t_buy+1; earliest sale is t_buy+2.
    """
    assume(t_buy + 4 < T)
    p = dp.round_tick_hose(price)
    ro = np.full((T, 1), p, dtype=np.float64)
    rc = np.full((T, 1), p, dtype=np.float64)
    ceil, flr = sim.compute_bands_and_ticks(ro, exchange="HOSE")
    adv = np.full((T, 1), 10_000_000.0, dtype=np.float64)
    vol = np.zeros((T, 1), dtype=np.float64)

    # Buy at t_buy, immediately set target weight to 0.0 from t_buy+1 onwards
    w = np.zeros((T, 1), dtype=np.float64)
    w[t_buy, 0] = 0.80

    base_date = datetime.date(2021, 1, 4) if is_t3 else datetime.date(2024, 1, 4)
    dates = [base_date + datetime.timedelta(days=i) for i in range(T)]

    res = sim.simulate_vietnam(
        raw_open=ro,
        raw_close=rc,
        ceiling=ceil,
        floor=flr,
        adv20_shares=adv,
        target_weights=w,
        initial_cash=500_000_000.0,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        dates=dates,
        split_date="2022-08-29",
        strict_t25_ato_lock=strict_t25,
        warn_unlagged=False,
    )

    sell_tax = res["sell_tax"]
    if strict_t25 or is_t3:
        assert sell_tax[t_buy] == 0.0
        assert sell_tax[t_buy + 1] == 0.0
        assert sell_tax[t_buy + 2] == 0.0
        assert sell_tax[t_buy + 3] > 0.0
    else:
        assert sell_tax[t_buy] == 0.0
        assert sell_tax[t_buy + 1] == 0.0
        assert sell_tax[t_buy + 2] > 0.0


# ==============================================================================
# 4. PROPERTY-BASED TESTS: ZERO LOOK-AHEAD BIAS & WALK-FORWARD SPLITS
# ==============================================================================

@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=hose_market_scenario(min_t=10, max_t=24, min_n=2, max_n=4))
def test_fuzz_simulate_vietnam_strict_causality_under_future_perturbation(scenario):
    """
    Property (Zero Look-Ahead in Simulator):
    Perturbing raw_open, raw_close, ceiling, floor, adv20_shares, and target_weights
    strictly after session t0 (t > t0) must have ZERO effect on any simulator state
    at sessions t <= t0.
    """
    T = scenario["T"]
    t0 = T // 2

    res_orig = sim.simulate_vietnam(
        raw_open=scenario["raw_open"],
        raw_close=scenario["raw_close"],
        ceiling=scenario["ceiling"],
        floor=scenario["floor"],
        adv20_shares=scenario["adv20_shares"],
        target_weights=scenario["target_weights"],
        initial_cash=scenario["initial_cash"],
        broker_fee=scenario["broker_fee"],
        slippage=scenario["slippage"],
        sell_tax=scenario["sell_tax"],
        advance_fee_rate=scenario["advance_fee_rate"],
        min_advance_fee=scenario["min_advance_fee"],
        max_adv_participation=scenario["max_adv_participation"],
        penalty_rate_per_day=scenario["penalty_rate_per_day"],
        strict_t25_ato_lock=scenario["strict_t25_ato_lock"],
        daily_vol=None,
        warn_unlagged=False,
    )

    ro_p = scenario["raw_open"].copy()
    rc_p = scenario["raw_close"].copy()
    ceil_p = scenario["ceiling"].copy()
    flr_p = scenario["floor"].copy()
    adv_p = scenario["adv20_shares"].copy()
    w_p = scenario["target_weights"].copy()

    ro_p[t0 + 1 :] *= 3.0
    rc_p[t0 + 1 :] *= 0.25
    ceil_p[t0 + 1 :] *= 3.0
    flr_p[t0 + 1 :] *= 0.25
    adv_p[t0 + 1 :] = 0.0
    w_p[t0 + 1 :] = 0.0

    res_pert = sim.simulate_vietnam(
        raw_open=ro_p,
        raw_close=rc_p,
        ceiling=ceil_p,
        floor=flr_p,
        adv20_shares=adv_p,
        target_weights=w_p,
        initial_cash=scenario["initial_cash"],
        broker_fee=scenario["broker_fee"],
        slippage=scenario["slippage"],
        sell_tax=scenario["sell_tax"],
        advance_fee_rate=scenario["advance_fee_rate"],
        min_advance_fee=scenario["min_advance_fee"],
        max_adv_participation=scenario["max_adv_participation"],
        penalty_rate_per_day=scenario["penalty_rate_per_day"],
        strict_t25_ato_lock=scenario["strict_t25_ato_lock"],
        daily_vol=None,
        warn_unlagged=False,
    )

    for key in (
        "equity",
        "cash_history",
        "trading_fees",
        "broker_fees",
        "sell_tax",
        "advance_fees",
        "realized_daily_turnover",
    ):
        assert np.allclose(res_orig[key][: t0 + 1], res_pert[key][: t0 + 1], atol=1e-9), (
            f"Look-ahead leakage in {key} at t <= {t0}"
        )


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    T=st.integers(min_value=15, max_value=45),
    N=st.integers(min_value=8, max_value=15),
    K=st.integers(min_value=2, max_value=5),
    window=st.integers(min_value=5, max_value=12),
    embargo=st.integers(min_value=1, max_value=5),
    ties=st.sampled_from(["ordinal", "average"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fuzz_purged_walk_forward_combiner_causality_and_weight_simplex(
    T: int,
    N: int,
    K: int,
    window: int,
    embargo: int,
    ties: str,
    seed: int,
):
    """
    Property: For any walk-forward configuration with T >= window + embargo and embargo >= 1:
    1. factor_weights_history[t] is on the probability simplex (w >= 0, sum(w) == 1.0) for all t.
    2. Perturbing raw_close or factor values strictly after t0 (t > t0) has ZERO effect on
       composite_alpha[:t0+1] and factor_weights_history[:t0+1].
    """
    assume(T > window + embargo + 2)
    rng = np.random.default_rng(seed)
    ortho_alphas = [rng.standard_normal((T, N)) for _ in range(K)]
    raw_close = 20_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(T, N)), axis=0))
    tradable_mask = rng.random((T, N)) > 0.10

    comp_orig, w_orig = mlc.purged_walk_forward_combiner(
        ortho_alphas=ortho_alphas,
        raw_close=raw_close,
        tradable_mask=tradable_mask,
        window=window,
        embargo=embargo,
        rank_ic_ties=ties,
    )

    assert np.all(np.isfinite(comp_orig))
    assert np.all(np.isfinite(w_orig))
    assert np.all(w_orig >= -1e-12)
    assert np.allclose(np.sum(w_orig, axis=1), 1.0, atol=1e-9)

    # Future perturbation at t > t0
    t0 = window + embargo + (T - (window + embargo)) // 2
    rc_pert = raw_close.copy()
    rc_pert[t0 + 1 :] *= 5.0
    alphas_pert = [a.copy() for a in ortho_alphas]
    for a in alphas_pert:
        a[t0 + 1 :] = -a[t0 + 1 :] * 100.0

    comp_pert, w_pert = mlc.purged_walk_forward_combiner(
        ortho_alphas=alphas_pert,
        raw_close=rc_pert,
        tradable_mask=tradable_mask,
        window=window,
        embargo=embargo,
        rank_ic_ties=ties,
    )

    assert np.allclose(w_orig[: t0 + 1], w_pert[: t0 + 1], atol=1e-12)
    assert np.allclose(comp_orig[: t0 + 1], comp_pert[: t0 + 1], atol=1e-12)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    T=st.integers(min_value=12, max_value=30),
    N=st.integers(min_value=4, max_value=12),
    d=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fuzz_alpha_ops_causality_and_bounds(T: int, N: int, d: int, seed: int):
    """
    Property: Cross-sectional and rolling time-series operators in alpha_ops:
    1. Exhibit 0.0 look-ahead leakage when future rows t > t0 are perturbed.
    2. Respect mathematical output bounds:
       - rank in [0, 1]
       - zscore in [-clip, clip]
       - scale absolute sum == target_sum (or 0 when all zero/NaN)
       - group_neutralize group means == 0
       - ts_corr in [-1, 1]
    """
    assume(d < T - 2)
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 50.0, size=(T, N))
    Y = rng.normal(0.0, 50.0, size=(T, N))
    # Inject sparse NaNs and flat slices
    X[rng.random((T, N)) < 0.08] = np.nan
    X[T // 3, :] = 42.0
    mask = rng.random((T, N)) > 0.15
    groups = rng.integers(0, 3, size=N)

    t0 = T // 2
    X_pert = X.copy()
    Y_pert = Y.copy()
    X_pert[t0 + 1 :] += 99_999.0
    Y_pert[t0 + 1 :] -= 77_777.0

    # 1. Cross-sectional operators
    rk = ops.rank(X, mask=mask)
    rk_p = ops.rank(X_pert, mask=mask)
    assert np.allclose(rk[: t0 + 1], rk_p[: t0 + 1], equal_nan=True)
    valid_rk = rk[np.isfinite(rk)]
    if len(valid_rk) > 0:
        assert np.all((valid_rk >= -1e-12) & (valid_rk <= 1.0 + 1e-12))

    zs = ops.zscore(X, clip=2.5, mask=mask)
    zs_p = ops.zscore(X_pert, clip=2.5, mask=mask)
    assert np.allclose(zs[: t0 + 1], zs_p[: t0 + 1], equal_nan=True)
    assert np.all(np.isfinite(zs)) and np.all(np.abs(zs) <= 2.5 + 1e-12)

    sc = ops.scale(X, target_sum=1.0, mask=mask)
    sc_p = ops.scale(X_pert, target_sum=1.0, mask=mask)
    assert np.allclose(sc[: t0 + 1], sc_p[: t0 + 1], equal_nan=True)
    for t in range(T):
        s_row = float(np.sum(np.abs(sc[t])))
        assert s_row == pytest.approx(0.0, abs=1e-9) or s_row == pytest.approx(1.0, abs=1e-9)

    gn = ops.group_neutralize(X, groups, mask=mask)
    gn_p = ops.group_neutralize(X_pert, groups, mask=mask)
    assert np.allclose(gn[: t0 + 1], gn_p[: t0 + 1], equal_nan=True)
    for t in range(T):
        valid_t = np.isfinite(X[t]) & mask[t]
        for g in np.unique(groups[valid_t]):
            g_mask = valid_t & (groups == g)
            if np.any(g_mask):
                assert float(np.mean(gn[t, g_mask])) == pytest.approx(0.0, abs=1e-9)

    # 2. Time-series operators
    for op_fn in (
        lambda a: ops.ts_delay(a, d),
        lambda a: ops.ts_delta(a, d),
        lambda a: ops.ts_mean(a, d),
        lambda a: ops.ts_std_dev(a, d),
        lambda a: ops.ts_rank(a, d),
        lambda a: ops.ts_decay_linear(a, d),
        lambda a: ops.apply_hawkes_ou_decay(a, tau_sessions=float(d)),
        lambda a: ops.van_der_waerden_transform(a, mask=mask, clip_limit=3.0),
    ):
        o1 = op_fn(X)
        o2 = op_fn(X_pert)
        assert np.allclose(o1[: t0 + 1], o2[: t0 + 1], equal_nan=True)

    c1 = ops.ts_corr(X, Y, d)
    c2 = ops.ts_corr(X_pert, Y_pert, d)
    assert np.allclose(c1[: t0 + 1], c2[: t0 + 1], equal_nan=True)
    valid_c = c1[np.isfinite(c1)]
    if len(valid_c) > 0:
        assert np.all((valid_c >= -1.0 - 1e-12) & (valid_c <= 1.0 + 1e-12))


@settings(max_examples=150, deadline=None)
@given(
    returns=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=5, max_value=120),
        elements=st.floats(min_value=-0.07, max_value=0.07, allow_nan=False, allow_infinity=False),
    ),
    n_trials=st.integers(min_value=1, max_value=50),
    lags=st.integers(min_value=1, max_value=10),
)
def test_fuzz_dsr_and_newey_west_invariants(returns: np.ndarray, n_trials: int, lags: int):
    """
    Property: For bounded daily returns in [-7%, +7%]:
    1. compute_deflated_sharpe_ratio returns finite DSR in [0, 1] with dsr + p_value == 1.0 and n_eff >= 1.0.
    2. compute_newey_west_sharpe_se returns a finite non-negative standard error.
    """
    dsr_res = sim.compute_deflated_sharpe_ratio(returns, n_trials=n_trials)
    assert np.isfinite(dsr_res["deflated_sharpe_ratio"])
    assert np.isfinite(dsr_res["p_value"])
    assert 0.0 <= dsr_res["deflated_sharpe_ratio"] <= 1.0
    assert dsr_res["deflated_sharpe_ratio"] + dsr_res["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert dsr_res["n_eff"] >= 1.0

    nw_se = sim.compute_newey_west_sharpe_se(returns, lags=lags)
    assert np.isfinite(nw_se) and nw_se >= 0.0


@settings(max_examples=150, deadline=None)
@given(
    q_shares=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    adv20_shares=st.floats(min_value=1_000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
)
def test_fuzz_slice_intraday_order_conservation_and_lot_100(q_shares: float, adv20_shares: float):
    """
    Property: slice_intraday_order always returns lot-100 multiples that sum to q_total
    and never exceed floor(0.03 * adv20_shares / 100) * 100.
    """
    res = ops.slice_intraday_order(q_shares, adv20_shares)
    assert res["q_total"] == pytest.approx(res["q_ato"] + res["q_twap"] + res["q_atc"])
    for key in ("q_total", "q_ato", "q_twap", "q_atc"):
        assert res[key] >= 0.0
        assert res[key] % 100.0 == pytest.approx(0.0, abs=1e-6)
    assert res["q_total"] <= adv20_shares * 0.03 + 1e-6


# ==============================================================================
# 5. CONFIRMED BUGS & EDGE CASES FOUND BY HYPOTHESIS FUZZING (REGRESSION TESTS)
# ==============================================================================

def test_bug1_partial_floor_trap_double_sell_on_unlock():
    T, N = 6, 1
    ro = np.full((T, N), 50_000.0)
    rc = np.full((T, N), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    # Session t=4 opens locked at floor; target weight drops from 1.0 (2,000 shares) to 0.5 (1,000 shares)
    ro[4, 0] = flr[4, 0]
    w = np.zeros((T, N))
    w[1:4, 0] = 1.0
    w[4:, 0] = 0.5

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.001,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        strict_t25_ato_lock=True,
        warn_unlagged=False,
    )
    # Only 1,000 shares (50M VND) should be sold at t=5 -> sell_tax == 50,000 VND (actual: 100,000 VND)
    assert res["sell_tax"][5] == pytest.approx(50_000.0)


def test_bug2_multi_day_floor_trap_resets_trapped_days_and_breaches_adv_cap():
    T, N = 8, 1
    ro = np.full((T, N), 50_000.0)
    rc = np.full((T, N), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    # Floor-locked on sessions t=4, 5, 6; unlocks at t=7
    ro[4:7, 0] = flr[4:7, 0]
    # ADV20 = 20,000 shares, max_adv_participation = 5% -> max 1,000 shares (50M VND) per session
    adv = np.full((T, N), 20_000.0)
    vol = np.zeros((T, N))
    w = np.zeros((T, N))
    w[1:4, 0] = 1.0  # Accumulates 3,000 shares (1,000/day at t=1, 2, 3)

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=150_000_000.0,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.001,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=0.05,
        penalty_rate_per_day=0.05,
        strict_t25_ato_lock=False,
        warn_unlagged=False,
    )
    # Single-session sell volume at t=7 must not exceed adv_limit = 1,000 shares (turnover <= 50M / 300M = 1/6)
    assert res["realized_daily_turnover"][7] <= (1.0 / 6.0) + 1e-6


def test_bug3_trapped_sell_executes_when_signal_reverses_to_hold():
    T, N = 7, 1
    ro = np.full((T, N), 50_000.0)
    rc = np.full((T, N), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    # Session t=4 is floor-locked when signal wants to exit (w=0.0); at t=5 signal wants to hold 100% (w=1.0)
    ro[4, 0] = flr[4, 0]
    w = np.zeros((T, N))
    w[1:4, 0] = 1.0
    w[4, 0] = 0.0
    w[5:, 0] = 1.0

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        strict_t25_ato_lock=True,
        warn_unlagged=False,
    )
    # Since t=4 could not sell and at t=5 target weight is 1.0 (already holding 100%), no sell should execute at t=5
    assert res["sell_tax"][5] == 0.0


def test_bug4_float_ulp_shortfall_causes_negative_cash_and_ghost_advance_fee():
    p = 10_000.13
    cps = p * (1.0 + 0.0015)
    actual = 1_500.0 * cps
    init_cash = float(np.nextafter(actual, -np.inf))

    T, N = 6, 1
    ro = np.full((T, N), p)
    rc = np.full((T, N), p)
    ceil = np.full((T, N), p * 1.07)
    flr = np.full((T, N), p * 0.93)
    adv = np.full((T, N), 1e9)
    vol = np.zeros((T, N))
    w = np.zeros((T, N))
    w[1, 0] = 1.0

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=init_cash,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        min_advance_fee=50_000.0,
        sanitize_inputs=True,
        warn_unlagged=False,
    )
    assert np.all(res["cash_history"] >= 0.0), f"Negative cash detected: {res['cash_history']}"
    assert res["advance_fees"][-1] == 0.0, f"Ghost advance fee charged with zero sales: {res['advance_fees']}"


def test_bug5_min_advance_notional_aborts_buy_instead_of_scaling_to_cash_avail():
    T, N = 5, 2
    p = 50_000.0
    ro = np.full((T, N), p)
    rc = np.full((T, N), p)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    # Start with 100M. At t=1 buy 20M of asset 0 (leaving ~80M cash_avail).
    # At t=3 sell asset 0 (~20M pending_cash_t0) and target 100% in asset 1 (~100M needed).
    # Initial shortfall ~ 20M; if we set initial_cash=160M, buy 20M of asset 0 at t=1 (leaving ~140M cash),
    # and at t=3 sell asset 0 (~20M pending_cash_t0) while targeting w[3, 1] = 2.0 (sanitize=False -> ~320M needed):
    # initial shortfall = 180M >= min_advance_notional (50M), but max_budget = 140M + 20M = 160M,
    # so scaled shortfall = 20M < min_advance_notional (50M) -> simulator buys 0 shares instead of 140M cash!
    w = np.zeros((T, N))
    w[1, 0] = 0.125
    w[2, 0] = 0.125
    w[3, 1] = 2.0

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=160_000_000.0,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        min_advance_notional=50_000_000.0,
        strict_t25_ato_lock=False,
        sanitize_inputs=False,
        warn_unlagged=False,
    )
    # Asset 1 should be purchased using the ~140M available cash at t=3
    assert res["cash_history"][3] < 10_000_000.0


def test_bug6_corporate_action_negative_sentinel_pay_date_pays_on_last_bar():
    T, N = 8, 1
    ro = np.full((T, N), 50_000.0)
    rc = np.full((T, N), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 10_000_000.0)
    w = np.zeros((T, N))
    w[1:, 0] = 1.0

    div_cash = np.zeros((T, N))
    div_cash[3, 0] = 2_000.0
    # Sentinel -1 (payout beyond simulation horizon T)
    div_pay_t = np.full((T, N), -1.0)

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        half_spread_floor=0.0,
        daily_vol=np.zeros((T, N)),
        corp_action_div_cash=div_cash,
        corp_action_div_pay_t=div_pay_t,
        warn_unlagged=False,
    )
    # With sentinel -1 (payout after horizon), dividend must remain in receivable at T-1, not paid via index -1
    assert res["dividend_cash_receivable"][-1] > 0.0


def test_bug7_sanitize_market_matrices_nan_weight_bypasses_row_normalization():
    ro = np.full((4, 3), 50_000.0)
    rc = np.full((4, 3), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((4, 3), 1_000_000.0)
    tw = np.array([[np.nan, 0.80, 0.80]] * 4, dtype=np.float64)

    _, _, _, _, _, clean_w, _ = sim.sanitize_market_matrices(ro, rc, ceil, flr, adv, tw)
    assert np.all(np.isfinite(clean_w))
    assert np.all(np.sum(clean_w, axis=1) <= 1.0 + 1e-9)


def test_bug8_upcom_band_float_truncation_on_exact_tick_multiples():
    ceil, flr = sim.compute_bands_and_ticks(np.array([6_000.0, 12_000.0, 24_000.0]), exchange="UPCOM")
    assert list(ceil) == [6_900.0, 13_800.0, 27_600.0]
    assert list(flr) == [5_100.0, 10_200.0, 20_400.0]
    ref_ulp = float(np.nextafter(1000.0, np.inf))
    _, sim_flr = sim.compute_bands_and_ticks(np.array([ref_ulp]), exchange="HOSE")
    _, dp_flr = dp.compute_hose_limits(ref_ulp)
    assert float(sim_flr[0]) == float(dp_flr)


def test_bug9_purged_walk_forward_combiner_index_error_on_short_series():
    T, N = 30, 8
    alphas = [np.ones((T, N)), np.ones((T, N))]
    rc = np.full((T, N), 50_000.0)
    mask = np.ones((T, N), dtype=bool)
    comp, w_hist = mlc.purged_walk_forward_combiner(alphas, rc, mask, window=40, embargo=3)
    assert comp.shape == (T, N)
    assert w_hist.shape == (T, 2)


def test_bug10_purged_walk_forward_combiner_zero_weight_nan_poisoning():
    T, N = 25, 10
    rng = np.random.default_rng(7)
    a_valid = rng.standard_normal((T, N))
    a_nan = np.full((T, N), np.nan)
    rc = 50_000.0 + np.cumsum(rng.normal(100.0, 50.0, size=(T, N)), axis=0)
    mask = np.ones((T, N), dtype=bool)

    comp, w_hist = mlc.purged_walk_forward_combiner([a_valid, a_nan], rc, mask, window=10, embargo=2)
    assert w_hist[-1, 1] == 0.0
    assert np.all(np.isfinite(comp[-1]))


def test_bug11_dampen_turnover_normalization_breaks_turnover_cap_and_locks_masked_stocks():
    alpha = np.ones((10, 4), dtype=np.float64)
    mask = np.ones((10, 4), dtype=bool)
    mask[0, :] = False  # Session 0: 100% cash (weights[0] == 0)
    mask[5:, :] = False  # Sessions 5..9: all stocks untradable

    w = ops.dampen_turnover(alpha, top_k_in=2, top_k_out=3, ema_span=10, max_daily_turnover=0.05, mask=mask)
    day1_turnover = 0.5 * float(np.sum(np.abs(w[1] - w[0])))
    assert day1_turnover <= 0.05 + 1e-9, f"Day 1 turnover {day1_turnover} breached max_daily_turnover=0.05"
    assert float(np.sum(w[9])) < 1.0 - 1e-6, f"Weights stayed 100% invested in masked-out stocks: {w[9]}"


def test_bug12_ts_corr_catastrophic_cancellation_on_constant_series():
    val = 1e10 / 7.0
    x_const = np.full((15, 1), val, dtype=np.float64)
    corr = ops.ts_corr(x_const, x_const, 7)
    assert corr[-1, 0] == 0.0


def test_bug13_compute_deflated_sharpe_ratio_nan_on_precision_loss():
    rets = np.full(30, 1e5, dtype=np.float64)
    rets[0] += 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sim.compute_deflated_sharpe_ratio(rets)
    assert np.isfinite(res["deflated_sharpe_ratio"])
    assert np.isfinite(res["p_value"])


def test_bug14_trapped_exit_unbounded_slippage_causes_negative_price_fees_cash_and_equity():
    T, N = 8, 1
    ro = np.full((T, N), 50_000.0)
    rc = np.full((T, N), 50_000.0)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    ro[4, 0] = flr[4, 0]  # Floor trap at session t=4

    adv = np.full((T, N), 1_000_000.0)
    adv[4, 0] = 50.0  # Liquidity dries up on session t=4 (used as t_lag=4 at t=5 unlock)
    vol = np.full((T, N), 0.05)

    w = np.zeros((T, N))
    w[1:4, 0] = 1.0

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=500_000_000.0,
        max_adv_participation=0.10,
        warn_unlagged=False,
    )
    assert np.all(np.diff(res["broker_fees"]) >= 0.0)
    assert np.all(res["sell_tax"] >= 0.0)
    assert np.all(res["cash_history"] >= 0.0)
    assert np.all(res["equity"] >= 0.0)

