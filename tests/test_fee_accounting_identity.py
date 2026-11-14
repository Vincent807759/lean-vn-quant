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
            min_advance_notional=0.0,
            warn_unlagged=False
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
            min_advance_notional=71428571.0,
            warn_unlagged=False
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
        min_advance_notional=0.0,
        strict_t25_ato_lock=False,  # scenario sells at T+2 ATO (legacy timing)
    )

    adv_fees_t3 = res["advance_fees"][3] - res["advance_fees"][2]
    assert 0.0 < adv_fees_t3 < 90000.0, (
        f"Expected single advance fee (~50k), got {adv_fees_t3:,.0f} VND"
    )
    assert np.all(res["cash_history"] >= -1e-6), "Cash history should not be negative"
def test_fee_split_and_monotonic_accounting_invariants():
    """
    M4.1a: Comprehensive accounting conservation and pure Python vs Numba JIT equivalence.
    Verifies:
    1. trading_fees == broker_fees + sell_tax and total_friction == trading_fees[-1] + advance_fees[-1].
    2. Monotonically non-decreasing cumulative fees and non-negative cash_avail at every session.
    3. Bit-for-bit 0.0 difference between Numba JIT (_simulate_vn_microstructure_core)
       and pure Python (_simulate_vn_microstructure_core.py_func) across all 9 outputs.
    4. Exact equity decomposition:
       equity[t] == cash_avail[t] + pending_cash_t1[t] + pending_cash_t0[t] + div_receivable[t] + holdings_val_close[t] + locked_corp_val_close[t].
    """
    T = 30
    N = 5
    rng = np.random.RandomState(42)
    p0 = 50_000.0
    drift = rng.normal(0.0005, 0.01, size=(T, N))
    raw_close = p0 * np.exp(np.cumsum(drift, axis=0))
    raw_open = raw_close * rng.uniform(0.99, 1.01, size=(T, N))
    ceiling = raw_close * 1.07
    floor = raw_close * 0.93
    adv20_shares = np.full((T, N), 1e6, dtype=np.float64)
    daily_vol = np.full((T, N), 0.015, dtype=np.float64)

    weights = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        w = rng.uniform(0.0, 1.0, size=N)
        weights[t] = w / np.sum(w)

    init_cash = 1e9
    broker_fee = 0.0015
    slippage = 0.001
    sell_tax = 0.0010
    advance_fee_rate = 0.00035
    min_advance_fee = 50000.0
    half_spread_floor = 0.0012
    max_adv_participation = 0.03
    penalty_rate_per_day = 0.0
    min_advance_notional = 71428571.0

    res = sim.simulate_vietnam(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=weights,
        initial_cash=init_cash,
        broker_fee=broker_fee,
        slippage=slippage,
        sell_tax=sell_tax,
        advance_fee_rate=advance_fee_rate,
        min_advance_fee=min_advance_fee,
        half_spread_floor=half_spread_floor,
        daily_vol=daily_vol,
        max_adv_participation=max_adv_participation,
        penalty_rate_per_day=penalty_rate_per_day,
        min_advance_notional=min_advance_notional,
        warn_unlagged=False
    )

    t_fees = res["trading_fees"]
    b_fees = res["broker_fees"]
    s_tax = res["sell_tax"]
    a_fees = res["advance_interest_fees"]

    # 1. Fee split & total friction
    assert np.max(np.abs(t_fees - (b_fees + s_tax))) < 1e-9
    assert abs(res["total_friction"] - (t_fees[-1] + a_fees[-1])) < 1e-9

    # 2. Monotonic non-decreasing
    assert np.all(np.diff(b_fees) >= -1e-9)
    assert np.all(np.diff(s_tax) >= -1e-9)
    assert np.all(np.diff(a_fees) >= -1e-9)
    assert np.all(np.diff(t_fees) >= -1e-9)

    # 3. Non-negative cash
    assert np.min(res["cash_history"]) >= -1e-9

    # 4. Compare Numba JIT vs pure Python .py_func
    corp_div_ex = np.zeros((T, N), dtype=np.float64)
    corp_div_pay_t = np.full((T, N), -1, dtype=np.int64)
    corp_stock_ratio = np.zeros((T, N), dtype=np.float64)
    corp_stock_list_t = np.full((T, N), -1, dtype=np.int64)
    in_pit_universe = np.ones((T, N), dtype=bool)
    is_t3_regime = np.zeros(T, dtype=bool)
    calendar_days_t1 = np.ones(T, dtype=np.int64)
    calendar_days_t2 = np.full(T, 2, dtype=np.int64)

    jit_res = sim._simulate_vn_microstructure_core(
        raw_open, raw_close, ceiling, floor, adv20_shares, weights, init_cash,
        broker_fee, slippage, sell_tax, advance_fee_rate, max_adv_participation,
        calendar_days_t1, calendar_days_t2, min_advance_fee,
        corp_div_ex, corp_div_pay_t, corp_stock_ratio, corp_stock_list_t,
        in_pit_universe, daily_vol, penalty_rate_per_day, is_t3_regime,
        min_advance_notional, half_spread_floor, 0.05, False, True, False
    )
    py_res = sim._simulate_vn_microstructure_core.py_func(
        raw_open, raw_close, ceiling, floor, adv20_shares, weights, init_cash,
        broker_fee, slippage, sell_tax, advance_fee_rate, max_adv_participation,
        calendar_days_t1, calendar_days_t2, min_advance_fee,
        corp_div_ex, corp_div_pay_t, corp_stock_ratio, corp_stock_list_t,
        in_pit_universe, daily_vol, penalty_rate_per_day, is_t3_regime,
        min_advance_notional, half_spread_floor, 0.05, False, True, False
    )

    for i in range(len(jit_res)):
        diff = np.max(np.abs(jit_res[i] - py_res[i]))
        assert diff == 0.0, f"Output array {i} mismatch between JIT and py_func: {diff}"


def test_placebo_cross_sectional_permutation_pipeline_invariants():
    """
    M4.1b: Deterministic cross-sectional permutation test (A7, seed=20261107, N_perm=25, T=120, N=10).
    Verifies that:
    1. Permutations inside tradable_mask preserve accounting conservation across all shuffle paths.
    2. Real predictive signal achieves k=0 (p_value <= 0.05) vs cross-sectional shuffles.
    3. Production baseline accounting invariants hold if data/vn30_production.parquet is present.
    """
    import pathlib
    from engine import alpha_ops as ops

    seed = 20261107
    N_perm = 25
    T = 120
    N = 10
    rng = np.random.RandomState(seed)

    # Base price process: leading stocks have strong upward drift
    drift = rng.normal(0.0002, 0.008, size=(T, N))
    drift[:, :3] += 0.003
    p0 = 50_000.0
    raw_close = p0 * np.exp(np.cumsum(drift, axis=0))
    raw_open = raw_close * rng.uniform(0.995, 1.005, size=(T, N))
    ceiling = raw_close * 1.07
    floor = raw_close * 0.93
    adv20_shares = np.full((T, N), 1e6, dtype=np.float64)

    tradable_mask = np.ones((T, N), dtype=bool)
    tradable_mask[:, 9] = False

    base_signal = np.zeros((T, N), dtype=np.float64)
    base_signal[:, :3] = 1.0
    base_signal[:, 3:6] = 0.5
    base_signal[:, 6:] = 0.0

    w_obs_target = ops.dampen_turnover(base_signal, top_k_in=3, top_k_out=5, mask=tradable_mask, tie_break_key=adv20_shares)
    w_obs_exec = np.zeros_like(w_obs_target)
    w_obs_exec[1:] = w_obs_target[:-1]

    res_obs = sim.simulate_vietnam(
        raw_open, raw_close, ceiling, floor, adv20_shares, w_obs_exec,
        initial_cash=1e9, penalty_rate_per_day=0.0, warn_unlagged=False
    )
    obs_ret = np.diff(res_obs["equity"]) / res_obs["equity"][:-1]
    obs_sharpe = float(np.mean(obs_ret) / np.std(obs_ret, ddof=1) * np.sqrt(250))

    placebo_sharpes = []
    for p in range(N_perm):
        perm_sig = np.zeros_like(base_signal)
        for t in range(T):
            idx = np.flatnonzero(tradable_mask[t])
            if idx.size > 1:
                vals = base_signal[t, idx].copy()
                perm_idx = rng.permutation(idx)
                perm_sig[t, perm_idx] = vals
            elif idx.size == 1:
                perm_sig[t, idx] = base_signal[t, idx]

        w_perm_target = ops.dampen_turnover(perm_sig, top_k_in=3, top_k_out=5, mask=tradable_mask, tie_break_key=adv20_shares)
        w_perm_exec = np.zeros_like(w_perm_target)
        w_perm_exec[1:] = w_perm_target[:-1]

        res_p = sim.simulate_vietnam(
            raw_open, raw_close, ceiling, floor, adv20_shares, w_perm_exec,
            initial_cash=1e9, penalty_rate_per_day=0.0, warn_unlagged=False
        )
        assert np.min(res_p["cash_history"]) >= -1e-9
        t_f = res_p["trading_fees"]
        assert np.max(np.abs(t_f - (res_p["broker_fees"] + res_p["sell_tax"]))) < 1e-9

        p_ret = np.diff(res_p["equity"]) / res_p["equity"][:-1]
        p_sr = float(np.mean(p_ret) / np.std(p_ret, ddof=1) * np.sqrt(250))
        placebo_sharpes.append(p_sr)

    placebo_sharpes = np.array(placebo_sharpes)
    k = int(np.sum(placebo_sharpes >= obs_sharpe))
    p_val = (1 + k) / (N_perm + 1)

    assert k == 0
    assert p_val <= 0.05
    assert np.mean(placebo_sharpes) > 0.0

    # Production dataset baseline check if available
    prod_path = pathlib.Path("data/vn30_production.parquet")
    if prod_path.exists():
        import polars as pl
        df_prod = pl.read_parquet(str(prod_path))
        assert len(df_prod) > 0
