"""
tests/test_dynamic_execution.py - Phase 2 & Phase 3 Microstructure, LOB & FinGraph Tests
Migrated from run_verification.py (Test 6 & Test 7) into pytest suite.
"""
import os
import datetime
import numpy as np
import polars as pl
import pytest
from engine import alpha_ops as ops
from engine import vn_simulator as sim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_phase2_pit_constituent_masking_and_perturbation():
    """Subtest 6.1: Vectorized PIT Constituent Masking & Perturbation Invariance."""
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(50)]
    df_aaa = pl.DataFrame({
        "ticker": ["AAA"] * 50,
        "date": dates,
        "raw_close": [50000.0] * 50,
        "volume": [300000.0] * 50  # 15B VND/day > 10B threshold
    })
    df_bbb = pl.DataFrame({
        "ticker": ["BBB"] * 50,
        "date": dates,
        "raw_close": [20000.0] * 50,
        "volume": [100000.0] * 50  # 2B VND/day < 10B threshold
    })
    df_ipo = pl.DataFrame({
        "ticker": ["IPO"] * 10,
        "date": dates[:10],
        "raw_close": [80000.0] * 10,
        "volume": [1000000.0] * 10
    })
    ohlcv = pl.concat([df_aaa, df_bbb, df_ipo])
    mem = pl.DataFrame({
        "ticker": ["AAA", "BBB", "IPO"],
        "effective_date": [datetime.date(2024, 1, 16), datetime.date(2024, 1, 11), datetime.date(2024, 1, 1)],
        "expiry_date": [datetime.date(2024, 1, 31), datetime.date(2024, 1, 26), datetime.date(2024, 1, 10)]
    })
    res_pit = ops.apply_pit_universe_mask(ohlcv, mem, adv_threshold_vnd=10_000_000_000.0, min_samples=15)

    bbb_valid = res_pit.filter(pl.col("ticker") == "BBB")["valid_alpha_mask"].sum()
    assert bbb_valid == 0, f"BBB should have 0 valid sessions due to ADV filter, got {bbb_valid}"

    ipo_valid = res_pit.filter(pl.col("ticker") == "IPO")["valid_alpha_mask"].sum()
    assert ipo_valid == 0, f"Newly listed IPO (< 15 sessions) must have 0 valid sessions, got {ipo_valid}"

    aaa_valid = res_pit.filter(pl.col("ticker") == "AAA")["valid_alpha_mask"].sum()
    assert aaa_valid == 16, f"AAA should have 16 valid sessions, got {aaa_valid}"

    aaa_day_after = res_pit.filter((pl.col("ticker") == "AAA") & (pl.col("date") == datetime.date(2024, 2, 1)))
    assert aaa_day_after["pit_masked_close"][0] is None, "Masked close must be None after index exclusion"

    # Perturbation invariance test
    df_perturbed = pl.concat([
        pl.DataFrame({
            "ticker": ["AAA"] * 50,
            "date": dates,
            "raw_close": [50000.0] * 40 + [99999.0] * 10,
            "volume": [300000.0] * 40 + [1000000.0] * 10
        }),
        df_bbb,
        df_ipo
    ])
    res_perturbed = ops.apply_pit_universe_mask(df_perturbed, mem, adv_threshold_vnd=10_000_000_000.0, min_samples=15)
    s_orig = res_pit.filter((pl.col("ticker") == "AAA") & (pl.col("date") <= datetime.date(2024, 1, 20)))["valid_alpha_mask"].to_numpy()
    s_pert = res_perturbed.filter((pl.col("ticker") == "AAA") & (pl.col("date") <= datetime.date(2024, 1, 20)))["valid_alpha_mask"].to_numpy()
    diff = int(np.sum(s_orig != s_pert))
    assert diff == 0, f"PIT masking violated perturbation invariance with diff={diff}"


def test_phase2_orderly_liquidation():
    """Subtest 6.2: Orderly Liquidation Mechanism & Grace Counter."""
    T_liq, N_liq = 6, 1
    ro_liq = np.full((T_liq, N_liq), 50000.0)
    rc_liq = np.full((T_liq, N_liq), 50000.0)
    ceil_liq = np.full((T_liq, N_liq), 53500.0)
    flr_liq = np.full((T_liq, N_liq), 46500.0)
    adv_liq = np.full((T_liq, N_liq), 1000000.0)
    w_liq = np.ones((T_liq, N_liq), dtype=np.float64)
    pit_liq = np.ones((T_liq, N_liq), dtype=bool)
    pit_liq[2:, 0] = False  # Excluded starting at t=2

    res_liq = sim.simulate_vietnam(
        raw_open=ro_liq,
        raw_close=rc_liq,
        ceiling=ceil_liq,
        floor=flr_liq,
        adv20_shares=adv_liq,
        target_weights=w_liq,
        initial_cash=1.0e9,
        in_pit_universe=pit_liq
    )
    assert res_liq["sell_tax"][1] == 0.0, "No sell at t=1"
    assert res_liq["sell_tax"][2] > 0.0, "Excluded stock must be liquidated starting at t=2"
    assert res_liq["equity"][-1] > 0.98e9, "Cash from liquidated stock must be preserved"


def test_phase2_hose_lob_sweep_and_market_impact():
    """Subtest 6.3: HOSE 3-Tier LOB Sweep & Square-Root Impact Model."""
    adv_val = 1_000_000.0
    vol_val = 0.02
    p_mid = 50000.0
    depths = np.array([5000.0, 5000.0, 5000.0])
    prices = np.array([49950.0, 49900.0, 49850.0])

    is_floor = ops.compute_hose_market_impact(100.0, p_mid, adv_val, vol_val)
    is_lob = ops.compute_hose_market_impact(10000.0, p_mid, adv_val, vol_val, depths, prices)
    is_tail = ops.compute_hose_market_impact(25000.0, p_mid, adv_val, vol_val, depths, prices)

    assert is_floor >= 0.0012, f"Half-spread floor (12 bps) violated: {is_floor}"
    assert is_tail > is_lob, f"Tail impact {is_tail} must exceed within-LOB impact {is_lob}"


def test_phase2_intraday_order_slicing():
    """Subtest 6.4: Intraday Execution Slicing Rule."""
    adv_val = 1_000_000.0
    s_small = ops.slice_intraday_order(4000.0, adv_val)
    assert not s_small["is_sliced"], "Small order (<= 0.5% ADV) should not be sliced"
    assert s_small["q_ato"] == 4000.0

    s_large = ops.slice_intraday_order(20000.0, adv_val)
    assert s_large["is_sliced"], "Large order (> 0.5% ADV) must be sliced"
    assert s_large["q_ato"] == 5000.0, f"Expected 5000 ATO, got {s_large['q_ato']}"
    assert s_large["q_twap"] == 11000.0, f"Expected 11000 TWAP, got {s_large['q_twap']}"
    assert s_large["q_atc"] == 4000.0, f"Expected 4000 ATC, got {s_large['q_atc']}"
    assert not s_large["cap_applied"]

    s_huge = ops.slice_intraday_order(50000.0, adv_val)
    assert s_huge["cap_applied"], "Order > 3% ADV must trigger participation cap"
    assert s_huge["q_total"] == 30000.0, f"Expected capped at 30,000, got {s_huge['q_total']}"


def test_phase3_hawkes_ou_continuous_decay():
    """Subtest 7.1: Hawkes / Ornstein-Uhlenbeck Continuous Exponential Decay."""
    T, N = 20, 2
    impulses = np.zeros((T, N), dtype=np.float64)
    impulses[0, 0] = 1.0
    impulses[0, 1] = 1.0

    decay_news = ops.apply_hawkes_ou_decay(impulses, tau_sessions=5.0)
    decay_debt = ops.apply_hawkes_ou_decay(impulses, tau_sessions=42.0)

    val_at_tau = decay_news[5, 0]
    expected_half = 0.50
    assert abs(val_at_tau - expected_half) < 1e-4, f"Expected 0.50 half-life decay at t=5, got {val_at_tau}"
    assert abs(decay_news[10, 0] - 0.25) < 1e-4, f"Expected 0.25 at t=10, got {decay_news[10, 0]}"
    assert decay_debt[5, 1] > 0.90, f"Expected debt decay > 0.90 at t=5, got {decay_debt[5, 1]}"


def test_phase3_vdw_transform():
    """Subtest 7.2: Cross-Sectional Van der Waerden Normal Score Transform."""
    np.random.seed(99)
    raw_scores = np.random.exponential(scale=2.0, size=(10, 30))
    vdw_scores = ops.van_der_waerden_transform(raw_scores)

    for t in range(10):
        row_mean = float(np.mean(vdw_scores[t]))
        row_std = float(np.std(vdw_scores[t]))
        assert abs(row_mean) < 1e-12, f"Van der Waerden mean is not 0: {row_mean}"
        assert abs(row_std - 1.0) < 0.05, f"Van der Waerden std is not ~1.0: {row_std}"
        assert np.all(np.abs(vdw_scores[t]) <= 3.0), "Values exceeded clip limit [-3.0, 3.0]"


def test_phase3_fingraph_parquet_bridge():
    """Subtest 7.3: Real VN-FinGraph Parquet Bridge (if available)."""
    fg_parquet = os.path.join(BASE_DIR, "fingraph", "data", "event_contagion_alpha.parquet")
    if os.path.exists(fg_parquet):
        df_fg = pl.read_parquet(fg_parquet)
        assert df_fg.shape[0] >= 900, f"Expected >= 900 sessions in event contagion, got {df_fg.shape[0]}"
        assert "ACB" in df_fg.columns and "VIC" in df_fg.columns, "VN30 tickers missing from event contagion"


def test_b6_no_look_ahead_close_to_ato_execution():
    """
    B6: Changing raw_close on day t must NOT alter ATO executed volume/trading on day t.
    Target weights calculated from close[t] are lagged to execute at ATO of day t+1.
    """
    np.random.seed(42)
    T, N = 20, 5
    raw_open = np.full((T, N), 50000.0)
    raw_close = np.full((T, N), 50000.0)
    ceiling = np.full((T, N), 53500.0)
    floor = np.full((T, N), 46500.0)
    adv = np.full((T, N), 1000000.0)

    # Signal generated from close
    alpha = np.random.randn(T, N)
    w_raw = ops.rank(alpha)
    w_raw = w_raw / np.sum(w_raw, axis=1, keepdims=True)

    # Causal lag: w_exec[t] = w_raw[t-1]
    w_exec = np.zeros_like(w_raw)
    w_exec[1:] = w_raw[:-1]
    w_exec[0] = 0.0

    res_orig = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv, w_exec)

    # Now perturb raw_close at day t0 = 10
    t0 = 10
    raw_close_pert = raw_close.copy()
    raw_close_pert[t0, :] *= 1.50  # 50% price change at close of day t0

    # Under causal execution, w_exec at day t0 only depends on day t0-1 close, so w_exec[:t0+1] is unchanged!
    res_pert = sim.simulate_vietnam(raw_open, raw_close_pert, ceiling, floor, adv, w_exec)

    # ATO executed turnover and trading fees up to morning of day t0 must be 100% IDENTICAL
    assert np.allclose(res_orig["realized_daily_turnover"][:t0 + 1], res_pert["realized_daily_turnover"][:t0 + 1]), \
        "B6 VIOLATION: Altering close[t] altered morning ATO trading on day t!"
    assert np.allclose(res_orig["broker_fees"][:t0 + 1], res_pert["broker_fees"][:t0 + 1]), \
        "B6 VIOLATION: Broker fees up to t0 ATO differ after altering close[t0]!"

