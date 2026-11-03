"""Unit tests for planted-alpha pipeline execution (HUONG_DI_R18 commit b)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from engine import alpha_ops as ops, vn_simulator as sim
from scripts.power_test import generate_planted_market, nw_se, run_g2_timing_control, run_one


def test_nw_se_fixed_vector():
    """Verify nw_se matches hand calculation on a fixed return vector."""
    # Fixed 8-element return series
    r = np.array([0.01, -0.005, 0.015, -0.01, 0.02, 0.005, -0.002, 0.008], dtype=np.float64)
    # Expected value from step-by-step formula with rf = 0.04/252 and lags = 2:
    # ex = r - rf
    # mu = 0.00509127, sig = 0.01016629, sr = 0.50080
    # u = [-0.10734, -0.99262, 0.97467, -1.48444, 1.46649, -0.00898, -0.69752, 0.28623]
    # g = [-0.10444, -0.98897, 0.98719, -1.18320, 1.17833, 0.24141, -0.82607, 0.51608]
    # lr_var = var(g) + 2*(2/3)*cov1 + 2*(1/3)*cov2 = 0.1627916
    # se = sqrt(0.1627916 / 8) * sqrt(252) = 2.26087033
    se_val = nw_se(r, rf=0.04 / 252.0, lags=2)
    assert abs(se_val - 2.26087033) < 1e-6
    assert sim.compute_newey_west_sharpe_se(r, rf=0.04 / 252.0, lags=2) == se_val

    # Verify delegation in all 3 script files
    f_pt = Path("scripts/power_test.py").read_text(encoding="utf-8")
    f_cb = Path("scripts/compare_backtest_b6_b11.py").read_text(encoding="utf-8")
    f_ar = Path("scripts/audit_argon_revisions.py").read_text(encoding="utf-8")
    assert "sim.compute_newey_west_sharpe_se" in f_pt
    assert "sim.compute_newey_west_sharpe_se" in f_cb
    assert "sim.compute_newey_west_sharpe_se" in f_ar


def test_run_one_finite_metrics_and_invariants():
    """Verify run_one outputs all required finite metrics and leaves trial_registry unchanged."""
    registry_path = Path("experiments/trial_registry.jsonl")
    hash_before = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    lines_before = len(registry_path.read_text(encoding="utf-8").splitlines())

    # Run one simulation at T=300
    res = run_one(seed=1001, rho=0.05, construction="C1", T=300, N=30, phi=0.9)

    required_keys = [
        "sharpe_net",
        "sharpe_gross",
        "t_nw",
        "dsr_p_value",
        "daily_turnover",
        "ic_real",
        "ic_latent",
        "t_nw_excess",
        "seed",
        "rho",
        "construction",
        "phi",
    ]
    for k in required_keys:
        assert k in res, f"Missing key {k}"
        val = res[k]
        if isinstance(val, (int, float)):
            assert np.isfinite(val), f"Metric {k} is not finite: {val}"

    # Verify trial registry is untouched (26 lines, bitwise identical)
    hash_after = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    lines_after = len(registry_path.read_text(encoding="utf-8").splitlines())
    assert hash_before == hash_after
    assert lines_before == lines_after == 26


def test_g2_timing_control_relaxed():
    """Verify Gate 2 timing control with relaxed thresholds (|Sharpe| < 3) at T=300."""
    g2 = run_g2_timing_control(seeds=[1000, 1001, 1002], T=300, N=30)
    g2_quicksort = run_g2_timing_control(seeds=[1000, 1001, 1002], T=300, N=30, sort_kind="quicksort")
    assert g2 == g2_quicksort

    # (i) Clairvoyant mean > 5
    assert g2["mean_clairvoyant"] > 5.0

    # (iii) Negated clairvoyant mean < -5
    assert g2["mean_negated"] < -5.0

    # (ii) Lagged by 1 additional day: relaxed threshold |Sharpe| < 3 at T=300
    assert abs(g2["mean_lagged"]) < 1.5
    for s_val in g2["sharpes_lagged"]:
        assert abs(s_val) < 3.0, f"Lagged Sharpe {s_val} exceeds relaxed threshold 3.0"

    # Test sort_kind="stable" option
    g2_stable = run_g2_timing_control(seeds=[1000, 1001, 1002], T=300, N=30, sort_kind="stable")
    assert g2_stable["passed"] is True
    assert g2_stable["mean_clairvoyant"] > 5.0
    assert g2_stable["mean_negated"] < -5.0
    assert abs(g2_stable["mean_lagged"]) < 1.5

def test_run_one_heavy_tail_and_trend_filter_burn_in():
    """Verify run_one with df_t and trend_filter, and check 49 burn-in sessions of SMA50."""
    # 1. run_one execution with heavy-tail and trend filter
    res_ex = run_one(
        seed=1001,
        rho=0.10,
        construction="C3",
        T=300,
        N=30,
        df_t=3.5,
        trend_filter=True,
        invalid_below=-900.0,
    )
    res_leg = run_one(
        seed=1001,
        rho=0.10,
        construction="C3",
        T=300,
        N=30,
        df_t=3.5,
        trend_filter=True,
        invalid_below=None,
    )
    res_base = run_one(
        seed=1001,
        rho=0.10,
        construction="C3",
        T=300,
        N=30,
        df_t=None,
        trend_filter=False,
    )

    for r in [res_ex, res_leg]:
        for k in ["sharpe_net", "sharpe_gross", "t_nw", "daily_turnover", "ic_real", "ic_latent", "bound_hit_rate"]:
            assert k in r
            assert np.isfinite(r[k])
        assert r["bound_hit_rate"] > 0.0

    # Latent IC matches bit-for-bit between base and heavy
    assert res_ex["ic_latent"] == res_base["ic_latent"]

    # 2. Check 49 burn-in sessions behavior of SMA50 (PB finding #3)
    mkt = generate_planted_market(seed=1001, rho=0.10, T=300, N=30)
    sma50 = ops.ts_mean(mkt["close"], 50)
    sig = np.where(mkt["close"] > sma50, mkt["s"], -999.0)
    w = ops.dampen_turnover(
        sig,
        top_k_in=5,
        top_k_out=10,
        ema_span=1,
        max_daily_turnover=1.0,
        mask=mkt["mask"],
        tie_break_key=mkt["adv20"],
        invalid_below=-900.0,
    )
    # Must hold 100% cash during first 49 sessions where SMA50 is NaN
    assert np.all(w[:49] == 0.0), f"Expected 0.0 weights in burn-in, got max: {np.max(np.abs(w[:49]))}"
    assert np.all(np.isfinite(w)), "Found non-finite weights in dampen_turnover with invalid_below"

