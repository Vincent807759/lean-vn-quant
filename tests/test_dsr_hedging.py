"""
tests/test_dsr_hedging.py - Phase 4 Dynamic VN30F Beta Hedge & Calibrated DSR Tests
Migrated from run_verification.py (Test 8) into pytest suite.
"""
import numpy as np
import pytest
from engine import vn_simulator as sim


def test_phase4_dynamic_regime_switch_hedge():
    """Subtest 8.1: Dynamic Contagion Regime Switch (Logistic Hedge Overlay)."""
    p_uptrend = np.linspace(1000, 1250, 70)
    p_shock = np.linspace(1250, 950, 70)

    h_up = sim.calculate_dynamic_vn30f_hedge(
        portfolio_value=1.0e9,
        beta_cw=0.98,
        futures_price=1280.0,
        lambda_contagion=0.40,
        vn30_close_history=p_uptrend
    )
    assert h_up["market_regime_M"] == 0, f"Expected regime 0 for uptrend, got {h_up['market_regime_M']}"
    assert h_up["contracts_to_short"] == 0, f"Uptrend must not short futures! Got {h_up['contracts_to_short']}"
    assert h_up["h_actual"] == 0.0, "Target hedge should be 0.0 in uptrend with lambda < 0.55"

    h_shock = sim.calculate_dynamic_vn30f_hedge(
        portfolio_value=1.0e9,
        beta_cw=0.98,
        futures_price=1280.0,
        lambda_contagion=0.40,
        vn30_close_history=p_shock
    )
    assert h_shock["market_regime_M"] >= 1, f"Expected regime >= 1 for breakdown, got {h_shock['market_regime_M']}"
    assert h_shock["contracts_to_short"] > 0, "Breakdown must activate futures short hedge!"
    assert h_shock["h_actual"] > 0.25, f"Target hedge should activate, got {h_shock['h_actual']}"

    h_small_move = sim.calculate_dynamic_vn30f_hedge(
        portfolio_value=1.0e9,
        beta_cw=0.98,
        futures_price=1280.0,
        lambda_contagion=0.42,
        vn30_close_history=p_shock,
        prev_h=h_shock["h_actual"]
    )
    assert h_small_move["h_actual"] == h_shock["h_actual"], "Deadband should have filtered small noise variation!"

    h_extreme = sim.calculate_dynamic_vn30f_hedge(
        portfolio_value=1.0e9,
        beta_cw=1.50,
        futures_price=1280.0,
        lambda_contagion=0.95,
        vn30_close_history=p_shock
    )
    max_allowed = int(np.floor((1.0e9 * 0.18) / (0.30 * 1280.0 * 100000.0)))
    assert h_extreme["contracts_to_short"] <= max_allowed, f"Contracts {h_extreme['contracts_to_short']} exceeded buffer limit {max_allowed}"


def test_phase4_calibrated_dsr_formulation_and_neff():
    """Subtest 8.2: Calibrated Deflated Sharpe Ratio (DSR) & Effective Trial Count N_eff."""
    np.random.seed(123)
    T_sim = 2100

    N_trials = 26
    common_factor = np.random.randn(T_sim)
    trial_returns = np.zeros((T_sim, N_trials))
    for j in range(N_trials):
        trial_returns[:, j] = 0.0006 + 0.009 * (np.sqrt(0.72) * common_factor + np.sqrt(0.28) * np.random.randn(T_sim))

    dsr_calibrated = sim.compute_deflated_sharpe_ratio(
        daily_returns=trial_returns[:, 0],
        n_trials=N_trials,
        trial_returns=trial_returns
    )

    n_eff_eig = dsr_calibrated["n_eff"]
    assert 1.0 <= n_eff_eig <= 10.0, f"Expected N_eff from eigenvalues in [1.0, 10.0], got {n_eff_eig}"

    # Calibrated via average correlation rho = 0.72
    dsr_rho = sim.compute_deflated_sharpe_ratio(
        daily_returns=trial_returns[:, 0],
        n_trials=N_trials,
        mean_trial_correlation=0.72,
        var_sharpe_trials=0.25**2
    )
    n_eff_rho = dsr_rho["n_eff"]
    assert abs(n_eff_rho - 8.0) < 1e-4, f"Expected N_eff = 8.0 with rho=0.72, got {n_eff_rho}"

    sr_0_calib = dsr_rho["expected_max_sharpe"]
    assert 0.20 <= sr_0_calib <= 0.60, f"Expected calibrated SR_0 in [0.20, 0.60], got {sr_0_calib}"

    high_sr_returns = 0.00095 + 0.010 * np.random.randn(T_sim)
    dsr_multi = sim.compute_deflated_sharpe_ratio(
        daily_returns=high_sr_returns,
        n_trials=N_trials,
        mean_trial_correlation=0.72,
        var_sharpe_trials=0.25**2
    )
    assert dsr_multi["deflated_sharpe_ratio"] >= 0.95, f"Multi-alpha DSR {dsr_multi['deflated_sharpe_ratio']} < 0.95"
    assert dsr_multi["p_value"] <= 0.05, f"Multi-alpha p-value {dsr_multi['p_value']} > 0.05"


def test_b11_trial_registry_integration(tmp_path):
    """
    B11: Test trial registry logging and calibrated DSR calculation from JSONL registry file.
    """
    registry_file = str(tmp_path / "trial_registry.jsonl")
    np.random.seed(42)
    T = 100

    r1 = np.random.normal(0.001, 0.01, T)
    r2 = 0.8 * r1 + 0.6 * np.random.normal(0.001, 0.01, T)
    r3 = 0.8 * r1 + 0.6 * np.random.normal(0.001, 0.01, T)

    sim.record_trial("alpha_1", r1, registry_file=registry_file)
    sim.record_trial("alpha_2", r2, registry_file=registry_file)

    res = sim.compute_dsr_from_registry(r3, registry_file=registry_file)
    assert res["n_trials"] == 3
    assert 1.0 <= res["n_eff"] <= 3.0
    assert 0.0 <= res["deflated_sharpe_ratio"] <= 1.0

