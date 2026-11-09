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
def test_compute_dsr_from_registry_active_returns_unit_consistency_and_benchmark_filter(tmp_path):
    """
    M2.1: Verify that compute_dsr_from_registry with benchmark_returns:
    1) Filters out degenerate zero-variance active columns (such as benchmark_vn30 - bm == 0)
       without raising LinAlgError in np.linalg.eigvalsh;
    2) Evaluates both numerator (daily_returns = r - bm) and denominator (trial_returns = matrix - bm)
       in active-return units with risk_free_rate = 0.0.
    """
    registry_file = str(tmp_path / "trial_registry.jsonl")
    rng = np.random.default_rng(123)
    T = 120

    bm = rng.normal(0.0004, 0.012, T)
    r1 = bm + rng.normal(0.0005, 0.006, T)
    r2 = bm + 0.5 * (r1 - bm) + rng.normal(0.0001, 0.006, T)
    r3 = bm + rng.normal(-0.0001, 0.007, T)

    sim.record_trial("benchmark_vn30", bm, metadata={"type": "benchmark"}, registry_file=registry_file)
    sim.record_trial("alpha_1", r1, registry_file=registry_file)
    sim.record_trial("alpha_2", r2, registry_file=registry_file)
    sim.record_trial("alpha_3", r3, registry_file=registry_file)

    res = sim.compute_dsr_from_registry(
        daily_returns=r1,
        registry_file=registry_file,
        risk_free_rate=0.04,
        benchmark_returns=bm
    )
    expected = sim.compute_deflated_sharpe_ratio(
        daily_returns=r1 - bm,
        n_trials=3,
        risk_free_rate=0.0,
        trial_returns=np.column_stack([r1 - bm, r2 - bm, r3 - bm])
    )

    assert res["n_trials"] == 3
    for k in ("deflated_sharpe_ratio", "p_value", "expected_max_sharpe", "annualized_sharpe", "n_eff"):
        assert np.isclose(res[k], expected[k], atol=1e-12), f"Mismatch on {k}: {res[k]} != {expected[k]}"


def test_compute_dsr_from_registry_canonical_active_mode():
    """
    M2.1: Verify compute_dsr_from_registry on the canonical 26-entry trial_registry.jsonl
    with benchmark_vn30 as benchmark_returns (filters 1 degenerate benchmark column -> 25 trials).
    """
    import json
    reg_path = "experiments/trial_registry.jsonl"
    by_name = {}
    with open(reg_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                by_name[obj["trial_name"]] = np.asarray(obj["returns"], dtype=np.float64)

    ret_bm = by_name["benchmark_vn30"]
    ret_s_a = by_name["single_after_b6_b11"]

    res = sim.compute_dsr_from_registry(
        daily_returns=ret_s_a,
        registry_file=reg_path,
        benchmark_returns=ret_bm
    )
    assert res["n_trials"] == 25
    assert np.isclose(res["n_eff"], 1.9879144564, atol=1e-4)
    assert np.isclose(res["expected_max_sharpe"], 0.2564143934, atol=1e-4)
    assert np.isclose(res["annualized_sharpe"], 0.6815294001, atol=1e-4)
    assert np.isclose(res["deflated_sharpe_ratio"], 0.7945034484, atol=1e-4)
def test_record_trial_idempotent_deduplication_and_overwrite(tmp_path):
    """
    M2.2a: Verify record_trial deduplicates by (trial_name, n_bars) when overwrite=False
    (leaving file bytes untouched) and replaces in-place preserving line order when overwrite=True.
    """
    import hashlib
    import json

    reg_path = tmp_path / "trial_registry.jsonl"
    reg_file = str(reg_path)
    r1 = np.array([0.01, -0.005, 0.002, 0.003], dtype=np.float64)
    r2 = np.array([0.002, 0.001, -0.001, 0.004], dtype=np.float64)
    r1_updated = np.array([0.02, 0.01, 0.005, 0.006], dtype=np.float64)

    sim.record_trial("trial_first", r1, registry_file=reg_file)
    sim.record_trial("trial_second", r2, registry_file=reg_file)
    sha_before = hashlib.sha256(reg_path.read_bytes()).hexdigest()

    # Calling record_trial again with default overwrite=False must not alter the file on disk
    returned = sim.record_trial("trial_first", r1_updated, registry_file=reg_file)
    sha_after = hashlib.sha256(reg_path.read_bytes()).hexdigest()
    assert sha_after == sha_before
    assert np.isclose(returned["mean_daily_return"], float(np.mean(r1)))

    # Calling with overwrite=True must update trial_first in-place at line 0 (preserving order)
    sim.record_trial("trial_first", r1_updated, registry_file=reg_file, overwrite=True)
    lines = [json.loads(line) for line in reg_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert [x["trial_name"] for x in lines] == ["trial_first", "trial_second"]
    assert np.isclose(lines[0]["mean_daily_return"], float(np.mean(r1_updated)))
def test_calculate_vn30f_hedge_static_overlay():
    """
    M4.2b: Unit test coverage for standalone static overlay calculate_vn30f_hedge.
    Covers exact contract sizing, under/over hedging rounding, beta <= 0, and degenerate inputs.
    """
    # 1. Exact sizing
    h = sim.calculate_vn30f_hedge(portfolio_value=1.0e9, beta=1.0, futures_price=1250.0)
    assert h["contracts_to_short"] == 8
    assert np.isclose(h["contract_value_vnd"], 1250.0 * 100000.0)
    assert np.isclose(h["total_hedge_notional_vnd"], 1.0e9)
    assert np.isclose(h["margin_required_vnd"], 2.0e8)
    assert np.isclose(h["original_beta"], 1.0)
    assert np.isclose(h["residual_beta"], 0.0)

    # 2. Under-hedged (round down)
    h_under = sim.calculate_vn30f_hedge(portfolio_value=1.0e9, beta=0.90, futures_price=1250.0)
    assert h_under["contracts_to_short"] == 7
    assert h_under["residual_beta"] > 0.0

    # 3. Over-hedged (round up)
    h_over = sim.calculate_vn30f_hedge(portfolio_value=1.0e9, beta=1.10, futures_price=1250.0)
    assert h_over["contracts_to_short"] == 9
    assert h_over["residual_beta"] < 0.0

    # 4. Small portfolio < half contract
    h_small = sim.calculate_vn30f_hedge(portfolio_value=5.0e7, beta=1.0, futures_price=1250.0)
    assert h_small["contracts_to_short"] == 0
    assert np.isclose(h_small["residual_beta"], 1.0)

    # 5. Non-positive beta
    h_zero = sim.calculate_vn30f_hedge(portfolio_value=1.0e9, beta=0.0, futures_price=1250.0)
    assert h_zero["contracts_to_short"] == 0
    h_neg = sim.calculate_vn30f_hedge(portfolio_value=1.0e9, beta=-0.5, futures_price=1250.0)
    assert h_neg["contracts_to_short"] == 0

    # 6. Degenerate inputs return all 6 standard keys without exception
    std_keys = {"contracts_to_short", "contract_value_vnd", "total_hedge_notional_vnd", "margin_required_vnd", "original_beta", "residual_beta"}
    for kwargs in [
        {"portfolio_value": -1.0, "beta": 1.0, "futures_price": 1250.0},
        {"portfolio_value": 1.0e9, "beta": 1.0, "futures_price": 0.0},
        {"portfolio_value": 1.0e9, "beta": 1.0, "futures_price": -100.0},
        {"portfolio_value": 1.0e9, "beta": 1.0, "futures_price": 1250.0, "multiplier": 0.0},
        {"portfolio_value": 1.0e9, "beta": 1.0, "futures_price": 1250.0, "margin_req": 0.0},
    ]:
        res_deg = sim.calculate_vn30f_hedge(**kwargs)
        assert std_keys.issubset(res_deg.keys())
        assert res_deg["contracts_to_short"] == 0


def test_load_and_decay_fingraph_alpha_standalone_bridge(tmp_path):
    """
    M4.2b: Unit test coverage for standalone overlay load_and_decay_fingraph_alpha.
    Covers missing file, missing date column, date/symbol matrix alignment, zero inputs, and exact transform.
    """
    import polars as pl
    from engine import ml_alpha_combiner as ml
    from engine import alpha_ops as ops

    # 1. FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ml.load_and_decay_fingraph_alpha("non_existent_path.parquet", ["2026-01-01"], ["HPG"])

    # 2. ValueError when missing required 'date' column
    bad_df = pl.DataFrame({"HPG": [1.0, 2.0]})
    bad_path = str(tmp_path / "missing_date.parquet")
    bad_df.write_parquet(bad_path)
    with pytest.raises(ValueError, match="missing required 'date' column"):
        ml.load_and_decay_fingraph_alpha(bad_path, ["2026-01-01"], ["HPG"])

    # 3. Alignment with pl.Utf8 and scrambled dates/symbols
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    df_utf8 = pl.DataFrame({
        "date": [dates[3], dates[0], dates[1]],  # scrambled, missing 2026-01-07
        "HPG": [0.5, 1.2, -0.4],
        "VNM": [1.1, -0.2, 0.8],
        "VIC": [0.0, 0.5, -0.5]
    })
    path_utf8 = str(tmp_path / "fingraph_utf8.parquet")
    df_utf8.write_parquet(path_utf8)

    target_dates = dates
    target_symbols = ["VIC", "HPG", "VNM", "MSN"]  # MSN not in parquet
    res_utf8 = ml.load_and_decay_fingraph_alpha(path_utf8, target_dates, target_symbols, tau_sessions=5.0)
    assert res_utf8.shape == (len(target_dates), len(target_symbols))

    # Check that bit-for-bit matches manual application
    raw_expected = np.zeros((4, 4), dtype=np.float64)
    raw_expected[0, :] = [0.5, 1.2, -0.2, 0.0]
    raw_expected[1, :] = [-0.5, -0.4, 0.8, 0.0]
    raw_expected[2, :] = [0.0, 0.0, 0.0, 0.0]
    raw_expected[3, :] = [0.0, 0.5, 1.1, 0.0]

    expected_alpha = ops.van_der_waerden_transform(ops.apply_hawkes_ou_decay(raw_expected, tau_sessions=5.0))
    assert np.allclose(res_utf8, expected_alpha, equal_nan=True, atol=1e-12)

    # 4. Uniform zero & tau_sessions <= 0
    zero_df = pl.DataFrame({"date": dates, "HPG": [0.0] * 4, "VNM": [0.0] * 4})
    path_zero = str(tmp_path / "fingraph_zero.parquet")
    zero_df.write_parquet(path_zero)
    res_zero = ml.load_and_decay_fingraph_alpha(path_zero, dates, ["HPG", "VNM"], tau_sessions=0.0)
    assert res_zero.shape == (4, 2)


def test_default_textbook_placeholder_signal_single_source_of_truth():
    """
    M4.2b: Single source of truth verification for default_textbook_placeholder_signal.
    Asserts bit-for-bit equality across engine/alpha_ops, run_verification, and services/daily_paper_trader,
    with zero RuntimeWarnings on startup t < 49 and robust handling of edge cases.
    """
    import warnings
    from engine import alpha_ops as ops
    import run_verification as r_verif
    import services.daily_paper_trader as d_paper

    rng = np.random.RandomState(42)
    T = 100
    N = 6
    raw_close = rng.lognormal(mean=10.0, sigma=0.05, size=(T, N))
    mask_bool = rng.rand(T, N) > 0.2

    raw_close[:, 0] = 50000.0
    raw_close[50:60, 1] = np.nan
    mask_bool[70, :] = False
    mask_bool[71, :] = False
    mask_bool[71, 2] = True
    mask_int = mask_bool.astype(int)

    with warnings.catch_warnings(record=True) as w_records:
        warnings.simplefilter("always")
        s_ops = ops.default_textbook_placeholder_signal(raw_close, mask_bool)
        s_verif = r_verif._default_textbook_placeholder_signal(raw_close, mask_bool)
        s_paper = d_paper._default_textbook_placeholder_signal(raw_close, mask_bool)
        s_int = ops.default_textbook_placeholder_signal(raw_close, mask_int)

        runtime_warns = [w for w in w_records if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) == 0, f"Unexpected RuntimeWarning: {runtime_warns}"

    assert np.array_equal(s_ops, s_verif, equal_nan=True)
    assert np.array_equal(s_ops, s_paper, equal_nan=True)
    assert np.array_equal(s_ops, s_int, equal_nan=True)
def test_dsr_registry_corrupted_line_emits_warning_and_run_verification_zero_warnings(tmp_path):
    """
    M6.3: Verifies that compute_dsr_from_registry emits a UserWarning when encountering
    a malformed JSON line in the registry file instead of failing silently, and that
    simulate_vietnam(..., warn_unlagged=False) suppresses false-positive unlagged warnings.
    """
    import warnings
    from engine import vn_simulator as sim

    bad_reg = tmp_path / "corrupted_registry.jsonl"
    bad_reg.write_text("{malformed_json_line\n", encoding="utf-8")

    rng = np.random.RandomState(7)
    daily_r = rng.normal(0.0005, 0.012, size=60)

    with pytest.warns(UserWarning, match="Failed to read trial registry"):
        res = sim.compute_dsr_from_registry(daily_r, registry_file=str(bad_reg))
    assert "deflated_sharpe_ratio" in res

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        ro = np.full((10, 2), 50000.0)
        rc = np.full((10, 2), 50000.0)
        ceil = np.full((10, 2), 53500.0)
        flr = np.full((10, 2), 46500.0)
        adv = np.full((10, 2), 1000000.0)
        w = np.full((10, 2), 0.5)
        sim.simulate_vietnam(ro, rc, ceil, flr, adv, w, initial_cash=1.0e9, warn_unlagged=False)
        assert len(caught) == 0
