"""
tests/test_alpha_ops.py - Hermetic unit tests for alpha_ops and PIT fundamental alignment
Migrated from run_verification.py (Test 1 & Test 2) into pytest format.
"""
import datetime
import numpy as np
import polars as pl
import pytest
from engine import alpha_ops as ops
from engine import data_pipeline as dp


def test_look_ahead_bias_ts_operators():
    np.random.seed(101)
    T, N = 100, 10
    t0 = 45  # Cutoff point

    X = np.random.randn(T, N)
    Y = np.random.randn(T, N)

    # Perturb ONLY future data (t > t0)
    X_pert = X.copy()
    Y_pert = Y.copy()
    X_pert[t0 + 1:, :] += 10000.0
    Y_pert[t0 + 1:, :] -= 5000.0

    ts_operators = [
        ("ts_delay(5)", lambda arr: ops.ts_delay(arr, 5)),
        ("ts_delta(5)", lambda arr: ops.ts_delta(arr, 5)),
        ("ts_mean(10)", lambda arr: ops.ts_mean(arr, 10)),
        ("ts_std_dev(10)", lambda arr: ops.ts_std_dev(arr, 10)),
        ("ts_rank(10)", lambda arr: ops.ts_rank(arr, 10)),
        ("ts_decay_linear(10)", lambda arr: ops.ts_decay_linear(arr, 10)),
        ("ts_corr(10)", lambda arr: ops.ts_corr(arr, Y if arr is X else Y_pert, 10)),
    ]

    for name, op_func in ts_operators:
        out_orig = op_func(X)
        out_pert = op_func(X_pert)
        # Values at t <= t0 must be 100% identical
        assert np.allclose(out_orig[:t0 + 1], out_pert[:t0 + 1], equal_nan=True), f"Look-ahead bias in {name}"


def test_look_ahead_bias_cs_operators():
    np.random.seed(101)
    T, N = 100, 10
    t0 = 45

    X = np.random.randn(T, N)
    groups = np.random.randint(1, 4, size=N)

    X_pert = X.copy()
    X_pert[t0 + 1:, :] += 10000.0

    cs_operators = [
        ("rank", lambda arr: ops.rank(arr)),
        ("zscore(3.0)", lambda arr: ops.zscore(arr, clip=3.0)),
        ("scale(1.0)", lambda arr: ops.scale(arr, target_sum=1.0)),
        ("group_neutralize", lambda arr: ops.group_neutralize(arr, groups)),
    ]

    for name, op_func in cs_operators:
        out_orig = op_func(X)
        out_pert = op_func(X_pert)
        assert np.allclose(out_orig[:t0 + 1], out_pert[:t0 + 1], equal_nan=True), f"Look-ahead bias in {name}"


def test_align_pit_fundamentals_no_look_ahead():
    calendar_dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]
    df_trading = pl.DataFrame({
        "date": calendar_dates,
        "symbol": ["HPG"] * 120,
        "price": [25000.0 + i * 50.0 for i in range(120)]
    })

    # Financial statements: Q1 effective on 2023-04-20, Q2 on 2023-08-15
    df_fin = pl.DataFrame({
        "period_end_date": [datetime.date(2023, 3, 31), datetime.date(2023, 6, 30)],
        "symbol": ["HPG", "HPG"],
        "report_type": ["Q1", "SEMI"],
        "actual_publish_date": [datetime.date(2023, 4, 20), datetime.date(2023, 8, 15)],
        "roe": [0.15, 0.22]
    })

    joined_orig = dp.align_pit_fundamentals(df_trading, df_fin)

    # Alter future Q2 ROE
    df_fin_pert = pl.DataFrame({
        "period_end_date": [datetime.date(2023, 3, 31), datetime.date(2023, 6, 30)],
        "symbol": ["HPG", "HPG"],
        "report_type": ["Q1", "SEMI"],
        "actual_publish_date": [datetime.date(2023, 4, 20), datetime.date(2023, 8, 15)],
        "roe": [0.15, 0.99]
    })

    joined_pert = dp.align_pit_fundamentals(df_trading, df_fin_pert)

    # Historical dates before Q2 publish date (2023-08-15) must remain 100% untouched
    cutoff_date = datetime.date(2023, 8, 15)
    hist_orig = joined_orig.filter(pl.col("date") < cutoff_date)["roe"].to_list()
    hist_pert = joined_pert.filter(pl.col("date") < cutoff_date)["roe"].to_list()

    assert hist_orig == hist_pert


def test_orthogonalization_crushes_correlation():
    np.random.seed(42)
    T, N = 100, 20

    base_alpha = np.random.randn(T, N)
    noise = np.random.randn(T, N)
    candidate_alpha = 0.86 * base_alpha + np.sqrt(1.0 - 0.86**2) * noise

    r_init = np.corrcoef(candidate_alpha.ravel(), base_alpha.ravel())[0, 1]
    assert r_init > 0.80

    ortho_alpha, max_corr_before, max_corr_after = ops.check_and_orthogonalize(
        new_alpha=candidate_alpha,
        existing_alphas=[base_alpha],
        threshold=0.50
    )
    assert max_corr_after < 0.50
    assert max_corr_after < 1e-10


def test_ridge_factor_projection_permutation_invariance():
    np.random.seed(42)
    T, N = 100, 20

    base_alpha = np.random.randn(T, N)
    f1 = base_alpha
    f2 = 0.40 * base_alpha + np.random.randn(T, N) * 0.90
    candidate_alpha = 0.86 * base_alpha + np.sqrt(1.0 - 0.86**2) * np.random.randn(T, N)

    ortho_p1, _, corr_p1 = ops.ridge_factor_projection(candidate_alpha, [f1, f2], threshold=0.10)
    ortho_p2, _, corr_p2 = ops.ridge_factor_projection(candidate_alpha, [f2, f1], threshold=0.10)

    perm_diff = float(np.max(np.abs(ortho_p1 - ortho_p2)))
    assert perm_diff < 1e-10
    assert corr_p1 < 0.05


def test_b7_orthogonalization_strict_causality_no_look_ahead():
    """
    B7: Cross-Sectional QR Orthogonalization must be strictly causal.
    Perturbing future data at t > t0 must have ZERO impact on orthogonalized alpha at t <= t0.
    """
    np.random.seed(101)
    T, N = 80, 15
    t0 = 35

    base_alpha = np.random.randn(T, N)
    noise = np.random.randn(T, N)
    candidate_alpha = 0.80 * base_alpha + 0.60 * noise

    # Compute baseline orthogonalization
    ortho_orig, _, _ = ops.check_and_orthogonalize(
        new_alpha=candidate_alpha,
        existing_alphas=[base_alpha],
        threshold=0.30
    )

    # Perturb candidate and base alpha strictly at t > t0
    cand_pert = candidate_alpha.copy()
    base_pert = base_alpha.copy()
    cand_pert[t0 + 1:, :] += 9999.0
    base_pert[t0 + 1:, :] -= 5555.0

    ortho_pert, _, _ = ops.check_and_orthogonalize(
        new_alpha=cand_pert,
        existing_alphas=[base_pert],
        threshold=0.30
    )

    # Values at t <= t0 must be 100% IDENTICAL (max error = 0.0)
    max_diff = float(np.max(np.abs(ortho_orig[:t0 + 1] - ortho_pert[:t0 + 1])))
    assert max_diff == 0.0, f"B7 VIOLATION: Look-ahead leakage detected in orthogonalization, max_diff={max_diff}"


def test_ridge_and_lowdin_strict_causality_no_look_ahead():
    """
    C1 Causality / Future Perturbation Test:
    Verifies that neither ridge_factor_projection nor lowdin_symmetric_orthogonalization
    leaks future data: perturbing data at t > t0 must have 0.0 effect on results at t <= t0.
    """
    np.random.seed(102)
    T, N = 80, 15
    t0 = 35

    base_f1 = np.random.randn(T, N)
    base_f2 = 0.50 * base_f1 + 0.50 * np.random.randn(T, N)
    cand_alpha = 0.70 * base_f1 + 0.30 * np.random.randn(T, N)

    # 1. Baseline Ridge Projection
    ridge_orig, _, _ = ops.ridge_factor_projection(
        alpha_target=cand_alpha,
        benchmark_factors=[base_f1, base_f2],
        threshold=0.10
    )

    # Perturb future data strictly at t > t0
    cand_pert = cand_alpha.copy()
    base_f1_pert = base_f1.copy()
    base_f2_pert = base_f2.copy()
    cand_pert[t0 + 1:, :] += 12345.0
    base_f1_pert[t0 + 1:, :] -= 67890.0
    base_f2_pert[t0 + 1:, :] += 45678.0

    ridge_pert, _, _ = ops.ridge_factor_projection(
        alpha_target=cand_pert,
        benchmark_factors=[base_f1_pert, base_f2_pert],
        threshold=0.10
    )

    diff_ridge = float(np.max(np.abs(ridge_orig[:t0 + 1] - ridge_pert[:t0 + 1])))
    assert diff_ridge < 1e-12, f"C1 VIOLATION: Look-ahead detected in ridge_factor_projection! diff={diff_ridge}"

    # 2. Baseline Löwdin Orthogonalization
    lowdin_orig = ops.lowdin_symmetric_orthogonalization([base_f1, base_f2, cand_alpha])
    lowdin_pert = ops.lowdin_symmetric_orthogonalization([base_f1_pert, base_f2_pert, cand_pert])

    for k in range(len(lowdin_orig)):
        diff_lowdin_k = float(np.max(np.abs(lowdin_orig[k][:t0 + 1] - lowdin_pert[k][:t0 + 1])))
        assert diff_lowdin_k < 1e-12, f"C1 VIOLATION: Look-ahead detected in Löwdin factor {k}! diff={diff_lowdin_k}"

def test_check_and_orthogonalize_per_session_gate_boundary_causality():
    """
    M5.2: Verifies that per_session_gate=True evaluates threshold per session t,
    eliminating boundary look-ahead / truncation drift on synthetic series,
    while per_session_gate=False reproduces whole-sample gate past difference.
    """
    import numpy as np
    import engine.alpha_ops as ops
    import engine.ml_alpha_combiner as ml_comb

    rng = np.random.default_rng(20261107)
    T, N, t0 = 200, 30, 100
    ex1 = rng.standard_normal((T, N))
    noise = rng.standard_normal((T, N)) * 0.85

    # [0..99] positively correlated (~0.42 >= 0.25)
    new_a = 0.40 * ex1 + noise
    # In new_b: [100..199] negatively correlated (-0.40) so pooled whole-sample corr < 0.25
    new_b = new_a.copy()
    new_b[t0:, :] = -0.40 * ex1[t0:, :] + noise[t0:, :]

    # Negative assertion (per_session_gate=False): whole-sample gate causes past diff > 1.0
    out_a_whole, _, _ = ops.check_and_orthogonalize(new_a, [ex1], threshold=0.25, per_session_gate=False)
    out_b_whole, _, _ = ops.check_and_orthogonalize(new_b, [ex1], threshold=0.25, per_session_gate=False)
    diff_whole = float(np.max(np.abs(out_a_whole[:t0] - out_b_whole[:t0])))
    assert diff_whole > 1.0, f"Expected past diff > 1.0 for whole-sample gate, got {diff_whole}"

    # Positive assertion (per_session_gate=True): zero past difference
    out_a_sess, _, _ = ops.check_and_orthogonalize(new_a, [ex1], threshold=0.25, per_session_gate=True)
    out_b_sess, _, _ = ops.check_and_orthogonalize(new_b, [ex1], threshold=0.25, per_session_gate=True)
    diff_sess = float(np.max(np.abs(out_a_sess[:t0] - out_b_sess[:t0])))
    assert diff_sess == 0.0, f"Expected 0.0 past diff for per_session_gate=True, got {diff_sess}"

    # Check ml_alpha_combiner with per_session_gate=True
    tradable_mask = np.ones((T, N), dtype=bool)
    ortho_a_list, _, _ = ml_comb.orthogonalize_factor_library([ex1, new_a], ["F1", "F2"], tradable_mask, threshold=0.25, per_session_gate=True)
    ortho_b_list, _, _ = ml_comb.orthogonalize_factor_library([ex1, new_b], ["F1", "F2"], tradable_mask, threshold=0.25, per_session_gate=True)
    diff_comb_sess = float(np.max(np.abs(ortho_a_list[1][:t0] - ortho_b_list[1][:t0])))
    assert diff_comb_sess == 0.0, f"Expected 0.0 comb diff, got {diff_comb_sess}"

    # Check per-session gate mechanics:
    # When corr < threshold, slice is untouched
    zero_corr_slice = rng.standard_normal((1, N))
    zero_ex = rng.standard_normal((1, N))
    actual_c = abs(float(np.corrcoef(zero_corr_slice[0], zero_ex[0])[0, 1]))
    zero_out, _, _ = ops.check_and_orthogonalize(zero_corr_slice, [zero_ex], threshold=actual_c + 0.10, per_session_gate=True)
    assert np.allclose(zero_out, zero_corr_slice, atol=1e-12)

    # When corr >= threshold, cross-sectional corr after projection < 1e-10
    high_corr_slice = zero_ex.copy() + 0.01 * rng.standard_normal((1, N))
    actual_high_c = abs(float(np.corrcoef(high_corr_slice[0], zero_ex[0])[0, 1]))
    high_out, _, _ = ops.check_and_orthogonalize(high_corr_slice, [zero_ex], threshold=actual_high_c - 0.10, per_session_gate=True)
    corr_after = abs(float(np.corrcoef(high_out[0], zero_ex[0])[0, 1]))
    assert corr_after < 1e-10

