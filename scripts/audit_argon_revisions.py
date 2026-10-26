import os
import sys
import json
import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import polars as pl
from scipy import stats

from engine import alpha_ops as ops
from engine import data_pipeline as dp
from engine import vn_simulator as sim
from engine.ml_alpha_combiner import (
    generate_five_alphas,
    orthogonalize_factor_library,
    purged_walk_forward_combiner
)

def run_all_argon_audits():
    print("=" * 80)
    print("STARTING ARGON 4 QUANT AUDIT EXPERIMENTS & VERIFICATION")
    print("=" * 80)

    # 1. Load Data
    parquet_path = "data/vn30_production.parquet"
    df_market = pl.read_parquet(parquet_path)
    df_masked = dp.build_tradable_mask(df_market, min_adv=1.0e10, window=20)

    raw_close, dates, syms = dp.pivot_to_matrix(df_masked, "raw_close")
    raw_open, _, _ = dp.pivot_to_matrix(df_masked, "raw_open")
    raw_high, _, _ = dp.pivot_to_matrix(df_masked, "raw_high")
    raw_low, _, _ = dp.pivot_to_matrix(df_masked, "raw_low")
    volume, _, _ = dp.pivot_to_matrix(df_masked, "volume")
    adv20, _, _ = dp.pivot_to_matrix(df_masked, "adv20")
    ceiling, _, _ = dp.pivot_to_matrix(df_masked, "ceiling_price")
    floor, _, _ = dp.pivot_to_matrix(df_masked, "floor_price")
    tradable_mask, _, _ = dp.pivot_to_matrix(df_masked, "tradable_mask")

    adv20_shares = np.where(raw_close > 0, adv20 / np.maximum(raw_close, 1e-4), 1e6)
    T, N = raw_close.shape
    years = T / 252.0

    # 2. Benchmark VN30 Equal-Weight Buy & Hold
    # Equal weight across valid tradable stocks at session t
    w_bm = np.where(tradable_mask, 1.0, 0.0)
    w_bm = w_bm / np.maximum(np.sum(w_bm, axis=1, keepdims=True), 1e-6)
    w_bm_exec = np.zeros_like(w_bm)
    w_bm_exec[1:] = w_bm[:-1]
    res_bm = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_bm_exec, dates=dates, penalty_rate_per_day=0.0)
    m_bm = sim.evaluate_strategy(res_bm["equity"], w_bm_exec)

    # Calculate Newey-West standard error for Sharpe
    def newey_west_sharpe_se(returns, rf_daily=0.04/252.0, max_lags=5):
        excess = returns - rf_daily
        mu = np.mean(excess)
        sigma = np.std(excess, ddof=1)
        sr_daily = mu / sigma if sigma > 1e-12 else 0.0
        # GMM influence function: g_t = (R_t - mu)/sigma - 0.5 * sr_daily * ((R_t - mu)^2 / sigma^2 - 1)
        u_t = (excess - mu) / sigma
        g_t = u_t - 0.5 * sr_daily * (u_t**2 - 1.0)
        n = len(g_t)
        gamma_0 = np.var(g_t)
        gamma_sum = 0.0
        for l in range(1, max_lags + 1):
            w_l = 1.0 - l / (max_lags + 1.0)
            cov_l = np.mean(g_t[l:] * g_t[:-l])
            gamma_sum += 2.0 * w_l * cov_l
        lr_var = gamma_0 + gamma_sum
        se_daily = np.sqrt(max(lr_var, 1e-12) / n)
        se_ann = se_daily * np.sqrt(252.0)
        return se_ann

    ret_bm = np.diff(res_bm["equity"]) / np.maximum(res_bm["equity"][:-1], 1e-6)
    se_bm = newey_west_sharpe_se(ret_bm)

    # 3. Generate factors & combiner
    raw_alphas, factor_names = generate_five_alphas(raw_open, raw_high, raw_low, raw_close, volume, tradable_mask)
    ortho_alphas, raw_corr, ortho_corr = orthogonalize_factor_library(raw_alphas, factor_names, tradable_mask)
    composite_alpha, _ = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=120, embargo=3)
    composite_ranked = ops.rank(composite_alpha, mask=tradable_mask)

    w_single = ops.dampen_turnover(raw_alphas[0], 2, 4, 20, 0.015, tradable_mask)
    w_multi = ops.dampen_turnover(composite_ranked, 2, 4, 20, 0.015, tradable_mask)

    # 4. Scenarios: Before vs After B6-B11
    # Before
    res_s_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_single, penalty_rate_per_day=0.07)
    res_m_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_multi, penalty_rate_per_day=0.07)
    naive_s_b = sim.simulate_naive(raw_close, w_single)
    naive_m_b = sim.simulate_naive(raw_close, w_multi)
    m_s_b = sim.evaluate_strategy(res_s_b["equity"], w_single, naive_s_b)
    m_m_b = sim.evaluate_strategy(res_m_b["equity"], w_multi, naive_m_b)

    # After
    w_s_exec = np.zeros_like(w_single); w_s_exec[1:] = w_single[:-1]
    w_m_exec = np.zeros_like(w_multi); w_m_exec[1:] = w_multi[:-1]
    res_s_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_s_exec, dates=dates, penalty_rate_per_day=0.0)
    res_m_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_m_exec, dates=dates, penalty_rate_per_day=0.0)
    naive_s_a = sim.simulate_naive(raw_close, w_s_exec)
    naive_m_a = sim.simulate_naive(raw_close, w_m_exec)
    m_s_a = sim.evaluate_strategy(res_s_a["equity"], w_s_exec, naive_s_a)
    m_m_a = sim.evaluate_strategy(res_m_a["equity"], w_m_exec, naive_m_a)

    ret_s_a = np.diff(res_s_a["equity"]) / np.maximum(res_s_a["equity"][:-1], 1e-6)
    ret_m_a = np.diff(res_m_a["equity"]) / np.maximum(res_m_a["equity"][:-1], 1e-6)
    se_s_a = newey_west_sharpe_se(ret_s_a)
    se_m_a = newey_west_sharpe_se(ret_m_a)

    ret_s_b = np.diff(res_s_b["equity"]) / np.maximum(res_s_b["equity"][:-1], 1e-6)
    ret_m_b = np.diff(res_m_b["equity"]) / np.maximum(res_m_b["equity"][:-1], 1e-6)
    se_s_b = newey_west_sharpe_se(ret_s_b)
    se_m_b = newey_west_sharpe_se(ret_m_b)

    # [HIGH-01] Implementation Shortfall metrics
    # Cumulative IS % initial capital = (Final Naive NAV - Final Real NAV) / Capital_initial * 100
    is_cum_s_b = (naive_s_b[-1] - res_s_b["equity"][-1]) / 1.0e9 * 100.0
    is_cum_s_a = (naive_s_a[-1] - res_s_a["equity"][-1]) / 1.0e9 * 100.0
    is_cum_m_b = (naive_m_b[-1] - res_m_b["equity"][-1]) / 1.0e9 * 100.0
    is_cum_m_a = (naive_m_a[-1] - res_m_a["equity"][-1]) / 1.0e9 * 100.0

    # Annualized IS % = CAGR_Naive - CAGR_Real
    cagr_naive_s_b = (naive_s_b[-1] / 1.0e9)**(1.0/years) - 1.0
    cagr_naive_s_a = (naive_s_a[-1] / 1.0e9)**(1.0/years) - 1.0
    cagr_naive_m_b = (naive_m_b[-1] / 1.0e9)**(1.0/years) - 1.0
    cagr_naive_m_a = (naive_m_a[-1] / 1.0e9)**(1.0/years) - 1.0

    is_ann_s_b = (cagr_naive_s_b - m_s_b["cagr"]) * 100.0
    is_ann_s_a = (cagr_naive_s_a - m_s_a["cagr"]) * 100.0
    is_ann_m_b = (cagr_naive_m_b - m_m_b["cagr"]) * 100.0
    is_ann_m_a = (cagr_naive_m_a - m_m_a["cagr"]) * 100.0

    print("\n--- [HIGH-01] & [HIGH-02] CORE METRICS COMPARISON TABLE ---")
    table_data = [
        ("Final Real NAV (VND)", f"{res_s_b['equity'][-1]:,.0f}", f"{res_s_a['equity'][-1]:,.0f}", f"{res_m_b['equity'][-1]:,.0f}", f"{res_m_a['equity'][-1]:,.0f}", f"{res_bm['equity'][-1]:,.0f}"),
        ("Final Naive NAV (VND)", f"{naive_s_b[-1]:,.0f}", f"{naive_s_a[-1]:,.0f}", f"{naive_m_b[-1]:,.0f}", f"{naive_m_a[-1]:,.0f}", "N/A"),
        ("Tổng tỷ suất sinh lời (Real Return)", f"{m_s_b['total_return']*100:.2f}%", f"{m_s_a['total_return']*100:.2f}%", f"{m_m_b['total_return']*100:.2f}%", f"{m_m_a['total_return']*100:.2f}%", f"{m_bm['total_return']*100:.2f}%"),
        ("CAGR Thực tế (%/năm)", f"{m_s_b['cagr']*100:.2f}%", f"{m_s_a['cagr']*100:.2f}%", f"{m_m_b['cagr']*100:.2f}%", f"{m_m_a['cagr']*100:.2f}%", f"{m_bm['cagr']*100:.2f}%"),
        ("CAGR Naive (%/năm)", f"{cagr_naive_s_b*100:.2f}%", f"{cagr_naive_s_a*100:.2f}%", f"{cagr_naive_m_b*100:.2f}%", f"{cagr_naive_m_a*100:.2f}%", "N/A"),
        ("Sharpe Ratio (Newey-West SE)", f"{m_s_b['sharpe_ratio']:.2f} (±{se_s_b:.2f})", f"{m_s_a['sharpe_ratio']:.2f} (±{se_s_a:.2f})", f"{m_m_b['sharpe_ratio']:.2f} (±{se_m_b:.2f})", f"{m_m_a['sharpe_ratio']:.2f} (±{se_m_a:.2f})", f"{m_bm['sharpe_ratio']:.2f} (±{se_bm:.2f})"),
        ("Deflated Sharpe Ratio (DSR)", f"{m_s_b['deflated_sharpe_ratio']:.4f}", f"{m_s_a['deflated_sharpe_ratio']:.4f}", f"{m_m_b['deflated_sharpe_ratio']:.4f}", f"{m_m_a['deflated_sharpe_ratio']:.4f}", "N/A"),
        ("Sortino Ratio (Downside Risk)", f"{m_s_b['sortino_ratio']:.2f}", f"{m_s_a['sortino_ratio']:.2f}", f"{m_m_b['sortino_ratio']:.2f}", f"{m_m_a['sortino_ratio']:.2f}", f"{m_bm['sortino_ratio']:.2f}"),
        ("Biến động danh mục năm hóa (Vol)", f"{m_s_b['annualized_vol']*100:.2f}%", f"{m_s_a['annualized_vol']*100:.2f}%", f"{m_m_b['annualized_vol']*100:.2f}%", f"{m_m_a['annualized_vol']*100:.2f}%", f"{m_bm['annualized_vol']*100:.2f}%"),
        ("Mức sụt giảm tối đa (MaxDD)", f"{m_s_b['max_drawdown']*100:.2f}%", f"{m_s_a['max_drawdown']*100:.2f}%", f"{m_m_b['max_drawdown']*100:.2f}%", f"{m_m_a['max_drawdown']*100:.2f}%", f"{m_bm['max_drawdown']*100:.2f}%"),
        ("Turnover trung bình ngày (4 decimals)", f"{m_s_b['daily_turnover']*100:.4f}%", f"{m_s_a['daily_turnover']*100:.4f}%", f"{m_m_b['daily_turnover']*100:.4f}%", f"{m_m_a['daily_turnover']*100:.4f}%", f"{m_bm['daily_turnover']*100:.4f}%"),
        ("IS Lũy kế (% Vốn gốc 1.0 tỷ)", f"{is_cum_s_b:+.2f}%", f"{is_cum_s_a:+.2f}%", f"{is_cum_m_b:+.2f}%", f"{is_cum_m_a:+.2f}%", "N/A"),
        ("IS Năm hóa (%/năm)", f"{is_ann_s_b:+.2f}%", f"{is_ann_s_a:+.2f}%", f"{is_ann_m_b:+.2f}%", f"{is_ann_m_a:+.2f}%", "N/A"),
        ("Tổng phí môi giới & Thuế bán", f"{res_s_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['trading_fees'][-1]/1e6:.1f}M VND", f"{res_bm['trading_fees'][-1]/1e6:.1f}M VND"),
        ("Chi phí lãi vay ứng trước T+2", f"{res_s_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_bm['advance_interest_fees'][-1]/1e6:.1f}M VND")
    ]
    hdr = f"{'Chỉ số':<38} | {'Single (Trước)':<18} | {'Single (Sau)':<18} | {'Multi (Trước)':<18} | {'Multi (Sau)':<18} | {'Benchmark (VN30)':<18}"
    print(hdr)
    print("-" * len(hdr))
    for row in table_data:
        print(f"{row[0]:<38} | {row[1]:>18} | {row[2]:>18} | {row[3]:>18} | {row[4]:>18} | {row[5]:>18}")

    # [MEDIUM-02] & Q2: Advance Fee Detailed Breakdown
    # Track variable vs floor fee directly
    def decompose_advance_fee(weights):
        # Run simulation and count how many times floor fee 50k triggered vs rate
        # We can simulate the exact loop or re-run with min_advance_fee=0
        res_with_floor = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, weights, dates=dates, min_advance_fee=50000.0)
        res_no_floor = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, weights, dates=dates, min_advance_fee=0.0)
        total_adv_fee = res_with_floor["advance_interest_fees"][-1]
        pure_rate_fee = res_no_floor["advance_interest_fees"][-1]
        floor_penalty_fee = total_adv_fee - pure_rate_fee
        return total_adv_fee, pure_rate_fee, floor_penalty_fee

    adv_tot_s, adv_rate_s, adv_floor_s = decompose_advance_fee(w_s_exec)
    adv_tot_m, adv_rate_m, adv_floor_m = decompose_advance_fee(w_m_exec)

    print("\n--- [MEDIUM-02] & (Q2) ADVANCE INTEREST FEE DECOMPOSITION ---")
    print(f"Single-Alpha (Sau B6-B11):")
    print(f"  • Tổng phí ứng trước thực tế : {adv_tot_s:,.0f} VND (100.0%)")
    print(f"  • Phần lãi suất thuần (0.035%): {adv_rate_s:,.0f} VND ({adv_rate_s/adv_tot_s*100:.1f}%)")
    print(f"  • Phần phụ trội sàn (50k floor): {adv_floor_s:,.0f} VND ({adv_floor_s/adv_tot_s*100:.1f}%)")
    print(f"Multi-Alpha (Sau B6-B11):")
    print(f"  • Tổng phí ứng trước thực tế : {adv_tot_m:,.0f} VND (100.0%)")
    print(f"  • Phần lãi suất thuần (0.035%): {adv_rate_m:,.0f} VND ({adv_rate_m/adv_tot_m*100:.1f}%)")
    print(f"  • Phần phụ trội sàn (50k floor): {adv_floor_m:,.0f} VND ({adv_floor_m/adv_tot_m*100:.1f}%)")

    # [MEDIUM-03] & (Q4) DSR Detailed Parameters
    print("\n--- [MEDIUM-03] & (Q4) DSR EXACT PARAMETERS ---")
    dsr_s = sim.compute_deflated_sharpe_ratio(ret_s_a, n_trials=26)
    dsr_m = sim.compute_deflated_sharpe_ratio(ret_m_a, n_trials=26)
    print("Parameter                     | Single-Alpha (Sau) | Multi-Alpha (Sau)")
    print("-" * 65)
    print(f"Tổng số thử nghiệm (N)        | {dsr_s['n_trials']:>18} | {dsr_m['n_trials']:>17}")
    print(f"Số thử nghiệm hữu hiệu (N_eff)| {dsr_s['n_eff']:>18.1f} | {dsr_m['n_eff']:>17.1f}")
    print(f"Ngưỡng SR kỳ vọng null (SR_0) | {dsr_s['expected_max_sharpe']:>18.4f} | {dsr_m['expected_max_sharpe']:>17.4f}")
    print(f"Hệ số bất đối xứng (Skewness) | {dsr_s['sample_skewness']:>18.4f} | {dsr_m['sample_skewness']:>17.4f}")
    print(f"Độ nhọn phân phối (Kurtosis)  | {dsr_s['sample_kurtosis']:>18.4f} | {dsr_m['sample_kurtosis']:>17.4f}")
    print(f"Annualized Sharpe (Sau phí)   | {dsr_s['annualized_sharpe']:>18.4f} | {dsr_m['annualized_sharpe']:>17.4f}")
    print(f"Deflated Sharpe Ratio (DSR)   | {dsr_s['deflated_sharpe_ratio']:>18.4f} | {dsr_m['deflated_sharpe_ratio']:>17.4f}")
    print(f"p-value (Multiple Testing)    | {dsr_s['p_value']:>18.4f} | {dsr_m['p_value']:>17.4f}")

    # (Q5) Ablation study on all 5 Alpha factors
    print("\n--- (Q5) & CHECK 4: ABLATION STUDY OF 5 SINGLE ALPHAS ---")
    alpha_metrics = []
    for idx, (alpha_mat, name) in enumerate(zip(raw_alphas, factor_names)):
        w_factor = ops.dampen_turnover(alpha_mat, 2, 4, 20, 0.015, tradable_mask)
        w_f_exec = np.zeros_like(w_factor); w_f_exec[1:] = w_factor[:-1]
        res_f = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_f_exec, dates=dates, penalty_rate_per_day=0.0)
        m_f = sim.evaluate_strategy(res_f["equity"], w_f_exec)
        alpha_metrics.append({
            "name": name,
            "final_nav": res_f["equity"][-1],
            "total_ret": m_f["total_return"],
            "cagr": m_f["cagr"],
            "sharpe": m_f["sharpe_ratio"],
            "sortino": m_f["sortino_ratio"],
            "vol": m_f["annualized_vol"],
            "maxdd": m_f["max_drawdown"],
            "turnover": m_f["daily_turnover"]
        })
        print(f"Alpha {idx+1} ({name:<22}): Final NAV={res_f['equity'][-1]:,.0f} | CAGR={m_f['cagr']*100:.2f}% | Sharpe={m_f['sharpe_ratio']:.2f} | MaxDD={m_f['max_drawdown']*100:.2f}% | Turnover={m_f['daily_turnover']*100:.4f}%")

    # CHECK 1: Lag-Signal Decay Test: Delta t in {1, 2, 3, 5}
    print("\n--- CHECK 1: LAG-SIGNAL DECAY TEST (Delta t in {1, 2, 3, 5}) ---")
    lags = [1, 2, 3, 5]
    print(f"{'Lag Delta t':<12} | {'Single CAGR':<14} | {'Single Sharpe':<14} | {'Single MaxDD':<14} | {'Multi CAGR':<14} | {'Multi Sharpe':<14} | {'Multi MaxDD':<14}")
    print("-" * 95)
    for l in lags:
        w_sl = np.zeros_like(w_single); w_sl[l:] = w_single[:-l]
        w_ml = np.zeros_like(w_multi); w_ml[l:] = w_multi[:-l]
        r_sl = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_sl, dates=dates, penalty_rate_per_day=0.0)
        r_ml = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_ml, dates=dates, penalty_rate_per_day=0.0)
        m_sl = sim.evaluate_strategy(r_sl["equity"], w_sl)
        m_ml = sim.evaluate_strategy(r_ml["equity"], w_ml)
        print(f"{l} session(s)  | {m_sl['cagr']*100:>12.2f}% | {m_sl['sharpe_ratio']:>14.2f} | {m_sl['max_drawdown']*100:>12.2f}% | {m_ml['cagr']*100:>12.2f}% | {m_ml['sharpe_ratio']:>14.2f} | {m_ml['max_drawdown']*100:>12.2f}%")

    # CHECK 2: Cost & Minimum Advance Fee Stress Test
    print("\n--- CHECK 2: COST & MIN-ADVANCE-FEE STRESS TEST ---")
    # Base: slippage = 0.0010, min_advance_fee = 50000.0
    # Stress 1: Double Slippage = 0.0020 (20 bps)
    r_s_slip2 = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_s_exec, dates=dates, slippage=0.0020, penalty_rate_per_day=0.0)
    r_m_slip2 = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_m_exec, dates=dates, slippage=0.0020, penalty_rate_per_day=0.0)
    m_s_slip2 = sim.evaluate_strategy(r_s_slip2["equity"], w_s_exec)
    m_m_slip2 = sim.evaluate_strategy(r_m_slip2["equity"], w_m_exec)

    # Stress 2: No 50k floor fee
    r_s_nofloor = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_s_exec, dates=dates, min_advance_fee=0.0, penalty_rate_per_day=0.0)
    r_m_nofloor = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_m_exec, dates=dates, min_advance_fee=0.0, penalty_rate_per_day=0.0)
    m_s_nofloor = sim.evaluate_strategy(r_s_nofloor["equity"], w_s_exec)
    m_m_nofloor = sim.evaluate_strategy(r_m_nofloor["equity"], w_m_exec)

    print(f"Kịch bản Thử nghiệm                          | Single CAGR | Single Sharpe | Single NAV      | Multi CAGR  | Multi Sharpe  | Multi NAV")
    print("-" * 115)
    print(f"Chuẩn (Slippage=10bps, Floor=50k)            | {m_s_a['cagr']*100:>9.2f}% | {m_s_a['sharpe_ratio']:>13.2f} | {res_s_a['equity'][-1]:>15,.0f} | {m_m_a['cagr']*100:>9.2f}% | {m_m_a['sharpe_ratio']:>13.2f} | {res_m_a['equity'][-1]:>15,.0f}")
    print(f"Gấp đôi trượt giá (Slippage=20bps, Floor=50k)| {m_s_slip2['cagr']*100:>9.2f}% | {m_s_slip2['sharpe_ratio']:>13.2f} | {r_s_slip2['equity'][-1]:>15,.0f} | {m_m_slip2['cagr']*100:>9.2f}% | {m_m_slip2['sharpe_ratio']:>13.2f} | {r_m_slip2['equity'][-1]:>15,.0f}")
    print(f"Tắt phí sàn 50k (Slippage=10bps, Floor=0)    | {m_s_nofloor['cagr']*100:>9.2f}% | {m_s_nofloor['sharpe_ratio']:>13.2f} | {r_s_nofloor['equity'][-1]:>15,.0f} | {m_m_nofloor['cagr']*100:>9.2f}% | {m_m_nofloor['sharpe_ratio']:>13.2f} | {r_m_nofloor['equity'][-1]:>15,.0f}")

    # CHECK 3: Yearly Regime Split (2023, 2024, 2025, 2026)
    print("\n--- CHECK 3: YEARLY SUBSAMPLE & REGIME SPLIT ---")
    years_list = ["2023", "2024", "2025", "2026"]
    date_strs = [str(d) for d in dates]
    print(f"{'Năm':<6} | {'Số phiên':<8} | {'Single Return':<13} | {'Single Sharpe':<13} | {'Single MaxDD':<13} | {'Multi Return':<13} | {'Multi Sharpe':<13} | {'Multi MaxDD':<13} | {'VN30 Return':<13}")
    print("-" * 115)
    for yr in years_list:
        mask_yr = [i for i, d in enumerate(date_strs) if d.startswith(yr)]
        if not mask_yr:
            continue
        idx_start, idx_end = mask_yr[0], mask_yr[-1]
        n_bars = len(mask_yr)
        
        # Sub-curve return
        eq_s_sub = res_s_a["equity"][idx_start:idx_end+1]
        eq_m_sub = res_m_a["equity"][idx_start:idx_end+1]
        eq_bm_sub = res_bm["equity"][idx_start:idx_end+1]
        
        ret_s_sub = (eq_s_sub[-1] / eq_s_sub[0]) - 1.0
        ret_m_sub = (eq_m_sub[-1] / eq_m_sub[0]) - 1.0
        ret_bm_sub = (eq_bm_sub[-1] / eq_bm_sub[0]) - 1.0
        
        # Daily returns in sub-period
        r_s_d = np.diff(eq_s_sub) / eq_s_sub[:-1]
        r_m_d = np.diff(eq_m_sub) / eq_m_sub[:-1]
        
        sr_s_sub = (np.mean(r_s_d) * 252.0 - 0.04) / (np.std(r_s_d) * np.sqrt(252.0)) if np.std(r_s_d) > 1e-6 else 0.0
        sr_m_sub = (np.mean(r_m_d) * 252.0 - 0.04) / (np.std(r_m_d) * np.sqrt(252.0)) if np.std(r_m_d) > 1e-6 else 0.0
        
        # MaxDD in sub-period
        def get_maxdd(arr):
            cm = np.maximum.accumulate(arr)
            return float(np.max((cm - arr) / cm))
            
        dd_s = get_maxdd(eq_s_sub)
        dd_m = get_maxdd(eq_m_sub)
        
        print(f"{yr:<6} | {n_bars:<8} | {ret_s_sub*100:>11.2f}% | {sr_s_sub:>13.2f} | {dd_s*100:>11.2f}% | {ret_m_sub*100:>11.2f}% | {sr_m_sub:>13.2f} | {dd_m*100:>11.2f}% | {ret_bm_sub*100:>11.2f}%")

    # CHECK 4: Graph Diffusion Parameter alpha Sensitivity Test
    print("\n--- [MEDIUM-01] & CHECK 4: GRAPH DIFFUSION ALPHA SENSITIVITY TEST ---")
    # Test Laplacian diffusion with alpha in {0.15, 0.25, 0.30, 0.50, 0.70}
    # Load knowledge graph if available or simulate synthetic shock on VIC
    kg_path = Path("local_docs/fingraph/knowledge_graph_engine.py")
    if kg_path.exists():
        sys.path.insert(0, str(kg_path.parent))
        try:
            from knowledge_graph_engine import FinancialKnowledgeGraph
            kg = FinancialKnowledgeGraph()
            alphas_test = [0.15, 0.25, 0.30, 0.50, 0.70]
            print(f"Hệ số alpha | Độ hấp thụ tại tâm (1-alpha) | Tỷ lệ lan truyền mạng | Bán kính phổ rho(alpha*S) | Tác động lan truyền lên VCB")
            print("-" * 105)
            for a in alphas_test:
                rho = a * float(np.max(np.abs(np.linalg.eigvals(kg.A_tilde))))
                res_vic = kg.simulate_shock("VIC", intensity=-1.0, alpha=a)
                vcb_spill = next(imp["alpha_spillover"] for imp in res_vic["impacts"] if imp["symbol"] == "VCB")
                print(f"alpha={a:<4.2f} | {1.0-a:>26.1%} | {a:>21.1%} | {rho:>25.4f} | {vcb_spill:>27.4f}")
        except Exception as e:
            print("Could not run KG direct test:", e)

    print("\n" + "=" * 80)
    print("ALL AUDIT EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_argon_audits()
