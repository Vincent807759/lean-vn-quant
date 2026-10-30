import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import polars as pl
from engine import alpha_ops as ops
from engine import data_pipeline as dp
from engine import vn_simulator as sim
from engine.ml_alpha_combiner import generate_five_alphas, orthogonalize_factor_library, purged_walk_forward_combiner

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

# Generate factors & purged walk-forward combiner
raw_alphas, factor_names = generate_five_alphas(raw_open, raw_high, raw_low, raw_close, volume, tradable_mask)
ortho_alphas, raw_corr, ortho_corr = orthogonalize_factor_library(raw_alphas, factor_names, tradable_mask)
composite_alpha, _ = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=120, embargo=3)
composite_ranked = ops.rank(composite_alpha, mask=tradable_mask)

w_single = ops.dampen_turnover(raw_alphas[0], 2, 4, 20, 0.015, tradable_mask)
w_multi = ops.dampen_turnover(composite_ranked, 2, 4, 20, 0.015, tradable_mask)

# Benchmark VN30 Equal-Weight
w_bm = np.where(tradable_mask, 1.0, 0.0)
w_bm = w_bm / np.maximum(np.sum(w_bm, axis=1, keepdims=True), 1e-6)
w_bm_exec = np.zeros_like(w_bm); w_bm_exec[1:] = w_bm[:-1]
res_bm = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_bm_exec, dates=dates, penalty_rate_per_day=0.0)
m_bm = sim.evaluate_strategy(res_bm["equity"], w_bm_exec)

# 1. Unlagged (Trước B6-B11)
res_s_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_single, penalty_rate_per_day=0.07)
res_m_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_multi, penalty_rate_per_day=0.07)
naive_s_b = sim.simulate_naive(raw_close, w_single, raw_open=raw_open)
naive_m_b = sim.simulate_naive(raw_close, w_multi, raw_open=raw_open)
m_s_b = sim.evaluate_strategy(res_s_b["equity"], w_single, naive_s_b)
m_m_b = sim.evaluate_strategy(res_m_b["equity"], w_multi, naive_m_b)

# 2. Lagged (Sau B6-B11)
w_s_exec = np.zeros_like(w_single); w_s_exec[1:] = w_single[:-1]
w_m_exec = np.zeros_like(w_multi); w_m_exec[1:] = w_multi[:-1]
res_s_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_s_exec, dates=dates, penalty_rate_per_day=0.0)
res_m_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_m_exec, dates=dates, penalty_rate_per_day=0.0)
naive_s_a = sim.simulate_naive(raw_close, w_s_exec, raw_open=raw_open)
naive_m_a = sim.simulate_naive(raw_close, w_m_exec, raw_open=raw_open)
m_s_a = sim.evaluate_strategy(res_s_a["equity"], w_s_exec, naive_s_a)
m_m_a = sim.evaluate_strategy(res_m_a["equity"], w_m_exec, naive_m_a)

# Implementation Shortfall
is_cum_s_b = (naive_s_b[-1] - res_s_b["equity"][-1]) / 1.0e9 * 100.0
is_cum_s_a = (naive_s_a[-1] - res_s_a["equity"][-1]) / 1.0e9 * 100.0
is_cum_m_b = (naive_m_b[-1] - res_m_b["equity"][-1]) / 1.0e9 * 100.0
is_cum_m_a = (naive_m_a[-1] - res_m_a["equity"][-1]) / 1.0e9 * 100.0

cagr_naive_s_b = (naive_s_b[-1] / 1.0e9)**(1.0/years) - 1.0
cagr_naive_s_a = (naive_s_a[-1] / 1.0e9)**(1.0/years) - 1.0
cagr_naive_m_b = (naive_m_b[-1] / 1.0e9)**(1.0/years) - 1.0
cagr_naive_m_a = (naive_m_a[-1] / 1.0e9)**(1.0/years) - 1.0

is_ann_s_b = (cagr_naive_s_b - m_s_b["cagr"]) * 100.0
is_ann_s_a = (cagr_naive_s_a - m_s_a["cagr"]) * 100.0
is_ann_m_b = (cagr_naive_m_b - m_m_b["cagr"]) * 100.0
is_ann_m_a = (cagr_naive_m_a - m_m_a["cagr"]) * 100.0

# Newey-West standard error for Sharpe
def nw_se(returns, rf=0.04/252.0, lags=5):
    ex = returns - rf
    mu, sig = np.mean(ex), np.std(ex, ddof=1)
    sr = mu / sig if sig > 1e-12 else 0.0
    u = (ex - mu) / sig
    g = u - 0.5 * sr * (u**2 - 1.0)
    n = len(g)
    lr_var = np.var(g)
    for l in range(1, lags + 1):
        w_l = 1.0 - l / (lags + 1.0)
        lr_var += 2.0 * w_l * np.mean(g[l:] * g[:-l])
    return float(np.sqrt(max(lr_var, 1e-12) / n) * np.sqrt(252.0))

ret_s_b = np.diff(res_s_b["equity"]) / np.maximum(res_s_b["equity"][:-1], 1e-6)
ret_s_a = np.diff(res_s_a["equity"]) / np.maximum(res_s_a["equity"][:-1], 1e-6)
ret_m_b = np.diff(res_m_b["equity"]) / np.maximum(res_m_b["equity"][:-1], 1e-6)
ret_m_a = np.diff(res_m_a["equity"]) / np.maximum(res_m_a["equity"][:-1], 1e-6)
ret_bm = np.diff(res_bm["equity"]) / np.maximum(res_bm["equity"][:-1], 1e-6)

print("\n" + "=" * 140)
print("BẢNG SO SÁNH HIỆU QUẢ BACKTEST 947 PHIÊN VN30 (2023-01-03 ĐẾN 2026-10-23): TRƯỚC VS SAU KHI VÁ B6 - B11")
print("=" * 140)

rows = [
    ("Final Real NAV (VND)", f"{res_s_b['equity'][-1]:,.0f}", f"{res_s_a['equity'][-1]:,.0f}", f"{res_m_b['equity'][-1]:,.0f}", f"{res_m_a['equity'][-1]:,.0f}", f"{res_bm['equity'][-1]:,.0f}"),
    ("Final Naive NAV (VND)", f"{naive_s_b[-1]:,.0f}", f"{naive_s_a[-1]:,.0f}", f"{naive_m_b[-1]:,.0f}", f"{naive_m_a[-1]:,.0f}", "N/A"),
    ("Tổng tỷ suất sinh lời (Total Return)", f"{m_s_b['total_return']*100:.2f}%", f"{m_s_a['total_return']*100:.2f}%", f"{m_m_b['total_return']*100:.2f}%", f"{m_m_a['total_return']*100:.2f}%", f"{m_bm['total_return']*100:.2f}%"),
    ("Tỷ suất sinh lời năm hóa (CAGR Thực tế)", f"{m_s_b['cagr']*100:.2f}%", f"{m_s_a['cagr']*100:.2f}%", f"{m_m_b['cagr']*100:.2f}%", f"{m_m_a['cagr']*100:.2f}%", f"{m_bm['cagr']*100:.2f}%"),
    ("CAGR Naive (%/năm)", f"{cagr_naive_s_b*100:.2f}%", f"{cagr_naive_s_a*100:.2f}%", f"{cagr_naive_m_b*100:.2f}%", f"{cagr_naive_m_a*100:.2f}%", "N/A"),
    ("Sharpe Ratio (Newey-West SE)", f"{m_s_b['sharpe_ratio']:.2f} (±{nw_se(ret_s_b):.2f})", f"{m_s_a['sharpe_ratio']:.2f} (±{nw_se(ret_s_a):.2f})", f"{m_m_b['sharpe_ratio']:.2f} (±{nw_se(ret_m_b):.2f})", f"{m_m_a['sharpe_ratio']:.2f} (±{nw_se(ret_m_a):.2f})", f"{m_bm['sharpe_ratio']:.2f} (±{nw_se(ret_bm):.2f})"),
    ("Deflated Sharpe Ratio (DSR)", f"{m_s_b['deflated_sharpe_ratio']:.4f}", f"{m_s_a['deflated_sharpe_ratio']:.4f}", f"{m_m_b['deflated_sharpe_ratio']:.4f}", f"{m_m_a['deflated_sharpe_ratio']:.4f}", "N/A"),
    ("Sortino Ratio (Downside Risk)", f"{m_s_b['sortino_ratio']:.2f}", f"{m_s_a['sortino_ratio']:.2f}", f"{m_m_b['sortino_ratio']:.2f}", f"{m_m_a['sortino_ratio']:.2f}", f"{m_bm['sortino_ratio']:.2f}"),
    ("Biến động danh mục năm hóa (Vol)", f"{m_s_b['annualized_vol']*100:.2f}%", f"{m_s_a['annualized_vol']*100:.2f}%", f"{m_m_b['annualized_vol']*100:.2f}%", f"{m_m_a['annualized_vol']*100:.2f}%", f"{m_bm['annualized_vol']*100:.2f}%"),
    ("Mức sụt giảm tối đa (MaxDD)", f"{m_s_b['max_drawdown']*100:.2f}%", f"{m_s_a['max_drawdown']*100:.2f}%", f"{m_m_b['max_drawdown']*100:.2f}%", f"{m_m_a['max_drawdown']*100:.2f}%", f"{m_bm['max_drawdown']*100:.2f}%"),
    ("Vòng quay ngày (Turnover 4 decimals)", f"{m_s_b['daily_turnover']*100:.4f}%", f"{m_s_a['daily_turnover']*100:.4f}%", f"{m_m_b['daily_turnover']*100:.4f}%", f"{m_m_a['daily_turnover']*100:.4f}%", f"{m_bm['daily_turnover']*100:.4f}%"),
    ("IS Lũy kế (% Vốn gốc 1.0 tỷ)", f"{is_cum_s_b:+.2f}%", f"{is_cum_s_a:+.2f}%", f"{is_cum_m_b:+.2f}%", f"{is_cum_m_a:+.2f}%", "N/A"),
    ("IS Năm hóa (%/năm)", f"{is_ann_s_b:+.2f}%", f"{is_ann_s_a:+.2f}%", f"{is_ann_m_b:+.2f}%", f"{is_ann_m_a:+.2f}%", "N/A"),
    ("Tổng phí môi giới & Thuế bán", f"{res_s_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['trading_fees'][-1]/1e6:.1f}M VND", f"{res_bm['trading_fees'][-1]/1e6:.1f}M VND"),
    ("Chi phí lãi vay ứng trước T+2", f"{res_s_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_bm['advance_interest_fees'][-1]/1e6:.1f}M VND")
]

header = f"{'Chỉ số Thực nghiệm':<40} | {'Single (Trước)':<18} | {'Single (Sau)':<18} | {'Multi (Trước)':<18} | {'Multi (Sau)':<18} | {'Benchmark (VN30)':<18}"
print(header)
print("-" * 140)
for row in rows:
    print(f"{row[0]:<40} | {row[1]:>18} | {row[2]:>18} | {row[3]:>18} | {row[4]:>18} | {row[5]:>18}")
print("=" * 140)
