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

# Generate factors & purged walk-forward combiner
raw_alphas, factor_names = generate_five_alphas(raw_open, raw_high, raw_low, raw_close, volume, tradable_mask)
ortho_alphas, raw_corr, ortho_corr = orthogonalize_factor_library(raw_alphas, factor_names, tradable_mask)
composite_alpha, _ = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=120, embargo=3)
composite_ranked = ops.rank(composite_alpha, mask=tradable_mask)

w_single = ops.dampen_turnover(raw_alphas[0], 2, 4, 20, 0.015, tradable_mask)
w_multi = ops.dampen_turnover(composite_ranked, 2, 4, 20, 0.015, tradable_mask)

# 1. Unlagged (Mô phỏng Trước khi vá B6-B11: same-day execution, hardcoded penalty 7%)
res_s_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_single, penalty_rate_per_day=0.07)
res_m_b = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_multi, penalty_rate_per_day=0.07)
naive_s_b = sim.simulate_naive(raw_close, w_single)
naive_m_b = sim.simulate_naive(raw_close, w_multi)
m_s_b = sim.evaluate_strategy(res_s_b["equity"], w_single, naive_s_b)
m_m_b = sim.evaluate_strategy(res_m_b["equity"], w_multi, naive_m_b)

# 2. Lagged (Sau khi vá B6-B11: w_exec lagged 1 session, causal ADV/vol, T+3/T+2.5 split, penalty=0.0)
w_s_exec = np.zeros_like(w_single); w_s_exec[1:] = w_single[:-1]
w_m_exec = np.zeros_like(w_multi); w_m_exec[1:] = w_multi[:-1]
res_s_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_s_exec, dates=dates, penalty_rate_per_day=0.0)
res_m_a = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_m_exec, dates=dates, penalty_rate_per_day=0.0)
naive_s_a = sim.simulate_naive(raw_close, w_s_exec)
naive_m_a = sim.simulate_naive(raw_close, w_m_exec)
m_s_a = sim.evaluate_strategy(res_s_a["equity"], w_s_exec, naive_s_a)
m_m_a = sim.evaluate_strategy(res_m_a["equity"], w_m_exec, naive_m_a)

print("\n" + "=" * 115)
print("BẢNG SO SÁNH HIỆU QUẢ BACKTEST 947 PHIÊN VN30 (2023-01-03 ĐẾN 2026-10-23): TRƯỚC VS SAU KHI VÁ B6 - B11")
print("=" * 115)
rows = [
    ("Final Real NAV (VND)", f"{res_s_b['equity'][-1]:,.0f}", f"{res_s_a['equity'][-1]:,.0f}", f"{res_m_b['equity'][-1]:,.0f}", f"{res_m_a['equity'][-1]:,.0f}"),
    ("Tổng tỷ suất sinh lời (Total Return)", f"{m_s_b['total_return']*100:.2f}%", f"{m_s_a['total_return']*100:.2f}%", f"{m_m_b['total_return']*100:.2f}%", f"{m_m_a['total_return']*100:.2f}%"),
    ("Tỷ suất sinh lời năm hóa (CAGR)", f"{m_s_b['cagr']*100:.2f}%", f"{m_s_a['cagr']*100:.2f}%", f"{m_m_b['cagr']*100:.2f}%", f"{m_m_a['cagr']*100:.2f}%"),
    ("Sharpe Ratio (Sau thuế & phí)", f"{m_s_b['sharpe_ratio']:.2f}", f"{m_s_a['sharpe_ratio']:.2f}", f"{m_m_b['sharpe_ratio']:.2f}", f"{m_m_a['sharpe_ratio']:.2f}"),
    ("Deflated Sharpe Ratio (DSR)", f"{m_s_b['deflated_sharpe_ratio']:.4f}", f"{m_s_a['deflated_sharpe_ratio']:.4f}", f"{m_m_b['deflated_sharpe_ratio']:.4f}", f"{m_m_a['deflated_sharpe_ratio']:.4f}"),
    ("Sortino Ratio (Downside Risk)", f"{m_s_b['sortino_ratio']:.2f}", f"{m_s_a['sortino_ratio']:.2f}", f"{m_m_b['sortino_ratio']:.2f}", f"{m_m_a['sortino_ratio']:.2f}"),
    ("Biến động danh mục năm hóa (Vol)", f"{m_s_b['annualized_vol']*100:.2f}%", f"{m_s_a['annualized_vol']*100:.2f}%", f"{m_m_b['annualized_vol']*100:.2f}%", f"{m_m_a['annualized_vol']*100:.2f}%"),
    ("Mức sụt giảm tối đa (MaxDD)", f"{m_s_b['max_drawdown']*100:.2f}%", f"{m_s_a['max_drawdown']*100:.2f}%", f"{m_m_b['max_drawdown']*100:.2f}%", f"{m_m_a['max_drawdown']*100:.2f}%"),
    ("Vòng quay danh mục ngày (Turnover)", f"{m_s_b['daily_turnover']*100:.2f}%", f"{m_s_a['daily_turnover']*100:.2f}%", f"{m_m_b['daily_turnover']*100:.2f}%", f"{m_m_a['daily_turnover']*100:.2f}%"),
    ("Tổn thất thực thi (Implementation Shortfall)", f"{m_s_b['implementation_shortfall_pct']:.2f}%", f"{m_s_a['implementation_shortfall_pct']:.2f}%", f"{m_m_b['implementation_shortfall_pct']:.2f}%", f"{m_m_a['implementation_shortfall_pct']:.2f}%"),
    ("Tổng phí môi giới & Thuế bán", f"{res_s_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['trading_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['trading_fees'][-1]/1e6:.1f}M VND"),
    ("Chi phí lãi vay ứng trước T+2", f"{res_s_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_s_a['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_b['advance_interest_fees'][-1]/1e6:.1f}M VND", f"{res_m_a['advance_interest_fees'][-1]/1e6:.1f}M VND"),
]

header = f"{'Chỉ số Thực nghiệm':<42} | {'Single (Trước B6-B11)':<22} | {'Single (Sau B6-B11)':<20} | {'Multi (Trước B6-B11)':<22} | {'Multi (Sau B6-B11)':<20}"
print(header)
print("-" * 115)
for title, s_b, s_a, m_b, m_a in rows:
    print(f"{title:<42} | {s_b:>22} | {s_a:>20} | {m_b:>22} | {m_a:>20}")
print("=" * 115)
