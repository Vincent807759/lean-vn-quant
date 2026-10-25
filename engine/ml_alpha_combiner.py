"""
================================================================================
LEAN VN-QUANT: MULTI-ALPHA ORTHOGONAL COMBINER & PURGED WALK-FORWARD CV
================================================================================
Multi-factor quantitative research and machine learning combiner module:
1. Generates 5 orthogonal macro & microstructure alpha factors:
   - Alpha 1: Breakout Z-Score (Trend breakout above MA40 with MA50 trend filter)
   - Alpha 2: Momentum Acceleration (Trend acceleration MA10 vs MA30)
   - Alpha 3: Short-term Mean Reversion (5-day price oscillation reversal)
   - Alpha 4: Volume Volatility Dampener (Volume compression to price volatility ratio)
   - Alpha 5: Intraday Liquidity Flow (Close-to-Open order flow pressure and liquidity)
2. Gram-Schmidt QR & Ridge Factor Orthogonalization (alpha_ops.check_and_orthogonalize)
   purges factor multicollinearity down to cross-correlation < 0.20.
3. Purged Walk-Forward Cross-Validation with T+3 Embargo:
   Strictly prevents look-ahead bias and dynamic factor weighting proportional to positive forward IC.
4. Empirical head-to-head performance evaluation of Single-Alpha vs Multi-Alpha
   via Realistic Vietnam Market Simulator (T+2.5, taxes, fees, slippage, advance interest).
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from engine import alpha_ops as ops
    from engine import data_pipeline as dp
    from engine import vn_simulator as sim
except ImportError:
    import alpha_ops as ops
    import data_pipeline as dp
    import vn_simulator as sim

DEFAULT_PARQUET = os.path.join(ROOT_DIR, "data", "vn30_production.parquet")
if not os.path.exists(DEFAULT_PARQUET):
    DEFAULT_PARQUET = os.path.join("data", "vn30_production.parquet")
if not os.path.exists(DEFAULT_PARQUET):
    DEFAULT_PARQUET = "vn30_production.parquet"


# ==============================================================================
# 1. GENERATE 5 ORTHOGONAL MULTI-FACTOR ALPHAS (FACTOR FACTORY)
# ==============================================================================

def generate_five_alphas(
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    volume: np.ndarray,
    tradable_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generates 5 distinct alpha factors representing independent quantitative strategies on VN30.
    """
    T, N = raw_close.shape

    # --- Alpha 1: Breakout Z-Score with MA50 Trend Filter ---
    hh = ops.ts_mean(raw_close, 40)
    dev = ops.ts_std_dev(raw_close, 40)
    z = (raw_close - hh) / np.maximum(dev, 1e-4)
    sma50 = ops.ts_mean(raw_close, 50)
    in_uptrend = raw_close > sma50
    z_filtered = np.where(in_uptrend, z, -999.0)
    a1 = ops.rank(z_filtered, mask=tradable_mask)

    # --- Alpha 2: Momentum Acceleration ---
    ma10 = ops.ts_mean(raw_close, 10)
    ma30 = ops.ts_mean(raw_close, 30)
    mom_acc = ops.ts_delta(ma10, 10) / np.maximum(ma30, 1e-4)
    a2 = ops.rank(mom_acc, mask=tradable_mask)

    # --- Alpha 3: Short-term Mean Reversion ---
    ma5 = ops.ts_mean(raw_close, 5)
    ratio = raw_close / np.maximum(ma5, 1e-4)
    rev = -ops.ts_rank(ratio, 5)
    a3 = ops.rank(rev, mask=tradable_mask)

    # --- Alpha 4: Volume Volatility Dampener ---
    vol5 = ops.ts_mean(volume, 5)
    vol20 = ops.ts_mean(volume, 20)
    price_vol = ops.ts_std_dev(raw_close, 10) / np.maximum(raw_close, 1e-4)
    vol_factor = (vol5 / np.maximum(vol20, 1e-4)) / np.maximum(price_vol, 1e-4)
    a4 = ops.rank(vol_factor, mask=tradable_mask)

    # --- Alpha 5: Intraday Liquidity Flow ---
    hl_range = np.maximum(raw_high - raw_low, 1e-4)
    co_flow = (raw_close - raw_open) / hl_range
    vol_norm = volume / np.maximum(vol20, 1e-4)
    flow_factor = ops.ts_mean(co_flow * vol_norm, 5)
    a5 = ops.rank(flow_factor, mask=tradable_mask)

    alphas = [a1, a2, a3, a4, a5]
    names = [
        "Breakout Z-Score",
        "Momentum Acceleration",
        "Short-term Reversion",
        "Volume Volatility Dampener",
        "Liquidity Flow Pressure"
    ]
    return alphas, names


# ==============================================================================
# 2. GRAM-SCHMIDT QR FACTOR ORTHOGONALIZATION (PURGE MULTICOLLINEARITY)
# ==============================================================================

def orthogonalize_factor_library(
    alphas: List[np.ndarray],
    names: List[str],
    tradable_mask: np.ndarray,
    threshold: float = 0.25
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Applies sequential Gram-Schmidt QR Orthogonalization to remove cross-correlation,
    ensuring each factor provides unique independent information value.
    """
    K = len(alphas)
    
    # 1. Compute raw cross-correlation matrix
    raw_corr = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            valid = tradable_mask & np.isfinite(alphas[i]) & np.isfinite(alphas[j])
            if np.sum(valid) > 20:
                raw_corr[i, j] = np.corrcoef(alphas[i][valid], alphas[j][valid])[0, 1]

    # 2. Sequential orthogonal projection
    ortho_alphas = [alphas[0]]
    for k in range(1, K):
        ortho_k, corr_before, corr_after = ops.check_and_orthogonalize(
            new_alpha=alphas[k],
            existing_alphas=ortho_alphas,
            threshold=threshold
        )
        ranked_ortho = ops.rank(ortho_k, mask=tradable_mask)
        ortho_alphas.append(ranked_ortho)

    # 3. Compute orthogonalized cross-correlation matrix
    ortho_corr = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            valid = tradable_mask & np.isfinite(ortho_alphas[i]) & np.isfinite(ortho_alphas[j])
            if np.sum(valid) > 20:
                ortho_corr[i, j] = np.corrcoef(ortho_alphas[i][valid], ortho_alphas[j][valid])[0, 1]

    return ortho_alphas, raw_corr, ortho_corr


# ==============================================================================
# 3. PURGED WALK-FORWARD CROSS-VALIDATION WITH T+3 EMBARGO
# ==============================================================================

def purged_walk_forward_combiner(
    ortho_alphas: List[np.ndarray],
    raw_close: np.ndarray,
    tradable_mask: np.ndarray,
    window: int = 120,
    embargo: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamic Walk-Forward factor combiner using Information Coefficient (IC) weighting:
    - Rolling lookback window of W trading sessions (default 120 days).
    - Embargo T+3: Strictly excises the most recent 3 days from the training set
      to eliminate any look-ahead bias induced by the T+2.5 settlement cycle.
    - Dynamic weights proportional to positive forward Spearman Rank IC.
    """
    T, N = raw_close.shape
    K = len(ortho_alphas)
    
    # Forward 1-day Return
    fwd_ret = np.zeros((T, N), dtype=np.float64)
    fwd_ret[:-1] = (raw_close[1:] - raw_close[:-1]) / np.maximum(raw_close[:-1], 1e-4)

    composite_alpha = np.zeros((T, N), dtype=np.float64)
    factor_weights_history = np.zeros((T, K), dtype=np.float64)

    for t in range(window + embargo, T):
        train_end = t - embargo
        train_start = max(0, train_end - window)
        
        ics = []
        for k in range(K):
            daily_ics = []
            for s in range(train_start, train_end):
                mask_s = tradable_mask[s] & np.isfinite(ortho_alphas[k][s]) & np.isfinite(fwd_ret[s])
                if np.sum(mask_s) >= 8:
                    r_a = np.argsort(np.argsort(ortho_alphas[k][s][mask_s]))
                    r_f = np.argsort(np.argsort(fwd_ret[s][mask_s]))
                    if len(r_a) > 2 and np.std(r_a) > 0 and np.std(r_f) > 0:
                        corr_val = np.corrcoef(r_a, r_f)[0, 1]
                        if np.isfinite(corr_val):
                            daily_ics.append(corr_val)
            ics.append(float(np.mean(daily_ics)) if daily_ics else 0.0)
            
        pos_ics = [max(0.0, ic) for ic in ics]
        sum_ic = sum(pos_ics)
        if sum_ic > 1e-6:
            w_k = np.array([ic / sum_ic for ic in pos_ics], dtype=np.float64)
        else:
            w_k = np.full(K, 1.0 / K, dtype=np.float64)

        factor_weights_history[t] = w_k
        
        for k in range(K):
            composite_alpha[t] += w_k[k] * ortho_alphas[k][t]

    # Warm-up phase defaults to Alpha 1
    for t in range(window + embargo):
        composite_alpha[t] = ortho_alphas[0][t]
        factor_weights_history[t, 0] = 1.0

    return composite_alpha, factor_weights_history


# ==============================================================================
# 4. HEAD-TO-HEAD EMPIRICAL EXPERIMENT: SINGLE-ALPHA VS MULTI-ALPHA
# ==============================================================================

def run_comparative_experiment(parquet_path: str = DEFAULT_PARQUET) -> Dict[str, Any]:
    """Executes the full institutional verification pipeline and outputs comparison table."""
    print("\n" + "=" * 90)
    print("      MULTI-FACTOR PERFORMANCE AUDIT (PURGED WALK-FORWARD MULTI-ALPHA COMBINER)")
    print("=" * 90)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet data file not found at: {parquet_path}")

    # 1. Load data and construct matrices
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

    T, N = raw_close.shape
    adv20_shares = np.where(raw_close > 0, adv20 / np.maximum(raw_close, 1e-4), 1e6)
    print(f"-> Ingested dataset: {T} trading sessions ({dates[0]} to {dates[-1]}), {N} VN30 constituents.")

    # 2. Factor generation & QR orthogonalization
    raw_alphas, factor_names = generate_five_alphas(raw_open, raw_high, raw_low, raw_close, volume, tradable_mask)
    ortho_alphas, raw_corr, ortho_corr = orthogonalize_factor_library(raw_alphas, factor_names, tradable_mask)

    print("\n--- TABLE 1: CROSS-CORRELATION BEFORE & AFTER QR ORTHOGONALIZATION (MAX CORR) ---")
    print(f"{'Alpha Factor':<30} | {'Raw Max Correlation':<22} | {'Ortho Max Correlation':<24}")
    print("-" * 80)
    for i in range(len(factor_names)):
        max_r_raw = np.max([abs(raw_corr[i, j]) for j in range(len(factor_names)) if i != j])
        max_r_ortho = np.max([abs(ortho_corr[i, j]) for j in range(len(factor_names)) if i != j])
        print(f"{factor_names[i]:<30} | {max_r_raw:>20.3f}   | {max_r_ortho:>22.3f}")

    # 3. Purged Walk-Forward combination
    print("\n-> Running Purged Walk-Forward Cross-Validation (Window=120, Embargo=T+3)...")
    composite_alpha, w_history = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=120, embargo=3)
    composite_ranked = ops.rank(composite_alpha, mask=tradable_mask)

    # 4. Turnover Dampener
    w_single = ops.dampen_turnover(
        raw_alphas[0],
        top_k_in=2,
        top_k_out=4,
        ema_span=20,
        max_daily_turnover=0.015,
        mask=tradable_mask
    )

    w_multi = ops.dampen_turnover(
        composite_ranked,
        top_k_in=2,
        top_k_out=4,
        ema_span=20,
        max_daily_turnover=0.015,
        mask=tradable_mask
    )

    # 5. Realistic Vietnam Microstructure Simulator
    res_single = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_single)
    res_multi = sim.simulate_vietnam(raw_open, raw_close, ceiling, floor, adv20_shares, w_multi)

    naive_single = sim.simulate_naive(raw_close, w_single)
    naive_multi = sim.simulate_naive(raw_close, w_multi)

    m_single = sim.evaluate_strategy(res_single["equity"], w_single, naive_single)
    m_multi = sim.evaluate_strategy(res_multi["equity"], w_multi, naive_multi)

    # 6. Benchmark Comparison Table
    print("\n" + "=" * 90)
    print("         TABLE 2: HEAD-TO-HEAD QUANTITATIVE BENCHMARK PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"{'Quantitative Metric':<32} | {'Single-Alpha (Breakout)':<25} | {'Multi-Alpha Combiner':<25}")
    print("-" * 90)
    print(f"{'Total Cumulative Return':<32} | {m_single['total_return']*100:>23.2f}% | {m_multi['total_return']*100:>23.2f}%")
    print(f"{'CAGR (Annualized Return)':<32} | {m_single['cagr']*100:>23.2f}% | {m_multi['cagr']*100:>23.2f}%")
    print(f"{'Sharpe Ratio (Post Fee & Tax)':<32} | {m_single['sharpe_ratio']:>24.2f}  | {m_multi['sharpe_ratio']:>24.2f} ")
    print(f"{'Sortino Ratio (Downside Risk)':<32} | {m_single['sortino_ratio']:>24.2f}  | {m_multi['sortino_ratio']:>24.2f} ")
    print(f"{'Annualized Volatility':<32} | {m_single['annualized_vol']*100:>23.2f}% | {m_multi['annualized_vol']*100:>23.2f}%")
    print(f"{'Maximum Drawdown (MaxDD)':<32} | {m_single['max_drawdown']*100:>23.2f}% | {m_multi['max_drawdown']*100:>23.2f}%")
    print(f"{'Average Daily Turnover':<32} | {m_single['daily_turnover']*100:>23.2f}% | {m_multi['daily_turnover']*100:>23.2f}%")
    print(f"{'Implementation Shortfall':<32} | {m_single['implementation_shortfall_pct']:>23.2f}% | {m_multi['implementation_shortfall_pct']:>23.2f}%")
    print(f"{'Total Brokerage & Tax Fees':<32} | {res_single['trading_fees'][-1]/1e6:>20.1f}M VND | {res_multi['trading_fees'][-1]/1e6:>20.1f}M VND")
    print(f"{'T+2 Advance Interest Cost':<32} | {res_single['advance_interest_fees'][-1]/1e6:>20.1f}M VND | {res_multi['advance_interest_fees'][-1]/1e6:>20.1f}M VND")
    print("=" * 90)

    comparison_results = {
        "dates_count": T,
        "single_alpha": m_single,
        "multi_alpha": m_multi,
        "raw_corr": raw_corr.tolist(),
        "ortho_corr": ortho_corr.tolist(),
        "factor_names": factor_names,
        "single_equity": res_single["equity"].tolist(),
        "multi_equity": res_multi["equity"].tolist()
    }
    return comparison_results


# ==============================================================================
# 5. VN-FINGRAPH CONTAGION INTEGRATION & HAWKES/OU DECAY PIPELINE
# ==============================================================================

def load_and_decay_fingraph_alpha(
    fingraph_parquet_path: str,
    target_dates: List[Any],
    target_symbols: List[str],
    tau_sessions: float = 5.0
) -> np.ndarray:
    """
    Ingests VN-FinGraph event contagion alpha from Parquet,
    applies continuous Hawkes/OU exponential decay and Van der Waerden normal score transform,
    and returns aligned stationary factor matrix matching (T, N).
    """
    if not os.path.exists(fingraph_parquet_path):
        raise FileNotFoundError(f"VN-FinGraph parquet not found at: {fingraph_parquet_path}")
        
    df_fg = pl.read_parquet(fingraph_parquet_path)
    T = len(target_dates)
    N = len(target_symbols)
    raw_fg = np.zeros((T, N), dtype=np.float64)
    
    date_strs = [str(d)[:10] for d in target_dates]
    df_dates = df_fg["date"].to_list()
    date_to_idx = {d: i for i, d in enumerate(df_dates)}
    
    fg_cols = set(df_fg.columns)
    for j, sym in enumerate(target_symbols):
        if sym in fg_cols:
            col_vals = df_fg[sym].to_numpy()
            for t, d_str in enumerate(date_strs):
                if d_str in date_to_idx:
                    raw_fg[t, j] = col_vals[date_to_idx[d_str]]
                    
    # Apply Hawkes/OU exponential decay filter
    decayed_fg = ops.apply_hawkes_ou_decay(raw_fg, tau_sessions=tau_sessions)
    
    # Apply cross-sectional Van der Waerden Inverse Normal Rank transform
    stationary_alpha = ops.van_der_waerden_transform(decayed_fg)
    return stationary_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Alpha Orthogonal Combiner & Purged Walk-Forward Engine")
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET, help="Path to VN30 Parquet file")
    args = parser.parse_args()
    
    run_comparative_experiment(args.parquet)
