"""
vn_simulator.py - Realistic Vietnam Equity Market Microstructure Execution Engine

Features:
- Numba @numba.njit accelerated execution core
- T+2.5 settlement ledger (signal at close d -> execute at raw_open d+1; locked until T+2)
- Pending cash queue with 0.035%/day advance cash interest fees
- HOSE/HNX/UPCOM price bands, tick size rounding
- Conservative liquidity trap:
    * raw_open >= ceiling -> refuse BUY (locked at limit up)
    * raw_open <= floor -> refuse SELL (floor trapped, forced hold)
- Lot 100 round-down and <= 5% ADV20 participation constraint
- Asymmetric costs: Broker fee (0.15%), Slippage (0.10%), Personal Income Tax (0.10% on SELL only)
- VN30F Derivative Beta Hedge Overlay
- Performance evaluation with Implementation Shortfall & Kill-Switch monitoring
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import os
import json
import datetime
import numpy as np
import bottleneck as bn
import numba
from scipy import stats


# ==============================================================================
# DEFLATED SHARPE RATIO (BAILEY & LOPEZ DE PRADO, 2014)
# ==============================================================================

def compute_deflated_sharpe_ratio(
    daily_returns: np.ndarray,
    n_trials: int = 26,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 252.0,
    trial_returns: Optional[np.ndarray] = None,
    trial_corr_matrix: Optional[np.ndarray] = None,
    mean_trial_correlation: Optional[float] = None,
    var_sharpe_trials: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes the Deflated Sharpe Ratio (DSR) to account for selection bias / multiple testing,
    non-normality (skewness, kurtosis), sample length, and factor cross-correlation following:
    Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management.
    
    Calibrated Enhancements (Phase 4):
    - Calculates effective number of independent trials N_eff via Eigenvalue Participation Ratio:
        N_eff = (sum lambda_i)^2 / sum (lambda_i^2)
      or via correlation formula N_eff = 1 + (N - 1)(1 - rho_bar).
    - Uses empirical Var[{SR_n}] for accurate SR_0 threshold determination.
    """
    returns = np.asarray(daily_returns, dtype=np.float64)
    T = len(returns)
    if T < 10:
        return {
            "deflated_sharpe_ratio": 0.0,
            "p_value": 1.0,
            "expected_max_sharpe": 0.0,
            "annualized_sharpe": 0.0,
            "n_eff": 1.0
        }
        
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r <= 1e-12:
        return {
            "deflated_sharpe_ratio": 0.0,
            "p_value": 1.0,
            "expected_max_sharpe": 0.0,
            "annualized_sharpe": 0.0,
            "n_eff": 1.0
        }
        
    daily_rf = risk_free_rate / annualization_factor
    daily_sr = (mean_r - daily_rf) / std_r
    ann_sr = daily_sr * np.sqrt(annualization_factor)
    
    # Skewness and Kurtosis
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # Pearson kurtosis (normal=3)
    
    # Variance of Sharpe Ratio estimator under non-normality
    # Var(SR) = (1 - skew * SR + 0.25 * (kurt - 1) * SR^2) / (T - 1)
    sr_var = (1.0 - skew * daily_sr + 0.25 * (kurt - 1.0) * (daily_sr ** 2)) / max(T - 1, 1)
    sr_std = np.sqrt(max(sr_var, 1e-12))
    
    # Determine N_eff (Effective number of independent trials)
    n_trials = max(n_trials, 1)
    if trial_returns is not None and trial_returns.shape[1] > 1:
        C = np.corrcoef(trial_returns.T)
        eigvals = np.maximum(np.linalg.eigvalsh(C), 0.0)
        sum_l = float(np.sum(eigvals))
        sum_l2 = float(np.sum(eigvals ** 2))
        n_eff = (sum_l ** 2) / max(sum_l2, 1e-6)
        
        # Calculate empirical variance across trial Sharpe ratios
        t_means = np.mean(trial_returns, axis=0) * annualization_factor - risk_free_rate
        t_stds = np.std(trial_returns, axis=0, ddof=1) * np.sqrt(annualization_factor)
        t_srs = t_means / np.maximum(t_stds, 1e-4)
        var_sr = float(np.var(t_srs, ddof=1)) if len(t_srs) > 1 else (0.25 ** 2)
    elif trial_corr_matrix is not None and trial_corr_matrix.shape[0] > 1:
        eigvals = np.maximum(np.linalg.eigvalsh(trial_corr_matrix), 0.0)
        sum_l = float(np.sum(eigvals))
        sum_l2 = float(np.sum(eigvals ** 2))
        n_eff = (sum_l ** 2) / max(sum_l2, 1e-6)
        var_sr = var_sharpe_trials if var_sharpe_trials is not None else (0.25 ** 2)
    elif mean_trial_correlation is not None:
        rho_bar = float(np.clip(mean_trial_correlation, 0.0, 0.99))
        n_eff = 1.0 + (n_trials - 1.0) * (1.0 - rho_bar)
        var_sr = var_sharpe_trials if var_sharpe_trials is not None else (0.25 ** 2)
    else:
        # Standard unadjusted multiple testing
        n_eff = float(n_trials)
        var_sr = sr_var * annualization_factor
        
    n_eff = max(1.0, n_eff)
    euler_mascheroni = 0.5772156649
    
    if n_eff > 1.0001:
        p1 = max(0.5, 1.0 - 1.0 / n_eff)
        p2 = max(0.5, 1.0 - 1.0 / (n_eff * np.e))
        z1 = stats.norm.ppf(p1)
        z2 = stats.norm.ppf(p2)
        sigma_trials = np.sqrt(max(var_sr, 1e-6))
        # Expected max annualized Sharpe ratio under the null hypothesis of selection bias
        expected_max_ann_sr = max(0.0, float(sigma_trials * ((1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2)))
    else:
        expected_max_ann_sr = 0.0
        
    expected_max_daily_sr = expected_max_ann_sr / np.sqrt(annualization_factor)
    
    # DSR Statistic and Probability
    dsr_stat = (daily_sr - expected_max_daily_sr) / sr_std
    dsr_prob = float(stats.norm.cdf(dsr_stat))
    p_value = float(1.0 - dsr_prob)
    
    return {
        "deflated_sharpe_ratio": dsr_prob,
        "p_value": p_value,
        "expected_max_sharpe": expected_max_ann_sr,
        "annualized_sharpe": ann_sr,
        "n_trials": n_trials,
        "n_eff": float(n_eff),
        "sample_skewness": skew,
        "sample_kurtosis": kurt
    }


def record_trial(
    trial_name: str,
    daily_returns: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    registry_file: str = "experiments/trial_registry.jsonl"
) -> Dict[str, Any]:
    """
    Appends a quantitative backtest trial to the persistent JSONL trial registry.
    """
    os.makedirs(os.path.dirname(os.path.abspath(registry_file)), exist_ok=True)
    r_arr = np.asarray(daily_returns, dtype=np.float64)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "trial_name": trial_name,
        "n_bars": len(r_arr),
        "mean_daily_return": float(np.mean(r_arr)),
        "vol_annualized": float(np.std(r_arr, ddof=1) * np.sqrt(252.0)) if len(r_arr) > 1 else 0.0,
        "metadata": metadata or {},
        "returns": [float(x) for x in r_arr]
    }
    with open(registry_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def compute_dsr_from_registry(
    daily_returns: np.ndarray,
    registry_file: str = "experiments/trial_registry.jsonl",
    risk_free_rate: float = 0.04,
    annualization_factor: float = 252.0,
    benchmark_returns: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes Calibrated Deflated Sharpe Ratio (DSR) using real historical trials
    stored in experiments/trial_registry.jsonl to derive effective trial count N_eff
    via Eigenvalue Participation Ratio.
    
    If benchmark_returns is provided, computes DSR on Active Returns:
      r_active(t) = r(t) - r_bench(t)
    stripping market beta and providing an unconstrained N_eff > 2.0.
    """
    current_r = np.asarray(daily_returns, dtype=np.float64)
    T = len(current_r)
    
    trials_returns_list = []
    if os.path.exists(registry_file):
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    ret_seq = data.get("returns")
                    if ret_seq and len(ret_seq) == T:
                        trials_returns_list.append(np.asarray(ret_seq, dtype=np.float64))
        except Exception:
            pass
            
    if not trials_returns_list:
        trials_returns_list = [current_r]
    else:
        is_present = any(np.allclose(current_r, tr, atol=1e-8) for tr in trials_returns_list)
        if not is_present:
            trials_returns_list.append(current_r)
            
    if len(trials_returns_list) > 1:
        matrix = np.column_stack(trials_returns_list)
        if benchmark_returns is not None and len(benchmark_returns) == T:
            bm_arr = np.asarray(benchmark_returns, dtype=np.float64)
            active_matrix = matrix - bm_arr[:, None]
            return compute_deflated_sharpe_ratio(
                daily_returns=current_r,
                n_trials=len(trials_returns_list),
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
                trial_returns=active_matrix
            )
        else:
            return compute_deflated_sharpe_ratio(
                daily_returns=current_r,
                n_trials=len(trials_returns_list),
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
                trial_returns=matrix
            )
    else:
        return compute_deflated_sharpe_ratio(
            daily_returns=current_r,
            n_trials=1,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )


# ==============================================================================
# 1. PRICE LIMITS AND TICK SIZES
# ==============================================================================

def get_exchange_band(exchange: str = "HOSE") -> float:
    """Returns price limit band for Vietnamese exchanges."""
    ex = exchange.upper()
    if ex == "HOSE":
        return 0.07  # +-7%
    elif ex == "HNX":
        return 0.10  # +-10%
    elif ex == "UPCOM":
        return 0.15  # +-15%
    return 0.07


def compute_bands_and_ticks(
    ref_prices: np.ndarray,
    exchange: str = "HOSE"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes ceiling and floor prices for all assets according to exchange rules.
    HOSE tick sizes:
        < 10k VND: tick = 10 VND
        10k - 50k VND: tick = 50 VND
        >= 50k VND: tick = 100 VND
    Ceiling is rounded DOWN to tick size; Floor is rounded UP to tick size.
    """
    band = get_exchange_band(exchange)
    ref = np.asarray(ref_prices, dtype=np.float64)
    
    raw_ceil = ref * (1.0 + band)
    raw_flr = ref * (1.0 - band)
    
    if exchange.upper() == "HOSE":
        # Vectorized tick calculation
        ceil_tick = np.where(raw_ceil < 10000.0, 10.0, np.where(raw_ceil < 50000.0, 50.0, 100.0))
        flr_tick = np.where(raw_flr < 10000.0, 10.0, np.where(raw_flr < 50000.0, 50.0, 100.0))
        
        ceiling = np.floor(raw_ceil / ceil_tick) * ceil_tick
        floor = np.ceil(raw_flr / flr_tick) * flr_tick
    else:
        # HNX / UPCOM standard 100 VND tick
        ceiling = np.floor(raw_ceil / 100.0) * 100.0
        floor = np.ceil(raw_flr / 100.0) * 100.0
        
    return ceiling, floor


# ==============================================================================
# 2. NUMBA JIT MICROSTRUCTURE SIMULATION CORE
# ==============================================================================

@numba.njit(fastmath=True, cache=True)
def _calc_borrow_cap(pending: float, rate: float, days: float, min_fee: float) -> float:
    """
    Computes maximum advance cash withdrawal capacity given pending receivables,
    daily advance borrowing interest rate, calendar days, and minimum floor fee.
    """
    if pending <= min_fee:
        return 0.0
    eff_days = max(days, 1.0)
    thresh = min_fee / max(rate * eff_days, 1e-8)
    if (pending - min_fee) < thresh:
        return pending - min_fee
    else:
        return pending / (1.0 + rate * eff_days)


@numba.njit(fastmath=True, cache=True)
def _calc_dynamic_slippage(
    qty: float,
    adv20: float,
    vol: float,
    base_slippage: float = 0.0010,
    half_spread_floor: float = 0.0012,
    eta: float = 0.75,
    delta: float = 0.55,
    gamma_perm: float = 0.30
) -> float:
    """
    Computes dynamic market impact slippage under HOSE microstructure:
    IS(Q) = half_spread_floor (min 12 bps) + eta * vol * (Q / ADV20)^0.55 + gamma_perm * vol * (Q / ADV20).
    """
    if adv20 <= 1.0 or qty <= 0.0:
        return max(base_slippage, half_spread_floor)
    q_ratio = qty / adv20
    temp_impact = eta * vol * (q_ratio ** delta)
    perm_impact = gamma_perm * vol * q_ratio
    total_slip = half_spread_floor + temp_impact + perm_impact
    return max(base_slippage, max(half_spread_floor, total_slip))


@numba.njit(cache=True)
def _simulate_vn_microstructure_core(
    raw_open: np.ndarray,
    raw_close: np.ndarray,
    ceiling: np.ndarray,
    floor: np.ndarray,
    adv20_shares: np.ndarray,
    target_weights: np.ndarray,
    initial_cash: float,
    broker_fee: float,
    slippage: float,
    sell_tax: float,
    advance_fee_rate: float,
    max_adv_participation: float,
    calendar_days_t1: np.ndarray,
    calendar_days_t2: np.ndarray,
    min_advance_fee: float,
    corp_div_ex: np.ndarray,
    corp_div_pay_t: np.ndarray,
    corp_stock_ratio: np.ndarray,
    corp_stock_list_t: np.ndarray,
    in_pit_universe: np.ndarray,
    daily_vol: np.ndarray,
    penalty_rate_per_day: float,
    is_t3_regime: np.ndarray,
    min_advance_notional: float = 0.0,
    half_spread_floor: float = 0.0012
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-performance Numba engine simulating Vietnam microstructure:
    - Dynamic T+3 (pre-2022-08-29) vs T+2.5 (post-2022-08-29) settlement
    - Independent accounting of Brokerage Fees (Buy/Sell 0.15%) and Transfer Tax (Sell 0.10%)
    - Dynamic advance cash borrowing interest based on calendar days with 50,000 VND floor fee
    - Dual corporate actions ledger: dividend_cash_receivable and locked_corp_action_shares
    - Priority queue for floor-trapped sell orders with parameterized adverse penalty upon exit
    - Realized daily turnover directly computed from executed transaction volume
    - Point-in-Time constituent masking with liquidation_grace_mask (orderly exit <= 5 sessions)
    - 3-tier order book market impact model with 12 bps half-spread floor
    - Causal liquidity & volatility lagging (t-1) for sizing and slippage
    """
    T, N = raw_open.shape
    equity = np.zeros(T, dtype=np.float64)
    cash_history = np.zeros(T, dtype=np.float64)
    realized_fees = np.zeros(T, dtype=np.float64)
    advance_fees = np.zeros(T, dtype=np.float64)
    broker_fees_hist = np.zeros(T, dtype=np.float64)
    sell_tax_hist = np.zeros(T, dtype=np.float64)
    daily_turnover_hist = np.zeros(T, dtype=np.float64)
    div_receivable_hist = np.zeros(T, dtype=np.float64)
    locked_corp_hist = np.zeros(T, dtype=np.float64)
    
    cash_avail = initial_cash
    pending_cash_t1 = 0.0
    pending_cash_t0 = 0.0
    
    shares_held = np.zeros(N, dtype=np.float64)
    shares_sellable = np.zeros(N, dtype=np.float64)
    buy_t2 = np.zeros(N, dtype=np.float64)
    buy_t1 = np.zeros(N, dtype=np.float64)
    buy_t0 = np.zeros(N, dtype=np.float64)
    
    # Corporate Actions and Floor-trapped state trackers
    div_payout_schedule = np.zeros(T, dtype=np.float64)
    stock_listing_schedule = np.zeros((T, N), dtype=np.float64)
    trapped_sell_shares = np.zeros(N, dtype=np.float64)
    trapped_days = np.zeros(N, dtype=np.int64)
    liquidation_grace_counter = np.zeros(N, dtype=np.int64)
    locked_corp_shares = np.zeros(N, dtype=np.float64)
    cum_div_receivable = 0.0
    
    cum_broker_fees = 0.0
    cum_sell_tax = 0.0
    cum_adv_fees = 0.0
    
    for t in range(T):
        if t == 0:
            # Session t=0 is the initialization baseline session.
            # 100% cash is held, zero orders executed to prevent look-ahead bias
            # from same-day volume (daily_vol[0] / adv20[0] known only post-close).
            equity[0] = initial_cash
            cash_history[0] = initial_cash
            realized_fees[0] = 0.0
            advance_fees[0] = 0.0
            broker_fees_hist[0] = 0.0
            sell_tax_hist[0] = 0.0
            daily_turnover_hist[0] = 0.0
            div_receivable_hist[0] = 0.0
            locked_corp_hist[0] = 0.0
            continue

        t_lag = t - 1
        # ----------------------------------------------------------------------
        # A. SETTLEMENT UPDATES & CORPORATE ACTIONS (At Open of day t)
        # ----------------------------------------------------------------------
        # 1. Stocks settlement: T+3 ATO (before 2022-08-29) vs T+2.5 (from 2022-08-29 onwards)
        if is_t3_regime[t]:
            # T+3 regime: 3-step queue (buy_t2 -> buy_t1 -> buy_t0), unlocks at ATO morning of T+3
            shares_sellable += buy_t2
            buy_t2 = buy_t1.copy()
            buy_t1 = buy_t0.copy()
            buy_t0[:] = 0.0
        else:
            # T+2.5 regime: shares bought at T settle into available at T+2
            shares_sellable += buy_t2 + buy_t1
            buy_t2[:] = 0.0
            buy_t1 = buy_t0.copy()
            buy_t0[:] = 0.0
        
        # 2. Cash from sales at t-2 settles into available cash today
        cash_avail += pending_cash_t1
        pending_cash_t1 = pending_cash_t0
        pending_cash_t0 = 0.0
        
        # 3. Process corporate action cash dividend payouts scheduled for today
        payout_today = div_payout_schedule[t]
        if payout_today > 0.0:
            cash_avail += payout_today
            cum_div_receivable -= payout_today
            if cum_div_receivable < 0.0:
                cum_div_receivable = 0.0
                
        # 4. Process corporate action stock dividend listings scheduled for today
        for i in range(N):
            unlocked_stock = stock_listing_schedule[t, i]
            if unlocked_stock > 0.0:
                shares_sellable[i] += unlocked_stock
                shares_held[i] += unlocked_stock
                locked_corp_shares[i] -= unlocked_stock
                if locked_corp_shares[i] < 0.0:
                    locked_corp_shares[i] = 0.0
                    
        # 5. Check any new corporate actions taking effect on ex_date t
        for i in range(N):
            c_div = corp_div_ex[t, i]
            if c_div > 0.0 and shares_held[i] > 0.0:
                p_day = int(corp_div_pay_t[t, i])
                div_amt = shares_held[i] * c_div
                cum_div_receivable += div_amt
                if p_day < T:
                    div_payout_schedule[p_day] += div_amt
                    
            s_ratio = corp_stock_ratio[t, i]
            if s_ratio > 0.0 and shares_held[i] > 0.0:
                l_day = int(corp_stock_list_t[t, i])
                bonus_sh = np.floor(shares_held[i] * s_ratio)
                locked_corp_shares[i] += bonus_sh
                if l_day < T:
                    stock_listing_schedule[l_day, i] += bonus_sh

        # ----------------------------------------------------------------------
        # B. POINT-IN-TIME CONSTITUENT CHECK & ORDER SIZING
        # ----------------------------------------------------------------------
        holdings_val_open = 0.0
        locked_corp_val_open = 0.0
        for i in range(N):
            if shares_held[i] > 0.0:
                holdings_val_open += shares_held[i] * raw_open[t, i]
            if locked_corp_shares[i] > 0.0:
                locked_corp_val_open += locked_corp_shares[i] * raw_open[t, i]
            
            # Liquidation grace check: stock excluded from PIT universe
            if not in_pit_universe[t, i]:
                if shares_held[i] > 0.0:
                    target_weights[t, i] = 0.0
                    liquidation_grace_counter[i] += 1
                else:
                    liquidation_grace_counter[i] = 0
            else:
                liquidation_grace_counter[i] = 0
            
        port_val_open = (
            cash_avail + pending_cash_t1 + pending_cash_t0
            + cum_div_receivable + holdings_val_open + locked_corp_val_open
        )
        
        delta_shares = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if not in_pit_universe[t, i] and shares_held[i] > 0.0:
                # Forcibly exit excluded stock
                target_sh = 0.0
            else:
                target_val = target_weights[t, i] * port_val_open
                price_i = raw_open[t, i]
                if np.isfinite(price_i) and price_i > 0.0 and np.isfinite(target_val):
                    target_sh = np.floor(target_val / (price_i * 100.0)) * 100.0
                else:
                    target_sh = 0.0
            delta_shares[i] = target_sh - shares_held[i]
            
        daily_turnover_traded = 0.0

        # ----------------------------------------------------------------------
        # C1. EXECUTE FLOOR-TRAPPED SELLS QUEUE FIRST (Priority Exit)
        # ----------------------------------------------------------------------
        for i in range(N):
            if trapped_sell_shares[i] > 0.0:
                p_open = raw_open[t, i]
                flr_p = floor[t, i]
                
                # Check if still trapped at floor limit
                if p_open <= flr_p + 1e-4:
                    trapped_days[i] += 1
                else:
                    # Liquidity returned! Execute with parameterized adverse penalty slippage
                    q_trap = trapped_sell_shares[i]
                    penalty_rate = min(0.40, penalty_rate_per_day * float(trapped_days[i]))
                    vol_i = daily_vol[t_lag, i]
                    adv_i = adv20_shares[t_lag, i]
                    dyn_slip = _calc_dynamic_slippage(q_trap, adv_i, vol_i, base_slippage=slippage, half_spread_floor=half_spread_floor)
                    exec_price = p_open * (1.0 - dyn_slip - penalty_rate)
                    gross_val = q_trap * exec_price
                    b_fee = gross_val * broker_fee
                    s_tax = gross_val * sell_tax
                    cum_broker_fees += b_fee
                    cum_sell_tax += s_tax
                    net_proceeds = gross_val - b_fee - s_tax
                    pending_cash_t0 += net_proceeds
                    shares_held[i] -= q_trap
                    daily_turnover_traded += q_trap * p_open
                    trapped_sell_shares[i] = 0.0
                    trapped_days[i] = 0

        # ----------------------------------------------------------------------
        # C2. EXECUTE NORMAL & GRACE LIQUIDATION SELLS (Frees cash for buys)
        # ----------------------------------------------------------------------
        for i in range(N):
            # Liquidation grace check: if stock excluded for >= 5 sessions, liquidate all sellable
            if liquidation_grace_counter[i] >= 5 and shares_sellable[i] > 0.0:
                desired_sell = shares_sellable[i]
            elif delta_shares[i] < -50.0:
                desired_sell = -delta_shares[i]
            else:
                desired_sell = 0.0
                
            if desired_sell > 0.0:
                p_open = raw_open[t, i]
                flr_p = floor[t, i]
                if p_open <= 0.0 or not np.isfinite(p_open):
                    continue
                
                avail_sell = max(0.0, shares_sellable[i] - trapped_sell_shares[i])
                qty_sell = min(desired_sell, avail_sell)
                qty_sell = np.floor(qty_sell / 100.0) * 100.0
                
                # Expand participation limit to 10% ADV if grace period reached to ensure complete exit
                eff_max_adv = max_adv_participation if liquidation_grace_counter[i] < 5 else max(max_adv_participation, 0.10)
                adv_limit = np.floor((adv20_shares[t_lag, i] * eff_max_adv) / 100.0) * 100.0
                qty_sell = min(qty_sell, adv_limit)
                
                if qty_sell > 0.0:
                    # Check floor trap at ATO open
                    if p_open <= flr_p + 1e-4:
                        trapped_sell_shares[i] += qty_sell
                        trapped_days[i] = 1
                    else:
                        shares_held[i] -= qty_sell
                        shares_sellable[i] -= qty_sell
                        
                        vol_i = daily_vol[t_lag, i]
                        adv_i = adv20_shares[t_lag, i]
                        dyn_slip = _calc_dynamic_slippage(qty_sell, adv_i, vol_i, base_slippage=slippage, half_spread_floor=half_spread_floor)
                        exec_price = p_open * (1.0 - dyn_slip)
                        gross_val = qty_sell * exec_price
                        b_fee = gross_val * broker_fee
                        s_tax = gross_val * sell_tax
                        cum_broker_fees += b_fee
                        cum_sell_tax += s_tax
                        
                        net_proceeds = gross_val - b_fee - s_tax
                        pending_cash_t0 += net_proceeds
                        daily_turnover_traded += qty_sell * p_open

        # ----------------------------------------------------------------------
        # D. EXECUTE BUYS SECOND
        # ----------------------------------------------------------------------
        for i in range(N):
            # Only allow opening buys for valid in-universe constituents
            if not in_pit_universe[t, i]:
                continue
                
            if delta_shares[i] > 50.0:
                p_open = raw_open[t, i]
                ceil_p = ceiling[t, i]
                
                # Check valid positive open price and ceiling lock
                if p_open <= 0.0 or not np.isfinite(p_open):
                    continue
                if p_open >= ceil_p - 1e-4:
                    continue  # Ceiling locked, cannot purchase
                    
                desired_buy = delta_shares[i]
                adv_limit = np.floor((adv20_shares[t_lag, i] * max_adv_participation) / 100.0) * 100.0
                qty_buy = min(desired_buy, adv_limit)
                qty_buy = np.floor(qty_buy / 100.0) * 100.0
                
                if qty_buy <= 0.0:
                    continue
                    
                vol_i = daily_vol[t_lag, i]
                adv_i = adv20_shares[t_lag, i]
                dyn_slip = _calc_dynamic_slippage(qty_buy, adv_i, vol_i, base_slippage=slippage, half_spread_floor=half_spread_floor)
                exec_price = p_open * (1.0 + dyn_slip)
                cost_per_share = exec_price * (1.0 + broker_fee)
                total_needed = qty_buy * cost_per_share
                
                days_t1 = calendar_days_t1[t]
                days_t0 = calendar_days_t2[t]
                
                if total_needed <= cash_avail:
                    cash_avail -= total_needed
                    b_fee = qty_buy * exec_price * broker_fee
                    cum_broker_fees += b_fee
                    daily_turnover_traded += qty_buy * p_open
                else:
                    shortfall = total_needed - cash_avail
                    if min_advance_notional > 0.0 and shortfall < min_advance_notional:
                        denom = cost_per_share * 100.0
                        if denom > 0.0 and np.isfinite(denom):
                            scaled_qty = np.floor(cash_avail / denom) * 100.0
                            if scaled_qty > 0.0:
                                qty_buy = scaled_qty
                                vol_i = daily_vol[t_lag, i]
                                adv_i = adv20_shares[t_lag, i]
                                dyn_slip = _calc_dynamic_slippage(qty_buy, adv_i, vol_i, base_slippage=slippage, half_spread_floor=half_spread_floor)
                                exec_price = p_open * (1.0 + dyn_slip)
                                cost_per_share = exec_price * (1.0 + broker_fee)
                                actual_needed = qty_buy * cost_per_share
                                if actual_needed <= cash_avail:
                                    cash_avail -= actual_needed
                                    b_fee = qty_buy * exec_price * broker_fee
                                    cum_broker_fees += b_fee
                                    daily_turnover_traded += qty_buy * p_open
                                    shares_held[i] += qty_buy
                                    buy_t0[i] += qty_buy
                        continue
                        
                    cap_t1 = _calc_borrow_cap(pending_cash_t1, advance_fee_rate, days_t1, min_advance_fee)
                    cap_t0 = _calc_borrow_cap(pending_cash_t0, advance_fee_rate, days_t0, min_advance_fee)
                    total_adv_capacity = cap_t1 + cap_t0
                    
                    if total_adv_capacity >= shortfall:
                        cash_avail = 0.0
                        b_fee = qty_buy * exec_price * broker_fee
                        cum_broker_fees += b_fee
                        daily_turnover_traded += qty_buy * p_open
                        
                        take_t1 = min(shortfall, cap_t1)
                        adv_cost_t1 = max(min_advance_fee, take_t1 * advance_fee_rate * days_t1)
                        pending_cash_t1 -= (take_t1 + adv_cost_t1)
                        cum_adv_fees += adv_cost_t1
                        
                        rem = shortfall - take_t1
                        if rem > 0.0:
                            take_t0 = rem
                            adv_cost_t0 = max(min_advance_fee, take_t0 * advance_fee_rate * days_t0)
                            pending_cash_t0 -= (take_t0 + adv_cost_t0)
                            cum_adv_fees += adv_cost_t0
                    else:
                        max_budget = cash_avail + total_adv_capacity
                        denom = cost_per_share * 100.0
                        if denom <= 0.0 or not np.isfinite(denom):
                            continue
                        scaled_qty = np.floor(max_budget / denom) * 100.0
                        if scaled_qty <= 0.0:
                            continue
                            
                        qty_buy = scaled_qty
                        vol_i = daily_vol[t_lag, i]
                        adv_i = adv20_shares[t_lag, i]
                        dyn_slip = _calc_dynamic_slippage(qty_buy, adv_i, vol_i, base_slippage=slippage, half_spread_floor=half_spread_floor)
                        exec_price = p_open * (1.0 + dyn_slip)
                        cost_per_share = exec_price * (1.0 + broker_fee)
                        actual_needed = qty_buy * cost_per_share
                        b_fee = qty_buy * exec_price * broker_fee
                        cum_broker_fees += b_fee
                        daily_turnover_traded += qty_buy * p_open
                        
                        if actual_needed <= cash_avail:
                            cash_avail -= actual_needed
                        else:
                            shortfall = actual_needed - cash_avail
                            if min_advance_notional > 0.0 and shortfall < min_advance_notional:
                                continue
                            cash_avail = 0.0
                            
                            take_t1 = min(shortfall, cap_t1)
                            adv_cost_t1 = max(min_advance_fee, take_t1 * advance_fee_rate * days_t1)
                            pending_cash_t1 -= (take_t1 + adv_cost_t1)
                            cum_adv_fees += adv_cost_t1
                            
                            rem = shortfall - take_t1
                            if rem > 0.0:
                                take_t0 = rem
                                adv_cost_t0 = max(min_advance_fee, take_t0 * advance_fee_rate * days_t0)
                                pending_cash_t0 -= (take_t0 + adv_cost_t0)
                                cum_adv_fees += adv_cost_t0
                                
                shares_held[i] += qty_buy
                buy_t0[i] += qty_buy  # Locked until day t+2
                
        # ----------------------------------------------------------------------
        # E. END-OF-DAY VALUATION AT CLOSE
        # ----------------------------------------------------------------------
        holdings_val_close = 0.0
        locked_corp_val_close = 0.0
        for i in range(N):
            if shares_held[i] > 0.0:
                holdings_val_close += shares_held[i] * raw_close[t, i]
            if locked_corp_shares[i] > 0.0:
                locked_corp_val_close += locked_corp_shares[i] * raw_close[t, i]
            
        equity[t] = (
            cash_avail + pending_cash_t1 + pending_cash_t0
            + cum_div_receivable + holdings_val_close + locked_corp_val_close
        )
        cash_history[t] = cash_avail
        realized_fees[t] = cum_broker_fees + cum_sell_tax
        advance_fees[t] = cum_adv_fees
        broker_fees_hist[t] = cum_broker_fees
        sell_tax_hist[t] = cum_sell_tax
        div_receivable_hist[t] = cum_div_receivable
        locked_corp_hist[t] = locked_corp_val_close
        
        if port_val_open > 0.0:
            daily_turnover_hist[t] = daily_turnover_traded / (2.0 * port_val_open)
        else:
            daily_turnover_hist[t] = 0.0
            
    return (
        equity,
        cash_history,
        realized_fees,
        advance_fees,
        broker_fees_hist,
        sell_tax_hist,
        daily_turnover_hist,
        div_receivable_hist,
        locked_corp_hist
    )


# ==============================================================================
# 3. NAIVE FRICTIONLESS SIMULATOR (US/CRYPTO T+0 BASELINE)
# ==============================================================================

def simulate_naive(
    raw_close: np.ndarray,
    target_weights: np.ndarray,
    initial_cash: float = 1.0e9,
    raw_open: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Theoretical frictionless baseline backtest (T+0, 0% fees, 0% taxes, continuous execution).
    Used to benchmark Implementation Shortfall against the real Vietnam execution engine.
    """
    raw_close = np.asarray(raw_close, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    T, N = raw_close.shape
    
    equity = np.zeros(T, dtype=np.float64)
    equity[0] = initial_cash
    
    if raw_open is not None:
        raw_open = np.asarray(raw_open, dtype=np.float64)
        for t in range(1, T):
            w_pre = np.zeros(N, dtype=np.float64) if t == 1 else target_weights[t - 1]
            denom_c = np.maximum(raw_close[t - 1], 1e-6)
            ret_overnight = np.where(
                np.isfinite(raw_open[t]) & np.isfinite(raw_close[t - 1]),
                (raw_open[t] - raw_close[t - 1]) / denom_c,
                0.0
            )
            r_overnight = float(np.sum(np.nan_to_num(w_pre * ret_overnight)))

            denom_o = np.maximum(raw_open[t], 1e-6)
            ret_intraday = np.where(
                np.isfinite(raw_close[t]) & np.isfinite(raw_open[t]),
                (raw_close[t] - raw_open[t]) / denom_o,
                0.0
            )
            r_intraday = float(np.sum(np.nan_to_num(target_weights[t] * ret_intraday)))

            equity[t] = equity[t - 1] * (1.0 + r_overnight) * (1.0 + r_intraday)
    else:
        # Daily asset returns (legacy close-to-close)
        returns = np.zeros((T, N), dtype=np.float64)
        returns[1:] = (raw_close[1:] - raw_close[:-1]) / np.maximum(raw_close[:-1], 1e-6)
        
        for t in range(1, T):
            port_ret = np.sum(target_weights[t - 1] * returns[t])
            equity[t] = equity[t - 1] * (1.0 + port_ret)
        
    return equity


# ==============================================================================
# 4. VIETNAM REALISTIC SIMULATOR WRAPPER & INPUT SANITATION
# ==============================================================================

def sanitize_market_matrices(
    raw_open: np.ndarray,
    raw_close: np.ndarray,
    ceiling: np.ndarray,
    floor: np.ndarray,
    adv20_shares: np.ndarray,
    target_weights: np.ndarray,
    in_pit_universe: Optional[np.ndarray] = None,
    max_missing_sessions: int = 5,
    halt_haircut: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sanitizes market input matrices before feeding into Numba simulator.
    Adheres to B3, S4, T5, T13:
    1. Replaces NaNs in raw_close via forward-fill; if raw_open is NaN, fills with prior raw_close.
    2. Tracks consecutive missing/NaN sessions per asset:
       - If missing <= max_missing_sessions (5): mask as untradable (target_weights=0, in_pit_universe=False).
       - If missing > max_missing_sessions: asset transitions to 'suspended' state (A8).
    3. Mark-to-market halt valuation (T13):
       - If halt_haircut > 0 and asset is suspended, position valuation uses raw_close * (1 - halt_haircut)
         starting from the observed halt session.
    """
    T, N = raw_close.shape
    clean_open = np.asarray(raw_open, dtype=np.float64).copy()
    clean_close = np.asarray(raw_close, dtype=np.float64).copy()
    clean_ceil = np.asarray(ceiling, dtype=np.float64).copy()
    clean_floor = np.asarray(floor, dtype=np.float64).copy()
    clean_adv = np.asarray(adv20_shares, dtype=np.float64).copy()
    clean_weights = np.asarray(target_weights, dtype=np.float64).copy()
    
    if in_pit_universe is None:
        clean_pit = np.ones((T, N), dtype=np.bool_)
    else:
        clean_pit = np.asarray(in_pit_universe, dtype=np.bool_).copy()

    for j in range(N):
        consecutive_missing = 0
        last_valid_close = np.nan
        for t in range(T):
            is_close_nan = np.isnan(clean_close[t, j])
            is_open_nan = np.isnan(clean_open[t, j])
            
            if is_close_nan or is_open_nan:
                consecutive_missing += 1
                clean_pit[t, j] = False
                clean_weights[t, j] = 0.0
                
                # Forward fill close if available
                if not np.isnan(last_valid_close):
                    fill_price = last_valid_close
                    # T13: If suspended (> max_missing_sessions), apply haircut from observed halt session
                    if consecutive_missing > max_missing_sessions and halt_haircut > 0.0:
                        fill_price = last_valid_close * (1.0 - halt_haircut)
                    clean_close[t, j] = fill_price
                    clean_open[t, j] = fill_price
                    clean_ceil[t, j] = fill_price * 1.07
                    clean_floor[t, j] = fill_price * 0.93
                else:
                    clean_close[t, j] = 0.0
                    clean_open[t, j] = 0.0
                    clean_ceil[t, j] = 0.0
                    clean_floor[t, j] = 0.0
                    
                clean_adv[t, j] = 0.0
            else:
                consecutive_missing = 0
                last_valid_close = clean_close[t, j]
                
    # Normalize weights row-wise if positive sum
    for t in range(T):
        w_sum = np.sum(clean_weights[t, :])
        if w_sum > 1.0:
            clean_weights[t, :] /= w_sum
            
    return clean_open, clean_close, clean_ceil, clean_floor, clean_adv, clean_weights, clean_pit


def simulate_vietnam(
    raw_open: np.ndarray,
    raw_close: np.ndarray,
    ceiling: np.ndarray,
    floor: np.ndarray,
    adv20_shares: np.ndarray,
    target_weights: np.ndarray,
    initial_cash: float = 1.0e9,
    broker_fee: float = 0.0015,
    slippage: float = 0.0010,
    sell_tax: float = 0.0010,
    advance_fee_rate: float = 0.00035,
    max_adv_participation: float = 0.03,
    calendar_days_t1: Optional[np.ndarray] = None,
    calendar_days_t2: Optional[np.ndarray] = None,
    min_advance_fee: float = 50000.0,
    corp_action_div_cash: Optional[np.ndarray] = None,
    corp_action_div_pay_t: Optional[np.ndarray] = None,
    corp_action_stock_ratio: Optional[np.ndarray] = None,
    corp_action_stock_list_t: Optional[np.ndarray] = None,
    dates: Optional[Any] = None,
    in_pit_universe: Optional[np.ndarray] = None,
    daily_vol: Optional[np.ndarray] = None,
    halt_haircut: float = 0.0,
    sanitize_inputs: bool = True,
    split_date: Optional[Union[str, datetime.date]] = "2022-08-29",
    penalty_rate_per_day: float = 0.0,
    min_advance_notional: float = 0.0,
    half_spread_floor: float = 0.0012
) -> Dict[str, Any]:
    """
    Full simulation wrapper executing the Numba-compiled realistic Vietnam microstructure engine:
    - Split T+3 / T+2.5 settlement (split_date: default '2022-08-29')
    - Parameterized floor trapped penalty (penalty_rate_per_day: default 0.0)
    - Independent tax & fee accounting
    - Calendar day advance loan financing (min 50k floor)
    - Corporate actions dual ledger
    - 3-tier market impact model with 12 bps half-spread floor
    - Liquidation grace mask (<= 5 sessions orderly liquidation)
    """
    raw_open = np.asarray(raw_open, dtype=np.float64)
    raw_close = np.asarray(raw_close, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    adv20_shares = np.asarray(adv20_shares, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64).copy()
    
    if sanitize_inputs:
        (
            raw_open,
            raw_close,
            ceiling,
            floor,
            adv20_shares,
            target_weights,
            in_pit_universe
        ) = sanitize_market_matrices(
            raw_open=raw_open,
            raw_close=raw_close,
            ceiling=ceiling,
            floor=floor,
            adv20_shares=adv20_shares,
            target_weights=target_weights,
            in_pit_universe=in_pit_universe,
            max_missing_sessions=5,
            halt_haircut=halt_haircut
        )
        
    T, N = raw_open.shape
    
    # 1. Setup Calendar Days
    if calendar_days_t1 is None:
        c_days_t1 = np.ones(T, dtype=np.float64)
        if dates is not None and len(dates) == T:
            for t in range(T - 1):
                d0 = dates[t]
                d1 = dates[t + 1]
                if isinstance(d0, str):
                    d0 = datetime.date.fromisoformat(d0)
                if isinstance(d1, str):
                    d1 = datetime.date.fromisoformat(d1)
                c_days_t1[t] = max(1.0, float((d1 - d0).days))
            d_last = dates[-1]
            if isinstance(d_last, str):
                d_last = datetime.date.fromisoformat(d_last)
            c_days_t1[-1] = 3.0 if d_last.weekday() == 4 else 1.0
        calendar_days_t1 = c_days_t1
    else:
        calendar_days_t1 = np.asarray(calendar_days_t1, dtype=np.float64)
        
    if calendar_days_t2 is None:
        c_days_t2 = np.full(T, 2.0, dtype=np.float64)
        if dates is not None and len(dates) == T:
            for t in range(T - 2):
                d0 = dates[t]
                d2 = dates[t + 2]
                if isinstance(d0, str):
                    d0 = datetime.date.fromisoformat(d0)
                if isinstance(d2, str):
                    d2 = datetime.date.fromisoformat(d2)
                c_days_t2[t] = max(2.0, float((d2 - d0).days))
            for t in range(max(0, T - 2), T):
                d_t = dates[t]
                if isinstance(d_t, str):
                    d_t = datetime.date.fromisoformat(d_t)
                c_days_t2[t] = 4.0 if d_t.weekday() in (3, 4) else 2.0
        calendar_days_t2 = c_days_t2
    else:
        calendar_days_t2 = np.asarray(calendar_days_t2, dtype=np.float64)
        
    # 2. Setup Corporate Actions matrices
    if corp_action_div_cash is None:
        corp_div_ex = np.zeros((T, N), dtype=np.float64)
    else:
        corp_div_ex = np.asarray(corp_action_div_cash, dtype=np.float64)
        
    if corp_action_div_pay_t is None:
        corp_div_pay_t = np.zeros((T, N), dtype=np.float64)
    else:
        corp_div_pay_t = np.asarray(corp_action_div_pay_t, dtype=np.float64)
        
    if corp_action_stock_ratio is None:
        corp_stock_ratio = np.zeros((T, N), dtype=np.float64)
    else:
        corp_stock_ratio = np.asarray(corp_action_stock_ratio, dtype=np.float64)
        
    if corp_action_stock_list_t is None:
        corp_stock_list_t = np.zeros((T, N), dtype=np.float64)
    else:
        corp_stock_list_t = np.asarray(corp_action_stock_list_t, dtype=np.float64)
        
    # 3. Setup PIT Universe Mask and Daily Volatility
    if in_pit_universe is None:
        in_pit_universe_arr = np.ones((T, N), dtype=np.bool_)
    else:
        in_pit_universe_arr = np.asarray(in_pit_universe, dtype=np.bool_)
        
    if daily_vol is None:
        ret = np.zeros((T, N), dtype=np.float64)
        ret[1:] = (raw_close[1:] - raw_close[:-1]) / np.maximum(raw_close[:-1], 1.0)
        daily_vol_arr = np.zeros((T, N), dtype=np.float64)
        win = min(20, max(1, T))
        for j in range(N):
            daily_vol_arr[:, j] = bn.move_std(ret[:, j], window=win, min_count=1)
        daily_vol_arr[np.isnan(daily_vol_arr) | (daily_vol_arr <= 1e-4)] = 0.015
    else:
        daily_vol_arr = np.asarray(daily_vol, dtype=np.float64)

    # 4. Setup T+3 (pre-split) vs T+2.5 (post-split) settlement regimes
    is_t3_regime = np.zeros(T, dtype=np.bool_)
    if dates is not None and split_date is not None and len(dates) == T:
        split_dt = datetime.date.fromisoformat(split_date) if isinstance(split_date, str) else split_date
        for t in range(T):
            d_t = dates[t]
            if isinstance(d_t, str):
                d_t = datetime.date.fromisoformat(d_t)
            elif hasattr(d_t, "date"):
                d_t = d_t.date()
            if d_t < split_dt:
                is_t3_regime[t] = True
        
    (
        eq,
        cash,
        fees,
        adv_fees,
        b_fees,
        s_tax,
        to_hist,
        div_rec,
        corp_val
    ) = _simulate_vn_microstructure_core(
        raw_open=raw_open,
        raw_close=raw_close,
        ceiling=ceiling,
        floor=floor,
        adv20_shares=adv20_shares,
        target_weights=target_weights,
        initial_cash=initial_cash,
        broker_fee=broker_fee,
        slippage=slippage,
        sell_tax=sell_tax,
        advance_fee_rate=advance_fee_rate,
        max_adv_participation=max_adv_participation,
        calendar_days_t1=calendar_days_t1,
        calendar_days_t2=calendar_days_t2,
        min_advance_fee=min_advance_fee,
        corp_div_ex=corp_div_ex,
        corp_div_pay_t=corp_div_pay_t,
        corp_stock_ratio=corp_stock_ratio,
        corp_stock_list_t=corp_stock_list_t,
        in_pit_universe=in_pit_universe_arr,
        daily_vol=daily_vol_arr,
        penalty_rate_per_day=penalty_rate_per_day,
        is_t3_regime=is_t3_regime,
        min_advance_notional=min_advance_notional,
        half_spread_floor=half_spread_floor
    )
    
    if not np.all(np.isfinite(eq)):
        nan_indices = np.where(~np.isfinite(eq))[0]
        raise FloatingPointError(
            f"Simulator Invariant B3 Violated: Equity curve contains non-finite values (NaN/Inf) "
            f"at {len(nan_indices)} session(s), first detected at index {nan_indices[0]}."
        )
    
    return {
        "equity": eq,
        "cash_history": cash,
        "trading_fees": fees,
        "broker_fees": b_fees,
        "sell_tax": s_tax,
        "advance_interest_fees": adv_fees,
        "advance_fees": adv_fees,
        "realized_daily_turnover": to_hist,
        "mean_daily_turnover": float(np.mean(to_hist)),
        "dividend_cash_receivable": div_rec,
        "locked_corp_action_shares_val": corp_val,
        "total_friction": fees[-1] + adv_fees[-1]
    }


# ==============================================================================
# 5. VN30F FUTURES DERIVATIVE BETA HEDGE OVERLAY
# ==============================================================================

def calculate_vn30f_hedge(
    portfolio_value: float,
    beta: float,
    futures_price: float,
    multiplier: float = 100000.0,
    margin_req: float = 0.20
) -> Dict[str, Any]:
    """
    Calculates VN30F futures contract overlay to hedge broad market systemic risk:
    - Point value multiplier: 100,000 VND / point
    - Initial margin requirement: 20%
    
    Parameters:
    -----------
    portfolio_value : float (in VND)
    beta : float (portfolio beta to VN30 Index)
    futures_price : float (in index points, e.g. 1280.0)
    multiplier : float, default 100,000 VND/pt
    margin_req : float, default 0.20 (20%)
    
    Returns:
    --------
    Dict containing hedge specification.
    """
    contract_val = futures_price * multiplier
    if contract_val <= 0:
        return {"contracts": 0, "margin_required": 0.0, "effective_beta": beta}
        
    hedge_notional = beta * portfolio_value
    num_contracts = int(round(hedge_notional / contract_val))
    margin_required = num_contracts * contract_val * margin_req
    actual_hedge_notional = num_contracts * contract_val
    hedged_beta = beta - (actual_hedge_notional / portfolio_value) if portfolio_value > 0 else 0.0
    
    return {
        "contracts_to_short": num_contracts,
        "contract_value_vnd": contract_val,
        "total_hedge_notional_vnd": actual_hedge_notional,
        "margin_required_vnd": margin_required,
        "original_beta": beta,
        "residual_beta": hedged_beta
    }


def calculate_dynamic_vn30f_hedge(
    portfolio_value: float,
    beta_cw: float,
    futures_price: float,
    lambda_contagion: float,
    vn30_close_history: Optional[np.ndarray] = None,
    prev_h: float = 0.0,
    deadband: float = 0.20,
    multiplier: float = 100000.0,
    deriv_cash_buffer_pct: float = 0.18,
    margin_req: float = 0.20,
    total_margin_buffer: float = 0.30
) -> Dict[str, Any]:
    """
    Dynamic Contagion Regime Hedge Overlay with VN30F1M Futures (Phase 4 / Section 4.1 Q2):
    - Replaces static 15-contract short hedge with regime-aware logistic switching.
    - Market regime M in {0, 1, 2}:
        M = I(P_VN30 < SMA_50) + I(vol_10d > 1.35 * vol_60d)
    - Logistic switching target hedge ratio h*(t):
        * Uptrend (M=0): theta_low=0.55, theta_mid=0.70, k=12 (Hedge OFF, 0 roll cost)
        * Shock (M>=1): theta_low=0.25, theta_mid=0.45, k=15 (Activates early downside hedge)
    - Whipsaw Deadband Filter (delta_h = 0.20):
        Updates only if |h*(t) - h(t-1)| >= 0.20, or lambda_tilde >= 0.80, or (h* == 0 and h(t-1) <= 0.25)
    - Sizing: N_contracts bounded by 18% NAV derivative cash buffer.
    """
    contract_val = futures_price * multiplier
    if contract_val <= 0 or portfolio_value <= 0:
        return {
            "contracts_to_short": 0,
            "h_target": 0.0,
            "h_actual": 0.0,
            "market_regime_M": 0,
            "margin_required": 0.0,
            "residual_beta": beta_cw
        }
        
    # 1. Determine Market Regime M in {0, 1, 2}
    M = 0
    if vn30_close_history is not None and len(vn30_close_history) >= 50:
        p_hist = np.asarray(vn30_close_history, dtype=np.float64)
        p_last = p_hist[-1]
        sma50 = float(np.mean(p_hist[-50:]))
        cond1 = p_last < sma50
        
        cond2 = False
        if len(p_hist) >= 61:
            rets = np.diff(np.log(np.maximum(p_hist[-61:], 1.0)))
            vol10 = float(np.std(rets[-10:]))
            vol60 = float(np.std(rets))
            cond2 = vol10 > 1.35 * vol60
            
        M = int(cond1) + int(cond2)
        
    # 2. Logistic Regime Parameters
    lambda_val = float(np.clip(lambda_contagion, 0.0, 1.0))
    if M == 0:
        theta_low = 0.55
        theta_mid = 0.70
        k = 12.0
    else:
        theta_low = 0.25
        theta_mid = 0.45
        k = 15.0
        
    # 3. Theoretical Target Hedge Ratio h*(t)
    if lambda_val < theta_low:
        h_star = 0.0
    else:
        h_star = 1.0 / (1.0 + np.exp(-k * (lambda_val - theta_mid)))
        
    # 4. Whipsaw Deadband Filter (delta_h = 0.20)
    deadband_triggered = False
    if abs(h_star - prev_h) >= deadband:
        h_actual = h_star
        deadband_triggered = True
    elif lambda_val >= 0.80:
        h_actual = h_star
        deadband_triggered = True
    elif h_star == 0.0 and prev_h <= 0.25:
        h_actual = 0.0
        deadband_triggered = True
    else:
        h_actual = prev_h
        
    # 5. Contract Sizing with Strict Margin Buffer Cap
    hedge_notional = h_actual * beta_cw * portfolio_value
    num_contracts = int(np.floor(hedge_notional / contract_val + 0.5))
    
    max_deriv_cash = portfolio_value * deriv_cash_buffer_pct
    max_contracts = int(np.floor(max_deriv_cash / (total_margin_buffer * contract_val)))
    num_contracts = max(0, min(num_contracts, max_contracts))
    
    margin_required = num_contracts * contract_val * margin_req
    actual_notional = num_contracts * contract_val
    residual_beta = beta_cw - (actual_notional / portfolio_value) if portfolio_value > 0 else 0.0
    
    return {
        "contracts_to_short": num_contracts,
        "max_contracts_allowed": max_contracts,
        "h_target": float(h_star),
        "h_actual": float(h_actual),
        "market_regime_M": M,
        "lambda_contagion": lambda_val,
        "deadband_triggered": deadband_triggered,
        "contract_value_vnd": contract_val,
        "margin_required_vnd": margin_required,
        "total_hedge_notional_vnd": actual_notional,
        "original_beta": beta_cw,
        "residual_beta": float(residual_beta)
    }


# ==============================================================================
# 6. STRATEGY EVALUATION & RISK MANAGEMENT (KILL-SWITCH)
# ==============================================================================

def evaluate_strategy(
    equity_curve: np.ndarray,
    daily_weights: np.ndarray,
    naive_equity_curve: Optional[np.ndarray] = None,
    benchmark_equity_curve: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.04,
    n_trials: int = 26,
    initial_capital: Optional[float] = 1.0e9,
    realized_daily_turnover: Optional[np.ndarray] = None,
    trial_returns: Optional[np.ndarray] = None,
    mean_trial_correlation: Optional[float] = None,
    var_sharpe_trials: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes institutional quant metrics based on 252 trading sessions/year:
    - CAGR, Annualized Volatility
    - Sharpe Ratio (post fee & tax), Sortino Ratio
    - Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) for multiple testing
    - Maximum Drawdown (MaxDD)
    - Realized Daily Turnover (%/day) from actual trade ledger
    - Implementation Shortfall decomposition
    - Market Beta & Alpha relative to benchmark
    - Automated Kill-Switch monitor (Triggers if MaxDD > 25% or Shortfall > 30%)
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    T = len(eq)
    
    if T < 2:
        return {}
        
    daily_returns = (eq[1:] - eq[:-1]) / np.maximum(eq[:-1], 1e-6)
    
    initial_eq = float(initial_capital) if initial_capital is not None else float(eq[0])
    final_eq = float(eq[-1])
    total_return = (final_eq / initial_eq) - 1.0
    
    # CAGR
    years = max(T / 252.0, 1.0 / 252.0)
    cagr = (final_eq / initial_eq) ** (1.0 / years) - 1.0
    
    # Volatility
    ann_vol = float(np.std(daily_returns) * np.sqrt(252.0))
    
    # Sharpe
    mean_ret = float(np.mean(daily_returns) * 252.0)
    sharpe = (mean_ret - risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0
    
    # Sortino (Downside deviation relative to risk free)
    downside_returns = daily_returns[daily_returns < 0.0]
    downside_vol = float(np.std(downside_returns) * np.sqrt(252.0)) if len(downside_returns) > 1 else ann_vol
    sortino = (mean_ret - risk_free_rate) / downside_vol if downside_vol > 1e-12 else 0.0
    
    # Deflated Sharpe Ratio (DSR)
    dsr_dict = compute_deflated_sharpe_ratio(
        daily_returns,
        n_trials=n_trials,
        risk_free_rate=risk_free_rate,
        annualization_factor=252.0,
        trial_returns=trial_returns,
        mean_trial_correlation=mean_trial_correlation,
        var_sharpe_trials=var_sharpe_trials
    )
    
    # Maximum Drawdown
    cummax = np.maximum.accumulate(eq)
    drawdowns = (cummax - eq) / cummax
    max_dd = float(np.max(drawdowns))
    
    # Daily Turnover (Realized ledger turnover preferred over theoretical weights difference)
    if realized_daily_turnover is not None:
        realized_arr = np.asarray(realized_daily_turnover, dtype=np.float64)
        daily_turnover = float(np.mean(realized_arr))
    else:
        w = np.asarray(daily_weights, dtype=np.float64)
        turnovers = 0.5 * np.sum(np.abs(w[1:] - w[:-1]), axis=1)
        daily_turnover = float(np.mean(turnovers))
    
    # Implementation Shortfall
    if naive_equity_curve is not None:
        naive_eq = np.asarray(naive_equity_curve, dtype=np.float64)
        naive_init = float(initial_capital) if initial_capital is not None else float(naive_eq[0])
        naive_cagr = (naive_eq[-1] / naive_init) ** (1.0 / years) - 1.0
        shortfall_pct = naive_cagr - cagr
        shortfall_ratio = (shortfall_pct / naive_cagr) if abs(naive_cagr) > 1e-12 else 0.0
    else:
        naive_cagr = cagr
        shortfall_pct = 0.0
        shortfall_ratio = 0.0
        
    # Benchmark Comparison (Beta & Alpha)
    market_beta = 1.0
    market_alpha = 0.0
    if benchmark_equity_curve is not None:
        bm_eq = np.asarray(benchmark_equity_curve, dtype=np.float64)
        bm_ret = (bm_eq[1:] - bm_eq[:-1]) / np.maximum(bm_eq[:-1], 1e-6)
        cov_mat = np.cov(daily_returns, bm_ret)
        if cov_mat[1, 1] > 1e-12:
            market_beta = float(cov_mat[0, 1] / cov_mat[1, 1])
            bm_mean = float(np.mean(bm_ret) * 252.0)
            market_alpha = float(mean_ret - (risk_free_rate + market_beta * (bm_mean - risk_free_rate)))
            
    # Kill-Switch Monitor
    kill_switch_triggered = False
    kill_switch_reasons = []
    
    if max_dd > 0.25:
        kill_switch_triggered = True
        kill_switch_reasons.append(f"Max Drawdown {max_dd * 100:.2f}% exceeded 25% threshold")
        
    if shortfall_ratio > 0.30:
        kill_switch_triggered = True
        kill_switch_reasons.append(f"Implementation Shortfall ratio {shortfall_ratio * 100:.2f}% exceeded 30% limit")
        
    return {
        "initial_equity": initial_eq,
        "final_equity": final_eq,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "deflated_sharpe_ratio": dsr_dict["deflated_sharpe_ratio"],
        "dsr_p_value": dsr_dict["p_value"],
        "expected_max_sharpe_null": dsr_dict["expected_max_sharpe"],
        "n_trials_evaluated": n_trials,
        "n_eff": dsr_dict.get("n_eff", float(n_trials)),
        "max_drawdown": max_dd,
        "daily_turnover": daily_turnover,
        "naive_cagr": naive_cagr,
        "implementation_shortfall_pct": shortfall_pct,
        "implementation_shortfall_ratio": shortfall_ratio,
        "market_beta": market_beta,
        "market_alpha": market_alpha,
        "kill_switch_triggered": kill_switch_triggered,
        "kill_switch_reasons": kill_switch_reasons
    }
