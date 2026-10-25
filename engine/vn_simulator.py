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

from typing import Dict, Any, Optional, Tuple
import numpy as np
import numba
from scipy import stats


# ==============================================================================
# DEFLATED SHARPE RATIO (BAILEY & LOPEZ DE PRADO, 2014)
# ==============================================================================

def compute_deflated_sharpe_ratio(
    daily_returns: np.ndarray,
    n_trials: int = 26,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 252.0
) -> Dict[str, float]:
    """
    Computes the Deflated Sharpe Ratio (DSR) to account for selection bias / multiple testing,
    non-normality (skewness, kurtosis), and sample length following:
    Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management.
    """
    returns = np.asarray(daily_returns, dtype=np.float64)
    T = len(returns)
    if T < 10:
        return {
            "deflated_sharpe_ratio": 0.0,
            "p_value": 1.0,
            "expected_max_sharpe": 0.0,
            "annualized_sharpe": 0.0
        }
        
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r <= 1e-12:
        return {
            "deflated_sharpe_ratio": 0.0,
            "p_value": 1.0,
            "expected_max_sharpe": 0.0,
            "annualized_sharpe": 0.0
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
    
    # Expected maximum Sharpe Ratio under Null Hypothesis of zero alpha across N trials
    # E[max_N {SR}] approx sr_std * ((1 - euler) * Z^-1(1 - 1/N) + euler * Z^-1(1 - 1/(N*e)))
    euler_mascheroni = 0.5772156649
    n_trials = max(n_trials, 1)
    if n_trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_daily_sr = sr_std * ((1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2)
    else:
        expected_max_daily_sr = 0.0
        
    expected_max_ann_sr = float(expected_max_daily_sr * np.sqrt(annualization_factor))
    
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
        "sample_skewness": skew,
        "sample_kurtosis": kurt
    }


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

@numba.njit(fastmath=True)
def _simulate_vn_microstructure_core(
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
    max_adv_participation: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-performance Numba engine simulating Vietnam microstructure:
    - T+2.5 settlement for stocks and sales proceeds
    - Liquidity traps at ceiling and floor
    - Lot size 100 constraint
    - Advance cash financing costs
    
    Returns:
    --------
    Tuple:
      - equity: shape (T,) daily portfolio equity at close
      - cash_history: shape (T,) available cash
      - realized_fees: shape (T,) cumulative trading fees and taxes
      - advance_fees: shape (T,) cumulative advance interest fees
    """
    T, N = raw_open.shape
    equity = np.zeros(T, dtype=np.float64)
    cash_history = np.zeros(T, dtype=np.float64)
    realized_fees = np.zeros(T, dtype=np.float64)
    advance_fees = np.zeros(T, dtype=np.float64)
    
    cash_avail = initial_cash
    pending_cash_t1 = 0.0
    pending_cash_t0 = 0.0
    
    shares_held = np.zeros(N, dtype=np.float64)
    shares_sellable = np.zeros(N, dtype=np.float64)
    buy_t1 = np.zeros(N, dtype=np.float64)
    buy_t0 = np.zeros(N, dtype=np.float64)
    
    cum_trade_fees = 0.0
    cum_adv_fees = 0.0
    
    for t in range(T):
        # ----------------------------------------------------------------------
        # A. SETTLEMENT UPDATES (At Open of day t)
        # ----------------------------------------------------------------------
        # Stocks bought at t-2 unlock into sellable today
        shares_sellable += buy_t1
        buy_t1 = buy_t0.copy()
        buy_t0[:] = 0.0
        
        # Cash from sales at t-2 settles into available cash today
        cash_avail += pending_cash_t1
        pending_cash_t1 = pending_cash_t0
        pending_cash_t0 = 0.0
        
        # ----------------------------------------------------------------------
        # B. VALUATION AT OPEN & TARGET ORDER SIZING
        # ----------------------------------------------------------------------
        holdings_val_open = 0.0
        for i in range(N):
            holdings_val_open += shares_held[i] * raw_open[t, i]
            
        port_val_open = cash_avail + pending_cash_t1 + pending_cash_t0 + holdings_val_open
        
        delta_shares = np.zeros(N, dtype=np.float64)
        for i in range(N):
            target_val = target_weights[t, i] * port_val_open
            # Round target shares down to round lot 100
            price_i = raw_open[t, i]
            if price_i > 0:
                target_sh = np.floor(target_val / (price_i * 100.0)) * 100.0
            else:
                target_sh = 0.0
            delta_shares[i] = target_sh - shares_held[i]
            
        # ----------------------------------------------------------------------
        # C. EXECUTE SELLS FIRST (Frees cash for buys)
        # ----------------------------------------------------------------------
        for i in range(N):
            if delta_shares[i] < -50.0:
                p_open = raw_open[t, i]
                flr_p = floor[t, i]
                
                # Liquidity Trap: Floor trapped (White on buy side, cannot sell)
                if p_open <= flr_p + 1e-4:
                    continue  # Order rejected, shares remain locked
                    
                desired_sell = -delta_shares[i]
                # T+2.5 Constraint: Can only sell unlocked shares
                qty_sell = min(desired_sell, shares_sellable[i])
                qty_sell = np.floor(qty_sell / 100.0) * 100.0
                
                # ADV participation limit
                adv_limit = np.floor((adv20_shares[t, i] * max_adv_participation) / 100.0) * 100.0
                qty_sell = min(qty_sell, adv_limit)
                
                if qty_sell > 0.0:
                    shares_held[i] -= qty_sell
                    shares_sellable[i] -= qty_sell
                    
                    gross_val = qty_sell * p_open
                    # Asymmetric sell costs: broker (0.15%) + slippage (0.10%) + tax (0.10%)
                    sell_cost_rate = broker_fee + slippage + sell_tax
                    cost_val = gross_val * sell_cost_rate
                    cum_trade_fees += cost_val
                    
                    net_proceeds = gross_val - cost_val
                    # Sales proceeds are pending for 2 days
                    pending_cash_t0 += net_proceeds
                    
        # ----------------------------------------------------------------------
        # D. EXECUTE BUYS SECOND
        # ----------------------------------------------------------------------
        for i in range(N):
            if delta_shares[i] > 50.0:
                p_open = raw_open[t, i]
                ceil_p = ceiling[t, i]
                
                # Liquidity Trap: Ceiling locked (White on sell side, cannot buy)
                if p_open >= ceil_p - 1e-4:
                    continue  # Order rejected, cannot purchase
                    
                desired_buy = delta_shares[i]
                # ADV participation limit
                adv_limit = np.floor((adv20_shares[t, i] * max_adv_participation) / 100.0) * 100.0
                qty_buy = min(desired_buy, adv_limit)
                qty_buy = np.floor(qty_buy / 100.0) * 100.0
                
                if qty_buy <= 0.0:
                    continue
                    
                cost_per_share = p_open * (1.0 + broker_fee + slippage)
                total_needed = qty_buy * cost_per_share
                
                if total_needed <= cash_avail:
                    cash_avail -= total_needed
                    cum_trade_fees += qty_buy * p_open * (broker_fee + slippage)
                else:
                    # Shortfall: requires Advance Cash from pending sales proceeds
                    shortfall = total_needed - cash_avail
                    cap_t1 = pending_cash_t1 / (1.0 + advance_fee_rate * 1.0)
                    cap_t0 = pending_cash_t0 / (1.0 + advance_fee_rate * 2.0)
                    total_adv_capacity = cap_t1 + cap_t0
                    
                    if total_adv_capacity >= shortfall:
                        # Full order can be funded via advance cash
                        cum_trade_fees += qty_buy * p_open * (broker_fee + slippage)
                        cash_avail = 0.0
                        
                        # Consume pending_cash_t1 first (settles in 1 day)
                        take_t1 = min(shortfall, cap_t1)
                        adv_cost_t1 = take_t1 * advance_fee_rate * 1.0
                        pending_cash_t1 -= (take_t1 + adv_cost_t1)
                        cum_adv_fees += adv_cost_t1
                        
                        rem = shortfall - take_t1
                        if rem > 0.0:
                            take_t0 = rem
                            adv_cost_t0 = take_t0 * advance_fee_rate * 2.0
                            pending_cash_t0 -= (take_t0 + adv_cost_t0)
                            cum_adv_fees += adv_cost_t0
                    else:
                        # Budget constrained: scale down buy order to exact purchasing power
                        max_budget = cash_avail + total_adv_capacity
                        scaled_qty = np.floor(max_budget / (cost_per_share * 100.0)) * 100.0
                        if scaled_qty <= 0.0:
                            continue
                            
                        qty_buy = scaled_qty
                        actual_needed = qty_buy * cost_per_share
                        cum_trade_fees += qty_buy * p_open * (broker_fee + slippage)
                        
                        if actual_needed <= cash_avail:
                            cash_avail -= actual_needed
                        else:
                            shortfall = actual_needed - cash_avail
                            cash_avail = 0.0
                            
                            take_t1 = min(shortfall, cap_t1)
                            adv_cost_t1 = take_t1 * advance_fee_rate * 1.0
                            pending_cash_t1 -= (take_t1 + adv_cost_t1)
                            cum_adv_fees += adv_cost_t1
                            
                            rem = shortfall - take_t1
                            if rem > 0.0:
                                take_t0 = rem
                                adv_cost_t0 = take_t0 * advance_fee_rate * 2.0
                                pending_cash_t0 -= (take_t0 + adv_cost_t0)
                                cum_adv_fees += adv_cost_t0
                                
                shares_held[i] += qty_buy
                buy_t0[i] += qty_buy  # Locked until day t+2
                
        # ----------------------------------------------------------------------
        # E. END-OF-DAY VALUATION AT CLOSE
        # ----------------------------------------------------------------------
        holdings_val_close = 0.0
        for i in range(N):
            holdings_val_close += shares_held[i] * raw_close[t, i]
            
        equity[t] = cash_avail + pending_cash_t1 + pending_cash_t0 + holdings_val_close
        cash_history[t] = cash_avail
        realized_fees[t] = cum_trade_fees
        advance_fees[t] = cum_adv_fees
        
    return equity, cash_history, realized_fees, advance_fees


# ==============================================================================
# 3. NAIVE FRICTIONLESS SIMULATOR (US/CRYPTO T+0 BASELINE)
# ==============================================================================

def simulate_naive(
    raw_close: np.ndarray,
    target_weights: np.ndarray,
    initial_cash: float = 1.0e9
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
    
    # Daily asset returns
    returns = np.zeros((T, N), dtype=np.float64)
    returns[1:] = (raw_close[1:] - raw_close[:-1]) / np.maximum(raw_close[:-1], 1e-6)
    
    for t in range(1, T):
        port_ret = np.sum(target_weights[t - 1] * returns[t])
        equity[t] = equity[t - 1] * (1.0 + port_ret)
        
    return equity


# ==============================================================================
# 4. VIETNAM REALISTIC SIMULATOR WRAPPER
# ==============================================================================

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
    max_adv_participation: float = 0.05
) -> Dict[str, Any]:
    """
    Full simulation wrapper executing the Numba-compiled realistic Vietnam microstructure engine.
    """
    raw_open = np.asarray(raw_open, dtype=np.float64)
    raw_close = np.asarray(raw_close, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    adv20_shares = np.asarray(adv20_shares, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    
    eq, cash, fees, adv_fees = _simulate_vn_microstructure_core(
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
        max_adv_participation=max_adv_participation
    )
    
    return {
        "equity": eq,
        "cash_history": cash,
        "trading_fees": fees,
        "advance_interest_fees": adv_fees,
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


# ==============================================================================
# 6. STRATEGY EVALUATION & RISK MANAGEMENT (KILL-SWITCH)
# ==============================================================================

def evaluate_strategy(
    equity_curve: np.ndarray,
    daily_weights: np.ndarray,
    naive_equity_curve: Optional[np.ndarray] = None,
    benchmark_equity_curve: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.04,
    n_trials: int = 26
) -> Dict[str, Any]:
    """
    Computes institutional quant metrics based on 252 trading sessions/year:
    - CAGR, Annualized Volatility
    - Sharpe Ratio (post fee & tax), Sortino Ratio
    - Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) for multiple testing
    - Maximum Drawdown (MaxDD)
    - Daily Turnover (%/day)
    - Implementation Shortfall decomposition
    - Market Beta & Alpha relative to benchmark
    - Automated Kill-Switch monitor (Triggers if MaxDD > 25% or Shortfall > 30%)
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    T = len(eq)
    
    if T < 2:
        return {}
        
    daily_returns = (eq[1:] - eq[:-1]) / np.maximum(eq[:-1], 1e-6)
    
    initial_eq = float(eq[0])
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
        annualization_factor=252.0
    )
    
    # Maximum Drawdown
    cummax = np.maximum.accumulate(eq)
    drawdowns = (cummax - eq) / cummax
    max_dd = float(np.max(drawdowns))
    
    # Daily Turnover
    w = np.asarray(daily_weights, dtype=np.float64)
    turnovers = 0.5 * np.sum(np.abs(w[1:] - w[:-1]), axis=1)
    daily_turnover = float(np.mean(turnovers))
    
    # Implementation Shortfall
    if naive_equity_curve is not None:
        naive_eq = np.asarray(naive_equity_curve, dtype=np.float64)
        naive_cagr = (naive_eq[-1] / naive_eq[0]) ** (1.0 / years) - 1.0
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
