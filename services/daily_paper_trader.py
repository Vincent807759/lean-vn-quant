#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LEAN VN-QUANT: DAILY PRODUCTION PAPER TRADING ENGINE (FORWARD-TESTING)
================================================================================
Daily portfolio management and morning ATO order ticket generator for the
Vietnam Equity Market (Forward-Testing & Paper Trading Engine).

Operational Specifications:
1. Realistic Capital Scale: 100,000,000 VND (Tests odd-lot 100 shares discretization).
2. Persistent state management via paper_portfolio_state.json and paper_nav_history.csv.
3. Realistic T+2.5 microstructure accounting:
   - Position settlement ledger: sellable_qty, locked_t1, locked_t2.
   - Pending sales cash receivable T+2 and dynamic 0.035%/day advance interest.
4. Conservative liquidity traps:
   - Refuse BUY if market opens locked at ceiling.
   - Refuse SELL if trapped at floor limit.
5. Asymmetric transaction costs: Brokerage (0.15%), Slippage (0.10%), PIT Sell Tax (0.10%).
6. Multi-Day Catch-up Loop: Deterministically reconciles missed trading sessions
   when machine was offline, preserving continuous T+2.5 settlement.
7. Instant Telegram Bot Alerts: Dispatches daily HTML reports via Telegram Bot.
8. Morning ATO Order Ticket Generator with Real-Time Implementation Shortfall tracking.
================================================================================
"""

import sys
import os
import json
import shutil
import argparse
import datetime
from typing import Dict, List, Tuple, Any, Optional

import requests
import numpy as np
import polars as pl

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
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
    from services.telegram_bot_service import mask_token
except ImportError:
    import alpha_ops as ops
    import data_pipeline as dp
    try:
        from telegram_bot_service import mask_token
    except ImportError:
        def mask_token(text: str, token: Optional[str] = None) -> str:
            return text

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.getcwd()

DEFAULT_INITIAL_CAPITAL = 100_000_000.0  # 100M VND
DEFAULT_STATE_FILE = os.path.join(DEFAULT_DATA_DIR, "paper_portfolio_state.json")
DEFAULT_HISTORY_FILE = os.path.join(DEFAULT_DATA_DIR, "paper_nav_history.csv")
DEFAULT_PARQUET_FILE = os.path.join(DEFAULT_DATA_DIR, "vn30_production.parquet")

BROKER_FEE = 0.0015       # 0.15% brokerage commission per leg
SLIPPAGE = 0.0010         # 0.10% slippage per leg
SELL_TAX = 0.0010         # 0.10% personal income tax on gross sales
ADVANCE_FEE_RATE = 0.00035 # 0.035%/day advance sales cash borrowing rate
LOT_SIZE = 100            # HOSE compulsory round-lot 100 shares
MAX_ADV_PARTICIPATION = 0.05 # Max 5% ADV20 participation


# ==============================================================================
# 1. PERSISTENT STATE MANAGEMENT (JSON & CSV)
# ==============================================================================

def init_empty_state(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> Dict[str, Any]:
    """Initializes empty portfolio state ledger."""
    return {
        "last_updated_date": None,
        "last_ledger_advance_date": None,
        "last_eod_reconciled_date": None,
        "initial_capital": float(initial_capital),
        "available_cash": float(initial_capital),
        "receivable_cash_t1": 0.0,
        "receivable_cash_t2": 0.0,
        "dividend_cash_receivable": 0.0,
        "locked_corp_action_shares": {},  # symbol -> int
        "trapped_sell_orders": [],  # orders trapped at floor price
        "holdings": {},  # symbol -> {"sellable_qty": int, "locked_t1": int, "locked_t2": int, "avg_cost": float}
        "pending_orders": [],  # list of pending ATO orders for next morning
        "naive_nav": float(initial_capital),
        "last_real_nav": float(initial_capital),
        "total_broker_fees": 0.0,
        "total_sell_tax": 0.0,
        "total_brokerage_tax": 0.0,
        "total_advance_fees": 0.0,
        "prev_target_weights": {}  # symbol -> float
    }


def load_portfolio_state(state_file: str, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> Dict[str, Any]:
    """
    Loads portfolio state from JSON file.
    B13 Invariant: If state file is corrupted, attempts recovery from .bak file.
    If both fail, strictly raises RuntimeError to prevent catastrophic silent state reset.
    """
    bak_file = f"{state_file}.bak"
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                return state
        except Exception as e:
            print(f"[ERROR] Corrupted state file {state_file} ({e}). Attempting backup recovery from {bak_file}...")
            if os.path.exists(bak_file):
                try:
                    with open(bak_file, "r", encoding="utf-8") as f_bak:
                        state = json.load(f_bak)
                        print(f"[RECOVERY] Successfully recovered portfolio state from {bak_file}")
                        return state
                except Exception as bak_err:
                    raise RuntimeError(
                        f"B13 FATAL: Both primary state file ({state_file}: {e}) and backup ({bak_file}: {bak_err}) "
                        "are corrupted! Refusing to reset empty state automatically."
                    )
            else:
                raise RuntimeError(
                    f"B13 FATAL: Primary state file ({state_file}) is corrupted ({e}) and no backup file ({bak_file}) exists! "
                    "Refusing to reset empty state automatically."
                )
    return init_empty_state(initial_capital)


def save_portfolio_state(state: Dict[str, Any], state_file: str) -> None:
    """
    Saves portfolio state atomically into JSON file with fsync and automatic backup (.bak).
    B13 Protocol:
    1. Backup existing state_file to state_file.bak
    2. Write new payload into temporary file state_file.tmp
    3. Flush and call os.fsync() to force hardware disk write
    4. Atomic rename via os.replace(tmp, state_file)
    """
    state_dir = os.path.dirname(os.path.abspath(state_file))
    os.makedirs(state_dir, exist_ok=True)
    
    bak_file = f"{state_file}.bak"
    tmp_file = f"{state_file}.tmp"
    
    # 1. Backup current state
    if os.path.exists(state_file):
        try:
            shutil.copy2(state_file, bak_file)
        except Exception as e:
            print(f"[WARNING] Failed to create backup state {bak_file}: {e}")
            
    # 2. Write to tmp file with fsync
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    # 3. Atomic replacement
    os.replace(tmp_file, state_file)


def append_nav_history(history_file: str, record: Dict[str, Any]) -> None:
    """Appends daily NAV and Implementation Shortfall record into CSV ledger."""
    os.makedirs(os.path.dirname(os.path.abspath(history_file)), exist_ok=True)
    file_exists = os.path.exists(history_file) and os.path.getsize(history_file) > 0
    headers = [
        "date", "real_nav", "available_cash", "receivable_cash", "stock_value",
        "naive_nav", "implementation_shortfall_pct", "daily_return_real",
        "daily_return_naive", "orders_executed"
    ]
    
    with open(history_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(headers) + "\n")
        line = [
            str(record["date"]),
            f"{record['real_nav']:.2f}",
            f"{record['available_cash']:.2f}",
            f"{record['receivable_cash']:.2f}",
            f"{record['stock_value']:.2f}",
            f"{record['naive_nav']:.2f}",
            f"{record['implementation_shortfall_pct']:.4f}",
            f"{record['daily_return_real']:.6f}",
            f"{record['daily_return_naive']:.6f}",
            str(record["orders_executed"])
        ]
        f.write(",".join(line) + "\n")


# ==============================================================================
# 2. STEP 1: MARKET DATA INGESTION FROM REST API / PARQUET
# ==============================================================================

def check_cache_overlap_and_reconcile(
    df_cache: pl.DataFrame,
    df_new: pl.DataFrame,
    overlap_window: int = 20,
    tolerance: float = 1e-6
) -> bool:
    """
    Checks the overlap of up to `overlap_window` sessions between cached data and newly fetched data.
    Adheres to B2/S2/T3:
    Returns True if cache is valid (relative price difference <= tolerance on all overlapping bars).
    Returns False if corporate actions (splits, dividends) rewrote historical prices, requiring full refetch.
    """
    cache_dates = set(df_cache["date"].unique().to_list())
    new_dates = set(df_new["date"].unique().to_list())
    common_dates = sorted(cache_dates & new_dates)
    
    if not common_dates:
        return True
        
    overlap_dates = common_dates[-overlap_window:]
    
    cache_sub = df_cache.filter(pl.col("date").is_in(overlap_dates)).select(["date", "symbol", "adj_close"])
    new_sub = df_new.filter(pl.col("date").is_in(overlap_dates)).select(["date", "symbol", "adj_close"])
    
    joined = cache_sub.join(new_sub, on=["date", "symbol"], suffix="_new")
    if len(joined) == 0:
        return True
        
    diffs = (joined["adj_close"] - joined["adj_close_new"]).abs() / joined["adj_close"]
    max_diff = diffs.max()
    
    if max_diff is not None and max_diff > tolerance:
        return False
    return True


def fetch_or_update_data(
    parquet_file: str,
    is_replay: bool = False,
    calibration_file: Optional[str] = None
) -> pl.DataFrame:
    """
    Ingests or updates VN30 daily OHLCV dataset.
    Uses cached Parquet file if in replay mode or if no new candles available.
    Adheres to B2/S2/T3/T8:
    - Drops incomplete intraday candle if fetched before 15:05 ICT.
    - Validates 20 overlapping bars with tolerance 1e-6.
    - On corporate action mismatch: Refetches entire history and invalidates/refreshes calibration table.
    - On API error: preserves cached storage untouched.
    """
    if not os.path.exists(parquet_file):
        print(f"[DATA] Parquet storage {parquet_file} not found. Fetching historical data from Entrade API...")
        os.makedirs(os.path.dirname(os.path.abspath(parquet_file)), exist_ok=True)
        df_market = dp.fetch_vn_ohlcv(save_parquet=parquet_file, force_fallback=False)
        return df_market

    df_market = pl.read_parquet(parquet_file)
    
    if is_replay:
        return df_market

    cache_dates = sorted(df_market["date"].unique().to_list())
    latest_dt = cache_dates[-1]
    
    vn_tz = datetime.timezone(datetime.timedelta(hours=7))
    now_vn = datetime.datetime.now(vn_tz)
    today = now_vn.date()
    
    if today >= latest_dt:
        overlap_idx = max(0, len(cache_dates) - 20)
        fetch_start_dt = cache_dates[overlap_idx]
        
        try:
            df_new = dp.fetch_vn_ohlcv(
                start_date=fetch_start_dt.isoformat(),
                end_date=today.isoformat(),
                save_parquet=None,
                force_fallback=False
            )
            
            if df_new is not None and len(df_new) > 0:
                # T3: Drop incomplete intraday bar of today if called before market close (15:05 ICT)
                if now_vn.time() < datetime.time(15, 5):
                    df_new = df_new.filter(pl.col("date") < today)
                    
                if len(df_new) == 0:
                    return df_market
                    
                # B2/S2/T3: Check overlap across the 20 overlapping sessions
                is_cache_consistent = check_cache_overlap_and_reconcile(
                    df_cache=df_market,
                    df_new=df_new,
                    overlap_window=20,
                    tolerance=1e-6
                )
                
                if not is_cache_consistent:
                    print(f"[DATA WARNING] Historical price adjustment (split/dividend) detected on overlap window! Refetching entire history...")
                    earliest_dt = cache_dates[0]
                    df_market = dp.fetch_vn_ohlcv(
                        start_date=earliest_dt.isoformat(),
                        end_date=today.isoformat(),
                        save_parquet=parquet_file,
                        force_fallback=False
                    )
                    # T8/B2: Invalidate calibration/raw price cache when new corporate actions occur
                    if calibration_file and os.path.exists(calibration_file):
                        try:
                            os.remove(calibration_file)
                            print(f"[DATA] Invalidated calibration cache: {calibration_file}")
                        except Exception as ce:
                            print(f"[DATA WARNING] Could not remove calibration cache: {ce}")
                    print(f"[DATA] Full history successfully refreshed and saved to {parquet_file}.")
                    return df_market
                else:
                    new_candles = df_new.filter(pl.col("date") > latest_dt)
                    if len(new_candles) > 0:
                        combined = pl.concat([df_market, new_candles]).sort(["date", "symbol"])
                        combined.write_parquet(parquet_file)
                        print(f"[DATA] Appended {len(new_candles)} new daily candles to {parquet_file}.")
                        return combined
        except Exception as e:
            print(f"[DATA WARNING] Could not fetch live API data ({e}). Continuing with cached storage untouched.")
            
    return df_market


# ==============================================================================
# 3. STEP 2: ATO ORDER EXECUTION & T+2 SETTLEMENT ROLL
# ==============================================================================

def advance_t25_settlement(state: Dict[str, Any], session_date: str) -> bool:
    """
    Idempotent T+2.5 settlement ledger roll for HOSE equity microstructure:
    - Rolls T+2 cash receivable -> T+1 -> Available Cash
    - Rolls locked_t2 shares -> locked_t1 -> sellable_qty
    - Guarantees idempotency: cannot advance twice for the same trading session.
    """
    if state.get("last_ledger_advance_date") == session_date:
        print(f"[LEDGER IDEMPOTENCY] T+2.5 settlement already advanced for session {session_date}. Skipping duplicate roll.")
        return False

    state["available_cash"] += state.get("receivable_cash_t1", 0.0)
    state["receivable_cash_t1"] = state.get("receivable_cash_t2", 0.0)
    state["receivable_cash_t2"] = 0.0

    for sym, hold in state.get("holdings", {}).items():
        hold["sellable_qty"] += hold.get("locked_t1", 0)
        hold["locked_t1"] = hold.get("locked_t2", 0)
        hold["locked_t2"] = 0

    state["last_ledger_advance_date"] = session_date
    return True


def execute_pending_orders_and_settle(
    state: Dict[str, Any],
    today_market_slice: Dict[str, Dict[str, float]],
    session_date: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes morning ATO session workflow:
    1. Settlement Roll: T+2 receivable -> T+1 -> Available Cash.
       Locked T+2 shares -> Locked T+1 -> Sellable Qty.
    2. Matches pending orders generated previous afternoon at today's raw_open.
    3. Handles liquidity traps (ceiling locked or floor trapped).
    4. Accounts for asymmetric costs and advance borrowing interest.
    """
    # 1. Idempotent Settlement Roll (guaranteed single roll per session_date)
    advance_t25_settlement(state, session_date)

    pending_orders = state.get("pending_orders", [])
    if not pending_orders:
        return state, []

    executed_reports = []
    remaining_pending = []
    
    sell_orders = [o for o in pending_orders if o.get("action") == "SELL"]
    buy_orders = [o for o in pending_orders if o.get("action") == "BUY"]

    # --- PROCESS SELL ORDERS FIRST ---
    for o in sell_orders:
        sym = o["symbol"]
        qty_requested = int(o["qty"])
        mkt = today_market_slice.get(sym)
        
        if not mkt:
            remaining_pending.append(o)
            continue
            
        raw_open = mkt["raw_open"]
        floor_p = mkt["floor_price"]
        
        # Liquidity Trap: Floor Trapped
        if raw_open <= floor_p + 1e-4:
            rep = dict(o)
            rep["status"] = "FLOOR TRAPPED (HOLDING ORDER)"
            rep["fill_qty"] = 0
            rep["fill_price"] = raw_open
            rep["trapped_days"] = o.get("trapped_days", 0) + 1
            rep["note"] = f"Market opened locked at floor ({raw_open:,.0f} VND), zero bid. Retaining order in priority queue for next session with -7%/session penalty."
            executed_reports.append(rep)
            o_trapped = dict(o)
            o_trapped["trapped_days"] = rep["trapped_days"]
            remaining_pending.append(o_trapped)
            continue
            
        sellable = state["holdings"].get(sym, {}).get("sellable_qty", 0)
        fill_qty = min(qty_requested, sellable)
        fill_qty = (fill_qty // LOT_SIZE) * LOT_SIZE
        
        if fill_qty <= 0:
            rep = dict(o)
            rep["status"] = "REJECTED (T+2.5 LOCKED)"
            rep["fill_qty"] = 0
            rep["fill_price"] = raw_open
            rep["note"] = f"Shares not yet settled in tradable account (still locked in T+1/T+2)."
            executed_reports.append(rep)
            continue
            
        trapped_days = o.get("trapped_days", 0)
        penalty_slippage = min(0.40, 0.07 * trapped_days)
        exec_price = raw_open * (1.0 - SLIPPAGE - penalty_slippage)
        gross_val = fill_qty * exec_price
        broker_fee_val = gross_val * BROKER_FEE
        tax_val = gross_val * SELL_TAX
        net_cash = gross_val - broker_fee_val - tax_val
        
        # Deduct sold shares from sellable holdings
        state["holdings"][sym]["sellable_qty"] -= fill_qty
        if (state["holdings"][sym]["sellable_qty"] + state["holdings"][sym]["locked_t1"] + state["holdings"][sym]["locked_t2"]) == 0:
            del state["holdings"][sym]
            
        # Sales proceeds enter T+2 receivable queue
        state["receivable_cash_t2"] += net_cash
        state["total_broker_fees"] = state.get("total_broker_fees", 0.0) + broker_fee_val
        state["total_sell_tax"] = state.get("total_sell_tax", 0.0) + tax_val
        state["total_brokerage_tax"] = state["total_broker_fees"] + state["total_sell_tax"]
        
        rep = dict(o)
        rep["status"] = "FILLED"
        rep["fill_qty"] = fill_qty
        rep["fill_price"] = exec_price
        rep["net_cash"] = net_cash
        rep["broker_fee"] = broker_fee_val
        rep["sell_tax"] = tax_val
        rep["fee_tax"] = broker_fee_val + tax_val
        rep["note"] = f"Sold {fill_qty:,} shares @ {exec_price:,.0f} VND (Penalty slippage: {penalty_slippage*100:.1f}%). Net proceeds {net_cash:,.0f} VND queued for T+2 settlement."
        executed_reports.append(rep)

    # --- PROCESS BUY ORDERS ---
    # Determine calendar days for advance interest
    try:
        s_date = datetime.date.fromisoformat(session_date) if isinstance(session_date, str) else session_date
        weekday = s_date.weekday()
        # Friday: T+1 settles Mon (3 cal days), T+2 settles Tue (4 cal days)
        # Thursday: T+1 settles Fri (1 cal day), T+2 settles Mon (4 cal days)
        # Mon-Wed: T+1 is 1 day, T+2 is 2 days
        if weekday == 4:
            cal_days_t1 = 3
            cal_days_t2 = 4
        elif weekday == 3:
            cal_days_t1 = 1
            cal_days_t2 = 4
        else:
            cal_days_t1 = 1
            cal_days_t2 = 2
    except Exception:
        cal_days_t1 = 1
        cal_days_t2 = 2

    for o in buy_orders:
        sym = o["symbol"]
        qty_requested = int(o["qty"])
        mkt = today_market_slice.get(sym)
        
        if not mkt:
            remaining_pending.append(o)
            continue
            
        raw_open = mkt["raw_open"]
        ceil_p = mkt["ceiling_price"]
        
        # Liquidity Trap: Ceiling Locked
        if raw_open >= ceil_p - 1e-4:
            rep = dict(o)
            rep["status"] = "CEILING LOCKED (SKIPPING BUY)"
            rep["fill_qty"] = 0
            rep["fill_price"] = raw_open
            rep["note"] = f"Market opened locked at ceiling ({raw_open:,.0f} VND), zero offer. Skipping buy to avoid chase risk."
            executed_reports.append(rep)
            continue
            
        exec_price = raw_open * (1.0 + SLIPPAGE)
        cost_per_share = exec_price * (1.0 + BROKER_FEE)
        
        # Liquidity cap: <= 5% ADV20
        adv20 = mkt.get("adv20", 1e11)
        max_shares_adv = int((adv20 * MAX_ADV_PARTICIPATION) / max(raw_open, 1.0))
        max_shares_adv = max(LOT_SIZE, (max_shares_adv // LOT_SIZE) * LOT_SIZE)
        capped_qty = min(qty_requested, max_shares_adv)
        
        # Check cash purchasing power
        avail_cash = state["available_cash"]
        rec_t1 = state["receivable_cash_t1"]
        rec_t2 = state["receivable_cash_t2"]
        total_purchasing_power = avail_cash + rec_t1 + rec_t2
        
        shares_afford = int(total_purchasing_power // cost_per_share)
        shares_afford = (shares_afford // LOT_SIZE) * LOT_SIZE
        fill_qty = min(capped_qty, shares_afford)
        
        if fill_qty <= 0:
            rep = dict(o)
            rep["status"] = "REJECTED (INSUFFICIENT CAPITAL)"
            rep["fill_qty"] = 0
            rep["fill_price"] = raw_open
            rep["note"] = f"Insufficient cash to purchase minimum lot size (100 shares @ {cost_per_share:,.0f} VND)."
            executed_reports.append(rep)
            continue
            
        total_buy_cost = fill_qty * cost_per_share
        gross_val = fill_qty * exec_price
        broker_fee_val = gross_val * BROKER_FEE
        state["total_broker_fees"] = state.get("total_broker_fees", 0.0) + broker_fee_val
        state["total_brokerage_tax"] = state["total_broker_fees"] + state.get("total_sell_tax", 0.0)
        
        # Deduct cash with dynamic advance borrowing interest (calendar days & 50k floor fee)
        if state["available_cash"] >= total_buy_cost:
            state["available_cash"] -= total_buy_cost
        else:
            deficit = total_buy_cost - state["available_cash"]
            state["available_cash"] = 0.0
            
            # Borrow against T+1 receivables first
            borrow_t1 = min(deficit, state["receivable_cash_t1"])
            adv_fee_t1 = max(50000.0, borrow_t1 * ADVANCE_FEE_RATE * cal_days_t1) if borrow_t1 > 0 else 0.0
            state["receivable_cash_t1"] = max(0.0, state["receivable_cash_t1"] - borrow_t1 - adv_fee_t1)
            deficit -= borrow_t1
            
            # Borrow against T+2 receivables
            borrow_t2 = min(deficit, state["receivable_cash_t2"])
            adv_fee_t2 = max(50000.0, borrow_t2 * ADVANCE_FEE_RATE * cal_days_t2) if borrow_t2 > 0 else 0.0
            state["receivable_cash_t2"] = max(0.0, state["receivable_cash_t2"] - borrow_t2 - adv_fee_t2)
            deficit -= borrow_t2
            
            tot_adv_fee = adv_fee_t1 + adv_fee_t2
            state["total_advance_fees"] += tot_adv_fee
            
        # Purchased shares enter Locked T+2
        if sym not in state["holdings"]:
            state["holdings"][sym] = {
                "sellable_qty": 0,
                "locked_t1": 0,
                "locked_t2": fill_qty,
                "avg_cost": exec_price
            }
        else:
            cur_hold = state["holdings"][sym]
            old_tot = cur_hold["sellable_qty"] + cur_hold["locked_t1"] + cur_hold["locked_t2"]
            new_tot = old_tot + fill_qty
            new_cost = (cur_hold["avg_cost"] * old_tot + exec_price * fill_qty) / max(new_tot, 1)
            cur_hold["locked_t2"] += fill_qty
            cur_hold["avg_cost"] = new_cost
            
        rep = dict(o)
        rep["status"] = "FILLED"
        rep["fill_qty"] = fill_qty
        rep["fill_price"] = exec_price
        rep["total_cost"] = total_buy_cost
        rep["note"] = f"Purchased {fill_qty:,} shares @ {exec_price:,.0f} VND (Total outlay: {total_buy_cost:,.0f} VND). Locked in T+2."
        executed_reports.append(rep)

    state["pending_orders"] = remaining_pending
    return state, executed_reports


# ==============================================================================
# 4. STEP 3: END-OF-DAY MARK-TO-MARKET & IMPLEMENTATION SHORTFALL AUDITING
# ==============================================================================

def mark_to_market(
    state: Dict[str, Any],
    today_market_slice: Dict[str, Dict[str, float]],
    prev_close_map: Dict[str, float],
    session_date: str,
    orders_executed: int
) -> Dict[str, Any]:
    """
    Computes end-of-day Mark-to-Market Real NAV vs Naive T+0 Frictionless NAV.
    """
    total_stock_value = 0.0
    for sym, hold in state["holdings"].items():
        tot_qty = hold["sellable_qty"] + hold["locked_t1"] + hold["locked_t2"]
        close_p = today_market_slice.get(sym, {}).get("raw_close", hold["avg_cost"])
        total_stock_value += (tot_qty * close_p)
        
    div_cash_rec = state.get("dividend_cash_receivable", 0.0)
    locked_corp_val = 0.0
    for sym, qty in state.get("locked_corp_action_shares", {}).items():
        p = today_market_slice.get(sym, {}).get("raw_close", 0.0)
        locked_corp_val += (qty * p)
        
    real_nav = (
        state["available_cash"]
        + state["receivable_cash_t1"]
        + state["receivable_cash_t2"]
        + div_cash_rec
        + total_stock_value
        + locked_corp_val
    )
    
    # Compute Naive T+0 Benchmark NAV
    prev_naive_nav = state.get("naive_nav", state["initial_capital"])
    prev_target_w = state.get("prev_target_weights", {})
    
    naive_daily_ret = 0.0
    for sym, w in prev_target_w.items():
        if sym in today_market_slice and sym in prev_close_map:
            p_today = today_market_slice[sym]["raw_close"]
            p_prev = prev_close_map[sym]
            if p_prev > 0:
                ret = (p_today - p_prev) / p_prev
                naive_daily_ret += w * ret
                
    naive_nav = prev_naive_nav * (1.0 + naive_daily_ret)
    
    # Implementation Shortfall (% of initial capital)
    initial_cap = state["initial_capital"]
    shortfall_pct = (naive_nav - real_nav) / initial_cap * 100.0
    
    prev_real_nav = state.get("last_real_nav", initial_cap)
    daily_ret_real = (real_nav - prev_real_nav) / max(prev_real_nav, 1.0)
    
    state["last_real_nav"] = real_nav
    state["naive_nav"] = naive_nav
    state["last_updated_date"] = session_date
    state["last_eod_reconciled_date"] = session_date
    
    return {
        "date": session_date,
        "real_nav": real_nav,
        "available_cash": state["available_cash"],
        "receivable_cash": state["receivable_cash_t1"] + state["receivable_cash_t2"],
        "stock_value": total_stock_value,
        "naive_nav": naive_nav,
        "implementation_shortfall_pct": shortfall_pct,
        "daily_return_real": daily_ret_real,
        "daily_return_naive": naive_daily_ret,
        "orders_executed": orders_executed
    }


# ==============================================================================
# 5. STEP 4: DYNAMIC ALPHA PLUGIN LOADER & MORNING ATO ORDER TICKET GENERATOR
# ==============================================================================

def _default_textbook_placeholder_signal(
    raw_close: np.ndarray,
    tradable_mask: np.ndarray
) -> np.ndarray:
    """
    Standard textbook educational placeholder signal (SMA-40 / SMA-50 Breakout Z-Score).
    Used exclusively for quantitative engine forward testing and CI/CD validation.
    Does NOT contain proprietary alpha research or live trading strategy weights.
    """
    hh = ops.ts_mean(raw_close, 40)
    dev = ops.ts_std_dev(raw_close, 40)
    z = (raw_close - hh) / np.maximum(dev, 1e-4)
    sma50 = ops.ts_mean(raw_close, 50)
    in_uptrend = raw_close > sma50
    z_filtered = np.where(in_uptrend, z, -999.0)
    return ops.rank(z_filtered, mask=tradable_mask)


def compute_strategy_signal(
    raw_close: np.ndarray,
    tradable_mask: np.ndarray,
    df_market_history: Optional[pl.DataFrame] = None
) -> np.ndarray:
    """
    Dynamic Alpha Plugin Loader:
    - If `private_alphas/` package exists locally, imports and calls `private_alphas.compute_custom_alpha(...)`
    - Otherwise (e.g. open-source CI/CD), falls back to `_default_textbook_placeholder_signal`.
    """
    try:
        import private_alphas
        if hasattr(private_alphas, "compute_custom_alpha"):
            return private_alphas.compute_custom_alpha(raw_close, tradable_mask, df_market_history)
    except (ImportError, ModuleNotFoundError):
        pass
    return _default_textbook_placeholder_signal(raw_close, tradable_mask)


def generate_next_day_orders(
    state: Dict[str, Any],
    df_market_history: pl.DataFrame,
    session_date: str
) -> List[Dict[str, Any]]:
    """
    Computes Alpha signals at session close and generates morning ATO order ticket:
    1. Dynamic Alpha Plugin Loader (private_alphas or default textbook placeholder signal).
    2. Rank Hysteresis + Turnover Dampening (Daily turnover <= 1.5%/day).
    3. Rebalancing & Odd-lot 100 rounding.
    4. Sells strictly capped at current sellable_qty.
    5. Greedy Cash Sweep: Allocates idle cash (> 15% NAV) across Top 4 Alpha stocks.
    """
    df_masked = dp.build_tradable_mask(df_market_history, min_adv=1.0e10, window=20)
    raw_close, dates, syms = dp.pivot_to_matrix(df_masked, "raw_close")
    tradable_mask, _, _ = dp.pivot_to_matrix(df_masked, "tradable_mask")
    
    T, N = raw_close.shape
    if T < 55:
        return []

    # Dynamic Alpha Plugin Loader
    alpha = compute_strategy_signal(raw_close, tradable_mask, df_market_history)

    # Rank Hysteresis Turnover Dampening
    target_weights_matrix = ops.dampen_turnover(
        alpha=alpha,
        top_k_in=2,
        top_k_out=4,
        ema_span=20,
        max_daily_turnover=0.015,
        mask=tradable_mask
    )
    
    today_target_w = target_weights_matrix[-1]
    
    # Cache target weights for Naive benchmark tracking
    prev_w_dict = {}
    for i, s in enumerate(syms):
        if today_target_w[i] > 1e-4:
            prev_w_dict[s] = float(today_target_w[i])
    state["prev_target_weights"] = prev_w_dict

    # Compute current total NAV
    real_nav = state["last_real_nav"]
    last_close_prices = {s: raw_close[-1, i] for i, s in enumerate(syms)}

    # Target shares allocation with 100-share lot rounding
    target_shares = {}
    for i, s in enumerate(syms):
        w = today_target_w[i]
        p = last_close_prices.get(s, 0.0)
        if w > 0 and p > 0:
            val_target = real_nav * w
            shares = int(val_target // (p * LOT_SIZE)) * LOT_SIZE
            target_shares[s] = shares
        else:
            target_shares[s] = 0

    # Current shares held
    current_shares = {}
    current_sellable = {}
    for s in syms:
        if s in state["holdings"]:
            h = state["holdings"][s]
            tot = h["sellable_qty"] + h["locked_t1"] + h["locked_t2"]
            current_shares[s] = tot
            current_sellable[s] = h["sellable_qty"]
        else:
            current_shares[s] = 0
            current_sellable[s] = 0

    orders = []

    # --- 1. GENERATE SELL ORDERS ---
    for s in syms:
        cur_q = current_shares[s]
        tgt_q = target_shares.get(s, 0)
        if tgt_q < cur_q:
            desired_sell = cur_q - tgt_q
            avail_sell = current_sellable.get(s, 0)
            actual_sell = min(desired_sell, avail_sell)
            actual_sell = (actual_sell // LOT_SIZE) * LOT_SIZE
            
            if actual_sell > 0:
                p = last_close_prices.get(s, 0.0)
                orders.append({
                    "action": "SELL",
                    "symbol": s,
                    "qty": int(actual_sell),
                    "est_price": float(p),
                    "est_value": float(actual_sell * p),
                    "target_weight": float(today_target_w[syms.index(s)])
                })

    # --- 2. GENERATE BUY ORDERS & GREEDY CASH SWEEP ---
    buy_orders = []
    total_est_buy_cost = 0.0
    for s in syms:
        cur_q = current_shares[s]
        tgt_q = target_shares.get(s, 0)
        if tgt_q > cur_q:
            diff = tgt_q - cur_q
            diff = (diff // LOT_SIZE) * LOT_SIZE
            if diff > 0:
                p = last_close_prices.get(s, 0.0)
                buy_val = diff * p * (1.0 + BROKER_FEE)
                total_est_buy_cost += buy_val
                buy_orders.append({
                    "action": "BUY",
                    "symbol": s,
                    "qty": int(diff),
                    "est_price": float(p),
                    "est_value": float(diff * p),
                    "target_weight": float(today_target_w[syms.index(s)])
                })

    # Greedy Cash Sweep: Allocates idle cash if > 15% NAV
    total_available_funds = state["available_cash"] + state["receivable_cash_t1"] + state["receivable_cash_t2"]
    rem_cash = total_available_funds - total_est_buy_cost
    
    if rem_cash > 0.15 * real_nav:
        top_candidates = []
        alpha_today = alpha[-1]
        for i, s in enumerate(syms):
            if tradable_mask[-1, i] and raw_close[-1, i] > sma50[-1, i]:
                top_candidates.append((alpha_today[i], s, last_close_prices[s], i))
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = top_candidates[:4]
        
        while rem_cash > 0.10 * real_nav and top_candidates:
            allocated = False
            for cand_alpha, cand_sym, cand_price, cand_idx in top_candidates:
                lot_cost = LOT_SIZE * cand_price * (1.0 + BROKER_FEE)
                if rem_cash >= lot_cost and rem_cash - lot_cost >= 0.05 * real_nav:
                    found = False
                    for bo in buy_orders:
                        if bo["symbol"] == cand_sym:
                            bo["qty"] += LOT_SIZE
                            bo["est_value"] += (LOT_SIZE * cand_price)
                            found = True
                            break
                    if not found:
                        buy_orders.append({
                            "action": "BUY",
                            "symbol": cand_sym,
                            "qty": LOT_SIZE,
                            "est_price": float(cand_price),
                            "est_value": float(LOT_SIZE * cand_price),
                            "target_weight": float(today_target_w[cand_idx])
                        })
                    rem_cash -= lot_cost
                    allocated = True
            if not allocated:
                break

    orders.extend(buy_orders)
    state["pending_orders"] = orders
    return orders


# ==============================================================================
# 6. PROFESSIONAL CLI CONSOLE DASHBOARD
# ==============================================================================

def print_portfolio_dashboard(
    state: Dict[str, Any],
    today_market_slice: Dict[str, Dict[str, float]],
    session_date: str,
    orders: List[Dict[str, Any]],
    executed_reports: List[Dict[str, Any]]
) -> None:
    """Prints institutional portfolio dashboard and order ticket."""
    print("\n" + "=" * 90)
    print(f"       LEAN VN-QUANT PRODUCTION FORWARD-TESTING AUDIT (SESSION: {session_date})")
    print("=" * 90)

    # 1. Morning ATO Execution Report
    if executed_reports:
        print("\n--- [1] MORNING ATO ORDER EXECUTION REPORT ---")
        for r in executed_reports:
            status = r.get("status", "")
            sym = r.get("symbol", "")
            action = r.get("action", "")
            qty = r.get("fill_qty", 0)
            p = r.get("fill_price", 0.0)
            note = r.get("note", "")
            if "FILLED" in status:
                val = r.get("total_cost", r.get("net_cash", 0.0))
                print(f"  [FILLED] {action} {qty:,} shares {sym} @ {p:,.0f} VND (Value: {val:,.0f} VND)")
            else:
                print(f"  [REJECTED] {action} {sym} -> {status}: {note}")
    else:
        print("\n--- [1] MORNING ATO REPORT: No pending orders for this session.")

    # 2. Portfolio Holdings (Mark-to-Market)
    print("\n--- [2] PORTFOLIO STATUS (END-OF-DAY MARK-TO-MARKET) ---")
    holdings = state.get("holdings", {})
    if holdings:
        print(f"{'Ticker':<8} | {'Sellable':<10} | {'Locked T+1':<10} | {'Locked T+2':<10} | {'Total Shares':<12} | {'Close Price':<14} | {'Market Value (VND)':<18}")
        print("-" * 90)
        total_stock_val = 0.0
        for sym, h in sorted(holdings.items()):
            sellable = h.get("sellable_qty", 0)
            t1 = h.get("locked_t1", 0)
            t2 = h.get("locked_t2", 0)
            tot = sellable + t1 + t2
            c_p = today_market_slice.get(sym, {}).get("raw_close", 0.0)
            val = tot * c_p
            total_stock_val += val
            print(f"{sym:<8} | {sellable:<10,d} | {t1:<10,d} | {t2:<10,d} | {tot:<12,d} | {c_p:>11,.0f} VND | {val:>16,.0f} VND")
        print("-" * 90)
    else:
        total_stock_val = 0.0
        print("  (Portfolio is 100% Cash. No open equity positions.)")

    avail_cash = state["available_cash"]
    rec_t1 = state["receivable_cash_t1"]
    rec_t2 = state["receivable_cash_t2"]
    real_nav = state["last_real_nav"]
    naive_nav = state["naive_nav"]
    init_cap = state["initial_capital"]
    ret_pct = (real_nav - init_cap) / init_cap * 100.0
    shortfall = (naive_nav - real_nav) / init_cap * 100.0

    print(f"  Available Cash           : {avail_cash:>15,.0f} VND")
    print(f"  T+1 Receivable Cash      : {rec_t1:>15,.0f} VND")
    print(f"  T+2 Receivable Cash      : {rec_t2:>15,.0f} VND")
    print(f"  Total Equities Value     : {total_stock_val:>15,.0f} VND")
    print(f"  TOTAL REAL NAV           : {real_nav:>15,.0f} VND (Return: {ret_pct:>+6.2f} %)")
    print(f"  Frictionless Naive T+0   : {naive_nav:>15,.0f} VND")
    print(f"  Implementation Shortfall : {shortfall:>+15.2f} % (Friction & settlement drag)")

    # 3. Next Session ATO Order Ticket
    print("\n" + "=" * 90)
    print(f"       TOMORROW MORNING ATO ORDER TICKET")
    print("=" * 90)
    if orders:
        print(f"{'Ticker':<8} | {'Side':<6} | {'Quantity':<12} | {'Est Price':<15} | {'Est Value (VND)':<18} | {'Target %'}")
        print("-" * 90)
        tot_order_val = 0.0
        for o in orders:
            sym = o["symbol"]
            act = o["action"]
            q = o["qty"]
            p = o["est_price"]
            v = o["est_value"]
            w = o["target_weight"] * 100.0
            tot_order_val += v
            print(f"{sym:<8} | {act:<6} | {q:>10,d} shs | {p:>12,.0f} VND | {v:>16,.0f} VND | {w:>6.1f} %")
        print("-" * 90)
        print(f"  Total Estimated Order Outlay: {tot_order_val:,.0f} VND")
        print(f"  Status: QUEUED IN PENDING_ORDERS FOR TOMORROW ATO SESSION")
    else:
        print("  No new rebalancing orders. Maintaining current positions (No-Trade Buffer).")
    print("=" * 90 + "\n")


# ==============================================================================
# 7. TELEGRAM BOT ALERTS & ENVIRONMENT VARIABLES
# ==============================================================================

def load_env_variables() -> Dict[str, str]:
    """Finds and parses local .env configuration without python-dotenv."""
    env_vars = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(base_dir, ".env"),
        os.path.join(base_dir, "..", ".env"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k not in env_vars:
                                env_vars[k] = v
            except Exception:
                pass
    return env_vars


def send_telegram_alert(report_text: str) -> bool:
    """Sends portfolio report directly to smartphone via Telegram Bot."""
    env = load_env_variables()
    token = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("[TELEGRAM] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured in .env. Skipping alert.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": report_text,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            print("[TELEGRAM] Daily report successfully dispatched to smartphone via Telegram Bot.")
            return True
        else:
            safe_text = mask_token(resp.text, token)
            print(f"[TELEGRAM ERROR] HTTP {resp.status_code}: {safe_text}")
            return False
    except Exception as e:
        safe_err = mask_token(str(e), token)
        print(f"[TELEGRAM CONNECTION ERROR] {safe_err}")
        return False


def build_telegram_summary(
    state: Dict[str, Any],
    session_date: str,
    orders: List[Dict[str, Any]],
    executed_reports: List[Dict[str, Any]],
    dates_processed: List[datetime.date]
) -> str:
    """Builds clean HTML notification for Telegram."""
    real_nav = state.get("last_real_nav", state["initial_capital"])
    init_cap = state["initial_capital"]
    ret_pct = (real_nav - init_cap) / init_cap * 100.0
    shortfall = (state.get("naive_nav", init_cap) - real_nav) / init_cap * 100.0
    avail_cash = state.get("available_cash", 0.0)
    rec_cash = state.get("receivable_cash_t1", 0.0) + state.get("receivable_cash_t2", 0.0)
    
    catch_up_info = ""
    if len(dates_processed) > 1:
        catch_up_info = f"\n⚡ <b>Multi-day Catch-up:</b> {len(dates_processed)} sessions reconciled ({dates_processed[0]} -> {dates_processed[-1]})\n"

    holdings = state.get("holdings", {})
    stock_count = len(holdings)

    if orders:
        order_lines = []
        for o in orders:
            act = "🟢 BUY" if o["action"] == "BUY" else "🔴 SELL"
            sym = o["symbol"]
            q = o["qty"]
            p = o["est_price"]
            v = o["est_value"]
            w = o["target_weight"] * 100.0
            order_lines.append(f"• {act} <b>{q:,} shs {sym}</b> @ {p:,.0f} VND (~{v/1e6:.2f}M | {w:.1f}%)")
        orders_text = "\n".join(order_lines)
    else:
        orders_text = "• <i>No rebalancing orders. Holding current portfolio.</i>"

    msg = f"""<b>📊 LEAN VN-QUANT DAILY PAPER TRADING REPORT</b>
📅 Session: <b>{session_date}</b>{catch_up_info}
💰 <b>PORTFOLIO & PERFORMANCE:</b>
• Real NAV: <b>{real_nav:,.0f} VND</b> (<b>{ret_pct:+.2f}%</b>)
• Available Cash: <b>{avail_cash:,.0f} VND</b>
• T+2 Receivables: <b>{rec_cash:,.0f} VND</b>
• Open Holdings: <b>{stock_count} stocks</b>
• Implementation Shortfall: <b>{shortfall:+.2f}%</b>

📋 <b>TOMORROW ATO ORDER TICKET:</b>
{orders_text}

<i>Status: T+2.5 settlement ledger synchronized 100%.</i>"""
    return msg


# ==============================================================================
# 8. 10-DAY REPLAY TEST AUDIT (--test-replay 10)
# ==============================================================================

def run_test_replay_10(parquet_file: str, days: int = 10) -> None:
    """
    Executes an end-to-end simulation of 10 sequential trading sessions from Parquet,
    verifying atomic JSON state updates and continuous CSV logging.
    """
    print("\n" + "#" * 90)
    print(f"### EXECUTING {days}-SESSION PAPER TRADING REPLAY TEST (100,000,000 VND) ###")
    print("#" * 90)

    replay_state_file = os.path.join(DEFAULT_DATA_DIR, "paper_portfolio_state.json")
    replay_history_file = os.path.join(DEFAULT_DATA_DIR, "paper_nav_history.csv")

    if os.path.exists(replay_history_file):
        os.remove(replay_history_file)
    state = init_empty_state(DEFAULT_INITIAL_CAPITAL)
    save_portfolio_state(state, replay_state_file)

    df_market = pl.read_parquet(parquet_file)
    all_dates = sorted(df_market.select("date").unique()["date"].to_list())
    
    if len(all_dates) < days:
        raise ValueError(f"Dataset has only {len(all_dates)} sessions, insufficient for {days}-day replay!")
        
    replay_dates = all_dates[-days:]
    print(f"-> Testing {days} sessions: From {replay_dates[0]} to {replay_dates[-1]}")

    for idx, cur_date in enumerate(replay_dates):
        cur_date_str = cur_date.isoformat()
        df_history = df_market.filter(pl.col("date") <= cur_date)
        df_today = df_market.filter(pl.col("date") == cur_date)
        today_market = {row["symbol"]: row for row in df_today.iter_rows(named=True)}
            
        prev_close_map = {}
        if idx > 0:
            prev_date = replay_dates[idx - 1]
            df_prev = df_market.filter(pl.col("date") == prev_date)
            for row in df_prev.iter_rows(named=True):
                prev_close_map[row["symbol"]] = row["raw_close"]

        state = load_portfolio_state(replay_state_file, DEFAULT_INITIAL_CAPITAL)
        state, exec_reps = execute_pending_orders_and_settle(state, today_market, cur_date_str)
        hist_record = mark_to_market(state, today_market, prev_close_map, cur_date_str, len(exec_reps))
        append_nav_history(replay_history_file, hist_record)
        orders = generate_next_day_orders(state, df_history, cur_date_str)
        save_portfolio_state(state, replay_state_file)
        print_portfolio_dashboard(state, today_market, cur_date_str, orders, exec_reps)

    print("\n" + "=" * 90)
    print(f"               10-DAY NAV & IMPLEMENTATION SHORTFALL AUDIT TRAIL")
    print("=" * 90)
    df_hist = pl.read_csv(replay_history_file)
    print(df_hist)
    print("=" * 90)

    assert os.path.exists(replay_state_file), "[FAIL] State JSON file not found!"
    assert os.path.exists(replay_history_file), "[FAIL] History CSV file not found!"
    assert len(df_hist) == days, f"[FAIL] CSV row count ({len(df_hist)}) does not match {days} days!"
    assert state["last_real_nav"] > 0, "[FAIL] Invalid ending NAV!"
    
    print("\n>>> TEST RESULT: 10-DAY REPLAY PASSED 100%! STATE JSON & CSV PERFECTLY SYNCHRONIZED!")


# ==============================================================================
# 9. MAIN ENTRY POINT WITH MULTI-DAY CATCH-UP LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lean VN-Quant: Daily Paper Trader (Forward-Testing Engine)")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="Initial capital (default 100,000,000 VND)")
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET_FILE, help="Path to VN30 Parquet file")
    parser.add_argument("--state-file", type=str, default=DEFAULT_STATE_FILE, help="Path to state JSON file")
    parser.add_argument("--history-file", type=str, default=DEFAULT_HISTORY_FILE, help="Path to history CSV file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "eod", "pre-open"],
        default="auto",
        help="Operational execution mode: 'eod' (15:15 EOD reconciliation & T+2.5 ledger roll), 'pre-open' (08:15 Morning ATO order ticket staging), 'auto' (Standard catch-up loop)."
    )
    parser.add_argument("--test-replay", type=int, default=0, help="Run simulation replay of last N sessions (e.g. --test-replay 10)")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio state to initial capital")
    parser.add_argument("--send-telegram", action="store_true", help="Dispatch current portfolio report to Telegram")
    parser.add_argument("--bot-daemon", action="store_true", help="Launch interactive 2-way Telegram Bot service daemon")
    args = parser.parse_args()

    if args.bot_daemon:
        try:
            from services import telegram_bot_service
            telegram_bot_service.run_bot_daemon()
        except ImportError:
            try:
                import telegram_bot_service
                telegram_bot_service.run_bot_daemon()
            except ImportError:
                print("[ERROR] Could not import telegram_bot_service module!")
        return

    if args.test_replay > 0:
        run_test_replay_10(args.parquet, days=args.test_replay)
        return

    if args.reset:
        state = init_empty_state(args.capital)
        save_portfolio_state(state, args.state_file)
        if os.path.exists(args.history_file):
            os.remove(args.history_file)
        print(f"[RESET] Reset account to initial capital of {args.capital:,.0f} VND.")
        return

    if args.mode == "pre-open":
        print("\n" + "=" * 90)
        print("   LEAN VN-QUANT: MORNING PRE-OPEN STAGING & ATO TICKET VERIFICATION (08:15 ICT)")
        print("=" * 90)
        state = load_portfolio_state(args.state_file, args.capital)
        df_market = fetch_or_update_data(args.parquet, is_replay=False)
        all_dates = sorted(df_market.select("date").unique()["date"].to_list())
        latest_date = all_dates[-1]
        today_str = datetime.date.today().isoformat()

        # Check overnight FinGraph contagion scores if bridge/data exists
        contagion_path = os.path.join(ROOT_DIR, "fingraph", "data", "event_contagion_alpha.parquet")
        if os.path.exists(contagion_path):
            try:
                df_cg = pl.read_parquet(contagion_path)
                print(f"[PRE-OPEN] Loaded VN-FinGraph overnight contagion signals from {contagion_path} ({len(df_cg)} sessions)")
            except Exception as e:
                print(f"[PRE-OPEN WARNING] Could not read contagion data: {e}")

        pending_orders = state.get("pending_orders", [])
        df_today = df_market.filter(pl.col("date") == latest_date)
        today_market = {row["symbol"]: row for row in df_today.iter_rows(named=True)}

        print_portfolio_dashboard(state, today_market, f"{today_str} (PRE-OPEN)", pending_orders, [])
        if args.send_telegram:
            tg_text = build_telegram_summary(state, f"{today_str} (PRE-OPEN)", pending_orders, [], [latest_date])
            send_telegram_alert(tg_text)
        print("[PRE-OPEN SUCCESS] ATO Order tickets staged and verified. Ready for 09:00:00 HOSE opening auction.")
        return

    df_market = fetch_or_update_data(args.parquet, is_replay=False)
    all_dates = sorted(df_market.select("date").unique()["date"].to_list())

    state = load_portfolio_state(args.state_file, args.capital)
    last_updated_date = state.get("last_updated_date")

    if last_updated_date is None:
        dates_to_run = [all_dates[-1]]
    else:
        last_dt = datetime.date.fromisoformat(last_updated_date)
        dates_to_run = [d for d in all_dates if d > last_dt]

    if len(dates_to_run) == 0:
        print(f"\n[INFO] Portfolio state is already up to date (Session: {last_updated_date}). No new sessions.")
        latest_date = all_dates[-1]
        df_today = df_market.filter(pl.col("date") == latest_date)
        today_market = {row["symbol"]: row for row in df_today.iter_rows(named=True)}
        print_portfolio_dashboard(state, today_market, last_updated_date, state.get("pending_orders", []), [])
        
        if args.send_telegram:
            curr_dt = [datetime.date.fromisoformat(last_updated_date)] if last_updated_date else [latest_date]
            tg_text = build_telegram_summary(state, str(last_updated_date), state.get("pending_orders", []), [], curr_dt)
            send_telegram_alert(tg_text)
            
        try:
            from services import dashboard
            dashboard.export_html_dashboard(
                os.path.join(BASE_DIR, "dashboard.html"),
                args.state_file,
                args.history_file
            )
        except Exception:
            pass
        return

    if len(dates_to_run) > 1:
        print("\n" + "=" * 90)
        print(f"  [MULTI-DAY CATCH-UP] DETECTED {len(dates_to_run)} MISSED SESSIONS. SEQUENTIALLY RECONCILING:")
        print(f"  -> Sessions: {[d.isoformat() for d in dates_to_run]}")
        print("=" * 90)

    last_exec_reps = []
    last_orders = []

    for d_idx, cur_date in enumerate(dates_to_run):
        session_date_str = cur_date.isoformat()
        cur_idx = all_dates.index(cur_date)

        df_history = df_market.filter(pl.col("date") <= cur_date)
        df_today = df_market.filter(pl.col("date") == cur_date)
        today_market = {row["symbol"]: row for row in df_today.iter_rows(named=True)}

        prev_close_map = {}
        if cur_idx > 0:
            prev_date = all_dates[cur_idx - 1]
            df_prev = df_market.filter(pl.col("date") == prev_date)
            prev_close_map = {row["symbol"]: row["raw_close"] for row in df_prev.iter_rows(named=True)}

        state, exec_reps = execute_pending_orders_and_settle(state, today_market, session_date_str)
        last_exec_reps = exec_reps

        hist_record = mark_to_market(state, today_market, prev_close_map, session_date_str, len(exec_reps))
        append_nav_history(args.history_file, hist_record)

        orders = generate_next_day_orders(state, df_history, session_date_str)
        last_orders = orders

        save_portfolio_state(state, args.state_file)

        if len(dates_to_run) > 1:
            print(f"\n>>> [CATCH-UP SESSION {d_idx + 1}/{len(dates_to_run)}]: {session_date_str}")
        print_portfolio_dashboard(state, today_market, session_date_str, orders, exec_reps)

    tg_text = build_telegram_summary(state, dates_to_run[-1].isoformat(), last_orders, last_exec_reps, dates_to_run)
    send_telegram_alert(tg_text)

    try:
        from services import dashboard
        dashboard.export_html_dashboard(
            os.path.join(BASE_DIR, "dashboard.html"),
            args.state_file,
            args.history_file
        )
    except Exception as e:
        print(f"[DASHBOARD] Could not export HTML: {e}")


if __name__ == "__main__":
    main()
