"""
================================================================================
LEAN VN-QUANT: TELEGRAM BOT 2-WAY INTERACTIVE SERVICE (TELEGRAM_BOT_SERVICE.PY)
================================================================================
Interactive 2-way Telegram Bot service daemon (zero-dependency):
1. Auto-Discovery of TELEGRAM_CHAT_ID via getUpdates API.
2. Two-way commands from smartphone:
   - /start, /help: Instructions and available command list.
   - /status, /nav: Real NAV, cash balance, T+1/T+2 receivables & Implementation Shortfall.
   - /holdings: Detailed breakdown of T+2.5 stock holdings (sellable, locked, cost, PnL).
   - /orders: Morning ATO order ticket for tomorrow.
   - /sync: Remote execution trigger to fetch market data and settle ledger from mobile.
3. Continuous background daemon or single-shot execution mode.
================================================================================
"""

import os
import sys
import time
import json
import datetime
import subprocess
import requests
from typing import Dict, Any, List, Optional, Tuple

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ENV_FILE_PATHS = [
    os.path.join(os.getcwd(), ".env"),
    os.path.join(ROOT_DIR, ".env"),
    os.path.join(BASE_DIR, ".env"),
    os.path.join(BASE_DIR, "..", ".env"),
]


# ==============================================================================
# 1. ENVIRONMENT VARIABLES & SECURITY FILTERING
# ==============================================================================

def load_env() -> Dict[str, str]:
    """Finds and parses local .env file without external dependencies."""
    env = {}
    for p in ENV_FILE_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k not in env:
                                env[k] = v
            except Exception:
                pass
    return env


def mask_token(text: str, token: Optional[str] = None) -> str:
    """Masks bot tokens and sensitive patterns from logs and exception messages."""
    if not text:
        return ""
    if token and len(token) > 6:
        text = text.replace(token, f"{token[:4]}...***")
    import re
    return re.sub(r'bot\d+:[A-Za-z0-9_-]+', 'bot[MASKED_TOKEN]', text)


def send_message(token: str, chat_id: str, text: str) -> bool:
    """Sends HTML formatted message to Telegram chat with masked error handling."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        return resp.status_code == 200
    except Exception as e:
        safe_msg = mask_token(str(e), token)
        print(f"[SEND ERROR] Failed to deliver message: {safe_msg}")
        return False


# ==============================================================================
# 2. BOT COMMAND HANDLERS
# ==============================================================================

def get_state_file_path() -> str:
    candidates = [
        os.path.join(ROOT_DIR, "data", "paper_portfolio_state.json"),
        os.path.join(os.getcwd(), "data", "paper_portfolio_state.json"),
        os.path.join(BASE_DIR, "..", "data", "paper_portfolio_state.json"),
        os.path.join(os.getcwd(), "paper_portfolio_state.json"),
        "paper_portfolio_state.json"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(ROOT_DIR, "data", "paper_portfolio_state.json")


def get_history_file_path() -> str:
    candidates = [
        os.path.join(ROOT_DIR, "data", "paper_nav_history.csv"),
        os.path.join(os.getcwd(), "data", "paper_nav_history.csv"),
        os.path.join(BASE_DIR, "..", "data", "paper_nav_history.csv"),
        os.path.join(os.getcwd(), "paper_nav_history.csv"),
        "paper_nav_history.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(ROOT_DIR, "data", "paper_nav_history.csv")


def handle_start_or_help() -> str:
    """Handles /start and /help commands."""
    return """🤖 <b>VN-QUANT INTERACTIVE BOT (v2.0 PRODUCTION)</b>
Quantitative portfolio execution & monitoring engine for VN30.

📋 <b>AVAILABLE COMMANDS:</b>
• <b>/status</b> or <b>/nav</b> : View total Real NAV, cash buffer, receivables & Shortfall
• <b>/holdings</b> : Detailed breakdown of T+2.5 stock holdings
• <b>/orders</b> : Tomorrow morning's ATO order ticket
• <b>/sync</b> : Trigger data update & settle portfolio immediately
• <b>/help</b> : Display this command menu

<i>Automated reports are dispatched daily after 15:15 Mon-Fri.</i>"""


def handle_status() -> str:
    """Handles /status and /nav commands."""
    state_file = get_state_file_path()
    if not os.path.exists(state_file):
        return "⚠️ <i>State file paper_portfolio_state.json not found.</i>"
        
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        real_nav = state.get("last_real_nav", state.get("initial_capital", 1e8))
        init_cap = state.get("initial_capital", 1e8)
        tot_ret = (real_nav - init_cap) / init_cap * 100.0
        
        avail_cash = state.get("available_cash", 0.0)
        rec_t1 = state.get("receivable_cash_t1", 0.0)
        rec_t2 = state.get("receivable_cash_t2", 0.0)
        tot_rec = rec_t1 + rec_t2
        
        naive_nav = state.get("naive_nav", init_cap)
        shortfall = (naive_nav - real_nav) / init_cap * 100.0
        
        fees = state.get("total_brokerage_tax", 0.0)
        adv_fees = state.get("total_advance_fees", 0.0)
        last_date = state.get("last_updated_date", "Pending initial run")
        holdings = state.get("holdings", {})
        
        stock_val = 0.0
        for sym, h in holdings.items():
            tot_s = h.get("sellable_qty", 0) + h.get("locked_t1", 0) + h.get("locked_t2", 0)
            stock_val += (tot_s * h.get("avg_cost", 0.0))
            
        invested_pct = (stock_val / real_nav * 100.0) if real_nav > 0 else 0.0
        cash_pct = ((avail_cash + tot_rec) / real_nav * 100.0) if real_nav > 0 else 0.0
        
        msg = f"""📊 <b>PORTFOLIO STATUS REPORT (MARK-TO-MARKET)</b>
📅 Last Session: <b>{last_date}</b>

💰 <b>CAPITAL & PERFORMANCE:</b>
• <b>Real NAV:</b> <code>{real_nav:,.0f} VND</code> (<b>{tot_ret:+.2f}%</b>)
• <b>Naive T+0 Benchmark:</b> <code>{naive_nav:,.0f} VND</code>
• <b>Implementation Shortfall:</b> <b>{shortfall:+.2f}%</b> (Friction & settlement drag)

📦 <b>ASSET ALLOCATION:</b>
• <b>Equities ({len(holdings)} stocks):</b> <code>{stock_val:,.0f} VND</code> (<b>{invested_pct:.1f}%</b>)
• <b>Available Cash:</b> <code>{avail_cash:,.0f} VND</code>
• <b>T+1 Receivable Cash:</b> <code>{rec_t1:,.0f} VND</code>
• <b>T+2 Receivable Cash:</b> <code>{rec_t2:,.0f} VND</code>
• <b>Total Cash Buffer:</b> <code>{avail_cash + tot_rec:,.0f} VND</code> (<b>{cash_pct:.1f}%</b>)

💳 <b>CUMULATIVE FRICTION COSTS:</b>
• Brokerage Fees & Taxes: <code>{fees:,.0f} VND</code>
• T+2 Advance Interest Cost: <code>{adv_fees:,.0f} VND</code>"""
        return msg
    except Exception as e:
        return f"⚠️ Error reading portfolio state: {e}"


def handle_holdings() -> str:
    """Handles /holdings command."""
    state_file = get_state_file_path()
    if not os.path.exists(state_file):
        return "⚠️ <i>Holdings data not found.</i>"
        
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        holdings = state.get("holdings", {})
        if not holdings:
            return "📦 <b>CURRENT HOLDINGS:</b> 100% Cash. No open equity positions."
            
        lines = [
            f"📦 <b>T+2.5 EQUITY HOLDINGS ({len(holdings)} STOCKS)</b>",
            f"<i>Settlement Breakdown: Sellable | T+1 | T+2</i>\n"
        ]
        
        tot_val = 0.0
        for sym, h in sorted(holdings.items()):
            sellable = h.get("sellable_qty", 0)
            t1 = h.get("locked_t1", 0)
            t2 = h.get("locked_t2", 0)
            tot_qty = sellable + t1 + t2
            cost = h.get("avg_cost", 0.0)
            pos_val = tot_qty * cost
            tot_val += pos_val
            
            lines.append(f"• <b>{sym}</b>: <b>{tot_qty:,}</b> shares (Avg: <code>{cost:,.0f} VND</code>)")
            lines.append(f"  └ Tradable: <code>{sellable:,}</code> | Locked T+1: <code>{t1:,}</code> | Locked T+2: <code>{t2:,}</code>")
            lines.append(f"  └ Position Value: <code>{pos_val:,.0f} VND</code>")
            
        lines.append(f"\n<b>Total Market Exposure:</b> <code>{tot_val:,.0f} VND</code>")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Error reading holdings: {e}"


def handle_orders() -> str:
    """Handles /orders command."""
    state_file = get_state_file_path()
    if not os.path.exists(state_file):
        return "⚠️ <i>Order ticket data not found.</i>"
        
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        orders = state.get("pending_orders", [])
        if not orders:
            return "📝 <b>TOMORROW ATO ORDER TICKET:</b> No pending orders. Maintaining positions (No-Trade Buffer)."
            
        lines = [
            f"📝 <b>PENDING ATO ORDER TICKET ({len(orders)} ORDERS)</b>",
            f"<i>Queued for execution at tomorrow morning's ATO</i>\n"
        ]
        
        tot_val = 0.0
        for o in orders:
            act = "🟢 BUY" if o["action"] == "BUY" else "🔴 SELL"
            sym = o["symbol"]
            q = o["qty"]
            p = o["est_price"]
            v = o["est_value"]
            w = o.get("target_weight", 0.0) * 100.0
            tot_val += v
            lines.append(f"• {act} <b>{q:,} shs {sym}</b> @ {p:,.0f} VND")
            lines.append(f"  └ Est Value: <code>{v:,.0f} VND</code> (Target: {w:.1f}%)")
            
        lines.append(f"\n<b>Total Estimated Outlay:</b> <code>{tot_val:,.0f} VND</code>")
        lines.append("<i>Status: Queued in pending_orders ready for morning ATO session.</i>")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Error reading orders: {e}"


_last_sync_timestamp: float = 0.0
SYNC_RATE_LIMIT_SECONDS: float = 60.0


def handle_sync() -> str:
    """
    Handles /sync command: triggers daily_paper_trader.py
    with rate-limiting (max 1 trigger per 60 seconds).
    """
    global _last_sync_timestamp
    now = time.time()
    elapsed = now - _last_sync_timestamp
    if elapsed < SYNC_RATE_LIMIT_SECONDS:
        remaining = int(SYNC_RATE_LIMIT_SECONDS - elapsed)
        return f"⏳ <b>RATE LIMIT EXCEEDED:</b> Please wait {remaining}s before triggering <b>/sync</b> again."
        
    _last_sync_timestamp = now
    python_exe = sys.executable
    script_path = os.path.join(BASE_DIR, "daily_paper_trader.py")
    if not os.path.exists(script_path):
        script_path = os.path.join(ROOT_DIR, "services", "daily_paper_trader.py")
        
    cmd = [python_exe, script_path, "--send-telegram"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            return "⚡ <b>SYNCHRONIZATION COMPLETED!</b>\nMarket data updated and T+2.5 ledger settled.\nFull report dispatched via Telegram."
        else:
            return f"⚠️ <b>SYNC ERROR:</b>\n{proc.stderr[:400]}"
    except Exception as e:
        return f"⚠️ Sync execution error: {e}"


def process_command(cmd: str) -> str:
    """Dispatches incoming slash command to appropriate handler."""
    cmd_clean = cmd.strip().lower().split("@")[0]
    if cmd_clean in ["/start", "/help"]:
        return handle_start_or_help()
    elif cmd_clean in ["/status", "/nav"]:
        return handle_status()
    elif cmd_clean == "/holdings":
        return handle_holdings()
    elif cmd_clean == "/orders":
        return handle_orders()
    elif cmd_clean == "/sync":
        return handle_sync()
    else:
        return f"❓ Unrecognized command <code>{cmd}</code>. Send <b>/help</b> for available command list."


# ==============================================================================
# 3. BACKGROUND POLLING DAEMON
# ==============================================================================

def run_bot_daemon(poll_interval: int = 2) -> None:
    """
    Continuous background daemon polling Telegram getUpdates.
    """
    print("\n" + "=" * 80)
    print("      LEAN VN-QUANT: TELEGRAM BOT 2-WAY INTERACTIVE SERVICE DAEMON")
    print("=" * 80)
    
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
    admin_chat_id = (
        env.get("TELEGRAM_ADMIN_CHAT_ID")
        or os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
        or env.get("TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
    )
    
    if not token:
        print("[FATAL ERROR] TELEGRAM_BOT_TOKEN not found in .env or environment!")
        sys.exit(1)
        
    if not admin_chat_id:
        print("[FATAL ERROR] TELEGRAM_ADMIN_CHAT_ID not configured! Auto-discovery is disabled for security.")
        sys.exit(1)

    print(f"-> Authorized Admin Chat ID: {admin_chat_id}")
    print(f"-> Bot daemon is listening for incoming commands (/status, /holdings, /orders, /sync)...")
    print("=" * 80 + "\n")
    
    last_update_id = 0
    try:
        warmup_resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates?offset=-1", timeout=5)
        if warmup_resp.status_code == 200:
            res = warmup_resp.json().get("result", [])
            if res:
                last_update_id = res[-1]["update_id"]
    except Exception as e:
        print(f"[WARMUP WARNING] {mask_token(str(e), token)}")

    send_message(
        token, admin_chat_id,
        "🟢 <b>LEAN VN-QUANT BOT ONLINE!</b>\nInteractive service active. Send <b>/status</b> or <b>/help</b> to manage portfolio."
    )

    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={last_update_id + 1}&timeout=10"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                updates = resp.json().get("result", [])
                for u in updates:
                    last_update_id = u["update_id"]
                    msg = u.get("message")
                    if not msg:
                        continue
                    
                    sender_chat_id = str(msg["chat"]["id"])
                    text = msg.get("text", "").strip()
                    
                    # Strict Authorization Check
                    if sender_chat_id != str(admin_chat_id):
                        print(f"[SECURITY WARNING] Rejected command from unauthorized chat_id {sender_chat_id}: {text}")
                        continue
                        
                    if text.startswith("/"):
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Received command: {text}")
                        reply_html = process_command(text)
                        send_message(token, sender_chat_id, reply_html)
                        
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Telegram Bot Daemon stopped by user.")
            break
        except Exception as e:
            print(f"[DAEMON ERROR] {mask_token(str(e), token)}")
            time.sleep(3)


if __name__ == "__main__":
    run_bot_daemon()
