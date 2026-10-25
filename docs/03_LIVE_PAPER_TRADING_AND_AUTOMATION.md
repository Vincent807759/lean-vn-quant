# Production Paper Trading & Operational Automation
**Lean VN-Quant: Automated Forward-Testing, Telegram Bot & Web Dashboard**  
**Classification:** Technical Whitepaper / Operations Manual  
**Core Services:** `services/daily_paper_trader.py`, `services/telegram_bot_service.py`, `services/dashboard.py`  
**Status:** ✅ PRODUCTION ACTIVE (100% OPERATIONAL)  

---

## 1. Executive Summary

Moving from historical backtesting to live production introduces operational failure modes:
- **Discretization Cash Drag:** Rounding target continuous weights to 100-share lots leaves substantial capital idle in small accounts ($100\text{M VND}$).
- **Intermittent Offline Gaps:** Personal computers or edge servers are occasionally powered off over weekends or holidays, risking broken $T+2.5$ settlement state tracking.
- **Remote Monitoring Need:** Quants require real-time portfolio oversight, manual override capabilities, and automated order generation without opening complex IDEs.

Lean VN-Quant addresses these operational challenges through a modular services layer:
1. `services/daily_paper_trader.py`: Autonomous forward-testing engine with multi-day catch-up loop and greedy cash sweep.
2. `services/telegram_bot_service.py`: Zero-dependency, two-way interactive Telegram Bot daemon.
3. `services/dashboard.py`: Responsive real-time dark-theme web dashboard powered by Chart.js.

---

## 2. Forward-Testing Execution Flow

```
Daily Trigger at 15:15 Mon-Fri (Windows Task Scheduler / Cron)
               │
               ▼
[Step 1] Ingest Latest Daily OHLCV Candle from Entrade/DNSE REST API
               │
               ▼
[Step 2] Read last_updated_date from data/paper_portfolio_state.json
               ├─ If last_updated_date == today:
               │   └─ State is fresh -> Print status & Exit safely (Anti-Duplicate Guard)
               │
               └─ If missed sessions detected (k > 1):
                   └─ Trigger Deterministic Multi-Day Catch-up Loop across d_1, ..., d_k
               │
               ▼
[Step 3] Morning ATO Matching & Settlement Roll Engine:
         • Roll Cash: Receivable T+2 -> Receivable T+1 -> Available Cash
         • Roll Holdings: Locked T+2 -> Locked T+1 -> Sellable Shares
         • Match pending orders at raw_open (Ceiling/Floor liquidity guards)
         • Deduct brokerage (0.15%), tax (0.10%), slippage (0.10%), advance fee (0.035%)
               │
               ▼
[Step 4] End-of-Day Mark-to-Market & Audit Trail:
         • Calculate Real NAV vs Naive T+0 Benchmark
         • Append audit row to data/paper_nav_history.csv
               │
               ▼
[Step 5] Generate Next-Day ATO Order Ticket:
         • Compute Breakout Z-Score with MA50 Trend Filter
         • Dampen turnover via Rank Hysteresis
         • Sells strictly capped at current sellable_qty
         • Greedy Cash Sweep: Allocates idle cash (> 15% NAV) across Top 4 Alphas
               │
               ▼
[Step 6] Dispatch Instant Alert & Update UI:
         • Dispatch HTML summary to smartphone via Telegram Bot API
         • Regenerate services/dashboard.html
```

---

## 3. Greedy Cash Sweep Mechanics

In small production portfolios ($100\text{M VND}$), rounding continuous weights down to 100-share lots leaves up to **$53.8\text{M VND}$ ($53.8\%$)** in uninvested cash.

When market regime is confirmed bullish ($\text{VN30} > \text{MA50}$), the **Greedy Cash Sweep** algorithm is triggered:
1. Calculates remaining cash buffer: $\text{RemCash} = \text{TotalFunds} - \text{EstBuyOutlay}$.
2. If $\text{RemCash} > 0.15 \times \text{RealNAV}$, selects the Top 4 Alpha candidate stocks.
3. Incrementally adds $100$-share lots across candidates until cash buffer drops to $5\% - 10\%$ of NAV.
4. Capital deployment increases from **$46.2\%$** to **$88.5\% - 95.0\%$**, dramatically reducing cash drag while preserving a prudent cash buffer.

---

## 4. Interactive 2-Way Telegram Bot Daemon

The Telegram Bot daemon (`services/telegram_bot_service.py`) operates with zero external dependencies (pure `requests`):

### 4.1. Supported Mobile Commands
- `/status` or `/nav`: View Real NAV, available cash, T+1/T+2 receivables, and Implementation Shortfall.
- `/holdings`: Full breakdown of T+2.5 holdings (`sellable_qty`, `locked_t1`, `locked_t2`, cost price, PnL).
- `/orders`: Tomorrow morning's ATO order ticket.
- `/sync`: Remote-triggers market data update and ledger reconciliation directly from smartphone.
- `/help`: Displays interactive command menu.

### 4.2. Running the Daemon
```bash
# Launch background daemon
python services/telegram_bot_service.py

# Or via daily_paper_trader CLI
python services/daily_paper_trader.py --bot-daemon
```

---

## 5. Real-Time Interactive Web Dashboard

Run the built-in HTTP server to inspect portfolio status on any browser:

```bash
python services/dashboard.py --serve --port 8050
```
Open `http://localhost:8050/dashboard.html` to view:
- Responsive equity curves (Real NAV vs Naive T+0 vs VN30 Benchmark).
- Capital allocation doughnut charts.
- Live T+2.5 settlement holdings.
- Morning ATO order tickets with target weights and factor signals.
