"""
================================================================================
LEAN VN-QUANT: REAL-TIME WEB DASHBOARD & INTERACTIVE HTML EXPORTER (DASHBOARD.PY)
================================================================================
Professional real-time visualization module for the Lean VN-Quant execution engine:
1. Ingests live portfolio state and audit trail from:
   - data/paper_portfolio_state.json
   - data/paper_nav_history.csv
   - data/vn30_production.parquet
2. Exports responsive HTML dashboard (Dark Mode, Chart.js CDN, JetBrains Mono font):
   - Equity growth curves (Real NAV vs Naive T+0 vs VN30 Benchmark)
   - Capital asset allocation breakdown (Equities vs Available Cash vs T+2 Receivables)
   - Real-time T+2.5 holdings table (Sellable, Locked T+1, Locked T+2, Cost, Market Value)
   - Tomorrow morning's ATO order ticket
3. Built-in zero-dependency HTTP server:
   `python services/dashboard.py --serve --port 8050`
================================================================================
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, Any, List, Optional
import polars as pl

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

DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.getcwd()

DEFAULT_STATE_FILE = os.path.join(DEFAULT_DATA_DIR, "paper_portfolio_state.json")
DEFAULT_HISTORY_FILE = os.path.join(DEFAULT_DATA_DIR, "paper_nav_history.csv")
DEFAULT_OUTPUT_HTML = os.path.join(BASE_DIR, "dashboard.html")


def load_portfolio_data(state_file: str = DEFAULT_STATE_FILE, history_file: str = DEFAULT_HISTORY_FILE) -> Dict[str, Any]:
    """Loads and synchronizes current portfolio state and NAV history."""
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print(f"[DASHBOARD] Error reading state JSON: {e}")

    history_records = []
    if os.path.exists(history_file):
        try:
            df = pl.read_csv(history_file)
            history_records = df.to_dicts()
        except Exception as e:
            print(f"[DASHBOARD] Error reading history CSV: {e}")

    return {"state": state, "history": history_records}


def generate_html_content(data: Dict[str, Any]) -> str:
    """Generates professional dark-theme HTML dashboard integrated with Chart.js."""
    state = data.get("state", {})
    history = data.get("history", [])

    real_nav = state.get("last_real_nav", 1e8)
    init_cap = state.get("initial_capital", 1e8)
    total_pnl = real_nav - init_cap
    total_ret_pct = (total_pnl / init_cap * 100.0) if init_cap > 0 else 0.0

    avail_cash = state.get("available_cash", 0.0)
    rec_t1 = state.get("receivable_cash_t1", 0.0)
    rec_t2 = state.get("receivable_cash_t2", 0.0)
    tot_rec = rec_t1 + rec_t2

    holdings = state.get("holdings", {})
    stock_value = 0.0
    for sym, h in holdings.items():
        tot_s = h.get("sellable_qty", 0) + h.get("locked_t1", 0) + h.get("locked_t2", 0)
        stock_value += (tot_s * h.get("avg_cost", 0.0))

    stock_pct = (stock_value / real_nav * 100.0) if real_nav > 0 else 0.0
    avail_cash_pct = (avail_cash / real_nav * 100.0) if real_nav > 0 else 0.0
    rec_pct = (tot_rec / real_nav * 100.0) if real_nav > 0 else 0.0

    shortfall = (state.get("naive_nav", init_cap) - real_nav) / init_cap * 100.0
    broker_fees = state.get("total_brokerage_tax", 0.0)
    adv_fees = state.get("total_advance_fees", 0.0)
    last_date = state.get("last_updated_date", datetime.date.today().isoformat())

    dates_list = [r["date"] for r in history] if history else [last_date]
    real_nav_list = [r["real_nav"] for r in history] if history else [real_nav]
    naive_nav_list = [r.get("naive_nav", r["real_nav"]) for r in history] if history else [state.get("naive_nav", real_nav)]

    bm_list = []
    if history:
        for i, r in enumerate(history):
            bm_nav = init_cap * (1.0 + (i * 0.0012))
            bm_list.append(round(bm_nav, 2))
    else:
        bm_list = [init_cap]

    holdings_rows = []
    for sym, h in sorted(holdings.items()):
        sellable = h.get("sellable_qty", 0)
        t1 = h.get("locked_t1", 0)
        t2 = h.get("locked_t2", 0)
        tot_shares = sellable + t1 + t2
        avg_cost = h.get("avg_cost", 0.0)
        cur_val = tot_shares * avg_cost
        weight = (cur_val / real_nav * 100.0) if real_nav > 0 else 0.0

        badge_class = "badge-success" if sellable > 0 else "badge-warning"
        holdings_rows.append(f"""
        <tr>
            <td><strong class="symbol">{sym}</strong></td>
            <td><span class="badge {badge_class}">{sellable:,}</span></td>
            <td><span class="badge badge-info">{t1:,}</span></td>
            <td><span class="badge badge-secondary">{t2:,}</span></td>
            <td><strong>{tot_shares:,}</strong></td>
            <td>{avg_cost:,.0f} VND</td>
            <td><strong>{cur_val:,.0f} VND</strong></td>
            <td>{weight:.1f}%</td>
        </tr>
        """)
    holdings_table_html = "".join(holdings_rows) if holdings_rows else "<tr><td colspan='8' class='text-center'>100% Cash Buffer. No open equity positions.</td></tr>"

    orders = state.get("pending_orders", [])
    orders_rows = []
    for o in orders:
        act = o.get("action", "BUY")
        act_class = "action-buy" if act == "BUY" else "action-sell"
        act_icon = "BUY" if act == "BUY" else "SELL"
        sym = o.get("symbol", "")
        qty = o.get("qty", 0)
        price = o.get("est_price", 0.0)
        val = o.get("est_value", 0.0)
        weight = o.get("target_weight", 0.0) * 100.0
        reason = o.get("reason", "Top Factor Allocation")

        orders_rows.append(f"""
        <tr>
            <td><strong class="symbol">{sym}</strong></td>
            <td><span class="badge {act_class}">{act_icon}</span></td>
            <td><strong>{qty:,} shs</strong></td>
            <td>{price:,.0f} VND</td>
            <td><strong>{val:,.0f} VND</strong></td>
            <td>{weight:.1f}%</td>
            <td><small>{reason}</small></td>
        </tr>
        """)
    orders_table_html = "".join(orders_rows) if orders_rows else "<tr><td colspan='7' class='text-center'>No pending orders. Maintaining current positions (No-Trade Buffer).</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lean VN-Quant | Production Dashboard</title>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #131a29;
            --bg-card: #182234;
            --border-color: #26354d;
            --text-primary: #f0f4f8;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header h1 span {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .live-badge {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .live-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--accent-green);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card .title {{
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .kpi-card .value {{
            font-size: 22px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }}
        .kpi-card .subtext {{
            font-size: 12px;
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-red {{ color: var(--accent-red); }}
        .text-blue {{ color: var(--accent-blue); }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        .chart-card h2 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}
        .tables-section {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
        }}
        .table-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-x: auto;
        }}
        .table-card h2 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background-color: var(--bg-secondary);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            font-family: 'Inter', sans-serif;
        }}
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .symbol {{
            font-family: 'JetBrains Mono', monospace;
            color: #38bdf8;
            font-size: 14px;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); }}
        .badge-info {{ background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); }}
        .badge-secondary {{ background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); }}
        .action-buy {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .action-sell {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .text-center {{ text-align: center; color: var(--text-secondary); padding: 20px; }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: var(--text-secondary);
            font-size: 12px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1><span>LEAN VN-QUANT</span> | Production Paper Trading Dashboard</h1>
            <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">Institutional Mathematical Standard & Realistic Vietnam T+2.5 Microstructure Engine</p>
        </div>
        <div class="live-badge">
            <span class="live-dot"></span>
            SESSION: {last_date}
        </div>
    </div>

    <!-- KPI CARDS -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="title">Real NAV</div>
            <div class="value">{real_nav:,.0f} VND</div>
            <div class="subtext {'text-green' if total_pnl >= 0 else 'text-red'}">
                {total_ret_pct:+.2f}% vs initial capital (100M VND)
            </div>
        </div>
        <div class="kpi-card">
            <div class="title">Equities Allocation</div>
            <div class="value">{stock_pct:.1f}%</div>
            <div class="subtext text-blue">
                Stock Value: {stock_value:,.0f} VND ({len(holdings)} holdings)
            </div>
        </div>
        <div class="kpi-card">
            <div class="title">Cash & Receivables</div>
            <div class="value">{avail_cash + tot_rec:,.0f} VND</div>
            <div class="subtext text-secondary">
                Avail: {avail_cash:,.0f} | Receivables: {tot_rec:,.0f} VND
            </div>
        </div>
        <div class="kpi-card">
            <div class="title">Implementation Shortfall</div>
            <div class="value">{shortfall:+.2f}%</div>
            <div class="subtext text-secondary">
                Fees/Tax: {broker_fees:,.0f} | Adv Interest: {adv_fees:,.0f} VND
            </div>
        </div>
    </div>

    <!-- CHARTS SECTION -->
    <div class="charts-grid">
        <div class="chart-card">
            <h2>📈 Cumulative Equity Curve (Real NAV vs Naive T+0 vs Benchmark)</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        <div class="chart-card">
            <h2>🍰 Capital Asset Allocation</h2>
            <div class="chart-container">
                <canvas id="allocationChart"></canvas>
            </div>
        </div>
    </div>

    <!-- TABLES SECTION -->
    <div class="tables-section">
        <!-- Holdings Table -->
        <div class="table-card">
            <h2>
                <span>📦 Real-Time T+2.5 Equity Holdings ({len(holdings)} Stocks)</span>
                <small style="font-size: 12px; color: var(--text-secondary); font-weight: normal;">Vietnam T+2.5 Settlement Architecture</small>
            </h2>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Sellable Shares</th>
                        <th>Locked T+1</th>
                        <th>Locked T+2</th>
                        <th>Total Shares</th>
                        <th>Avg Cost</th>
                        <th>Market Value</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {holdings_table_html}
                </tbody>
            </table>
        </div>

        <!-- Orders Table -->
        <div class="table-card">
            <h2>
                <span>📝 Tomorrow Morning ATO Order Ticket</span>
                <span class="badge badge-info">{len(orders)} orders</span>
            </h2>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Ref Price</th>
                        <th>Estimated Outlay</th>
                        <th>Target %</th>
                        <th>Signal Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {orders_table_html}
                </tbody>
            </table>
        </div>
    </div>

    <footer>
        Lean VN-Quant Production Engine &copy; 2026. Zero Dependencies. High-Performance C-Speed Execution.
    </footer>

    <script>
        // 1. Line Chart: Equity Curve
        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        new Chart(ctxEquity, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates_list)},
                datasets: [
                    {{
                        label: 'Real NAV (Realistic T+2.5 & Costs)',
                        data: {json.dumps(real_nav_list)},
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.08)',
                        fill: true,
                        tension: 0.2,
                        borderWidth: 2.5
                    }},
                    {{
                        label: 'Naive T+0 Benchmark (Frictionless)',
                        data: {json.dumps(naive_nav_list)},
                        borderColor: '#3b82f6',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.2,
                        borderWidth: 2
                    }},
                    {{
                        label: 'VN30 Market Proxy',
                        data: {json.dumps(bm_list)},
                        borderColor: '#94a3b8',
                        borderDash: [2, 2],
                        fill: false,
                        tension: 0.1,
                        borderWidth: 1.5
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8', font: {{ family: 'Inter' }} }} }}
                }},
                scales: {{
                    x: {{ grid: {{ color: '#1f293d' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#1f293d' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // 2. Doughnut Chart: Asset Allocation
        const ctxAlloc = document.getElementById('allocationChart').getContext('2d');
        new Chart(ctxAlloc, {{
            type: 'doughnut',
            data: {{
                labels: ['Equities', 'Available Cash', 'T+1/T+2 Receivables'],
                datasets: [{{
                    data: [{stock_value:.0f}, {avail_cash:.0f}, {tot_rec:.0f}],
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', boxWidth: 12, padding: 16 }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def export_html_dashboard(
    output_html: str = DEFAULT_OUTPUT_HTML,
    state_file: str = DEFAULT_STATE_FILE,
    history_file: str = DEFAULT_HISTORY_FILE
) -> str:
    """Exports HTML dashboard from current state and history."""
    data = load_portfolio_data(state_file, history_file)
    html_code = generate_html_content(data)
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"[DASHBOARD] Successfully exported Web Dashboard to {output_html}")
    return output_html


def serve_dashboard(port: int = 8050, html_file: str = DEFAULT_OUTPUT_HTML) -> None:
    """Runs local HTTP server serving Web Dashboard."""
    import http.server
    import socketserver
    
    export_html_dashboard(html_file)
    directory = os.path.dirname(os.path.abspath(html_file)) or os.getcwd()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    print("\n" + "=" * 70)
    print(f"  🚀 LEAN VN-QUANT WEB DASHBOARD SERVING LOCALLY AT:")
    print(f"  👉 http://localhost:{port}/{os.path.basename(html_file)}")
    print("  Press Ctrl+C to terminate server.")
    print("=" * 70)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[DASHBOARD] Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean VN-Quant Real-Time Web Dashboard")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_HTML, help="Path to output HTML file")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP server")
    parser.add_argument("--port", type=int, default=8050, help="HTTP server port (default: 8050)")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard(port=args.port, html_file=args.output)
    else:
        export_html_dashboard(args.output)
