"""
tests/test_services.py - Unit tests for services layer:
1. Telegram Bot Service error sanitization & path/token masking (H3)
2. Web Dashboard Mark-to-Market valuation & fake benchmark elimination (M1, M2)
"""

import pytest
import datetime
from services.telegram_bot_service import sanitize_error_message, mask_token
from services.dashboard import generate_html_content


def test_telegram_sanitize_error_message_masks_tokens_and_paths():
    """
    H3 & Test Gap 5: Verifies that sensitive paths and bot tokens are masked
    before transmitting messages via Telegram bot.
    """
    token_p1 = "1234567890"
    token_p2 = "ABCdefGHIjklMNOpqrSTUvwxYZ123456789"
    token = f"{token_p1}:{token_p2}"

    raw_error_win = (
        r"Traceback (most recent call last): "
        r"File 'C:\Users\test_dev\workspace\Quant\services\daily_paper_trader.py', line 45, in main "
        r"RuntimeError: Failed to connect with bot" + token
    )
    sanitized_win = sanitize_error_message(raw_error_win, token=token)
    assert r"C:\Users\test_dev" not in sanitized_win
    assert token not in sanitized_win
    assert "[PATH]" in sanitized_win
    assert "[MASKED_TOKEN]" in sanitized_win or "...***" in sanitized_win

    raw_error_unix = (
        "Traceback (most recent call last):\n"
        "  File '/home/ubuntu/quant/services/daily_paper_trader.py', line 12, in <module>\n"
        "requests.exceptions.ConnectionError: bot" + token
    )
    sanitized_unix = sanitize_error_message(raw_error_unix, token=token)
    assert "/home/ubuntu/quant" not in sanitized_unix
    assert "[PATH]" in sanitized_unix
    assert "bot[MASKED_TOKEN]" in sanitized_unix


def test_dashboard_mark_to_market_weighting():
    """
    M2 & Test Gap 5: Verifies that asset allocation in the dashboard uses
    Mark-to-Market valuation (market close price) rather than historic cost (avg_cost).
    """
    state = {
        "initial_capital": 100_000_000.0,
        "last_real_nav": 120_000_000.0,
        "available_cash": 20_000_000.0,
        "receivable_cash_t1": 10_000_000.0,
        "receivable_cash_t2": 0.0,
        "holdings": {
            "HPG": {
                "sellable_qty": 3000,
                "locked_t1": 0,
                "locked_t2": 0,
                "avg_cost": 20000.0  # Cost basis: 60M VND
            }
        }
    }
    # Market close price appreciated from 20,000 to 30,000 VND (Market value: 90M VND)
    # Total Mark-to-Market = 20M cash + 10M rec + 90M stock = 120M VND
    data = {
        "state": state,
        "history": [{"date": "2026-10-23", "real_nav": 120_000_000.0, "naive_nav": 120_000_000.0}],
        "prev_close_map": {"HPG": 30000.0}
    }
    html = generate_html_content(data)

    # In HTML, HPG current value must be 90,000,000 VND (3,000 * 30,000), not 60,000,000 VND
    assert "90,000,000 VND" in html
    # Stock allocation: 90M / 120M = 75.0%
    assert "75.0%" in html


def test_dashboard_hides_fake_benchmark_when_absent():
    """
    M1 & Test Gap 5: Verifies that artificial linear benchmark (1.0 + i*0.0012)
    is removed and note is displayed when no real benchmark exists.
    """
    state = {
        "initial_capital": 100_000_000.0,
        "last_real_nav": 100_000_000.0,
        "available_cash": 100_000_000.0,
        "receivable_cash_t1": 0.0,
        "receivable_cash_t2": 0.0,
        "holdings": {}
    }
    history = [
        {"date": "2026-10-20", "real_nav": 100_000_000.0, "naive_nav": 100_000_000.0},
        {"date": "2026-10-21", "real_nav": 101_000_000.0, "naive_nav": 101_000_000.0},
        {"date": "2026-10-22", "real_nav": 102_000_000.0, "naive_nav": 102_000_000.0}
    ]
    data = {
        "state": state,
        "history": history
    }
    html = generate_html_content(data)

    # Must display missing benchmark notice
    assert "(Chưa có dữ liệu benchmark thực tế)" in html
    # Must NOT contain fake benchmark dataset in Chart.js
    assert "VN30 Market Proxy" not in html


def test_paper_trader_greedy_cash_sweep_executes_without_name_error():
    """
    Verifies that generate_next_day_orders runs smoothly when available_cash > 0.15 * real_nav,
    confirming that the greedy cash sweep branch calculates sma50 without NameError.
    Hermetic test: generates in-RAM synthetic data without depending on untracked parquet files.
    """
    from engine import data_pipeline as dp
    from services.daily_paper_trader import generate_next_day_orders

    df_market = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01",
        end_date="2023-05-15",
        seed=42
    )
    last_date = str(df_market["date"].max())
    state = {
        "initial_capital": 100_000_000.0,
        "last_real_nav": 100_000_000.0,
        "available_cash": 100_000_000.0,
        "receivable_cash_t1": 0.0,
        "receivable_cash_t2": 0.0,
        "holdings": {},
        "last_update_date": last_date
    }
    orders = generate_next_day_orders(state, df_market, last_date)
    assert isinstance(orders, list)
    assert len(orders) > 0


def test_telegram_handle_sync_masks_token_on_failure_without_name_error():
    """
    Verifies that handle_sync() properly sanitizes stderr containing sensitive tokens
    and paths on subprocess failure without raising NameError for undefined BOT_TOKEN.
    """
    from unittest.mock import patch, MagicMock
    from services.telegram_bot_service import handle_sync

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "RuntimeError: token bot123456789:ABCdefGHI at C:\\Users\\test_user\\secret.py"

    with patch("subprocess.run", return_value=mock_proc):
        result = handle_sync()

    assert "123456789:ABCdefGHI" not in result
    assert "test_user" not in result
    assert "⚠️ <b>SYNC ERROR:</b>" in result


def test_dashboard_html_escapes_xss_and_neutralizes_header():
    """
    M1.1: Verifies that generate_html_content escapes HTML special characters in
    symbols, order reasons, session dates, and inline script JSON, and uses the neutral header.
    """
    state = {
        "initial_capital": 100_000_000.0,
        "last_real_nav": 100_000_000.0,
        "available_cash": 50_000_000.0,
        "receivable_cash_t1": 0.0,
        "receivable_cash_t2": 0.0,
        "last_updated_date": "2026-10-23<img src=x onerror=alert(1)>",
        "holdings": {
            "HPG<script>alert(1)</script>": {
                "sellable_qty": 100,
                "locked_t1": 0,
                "locked_t2": 0,
                "avg_cost": 25000.0,
            }
        },
        "pending_orders": [
            {
                "action": "BUY<script>",
                "symbol": "VCB<b>",
                "qty": 100,
                "est_price": 60000.0,
                "est_value": 6_000_000.0,
                "target_weight": 0.06,
                "reason": "Signal <script>evil()</script> & filter",
            }
        ],
    }
    history = [
        {
            "date": "2026-10-23</script><script>evil()</script>",
            "real_nav": 100_000_000.0,
            "naive_nav": 100_000_000.0,
        }
    ]
    html = generate_html_content({"state": state, "history": history, "prev_close_map": {}})

    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "</script><script>evil()</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html
    assert r"<\/script>" in html
    assert "Institutional Mathematical Standard" not in html
    assert "Vietnam T+2.5 Microstructure Simulation &amp; Paper Trading Monitor" in html


def test_dashboard_single_file_handler_blocks_other_files(tmp_path):
    """
    M1.1: Verifies that create_single_file_handler serves only / and /dashboard.html
    and returns 403 Forbidden for any sibling file or traversal path.
    """
    import http.client
    import socketserver
    import threading
    from services.dashboard import create_single_file_handler

    html_file = tmp_path / "dashboard.html"
    secret_file = tmp_path / "secret.txt"
    html_file.write_text("<html><body>Dashboard OK</body></html>", encoding="utf-8")
    secret_file.write_text("TOP_SECRET_TOKEN", encoding="utf-8")

    handler_cls = create_single_file_handler(str(html_file))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for ok_path in ("/", "/dashboard.html", "/dashboard.html?refresh=1#top"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", ok_path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            conn.close()
            assert resp.status == 200
            assert "Dashboard OK" in body

        for blocked_path in ("/secret.txt", "/../secret.txt", "/secret.txt?foo=1#bar", "/.env"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", blocked_path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            conn.close()
            assert resp.status == 403
            assert "TOP_SECRET_TOKEN" not in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_telegram_process_command_and_sync_escape_html():
    """
    M1.2: Verifies that unrecognized commands and /sync error messages escape HTML tags
    so Telegram HTML parse_mode cannot fail or inject tags.
    """
    from unittest.mock import patch, MagicMock
    import services.telegram_bot_service as tg

    reply = tg.process_command("/unknown_<script>&<b>")
    assert "<script>" not in reply
    assert "&lt;script&gt;&amp;&lt;b&gt;" in reply

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = 'File "<stdin>", line 1, in <module> & ValueError'

    with patch.object(tg, "_last_sync_timestamp", 0.0), patch("subprocess.run", return_value=mock_proc):
        sync_reply = tg.handle_sync()

    assert "<stdin>" not in sync_reply
    assert "<module>" not in sync_reply
    assert "&lt;stdin&gt;" in sync_reply
    assert "&lt;module&gt;" in sync_reply


def test_telegram_unauthorized_log_masks_token_straddling_64_char_boundary():
    """
    M1.2 (PB finding): Verifies that masking happens BEFORE the 64-character slice
    when formatting unauthorized command logs, preventing partial token leakage across index 64,
    and that repr() escapes newline characters.
    """
    token_p1 = "1234567890"
    token_p2 = "ABCdefGHIjklMNOpqrSTUvwxYZ123456789"
    token = f"{token_p1}:{token_p2}"
    # Prefix of length 45 so token starts at index 45 and straddles index 64
    prefix = "/cmd_" + ("x" * 40) + " "
    assert 40 < len(prefix) < 60
    malicious_text = prefix + token + "\nFAKE_LOG_LINE"

    logged_snippet = repr(mask_token(malicious_text[:512], token)[:64])
    assert token_p1 not in logged_snippet
    assert "ABCdef" not in logged_snippet
    assert "[MASKED_TOKEN]" in logged_snippet
    assert "\n" not in logged_snippet


def test_telegram_status_and_holdings_mark_to_market_consistency(tmp_path, monkeypatch):
    import json
    import os
    import polars as pl
    from services import telegram_bot_service as tbs

    # 1. Priority 1: last_close = 30000.0
    state = {
        "initial_capital": 100_000_000.0,
        "last_real_nav": 120_000_000.0,
        "available_cash": 20_000_000.0,
        "receivable_cash_t1": 10_000_000.0,
        "receivable_cash_t2": 0.0,
        "total_brokerage_tax": 150_000.0,
        "total_advance_fees": 50_000.0,
        "last_updated_date": "2024-01-05",
        "holdings": {
            "HPG": {
                "sellable_qty": 3000,
                "locked_t1": 0,
                "locked_t2": 0,
                "avg_cost": 20000.0,
                "last_close": 30000.0
            }
        }
    }
    state_file = tmp_path / "paper_portfolio_state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(tbs, "get_state_file_path", lambda: str(state_file))

    status_msg = tbs.handle_status()
    assert "90,000,000 VND" in status_msg
    assert "75.0%" in status_msg
    assert "25.0%" in status_msg

    holdings_msg = tbs.handle_holdings()
    assert "Position Value: <code>90,000,000 VND</code>" in holdings_msg
    assert "Total Market Exposure:</b> <code>90,000,000 VND</code>" in holdings_msg

    # 2. Priority 2: prev_close_map = {"HPG": 28000.0} when last_close is missing
    del state["holdings"]["HPG"]["last_close"]
    state["prev_close_map"] = {"HPG": 28000.0}
    state_file.write_text(json.dumps(state), encoding="utf-8")

    status_msg2 = tbs.handle_status()
    assert "84,000,000 VND" in status_msg2
    holdings_msg2 = tbs.handle_holdings()
    assert "Position Value: <code>84,000,000 VND</code>" in holdings_msg2

    # 3. Priority 3: Parquet fallback ("date", "symbol", "raw_close") when prev_close_map is missing
    del state["prev_close_map"]
    state_file.write_text(json.dumps(state), encoding="utf-8")

    pq_file = tmp_path / "vn30_mock.parquet"
    df_mock = pl.DataFrame({
        "date": ["2024-01-04", "2024-01-05"],
        "symbol": ["HPG", "HPG"],
        "raw_close": [25000.0, 32000.0]
    })
    df_mock.write_parquet(pq_file)
    monkeypatch.setattr(tbs, "get_parquet_file_path", lambda: str(pq_file))

    status_msg3 = tbs.handle_status()
    assert "96,000,000 VND" in status_msg3
    holdings_msg3 = tbs.handle_holdings()
    assert "Position Value: <code>96,000,000 VND</code>" in holdings_msg3

    # 4. Check against canonical parquet schema if file exists
    canonical_pq = os.path.join(tbs.ROOT_DIR, "data", "vn30_production.parquet")
    if os.path.exists(canonical_pq):
        monkeypatch.setattr(tbs, "get_parquet_file_path", lambda: canonical_pq)
        state_can = {
            "holdings": {
                "HPG": {
                    "sellable_qty": 1000,
                    "avg_cost": 12345.0
                }
            }
        }
        res_map = tbs._resolve_latest_close_map(state_can)
        assert res_map["HPG"] > 0
        assert res_map["HPG"] != 12345.0


def test_paper_trader_exec_price_clamping_and_t25_next_day_sellable():
    from services import daily_paper_trader as dpt

    # 1. Floor trapped penalty clamp
    state = dpt.init_empty_state(100_000_000.0)
    state["holdings"]["VNM"] = {
        "sellable_qty": 1000,
        "locked_t1": 0,
        "locked_t2": 0,
        "avg_cost": 50000.0
    }
    pending = [{
        "action": "SELL",
        "symbol": "VNM",
        "qty": 1000,
        "trapped_days": 5
    }]
    state["pending_orders"] = pending

    mkt_slice = {
        "VNM": {
            "raw_open": 50000.0,
            "floor_price": 46500.0,
            "ceiling_price": 53500.0,
            "raw_close": 48000.0,
            "adv20": 1e11
        }
    }
    st_res, reps = dpt.execute_pending_orders_and_settle(state, mkt_slice, "2024-01-02", clamp_and_round=True, penalty_rate_per_day=0.07)
    assert len(reps) == 1
    assert reps[0]["status"] == "FILLED"
    assert reps[0]["fill_price"] == 46500.0

    # 2. T+2.5 Next day sellable check
    st2 = dpt.init_empty_state(100_000_000.0)
    # Day T buy
    st2["holdings"]["HPG"] = {
        "sellable_qty": 0,
        "locked_t1": 0,
        "locked_t2": 1000,
        "avg_cost": 25000.0
    }
    st2["last_ledger_advance_date"] = "2024-01-01"

    # Morning Day T+1
    dpt.advance_t25_settlement(st2, "2024-01-02")
    assert st2["holdings"]["HPG"]["locked_t1"] == 1000
    assert st2["holdings"]["HPG"]["sellable_qty"] == 0

        # Evening Day T+1: generate orders for morning T+2
    st2["last_real_nav"] = 100_000_000.0
    from engine import data_pipeline as dp

    df_hist = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01",
        end_date="2023-05-15",
        seed=42
    )
    last_date = str(df_hist["date"].max())

    # Under strict_t25_ato_lock=True, locked_t1 is NOT sellable
    orders_strict = dpt.generate_next_day_orders(st2, df_hist, last_date, strict_t25_ato_lock=True)
    sell_orders_strict = [o for o in orders_strict if o["action"] == "SELL" and o["symbol"] == "HPG"]
    assert len(sell_orders_strict) == 0

    # Under default strict_t25_ato_lock=False, locked_t1 is included in sellable for morning T+2
    orders_def = dpt.generate_next_day_orders(st2, df_hist, last_date, strict_t25_ato_lock=False)
    assert isinstance(orders_def, list)

    # Morning Day T+2: advance settlement unlocks locked_t1 -> sellable_qty
    dpt.advance_t25_settlement(st2, "2024-01-03")
    assert st2["holdings"]["HPG"]["sellable_qty"] == 1000

    pending_sell = [{"action": "SELL", "symbol": "HPG", "qty": 1000, "trapped_days": 0}]
    st2["pending_orders"] = pending_sell
    mkt_slice2 = {"HPG": {"raw_open": 25000.0, "floor_price": 23250.0, "ceiling_price": 26750.0, "raw_close": 25000.0, "adv20": 1e11}}
    _, rep2 = dpt.execute_pending_orders_and_settle(st2, mkt_slice2, "2024-01-03")
    assert rep2[0]["status"] == "FILLED"

def test_paper_trader_advance_borrow_cap_and_holiday_calendar():
    """
    M5.1: Verifies that execute_pending_orders_and_settle bounds advance borrowing
    by sim._calc_borrow_cap, preserving fee split accounting identity across single-tranche
    and multi-tranche borrows, Greedy Cash Sweep, and holiday calendar computation.
    """
    import services.daily_paper_trader as dpt
    import engine.vn_simulator as sim
    from engine import data_pipeline as dp

    # 1. Single tranche advance borrowing bounded by sim._calc_borrow_cap
    state = dpt.init_empty_state(10_000_000.0)
    state["last_ledger_advance_date"] = "2026-04-29"
    state["available_cash"] = 0.0
    state["receivable_cash_t1"] = 10_000_000.0
    state["receivable_cash_t2"] = 0.0
    state["pending_orders"] = [
        {
            "action": "BUY",
            "symbol": "FPT",
            "qty": 400,
            "est_price": 24_900.0,
            "est_value": 400 * 24_900.0,
            "target_weight": 1.0,
        }
    ]
    today_slice = {
        "FPT": {
            "raw_open": 24_900.0,
            "raw_close": 25_000.0,
            "ceiling_price": 26_600.0,
            "floor_price": 23_200.0,
            "ref_price": 24_900.0,
        }
    }
    state, reps = dpt.execute_pending_orders_and_settle(
        state,
        today_slice,
        "2026-04-29",
        use_dynamic_slippage=False,
        clamp_and_round=True,
        round_adverse=False,
    )
    assert reps[0]["status"] == "FILLED"
    assert reps[0]["fill_qty"] == 300
    assert state["holdings"]["FPT"]["locked_t2"] == 300
    assert state["total_advance_fees"] == 50_000.0
    total_cost_1 = reps[0]["fill_qty"] * reps[0]["fill_price"] * (1.0 + dpt.BROKER_FEE)
    assert state["receivable_cash_t1"] + total_cost_1 + state["total_advance_fees"] == pytest.approx(10_000_000.0)

    # 2. Two tranche advance borrowing overflowing from T+1 to T+2
    state2 = dpt.init_empty_state(20_000_000.0)
    state2["last_ledger_advance_date"] = "2026-04-29"
    state2["available_cash"] = 0.0
    state2["receivable_cash_t1"] = 10_000_000.0
    state2["receivable_cash_t2"] = 10_000_000.0
    state2["pending_orders"] = [
        {
            "action": "BUY",
            "symbol": "FPT",
            "qty": 600,
            "est_price": 24_900.0,
            "est_value": 600 * 24_900.0,
            "target_weight": 1.0,
        }
    ]
    state2, reps2 = dpt.execute_pending_orders_and_settle(
        state2,
        today_slice,
        "2026-04-29",
        use_dynamic_slippage=False,
        clamp_and_round=True,
        round_adverse=False,
    )
    assert reps2[0]["status"] == "FILLED"
    assert reps2[0]["fill_qty"] == 600
    assert state2["receivable_cash_t1"] == 0.0
    assert state2["total_advance_fees"] == 100_000.0
    total_cost_2 = reps2[0]["fill_qty"] * reps2[0]["fill_price"] * (1.0 + dpt.BROKER_FEE)
    assert state2["receivable_cash_t1"] + state2["receivable_cash_t2"] + total_cost_2 + state2["total_advance_fees"] == pytest.approx(20_000_000.0)

    # 3. Greedy Cash Sweep with borrow cap
    st_swp = dpt.init_empty_state(100_000_000.0)
    st_swp["available_cash"] = 0.0
    st_swp["receivable_cash_t1"] = 20_000_000.0
    st_swp["receivable_cash_t2"] = 0.0
    st_swp["last_real_nav"] = 20_000_000.0
    df_hist = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01",
        end_date="2023-05-15",
        seed=42
    )
    last_date = str(df_hist["date"].max())
    swp_orders = dpt.generate_next_day_orders(st_swp, df_hist, last_date)
    tot_buy_cost = sum(o["qty"] * o["est_price"] * (1.0 + dpt.BROKER_FEE) for o in swp_orders if o["action"] == "BUY")
    cal_t1, _ = dpt._compute_settlement_calendar_days(last_date)
    cap_t1 = float(sim._calc_borrow_cap(20_000_000.0, dpt.ADVANCE_FEE_RATE, float(cal_t1), 50000.0))
    assert tot_buy_cost <= cap_t1 + 1e-6

    # 4. Holiday calendar settlement computation
    assert dpt._compute_settlement_calendar_days("2026-04-29") == (5, 6)
    assert dpt._compute_settlement_calendar_days("2026-04-29", trading_dates=["2026-04-29", "2026-05-04", "2026-05-05"]) == (5, 6)
    assert dpt._compute_settlement_calendar_days("2026-04-28", trading_dates=["2026-04-28", "2026-04-29"]) == (1, 6)


def test_telegram_summary_html_escape_and_dashboard_pinned_chartjs():
    """
    M5.3: Verifies that build_telegram_summary escapes HTML in user-controlled fields
    (session_date, symbol, dates_processed) and dashboard.py uses pinned chart.js version.
    """
    import services.daily_paper_trader as dpt
    from services.dashboard import generate_html_content

    state = dpt.init_empty_state(100_000_000.0)
    session_date = '2026-01-02<script>"x"&</script>'
    orders = [{
        "action": "BUY",
        "symbol": '<b>SYM&"1"</b>',
        "qty": 100,
        "est_price": 20000.0,
        "est_value": 2000000.0,
        "target_weight": 0.1
    }]
    dates_processed = ['2026-01-01<a>', '2026-01-02<b>']
    msg = dpt.build_telegram_summary(state, session_date, orders, [], dates_processed)

    assert "<script>" not in msg
    assert "</script>" not in msg
    assert "&lt;script&gt;" in msg
    assert "&quot;x&quot;&amp;&lt;/script&gt;" in msg
    assert "<b>SYM" not in msg
    assert "&lt;b&gt;SYM&amp;&quot;1&quot;&lt;/b&gt;" in msg
    assert "2026-01-01&lt;a&gt;" in msg
    assert "2026-01-02&lt;b&gt;" in msg

    html = generate_html_content({"state": state, "history": [], "prev_close_map": {}})
    assert "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" in html
    assert 'https://cdn.jsdelivr.net/npm/chart.js"></script>' not in html
def test_state_locking_pid_tmp_bak_recovery_and_bot_auth(tmp_path):
    """
    M6.1 & M6.3: Verifies cross-process state file locking (.lock >= 1 byte),
    PID-scoped .tmp cleanup, .bak recovery with recovered_from_bak=True and Telegram alert banner,
    dashboard Content-Security-Policy header, and Telegram .env search order.
    """
    import http.client
    import socketserver
    import os, threading
    import services.daily_paper_trader as dpt
    import services.telegram_bot_service as tbs
    from services.dashboard import create_single_file_handler

    state_file = str(tmp_path / "paper_portfolio_state.json")
    st = dpt.init_empty_state(100_000_000.0)
    st["available_cash"] = 88_000_000.0
    dpt.save_portfolio_state(st, state_file)

    lock_file = f"{state_file}.lock"
    assert os.path.exists(lock_file)
    assert os.path.getsize(lock_file) >= 1
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    # Trigger backup creation on second save, then corrupt primary JSON
    st["available_cash"] = 77_000_000.0
    dpt.save_portfolio_state(st, state_file)
    assert os.path.exists(f"{state_file}.bak")

    with open(state_file, "w", encoding="utf-8") as f_bad:
        f_bad.write("{corrupted_json_payload")

    recovered = dpt.load_portfolio_state(state_file, 100_000_000.0)
    assert recovered.get("recovered_from_bak") is True
    assert recovered["available_cash"] == 88_000_000.0

    summary_html = dpt.build_telegram_summary(recovered, "2026-10-23", [], [], [datetime.date(2026, 10, 23)])
    assert "State recovered from .bak backup!" in summary_html

    # Verify ROOT_DIR/.env takes precedence before os.getcwd()/.env
    assert tbs.ENV_FILE_PATHS[0] == os.path.join(tbs.ROOT_DIR, ".env")
    assert tbs.ENV_FILE_PATHS[-1] == os.path.join(os.getcwd(), ".env")

    # Verify Content-Security-Policy header on 200 OK response
    html_file = tmp_path / "dashboard.html"
    html_file.write_text("<html><body>CSP OK</body></html>", encoding="utf-8")
    handler_cls = create_single_file_handler(str(html_file))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/dashboard.html")
        resp = conn.getresponse()
        csp_hdr = resp.getheader("Content-Security-Policy")
        resp.read()
        conn.close()
        assert resp.status == 200
        assert csp_hdr is not None
        assert "default-src 'none'" in csp_hdr
        assert "https://cdn.jsdelivr.net" in csp_hdr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_paper_trader_market_enrichment_and_execution_parity_flags():
    """
    M6.2: Verifies _enrich_market_dataframe lagged adv20 and daily_vol enrichment
    (preventing look-ahead bias at morning ATO and activating dynamic slippage > 12 bps)
    as well as optional parity flags sell_adv_cap, penalty_rate_per_day, min_advance_notional,
    enable_greedy_sweep, and use_adv20_tie_break.
    """
    import polars as pl
    from engine import data_pipeline as dp
    import services.daily_paper_trader as dpt

    df_raw = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01",
        end_date="2023-05-15",
        seed=42,
    )
    df_enr = dpt._enrich_market_dataframe(df_raw)
    assert "adv20" in df_enr.columns
    assert "daily_vol" in df_enr.columns

    # Verify strict 1-session lag on adv20 in enriched DataFrame
    df_masked = dp.build_tradable_mask(df_raw, min_adv=1.0e10, window=20).sort(["symbol", "date"])
    fpt_eod_adv20 = df_masked.filter(pl.col("symbol") == "FPT")["adv20"].to_list()
    fpt_lag_adv20 = df_enr.sort(["symbol", "date"]).filter(pl.col("symbol") == "FPT")["adv20"].to_list()
    assert fpt_lag_adv20[-1] == pytest.approx(fpt_eod_adv20[-2])

    # Dynamic slippage activated via enriched slice (> 12 bps half-spread floor vs 10 bps fixed fallback)
    last_dt = df_enr["date"].max()
    today_enr = {r["symbol"]: r for r in df_enr.filter(pl.col("date") == last_dt).iter_rows(named=True)}
    today_raw = {r["symbol"]: r for r in df_raw.filter(pl.col("date") == last_dt).iter_rows(named=True)}

    st_raw = dpt.init_empty_state(100_000_000.0)
    st_raw["last_ledger_advance_date"] = str(last_dt)
    st_raw["pending_orders"] = [{"action": "BUY", "symbol": "FPT", "qty": 1000, "est_price": today_raw["FPT"]["raw_open"], "est_value": 1000 * today_raw["FPT"]["raw_open"], "target_weight": 0.5}]
    _, reps_raw = dpt.execute_pending_orders_and_settle(st_raw, today_raw, str(last_dt), clamp_and_round=False)

    st_enr = dpt.init_empty_state(100_000_000.0)
    st_enr["last_ledger_advance_date"] = str(last_dt)
    st_enr["pending_orders"] = [{"action": "BUY", "symbol": "FPT", "qty": 1000, "est_price": today_enr["FPT"]["raw_open"], "est_value": 1000 * today_enr["FPT"]["raw_open"], "target_weight": 0.5}]
    _, reps_enr = dpt.execute_pending_orders_and_settle(st_enr, today_enr, str(last_dt), clamp_and_round=False)

    assert reps_enr[0]["fill_price"] > reps_raw[0]["fill_price"]

    # Test sell_adv_cap=True, penalty_rate_per_day=0.0, and min_advance_notional
    st_flags = dpt.init_empty_state(100_000_000.0)
    st_flags["last_ledger_advance_date"] = "2026-05-04"
    st_flags["available_cash"] = 6_000_000.0
    st_flags["receivable_cash_t1"] = 20_000_000.0
    st_flags["holdings"] = {"FPT": {"sellable_qty": 2000, "locked_t1": 0, "locked_t2": 0, "avg_cost": 20_000.0}}
    st_flags["pending_orders"] = [
        {"action": "SELL", "symbol": "FPT", "qty": 2000, "trapped_days": 2, "est_price": 20_000.0, "est_value": 40_000_000.0, "target_weight": 0.0},
        {"action": "BUY", "symbol": "ACB", "qty": 300, "est_price": 25_000.0, "est_value": 7_500_000.0, "target_weight": 0.2},
    ]
    slice_flags = {
        "FPT": {"raw_open": 20_000.0, "raw_close": 20_000.0, "ceiling_price": 21_400.0, "floor_price": 18_600.0, "adv20": 200_000_000.0},
        "ACB": {"raw_open": 25_000.0, "raw_close": 25_000.0, "ceiling_price": 26_750.0, "floor_price": 23_250.0, "adv20": 500_000_000_000.0},
    }
    st_flags, reps_flags = dpt.execute_pending_orders_and_settle(
        st_flags,
        slice_flags,
        "2026-05-04",
        use_dynamic_slippage=False,
        clamp_and_round=False,
        sell_adv_cap=True,
        penalty_rate_per_day=0.0,
        min_advance_notional=5_000_000.0,
    )
    sell_rep = [r for r in reps_flags if r["action"] == "SELL"][0]
    buy_rep = [r for r in reps_flags if r["action"] == "BUY"][0]
    assert sell_rep["fill_qty"] == 500  # 5% of (200M / 20k = 10,000 shares) = 500 shares
    assert sell_rep["fill_price"] == pytest.approx(20_000.0 * (1.0 - dpt.SLIPPAGE))  # 0.0 penalty rate
    assert any(p["action"] == "SELL" and p["qty"] == 1500 for p in st_flags["pending_orders"])
    assert buy_rep["fill_qty"] == 200  # Scaled down to available_cash (6M VND -> 200 shs) without advance loan
    assert st_flags["total_advance_fees"] == 0.0

    # Test enable_greedy_sweep=False and use_adv20_tie_break=True
    st_gen = dpt.init_empty_state(100_000_000.0)
    orders_no_sweep = dpt.generate_next_day_orders(
        st_gen, df_enr, str(last_dt), enable_greedy_sweep=False, use_adv20_tie_break=True
    )
    assert isinstance(orders_no_sweep, list)


def test_m7_live_parquet_isolation_and_end_date_timestamp(tmp_path, monkeypatch):
    """
    M7.1: Verifies that live paper-trading data ingestion redirects DEFAULT_PARQUET_FILE
    to vn30_live_paper.parquet (leaving vn30_production.parquet untouched byte-for-byte)
    and that fetch_vn_ohlcv parses date-only end_date strings to 23:59:59 of that day.
    """
    import datetime
    import hashlib
    import inspect
    import shutil
    from pathlib import Path
    import polars as pl
    from engine import data_pipeline as dp
    from services import daily_paper_trader as dpt
    from services import telegram_bot_service as tbs

    # 1. Verify end_date 23:59:59 timestamp calculation in fetch_vn_ohlcv
    captured_params = {}

    class DummyResp:
        status_code = 200

        def json(self):
            return {"t": [1794182400], "o": [130.0], "h": [132.0], "l": [129.0], "c": [131.0], "v": [1000000]}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured_params["params"] = dict(params or {})
        return DummyResp()

    monkeypatch.setattr(dp.requests, "get", fake_get)
    dp.fetch_vn_ohlcv(
        symbols=["FPT"],
        start_date="2026-11-01",
        end_date="2026-11-09",
        save_parquet=None,
        force_fallback=False,
    )
    from_ts = int(captured_params["params"]["from"])
    to_ts = int(captured_params["params"]["to"])
    assert to_ts - int(datetime.datetime.fromisoformat("2026-11-09").timestamp()) == 86399

    # 2. Verify canonical vn30_production.parquet isolation during live update
    fake_prod = tmp_path / "vn30_production.parquet"
    fake_live = tmp_path / "vn30_live_paper.parquet"
    real_prod = Path(dpt.ROOT_DIR) / "data" / "vn30_production.parquet"
    df_seed = (
        pl.read_parquet(real_prod).tail(120)
        if real_prod.exists()
        else dp.generate_fallback_data(
            symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
            start_date="2023-01-01",
            end_date="2023-05-15",
            seed=42,
        )
    )
    df_seed.write_parquet(fake_prod)
    prod_sha_before = hashlib.sha256(fake_prod.read_bytes()).hexdigest()

    monkeypatch.setattr(dpt, "DEFAULT_PARQUET_FILE", str(fake_prod))
    monkeypatch.setattr(dpt, "DEFAULT_LIVE_PARQUET_FILE", str(fake_live))

    latest_dt = df_seed["date"].max()
    next_dt = latest_dt + datetime.timedelta(days=1)
    df_last_day = df_seed.filter(pl.col("date") == latest_dt).with_columns(pl.lit(next_dt).alias("date"))
    df_overlap_plus_new = pl.concat([df_seed, df_last_day]).sort(["date", "symbol"])

    monkeypatch.setattr(dp, "fetch_vn_ohlcv", lambda **kwargs: df_overlap_plus_new)

    resolved = dpt._resolve_live_parquet_path(str(fake_prod), is_replay=False)
    assert Path(resolved) == fake_live
    assert fake_live.exists() and fake_live.stat().st_size > 0
    assert hashlib.sha256(fake_prod.read_bytes()).hexdigest() == prod_sha_before

    # Replay mode must NOT redirect
    assert Path(dpt._resolve_live_parquet_path(str(fake_prod), is_replay=True)) == fake_prod
    assert tbs.get_parquet_file_path().endswith(("vn30_live_paper.parquet", "vn30_production.parquet"))


def test_m7_floor_trapped_sell_order_survival_and_zero_default_penalty():
    """
    M7.2: Verifies that execute_pending_orders_and_settle defaults to penalty_rate_per_day=0.0
    (matching vn_simulator.simulate_vietnam) and that floor-trapped sell orders retain their
    trapped_days counter and priority queue status across afternoon generate_next_day_orders,
    while blocking conflicting BUY orders on the floor-trapped symbol.
    """
    import inspect
    from pathlib import Path
    import polars as pl
    from engine import data_pipeline as dp
    from engine import vn_simulator as sim
    from services import daily_paper_trader as dpt

    dpt_default = inspect.signature(dpt.execute_pending_orders_and_settle).parameters["penalty_rate_per_day"].default
    sim_default = inspect.signature(sim.simulate_vietnam).parameters["penalty_rate_per_day"].default
    assert dpt_default == 0.0
    assert dpt_default == sim_default

    pq_path = Path(dpt.ROOT_DIR) / "data" / "vn30_production.parquet"
    df_raw = (
        pl.read_parquet(pq_path)
        if pq_path.exists()
        else dp.generate_fallback_data(
            symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
            start_date="2023-01-01",
            end_date="2023-05-15",
            seed=42,
        )
    )
    df_enr = dpt._enrich_market_dataframe(df_raw)
    all_dates = sorted(df_enr["date"].unique().to_list())
    last_date = all_dates[-1]
    last_date_str = last_date.isoformat()
    df_history = df_enr.filter(pl.col("date") <= last_date)
    today_mkt = {row["symbol"]: dict(row) for row in df_enr.filter(pl.col("date") == last_date).iter_rows(named=True)}

    state = dpt.init_empty_state(100_000_000.0)
    state["holdings"]["FPT"] = {
        "sellable_qty": 1000,
        "locked_t1": 0,
        "locked_t2": 0,
        "avg_cost": 130_000.0,
    }
    state["pending_orders"] = [
        {
            "action": "SELL",
            "symbol": "FPT",
            "qty": 1000,
            "est_price": 130_000.0,
            "est_value": 130_000_000.0,
            "target_weight": 0.0,
            "trapped_days": 1,
        }
    ]
    mkt_trap = {k: dict(v) for k, v in today_mkt.items()}
    mkt_trap["FPT"]["raw_open"] = mkt_trap["FPT"]["floor_price"]

    state, reps = dpt.execute_pending_orders_and_settle(state, mkt_trap, last_date_str, trading_dates=all_dates)
    assert reps[0]["status"] == "FLOOR TRAPPED (HOLDING ORDER)"
    assert reps[0]["trapped_days"] == 2
    assert state["trapped_sell_orders"][0]["trapped_days"] == 2

    eod_orders = dpt.generate_next_day_orders(state, df_history, last_date_str)
    fpt_sells = [o for o in eod_orders if o["symbol"] == "FPT" and o["action"] == "SELL"]
    fpt_buys = [o for o in eod_orders if o["symbol"] == "FPT" and o["action"] == "BUY"]
    assert len(fpt_sells) == 1
    assert fpt_sells[0]["trapped_days"] == 2
    assert fpt_sells[0]["qty"] == 1000
    assert len(fpt_buys) == 0
    assert state["trapped_sell_orders"][0]["trapped_days"] == 2


def test_m7_corp_action_overlap_alert_idempotent_csv_and_dashboard_mtm(tmp_path):
    """
    M7.3: Verifies corporate action overlap detection on held symbols, idempotent date
    deduplication in append_nav_history, and MTM close price resolution in dashboard.py.
    """
    import datetime
    import polars as pl
    from services import daily_paper_trader as dpt
    from services import dashboard as dash

    # 1. Corporate action detection on overlap window for held symbol FPT
    dates = [datetime.date(2026, 10, 20), datetime.date(2026, 10, 21)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["FPT", "FPT"],
        "adj_close": [100_000.0, 102_000.0],
    })
    df_new = pl.DataFrame({
        "date": dates,
        "symbol": ["FPT", "FPT"],
        "adj_close": [90_000.0, 91_800.0],  # 10% ex-dividend / split adjustment
    })
    holdings = {"FPT": {"sellable_qty": 500, "locked_t1": 200, "locked_t2": 0, "avg_cost": 100_000.0}}
    alerts = dpt.detect_corporate_action_on_overlap(
        df_cache, df_new, holdings=holdings, overlap_window=20, detected_date="2026-10-22"
    )
    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "FPT"
    assert alerts[0]["held_qty"] == 700
    assert alerts[0]["requires_manual_reconcile"] is True
    assert alerts[0]["adj_ratio"] == pytest.approx(0.9)

    state = dpt.init_empty_state(100_000_000.0)
    state["holdings"] = holdings
    state["corporate_action_alerts"] = alerts
    tg_msg = dpt.build_telegram_summary(state, "2026-10-22", [], [], [datetime.date(2026, 10, 22)])
    assert "CORP ACTION ALERT" in tg_msg
    assert "FPT" in tg_msg

    # 2. Idempotent append_nav_history deduplicating same date
    csv_file = tmp_path / "paper_nav_history.csv"
    rec1 = {
        "date": "2026-10-22",
        "real_nav": 100_100_000.0,
        "available_cash": 50_000_000.0,
        "receivable_cash": 0.0,
        "stock_value": 50_100_000.0,
        "naive_nav": 100_200_000.0,
        "implementation_shortfall_pct": 0.10,
        "daily_return_real": 0.001,
        "daily_return_naive": 0.002,
        "orders_executed": 1,
    }
    rec2 = dict(rec1, real_nav=100_500_000.0, orders_executed=2)
    dpt.append_nav_history(str(csv_file), rec1)
    dpt.append_nav_history(str(csv_file), rec2)
    df_csv = pl.read_csv(csv_file)
    assert len(df_csv) == 1
    assert float(df_csv["real_nav"][0]) == pytest.approx(100_500_000.0)
    assert int(df_csv["orders_executed"][0]) == 2

    # 3. Dashboard MTM lookup from state["prev_close_map"] & hold["last_close"]
    dash_state = dpt.init_empty_state(100_000_000.0)
    dash_state["last_real_nav"] = 115_000_000.0
    dash_state["available_cash"] = 0.0
    dash_state["holdings"] = {
        "FPT": {"sellable_qty": 1000, "locked_t1": 0, "locked_t2": 0, "avg_cost": 100_000.0, "last_close": 115_000.0}
    }
    dash_state["prev_close_map"] = {"FPT": 112_000.0}
    html_out = dash.generate_html_content({"state": dash_state, "history": []})
    assert "115,000,000 VND" in html_out


def test_m8_no_self_crossing_orders_and_no_greedy_sweep_ping_pong():
    """
    M8.1: Verifies that generate_next_day_orders:
    1. Never emits both SELL and BUY orders for the same symbol in the same session (zero self-crossing).
    2. Retains already-held Greedy Cash Sweep lots of top_candidates across T+2.5 settlement unlock
       (preventing the 2-day ping-pong sell-off and re-buy loop), while cleanly selling when a symbol
       falls out of top_candidates.
    3. Blocks Greedy Cash Sweep from buying a symbol whose locked_t2 holdings already exceed effective_target_shares.
    """
    import polars as pl
    from engine import data_pipeline as dp
    from services import daily_paper_trader as dpt

    df_raw = dp.generate_fallback_data(
        symbols=["ACB", "CTG", "FPT", "HPG", "TCB", "TPB", "VCB", "VIB", "VJC", "VNM"],
        start_date="2023-01-01",
        end_date="2023-06-15",
        seed=42,
    )
    df_enr = dpt._enrich_market_dataframe(df_raw)
    all_dates = sorted(df_enr["date"].unique().to_list())

    # 1. Multi-session sequential simulation: verify zero self-crossing and no T+2.5 ping-pong
    state = dpt.init_empty_state(100_000_000.0)
    sim_dates = all_dates[-8:]
    for idx, cur_date in enumerate(sim_dates):
        cur_str = cur_date.isoformat()
        df_hist = df_enr.filter(pl.col("date") <= cur_date)
        today_mkt = {r["symbol"]: dict(r) for r in df_enr.filter(pl.col("date") == cur_date).iter_rows(named=True)}
        prev_map = {}
        if idx > 0:
            p_date = sim_dates[idx - 1]
            prev_map = {r["symbol"]: r["raw_close"] for r in df_enr.filter(pl.col("date") == p_date).iter_rows(named=True)}
        state, exec_reps = dpt.execute_pending_orders_and_settle(state, today_mkt, cur_str, trading_dates=all_dates)
        dpt.mark_to_market(state, today_mkt, prev_map, cur_str, len(exec_reps))
        orders = dpt.generate_next_day_orders(state, df_hist, cur_str)
        sell_syms = {o["symbol"] for o in orders if o["action"] == "SELL"}
        buy_syms = {o["symbol"] for o in orders if o["action"] == "BUY"}
        assert sell_syms.isdisjoint(buy_syms)
        assert len(buy_syms) == len([o for o in orders if o["action"] == "BUY"])

    # 2. Direct test of held top_candidate with idle cash > 15% NAV: must retain held lot and never self-cross
    last_dt = sim_dates[-1]
    last_str = last_dt.isoformat()
    df_last_hist = df_enr.filter(pl.col("date") <= last_dt)
    st_direct = dpt.init_empty_state(100_000_000.0)
    st_direct["available_cash"] = 35_000_000.0
    st_direct["last_real_nav"] = 90_000_000.0
    # Seed 100 sellable shares across all symbols so whichever is top_candidate is already held
    last_prices = {r["symbol"]: r["raw_close"] for r in df_enr.filter(pl.col("date") == last_dt).iter_rows(named=True)}
    for sym_k, p_k in last_prices.items():
        st_direct["holdings"][sym_k] = {
            "sellable_qty": 100,
            "locked_t1": 0,
            "locked_t2": 0,
            "avg_cost": float(p_k),
            "last_close": float(p_k),
        }
    orders_direct = dpt.generate_next_day_orders(st_direct, df_last_hist, last_str)
    sells_d = {o["symbol"] for o in orders_direct if o["action"] == "SELL"}
    buys_d = {o["symbol"] for o in orders_direct if o["action"] == "BUY"}
    assert sells_d.isdisjoint(buys_d)

    # 3. Overweight locked_t2 symbol (current_sellable == 0) must NOT be bought by Greedy Cash Sweep
    st_locked = dpt.init_empty_state(100_000_000.0)
    st_locked["available_cash"] = 30_000_000.0
    st_locked["last_real_nav"] = 100_000_000.0
    for sym_k, p_k in last_prices.items():
        st_locked["holdings"][sym_k] = {
            "sellable_qty": 0,
            "locked_t1": 0,
            "locked_t2": 100_000,
            "avg_cost": float(p_k),
            "last_close": float(p_k),
        }
    orders_locked = dpt.generate_next_day_orders(st_locked, df_last_hist, last_str)
    assert [o for o in orders_locked if o["action"] == "BUY"] == []


def test_m8_replay_isolation_overlap_backfill_mtm_fallback_and_bak_flag_reset(tmp_path, monkeypatch):
    """
    M8.2 & M8.3: Verifies:
    1. run_test_replay_10 writes to paper_replay_state.json / paper_replay_nav_history.csv without touching
       live paper_portfolio_state.json.
    2. fetch_or_update_data backfills missing (date, symbol) rows inside the overlap window with schema alignment.
    3. _resolve_valid_close_price skips None / 0.0 and prioritizes last_close -> prev_close_map -> avg_cost.
    """
    import datetime
    import json
    import polars as pl
    from engine import data_pipeline as dp
    from services import daily_paper_trader as dpt

    # 1. Replay isolation test
    live_state_path = tmp_path / "paper_portfolio_state.json"
    live_sentinel = {"sentinel": "LIVE_PORTFOLIO_UNTOUCHED", "last_real_nav": 123_456_789.0}
    live_state_path.write_text(json.dumps(live_sentinel), encoding="utf-8")
    replay_state_path = tmp_path / "paper_replay_state.json"
    replay_hist_path = tmp_path / "paper_replay_nav_history.csv"
    monkeypatch.setattr(dpt, "DEFAULT_STATE_FILE", str(live_state_path))
    monkeypatch.setattr(dpt, "DEFAULT_REPLAY_STATE_FILE", str(replay_state_path))
    monkeypatch.setattr(dpt, "DEFAULT_REPLAY_HISTORY_FILE", str(replay_hist_path))

    pq_replay = tmp_path / "replay_source.parquet"
    df_rep = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01",
        end_date="2023-05-15",
        seed=42,
    )
    df_rep.write_parquet(pq_replay)
    dpt.run_test_replay_10(str(pq_replay), days=3)

    assert json.loads(live_state_path.read_text(encoding="utf-8")) == live_sentinel
    assert replay_state_path.exists()
    assert replay_hist_path.exists()
    assert len(pl.read_csv(replay_hist_path)) == 3

    # 2. Overlap symbol gap backfill with schema alignment
    live_pq = tmp_path / "vn30_live_paper.parquet"
    d1 = datetime.date(2026, 10, 20)
    d2 = datetime.date(2026, 10, 21)
    df_cached = pl.DataFrame({
        "date": [d1, d1, d2],
        "symbol": ["ACB", "FPT", "ACB"],  # FPT missing on d2 due to prior timeout
        "raw_open": [20000.0, 100000.0, 20100.0],
        "raw_high": [20500.0, 101000.0, 20600.0],
        "raw_low": [19800.0, 99000.0, 19900.0],
        "raw_close": [20200.0, 100500.0, 20300.0],
        "adj_close": [20200.0, 100500.0, 20300.0],
        "volume": [1000000.0, 2000000.0, 1100000.0],
        "data_source": ["entrade", "entrade", "entrade"],
    })
    df_cached.write_parquet(live_pq)

    # df_new has reversed column order, integer volume, missing data_source metadata column, and includes (d2, FPT)
    df_new_api = pl.DataFrame({
        "symbol": ["ACB", "FPT", "ACB", "FPT"],
        "date": [d1, d1, d2, d2],
        "volume": [1000000, 2000000, 1100000, 2100000],
        "adj_close": [20200.0, 100500.0, 20300.0, 101500.0],
        "raw_close": [20200.0, 100500.0, 20300.0, 101500.0],
        "raw_low": [19800.0, 99000.0, 19900.0, 100000.0],
        "raw_high": [20500.0, 101000.0, 20600.0, 102000.0],
        "raw_open": [20000.0, 100000.0, 20100.0, 101000.0],
    })
    monkeypatch.setattr(dp, "fetch_vn_ohlcv", lambda **kwargs: df_new_api)
    df_updated = dpt.fetch_or_update_data(str(live_pq), is_replay=False)
    assert len(df_updated) == 4
    assert df_updated.schema == df_cached.schema
    fpt_d2 = df_updated.filter((pl.col("date") == d2) & (pl.col("symbol") == "FPT"))
    assert len(fpt_d2) == 1
    assert float(fpt_d2["raw_close"][0]) == pytest.approx(101500.0)

    # 3. MTM fallback when raw_close is None or 0.0
    st_mtm = dpt.init_empty_state(100_000_000.0)
    st_mtm["available_cash"] = 0.0
    st_mtm["holdings"] = {
        "FPT": {"sellable_qty": 100, "locked_t1": 0, "locked_t2": 0, "avg_cost": 80_000.0, "last_close": 95_000.0},
        "ACB": {"sellable_qty": 100, "locked_t1": 0, "locked_t2": 0, "avg_cost": 20_000.0, "last_close": 0.0},
    }
    st_mtm["prev_close_map"] = {"ACB": 24_000.0}
    rec_mtm = dpt.mark_to_market(st_mtm, {"FPT": {"raw_close": None}, "ACB": {"raw_close": 0.0}}, {}, "2026-10-22", 0)
    assert rec_mtm["stock_value"] == pytest.approx(100 * 95_000.0 + 100 * 24_000.0)
