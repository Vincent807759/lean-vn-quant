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
    st_res, reps = dpt.execute_pending_orders_and_settle(state, mkt_slice, "2024-01-02", clamp_and_round=True)
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
