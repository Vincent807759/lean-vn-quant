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
