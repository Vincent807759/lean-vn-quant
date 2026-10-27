import os
import pytest
import requests
from unittest.mock import MagicMock
from services.daily_paper_trader import send_telegram_alert


def test_send_telegram_alert_connection_error_masks_token(monkeypatch, capsys):
    """
    F11a Security Test 1:
    When a network ConnectionError occurs containing the Telegram API URL with raw bot token,
    the raw bot token and secret payload must NOT be leaked to stdout or stderr.
    """
    fake_token = "123456:ABC-FAKE-TOKEN"
    fake_chat_id = "987654321"

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", fake_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", fake_chat_id)

    def mock_post_raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='api.telegram.org', port=443): "
            f"Max retries exceeded with url: /bot{fake_token}/sendMessage "
            f"(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object>: Failed to establish a new connection'))"
        )

    monkeypatch.setattr(requests, "post", mock_post_raise)

    success = send_telegram_alert("Daily Portfolio Report Test")
    assert success is False

    captured = capsys.readouterr()
    all_output = captured.out + captured.err

    # Verify no leak
    assert fake_token not in all_output, "SECURITY LEAK: Full bot token found in output!"
    assert "ABC-FAKE-TOKEN" not in all_output, "SECURITY LEAK: Token secret found in output!"

    # Verify error indicator remains visible for operational monitoring
    assert "[TELEGRAM CONNECTION ERROR]" in all_output


def test_send_telegram_alert_http_error_masks_token(monkeypatch, capsys):
    """
    F11a Security Test 2:
    When Telegram API returns an HTTP error body echoing the bot token,
    the token must be masked before printing to console logs.
    """
    fake_token = "123456:ABC-FAKE-TOKEN"
    fake_chat_id = "987654321"

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", fake_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", fake_chat_id)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = f'{{"ok":false,"error_code":401,"description":"Unauthorized: /bot{fake_token}/sendMessage"}}'

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    success = send_telegram_alert("Daily Portfolio Report Test")
    assert success is False

    captured = capsys.readouterr()
    all_output = captured.out + captured.err

    assert fake_token not in all_output, "SECURITY LEAK: Bot token echoed in HTTP error log!"
    assert "ABC-FAKE-TOKEN" not in all_output, "SECURITY LEAK: Token secret echoed in HTTP error log!"
    assert "[TELEGRAM ERROR] HTTP 401" in all_output


def test_send_telegram_alert_success_no_token_leak(monkeypatch, capsys):
    """
    F11a Security Test 3:
    Successful delivery logs confirmation without printing any token secrets.
    """
    fake_token = "123456:ABC-FAKE-TOKEN"
    fake_chat_id = "987654321"

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", fake_token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", fake_chat_id)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"ok":true,"result":{"message_id":1001}}'

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    success = send_telegram_alert("Daily Portfolio Report Test")
    assert success is True

    captured = capsys.readouterr()
    all_output = captured.out + captured.err

    assert fake_token not in all_output
    assert "ABC-FAKE-TOKEN" not in all_output
    assert "[TELEGRAM] Daily report successfully dispatched" in all_output
