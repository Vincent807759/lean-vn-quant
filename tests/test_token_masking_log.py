import os
import pytest
import requests
from unittest.mock import MagicMock
from services.daily_paper_trader import send_telegram_alert, mask_token as daily_mask_token
from services.telegram_bot_service import mask_token, mask_chat_id


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


def test_mask_token_hides_prefix_and_bare_token_without_false_positive():
    # 1. Bare token without 'bot' prefix
    raw_bare = "Error connecting with 123456789:ABCdefGHI_123 in request."
    masked_bare = mask_token(raw_bare)
    assert "123456789:ABCdefGHI_123" not in masked_bare
    assert "[MASKED_TOKEN]" in masked_bare
    assert "1234" not in masked_bare

    # 2. Token with special characters / suffix when passing explicit token
    custom_token = "123456789:special!secret?value"
    msg = f"Failed to send to bot{custom_token} or token {custom_token}"
    masked_custom = mask_token(msg, token=custom_token)
    assert custom_token not in masked_custom
    assert f"bot{custom_token}" not in masked_custom
    assert "bot[MASKED_TOKEN]" in masked_custom
    assert "[MASKED_TOKEN]" in masked_custom

    # 3. Negative test: do not mask timestamps or large integer pairs
    timestamp_msg = "Session timestamp 20261103:180530 and coordinate 1000000:2000000 recorded."
    assert mask_token(timestamp_msg) == timestamp_msg


def test_daily_paper_trader_fallback_mask_token():
    # Test with bot prefix
    msg_with_bot = "Connection error to https://api.telegram.org/bot123456789:ABCdefGHI_123/sendMessage"
    masked1 = daily_mask_token(msg_with_bot)
    assert "123456789:ABCdefGHI_123" not in masked1
    assert "bot[MASKED_TOKEN]" in masked1

    # Test without bot prefix (bare token)
    msg_bare = "Unauthorized: 123456789:ABCdefGHI_123"
    masked2 = daily_mask_token(msg_bare)
    assert "123456789:ABCdefGHI_123" not in masked2
    assert "[MASKED_TOKEN]" in masked2


def test_mask_chat_id_hides_raw_id():
    raw_id = "987654321"
    masked = mask_chat_id(raw_id)
    assert masked == "[MASKED_CHAT_ID]"
    assert raw_id not in masked
    assert mask_chat_id("") == ""
    assert mask_chat_id(None) == ""

def test_telegram_bot_docstring_no_auto_discovery_claim():
    import services.telegram_bot_service as tbs
    doc = tbs.__doc__ or ""
    assert "Auto-Discovery" not in doc
    assert "Mandatory admin_chat_id authentication check on all incoming commands." in doc
