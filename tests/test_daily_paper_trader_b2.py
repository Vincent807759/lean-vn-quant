"""
tests/test_daily_paper_trader_b2.py - Unit tests for B2, S2, T3, T8
"""
import os
import hashlib
import datetime
import pytest
import polars as pl
from unittest.mock import patch
from services.daily_paper_trader import (
    check_cache_overlap_and_reconcile,
    fetch_or_update_data,
    load_portfolio_state,
    save_portfolio_state
)

def test_cache_overlap_consistent():
    # Cache has 5 days of HPG
    dates = [datetime.date(2026, 1, i) for i in range(1, 6)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG"] * 5,
        "adj_close": [25000.0, 25200.0, 25100.0, 25300.0, 25500.0]
    })
    
    # New fetch overlaps on dates 3, 4, 5 with identical prices + adds date 6
    dates_new = [datetime.date(2026, 1, i) for i in range(3, 7)]
    df_new = pl.DataFrame({
        "date": dates_new,
        "symbol": ["HPG"] * 4,
        "adj_close": [25100.0, 25300.0, 25500.0, 25800.0]
    })
    
    # Must be valid
    is_valid = check_cache_overlap_and_reconcile(df_cache, df_new, overlap_window=20, tolerance=1e-6)
    assert is_valid is True

def test_cache_overlap_detects_small_cash_dividend():
    # Cache has HPG at 25,000 VND
    dates = [datetime.date(2026, 1, i) for i in range(1, 6)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG"] * 5,
        "adj_close": [25000.0, 25200.0, 25100.0, 25300.0, 25500.0]
    })
    
    # A small cash dividend of 100 VND (0.4%) occurred.
    # New API adjusted past prices by (25000 - 100)/25000 = 0.996
    factor = 24900.0 / 25000.0 # 0.996
    dates_new = [datetime.date(2026, 1, i) for i in range(3, 7)]
    df_new = pl.DataFrame({
        "date": dates_new,
        "symbol": ["HPG"] * 4,
        "adj_close": [25100.0 * factor, 25300.0 * factor, 25500.0 * factor, 25800.0]
    })
    
    # With 1e-6 tolerance, this 0.4% discrepancy must be detected as an invalid cache!
    is_valid = check_cache_overlap_and_reconcile(df_cache, df_new, overlap_window=20, tolerance=1e-6)
    assert is_valid is False

def test_cache_overlap_tolerates_ieee754_float_noise():
    # Cache and new differ only by 1e-9 (sub-micro floating noise)
    dates = [datetime.date(2026, 1, i) for i in range(1, 4)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG"] * 3,
        "adj_close": [25000.0, 25200.0, 25100.0]
    })
    df_new = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG"] * 3,
        "adj_close": [25000.00000001, 25200.00000001, 25100.00000001]
    })
    # Within 1e-6 tolerance, this is valid
    is_valid = check_cache_overlap_and_reconcile(df_cache, df_new, overlap_window=20, tolerance=1e-6)
    assert is_valid is True

def test_fetch_or_update_api_failure_preserves_cache_intact(tmp_path):
    parquet_path = os.path.join(tmp_path, "cache.parquet")
    dates = [datetime.date(2026, 1, i) for i in range(1, 4)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG"] * 3,
        "adj_close": [25000.0, 25200.0, 25100.0],
        "data_source": ["entrade"] * 3
    })
    df_cache.write_parquet(parquet_path)
    
    with open(parquet_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
        
    # Simulate API failure during update
    with patch("engine.data_pipeline.fetch_vn_ohlcv", side_effect=RuntimeError("API Outage")):
        df_result = fetch_or_update_data(parquet_path, is_replay=False)
        
    with open(parquet_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
        
    assert sha_before == sha_after
    assert len(df_result) == len(df_cache)

def test_b2_drop_incomplete_bar_before_1505_ict(tmp_path):
    parquet_path = os.path.join(tmp_path, "cache.parquet")
    dates = [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG", "HPG"],
        "adj_close": [25000.0, 25200.0],
        "data_source": ["entrade", "entrade"]
    })
    df_cache.write_parquet(parquet_path)

    today = datetime.date(2026, 1, 3)
    vn_tz = datetime.timezone(datetime.timedelta(hours=7))
    mock_now = datetime.datetime(2026, 1, 3, 10, 30, 0, tzinfo=vn_tz)  # 10:30 ICT < 15:05

    # API returns intraday unfinished candle for today
    df_api = pl.DataFrame({
        "date": [datetime.date(2026, 1, 2), today],
        "symbol": ["HPG", "HPG"],
        "adj_close": [25200.0, 25800.0],
        "data_source": ["entrade", "entrade"]
    })

    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        mock_dt.time = datetime.time
        mock_dt.date = datetime.date
        with patch("engine.data_pipeline.fetch_vn_ohlcv", return_value=df_api):
            res = fetch_or_update_data(parquet_path, is_replay=False)

    # Incomplete bar of today must have been dropped; cache still ends on 2026-01-02
    assert res["date"].max() == datetime.date(2026, 1, 2)

def test_b2_corporate_action_invalidates_calibration_and_reloads_history(tmp_path):
    parquet_path = os.path.join(tmp_path, "cache.parquet")
    calib_path = os.path.join(tmp_path, "calibration.parquet")
    
    dates = [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]
    df_cache = pl.DataFrame({
        "date": dates,
        "symbol": ["HPG", "HPG"],
        "adj_close": [25000.0, 25200.0],
        "data_source": ["entrade", "entrade"]
    })
    df_cache.write_parquet(parquet_path)

    # Create dummy calibration file
    df_calib = pl.DataFrame({"symbol": ["HPG"], "f_factor": [1.0]})
    df_calib.write_parquet(calib_path)
    assert os.path.exists(calib_path)

    today = datetime.date(2026, 1, 3)
    vn_tz = datetime.timezone(datetime.timedelta(hours=7))
    mock_now = datetime.datetime(2026, 1, 3, 15, 30, 0, tzinfo=vn_tz)  # 15:30 ICT > 15:05

    # API returns adjusted historical prices (dividend happened)
    # 25200 * 0.95 = 23940 (mismatch > 1e-6)
    df_api_overlap = pl.DataFrame({
        "date": [datetime.date(2026, 1, 2), today],
        "symbol": ["HPG", "HPG"],
        "adj_close": [23940.0, 24000.0],
        "data_source": ["entrade", "entrade"]
    })
    df_api_full = pl.DataFrame({
        "date": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2), today],
        "symbol": ["HPG", "HPG", "HPG"],
        "adj_close": [23750.0, 23940.0, 24000.0],
        "data_source": ["entrade", "entrade", "entrade"]
    })

    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        mock_dt.time = datetime.time
        mock_dt.date = datetime.date
        with patch("engine.data_pipeline.fetch_vn_ohlcv", side_effect=[df_api_overlap, df_api_full]):
            res = fetch_or_update_data(parquet_path, is_replay=False, calibration_file=calib_path)

    # Calibration file must have been invalidated (removed)
    assert not os.path.exists(calib_path)
    # Parquet must have been refreshed with full reloaded history
    assert len(res) == 3
    assert res.filter(pl.col("date") == datetime.date(2026, 1, 2))["adj_close"][0] == 23940.0


def test_b13_atomic_save_and_backup_recovery(tmp_path):
    """
    B13: Verify atomic save creates backup .bak and recovers successfully if main file corrupts.
    """
    state_file = str(tmp_path / "paper_portfolio_state.json")
    state1 = {"real_nav": 105000000.0, "available_cash": 50000000.0}
    save_portfolio_state(state1, state_file)

    # Second save should create .bak containing state1
    state2 = {"real_nav": 110000000.0, "available_cash": 60000000.0}
    save_portfolio_state(state2, state_file)

    assert os.path.exists(f"{state_file}.bak")

    # Corrupt main file
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("{invalid_json: true,")

    # Loading should recover state1 from .bak rather than resetting to empty state
    recovered = load_portfolio_state(state_file)
    assert recovered["real_nav"] == 105000000.0


def test_b13_corrupted_state_without_backup_raises_runtime_error(tmp_path):
    """
    B13: Verify that if state is corrupted and no backup exists, it raises RuntimeError
    instead of silently resetting account to 100M VND.
    """
    state_file = str(tmp_path / "corrupted_state.json")
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("corrupted garbage content")

    with pytest.raises(RuntimeError) as exc_info:
        load_portfolio_state(state_file)
    assert "B13 FATAL" in str(exc_info.value)


