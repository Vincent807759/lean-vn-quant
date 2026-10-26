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
    fetch_or_update_data
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
