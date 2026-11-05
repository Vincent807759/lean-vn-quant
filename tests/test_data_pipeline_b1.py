"""
tests/test_data_pipeline_b1.py - Unit tests for B1 (Data pipeline integrity) & S1 invariant
"""
import os
import pytest
import polars as pl
from unittest.mock import patch, MagicMock
from engine.data_pipeline import (
    generate_fallback_data,
    fetch_vn_ohlcv,
    validate_real_data_source
)

def test_fallback_data_has_synthetic_data_source():
    df = generate_fallback_data(symbols=["HPG", "VCB"], start_date="2026-01-01", end_date="2026-01-10")
    assert "data_source" in df.columns
    assert (df["data_source"] == "synthetic").all()

def test_s1_synthetic_write_allowed_in_tests_tmp(tmp_path):
    # Allowed in tests/tmp
    test_tmp_path = os.path.join("tests", "tmp", "test_allowed.parquet")
    df = fetch_vn_ohlcv(
        symbols=["HPG"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        save_parquet=test_tmp_path,
        force_fallback=True
    )
    assert os.path.exists(test_tmp_path)
    loaded = pl.read_parquet(test_tmp_path)
    assert (loaded["data_source"] == "synthetic").all()
    # Cleanup
    if os.path.exists(test_tmp_path):
        os.remove(test_tmp_path)

def test_s1_synthetic_write_rejected_outside_tests_tmp():
    # Should reject saving synthetic to data/ or anywhere outside tests/tmp
    with pytest.raises(PermissionError) as excinfo:
        fetch_vn_ohlcv(
            symbols=["HPG"],
            start_date="2026-01-01",
            end_date="2026-01-05",
            save_parquet="data/vn30_production.parquet",
            force_fallback=True
        )
    assert "S1 Invariant Violated" in str(excinfo.value)

    with pytest.raises(PermissionError) as excinfo2:
        fetch_vn_ohlcv(
            symbols=["HPG"],
            start_date="2026-01-01",
            end_date="2026-01-05",
            save_parquet="custom_folder/output.parquet",
            force_fallback=True
        )
    assert "S1 Invariant Violated" in str(excinfo2.value)

def test_s1_validate_real_data_source_acceptance_and_rejection():
    # Acceptance
    real_df = pl.DataFrame({
        "date": ["2026-01-02"],
        "symbol": ["HPG"],
        "data_source": ["entrade"]
    })
    validate_real_data_source(real_df) # Should pass without exception

    # Rejection: synthetic data
    synth_df = pl.DataFrame({
        "date": ["2026-01-02"],
        "symbol": ["HPG"],
        "data_source": ["synthetic"]
    })
    with pytest.raises(ValueError) as excinfo:
        validate_real_data_source(synth_df)
    assert "Real backtest cannot read non-entrade data" in str(excinfo.value)

    # Rejection: missing column
    missing_df = pl.DataFrame({
        "date": ["2026-01-02"],
        "symbol": ["HPG"]
    })
    with pytest.raises(ValueError) as excinfo2:
        validate_real_data_source(missing_df)
    assert "missing 'data_source' column" in str(excinfo2.value)

def test_b1_api_failure_raises_runtime_error_no_silent_fallback():
    # Mock requests to fail (raising Exception or returning 500)
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection Timeout")
        with pytest.raises(RuntimeError) as excinfo:
            fetch_vn_ohlcv(symbols=["HPG", "VCB"], start_date="2026-01-01", end_date="2026-01-05")
        assert "Failed to fetch market data from Entrade API" in str(excinfo.value)


def test_query_parquet_duckdb_blocks_external_file_access(tmp_path):
    """
    M1.3: Verifies that query_parquet_duckdb queries the registered ohlcv_table
    while blocking external file reads (read_csv_auto) via enable_external_access=false.
    """
    import duckdb
    from engine.data_pipeline import query_parquet_duckdb

    parquet_path = tmp_path / "sample.parquet"
    csv_path = tmp_path / "secret.csv"
    pl.DataFrame({"symbol": ["HPG", "VCB"], "raw_close": [28000.0, 85000.0]}).write_parquet(parquet_path)
    csv_path.write_text("secret_col\nTOP_SECRET_VALUE\n", encoding="utf-8")

    res = query_parquet_duckdb(
        str(parquet_path),
        "SELECT symbol, raw_close FROM ohlcv_table WHERE symbol = 'HPG'",
    )
    assert res.height == 1
    assert res["symbol"][0] == "HPG"

    with pytest.raises(duckdb.PermissionException) as excinfo:
        query_parquet_duckdb(
            str(parquet_path),
            f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}')",
        )
    err_text = str(excinfo.value)
    assert "Permission Error" in err_text or "disabled through configuration" in err_text


def test_fetch_vn_ohlcv_passes_query_params_to_requests_get():
    """
    M1.3: Verifies that fetch_vn_ohlcv passes symbol/resolution/from/to via the
    requests.get params= dictionary rather than raw URL string concatenation.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "t": [1704153600],
        "o": [28.0],
        "h": [28.5],
        "l": [27.8],
        "c": [28.2],
        "v": [1000000.0],
    }
    with patch("requests.get", return_value=mock_resp) as mock_get:
        df = fetch_vn_ohlcv(
            symbols=["HPG"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            save_parquet=None,
            force_fallback=False,
        )
    assert df.height == 1
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
    assert "params" in kwargs
    assert kwargs["params"]["symbol"] == "HPG"
    assert kwargs["params"]["resolution"] == "1D"
    assert isinstance(kwargs["params"]["from"], int)
    assert isinstance(kwargs["params"]["to"], int)
