import polars as pl
from engine.data_pipeline import data_provenance, fetch_vn_ohlcv


def test_data_provenance_entrade():
    df = pl.DataFrame({"data_source": ["entrade", "entrade"]})
    assert data_provenance(df) == "entrade"


def test_data_provenance_synthetic():
    df = fetch_vn_ohlcv(
        symbols=["HPG"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        save_parquet=None,
        force_fallback=True
    )
    assert data_provenance(df) == "synthetic"


def test_data_provenance_mixed():
    df1 = pl.DataFrame({"data_source": ["entrade", "entrade"]})
    df2 = fetch_vn_ohlcv(
        symbols=["HPG"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        save_parquet=None,
        force_fallback=True
    )
    df_mixed = pl.concat([df1, df2.select(["data_source"])])
    assert data_provenance(df_mixed) == "mixed"


def test_data_provenance_unknown():
    # DataFrame thiếu cột data_source
    df_no_col = pl.DataFrame({"symbol": ["HPG", "FPT"]})
    assert data_provenance(df_no_col) == "unknown"

    # DataFrame rỗng
    df_empty = pl.DataFrame({"data_source": []})
    assert data_provenance(df_empty) == "unknown"

    # Cột data_source = [None, None]
    df_none = pl.DataFrame({"data_source": [None, None]})
    assert data_provenance(df_none) == "unknown"

    # Cột data_source = ["unknown_vendor"]
    df_vendor = pl.DataFrame({"data_source": ["unknown_vendor"]})
    assert data_provenance(df_vendor) == "unknown"

    # Cột data_source = ["entrade", None]
    df_entrade_none = pl.DataFrame({"data_source": ["entrade", None]})
    assert data_provenance(df_entrade_none) == "unknown"
