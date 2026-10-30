import os
import datetime
import tempfile
import polars as pl
import pytest
from engine.data_pipeline import fetch_vn_ohlcv, _apply_universe_membership


def test_universe_membership_path_updates_is_listed_csv(tmp_path):
    """
    Verifies that providing universe_membership_path (CSV) updates 'is_listed'
    according to presence of (date, symbol) in the Point-in-Time membership file,
    and preserves is_listed=False for halted/suspended stocks.
    """
    # Create membership table where ACB and HALTED are present on 2023-01-03, but FPT is not.
    csv_file = str(tmp_path / "membership.csv")
    mem_df = pl.DataFrame({
        "date": ["2023-01-03", "2023-01-03"],
        "symbol": ["ACB", "HALTED"]
    })
    mem_df.write_csv(csv_file)

    df = fetch_vn_ohlcv(
        symbols=["ACB", "FPT"],
        start_date="2023-01-01",
        end_date="2023-01-05",
        save_parquet=None,
        force_fallback=True,
        universe_membership_path=csv_file
    )

    acb_listed = df.filter(
        (pl.col("date") == pl.lit("2023-01-03").str.to_date("%Y-%m-%d")) &
        (pl.col("symbol") == "ACB")
    )["is_listed"][0]

    fpt_listed = df.filter(
        (pl.col("date") == pl.lit("2023-01-03").str.to_date("%Y-%m-%d")) &
        (pl.col("symbol") == "FPT")
    )["is_listed"][0]

    assert acb_listed is True, "ACB should be listed on 2023-01-03 per membership table"
    assert fpt_listed is False, "FPT should NOT be listed on 2023-01-03 per membership table"

    # Assertion: if a row initially has is_listed == False (halted), it preserves False
    d = datetime.date(2023, 1, 3)
    halt_df = pl.DataFrame({
        "date": [d, d],
        "symbol": ["ACB", "HALTED"],
        "is_listed": [True, False]
    })
    updated_halt = _apply_universe_membership(halt_df, csv_file)
    assert updated_halt.filter(pl.col("symbol") == "HALTED")["is_listed"][0] is False, (
        "Halted stock must preserve is_listed=False even when present in membership table"
    )


def test_universe_membership_path_updates_is_listed_parquet(tmp_path):
    """
    Verifies that providing universe_membership_path (Parquet) updates 'is_listed'.
    """
    parquet_file = str(tmp_path / "membership.parquet")
    mem_df = pl.DataFrame({
        "date": ["2023-01-03", "2023-01-04"],
        "symbol": ["FPT", "FPT"]
    })
    mem_df.write_parquet(parquet_file)

    df = fetch_vn_ohlcv(
        symbols=["ACB", "FPT"],
        start_date="2023-01-01",
        end_date="2023-01-05",
        save_parquet=None,
        force_fallback=True,
        universe_membership_path=parquet_file
    )

    fpt_03 = df.filter(
        (pl.col("date") == pl.lit("2023-01-03").str.to_date("%Y-%m-%d")) &
        (pl.col("symbol") == "FPT")
    )["is_listed"][0]

    acb_03 = df.filter(
        (pl.col("date") == pl.lit("2023-01-03").str.to_date("%Y-%m-%d")) &
        (pl.col("symbol") == "ACB")
    )["is_listed"][0]

    assert fpt_03 is True
    assert acb_03 is False
