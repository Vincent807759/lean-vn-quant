import json
import datetime
import pytest
import numpy as np
import polars as pl
from engine.data_pipeline import (
    fetch_vn_ohlcv,
    dataset_fingerprint,
    manifest_status,
    FINGERPRINT_COLUMNS,
)


@pytest.fixture(scope="module")
def sample_df():
    return fetch_vn_ohlcv(
        symbols=["HPG", "FPT"],
        start_date="2026-01-01",
        end_date="2026-03-31",
        save_parquet=None,
        force_fallback=True,
    )


def test_dataset_fingerprint_deterministic(sample_df):
    fp1 = dataset_fingerprint(sample_df)
    fp2 = dataset_fingerprint(sample_df)
    assert fp1 == fp2
    assert fp1["rows"] == sample_df.height
    assert fp1["symbols"] == 2
    assert set(fp1["per_symbol_rows"].keys()) == {"HPG", "FPT"}


def test_dataset_fingerprint_row_order_invariant(sample_df):
    fp_orig = dataset_fingerprint(sample_df)
    df_shuffled = sample_df.sample(fraction=1.0, shuffle=True, seed=42)
    fp_shuffled = dataset_fingerprint(df_shuffled)
    assert fp_orig["content_sha256"] == fp_shuffled["content_sha256"]
    assert fp_orig["per_column_content_sha256"] == fp_shuffled["per_column_content_sha256"]
    assert fp_orig["per_symbol_content_sha256"] == fp_shuffled["per_symbol_content_sha256"]


def test_dataset_fingerprint_parquet_compression_invariant(sample_df, tmp_path):
    p1 = tmp_path / "zstd.parquet"
    p2 = tmp_path / "uncompressed.parquet"
    sample_df.write_parquet(p1, compression="zstd")
    sample_df.write_parquet(p2, compression="uncompressed")
    df1 = pl.read_parquet(p1)
    df2 = pl.read_parquet(p2)
    assert dataset_fingerprint(df1)["content_sha256"] == dataset_fingerprint(df2)["content_sha256"]


def test_dataset_fingerprint_single_cell_edit(sample_df):
    fp_orig = dataset_fingerprint(sample_df)
    first_hpg_date = sample_df.filter(pl.col("symbol") == "HPG")["date"][0]
    df_mod = sample_df.with_columns(
        raw_close=pl.when(
            (pl.col("symbol") == "HPG") & (pl.col("date") == first_hpg_date)
        )
        .then(pl.col("raw_close") + 100.0)
        .otherwise(pl.col("raw_close"))
    )
    fp_mod = dataset_fingerprint(df_mod)
    assert fp_mod["content_sha256"] != fp_orig["content_sha256"]
    assert fp_mod["per_symbol_content_sha256"]["HPG"] != fp_orig["per_symbol_content_sha256"]["HPG"]
    assert fp_mod["per_symbol_content_sha256"]["FPT"] == fp_orig["per_symbol_content_sha256"]["FPT"]
    diff_cols = [
        c
        for c in FINGERPRINT_COLUMNS
        if fp_mod["per_column_content_sha256"][c] != fp_orig["per_column_content_sha256"][c]
    ]
    assert diff_cols == ["raw_close"]


def test_dataset_fingerprint_extra_columns_invariant(sample_df):
    fp_orig = dataset_fingerprint(sample_df)
    df_mod = sample_df.with_columns(
        sector=pl.lit("DIFFERENT_SECTOR"),
        data_source=pl.lit("custom_source"),
        extra_dummy_column=pl.lit(12345.67),
    )
    fp_mod = dataset_fingerprint(df_mod)
    assert fp_orig["content_sha256"] == fp_mod["content_sha256"]


def test_dataset_fingerprint_validation_errors(sample_df):
    # Missing column
    with pytest.raises(ValueError, match="Missing fingerprint columns"):
        dataset_fingerprint(sample_df.drop("raw_close"))

    # Empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        dataset_fingerprint(sample_df.filter(pl.col("symbol") == "NONEXISTENT"))

    # Date contains null
    with pytest.raises(ValueError, match="date column contains null"):
        dataset_fingerprint(
            sample_df.with_columns(
                date=pl.when(pl.int_range(0, pl.len()) == 0).then(None).otherwise(pl.col("date"))
            )
        )

    # Date cannot be cast
    with pytest.raises(ValueError, match="date column cannot be strictly cast"):
        dataset_fingerprint(sample_df.with_columns(date=pl.lit("invalid_date_format")))

    # Symbol contains null
    with pytest.raises(ValueError, match="symbol column contains null"):
        dataset_fingerprint(
            sample_df.with_columns(
                symbol=pl.when(pl.int_range(0, pl.len()) == 0).then(None).otherwise(pl.col("symbol"))
            )
        )

    # Duplicate (date, symbol) pair
    with pytest.raises(ValueError, match="Duplicate"):
        dataset_fingerprint(pl.concat([sample_df, sample_df.slice(0, 1)]))

    # Datetime with 2 different hours on same day and same symbol
    dt_series = pl.Series(
        "date",
        [
            datetime.datetime(2026, 1, 1, 9, 0, 0),
            datetime.datetime(2026, 1, 1, 15, 0, 0),
        ],
    )
    df_dt_dup = sample_df.slice(0, 2).with_columns(
        date=dt_series,
        symbol=pl.lit("HPG"),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        dataset_fingerprint(df_dt_dup)


def test_dataset_fingerprint_normalization(sample_df):
    # -0.0 vs +0.0
    df_zero = sample_df.with_columns(
        adv=pl.when(pl.int_range(0, pl.len()) == 0).then(0.0).otherwise(pl.col("adv"))
    )
    df_neg_zero = sample_df.with_columns(
        adv=pl.when(pl.int_range(0, pl.len()) == 0).then(-0.0).otherwise(pl.col("adv"))
    )
    assert (
        dataset_fingerprint(df_zero)["content_sha256"]
        == dataset_fingerprint(df_neg_zero)["content_sha256"]
    )

    # Canonical NaN vs custom NaN payload
    custom_nan_bytes = b"\x01\x02\x03\x04\x00\x00\xf8\x7f"
    custom_nan = float(np.frombuffer(custom_nan_bytes, dtype="<f8")[0])
    df_nan1 = sample_df.with_columns(
        adv=pl.when(pl.int_range(0, pl.len()) == 0).then(float("nan")).otherwise(pl.col("adv"))
    )
    df_nan2 = sample_df.with_columns(
        adv=pl.when(pl.int_range(0, pl.len()) == 0).then(custom_nan).otherwise(pl.col("adv"))
    )
    assert (
        dataset_fingerprint(df_nan1)["content_sha256"]
        == dataset_fingerprint(df_nan2)["content_sha256"]
    )


def test_manifest_status_states(sample_df, tmp_path):
    fp = dataset_fingerprint(sample_df)

    # 1. NO_MANIFEST
    nonexistent = str(tmp_path / "does_not_exist.json")
    assert manifest_status(sample_df, nonexistent) == ("NO_MANIFEST", None)

    # 2. MATCH
    manifest_data = dict(fp)
    manifest_data["file_sha256"] = "dummy_file_sha256"
    p_match = tmp_path / "manifest_match.json"
    p_match.write_text(json.dumps(manifest_data), encoding="utf-8")
    assert manifest_status(sample_df, str(p_match)) == ("MATCH", fp["content_sha256"])

    # 3. MISMATCH
    manifest_mismatch = dict(manifest_data)
    manifest_mismatch["content_sha256"] = "0" * 64
    p_mismatch = tmp_path / "manifest_mismatch.json"
    p_mismatch.write_text(json.dumps(manifest_mismatch), encoding="utf-8")
    assert manifest_status(sample_df, str(p_mismatch)) == ("MISMATCH", fp["content_sha256"])

    # 4. UNAVAILABLE
    # Bad JSON
    p_bad_json = tmp_path / "bad.json"
    p_bad_json.write_text("invalid json {[[", encoding="utf-8")
    assert manifest_status(sample_df, str(p_bad_json)) == ("UNAVAILABLE", None)

    # Missing key
    p_missing = tmp_path / "missing_key.json"
    m_miss = dict(manifest_data)
    del m_miss["first_date"]
    p_missing.write_text(json.dumps(m_miss), encoding="utf-8")
    assert manifest_status(sample_df, str(p_missing)) == ("UNAVAILABLE", None)

    # Invalid schema_version
    p_bad_ver = tmp_path / "bad_ver.json"
    m_ver = dict(manifest_data)
    m_ver["schema_version"] = 2
    p_bad_ver.write_text(json.dumps(m_ver), encoding="utf-8")
    assert manifest_status(sample_df, str(p_bad_ver)) == ("UNAVAILABLE", None)

    # Invalid fingerprint_columns
    p_bad_cols = tmp_path / "bad_cols.json"
    m_cols = dict(manifest_data)
    m_cols["fingerprint_columns"] = ["date", "symbol"]
    p_bad_cols.write_text(json.dumps(m_cols), encoding="utf-8")
    assert manifest_status(sample_df, str(p_bad_cols)) == ("UNAVAILABLE", None)

    # Invalid per_symbol_rows type (list instead of dict)
    p_bad_dict = tmp_path / "bad_dict.json"
    m_dict = dict(manifest_data)
    m_dict["per_symbol_rows"] = [947, 947]
    p_bad_dict.write_text(json.dumps(m_dict), encoding="utf-8")
    assert manifest_status(sample_df, str(p_bad_dict)) == ("UNAVAILABLE", None)

    # dataset_fingerprint raises exception on invalid DataFrame
    assert manifest_status(pl.DataFrame(), str(p_match)) == ("UNAVAILABLE", None)
