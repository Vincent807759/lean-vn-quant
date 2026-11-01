import json
import os
import subprocess
import sys
import pytest
import polars as pl
from engine.data_pipeline import fetch_vn_ohlcv

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "verify_data.py",
)


@pytest.fixture(scope="module")
def base_df():
    return fetch_vn_ohlcv(
        symbols=["HPG", "FPT", "VIC"],
        start_date="2026-01-01",
        end_date="2026-03-31",
        save_parquet=None,
        force_fallback=True,
    )


def run_script(*args):
    cmd = [sys.executable, SCRIPT_PATH] + list(args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_verify_data_write_manifest_and_refuse_overwrite(base_df, tmp_path):
    parquet_file = str(tmp_path / "base.parquet")
    manifest_file = str(tmp_path / "manifest.json")
    base_df.write_parquet(parquet_file)

    # 1. Write manifest successfully
    code, stdout, stderr = run_script(parquet_file, "--write-manifest", manifest_file)
    assert code == 0
    assert "Manifest successfully written" in stdout
    assert os.path.exists(manifest_file)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["rows"] == base_df.height
    assert manifest_data["symbols"] == 3

    # 2. Re-run same write command -> exit code 2 (refuse overwrite) and manifest untouched
    mtime_before = os.path.getmtime(manifest_file)
    code2, stdout2, stderr2 = run_script(parquet_file, "--write-manifest", manifest_file)
    assert code2 == 2
    assert "refusing to overwrite" in stderr2
    assert os.path.getmtime(manifest_file) == mtime_before


def test_verify_data_match(base_df, tmp_path):
    parquet_file = str(tmp_path / "match.parquet")
    manifest_file = str(tmp_path / "match_manifest.json")
    base_df.write_parquet(parquet_file)

    code, _, _ = run_script(parquet_file, "--write-manifest", manifest_file)
    assert code == 0

    code, stdout, stderr = run_script(parquet_file, "--manifest", manifest_file)
    assert code == 0
    assert "RESULT: MATCH" in stdout


def test_verify_data_mismatch_modified_price(base_df, tmp_path):
    orig_parquet = str(tmp_path / "orig.parquet")
    mod_parquet = str(tmp_path / "mod_price.parquet")
    manifest_file = str(tmp_path / "mod_price_manifest.json")

    base_df.write_parquet(orig_parquet)
    code, _, _ = run_script(orig_parquet, "--write-manifest", manifest_file)
    assert code == 0

    # Modify raw_close for HPG
    first_hpg_date = base_df.filter(pl.col("symbol") == "HPG")["date"][0]
    df_mod = base_df.with_columns(
        raw_close=pl.when(
            (pl.col("symbol") == "HPG") & (pl.col("date") == first_hpg_date)
        )
        .then(pl.col("raw_close") + 500.0)
        .otherwise(pl.col("raw_close"))
    )
    df_mod.write_parquet(mod_parquet)

    code, stdout, stderr = run_script(mod_parquet, "--manifest", manifest_file)
    assert code == 1
    assert "RESULT: MISMATCH" in stdout
    assert "Differing columns" in stdout and "raw_close" in stdout
    assert "Differing symbols" in stdout and "HPG" in stdout
    assert "FPT" not in stdout.split("Differing symbols")[1]


def test_verify_data_mismatch_symbol_set(base_df, tmp_path):
    full_parquet = str(tmp_path / "full.parquet")
    manifest_file = str(tmp_path / "full_manifest.json")
    base_df.write_parquet(full_parquet)
    code, _, _ = run_script(full_parquet, "--write-manifest", manifest_file)
    assert code == 0

    # Test file with missing symbol (only HPG, FPT; missing VIC)
    subset_parquet = str(tmp_path / "subset.parquet")
    base_df.filter(pl.col("symbol") != "VIC").write_parquet(subset_parquet)

    code, stdout, stderr = run_script(subset_parquet, "--manifest", manifest_file)
    assert code == 1
    assert "RESULT: MISMATCH" in stdout
    assert "only in manifest" in stdout and "VIC" in stdout

    # Test file with extra symbol
    extra_df = fetch_vn_ohlcv(
        symbols=["HPG", "FPT", "VIC", "SSI"],
        start_date="2026-01-01",
        end_date="2026-03-31",
        save_parquet=None,
        force_fallback=True,
    )
    extra_parquet = str(tmp_path / "extra.parquet")
    extra_df.write_parquet(extra_parquet)

    code2, stdout2, stderr2 = run_script(extra_parquet, "--manifest", manifest_file)
    assert code2 == 1
    assert "RESULT: MISMATCH" in stdout2
    assert "only in file" in stdout2 and "SSI" in stdout2


def test_verify_data_mismatch_row_count(base_df, tmp_path):
    orig_parquet = str(tmp_path / "orig_rows.parquet")
    manifest_file = str(tmp_path / "rows_manifest.json")
    base_df.write_parquet(orig_parquet)
    code, _, _ = run_script(orig_parquet, "--write-manifest", manifest_file)
    assert code == 0

    # Drop 1 row of HPG
    hpg_dates = base_df.filter(pl.col("symbol") == "HPG")["date"].to_list()
    df_fewer_hpg = base_df.filter(
        ~((pl.col("symbol") == "HPG") & (pl.col("date") == hpg_dates[0]))
    )
    fewer_parquet = str(tmp_path / "fewer_hpg.parquet")
    df_fewer_hpg.write_parquet(fewer_parquet)

    code, stdout, stderr = run_script(fewer_parquet, "--manifest", manifest_file)
    assert code == 1
    assert "RESULT: MISMATCH" in stdout
    assert "Symbols with different row count" in stdout
    assert "HPG" in stdout


def test_verify_data_missing_parquet(tmp_path):
    nonexistent_parquet = str(tmp_path / "no_such_file.parquet")
    dummy_manifest = str(tmp_path / "dummy_manifest.json")

    # When manifest also doesn't exist, manifest validation catches it first
    # So create a valid manifest first
    manifest_data = {
        "schema_version": 1,
        "fingerprint_columns": [
            "date", "symbol", "raw_open", "raw_high", "raw_low", "raw_close",
            "adj_close", "volume", "adv", "ceiling_price", "floor_price", "is_listed"
        ],
        "file_sha256": "dummy",
        "content_sha256": "dummy",
        "rows": 0,
        "sessions": 0,
        "symbols": 0,
        "first_date": "2026-01-01",
        "last_date": "2026-03-31",
        "per_column_content_sha256": {},
        "per_symbol_rows": {},
        "per_symbol_content_sha256": {},
    }
    with open(dummy_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    code, stdout, stderr = run_script(nonexistent_parquet, "--manifest", dummy_manifest)
    assert code == 2
    assert "Parquet file not found" in stderr


def test_verify_data_parquet_missing_column(base_df, tmp_path):
    bad_parquet = str(tmp_path / "bad_cols.parquet")
    manifest_file = str(tmp_path / "valid_manifest.json")
    base_df.write_parquet(str(tmp_path / "tmp.parquet"))
    run_script(str(tmp_path / "tmp.parquet"), "--write-manifest", manifest_file)

    # Parquet without raw_close
    base_df.drop("raw_close").write_parquet(bad_parquet)
    code, stdout, stderr = run_script(bad_parquet, "--manifest", manifest_file)
    assert code == 2
    assert "Error computing dataset fingerprint" in stderr or "Missing fingerprint columns" in stderr


def test_verify_data_manifest_invalid_cases(base_df, tmp_path):
    parquet_file = str(tmp_path / "valid.parquet")
    base_df.write_parquet(parquet_file)
    valid_manifest_path = str(tmp_path / "valid_manifest.json")
    run_script(parquet_file, "--write-manifest", valid_manifest_path)
    with open(valid_manifest_path, "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    # 1. Broken JSON
    p_bad_json = str(tmp_path / "broken.json")
    with open(p_bad_json, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    code, stdout, stderr = run_script(parquet_file, "--manifest", p_bad_json)
    assert code == 2
    assert "manifest invalid" in stderr

    # 2. Missing key
    p_miss = str(tmp_path / "miss_key.json")
    miss_data = dict(valid_data)
    del miss_data["content_sha256"]
    with open(p_miss, "w", encoding="utf-8") as f:
        json.dump(miss_data, f)
    code, stdout, stderr = run_script(parquet_file, "--manifest", p_miss)
    assert code == 2
    assert "manifest invalid" in stderr and "missing required key" in stderr

    # 3. Wrong schema_version
    p_ver = str(tmp_path / "wrong_ver.json")
    ver_data = dict(valid_data)
    ver_data["schema_version"] = 99
    with open(p_ver, "w", encoding="utf-8") as f:
        json.dump(ver_data, f)
    code, stdout, stderr = run_script(parquet_file, "--manifest", p_ver)
    assert code == 2
    assert "manifest invalid" in stderr and "schema_version" in stderr

    # 4. Wrong fingerprint_columns
    p_cols = str(tmp_path / "wrong_cols.json")
    cols_data = dict(valid_data)
    cols_data["fingerprint_columns"] = ["date", "symbol"]
    with open(p_cols, "w", encoding="utf-8") as f:
        json.dump(cols_data, f)
    code, stdout, stderr = run_script(parquet_file, "--manifest", p_cols)
    assert code == 2
    assert "manifest invalid" in stderr and "fingerprint_columns" in stderr
