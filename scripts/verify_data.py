import argparse
import hashlib
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import polars as pl
from engine import data_pipeline as dp


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    parser = argparse.ArgumentParser(description="Verify Parquet data against recorded content fingerprint manifest.")
    parser.add_argument("parquet", nargs="?", default="data/vn30_production.parquet", help="Path to Parquet file")
    parser.add_argument("--manifest", default=os.path.join(ROOT_DIR, "experiments", "data_manifest.json"), help="Path to manifest JSON")
    parser.add_argument("--write-manifest", default=None, help="Write computed fingerprint to specified manifest path (refuses overwrite)")
    args = parser.parse_args()

    parquet_path = args.parquet

    # Mode 1: Write manifest
    if args.write_manifest:
        if os.path.exists(args.write_manifest):
            print(f"Error: manifest already exists at {args.write_manifest}, refusing to overwrite", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(parquet_path):
            print(f"Error: Parquet file not found: {parquet_path}", file=sys.stderr)
            sys.exit(2)
        try:
            file_sha = compute_file_sha256(parquet_path)
            df = pl.read_parquet(parquet_path)
            fp = dp.dataset_fingerprint(df)
        except ValueError as e:
            print(f"Error computing dataset fingerprint: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"Error reading Parquet file: {e}", file=sys.stderr)
            sys.exit(2)

        manifest = {
            "schema_version": 1,
            "fingerprint_columns": list(dp.FINGERPRINT_COLUMNS),
            "file_sha256": file_sha,
            "content_sha256": fp["content_sha256"],
            "rows": fp["rows"],
            "sessions": fp["sessions"],
            "symbols": fp["symbols"],
            "first_date": fp["first_date"],
            "last_date": fp["last_date"],
            "per_column_content_sha256": fp["per_column_content_sha256"],
            "per_symbol_rows": fp["per_symbol_rows"],
            "per_symbol_content_sha256": fp["per_symbol_content_sha256"],
        }

        out_dir = os.path.dirname(os.path.abspath(args.write_manifest))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        with open(args.write_manifest, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"Manifest successfully written to: {args.write_manifest}")
        sys.exit(0)

    # Mode 2: Verify against manifest
    # Validate manifest first
    if not os.path.isfile(args.manifest):
        print(f"manifest invalid: file not found at {args.manifest}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"manifest invalid: JSON error: {e}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(manifest, dict):
        print("manifest invalid: root JSON element is not an object", file=sys.stderr)
        sys.exit(2)

    for k in dp.REQUIRED_MANIFEST_KEYS:
        if k not in manifest:
            print(f"manifest invalid: missing required key '{k}'", file=sys.stderr)
            sys.exit(2)

    if manifest.get("schema_version") != 1:
        print(f"manifest invalid: unsupported schema_version {manifest.get('schema_version')}", file=sys.stderr)
        sys.exit(2)

    if manifest.get("fingerprint_columns") != dp.FINGERPRINT_COLUMNS:
        print("manifest invalid: fingerprint_columns do not match expected columns", file=sys.stderr)
        sys.exit(2)

    # Read and fingerprint parquet
    if not os.path.isfile(parquet_path):
        print(f"Error: Parquet file not found: {parquet_path}", file=sys.stderr)
        sys.exit(2)

    try:
        file_sha = compute_file_sha256(parquet_path)
        df = pl.read_parquet(parquet_path)
        fp = dp.dataset_fingerprint(df)
    except ValueError as e:
        print(f"Error computing dataset fingerprint: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading Parquet file: {e}", file=sys.stderr)
        sys.exit(2)

    manifest_file_sha = str(manifest.get("file_sha256", "")).lower()
    file_sha_cmp = "same" if file_sha == manifest_file_sha else "different"

    # Print summary (ASCII only)
    print(f"Parquet Path : {parquet_path}")
    print(f"Manifest Path: {args.manifest}")
    print(f"File SHA256  : file={file_sha} manifest={manifest_file_sha} ({file_sha_cmp})")
    print(f"Rows         : file={fp['rows']} manifest={manifest['rows']}")
    print(f"Sessions     : file={fp['sessions']} manifest={manifest['sessions']}")
    print(f"Symbols      : file={fp['symbols']} manifest={manifest['symbols']}")
    print(f"First Date   : file={fp['first_date']} manifest={manifest['first_date']}")
    print(f"Last Date    : file={fp['last_date']} manifest={manifest['last_date']}")
    print(f"Content SHA  : file={fp['content_sha256']}")
    print(f"               manifest={manifest['content_sha256']}")

    is_match = (fp["content_sha256"] == manifest["content_sha256"])
    if is_match:
        print("RESULT: MATCH")
        sys.exit(0)
    else:
        print("RESULT: MISMATCH")
        # Differing columns
        diff_cols = [
            c for c in dp.FINGERPRINT_COLUMNS
            if fp["per_column_content_sha256"].get(c) != manifest.get("per_column_content_sha256", {}).get(c)
        ]
        if diff_cols:
            print(f"  Differing columns ({len(diff_cols)}): {', '.join(diff_cols)}")

        file_symbols = set(fp["per_symbol_content_sha256"].keys())
        manifest_symbols = set(manifest.get("per_symbol_content_sha256", {}).keys())
        only_in_file = sorted(file_symbols - manifest_symbols)
        only_in_manifest = sorted(manifest_symbols - file_symbols)
        common_symbols = sorted(file_symbols & manifest_symbols)

        if only_in_file:
            print(f"  Symbols only in file ({len(only_in_file)}): {', '.join(only_in_file)}")
        if only_in_manifest:
            print(f"  Symbols only in manifest ({len(only_in_manifest)}): {', '.join(only_in_manifest)}")

        diff_sym_hashes = [
            s for s in common_symbols
            if fp["per_symbol_content_sha256"].get(s) != manifest.get("per_symbol_content_sha256", {}).get(s)
        ]
        if diff_sym_hashes:
            print(f"  Differing symbols ({len(diff_sym_hashes)}): {', '.join(diff_sym_hashes)}")

        diff_sym_rows = [
            (s, fp["per_symbol_rows"].get(s), manifest.get("per_symbol_rows", {}).get(s))
            for s in common_symbols
            if fp["per_symbol_rows"].get(s) != manifest.get("per_symbol_rows", {}).get(s)
        ]
        if diff_sym_rows:
            row_diff_strs = [f"{s} (file={fr}, manifest={mr})" for s, fr, mr in diff_sym_rows]
            print(f"  Symbols with different row count ({len(diff_sym_rows)}): {', '.join(row_diff_strs)}")

        sys.exit(1)


if __name__ == "__main__":
    main()
