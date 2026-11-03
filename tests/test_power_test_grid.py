"""Unit tests for planted-alpha grid runner and CLI (HUONG_DI_R18 commit c)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.power_test import (
    interpolate_rho_star,
    run_grid,
    save_results_json,
    wilson_ci,
)


def test_interpolate_rho_star_known_curve():
    """Verify interpolate_rho_star on synthetic curves with known crossings."""
    grid = [0.0, 0.05, 0.10, 0.15]

    # Curve crossing 0.8 between 0.05 (0.50) and 0.10 (0.85)
    # fraction = (0.8 - 0.5) / (0.85 - 0.5) = 0.30 / 0.35 = 6/7
    # rho* = 0.05 + (6/7) * 0.05 = 0.05 + 0.042857 = 0.092857
    curve1 = [0.05, 0.50, 0.85, 0.99]
    res1 = interpolate_rho_star(grid, curve1, threshold=0.8)
    assert isinstance(res1, float)
    assert abs(res1 - 0.092857) < 1e-4

    # Curve starting above threshold
    curve2 = [0.85, 0.90, 0.95, 1.0]
    assert interpolate_rho_star(grid, curve2, threshold=0.8) == 0.0

    # Curve never reaching threshold
    curve3 = [0.05, 0.20, 0.40, 0.60]
    assert interpolate_rho_star(grid, curve3, threshold=0.8) == "chưa đạt trong lưới"


def test_wilson_ci_properties():
    """Verify Wilson confidence interval properties."""
    low, high = wilson_ci(80, 100)
    assert 0.70 < low < 0.72
    assert 0.86 < high < 0.88

    # Edge cases
    assert wilson_ci(0, 10)[0] == 0.0
    assert wilson_ci(10, 10)[1] == 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_run_grid_small_and_json_serialization(tmp_path: Path):
    """Verify small grid run (R=2, R0=4, T=200) saves JSON, protects overwrite, and has no absolute paths."""
    out_file = tmp_path / "test_power_grid.json"

    data = run_grid(blocks=["main"], R=2, R0=4, T=200, N=30)

    # 1. Required top-level keys
    assert "metadata" in data
    assert "gates" in data
    assert "primary_answers" in data
    assert "cells" in data

    # Metadata keys
    meta = data["metadata"]
    assert meta["T"] == 200
    assert meta["R"] == 2
    assert meta["R0"] == 4
    assert "versions" in meta
    assert "numpy" in meta["versions"]
    assert "scipy" in meta["versions"]
    assert "numba" in meta["versions"]

    # 2. Save JSON
    save_results_json(data, out_file)
    assert out_file.exists()

    # 3. Reject overwrite
    with pytest.raises(FileExistsError):
        save_results_json(data, out_file)

    # 4. JSON contains no absolute paths
    raw_json_text = out_file.read_text(encoding="utf-8")
    assert str(tmp_path) not in raw_json_text
    assert "C:\\" not in raw_json_text
    assert "c:/" not in raw_json_text.lower()
    assert "/home/" not in raw_json_text
    assert "/Users/" not in raw_json_text

    # Parse back
    loaded = json.loads(raw_json_text)
    assert loaded["metadata"]["T"] == 200
