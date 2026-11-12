"""
tests/test_paper_fill_log_benchmark.py - Observational instrumentation for live paper evaluation:
1. Per-order fill audit log (est_price vs ATO raw_open vs fill_price) - criterion E3.
2. VN30 Equal-Weight benchmark NAV in paper_nav_history.csv - criteria B_N / A_N.
Both must be strictly observational: they must never change generated orders or fills.
"""

import copy

import polars as pl
import pytest

from engine import data_pipeline as dp
from services import daily_paper_trader as dpt


def _run_sessions(n_sessions: int = 8):
    df_raw = dp.generate_fallback_data(
        symbols=["ACB", "CTG", "FPT", "HPG", "TCB", "TPB", "VCB", "VIB", "VJC", "VNM"],
        start_date="2023-01-01",
        end_date="2023-06-15",
        seed=42,
    )
    df_enr = dpt._enrich_market_dataframe(df_raw)
    all_dates = sorted(df_enr["date"].unique().to_list())
    sim_dates = all_dates[-n_sessions:]
    state = dpt.init_empty_state(100_000_000.0)
    steps = []
    for idx, cur_date in enumerate(sim_dates):
        cur_str = cur_date.isoformat()
        df_hist = df_enr.filter(pl.col("date") <= cur_date)
        today_mkt = {r["symbol"]: dict(r) for r in df_enr.filter(pl.col("date") == cur_date).iter_rows(named=True)}
        prev_map = {}
        if idx > 0:
            p_date = sim_dates[idx - 1]
            prev_map = {r["symbol"]: r["raw_close"] for r in df_enr.filter(pl.col("date") == p_date).iter_rows(named=True)}
        state, exec_reps = dpt.execute_pending_orders_and_settle(state, today_mkt, cur_str, trading_dates=all_dates)
        rec = dpt.mark_to_market(state, today_mkt, prev_map, cur_str, len(exec_reps))
        orders = dpt.generate_next_day_orders(state, df_hist, cur_str)
        steps.append((cur_str, copy.deepcopy(state), df_hist, today_mkt, exec_reps, rec, orders))
    return steps


def test_instrumentation_never_changes_orders_or_fills():
    steps = _run_sessions()
    assert any(len(s[4]) > 0 for s in steps), "fixture must produce at least one executed report"
    for cur_str, state, df_hist, today_mkt, exec_reps, rec, orders in steps:
        # Orders must be identical whether or not the benchmark state key exists.
        st_wo = copy.deepcopy(state)
        st_wo.pop("benchmark_nav", None)
        assert dpt.generate_next_day_orders(st_wo, df_hist, cur_str) == orders
        # raw_open attached to every report equals that session's ATO open.
        for rep in exec_reps:
            assert rep["raw_open"] == pytest.approx(float(today_mkt[rep["symbol"]]["raw_open"]))


def test_benchmark_equal_weight_close_to_close_and_compounding():
    st = dpt.init_empty_state(100_000_000.0)
    prev = {"AAA": 100.0, "BBB": 200.0, "CCC": 50.0}
    today = {"AAA": {"raw_close": 110.0}, "BBB": {"raw_close": 190.0}, "CCC": {"raw_close": None}, "DDD": {"raw_close": 10.0}}
    rec1 = dpt.mark_to_market(st, today, prev, "2026-11-13", 0)
    # Only AAA (+10%) and BBB (-5%) have valid closes on both sessions -> mean +2.5%
    assert rec1["daily_return_benchmark"] == pytest.approx(0.025)
    assert rec1["benchmark"] == pytest.approx(102_500_000.0)
    assert st["benchmark_nav"] == pytest.approx(102_500_000.0)

    rec2 = dpt.mark_to_market(st, {"AAA": {"raw_close": 121.0}}, {"AAA": 110.0}, "2026-11-14", 0)
    assert rec2["benchmark"] == pytest.approx(102_500_000.0 * 1.10)

    st_empty = dpt.init_empty_state(100_000_000.0)
    rec3 = dpt.mark_to_market(st_empty, {"AAA": {"raw_close": 110.0}}, {}, "2026-11-13", 0)
    assert rec3["daily_return_benchmark"] == 0.0
    assert rec3["benchmark"] == pytest.approx(100_000_000.0)


def test_nav_history_pads_legacy_rows_and_accepts_records_without_benchmark(tmp_path):
    csv_file = tmp_path / "paper_nav_history.csv"
    csv_file.write_text(
        "date,real_nav,available_cash,receivable_cash,stock_value,naive_nav,implementation_shortfall_pct,"
        "daily_return_real,daily_return_naive,orders_executed\n"
        "2026-11-09,100000000.00,100000000.00,0.00,0.00,100000000.00,0.0000,0.000000,0.000000,0\n",
        encoding="utf-8",
    )
    base = {
        "real_nav": 100_100_000.0, "available_cash": 1.0, "receivable_cash": 0.0, "stock_value": 1.0,
        "naive_nav": 100_200_000.0, "implementation_shortfall_pct": 0.1, "daily_return_real": 0.001,
        "daily_return_naive": 0.002, "orders_executed": 1,
    }
    dpt.append_nav_history(str(csv_file), dict(base, date="2026-11-10", daily_return_benchmark=0.004, benchmark=100_400_000.0))
    dpt.append_nav_history(str(csv_file), dict(base, date="2026-11-11"))  # legacy-shaped record (no benchmark keys)
    df = pl.read_csv(csv_file)
    assert df.columns[-2:] == ["daily_return_benchmark", "benchmark"]
    assert df["date"].cast(pl.Utf8).to_list() == ["2026-11-09", "2026-11-10", "2026-11-11"]
    assert df["benchmark"][0] is None
    assert float(df["benchmark"][1]) == pytest.approx(100_400_000.0)
    assert df["benchmark"][2] is None


def test_fill_log_signs_and_idempotent_by_date(tmp_path):
    reports = [
        {"action": "BUY", "symbol": "FPT", "qty": 200, "est_price": 100_000.0, "raw_open": 101_000.0,
         "fill_price": 101_100.0, "fill_qty": 200, "status": "FILLED"},
        {"action": "SELL", "symbol": "ACB", "qty": 300, "est_price": 20_000.0, "raw_open": 19_800.0,
         "fill_price": 19_780.0, "fill_qty": 300, "status": "FILLED"},
        {"action": "SELL", "symbol": "VNM", "qty": 100, "est_price": 60_000.0, "raw_open": 55_800.0,
         "fill_price": 55_800.0, "fill_qty": 0, "status": "FLOOR TRAPPED (HOLDING ORDER)"},
    ]
    log_file = tmp_path / "paper_fill_log.csv"
    dpt.append_fill_log(str(log_file), "2026-11-13", reports)
    dpt.append_fill_log(str(log_file), "2026-11-13", reports[:1])  # rerun same session replaces rows
    dpt.append_fill_log(str(log_file), "2026-11-14", reports[1:])
    df = pl.read_csv(log_file)
    assert df.columns == dpt.FILL_LOG_HEADERS
    assert len(df) == 3
    buy = df.row(0, named=True)
    assert buy["overnight_gap_pct"] == pytest.approx(1.0)
    assert buy["exec_vs_open_bps"] == pytest.approx(9.90, abs=0.01)     # paid above open -> adverse +
    assert buy["exec_vs_est_bps"] == pytest.approx(110.0, abs=0.01)
    sell = df.row(1, named=True)
    assert sell["exec_vs_open_bps"] == pytest.approx(10.10, abs=0.01)   # sold below open -> adverse +
    assert sell["exec_vs_est_bps"] == pytest.approx(110.0, abs=0.01)
    trapped = df.row(2, named=True)
    assert trapped["fill_qty"] == 0
    assert trapped["exec_vs_open_bps"] is None
    assert trapped["overnight_gap_pct"] == pytest.approx(-7.0)
