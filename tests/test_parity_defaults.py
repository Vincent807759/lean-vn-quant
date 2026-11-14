"""
tests/test_parity_defaults.py - The live paper trader and the backtest simulator
must run the same rules when called with their DEFAULT arguments.
"""

import copy
import inspect

import numpy as np

from engine import data_pipeline as dp
from engine import vn_simulator as sim
from services import daily_paper_trader as dpt


def _default(func, name):
    return inspect.signature(func).parameters[name].default


def test_default_flags_match_between_paper_trader_and_simulator():
    """Cùng một bộ luật mặc định: T+2.5 nghiêm ngặt, trần 3% ADV20, tắt greedy sweep, tie-break theo ADV20, giới hạn ADV khi bán."""
    assert _default(sim.simulate_vietnam, "strict_t25_ato_lock") is True
    assert _default(dpt.generate_next_day_orders, "strict_t25_ato_lock") is True
    assert dpt.MAX_ADV_PARTICIPATION == _default(sim.simulate_vietnam, "max_adv_participation")
    assert _default(dpt.generate_next_day_orders, "enable_greedy_sweep") is False
    assert _default(dpt.generate_next_day_orders, "use_adv20_tie_break") is True
    assert _default(dpt.execute_pending_orders_and_settle, "sell_adv_cap") is True


def test_simulator_default_first_sale_is_t_plus_3_ato():
    """Mua ATO phiên T thì với tham số mặc định, lệnh bán sớm nhất khớp ở ATO phiên T+3, không phải T+2."""
    T, N = 8, 1
    ro = np.full((T, N), 50000.0)
    rc = ro.copy()
    ceil = np.full((T, N), 53500.0)
    flr = np.full((T, N), 46500.0)
    adv = np.full((T, N), 10_000_000.0)
    w = np.zeros((T, N))
    w[1, 0] = 1.0  # buy fills at ATO of t=1 (T)

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0, daily_vol=np.zeros((T, N)),
        max_adv_participation=1.0, warn_unlagged=False,
    )
    tax = np.diff(np.concatenate([[0.0], res["sell_tax"]]))
    assert tax[3] == 0.0  # T+2 ATO: shares not delivered yet
    assert tax[4] > 0.0   # T+3 ATO: first legal sale


def test_paper_trader_default_does_not_sell_locked_t1_shares():
    """Cuối phiên T+1, cổ phiếu còn ở locked_t1 không được đưa vào lệnh bán ATO sáng T+2 khi dùng tham số mặc định."""
    st = dpt.init_empty_state(100_000_000.0)
    st["holdings"]["HPG"] = {"sellable_qty": 0, "locked_t1": 0, "locked_t2": 1000, "avg_cost": 25000.0}
    st["last_ledger_advance_date"] = "2024-01-01"
    dpt.advance_t25_settlement(st, "2024-01-02")
    st["last_real_nav"] = 100_000_000.0
    df_hist = dp.generate_fallback_data(
        symbols=["ACB", "FPT", "HPG", "VCB", "MBB", "TCB"],
        start_date="2023-01-01", end_date="2023-05-15", seed=42,
    )
    last_date = str(df_hist["date"].max())

    orders_default = dpt.generate_next_day_orders(copy.deepcopy(st), df_hist, last_date)
    orders_strict = dpt.generate_next_day_orders(copy.deepcopy(st), df_hist, last_date, strict_t25_ato_lock=True)
    assert not [o for o in orders_default if o["action"] == "SELL" and o["symbol"] == "HPG"]
    assert orders_default == orders_strict
