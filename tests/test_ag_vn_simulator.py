"""
tests/test_ag_vn_simulator.py - Kiểm thử bất biến và biên cho engine/vn_simulator.py.
"""

import datetime
import json
import warnings
import numpy as np
import pytest
from engine import vn_simulator as sim


# ------------------------------------------------------------------------------
# 1. PRICE LIMITS & HOSE TICK HELPERS
# ------------------------------------------------------------------------------

# engine/vn_simulator.py:340-349
def test_get_exchange_band_all_exchanges():
    """Bảo vệ biên độ dao động giá trần/sàn của HOSE, HNX, UPCOM và giá trị mặc định."""
    assert sim.get_exchange_band("HOSE") == pytest.approx(0.07)
    assert sim.get_exchange_band("hose") == pytest.approx(0.07)
    assert sim.get_exchange_band("HNX") == pytest.approx(0.10)
    assert sim.get_exchange_band("hnx") == pytest.approx(0.10)
    assert sim.get_exchange_band("UPCOM") == pytest.approx(0.15)
    assert sim.get_exchange_band("upcom") == pytest.approx(0.15)
    assert sim.get_exchange_band("UNKNOWN") == pytest.approx(0.07)


# engine/vn_simulator.py:352-382
def test_compute_bands_and_ticks_boundaries_and_invariants():
    """Bảo vệ quy tắc làm tròn trần/sàn theo đúng bước giá HOSE tại các mốc 10k và 50k VND."""
    refs = np.array([8000.0, 9350.0, 10700.0, 25000.0, 46750.0, 53700.0, 80000.0])
    ceil, flr = sim.compute_bands_and_ticks(refs, exchange="HOSE")

    # Kiểm tra mốc nhảy bước giá: 9350*1.07=10004.5 -> tick 50 -> 10000; 46750*1.07=50022.5 -> tick 100 -> 50000
    assert ceil[1] == 10000.0
    assert ceil[4] == 50000.0
    # 10700*0.93=9951 -> tick 10 -> 9960; 53700*0.93=49941 -> tick 50 -> 49950
    assert flr[2] == 9960.0
    assert flr[5] == 49950.0

    # Bất biến chung cho mọi mã HOSE
    assert np.all(flr <= refs)
    assert np.all(refs <= ceil)
    assert np.all(ceil <= refs * 1.07 + 1e-9)
    assert np.all(flr >= refs * 0.93 - 1e-9)
    for c, f in zip(ceil, flr):
        c_tk = 10.0 if c < 10000.0 else (50.0 if c < 50000.0 else 100.0)
        f_tk = 10.0 if f < 10000.0 else (50.0 if f < 50000.0 else 100.0)
        assert c % c_tk == pytest.approx(0.0)
        assert f % f_tk == pytest.approx(0.0)

    # HNX (+-10%) và UPCOM (+-15%) dùng bước giá 100 VND trong code
    c_hnx, f_hnx = sim.compute_bands_and_ticks(np.array([15400.0]), exchange="HNX")
    assert c_hnx[0] == 16900.0
    assert f_hnx[0] == 13900.0
    c_upc, f_upc = sim.compute_bands_and_ticks(np.array([20000.0]), exchange="UPCOM")
    assert c_upc[0] == 23000.0
    assert f_upc[0] == 17000.0


# engine/vn_simulator.py:390-397
def test_get_hose_tick_size_thresholds_and_jit_parity():
    """Bảo vệ ngưỡng bước giá 10/50/100 VND của HOSE và tính tương đương giữa Numba JIT với Python."""
    cases = [
        (1000.0, 10.0),
        (9999.9, 10.0),
        (10000.0, 50.0),
        (49999.9, 50.0),
        (50000.0, 100.0),
        (120000.0, 100.0),
    ]
    for price, expected_tick in cases:
        assert sim._get_hose_tick_size(price) == expected_tick
        assert sim._get_hose_tick_size.py_func(price) == expected_tick


# engine/vn_simulator.py:401-443
def test_finalize_exec_price_clamp_and_adverse_invariant():
    """Bảo vệ kẹp biên giá [sàn, trần] theo bước giá HOSE và tính bất lợi của round_adverse."""
    # clamp_and_round=False trả về nguyên giá gốc
    assert sim._finalize_exec_price(50012.34, 1, 46500.0, 53500.0, False, True) == pytest.approx(50012.34)

    # Vượt trần/sàn bị kẹp chặt vào trong dải tick hợp lệ
    p_over_ceil = sim._finalize_exec_price(54000.0, 1, 46500.0, 53520.0, True, False)
    assert p_over_ceil == 53500.0
    p_under_flr = sim._finalize_exec_price(46000.0, -1, 46510.0, 53500.0, True, False)
    assert p_under_flr == 46550.0

    # Bất biến: round_adverse BUY >= round_nearest, SELL <= round_nearest, và luôn thuộc [eff_flr, eff_ceil]
    rng = np.random.default_rng(7)
    for raw_p in rng.uniform(5000.0, 90000.0, size=100):
        flr_p = float(raw_p * 0.93)
        ceil_p = float(raw_p * 1.07)
        for side in (1, -1):
            p_near = sim._finalize_exec_price(float(raw_p), side, flr_p, ceil_p, True, False)
            p_adv = sim._finalize_exec_price(float(raw_p), side, flr_p, ceil_p, True, True)
            assert p_near % sim._get_hose_tick_size(p_near) == pytest.approx(0.0)
            assert p_adv % sim._get_hose_tick_size(p_adv) == pytest.approx(0.0)
            if side > 0:
                assert p_adv >= p_near - 1e-9
            else:
                assert p_adv <= p_near + 1e-9


# engine/vn_simulator.py:447-459
def test_calc_borrow_cap_edge_and_repayment_invariant():
    """Bảo vệ công thức giới hạn ứng trước tiền bán để tổng gốc và lãi không vượt tiền chờ về."""
    assert sim._calc_borrow_cap(50000.0, 0.00035, 2.0, 50000.0) == 0.0
    assert sim._calc_borrow_cap(40000.0, 0.00035, 2.0, 50000.0) == 0.0

    # days <= 0 được nâng về tối thiểu 1 ngày
    cap_d0 = sim._calc_borrow_cap(10_000_000.0, 0.00035, 0.0, 50000.0)
    cap_d1 = sim._calc_borrow_cap(10_000_000.0, 0.00035, 1.0, 50000.0)
    assert cap_d0 == pytest.approx(cap_d1)

    for pending in (100_000.0, 5_000_000.0, 71_428_571.0, 500_000_000.0):
        for days in (1.0, 2.0, 4.0):
            cap = sim._calc_borrow_cap(pending, 0.00035, days, 50000.0)
            fee = max(50000.0, cap * 0.00035 * days)
            assert cap >= 0.0
            assert cap + fee <= pending + 1e-6


# engine/vn_simulator.py:463-483
def test_calc_dynamic_slippage_monotonic_and_floor_invariant():
    """Bảo vệ sàn trượt giá tối thiểu và tính đơn điệu tăng của trượt giá theo khối lượng và biến động."""
    # Biên suy biến trả về max(base_slippage, half_spread_floor)
    assert sim._calc_dynamic_slippage(0.0, 100000.0, 0.02) == pytest.approx(0.0012)
    assert sim._calc_dynamic_slippage(1000.0, 0.0, 0.02) == pytest.approx(0.0012)
    assert sim._calc_dynamic_slippage(np.nan, 100000.0, 0.02) == pytest.approx(0.0012)
    assert sim._calc_dynamic_slippage(0.0, 100000.0, 0.02, base_slippage=0.0020) == pytest.approx(0.0020)

    # Đơn điệu tăng theo qty và vol
    s_small = sim._calc_dynamic_slippage(1000.0, 100000.0, 0.015)
    s_large = sim._calc_dynamic_slippage(20000.0, 100000.0, 0.015)
    s_high_vol = sim._calc_dynamic_slippage(20000.0, 100000.0, 0.040)
    assert s_small >= 0.0012
    assert s_large > s_small
    assert s_high_vol > s_large


# ------------------------------------------------------------------------------
# 2. INPUT SANITATION & NAIVE SIMULATOR
# ------------------------------------------------------------------------------

# engine/vn_simulator.py:1024-1103
def test_sanitize_market_matrices_edge_cases_and_invariants():
    """Bảo vệ xử lý NaN ngay phiên đầu, ADV âm, reset thiếu giá khi phục hồi và chuẩn hóa tỷ trọng."""
    ro = np.array([
        [np.nan, 20000.0],
        [np.nan, 20500.0],
        [15000.0, np.nan],
        [15200.0, 20600.0],
    ], dtype=np.float64)
    rc = ro.copy()
    ceil, flr = sim.compute_bands_and_ticks(np.nan_to_num(rc, nan=20000.0), "HOSE")
    adv = np.array([
        [1e5, -50.0],
        [np.nan, 1e5],
        [1e5, 1e5],
        [1e5, 1e5],
    ], dtype=np.float64)
    w = np.array([
        [0.6, 0.6],   # asset 0 NaN -> w[0,0]=0 nên tổng còn 0.6 (không chuẩn hóa)
        [0.3, 0.4],   # asset 0 NaN -> w[1,0]=0, asset 1 giữ 0.4
        [0.5, 0.5],   # asset 1 bị NaN nên bị đưa về 0.0
        [0.7, 0.7],   # cả hai hợp lệ, tổng 1.4 > 1.0 -> chuẩn hóa về [0.5, 0.5]
    ], dtype=np.float64)

    c_open, c_close, c_ceil, c_floor, c_adv, c_w, c_pit = sim.sanitize_market_matrices(
        ro, rc, ceil, flr, adv, w, max_missing_sessions=5, halt_haircut=0.30
    )

    assert np.all(np.isfinite(c_open))
    assert np.all(np.isfinite(c_close))
    assert np.all(np.isfinite(c_adv))
    assert np.all(c_adv >= 0.0)
    # Asset 0 NaN từ đầu -> 0.0 trước khi có giá đầu tiên
    assert c_open[0, 0] == 0.0 and c_close[1, 0] == 0.0 and c_pit[1, 0] == False
    # Asset 1 ADV âm ở t=0 -> bị ép về 0.0; thiếu ở t=2 -> ffill 20500 và pit=False, weight=0
    assert c_adv[0, 1] == 0.0
    assert c_close[2, 1] == 20500.0
    assert c_pit[2, 1] == False
    assert c_w[2, 1] == 0.0
    # Khử tỷ trọng NaN trước rồi mới chuẩn hóa hàng > 1.0
    assert c_w[0, 0] == 0.0 and c_w[0, 1] == pytest.approx(0.6)
    assert c_w[1, 0] == 0.0 and c_w[1, 1] == pytest.approx(0.4)
    assert c_w[3] == pytest.approx(np.array([0.5, 0.5]))
    assert np.all(np.sum(c_w, axis=1) <= 1.0 + 1e-12)


# engine/vn_simulator.py:970-1017
def test_simulate_naive_close_to_close_and_zero_weight_invariant():
    """Bảo vệ chế độ close-to-close của simulate_naive và tính bảo toàn vốn khi tỷ trọng bằng 0."""
    rc = np.array([[100.0, 200.0], [110.0, 180.0], [121.0, 180.0]], dtype=np.float64)
    w_zero = np.zeros_like(rc)
    eq_zero = sim.simulate_naive(rc, w_zero, initial_cash=1.0e8)
    assert np.all(eq_zero == 1.0e8)

    w_hold = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    eq_c2c = sim.simulate_naive(rc, w_hold, initial_cash=1.0e8)
    assert eq_c2c[0] == pytest.approx(1.0e8)
    assert eq_c2c[1] == pytest.approx(1.1e8)
    assert eq_c2c[2] == pytest.approx(1.21e8)

    # Khi raw_open có NaN, nhánh raw_open vẫn cho kết quả hữu hạn
    ro_nan = np.array([[100.0, 200.0], [np.nan, 190.0], [115.0, 180.0]], dtype=np.float64)
    eq_ro = sim.simulate_naive(rc, w_hold, initial_cash=1.0e8, raw_open=ro_nan)
    assert np.all(np.isfinite(eq_ro))


# ------------------------------------------------------------------------------
# 3. MICROSTRUCTURE CORE & SIMULATE_VIETNAM INVARIANTS
# ------------------------------------------------------------------------------

# engine/vn_simulator.py:487-963,1106-1346
def test_simulate_vietnam_conservation_of_value_all_sessions():
    """Bảo vệ bất biến bảo toàn giá trị tiền + cổ phiếu trừ phí tại mọi phiên khi giá đi ngang."""
    T, N = 12, 3
    p = 25000.0
    ro = np.full((T, N), p, dtype=np.float64)
    rc = np.full((T, N), p, dtype=np.float64)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 5_000_000.0, dtype=np.float64)
    vol = np.zeros((T, N), dtype=np.float64)

    rng = np.random.default_rng(101)
    init_cash = 500_000_000.0
    w = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        raw_w = rng.uniform(0.0, 1.0, size=N)
        w[t] = raw_w / np.sum(raw_w)

    res = sim.simulate_vietnam(
        raw_open=ro,
        raw_close=rc,
        ceiling=ceil,
        floor=flr,
        adv20_shares=adv,
        target_weights=w,
        initial_cash=init_cash,
        broker_fee=0.0015,
        slippage=0.0,
        sell_tax=0.0010,
        advance_fee_rate=0.00035,
        min_advance_fee=50000.0,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=0.03,
        min_advance_notional=0.0,
    )

    total_fees_by_t = res["broker_fees"] + res["sell_tax"] + res["advance_fees"]
    assert np.allclose(init_cash - res["equity"], total_fees_by_t, atol=1.0)
    assert np.all(res["cash_history"] >= -1e-6)


# engine/vn_simulator.py:786,747,690,401-443
def test_simulate_vietnam_no_fills_outside_price_band_and_limit_locks():
    """Bảo vệ nguyên tắc không khớp mua tại giá trần, không khớp bán tại giá sàn và không khớp ngoài biên."""
    T, N = 8, 1
    ro = np.full((T, N), 50000.0, dtype=np.float64)
    rc = np.full((T, N), 50000.0, dtype=np.float64)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 5_000_000.0, dtype=np.float64)

    # Kịch bản A: t=1 mở cửa tại giá trần (53,500) -> mua phải bị từ chối hoàn toàn
    ro_ceil = ro.copy()
    ro_ceil[1, 0] = ceil[1, 0]
    w_buy = np.zeros((T, N), dtype=np.float64)
    w_buy[1, 0] = 1.0
    res_ceil = sim.simulate_vietnam(
        ro_ceil, rc, ceil, flr, adv, w_buy,
        initial_cash=200_000_000.0, clamp_and_round_exec_price=True
    )
    assert res_ceil["broker_fees"][1] == 0.0
    assert res_ceil["realized_daily_turnover"][1] == 0.0
    assert res_ceil["cash_history"][1] == 200_000_000.0

    # Kịch bản B: Mua hợp lệ ở t=1 gần trần (53,400) kèm trượt giá lớn -> giá khớp phải bị kẹp <= trần (53,500)
    ro_near_ceil = ro.copy()
    ro_near_ceil[1, 0] = 53400.0
    rc_near_ceil = rc.copy()
    rc_near_ceil[1, 0] = 53400.0
    w_one = np.zeros((T, N), dtype=np.float64)
    w_one[1, 0] = 0.5
    res_clamped_buy = sim.simulate_vietnam(
        ro_near_ceil, rc_near_ceil, ceil, flr, adv, w_one,
        initial_cash=200_000_000.0, broker_fee=0.0015, slippage=0.05,
        half_spread_floor=0.0, daily_vol=np.zeros((T, N)),
        clamp_and_round_exec_price=True, round_exec_adverse=False
    )
    cash_spent = 200_000_000.0 - res_clamped_buy["cash_history"][1]
    b_fee = res_clamped_buy["broker_fees"][1]
    gross_buy = cash_spent - b_fee
    shares_bought = round(gross_buy / ceil[1, 0])
    implied_exec = gross_buy / max(shares_bought, 1)
    assert implied_exec <= ceil[1, 0] + 1e-6
    assert implied_exec >= flr[1, 0] - 1e-6


# engine/vn_simulator.py:674,734,796,821,874
def test_simulate_vietnam_bought_shares_always_lot_100_multiples():
    """Bảo vệ quy tắc khối lượng mua luôn là bội số nguyên của lô 100 cổ phiếu trong mọi nhánh ngân sách."""
    T, N = 6, 2
    p = 27350.0  # giá không tròn triệu để ép phần lẻ cổ phiếu
    ro = np.full((T, N), p, dtype=np.float64)
    rc = np.full((T, N), p, dtype=np.float64)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 1_000_000.0, dtype=np.float64)
    vol = np.zeros((T, N), dtype=np.float64)

    w = np.zeros((T, N), dtype=np.float64)
    # t=1 mua 2 mã; t=2 tiếp tục mua thêm; t=3 bán mã 0 mua mã 1 (kích hoạt nhánh ngân sách hạn chế)
    w[1] = [0.63, 0.37]
    w[2] = [0.77, 0.23]
    w[3] = [0.0, 1.0]
    w[4] = [0.5, 0.5]

    for min_adv_notional in (0.0, 50_000_000.0):
        res = sim.simulate_vietnam(
            ro, rc, ceil, flr, adv, w,
            initial_cash=153_700_000.0,
            broker_fee=0.0015,
            slippage=0.0,
            sell_tax=0.0010,
            half_spread_floor=0.0,
            daily_vol=vol,
            max_adv_participation=0.03,
            min_advance_notional=min_adv_notional,
        )
        # Khi slippage=0, half_spread_floor=0: tổng tiền mua ngày t trừ tiền bán chưa về = cash_spent; với ngày chỉ mua (t=1,2):
        for t in (1, 2):
            fee_t = res["broker_fees"][t] - res["broker_fees"][t - 1]
            assert fee_t > 0.0
            qty_approx = fee_t / (p * 0.0015)
            assert abs(qty_approx - round(qty_approx)) < 1e-6
            assert round(qty_approx) % 100 == 0


# engine/vn_simulator.py:584-596,732-753
def test_simulate_vietnam_sold_or_unsettled_shares_cannot_be_resold_early():
    """Bảo vệ ràng buộc không thể bán cổ phiếu chưa về tài khoản hoặc bán lặp lại cổ phiếu đã bán."""
    T, N = 7, 1
    ro = np.array([[50000.0], [50000.0], [52000.0], [54000.0], [56000.0], [58000.0], [60000.0]])
    rc = ro.copy()
    ceil = np.full((T, N), 100000.0)
    flr = np.full((T, N), 10000.0)
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    w = np.zeros((T, N))
    w[1, 0] = 1.0  # mua ở t=1

    # Chế độ T+2 mặc định: mua t=1 -> khóa t=1, t=2 -> mở bán ở t=3; không thể bán thêm ở t=4 nếu đã bán hết ở t=3
    res_def = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0, broker_fee=0.0015, slippage=0.0, sell_tax=0.0010,
        half_spread_floor=0.0, daily_vol=vol, max_adv_participation=1.0, strict_t25_ato_lock=False
    )
    assert res_def["sell_tax"][1] == 0.0
    assert res_def["sell_tax"][2] == 0.0
    assert res_def["sell_tax"][3] > 0.0
    assert res_def["sell_tax"][4] == res_def["sell_tax"][3]
    assert res_def["cash_history"][5] == res_def["equity"][5]

    # Chế độ T+3 lịch sử (split_date sau ngày giao dịch): mua t=1 -> mở bán ở t=4 dù strict_t25_ato_lock=False
    dates_t3 = [datetime.date(2022, 8, 1) + datetime.timedelta(days=i) for i in range(T)]
    res_t3 = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0, broker_fee=0.0015, slippage=0.0, sell_tax=0.0010,
        half_spread_floor=0.0, daily_vol=vol, max_adv_participation=1.0,
        dates=dates_t3, split_date="2022-08-29", strict_t25_ato_lock=False
    )
    assert res_t3["sell_tax"][2] == 0.0
    assert res_t3["sell_tax"][3] == 0.0
    assert res_t3["sell_tax"][4] > 0.0


# engine/vn_simulator.py:612-637,647-648,933-934
def test_simulate_vietnam_stock_dividend_locked_until_listing_date():
    """Bảo vệ quy trình khóa cổ tức bằng cổ phiếu đến ngày niêm yết bổ sung rồi mới cho phép bán."""
    T, N = 8, 1
    ro = np.full((T, N), 50000.0)
    rc = np.full((T, N), 50000.0)
    # Giả lập điều chỉnh giá tham chiếu ngày hưởng quyền t=3 do chia 1:1 (giá giảm 50k -> 25k)
    ro[3:] = 25000.0
    rc[3:] = 25000.0
    ceil = np.full((T, N), 60000.0)
    flr = np.full((T, N), 10000.0)
    adv = np.full((T, N), 10_000_000.0)

    w = np.zeros((T, N))
    w[1, 0] = 1.0  # mua tại t=1 (2000 cp giá 50k = 100M)
    w[2, 0] = 1.0  # giữ qua ngày hưởng quyền t=3
    # từ t=3 muốn bán hết

    stock_ratio = np.zeros((T, N))
    stock_list_t = np.zeros((T, N))
    stock_ratio[3, 0] = 1.0     # chia 1:1 -> +2000 cp thưởng bị khóa
    stock_list_t[3, 0] = 6.0    # về tài khoản giao dịch ở phiên t=6

    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=100_000_000.0, broker_fee=0.0, slippage=0.0, sell_tax=0.001,
        half_spread_floor=0.0, daily_vol=np.zeros((T, N)), max_adv_participation=1.0,
        corp_action_stock_ratio=stock_ratio, corp_action_stock_list_t=stock_list_t
    )
    # Tổng tài sản bảo toàn qua ex-date t=3 nhờ locked_corp_action_shares_val
    assert res["locked_corp_action_shares_val"][3] == pytest.approx(2000.0 * 25000.0)
    assert res["equity"][3] == pytest.approx(100_000_000.0 - 2000.0 * 25000.0 * 0.001)
    # Trước t=6 vẫn còn cổ phiếu thưởng bị khóa (chỉ bán được 2000 cp gốc ở t=3)
    assert res["locked_corp_action_shares_val"][5] == pytest.approx(2000.0 * 25000.0)
    # Tại t=6 cổ phiếu thưởng mở khóa và bán nốt
    assert res["locked_corp_action_shares_val"][6] == 0.0
    assert res["sell_tax"][6] > res["sell_tax"][5]


# engine/vn_simulator.py:1106-1346
def test_simulate_vietnam_strict_no_look_ahead_on_future_perturbation():
    """Bảo vệ tính nhân quả chặt: thay đổi giá, khối lượng và tỷ trọng ở t > t0 không đổi kết quả tới t0."""
    rng = np.random.default_rng(2026)
    T, N, t0 = 25, 4, 12
    ro = 30000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(T, N)), axis=0))
    rc = ro * (1.0 + rng.normal(0.0, 0.005, size=(T, N)))
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 500_000.0)
    w = rng.uniform(0.0, 1.0, size=(T, N))
    w[0] = 0.0
    w[1:] /= np.sum(w[1:], axis=1, keepdims=True)

    ro_pert = ro.copy()
    rc_pert = rc.copy()
    adv_pert = adv.copy()
    w_pert = w.copy()
    ro_pert[t0 + 1:] *= 2.5
    rc_pert[t0 + 1:] *= 0.4
    adv_pert[t0 + 1:] *= 0.01
    w_pert[t0 + 1:] = 0.0

    # Cố ý truyền daily_vol=None để kiểm tra luôn bộ tính biến động nội bộ
    res_orig = sim.simulate_vietnam(ro, rc, ceil, flr, adv, w, initial_cash=5e8, daily_vol=None)
    res_pert = sim.simulate_vietnam(ro_pert, rc_pert, ceil, flr, adv_pert, w_pert, initial_cash=5e8, daily_vol=None)

    for key in ("equity", "cash_history", "broker_fees", "sell_tax", "advance_fees", "realized_daily_turnover"):
        assert np.allclose(res_orig[key][:t0 + 1], res_pert[key][:t0 + 1], atol=1e-9), f"Rò rỉ tương lai ở {key}"


# engine/vn_simulator.py:1194-1231
def test_simulate_vietnam_weekend_calendar_days_affect_advance_interest():
    """Bảo vệ cơ chế tính số ngày lịch thực tế qua cuối tuần khi tính lãi ứng trước tiền bán."""
    T, N = 6, 2
    p = 50000.0
    ro = np.full((T, N), p)
    rc = np.full((T, N), p)
    ceil, flr = sim.compute_bands_and_ticks(ro, "HOSE")
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))
    w = np.zeros((T, N))
    w[1] = [1.0, 0.0]
    w[2] = [1.0, 0.0]
    w[3] = [0.0, 1.0]  # Bán mã 0 và ứng tiền mua mã 1 tại t=3

    # Bộ ngày liên tục 1 ngày/phiên vs bộ ngày mà t=3 là Thứ Sáu -> t+2 vượt qua Thứ Bảy/Chủ Nhật (4 ngày)
    dates_flat = [datetime.date(2026, 1, 5) + datetime.timedelta(days=i) for i in range(T)]
    dates_wknd = [
        datetime.date(2026, 1, 6),  # T3
        datetime.date(2026, 1, 7),  # T4
        datetime.date(2026, 1, 8),  # T5
        datetime.date(2026, 1, 9),  # T6 (t=3 bán & ứng tiền T+2 -> sang T3 tuần sau = 4 ngày)
        datetime.date(2026, 1, 12), # T2
        datetime.date(2026, 1, 13), # T3
    ]

    r_flat = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=500_000_000.0, slippage=0.0, half_spread_floor=0.0, daily_vol=vol,
        max_adv_participation=1.0, dates=dates_flat, min_advance_fee=0.0
    )
    r_wknd = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=500_000_000.0, slippage=0.0, half_spread_floor=0.0, daily_vol=vol,
        max_adv_participation=1.0, dates=dates_wknd, min_advance_fee=0.0
    )
    assert r_wknd["advance_fees"][3] > r_flat["advance_fees"][3] * 1.5


# engine/vn_simulator.py:693-712,719-753
def test_floor_trapped_c1_exit_decrements_shares_sellable():
    """Bảo vệ nguyên tắc không bán trùng một lô cổ phiếu khi thoát hàng kẹt sàn đồng thời hết ân hạn PIT."""
    T, N = 9, 1
    ro = np.full((T, N), 50000.0)
    rc = np.full((T, N), 50000.0)
    ceil = np.full((T, N), 53500.0)
    flr = np.full((T, N), 46500.0)
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    # Kẹt sàn từ t=3..6 (4 phiên), mở cửa lại ở t=7 (50k)
    ro[3:7, 0] = 46500.0
    rc[3:7, 0] = 46500.0

    # Bị loại khỏi PIT universe từ t=3..8 -> tại t=7 liquidation_grace_counter == 5
    pit = np.ones((T, N), dtype=bool)
    pit[3:, 0] = False

    w = np.zeros((T, N))
    w[1, 0] = 1.0  # mua tại t=1 (settle ở t=3)
    w[2, 0] = 1.0

    init_cash = 100_000_000.0
    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=init_cash,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        in_pit_universe=pit,
        penalty_rate_per_day=0.0,
    )

    # Giá chỉ biến động 50k -> 46.5k -> 50k, không phí, tài sản cuối không thể vượt quá vốn gốc 100M
    assert res["equity"][-1] <= init_cash + 1.0


# ------------------------------------------------------------------------------
# 4. STRATEGY EVALUATION, DSR & DYNAMIC HEDGE EDGE CASES
# ------------------------------------------------------------------------------

# engine/vn_simulator.py:1527-1662
def test_evaluate_strategy_edge_cases_and_kill_switch():
    """Bảo vệ chỉ số đánh giá chiến lược ở các trường hợp T<2, không có ngày lỗ và kích hoạt kill-switch."""
    assert sim.evaluate_strategy(np.array([1e9]), np.zeros((1, 2))) == {}

    # Toàn bộ ngày tăng trưởng dương -> downside_returns rỗng -> fallback ann_vol nên Sortino == Sharpe
    eq_up = np.array([100.0, 101.0, 102.0, 103.5, 105.0])
    w_dummy = np.zeros((5, 2))
    bm_up = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
    m_up = sim.evaluate_strategy(eq_up, w_dummy, benchmark_equity_curve=bm_up, initial_capital=100.0)
    assert m_up["max_drawdown"] == pytest.approx(0.0)
    assert m_up["sortino_ratio"] == pytest.approx(m_up["sharpe_ratio"])
    assert not m_up["kill_switch_triggered"]

    # MaxDD > 25% và Implementation Shortfall > 30% đều kích hoạt kill-switch
    eq_dd = np.array([100.0, 110.0, 70.0, 80.0])  # DD = 40/110 = 36.36% > 25%
    naive_eq = np.array([100.0, 110.0, 120.0, 140.0])
    m_ks = sim.evaluate_strategy(eq_dd, w_dummy, naive_equity_curve=naive_eq, initial_capital=100.0)
    assert m_ks["max_drawdown"] > 0.25
    assert m_ks["kill_switch_triggered"] is True
    assert len(m_ks["kill_switch_reasons"]) == 2


# engine/vn_simulator.py:33-157,228-332
def test_dsr_and_newey_west_degenerate_inputs(tmp_path):
    """Bảo vệ compute_deflated_sharpe_ratio, compute_dsr_from_registry và Newey-West với chuỗi ngắn hoặc phẳng."""
    # T < 10 và chuỗi phẳng
    d_short = sim.compute_deflated_sharpe_ratio(np.array([0.01, 0.02, -0.01]))
    assert d_short["deflated_sharpe_ratio"] == 0.0 and d_short["p_value"] == 1.0
    d_flat = sim.compute_deflated_sharpe_ratio(np.full(30, 0.001))
    assert d_flat["deflated_sharpe_ratio"] == 0.0 and d_flat["p_value"] == 1.0

    # Chuỗi đủ dài -> dsr + p_value == 1.0 và n_eff >= 1.0
    rng = np.random.default_rng(11)
    rets = rng.normal(0.001, 0.01, size=60)
    corr_mat = np.eye(4)
    d_mat = sim.compute_deflated_sharpe_ratio(rets, n_trials=4, trial_corr_matrix=corr_mat)
    assert 0.0 <= d_mat["deflated_sharpe_ratio"] <= 1.0
    assert d_mat["deflated_sharpe_ratio"] + d_mat["p_value"] == pytest.approx(1.0)
    assert d_mat["n_eff"] == pytest.approx(4.0)

    # Newey-West: rỗng và chuỗi hằng
    assert sim.compute_newey_west_sharpe_se(np.array([])) == 0.0
    se_const = sim.compute_newey_west_sharpe_se(np.full(10, 0.001), lags=20)
    assert np.isfinite(se_const) and se_const > 0.0

    # Registry không tồn tại vẫn tính hợp lệ
    res_missing = sim.compute_dsr_from_registry(rets, registry_file=str(tmp_path / "missing.jsonl"))
    assert res_missing["n_trials"] == 1
    assert 0.0 <= res_missing["deflated_sharpe_ratio"] <= 1.0


# engine/vn_simulator.py:1411-1520
def test_calculate_dynamic_vn30f_hedge_edge_branches():
    """Bảo vệ các nhánh biên của calculate_dynamic_vn30f_hedge: vốn<=0, lịch sử ngắn, giải phóng hedge."""
    # Degenerate portfolio_value <= 0
    z_res = sim.calculate_dynamic_vn30f_hedge(0.0, 1.0, 1250.0, 0.5)
    assert z_res["contracts_to_short"] == 0 and z_res["h_actual"] == 0.0

    # Lịch sử ngắn (<50) mặc định regime M=0 -> theta_low=0.55; với lambda=0.3 -> h_star=0
    # Khi h_star==0 và prev_h=0.20 <= 0.25 -> giải phóng hedge về 0.0
    rel = sim.calculate_dynamic_vn30f_hedge(
        portfolio_value=1.0e9,
        beta_cw=1.0,
        futures_price=1250.0,
        lambda_contagion=0.30,
        vn30_close_history=np.full(20, 1200.0),
        prev_h=0.20,
    )
    assert rel["market_regime_M"] == 0
    assert rel["h_target"] == 0.0
    assert rel["h_actual"] == 0.0
    assert rel["deadband_triggered"] is True


# engine/vn_simulator.py:666-712,719-753
def test_floor_trapped_exit_normal_path_no_double_sell():
    """Bảo vệ: lệnh bán bị kẹt sàn rồi thoát sàn không bị bán trùng lần nữa bởi tín hiệu bán thường (không cần ân hạn PIT)."""
    T, N = 9, 1
    ro = np.full((T, N), 50000.0)
    rc = np.full((T, N), 50000.0)
    ceil = np.full((T, N), 53500.0)
    flr = np.full((T, N), 46500.0)
    adv = np.full((T, N), 10_000_000.0)
    vol = np.zeros((T, N))

    # Kẹt sàn t=4..5, mở cửa lại ở t=6
    ro[4:6, 0] = 46500.0
    rc[4:6, 0] = 46500.0

    w = np.zeros((T, N))
    w[1:4, 0] = 1.0  # giữ đến t=3, tín hiệu bán từ t=4 trở đi

    init_cash = 100_000_000.0
    res = sim.simulate_vietnam(
        ro, rc, ceil, flr, adv, w,
        initial_cash=init_cash,
        broker_fee=0.0,
        slippage=0.0,
        sell_tax=0.0,
        half_spread_floor=0.0,
        daily_vol=vol,
        max_adv_participation=1.0,
        penalty_rate_per_day=0.0,
    )
    assert res["equity"].max() <= init_cash + 1.0  # không có ngày nào tài sản tăng ảo
