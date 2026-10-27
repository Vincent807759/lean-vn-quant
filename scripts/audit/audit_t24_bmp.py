"""
scripts/audit/audit_t24_bmp.py
Kiểm toán độc lập toàn bộ lịch sử BMP giữa Entrade và CafeF (2018 - 2026):
- Nạp đầy đủ lịch sử với phân trang chuẩn định dạng MM/DD/YYYY trên CafeF.
- So sánh tỷ số R(t) = P_adj_Entrade(t) / P_adj_CafeF(t).
- Xác định toàn bộ các ngày GDKHQ (Ex-Dividend Dates) của BMP từ 2018 đến 2026.
- Đối chiếu: Tỷ suất ngầm định theo nhà cung cấp (Vendor-Implied Econ Return) vs CafeF Adj Return vs Entrade Adj Return.
- Lưu ý phương pháp luận (Điều chỉnh A2): Mức cổ tức ước lượng (ví dụ 2,800 VND ngày 14/05/2018 và 5,000 VND ngày 19/05/2023)
  được suy ra từ hệ số điều chỉnh của CafeF, phản ánh tỷ suất ngầm định của nhà cung cấp, không phải số liệu nghị quyết
  chính thức độc lập từ VSDC/HOSE. Không kết luận nguồn nào đúng hơn.
- Giải thích chi tiết sự kiện ngày 20/11/2023 (0.4513% vs 0.4522%).
Lưu JSON gốc vào scripts/audit/results/t24_bmp_audit.json và in kèm SHA-256 hash.
"""
import os
import sys
import json
import time
import hashlib
import datetime
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, "t24_bmp_audit.json")

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))

print("=== BẮT ĐẦU KIỂM TOÁN T24: BÓC TÁCH LỆCH BMP VÀ TỶ SUẤT KINH TẾ ===")

# 1. Kéo toàn bộ lịch sử BMP từ Entrade
t_from = int(datetime.datetime(2018, 1, 1, tzinfo=VN_TZ).timestamp())
t_to = int(datetime.datetime(2026, 10, 26, tzinfo=VN_TZ).timestamp())
url_en = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?symbol=BMP&resolution=1D&from={t_from}&to={t_to}"
r_en = requests.get(url_en, timeout=15).json()

en_by_date = {}
for i in range(len(r_en.get("t", []))):
    dt_str = datetime.datetime.fromtimestamp(r_en["t"][i], tz=VN_TZ).strftime("%Y-%m-%d")
    en_by_date[dt_str] = {
        "open": float(r_en["o"][i]) * 1000.0,
        "high": float(r_en["h"][i]) * 1000.0,
        "low": float(r_en["l"][i]) * 1000.0,
        "close": float(r_en["c"][i]) * 1000.0,
        "volume": float(r_en["v"][i])
    }

# 2. Kéo toàn bộ lịch sử BMP từ CafeF qua phân trang định dạng MM/DD/YYYY
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

cf_by_date = {}
for year in range(2018, 2027):
    # Mỗi năm chia 2 nửa để tránh quá tải phân trang
    for m_start, m_end in [("01/01", "06/30"), ("07/01", "12/31")]:
        page = 1
        while True:
            url_cf = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol=BMP&StartDate={m_start}/{year}&EndDate={m_end}/{year}&PageIndex={page}&PageSize=100"
            resp = s.get(url_cf, timeout=15).json()
            time.sleep(0.3)
            items = resp.get("Data", {}).get("Data", [])
            if not items:
                break
            for it in items:
                dp = it["Ngay"].split("/")
                d_iso = f"{dp[2]}-{dp[1]}-{dp[0]}"
                cf_by_date[d_iso] = {
                    "raw_close": float(it["GiaDongCua"]) * 1000.0,
                    "raw_open": float(it["GiaMoCua"]) * 1000.0,
                    "adj_close": float(it["GiaDieuChinh"]) * 1000.0,
                    "volume": float(it["KhoiLuongKhopLenh"]),
                    "pt_volume": float(it["KLThoaThuan"])
                }
            if len(items) < 30:
                break
            page += 1

common_dates = sorted(set(en_by_date.keys()) & set(cf_by_date.keys()))
print(f"Tổng số phiên chung giữa Entrade và CafeF: {len(common_dates)}")

# 3. Phân tích chuỗi tỷ số R(t) = Entrade_adj / CafeF_adj
ratios = []
for d in common_dates:
    en_c = en_by_date[d]["close"]
    cf_adj = cf_by_date[d]["adj_close"]
    cf_raw = cf_by_date[d]["raw_close"]
    r = en_c / cf_adj
    ratios.append((d, r, en_c, cf_adj, cf_raw))

# 4. Tìm các ngày GDKHQ (Ex-Dividend Dates)
ex_dates = []
for i in range(1, len(ratios)):
    prev_d, prev_r, prev_en, prev_cf_adj, prev_cf_raw = ratios[i - 1]
    curr_d, curr_r, curr_en, curr_cf_adj, curr_cf_raw = ratios[i]

    raw_ret = (curr_cf_raw - prev_cf_raw) / prev_cf_raw
    adj_ret_cf = (curr_cf_adj - prev_cf_adj) / prev_cf_adj
    adj_ret_en = (curr_en - prev_en) / prev_en
    r_step = (curr_r / prev_r) - 1.0

    # Nếu raw return lệch so với adj return > 1.5% -> Có sự kiện doanh nghiệp (cổ tức)
    if abs(raw_ret - adj_ret_cf) > 0.015:
        # Cổ tức ước lượng theo nhà cung cấp (vendor-implied)
        div_est = prev_cf_raw - curr_cf_raw + (prev_cf_raw * adj_ret_cf)
        div_est_rounded = round(div_est / 100.0) * 100.0
        vendor_implied_econ_ret = (curr_cf_raw + div_est_rounded - prev_cf_raw) / prev_cf_raw
        
        ex_dates.append({
            "ex_date": curr_d,
            "prev_date": prev_d,
            "div_estimate": div_est_rounded,
            "prev_raw": prev_cf_raw,
            "curr_raw": curr_cf_raw,
            "raw_return_pct": raw_ret * 100.0,
            "vendor_implied_econ_return_pct": vendor_implied_econ_ret * 100.0,
            "cafef_adj_return_pct": adj_ret_cf * 100.0,
            "entrade_adj_return_pct": adj_ret_en * 100.0,
            "diff_cafef_vs_vendor_implied_bps": (adj_ret_cf - vendor_implied_econ_ret) * 10000.0,
            "diff_entrade_vs_vendor_implied_bps": (adj_ret_en - vendor_implied_econ_ret) * 10000.0,
            "diff_entrade_vs_cafef_bps": (adj_ret_en - adj_ret_cf) * 10000.0,
            "ratio_prev": prev_r,
            "ratio_curr": curr_r,
            "ratio_step_pct": r_step * 100.0
        })

# 5. Bóc tách phiên 2023-11-20 cụ thể
p_en_2023_11_17 = en_by_date.get("2023-11-17", {}).get("close")
p_en_2023_11_20 = en_by_date.get("2023-11-20", {}).get("close")
en_ret_current_data = (p_en_2023_11_20 - p_en_2023_11_17) / p_en_2023_11_17 if (p_en_2023_11_17 and p_en_2023_11_20) else None

audit_output = {
    "generated_at": datetime.datetime.now(VN_TZ).isoformat(),
    "methodology_note": "Mức cổ tức ước lượng div_estimate được suy ra từ chính hệ số điều chỉnh của CafeF, phản ánh tỷ suất kinh tế ngầm định theo nhà cung cấp (vendor-implied economic return). Do chưa có dữ liệu nghị quyết chia cổ tức chính thức độc lập từ VSDC/HOSE, không thể kết luận CafeF hay Entrade đúng hơn.",
    "ticker": "BMP",
    "total_common_dates": len(common_dates),
    "ex_dividend_events": ex_dates,
    "case_2023_11_20_reconciliation": {
        "entrade_close_2023_11_17_vnd": p_en_2023_11_17,
        "entrade_close_2023_11_20_vnd": p_en_2023_11_20,
        "entrade_return_from_current_api": en_ret_current_data,
        "explanation": "Trong Bản 2.6, công thức viết '(76,120 - 75,778)/75,778 = +0.4522%' là lỗi đánh máy: phép tính (76120 - 75778)/75778 ra 0.4513%, trong khi 0.4522% là kết quả từ chuỗi giá Entrade mới (71,090 - 70,770)/70,770 = 0.0045217 = 0.4522%."
    }
}

with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
    json.dump(audit_output, f, ensure_ascii=False, indent=2)
    f.write("\n")

with open(OUT_JSON, "rb") as f:
    raw_bytes = f.read()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

print(f"-> Ghi thành công tệp: {OUT_JSON}")
print(f"-> SHA-256: {sha256}")
