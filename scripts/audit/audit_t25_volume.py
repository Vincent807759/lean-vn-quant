"""
scripts/audit/audit_t25_volume.py
Kiểm toán độc lập chênh lệch Volume giữa Entrade và CafeF trên 10 mã cổ phiếu lớn:
HPG, VCB, SSI, FPT, VNM, MWG, VIC, VHM, MSN, BMP.
Đối chiếu:
1. diff = Entrade_Vol - CafeF_Match_Vol
2. So sánh diff với KLThoaThuan (Put-through Volume)
3. Kiểm tra thành phần lô lẻ: diff % 100 != 0
4. Phân bố lệch trước mốc 12/09/2022 (trước khi mở lại lô lẻ) vs sau 12/09/2022.
5. Đưa ra định nghĩa Volume duy nhất chuẩn hóa cho toàn bộ hệ thống.
Lưu JSON gốc vào scripts/audit/results/t25_volume_audit.json kèm SHA-256 hash.
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
OUT_JSON = os.path.join(RESULTS_DIR, "t25_volume_audit.json")

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))
TICKERS = ["HPG", "VCB", "SSI", "FPT", "VNM", "MWG", "VIC", "VHM", "MSN", "BMP"]

print("=== BẮT ĐẦU KIỂM TOÁN T25: ĐỐI CHIẾU VOLUME 10 MÃ VS THỎA THUẬN & LÔ LẺ ===")

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

t_from = int(datetime.datetime(2018, 1, 1, tzinfo=VN_TZ).timestamp())
t_to = int(datetime.datetime(2026, 10, 26, tzinfo=VN_TZ).timestamp())

cutoff_odd_lot = "2022-09-12"  # Mốc HOSE mở lại giao dịch lô lẻ

audit_summary = {}
raw_details = {}

for ticker in TICKERS:
    # 1. Kéo Entrade
    url_en = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?symbol={ticker}&resolution=1D&from={t_from}&to={t_to}"
    r_en = requests.get(url_en, timeout=15).json()
    en_by_date = {}
    for i in range(len(r_en.get("t", []))):
        d_str = datetime.datetime.fromtimestamp(r_en["t"][i], tz=VN_TZ).strftime("%Y-%m-%d")
        en_by_date[d_str] = float(r_en["v"][i])

    # 2. Kéo CafeF (2018 - 2026)
    cf_by_date = {}
    for year in range(2018, 2027):
        for m_start, m_end in [("01/01", "06/30"), ("07/01", "12/31")]:
            page = 1
            while True:
                url_cf = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol={ticker}&StartDate={m_start}/{year}&EndDate={m_end}/{year}&PageIndex={page}&PageSize=100"
                resp = s.get(url_cf, timeout=15).json()
                time.sleep(0.3)
                items = resp.get("Data", {}).get("Data", [])
                if not items:
                    break
                for it in items:
                    dp = it["Ngay"].split("/")
                    d_iso = f"{dp[2]}-{dp[1]}-{dp[0]}"
                    cf_by_date[d_iso] = {
                        "match_vol": float(it["KhoiLuongKhopLenh"]),
                        "pt_vol": float(it["KLThoaThuan"])
                    }
                if len(items) < 30:
                    break
                page += 1

    common = sorted(set(en_by_date.keys()) & set(cf_by_date.keys()))
    
    total_days = len(common)
    mismatch_days = 0
    diff_equals_pt = 0
    diff_has_oddlot = 0
    mismatch_pre_cutoff = 0
    mismatch_post_cutoff = 0
    
    ticker_diffs = []
    
    for d in common:
        v_en = en_by_date[d]
        v_cf_match = cf_by_date[d]["match_vol"]
        v_cf_pt = cf_by_date[d]["pt_vol"]
        
        diff = v_en - v_cf_match
        if abs(diff) > 0:
            mismatch_days += 1
            is_odd = (diff % 100 != 0)
            if is_odd:
                diff_has_oddlot += 1
            
            # Kiểm tra xem diff có bằng chính xác KL thỏa thuận không
            if abs(diff - v_cf_pt) < 1.0:
                diff_equals_pt += 1
                
            if d < cutoff_odd_lot:
                mismatch_pre_cutoff += 1
            else:
                mismatch_post_cutoff += 1
                
            ticker_diffs.append({
                "date": d,
                "entrade_vol": v_en,
                "cafef_match_vol": v_cf_match,
                "cafef_pt_vol": v_cf_pt,
                "diff": diff,
                "diff_pct": (diff / v_cf_match * 100.0) if v_cf_match > 0 else 0.0,
                "is_odd_lot": is_odd,
                "equals_pt": (abs(diff - v_cf_pt) < 1.0)
            })
            
    pct_mismatch = (mismatch_days / total_days * 100.0) if total_days > 0 else 0.0
    
    audit_summary[ticker] = {
        "total_days": total_days,
        "mismatch_days": mismatch_days,
        "pct_mismatch": pct_mismatch,
        "diff_equals_pt_count": diff_equals_pt,
        "diff_has_oddlot_count": diff_has_oddlot,
        "mismatch_pre_2022_09_12": mismatch_pre_cutoff,
        "mismatch_post_2022_09_12": mismatch_post_cutoff
    }
    raw_details[ticker] = ticker_diffs[:50]  # Lưu mẫu 50 ngày lệch đầu tiên
    print(f"[{ticker:<4}] Tổng: {total_days:>4} | Lệch: {mismatch_days:>3} ({pct_mismatch:>5.2f}%) | Bằng Thỏa thuận: {diff_equals_pt:>2} | Chứa Lô lẻ: {diff_has_oddlot:>3} | Trước 12/09/22: {mismatch_pre_cutoff:>2} | Sau 12/09/22: {mismatch_post_cutoff:>3}")

full_output = {
    "generated_at": datetime.datetime.now(VN_TZ).isoformat(),
    "summary": audit_summary,
    "details_sample": raw_details
}

with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
    json.dump(full_output, f, ensure_ascii=False, indent=2)
    f.write("\n")

with open(OUT_JSON, "rb") as f:
    raw_bytes = f.read()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

print(f"\n-> Ghi thành công tệp: {OUT_JSON}")
print(f"-> SHA-256: {sha256}")
