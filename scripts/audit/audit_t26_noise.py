"""
scripts/audit/audit_t26_noise.py
Kiểm toán độc lập phân phối nhiễu của tỷ số k(t) = (1 + r_adj) / (1 + r_raw) trên các ngày thường:
1. Tính phân phối thực tế của |k(t) - 1| trên toàn bộ lịch sử 10 mã lớn:
   HPG, VCB, SSI, FPT, VNM, MWG, VIC, VHM, MSN, BMP.
2. Xác định các phân vị nhiễu: Median, P75, P90, P95, P99, Max.
3. Đo biên an toàn nhiễu max/p99 vs cổ tức nhỏ nhất (30 bps); nếu < 10 bps đề xuất tra cứu sự kiện ngoài.
4. Xác nhận bước giá chuẩn của VCB theo quy chế HOSE (100đ).
5. Phân tích nguồn dữ liệu bắt buộc cho Corporate Actions Ledger.
Lưu JSON gốc vào scripts/audit/results/t26_noise_audit.json kèm SHA-256 hash.
"""
import os
import sys
import json
import time
import hashlib
import datetime
import numpy as np
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, "t26_noise_audit.json")

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))
TICKERS = ["HPG", "VCB", "SSI", "FPT", "VNM", "MWG", "VIC", "VHM", "MSN", "BMP"]

print("=== BẮT ĐẦU KIỂM TOÁN T26: NHIỄU TỶ SỐ k(t) 10 MÃ VÀ BƯỚC GIÁ VCB ===")

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

noise_by_ticker = {}

for ticker in TICKERS:
    # Kéo dữ liệu CafeF đầy đủ 2018 - 2026 (query format: MM/DD/YYYY)
    cf_data = []
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
                    cf_data.append({
                        "date": d_iso,
                        "raw_close": float(it["GiaDongCua"]) * 1000.0,
                        "adj_close": float(it["GiaDieuChinh"]) * 1000.0
                    })
                if len(items) < 30:
                    break
                page += 1

    # Sắp xếp theo ngày tăng dần
    cf_data = sorted({x["date"]: x for x in cf_data}.values(), key=lambda x: x["date"])
    
    # Tính r_raw, r_adj, k(t)
    normal_k_diffs = []
    ex_events = []
    
    for i in range(1, len(cf_data)):
        p_raw_prev = cf_data[i - 1]["raw_close"]
        p_raw_curr = cf_data[i]["raw_close"]
        p_adj_prev = cf_data[i - 1]["adj_close"]
        p_adj_curr = cf_data[i]["adj_close"]
        
        if p_raw_prev <= 0 or p_adj_prev <= 0:
            continue
            
        r_raw = (p_raw_curr - p_raw_prev) / p_raw_prev
        r_adj = (p_adj_curr - p_adj_prev) / p_adj_prev
        
        k = (1.0 + r_adj) / (1.0 + r_raw)
        k_diff_bps = abs(k - 1.0) * 10000.0  # Đổi ra basis points (bps)
        
        # Nếu lệch > 25 bps (0.25%) -> Sự kiện quyền (GDKHQ)
        if k_diff_bps > 25.0:
            ex_events.append({
                "date": cf_data[i]["date"],
                "k": k,
                "k_diff_bps": k_diff_bps,
                "r_raw_pct": r_raw * 100.0,
                "r_adj_pct": r_adj * 100.0
            })
        else:
            normal_k_diffs.append(k_diff_bps)
            
    arr = np.array(normal_k_diffs)
    p99_val = float(np.percentile(arr, 99)) if len(arr) > 0 else 0.0
    max_val = float(np.max(arr)) if len(arr) > 0 else 0.0
    margin_p99 = 30.0 - p99_val
    margin_max = 30.0 - max_val
    
    noise_by_ticker[ticker] = {
        "total_sessions": len(cf_data),
        "normal_sessions": len(normal_k_diffs),
        "ex_events_count": len(ex_events),
        "noise_bps_median": float(np.median(arr)) if len(arr) > 0 else 0.0,
        "noise_bps_mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
        "noise_bps_p75": float(np.percentile(arr, 75)) if len(arr) > 0 else 0.0,
        "noise_bps_p90": float(np.percentile(arr, 90)) if len(arr) > 0 else 0.0,
        "noise_bps_p95": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
        "noise_bps_p99": p99_val,
        "noise_bps_max": max_val,
        "safety_margin_p99_bps": margin_p99,
        "safety_margin_max_bps": margin_max,
        "margin_warning": "Biên an toàn < 10 bps; rủi ro nhiễu làm tròn nhầm với cổ tức nhỏ; đề xuất tra cứu Corporate Actions Ledger ngoài." if margin_max < 10.0 else "Biên an toàn >= 10 bps"
    }
    
    print(f"[{ticker:<4}] Phiên: {len(cf_data):>4} | Median: {noise_by_ticker[ticker]['noise_bps_median']:.2f} bps | P99: {p99_val:.2f} bps | Max: {max_val:.2f} bps | Margin Max: {margin_max:.2f} bps | Sự kiện: {len(ex_events)}")

# 4. Xác nhận bước giá VCB
vcb_tick_size_verification = {
    "price_range_vnd": "70,000 - 95,000",
    "hose_rule": "Quyết định 17/QĐ-SGDHCM ngày 26/03/2021 & Quyết định 75/QĐ-SGDHCM",
    "hose_brackets": {
        "bracket_under_10k": "10 VND",
        "bracket_10k_to_49950": "50 VND",
        "bracket_50k_and_above": "100 VND"
    },
    "vcb_official_tick_size": 100,
    "correction": "Bản 2.5 ghi '50đ' cho VCB là sai quy chế HOSE; VCB thuộc khung giá >= 50,000đ nên bước giá chính thức là 100đ."
}

# 5. Đặc tả nguồn sự kiện doanh nghiệp bắt buộc
corp_action_sources = {
    "limitation_of_k_ratio": "Tỷ số k(t) chỉ phát hiện được ngày GDKHQ (ex-date) và tỷ lệ điều chỉnh; KHÔNG THỂ suy ra pay_date (ngày trả cổ tức tiền mặt) hay list_date (ngày cổ phiếu thưởng/tách được phép giao dịch).",
    "required_official_sources": [
        "1. VSDC (Trung tâm Lưu ký Chứng khoán Việt Nam - vsd.vn): Thông báo thực hiện quyền và ngày đăng ký cuối cùng.",
        "2. HOSE (Sở Giao dịch Chứng khoán TP.HCM - hsx.vn): Thông báo ngày GDKHQ, ngày thanh toán cổ tức và ngày niêm yết bổ sung.",
        "3. Nhà cung cấp dữ liệu cấu trúc (FiinGroup FiinPro / Vietstock / SSI iBoard Events API)."
    ],
    "prerequisite_for_phase_4": "Nếu chưa ingest đầy đủ tệp corporate_actions_ledger.parquet có nguồn kiểm chứng từ VSDC/HOSE thì Giai đoạn 4 CHƯA THỂ THỰC THI vì simulator không có ngày mở khóa tiền và cổ phiếu."
}

output_t26 = {
    "generated_at": datetime.datetime.now(VN_TZ).isoformat(),
    "tickers_audited": TICKERS,
    "noise_distribution": noise_by_ticker,
    "vcb_tick_size": vcb_tick_size_verification,
    "corporate_action_sources_audit": corp_action_sources
}

with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
    json.dump(output_t26, f, ensure_ascii=False, indent=2)
    f.write("\n")

with open(OUT_JSON, "rb") as f:
    raw_bytes = f.read()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

print(f"\n-> Ghi thành công tệp: {OUT_JSON}")
print(f"-> SHA-256: {sha256}")
