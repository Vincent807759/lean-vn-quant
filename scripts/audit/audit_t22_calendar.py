"""
scripts/audit/audit_t22_calendar.py
Kiểm toán độc lập lịch giao dịch VNINDEX và các mã cổ phiếu trên CafeF và Entrade.
Lưu toàn bộ JSON thô vào scripts/audit/results/t22_calendar_audit.json và in kèm SHA-256 hash.
"""
import os
import json
import hashlib
import datetime
import requests
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, "t22_calendar_audit.json")

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

dates_to_check = [
    "2019-12-31",
    "2020-03-11",
    "2020-12-31",
    "2021-12-31",
    "2025-02-03",
    "2026-01-29",
    "2026-01-30"
]

audit_results = {
    "generated_at": datetime.datetime.now(VN_TZ).isoformat(),
    "dates_audited": {}
}

print("=== BẮT ĐẦU KIỂM TOÁN T22: LỊCH GIAO DỊCH VÀ DỮ LIỆU THÔ ===")

for d in dates_to_check:
    dp = d.split('-')
    d_m_y = f"{dp[1]}/{dp[2]}/{dp[0]}"
    
    # 1. CafeF VNINDEX
    url_cf_vni = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol=VNINDEX&StartDate={d_m_y}&EndDate={d_m_y}&PageIndex=1&PageSize=5"
    r_cf_vni = s.get(url_cf_vni, timeout=15).json()
    items_cf_vni = r_cf_vni.get("Data", {}).get("Data", [])
    
    # 2. CafeF HPG
    url_cf_hpg = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol=HPG&StartDate={d_m_y}&EndDate={d_m_y}&PageIndex=1&PageSize=5"
    r_cf_hpg = s.get(url_cf_hpg, timeout=15).json()
    items_cf_hpg = r_cf_hpg.get("Data", {}).get("Data", [])
    
    # 3. CafeF VCB
    url_cf_vcb = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol=VCB&StartDate={d_m_y}&EndDate={d_m_y}&PageIndex=1&PageSize=5"
    r_cf_vcb = s.get(url_cf_vcb, timeout=15).json()
    items_cf_vcb = r_cf_vcb.get("Data", {}).get("Data", [])

    # 4. Entrade HPG on that date
    dt_day = datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=VN_TZ)
    t_start = int(dt_day.timestamp())
    t_end = int((dt_day + datetime.timedelta(days=1)).timestamp())
    url_en_hpg = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?symbol=HPG&resolution=1D&from={t_start}&to={t_end}"
    r_en_hpg = requests.get(url_en_hpg, timeout=15).json()
    en_has_hpg = bool(r_en_hpg.get("t") and len(r_en_hpg["t"]) > 0)
    en_hpg_bar = None
    if en_has_hpg:
        en_hpg_bar = {
            "t": r_en_hpg["t"][0],
            "o": r_en_hpg["o"][0],
            "h": r_en_hpg["h"][0],
            "l": r_en_hpg["l"][0],
            "c": r_en_hpg["c"][0],
            "v": r_en_hpg["v"][0]
        }

    # 5. Entrade VNINDEX on that date
    url_en_vni = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?symbol=VNINDEX&resolution=1D&from={t_start}&to={t_end}"
    r_en_vni = requests.get(url_en_vni, timeout=15).json()
    en_has_vni = bool(r_en_vni.get("t") and len(r_en_vni["t"]) > 0)
    en_vni_bar = None
    if en_has_vni:
        en_vni_bar = {
            "t": r_en_vni["t"][0],
            "o": r_en_vni["o"][0],
            "h": r_en_vni["h"][0],
            "l": r_en_vni["l"][0],
            "c": r_en_vni["c"][0],
            "v": r_en_vni["v"][0]
        }

    audit_results["dates_audited"][d] = {
        "cafef_vnindex": items_cf_vni[0] if items_cf_vni else None,
        "cafef_hpg": items_cf_hpg[0] if items_cf_hpg else None,
        "cafef_vcb": items_cf_vcb[0] if items_cf_vcb else None,
        "entrade_hpg": en_hpg_bar,
        "entrade_vnindex": en_vni_bar
    }

# Ghi ra tệp JSON
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, ensure_ascii=False, indent=2)

# Tính SHA-256
with open(OUT_JSON, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"-> Ghi thành công tệp: {OUT_JSON}")
print(f"-> SHA-256: {sha256}")
print("\n=== KẾT QUẢ TRÍCH XUẤT NGUYÊN BẢN TỪNG PHIÊN ===")
for d in dates_to_check:
    rec = audit_results["dates_audited"][d]
    cf_vni = rec["cafef_vnindex"]
    en_hpg = rec["entrade_hpg"]
    cf_hpg = rec["cafef_hpg"]
    print(f"[{d}]")
    if cf_vni:
        print(f"   CafeF VNINDEX: Close={cf_vni.get('GiaDongCua')}, MatchVol={cf_vni.get('KhoiLuongKhopLenh')}, PTVol={cf_vni.get('KLThoaThuan')}")
    else:
        print(f"   CafeF VNINDEX: KHÔNG CÓ DỮ LIỆU (NOT FOUND)")
    print(f"   CafeF HPG    : {'CÓ' if cf_hpg else 'KHÔNG CÓ'}")
    print(f"   Entrade HPG  : {'CÓ, Vol=' + str(en_hpg.get('v')) if en_hpg else 'KHÔNG CÓ'}")
    print(f"   Entrade VNI  : {'CÓ' if rec['entrade_vnindex'] else 'KHÔNG CÓ'}")
