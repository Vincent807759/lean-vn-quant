"""
scripts/audit/audit_t22_calendar.py
Kiểm toán độc lập lịch giao dịch VNINDEX và các mã cổ phiếu trên CafeF và Entrade:
- Định dạng truy vấn CafeF chuẩn: MM/DD/YYYY.
- Xử lý biên truy vấn: StartDate = d, EndDate = d + 1 ngày để đảm bảo bắt đủ bản ghi; sau đó lọc chính xác theo Ngay == d (DD/MM/YYYY).
  (Lưu ý: Giả thuyết về thành phần giờ "15:00:00" trong CSDL CafeF chỉ là suy luận thực nghiệm).
- Bổ sung 3 ngày giao dịch bình thường (2026-10-21, 2026-10-22, 2026-10-23) làm nhóm đối chứng độc lập.
- Lưu toàn bộ JSON thô nguyên bản vào scripts/audit/results/t22_calendar_audit.json và in kèm SHA-256 hash.
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
OUT_JSON = os.path.join(RESULTS_DIR, "t22_calendar_audit.json")

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

dates_to_check = [
    # 3 ngày đối chứng thường gần nhất
    "2026-10-21",
    "2026-10-22",
    "2026-10-23",
    # 7 ngày kiểm toán / tranh chấp
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
    "methodology_note": "Truy vấn CafeF sử dụng StartDate=d và EndDate=d+1 (định dạng MM/DD/YYYY) rồi lọc chính xác theo trường Ngay == d (định dạng DD/MM/YYYY). Giả thuyết thành phần giờ '15:00:00' trong CSDL CafeF chỉ là suy luận thực nghiệm.",
    "dates_audited": {}
}

def fetch_cafef_bar(symbol: str, start_str: str, end_str: str, expected_date_str: str):
    url = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?ExchangeType=hose&Symbol={symbol}&StartDate={start_str}&EndDate={end_str}&PageIndex=1&PageSize=10"
    try:
        resp = s.get(url, timeout=15).json()
        data_obj = resp.get("Data") or {}
        items = data_obj.get("Data") or []
        matching = [it for it in items if it.get("Ngay") == expected_date_str]
        return matching[0] if matching else None
    except Exception as exc:
        print(f"   [CẢNH BÁO] Lỗi truy vấn CafeF {symbol} ({start_str}): {exc}")
        return None

def fetch_entrade_bar(symbol: str, is_index: bool, t_start: int, t_end: int):
    endpoint = "index" if is_index else "stock"
    url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{endpoint}?symbol={symbol}&resolution=1D&from={t_start}&to={t_end}"
    try:
        resp = requests.get(url, timeout=15).json()
        if resp.get("t") and len(resp["t"]) > 0:
            return {
                "t": resp["t"][0],
                "o": resp["o"][0],
                "h": resp["h"][0],
                "l": resp["l"][0],
                "c": resp["c"][0],
                "v": resp["v"][0]
            }
        return None
    except Exception as exc:
        print(f"   [CẢNH BÁO] Lỗi truy vấn Entrade {symbol}: {exc}")
        return None

print("=== BẮT ĐẦU KIỂM TOÁN T22: LỊCH GIAO DỊCH VÀ DỮ LIỆU THÔ ===")

total_requests = 0
start_time = time.time()

for d in dates_to_check:
    dt_day = datetime.datetime.strptime(d, "%Y-%m-%d").date()
    dt_next = dt_day + datetime.timedelta(days=1)
    start_m_d_y = dt_day.strftime("%m/%d/%Y")
    end_m_d_y = dt_next.strftime("%m/%d/%Y")
    expected_ngay = dt_day.strftime("%d/%m/%Y")

    # 1. CafeF VNINDEX
    cf_vni = fetch_cafef_bar("VNINDEX", start_m_d_y, end_m_d_y, expected_ngay)
    total_requests += 1
    time.sleep(0.5)

    # 2. CafeF HPG
    cf_hpg = fetch_cafef_bar("HPG", start_m_d_y, end_m_d_y, expected_ngay)
    total_requests += 1
    time.sleep(0.5)

    # 3. CafeF VCB
    cf_vcb = fetch_cafef_bar("VCB", start_m_d_y, end_m_d_y, expected_ngay)
    total_requests += 1
    time.sleep(0.5)

    # Entrade timestamps
    dt_dt = datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=VN_TZ)
    t_start = int(dt_dt.timestamp())
    t_end = int((dt_dt + datetime.timedelta(days=1)).timestamp())

    # 4. Entrade HPG
    en_hpg_bar = fetch_entrade_bar("HPG", False, t_start, t_end)
    total_requests += 1
    time.sleep(0.5)

    # 5. Entrade VNINDEX
    en_vni_bar = fetch_entrade_bar("VNINDEX", True, t_start, t_end)
    total_requests += 1
    time.sleep(0.5)

    audit_results["dates_audited"][d] = {
        "cafef_vnindex": cf_vni,
        "cafef_hpg": cf_hpg,
        "cafef_vcb": cf_vcb,
        "entrade_hpg": en_hpg_bar,
        "entrade_vnindex": en_vni_bar
    }

# Ghi ra tệp JSON với định dạng LF và có trailing newline chuẩn
with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
    json.dump(audit_results, f, ensure_ascii=False, indent=2)
    f.write("\n")

# Tính SHA-256 trên đúng chuỗi byte vừa ghi
with open(OUT_JSON, "rb") as f:
    raw_bytes = f.read()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

elapsed = time.time() - start_time
print(f"-> Hoàn tất trong {elapsed:.1f}s | Tổng số request: {total_requests}")
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
    print(f"   CafeF HPG    : {'CÓ, Close=' + str(cf_hpg.get('GiaDongCua')) if cf_hpg else 'KHÔNG CÓ'}")
    print(f"   Entrade HPG  : {'CÓ, Vol=' + str(en_hpg.get('v')) if en_hpg else 'KHÔNG CÓ'}")
    print(f"   Entrade VNI  : {'CÓ, Close=' + str(rec['entrade_vnindex'].get('c')) if rec['entrade_vnindex'] else 'KHÔNG CÓ'}")
