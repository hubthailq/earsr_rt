#!/usr/bin/env python3
"""Áp tiêu chí đạt ghi trước (configs/criteria.yaml) lên các bảng số đo theo từng ảnh.
Kết quả được in ra và ghi vào ``results/criteria/<tên>.json``.

Mỗi nhóm file là các file CSV số đo theo từng ảnh của MỘT phương pháp (thường một file mỗi fold).

  # N5b (b): biến thể so với thân tham chiếu, cùng ngân sách tiền huấn luyện
  python scripts/check_criteria.py n5b-gain --a runs/T6_span-replicate_ps20_*/test_hr144_x4_bic.csv \
                                           --b runs/T6_span-zero_ps20_*/test_hr144_x4_bic.csv
  # N5b (c): phép thử tương tác
  python scripts/check_criteria.py n5b-interaction --a VAR_SMALL.csv --b REF_SMALL.csv --a2 VAR_LARGE.csv --b2 REF_LARGE.csv
  # so sánh bất kỳ (a − b), ví dụ mô hình đề xuất so với mốc phải vượt
  python scripts/check_criteria.py gain --a ... --b ... --metric lpips
  # N1 (cần evaluate.py --landmark-ckpt --ridge)
  python scripts/check_criteria.py n1 --a AUX*.csv --b BASE*.csv --control CONTROL*.csv
  # N5a: giao thức (lặp lại --tier cho từng cỡ)
  python scripts/check_criteria.py n5a --tier 96 --a RAND96*.csv --b FIXED96*.csv --tier 144 --a ... --b ... --tier 192 --a ... --b ...
  # N3
  python scripts/check_criteria.py n3 --a P*.csv --b BASE*.csv --blend BLEND*.csv --ldl LDL*.csv
  # bản thiên trung thực
  python scripts/check_criteria.py fidelity --a F*.csv --b BASE*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.report import criteria as C  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gain", "n5b-gain", "n5b-interaction", "n5a", "n1", "n3", "fidelity"])
    ap.add_argument("--a", nargs="+", action="append", required=True)
    ap.add_argument("--b", nargs="+", action="append", required=True)
    ap.add_argument("--a2", nargs="+")
    ap.add_argument("--b2", nargs="+")
    ap.add_argument("--control", nargs="+")
    ap.add_argument("--blend", nargs="+")
    ap.add_argument("--ldl", nargs="+")
    ap.add_argument("--tier", type=int, action="append")
    ap.add_argument("--metric", default="psnr_y")
    ap.add_argument("--relative", action="store_true")
    ap.add_argument("--criteria", default="configs/criteria.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results/criteria")
    a = ap.parse_args(argv)
    crit = C.load_criteria(a.criteria)
    A, B = a.a[0], a.b[0]
    if a.what == "gain":
        r = C.gain(A, B, a.metric, a.relative)
    elif a.what == "n5b-gain":
        r = C.n5b_variant_gain(A, B, crit, a.metric)
    elif a.what == "n5b-interaction":
        if not (a.a2 and a.b2):
            raise SystemExit("cần --a2 và --b2 (điều kiện ảnh vào lớn)")
        r = C.n5b_interaction(A, B, a.a2, a.b2, a.metric)
    elif a.what == "n5a":
        if not a.tier or len(a.tier) != len(a.a) or len(a.tier) != len(a.b):
            raise SystemExit("n5a: mỗi --tier đi kèm một --a và một --b")
        r = C.n5a_protocol(dict(zip(a.tier, a.a)), dict(zip(a.tier, a.b)), crit, metric=a.metric)
    elif a.what == "n1":
        if not a.control:
            raise SystemExit("n1 cần --control (nhánh 'thân + ảnh thêm, không nhãn')")
        r = C.n1_aux_head(A, B, a.control, crit)
    elif a.what == "n3":
        if not (a.blend and a.ldl):
            raise SystemExit("n3 cần --blend và --ldl")
        r = C.n3_gate(A, B, a.blend, a.ldl, crit)
    else:
        r = C.fidelity_point(A, B, crit)
    r["criteria_version"] = crit.get("version")
    r["files"] = {"a": a.a, "b": a.b}
    name = a.name or a.what
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    if "pass" in r:
        print("KẾT LUẬN:", {True: "ĐẠT", False: "KHÔNG ĐẠT", None: "THIẾU DỮ LIỆU"}[r["pass"]])
    return r


if __name__ == "__main__":
    main()
