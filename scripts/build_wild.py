#!/usr/bin/env python3
"""Dựng benchmark từ ảnh tai ngoài thực tế (EarVN1.0, AWEx): lọc, tạo đáp án, ghi manifest.

EarVN1.0, nhóm test giữ riêng, biên an toàn 2 lần:
  python scripts/build_wild.py --root /path/EarVN1.0 --name earvn --out data/bench/earvn \
      --tiers 96 --safety 2.0 --roles splits/earvn_roles.json --role test
AWEx (không bao giờ dùng để huấn luyện, nên không cần vai), hai mức biên:
  python scripts/build_wild.py --root /path/awex --name awex --out data/bench/awex_s20 --tiers 96 144 --safety 2.0
  python scripts/build_wild.py --root /path/awex --name awex --out data/bench/awex_s15 --tiers 96 144 --safety 1.5

Ảnh tự nhiên thu nhỏ cho phép thử ngữ cảnh và phép thử tương tác ngoài ảnh tai (T6 v):
  python scripts/build_wild.py --root /path/DIV2K_valid_HR --name div2k --out data/bench/div2k \
      --tiers 144 576 --safety 2.0 --subject-per-image

Cấu trúc thư mục phải là ``root/<người>/<ảnh>`` (trừ khi dùng --subject-per-image). Sau đó chấm bằng
``scripts/evaluate.py --bench ... --folds none``.

Ảnh bị loại được ghi kèm lý do vào ``<out>/dropped_hr<tier>.csv``. Lần đầu script
quét cả bộ (đọc mọi ảnh) và lưu kết quả vào ``<out>/scan.csv`` để lần sau khỏi quét lại.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.data import wild  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True, help="tên bộ dữ liệu ghi vào manifest (earvn, awex)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", type=int, nargs="+", default=[96])
    ap.add_argument("--safety", type=float, default=2.0, help="ảnh gốc phải được thu nhỏ ít nhất ngần này lần")
    ap.add_argument("--roles", default=None, help="file vai theo người (tạo nếu chưa có)")
    ap.add_argument("--role", default="test")
    ap.add_argument("--subject-per-image", action="store_true",
                    help="coi mỗi ảnh là một 'người' (cho bộ ảnh tự nhiên như DIV2K, để bootstrap theo ảnh)")
    ap.add_argument("--max-per-subject", type=int, default=0, help="giới hạn số ảnh mỗi người (0: không giới hạn)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    scan_csv = out / "scan.csv"
    if scan_csv.exists():
        rows = wild.load_scan(scan_csv)
        print(f"dùng lại {scan_csv} ({len(rows)} ảnh)")
    else:
        rows = wild.scan(a.root)
        if not rows:
            raise SystemExit(f"không thấy ảnh nào dưới {a.root}")
        wild.save_scan(rows, scan_csv)
    if a.subject_per_image:
        for r in rows:
            r["subject"] = "".join(c if c.isalnum() else "-" for c in Path(r["path"]).stem)
    print(f"{len(rows)} ảnh, {len({r['subject'] for r in rows})} người; không đọc được: {sum(not r['ok'] for r in rows)}")
    if a.roles:
        roles, made = wild.load_or_make_roles(sorted({r["subject"] for r in rows}), a.roles)
        if made:
            print(f"đã tạo {a.roles}: hãy commit file này")
        rows = [r for r in rows if roles.get(r["subject"]) == a.role]
        print(f"vai '{a.role}': {len(rows)} ảnh, {len({r['subject'] for r in rows})} người")
    report = {}
    for tier in a.tiers:
        keep, drop = wild.select(rows, tier, a.safety)
        if a.max_per_subject:
            seen, kept = Counter(), []
            for r in keep:
                seen[r["subject"]] += 1
                if seen[r["subject"]] <= a.max_per_subject:
                    kept.append(r)
                else:
                    drop.append({**r, "why": "quá số ảnh mỗi người"})
            keep = kept
        with open(out / f"dropped_hr{tier}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "subject", "why"], extrasaction="ignore")
            w.writeheader()
            w.writerows(drop)
        why = Counter(" ".join(r["why"].split(" ")[:2]) for r in drop)
        if not keep:
            print(f"tầng {tier}: KHÔNG còn ảnh nào (loại {len(drop)}: {dict(why)})")
            report[tier] = 0
            continue
        wild.build_hr(a.root, keep, out, tier, a.name)
        report[tier] = len(keep)
        print(f"tầng {tier}: giữ {len(keep)} ảnh của {len({r['subject'] for r in keep})} người; loại {len(drop)} ({dict(why)})")
    return report


if __name__ == "__main__":
    main()
