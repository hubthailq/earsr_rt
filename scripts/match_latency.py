#!/usr/bin/env python3
"""Tìm số kênh của thân SPAN để độ trễ bằng một mức cho trước (chênh không quá 10%).

Dùng cho hai việc của kế hoạch:
1. "Mốc nới rộng cho bằng độ trễ của mô hình đề xuất" (mục 1.8):
     python scripts/match_latency.py --target-variant replicate-deep --base zero
   -> in ra variant dạng ``zero-cNN`` có độ trễ gần nhất.
2. Chỉnh hai hình dạng "wide" và "deep" cho bằng độ trễ thân tham chiếu (N5b):
     python scripts/match_latency.py --target-variant zero --base zero-b3     # nông và rộng
     python scripts/match_latency.py --target-variant zero --base zero-b11    # sâu và hẹp

Số đo ở đây là trên máy đang chạy. Sau khi chọn, xuất variant đó
(``export_for_device.py --variants ...``) và xác nhận lại trên thiết bị.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from earsr.deploy.latency import measure_latency  # noqa: E402
from earsr.eval.complexity import count_params  # noqa: E402
from earsr.models.variants import arch_part, build_span_variant, parse_variant_full  # noqa: E402

INPUT_HW = {4: (68, 48), 2: (136, 96)}


def latency_of(variant: str, scale: int, device: str, warmup: int, runs: int, half: bool) -> float:
    m = build_span_variant(variant, scale=scale).eval().switch_to_deploy()
    return measure_latency(m, INPUT_HW[scale], device, warmup, runs, half)["median_ms"]


def with_channels(base: str, c: int) -> str:
    parts = [p for p in arch_part(base).split("-") if not (p[0] == "c" and p[1:].isdigit()) and p not in ("wide", "deep")]
    _, _, nb = parse_variant_full(base)
    if not any(p[0] == "b" and p[1:].isdigit() for p in parts) and nb != 6:
        parts.append(f"b{nb}")
    return "-".join(parts + [f"c{c}"])


def search(target_ms: float, base: str, measure, c_min: int = 16, c_max: int = 128, step: int = 2,
           tol: float = 0.10) -> dict:
    """Tìm nhị phân số kênh (độ trễ tăng theo số kênh). ``measure(variant) -> ms``."""
    lo, hi = c_min // step, c_max // step
    tried = {}
    while lo <= hi:
        mid = (lo + hi) // 2
        c = mid * step
        tried[c] = measure(with_channels(base, c))
        if tried[c] < target_ms:
            lo = mid + 1
        else:
            hi = mid - 1
    best = min(tried, key=lambda c: abs(tried[c] - target_ms))
    rel = (tried[best] - target_ms) / target_ms
    return {"variant": with_channels(base, best), "channels": best, "ms": tried[best], "target_ms": target_ms,
            "rel_diff": rel, "within_tol": abs(rel) <= tol, "tried": dict(sorted(tried.items()))}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target-variant", help="variant có độ trễ cần bằng")
    g.add_argument("--target-ms", type=float)
    ap.add_argument("--base", default="zero", help="variant được đổi số kênh (kiểu đệm và số khối giữ nguyên)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    meas = lambda v: latency_of(v, a.scale, a.device, a.warmup, a.runs, a.half)
    target = a.target_ms if a.target_ms else meas(a.target_variant)
    print(f"độ trễ đích: {target:.3f} ms" + (f" ({a.target_variant})" if a.target_variant else ""))
    r = search(target, a.base, meas)
    for c, ms in r["tried"].items():
        print(f"  {with_channels(a.base, c):22s} {ms:8.3f} ms")
    p = count_params(build_span_variant(r["variant"], a.scale).eval().switch_to_deploy())
    print(f"chọn {r['variant']}: {r['ms']:.3f} ms, lệch {r['rel_diff']:+.1%}, {p / 1e3:.0f} nghìn tham số"
          + ("" if r["within_tol"] else "  (CHƯA trong 10%: thử --base với số khối khác)"))
    return r


if __name__ == "__main__":
    main()
