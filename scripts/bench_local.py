#!/usr/bin/env python3
"""Đo độ trễ trên máy đang chạy (CPU hoặc GPU của PyTorch), theo giao thức mục 1.5.

Dùng để so tương đối và để ghép biến thể theo độ trễ. KHÔNG thay cho số đo
trên thiết bị (deploy_jetson/).

  python scripts/bench_local.py --out results/latency_local.csv
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from earsr.deploy.latency import measure_latency  # noqa: E402
from earsr.device import describe_device, free_gpu_cache, is_oom  # noqa: E402
from earsr.eval.complexity import count_flops, count_params  # noqa: E402
from earsr.models.registry import SPECS, Bicubic, build_model, list_models  # noqa: E402
from earsr.models.variants import all_variants, build_span_variant  # noqa: E402

INPUT_HW = {4: (68, 48), 2: (136, 96)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latency_local.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--half", action="store_true", help="FP16 (chỉ có nghĩa trên GPU)")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--oom-retries", type=int, default=3, help="số lần đo lại một mô hình khi GPU hết bộ nhớ")
    ap.add_argument("--oom-wait", type=float, default=30.0, help="số giây chờ trước mỗi lần đo lại")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    print(describe_device(a.device, on_oom="hết bộ nhớ thì chờ rồi đo lại, không đo thay trên CPU"), flush=True)
    host = f"{platform.processor() or platform.machine()} | torch {torch.__version__} | threads {torch.get_num_threads()}"
    if a.device.startswith("cuda"):
        host += f" | {torch.cuda.get_device_name(0)}"
    rows = []

    def add(name, kind, model, scale, extra=None):
        # Độ trễ không được đo trên CPU thay cho GPU (hai số không so được). GPU hết bộ nhớ thì chờ rồi
        # thử lại; vẫn không được thì ghi một dòng lỗi và đi tiếp, không làm chết cả lượt đo.
        r = None
        for attempt in range(1, a.oom_retries + 2):
            oom = False
            try:
                r = measure_latency(model, INPUT_HW[scale], a.device, a.warmup, a.runs, a.half)
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = True
            if not oom:
                break
            model.cpu()
            free_gpu_cache()
            if attempt <= a.oom_retries:
                print(f"[thiết bị] {name}: {a.device} hết bộ nhớ, chờ {a.oom_wait:.0f} giây rồi đo lại "
                      f"(lần {attempt}/{a.oom_retries})", flush=True)
                time.sleep(a.oom_wait)
        if r is None:
            r = {"error": f"{a.device} hết bộ nhớ sau {a.oom_retries + 1} lần thử; chưa đo", "device": a.device}
        r.update(name=name, kind=kind, scale=scale, host=host, **(extra or {}))
        rows.append(r)
        if "error" in r:
            print(f"{name:24s} CHƯA ĐO: {r['error']}", flush=True)
        else:
            print(f"{name:24s} median {r['median_ms']:8.2f} ms   p95 {r['p95_ms']:8.2f} ms", flush=True)

    add("bicubic", "baseline", Bicubic(4), 4)
    for n in list_models(available_only=True):
        s = SPECS[n]
        if a.groups and s.group not in a.groups:
            continue
        m = build_model(n)
        add(n, "zoo", m, s.scale, {"group": s.group, "params": count_params(m), "flops_g_256": count_flops(m)["flops_g"]})
    for v in all_variants():
        m = build_span_variant(v).eval().switch_to_deploy()
        add(f"spanvar_{v}", "variant", m, 4, {"params": count_params(m), "flops_g_256": count_flops(m)["flops_g"]})
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("đã ghi", a.out)


if __name__ == "__main__":
    main()
