#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do do tre TensorRT FP16 tren Jetson cho moi file ONNX trong ./onnx (T3).

CHUA DUOC CHAY THU TREN THIET BI THAT. File nay duoc viet theo tai lieu cua
trtexec (TensorRT 8.x). Neu dong "Latency" cua trtexec tren may ban co dang
khac, ket qua tho van duoc luu trong ./logs de doc tay.

Yeu cau: JetPack co san trtexec (/usr/src/tensorrt/bin/trtexec). Khong can
PyTorch. Chay duoc voi Python 3.6 (JetPack 4.6, Jetson Nano ban 2019).

Giao thuc (muc 1.5 cua ke hoach):
  - che do nguon cao nhat va khoa xung nhip TRUOC khi chay:
        sudo nvpmodel -m 0
        sudo jetson_clocks
  - FP16, lo 1, kich thuoc vao co dinh trong file ONNX
  - 50 luot lam nong (uoc luong bang thoi gian), 500 luot do
  - bao trung vi va phan vi 95; chi tinh luot chay cua mo hinh (GPU compute)

Cach chay:
    python3 bench_trtexec.py --note "Jetson Nano 4GB, JetPack 4.6.1, MAXN, jetson_clocks"

Ket qua: latency_trt.csv va thu muc logs/. Gui lai ca hai.
Mo hinh nao khong dung duoc engine (vi du TensorRT khong ho tro kieu dem) se co
status = build_failed; do la thong tin can biet, khong phai loi cua script.
"""
from __future__ import print_function

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

LAT_RE = re.compile(
    r"Latency:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*mean\s*=\s*([\d.]+)\s*ms,"
    r"\s*median\s*=\s*([\d.]+)\s*ms,\s*percentile\((\d+)%\)\s*=\s*([\d.]+)\s*ms")
GPU_RE = re.compile(
    r"GPU Compute Time:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*mean\s*=\s*([\d.]+)\s*ms,"
    r"\s*median\s*=\s*([\d.]+)\s*ms,\s*percentile\((\d+)%\)\s*=\s*([\d.]+)\s*ms")


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""


def device_info():
    info = {}
    info["tegra_release"] = read_text("/etc/nv_tegra_release").split("\n")[0]
    info["model"] = read_text("/proc/device-tree/model").replace("\x00", "")
    try:
        info["nvpmodel"] = subprocess.check_output(["nvpmodel", "-q"], stderr=subprocess.STDOUT).decode().strip().replace("\n", " | ")
    except Exception as e:
        info["nvpmodel"] = "khong doc duoc: %s" % e
    temps = []
    for z in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp")):
        t = read_text(z)
        if t.isdigit():
            temps.append(int(t) / 1000.0)
    info["temp_max_c"] = max(temps) if temps else ""
    return info


def run_one(trtexec, onnx_path, log_dir, iterations, warmup_ms, workspace_mb, fp16):
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    log_path = os.path.join(log_dir, name + ".log")
    cmd = [trtexec, "--onnx=" + onnx_path, "--workspace=%d" % workspace_mb,
           "--iterations=%d" % iterations, "--avgRuns=1", "--duration=0",
           "--warmUp=%d" % warmup_ms, "--percentile=95"]
    if fp16:
        cmd.append("--fp16")
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "replace")
        ok = True
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", "replace")
        ok = False
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + out)
    row = {"name": name, "status": "ok" if ok else "build_failed", "wall_s": round(time.time() - t0, 1), "log": log_path}
    m = GPU_RE.search(out) or LAT_RE.search(out)
    if m:
        row.update(min_ms=m.group(1), max_ms=m.group(2), mean_ms=m.group(3), median_ms=m.group(4),
                   pctl=m.group(5), p_ms=m.group(6),
                   source="gpu_compute" if GPU_RE.search(out) else "latency")
    elif ok:
        row["status"] = "ok_but_unparsed"
    if not ok:
        errs = [l for l in out.split("\n") if "[E]" in l]
        row["error"] = (errs[0] if errs else out[-300:]).strip()[:300]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    ap.add_argument("--out", default="latency_trt.csv")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--warmup-ms", type=int, default=2000, help="thoi gian lam nong; 2 giay la hon 50 luot voi mo hinh duoi 40 ms")
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--note", default="", help="BAT BUOC: doi may, JetPack, che do nguon, da chay jetson_clocks chua")
    a = ap.parse_args()
    if not os.path.isfile(a.trtexec):
        sys.exit("khong thay trtexec o %s (dung --trtexec de chi duong dan)" % a.trtexec)
    if not a.note:
        print("CANH BAO: --note dang trong. Hay ghi doi may, JetPack va che do nguon.")
    files = sorted(glob.glob(os.path.join(a.onnx_dir, "*.onnx")))
    if not files:
        sys.exit("khong co file .onnx trong %s" % a.onnx_dir)
    log_dir = "logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    info = device_info()
    print("thiet bi:", info)
    rows = []
    for p in files:
        print("== %s" % os.path.basename(p))
        r = run_one(a.trtexec, p, log_dir, a.iterations, a.warmup_ms, a.workspace_mb, not a.no_fp16)
        r.update(precision="fp32" if a.no_fp16 else "fp16", iterations=a.iterations, note=a.note)
        r.update(info)
        r["temp_after_c"] = device_info()["temp_max_c"]
        print("   %s  median=%s ms  p%s=%s ms" % (r["status"], r.get("median_ms", "?"), r.get("pctl", "95"), r.get("p_ms", "?")))
        rows.append(r)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("da ghi %s (%d mo hinh). Gui lai file nay va thu muc logs/." % (a.out, len(rows)))


if __name__ == "__main__":
    main()
