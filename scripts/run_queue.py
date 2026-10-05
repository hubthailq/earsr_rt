#!/usr/bin/env python3
"""Chạy tuần tự một danh sách lệnh (một GPU, một việc một lúc).

  python scripts/run_queue.py jobs/t6ii.txt

- Dòng trống và dòng bắt đầu bằng # bị bỏ qua.
- Trạng thái từng dòng được lưu ở ``<file>.state.json``; chạy lại sẽ bỏ qua dòng đã
  xong và thử lại dòng lỗi. Lệnh ``train.py`` còn tự tiếp tục từ checkpoint gần nhất.
- Một lệnh lỗi không dừng hàng đợi; đầu ra của từng lệnh ở ``<file>.logs/NNN.log``.
- Dừng êm: tạo file ``<file>.stop`` (hàng đợi dừng sau lệnh đang chạy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", default=True)
    a = ap.parse_args(argv)
    jf = Path(a.jobs)
    lines = [l.strip() for l in jf.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    state_f, log_d, stop_f = Path(str(jf) + ".state.json"), Path(str(jf) + ".logs"), Path(str(jf) + ".stop")
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    log_d.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "failed": 0, "skipped": 0}
    for i, cmd in enumerate(lines, 1):
        h = hashlib.sha1(cmd.encode()).hexdigest()[:12]
        if state.get(h, {}).get("status") == "done":
            counts["skipped"] += 1
            continue
        if stop_f.exists():
            print(f"thấy {stop_f}: dừng trước lệnh {i}")
            break
        print(f"[{i}/{len(lines)}] {cmd}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        with open(log_d / f"{i:03d}.log", "w") as lf:
            rc = subprocess.call(shlex.split(cmd), stdout=lf, stderr=subprocess.STDOUT)
        status = "done" if rc == 0 else "failed"
        counts[status] += 1
        state[h] = {"status": status, "returncode": rc, "minutes": round((time.time() - t0) / 60, 1), "cmd": cmd,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        state_f.write_text(json.dumps(state, indent=1, ensure_ascii=False))
        print(f"    -> {status} sau {state[h]['minutes']} phút" + ("" if rc == 0 else f" (xem {log_d / f'{i:03d}.log'})"),
              flush=True)
    print(counts)
    return counts


if __name__ == "__main__":
    sys.exit(0 if main()["failed"] == 0 else 1)
