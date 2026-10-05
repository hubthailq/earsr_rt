"""Sổ ghi các lần chạy: ``results/runs.csv``, một dòng mỗi lần chạy.

Trạng thái: running | done | failed | excluded. Lần chạy bị loại (excluded) không
bị xóa; lý do ghi ở cột ``reason``. Ghi lại cùng ``run_id`` thì cập nhật dòng cũ.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

BASE = ["run_id", "status", "reason", "started", "updated", "git_commit", "host"]
STATUSES = ("running", "done", "failed", "excluded")


def read(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def record(path: str | Path, run_id: str, status: str, reason: str = "", **fields) -> dict:
    """Thêm hoặc cập nhật dòng của ``run_id``. Trả về dòng sau khi ghi."""
    if status not in STATUSES:
        raise ValueError(f"status phải thuộc {STATUSES}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read(path)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    row = next((r for r in rows if r["run_id"] == run_id), None)
    if row is None:
        row = {"run_id": run_id, "started": now}
        rows.append(row)
    if status == "running":
        row["started"] = now
    row.update(status=status, reason=reason, updated=now, **{k: str(v) for k, v in fields.items()})
    row.setdefault("host", os.uname().nodename if hasattr(os, "uname") else "")
    cols = list(BASE)
    for r in rows:
        cols += [k for k in r if k not in cols]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return row


def status_of(path: str | Path, run_id: str) -> str | None:
    return next((r["status"] for r in read(path) if r["run_id"] == run_id), None)
