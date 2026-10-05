"""Tập ảnh tai ngoài thực tế (EarVN1.0, AWEx): lọc, chia vai theo người, tạo đáp án.

Chỉ dùng numpy và OpenCV (không cần torch), để chạy được ở nơi có dữ liệu thô.

Quy ước thư mục: ``root/<người>/<ảnh>``; tên thư mục cha là mã người.

Lọc (mọi ảnh bị loại đều được ghi kèm lý do):
- không đọc được, không phải 8 bit 3 kênh;
- cạnh ngắn nhỏ hơn ``tier × safety`` (biên an toàn: ảnh phải được thu nhỏ ít
  nhất ``safety`` lần để thành đáp án cỡ ``tier``);
- tỉ lệ khung ngoài [ar_min, ar_max];
- trùng hẳn (cùng mã băm điểm ảnh) hoặc gần trùng trong cùng một người (dHash).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .resize import center_crop_to_multiple, imresize

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# vai của người trong EarVN1.0 (mục 3.4 của kế hoạch): các nhóm tách rời
ROLE_FRACTIONS = {"train": 0.45, "test": 0.25, "fit": 0.10, "clf": 0.10, "viewer": 0.10}


def _read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, "không đọc được"
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return None, f"không phải ảnh 8 bit 3 kênh ({img.dtype}, {img.shape})"
    return np.ascontiguousarray(img[:, :, ::-1]), ""


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def scan(root: str | Path) -> list[dict]:
    """Liệt kê ảnh kèm kích thước; không lọc gì."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img, why = _read(p)
        row = {"path": str(p.relative_to(root)), "subject": p.parent.name, "ok": img is not None, "why": why}
        if img is not None:
            row.update(h=img.shape[0], w=img.shape[1], sha1=hashlib.sha1(img.tobytes()).hexdigest(), dhash=dhash(img))
        rows.append(row)
    return rows


def select(rows: list[dict], tier: int, safety: float = 2.0, ar_min: float = 0.4, ar_max: float = 2.5,
           near_dup_bits: int = 2) -> tuple[list[dict], list[dict]]:
    """Trả về (giữ, loại). Mỗi dòng loại có khóa ``why``."""
    keep, drop = [], []
    seen_sha: set = set()
    seen_hash: dict[str, list[int]] = {}
    for r in rows:
        if not r["ok"]:
            drop.append(r)
            continue
        h, w = r["h"], r["w"]
        why = ""
        if min(h, w) < tier * safety:
            why = f"cạnh ngắn {min(h, w)} < {tier}×{safety}"
        elif not ar_min <= h / w <= ar_max:
            why = f"tỉ lệ khung {h / w:.2f}"
        elif r["sha1"] in seen_sha:
            why = "trùng hẳn"
        elif any(bin(r["dhash"] ^ x).count("1") <= near_dup_bits for x in seen_hash.get(r["subject"], [])):
            why = "gần trùng trong cùng người"
        if why:
            drop.append({**r, "why": why})
        else:
            keep.append(r)
            seen_sha.add(r["sha1"])
            seen_hash.setdefault(r["subject"], []).append(r["dhash"])
    return keep, drop


def assign_roles(subjects: list[str], fractions: dict[str, float], seed: int = 20261005) -> dict[str, str]:
    """Chia người thành các nhóm vai tách rời, ví dụ
    {'train': .5, 'test': .2, 'fit': .1, 'clf': .1, 'viewer': .1}."""
    subjects = sorted(set(subjects))
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("tổng tỉ lệ các vai phải bằng 1")
    rng = np.random.default_rng(seed)
    perm = [subjects[i] for i in rng.permutation(len(subjects))]
    out, start = {}, 0
    names = list(fractions)
    for i, n in enumerate(names):
        k = len(perm) - start if i == len(names) - 1 else int(round(fractions[n] * len(perm)))
        for s in perm[start:start + k]:
            out[s] = n
        start += k
    return out


def build_hr(root: str | Path, keep: list[dict], out_dir: str | Path, tier: int, name: str) -> Path:
    """Thu các ảnh được giữ về cạnh ngắn ``tier``, cắt cho chia hết cho 4, lưu PNG và manifest."""
    root, out_dir = Path(root), Path(out_dir)
    (out_dir / f"hr{tier}").mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for i, r in enumerate(keep):
        img, _ = _read(root / r["path"])
        h, w = img.shape[:2]
        out = (int(round(h * tier / w)), tier) if h >= w else (tier, int(round(w * tier / h)))
        hr = center_crop_to_multiple(imresize(img, out_size=out), 4)
        # "view" lấy từ tên file gốc, để cùng một ảnh có cùng khóa ở mọi tầng
        view = "".join(c if c.isalnum() else "-" for c in Path(r["path"]).stem) or f"{i:05d}"
        if (r["subject"], view) in used:
            view = f"{view}-{i:05d}"
        used.add((r["subject"], view))
        rel = f"hr{tier}/{r['subject']}_{view}.png"
        cv2.imwrite(str(out_dir / rel), hr[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        rows.append({"dataset": name, "subject": r["subject"], "view": view, "tier": tier, "file": rel,
                     "h": hr.shape[0], "w": hr.shape[1], "sha1": hashlib.sha1(hr.tobytes()).hexdigest(),
                     "src": r["path"], "src_h": h, "src_w": w})
    # một manifest.csv cho cả bộ, cùng dạng với benchmark AMI; giữ các tầng đã có
    man = out_dir / "manifest.csv"
    old = []
    if man.exists():
        with open(man, newline="") as f:
            old = [r for r in csv.DictReader(f) if int(r["tier"]) != tier]
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(old + rows)
    return man


def load_or_make_roles(subjects: list[str], path: str | Path, fractions: dict | None = None) -> tuple[dict, bool]:
    """Đọc file vai; nếu chưa có thì tạo và ghi ra. Trả về (vai, vừa_tạo)."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text()), False
    roles = assign_roles(subjects, fractions or ROLE_FRACTIONS)
    save_json(roles, path)
    return roles, True


def save_scan(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "subject", "ok", "why", "h", "w", "sha1", "dhash"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


def load_scan(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ok"] = r["ok"] == "True"
        if r["ok"]:
            r["h"], r["w"], r["dhash"] = int(r["h"]), int(r["w"]), int(r["dhash"])
    return rows


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
