"""Chia 5 fold theo người cho AMI. File fold sinh một lần, đưa vào git.

Fold i (1..5): nhóm i là test (20 người); 10 người lấy ngẫu nhiên từ 80 người
còn lại là validation; 70 người là train. Seed huấn luyện của fold i bằng i.
Fold 1 và 5 là hai fold giữ kín: không dùng cho quyết định nào trước lần chạy
cuối (mục 1.6 của kế hoạch).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 20261005
HELD_OUT_FOLDS = (1, 5)
DEV_FOLDS = (2, 3, 4)


def make_folds(subjects: list[str], n_folds: int = 5, n_val: int = 10, seed: int = SPLIT_SEED) -> dict:
    subjects = sorted(set(subjects))
    if len(subjects) % n_folds:
        raise ValueError(f"{len(subjects)} người không chia đều cho {n_folds} fold")
    rng = np.random.default_rng(seed)
    perm = [subjects[i] for i in rng.permutation(len(subjects))]
    size = len(subjects) // n_folds
    groups = [sorted(perm[i * size:(i + 1) * size]) for i in range(n_folds)]
    folds = {}
    for i in range(n_folds):
        test = groups[i]
        rest = sorted(s for j, g in enumerate(groups) if j != i for s in g)
        r = np.random.default_rng(seed + 1000 + i)
        val = sorted(rest[k] for k in r.permutation(len(rest))[:n_val])
        train = sorted(s for s in rest if s not in set(val))
        folds[str(i + 1)] = {"train": train, "val": val, "test": test}
    out = {"seed": seed, "n_folds": n_folds, "held_out_folds": list(HELD_OUT_FOLDS),
           "dev_folds": list(DEV_FOLDS), "folds": folds}
    out["sha1"] = folds_hash(out)
    return out


def folds_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d["folds"], sort_keys=True).encode()).hexdigest()


def check_folds(d: dict) -> None:
    """Dừng nếu có rò rỉ: người nằm ở hai phần của một fold, hoặc test hai lần."""
    seen_test: list[str] = []
    all_subj = None
    for k, f in d["folds"].items():
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        if tr & va or tr & te or va & te:
            raise AssertionError(f"fold {k}: có người nằm ở hai phần")
        u = tr | va | te
        if all_subj is None:
            all_subj = u
        elif u != all_subj:
            raise AssertionError(f"fold {k}: tập người khác các fold khác")
        seen_test += sorted(te)
    if sorted(seen_test) != sorted(all_subj or []):
        raise AssertionError("mỗi người phải nằm trong test đúng một lần")
    if d.get("sha1") and d["sha1"] != folds_hash(d):
        raise AssertionError("mã băm của file fold không khớp nội dung")


def save_folds(d: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)


def load_folds(path: str | Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    check_folds(d)
    return d


def fold_of_subject(d: dict) -> dict[str, int]:
    """Người -> fold mà người đó nằm trong test."""
    return {s: int(k) for k, f in d["folds"].items() for s in f["test"]}
