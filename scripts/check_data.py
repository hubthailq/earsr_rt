#!/usr/bin/env python3
"""Kiểm tra các bộ dữ liệu đã đặt đúng chỗ và đúng cấu trúc chưa (xem docs/DATA_AND_OUTPUTS.md).

  python scripts/check_data.py                 # dùng bố cục mặc định dưới data/raw
  python scripts/check_data.py --root /data    # nếu dữ liệu nằm ở nơi khác

Chỉ đọc tên file và phần đầu file ảnh; không sửa gì. AMI là bộ bắt buộc; các bộ
khác chỉ cần khi tới bước dùng chúng.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from earsr.data import ami  # noqa: E402
from earsr.landmarks.pts import find_pairs  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# (thư mục con, bắt buộc?, dùng cho việc gì)
LAYOUT = [("AMI", True, "benchmark chính; mọi giai đoạn"),
          ("DIV2K_train_HR", False, "tiền huấn luyện (T6 iv)"),
          ("DIV2K_valid_HR", False, "validation của tiền huấn luyện; phép thử ngữ cảnh và tương tác (T6 v)"),
          ("EarVN1.0", False, "N2: ước lượng suy giảm, bộ phân loại, ảnh thêm, test ngoài thực tế"),
          ("awex", False, "test chéo ngoài thực tế"),
          ("landmarks", False, "N1 và độ lệch điểm mốc (tùy chọn)")]


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)


def check_ami(d: Path) -> tuple[bool, str]:
    items = ami.scan_ami(d)
    rep = ami.check_ami(items)
    if rep["n_images"] != 700 or rep["n_subjects"] != 100 or rep["incomplete_subjects"] or rep["duplicate_keys"]:
        return False, f"cần 700 file NNN_<góc>_ear.jpg của 100 người; thấy {rep['n_images']} ảnh, {rep['n_subjects']} người"
    sizes = Counter()
    for it in items[::25]:
        with Image.open(it.path) as im:
            sizes[im.size] += 1
    if set(sizes) != {(492, 702)}:
        return False, f"ảnh phải cỡ 492×702 (rộng × cao); thấy {dict(sizes)}"
    other = len(_images(d)) - 700
    return True, "700 ảnh, 100 người, 492×702" + (f" (bỏ qua {other} file ảnh khác)" if other else "")


def check_subject_tree(d: Path, min_subjects: int) -> tuple[bool, str]:
    imgs = _images(d)
    if not imgs:
        return False, "không có ảnh nào"
    loose = [p for p in imgs if p.parent == d]
    subj = Counter(p.parent.name for p in imgs if p.parent != d)
    if loose and not subj:
        return False, "ảnh nằm thẳng trong thư mục gốc; cần dạng <gốc>/<người>/<ảnh>"
    depth = Counter(len(p.relative_to(d).parts) for p in imgs)
    msg = f"{len(imgs)} ảnh, {len(subj)} thư mục người"
    if len(depth) > 1 or 2 not in depth:
        msg += f"; CHÚ Ý: độ sâu thư mục không đều {dict(depth)} (mã người lấy từ tên thư mục chứa ảnh)"
    if len(subj) < min_subjects:
        return False, msg + f"; ít hơn {min_subjects} người, kiểm lại cấu trúc"
    return True, msg


def check_flat(d: Path, min_images: int) -> tuple[bool, str]:
    n = len(_images(d))
    return n >= min_images, f"{n} ảnh" + ("" if n >= min_images else f" (mong đợi từ {min_images})")


def check_landmarks(d: Path) -> tuple[bool, str]:
    pairs = find_pairs(d)
    return bool(pairs), f"{len(pairs)} cặp ảnh và file .pts cùng tên"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw")
    for name, _, _ in LAYOUT:
        ap.add_argument(f"--{name.lower().replace('_', '-').replace('.', '')}", default=None,
                        help=f"đường dẫn riêng cho {name} (mặc định <root>/{name})")
    a = ap.parse_args(argv)
    checks = {"AMI": check_ami, "DIV2K_train_HR": lambda d: check_flat(d, 800),
              "DIV2K_valid_HR": lambda d: check_flat(d, 100), "EarVN1.0": lambda d: check_subject_tree(d, 100),
              "awex": lambda d: check_subject_tree(d, 100), "landmarks": check_landmarks}
    bad = 0
    for name, required, use in LAYOUT:
        d = Path(getattr(a, name.lower().replace(".", "")) or Path(a.root) / name)
        if not d.is_dir():
            status, msg = ("THIẾU" if required else "chưa có"), "không thấy thư mục"
            bad += required
        else:
            ok, msg = checks[name](d)
            status = "ĐẠT" if ok else "SAI"
            bad += (not ok) and required
        print(f"[{status:7s}] {name:15s} {d}\n            {msg}\n            dùng cho: {use}")
    w = Path("weights")
    n = len(list(w.glob("*.pth"))) if w.is_dir() else 0
    print(f"[{'ĐẠT' if n >= 24 else 'THIẾU':7s}] trọng số        {w}  ({n}/24 file .pth; chạy bash scripts/get_weights.sh)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
