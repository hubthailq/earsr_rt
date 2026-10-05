#!/usr/bin/env python3
"""Tiền huấn luyện một biến thể thân trên ảnh tự nhiên (DIV2K hoặc tương đương).

  python scripts/pretrain.py --variant reflect --budget 20 --full-iters 500000 \
      --train-dir /path/DIV2K_train_HR --val-dir /path/DIV2K_valid_HR

``--budget N``: dùng N% của ``--full-iters`` (ngân sách rút gọn). Checkpoint
được lưu thêm ở 50% và 100% ngân sách, để kiểm điều kiện "thứ hạng các biến
thể giống nhau ở hai mốc". Sau khi chạy, script in kết quả kiểm "đường học đã
phẳng" (10% số bước cuối thêm dưới 0,02 dB).

Ảnh LR của patch được tạo bằng cách thu nhỏ một vùng cắt rộng hơn patch rồi bỏ
rìa, nên không sai ở mép patch và không cần lưu sẵn ảnh LR của cả bộ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from earsr.data.datasets import list_images  # noqa: E402
from earsr.data.resize import imresize, modcrop  # noqa: E402
from earsr.eval.metrics import crop_border, psnr, rgb_to_y  # noqa: E402
from earsr.io import imread_rgb, to_tensor, to_uint8  # noqa: E402
from earsr.train.finetune import build_trainable  # noqa: E402
from earsr.runid import RunId  # noqa: E402
from earsr.train.trainer import TrainConfig, curve_is_flat, train  # noqa: E402


class GenericPatchDataset(Dataset):
    """Patch ngẫu nhiên từ ảnh HR lớn. LR = bicubic của vùng cắt có rìa ``pad`` px LR."""

    def __init__(self, paths, scale=4, patch_lr=48, pad=4, seed=0, cache=256):
        self.paths, self.scale, self.p, self.pad, self.seed = list(paths), scale, patch_lr, pad, seed
        self._cache: dict = {}
        self._cache_max = cache

    def __len__(self):
        return 1 << 40

    def _img(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = imread_rgb(self.paths[i])
        return self._cache[i]

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, int(idx)])
        s, p, pad = self.scale, self.p, self.pad
        big = (p + 2 * pad) * s
        for _ in range(20):
            img = self._img(int(rng.integers(len(self.paths))))
            if img.shape[0] >= big and img.shape[1] >= big:
                break
        else:
            raise RuntimeError("không tìm được ảnh đủ lớn cho patch")
        y = int(rng.integers(img.shape[0] - big + 1))
        x = int(rng.integers(img.shape[1] - big + 1))
        crop = img[y:y + big, x:x + big]
        lr = imresize(crop, out_size=(p + 2 * pad, p + 2 * pad))[pad:pad + p, pad:pad + p]
        hr = crop[pad * s:(pad + p) * s, pad * s:(pad + p) * s]
        if rng.random() < 0.5:
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        k = int(rng.integers(4))
        lr, hr = np.rot90(lr, k), np.rot90(hr, k)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)
        return t(lr), t(hr)


def make_validator(val_paths, scale, device, max_images=20, max_side=480):
    pairs = []
    for p in list(val_paths)[:max_images]:
        hr = imread_rgb(p)
        h, w = hr.shape[:2]
        t, l = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
        hr = modcrop(hr[t:t + max_side, l:l + max_side], scale)
        pairs.append((imresize(hr, out_size=(hr.shape[0] // scale, hr.shape[1] // scale)), hr))

    @torch.no_grad()
    def validate(model):
        v = []
        for lr, hr in pairs:
            sr = to_uint8(model(to_tensor(lr).to(device)))
            v.append(psnr(crop_border(rgb_to_y(sr), scale), crop_border(rgb_to_y(hr), scale)))
        return float(np.mean(v))

    return validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="span",
                    help="span (cấu hình bằng --variant) hoặc một mô hình trong kho (rlfn, efdn, ...) học lại từ đầu "
                         "cho nhánh so sánh có kiểm soát: cùng dữ liệu, cùng số bước với mô hình đề xuất")
    ap.add_argument("--variant", default="zero")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--budget", type=int, default=20, help="phần trăm của --full-iters; 100 là tiền huấn luyện đủ")
    ap.add_argument("--full-iters", type=int, default=500000)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patch-lr", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    iters = max(1, a.full_iters * a.budget // 100)
    tag = "pf" if a.budget >= 100 else f"ps{a.budget}"
    if a.backbone != "span" and a.variant != "zero":
        raise SystemExit("--variant chỉ áp cho --backbone span")
    rid = RunId("PRE", a.backbone, a.variant, tag, "na", a.scale, "all", "bic", 0)
    model = build_trainable(a.backbone, a.variant, "none", a.scale)   # khởi tạo ngẫu nhiên, không nạp trọng số công bố
    train_set = GenericPatchDataset(list_images(a.train_dir), a.scale, a.patch_lr, seed=a.seed)
    validate = make_validator(list_images(a.val_dir), a.scale, a.device)
    cfg = TrainConfig(run_id=str(rid), out_dir=a.out, iters=iters, batch_size=a.batch_size, lr=a.lr,
                      val_every=min(a.val_every, max(1, iters // 20)), save_every=min(a.val_every, max(1, iters // 20)),
                      num_workers=a.workers, seed=a.seed, device=a.device, checkpoints_at=[iters // 2, iters],
                      extra={"backbone": a.backbone, "variant": a.variant, "scale": a.scale, "model_kind": "plain",
                             "objective": "l1", "train_dir": a.train_dir, "budget_pct": a.budget,
                             "full_iters": a.full_iters})
    print(train(model, train_set, validate, cfg))
    print("đường học:", curve_is_flat(Path(a.out) / str(rid) / "log.csv"))


if __name__ == "__main__":
    main()
