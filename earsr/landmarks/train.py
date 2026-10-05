"""Huấn luyện một bộ dò điểm mốc. Loss: L1 trên toạ độ (chuẩn hóa theo đường
chéo ảnh vào). Checkpoint chọn theo sai số trung bình chuẩn hóa (NME) trên phần
validation: khoảng cách trung bình chia cho đường chéo hộp bao điểm mốc.

CHƯA HUẤN LUYỆN trên bộ điểm mốc thật; số bước và tốc độ học là giá trị khởi đầu.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import LandmarkDataset
from .metric import save_detector
from .nets import build_detector


@torch.no_grad()
def nme(net, loader, device: str) -> float:
    net.eval()
    vals = []
    for x, y in loader:
        p = net.predict(x.to(device)).cpu()
        diag = (y.max(1).values - y.min(1).values).norm(dim=1).clamp(min=1e-6)
        vals += ((p - y).norm(dim=2).mean(1) / diag).tolist()
    return float(np.mean(vals))


def train_detector(train_items: list, val_items: list, out: str | Path, arch: str = "heatmap", width: int = 32,
                   input_hw: tuple[int, int] = (136, 96), iters: int = 20000, batch_size: int = 32,
                   lr: float = 1e-3, val_every: int = 500, workers: int = 4, device: str = "cpu",
                   seed: int = 0, extra: dict | None = None) -> dict:
    torch.manual_seed(seed)
    n_points = len(train_items[0][1])
    net = build_detector(arch, n_points, width, input_hw).to(device)
    ds = LandmarkDataset(train_items, input_hw, train=True, seed=seed)
    val = DataLoader(LandmarkDataset(val_items, input_hw, train=False), batch_size=batch_size, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    diag = float(np.hypot(*input_hw))
    step, best, best_step, hist = 0, float("inf"), -1, []
    while step < iters:
        ds.epoch += 1
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                            drop_last=len(ds) >= batch_size)
        for x, y in loader:
            net.train()
            loss = F.l1_loss(net.predict(x.to(device)), y.to(device)) / diag
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % val_every == 0 or step == iters:
                v = nme(net, val, device)
                hist.append({"step": step, "loss": loss.item(), "val_nme": v})
                if v < best:
                    best, best_step = v, step
                    save_detector(out, net, {"val_nme": v, "step": step, "n_train": len(train_items),
                                             "n_val": len(val_items), **(extra or {})})
                print(f"[điểm mốc {arch}] bước {step}/{iters}  loss {loss.item():.5f}  NME val {v:.4f} "
                      f"(tốt nhất {best:.4f})", flush=True)
            if step >= iters:
                break
    return {"best_val_nme": best, "best_step": best_step, "history": hist, "out": str(out)}
