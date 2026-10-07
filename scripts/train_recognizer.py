#!/usr/bin/env python3
"""Huấn luyện mạng nhận dạng tai dùng để ĐO (không phải đóng góp của bài).

  python scripts/train_recognizer.py --root data/raw/EarVN1.0 --roles splits/earvn_roles.json --out runs/RECOG_resnet18

Chỉ học trên ảnh LỚN (cạnh ngắn từ --min-short) của những người có vai train, fit, clf. Nhóm test và viewer không bao
giờ được dùng ở đây: đó là những người sẽ được chấm. Mạng không thấy ảnh nhỏ hay ảnh đã phóng lúc học, nên không
thiên về kiểu ảnh của một phương pháp phóng nào. Mười phần trăm ảnh của mỗi người được giữ lại để theo dõi.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from earsr.io import imread_rgb  # noqa: E402
from earsr.recog.data import INPUT_HW, MEAN, STD, list_images, to_input  # noqa: E402
from earsr.recog.model import EarEmbedder, save_embedder  # noqa: E402

EVAL_ROLES = ("test", "viewer")


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, root, rows, label_of, train: bool, hw=INPUT_HW):
        import torchvision.transforms as T

        self.root, self.rows, self.label_of, self.train, self.hw = Path(root), rows, label_of, train, hw
        self.aug = T.Compose([
            T.RandomResizedCrop(hw, scale=(0.7, 1.0), ratio=(0.55, 0.9), interpolation=T.InterpolationMode.BICUBIC,
                                antialias=True),
            T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = imread_rgb(self.root / r["path"])
        x = self.aug(Image.fromarray(img)) if self.train else to_input(img, self.hw)
        return x, self.label_of[r["subject"]]


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    hit = n = 0
    for x, y in loader:
        hit += int((model(x.to(device)).argmax(1).cpu() == y).sum())
        n += len(y)
    return hit / max(n, 1)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--train-roles", nargs="+", default=["train", "fit", "clf"])
    ap.add_argument("--min-short", type=int, default=96)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/RECOG_resnet18")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N ảnh đầu của mỗi người (để thử)")
    a = ap.parse_args(argv)
    bad = set(a.train_roles) & set(EVAL_ROLES)
    if bad:
        raise SystemExit(f"DỪNG: vai {sorted(bad)} là nhóm được chấm, không được dùng để huấn luyện mạng nhận dạng")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rows = [r for r in list_images(a.root, a.roles, tuple(a.train_roles), cache=out / "scan.csv")
            if r["short"] >= a.min_short]
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    by = {s: v[:a.limit] if a.limit else v for s, v in by.items() if len(v) >= 2}
    subjects = sorted(by)
    label_of = {s: i for i, s in enumerate(subjects)}
    rng = np.random.default_rng(a.seed)
    tr, va = [], []
    for s in subjects:
        order = rng.permutation(len(by[s]))
        n_val = max(1, len(order) // 10)
        va += [by[s][i] for i in order[:n_val]]
        tr += [by[s][i] for i in order[n_val:]]
    print(f"mạng nhận dạng: {len(subjects)} người, {len(tr)} ảnh học, {len(va)} ảnh theo dõi "
          f"(vai {a.train_roles}, cạnh ngắn >= {a.min_short}); thiết bị {a.device}", flush=True)
    dl = torch.utils.data.DataLoader(TrainSet(a.root, tr, label_of, True), batch_size=a.batch_size, shuffle=True,
                                     num_workers=a.workers, drop_last=len(tr) > a.batch_size)
    dv = torch.utils.data.DataLoader(TrainSet(a.root, va, label_of, False), batch_size=128, num_workers=a.workers)
    model = EarEmbedder(len(subjects), a.arch, a.emb, a.init).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, a.epochs * len(dl)))
    t0, log = time.time(), []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = 0
        for x, y in dl:
            x, y = x.to(a.device), y.to(a.device)
            loss = F.cross_entropy(model(x, y), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(y)
            n += len(y)
        acc = accuracy(model, dv, a.device)
        log.append({"epoch": ep, "loss": tot / max(n, 1), "val_acc": acc})
        print(f"epoch {ep}/{a.epochs}  loss {tot / max(n, 1):.4f}  val_acc {acc:.4f}", flush=True)
    info = {"subjects": subjects, "train_roles": a.train_roles, "min_short": a.min_short, "n_train": len(tr),
            "n_val": len(va), "val_acc": log[-1]["val_acc"] if log else None, "init": a.init, "epochs": a.epochs,
            "seed": a.seed, "input_hw": list(INPUT_HW), "elapsed_s": round(time.time() - t0, 1)}
    save_embedder(model.cpu(), out / "ckpt.pt", extra=info)
    (out / "train_log.json").write_text(json.dumps({"info": info, "log": log}, indent=1, ensure_ascii=False))
    print(f"đã lưu {out / 'ckpt.pt'} (val_acc {info['val_acc']})")
    return info


if __name__ == "__main__":
    main()
