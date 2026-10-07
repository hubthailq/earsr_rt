#!/usr/bin/env python3
"""Đo nhận dạng tai trên ảnh nhỏ THẬT sau khi phóng bằng từng phương pháp (việc 2 của docs/story-imavis.md).

  python scripts/recog_eval.py --root data/raw/EarVN1.0 --roles splits/earvn_roles.json \
      --recognizer runs/RECOG_resnet18/ckpt.pt --models bicubic span_ch48 disp26 --runs runs/N2_*+* \
      --out results/recog/resnet18

Giao thức: với mỗi người của nhóm chấm (chưa từng dùng để huấn luyện SR hay mạng nhận dạng), một nửa số ảnh lớn làm
ảnh đăng ký, mẫu của người đó là trung bình đặc trưng. Ảnh dò là ảnh nhỏ thật (cạnh ngắn trong --probe-short), phóng ×4
bằng từng phương pháp rồi đưa về cỡ vào của mạng nhận dạng bằng cùng một phép thu phóng. Phương pháp ``direct`` bỏ
bước phóng; ``ref_large`` dùng nửa ảnh lớn còn lại làm ảnh dò, cho biết mạng nhận dạng làm được tới đâu.
Mỗi phương pháp ghi <tên>.csv (theo ảnh dò, cùng thứ tự với probes.csv) và <tên>.npy (điểm với mẫu của mọi người;
không đưa vào git). Tổng hợp: summarize_recog.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from earsr.eval.infer import make_predictor  # noqa: E402
from earsr.io import imread_rgb, imwrite_rgb  # noqa: E402
from earsr.recog.data import INPUT_HW, list_images, split_gallery_probe, to_input  # noqa: E402
from earsr.recog.metrics import score_probes, templates  # noqa: E402
from earsr.recog.model import load_recognizer  # noqa: E402


@torch.no_grad()
def embed(model, root: Path, rows: list[dict], device: str, upscale=None, save_dir: Path | None = None,
          save_n: int = 0, batch: int = 128) -> np.ndarray:
    out, buf = [], []

    def flush():
        if buf:
            out.append(model.embed(torch.stack(buf).to(device)).float().cpu().numpy())
            buf.clear()

    for i, r in enumerate(rows):
        img = imread_rgb(root / r["path"])
        if upscale is not None:
            img = upscale(img)
            if save_dir is not None and i < save_n:
                imwrite_rgb(save_dir / (Path(r["path"]).as_posix().replace("/", "__").rsplit(".", 1)[0] + ".png"), img)
        buf.append(to_input(img, INPUT_HW))
        if len(buf) == batch:
            flush()
    flush()
    return np.concatenate(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--role", nargs="+", default=["test", "viewer"],
                    help="nhóm được chấm: những người không dùng để huấn luyện SR hay mạng nhận dạng")
    ap.add_argument("--recognizer", required=True, help="ckpt của train_recognizer.py, hoặc imagenet-resnet50")
    ap.add_argument("--recognizer-init", default="imagenet", choices=["imagenet", "none"],
                    help="chỉ cho mạng đối chứng imagenet-*; 'none' để thử khi không tải được trọng số")
    ap.add_argument("--models", nargs="*", default=["bicubic", "span_ch48", "disp26"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--probe-short", type=int, nargs=2, default=[24, 48])
    ap.add_argument("--gallery-min-short", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-sr", default=None, help="lưu ảnh đã phóng của --save-n ảnh dò đầu (cho hình minh họa)")
    ap.add_argument("--save-n", type=int, default=16)
    ap.add_argument("--save-match", default=None, help="chỉ lưu ảnh của phương pháp có tên khớp biểu thức này")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N ảnh dò đầu của mỗi người (để thử)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ck = load_recognizer(a.recognizer, a.device, init=a.recognizer_init)
    seen = set(ck.get("extra", {}).get("subjects", []))
    rows = list_images(root, a.roles, tuple(a.role), cache=out / "scan.csv")
    leak = seen & {r["subject"] for r in rows}
    if leak:
        raise SystemExit(f"DỪNG: mạng nhận dạng đã học trên {len(leak)} người của nhóm chấm (ví dụ {sorted(leak)[:3]})")
    sp = split_gallery_probe(rows, tuple(a.probe_short), a.gallery_min_short)
    if a.limit:
        cnt, keep = {}, []
        for r in sp["probe"]:
            cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
            if cnt[r["subject"]] <= a.limit:
                keep.append(r)
        sp["probe"] = keep
    if not sp["probe"]:
        raise SystemExit("không có ảnh dò nào")
    g_subj = np.array([r["subject"] for r in sp["gallery"]])
    uniq, temp = templates(embed(model, root, sp["gallery"], a.device), g_subj)
    meta = {"role": a.role, "recognizer": a.recognizer, "recognizer_cfg": ck["cfg"], "subjects": uniq.tolist(),
            "n_gallery": len(sp["gallery"]), "n_ref": len(sp["ref"]), "n_probe": len(sp["probe"]),
            "dropped_subjects": sp["dropped"], "gallery_only_subjects": sp["gallery_only"],
            "n_probe_subjects": len({r["subject"] for r in sp["probe"]}), "probe_short": a.probe_short, "gallery_min_short": a.gallery_min_short,
            "scale": a.scale, "input_hw": list(INPUT_HW), "limit": a.limit}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    for nm in ("probe", "ref", "gallery"):
        with open(out / f"{nm}s.csv" if nm != "gallery" else out / "gallery.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["path", "subject", "short"])
            wr.writerows([r["path"], r["subject"], r["short"]] for r in sp[nm])
    print(f"nhóm chấm {a.role}: {len(uniq)} người được đăng ký ({len(sp['gallery'])} ảnh), trong đó "
          f"{meta['n_probe_subjects']} người có ảnh dò nhỏ thật ({len(sp['probe'])} ảnh); {len(sp['ref'])} ảnh dò lớn; "
          f"bỏ {len(sp['dropped'])} người thiếu ảnh lớn", flush=True)

    # (tên, hàm dựng bộ phóng hoặc None, danh sách ảnh dò)
    jobs = [("ref_large", None, sp["ref"]), ("direct", None, sp["probe"])]
    jobs += [(m, (lambda m=m: make_predictor(m, a.scale, a.device)), sp["probe"]) for m in a.models]
    for rd in a.runs:
        from earsr.train.finetune import predictor_from_run

        jobs.append((Path(rd).name, (lambda rd=rd: predictor_from_run(rd, a.device)), sp["probe"]))
    for name, make, probes in jobs:
        f = out / f"{name}.csv"
        if f.exists() and not a.overwrite:
            continue
        up = make() if make else None
        save = Path(a.save_sr) / name if (a.save_sr and up is not None and
                                          (not a.save_match or re.search(a.save_match, name))) else None
        emb = embed(model, root, probes, a.device, upscale=up, save_dir=save, save_n=a.save_n)
        s = score_probes(emb, np.array([r["subject"] for r in probes]), uniq, temp)
        np.save(out / f"{name}.npy", s["scores"].astype(np.float32))
        with open(f, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["subject", "rank", "genuine", "max_impostor"])   # cùng thứ tự với probes.csv (hoặc ref.csv)
            for r, rk, g, mi in zip(probes, s["rank"], s["genuine"], s["max_impostor"]):
                wr.writerow([r["subject"], int(rk), f"{g:.5f}", f"{mi:.5f}"])
        print(f"xong {name}: rank-1 {100 * float((s['rank'] == 1).mean()):.2f}% trên {len(probes)} ảnh dò", flush=True)
        del up


if __name__ == "__main__":
    main()
