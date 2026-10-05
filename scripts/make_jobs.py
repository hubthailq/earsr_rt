#!/usr/bin/env python3
"""Sinh danh sách lệnh cho từng khối thí nghiệm của giai đoạn 2 (mục 1.10 của kế hoạch).
Mỗi dòng là một lệnh; chạy tuần tự bằng ``scripts/run_queue.py``.

  python scripts/make_jobs.py lr    --ami-raw /path/AMI --bench data/bench/ami > jobs/lr.txt
  python scripts/make_jobs.py t6ii  --ami-raw ... --bench ...                  > jobs/t6ii.txt
  python scripts/make_jobs.py t6iii ...                                        > jobs/t6iii.txt
  python scripts/make_jobs.py pre   --div2k-train /path/DIV2K_train_HR --div2k-val /path/DIV2K_valid_HR > jobs/pre.txt
  python scripts/make_jobs.py t6iv  ... > jobs/t6iv.txt
  python scripts/make_jobs.py s2    ... > jobs/s2.txt

Các khối:
  n2    phép thử sớm của N2: hai mốc × ba kiểu suy giảm lúc huấn luyện (bic, generic, est), fold 2
  pad   phép thử rẻ của N5b: trọng số công bố, ba kiểu đệm, tinh chỉnh, fold 2, 3, 4 (không tiền huấn luyện)
  lr    dò tốc độ học: mỗi mốc ba mức, chỉ fold 2 (không chấm test)
  t6ii  ba giao thức (native, fixed, rand), fold 2, 3, 4
  t6iii học từ đầu trên ảnh tai so với trọng số công bố
  pre   tiền huấn luyện với CÙNG công thức: năm biến thể SPAN, mốc nới rộng (--ctl-variants) và
        mốc khác họ kiến trúc (--ctl-backbones, mặc định rlfn). --budget 100 cho ngân sách đủ
  t6iv  tinh chỉnh mọi mô hình của 'pre' từ checkpoint tương ứng (nhánh có kiểm soát)
  s2    ô chính: các mốc real-time × 5 fold (CHỈ chạy sau điểm kiểm tra 2: có fold 1 và 5)

Danh sách mốc, số bước và ba mức tốc độ học là giá trị khởi đầu; sửa bằng tham số
dòng lệnh sau lần chạy thử đầu tiên. Tốc độ học tốt nhất của từng mốc (từ khối
'lr') được truyền lại bằng ``--lr-of span=2e-4 rlfn=1e-4``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.models.variants import all_variants  # noqa: E402

DEV_FOLDS = (2, 3, 4)
ALL_FOLDS = (1, 2, 3, 4, 5)
LRS = ("5e-5", "1e-4", "2e-4")


def main(argv=None) -> list[str]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("block", choices=["n2", "pad", "lr", "t6ii", "t6iii", "pre", "t6iv", "s2"])
    ap.add_argument("--ami-raw", default="/path/AMI")
    ap.add_argument("--bench", default="data/bench/ami")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--baselines", nargs="+", default=["span", "disp26", "errn26"],
                    help="các mốc real-time (span = SPAN 48 kênh cấu hình được; tên khác lấy từ kho mô hình)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--scratch-iters", type=int, default=100000)
    ap.add_argument("--degrade", default="bic")
    ap.add_argument("--degrade-params", default=None)
    ap.add_argument("--protocol", default="rand", help="giao thức đã chọn ở T6 (ii), cho các khối sau đó")
    ap.add_argument("--lr-of", nargs="*", default=[], help="tốc độ học đã chọn, dạng tên=giá_trị")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--full-iters", type=int, default=500000)
    ap.add_argument("--div2k-train", default="/path/DIV2K_train_HR")
    ap.add_argument("--div2k-val", default="/path/DIV2K_valid_HR")
    ap.add_argument("--variants", nargs="*", default=None, help="mặc định: năm biến thể của T6 (iv)")
    ap.add_argument("--ctl-backbones", nargs="*", default=["rlfn"],
                    help="mốc khác họ kiến trúc được học lại từ đầu với cùng công thức (nhánh có kiểm soát)")
    ap.add_argument("--ctl-variants", nargs="*", default=[],
                    help="variant SPAN thêm cho nhánh có kiểm soát, ví dụ mốc nới rộng zero-c56 (từ match_latency.py)")
    ap.add_argument("--extra", default="", help="tham số thêm cho mọi lệnh train.py")
    a = ap.parse_args(argv)
    lr_of = dict(kv.split("=") for kv in a.lr_of)
    deg = f"--degrade {a.degrade}" + (f" --degrade-params {a.degrade_params}" if a.degrade_params else "")
    base = (f"python scripts/train.py --ami-raw {a.ami_raw} --bench {a.bench} --out {a.out} {deg}"
            + (f" {a.extra}" if a.extra else ""))
    variants = a.variants or all_variants()

    def tr(exp, backbone, fold, protocol, tier=144, variant="zero", pretrain="pub", more=""):
        lr = f" --lr {lr_of[backbone]}" if backbone in lr_of and "--lr " not in more else ""
        return (f"{base} --exp {exp} --backbone {backbone} --variant {variant} --pretrain {pretrain} "
                f"--protocol {protocol} --tier {tier} --fold {fold} --iters {a.iters}{lr}{(' ' + more) if more else ''}")

    def pre_dir(v, backbone="span"):
        tag = "pf" if a.budget >= 100 else f"ps{a.budget}"
        return f"{a.out}/PRE_{backbone}-{v}_{tag}_na_x4_hrall_bic_f0/ckpt/best.pt", tag

    # nhánh có kiểm soát: mọi mô hình học lại từ đầu với cùng dữ liệu và cùng số bước
    controlled = [("span", v) for v in list(variants) + list(a.ctl_variants)] + [(b, "zero") for b in a.ctl_backbones]

    jobs = []
    if a.block == "n2":
        # Phép thử sớm của N2 (bản 22): cùng một mốc, ba kiểu suy giảm lúc huấn luyện, fold 2.
        if not a.degrade_params:
            raise SystemExit("khối n2 cần --degrade-params (configs/degrade_estimated.json từ fit_degradation.py)")
        for b in a.baselines[:2]:
            for kind in ("bic", "generic", "est"):
                j = tr("N2", b, 2, a.protocol, more="--no-test")
                j = j.replace(deg, f"--degrade {kind}" + (f" --degrade-params {a.degrade_params}" if kind == "est" else ""))
                jobs.append(j)
    elif a.block == "pad":
        # Phép thử rẻ của N5b (bản 22): trọng số công bố, đổi kiểu đệm, rồi tinh chỉnh; không cần tiền huấn luyện.
        for b in a.baselines:
            for v in ("zero", "replicate", "reflect"):
                for f in DEV_FOLDS:
                    jobs.append(tr("T6pad", b, f, a.protocol, variant=v))
    elif a.block == "lr":
        for b in a.baselines:
            for lr in LRS:
                jobs.append(tr("LR", b, 2, a.protocol, more=f"--lr {lr} --tag lr{lr} --no-test"))
    elif a.block == "t6ii":
        for f in DEV_FOLDS:
            for b in a.baselines:
                jobs.append(tr("T6ii", b, f, "native"))
                jobs.append(tr("T6ii", b, f, "rand"))
            for i, b in enumerate(a.baselines):
                for t in ((96, 144, 192) if i < 2 else (144,)):
                    jobs.append(tr("T6ii", b, f, "fixed", tier=t))
    elif a.block == "t6iii":
        for f in DEV_FOLDS:
            jobs.append(tr("T6iii", "span", f, a.protocol))
            jobs.append(tr("T6iii", "span", f, a.protocol, pretrain="none").replace(
                f"--iters {a.iters}", f"--iters {a.scratch_iters}"))
    elif a.block == "pre":
        for b, v in controlled:
            jobs.append(f"python scripts/pretrain.py --backbone {b} --variant {v} --budget {a.budget} "
                        f"--full-iters {a.full_iters} --train-dir {a.div2k_train} --val-dir {a.div2k_val} --out {a.out}")
    elif a.block == "t6iv":
        for b, v in controlled:
            ck, tag = pre_dir(v, b)
            for f in DEV_FOLDS:
                jobs.append(tr("T6iv", b, f, a.protocol, variant=v, pretrain=tag, more=f"--init-ckpt {ck}"))
    elif a.block == "s2":
        for b in a.baselines:
            for f in ALL_FOLDS:
                jobs.append(tr("S2", b, f, a.protocol))
    print("\n".join(jobs))
    return jobs


if __name__ == "__main__":
    main()
