#!/usr/bin/env python3
"""Xuất các file ONNX cần đo trên thiết bị (T3) vào ``deploy_jetson/onnx/``.

Hai nhóm:
1. Mọi mô hình trong kho có trọng số, ở cỡ ảnh vào của mục 1.5:
   ×4: 68×48 (cao × rộng); ×2: 136×96.
2. Ba kiểu đệm (và hai hình dạng) của thân SPAN, trọng số ngẫu nhiên. Độ trễ
   không phụ thuộc giá trị trọng số; nhóm này trả lời câu hỏi "thiết bị có hỗ
   trợ kiểu đệm này không và chậm hơn bao nhiêu" trước khi tốn tiền huấn luyện.

Sau đó chép cả thư mục ``deploy_jetson`` sang thiết bị và chạy
``python3 bench_trtexec.py`` ở đó.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from earsr.deploy.export_onnx import export_onnx  # noqa: E402
from earsr.eval.complexity import count_flops, count_params  # noqa: E402
from earsr.models.registry import SPECS, build_model, list_models  # noqa: E402
from earsr.models.span import switch_all_to_deploy  # noqa: E402
from earsr.models.variants import all_variants, build_span_variant  # noqa: E402

INPUT_HW = {4: (68, 48), 2: (136, 96)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy_jetson/onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--skip-zoo", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variant SPAN cần xuất (mặc định: năm biến thể của T6 iv); ví dụ thêm zero-c56")
    ap.add_argument("--runs", nargs="*", default=[], help="thư mục lần chạy của train.py: xuất mô hình đã huấn luyện")
    a = ap.parse_args()
    out = Path(a.out)
    index = []
    if not a.skip_zoo:
        for name in list_models(available_only=True):
            spec = SPECS[name]
            m = build_model(name)
            hw = INPUT_HW[spec.scale]
            try:
                info = export_onnx(m, out / f"{name}.onnx", hw, a.opset)
            except Exception as e:  # ví dụ toán tử không xuất được ở opset này
                index.append({"name": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"KHÔNG xuất được {name}: {e}")
                continue
            info.update(name=name, kind="zoo", scale=spec.scale, group=spec.group, params=count_params(m),
                        flops_g_256=count_flops(m)["flops_g"])
            index.append(info)
            print(f"{name}: lệch ONNX so với PyTorch {info.get('max_abs_diff')}")
    for rd in a.runs:
        from earsr.train.finetune import load_run_model

        m, cfg = load_run_model(rd)
        switch_all_to_deploy(m.eval())
        sc = int(cfg["extra"]["scale"])
        name = cfg["run_id"].replace("+", "p")
        info = export_onnx(m, out / f"{name}.onnx", INPUT_HW[sc], a.opset)
        info.update(name=name, run_id=cfg["run_id"], kind="trained", scale=sc, params=count_params(m),
                    flops_g_256=count_flops(m)["flops_g"])
        index.append(info)
        print(f"{name}: lệch ONNX so với PyTorch {info.get('max_abs_diff')}")
    for v in (a.variants if a.variants is not None else all_variants()):
        torch.manual_seed(0)
        m = build_span_variant(v, scale=4, deploy=False).eval().switch_to_deploy()
        info = export_onnx(m, out / f"spanvar_{v}.onnx", INPUT_HW[4], a.opset)
        info.update(name=f"spanvar_{v}", kind="variant_random_init", scale=4, params=count_params(m),
                    flops_g_256=count_flops(m)["flops_g"])
        index.append(info)
        print(f"spanvar_{v}: toán tử {info.get('ops')}")
    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    print(f"đã ghi {len(index)} mục vào {out}/index.json")


if __name__ == "__main__":
    main()
