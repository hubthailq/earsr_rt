#!/usr/bin/env python3
"""Xuất mô hình sang Core ML (.mlpackage) để đo độ trễ trên iPhone bằng Xcode (việc 4 của docs/story-imavis.md).

  .venv-coreml/bin/python scripts/export_coreml.py                       # các mốc mặc định, trọng số công bố
  .venv-coreml/bin/python scripts/export_coreml.py --runs runs/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f2

Ảnh vào cố định 68×48 (cao × rộng), lô 1: cỡ ảnh vào lớn nhất của bài ở ×4 (cạnh ngắn 48 px). Mô hình được gộp nhánh
(dạng deploy) trước khi xuất. Sau khi xuất, đầu ra Core ML được so với PyTorch trên cùng một ảnh, ngay trên máy Mac.
Độ trễ KHÔNG đo ở đây: mở từng .mlpackage bằng Xcode và chạy trên iPhone (deploy_ios/README.md).
Cần coremltools (không có sẵn trong Python của hệ thống): python3 -m venv --system-site-packages .venv-coreml &&
.venv-coreml/bin/python -m pip install coremltools
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from earsr.eval.complexity import count_params  # noqa: E402
from earsr.models.registry import SPECS, build_model  # noqa: E402

INPUT_HW = (68, 48)
DEFAULT = ["span_ch48", "disp26", "edsr_baseline", "swinir_light", "rrdb_psnr", "bsrgan"]


def to_deploy(model: torch.nn.Module) -> torch.nn.Module:
    model = model.cpu().eval()
    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
    return model


def test_image(hw: tuple[int, int]) -> torch.Tensor:
    """Ảnh thử trơn (nhiễu tần thấp), như ở export_onnx.py: với nhiễu trắng phép so sai số tuyệt đối mất nghĩa."""
    g = torch.Generator().manual_seed(0)
    low = torch.rand(1, 3, max(2, hw[0] // 8), max(2, hw[1] // 8), generator=g)
    return torch.nn.functional.interpolate(low, size=hw, mode="bicubic", align_corners=False).clamp(0, 1)


def export(model: torch.nn.Module, path: Path, precision: str, target: str) -> dict:
    import coremltools as ct

    model = to_deploy(model)
    x = test_image(INPUT_HW)
    with torch.no_grad():
        y = model(x).numpy()
        traced = torch.jit.trace(model, x, check_trace=False)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="lr", shape=tuple(x.shape))], outputs=[ct.TensorType(name="sr")],
                    convert_to="mlprogram", minimum_deployment_target=getattr(ct.target, target),
                    compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32)
    path.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(path))
    info = {"path": str(path), "precision": precision, "target": target, "input_h": INPUT_HW[0], "input_w": INPUT_HW[1],
            "params": count_params(model), "checked": False}
    try:   # chạy thử trên máy Mac (chỉ CPU) và so với PyTorch
        out = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY).predict({"lr": x.numpy()})["sr"]
        d = np.abs(out.astype(np.float64) - y)
        mse = float(((np.clip(out, 0, 1) - np.clip(y, 0, 1)) ** 2).mean())
        info.update(checked=True, max_abs_diff=float(d.max()), psnr_vs_torch_db=float(10 * np.log10(1.0 / max(mse, 1e-12))),
                    out_shape=list(out.shape))
    except Exception as e:   # ví dụ máy không chạy được Core ML
        info["check_skipped"] = f"{type(e).__name__}: {str(e)[:160]}"
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=DEFAULT, help="tên mô hình trong kho (trọng số công bố)")
    ap.add_argument("--runs", nargs="*", default=[], help="thư mục lần chạy của train.py (mô hình của bài)")
    ap.add_argument("--out", default="deploy_ios/models")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--target", default="iOS17", help="phiên bản iOS tối thiểu (tên trong coremltools.target)")
    a = ap.parse_args(argv)
    out, index = Path(a.out), []
    jobs = [(n, "zoo", (lambda n=n: build_model(n))) for n in a.models if SPECS[n].scale == 4]
    for rd in a.runs:
        from earsr.train.finetune import load_run_model

        cfg = json.loads((Path(rd) / "config.json").read_text())
        jobs.append((cfg["run_id"].replace("+", "p"), "trained", (lambda rd=rd: load_run_model(rd)[0])))
    for name, kind, make in jobs:
        try:
            info = export(make(), out / f"{name}.mlpackage", a.precision, a.target)
        except Exception as e:   # toán tử không chuyển được: ghi lại, không dừng cả lượt
            info = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"KHÔNG chuyển được {name}: {info['error']}", flush=True)
        info.update(name=name, kind=kind)
        index.append(info)
        if "error" not in info:
            print(f"{name}: {info['params'] / 1e3:.0f} nghìn tham số; lệch tối đa so với PyTorch "
                  f"{info.get('max_abs_diff', float('nan')):.4f}; PSNR giữa hai đầu ra "
                  f"{info.get('psnr_vs_torch_db', float('nan')):.1f} dB", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    # bản sao trong results/ (đi theo git) để bản thảo lấy số của phép so với PyTorch; gộp với các lượt xuất trước
    keep = Path("results/coreml_export.json")
    old = {d["name"]: d for d in json.loads(keep.read_text())} if keep.exists() else {}
    old.update({d["name"]: d for d in index})
    keep.write_text(json.dumps(list(old.values()), indent=1))
    print(f"đã ghi {len(index)} mục vào {out / 'index.json'} và cập nhật {keep}")


if __name__ == "__main__":
    main()
