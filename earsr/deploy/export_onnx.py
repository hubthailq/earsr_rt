"""Xuất mô hình sang ONNX với kích thước vào cố định, để đo trên thiết bị.

Mặc định opset 13: TensorRT 8.2 của JetPack 4.6 (Jetson Nano bản 2019) không
đọc được opset cao hơn. Mô hình được xuất ở dạng deploy (đã gộp nhánh).
"""
from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(model: torch.nn.Module, path: str | Path, input_hw: tuple[int, int], opset: int = 13,
                check: bool = True) -> dict:
    """Xuất và (nếu có onnxruntime) so đầu ra với PyTorch. Trả về thông tin kiểm tra."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    for m in model.modules():
        if hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
    # Ảnh thử trơn (nhiễu tần thấp), giống ảnh thật hơn nhiễu trắng. Với nhiễu
    # trắng, mạng đã huấn luyện cho đầu ra rất lớn và phép so sai số tuyệt đối mất nghĩa.
    g = torch.Generator().manual_seed(0)
    low = torch.rand(1, 3, max(2, input_hw[0] // 8), max(2, input_hw[1] // 8), generator=g)
    x = torch.nn.functional.interpolate(low, size=tuple(input_hw), mode="bicubic", align_corners=False).clamp(0, 1)
    kw = dict(input_names=["lr"], output_names=["sr"], opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, x, str(path), dynamo=False, **kw)
    except TypeError:  # torch cũ không có tham số dynamo
        torch.onnx.export(model, x, str(path), **kw)
    info = {"path": str(path), "opset": opset, "input_h": input_hw[0], "input_w": input_hw[1], "checked": False}
    if check:
        try:
            import onnx
            import onnxruntime as ort

            ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            y_onnx = sess.run(None, {"lr": x.numpy()})[0]
            with torch.no_grad():
                y_torch = model(x).numpy()
            d = float(abs(y_onnx - y_torch).max())
            info.update(checked=True, max_abs_diff=d, rel_diff=d / max(1e-12, float(abs(y_torch).max())), ops=ops)
        except ImportError as e:
            info["check_skipped"] = str(e)
    return info
