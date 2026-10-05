"""Đo vùng nhìn (receptive field) của một mạng SR bằng gradient.

Lấy gradient của ô đầu ra ở giữa theo ảnh vào. ``radius`` là khoảng cách
Chebyshev xa nhất (tính theo điểm ảnh LR) mà gradient còn khác 0: vùng nhìn lý
thuyết. ``r95`` là bán kính nhỏ nhất chứa 95% năng lượng gradient: vùng nhìn
hiệu dụng. Nếu ``radius`` chạm mép ảnh thử thì mạng có phép toán toàn cục (hoặc
vùng nhìn lớn hơn ảnh thử) và ``saturated`` là True.
"""
from __future__ import annotations

import numpy as np
import torch


def measure_receptive_field(model: torch.nn.Module, scale: int, size: int = 97, n_inputs: int = 3,
                            seed: int = 0, tol: float = 0.0) -> dict:
    """Tính trên một bản sao float64 của mô hình và đếm mọi gradient khác 0
    (``tol=0``), để kết quả không phụ thuộc độ lớn của trọng số."""
    import copy

    if size % 2 == 0:
        raise ValueError("size phải lẻ để có điểm giữa")
    model = copy.deepcopy(model).double().eval()
    g = torch.Generator().manual_seed(seed)
    c = size // 2
    acc = torch.zeros(size, size, dtype=torch.float64)
    for _ in range(n_inputs):
        x = torch.rand(1, 3, size, size, generator=g, dtype=torch.float64).requires_grad_(True)
        try:
            y = model(x)
            y[..., c * scale:(c + 1) * scale, c * scale:(c + 1) * scale].sum().backward()
        except RuntimeError as e:  # ví dụ mạng có phép toán inplace không lấy được gradient
            return {"radius": None, "r95": None, "saturated": None, "test_size": size,
                    "error": str(e).split("\n")[0][:160]}
        acc += x.grad.detach().abs().sum(1)[0].double()
    acc = acc.numpy()
    yy, xx = np.mgrid[0:size, 0:size]
    cheb = np.maximum(np.abs(yy - c), np.abs(xx - c))
    nz = acc > tol * acc.max()
    radius = int(cheb[nz].max()) if nz.any() else 0
    e = acc ** 2
    tot = e.sum()
    r95 = 0
    for r in range(c + 1):
        if e[cheb <= r].sum() >= 0.95 * tot:
            r95 = r
            break
    return {"radius": radius, "r95": int(r95), "saturated": bool(radius >= c), "test_size": size}
