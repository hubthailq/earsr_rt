"""Đo độ trễ theo giao thức mục 1.5: làm nóng, đo nhiều lượt, báo trung vị và
phân vị 95. Chỉ tính lượt chạy của mô hình, lô 1.

Hàm này đo trên máy đang chạy (CPU hoặc GPU của PyTorch). Nó dùng để so tương
đối giữa các mô hình trên cùng một máy và để ghép các biến thể theo độ trễ. Số
đo cho bài báo phải lấy trên thiết bị (deploy_jetson/).
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_hw: tuple[int, int], device: str = "cpu",
                    warmup: int = 50, runs: int = 500, half: bool = False) -> dict:
    model = model.to(device).eval()
    x = torch.rand(1, 3, *input_hw, device=device)
    if half:
        model, x = model.half(), x.half()
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    ts = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        model(x)
        sync()
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.quantile(ts, 0.95)),
            "mean_ms": float(ts.mean()), "min_ms": float(ts.min()), "warmup": warmup, "runs": runs,
            "input_h": input_hw[0], "input_w": input_hw[1], "device": device, "half": half}
