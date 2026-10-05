"""Độ lệch điểm mốc, sàn nhiễu của bộ dò, và hộp bao vùng tai (mục 1.9).

Độ lệch điểm mốc của một ảnh SR = trung bình theo điểm của khoảng cách giữa
điểm mốc dò trên ảnh SR và trên ảnh đáp án, chia cho đường chéo hộp bao điểm
mốc trên ảnh đáp án. Bộ dò chạy trên ảnh đã đưa về cỡ ``input_hw`` bằng hàm
thu phóng chung của project.

Sàn nhiễu = độ lệch giữa ảnh đáp án và chính nó sau một nhiễu nhẹ (nhiễu Gauss
σ = 2/255; nén JPEG mức 90). Thước đo chỉ dùng được ở một cỡ ảnh nếu độ lệch
của thân tinh chỉnh ít nhất gấp 2 lần sàn này.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch

from ..data.resize import imresize
from ..degrade import ops
from .nets import build_detector


def save_detector(path: str | Path, net, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": net.arch, "n_points": net.n_points, "input_hw": list(net.input_hw),
                "width": net.e0[0][0].out_channels if net.arch == "heatmap" else net.body[0][0].out_channels,
                "model": net.state_dict(), **(extra or {})}, path)


def load_detector(path: str | Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_detector(ck["arch"], ck["n_points"], ck["width"], tuple(ck["input_hw"]))
    net.load_state_dict(ck["model"])
    return net.eval(), ck


class LandmarkScorer:
    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        self.net, self.info = load_detector(ckpt)
        self.net.to(device)
        self.device = device
        self.input_hw = tuple(self.net.input_hw)

    @torch.no_grad()
    def points(self, img: np.ndarray) -> np.ndarray:
        """Điểm mốc (K, 2) dạng (x, y) theo toạ độ điểm ảnh của ``img``."""
        h, w = img.shape[:2]
        ih, iw = self.input_hw
        x = img if (h, w) == (ih, iw) else imresize(img, out_size=(ih, iw))
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        p = self.net.predict(t)[0].float().cpu().numpy()
        # về toạ độ ảnh gốc (quy ước tâm điểm ảnh)
        return np.stack([(p[:, 0] + 0.5) * w / iw - 0.5, (p[:, 1] + 0.5) * h / ih - 0.5], 1)

    @staticmethod
    def _dev(p: np.ndarray, ref: np.ndarray) -> float:
        diag = float(np.hypot(*(ref.max(0) - ref.min(0))))
        return float(np.linalg.norm(p - ref, axis=1).mean() / max(diag, 1e-6))

    def deviation(self, sr: np.ndarray, hr: np.ndarray, hr_points: np.ndarray | None = None) -> float:
        ref = self.points(hr) if hr_points is None else hr_points
        return self._dev(self.points(sr), ref)

    def noise_floor(self, hr: np.ndarray, seed: int = 0, hr_points: np.ndarray | None = None) -> dict:
        ref = self.points(hr) if hr_points is None else hr_points
        rng = np.random.default_rng(seed)
        return {"lm_floor_noise": self._dev(self.points(ops.gaussian_noise(hr, 2.0, rng)), ref),
                "lm_floor_jpeg": self._dev(self.points(ops.jpeg(hr, 90)), ref)}

    def box(self, hr: np.ndarray, margin: float = 0.10) -> tuple[float, float, float, float]:
        """Hộp bao vùng tai (trên, trái, dưới, phải) theo TỈ LỆ của ảnh, nới ``margin``
        mỗi phía, cắt về [0, 1]. Lưu theo tỉ lệ để dùng được cho mọi cỡ ảnh."""
        p = self.points(hr)
        h, w = hr.shape[:2]
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        c = lambda v: float(min(1.0, max(0.0, v)))
        return c((y0 - my) / h), c((x0 - mx) / w), c((y1 + my + 1) / h), c((x1 + mx + 1) / w)

    def metric_fn(self, floors: bool = True):
        """Hàm ``f(sr, hr, row) -> dict`` cho ``evaluate_on_bench(extra_metrics=...)``."""
        def f(sr, hr, row):
            ref = self.points(hr)
            out = {"lm_dev": self._dev(self.points(sr), ref)}
            if floors:
                out.update(self.noise_floor(hr, seed=zlib.crc32(str(row.get("file", "")).encode()), hr_points=ref))
            return out
        return f


def box_to_pixels(box_frac, h: int, w: int) -> tuple[int, int, int, int]:
    t, l, b, r = box_frac
    t, l = int(np.floor(t * h)), int(np.floor(l * w))
    b, r = int(np.ceil(b * h)), int(np.ceil(r * w))
    return max(0, t), max(0, l), min(h, max(b, t + 1)), min(w, max(r, l + 1))
