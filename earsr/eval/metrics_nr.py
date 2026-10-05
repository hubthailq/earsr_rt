"""Chỉ số không tham chiếu của NTIRE Image SR (NIQE, MANIQA, MUSIQ, CLIP-IQA) và
điểm cảm nhận tổng.

    Score = (1 - LPIPS) + (1 - DISTS) + CLIPIQA + MANIQA + MUSIQ / 100 + max(0, (10 - NIQE) / 10)

Các chỉ số này cần thư viện ``pyiqa`` và trọng số tải từ mạng. Chúng chỉ đáng
tin ở ảnh đủ lớn, nên kế hoạch chỉ tính ở cỡ 244×348 px. Chỉ số nào không nạp
được thì trả về NaN kèm lý do, không dừng chương trình.

CHƯA CHẠY THẬT: máy dựng project không tải được trọng số. Tên chỉ số theo
pyiqa; nếu phiên bản pyiqa của bạn đặt tên khác, sửa bảng ``PYIQA_NAMES``.
"""
from __future__ import annotations

import numpy as np

PYIQA_NAMES = {"niqe": "niqe", "maniqa": "maniqa", "musiq": "musiq", "clipiqa": "clipiqa"}


def ntire_perceptual_score(lpips: float, dists: float, clipiqa: float, maniqa: float, musiq: float,
                           niqe: float) -> float:
    """Điểm cảm nhận tổng của NTIRE 2025 Image SR ×4. Cao hơn là tốt hơn."""
    return (1 - lpips) + (1 - dists) + clipiqa + maniqa + musiq / 100.0 + max(0.0, (10.0 - niqe) / 10.0)


def add_ntire_score(row: dict) -> dict:
    """Thêm cột ``ntire_score`` nếu dòng có đủ sáu chỉ số và không cái nào là NaN."""
    keys = ("lpips", "dists", "clipiqa", "maniqa", "musiq", "niqe")
    if all(k in row and row[k] == row[k] for k in keys):
        row["ntire_score"] = ntire_perceptual_score(*(float(row[k]) for k in keys))
    return row


class NoReferenceMetrics:
    def __init__(self, device: str = "cpu", want: tuple[str, ...] = ("niqe", "maniqa", "musiq", "clipiqa")):
        self.device = device
        self.fn: dict = {}
        self.reason: dict = {}
        self.info: dict = {}
        try:
            import pyiqa  # type: ignore
        except Exception as e:
            self.reason = {k: f"{type(e).__name__}: {e}" for k in want}
            return
        for name in want:
            try:
                self.fn[name] = pyiqa.create_metric(PYIQA_NAMES[name], device=device)
                self.info[name] = f"pyiqa {getattr(pyiqa, '__version__', '?')}, {PYIQA_NAMES[name]}"
            except Exception as e:
                self.reason[name] = f"{type(e).__name__}: {str(e)[:200]}"

    @property
    def available(self) -> list[str]:
        return list(self.fn)

    def __call__(self, sr: np.ndarray) -> dict:
        import torch

        out = {}
        if not self.fn:
            return out
        with torch.no_grad():
            x = torch.from_numpy(sr).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
            for k, f in self.fn.items():
                try:
                    out[k] = float(f(x).item())
                except Exception:
                    out[k] = float("nan")
        return out
