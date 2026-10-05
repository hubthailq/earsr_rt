"""Mã lần chạy. Hai cấu hình khác nhau không bao giờ cho cùng một mã.

    {exp}_{backbone}-{variant}_{pretrain}_{protocol}_x{scale}_hr{tier}_{degrade}_f{fold}

variant : zero | replicate | reflect (kiểu đệm), có thể kèm wide | deep, ví dụ ``reflect-wide``
pretrain: pub (trọng số công bố) | ps{N} (rút gọn, N% số bước) | pf (đủ) | none (học từ đầu)
protocol: native (cắt từ ảnh gốc) | fixed (ảnh HR đúng cỡ) | rand (tỉ lệ ngẫu nhiên) | na (không huấn luyện)
tier    : cạnh ngắn của ảnh đáp án khi test; ``all`` nếu một mô hình dùng cho mọi cỡ
Seed bằng số fold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"^[A-Za-z0-9.+-]+$")
_PRETRAIN = re.compile(r"^(pub|pf|none|ps\d{1,3})$")
PROTOCOLS = ("native", "fixed", "rand", "na")


@dataclass(frozen=True)
class RunId:
    exp: str
    backbone: str
    variant: str
    pretrain: str
    protocol: str
    scale: int
    tier: str
    degrade: str
    fold: int

    def __post_init__(self):
        for name in ("exp", "backbone", "variant", "degrade"):
            v = getattr(self, name)
            if not _TOKEN.match(v) or "_" in v:
                raise ValueError(f"{name}='{v}' không hợp lệ (chỉ chữ, số, dấu chấm, cộng, gạch nối; không gạch dưới)")
        if "-" in self.backbone:
            raise ValueError("backbone không được chứa gạch nối (gạch nối ngăn backbone với variant)")
        if not _PRETRAIN.match(self.pretrain):
            raise ValueError(f"pretrain='{self.pretrain}' không hợp lệ")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol phải thuộc {PROTOCOLS}")
        if not re.match(r"^(\d+|all)$", str(self.tier)):
            raise ValueError("tier phải là số hoặc 'all'")
        if not 0 <= int(self.fold) <= 5:
            raise ValueError("fold từ 1 đến 5 (0 cho lần chạy không theo fold, ví dụ tiền huấn luyện)")

    def __str__(self) -> str:
        return (f"{self.exp}_{self.backbone}-{self.variant}_{self.pretrain}_{self.protocol}"
                f"_x{self.scale}_hr{self.tier}_{self.degrade}_f{self.fold}")

    @property
    def seed(self) -> int:
        return int(self.fold)

    @staticmethod
    def parse(s: str) -> "RunId":
        m = re.match(r"^([^_]+)_([^_]+?)-([^_]+)_([^_]+)_([^_]+)_x(\d+)_hr([^_]+)_([^_]+)_f(\d+)$", s)
        if not m:
            raise ValueError(f"không đọc được mã lần chạy: {s}")
        e, b, v, p, pr, sc, t, d, f = m.groups()
        return RunId(e, b, v, p, pr, int(sc), t, d, int(f))
