"""Đổi kiểu đệm viền của một mạng bất kỳ (N5b), không phụ thuộc kiến trúc.

Mọi ``nn.Conv2d`` có nhân lớn hơn 1×1 và có đệm được chuyển sang kiểu đệm mới.
Kiểu đệm không có tham số học, nên trọng số đã có vẫn nạp được; mạng chỉ cho
kết quả khác ở vùng sát mép. Dùng cho các thân ở dạng đã gộp nhánh (mọi mô hình
NTIRE trong kho). Thân SPAN cấu hình được của project tự lo kiểu đệm ở cả nhánh
huấn luyện (``span.py``), không qua hàm này.

Giới hạn: phép đệm viết tay bằng ``F.pad`` hoặc ``F.conv2d`` bên trong ``forward``
không bị đổi. ``count_padded_convs`` cho biết đã đổi bao nhiêu lớp; kiểm thử so
đầu ra ở lòng ảnh để chắc không lớp nào bị sót hoặc đổi sai.
"""
from __future__ import annotations

import torch.nn as nn

MODES = ("zeros", "replicate", "reflect")
ALIAS = {"zero": "zeros", "zeros": "zeros", "replicate": "replicate", "reflect": "reflect"}


def _targets(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and max(m.kernel_size) > 1 and isinstance(m.padding, tuple) and max(m.padding) > 0:
            yield m


def count_padded_convs(model: nn.Module) -> int:
    return sum(1 for _ in _targets(model))


def set_padding_mode(model: nn.Module, mode: str) -> int:
    """Đặt kiểu đệm cho mọi tích chập có đệm. Trả về số lớp đã đổi."""
    mode = ALIAS.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"kiểu đệm phải thuộc {MODES}")
    n = 0
    for m in _targets(model):
        m.padding_mode = mode
        # nn.Conv2d dùng thuộc tính này khi padding_mode khác 'zeros'
        m._reversed_padding_repeated_twice = tuple(x for p in reversed(m.padding) for x in (p, p))
        n += 1
    return n
