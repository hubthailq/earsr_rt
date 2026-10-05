"""Biến thể thân cho N5b: kiểu đệm viền và tỉ lệ sâu, rộng.

Ba hình dạng có số phép tính xấp xỉ bằng nhau (số kênh bình phương nhân số tích
chập 3×3 ở thân):
    ref : 48 kênh, 6 khối  (SPAN gốc)        48² × 20 = 46.080
    wide: 64 kênh, 3 khối  (nông và rộng)    64² × 11 = 45.056
    deep: 36 kênh, 11 khối (sâu và hẹp)      36² × 35 = 45.360
Số phép tính bằng nhau chưa chắc cho độ trễ bằng nhau. Trước khi dùng, đo độ
trễ thật (scripts/bench_local.py, deploy/jetson) và chỉnh số kênh cho tới khi
chênh không quá 10% (scripts/match_latency.py tìm số kênh).

Ngoài ba tên hình dạng, variant nhận hai thẻ số: ``cNN`` (số kênh) và ``bNN``
(số khối), ví dụ ``zero-c56`` là "mốc nới rộng cho bằng độ trễ", ``reflect-c40-b9``.
Phần sau dấu ``+`` (ví dụ ``zero+gan``, ``zero+lr1e-4``) chỉ là nhãn của lần
chạy, không đổi kiến trúc.
"""
from __future__ import annotations

import re

from .span import PADDING_MODES, SPAN

SHAPES: dict[str, tuple[int, int]] = {"ref": (48, 6), "wide": (64, 3), "deep": (36, 11)}
_PAD_ALIAS = {"zero": "zeros", "zeros": "zeros", "replicate": "replicate", "reflect": "reflect"}


_NUM = re.compile(r"^([cb])(\d{1,3})$")


def arch_part(variant: str) -> str:
    """Bỏ phần nhãn sau dấu '+': 'zero-c56+gan' -> 'zero-c56'."""
    return variant.split("+")[0]


def parse_variant_full(variant: str) -> tuple[str, int, int]:
    """Trả về (kiểu đệm, số kênh, số khối)."""
    pad, (ch, nb) = "zeros", SHAPES["ref"]
    for p in arch_part(variant).split("-"):
        m = _NUM.match(p)
        if p in _PAD_ALIAS:
            pad = _PAD_ALIAS[p]
        elif p in SHAPES:
            ch, nb = SHAPES[p]
        elif m:
            if m.group(1) == "c":
                ch = int(m.group(2))
            else:
                nb = int(m.group(2))
        else:
            raise ValueError(f"không hiểu '{p}' trong variant '{variant}'")
    assert pad in PADDING_MODES
    if ch < 4 or nb < 2:
        raise ValueError(f"variant '{variant}': số kênh phải từ 4, số khối từ 2")
    return pad, ch, nb


def parse_variant(variant: str) -> tuple[str, str]:
    """'reflect-wide' -> ('reflect', 'wide'); 'zero' -> ('zeros', 'ref'). Hình dạng
    không trùng tên nào trong SHAPES thì trả về 'custom'."""
    pad, ch, nb = parse_variant_full(variant)
    shape = next((k for k, v in SHAPES.items() if v == (ch, nb)), "custom")
    return pad, shape


def is_published_shape(variant: str) -> bool:
    """True nếu variant có đúng kiến trúc của trọng số công bố (đệm số 0, 48 kênh, 6 khối)."""
    return parse_variant_full(variant) == ("zeros", *SHAPES["ref"])


def has_published_channels(variant: str) -> bool:
    """True nếu số kênh và số khối khớp trọng số công bố (kiểu đệm không có tham số, nên không cần khớp)."""
    return parse_variant_full(variant)[1:] == SHAPES["ref"]


def build_span_variant(variant: str = "zero", scale: int = 4, deploy: bool = False, img_range: float = 1.0) -> SPAN:
    """``img_range`` chỉ cần khác 1 khi sắp nạp trọng số công bố học ở dải khác; nạp checkpoint
    của project thì dải giá trị đến từ checkpoint."""
    pad, ch, nb = parse_variant_full(variant)
    return SPAN(feature_channels=ch, n_blocks=nb, upscale=scale, padding_mode=pad, deploy=deploy, img_range=img_range)


def all_variants() -> list[str]:
    """Năm biến thể của T6 (iv): thân tham chiếu, hai kiểu đệm khác, hai hình dạng khác."""
    return ["zero", "replicate", "reflect", "zero-wide", "zero-deep"]
