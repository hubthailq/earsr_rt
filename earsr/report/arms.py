"""Gộp các lần chạy của khối n2 thành "nhánh": cùng thân, cùng kiểu suy giảm huấn luyện, khác fold."""
from __future__ import annotations

import re

RUN = re.compile(r"^N2_(?P<bb>[a-z0-9]+)-zero\+(?P<tag>[a-z]+)_pub_rand_x4_hrall_(?P<kind>[a-z0-9]+)_f(?P<fold>\d)$")
AMI_ONLY = " [AMI only]"
PUBLISHED = " (published)"


def arm_of(model: str) -> tuple[str, int]:
    """Tên lần chạy hoặc tên mô hình -> (nhánh, fold). Mô hình có trọng số công bố và mốc nội suy có fold 0.

    ``N2_span-zero+xearvn_pub_rand_x4_hrall_est_f3`` -> (``span/est``, 3); nhãn ``+pa`` (chỉ AMI) thêm hậu tố.
    """
    g = RUN.match(model)
    if not g:
        return (model if model in ("bicubic", "bicubic_sharp", "direct", "ref_large") else model + PUBLISHED), 0
    return f"{g['bb']}/{g['kind']}" + ("" if g["tag"] == "xearvn" else AMI_ONLY), int(g["fold"])
