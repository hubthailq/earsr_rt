"""Cấu hình pytest. Một số kiểm thử cần trọng số công bố hoặc bộ Set5; chúng tự
bỏ qua (skip) nếu thiếu:

    EARSR_WEIGHTS=/path/weights  EARSR_SET5=/path/Set5_HR  pytest -q
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Kiểm thử chỉ chạy mô hình rất nhỏ. Một luồng nhanh hơn nhiều luồng ở cỡ này và
# tránh tranh chấp luồng trên máy ít lõi. Đặt EARSR_TEST_THREADS để đổi.
try:
    import torch

    torch.set_num_threads(int(os.environ.get("EARSR_TEST_THREADS", "1")))
except ImportError:
    pass


@pytest.fixture(scope="session")
def weights_dir():
    d = Path(os.environ.get("EARSR_WEIGHTS", ROOT / "weights"))
    if not d.is_dir() or not any(d.iterdir()):
        pytest.skip("không có thư mục trọng số (EARSR_WEIGHTS)")
    return d


@pytest.fixture(scope="session")
def set5_dir():
    d = os.environ.get("EARSR_SET5")
    if not d or not Path(d).is_dir():
        pytest.skip("không có bộ Set5 (EARSR_SET5)")
    return Path(d)
