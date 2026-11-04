"""Kiểu suy giảm cho đối chứng thứ tự (phản biện 04/11) và phép tách phần dư của scripts/analyze_blocks.py."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from earsr.degrade import ops  # noqa: E402
from earsr.degrade.pipelines import SHIFT_RANGE, _shift, degrade  # noqa: E402


def _img(h=96, w=64, seed=0):
    r = np.random.default_rng(seed)
    x = cv2.GaussianBlur(r.uniform(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 3)
    return np.clip(x, 0, 255).astype(np.uint8)


def test_default_jpeg_is_420():
    ok, buf = cv2.imencode(".jpg", _img()[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
    b = buf.tobytes()
    k = b.find(b"\xff\xc0")
    assert b[k + 11] == 0x22 and b[k + 14] == 0x11   # Y 2×2, Cb 1×1: 4:2:0


def test_chroma444_differs_and_bicjpeg_unchanged():
    hr = _img()
    a = degrade(hr, 4, "bicjpeg75")
    assert np.array_equal(a, ops.jpeg(ops.bicubic_down(hr, 4), 75))
    assert not np.array_equal(a, degrade(hr, 4, "bicjpeg75c444"))


def test_shift_kinds_keep_size_and_are_deterministic():
    hr = _img()
    for k in ("shiftjpeg75", "dbljpeg93to75", "jpeg75bic"):
        x = degrade(hr, 4, k, key="a")
        assert x.shape == (24, 16, 3)
        assert np.array_equal(x, degrade(hr, 4, k, key="a"))
    dy, dx = _shift(0, "a")
    assert SHIFT_RANGE[0] <= dy <= SHIFT_RANGE[1] and SHIFT_RANGE[0] <= dx <= SHIFT_RANGE[1]
    assert not np.array_equal(degrade(hr, 4, "shiftjpeg75", key="a"), degrade(hr, 4, "bicjpeg75"))


def test_jpeg_shifted_zero_offset_is_plain_jpeg():
    x = _img(32, 32)
    assert np.array_equal(ops.jpeg_shifted(x, 75, 0, 0), ops.jpeg(x, 75))


def test_block_and_frequency_splits_add_up():
    from analyze_blocks import block_class, stats_one

    c = block_class(64, 64)
    assert set(np.unique(c)) == {0, 1, 2, 3}
    assert c[0, 0] == 0 and c[4 * 3, 4 * 3] == 3
    hr, sr = _img(), _img(seed=1)
    pos, frq, mse = stats_one(sr, hr)
    assert np.isclose(pos.sum(), mse) and np.isclose(frq.sum(), mse)
