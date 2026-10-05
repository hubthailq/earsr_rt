"""Kiểm thử kho mô hình: nạp chặt trọng số công bố và tái lập PSNR trên Set5 ×4.

Cần EARSR_WEIGHTS và EARSR_SET5 (thư mục chứa ảnh HR của Set5; file có đuôi
``_HR.png`` hoặc mọi file .png nếu không có đuôi đó). Thiếu thì bỏ qua.
"""
import numpy as np
import pytest
import torch

from earsr.data.resize import imresize, modcrop
from earsr.eval.metrics import fr_metrics
from earsr.io import imread_rgb, to_tensor, to_uint8
from earsr.models.registry import SPECS, build_model

# PSNR-Y Set5 ×4 do tác giả công bố
# span_ch48: bảng 1 của bài SPAN (arXiv 2311.12770), dòng SPAN-S ×4 (48 kênh, 426 nghìn tham số); đo được 32,2005
PUBLISHED = {"swinir_light": 32.44, "rrdb_psnr": 32.73, "span_ch48": 32.20}
# PSNR-Y Set5 ×4 đo bằng chính project này lúc dựng kho (05/10/2026); dùng để phát hiện hồi quy
MEASURED = {"span_ch48_t44": 32.13, "span_ch28": 31.73, "span_ch26": 31.74, "rlfn": 32.10, "efdn": 32.08,
            "safmnpp": 32.01, "smfan": 32.04, "msrresnet": 32.22, "esrgan": 30.47, "bsrgan": 27.71,
            "realesrgan": 26.62, "realesr_compact": 26.64,
            # NTIRE 2026 ESR (đo ngày 05/10/2026; các mô hình cỡ 0,13 đến 0,25 triệu tham số)
            "span26": 31.73, "pds26": 31.76, "pkdsr26": 31.77, "dscf26": 31.74, "disp26": 31.94, "errn26": 31.89,
            # EDSR-baseline, trọng số chính thức (đo ngày 06/10/2026: 32,096). Không nằm trong PUBLISHED vì tác giả
            # không công bố số Set5 cho bản baseline; số họ công bố là 28,95 dB trên DIV2K 0801..0900 (RGB), và
            # project tái lập được 28,96 dB (đo tay một lần, xem docs/STATUS.md; quá chậm cho bộ kiểm thử).
            "edsr_baseline": 32.10}


def _set5(set5_dir, scale):
    files = sorted(set5_dir.glob("*_HR.png")) or sorted(set5_dir.glob("*.png"))
    hrs = [modcrop(imread_rgb(f), scale) for f in files]
    assert len(hrs) == 5, "Set5 phải có đúng 5 ảnh"
    return [(imresize(h, 1 / scale), h) for h in hrs]


def _psnr(name, pairs, wdir, scale=4):
    m = build_model(name, wdir=wdir)
    with torch.no_grad():
        return float(np.mean([fr_metrics(to_uint8(m(to_tensor(l))), h, scale)["psnr_y"] for l, h in pairs]))


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_reproduces_published_psnr(name, weights_dir, set5_dir):
    if not (weights_dir / SPECS[name].weights).is_file():
        pytest.skip("thiếu trọng số")
    assert abs(_psnr(name, _set5(set5_dir, 4), weights_dir) - PUBLISHED[name]) <= 0.05


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_no_regression_against_first_measurement(name, weights_dir, set5_dir):
    if not (weights_dir / SPECS[name].weights).is_file():
        pytest.skip("thiếu trọng số")
    assert abs(_psnr(name, _set5(set5_dir, 4), weights_dir) - MEASURED[name]) <= 0.02


def test_every_spec_is_consistent():
    for n, s in SPECS.items():
        assert s.scale in (2, 4) and s.group in ("light", "mid", "upper", "perceptual") and s.weights
        assert s.objective in ("psnr", "gan")
