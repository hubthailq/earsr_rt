import numpy as np
import torch

from earsr.eval.context_test import RING_EDGES, ContextConfig, make_context_frame, ring_index, run_context_test
from earsr.io import to_tensor, to_uint8
from earsr.data.resize import imresize
from earsr.models.span import SPAN


def _model():
    torch.manual_seed(0)
    m = SPAN(feature_channels=8, n_blocks=2, deploy=True).eval()  # vùng nhìn 9 px
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return m


def _predict(m):
    @torch.no_grad()
    def f(lr):
        return to_uint8(m(to_tensor(lr)))
    return f


def test_ring_index():
    r = ring_index((20, 20), 2)
    assert r.shape == (40, 40) and r[0, 0] == 0 and r[2, 2] == 1 and r[8, 8] == 4 and r[20, 20] == len(RING_EDGES) - 1


def test_frame_geometry():
    cfg = ContextConfig(win_hw=(51, 36), margins=(0, 4, 24))
    rng = np.random.default_rng(0)
    frame, c = make_context_frame(rng.integers(0, 256, (702, 492, 3), dtype=np.uint8), cfg)
    assert frame.shape == (396, 336, 3) and c.win_hw == (51, 36)
    frame, c = make_context_frame(rng.integers(0, 256, (492, 702, 3), dtype=np.uint8), cfg)
    assert frame.shape == (336, 396, 3) and c.win_hw == (36, 51)  # ảnh ngang: cửa sổ xoay ngang


def test_context_test_measures_what_it_claims():
    """m = 0 phải trùng với chạy thẳng trên cửa sổ; ngữ cảnh rộng hơn vùng nhìn
    phải trùng với phần tương ứng khi chạy cả khung."""
    m = _model()
    predict = _predict(m)
    cfg = ContextConfig(win_hw=(28, 24), margins=(0, 4, 12), prepad=(("replicate", 12),), scale=4)
    rng = np.random.default_rng(1)
    import cv2

    frame = cv2.GaussianBlur(rng.integers(0, 256, ((28 + 24) * 4, (24 + 24) * 4, 3), dtype=np.uint8), (0, 0), 2.0)
    rows = {r["cond"]: r for r in run_context_test(predict, frame, cfg)}
    assert set(rows) == {"m0", "m4", "m12", "rep12"}
    lr_full = imresize(frame, out_size=(52, 48))
    hr_win = frame[48:48 + 112, 48:48 + 96]
    from earsr.eval.metrics import rgb_to_y

    def mse(sr):
        return float(((rgb_to_y(sr) - rgb_to_y(hr_win)) ** 2).mean())

    direct = predict(np.ascontiguousarray(lr_full[12:40, 12:36]))
    assert abs(rows["m0"]["mse_y"] - mse(direct)) < 1e-9
    whole = predict(lr_full)[48:48 + 112, 48:48 + 96]
    assert abs(rows["m12"]["mse_y"] - mse(whole)) < 1e-9  # 12 px > vùng nhìn 9 px
    # ngữ cảnh 4 px, vùng nhìn 9 px: chỉ các điểm cách mép dưới 5 px bị ảnh hưởng,
    # nên vành trong cùng (từ 8 px trở vào) phải giống hệt khi có đủ ngữ cảnh
    assert abs(rows["m4"]["mse_ring5"] - rows["m12"]["mse_ring5"]) < 1e-9
    assert rows["m0"]["mse_ring0"] != rows["m12"]["mse_ring0"]
