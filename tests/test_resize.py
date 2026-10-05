import numpy as np

from earsr.data.resize import center_crop_to_multiple, imresize, modcrop

REF = np.load(__file__.replace("test_resize.py", "fixtures/resize_ref.npz"))


def test_matches_matlab_compatible_reference():
    """So với hàm imresize tương thích MATLAB của BasicSR (kết quả lưu sẵn)."""
    x = REF["x"].astype(np.float64) / 255
    for key, scale in (("d4", 0.25), ("d2", 0.5), ("u2", 2.0), ("d3", 1 / 3)):
        out = imresize(x, scale)
        assert out.shape == REF[key].shape
        assert np.abs(out - REF[key]).max() < 2e-6, key


def test_uint8_rounding_and_out_size():
    x = REF["x"]
    a = imresize(x, 0.25)
    assert a.dtype == np.uint8 and a.shape == (10, 14, 3)
    b = imresize(x, out_size=(10, 14))
    assert np.array_equal(a, b)
    ref = np.clip(np.round(REF["d4"] * 255), 0, 255)
    assert np.abs(a.astype(int) - ref).max() <= 1


def test_constant_image_stays_constant():
    x = np.full((33, 47, 3), 137, dtype=np.uint8)
    for kw in ({"scale": 0.25}, {"scale": 2.0}, {"out_size": (9, 13)}):
        assert np.all(imresize(x, **kw) == 137)


def test_crops():
    x = np.zeros((30, 43, 3), np.uint8)
    assert modcrop(x, 4).shape == (28, 40, 3)
    assert center_crop_to_multiple(x, 4).shape == (28, 40, 3)
