"""Chuẩn nén khác JPEG, ảnh dò tổng hợp và nhãn seed (TODO.md, việc Q1 đến Q3)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from earsr.degrade import ops  # noqa: E402
from earsr.degrade.pipelines import degrade  # noqa: E402
from earsr.report.arms import arm_of  # noqa: E402


def _img(h=96, w=64, seed=0):
    r = np.random.default_rng(seed)
    x = cv2.GaussianBlur(r.uniform(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 3)
    return np.clip(x, 0, 255).astype(np.uint8)


def test_webp_avif_kinds_keep_size_and_lose_more_at_low_quality():
    hr = _img(100, 68)
    lr = ops.bicubic_down(hr, 4)
    for codec, hi, lo in (("webp", 95, 60), ("avif", 90, 60)):
        a, b = degrade(hr, 4, f"bic{codec}{hi}"), degrade(hr, 4, f"bic{codec}{lo}")
        assert a.shape == b.shape == lr.shape
        err = [float(((x.astype(float) - lr) ** 2).mean()) for x in (a, b)]
        assert 0 < err[0] < err[1]


def test_synth_prep_gives_short_side_and_compression():
    from recog_eval import synth_prep

    big = _img(300, 210)
    r = {"path": "x/1.jpg"}
    clean = synth_prep("bic", 36, 4)(big, r)
    comp = synth_prep("bicjpeg75", 36, 4)(big, r)
    assert min(clean.shape[:2]) == 36 and clean.shape == comp.shape
    assert not np.array_equal(clean, comp)


def test_seed_tag_gives_separate_arm():
    assert arm_of("N2_span-zero+xearvn_pub_rand_x4_hrall_est_f3") == ("span/est", 3)
    assert arm_of("N2_span-zero+pa_pub_rand_x4_hrall_bic_f1") == ("span/bic [AMI only]", 1)
    assert arm_of("N2_span-zero+xearvn+s101_pub_rand_x4_hrall_est_f1") == ("span/est [seed 101]", 1)
