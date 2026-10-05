import numpy as np
import pandas as pd

from earsr.stats.bootstrap import interaction, mean_ci, paired_diff
from earsr.stats.mde import mde_paired
from earsr.stats.multiple import holm
from earsr.stats.ranking import kendall_tau_b, pairwise_signs


def _sim(seed, n_subj=60, per=7, effect=0.05):
    r = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_subj), per)
    d = effect + 0.1 * r.normal(0, 1, n_subj)[cl] + r.normal(0, 0.3, n_subj * per)
    return d, cl


def test_cluster_bootstrap_coverage_is_nominal():
    """Trên dữ liệu giả có hiệu ứng biết trước, khoảng tin cậy 95% phải phủ giá trị thật khoảng 95% số lần."""
    hit = 0
    for i in range(200):
        d, cl = _sim(i)
        e = paired_diff(d, np.zeros_like(d), cl, n_boot=400, seed=i)
        hit += e.lo <= 0.05 <= e.hi
    assert 0.90 <= hit / 200 <= 0.99


def test_image_level_bootstrap_would_be_too_narrow():
    """Lấy mẫu lại theo ảnh (bỏ qua cụm) cho khoảng hẹp hơn: lý do phải lấy mẫu lại theo người."""
    d, cl = _sim(3, effect=0.0)
    by_subject = paired_diff(d, np.zeros_like(d), cl, n_boot=800)
    by_image = paired_diff(d, np.zeros_like(d), np.arange(len(d)), n_boot=800)
    assert (by_subject.hi - by_subject.lo) > (by_image.hi - by_image.lo)


def test_relative_difference_and_nan_handling():
    b = np.array([10.0, 10, 10, 10, np.nan, 10])
    a = b * 0.9
    e = paired_diff(a, b, np.array([0, 0, 1, 1, 2, 2]), relative=True, n_boot=200)
    assert abs(e.point + 0.1) < 1e-9 and e.n_obs == 5
    m = mean_ci([1.0, 2.0, np.nan, 3.0], [0, 0, 1, 1], n_boot=100)
    assert abs(m.point - 2.0) < 1e-9


def test_interaction_detects_difference_of_gains():
    r = np.random.default_rng(0)
    cl = np.repeat(np.arange(50), 4)
    b1 = r.normal(30, 1, 200)
    b2 = r.normal(30, 1, 200)
    a1 = b1 + 0.30 + r.normal(0, 0.05, 200)   # phần hơn lớn ở điều kiện 1
    a2 = b2 + 0.05 + r.normal(0, 0.05, 200)   # phần hơn nhỏ ở điều kiện 2
    e = interaction(a1, b1, cl, a2, b2, cl, n_boot=500)
    assert abs(e.point - 0.25) < 0.02 and e.excludes_zero()
    e0 = interaction(a1, b1, cl, a1, b1, cl, n_boot=200)
    assert abs(e0.point) < 1e-12


def test_holm_known_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.5]), [1.0, 1.0])


def test_mde_formula():
    d, cl = _sim(1)
    r = mde_paired(d, cl, n_subjects=60)
    per = np.array([d[cl == u].mean() for u in np.unique(cl)])
    assert abs(r["mde"] - (1.959964 + 0.841621) * per.std(ddof=1) / np.sqrt(60)) < 1e-4


def test_tau_b_treats_inseparable_pairs_as_ties():
    r = np.random.default_rng(0)
    cl = np.repeat(np.arange(40), 5)
    base = r.normal(30, 1, 200)
    a = pd.DataFrame({"m1": base + 0.5, "m2": base + 0.3, "m3": base + 0.3 + r.normal(0, 0.2, 200)})
    b = pd.DataFrame({"m1": base + 0.1, "m2": base + 0.6, "m3": base + 0.6 + r.normal(0, 0.2, 200)})
    sa, sb = pairwise_signs(a, cl, n_boot=300), pairwise_signs(b, cl, n_boot=300)
    assert sa[("m1", "m2")] == 1 and sb[("m1", "m2")] == -1
    assert sa[("m2", "m3")] == 0  # không tách được: hòa
    t = kendall_tau_b(sa, sb)
    assert t["discordant"] == 2 and ("m1", "m2") in t["reversals"] and t["tau_b"] < 0
