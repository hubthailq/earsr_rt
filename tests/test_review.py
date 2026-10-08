"""Các mốc thêm cho vòng phản biện: khử nén rồi phóng, RRDB học có nén, RRDB tinh chỉnh (scripts/run_review.sh)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from earsr.data import ami
from earsr.models.registry import CHAINS, RESTORERS, SPECS
from earsr.report.arms import arm_of

ROOT = Path(__file__).resolve().parents[1]


def test_chains_point_to_known_models():
    for name, (pre, post) in CHAINS.items():
        assert pre in RESTORERS and (post == "bicubic" or SPECS[post].scale == 4), name
        assert name not in SPECS            # tên chuỗi không được trùng tên một mạng phóng ảnh


def test_new_published_models_are_outside_the_sixteen():
    # bsrnet và realesrnet học có nén: không được lọt vào nhóm mô hình học bằng bicubic mà summarize_t2.py gộp (light, mid)
    for n in ("bsrnet", "realesrnet"):
        assert SPECS[n].group == "upper" and SPECS[n].objective == "psnr" and SPECS[n].scale == 4


def test_review_runs_are_pooled_like_n2_runs():
    assert arm_of("REV_rrdb-zero+xearvn_pub_rand_x4_hrall_est_f4") == ("rrdb/est", 4)
    assert arm_of("N2_span-zero+xearvn_pub_rand_x4_hrall_est_f4") == ("span/est", 4)
    assert arm_of("fbcnn_span_ch48") == ("fbcnn_span_ch48 (published)", 0)


def test_fbcnn_keeps_size_for_any_input():
    from earsr.models.zoo.fbcnn import FBCNNBlind

    torch.manual_seed(0)
    net = FBCNNBlind().eval()
    with torch.no_grad():
        for hw in ((51, 37), (24, 24), (68, 48)):
            assert net(torch.rand(1, 3, *hw)).shape == (1, 3, *hw)


def test_rrdb_alias_builds_the_published_backbone():
    from earsr.train.finetune import build_trainable

    m = build_trainable("rrdb", "zero", "none", 4)
    assert sum(p.numel() for p in m.parameters()) == 16_697_987
    with torch.no_grad():
        assert m(torch.rand(1, 3, 12, 10)).shape == (1, 3, 48, 40)


def test_rrdb_job_list_keeps_held_out_folds_closed(tmp_path):
    out = tmp_path / "jobs.txt"
    r = subprocess.run(["bash", "scripts/make_rrdb_jobs.sh", str(out)], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHON": sys.executable, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3 and all("--exp REV --backbone rrdb " in l and "--degrade est " in l and l.endswith("--amp") for l in lines)
    assert [l.split("--fold ")[1].split()[0] for l in lines] == ["2", "3", "4"]
    r = subprocess.run(["bash", "scripts/make_rrdb_jobs.sh", str(out)], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHON": sys.executable, "PATH": "/usr/bin:/bin:/usr/local/bin", "FOLDS": "1 5"})
    assert r.returncode != 0 and "FINAL_RUN=1" in r.stderr


def test_deblock_then_upscale_end_to_end(tmp_path, weights_dir, monkeypatch):
    need = [RESTORERS["fbcnn"].weights, SPECS["span_ch48"].weights]
    if not all((weights_dir / w).is_file() for w in need):
        pytest.skip("thiếu trọng số")
    monkeypatch.setenv("EARSR_WEIGHTS", str(weights_dir))
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for s in ("000", "001"):
        for v in ami.VIEWS:
            img = cv2.GaussianBlur(rng.integers(0, 256, (702, 492, 3), dtype=np.uint8), (0, 0), 6.0)
            cv2.imwrite(str(raw / f"{s}_{v}_ear.jpg"), img)
    bench = tmp_path / "bench"
    ami.build_ami_benchmark(raw, bench, tiers=(96,), strict=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate
    import summarize_review

    res = tmp_path / "res"
    evaluate.main(["--bench", str(bench), "--folds", "none", "--tiers", "96", "--kinds", "bicjpeg75", "--no-perceptual",
                   "--models", "bicubic", "fbcnn_bicubic", "fbcnn_span_ch48", "--out", str(res / "rev"), "--device", "cpu",
                   "--limit", "4"])
    assert (res / "rev" / "fbcnn_span_ch48__hr96_x4_bicjpeg75.csv").is_file()
    summarize_review.main(["--results", str(res)])
    text = (res / "rev_summary" / "summary.md").read_text()
    assert "fbcnn_span_ch48 (published)" in text and "so với bicubic" in text
