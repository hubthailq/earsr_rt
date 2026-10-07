"""Đo nhận dạng: số đo, cách chia ảnh, và một lượt chạy đủ trên dữ liệu giả."""
import json

import cv2
import numpy as np
import pytest
import torch

from earsr.recog.data import INPUT_HW, list_images, split_gallery_probe, to_input
from earsr.recog.metrics import boot_stat, eer, eer_from_hist, score_probes, subject_hists, tar_from_hist, templates
from earsr.recog.model import EarEmbedder, load_recognizer, save_embedder
from earsr.report.arms import arm_of


def test_eer_and_rank_on_known_cases():
    rng = np.random.default_rng(0)
    assert eer(np.full(100, 0.9), np.full(500, 0.1)) == 0.0                      # tách hẳn
    same = rng.uniform(-1, 1, 20000)
    assert abs(eer(same[:10000], same[10000:]) - 0.5) < 0.02                      # không phân biệt được
    g, i = rng.normal(0.5, 0.1, 20000), rng.normal(0.3, 0.1, 20000)               # hai chuẩn cách nhau 2 sigma
    assert abs(eer(g, i) - 0.1587) < 0.01                                         # Phi(-1)
    uniq, temp = np.array(["a", "b", "c"]), np.eye(3)
    s = score_probes(np.array([[1.0, 0, 0], [0.2, 0.1, 0.9], [0.5, 0.6, 0.0]]), np.array(["a", "b", "b"]), uniq, temp)
    assert s["rank"].tolist() == [1, 3, 1] and np.allclose(s["genuine"], [1.0, 0.1, 0.6])
    assert np.allclose(s["max_impostor"], [0.0, 0.9, 0.5])
    with pytest.raises(ValueError):
        score_probes(np.eye(3)[:1], np.array(["z"]), uniq, temp)


def test_templates_and_bootstrap_by_subject():
    emb = np.array([[1.0, 0], [0.8, 0.6], [0, 1.0]])
    uniq, t = templates(emb, np.array(["a", "a", "b"]))
    assert uniq.tolist() == ["a", "b"] and np.allclose(np.linalg.norm(t, axis=1), 1) and np.allclose(t[1], [0, 1])
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    col, cl = np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"])
    u, hg, hi = subject_hists(scores, col, cl)
    assert hg.sum(1).tolist() == [2, 2] and hi.sum(1).tolist() == [2, 2]
    b = boot_stat([hg, hi], eer_from_hist, n_boot=200)
    assert b["lo"] <= b["point"] <= b["hi"] and b["n_clusters"] == 2
    # điểm đúng người: 0.9, 0.8, 0.7, 0.4; sai người: 0.1, 0.2, 0.3, 0.6. FAR = 0 cần ngưỡng trên 0.6: nhận ba trong bốn
    assert tar_from_hist(hg.sum(0), hi.sum(0), 0.0) == 0.75


def test_split_keeps_sizes_apart_and_drops_poor_subjects():
    rows = [{"path": f"s1/{i}.jpg", "subject": "s1", "short": 120} for i in range(6)]
    rows += [{"path": f"s1/p{i}.jpg", "subject": "s1", "short": 30} for i in range(3)]
    rows += [{"path": "s1/mid.jpg", "subject": "s1", "short": 70}]                 # không lớn, không nhỏ: bỏ
    rows += [{"path": f"s2/{i}.jpg", "subject": "s2", "short": 200} for i in range(2)]   # thiếu ảnh lớn
    rows += [{"path": "s2/p.jpg", "subject": "s2", "short": 40}]
    rows += [{"path": f"s3/{i}.jpg", "subject": "s3", "short": 150} for i in range(4)]   # đủ ảnh lớn, không có ảnh nhỏ
    sp = split_gallery_probe(rows)
    assert sp["dropped"] == ["s2"] and sp["gallery_only"] == ["s3"]
    assert len(sp["gallery"]) == 5 and len(sp["ref"]) == 5 and len(sp["probe"]) == 3
    assert {r["subject"] for r in sp["probe"]} == {"s1"}
    assert not {r["path"] for r in sp["gallery"]} & {r["path"] for r in sp["ref"]}
    assert all(r["short"] <= 48 for r in sp["probe"]) and all(r["short"] >= 96 for r in sp["gallery"] + sp["ref"])


def test_embedder_roundtrip_and_arm_names(tmp_path):
    m = EarEmbedder(5, emb=32).eval()
    x = torch.stack([to_input(np.random.default_rng(i).integers(0, 256, (40 + i, 30, 3), dtype=np.uint8)) for i in range(3)])
    assert x.shape == (3, 3, *INPUT_HW)
    e = m.embed(x)
    assert e.shape == (3, 32) and torch.allclose(e.norm(dim=1), torch.ones(3), atol=1e-5)
    y = torch.tensor([0, 1, 2])
    assert torch.all(m(x, y)[torch.arange(3), y] < m(x)[torch.arange(3), y])       # biên chỉ trừ ở lớp đúng
    save_embedder(m, tmp_path / "c.pt", extra={"subjects": ["a"]})
    m2, ck = load_recognizer(str(tmp_path / "c.pt"))
    assert torch.allclose(m2.embed(x), e, atol=1e-6) and ck["extra"]["subjects"] == ["a"]
    assert arm_of("N2_span-zero+xearvn_pub_rand_x4_hrall_est_f3") == ("span/est", 3)
    assert arm_of("N2_disp26-zero+pa_pub_rand_x4_hrall_bic_f2") == ("disp26/bic [AMI only]", 2)
    assert arm_of("span_ch48") == ("span_ch48 (published)", 0) and arm_of("bicubic") == ("bicubic", 0)


def _fake_set(root, roles_path):
    """6 người; mỗi người một màu và một hoa văn riêng, có ảnh lớn và ảnh nhỏ."""
    rng = np.random.default_rng(0)
    roles = {}
    for k in range(6):
        s = f"p{k}"
        roles[s] = "test" if k < 3 else "train"
        (root / s).mkdir(parents=True)
        base = rng.integers(0, 256, (12, 8, 3), dtype=np.uint8)
        for i in range(8):
            big = cv2.resize(base, (104, 152), interpolation=cv2.INTER_NEAREST)
            big = np.clip(big.astype(int) + rng.integers(-12, 13, big.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(root / s / f"big{i}.jpg"), big)
        for i in range(4):
            cv2.imwrite(str(root / s / f"small{i}.jpg"), cv2.resize(base, (28, 40), interpolation=cv2.INTER_AREA))
    roles_path.write_text(json.dumps(roles))


def test_recognition_pipeline_end_to_end(tmp_path):
    import sys

    sys.path.insert(0, "scripts")
    import recog_eval
    import summarize_recog
    import train_recognizer

    root, roles = tmp_path / "data", tmp_path / "roles.json"
    _fake_set(root, roles)
    assert len(list_images(root, roles, ("test",))) == 36
    with pytest.raises(SystemExit):                                               # nhóm chấm không được vào tập học
        train_recognizer.main(["--root", str(root), "--roles", str(roles), "--train-roles", "train", "test",
                               "--out", str(tmp_path / "bad")])
    info = train_recognizer.main(["--root", str(root), "--roles", str(roles), "--train-roles", "train", "--init", "none",
                                  "--epochs", "2", "--batch-size", "8", "--workers", "0", "--device", "cpu",
                                  "--out", str(tmp_path / "rec")])
    assert info["subjects"] == ["p3", "p4", "p5"] and (tmp_path / "rec" / "ckpt.pt").exists()
    out = tmp_path / "res"
    args = ["--root", str(root), "--roles", str(roles), "--recognizer", str(tmp_path / "rec" / "ckpt.pt"),
            "--role", "test", "--models", "bicubic", "--out", str(out), "--device", "cpu", "--save-sr", str(tmp_path / "sr"), "--save-n", "2"]
    recog_eval.main(args)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["subjects"] == ["p0", "p1", "p2"] and meta["n_probe"] == 12 and meta["n_gallery"] == 12
    for name, n in (("ref_large", 12), ("direct", 12), ("bicubic", 12)):
        assert np.load(out / f"{name}.npy").shape == (n, 3)
    assert len(list((tmp_path / "sr" / "bicubic").glob("*.png"))) == 2
    with pytest.raises(SystemExit):                                               # mạng nhận dạng đã thấy người được chấm
        recog_eval.main(["--role" if x == "--role" else ("train" if args[i - 1] == "--role" else x) for i, x in enumerate(args)])
    summarize_recog.main(["--in", str(out), "--out", str(tmp_path / "sum"), "--n-boot", "50"])
    import pandas as pd

    s = pd.read_csv(tmp_path / "sum" / "summary.csv")
    assert set(s.arm) == {"ref_large", "direct", "bicubic"} and s.rank1.between(0, 100).all()
    p = pd.read_csv(tmp_path / "sum" / "paired.csv")
    assert set(p.minus) == {"bicubic", "direct"} and "ref_large" not in set(p.arm)
