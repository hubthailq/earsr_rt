"""Áp các tiêu chí đạt ghi trước trong ``configs/criteria.yaml`` lên bảng số đo
theo từng ảnh. Mỗi hàm trả về một dict có khóa ``pass`` (True, False, hoặc None
nếu thiếu dữ liệu) và các con số dẫn tới kết luận đó.

Mọi so sánh là ghép cặp theo từng ảnh (cùng khóa ảnh), bootstrap theo người.
Nhiều file (một file mỗi fold) được nối lại; mỗi người chỉ được có mặt ở một fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..stats.bootstrap import interaction, paired_diff


def load_criteria(path: str | Path = "configs/criteria.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def load(files) -> pd.DataFrame:
    if isinstance(files, (str, Path)):
        files = [files]
    d = pd.concat([pd.read_csv(f, dtype={"subject": str, "view": str, "key": str}) for f in files], ignore_index=True)
    if d.key.duplicated().any():
        raise ValueError(f"trùng khóa ảnh khi nối {len(files)} file (một ảnh có mặt ở hai fold?)")
    return d.sort_values("key").reset_index(drop=True)


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(a.key) != list(b.key):
        raise ValueError(f"hai bảng không cùng tập ảnh ({len(a)} và {len(b)} ảnh, chung {len(set(a.key) & set(b.key))})")
    return a, b


def gain(files_a, files_b, metric: str = "psnr_y", relative: bool = False, n_boot: int = 2000) -> dict:
    """Chênh lệch ghép cặp a − b của một số đo."""
    a, b = align(load(files_a), load(files_b))
    e = paired_diff(a[metric].values, b[metric].values, a.subject.values, relative=relative, n_boot=n_boot)
    return {"metric": metric, "relative": relative, "mean_a": float(a[metric].mean()), "mean_b": float(b[metric].mean()),
            **e.as_dict(), "n_folds": int(a.fold.nunique()) if "fold" in a else None}


def at_least(files_a, files_b, metric: str, minimum: float, relative: bool = False, higher_is_better: bool = True,
             n_boot: int = 2000) -> dict:
    """Đạt khi a hơn b ít nhất ``minimum`` và khoảng tin cậy của chênh lệch không chứa 0."""
    g = gain(files_a, files_b, metric, relative, n_boot)
    sign = 1.0 if higher_is_better else -1.0
    imp, lo = sign * g["point"], (g["lo"] if higher_is_better else -g["hi"])
    g.update(improvement=imp, improvement_lo=lo, minimum=minimum, **{"pass": bool(imp >= minimum and lo > 0)})
    return g


def not_worse(files_a, files_b, metric: str, max_deficit: float, relative: bool = False,
              higher_is_better: bool = True, n_boot: int = 2000) -> dict:
    """Đạt khi cận xấu của khoảng tin cậy cho thấy a không kém b quá ``max_deficit``."""
    g = gain(files_a, files_b, metric, relative, n_boot)
    worst = -g["lo"] if higher_is_better else g["hi"]     # mức kém lớn nhất còn nằm trong khoảng tin cậy
    g.update(worst_case_deficit=worst, max_deficit=max_deficit, **{"pass": bool(worst <= max_deficit)})
    return g


def same_sign_each_fold(files_a, files_b, metric: str, higher_is_better: bool = True) -> dict:
    a, b = align(load(files_a), load(files_b))
    d = (a[metric] - b[metric]) * (1 if higher_is_better else -1)
    per = d.groupby(a.fold).mean()
    return {"per_fold": {int(k): float(v) for k, v in per.items()}, "pass": bool((per > 0).all())}


# ----------------------------------------------------------------- tiêu chí của từng điểm mới

def n5b_variant_gain(variant_files, ref_files, crit: dict, metric: str = "psnr_y") -> dict:
    """N5b (b): biến thể hơn thân tham chiếu ít nhất 0,1 dB, cùng ngân sách tiền huấn luyện."""
    return at_least(variant_files, ref_files, metric, crit["N5b"]["b_variant_gain_db"]["min"])


def n5b_interaction(var_small, ref_small, var_large, ref_large, metric: str = "psnr_y", n_boot: int = 2000) -> dict:
    """N5b (c): phần hơn ở ảnh vào nhỏ lớn hơn phần hơn ở ảnh vào lớn."""
    a1, b1 = align(load(var_small), load(ref_small))
    a2, b2 = align(load(var_large), load(ref_large))
    same = sorted(set(a1.subject)) == sorted(set(a2.subject))
    e = interaction(a1[metric].values, b1[metric].values, a1.subject.values, a2[metric].values, b2[metric].values,
                    a2.subject.values, n_boot=n_boot, same_clusters=same)
    return {"gain_small": float((a1[metric] - b1[metric]).mean()), "gain_large": float((a2[metric] - b2[metric]).mean()),
            **e.as_dict(), "same_subjects": same, "pass": bool(e.lo > 0)}


def n5a_protocol(rand_by_tier: dict, fixed_by_tier: dict, crit: dict, main_tier: int = 144,
                 metric: str = "psnr_y") -> dict:
    """N5a: (1) rand hơn fixed ít nhất 0,1 dB ở ô chính; HOẶC (2) một mô hình rand
    không kém mô hình fixed của từng cỡ quá 0,05 dB ở mọi cỡ. ``*_by_tier``: dict cỡ -> danh sách file."""
    c = crit["N5a"]["claim_if_any"]
    r1 = at_least(rand_by_tier[main_tier], fixed_by_tier[main_tier], metric, c[0]["min_db"])
    per = {t: not_worse(rand_by_tier[t], fixed_by_tier[t], metric, c[1]["max_deficit_db"]) for t in c[1]["tiers"]
           if t in rand_by_tier and t in fixed_by_tier}
    r2 = bool(per) and len(per) == len(c[1]["tiers"]) and all(v["pass"] for v in per.values())
    return {"rand_beats_fixed": r1, "one_model_all_sizes": {"per_tier": per, "pass": r2},
            "pass": bool(r1["pass"] or r2)}


def n1_aux_head(aux_files, base_files, control_files, crit: dict) -> dict:
    """N1: sáu điều kiện của mục 1.2b. Các file cần cột lm_dev, lm_floor_noise,
    lm_floor_jpeg (evaluate.py --landmark-ckpt), ridge_f1 (--ridge), lpips, psnr_y."""
    c = crit["N1"]
    base = load(base_files)
    out = {}
    out["1_landmark_reduction"] = at_least(aux_files, base_files, "lm_dev", c["landmark_dev_rel_reduction"]["min"],
                                           relative=True, higher_is_better=False)
    out["2_same_sign_each_seed"] = same_sign_each_fold(aux_files, base_files, "lm_dev", higher_is_better=False)
    floor = float(max(base.lm_floor_noise.mean(), base.lm_floor_jpeg.mean()))
    ratio = float(base.lm_dev.mean() / max(floor, 1e-12))
    out["3_dynamic_range"] = {"baseline_dev": float(base.lm_dev.mean()), "noise_floor": floor, "ratio": ratio,
                              "pass": bool(ratio >= c["dynamic_range"]["baseline_dev_over_noise_floor_min"])}
    tol = c["no_harm"]["ridge_and_lpips_max_rel_worsening"]
    harm = {"ridge_f1": not_worse(aux_files, base_files, "ridge_f1", tol, relative=True, higher_is_better=True)}
    if "lpips" in base.columns and base.lpips.notna().any():
        harm["lpips"] = not_worse(aux_files, base_files, "lpips", tol, relative=True, higher_is_better=False)
    out["4_no_harm"] = {**harm, "pass": all(v["pass"] for v in harm.values()), "lpips_checked": "lpips" in harm}
    g5 = gain(aux_files, base_files, "psnr_y")   # "không giảm" = không có mức giảm có ý nghĩa thống kê
    out["5_psnr_not_lower"] = {**g5, "pass": bool(g5["hi"] >= 0)}
    g = gain(aux_files, control_files, "lm_dev")
    out["6_beats_unlabeled_control"] = {**g, "pass": bool(g["hi"] < 0)}
    out["pass"] = all(v["pass"] for v in out.values())
    if not out["3_dynamic_range"]["pass"]:
        out["note"] = "thước đo điểm mốc không có dải động ở cỡ ảnh này: xét N1 bằng bảng kiểm tra gờ (mục 1.9)"
    return out


def fidelity_point(model_files, base_files, crit: dict) -> dict:
    """Bản thiên trung thực: PSNR không thấp hơn thân tinh chỉnh quá 0,05 dB (trung bình)."""
    g = gain(model_files, base_files, "psnr_y")
    d = crit["model"]["fidelity_point_max_psnr_deficit_db"]
    g.update(max_deficit=d, **{"pass": bool(g["point"] >= -d)})
    return g


def n3_gate(p_files, base_files, blend_same_psnr_files, ldl_files, crit: dict, metric: str = "lpips") -> dict:
    """N3: bản thiên sắc nét (P) trong ngân sách PSNR; LPIPS thấp hơn đường trộn ở
    cùng PSNR ít nhất 3% tương đối; cùng điều kiện so với thân + LDL.
    ``blend_same_psnr_files``: ảnh trộn với α đã chọn (trên validation) để PSNR bằng bản P."""
    c = crit["N3"]
    g = gain(p_files, base_files, "psnr_y")
    out = {"psnr_budget": {**g, "pass": bool(g["point"] >= -c["psnr_budget_db"])},
           "vs_blend": at_least(p_files, blend_same_psnr_files, metric, c["lpips_vs_blend_rel"]["min"], relative=True,
                                higher_is_better=False),
           "vs_ldl": at_least(p_files, ldl_files, metric, c["lpips_vs_ldl_rel"]["min"], relative=True,
                              higher_is_better=False)}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def choose_n1_n3(n1_improvement: float | None, n3_improvement: float | None, crit: dict) -> dict:
    """Quy tắc chọn một trong N1, N3 (mục 1.2b). ``*_improvement``: phần hơn đo được
    (tỉ lệ), None nếu thành phần không đạt tiêu chí của nó."""
    t1, t3 = crit["N1"]["landmark_dev_rel_reduction"]["min"], crit["N3"]["lpips_vs_blend_rel"]["min"]
    if n1_improvement is None and n3_improvement is None:
        return {"keep": None, "why": "không thành phần nào đạt"}
    if n1_improvement is None:
        return {"keep": "N3", "why": "chỉ N3 đạt"}
    if n3_improvement is None:
        return {"keep": "N1", "why": "chỉ N1 đạt"}
    if n1_improvement < 1.5 * t1 and n3_improvement >= 2.0 * t3:
        return {"keep": "N3", "why": "ngoại lệ ghi trước: N1 dưới 1,5 lần ngưỡng, N3 từ 2 lần ngưỡng"}
    return {"keep": "N1", "why": "mặc định ưu tiên N1"}


def realtime_class(median_ms: float, p95_ms: float, crit: dict) -> dict:
    """Xếp nhóm theo trung vị; phân vị 95 vượt ngưỡng của nhóm thì gắn 'không ổn định'."""
    t = crit["realtime"]["thresholds_ms"]
    if median_ms <= t["realtime"]:
        cls, lim = "realtime", t["realtime"]
    elif median_ms <= t["near_realtime"]:
        cls, lim = "near_realtime", t["near_realtime"]
    else:
        return {"class": "reference_only", "stable": None}
    return {"class": cls, "stable": bool(p95_ms <= lim)}


def matched_latency(ms_a: float, ms_b: float, crit: dict) -> dict:
    tol = crit["realtime"]["matched_latency_tolerance"]
    rel = abs(ms_a - ms_b) / min(ms_a, ms_b)
    return {"rel_diff": rel, "tolerance": tol, "pass": bool(rel <= tol)}


def pretrain_rank_stable(psnr_at_50: dict, psnr_at_100: dict) -> dict:
    """Thứ hạng các biến thể giống nhau ở 50% và 100% ngân sách (dict variant -> PSNR validation)."""
    if set(psnr_at_50) != set(psnr_at_100):
        raise ValueError("hai mốc phải có cùng tập biến thể")
    r50 = sorted(psnr_at_50, key=psnr_at_50.get, reverse=True)
    r100 = sorted(psnr_at_100, key=psnr_at_100.get, reverse=True)
    return {"rank_50": r50, "rank_100": r100, "pass": r50 == r100}
