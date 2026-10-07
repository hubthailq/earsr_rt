#!/usr/bin/env python3
"""Sinh mọi con số, bảng và hình của bản thảo từ results/ (không gõ tay số nào vào bài).

  python scripts/make_paper.py            # ghi paper/generated/*.tex và paper/figures/*.pdf
  cd paper && latexmk -pdf main.tex

Mỗi con số trong bài là một macro LaTeX ở paper/generated/numbers.tex. Số nào chưa có kết quả thì macro in ra một ô
đỏ [TBD: ...], và script liệt kê các ô đó ở cuối. Có thêm kết quả (khối n2c, nhận dạng, fold 1 và 5) thì chạy lại
script: bảng và số tự cập nhật, không phải sửa bài.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earsr.report.arms import arm_of  # noqa: E402
from earsr.stats.bootstrap import paired_diff  # noqa: E402

# 16 mô hình tối ưu PSNR, khác nhau về trọng số (span26 trùng span_ch28; span_ch48_t44 là bản phụ của SPAN 48 kênh)
PSNR16 = ["span_ch48", "span_ch28", "span_ch26", "rlfn", "efdn", "safmnpp", "smfan", "msrresnet", "edsr_baseline",
          "swinir_light", "rrdb_psnr", "pds26", "pkdsr26", "dscf26", "disp26", "errn26"]
GAN = ["esrgan", "bsrgan", "realesrgan", "realesr_compact"]
DISPLAY = {"bicubic": "Bicubic", "span_ch48": "SPAN-S (48 ch.)", "span_ch28": "SPAN (28 ch.)", "span_ch26": "SPAN (26 ch.)",
           "rlfn": "RLFN", "efdn": "EFDN", "safmnpp": "SAFMN++", "smfan": "SMFANet", "msrresnet": "MSRResNet",
           "edsr_baseline": "EDSR-baseline", "swinir_light": "SwinIR-light", "rrdb_psnr": "RRDB (PSNR)",
           "pds26": "PDS", "pkdsr26": "PKDSR", "dscf26": "DSCF", "disp26": "DISP", "errn26": "ERRN",
           "esrgan": "ESRGAN", "bsrgan": "BSRGAN", "realesrgan": "Real-ESRGAN", "realesr_compact": "Real-ESRGAN (compact)"}
YEAR = {"pds26": "NTIRE'26", "pkdsr26": "NTIRE'26", "dscf26": "NTIRE'26", "disp26": "NTIRE'26", "errn26": "NTIRE'26",
        "span_ch28": "NTIRE'24", "span_ch26": "NTIRE'24", "safmnpp": "NTIRE'24", "smfan": "NTIRE'24", "efdn": "NTIRE'23"}
TRAIN_LABEL = {"bic": "Bicubic", "generic": "Generic real-world", "bicjpeg75": "Bicubic + JPEG 75",
               "jpegu": "Bicubic + JPEG U(60, 95)", "jpegmix": "Bicubic + measured JPEG levels",
               "est": "Measured (ours)"}
KIND_LABEL = {"bic": "Clean", "bicjpeg93": "JPEG 93", "bicjpeg85": "JPEG 85", "bicjpeg75": "JPEG 75",
              "bicjpeg60": "JPEG 60", "est": "Measured", "generic": "Generic", "jpegmix": "JPEG mix"}
CELLS = [("ami", 144, "AMI, 36 px"), ("earvn", 96, "EarVN1.0, 24 px"), ("awex", 96, "AWEx, 24 px"),
         ("awex", 144, "AWEx, 36 px")]
PUB = "span_ch48 (published)"
M: dict[str, str] = {}
TODO: list[str] = []


# ------------------------------------------------------------------ định dạng
def num(x, nd: int = 2, sign: bool = False) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return ("$+$" if sign else "") + s


def put(name: str, value, why: str = "") -> None:
    """Đăng ký một macro. ``value`` là None thì macro in ô [TBD]."""
    assert name.isalpha(), name
    if value is None:
        M[name] = "\\todo{" + (why or name).replace("_", "\\_") + "}"
        TODO.append(f"{name}: {why}")
    else:
        M[name] = str(value)


def est_cell(e, nd: int = 2) -> str:
    """Chênh lệch ghép cặp kèm khoảng tin cậy, cho ô bảng."""
    if e is None:
        return "\\todo{--}"
    return f"{num(e.point, nd, True)} {{\\scriptsize[{num(e.lo, nd)}, {num(e.hi, nd)}]}}"


# ------------------------------------------------------------------ nạp dữ liệu
def load_dir(folder: Path) -> pd.DataFrame | None:
    fs = sorted(folder.glob("*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, dtype={"subject": str, "view": str}) for f in fs], ignore_index=True)
    a = d.model.map(arm_of)
    d["arm"], d["runfold"] = [x[0] for x in a], [x[1] for x in a]
    return d


METRICS = ["psnr_y", "ssim_y", "lpips", "dists"]


def pool(d: pd.DataFrame, folds_path: Path | None) -> tuple[pd.DataFrame, list[int]]:
    """Gộp các fold. AMI (có ``folds_path``): mô hình tự huấn luyện chỉ có dòng của người test fold nó, nên các mốc được
    giới hạn về người test của các fold đã có. Bộ ngoài thực tế: trung bình các mô hình (các fold) theo từng ảnh."""
    folds = sorted(int(f) for f in d.loc[d.arm == "span/est", "runfold"].unique())
    if folds_path is not None:
        fj = json.loads(folds_path.read_text())["folds"]
        keep = {s for f in folds for s in fj[str(f)]["test"]}
        d = d[d.subject.isin(keep)]
    cols = [c for c in METRICS if c in d.columns]
    return d.groupby(["arm", "tier", "degrade", "key", "subject"], as_index=False)[cols].mean(), folds


def paired(p, a: str, b: str, tier: int, kind: str, metric: str = "psnr_y"):
    if p is None:
        return None
    A = p[(p.arm == a) & (p.tier == tier) & (p.degrade == kind)].set_index("key")
    B = p[(p.arm == b) & (p.tier == tier) & (p.degrade == kind)].set_index("key")
    if A.empty or B.empty or len(A) != len(B) or not A.index.sort_values().equals(B.index.sort_values()):
        return None
    B = B.loc[A.index]
    return paired_diff(A[metric].to_numpy(), B[metric].to_numpy(), A["subject"].to_numpy(), n_boot=2000)


def mean_of(p, arm: str, tier: int, kind: str, metric: str = "psnr_y"):
    if p is None:
        return None
    x = p[(p.arm == arm) & (p.tier == tier) & (p.degrade == kind)][metric]
    return float(x.mean()) if len(x) else None


def rng_put(prefix: str, ests: list, nd: int = 2, magnitude: bool = False, why: str = "", sign: bool = False) -> None:
    """Hai macro <prefix>Min và <prefix>Max từ một danh sách chênh lệch (None ở đâu thì cả hai thành TBD)."""
    if not ests or any(e is None for e in ests):
        put(prefix + "Min", None, why)
        put(prefix + "Max", None, why)
        return
    v = [abs(e.point) if magnitude else e.point for e in ests]
    put(prefix + "Min", num(min(v), nd, sign))
    put(prefix + "Max", num(max(v), nd, sign))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ phần 1: mô hình có trọng số công bố
def published(res: Path, out: Path) -> dict:
    q = pd.read_csv(res / "t2_summary" / "quality.csv")
    q = q[q.scale == 4]
    wild = {"earvn": pd.read_csv(res / "t2_earvn_summary" / "quality.csv"),
            "awex": pd.read_csv(res / "t2_awex_summary" / "quality.csv")}
    lat = pd.read_csv(res / "latency_local.csv")
    lat = lat[(lat.input_h == 68) & (lat.input_w == 48) & (lat.device == "cuda")].set_index("name")

    def cell(df, tier, kind, models=PSNR16):
        return df[(df.tier == tier) & (df.degrade == kind) & df.model.isin(models)].set_index("model")

    main = {k: cell(q, 144, k) for k in ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "bicjpeg60")}
    assert all(len(v) == 16 for v in main.values()), {k: len(v) for k, v in main.items()}
    put("NPub", 16)
    put("NAmiAllImg", int(main["bic"].n.iloc[0]))
    put("AmiCleanGainMin", num(main["bic"].gain.min()))
    put("AmiCleanGainMax", num(main["bic"].gain.max()))
    put("AmiQLowGapMin", num((-main["bicjpeg75"].gain).min()))
    put("AmiQLowGapMax", num((-main["bicjpeg75"].gain).max()))
    put("AmiQMidGapMin", num((-main["bicjpeg85"].gain).min()))
    put("AmiQMidGapMax", num((-main["bicjpeg85"].gain).max()))
    put("AmiQHighGainMin", num(main["bicjpeg93"].gain.min()))
    put("AmiQHighGainMax", num(main["bicjpeg93"].gain.max()))
    put("SpreadClean", num(main["bic"].psnr_y.max() - main["bic"].psnr_y.min()))
    put("SpreadQLow", num(main["bicjpeg75"].psnr_y.max() - main["bicjpeg75"].psnr_y.min()))
    for tier, tag in ((96, "S"), (144, "M"), (192, "L")):
        c, j = cell(q, tier, "bic"), cell(q, tier, "bicjpeg75")
        put("RrdbVsSpanClean" + tag, num(c.psnr_y["rrdb_psnr"] - c.psnr_y["span_ch48"], 2, True))
        put("RrdbVsSpanQLow" + tag, num(j.psnr_y["rrdb_psnr"] - j.psnr_y["span_ch48"], 2, True))
        put("SpanCleanGain" + tag, num(c.gain["span_ch48"]))
        put("SpanQLowGap" + tag, num(-j.gain["span_ch48"]))
    from scipy.stats import kendalltau

    put("TauRank", num(kendalltau(main["bic"].psnr_y[PSNR16], main["bicjpeg75"].psnr_y[PSNR16]).statistic))
    put("SpanCleanPsnr", num(main["bic"].psnr_y["span_ch48"]))
    put("SpanQLowPsnr", num(main["bicjpeg75"].psnr_y["span_ch48"]))
    bic = {k: q[(q.tier == 144) & (q.degrade == k) & (q.model == "bicubic")].iloc[0] for k in main}
    put("BicCleanPsnr", num(bic["bic"].psnr_y))
    put("BicQLowPsnr", num(bic["bicjpeg75"].psnr_y))
    put("BicQLowLpips", num(bic["bicjpeg75"].lpips, 3))
    put("SpanQLowLpips", num(main["bicjpeg75"].lpips["span_ch48"], 3))
    put("LpipsQLowBetter", int((main["bicjpeg75"].lpips < bic["bicjpeg75"].lpips).sum()))
    g = cell(q, 144, "bicjpeg75", GAN)
    put("BsrganQLowGap", num(-g.gain["bsrgan"]))
    put("BsrganQLowLpips", num(g.lpips["bsrgan"], 3))
    put("ParamsSpan", num(lat.params["span_ch48"] / 1e6, 2))
    put("ParamsRrdb", num(lat.params["rrdb_psnr"] / 1e6, 1))
    put("ParamsRatio", int(round(lat.params["rrdb_psnr"] / lat.params["span_ch48"])))
    put("LatSpan", num(lat.median_ms["span_ch48"], 2))
    put("LatDisp", num(lat.median_ms["disp26"], 2))
    put("LatBsrgan", num(lat.median_ms["bsrgan"], 1))
    put("LatRatio", int(round(lat.median_ms["bsrgan"] / lat.median_ms["span_ch48"])))

    # bảng: từng mô hình ở ô chính của AMI
    rows = []
    order = ["bicubic"] + sorted(PSNR16, key=lambda m: lat.params.get(m, 0)) + GAN
    full = q[(q.tier == 144)].set_index(["model", "degrade"])
    for m in order:
        if m == GAN[0]:
            rows.append("\\midrule")
        r = {k: full.loc[(m, k)] for k in ("bic", "bicjpeg93", "bicjpeg75")}

        def ps(k):
            v = num(r[k].psnr_y)
            if m == "bicubic":
                return v
            return ("\\textbf{" + v + "}") if r[k].gain > 0 else v

        par = "--" if m == "bicubic" else num(lat.params[m] / 1e3, 0)
        ms = "--" if m == "bicubic" else num(lat.median_ms[m], 2)
        name = DISPLAY[m] + (f" {{\\scriptsize({YEAR[m]})}}" if m in YEAR else "")
        rows.append(f"{name} & {par} & {ms} & {ps('bic')} & {ps('bicjpeg93')} & {ps('bicjpeg75')} & "
                    f"{num(r['bic'].lpips, 3)} & {num(r['bicjpeg75'].lpips, 3)} \\\\")
        if m == "bicubic":
            rows.append("\\midrule")
    write(out / "tab_published.tex", "\n".join(rows) + "\n")

    # bảng: độ tổng quát qua các bộ ảnh và cỡ ảnh vào
    gen_rows, below = [], {}
    spec = [("AMI", q, 96, "24"), ("AMI", q, 144, "36"), ("AMI", q, 192, "48"), ("EarVN1.0", wild["earvn"], 96, "24"),
            ("AWEx", wild["awex"], 96, "24"), ("AWEx", wild["awex"], 144, "36")]
    for ds, df, tier, px in spec:
        cells = []
        for k in ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "est"):
            c = cell(df, tier, k)
            if len(c) != 16:
                cells.append("--")
                continue
            nb, ns = int((c.gain < 0).sum()), int((c.gain_hi < 0).sum())
            below[(ds, tier, k)] = (nb, ns, float(c.gain.min()), float(c.gain.max()))
            cells.append(f"{num(c.gain.median(), 2, True)} ({nb}/{ns})")
        n = int(cell(df, tier, "bic").n.iloc[0])
        gen_rows.append(f"{ds} & {px} & {n} & " + " & ".join(cells) + " \\\\")
    write(out / "tab_generality.tex", "\n".join(gen_rows) + "\n")
    put("EarvnQLowBelow", below[("EarVN1.0", 96, "bicjpeg75")][0])
    put("EarvnQLowSig", below[("EarVN1.0", 96, "bicjpeg75")][1])
    put("AwexSQLowSig", below[("AWEx", 96, "bicjpeg75")][1])
    put("AwexMQLowSig", below[("AWEx", 144, "bicjpeg75")][1])
    put("AmiQLowSig", min(below[("AMI", t, "bicjpeg75")][1] for t in (96, 144, 192)))
    put("AllQHighBelow", sum(v[0] for k, v in below.items() if k[2] == "bicjpeg93"))
    put("WildEstGainMin", num(min(v[2] for k, v in below.items() if k[2] == "est"), 2, True))
    put("WildEstGainMax", num(max(v[3] for k, v in below.items() if k[2] == "est"), 2, True))
    put("WildCleanGainMin", num(min(v[2] for k, v in below.items() if k[2] == "bic" and k[0] != "AMI")))
    put("WildCleanGainMax", num(max(v[3] for k, v in below.items() if k[2] == "bic" and k[0] != "AMI")))
    for ds, tag in (("EarVN1.0", "Earvn"), ("AWEx", "AwexS")):
        put(f"N{tag}Img", int(cell(wild["earvn" if tag == "Earvn" else "awex"], 96, "bic").n.iloc[0]))
    put("NAwexMImg", int(cell(wild["awex"], 144, "bic").n.iloc[0]))
    return {"q": q, "main": main, "lat": lat}


# ------------------------------------------------------------------ phần 2: các nhánh tự huấn luyện
def trained(res: Path, out: Path, folds_path: Path) -> dict:
    raw = {"ami": load_dir(res / "n2"), "earvn": load_dir(res / "n2_earvn"), "awex": load_dir(res / "n2_awex")}
    P, folds = {}, []
    for k, d in raw.items():
        P[k], f = pool(d, folds_path if k == "ami" else None)
        folds = f if k == "ami" else folds
    put("NFolds", len(folds))
    a = P["ami"][(P["ami"].arm == "bicubic") & (P["ami"].tier == 144) & (P["ami"].degrade == "bic")]
    put("NAmiImg", len(a))
    put("NAmiSubj", a.subject.nunique())
    for k, t, tag in (("earvn", 96, "Earvn"), ("awex", 96, "AwexS"), ("awex", 144, "AwexM")):
        x = P[k][(P[k].arm == "bicubic") & (P[k].tier == t) & (P[k].degrade == "bic")]
        put(f"N{tag}Subj", x.subject.nunique())
    runs = pd.read_csv(res / "runs.csv")
    done = runs[(runs.status == "done") & runs.run_id.str.contains(r"\+", regex=True)]
    put("NRuns", len(done))
    put("NRunsStable", int((done.stress_stable.astype(str) == "True").sum()))
    put("StressMaxDrop", num(done.stress_gap_drop.max()))

    def across(a_, b_, kind, metric="psnr_y"):
        return [paired(P[k], a_, b_, t, kind, metric) for k, t, _ in CELLS]

    rng_put("EstVsBic", across("span/est", "bicubic", "est"))
    rng_put("EstVsPub", across("span/est", PUB, "est"))
    rng_put("EstVsGen", across("span/est", "span/generic", "est"))
    rng_put("EstVsJpg", across("span/est", "span/bicjpeg75", "est"))
    rng_put("EstVsJpgHigh", across("span/est", "span/bicjpeg75", "bicjpeg93"))
    rng_put("EstVsJpgLow", across("span/est", "span/bicjpeg75", "bicjpeg75"), magnitude=True)
    rng_put("EstVsPubClean", across("span/est", PUB, "bic"), magnitude=True)
    rng_put("DegEff", across("span/est", "span/bic", "bicjpeg75"))
    rng_put("EstVsBicQLow", across("span/est", "bicubic", "bicjpeg75"))
    rng_put("EstVsBicQHigh", across("span/est", "bicubic", "bicjpeg93"))
    rng_put("EstLpipsVsBic", across("span/est", "bicubic", "est", "lpips"), 3, True)
    rng_put("EstLpipsVsPub", across("span/est", PUB, "est", "lpips"), 3, True)
    rng_put("EstLpipsVsGen", across("span/est", "span/generic", "est", "lpips"), 3, True)
    rng_put("DispEstVsBic", across("disp26/est", "bicubic", "est"))
    rng_put("DispEstVsGen", across("disp26/est", "disp26/generic", "est"))
    rng_put("GenVsBicClean", across("span/generic", "bicubic", "bic"), magnitude=True)
    rng_put("PubVsBicEst", across(PUB, "bicubic", "est"))
    rng_put("GenVsBicEst", across("span/generic", "bicubic", "est"), sign=True)
    rng_put("JpgEff", across("span/bicjpeg75", "span/bic", "bicjpeg75"))
    for t, tag in ((96, "S"), (144, "M"), (192, "L")):
        e = paired(P["ami"], "span/est", "bicubic", t, "est")
        put("EstVsBicAmi" + tag, num(e.point) if e else None)
    ex = across("span/bic", "span/bic [AMI only]", "bic")
    for e, tag in zip(ex, ("Ami", "Earvn", "AwexS", "AwexM")):
        put("Extra" + tag, num(e.point, 2, True) if e else None)
    ao = across("span/bic [AMI only]", PUB, "bic")
    put("AmiOnlyVsPubEarvn", num(ao[1].point, 2, True) if ao[1] else None)
    # theo từng fold trên AMI: nhánh ước lượng trừ nhánh tổng quát
    d = raw["ami"]
    fj = json.loads(folds_path.read_text())["folds"]
    per = []
    for f in folds:
        x = d[(d.tier == 144) & (d.degrade == "est") & d.subject.isin(fj[str(f)]["test"])]
        per.append(float(x[x.arm == "span/est"].psnr_y.mean() - x[x.arm == "span/generic"].psnr_y.mean()))
    put("EstVsGenFoldMin", num(min(per)))
    put("EstVsGenFoldMax", num(max(per)))

    # bảng: ma trận suy giảm huấn luyện × ảnh vào lúc chấm (SPAN, AMI, ô chính)
    kinds = ["bic", "bicjpeg93", "bicjpeg75", "est", "generic"]
    arms = [("bicubic", "Bicubic interpolation"), (PUB, "Published weights"), None] + \
           [(f"span/{k}", TRAIN_LABEL[k]) for k in ("bic", "generic", "bicjpeg75", "jpegu", "jpegmix", "est")]
    best = {k: max((mean_of(P["ami"], a_[0], 144, k) or -1) for a_ in arms if a_) for k in kinds}
    rows = []
    for a_ in arms:
        if a_ is None:
            rows.append("\\midrule")
            continue
        cells = []
        for k in kinds:
            v = mean_of(P["ami"], a_[0], 144, k)
            s = num(v) if v is not None else "\\todo{--}"
            cells.append("\\textbf{" + s + "}" if v is not None and abs(v - best[k]) < 5e-3 else s)
        rows.append(f"{a_[1]} & " + " & ".join(cells) + " \\\\")
    write(out / "tab_matrix.tex", "\n".join(rows) + "\n")

    # bảng: kết quả chính, chênh lệch ghép cặp trên bốn ô
    rows = []
    spec = [("\\multicolumn{5}{l}{\\emph{PSNR-Y (dB), higher is better}}", None),
            ("Bicubic interpolation", ("bicubic", "psnr_y", 2)), ("Published weights", (PUB, "psnr_y", 2)),
            ("Fine-tuned, bicubic", ("span/bic", "psnr_y", 2)), ("Fine-tuned, generic", ("span/generic", "psnr_y", 2)),
            ("Fine-tuned, JPEG 75", ("span/bicjpeg75", "psnr_y", 2)),
            ("\\multicolumn{5}{l}{\\emph{LPIPS, lower is better}}", None),
            ("Bicubic interpolation", ("bicubic", "lpips", 3)), ("Published weights", (PUB, "lpips", 3)),
            ("Fine-tuned, generic", ("span/generic", "lpips", 3)), ("Fine-tuned, JPEG 75", ("span/bicjpeg75", "lpips", 3))]
    for label, s in spec:
        if s is None:
            rows.append(("\\midrule\n" if rows else "") + label + " \\\\")
            continue
        rows.append(f"\\quad {label} & " + " & ".join(est_cell(e, s[2]) for e in across("span/est", s[0], "est", s[1]))
                    + " \\\\")
    write(out / "tab_main.tex", "\n".join(rows) + "\n")

    # bảng: bền theo mức nén (ước lượng trừ JPEG 75 cố định, theo ảnh vào lúc chấm), hai thân
    rows = []
    for bb, name in (("span", "SPAN-S"), ("disp26", "DISP")):
        for kind in ("bicjpeg93", "est", "bicjpeg75", "bic"):
            rows.append(f"{name} & {KIND_LABEL[kind]} & " +
                        " & ".join(est_cell(e) for e in across(f"{bb}/est", f"{bb}/bicjpeg75", kind)) + " \\\\")
        if bb == "span":
            rows.append("\\midrule")
    write(out / "tab_levels.tex", "\n".join(rows) + "\n")

    # bảng: tách "có nén" khỏi "đo từ dữ liệu" (khối n2c; chưa có thì toàn ô TBD)
    rows = []
    for label, b_ in (("measured JPEG levels only", "span/jpegmix"), ("uniform JPEG levels", "span/jpegu")):
        for kind in ("est", "jpegmix", "bicjpeg75", "bicjpeg93", "bic"):
            rows.append(f"Ours $-$ {label} & {KIND_LABEL[kind]} & " +
                        " & ".join(est_cell(e) for e in across("span/est", b_, kind)) + " \\\\")
        rows.append("\\midrule")
    write(out / "tab_ablation.tex", "\n".join(rows[:-1]) + "\n")
    why = "n2c"   # khối n2c (scripts/make_n2c_jobs.sh) chưa có kết quả
    rng_put("EstVsMix", across("span/est", "span/jpegmix", "est"), why=why)
    rng_put("EstVsUni", across("span/est", "span/jpegu", "est"), why=why)
    rng_put("MixVsUni", across("span/jpegmix", "span/jpegu", "est"), why=why, sign=True)
    rng_put("MixVsEstClean", across("span/jpegmix", "span/est", "bic"), why=why)
    rng_put("MixVsEstHigh", across("span/jpegmix", "span/est", "bicjpeg93"), why=why)
    rng_put("UniVsEstClean", across("span/jpegu", "span/est", "bic"), why=why)
    rng_put("MixVsEstOnMix", across("span/jpegmix", "span/est", "jpegmix"), why=why)

    # bảng: công thức tinh chỉnh ổn định (ảnh vào sạch; các nhánh học với bicubic)
    amo = res / "n2_amionly"
    col = {}
    for name, f in (("collapse", "N2_span-zero_pub_rand_x4_hrall_bic_f2__hr96_x4_bic.csv"),
                    ("pub", "span_ch48__hr96_x4_bic.csv"), ("bic", "bicubic__hr96_x4_bic.csv")):
        x = pd.read_csv(amo / f)
        col[name] = (float(x.psnr_y.mean()), int((x.psnr_y < 20).sum()), len(x))
    put("CollapsePsnr", num(col["collapse"][0]))
    put("CollapseBroken", col["collapse"][1])
    put("CollapsePub", num(col["pub"][0]))
    put("CollapseBic", num(col["bic"][0]))
    rows = [f"Published weights & {num(mean_of(P['ami'], PUB, 144, 'bic'))} & {num(mean_of(P['earvn'], PUB, 96, 'bic'))} & "
            f"{num(mean_of(P['awex'], PUB, 96, 'bic'))} & {num(mean_of(P['awex'], PUB, 144, 'bic'))} \\\\",
            f"Fine-tuned on AMI only$^\\dagger$ & -- & {num(col['collapse'][0])} & -- & -- \\\\"]
    for arm_, label in (("span/bic [AMI only]", "\\quad + photometric augmentation"),
                        ("span/bic", "\\quad + in-the-wild ear images")):
        rows.append(f"{label} & {num(mean_of(P['ami'], arm_, 144, 'bic'))} & {num(mean_of(P['earvn'], arm_, 96, 'bic'))} & "
                    f"{num(mean_of(P['awex'], arm_, 96, 'bic'))} & {num(mean_of(P['awex'], arm_, 144, 'bic'))} \\\\")
    write(out / "tab_stability.tex", "\n".join(rows) + "\n")
    return {"P": P, "folds": folds}


# ------------------------------------------------------------------ phần 3: bộ phân loại, tham số suy giảm, nhận dạng
def realism_and_params(res: Path, out: Path, params_path: Path) -> None:
    r = json.loads((res / "n2_realism.json").read_text())
    rows = []
    for k, label, tag in (("bic", "Bicubic", "Bic"), ("generic", "Generic real-world pipeline", "Gen"),
                          ("bicjpeg75", "Bicubic + JPEG 75", "Jpg"), ("est", "Measured (ours)", "Est")):
        v = r["kinds"][k]
        put("Real" + tag, num(100 * v["balanced_acc"], 1))
        rows.append(f"{label} & {num(100 * v['balanced_acc'], 1)} & [{num(100 * v['lo'], 1)}, {num(100 * v['hi'], 1)}] \\\\")
    write(out / "tab_realism.tex", "\n".join(rows) + "\n")
    put("RealTestSubj", r["n_test_subjects"])
    put("RealTrainSubj", r["n_train_subjects"])
    p = json.loads(params_path.read_text())
    q = dict((int(a), float(b)) for a, b in p["jpeg_q"])
    put("QLowShare", num(100 * q.get(75, 0), 1))
    put("QHighShare", num(100 * q.get(93, 0), 1))
    put("NoiseLo", num(p["noise_sigma"][0], 2))
    put("NoiseHi", num(p["noise_sigma"][1], 2))
    put("BlurLo", num(p["blur_sigma"][0], 1))
    put("BlurHi", num(p["blur_sigma"][1], 1))
    roles = json.loads(Path("splits/earvn_roles.json").read_text())
    for role, tag in (("train", "Train"), ("test", "Test"), ("viewer", "Viewer"), ("fit", "Fit"), ("clf", "Clf")):
        put("Role" + tag, sum(v == role for v in roles.values()))
    put("RoleAll", len(roles))
    import re

    m = re.search(r"(\d+) ảnh nhỏ thật, (\d+) ảnh lớn", p.get("source", ""))
    put("FitSmall", m.group(1) if m else None, "number of real small images used for fitting")
    put("FitLarge", m.group(2) if m else None, "number of large images used for fitting")


def recognition(res: Path, out: Path) -> None:
    why = "recog"   # kết quả nhận dạng (scripts/run_recog.sh) chưa có
    order = [("ref_large", "Large probes (reference)"), ("direct", "No upscaling"), ("bicubic", "Bicubic $\\times$4"),
             (PUB, "SPAN-S, published"), ("rrdb_psnr (published)", "RRDB (PSNR), published"),
             ("bsrgan (published)", "BSRGAN"), ("realesrgan (published)", "Real-ESRGAN"),
             ("span/bic", "SPAN-S, fine-tuned bicubic"), ("span/generic", "SPAN-S, generic"),
             ("span/bicjpeg75", "SPAN-S, JPEG 75"), ("span/jpegu", "SPAN-S, uniform JPEG"),
             ("span/jpegmix", "SPAN-S, measured JPEG levels"), ("span/est", "SPAN-S, measured (ours)"),
             ("disp26/est", "DISP, measured (ours)")]
    S, Pd, meta = {}, {}, None
    for tag in ("resnet18", "imagenet50"):
        f = res / "recog" / f"{tag}_summary" / "summary.csv"
        if f.exists():
            S[tag] = pd.read_csv(f).set_index("arm")
            Pd[tag] = pd.read_csv(res / "recog" / f"{tag}_summary" / "paired.csv").set_index(["arm", "minus"])
            meta = json.loads((res / "recog" / tag / "meta.json").read_text())
    rows = []
    for arm_, label in order:
        cells = []
        for tag in ("resnet18", "imagenet50"):
            if tag in S and arm_ in S[tag].index:
                r = S[tag].loc[arm_]
                cells += [f"{num(r['rank1'], 1)} {{\\scriptsize[{num(r['rank1_lo'], 1)}, {num(r['rank1_hi'], 1)}]}}",
                          num(r["eer"], 1)]
            else:
                cells += ["\\todo{--}", "\\todo{--}"]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if arm_ in ("ref_large", "realesrgan (published)"):
            rows.append("\\midrule")
    write(out / "tab_recog.tex", "\n".join(rows) + "\n")
    put("RecSubj", meta["n_probe_subjects"] if meta else None, why)
    put("RecEnrolled", len(meta["subjects"]) if meta else None, why)
    put("RecProbes", meta["n_probe"] if meta else None, why)
    put("RecGallery", meta["n_gallery"] if meta else None, why)
    def r1(tag_, arm_, col="rank1"):
        t = S.get(tag_)
        return num(t.loc[arm_, col], 1) if t is not None and arm_ in t.index else None

    def gap(tag_, arm_, base, col="d_rank1"):
        t = Pd.get(tag_)
        if t is None or (arm_, base) not in t.index:
            return None
        r = t.loc[(arm_, base)]
        return f"{num(r[col], 1, True)} [{num(r[col + '_lo'], 1)}, {num(r[col + '_hi'], 1)}]"

    names = (("ref_large", "Ref"), ("direct", "Direct"), ("bicubic", "Bic"), (PUB, "Pub"), ("span/bic", "Bict"),
             ("span/est", "Est"), ("span/generic", "Gen"), ("span/bicjpeg75", "Jpg"), ("span/jpegmix", "Mix"),
             ("span/jpegu", "Uni"), ("bsrgan (published)", "Bsrgan"), ("realesrgan (published)", "Realesrgan"),
             ("disp26/est", "DispEst"), ("disp26 (published)", "DispPub"))
    for arm_, tag in names:
        put("RecRank" + tag, r1("resnet18", arm_), why)
    for arm_, tag in (("ref_large", "Ref"), ("bicubic", "Bic"), (PUB, "Pub"), ("span/est", "Est"),
                      ("bsrgan (published)", "Bsrgan")):
        put("RecCtlRank" + tag, r1("imagenet50", arm_), why)
    put("RecEerBic", r1("resnet18", "bicubic", "eer"), why)
    put("RecEerEst", r1("resnet18", "span/est", "eer"), why)
    put("RecEerRef", r1("resnet18", "ref_large", "eer"), why)
    for arm_, base, tag in (("span/est", "bicubic", "EstVsBic"), ("span/est", PUB, "EstVsPub"), (PUB, "bicubic", "PubVsBic"),
                            ("span/bic", "bicubic", "BictVsBic"), ("span/generic", "span/est", "GenVsEst"),
                            ("span/bicjpeg75", "span/est", "JpgVsEst"), ("span/jpegmix", "span/est", "MixVsEst"),
                            ("span/jpegu", "span/est", "UniVsEst"), ("bsrgan (published)", "span/est", "BsrganVsEst"),
                            ("disp26/est", "bicubic", "DispEstVsBic")):
        put("Rec" + tag, gap("resnet18", arm_, base), why)
    put("RecEerEstVsBic", gap("resnet18", "span/est", "bicubic", "d_eer"), why)
    put("RecCtlEstVsBic", gap("imagenet50", "span/est", "bicubic"), why)
    put("RecCtlPubVsBic", gap("imagenet50", PUB, "bicubic"), why)
    put("RecCtlGenVsEst", gap("imagenet50", "span/generic", "span/est"), why)
    f = res / "recog" / "recognizer_train_log.json"
    info = json.loads(f.read_text())["info"] if f.exists() else None
    put("RecTrainSubj", len(info["subjects"]) if info else None, why)
    put("RecTrainImg", info["n_train"] if info else None, why)
    put("RecValAcc", num(100 * info["val_acc"], 1) if info else None, why)


def recognition_extra(res: Path, out: Path) -> None:
    """Ba phần đào sâu của phép đo nhận dạng: tách theo mức nén của file ảnh dò; bộ ảnh thật thứ hai (AWEx);
    mạng nhận dạng thứ hai (ResNet-50). Phần nào chưa có kết quả thì ra ô TBD."""
    why = "recog2"   # chạy lại scripts/run_recog.sh bằng mã mới (07/10/2026)

    def load(tag):
        d = res / "recog" / f"{tag}_summary"
        if not (d / "summary.csv").exists():
            return None
        return {"s": pd.read_csv(d / "summary.csv").set_index("arm"),
                "p": pd.read_csv(d / "paired.csv").set_index(["arm", "minus"]),
                "q": pd.read_csv(d / "by_quality.csv").set_index(["arm", "group"]) if (d / "by_quality.csv").exists() else None,
                "meta": json.loads((res / "recog" / tag / "meta.json").read_text())}

    def ci(r, col, nd=1):
        return f"{num(r[col], nd, True)} {{\\scriptsize[{num(r[col + '_lo'], nd)}, {num(r[col + '_hi'], nd)}]}}"

    def txt(r, col, nd=1):
        return f"{num(r[col], nd, True)} [{num(r[col + '_lo'], nd)}, {num(r[col + '_hi'], nd)}]"

    rows_spec = [(PUB, "SPAN-S, published"), ("span/bic", "SPAN-S, fine-tuned bicubic"), ("span/generic", "SPAN-S, generic"),
                 ("span/bicjpeg75", "SPAN-S, JPEG 75"), ("span/est", "SPAN-S, measured (ours)"),
                 ("bsrgan (published)", "BSRGAN")]
    # --- theo mức nén của file ảnh dò (EarVN1.0, mạng ResNet-18)
    R = load("resnet18")
    q = R["q"] if R else None
    rows = []
    if q is not None and ("bicubic", "low") in q.index and ("bicubic", "high") in q.index:
        rows.append(f"Bicubic $\\times$4 & {num(q.loc[('bicubic', 'low'), 'rank1'], 1)} & -- & "
                    f"{num(q.loc[('bicubic', 'high'), 'rank1'], 1)} & -- \\\\\n\\midrule")
        for arm_, label in rows_spec:
            lo, hi = q.loc[(arm_, "low")], q.loc[(arm_, "high")]
            rows.append(f"{label} & {num(lo['rank1'], 1)} & {ci(lo, 'd_bic')} & {num(hi['rank1'], 1)} & {ci(hi, 'd_bic')} \\\\")
        for g, tag in (("low", "QLow"), ("high", "QHigh")):
            b = q.loc[("bicubic", g)]
            put(f"Rec{tag}N", int(b["n_probe"]))
            put(f"Rec{tag}Subj", int(b["n_subjects"]))
            put(f"Rec{tag}RankBic", num(b["rank1"], 1))
            for arm_, t2 in ((PUB, "Pub"), ("span/est", "Est"), ("bsrgan (published)", "Bsrgan"), ("span/bic", "Bict")):
                put(f"Rec{tag}{t2}", txt(q.loc[(arm_, g)], "d_bic"))
    else:
        rows = ["\\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} \\\\"]
        for tag in ("QLow", "QHigh"):
            for k in ("N", "Subj", "RankBic", "Pub", "Est", "Bsrgan", "Bict"):
                put(f"Rec{tag}{k}", None, why)
    write(out / "tab_recog_quality.tex", "\n".join(rows) + "\n")

    # --- bộ ảnh thứ hai và mạng nhận dạng thứ hai: chênh lệch rank-1 so với bicubic
    cols = [("resnet18", "EarvnRe"), ("resnet50", "EarvnRf"), ("awex_resnet18", "AwexRe"), ("awex_resnet50", "AwexRf")]
    L = {tag: load(tag) for tag, _ in cols}

    def cell(tag, arm_, kind):
        d = L[tag]
        if d is None:
            return "\\todo{--}"
        if kind == "rank":
            return num(d["s"].loc[arm_, "rank1"], 1) if arm_ in d["s"].index else "--"
        return ci(d["p"].loc[(arm_, "bicubic")], "d_rank1") if (arm_, "bicubic") in d["p"].index else "--"

    rows = ["Large probes (rank-1, \\%) & " + " & ".join(cell(t, "ref_large", "rank") for t, _ in cols) + " \\\\",
            "Bicubic $\\times$4 (rank-1, \\%) & " + " & ".join(cell(t, "bicubic", "rank") for t, _ in cols) + " \\\\", "\\midrule"]
    for arm_, label in rows_spec:
        rows.append(f"{label} & " + " & ".join(cell(t, arm_, "gap") for t, _ in cols) + " \\\\")
    write(out / "tab_recog_robust.tex", "\n".join(rows) + "\n")
    for tag, name in cols[1:]:
        d = L[tag]
        ok = d is not None
        put(f"Rec{name}RankRef", num(d["s"].loc["ref_large", "rank1"], 1) if ok else None, why)
        put(f"Rec{name}RankBic", num(d["s"].loc["bicubic", "rank1"], 1) if ok else None, why)
        put(f"Rec{name}RankEst", num(d["s"].loc["span/est", "rank1"], 1) if ok else None, why)
        put(f"Rec{name}EstVsBic", txt(d["p"].loc[("span/est", "bicubic")], "d_rank1") if ok else None, why)
        put(f"Rec{name}PubVsBic", txt(d["p"].loc[(PUB, "bicubic")], "d_rank1") if ok else None, why)
    a = L["awex_resnet18"]
    put("RecAwexEnrolled", len(a["meta"]["subjects"]) if a else None, why)
    put("RecAwexSubj", a["meta"]["n_probe_subjects"] if a else None, why)
    put("RecAwexProbes", a["meta"]["n_probe"] if a else None, why)
    f = res / "recog" / "recognizer_resnet50_train_log.json"
    put("RecRfValAcc", num(100 * json.loads(f.read_text())["info"]["val_acc"], 1) if f.exists() else None, why)


# ------------------------------------------------------------------ hình
def figures(pub: dict, tr: dict, fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig.mkdir(parents=True, exist_ok=True)
    q, P = pub["q"], tr["P"]["ami"]
    kinds = ["bicjpeg60", "bicjpeg75", "bicjpeg85", "bicjpeg93", "bic"]
    f, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, tier in zip(axes, (96, 144)):
        g = q[(q.tier == tier) & q.model.isin(PSNR16) & q.degrade.isin(kinds)].pivot(index="model", columns="degrade",
                                                                                      values="gain")[kinds]
        for m in g.index:
            ax.plot(range(5), g.loc[m], color="0.65", lw=0.7, zorder=1)
        ax.plot(range(5), g.median(), color="k", lw=1.6, marker="o", ms=3, label="16 published models (median)", zorder=3)
        ours = [paired(P, "span/est", "bicubic", tier, k) for k in ("bicjpeg75", "bicjpeg93", "bic")]
        if all(ours):
            ax.plot([1, 3, 4], [e.point for e in ours], color="C3", lw=1.6, marker="s", ms=3.5,
                    label="SPAN-S, measured degradation (ours)", zorder=4)
        ax.axhline(0, color="C0", lw=0.9, ls="--", zorder=2)
        ax.text(0.02, 0.06, "bicubic", color="C0", fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_xticks(range(5), ["60", "75", "85", "93", "none"])
        ax.set_xlabel("JPEG quality of the input")
        ax.set_title(f"AMI, {tier // 4} px inputs", fontsize=8)
    axes[0].set_ylabel("PSNR-Y gain over bicubic (dB)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    f.tight_layout()
    f.savefig(fig / "fig_jpeg_sweep.pdf")
    plt.close(f)

    f, ax = plt.subplots(figsize=(3.6, 3.0))
    xs = [24, 36, 48]
    for a_, kind, label, style in ((PUB, "bic", "Published, clean input", dict(color="0.4", ls="-")),
                                   (PUB, "bicjpeg75", "Published, JPEG 75", dict(color="0.4", ls=":")),
                                   ("span/est", "bicjpeg75", "Ours, JPEG 75", dict(color="C3", ls=":")),
                                   ("span/est", "est", "Ours, measured degradation", dict(color="C3", ls="-"))):
        es = [paired(P, a_, "bicubic", t, kind) for t in (96, 144, 192)]
        if all(es):
            ax.errorbar(xs, [e.point for e in es], yerr=[[e.point - e.lo for e in es], [e.hi - e.point for e in es]],
                        marker="o", ms=3, lw=1.3, capsize=2, label=label, **style)
    ax.axhline(0, color="C0", lw=0.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xlabel("Short side of the input (px)")
    ax.set_ylabel("PSNR-Y gain over bicubic (dB)")
    f.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=2)
    f.tight_layout(rect=(0, 0.12, 1, 1))
    f.savefig(fig / "fig_size.pdf")
    plt.close(f)


QUAL = [("bicubic", "Bicubic"), ("span_ch48", "SPAN-S\npublished"), ("bsrgan", "BSRGAN"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_generic_f2", "SPAN-S\ngeneric"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_bicjpeg75_f2", "SPAN-S\nJPEG 75"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_est_f2", "SPAN-S\nmeasured")]


def qualitative(res: Path, fig: Path) -> bool:
    """Ảnh nhỏ thật đã phóng ×4 bằng sáu phương pháp. Trả về False nếu chưa có ảnh (results/recog/sr)."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # sr2: 12 ảnh rải trên nhiều người (mã từ 07/10); sr: 16 ảnh đầu, đều của một người (mã cũ)
    root, picks = (res / "recog" / "sr2", (0, 3, 6, 9)) if (res / "recog" / "sr2" / "bicubic").is_dir() else (res / "recog" / "sr", (3, 4, 9, 14))
    if not all((root / m).is_dir() for m, _ in QUAL):
        return False
    names = sorted(p.name for p in (root / QUAL[0][0]).glob("*.png"))
    names = [names[i] for i in picks if i < len(names)]
    if not names:
        return False
    f, axes = plt.subplots(len(names), len(QUAL), figsize=(6.6, 2.1 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, (m, label) in enumerate(QUAL):
            img = cv2.imread(str(root / m / n))
            h, w = img.shape[:2]
            c = img[h // 2 - min(h, int(1.4 * w)) // 2: h // 2 + min(h, int(1.4 * w)) // 2]   # cắt giữa, bỏ bớt nền
            axes[i, j].imshow(c[:, :, ::-1], interpolation="nearest")
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(label, fontsize=7)
    f.tight_layout(pad=0.2)
    f.savefig(fig / "fig_qualitative.pdf", dpi=200)
    plt.close(f)
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--degrade-params", default="configs/degrade_estimated.json")
    a = ap.parse_args(argv)
    res, out, fig = Path(a.results), Path(a.paper) / "generated", Path(a.paper) / "figures"
    pub = published(res, out)
    tr = trained(res, out, Path(a.folds))
    realism_and_params(res, out, Path(a.degrade_params))
    recognition(res, out)
    recognition_extra(res, out)
    figures(pub, tr, fig)
    write(out / "fig_qualitative.tex", "\\includegraphics[width=\\linewidth]{figures/fig_qualitative.pdf}\n" if qualitative(res, fig)
          else "\\fbox{\\parbox[c][3cm][c]{0.85\\linewidth}{\\centering\\todo{qualitative figure}}}\n")
    lines = ["% Sinh bởi scripts/make_paper.py. KHÔNG sửa tay: chạy lại script khi có kết quả mới."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    write(out / "numbers.tex", "\n".join(lines) + "\n")
    print(f"đã ghi {len(M)} macro vào {out / 'numbers.tex'}; fold đã gộp trên AMI: {tr['folds']}")
    print(f"còn {len(TODO)} macro chưa có số:")
    for t in TODO:
        print("  -", t)


if __name__ == "__main__":
    main()
