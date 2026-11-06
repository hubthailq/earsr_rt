#!/usr/bin/env python3
"""Sinh mọi con số, bảng và hình của bản thảo từ results/ (không gõ tay số nào vào bài).

  python scripts/make_paper.py            # ghi paper/generated/*.tex và paper/figures/*.png (hình là ảnh PNG 600 dpi)
  cd paper && latexmk -pdf main.tex

Mỗi con số trong bài là một macro LaTeX ở paper/generated/numbers.tex. Số nào chưa có kết quả thì macro in ra một ô
đỏ [TBD: ...], và script liệt kê các ô đó ở cuối. Có thêm kết quả (khối n2c, nhận dạng, fold 1 và 5) thì chạy lại
script: bảng và số tự cập nhật, không phải sửa bài.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from earsr.report.arms import PUBLISHED, arm_of  # noqa: E402
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
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


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
    nb = int((main["bicjpeg75"].lpips < bic["bicjpeg75"].lpips).sum())
    put("LpipsQLowBetter", f"all {len(main['bicjpeg75'])}" if nb == len(main["bicjpeg75"]) else f"{nb} of the {len(main['bicjpeg75'])}")
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
        for k in ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "bicjpeg60", "est"):   # cùng cột với các hàng DIV2K
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
    put("AllQHighBelow", sum(v[0] for k, v in below.items() if k[2] == "bicjpeg93"))   # mọi bộ ảnh tai, mọi cỡ
    put("AllQHighSig", sum(v[1] for k, v in below.items() if k[2] == "bicjpeg93"))
    put("AllQHighCells", sum(16 for k in below if k[2] == "bicjpeg93"))
    ae = [v for k, v in below.items() if k[0] == "AMI" and k[2] == "est"]   # suy giảm đo được trên AMI (05/11; TODO P6)
    put("AmiEstSig", min(v[1] for v in ae) if ae else None, "ami-est")
    me = [cell(q, t, "est").gain.median() for t in (96, 144, 192) if len(cell(q, t, "est")) == 16]
    put("AmiEstMedMin", num(-max(me), 2) if me else None, "ami-est")
    put("AmiEstMedMax", num(-min(me), 2) if me else None, "ami-est")
    put("WildEstGainMin", num(min(v[2] for k, v in below.items() if k[2] == "est" and k[0] != "AMI"), 2, True))
    put("WildEstGainMax", num(max(v[3] for k, v in below.items() if k[2] == "est" and k[0] != "AMI"), 2, True))
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
    rows = [f"Bicubic interpolation & {num(mean_of(P['ami'], 'bicubic', 144, 'bic'))} & "   # mốc, để bảng tự đọc được
            f"{num(mean_of(P['earvn'], 'bicubic', 96, 'bic'))} & {num(mean_of(P['awex'], 'bicubic', 96, 'bic'))} & "
            f"{num(mean_of(P['awex'], 'bicubic', 144, 'bic'))} \\\\\n\\midrule",
            f"Published weights & {num(mean_of(P['ami'], PUB, 144, 'bic'))} & {num(mean_of(P['earvn'], PUB, 96, 'bic'))} & "
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

    g1, g2 = r1("resnet18", "bsrgan (published)"), r1("resnet18", "realesrgan (published)")
    put("RecGanPair", (f"both reach {g1}\\%" if g1 == g2 else f"reach {g1}\\% and {g2}\\%") if g1 and g2 else None, why)
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
    # EER của ba mô hình học với bicubic so với phóng bicubic (phản biện 04/11, TODO P4): câu ở 6.4 nói khoảng tin cậy chứa 0
    t = Pd.get("resnet18")
    bt = [(PUB, "bicubic"), ("rrdb_psnr (published)", "bicubic"), ("span/bic", "bicubic")]
    if t is not None and all(k in t.index for k in bt):
        de = t.loc[bt]
        if not ((de.d_eer_lo < 0) & (de.d_eer_hi > 0)).all():
            print("CHÚ Ý: EER của mô hình học với bicubic khác bicubic có ý nghĩa; sửa câu EER trong experiments.tex")
        put("RecEerBicTrMin", num(de.d_eer.min(), 1))
        put("RecEerBicTrMax", num(de.d_eer.max(), 1))
    else:
        put("RecEerBicTrMin", None, why)
        put("RecEerBicTrMax", None, why)
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
        # ảnh dò ngoài hai nhóm (3 ảnh mức 65, 7 ảnh mức 81 đến 87): ghi trong chú thích Bảng 7 (phản biện 04/11, TODO P4)
        put("RecQMidN", int(R["meta"]["n_probe"] - q.loc[("bicubic", "low"), "n_probe"] - q.loc[("bicubic", "high"), "n_probe"]))
        for g, tag in (("low", "QLow"), ("high", "QHigh")):
            b = q.loc[("bicubic", g)]
            put(f"Rec{tag}N", int(b["n_probe"]))
            put(f"Rec{tag}Subj", int(b["n_subjects"]))
            put(f"Rec{tag}RankBic", num(b["rank1"], 1))
            for arm_, t2 in ((PUB, "Pub"), ("span/est", "Est"), ("bsrgan (published)", "Bsrgan"), ("span/bic", "Bict")):
                put(f"Rec{tag}{t2}", txt(q.loc[(arm_, g)], "d_bic"))
    else:
        rows = ["\\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} \\\\"]
        put("RecQMidN", None, why)
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
        put(f"Rec{name}BsrganVsBic", txt(d["p"].loc[("bsrgan (published)", "bicubic")], "d_rank1") if ok else None, why)
        put(f"Rec{name}GenVsEst", txt(d["p"].loc[("span/generic", "span/est")], "d_rank1") if ok else None, why)
        put(f"Rec{name}JpgVsEst", txt(d["p"].loc[("span/bicjpeg75", "span/est")], "d_rank1") if ok else None, why)
    a = L["awex_resnet18"]
    put("RecAwexEnrolled", len(a["meta"]["subjects"]) if a else None, why)
    put("RecAwexSubj", a["meta"]["n_probe_subjects"] if a else None, why)
    put("RecAwexProbes", a["meta"]["n_probe"] if a else None, why)
    f = res / "recog" / "blockiness.json"   # scripts/probe_blockiness.py
    bl = json.loads(f.read_text()) if f.exists() else None
    put("BlockEarvnLow", num(bl["earvn"]["groups"]["low"]["median"]) if bl else None, why)
    put("BlockEarvnHigh", num(bl["earvn"]["groups"]["high"]["median"]) if bl else None, why)
    put("BlockAwex", num(bl["awex"]["groups"]["all"]["median"]) if bl else None, why)
    put("RecEarvnGalMed", int(bl["earvn"]["gallery_per_subject_median"]) if bl else None, why)
    put("RecAwexGalMed", int(bl["awex"]["gallery_per_subject_median"]) if bl else None, why)
    f = res / "recog" / "recognizer_resnet50_train_log.json"
    put("RecRfValAcc", num(100 * json.loads(f.read_text())["info"]["val_acc"], 1) if f.exists() else None, why)


def recog_synth(res: Path, out: Path) -> None:
    """Ảnh dò tổng hợp từ ảnh lớn của người test (TODO.md, việc Q2b; scripts/run_recog_synth2.sh): cạnh ngắn 24 và 36 px,
    sạch / JPEG 75 / suy giảm đo được, hai bộ nhận dạng. Chênh lệch rank-1 so với bicubic, theo nhánh huấn luyện."""
    base = res / "recog_synth2"
    recs, shorts, kinds = ("resnet18", "resnet50"), (24, 36), ("bic", "bicjpeg75", "est")
    bic_tr, cmp_tr = (PUB, "span/bic"), ("span/bicjpeg75", "span/est")
    keys = ["SynProbes", "SynSubj", "SynBicCleanS", "SynBicEstS", "SynBicEstL", "SynEstEstSRe", "SynEstEstSRf", "SynSigCompS",
            "SynNCompS"] + [p + s for p in ("SynBicTrCleanS", "SynCmpTrCleanS", "SynCmpTrCompS", "SynBicTrCompS", "SynCmpTrCompL",
                                            "SynCmpTrCleanL") for s in ("Min", "Max")]
    D = {}
    for t in recs:
        for s in shorts:
            for k in kinds:
                d = base / f"{t}_{s}px_{k}_summary"
                if (d / "summary.csv").exists():
                    D[t, s, k] = (pd.read_csv(d / "summary.csv").set_index("arm"),
                                  pd.read_csv(d / "paired.csv").set_index(["arm", "minus"]))
    if len(D) != len(recs) * len(shorts) * len(kinds):
        for k in keys:
            put(k, None, "recog_synth2")
        write(out / "tab_recog_synth.tex", "\\todo{--} & & & & & & \\\\\n")
        return

    def d_(t, s, k, arm_):
        return D[t, s, k][1].loc[(arm_, "bicubic")]

    def ci(r):
        return f"{num(r['d_rank1'], 1, True)} {{\\scriptsize[{num(r['d_rank1_lo'], 1)}, {num(r['d_rank1_hi'], 1)}]}}"

    def vals(s, ks, arms):
        return [d_(t, s, k, a_) for t in recs for k in ks for a_ in arms]

    def rng(prefix, rows, magnitude=False):
        v = [abs(r["d_rank1"]) if magnitude else r["d_rank1"] for r in rows]
        put(prefix + "Min", num(min(v), 1))
        put(prefix + "Max", num(max(v), 1))

    rows = []
    labels = ((PUB, "SPAN-S, published"), ("span/bic", "SPAN-S, fine-tuned bicubic"), ("span/bicjpeg75", "SPAN-S, JPEG 75"),
              ("span/est", "SPAN-S, measured (ours)"))
    for t, name in zip(recs, ("Ear-R18", "Ear-R50")):
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
        rows.append("Bicubic $\\times$4 (rank-1, \\%) & " + " & ".join(num(D[t, s, k][0].loc["bicubic", "rank1"], 1)
                                                                   for s in shorts for k in kinds) + " \\\\")
        for a_, lab in labels:
            rows.append(f"{lab} & " + " & ".join(ci(d_(t, s, k, a_)) for s in shorts for k in kinds) + " \\\\")
        if t != recs[-1]:
            rows.append("\\midrule")
    write(out / "tab_recog_synth.tex", "\n".join(rows) + "\n")
    s0 = D["resnet18", 24, "bic"][0]
    put("SynProbes", int(s0.loc["bicubic", "n_probe"]))
    put("SynSubj", int(s0.loc["bicubic", "n_subjects"]))
    put("SynBicCleanS", num(s0.loc["bicubic", "rank1"], 1))
    put("SynBicEstS", num(D["resnet18", 24, "est"][0].loc["bicubic", "rank1"], 1))
    put("SynBicEstL", num(D["resnet18", 36, "est"][0].loc["bicubic", "rank1"], 1))

    def txt(r):
        return f"{num(r['d_rank1'], 1, True)} [{num(r['d_rank1_lo'], 1)}, {num(r['d_rank1_hi'], 1)}]"

    def sig(rows):
        return sum(int(r["d_rank1_lo"] > 0 or r["d_rank1_hi"] < 0) for r in rows)

    # mỗi nhóm câu ở mục 6.4 ứng với đúng các ô này của Bảng tab:recogsynth (phản biện 05/11: câu phải khớp từng ô)
    groups = {"SynJpgBicTr": (24, ("bicjpeg75",), bic_tr, True), "SynJpgCmpTr": (24, ("bicjpeg75",), cmp_tr, False),
              "SynCleanBicTr": (24, ("bic",), bic_tr, False), "SynCleanJpgTr": (24, ("bic",), ("span/bicjpeg75",), True),
              "SynCleanCmpTrL": (36, ("bic",), cmp_tr, True), "SynCompCmpTrL": (36, ("bicjpeg75", "est"), cmp_tr, False)}
    G = {}
    for name, (sz, ks, arms, mag) in groups.items():
        rows = vals(sz, ks, arms)
        G[name] = rows
        rng(name, rows, magnitude=mag)
        put(name + "Sig", sig(rows))
        put(name + "N", len(rows))
    put("SynEstEstSRe", txt(d_("resnet18", 24, "est", "span/est")))
    put("SynEstEstSRf", txt(d_("resnet50", 24, "est", "span/est")))
    put("SynJpgEstSRf", txt(d_("resnet50", 24, "est", "span/bicjpeg75")))
    put("SynCleanEstSRe", txt(d_("resnet18", 24, "bic", "span/est")))
    put("SynCleanEstSRf", txt(d_("resnet50", 24, "bic", "span/est")))
    # các câu giả định: ô JPEG 75 ở 24 px và ô sạch 36 px có ý nghĩa ở mọi khoảng, ô nén 36 px không có ô nào
    want = {"SynJpgBicTr": "all", "SynJpgCmpTr": "all", "SynCleanBicTr": "all", "SynCleanJpgTr": "all",
            "SynCleanCmpTrL": "all", "SynCompCmpTrL": "none"}
    for name, w in want.items():
        k = sig(G[name])
        if (w == "all" and k != len(G[name])) or (w == "none" and k != 0):
            print(f"CHÚ Ý: {name}: {k}/{len(G[name])} khoảng không chứa 0; sửa câu ở mục 6.4, tóm tắt, đóng góp 4")
    if not all(r["d_rank1"] < 0 for r in G["SynJpgBicTr"]) or not all(r["d_rank1"] > 0 for r in G["SynJpgCmpTr"]):
        print("CHÚ Ý: chiều chênh lệch ở ô JPEG 75, 24 px đã đổi; sửa câu ở mục 6.4")


def natural_control(res: Path, out: Path) -> None:
    """Đối chứng trên ảnh tự nhiên (scripts/run_div2k_control.sh): cùng 16 mô hình, cùng các mức nén, ảnh vào từ 24 đến
    192 px. Chưa có kết quả thì bảng và macro ra ô TBD."""
    why = "div2k"
    f = res / "t2_div2k_summary" / "quality.csv"
    kinds = ("bic", "bicjpeg93", "bicjpeg85", "bicjpeg75", "bicjpeg60")

    def cells(df, tier):
        outc, st = [], {}
        for k in kinds:
            c = df[(df.tier == tier) & (df.degrade == k) & df.model.isin(PSNR16)]
            if len(c) != 16:
                outc.append("--")
                continue
            nb, ns = int((c.gain < 0).sum()), int((c.gain_hi < 0).sum())
            st[k] = (float(c.gain.median()), nb, ns, float(c.gain.min()), float(c.gain.max()))
            outc.append(f"{num(c.gain.median(), 2, True)} ({nb}/{ns})")
        return outc, st

    rows, stats = [], {}
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.scale == 4] if "scale" in d.columns else d
        for tier in sorted(d.tier.unique()):
            c, st = cells(d, tier)
            stats[int(tier)] = st
            n = int(d[(d.tier == tier) & (d.model == "bicubic")].n.iloc[0])
            rows.append(f"DIV2K & {int(tier) // 4} & {n} & " + " & ".join(c) + " & -- \\\\")
    else:
        rows.append("DIV2K & \\todo{--} & \\todo{--} & " + " & ".join(["\\todo{--}"] * 6) + " \\\\")
    # Bảng A.1 gộp ảnh tai và DIV2K (phản biện 04/11, TODO P3): các hàng AMI nằm ở tab_generality, không in lại ở đây
    write(out / "tab_natural.tex", "\n".join(rows) + "\n")
    small, large = stats.get(144), stats.get(max(stats)) if stats else None
    put("NatLargePx", max(stats) // 4 if stats else None, why)
    for tag, st in (("S", small), ("L", large)):
        for k, kt in (("bicjpeg75", "QLow"), ("bicjpeg60", "QVLow"), ("bicjpeg85", "QMid"), ("bic", "Clean")):
            ok = st is not None and k in st
            put(f"Nat{tag}{kt}Med", num(st[k][0], 2, True) if ok else None, why)
            put(f"Nat{tag}{kt}Below", st[k][1] if ok else None, why)
            put(f"Nat{tag}{kt}Sig", st[k][2] if ok else None, why)
    put("NatN", int(pd.read_csv(f).n.max()) if f.exists() else None, why)
    if f.exists():
        d = pd.read_csv(f)
        x = d[(d.tier == 144) & d.model.isin(PSNR16)]
        put("NatSpreadCleanS", num(x[x.degrade == "bic"].psnr_y.max() - x[x.degrade == "bic"].psnr_y.min()))
        put("NatSpreadQLowS", num(x[x.degrade == "bicjpeg75"].psnr_y.max() - x[x.degrade == "bicjpeg75"].psnr_y.min()))
        put("NatSBicClean", num(float(d[(d.tier == 144) & (d.model == "bicubic") & (d.degrade == "bic")].psnr_y.iloc[0])))
        put("NatSmallPx", int(d.tier.min()) // 4)
    else:
        for k in ("NatSpreadCleanS", "NatSpreadQLowS", "NatSBicClean", "NatSmallPx"):
            put(k, None, why)


def error_ratio(res: Path, out: Path, fig: Path) -> None:
    """Sai số do nén so với sai số nội suy, trên mọi bộ ảnh.

    Với mỗi ô (bộ ảnh × cỡ ảnh vào × mức JPEG): E_i là MSE của nội suy bicubic trên ảnh vào sạch, E_c là phần MSE tăng
    thêm khi ảnh vào bị nén. Ba phép tính, đều từ các file theo ảnh đã có:
      1. tỉ số E_c / E_i và trung vị phần hơn bicubic của 16 mô hình, từng ô (hình và tương quan hạng);
      2. phân tích MSE của từng mô hình: MSE(nén) = a·E_i + b·E_c, với a = MSE(sạch) / E_i. b > 1 nghĩa là mô hình
         khuếch đại sai số nén;
      3. tỉ số (và mức JPEG) tại đó phần hơn đổi dấu, nội suy tuyến tính theo log tỉ số giữa hai mức nén kề nhau, cho
         từng bộ ảnh và cỡ ảnh, kèm khoảng tin cậy bootstrap theo người (theo ảnh với DIV2K).
    Chỉ lấy các ô nén ở một mức JPEG (ô 'est' trộn hai mức nén cùng nhiễu, mờ)."""
    import os

    from scipy.stats import spearmanr

    why = "div2k"
    sets = (("AMI", "t2"), ("DIV2K", "t2_div2k"), ("EarVN1.0", "t2_earvn"), ("AWEx", "t2_awex"))
    names = ("Cells", "Sets", "MaxAbove", "MinBelow", "Mixed", "Spearman", "SpearmanQ", "AmiQLow", "DivQLow", "AmiQHigh",
             "DivQVLow", "PxMin", "PxMax")

    def mse(per, model, t, k, scale=4):
        f = res / per / f"{model}__hr{t}_x{scale}_{k}.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f, dtype={"subject": str}).set_index("key")
        return 10 ** (-d.psnr_y / 10), d.subject

    if not (res / "t2_div2k").is_dir():
        for k in names:
            put("Ratio" + k, None, why)
        for k in ("CrossNatMin", "CrossNatMax", "CrossAmiMin", "CrossAmiMax", "CrossWildMin", "CrossWildMax", "CrossAllMin",
                  "CrossAllMax", "CrossQNatMin", "CrossQNatMax", "CrossQAmiMin", "CrossQAmiMax", "CrossQWildMin",
                  "CrossQWildMax", "AmpMin", "AmpMax", "ReduceMin", "ReduceMax", "XtwoRatio", "XtwoGainClean", "XtwoGainQLow"):
            put(k, None, why)
        write(out / "tab_crossover.tex", "\\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} & \\todo{--} \\\\\n")
        return

    rng = np.random.default_rng(0)
    cells, cross, amp_b, amp_a = [], [], [], []
    for name, per in sets:
        tiers = sorted({int(f.split("__hr")[1].split("_")[0]) for f in os.listdir(res / per)
                        if f.startswith("bicubic__hr") and "_x4_bic.csv" in f})
        for t in tiers:
            base = mse(per, "bicubic", t, "bic")
            if base is None:
                continue
            ei, subj = base
            sr_clean = {m_: mse(per, m_, t, "bic") for m_ in PSNR16}
            if any(v is None for v in sr_clean.values()):
                continue
            pts = {}
            for q in (93, 85, 75, 60):
                bj = mse(per, "bicubic", t, f"bicjpeg{q}")
                sj = {m_: mse(per, m_, t, f"bicjpeg{q}") for m_ in PSNR16}
                if bj is None or any(v is None for v in sj.values()):
                    continue
                J = np.stack([bj[0].loc[ei.index].to_numpy()] + [sj[m_][0].loc[ei.index].to_numpy() for m_ in PSNR16], 1)
                pts[q] = J
                Ei, Ec = float(ei.mean()), float(J[:, 0].mean() - ei.mean())
                a_ = np.array([float(sr_clean[m_][0].loc[ei.index].mean()) / Ei for m_ in PSNR16])
                b_ = (J[:, 1:].mean(0) - a_ * Ei) / Ec
                amp_a += a_.tolist()
                amp_b += b_.tolist()
                psnr = -10 * np.log10(J)
                g = psnr[:, 1:].mean(0) - psnr[:, 0].mean()
                # sai số dư tuyệt đối do nén, (b-1)E_c, và phần sai số nội suy mô hình tiết kiệm được, (1-a)E_i; cả hai chia
                # cho E_i. Mô hình hơn bicubic khi và chỉ khi phần dư nhỏ hơn phần tiết kiệm.
                cells.append({"set": name, "px": t // 4, "q": q, "ratio": Ec / Ei, "gain": float(np.median(g)),
                              "below": int((g < 0).sum()), "b_median": float(np.median(b_)), "b_min": float(b_.min()),
                              "excess": float(np.median((b_ - 1) * Ec / Ei)), "saving": float(np.median(1 - a_)),
                              **{f"ex_{m_}": float((b_[k_] - 1) * Ec / Ei) for k_, m_ in enumerate(PSNR16)}})
            if len(pts) < 2:
                continue
            qs = sorted(pts, reverse=True)
            E = ei.to_numpy()
            sv = subj.to_numpy()
            us = np.unique(sv)
            idx = {u: np.where(sv == u)[0] for u in us}

            def crossing(rows, qs=qs, pts=pts, E=E):
                e0 = E[rows].mean()
                r, g = [], []
                for q in qs:
                    J = pts[q][rows]
                    p = -10 * np.log10(J)
                    r.append((J[:, 0].mean() - e0) / e0)
                    g.append(np.median(p[:, 1:].mean(0) - p[:, 0].mean()))
                for i in range(len(g) - 1):
                    if g[i] > 0 >= g[i + 1] and r[i] > 0:
                        w = g[i] / (g[i] - g[i + 1])
                        return float(np.exp(np.log(r[i]) + w * (np.log(r[i + 1]) - np.log(r[i])))), qs[i] + w * (qs[i + 1] - qs[i])
                return np.nan, np.nan

            cr, cq = crossing(np.arange(len(E)))
            bs = np.array([crossing(np.concatenate([idx[u] for u in rng.choice(us, len(us))])) for _ in range(400)])
            # khoảng giữa hai mức đã đo chứa điểm đổi dấu (cột đầu của bảng; góp ý reviewer 06/11)
            b_hi_ = min([q for q in qs if q > cq], default=np.nan) if cq == cq else np.nan
            b_lo_ = max([q for q in qs if q < cq], default=np.nan) if cq == cq else np.nan
            cross.append({"set": name, "px": t // 4, "n": len(E), "levels": len(qs), "ratio": cr, "b_lo": b_lo_, "b_hi": b_hi_,
                          "ratio_lo": float(np.nanpercentile(bs[:, 0], 2.5)), "ratio_hi": float(np.nanpercentile(bs[:, 0], 97.5)),
                          "q": cq, "q_lo": float(np.nanpercentile(bs[:, 1], 2.5)), "q_hi": float(np.nanpercentile(bs[:, 1], 97.5))})
    R, X = pd.DataFrame(cells), pd.DataFrame(cross).dropna(subset=["ratio"])
    R.to_csv(res / "error_ratio.csv", index=False)
    X.to_csv(res / "error_ratio_crossover.csv", index=False)
    above, below = R[R.below == 0], R[R.below == 16]
    put("RatioCells", len(R))
    put("RatioSets", R.set.nunique())
    put("RatioPxMin", int(R.px.min()))
    put("RatioPxMax", int(R.px.max()))
    put("RatioMaxAbove", num(above.ratio.max(), 2))
    put("RatioMinBelow", num(below.ratio.min(), 2))
    put("RatioMixed", len(R) - len(above) - len(below))
    put("RatioSpearman", num(spearmanr(R.ratio, R.gain).statistic, 2))
    put("RatioSpearmanQ", num(spearmanr(R.q, R.gain).statistic, 2))

    def one(st, px, qq):
        x = R[(R.set == st) & (R.px == px) & (R.q == qq)]
        return num(float(x.ratio.iloc[0]), 2) if len(x) else None

    put("RatioAmiQLow", one("AMI", 36, 75))
    put("RatioAmiQHigh", one("AMI", 36, 93))
    put("RatioDivQLow", one("DIV2K", 36, 75))
    put("RatioDivQVLow", one("DIV2K", 36, 60))
    # phần dư tuyệt đối có tăng dần khi nén nặng hơn trong mọi nhóm (bộ ảnh, cỡ ảnh) có ít nhất hai mức không
    grp = [g.sort_values("q", ascending=False).excess.to_numpy() for _, g in R.groupby(["set", "px"]) if len(g) > 1]
    mono = sum(bool(np.all(np.diff(e) > 0)) for e in grp)
    put("ExcessMono", mono)
    # cùng phép kiểm cho từng mô hình riêng (không chỉ mô hình trung vị)
    per = [(m_, g.sort_values("q", ascending=False)[f"ex_{m_}"].to_numpy()) for _, g in R.groupby(["set", "px"]) if len(g) > 1
           for m_ in PSNR16]
    put("ExcessMonoModels", sum(bool(np.all(np.diff(e) > 0)) for _, e in per))
    put("ExcessModelGroups", len(per))
    put("ExcessGroups", len(grp))
    if mono != len(grp):
        print("CHÚ Ý: phần dư (b-1)E_c không tăng đều theo mức nén ở mọi nhóm; sửa câu ở mục 4.2")

    def cellv(st, px, qq, col_):
        x = R[(R.set == st) & (R.px == px) & (R.q == qq)]
        return float(x[col_].iloc[0]) if len(x) else None

    for st, tg in (("AMI", "Ami"), ("DIV2K", "Div")):
        put(f"Excess{tg}High", num(cellv(st, 36, 93, "excess"), 2))
        put(f"Excess{tg}Low", num(cellv(st, 36, 60, "excess"), 2))
        put(f"Saving{tg}", num(cellv(st, 36, 75, "saving"), 2))
    put("AmpMin", num(min(amp_b), 1))
    put("AmpMax", num(max(amp_b), 1))
    put("ReduceMin", num(min(amp_a), 2))
    put("ReduceMax", num(max(amp_a), 2))
    for tag, sel in (("Nat", X.set == "DIV2K"), ("Ami", X.set == "AMI"), ("Wild", X.set.isin(["EarVN1.0", "AWEx"])),
                     ("All", X.set == X.set)):
        put(f"Cross{tag}Min", num(X[sel].ratio.min(), 2))
        put(f"Cross{tag}Max", num(X[sel].ratio.max(), 2))
        if tag != "All":
            put(f"CrossQ{tag}Min", int(round(X[sel].q.min())))
            put(f"CrossQ{tag}Max", int(round(X[sel].q.max())))
    # Khoảng lưới đo chứa điểm đổi dấu (góp ý reviewer 06/11: số nội suy chính xác hơn dữ liệu cho phép; văn bản dùng
    # hai mức đo kề nhau, bảng giữ số nội suy). Mức JPEG của benchmark: 60, 75, 85, 93.
    L = [60, 75, 85, 93]
    for tag, sel in (("Nat", X.set == "DIV2K"), ("Ami", X.set == "AMI")):
        lo_, hi_ = [l for l in L if l < X[sel].q.min()], [l for l in L if l > X[sel].q.max()]
        if not lo_ or not hi_ or len([l for l in L if lo_[-1] < l < hi_[0]]):
            print(f"CHÚ Ý: điểm đổi dấu {tag} không nằm trong một khoảng lưới; sửa tóm tắt, phần mở đầu và mục 4.2")
        put(f"CrossGrid{tag}Lo", lo_[-1] if lo_ else "TBD")
        put(f"CrossGrid{tag}Hi", hi_[0] if hi_ else "TBD")
        hl_ = Path(__file__).resolve().parent.parent / "paper" / "highlights.txt"
        if lo_ and hi_ and hl_.exists() and f"quality {lo_[-1]} and {hi_[0]}" not in hl_.read_text():
            print(f"CHÚ Ý: highlights.txt không còn khớp khoảng {lo_[-1]}–{hi_[0]} ({tag}); sửa highlights")
    # AMI theo cỡ ảnh vào: điểm đổi dấu dời về nén nhẹ hơn khi ảnh vào lớn hơn (thêm cỡ 48 px ngày 05/11; TODO P6)
    A = X[X.set == "AMI"].sort_values("px")
    if not (np.all(np.diff(A.q) > 0) and np.all(np.diff(A.ratio) < 0)):
        print("CHÚ Ý: điểm đổi dấu trên AMI không còn dời đều theo cỡ ảnh; sửa câu ở mục 4.2 và phần mở đầu")
    put("CrossQAmiSmallPx", int(A.px.iloc[0]))
    put("CrossQAmiLargePx", int(A.px.iloc[-1]))
    put("CrossQAmiSmall", int(round(A.q.iloc[0])))
    put("CrossQAmiLarge", int(round(A.q.iloc[-1])))
    put("CrossAmiSmall", num(A.ratio.iloc[0], 2))
    put("CrossAmiLarge", num(A.ratio.iloc[-1], 2))
    rows = []
    for r in X.to_dict("records"):
        kind = "Natural" if r["set"] == "DIV2K" else "Ear"
        rows.append(f"{kind} ({r['set']}) & {r['px']} & {r['levels']} & {int(r['b_lo'])}--{int(r['b_hi'])} & {r['q']:.1f} {{\\scriptsize[{min(r['q_lo'], r['q_hi']):.1f}, "
                    f"{max(r['q_lo'], r['q_hi']):.1f}]}} & {num(r['ratio'], 2)} {{\\scriptsize[{num(r['ratio_lo'], 2)}, "
                    f"{num(r['ratio_hi'], 2)}]}} \\\\")
    write(out / "tab_crossover_full.tex", "\n".join(rows) + "\n")
    # thân bài chỉ in khoảng mức đã đo; số nội suy ở phụ lục (phản biện thứ tám, 06/11)
    short = [" & ".join(r_.split(" & ")[:4]) + " \\\\" for r_ in rows]
    write(out / "tab_crossover.tex", "\n".join(short) + "\n")
    # ô ×2 duy nhất có sẵn: SwinIR-light ×2 trên AMI (đáp án 144 px, ảnh vào 72 px)
    c2, j2 = mse("t2", "bicubic", 144, "bic", 2), mse("t2", "bicubic", 144, "bicjpeg75", 2)
    s2c, s2j = mse("t2", "swinir_light_x2", 144, "bic", 2), mse("t2", "swinir_light_x2", 144, "bicjpeg75", 2)
    if all(v is not None for v in (c2, j2, s2c, s2j)):
        e0 = float(c2[0].mean())
        put("XtwoRatio", num((float(j2[0].mean()) - e0) / e0, 2))
        ps = lambda v: float((-10 * np.log10(v[0])).mean())   # noqa: E731
        put("XtwoGainClean", num(ps(s2c) - ps(c2), 2, True))
        put("XtwoGainQLow", num(ps(s2j) - ps(j2), 2, True))
    else:
        for k in ("XtwoRatio", "XtwoGainClean", "XtwoGainQLow"):
            put(k, None, "x2")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    f, ax = plt.subplots(figsize=(5.2, 3.0))
    col = {"AMI": "C3", "EarVN1.0": "C1", "AWEx": "C4", "DIV2K": "0.35"}
    mk = {60: "v", 75: "o", 85: "s", 93: "^"}
    ax.axvspan(X.ratio.min(), X.ratio.max(), color="0.9", lw=0, zorder=0)
    for (st, qq), g in R.groupby(["set", "q"]):
        ax.scatter(g.ratio, g.gain, s=16 + g.px / 6, c=col[st], marker=mk.get(qq, "x"), alpha=0.85, linewidths=0, zorder=2)
    ax.axhline(0, color="C0", lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Compression error / interpolation error ($E_c/E_i$)")
    ax.set_ylabel("Median PSNR gain over bicubic (dB)")
    h1 = [plt.Line2D([], [], marker="o", ls="", color=c, label=("ear: " if n != "DIV2K" else "natural: ") + n) for n, c in col.items()]
    h2 = [plt.Line2D([], [], marker=v, ls="", color="k", label=f"JPEG {k}") for k, v in sorted(mk.items(), reverse=True)]
    ax.legend(handles=h1 + h2, frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    f.tight_layout()
    f.savefig(fig / "fig_ratio.png", dpi=600)
    plt.close(f)

    # phần dư tuyệt đối (b-1)E_c theo mức nén, cùng phần tiết kiệm (1-a)E_i (đường ngang), đều chia cho E_i
    f, ax = plt.subplots(figsize=(5.2, 2.8))
    show = (("AMI", 36, "ear: AMI, 36 px"), ("EarVN1.0", 24, "ear: EarVN1.0, 24 px"), ("AWEx", 24, "ear: AWEx, 24 px"),
            ("DIV2K", 36, "natural: DIV2K, 36 px"))
    for st, px, lab in show:
        g = R[(R.set == st) & (R.px == px)].sort_values("q", ascending=False)
        if g.empty:
            continue
        xs = [str(q) for q in g.q]
        ax.plot(xs, g.excess, color=col[st], marker="o", ms=3.5, lw=1.4, label=lab)
        ax.axhline(float(g.saving.median()), color=col[st], lw=0.9, ls="--")
    ax.set_xlabel("JPEG quality of the input")
    ax.set_ylabel("Error relative to $E_i$")
    ax.text(0.01, 0.97, "solid: excess $(b-1)E_c$; dashed: saving $(1-a)E_i$", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    f.tight_layout()
    f.savefig(fig / "fig_excess.png", dpi=600)
    plt.close(f)


def codec_control(res: Path, out: Path) -> None:
    """Chuẩn nén khác JPEG (TODO.md, việc Q1): WebP và AVIF trên AMI 36 px, so với JPEG ở cùng ảnh.
    Với mỗi chuẩn nén và mức chất lượng: E_c/E_i, trung vị phần hơn bicubic, số mô hình kém bicubic, hệ số b (phương trình
    phân tích sai số); rồi tỉ số và mức chất lượng tại đó phần hơn đổi dấu, kèm khoảng tin cậy bootstrap theo người."""
    codecs = (("JPEG", "Jpeg", "t2", "bicjpeg", (93, 85, 75, 60)), ("WebP", "Webp", "codec", "bicwebp", (95, 90, 85, 75, 60)),
              ("AVIF", "Avif", "codec", "bicavif", (90, 85, 80, 70, 60)))
    keys = ("BMin", "BMax", "Cross", "CrossLo", "CrossHi", "CrossQ")

    def mse(per, m, k):
        f = res / per / f"{m}__hr144_x4_{k}.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f, dtype={"subject": str}).set_index("key")
        return 10 ** (-d.psnr_y / 10), d.subject

    base = mse("t2", "bicubic", "bic")
    need = [mse(per, m, f"{pre}{q}") for _, _, per, pre, qs in codecs for q in qs for m in ["bicubic"] + PSNR16]
    if base is None or any(v is None for v in need):
        for _, tg, *_ in codecs:
            for k in keys:
                put(f"Codec{tg}{k}", None, "codec")
        put("CodecLevels", None, "codec")
        write(out / "tab_codec.tex", "\\todo{--} & & & & & \\\\\n")
        return
    ei, subj = base
    E, sv = ei.to_numpy(), subj.loc[ei.index].to_numpy()
    us = np.unique(sv)
    idx = {u: np.where(sv == u)[0] for u in us}
    a_ = np.array([float(mse("t2", m, "bic")[0].loc[ei.index].mean()) / ei.mean() for m in PSNR16])
    rng = np.random.default_rng(0)
    rows, nlev = [], 0
    for name, tg, per, pre, qs in codecs:
        pts = {q: np.stack([mse(per, m, f"{pre}{q}")[0].loc[ei.index].to_numpy() for m in ["bicubic"] + PSNR16], 1) for q in qs}
        bs_all = []
        for i, q in enumerate(qs):
            J = pts[q]
            Ec = J[:, 0].mean() - E.mean()
            b_ = (J[:, 1:].mean(0) - a_ * E.mean()) / Ec
            bs_all += b_.tolist()
            p = -10 * np.log10(J)
            g = p[:, 1:].mean(0) - p[:, 0].mean()
            rows.append(f"{name if i == 0 else ''} & {q} & {num(Ec / E.mean(), 2)} & {num(float(np.median(g)), 2, True)} & "
                        f"{int((g < 0).sum())} & {num(float(np.median(b_)), 2)} \\\\")
            nlev += 1
        rows[-1] += "\n\\addlinespace" if tg != "Avif" else ""

        def crossing(rr, qs=qs, pts=pts):
            e0, r, g = E[rr].mean(), [], []
            for q in qs:
                J = pts[q][rr]
                p = -10 * np.log10(J)
                r.append((J[:, 0].mean() - e0) / e0)
                g.append(np.median(p[:, 1:].mean(0) - p[:, 0].mean()))
            for i in range(len(g) - 1):
                if g[i] > 0 >= g[i + 1]:
                    w = g[i] / (g[i] - g[i + 1])
                    return float(np.exp(np.log(r[i]) + w * (np.log(r[i + 1]) - np.log(r[i])))), qs[i] + w * (qs[i + 1] - qs[i])
            return np.nan, np.nan

        cr, cq = crossing(np.arange(len(E)))
        bs = np.array([crossing(np.concatenate([idx[u] for u in rng.choice(us, len(us))])) for _ in range(400)])
        if min(bs_all) <= 1:
            print(f"CHÚ Ý: {name}: có mô hình với b <= 1; sửa câu về chuẩn nén khác ở mục 4.2")
        put(f"Codec{tg}BMin", num(min(bs_all), 1))
        put(f"Codec{tg}BMax", num(max(bs_all), 1))
        put(f"Codec{tg}Cross", num(cr, 2))
        put(f"Codec{tg}CrossLo", num(float(np.nanpercentile(bs[:, 0], 2.5)), 2))
        put(f"Codec{tg}CrossHi", num(float(np.nanpercentile(bs[:, 0], 97.5)), 2))
        put(f"Codec{tg}CrossQ", int(round(cq)))
    put("CodecLevels", nlev)
    write(out / "tab_codec.tex", "\n".join(rows) + "\n")


def seeds(res: Path, folds_path: Path) -> None:
    """Dao động do seed huấn luyện (TODO.md, việc Q3): fold 1, ba nhánh SPAN, seed 1 (lần chạy cuối), 101 và 102."""
    keys = ("SeedMaxRange", "SeedMedRange", "SeedCells", "SeedDiffs", "SeedFlips", "SeedMinDiff")
    sets = ("ami", "earvn", "awex")
    if not all((res / "seeds" / s).is_dir() for s in sets):
        for k in keys:
            put(k, None, "seeds")
        return
    ranges, diffs = [], []
    for s in sets:
        d = load_dir(res / "seeds" / s)
        if s == "ami":
            d = d[d.subject.isin(set(json.loads(folds_path.read_text())["folds"]["1"]["test"]))]
        d = d[d.arm != "bicubic"].copy()
        d["base"] = d.arm.str.replace(r" \[seed \d+\]", "", regex=True)
        d["seed"] = d.arm.str.extract(r"seed (\d+)")[0].fillna("1")
        t = d.groupby(["base", "seed", "tier", "degrade"]).psnr_y.mean()
        ranges += t.groupby(["base", "tier", "degrade"]).agg(lambda x: x.max() - x.min()).tolist()
        for _, g in t.groupby(["tier", "degrade"]):
            g = g.droplevel(["tier", "degrade"]).unstack("base")
            for b in ("span/bicjpeg75", "span/bic"):
                diffs.append((g["span/est"] - g[b]).to_numpy())
    flips = sum(bool(x.min() < 0 < x.max()) for x in diffs)
    if flips:
        print(f"CHÚ Ý: {flips} chênh lệch giữa nhánh đổi dấu theo seed; sửa câu ở mục 5.3")
    put("SeedMaxRange", num(max(ranges), 2))
    put("SeedMedRange", num(float(np.median(ranges)), 3))
    put("SeedCells", len(ranges))
    put("SeedDiffs", len(diffs))
    put("SeedFlips", "none" if flips == 0 else flips)
    put("SeedMinDiff", num(min(float(np.abs(x).min()) for x in diffs), 2))


def headroom(res: Path) -> None:
    """Phần tăng nhận dạng theo "dư địa" của từng người: rank-1 trên ảnh dò lớn trừ rank-1 trên ảnh dò nhỏ (bicubic).
    Tính từ các file theo ảnh dò của recog_eval.py; chỉ lấy người có ít nhất 5 ảnh dò nhỏ trên EarVN1.0."""
    from scipy.stats import spearmanr

    why = "recog2"
    for tag, name, min_n in (("resnet18", "EarvnRe", 5), ("resnet50", "EarvnRf", 5), ("awex_resnet18", "AwexRe", 1),
                             ("awex_resnet50", "AwexRf", 1)):
        d = res / "recog" / tag
        keys = ("Subj", "LowRoom", "LowGain", "HighRoom", "HighGain", "Rho", "P")
        if not (d / "probes.csv").exists():
            for k in keys:
                put(f"Head{name}{k}", None, why)
            continue
        hits = {}
        for fcsv in sorted(d.glob("*.csv")):
            if fcsv.stem in ("probes", "refs", "gallery", "scan"):
                continue
            x = pd.read_csv(fcsv, dtype={"subject": str})
            hits.setdefault(arm_of(fcsv.stem)[0], []).append((x["rank"] == 1).astype(float).groupby(x["subject"]).mean())
        S = pd.DataFrame({a_: pd.concat(v, axis=1).mean(1) for a_, v in hits.items() if a_ in ("ref_large", "bicubic", "span/est")})
        S["n"] = pd.read_csv(d / "probes.csv", dtype={"subject": str}).groupby("subject").size()
        S = S.dropna()
        S = S[S.n >= min_n]
        S["room"], S["gain"] = S["ref_large"] - S["bicubic"], S["span/est"] - S["bicubic"]
        med = S.room.median()
        lo, hi = S[S.room <= med], S[S.room > med]

        def w(x, c):
            return float((x[c] * x.n).sum() / x.n.sum())

        r = spearmanr(S.room, S.gain)
        put(f"Head{name}Subj", len(S))
        put(f"Head{name}LowRoom", num(100 * w(lo, "room"), 1, True))
        put(f"Head{name}LowGain", num(100 * w(lo, "gain"), 1, True))
        put(f"Head{name}HighRoom", num(100 * w(hi, "room"), 1, True))
        put(f"Head{name}HighGain", num(100 * w(hi, "gain"), 1, True))
        put(f"Head{name}Rho", num(r.statistic, 2))
        put(f"Head{name}P", num(r.pvalue, 3))


def latency_ios(res: Path, out: Path, lat: pd.DataFrame) -> None:
    """Độ trễ đo trên iPhone bằng Xcode (deploy_ios/README.md). File chưa có số thì bảng và macro ra ô TBD."""
    why = "ios"   # results/latency_ios.csv chưa được điền
    f = res / "latency_ios.csv"
    d = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["name", "compute_units", "median_ms"])
    d = d[pd.to_numeric(d.median_ms, errors="coerce").notna()] if len(d) else d
    v = {(r["name"], r["compute_units"]): float(r["median_ms"]) for r in d.to_dict("records")}

    def cell(name, cu):
        return num(v[(name, cu)], 1) if (name, cu) in v else "\\todo{--}"

    rows = []
    required = ("span_ch48", "disp26", "bsrgan")
    for name in sorted(PSNR16 + ["bsrgan"], key=lambda m_: lat.params.get(m_, 0)):
        if name not in required and (name, "cpu") not in v and (name, "all") not in v:
            continue   # mốc tùy chọn: không đo thì không có dòng
        rows.append(f"{DISPLAY[name]} & {num(lat.params[name] / 1e3, 0)} & {num(lat.median_ms[name], 2)} & "
                    f"{cell(name, 'cpu')} & {cell(name, 'all')} \\\\")
    write(out / "tab_latency_ios.tex", "\n".join(rows) + "\n")
    for name, tag in (("span_ch48", "Span"), ("disp26", "Disp"), ("bsrgan", "Bsrgan"), ("edsr_baseline", "Edsr"),
                      ("swinir_light", "Swin"), ("rrdb_psnr", "Rrdb")):
        for cu, t2 in (("cpu", "Cpu"), ("all", "All")):
            put(f"LatPhone{tag}{t2}", num(v[(name, cu)], 1) if (name, cu) in v else None, why)
    # số phép tính của SwinIR-light còn nằm lại trên CPU khi cho phép mọi đơn vị tính
    sw = d[(d.name == "swinir_light") & (d.compute_units == "all")] if len(d) and "ops_cpu" in d.columns else []
    put("PhoneSwinOpsCpu", int(sw.ops_cpu.iloc[0]) if len(sw) else None, why)
    put("PhoneSwinOps", int(sw.ops_cpu.iloc[0] + sw.ops_gpu.iloc[0] + sw.ops_ne.iloc[0]) if len(sw) else None, why)
    nm = len({k[0] for k in v if not k[0].endswith("_fp32")})
    put("PhoneNModels", WORDS.get(nm, str(nm)))   # bản FP32: chỉ để so với FBCNN (mục 6.6)
    for name, tag in (("span_ch48_fp32", "SpanF"), ("fbcnn_span_ch48_fp32", "Fb")):
        for cu, t2 in (("cpu", "Cpu"), ("all", "All")):
            put(f"LatPhone{tag}{t2}", num(v[(name, cu)], 1) if (name, cu) in v else None, why)
    ex = res / "coreml_export.json"   # scripts/export_coreml.py: phép so đầu ra Core ML với PyTorch
    ce = {x["name"]: x for x in json.loads(ex.read_text())} if ex.exists() else {}
    for name, tag in (("span_ch48", "Span"), ("disp26", "Disp"), ("fbcnn_span_ch48", "FbHalf")):
        put("CoremlPsnr" + tag, num(ce[name]["psnr_vs_torch_db"], 1) if name in ce and "psnr_vs_torch_db" in ce[name] else None, why)
    for cu, t2 in (("cpu", "Cpu"), ("all", "All")):   # BSRGAN chậm hơn SPAN bao nhiêu lần trên điện thoại
        ok = ("span_ch48", cu) in v and ("bsrgan", cu) in v
        put(f"LatPhoneRatio{t2}", int(round(v[("bsrgan", cu)] / v[("span_ch48", cu)])) if ok else None, why)
    first = d.iloc[0] if len(d) else None
    put("PhoneName", first["device"] if first is not None else "iPhone 12 Pro Max")
    put("PhoneChip", first["chip"] if first is not None else "A14 Bionic")
    put("PhoneOs", first["ios"] if first is not None else "18.7.8")


# ------------------------------------------------------------------ mốc thêm: khử nén rồi phóng, RRDB học có nén
REV_ROWS = [("bicubic", "Bicubic", None), (PUB, "SPAN-S, published", "span_ch48"),
            ("rrdb_psnr (published)", "RRDB, published", "rrdb_psnr"), ("bsrnet (published)", "BSRNet", "bsrnet"),
            ("realesrnet (published)", "Real-ESRNet", None),
            ("fbcnn_bicubic (published)", "FBCNN, then bicubic", "fbcnn_bicubic"),
            ("fbcnn_span_ch48 (published)", "FBCNN, then SPAN-S published", "fbcnn_span_ch48"),
            ("span/est", "SPAN-S, measured (ours)", "span_ch48"), ("rrdb/est", "RRDB, measured", "rrdb_psnr")]
REV_SETS = (("ami", "rev", (96, 144, 192)), ("earvn", "rev_earvn", (96,)), ("awex", "rev_awex", (96, 144)))
REV_COMP = ("bicjpeg93", "bicjpeg85", "bicjpeg75", "est")        # các kiểu ảnh vào có nén


def review(res: Path, out: Path) -> None:
    """Mốc cho vòng phản biện (scripts/run_review.sh): FBCNN rồi phóng; BSRNet, Real-ESRNet; RRDB tinh chỉnh (nếu đã có).

    Mọi phép so ghép cặp tính trên các ảnh mà cả hai phương pháp đều có (mô hình tự huấn luyện trên AMI chỉ có ảnh của
    người test các fold đã chạy)."""
    why = "review"   # chưa có kết quả của scripts/run_review.sh
    P = {}
    for name, sub, _ in REV_SETS:
        d = load_dir(res / sub)
        P[name] = None if d is None else d.groupby(["arm", "tier", "degrade", "key", "subject"], as_index=False)[
            [c for c in METRICS if c in d.columns]].mean()
    have = all(v is not None for v in P.values())

    def cellp(name, arm_, tier, kind):
        p_ = P[name]
        return p_[(p_.arm == arm_) & (p_.tier == tier) & (p_.degrade == kind)].set_index("key") if p_ is not None else None

    def diff(name, a_, b_, tier, kind):
        A, B = cellp(name, a_, tier, kind), cellp(name, b_, tier, kind)
        if A is None or A.empty or B.empty:
            return None
        k = A.index.intersection(B.index)
        return paired_diff(A.loc[k, "psnr_y"].to_numpy(), B.loc[k, "psnr_y"].to_numpy(), A.loc[k, "subject"].to_numpy(),
                           n_boot=2000) if len(k) else None

    def span_of(prefix, a_, b_, sets, kinds, **kw):
        es = [diff(n_, a_, b_, t, k) for n_, _, tiers in REV_SETS if n_ in sets for t in tiers for k in kinds
              if P[n_] is not None and k in set(P[n_].degrade)] if have else []
        rng_put(prefix, es, why=why, **kw)
        return es

    FB, BSR, RESR, EST = "fbcnn_span_ch48 (published)", "bsrnet (published)", "realesrnet (published)", "span/est"
    wild = ("earvn", "awex")
    span_of("RevFbVsEstAmi", FB, EST, ("ami",), REV_COMP, sign=True)                   # có nén, AMI
    span_of("RevFbVsEstWildJpeg", FB, EST, wild, ("bicjpeg93", "bicjpeg75"), sign=True)  # chỉ nén JPEG, ngoài thực tế
    span_of("RevFbVsEstWildEst", FB, EST, wild, ("est",), sign=True)                   # suy giảm đo được, ngoài thực tế
    span_of("RevFbVsEstCleanWild", FB, EST, wild, ("bic",), sign=True)
    span_of("RevFbVsEstCleanAmi", FB, EST, ("ami",), ("bic",), sign=True)
    span_of("RevFbVsBic", FB, "bicubic", ("ami",) + wild, REV_COMP, sign=True)
    span_of("RevBsrVsBic", BSR, "bicubic", ("ami",) + wild, REV_COMP, sign=True)
    span_of("RevBsrVsEst", BSR, EST, ("ami",) + wild, REV_COMP, magnitude=True)
    if P["ami"] is not None and "rrdb/est" in set(P["ami"].arm):       # RRDB tinh chỉnh với công thức của SPAN nhánh est
        e = span_of("RevRrdbVsEst", "rrdb/est", EST, ("ami",) + wild, REV_COMP)
        span_of("RevRrdbVsEstClean", "rrdb/est", EST, ("ami",) + wild, ("bic",), sign=True)
        # cùng ba ô như DegEff (AMI 36 px, EarVN1.0 và AWEx 24 px), ảnh vào JPEG 75: tác dụng của kiến trúc khi cùng
        # công thức huấn luyện, và tác dụng của công thức trên từng thân (so với trọng số công bố của chính thân đó)
        q75 = [("ami", 144), ("earvn", 96), ("awex", 96)]
        rng_put("RevRrdbVsEstQ", [diff(n_, "rrdb/est", EST, t, "bicjpeg75") for n_, t in q75])
        rng_put("RevRrdbGainQ", [diff(n_, "rrdb/est", "rrdb_psnr (published)", t, "bicjpeg75") for n_, t in q75])
        rng_put("RevSpanGainQ", [diff(n_, EST, PUB, t, "bicjpeg75") for n_, t in q75])
        if e and not all(x is not None and x.lo > 0 for x in e):
            print("CHÚ Ý: RRDB nhánh est không còn hơn SPAN nhánh est có ý nghĩa ở mọi ô có nén; sửa câu ở mục 6.6")
    e = span_of("RevResrVsBic", RESR, "bicubic", ("ami",) + wild, REV_COMP, magnitude=True)
    if e and not all(x is not None and x.point < 0 for x in e):
        print("CHÚ Ý: Real-ESRNet không còn kém bicubic ở mọi ô có nén; sửa câu trong experiments.tex (mục các mốc thêm)")

    # nhận dạng trên ảnh nhỏ thật của EarVN1.0 (hai mạng nhận dạng) và AWEx
    S, Pd = {}, {}
    for tag in ("resnet18", "resnet50", "awex_resnet18", "awex_resnet50"):
        f = res / "recog" / f"rev_{tag}_summary" / "summary.csv"
        if f.exists():
            S[tag] = pd.read_csv(f).set_index("arm")
            Pd[tag] = pd.read_csv(f.with_name("paired.csv")).set_index(["arm", "minus"])

    def r1(tag, arm_):
        return num(S[tag].loc[arm_, "rank1"], 1) if tag in S and arm_ in S[tag].index else None

    def gap(tag, arm_, base):
        if tag not in Pd or (arm_, base) not in Pd[tag].index:
            return None
        r = Pd[tag].loc[(arm_, base)]
        return f"{num(r['d_rank1'], 1, True)} [{num(r['d_rank1_lo'], 1)}, {num(r['d_rank1_hi'], 1)}]"

    for arm_, nm in ((FB, "Fb"), ("fbcnn_bicubic (published)", "Fbb"), (BSR, "Bsr"), (RESR, "Resr"), (EST, "Est"),
                     ("rrdb/est", "Rrdb")):
        for tag, sfx in (("resnet18", ""), ("resnet50", "Rf")):
            if arm_ == "rrdb/est" and not (tag in S and arm_ in S[tag].index):
                continue                                  # RRDB tinh chỉnh: chỉ có macro khi đã có kết quả
            put(f"RevRank{nm}{sfx}", r1(tag, arm_), why)
            if arm_ != EST:
                put(f"Rev{nm}VsEstRank{sfx}", gap(tag, arm_, EST), why)
                put(f"Rev{nm}VsBicRank{sfx}", gap(tag, arm_, "bicubic"), why)
    for arm_ in (BSR, RESR):                              # câu "both large models give a higher identification rate" (ResNet-50)
        r = Pd["resnet50"].loc[(arm_, EST)] if "resnet50" in Pd and (arm_, EST) in Pd["resnet50"].index else None
        if r is not None and not r["d_rank1_lo"] > 0:
            print(f"CHÚ Ý: {arm_} không còn hơn SPAN nhánh est có ý nghĩa với ResNet-50; sửa câu trong experiments.tex")
    sig = [(tag, a_) for tag in ("awex_resnet18", "awex_resnet50") if tag in Pd for (a_, m_), r in Pd[tag].iterrows()
           if m_ == "bicubic" and a_ not in ("direct", "ref_large") and (r["d_rank1_lo"] > 0 or r["d_rank1_hi"] < 0)]
    if sig:
        print(f"CHÚ Ý: trên AWEx có phương pháp đổi hạng 1 có ý nghĩa so với bicubic: {sig}; sửa câu trong experiments.tex")

    # chi phí: tham số và độ trễ trên GPU của máy chấm (cùng một lượt đo)
    lat = pd.read_csv(res / "latency_review.csv").set_index("name") if (res / "latency_review.csv").is_file() else None

    def cost(key, col):
        return float(lat.loc[key, col]) if lat is not None and key in lat.index and pd.notna(lat.loc[key, col]) else None

    # Độ trễ trong bảng: lấy từ lượt đo của Bảng 1 (latency_local.csv) cho mọi mô hình có ở đó; các mô hình kiến trúc RRDB
    # (BSRNet, Real-ESRNet) dùng chung số của RRDB như Hình 7. Chỉ FBCNN lấy từ lượt đo sau (latency_review.csv). Hai lượt
    # cho RRDB 13,39 và 13,21 ms: dao động giữa các lượt, không in hai số cho cùng một kiến trúc (phản biện 04/11, TODO P1).
    loc = pd.read_csv(res / "latency_local.csv")
    loc = loc[(loc.input_h == 68) & (loc.input_w == 48) & (loc.device == "cuda")].set_index("name")

    def ms_of(key):
        k = "rrdb_psnr" if key in ("bsrnet", "realesrnet") else key
        return float(loc.loc[k, "median_ms"]) if k in loc.index else cost(key, "median_ms")

    fp, sp = cost("fbcnn_span_ch48", "params"), cost("span_ch48", "params")
    fl, sl, bl = cost("fbcnn_span_ch48", "median_ms"), cost("span_ch48", "median_ms"), ms_of("bsrnet")
    put("RevFbParams", num(fp / 1e6, 1) if fp else None, why)
    put("RevFbParamsRatio", int(round(fp / sp)) if fp and sp else None, why)
    put("RevFbLat", num(fl, 2) if fl else None, why)
    put("RevSpanLat", num(sl, 2) if sl else None, why)
    put("RevFbLatRatio", num(fl / sl, 1) if fl and sl else None, why)
    put("RevBsrLat", num(bl, 1) if bl else None, why)
    rl, sl1 = ms_of("rrdb_psnr"), ms_of("span_ch48")   # cùng lượt đo với Bảng 1
    put("RevBsrLatRatio", int(round(rl / sl1)) if rl and sl1 else None, why)
    put("RevRrdbLatRatio", int(round(rl / sl1)) if rl and sl1 else None, why)
    put("RevFbcnnParams", num((fp - sp) / 1e6, 1) if fp and sp else None, why)

    rows = []
    for arm_, label, key in REV_ROWS:
        # RRDB tinh chỉnh chỉ vào bảng khi đã đủ fold như SPAN nhánh est (trung bình trên cùng tập ảnh của AMI)
        if arm_ == "rrdb/est" and (not have or len(cellp("ami", arm_, 144, "bic")) != len(cellp("ami", EST, 144, "bic"))):
            continue
        par = cost(key, "params") if key else None
        if arm_ == "realesrnet (published)":
            par = cost("bsrnet", "params")               # cùng kiến trúc RRDB với BSRNet (số tham số đã kiểm khi nạp)
        if arm_ == "realesrnet (published)":
            key_ms = "realesrnet"
        else:
            key_ms = key or ("bicubic" if arm_ == "bicubic" else None)
        ms = ms_of(key_ms) if key_ms else None
        cells = [num(par / 1e6, 2) if par else "--", num(ms, 2) if ms else "--"]
        for name, tier, kind in (("ami", 144, "bic"), ("ami", 144, "bicjpeg93"), ("ami", 144, "bicjpeg75"),
                                 ("earvn", 96, "bicjpeg75"), ("awex", 96, "bicjpeg75")):
            c = cellp(name, arm_, tier, kind) if have else None
            cells.append(num(float(c.psnr_y.mean())) if c is not None and len(c) else "\\todo{--}")
        cells += [r1("resnet18", arm_) or "\\todo{--}", r1("resnet50", arm_) or "\\todo{--}"]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if arm_ in ("rrdb_psnr (published)", "realesrnet (published)", "fbcnn_span_ch48 (published)"):
            rows.append("\\midrule")
    write(out / "tab_review.tex", "\n".join(rows) + "\n")


def fid_recog(res: Path, out: Path, fig: Path) -> None:
    """Độ trung thực trên ảnh EarVN1.0 mô phỏng (ảnh vào 'est', 24 px) so với hạng 1 trên ảnh dò thật, mọi phương pháp có cả
    hai số: tương quan hạng theo ba nhóm (tất cả; bỏ GAN; chỉ các mô hình L1 do bài tinh chỉnh). Kèm hình đánh đổi độ trễ CPU
    điện thoại và hạng 1. Phân tích làm sau khi có kết quả (phản biện gợi ý), nên trong bài gọi là mô tả."""
    from scipy.stats import spearmanr

    why = "fidrecog"
    dirs = [res / d for d in ("t2_earvn", "n2_earvn", "rev_earvn")]
    fs = [f for d in dirs if d.is_dir() for f in d.glob("*__hr96_x4_est.csv")]
    S = {}
    for tag in ("resnet18", "resnet50"):
        parts = [res / "recog" / f"{p_}{tag}_summary" / "summary.csv" for p_ in ("", "rev_")]
        if all(x.exists() for x in parts):
            a_, b_ = (pd.read_csv(x).set_index("arm").rank1 for x in parts)
            S[tag] = pd.concat([a_, b_[~b_.index.isin(a_.index)]])
    keys = [f"Corr{m}{g}{r}" for m in ("Psnr", "Lpips", "Dists") for g in ("All", "Nogan", "Lone") for r in ("", "Rf")] + \
           ["CorrNAll", "CorrNNogan", "CorrNLone"]
    if not fs or len(S) < 2:
        for k in keys:
            put(k, None, why)
        write(out / "tab_fidrecog.tex", "\\todo{--} \\\\\n")
        return
    d = pd.concat([pd.read_csv(f, dtype={"subject": str}) for f in fs], ignore_index=True)
    d["arm"] = d.model.map(lambda m_: arm_of(m_)[0])
    F = d.groupby(["arm", "key"])[["psnr_y", "lpips", "dists"]].mean().groupby("arm").mean()
    arms = [a_ for a_ in F.index if all(a_ in S[t].index for t in S) and a_ not in ("ref_large", "direct")]
    gan = {"bsrgan (published)", "realesrgan (published)", "esrgan (published)", "realesr_compact (published)"}
    groups = (("All", "All methods", arms), ("Nogan", "Without GAN-trained models", [a_ for a_ in arms if a_ not in gan]),
              ("Lone", "L1-trained models fine-tuned in this study", [a_ for a_ in arms if "/" in a_]))
    rows = []
    for g, label, sel in groups:
        put(f"CorrN{g}", len(sel))
        cells = []
        for m, col in (("Psnr", "psnr_y"), ("Lpips", "lpips"), ("Dists", "dists")):
            for r, tag in (("", "resnet18"), ("Rf", "resnet50")):
                t = spearmanr(F.loc[sel, col], S[tag].loc[sel])
                put(f"Corr{m}{g}{r}", num(t.statistic, 2, True))
                cells.append(f"{num(t.statistic, 2, True)} {{\\scriptsize({'$<$0.001' if t.pvalue < 0.001 else f'{t.pvalue:.3f}'})}}")
        rows.append(f"{label} & {len(sel)} & " + " & ".join(cells) + " \\\\")
    write(out / "tab_fidrecog.tex", "\n".join(rows) + "\n")

    # hình đánh đổi: độ trễ CPU điện thoại (16 bit; FBCNN rồi SPAN ở 32 bit) và hạng 1, hai bộ nhận dạng
    lf = res / "latency_ios.csv"
    if not lf.exists():
        return
    L = pd.read_csv(lf)
    L = L[(L.compute_units == "cpu") & pd.to_numeric(L.median_ms, errors="coerce").notna()].set_index("name").median_ms.astype(float)
    # mỗi phương pháp một ký hiệu riêng (màu theo nhóm), chú giải chung bên dưới thay cho nhãn cạnh điểm
    pts = [("SPAN-S, published", PUB, "span_ch48", "0.55", "o"), ("DISP, published", "disp26 (published)", "disp26", "0.55", "s"),
           ("RRDB, published", "rrdb_psnr (published)", "rrdb_psnr", "0.55", "D"),
           ("SPAN-S, ours", "span/est", "span_ch48", "C3", "o"), ("DISP, ours", "disp26/est", "disp26", "C3", "s"),
           ("RRDB, our training", "rrdb/est", "rrdb_psnr", "C3", "D"),
           ("BSRNet", "bsrnet (published)", "rrdb_psnr", "C0", "^"), ("Real-ESRNet", "realesrnet (published)", "rrdb_psnr", "C0", "v"),
           ("BSRGAN", "bsrgan (published)", "bsrgan", "C2", "P"),
           ("FBCNN + SPAN-S (32-bit)", "fbcnn_span_ch48 (published)", "fbcnn_span_ch48_fp32", "C1", "X")]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, tag, title in zip(axes, ("resnet18", "resnet50"), ("Ear-R18", "Ear-R50")):
        for lab, arm_, key, c, mk in pts:
            if arm_ not in S[tag].index or key not in L.index:
                continue
            if key.endswith("_fp32"):   # 32 bit: ký hiệu rỗng và ghi ngay cạnh điểm (các điểm khác là 16 bit)
                ax.scatter(L[key], S[tag][arm_], s=30, marker=mk, zorder=3, label=lab, facecolors="none",
                           edgecolors=c, linewidths=1.0)
                ax.annotate("32-bit", (L[key], S[tag][arm_]), xytext=(4, -2), textcoords="offset points",
                            fontsize=5.5, color=c, va="top")
            else:
                ax.scatter(L[key], S[tag][arm_], s=30, color=c, marker=mk, zorder=3, label=lab, edgecolors="none")
        ax.axhline(S[tag]["bicubic"], color="0.3", lw=0.8, ls=":")
        ax.axvline(33, color="0.3", lw=0.8, ls="--")
        ax.text(33, ax.get_ylim()[1] if False else 0.98, " 33 ms", transform=ax.get_xaxis_transform(), fontsize=6, va="top")
        ax.text(1.6, S[tag]["bicubic"], "bicubic", fontsize=6, va="bottom", color="0.3")
        ax.set_xscale("log")
        ax.set_xlim(1.5, 250)
        ax.set_xlabel("CPU latency on the smartphone (ms)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Rank-1 on real EarVN1.0 probes (%)")
    h, lab_ = axes[0].get_legend_handles_labels()
    f.legend(h, lab_, loc="lower center", ncol=5, frameon=False, fontsize=6.2)
    f.tight_layout(rect=(0, 0.13, 1, 1))
    f.savefig(fig / "fig_tradeoff.png", dpi=600)
    plt.close(f)


def gt_quality(res: Path, bench: Path = Path("data/bench"), raw: Path = Path("data/raw/EarVN1.0")) -> None:
    """Mức JPEG còn lại trong ảnh gốc dùng làm đáp án của EarVN1.0 (phần giới hạn). Tính một lần từ ảnh gốc rồi lưu ở
    results/gt_quality.json, để máy không có dữ liệu vẫn dựng được bài."""
    f = res / "gt_quality.json"
    if not f.exists() and (bench / "earvn" / "manifest.csv").exists():
        import csv
        from earsr.degrade.fit_estimated import jpeg_quality_from_file
        rows = [r for r in csv.DictReader(open(bench / "earvn" / "manifest.csv")) if r["tier"] == "96"]
        q = [jpeg_quality_from_file(raw / r["src"]) for r in rows]
        q = [x for x in q if x]
        f.write_text(json.dumps({"earvn_gt_sources": {"n": len(q), "min": min(q), "max": max(q), "median": float(np.median(q))}}, indent=1))
    d = json.loads(f.read_text())["earvn_gt_sources"] if f.exists() else None
    put("GtEarvnQMin", d["min"] if d else None, "gt")
    put("GtEarvnQMax", d["max"] if d else None, "gt")
    put("GtEarvnQMed", int(d["median"]) if d else None, "gt")
    # độ khối 8×8 còn sót ở đáp án AMI (phản biện 04/11): ảnh gốc JPEG 90 là đối chứng dương
    b = json.loads(f.read_text()).get("ami_blockiness") if f.exists() else None
    for key, mac in (("ami_originals_jpeg90", "BlockAmiRaw"), ("ami_gt_hr96", "BlockAmiGtS"),
                     ("ami_gt_hr144", "BlockAmiGtM"), ("ami_lr36_bicjpeg75", "BlockAmiLrQLow"),
                     ("ami_lr36_bic", "BlockAmiLrClean")):
        put(mac, num(b[key]["median"], 2) if b else None, "gt")


def mechanism(res: Path, fig: Path) -> None:
    """Phần dư nén theo vị trí trong khối 8×8 và theo tần số (scripts/analyze_blocks.py; TODO.md, việc O1)."""
    why = "mechanism"
    keys = ["MechBoundQLow", "MechBoundQHigh", "MechBoundShare", "MechBoundFrac", "MechBoundDivQLow",
            "MechNearNyq", "MechAboveNyq", "MechAboveNyqEc", "MechNearNyqDiv", "MechAboveNyqDiv"]
    F = {st: res / "mechanism" / f"blocks_{st}_hr144.json" for st in ("ami", "div2k")}
    if not all(f.exists() for f in F.values()):
        for k in keys:
            put(k, None, why)
        return
    out = {}
    for st, f in F.items():
        d = json.loads(f.read_text())
        frac, E = np.array(d["pos_frac"]), np.array(d["freq_edges"])
        B = d["models"]["bicubic"]
        r = {"frac": frac, "edges": E}
        for q in ("bicjpeg93", "bicjpeg75"):
            ec_pos = np.array(B[q]["pos"]) - np.array(B["bic"]["pos"])
            ec_f = np.array(B[q]["freq"]) - np.array(B["bic"]["freq"])
            pos, frq = [], []
            for m in PSNR16:
                M = d["models"][m]
                ex_pos = np.array(M[q]["pos"]) - np.array(M["bic"]["pos"]) - ec_pos
                ex_f = np.array(M[q]["freq"]) - np.array(M["bic"]["freq"]) - ec_f
                tot = ex_f.sum()
                pos.append((ex_pos / frac) / tot)       # phần dư mỗi điểm ảnh, chia cho trung bình toàn ảnh
                frq.append(ex_f / tot)                  # tỉ lệ phần dư trong từng vành
            r[q] = {"pos": np.array(pos), "freq": np.array(frq), "ec_freq": ec_f / ec_f.sum()}
        out[st] = r
    nyq = 0.125   # tần số Nyquist của ảnh vào ×4, theo chu kỳ trên điểm ảnh HR
    E = out["ami"]["edges"]
    lo_e, hi_e = E[:-1], E[1:]
    near = (lo_e >= 0.6 * nyq - 1e-9) & (hi_e <= 1.4 * nyq + 1e-9)
    above = lo_e >= nyq - 1e-9

    def bound(st, q):   # tỉ số phần dư mỗi điểm ảnh: sát biên khối so với phần còn lại, trung vị qua mô hình
        r = out[st]
        p = r[q]["pos"] * r["frac"]
        return np.median(p[:, 0] / r["frac"][0] / (p[:, 1:].sum(1) / r["frac"][1:].sum()))
    a75 = out["ami"]["bicjpeg75"]
    put("MechBoundQLow", num(bound("ami", "bicjpeg75"), 2))
    put("MechBoundQHigh", num(bound("ami", "bicjpeg93"), 2))
    put("MechBoundDivQLow", num(bound("div2k", "bicjpeg75"), 2))
    put("MechBoundShare", int(round(100 * np.median(a75["pos"][:, 0] * out["ami"]["frac"][0]))))
    put("MechBoundFrac", int(round(100 * out["ami"]["frac"][0])))
    put("MechNearNyq", int(round(100 * np.median(a75["freq"][:, near].sum(1)))))
    put("MechAboveNyq", int(round(100 * np.median(a75["freq"][:, above].sum(1)))))
    put("MechAboveNyqEc", int(round(100 * a75["ec_freq"][above].sum())))
    d75 = out["div2k"]["bicjpeg75"]
    put("MechNearNyqDiv", int(round(100 * np.median(d75["freq"][:, near].sum(1)))))
    put("MechAboveNyqDiv", int(round(100 * np.median(d75["freq"][:, above].sum(1)))))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"ami": "C3", "div2k": "0.35"}
    lab = {"ami": "AMI", "div2k": "DIV2K"}
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for st in ("ami", "div2k"):
        for q, ls in (("bicjpeg75", "-"), ("bicjpeg93", "--")):
            y = np.median(out[st][q]["pos"], 0)
            ax1.plot(range(4), y, color=col[st], ls=ls, lw=1.4, marker="o", ms=3.5,
                     label=f"{lab[st]}, JPEG {q[-2:]}")
    ax1.axhline(1, color="0.6", lw=0.7, ls=":")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(["0\n(edge)", "1", "2", "3\n(centre)"])
    ax1.set_xlabel("Distance to the nearest block boundary (input px)", fontsize=7)
    ax1.set_ylabel("Excess per pixel / image mean", fontsize=7)
    ax1.legend(fontsize=6, frameon=False, handlelength=3)
    ax1.set_title("(a) Position in the 8$\\times$8 block", fontsize=8)
    c = (lo_e + hi_e) / 2 / nyq
    w = (hi_e - lo_e) / nyq
    for st in ("ami", "div2k"):
        ax2.plot(c, np.median(out[st]["bicjpeg75"]["freq"], 0) / w, color=col[st], lw=1.4, label=f"{lab[st]}: model excess")
    ax2.plot(c, a75["ec_freq"] / w, color=col["ami"], lw=0.9, ls=":", label="AMI: compression error\nof interpolation")
    ax2.axvline(1, color="0.3", lw=0.8, ls="--")
    ax2.text(1.02, 0.97, "input Nyquist", transform=ax2.get_xaxis_transform(), fontsize=6, va="top")
    ax2.set_xlim(0, 3)
    ax2.set_xlabel("Spatial frequency / Nyquist frequency of the input", fontsize=7)
    ax2.set_ylabel("Share of the error per unit frequency", fontsize=7)
    ax2.legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax2.set_title("(b) Frequency, JPEG 75", fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    f.tight_layout()
    f.savefig(fig / "fig_mechanism.png", dpi=600)
    plt.close(f)


UP_EST = [f"N2_span-zero+xearvn_pub_rand_x4_hrall_est_f{k}" for k in range(1, 6)]


def recog_upscaled(res: Path, out: Path) -> None:
    """Bộ nhận dạng học thêm ảnh nhỏ thật đã phóng bằng đúng phương pháp được chấm (scripts/run_recog_upscaled.sh;
    TODO.md, việc O4). So với ResNet-18 chỉ học ảnh lớn, trên cùng ảnh dò, cùng lần chạy SPAN (5 fold, gộp như bài)."""
    why = "recog-up"
    meth = (("bicubic", ["bicubic"]), ("span_ch48", ["span_ch48"]), ("est", UP_EST))
    sets = (("earvn", "resnet18", "Earvn", "EarVN1.0"), ("awex", "awex_resnet18", "Awex", "AWEx"))
    names = [f"RecUp{t}{k}" for _, _, t, _ in sets for k in ("Bic", "Pub", "Est", "Gain", "GainLo", "GainHi", "Ref",
                                                             "LgBic", "LgEst", "LgGain")]

    def hits(fs):   # trung bình qua các mô hình fold, theo từng ảnh dò (như summarize_recog.py)
        ts = [pd.read_csv(f, dtype={"subject": str}) for f in fs]
        return np.mean([(t["rank"].to_numpy() == 1).astype(float) for t in ts], 0), ts[0]["subject"].to_numpy()
    rows = []
    for key, base, tag, label in sets:
        up = {m: [res / "recog_up" / f"{key}_{m}" / f"{c}.csv" for c in cs] for m, cs in meth}
        lg = {m: [res / "recog" / base / f"{c}.csv" for c in cs] for m, cs in meth}
        if not all(f.exists() for fs in list(up.values()) + list(lg.values()) for f in fs):
            for n_ in names:
                if n_.startswith(f"RecUp{tag}"):
                    put(n_, None, why)
            rows.append(f"{label} & \\todo{{--}} & & & & \\\\")
            continue
        H = {("up", m): hits(f) for m, f in up.items()} | {("lg", m): hits(f) for m, f in lg.items()}
        # giao thức chính (bộ nhận dạng riêng cho mỗi cách phóng) đứng trước; góp ý reviewer 06/11
        for kind_, lab_ in (("up", "large and upscaled small images"), ("lg", "large images only")):
            a, subj = H[(kind_, "est")]
            b, _ = H[(kind_, "bicubic")]
            if kind_ == "lg":   # cùng ước lượng in ở Bảng 8 (summarize_recog.py, 1000 mẫu, seed 0): đọc lại, không tính lại
                r_ = pd.read_csv(res / "recog" / f"{base}_summary" / "paired.csv").set_index(["arm", "minus"]).loc[
                    ("span/est", "bicubic")]
                e = SimpleNamespace(point=r_["d_rank1"], lo=r_["d_rank1_lo"], hi=r_["d_rank1_hi"])
            else:
                e = paired_diff(100 * a, 100 * b, subj, n_boot=1000)
            r1 = {m: 100 * H[(kind_, m)][0].mean() for m, _ in meth}
            rows.append(f"{label} & {lab_} & {r1['bicubic']:.1f} & {r1['span_ch48']:.1f} & {r1['est']:.1f} & "
                        f"{est_cell(e, 1)} \\\\")
            if kind_ == "up":
                put(f"RecUp{tag}Bic", f"{r1['bicubic']:.1f}")
                put(f"RecUp{tag}Pub", f"{r1['span_ch48']:.1f}")
                put(f"RecUp{tag}Est", f"{r1['est']:.1f}")
                put(f"RecUp{tag}Gain", num(e.point, 1))
                put(f"RecUp{tag}GainLo", num(e.lo, 1))
                put(f"RecUp{tag}GainHi", num(e.hi, 1))
                if tag == "Earvn" and not e.lo > 0:
                    print("CHÚ Ý: mức tăng rank-1 theo giao thức chính trên EarVN1.0 không còn có ý nghĩa; sửa tóm tắt, mở đầu, mục 6.4")
            else:
                put(f"RecUp{tag}LgBic", f"{r1['bicubic']:.1f}")
                put(f"RecUp{tag}LgEst", f"{r1['est']:.1f}")
                put(f"RecUp{tag}LgGain", num(e.point, 1))
        ref = res / "recog_up" / f"{key}_bicubic" / "ref_large.csv"
        put(f"RecUp{tag}Ref", f"{100 * hits([ref])[0].mean():.1f}" if ref.exists() else None, why)
    write(out / "tab_recogup.tex", "\n".join(rows) + "\n")
    lg_ = res / "recog_up" / "train_log_bicubic.json"
    put("RecUpNSmall", f"{json.loads(lg_.read_text())['info']['n_small_upscaled']:,}" if lg_.exists() else None, why)



UP_RRDB = [f"REV_rrdb-zero+xearvn_pub_rand_x4_hrall_est_f{k}" for k in range(1, 6)]
UP_ALL = (("bicubic", ["bicubic"], "Bicubic", "Bic"), ("span_ch48", ["span_ch48"], "SPAN-S, published", "Pub"),
          ("est", UP_EST, "SPAN-S, measured (ours)", "Est"), ("rrdb_est", UP_RRDB, "RRDB, measured", "Rrdb"),
          ("bsrnet", ["bsrnet"], "BSRNet", "Bsr"), ("realesrnet", ["realesrnet"], "Real-ESRNet", "Resr"),
          ("fbcnn_span_ch48", ["fbcnn_span_ch48"], "FBCNN, then SPAN-S published", "Fb"))


def recog_upscaled_all(res: Path, out: Path) -> None:
    """Giao thức chính (mỗi cách phóng một bộ nhận dạng học thêm ảnh nhỏ thật đã phóng bằng chính nó) cho cả bốn cách
    phóng có tính đến nén (phản biện thứ tám, 06/11; TODO.md, việc V; scripts/run_recog_upscaled.sh với METHODS=...).
    Ghép cặp theo ảnh dò như recog_upscaled. Chưa có kết quả thì in một dòng và để macro TBD; bài chưa dùng bảng này."""
    why = "recog-up-all"
    rows, ok = [], True
    for key, tag, label in (("earvn", "Earvn", "EarVN1.0"), ("awex", "Awex", "AWEx")):
        F = {m: [res / "recog_up" / f"{key}_{m}" / f"{c}.csv" for c in cs] for m, cs, _, _ in UP_ALL}
        miss = [m for m, fs in F.items() if not all(f.exists() for f in fs)]
        if miss:
            ok = False
            for _, _, _, t in UP_ALL:
                for k in ("Rank", "VsBic", "VsEst"):
                    put(f"RecAll{tag}{t}{k}", None, why)
            continue
        H = {}
        for m, fs in F.items():
            ts = [pd.read_csv(f, dtype={"subject": str}) for f in fs]
            H[m] = (np.mean([(t["rank"].to_numpy() == 1).astype(float) for t in ts], 0), ts[0]["subject"].to_numpy())
        subj = H["bicubic"][1]
        for m, (h, sj) in H.items():
            if len(h) != len(subj) or not np.array_equal(sj, subj):
                raise SystemExit(f"recog_upscaled_all: ảnh dò của {m} không khớp với bicubic ({key})")
        for m, _, name, t in UP_ALL:
            r1 = 100 * H[m][0].mean()
            put(f"RecAll{tag}{t}Rank", f"{r1:.1f}")
            cells = [f"{r1:.1f}"]
            for ref_, k in (("bicubic", "VsBic"), ("est", "VsEst")):
                if m == ref_ or (m == "bicubic" and ref_ == "est"):
                    cells.append("--")
                    continue
                e = paired_diff(100 * H[m][0], 100 * H[ref_][0], subj, n_boot=1000)
                put(f"RecAll{tag}{t}{k}", f"{est_cell(e, 1)}")
                cells.append(est_cell(e, 1))
            rows.append((label, name, cells))
    if not ok:
        print("recog_upscaled_all: chưa đủ kết quả giao thức chính cho bốn cách phóng mới (TODO.md, việc V)")
        write(out / "tab_recogup_all.tex", "\\todo{--} \\\\\n")
        return
    by = {}
    for label, name, cells in rows:
        by.setdefault(name, {})[label] = cells
    lines = [f"{n} & " + " & ".join(by[n]["EarVN1.0"]) + " & " + " & ".join(by[n]["AWEx"][:2]) + " \\\\"
             for _, _, n, _ in UP_ALL]
    lines.insert(3, "\\midrule")
    write(out / "tab_recogup_all.tex", "\n".join(lines) + "\n")


ORDER_KINDS = [("bicjpeg75", "Downsize, then JPEG 75 (as in the paper)"), ("bicjpeg75c444", "Same, 4:4:4 chroma"),
               ("shiftjpeg75", "Same, block grid offset 1--7 px"), ("dbljpeg93to75", "JPEG 93 with offset grid, then 75"),
               ("jpeg75bic", "JPEG 75, then downsize"), ("bicjpeg93", "Downsize, then JPEG 93 (as in the paper)"),
               ("shiftjpeg93", "Same, block grid offset 1--7 px")]


def order_control(res: Path, out: Path, folds_path: Path = Path("splits/ami_5fold.json")) -> None:
    """Đối chứng thứ tự suy giảm (scripts/run_order_control.sh; TODO.md, việc O2): AMI, 36 px."""
    why = "order"
    names = ["OrdShiftBelow", "OrdShiftGapMin", "OrdShiftGapMax", "OrdDblBelow", "OrdDblGapMin", "OrdDblGapMax",
             "OrdChromaMax", "OrdPreGainMin", "OrdPreGainMax", "OrdShiftHighBelow", "OrdEstShiftMin", "OrdEstShiftMax",
             "OrdShiftDeltaMax", "OrdEstDblVsBic"]
    d = load_dir(res / "order")
    if d is None or not {"bicubic"} <= set(d.arm):
        for n_ in names:
            put(n_, None, why)
        write(out / "tab_order.tex", "\\todo{--} \\\\\n")
        return
    pubs = [m + PUBLISHED for m in PSNR16]
    G = {}   # mức lợi so với bicubic của từng mô hình công bố, mọi ảnh
    for k, _ in ORDER_KINDS:
        b = d[(d.arm == "bicubic") & (d.degrade == k)].set_index("key").psnr_y
        G[k] = {m: float((d[(d.arm == m) & (d.degrade == k)].set_index("key").psnr_y - b).mean()) for m in pubs
                if ((d.arm == m) & (d.degrade == k)).any()}
    have = all(len(G[k]) == 16 for k, _ in ORDER_KINDS)
    est = "span/est" in set(d.arm)
    P = pool(d, folds_path)[0] if est else None
    rows = []
    for k, lab in ORDER_KINDS:
        g = np.array(list(G[k].values()))
        bic = float(d[(d.arm == "bicubic") & (d.degrade == k)].psnr_y.mean())
        e1 = paired(P, "span/est", "bicubic", 144, k) if est else None
        e2 = paired(P, "span/est", PUB, 144, k) if est else None
        rows.append(f"{lab} & {bic:.2f} & {num(float(np.median(g)), 2, True)} & {int((g < 0).sum())} & "
                    f"{est_cell(e1)} & {est_cell(e2)} \\\\")
    write(out / "tab_order.tex", "\n".join(rows) + "\n")
    if not have:
        for n_ in names:
            put(n_, None, why)
        return

    def gaps(k):
        return np.array([G[k][m] for m in pubs])
    put("OrdShiftBelow", int((gaps("shiftjpeg75") < 0).sum()))
    put("OrdShiftGapMin", num(float(-gaps("shiftjpeg75").max()), 2))
    put("OrdShiftGapMax", num(float(-gaps("shiftjpeg75").min()), 2))
    put("OrdDblBelow", int((gaps("dbljpeg93to75") < 0).sum()))
    put("OrdDblGapMin", num(float(-gaps("dbljpeg93to75").max()), 2))
    put("OrdDblGapMax", num(float(-gaps("dbljpeg93to75").min()), 2))
    put("OrdChromaMax", num(float(np.abs(gaps("bicjpeg75c444") - gaps("bicjpeg75")).max()), 2))
    put("OrdShiftDeltaMax", num(float(np.abs(gaps("shiftjpeg75") - gaps("bicjpeg75")).max()), 2))
    put("OrdPreGainMin", num(float(gaps("jpeg75bic").min()), 2))
    put("OrdPreGainMax", num(float(gaps("jpeg75bic").max()), 2))
    put("OrdShiftHighBelow", int((gaps("shiftjpeg93") < 0).sum()))
    if est:
        rng_put("OrdEstShift", [paired(P, "span/est", "bicubic", 144, k) for k in ("shiftjpeg75", "shiftjpeg93")])
        e = paired(P, "span/est", "bicubic", 144, "dbljpeg93to75")
        put("OrdEstDblVsBic", num(e.point, 2) if e else None, why)
        e = paired(P, "span/est", PUB, 144, "jpeg75bic")
        put("OrdEstPreVsPub", num(-e.point, 2) if e else None, why)   # mức kém SPAN công bố khi ảnh vào gần sạch
    else:
        for n_ in ("OrdEstShiftMin", "OrdEstShiftMax", "OrdEstDblVsBic", "OrdEstPreVsPub"):
            put(n_, None, why)


def selector(res: Path) -> None:
    """Luật chọn lúc chạy giữa bicubic và SPAN công bố (scripts/analyze_selector.py; TODO.md, việc N)."""
    why = "selector"   # chưa chạy scripts/analyze_selector.py
    f = res / "selector" / "comparisons_span_ch48.json"
    k = json.loads(f.read_text()) if f.exists() else None
    if k is None:
        for n_ in ("SelRhoMin", "SelRhoMax", "SelCrossMin", "SelCrossMax", "SelSame", "SelBack", "SelGainMax"):
            put(n_, None, why)
        return
    rho = [v["rhat_vs_ratio"] for v in k["corr"].values()]
    put("SelRhoMin", num(min(rho)))
    put("SelRhoMax", num(max(rho)))
    sc = k["scenarios"]
    cross = [sc["DIV2K"]["comparisons"][ds]["ratio-jpeg"] for ds in ("AMI", "EarVN1.0", "AWEx")]
    if not all(c["lo"] > 0 for c in cross):
        print("CHÚ Ý: luật tỉ số không còn hơn luật mức JPEG có ý nghĩa trên mọi bộ ảnh tai; sửa đoạn ở mục 4.3")
    put("SelCrossMin", num(min(c["point"] for c in cross)))
    put("SelCrossMax", num(max(c["point"] for c in cross)))
    e = sc["AMI"]["comparisons"]["AMI"]["ratio-jpeg"]
    put("SelSame", f"{num(e['point'], 3, True)} [{num(e['lo'], 3)}, {num(e['hi'], 3)}]")
    e = sc["AMI"]["comparisons"]["DIV2K"]["ratio-jpeg"]
    put("SelBack", num(e["point"]))
    put("SelGainMax", num(max(c["ratio-model"]["point"] for s_ in sc.values() for c in s_["comparisons"].values())))
    lv = [str(q) for q in sc["DIV2K"]["jpeg_levels_model"] if q < 93]
    put("SelLevelsDiv", ", ".join(lv[:-1]) + " and " + lv[-1] if len(lv) > 1 else (lv[0] if lv else "none"))


# ------------------------------------------------------------------ hình
def figures(pub: dict, tr: dict, fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
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
    f.savefig(fig / "fig_jpeg_sweep.png", dpi=600)
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
    f.savefig(fig / "fig_size.png", dpi=600)
    plt.close(f)


QUAL = [("bicubic", "Bicubic"), ("span_ch48", "SPAN-S\npublished"), ("bsrgan", "BSRGAN"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_generic_f2", "SPAN-S\ngeneric"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_bicjpeg75_f2", "SPAN-S\nJPEG 75"),
        ("N2_span-zero+xearvn_pub_rand_x4_hrall_est_f2", "SPAN-S\nmeasured")]


def qualitative(res: Path, fig: Path, raw: Path) -> bool:
    """Ảnh nhỏ thật (cột đầu, lặp điểm ảnh ×4) và ảnh đã phóng ×4 bằng sáu phương pháp.

    Trả về False nếu chưa có ảnh đã phóng (results/recog/sr2) hoặc không thấy ảnh gốc trong `raw`."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # sr2: 12 ảnh rải trên nhiều người (mã từ 07/10); sr: 16 ảnh đầu, đều của một người (mã cũ).
    # Không lấy ảnh 0 của sr2: ảnh 40×146 px, phần cắt giữa không còn thấy vành tai.
    root, picks = (res / "recog" / "sr2", (1, 3, 6, 9)) if (res / "recog" / "sr2" / "bicubic").is_dir() else (res / "recog" / "sr", (3, 4, 9, 14))
    if not all((root / m).is_dir() for m, _ in QUAL):
        return False
    names = sorted(p.name for p in (root / QUAL[0][0]).glob("*.png"))
    names = [names[i] for i in picks if i < len(names)]
    if not names:
        return False
    cols = [(None, "Input")] + QUAL
    f, axes = plt.subplots(len(names), len(cols), figsize=(6.6, 1.25 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        subject, stem = n.rsplit(".", 1)[0].split("__", 1)
        src = sorted((raw / subject).glob(stem + ".*"))
        if not src:
            print(f"hình minh họa: không thấy ảnh gốc {raw / subject / stem}.*")
            plt.close(f)
            return False
        for j, (m, label) in enumerate(cols):
            # cột đầu: ảnh gốc, mỗi điểm ảnh lặp 4×4 để cùng cỡ với các cột sau mà không nội suy
            img = cv2.imread(str(root / m / n)) if m else cv2.imread(str(src[0])).repeat(4, axis=0).repeat(4, axis=1)
            h, w = img.shape[:2]
            c = img[h // 2 - min(h, int(1.25 * w)) // 2: h // 2 + min(h, int(1.25 * w)) // 2]   # cắt giữa, bỏ bớt nền
            axes[i, j].imshow(c[:, :, ::-1], interpolation="nearest")
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(label, fontsize=7)
    f.tight_layout(pad=0.2)
    f.savefig(fig / "fig_qualitative.png", dpi=600)
    plt.close(f)
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--folds", default="splits/ami_5fold.json")
    ap.add_argument("--degrade-params", default="configs/degrade_estimated.json")
    ap.add_argument("--earvn-raw", default="data/raw/EarVN1.0", help="ảnh gốc EarVN1.0, cho cột đầu của hình minh họa")
    a = ap.parse_args(argv)
    res, out, fig = Path(a.results), Path(a.paper) / "generated", Path(a.paper) / "figures"
    pub = published(res, out)
    tr = trained(res, out, Path(a.folds))
    realism_and_params(res, out, Path(a.degrade_params))
    recognition(res, out)
    recognition_extra(res, out)
    recog_synth(res, out)
    latency_ios(res, out, pub["lat"])
    natural_control(res, out)
    error_ratio(res, out, fig)
    codec_control(res, out)
    seeds(res, Path(a.folds))
    headroom(res)
    review(res, out)
    selector(res)
    gt_quality(res)
    order_control(res, out)
    mechanism(res, fig)
    recog_upscaled(res, out)
    recog_upscaled_all(res, out)
    fid_recog(res, out, fig)
    figures(pub, tr, fig)
    write(out / "fig_qualitative.tex", "\\includegraphics[width=\\linewidth]{figures/fig_qualitative.png}\n" if qualitative(res, fig, Path(a.earvn_raw))
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
