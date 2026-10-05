"""Khảo sát người xem (P9): tạo một trang HTML tự chứa và phân tích phiếu trả lời.

Hai phần (mục 1.9 của kế hoạch):
- 'ref'   : có đáp án ở giữa; câu hỏi "ảnh nào giống ảnh ở giữa hơn" (độ trung thực).
- 'noref' : không có đáp án (ảnh nhỏ thật); câu hỏi "ảnh nào rõ hơn" (sở thích).

Người xem không biết mô hình: trang HTML chỉ chứa mã thử ngẫu nhiên; bảng giải
mã (mã thử -> mô hình bên trái, bên phải) nằm ở file riêng, KHÔNG gửi cho người
xem. Thứ tự câu hỏi được xáo riêng cho từng người xem (trong trình duyệt); vị trí
trái phải được gieo ngẫu nhiên lúc tạo trang.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#777;color:#fff;text-align:center}
#top{padding:10px 16px;background:#333;font-size:15px}
#q{font-size:20px;margin:14px 0 6px}
#row{display:flex;justify-content:center;align-items:flex-start;gap:24px;flex-wrap:wrap;padding:8px}
figure{margin:0}
img{image-rendering:pixelated;image-rendering:crisp-edges;background:#777;display:block;border:3px solid transparent}
figure.pick img{cursor:pointer}
figure.pick img:hover{border-color:#ffd54a}
figcaption{font-size:14px;margin-top:4px}
button{font-size:16px;padding:8px 18px;margin:8px;cursor:pointer}
input{font-size:16px;padding:6px}
#end{display:none;padding:30px;font-size:18px}
</style></head><body>
<div id="top"><span id="prog"></span></div>
<div id="start" style="padding:30px;font-size:17px">
<p>__INTRO__</p>
<p>Mã người xem: <input id="vid" placeholder="ví dụ V01"></p>
<p>Bấm vào ảnh bạn chọn, hoặc dùng phím ← và →. Không có giới hạn thời gian. Giữ nguyên độ phóng của trình duyệt (100%).</p>
<button onclick="begin()">Bắt đầu</button></div>
<div id="main" style="display:none"><div id="q"></div><div id="row"></div></div>
<div id="end">Xong. Cảm ơn bạn.<br><button onclick="save()">Tải phiếu trả lời (.json)</button>
<p style="font-size:14px">Gửi file vừa tải cho người tổ chức.</p></div>
<script>
const TRIALS = __TRIALS__;
const STUDY = "__STUDY__";
const Q = {ref: "Ảnh nào GIỐNG ảnh ở giữa hơn?", noref: "Ảnh nào RÕ hơn?"};
let order = [], i = 0, ans = [], t0 = 0, viewer = "";
function shuffle(a){for(let k=a.length-1;k>0;k--){const j=Math.floor(Math.random()*(k+1));[a[k],a[j]]=[a[j],a[k]];}return a;}
function fig(src, cap, side){const f=document.createElement("figure");const im=document.createElement("img");
  im.src=src;im.width=TRIALS[order[i]].w;im.height=TRIALS[order[i]].h;f.appendChild(im);
  const c=document.createElement("figcaption");c.textContent=cap;f.appendChild(c);
  if(side){f.className="pick";im.onclick=()=>choose(side);}return f;}
function show(){const t=TRIALS[order[i]];document.getElementById("prog").textContent="Câu "+(i+1)+" / "+order.length;
  document.getElementById("q").textContent=Q[t.part];const r=document.getElementById("row");r.innerHTML="";
  r.appendChild(fig(t.left,"Trái (←)","L"));if(t.ref){r.appendChild(fig(t.ref,"Ảnh gốc",null));}
  r.appendChild(fig(t.right,"Phải (→)","R"));t0=performance.now();}
function choose(side){ans.push({trial:TRIALS[order[i]].id,choice:side,ms:Math.round(performance.now()-t0),pos:i});
  i++;if(i<order.length){show();}else{document.getElementById("main").style.display="none";
  document.getElementById("end").style.display="block";}}
function begin(){viewer=document.getElementById("vid").value.trim();if(!viewer){document.getElementById("vid").focus();return;}
  order=shuffle([...TRIALS.keys()]);document.getElementById("start").style.display="none";
  document.getElementById("main").style.display="block";show();}
document.addEventListener("keydown",e=>{if(document.getElementById("main").style.display==="none")return;
  if(e.key==="ArrowLeft")choose("L");if(e.key==="ArrowRight")choose("R");});
function save(){const blob=new Blob([JSON.stringify({study:STUDY,viewer:viewer,date:new Date().toISOString(),
  screen:[screen.width,screen.height,window.devicePixelRatio],answers:ans},null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="answers_"+STUDY+"_"+viewer+".json";a.click();}
</script></body></html>
"""


def _data_uri(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img[:, :, ::-1])
    if not ok:
        raise RuntimeError("không mã hóa được PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def build_study(items: list[dict], pairs: list[tuple[str, str]], out_html: str | Path, out_key: str | Path,
                study: str = "earsr", zoom: int = 2, seed: int = 0, title: str = "Khảo sát chất lượng ảnh",
                intro: str = "Mỗi câu hiện hai ảnh. Hãy chọn theo câu hỏi ở đầu trang.") -> pd.DataFrame:
    """``items``: mỗi phần tử {'key': mã ảnh, 'part': 'ref' | 'noref', 'ref': ảnh uint8 hoặc None,
    'outputs': {tên mô hình: ảnh uint8}}. ``pairs``: các cặp mô hình cần so.
    Ảnh hiển thị ở độ phóng nguyên ``zoom`` lần, không nội suy (image-rendering: pixelated).
    Trả về bảng giải mã (cũng được ghi ra ``out_key``).
    """
    rng = np.random.default_rng(seed)
    trials, key_rows = [], []
    for it in items:
        if it["part"] not in ("ref", "noref"):
            raise ValueError("part phải là 'ref' hoặc 'noref'")
        if it["part"] == "ref" and it.get("ref") is None:
            raise ValueError(f"{it['key']}: phần 'ref' cần ảnh đáp án")
        for a, b in pairs:
            if a not in it["outputs"] or b not in it["outputs"]:
                continue
            ia, ib = it["outputs"][a], it["outputs"][b]
            if ia.shape != ib.shape or (it.get("ref") is not None and it["ref"].shape != ia.shape):
                raise ValueError(f"{it['key']}: các ảnh của một câu phải cùng kích thước")
            swap = bool(rng.random() < 0.5)
            left, right = (b, a) if swap else (a, b)
            tid = hashlib.sha1(f"{study}|{seed}|{it['key']}|{a}|{b}".encode()).hexdigest()[:10]
            trials.append({"id": tid, "part": it["part"], "w": ia.shape[1] * zoom, "h": ia.shape[0] * zoom,
                           "left": _data_uri(it["outputs"][left]), "right": _data_uri(it["outputs"][right]),
                           "ref": _data_uri(it["ref"]) if it["part"] == "ref" else None})
            key_rows.append({"trial": tid, "image": it["key"], "subject": it.get("subject", ""), "part": it["part"],
                             "left": left, "right": right, "pair": f"{a}|{b}"})
    if not trials:
        raise ValueError("không tạo được câu hỏi nào (kiểm tra tên mô hình trong pairs)")
    html = (HTML.replace("__TRIALS__", json.dumps(trials)).replace("__STUDY__", study)
            .replace("__TITLE__", title).replace("__INTRO__", intro))
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html, encoding="utf-8")
    key = pd.DataFrame(key_rows)
    Path(out_key).parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(out_key, index=False)
    return key


def load_answers(files: list, key: pd.DataFrame) -> pd.DataFrame:
    """Nối các phiếu trả lời với bảng giải mã. Cột ``chose_a`` = 1 nếu người xem chọn
    mô hình đứng trước trong cặp 'a|b'."""
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for a in d["answers"]:
            rows.append({"viewer": d["viewer"], "trial": a["trial"], "choice": a["choice"], "ms": a.get("ms")})
    ans = pd.DataFrame(rows)
    if ans.empty:
        raise ValueError("không có câu trả lời nào")
    if ans.duplicated(["viewer", "trial"]).any():
        raise ValueError("một người xem trả lời một câu hai lần (trùng mã người xem?)")
    m = ans.merge(key, on="trial", how="left")
    if m.pair.isna().any():
        raise ValueError("có câu trả lời không khớp bảng giải mã (sai file key?)")
    a_model = m.pair.str.split("|").str[0]
    chosen = np.where(m.choice == "L", m.left, m.right)
    m["chose_a"] = (chosen == a_model).astype(int)
    return m


def analyze(m: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Với mỗi (phần, cặp): tỉ lệ chọn mô hình thứ nhất, khoảng tin cậy 95% bằng
    bootstrap theo NGƯỜI XEM (đơn vị độc lập), và kiểm định dấu trên tỉ lệ của
    từng người xem. Kèm tỉ lệ chọn bên trái (kiểm thiên lệch vị trí)."""
    from scipy.stats import binomtest

    rng = np.random.default_rng(seed)
    out = []
    for (part, pair), g in m.groupby(["part", "pair"]):
        per = g.groupby("viewer").chose_a.agg(["sum", "count"])
        k = len(per)
        idx = rng.integers(0, k, size=(n_boot, k))
        s, c = per["sum"].values, per["count"].values
        boots = s[idx].sum(1) / c[idx].sum(1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rate = per["sum"] / per["count"]
        n_pos, n_neg = int((rate > 0.5).sum()), int((rate < 0.5).sum())
        p_sign = binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if n_pos + n_neg else 1.0
        a, b = pair.split("|")
        out.append({"part": part, "model_a": a, "model_b": b, "n_viewers": k, "n_answers": int(c.sum()),
                    "n_images": int(g.image.nunique()), "pref_a": float(s.sum() / c.sum()), "lo": float(lo),
                    "hi": float(hi), "ci_excludes_half": bool(lo > 0.5 or hi < 0.5), "p_sign_viewers": float(p_sign),
                    "left_rate": float((g.choice == "L").mean())})
    return pd.DataFrame(out)


def metric_agreement(m: pd.DataFrame, metric_by_model: dict, lower_is_better: bool = True) -> dict:
    """Tỉ lệ ảnh mà một số đo (ví dụ LPIPS) và đa số người xem cùng chọn (mục 1.9).
    ``metric_by_model``: dict (image, model) -> giá trị số đo. Ảnh hòa phiếu bị bỏ."""
    agree = n = 0
    for (image, pair), g in m.groupby(["image", "pair"]):
        a, b = pair.split("|")
        if (image, a) not in metric_by_model or (image, b) not in metric_by_model:
            continue
        share = g.chose_a.mean()
        if share == 0.5:
            continue
        va, vb = metric_by_model[(image, a)], metric_by_model[(image, b)]
        if va == vb:
            continue
        metric_a = (va < vb) if lower_is_better else (va > vb)
        agree += int(metric_a == (share > 0.5))
        n += 1
    return {"n_pairs": n, "agreement": agree / n if n else float("nan")}
