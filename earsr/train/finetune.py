"""Dựng mô hình, dữ liệu và hàm validation cho một lần tinh chỉnh trên ảnh tai;
nạp lại mô hình từ thư mục của một lần chạy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.ami import read_manifest, scan_ami
from ..data.datasets import MixedDataset, SRTrainDataset
from ..data.splits import load_folds
from ..degrade.pipelines import DegradeParams
from ..device import GpuFirst
from ..eval.metrics import crop_border, psnr, rgb_to_y
from ..io import imread_rgb, to_tensor, to_uint8
from ..models.optional.heads import TwoHeadGated
from ..models.registry import SPAN_PUB_IMG_RANGE, SPECS, SRModel, load_state
from ..models.span import SPAN, Conv3XC
from ..models.padding import ALIAS as PAD_ALIAS
from ..models.padding import set_padding_mode
from ..models.variants import arch_part, build_span_variant, has_published_channels


def _load_ckpt_model(path: str | Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd["model"] if isinstance(sd, dict) and "model" in sd else sd


def build_trainable(backbone: str, variant: str, pretrain: str, scale: int,
                    init_ckpt: str | Path | None = None, wdir: Path | None = None) -> nn.Module:
    """Mô hình ở dạng huấn luyện, nhận và trả tensor [0, 1].

    backbone 'span': SPAN cấu hình được (``variant`` chọn kiểu đệm và hình dạng).
      pretrain 'pub' chỉ hợp lệ khi số kênh và số khối khớp trọng số công bố
      (48 kênh, 6 khối) và hệ số ×4. Kiểu đệm không có tham số, nên 'pub' đi được
      với kiểu đệm khác: đó là phép thử rẻ "trọng số công bố, đổi kiểu đệm, rồi
      tinh chỉnh" (mốc và biến thể cùng xuất phát từ một bộ trọng số).
    backbone khác: một mô hình trong kho, bọc trong ``SRModel`` (lo dải giá trị
      và bội số kích thước). ``variant`` chỉ nhận kiểu đệm (zero, replicate,
      reflect), áp cho mọi tích chập có đệm của mạng.
    """
    variant = arch_part(variant)
    if backbone == "span":
        net = build_span_variant(variant, scale=scale, deploy=False,
                                 img_range=SPAN_PUB_IMG_RANGE if pretrain == "pub" else 1.0)
        if pretrain == "pub":
            if not has_published_channels(variant) or scale != 4:
                raise ValueError("trọng số công bố chỉ dùng được cho thân 48 kênh, 6 khối ở ×4; "
                                 "hình dạng khác cần init_ckpt từ tiền huấn luyện")
            net.load_state_dict(load_state(SPECS["span_ch48"], wdir), strict=True)
        elif pretrain != "none":
            if init_ckpt is None:
                raise ValueError(f"pretrain='{pretrain}' cần init_ckpt")
            net.load_state_dict(_load_ckpt_model(init_ckpt), strict=True)
        return net
    if backbone not in SPECS:
        raise KeyError(f"không có backbone '{backbone}'")
    spec = SPECS[backbone]
    if spec.scale != scale:
        raise ValueError(f"{backbone} là mô hình ×{spec.scale}")
    if variant not in PAD_ALIAS:
        raise ValueError(f"backbone '{backbone}' chỉ nhận variant là kiểu đệm {sorted(PAD_ALIAS)}; nhận '{variant}'")
    net = spec.build()
    if pretrain == "pub":
        net.load_state_dict(load_state(spec, wdir), strict=True)
    if PAD_ALIAS[variant] != "zeros":
        if not spec.local_only:
            raise ValueError(f"'{backbone}' có phép toán không cục bộ; đổi kiểu đệm chưa được kiểm cho mô hình này")
        if set_padding_mode(net, variant) == 0:
            raise RuntimeError(f"không tìm thấy tích chập có đệm nào trong '{backbone}'")
    model = SRModel(net, spec)
    if pretrain not in ("pub", "none"):
        if init_ckpt is None:
            raise ValueError(f"pretrain='{pretrain}' cần init_ckpt")
        model.load_state_dict(_load_ckpt_model(init_ckpt), strict=True)
    # trọng số công bố ở dạng đã gộp (SPAN 28 và 26 kênh): mở gradient cho tích chập đã gộp
    for m in model.modules():
        if isinstance(m, Conv3XC) and m.deploy:
            m.unfreeze_deploy()
    return model


def build_n3(variant: str, scale: int, backbone_ckpt: str | Path | None = None, hidden: int = 16) -> TwoHeadGated:
    """Mô hình hai đầu và cổng trên thân SPAN. ``backbone_ckpt``: thân đã tinh chỉnh
    (bắt buộc với cấu hình A). Đầu kết cấu khởi tạo bằng bản sao của đầu trung thực."""
    net = build_span_variant(arch_part(variant), scale=scale, deploy=False)
    if backbone_ckpt is not None:
        net.load_state_dict(_load_ckpt_model(backbone_ckpt), strict=True)
    model = TwoHeadGated(net, hidden=hidden)
    model.head_t[0].load_state_dict(net.upsampler[0].state_dict())
    return model


def trainable_fraction(model: nn.Module) -> float:
    """Tỉ lệ tham số có gradient, không tính bản gộp ``eval_conv`` của dạng huấn luyện."""
    tot = tr = 0
    for n, p in model.named_parameters():
        if ".eval_conv." in n and not p.requires_grad:
            continue
        tot += p.numel()
        tr += p.numel() if p.requires_grad else 0
    return tr / max(1, tot)


def ami_train_paths(ami_raw: str | Path, folds_path: str | Path, fold: int, part: str = "train") -> list[Path]:
    subj = set(load_folds(folds_path)["folds"][str(fold)][part])
    return [it.path for it in scan_ami(ami_raw) if it.subject in subj]


def make_train_set(ami_raw, folds_path, fold: int, scale: int, protocol: str, tier: int, patch_lr: int,
                   degrade_kind: str = "bic", params: DegradeParams | None = None, seed: int = 0,
                   extra_paths: list | None = None, extra_min_downscale: float = 2.0,
                   extra_prob: float | None = None, landmarks: dict | None = None, **ds_kw):
    """Tập huấn luyện của một fold: ảnh AMI của người train, cộng (tùy chọn) ảnh thêm.

    ``extra_paths``: ảnh ngoài AMI (EarVN1.0 nhóm train, ảnh của bộ điểm mốc). Chúng
    đi vào một bộ riêng với biên an toàn ``extra_min_downscale`` và được trộn với
    xác suất ``extra_prob`` (mặc định theo tỉ lệ số ảnh). Với giao thức 'native',
    ảnh thêm vẫn dùng 'rand' (cắt từ ảnh gốc không có nghĩa với ảnh cỡ bất kỳ).
    """
    common = dict(scale=scale, patch_lr=patch_lr, tier=tier, degrade_kind=degrade_kind, params=params,
                  seed=seed, landmarks=landmarks, **ds_kw)
    ami = SRTrainDataset(ami_train_paths(ami_raw, folds_path, fold, "train"), protocol=protocol, **common)
    if not extra_paths:
        return ami
    extra = SRTrainDataset(list(extra_paths), protocol="rand" if protocol == "native" else protocol,
                           min_downscale=extra_min_downscale, **common)
    probs = None if extra_prob is None else [1.0 - extra_prob, extra_prob]
    return MixedDataset([ami, extra], probs=probs, seed=seed)


def val_pairs(bench_root: str | Path, folds_path: str | Path, fold: int, scale: int, tiers: tuple[int, ...],
              kind: str = "bic") -> list[tuple[np.ndarray, np.ndarray]]:
    """Các cặp (LR, HR) đã lưu của người validation."""
    bench_root = Path(bench_root)
    subj = set(load_folds(folds_path)["folds"][str(fold)]["val"])
    pairs = []
    for r in read_manifest(bench_root / "manifest.csv"):
        if r["subject"] in subj and r["tier"] in tiers:
            lr_path = bench_root / "lr" / f"hr{r['tier']}_x{scale}_{kind}" / f"{r['subject']}_{r['view']}.png"
            pairs.append((imread_rgb(lr_path), imread_rgb(bench_root / r["file"])))
    if not pairs:
        raise RuntimeError("tập validation rỗng")
    return pairs


def make_validator(bench_root: str | Path, folds_path: str | Path, fold: int, scale: int, tiers: tuple[int, ...],
                   kind: str = "bic", device: str = "cpu", metric: str = "psnr", perceptual=None):
    """Hàm ``validate(model) -> số`` (càng cao càng tốt) trên ảnh của người validation.

    metric 'psnr': PSNR-Y trung bình. metric 'lpips': trừ LPIPS trung bình (cần
    ``perceptual`` là một ``PerceptualMetrics`` có lpips), dùng để chọn checkpoint
    cho mô hình cảm nhận. Dùng ảnh HR và LR đã lưu của benchmark, nên validation
    đo đúng thứ sẽ đo ở test, nhưng trên người khác.
    """
    pairs = val_pairs(bench_root, folds_path, fold, scale, tiers, kind)
    if metric == "lpips" and (perceptual is None or "lpips" not in perceptual.available):
        raise RuntimeError("metric='lpips' cần PerceptualMetrics có lpips")
    if metric not in ("psnr", "lpips"):
        raise ValueError("metric phải là psnr hoặc lpips")

    @torch.no_grad()
    def validate(model: nn.Module) -> float:
        dev = next(model.parameters()).device
        vals = []
        for lr, hr in pairs:
            sr = to_uint8(model(to_tensor(lr).to(dev)))
            if metric == "psnr":
                vals.append(psnr(crop_border(rgb_to_y(sr), scale), crop_border(rgb_to_y(hr), scale)))
            else:
                vals.append(-perceptual(sr, hr)["lpips"])
        return float(np.mean(vals))

    return validate


# ----------------------------------------------------------------- nạp lại từ một lần chạy

def load_run_model(run_dir: str | Path, ckpt: str = "best") -> tuple[nn.Module, dict]:
    """Dựng lại mô hình của một lần chạy (``train.py`` hoặc ``pretrain.py``) từ
    ``config.json`` và nạp ``ckpt/{ckpt}.pt``. Trả về (mô hình ở chế độ eval, cấu hình)."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    ex = cfg.get("extra", {})
    for k in ("backbone", "variant", "scale"):
        if k not in ex:
            raise KeyError(f"{run_dir}/config.json thiếu extra.{k}")
    kind = ex.get("model_kind", "plain")
    if kind == "n3":
        model = build_n3(ex["variant"], int(ex["scale"]))
    else:
        model = build_trainable(ex["backbone"], ex["variant"], "none", int(ex["scale"]))
    model.load_state_dict(_load_ckpt_model(run_dir / "ckpt" / f"{ckpt}.pt"), strict=True)
    return model.eval(), cfg


def predictor_from_run(run_dir: str | Path, device: str = "cpu", ckpt: str = "best",
                       gate_bias: float | None = None, gate: float | None = None):
    """Hàm ``lr uint8 -> sr uint8`` của một lần chạy. Với mô hình N3: ``gate_bias``
    chọn điểm vận hành; ``gate`` = 1,0 hoặc 0,0 ép lấy đúng một đầu."""
    model, cfg = load_run_model(run_dir, ckpt)
    two_heads = isinstance(model, TwoHeadGated)
    if two_heads and gate_bias is not None:
        model.gate_bias = float(gate_bias)
    # Bản CPU là ``model``; bản trên ``device`` là một bản sao. Hết bộ nhớ GPU thì ảnh đó chạy trên CPU
    # và GPU được thử lại sau (earsr/device.py).
    runner = GpuFirst(lambda dev: model if dev == "cpu" else copy.deepcopy(model).to(dev), device,
                      name=str(cfg.get("run_id", ""))).warm()

    @torch.no_grad()
    def predict(lr):
        if two_heads:
            return runner.run(lambda m, dev: to_uint8(m(to_tensor(lr).to(dev), gate=gate)))
        return runner.run(lambda m, dev: to_uint8(m(to_tensor(lr).to(dev))))

    predict.scale = int(cfg["extra"]["scale"])
    predict.run_id = cfg["run_id"]
    predict.model = model
    predict.runner = runner
    return predict
