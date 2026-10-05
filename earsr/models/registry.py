"""Kho mô hình: tên -> kiến trúc, trọng số, quy ước vào ra.

Mọi mô hình được bọc trong ``SRModel`` để có cùng giao diện: nhận tensor RGB
trong [0, 1] cỡ bất kỳ, trả tensor RGB trong [0, 1] (chưa cắt ngưỡng) lớn gấp
``scale`` lần.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .span import SPAN, is_deploy_state_dict


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scale: int
    build: Callable[[], nn.Module]
    weights: str | None = None          # tên file trong thư mục trọng số
    ckpt_key: str | None = None         # 'params', 'params_ema' ... hoặc None nếu file là state_dict
    convert: Callable[[dict], dict] | None = None
    data_range: float = 1.0             # 255 nếu mạng học trên dải 0..255
    window: int = 1                     # bội số kích thước mà mạng cần (SwinIR: 8)
    group: str = "light"                # light | mid | perceptual | upper
    objective: str = "psnr"             # psnr | gan
    pretrain: str = "?"                 # dữ liệu tiền huấn luyện, theo nguồn công bố
    local_only: bool = True             # True nếu thân chỉ có phép toán cục bộ (kiểm thử kiểu đệm áp được)
    source: str = ""
    note: str = ""


class SRModel(nn.Module):
    """Bọc một mạng SR để thống nhất dải giá trị và yêu cầu kích thước."""

    def __init__(self, net: nn.Module, spec: ModelSpec):
        super().__init__()
        self.net = net
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, win = self.spec.scale, self.spec.window
        h, w = x.shape[-2:]
        if win > 1:
            ph, pw = (win - h % win) % win, (win - w % win) % win
            if ph or pw:
                # cách của mã test SwinIR: nối ảnh lật rồi cắt
                x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : h + ph, :]
                x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : w + pw]
        y = self.net(x * self.spec.data_range) / self.spec.data_range
        return y[..., : h * s, : w * s]


# Dải giá trị mà trọng số SPAN 48 kênh chính thức được học (ảnh vào nhân 255). Mọi nơi nạp bộ trọng số
# đó vào một thân SPAN tự dựng phải dùng đúng hằng số này (sai dải thì PSNR Set5 tụt từ 32,20 xuống 12,70 dB
# mà không báo lỗi).
SPAN_PUB_IMG_RANGE = 255.0


def _span(ch: int, deploy: bool, scale: int = 4, **kw):
    return lambda: SPAN(feature_channels=ch, upscale=scale, deploy=deploy, **kw)


def _rlfn():
    from .zoo.rlfn import RLFN_Prune
    return RLFN_Prune()


def _efdn():
    from .zoo.efdn import EFDN
    return EFDN()


def _safmnpp():
    from .zoo.safmnpp import SAFMNPP
    return SAFMNPP(dim=36, n_blocks=6, ffn_scale=1.5, upscaling_factor=4)


def _smfan():
    from .zoo.smfan import SMFAN
    return SMFAN()


def _msrresnet():
    from .zoo.msrresnet import MSRResNet
    return MSRResNet()


def _edsr_baseline():
    from .zoo.edsr import EDSR
    return EDSR()


def _swinir_light(scale: int):
    def f():
        from .zoo.swinir import SwinIR
        return SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                      depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                      upsampler="pixelshuffledirect", resi_connection="1conv")
    return f


def _rrdb(scale: int = 4):
    def f():
        from .zoo.rrdbnet import RRDBNet
        return RRDBNet(scale=scale)
    return f


def _kair(sd: dict) -> dict:
    from .zoo.rrdbnet import convert_kair_keys
    return convert_kair_keys(sd)


def _srvgg():
    from .zoo.srvgg import SRVGGNetCompact
    return SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)


_NT24 = "https://github.com/Amazingren/NTIRE2024_ESR"
_NT25 = "https://github.com/Amazingren/NTIRE2025_ESR"
_KAIR = "https://github.com/cszn/KAIR/releases/tag/v1.0"
_RESR = "https://github.com/xinntao/Real-ESRGAN/releases"
_SWIN = "https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0"
_EDSR = "https://github.com/sanghyun-son/EDSR-PyTorch"
_SPAN = "https://github.com/hongyuanyu/SPAN"

SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    # --- nhẹ, tối ưu PSNR ---
    ModelSpec("span_ch48", 4, _span(48, False, img_range=SPAN_PUB_IMG_RANGE), "span_ch48_x4_official.pth", "params_ema",
              group="light", pretrain="DF2K", source=_SPAN,
              note="SPAN 48 kênh, trọng số chính thức của tác giả (spanx4_ch48.pth trong span.zip; dạng huấn luyện, "
                   "gộp khi nạp; img_range=255). Trong bài SPAN, bản 48 kênh (426 nghìn tham số) mang tên SPAN-S; "
                   "bản 52 kênh mới mang tên SPAN."),
    ModelSpec("span_ch48_t44", 4, _span(48, False), "team44_SPANx4.pth", "params", group="light",
              pretrain="theo đội 44 NTIRE 2025 (chưa xác minh)", source=_NT25,
              note="SPAN 48 kênh, trọng số của đội 44 NTIRE 2025 (img_range=1). Là 'span_ch48' của project tới "
                   "05/10/2026; kết quả sơ bộ mang tên span_ch48 là của bộ trọng số này."),
    ModelSpec("span_ch28", 4, _span(28, True, img_range=255.0), "team38_span_ch28_slim.pth", None, group="light",
              pretrain="theo đội 38 NTIRE 2024", source=_NT24, note="SPAN 28 kênh, bản thắng NTIRE 2024 ESR; img_range=255 như mã test của NTIRE"),
    ModelSpec("span_ch26", 4, _span(26, True, img_range=255.0), "team39_spantiny_ch26_slim.pth", None, group="light",
              pretrain="theo đội 39 NTIRE 2024", source=_NT24, note="SPAN-tiny 26 kênh"),
    ModelSpec("rlfn", 4, _rlfn, "team00_rlfn.pth", None, data_range=255.0, group="light", local_only=False,
              pretrain="DIV2K+Flickr2K (theo bài RLFN)", source=_NT24, note="mốc của NTIRE 2024 ESR; có ESA (max-pool bước lớn)"),
    ModelSpec("efdn", 4, _efdn, "team00_EFDN.pth", None, group="light", local_only=False,
              pretrain="theo NTIRE 2023", source=_NT25, note="mốc của NTIRE 2025 ESR; đầu ra bị cắt ngưỡng trong mạng"),
    ModelSpec("safmnpp", 4, _safmnpp, "team23_safmnpp.pth", "params_ema", group="light", local_only=False,
              pretrain="theo đội 23 NTIRE 2024", source=_NT24, note="SAFMN++; có gộp thích nghi nhiều tỉ lệ"),
    ModelSpec("smfan", 4, _smfan, "team24_smfan.pth", "params_ema", group="light", local_only=False,
              pretrain="theo đội 24 NTIRE 2024", source=_NT24, note="SMFANet bản NTIRE"),
    # --- cỡ vừa ---
    ModelSpec("msrresnet", 4, _msrresnet, "msrresnet_x4_psnr.pth", None, group="mid",
              pretrain="không rõ (KAIR không nêu; README của KAIR liệt kê DIV2K và Flickr2K)", source=_KAIR,
              note="1,5 triệu tham số, cùng cỡ EDSR-baseline"),
    ModelSpec("edsr_baseline", 4, _edsr_baseline, "edsr_baseline_x4.pth", None, data_range=255.0, group="mid",
              pretrain="DIV2K", source=_EDSR, note="EDSR-baseline, trọng số chính thức của tác giả; 1,5 triệu tham số"),
    ModelSpec("swinir_light", 4, _swinir_light(4), "swinir_light_x4.pth", "params", window=8, group="mid",
              local_only=False, pretrain="DIV2K", source=_SWIN, note="SwinIR-light; transformer"),
    # --- trần trên, tối ưu PSNR ---
    ModelSpec("rrdb_psnr", 4, _rrdb(4), "kair_RRDB_psnr_x4.pth", None, convert=_kair, group="upper",
              pretrain="DF2K (ESRGAN bản PSNR)", source=_KAIR, note="RRDB 16,7 triệu tham số, bản tối ưu PSNR"),
    # --- cảm nhận ---
    ModelSpec("esrgan", 4, _rrdb(4), "kair_ESRGAN_x4.pth", None, convert=_kair, group="perceptual",
              objective="gan", pretrain="DF2K, suy giảm bicubic", source=_KAIR, note="ESRGAN bản GAN"),
    ModelSpec("bsrgan", 4, _rrdb(4), "BSRGAN.pth", None, convert=_kair, group="perceptual", objective="gan",
              pretrain="suy giảm tổng hợp của BSRGAN", source=_KAIR),
    ModelSpec("realesrgan", 4, _rrdb(4), "RealESRGAN_x4plus.pth", "params_ema", group="perceptual",
              objective="gan", pretrain="suy giảm bậc cao của Real-ESRGAN", source=_RESR),
    ModelSpec("realesr_compact", 4, _srvgg, "realesr-general-x4v3.pth", "params", group="perceptual",
              objective="gan", pretrain="suy giảm của Real-ESRGAN", source=_RESR, note="SRVGGNetCompact 1,2 triệu tham số"),
    # --- ×2 ---
    ModelSpec("swinir_light_x2", 2, _swinir_light(2), "swinir_light_x2.pth", "params", window=8, group="mid",
              local_only=False, pretrain="DIV2K", source=_SWIN),
    ModelSpec("realesrgan_x2", 2, _rrdb(2), "RealESRGAN_x2plus.pth", "params_ema", group="perceptual",
              objective="gan", pretrain="suy giảm bậc cao của Real-ESRGAN", source=_RESR),
]}


# --- NTIRE 2026 Efficient SR (trọng số ở dạng đã gộp nhánh; https://github.com/Amazingren/NTIRE2026_ESR) ---
_NT26 = "https://github.com/Amazingren/NTIRE2026_ESR"


def _nt26(module: str, cls: str, **kw):
    def build():
        import importlib
        return getattr(importlib.import_module(f"{__package__}.zoo.{module}"), cls)(**kw)
    return build


for _s in [
    ModelSpec("span26", 4, _span(28, True, img_range=255.0), "nt26_team00_SPAN.pth", None, group="light",
              pretrain="DIV2K+LSDIR (baseline chính thức NTIRE 2026)", source=_NT26,
              note="SPAN 28 kênh, baseline chính thức của NTIRE 2026 ESR"),
    ModelSpec("pds26", 4, _nt26("nt26_pds", "PDS"), "nt26_team01_PDS.pth", None, group="light",
              pretrain="theo đội 01 NTIRE 2026", source=_NT26, note="hạng 2 NTIRE 2026; SPANF cắt kênh và chưng cất"),
    ModelSpec("pkdsr26", 4, _nt26("nt26_pkdsr", "SPANFPrunedKD", num_in_ch=3, num_out_ch=3, upscale=4,
                                  tail_channels=24, feature_channels=32), "nt26_team16_PKDSR.pth", "params_ema",
              group="light", pretrain="theo đội 16 NTIRE 2026", source=_NT26, note="hạng 3 NTIRE 2026; SPANF cắt hai giai đoạn"),
    ModelSpec("dscf26", 4, _nt26("nt26_dscf", "DSCF_Fused", num_in_ch=3, num_out_ch=3, feature_channels=26, upscale=4),
              "nt26_team15_DSCF_Fused.pth", None, group="light", pretrain="theo đội 15 NTIRE 2026", source=_NT26,
              note="DSCF bản gộp"),
    ModelSpec("disp26", 4, _nt26("nt26_disp", "DISP"), "nt26_team18_DISP.pth", None, group="light",
              pretrain="theo đội 18 NTIRE 2026", source=_NT26, note="hạng 4 NTIRE 2026; họ TSSR, khối tham số hóa lại"),
    ModelSpec("errn26", 4, _nt26("nt26_errn2", "ERRN2", in_channels=3, out_channels=3, feature_channels=32, upscale=4),
              "nt26_team20_ERRN2.pth", None, group="light", pretrain="theo đội 20 NTIRE 2026", source=_NT26,
              note="hạng 6 NTIRE 2026; ERRN, khác họ SPAN"),
]:
    SPECS[_s.name] = _s


def weights_dir() -> Path:
    """Thư mục trọng số: biến môi trường EARSR_WEIGHTS, mặc định ``./weights``."""
    return Path(os.environ.get("EARSR_WEIGHTS", "weights"))


def list_models(scale: int | None = None, available_only: bool = False) -> list[str]:
    out = []
    for n, s in SPECS.items():
        if scale is not None and s.scale != scale:
            continue
        if available_only and not (weights_dir() / (s.weights or "")).is_file():
            continue
        out.append(n)
    return out


def load_state(spec: ModelSpec, wdir: Path | None = None) -> dict:
    path = (wdir or weights_dir()) / spec.weights
    if not path.is_file():
        raise FileNotFoundError(f"thiếu trọng số của '{spec.name}': {path}. Xem scripts/get_weights.sh")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if spec.ckpt_key is not None:
        sd = sd[spec.ckpt_key]
    if spec.convert is not None:
        sd = spec.convert(sd)
    return sd


def build_model(name: str, pretrained: bool = True, wdir: Path | None = None, deploy: bool = True) -> SRModel:
    """Tạo mô hình theo tên. ``pretrained``: nạp trọng số công bố (nạp chặt,
    thiếu hay thừa khóa đều báo lỗi). ``deploy``: gộp nhánh tham số hóa lại."""
    if name not in SPECS:
        raise KeyError(f"không có mô hình '{name}'. Có: {sorted(SPECS)}")
    spec = SPECS[name]
    net = spec.build()
    if pretrained:
        sd = load_state(spec, wdir)
        if isinstance(net, SPAN) and is_deploy_state_dict(sd) != net.conv_1.deploy:
            raise RuntimeError(f"{name}: dạng trọng số (deploy hay huấn luyện) không khớp cấu hình")
        net.load_state_dict(sd, strict=True)
    net.eval()
    if deploy and hasattr(net, "switch_to_deploy"):
        net.switch_to_deploy()
    return SRModel(net, spec).eval()


class Bicubic(nn.Module):
    """Mốc dưới: nội suy bicubic của torch (chỉ để đo độ trễ; bảng chất lượng dùng
    ``imresize`` kiểu MATLAB)."""

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
