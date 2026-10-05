"""Mục tiêu huấn luyện. Vòng lặp ở ``trainer.train`` chỉ gọi giao diện này, nên
mọi cách huấn luyện (L1, GAN, LDL, đầu phụ N1, hai giai đoạn của N3) dùng chung
một vòng lặp, một cách lưu checkpoint và một cách chọn checkpoint.

    L1Objective        mốc phải vượt; mọi thân N5b
    GanObjective       "thân bản GAN" (L1 + cảm nhận + GAN); w_ldl > 0 cho "thân + LDL"
    AuxObjective       N1: L1 + đầu phụ dự đoán bản đồ điểm mốc (gỡ lúc suy luận)
    N3HeadsObjective   N3 giai đoạn 1: đầu trung thực (L1) và đầu kết cấu (L1 + cảm nhận + GAN)
    N3GateObjective    N3 giai đoạn 2: chỉ học cổng, nhãn có biên τ

CHƯA HUẤN LUYỆN THẬT với mục tiêu nào ngoài L1 (và L1 cũng mới chạy vài bước).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.optional.heads import AuxStructureHead, TwoHeadGated, gate_label
from ..models.span import SPAN
from .losses import PerceptualLoss, UNetDiscriminatorSN, gan_loss, ldl_loss


class Objective:
    name = "l1"
    select = "max"            # "max": giữ checkpoint có validate() cao nhất; "last": giữ checkpoint cuối
    supports_amp = False

    def to(self, device: str) -> "Objective":
        return self

    def prepare(self, model: nn.Module) -> None:
        """Đóng băng hoặc mở tham số của ``model`` trước khi tạo bộ tối ưu."""

    def parameters(self, model: nn.Module) -> list:
        return [p for p in model.parameters() if p.requires_grad]

    def loss(self, model: nn.Module, batch: list, step: int, ema_model: nn.Module | None):
        raise NotImplementedError

    def after_step(self, step: int) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass


class L1Objective(Objective):
    supports_amp = True

    def loss(self, model, batch, step, ema_model=None):
        lr, hr = batch[0], batch[1]
        return F.l1_loss(model(lr), hr), {}


class _Adversarial:
    """Phần dùng chung của các mục tiêu có GAN: bộ phân biệt, loss cảm nhận, bước D."""

    def _init_adv(self, w_percep: float, w_gan: float, lr_d: float, d_feat: int, percep_pretrained: bool):
        self.w_percep, self.w_gan = w_percep, w_gan
        self.percep = PerceptualLoss(pretrained=percep_pretrained) if w_percep > 0 else None
        self.disc = UNetDiscriminatorSN(num_feat=d_feat) if w_gan > 0 else None
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr_d, betas=(0.9, 0.99)) if self.disc else None
        self._pending = None

    def _to_adv(self, device):
        if self.percep is not None:
            self.percep.to(device)
        if self.disc is not None:
            self.disc.to(device)

    def _adv_terms(self, sr, hr, logs: dict):
        total = sr.new_zeros(())
        if self.percep is not None:
            lp = self.percep(sr, hr)
            total = total + self.w_percep * lp
            logs["percep"] = float(lp)
        if self.disc is not None:
            for p in self.disc.parameters():
                p.requires_grad_(False)
            lg = gan_loss(self.disc(sr), True)
            total = total + self.w_gan * lg
            logs["g_gan"] = float(lg)
            self._pending = (sr.detach(), hr.detach())
        return total

    def after_step(self, step: int) -> dict:
        if self.disc is None or self._pending is None:
            return {}
        sr, hr = self._pending
        self._pending = None
        for p in self.disc.parameters():
            p.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        l_real, l_fake = gan_loss(self.disc(hr), True), gan_loss(self.disc(sr), False)
        (l_real + l_fake).backward()
        self.opt_d.step()
        return {"d_real": float(l_real), "d_fake": float(l_fake)}

    def state_dict(self) -> dict:
        if self.disc is None:
            return {}
        return {"disc": self.disc.state_dict(), "opt_d": self.opt_d.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        if self.disc is not None and sd.get("disc") is not None:
            self.disc.load_state_dict(sd["disc"])
            self.opt_d.load_state_dict(sd["opt_d"])


class GanObjective(_Adversarial, Objective):
    """L1 + cảm nhận + GAN (trọng số của Real-ESRGAN: 1, 1, 0,1). ``w_ldl`` > 0
    thêm loss artifact của LDL (cần EMA bật trong TrainConfig)."""

    name, select = "gan", "last"

    def __init__(self, w_pix: float = 1.0, w_percep: float = 1.0, w_gan: float = 0.1, w_ldl: float = 0.0,
                 lr_d: float = 1e-4, d_feat: int = 64, percep_pretrained: bool = True):
        self.w_pix, self.w_ldl = w_pix, w_ldl
        self._init_adv(w_percep, w_gan, lr_d, d_feat, percep_pretrained)
        if w_ldl > 0:
            self.name = "ldl"

    def to(self, device):
        self._to_adv(device)
        return self

    def loss(self, model, batch, step, ema_model=None):
        lr, hr = batch[0], batch[1]
        sr = model(lr)
        logs = {}
        l_pix = F.l1_loss(sr, hr)
        logs["pix"] = float(l_pix)
        total = self.w_pix * l_pix + self._adv_terms(sr, hr, logs)
        if self.w_ldl > 0:
            if ema_model is None:
                raise RuntimeError("LDL cần mô hình EMA: đặt TrainConfig.ema > 0")
            with torch.no_grad():
                sr_ema = ema_model(lr)
            l_ldl = ldl_loss(sr, hr, sr_ema)
            total = total + self.w_ldl * l_ldl
            logs["ldl"] = float(l_ldl)
        return total, logs


def _span_of(model: nn.Module) -> SPAN:
    if isinstance(model, SPAN):
        return model
    raise TypeError("mục tiêu này cần thân SPAN (có forward_features và upsampler)")


class AuxObjective(Objective):
    """N1: L1 cộng loss của đầu phụ cấu trúc. Lô có dạng (lr, hr, bản đồ, có_nhãn).

    Đầu phụ nằm trong mục tiêu, không nằm trong mô hình: checkpoint của mô hình
    là thân SPAN thường, nên "gỡ đầu phụ lúc suy luận" không cần làm gì.
    """

    name = "aux"

    def __init__(self, feat_ch: int, n_maps: int, w_aux: float = 0.1, hidden: int = 32):
        self.head = AuxStructureHead(feat_ch, n_maps, hidden)
        self.w_aux = w_aux

    def to(self, device):
        self.head.to(device)
        return self

    def parameters(self, model):
        return [p for p in model.parameters() if p.requires_grad] + list(self.head.parameters())

    def loss(self, model, batch, step, ema_model=None):
        lr, hr, target, has = batch
        net = _span_of(model)
        feat = net.forward_features(lr)
        l_pix = F.l1_loss(net.upsampler(feat), hr)
        per = ((self.head(feat) - target) ** 2).mean((1, 2, 3))
        l_aux = (per * has).sum() / has.sum().clamp(min=1.0)
        return l_pix + self.w_aux * l_aux, {"pix": float(l_pix), "aux": float(l_aux), "labeled": float(has.mean())}

    def state_dict(self):
        return {"head": self.head.state_dict()}

    def load_state_dict(self, sd):
        if sd.get("head") is not None:
            self.head.load_state_dict(sd["head"])


class N3HeadsObjective(_Adversarial, Objective):
    """N3, giai đoạn 1. Mô hình là ``TwoHeadGated``; cổng chưa dùng.

    config 'A': thân và đầu trung thực đóng băng (đã tinh chỉnh từ trước); chỉ
                đầu kết cấu học. Đầu trung thực giữ nguyên PSNR của thân.
    config 'B': thân và hai đầu học chung.
    """

    name, select = "n3s1", "last"

    def __init__(self, config: str = "A", w_pix_t: float = 1.0, w_percep: float = 1.0, w_gan: float = 0.1,
                 lr_d: float = 1e-4, d_feat: int = 64, percep_pretrained: bool = True):
        if config not in ("A", "B"):
            raise ValueError("config phải là 'A' hoặc 'B'")
        self.config, self.w_pix_t = config, w_pix_t
        self._init_adv(w_percep, w_gan, lr_d, d_feat, percep_pretrained)

    def to(self, device):
        self._to_adv(device)
        return self

    def prepare(self, model):
        if not isinstance(model, TwoHeadGated):
            raise TypeError("N3 cần mô hình TwoHeadGated")
        for p in model.parameters():
            p.requires_grad_(False)
        train_parts = [model.head_t] + ([model.backbone] if self.config == "B" else [])
        for part in train_parts:
            for n, p in part.named_parameters():
                if "eval_conv" not in n:   # eval_conv là bản gộp, không phải tham số học
                    p.requires_grad_(True)

    def loss(self, model, batch, step, ema_model=None):
        lr, hr = batch[0], batch[1]
        _, f_s, f_t, _ = model(lr, gate=1.0, return_all=True)
        logs = {}
        total = f_t.new_zeros(())
        if self.config == "B":
            l_s = F.l1_loss(f_s, hr)
            total = total + l_s
            logs["pix_s"] = float(l_s)
        l_t = F.l1_loss(f_t, hr)
        logs["pix_t"] = float(l_t)
        total = total + self.w_pix_t * l_t + self._adv_terms(f_t, hr, logs)
        return total, logs


class N3GateObjective(Objective):
    """N3, giai đoạn 2: chỉ cổng học. Nhãn = 1 (giữ đầu trung thực) ở nơi đầu kết
    cấu sai hơn đầu trung thực quá ``tau`` (sai số L1 theo điểm ảnh, dải [0, 1])."""

    name, select = "n3s2", "last"

    def __init__(self, tau: float = 0.02):
        self.tau = tau

    def prepare(self, model):
        if not isinstance(model, TwoHeadGated):
            raise TypeError("N3 cần mô hình TwoHeadGated")
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.gate.parameters():
            p.requires_grad_(True)

    def loss(self, model, batch, step, ema_model=None):
        lr, hr = batch[0], batch[1]
        _, f_s, f_t, g = model(lr, return_all=True)
        label = gate_label(f_s.detach(), f_t.detach(), hr, self.tau)
        l = F.binary_cross_entropy(g.clamp(1e-6, 1 - 1e-6), label)
        return l, {"gate_bce": float(l), "label_keep": float(label.mean()), "gate_mean": float(g.mean())}


def make_objective(name: str, model: nn.Module | None = None, **kw) -> Objective:
    """Tạo mục tiêu theo tên: l1 | gan | ldl | aux | n3s1A | n3s1B | n3s2."""
    if name == "l1":
        return L1Objective()
    if name == "gan":
        return GanObjective(**kw)
    if name == "ldl":
        return GanObjective(w_ldl=kw.pop("w_ldl", 1.0), **kw)
    if name == "aux":
        net = _span_of(model)
        return AuxObjective(net.feature_channels, kw.pop("n_maps"), **kw)
    if name in ("n3s1A", "n3s1B"):
        return N3HeadsObjective(config=name[-1], **kw)
    if name == "n3s2":
        return N3GateObjective(**kw)
    raise ValueError(f"không có mục tiêu '{name}'")
