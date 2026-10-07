"""Mạng nhận dạng: một mạng học trên ảnh tai và một mạng đặc trưng tổng quát làm đối chứng."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _backbone(arch: str, init: str) -> tuple[nn.Module, int]:
    import torchvision

    if arch not in ("resnet18", "resnet34", "resnet50"):
        raise ValueError(f"kiến trúc không hỗ trợ: {arch}")
    net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if init == "imagenet" else None)
    dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, dim


class EarEmbedder(nn.Module):
    """ResNet -> đặc trưng ``emb`` chiều đã chuẩn hóa; đầu phân loại cosine có biên (CosFace) chỉ dùng lúc học."""

    def __init__(self, n_classes: int, arch: str = "resnet18", emb: int = 256, init: str = "none",
                 scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.cfg = {"kind": "ear", "n_classes": n_classes, "arch": arch, "emb": emb, "scale": scale, "margin": margin}
        self.net, dim = _backbone(arch, init)
        self.head = nn.Sequential(nn.Linear(dim, emb), nn.BatchNorm1d(emb))
        self.weight = nn.Parameter(torch.randn(n_classes, emb) * 0.01)
        self.scale, self.margin = scale, margin

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.net(x)), dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        cos = self.embed(x) @ F.normalize(self.weight, dim=1).t()
        if y is not None:
            cos = cos - self.margin * F.one_hot(y, cos.shape[1]).to(cos.dtype)
        return self.scale * cos


class ImageNetFeatures(nn.Module):
    """Đặc trưng ImageNet đóng băng, chưa từng thấy ảnh tai. Dùng để kiểm kết luận không phụ thuộc mạng nhận dạng."""

    def __init__(self, arch: str = "resnet50", init: str = "imagenet"):
        super().__init__()
        self.cfg = {"kind": "imagenet", "arch": arch}
        self.net, _ = _backbone(arch, init)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def save_embedder(model: EarEmbedder, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict(), "extra": extra or {}}, path)


def load_recognizer(spec: str, device: str = "cpu", init: str = "imagenet") -> tuple[nn.Module, dict]:
    """``spec``: đường dẫn checkpoint của train_recognizer.py, hoặc ``imagenet-<arch>`` cho mạng đối chứng."""
    if spec.startswith("imagenet-"):
        m = ImageNetFeatures(spec.split("-", 1)[1], init=init)
        return m.eval().to(device), {"cfg": m.cfg, "extra": {}}
    ck = torch.load(spec, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = EarEmbedder(c["n_classes"], c["arch"], c["emb"], "none", c["scale"], c["margin"])
    m.load_state_dict(ck["state"], strict=True)
    return m.eval().to(device), ck
