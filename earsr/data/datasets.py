"""Bộ dữ liệu huấn luyện, với ba giao thức tạo mẫu (T6 ii của kế hoạch).

native: cắt patch từ ảnh gốc (với AMI là 492×702). Sai tỉ lệ so với lúc test;
        chỉ là nhánh đối chứng.
fixed : thu ảnh gốc về đúng cỡ của ô đang xét (``tier``) rồi cắt patch. Đây là
        cách hiển nhiên mà một người làm SR sẽ dùng.
rand  : thu ảnh gốc về cạnh ngắn s rút ngẫu nhiên từ một lưới trong
        [tier_min, tier_max] rồi cắt patch; một mô hình cho mọi cỡ.

Ảnh LR được tạo bằng cách suy giảm cả ảnh HR (đã thu về cỡ s) rồi mới cắt patch
thẳng hàng. Cắt patch HR trước rồi thu nhỏ sẽ làm sai ở mép patch, vì bộ lọc
khử răng cưa phản chiếu ở mép.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..degrade.pipelines import DegradeParams, degrade
from ..io import imread_rgb
from .resize import center_crop_to_multiple, imresize

PROTOCOLS = ("native", "fixed", "rand")


def short_side_transform(h: int, w: int, s: int | None, multiple: int = 4) -> tuple[float, float, float, float]:
    """Phép biến đổi toạ độ ứng với ``resize_short_side`` (hoặc chỉ cắt giữa nếu
    ``s`` là None): trả về (fy, fx, oy, ox) sao cho điểm (x, y) của ảnh gốc, theo
    quy ước tâm điểm ảnh, sang ảnh mới là
        x' = (x + 0.5) * fx - 0.5 - ox,   y' = (y + 0.5) * fy - 0.5 - oy.
    """
    if s is None or min(h, w) == s:
        nh, nw, fy, fx = h, w, 1.0, 1.0
    else:
        nh, nw = (int(round(h * s / w)), s) if h >= w else (s, int(round(w * s / h)))
        fy, fx = nh / h, nw / w
    return fy, fx, float((nh % multiple) // 2), float((nw % multiple) // 2)


def point_heatmaps(xy: np.ndarray, size: int, sigma: float, mode: str = "heatmap") -> np.ndarray:
    """Bản đồ Gauss ``size``×``size`` cho các điểm ``xy`` (K, 2) theo toạ độ điểm
    ảnh của patch. mode 'heatmap': K kênh; 'points': 1 kênh (lớn nhất theo điểm)."""
    g = np.arange(size, dtype=np.float32)
    d2 = (g[None, None, :] - xy[:, 0, None, None]) ** 2 + (g[None, :, None] - xy[:, 1, None, None]) ** 2
    hm = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
    return hm if mode == "heatmap" else hm.max(0, keepdims=True)


def resize_short_side(img: np.ndarray, s: int, multiple: int = 4) -> np.ndarray:
    """Thu ảnh về cạnh ngắn ``s`` rồi cắt giữa cho chia hết cho ``multiple``."""
    h, w = img.shape[:2]
    if min(h, w) == s:
        return center_crop_to_multiple(img, multiple)
    if h >= w:
        out = (int(round(h * s / w)), s)
    else:
        out = (s, int(round(w * s / h)))
    return center_crop_to_multiple(imresize(img, out_size=out), multiple)


def _short_side(path) -> int:
    """Cạnh ngắn của ảnh, chỉ đọc phần đầu file."""
    from PIL import Image

    with Image.open(path) as im:
        return min(im.size)


class _LRU:
    def __init__(self, max_items: int):
        self.max, self.d = max_items, OrderedDict()

    def get(self, k, make):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        v = make()
        self.d[k] = v
        if len(self.d) > self.max:
            self.d.popitem(last=False)
        return v


class SRTrainDataset(Dataset):
    """Mỗi phần tử: (lr, hr) tensor float [0, 1], cỡ patch_lr và patch_lr × scale.

    paths     : danh sách ảnh gốc (đã lọc theo người train của fold).
    protocol  : native | fixed | rand.
    tier      : cỡ của ô đang xét (cho 'fixed').
    rand_grid : lưới cạnh ngắn cho 'rand', mặc định 96, 104, ..., 192.
    degrade_kind, params: kiểu suy giảm (mục 1.7).
    min_downscale: ảnh gốc phải được thu nhỏ ít nhất ngần này lần để thành ảnh HR
                (biên an toàn cho ảnh JPEG ngoài thực tế; AMI dùng 1,0). Ảnh không
                đủ lớn cho cỡ nào thì bị loại; ``self.dropped`` ghi số ảnh loại.
    landmarks : dict đường dẫn ảnh (str) -> mảng (K, 2) toạ độ (x, y) trên ảnh gốc
                (N1). Nếu có, mỗi phần tử là (lr, hr, bản đồ, có_nhãn): bản đồ Gauss
                ở độ phân giải LR; ``có_nhãn`` = 0 với ảnh không có điểm mốc.
    Mẫu thứ ``idx`` được rút bằng bộ sinh ngẫu nhiên gieo từ (seed, idx), nên
    kết quả không phụ thuộc số worker và huấn luyện tiếp tục được đúng chỗ dừng.
    Vòng huấn luyện duyệt idx = 0, 1, 2, ... một lần duy nhất (không có epoch).
    """

    def __init__(self, paths: list, scale: int = 4, protocol: str = "rand", patch_lr: int = 24,
                 tier: int = 144, rand_grid: tuple[int, ...] = tuple(range(96, 193, 8)),
                 degrade_kind: str = "bic", params: DegradeParams | None = None, hflip: bool = True,
                 rot90: bool = False, seed: int = 0, cache_items: int = 4096,
                 landmarks: dict | None = None, n_landmarks: int = 55, aux_mode: str = "heatmap",
                 aux_sigma: float = 1.0, min_downscale: float = 1.0):
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol phải thuộc {PROTOCOLS}")
        if not paths:
            raise ValueError("danh sách ảnh rỗng")
        self.paths = [Path(p) for p in paths]
        self.scale, self.protocol, self.patch_lr = scale, protocol, patch_lr
        self.tier, self.rand_grid = tier, tuple(rand_grid)
        self.min_downscale, self.dropped = float(min_downscale), 0
        # cỡ HR được phép của từng ảnh (không bao giờ phóng to ảnh gốc để làm đáp án)
        self.sizes: list[tuple[int, ...]] | None = None
        if protocol != "native":
            want = (tier,) if protocol == "fixed" else self.rand_grid
            keep, sizes = [], []
            for q in self.paths:
                short = _short_side(q)
                ok = tuple(v for v in want if v * self.min_downscale <= short)
                if ok:
                    keep.append(q)
                    sizes.append(ok)
            self.dropped = len(self.paths) - len(keep)
            if not keep:
                raise ValueError(f"không ảnh nào đủ lớn cho cỡ {want} với biên {min_downscale}")
            self.paths, self.sizes = keep, sizes
        self.kind, self.params = degrade_kind, params
        self.hflip, self.rot90 = hflip, rot90
        self.seed = seed
        self.landmarks = None if landmarks is None else {str(k): np.asarray(v, np.float32) for k, v in landmarks.items()}
        if self.landmarks is not None:
            if rot90:
                raise ValueError("rot90 chưa hỗ trợ khi có điểm mốc")
            if aux_mode not in ("heatmap", "points"):
                raise ValueError("aux_mode phải là 'heatmap' hoặc 'points'")
            bad = [k for k, v in self.landmarks.items() if v.shape != (n_landmarks, 2)]
            if bad:
                raise ValueError(f"{len(bad)} ảnh có số điểm mốc khác {n_landmarks}, ví dụ {bad[0]}")
        self.n_landmarks, self.aux_mode, self.aux_sigma = n_landmarks, aux_mode, aux_sigma
        self._shape: dict = {}
        self._orig = _LRU(max(64, cache_items // 4))
        self._hr = _LRU(cache_items)
        self._lr = _LRU(cache_items)

    def __len__(self) -> int:
        return 1 << 40  # coi như vô hạn; vòng huấn luyện tự cấp chỉ số

    def _hr_image(self, i: int, s: int | None) -> np.ndarray:
        def make():
            orig = self._orig.get(i, lambda: imread_rgb(self.paths[i]))
            self._shape[i] = orig.shape[:2]
            if s is None:  # native
                return center_crop_to_multiple(orig, self.scale)
            return resize_short_side(orig, s, multiple=max(4, self.scale))
        return self._hr.get((i, s), make)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx)])
        i = int(rng.integers(len(self.paths)))
        if self.protocol == "native":
            s = None
        elif self.protocol == "fixed":
            s = self.tier
        else:
            s = int(rng.choice(self.sizes[i]))
        hr = self._hr_image(i, s)
        if self.kind == "bic":
            lr = self._lr.get((i, s), lambda: degrade(hr, self.scale, "bic"))
        else:  # suy giảm ngẫu nhiên: sinh mới mỗi lần
            lr = degrade(hr, self.scale, self.kind, seed=int(rng.integers(1 << 31)), key=str(idx),
                         params=self.params)
        p = self.patch_lr
        lh, lw = lr.shape[:2]
        if lh < p or lw < p:
            raise RuntimeError(f"ảnh LR {lh}×{lw} nhỏ hơn patch {p}; giảm patch_lr hoặc tăng cỡ ảnh")
        y, x = int(rng.integers(lh - p + 1)), int(rng.integers(lw - p + 1))
        lp = lr[y:y + p, x:x + p]
        hp = hr[y * self.scale:(y + p) * self.scale, x * self.scale:(x + p) * self.scale]
        flipped = bool(self.hflip and rng.random() < 0.5)
        if flipped:
            lp, hp = lp[:, ::-1], hp[:, ::-1]
        if self.rot90:
            k = int(rng.integers(4))
            lp, hp = np.rot90(lp, k), np.rot90(hp, k)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)
        if self.landmarks is None:
            return to_t(lp), to_t(hp)
        n_ch = self.n_landmarks if self.aux_mode == "heatmap" else 1
        pts = self.landmarks.get(str(self.paths[i]))
        if pts is None:
            return to_t(lp), to_t(hp), torch.zeros(n_ch, p, p), torch.zeros(())
        if i not in self._shape:  # ảnh HR lấy từ bộ đệm nhưng cỡ ảnh gốc chưa ghi (không xảy ra trong một tiến trình)
            self._shape[i] = imread_rgb(self.paths[i]).shape[:2]
        oh, ow = self._shape[i]
        mult = self.scale if s is None else max(4, self.scale)
        fy, fx, oy, ox = short_side_transform(oh, ow, s, mult)
        # ảnh gốc -> ảnh HR -> ảnh LR -> patch
        px = ((pts[:, 0] + 0.5) * fx - ox) / self.scale - 0.5 - x
        py = ((pts[:, 1] + 0.5) * fy - oy) / self.scale - 0.5 - y
        if flipped:
            px = (p - 1) - px
        hm = point_heatmaps(np.stack([px, py], 1), p, self.aux_sigma, self.aux_mode)
        return to_t(lp), to_t(hp), torch.from_numpy(hm), torch.ones(())


class MixedDataset(Dataset):
    """Trộn nhiều ``SRTrainDataset`` (ví dụ AMI và ảnh thêm từ EarVN1.0). Mẫu thứ
    ``idx`` lấy từ bộ thứ k với xác suất ``probs[k]`` (mặc định theo số ảnh), rút
    bằng bộ sinh gieo từ (seed, idx) nên tái lập được."""

    def __init__(self, parts: list, probs: list[float] | None = None, seed: int = 0):
        if not parts:
            raise ValueError("cần ít nhất một bộ dữ liệu")
        self.parts = list(parts)
        w = np.array(probs if probs is not None else [len(d.paths) for d in self.parts], dtype=np.float64)
        if len(w) != len(self.parts) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("probs không hợp lệ")
        self.cum = np.cumsum(w / w.sum())
        self.seed = seed

    @property
    def paths(self) -> list:
        return [p for d in self.parts for p in d.paths]

    def __len__(self) -> int:
        return 1 << 40

    def __getitem__(self, idx: int):
        u = np.random.default_rng([self.seed, 7919, int(idx)]).random()
        k = min(int(np.searchsorted(self.cum, u, side="right")), len(self.parts) - 1)
        return self.parts[k][idx]


def list_images(root: str | Path, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)
