"""N2 (a): bộ phân loại "mô phỏng hay thật".

Câu hỏi: ảnh LR do một kiểu suy giảm tạo ra có giống ảnh nhỏ thật không? Một bộ
phân loại nhỏ được huấn luyện để tách patch của ảnh nhỏ thật khỏi patch của ảnh
mô phỏng. Kiểu suy giảm càng giống thật thì độ chính xác càng gần 50%.

Cách tránh để bộ phân loại dựa vào nội dung thay vì suy giảm (mục 1.7):
- ảnh mô phỏng được tạo từ ảnh LỚN của chính bộ ảnh ngoài thực tế, của cùng nhóm
  người, và được đưa về cùng phân bố cỡ với ảnh nhỏ thật;
- bộ phân loại chỉ nhìn patch nhỏ (mặc định 16×16);
- chia train và test theo người.

Giới hạn phải nêu trong bài: độ chính xác gần 50% chỉ nói bộ phân loại NÀY không
tách được; nó không chứng minh hai phân bố trùng nhau.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.datasets import resize_short_side
from ..degrade.pipelines import DegradeParams, degrade


def simulate_small(large: np.ndarray, short: int, kind: str, scale: int = 4, params: DegradeParams | None = None,
                   seed: int = 0, key: str = "") -> np.ndarray | None:
    """Ảnh lớn -> ảnh LR mô phỏng có cạnh ngắn ``short``. None nếu ảnh lớn không đủ lớn."""
    if min(large.shape[:2]) < short * scale:
        return None
    hr = resize_short_side(large, short * scale, multiple=scale)
    return degrade(hr, scale, kind, seed=seed, key=key, params=params)


class PatchNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.f = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(inplace=True),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * w, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x - 0.5).squeeze(1)


def _patches(img: np.ndarray, patch: int, n: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = rng.integers(0, h - patch + 1, n), rng.integers(0, w - patch + 1, n)
    return np.stack([img[y:y + patch, x:x + patch] for y, x in zip(ys, xs)])


def _grid_patches(img: np.ndarray, patch: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = sorted(set(list(range(0, h - patch + 1, patch)) + [h - patch]))
    xs = sorted(set(list(range(0, w - patch + 1, patch)) + [w - patch]))
    return np.stack([img[y:y + patch, x:x + patch] for y in ys for x in xs])


def _to_t(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(device)


def train_classifier(real: list, sim: list, patch: int = 16, steps: int = 2000, batch: int = 128,
                     lr: float = 1e-3, width: int = 32, device: str = "cpu", seed: int = 0) -> PatchNet:
    """``real``, ``sim``: danh sách ảnh uint8. Mỗi lô cân bằng hai lớp (nhãn 1 = thật)."""
    for name, lst in (("thật", real), ("mô phỏng", sim)):
        if not lst:
            raise ValueError(f"không có ảnh {name} để huấn luyện")
        if min(min(i.shape[:2]) for i in lst) < patch:
            raise ValueError(f"có ảnh {name} nhỏ hơn patch {patch}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PatchNet(width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    half = batch // 2
    y = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)
    net.train()
    for _ in range(steps):
        xs = [_patches(real[int(rng.integers(len(real)))], patch, 1, rng)[0] for _ in range(half)]
        xs += [_patches(sim[int(rng.integers(len(sim)))], patch, 1, rng)[0] for _ in range(half)]
        x = _to_t(np.stack(xs), device)
        if rng.random() < 0.5:
            x = x.flip(3)
        loss = F.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def score_images(net: PatchNet, images: list, patch: int = 16, device: str = "cpu") -> np.ndarray:
    """Với mỗi ảnh: tỉ lệ patch (lưới không chồng) được đoán là 'thật'."""
    out = []
    for img in images:
        out.append(float((net(_to_t(_grid_patches(img, patch), device)) > 0).float().mean()))
    return np.array(out)


def balanced_accuracy(p_real_on_real: np.ndarray, p_real_on_sim: np.ndarray) -> float:
    return 0.5 * (float(np.mean(p_real_on_real)) + float(1.0 - np.mean(p_real_on_sim)))


def realism_test(real: dict, sims: dict, patch: int = 16, steps: int = 2000, test_frac: float = 0.3,
                 n_seeds: int = 3, n_boot: int = 2000, device: str = "cpu", seed: int = 0, width: int = 32) -> dict:
    """Chạy phép thử cho từng kiểu suy giảm.

    ``real``: dict người -> danh sách ảnh nhỏ thật.
    ``sims``: dict kiểu -> (dict người -> danh sách ảnh mô phỏng).
    Người được chia train và test một lần (cùng cách chia cho mọi kiểu). Với mỗi
    kiểu: huấn luyện ``n_seeds`` bộ phân loại, lấy trung bình điểm theo ảnh test.
    Trả về độ chính xác cân bằng (theo patch) kèm khoảng tin cậy bootstrap theo
    người, và với mỗi cặp kiểu, hiệu của |độ chính xác − 0,5|.
    """
    subjects = sorted(set(real) & set.intersection(*(set(v) for v in sims.values())))
    if len(subjects) < 4:
        raise ValueError("cần ít nhất 4 người có cả ảnh thật lẫn ảnh mô phỏng")
    rng = np.random.default_rng(seed)
    perm = [subjects[i] for i in rng.permutation(len(subjects))]
    n_te = max(2, int(round(len(perm) * test_frac)))
    te, tr = perm[:n_te], perm[n_te:]
    flat = lambda d, ss: [im for s in ss for im in d[s]]
    scores = {}
    for kind, sim in sims.items():
        pr = np.zeros(len(flat(real, te)))
        ps = np.zeros(len(flat(sim, te)))
        for k in range(n_seeds):
            net = train_classifier(flat(real, tr), flat(sim, tr), patch, steps, device=device, seed=seed + k,
                                   width=width)
            pr += score_images(net, flat(real, te), patch, device) / n_seeds
            ps += score_images(net, flat(sim, te), patch, device) / n_seeds
        scores[kind] = (pr, ps)
    subj_r = np.array([s for s in te for _ in real[s]])
    out = {"n_train_subjects": len(tr), "n_test_subjects": len(te), "patch": patch, "kinds": {}, "pairs": {}}
    idx = np.random.default_rng(seed + 99).integers(0, len(te), size=(n_boot, len(te)))

    def boot_acc(kind):
        pr, ps = scores[kind]
        subj_s = np.array([s for s in te for _ in sims[kind][s]])
        sr = np.array([[pr[subj_r == s].sum(), (subj_r == s).sum()] for s in te], dtype=float)
        ss = np.array([[ps[subj_s == s].sum(), (subj_s == s).sum()] for s in te], dtype=float)
        a = sr[idx, 0].sum(1) / np.maximum(sr[idx, 1].sum(1), 1)
        b = ss[idx, 0].sum(1) / np.maximum(ss[idx, 1].sum(1), 1)
        return 0.5 * (a + 1 - b)

    boots = {k: boot_acc(k) for k in sims}
    for k in sims:
        acc = balanced_accuracy(*scores[k])
        lo, hi = np.quantile(boots[k], [0.025, 0.975])
        out["kinds"][k] = {"balanced_acc": acc, "lo": float(lo), "hi": float(hi), "dist_from_chance": abs(acc - 0.5),
                           "n_real_test": int(len(scores[k][0])), "n_sim_test": int(len(scores[k][1]))}
    kinds = list(sims)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            d = np.abs(boots[a] - 0.5) - np.abs(boots[b] - 0.5)
            lo, hi = np.quantile(d, [0.025, 0.975])
            out["pairs"][f"{a} - {b}"] = {
                "diff_dist_from_chance": out["kinds"][a]["dist_from_chance"] - out["kinds"][b]["dist_from_chance"],
                "lo": float(lo), "hi": float(hi)}
    return out
