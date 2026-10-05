"""Vòng huấn luyện dùng chung cho tiền huấn luyện và tinh chỉnh.

- Adam, lịch cosine. Có EMA của trọng số (tùy chọn). Loss do một ``Objective``
  quyết định (mặc định L1; xem ``objectives.py``).
- Checkpoint được chọn theo số đo trên tập validation (mặc định PSNR-Y), không
  bao giờ theo test. Mục tiêu có GAN giữ checkpoint cuối (``select = "last"``).
- Tiếp tục được từ checkpoint gần nhất (``last.pt``).
- Mỗi thư mục lần chạy giữ cấu hình, mã commit, log theo bước.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    run_id: str
    out_dir: str = "runs"
    iters: int = 20000
    batch_size: int = 64
    lr: float = 5e-4
    lr_min: float = 1e-6
    warmup_iters: int = 0
    weight_decay: float = 0.0
    ema: float = 0.999            # 0 để tắt
    val_every: int = 1000
    save_every: int = 1000
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    amp: bool = False
    grad_clip: float = 0.0
    select: str = "auto"          # auto: theo mục tiêu; max: validate() cao nhất; last: checkpoint cuối
    log_every: int = 0            # in tiến độ mỗi N bước (0: chỉ in ở mỗi lần validation)
    checkpoints_at: list = field(default_factory=list)   # lưu thêm ở các bước này (ví dụ 50% ngân sách)
    extra: dict = field(default_factory=dict)             # ghi kèm: giao thức, dữ liệu, mã băm fold ...


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_iters and step < cfg.warmup_iters:
        return cfg.lr * (step + 1) / cfg.warmup_iters
    t = (step - cfg.warmup_iters) / max(1, cfg.iters - cfg.warmup_iters)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * min(1.0, t)))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ps, p in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(d).add_(p.detach(), alpha=1 - d)
        for bs, b in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(b)


def train(model: nn.Module, train_set, validate, cfg: TrainConfig, objective=None) -> dict:
    """Huấn luyện ``model`` (nhận và trả tensor [0, 1]).

    ``validate(model) -> float``: số đo trên tập validation, càng cao càng tốt
    (mặc định PSNR-Y). ``objective``: một ``Objective``; None là L1.
    Trả về dict tóm tắt; checkpoint được chọn ở ``best.pt``.
    """
    from .objectives import L1Objective

    objective = objective or L1Objective()
    run_dir = Path(cfg.out_dir) / cfg.run_id
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    model = model.to(device)
    objective.to(device)
    objective.prepare(model)
    params = objective.parameters(model)
    if not params:
        raise RuntimeError("không có tham số nào để học")
    n_all = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad or ".eval_conv." not in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema) if cfg.ema > 0 else None
    use_amp = bool(cfg.amp and device.startswith("cuda") and objective.supports_amp)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    select = objective.select if cfg.select == "auto" else cfg.select
    if select not in ("max", "last"):
        raise ValueError("select phải là auto, max hoặc last")
    step, best, best_step = 0, -float("inf"), -1
    last = run_dir / "ckpt" / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if ema is not None and st.get("ema") is not None:
            ema.shadow.load_state_dict(st["ema"])
        objective.load_state_dict(st.get("objective") or {})
        step, best, best_step = st["step"], st["best"], st["best_step"]
    meta = {**asdict(cfg), "device_used": device, "git_commit": git_commit(), "objective": objective.name,
            "select_used": select, "amp_used": use_amp, "torch": torch.__version__,
            "n_params_trainable": n_train, "n_params_model": n_all, "started_at_step": step}
    with open(run_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss", "lr", "val_psnr_y", "elapsed_s", "terms"])

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow.state_dict() if ema is not None else None,
                    "objective": objective.state_dict(), "step": step, "best": best, "best_step": best_step}, path)

    def eval_model() -> nn.Module:
        return ema.shadow if ema is not None else model

    # Mẫu được đánh số toàn cục: bước k dùng các chỉ số [k·B, (k+1)·B). Nhờ đó
    # huấn luyện tiếp tục đúng chỗ dừng và không lặp lại mẫu giữa các "epoch".
    bs = cfg.batch_size
    loader = DataLoader(train_set, batch_size=bs, sampler=range(step * bs, cfg.iters * bs),
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=device.startswith("cuda"),
                        persistent_workers=False)
    t0 = time.time()
    start_step = step
    running, n_run, terms = 0.0, 0, {}
    needs_ema = bool(getattr(objective, "w_ldl", 0) > 0)
    model.train()
    for batch in loader:
        cur_lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        batch = [t.to(device, non_blocking=True) for t in batch]
        ema_model = None
        if needs_ema and ema is not None:
            ema_model = ema.shadow.eval()   # eval() gộp lại các nhánh tham số hóa theo trọng số EMA mới nhất
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss, logs = objective.loss(model, batch, step, ema_model)
        if not torch.isfinite(loss):
            log_f.close()
            raise FloatingPointError(f"loss không hữu hạn ở bước {step} ({logs}); giảm tốc độ học")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        logs.update(objective.after_step(step))
        if ema is not None:
            ema.update(model)
        step += 1
        running += float(loss.item())
        n_run += 1
        for k, v in logs.items():
            terms[k] = terms.get(k, 0.0) + v
        if cfg.log_every and step % cfg.log_every == 0:
            el = time.time() - t0
            print(f"[{cfg.run_id}] bước {step}/{cfg.iters}  loss {running / n_run:.5f}  "
                  f"{(step - start_step) / max(el, 1e-9):.2f} bước/s", flush=True)
        do_val = step % cfg.val_every == 0 or step == cfg.iters
        if do_val:
            em = eval_model()
            em.eval()
            v = float(validate(em))
            model.train()
            if v > best or select == "last":
                best, best_step = v, step
                torch.save({"model": em.state_dict(), "step": step, "val_psnr_y": v}, run_dir / "ckpt" / "best.pt")
            el = time.time() - t0
            log.writerow([step, running / max(1, n_run), cur_lr, v, round(el, 1),
                          json.dumps({k: round(x / max(1, n_run), 6) for k, x in terms.items()})])
            log_f.flush()
            rate = (step - start_step) / max(el, 1e-9)
            print(f"[{cfg.run_id}] bước {step}/{cfg.iters}  loss {running / max(1, n_run):.5f}  val {v:.4f}  "
                  f"(chọn: {best:.4f} ở bước {best_step})  {rate:.2f} bước/s, còn ~{(cfg.iters - step) / max(rate, 1e-9) / 60:.0f} phút",
                  flush=True)
            running, n_run, terms = 0.0, 0, {}
        if step in cfg.checkpoints_at:
            torch.save({"model": eval_model().state_dict(), "step": step}, run_dir / "ckpt" / f"step{step}.pt")
        if step % cfg.save_every == 0 or step == cfg.iters:
            save(last)
    log_f.close()
    summary = {"run_id": cfg.run_id, "best_val_psnr_y": best, "best_step": best_step, "iters": step,
               "elapsed_s": round(time.time() - t0, 1), "objective": objective.name, "select": select}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def curve_is_flat(log_csv: str | Path, last_frac: float = 0.10, tol_db: float = 0.02) -> dict:
    """Điều kiện 'đường học đã phẳng' của ngân sách tiền huấn luyện rút gọn:
    ``last_frac`` số bước cuối thêm dưới ``tol_db`` dB trên validation."""
    import pandas as pd

    d = pd.read_csv(log_csv).dropna(subset=["val_psnr_y"])
    if len(d) < 3:
        return {"flat": False, "reason": "quá ít điểm validation"}
    end = d.step.max()
    before = d[d.step <= end * (1 - last_frac)]
    if before.empty:
        return {"flat": False, "reason": "không có điểm trước đoạn cuối"}
    gain = float(d.val_psnr_y.iloc[-1] - before.val_psnr_y.iloc[-1])
    return {"flat": gain < tol_db, "gain_db": gain, "last_frac": last_frac, "tol_db": tol_db}
