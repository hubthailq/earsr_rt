"""Ưu tiên GPU; GPU hết bộ nhớ thì chạy trên CPU, rồi thử lại GPU sau một quãng chờ.

Dùng cho các bước suy luận (chấm mô hình, số đo cảm nhận) trên máy có GPU dùng
chung: một lượt hết bộ nhớ không làm chết tiến trình, và khi GPU trống trở lại thì
việc quay về GPU là tự động. Không dùng cho huấn luyện (chạy trên CPU chậm hơn
hàng chục lần) và không dùng cho đo độ trễ (số đo trên hai thiết bị không so được).

Chỉ lỗi hết bộ nhớ mới được đỡ; mọi lỗi khác vẫn nổi lên như cũ.
"""
from __future__ import annotations

import time
from typing import Callable

import torch

_OOM_TYPES = tuple({t for t in (getattr(torch, "OutOfMemoryError", None),
                                getattr(torch.cuda, "OutOfMemoryError", None)) if t is not None})


def is_oom(e: BaseException) -> bool:
    """True nếu ``e`` là lỗi hết bộ nhớ của thiết bị (kể cả dạng RuntimeError của CUDA)."""
    return isinstance(e, _OOM_TYPES) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def free_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe_device(device: str, on_oom: str = "hết bộ nhớ thì tạm chạy trên CPU rồi tự quay lại GPU") -> str:
    """Một dòng cho biết lần chạy sẽ dùng thiết bị nào (in ở đầu script)."""
    device = str(device)
    if not device.startswith("cuda"):
        if torch.cuda.is_available():
            return f"thiết bị: {device} (máy có GPU nhưng không được chọn)"
        return f"thiết bị: {device}. CẢNH BÁO: torch không thấy GPU nào, mọi thứ chạy trên CPU"
    try:
        idx = torch.device(device).index or 0
        free, total = torch.cuda.mem_get_info(idx)
        return (f"thiết bị: {device} ({torch.cuda.get_device_name(idx)}, trống {free / 2 ** 30:.1f} / "
                f"{total / 2 ** 30:.1f} GB); {on_oom}")
    except Exception as e:  # không hỏi được GPU (ví dụ đã đầy tới mức không tạo được ngữ cảnh)
        return f"thiết bị: {device} (chưa hỏi được trạng thái GPU: {type(e).__name__}); {on_oom}"


class GpuFirst:
    """Giữ một đối tượng (mô hình, số đo) trên thiết bị chính và, khi cần, một bản trên CPU.

    ``factory(device)`` dựng đối tượng trên ``device``. ``run(call)`` gọi
    ``call(obj, device)`` trên thiết bị chính; nếu hết bộ nhớ (lúc dựng hoặc lúc gọi)
    thì gọi lại trên CPU và chờ ``retry_s`` giây mới thử thiết bị chính lần nữa. Quãng
    chờ nhân đôi sau mỗi lần thử lại không thành, tối đa ``max_retry_s``; một lần chạy
    được trên thiết bị chính đưa nó về ``retry_s``.

    ``call`` phải trả về giá trị không còn nằm trên thiết bị (số, mảng numpy, tensor CPU).
    """

    def __init__(self, factory: Callable[[str], object], device: str = "cpu", retry_s: float = 15.0,
                 max_retry_s: float = 240.0, name: str = "", clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] | None = print):
        self.factory = factory
        self.primary = str(device)
        self.single = self.primary == "cpu"
        self.retry_s, self.max_retry_s = float(retry_s), float(max_retry_s)
        self.name = name
        self.clock, self.log = clock, log
        self._obj: dict = {}
        self._wait = self.retry_s
        self._next_try = float("-inf")
        self._on_cpu = False
        self.last_device: str | None = None
        self.calls = {"primary": 0, "cpu": 0}
        self.fallbacks = 0

    def _get(self, device: str):
        if device not in self._obj:
            self._obj[device] = self.factory(device)
        return self._obj[device]

    def _say(self, msg: str) -> None:
        if self.log is not None:
            self.log(f"[thiết bị] {self.name + ': ' if self.name else ''}{msg}")

    def warm(self) -> "GpuFirst":
        """Dựng đối tượng ngay (trên thiết bị chính, hoặc CPU nếu hết bộ nhớ) để lỗi nạp lộ ra sớm."""
        self.run(lambda obj, device: None, _count=False)
        return self

    def run(self, call: Callable[[object, str], object], _count: bool = True):
        if not self.single and self.clock() >= self._next_try:
            failed = False
            try:
                out = call(self._get(self.primary), self.primary)
            except Exception as e:
                if not is_oom(e):
                    raise
                failed = True
            # Ra khỏi khối except rồi mới dọn: biến lỗi giữ khung gọi, tức giữ cả tensor trên GPU.
            if not failed:
                if self._on_cpu:
                    self._say(f"{self.primary} dùng lại được, quay về {self.primary}")
                self._on_cpu, self._wait = False, self.retry_s
                self.last_device = self.primary
                self.calls["primary"] += _count
                return out
            free_gpu_cache()
            self.fallbacks += 1
            self._next_try = self.clock() + self._wait
            self._say(f"{self.primary} hết bộ nhớ, chạy trên CPU; thử lại {self.primary} sau {self._wait:.0f} giây")
            self._wait = min(self._wait * 2, self.max_retry_s)
            self._on_cpu = True
        out = call(self._get("cpu"), "cpu")
        self.last_device = "cpu"
        self.calls["cpu"] += _count
        return out
