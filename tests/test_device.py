"""Ưu tiên GPU, hết bộ nhớ thì xuống CPU, rồi quay lại GPU.

Máy kiểm thử không có GPU, nên "thiết bị chính" ở đây là ``cpu:0`` (torch coi là một
thiết bị hợp lệ) và lỗi hết bộ nhớ được dựng giả đúng kiểu torch ném ra.
"""
import numpy as np
import pytest
import torch

from earsr.device import GpuFirst, describe_device, is_oom

GPU = "cpu:0"


def _oom():
    # torch.cuda.OutOfMemoryError có từ torch 1.13 (torch.OutOfMemoryError chỉ có từ 2.5; trên bản mới là cùng một lớp)
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Flaky:
    """Đối tượng giả: ``fail`` lần gọi kế tiếp trên thiết bị chính sẽ hết bộ nhớ."""

    def __init__(self):
        self.fail = 0
        self.fail_build = 0
        self.built: list[str] = []
        self.tried: list[str] = []

    def factory(self, device):
        if device == GPU and self.fail_build:
            self.fail_build -= 1
            raise _oom()
        self.built.append(device)
        return device

    def call(self, obj, device):
        assert obj == device
        self.tried.append(device)
        if device == GPU and self.fail:
            self.fail -= 1
            raise _oom()
        return f"ok@{device}"


def _runner(f, clock, **kw):
    return GpuFirst(f.factory, GPU, retry_s=10, max_retry_s=40, clock=clock, log=None, **kw)


def test_is_oom_recognises_torch_and_cuda_messages():
    assert is_oom(_oom())
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_oom(RuntimeError("size mismatch")) and not is_oom(ValueError("out of memory"))


def test_cpu_only_never_touches_another_device():
    f = Flaky()
    r = GpuFirst(f.factory, "cpu", log=None)
    assert r.run(f.call) == "ok@cpu" and f.built == ["cpu"] and r.last_device == "cpu" and r.fallbacks == 0


def test_stays_on_gpu_while_it_works():
    f, c = Flaky(), Clock()
    r = _runner(f, c)
    assert [r.run(f.call) for _ in range(3)] == [f"ok@{GPU}"] * 3
    assert f.built == [GPU] and r.calls == {"primary": 3, "cpu": 0}      # bản CPU không bao giờ được dựng


def test_falls_back_then_returns_to_gpu():
    f, c = Flaky(), Clock()
    r = _runner(f, c)
    f.fail = 1
    assert r.run(f.call) == "ok@cpu" and r.last_device == "cpu" and r.fallbacks == 1
    f.tried.clear()
    c.t = 9.9                                    # còn trong quãng chờ: không đụng tới GPU
    assert r.run(f.call) == "ok@cpu" and f.tried == ["cpu"]
    c.t = 10.0                                   # hết quãng chờ: thử GPU, chạy được, ở lại GPU
    assert r.run(f.call) == f"ok@{GPU}" and r.last_device == GPU
    assert r.run(f.call) == f"ok@{GPU}" and r.calls == {"primary": 2, "cpu": 2}


def test_backoff_doubles_is_capped_and_resets():
    f, c = Flaky(), Clock()
    r = _runner(f, c)
    f.fail = 10
    waits = []
    for _ in range(5):
        before = len(f.tried)
        assert r.run(f.call) == "ok@cpu"
        assert f.tried[before:] == [GPU, "cpu"]                    # mỗi lần thử lại đều thử GPU trước
        waits.append(r._next_try - c.t)
        c.t = r._next_try
    assert waits == [10, 20, 40, 40, 40]
    f.fail = 0
    assert r.run(f.call) == f"ok@{GPU}"
    f.fail = 1
    r.run(f.call)
    assert r._next_try - c.t == 10                                 # một lần thành công đưa quãng chờ về ban đầu


def test_out_of_memory_while_building_on_gpu():
    f, c = Flaky(), Clock()
    f.fail_build = 1
    r = _runner(f, c).warm()
    assert f.built == ["cpu"] and r.last_device == "cpu" and r.calls == {"primary": 0, "cpu": 0}
    assert r.run(f.call) == "ok@cpu"
    c.t = 10
    assert r.run(f.call) == f"ok@{GPU}" and f.built == ["cpu", GPU]   # dựng lại trên GPU khi GPU trống


def test_other_errors_are_not_swallowed():
    r = GpuFirst(lambda d: d, GPU, log=None)
    with pytest.raises(ZeroDivisionError):
        r.run(lambda obj, d: 1 / 0)
    with pytest.raises(KeyError):
        GpuFirst(lambda d: {}["x"], GPU, log=None).warm()


def test_describe_device_warns_when_no_gpu():
    s = describe_device("cpu")
    assert "cpu" in s and (torch.cuda.is_available() or "CẢNH BÁO" in s)


# ----------------------------------------------------------------- chấm ở FP32 đầy đủ (TF32 tắt)

def _tf32():
    return torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32


@pytest.fixture
def tf32_on():
    """Giả lập mặc định của GPU đời mới: TF32 bật cho tích chập."""
    old = _tf32()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def test_full_precision_turns_tf32_off_and_restores(tf32_on):
    from earsr.device import full_precision

    with full_precision():
        assert _tf32() == (False, False)
        with full_precision():
            assert _tf32() == (False, False)
        assert _tf32() == (False, False)                    # khối lồng nhau không bật lại sớm
    assert _tf32() == (True, True)
    with pytest.raises(ZeroDivisionError):
        with full_precision():
            1 / 0
    assert _tf32() == (True, True)                          # có lỗi vẫn trả lại trạng thái cũ


class _Spy(torch.nn.Module):
    """Mô hình ×4 giả ghi lại trạng thái TF32 ở mỗi lượt gọi."""

    seen: list = []

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, gate=None):
        _Spy.seen.append(_tf32())
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.w


def test_every_scoring_path_runs_at_full_precision(tf32_on, monkeypatch, tmp_path):
    import earsr.models.registry as reg
    from earsr.eval.infer import evaluate_on_bench, make_predictor
    from earsr.eval.metrics import PerceptualMetrics
    from earsr.io import imwrite_rgb
    from earsr.train import finetune

    class Spec:
        scale = 4

    monkeypatch.setitem(reg.SPECS, "spy", Spec())
    monkeypatch.setattr(reg, "build_model", lambda name, **kw: _Spy().eval())
    lr = np.random.default_rng(0).integers(0, 256, (12, 10, 3), dtype=np.uint8)
    hr = np.repeat(np.repeat(lr, 4, 0), 4, 1)

    # 1. hàm dự đoán của kho mô hình (T2, phép thử ngữ cảnh)
    _Spy.seen = []
    make_predictor("spy", 4, "cpu")(lr)
    assert _Spy.seen == [(False, False)] and _tf32() == (True, True)

    # 2. validation lúc huấn luyện
    monkeypatch.setattr(finetune, "val_pairs", lambda *a, **k: [(lr, hr)])
    _Spy.seen = []
    assert finetune.make_validator("x", "x", 2, 4, (40,))(_Spy()) > 60
    assert _Spy.seen == [(False, False)] and _tf32() == (True, True)

    # 3. chấm bằng hàm dự đoán tự cấp (cách train.py chấm test) và số đo cảm nhận đi kèm
    import csv

    bench = tmp_path / "bench"
    imwrite_rgb(bench / "lr" / "hr40_x4_bic" / "000_front.png", lr)
    imwrite_rgb(bench / "hr40" / "000_front.png", hr)
    with open(bench / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "subject", "view", "tier", "file"])
        w.writeheader()
        w.writerow({"dataset": "ami", "subject": "000", "view": "front", "tier": 40, "file": "hr40/000_front.png"})
    spy, seen_metric = _Spy(), []
    pm = PerceptualMetrics("cpu", want=())
    pm._add("fake", lambda d: d, lambda net, x, y: seen_metric.append(_tf32()) or 0.0)

    @torch.no_grad()
    def own_predict(img):
        from earsr.io import to_tensor, to_uint8

        return to_uint8(spy(to_tensor(img)))

    _Spy.seen = []
    evaluate_on_bench("own", bench, 40, 4, "bic", None, tmp_path / "o.csv", predictor=own_predict, perceptual=pm)
    assert _Spy.seen == [(False, False)] and seen_metric == [(False, False)] and _tf32() == (True, True)


# ----------------------------------------------------------------- nối vào đường suy luận

class _FakeSR(torch.nn.Module):
    """Mô hình ×4 giả (nội suy gần nhất); hết bộ nhớ theo lịch khi đang ở thiết bị chính."""

    schedule: list = []     # dùng chung cho mọi bản sao: True = lượt gọi kế tiếp trên GPU hết bộ nhớ

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.where = "cpu"

    def to(self, device):
        self.where = str(device)
        return super().to(device)

    def forward(self, x):
        if self.where == GPU and _FakeSR.schedule and _FakeSR.schedule.pop(0):
            raise _oom()
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") * self.w


@pytest.fixture
def fake_zoo(monkeypatch):
    import earsr.models.registry as reg

    class Spec:
        scale = 4

    monkeypatch.setitem(reg.SPECS, "fake", Spec())
    monkeypatch.setattr(reg, "build_model", lambda name, **kw: _FakeSR().eval())
    _FakeSR.schedule = []
    yield
    _FakeSR.schedule = []


def test_predictor_survives_out_of_memory(fake_zoo):
    from earsr.eval.infer import make_predictor

    lr = np.random.default_rng(0).integers(0, 256, (6, 5, 3), dtype=np.uint8)
    want = np.repeat(np.repeat(lr, 4, 0), 4, 1)
    p = make_predictor("fake", 4, GPU)
    p.runner.log, p.runner.clock = None, Clock()
    _FakeSR.schedule = [False, True, False]
    devices = []
    for i in range(3):
        if i == 2:
            p.runner.clock.t = 1000.0             # hết quãng chờ
        assert np.array_equal(p(lr), want)
        devices.append(p.runner.last_device)
    assert devices == [GPU, "cpu", GPU] and p.runner.fallbacks == 1
    with pytest.raises(ValueError):
        make_predictor("fake", 2, GPU)


def test_evaluate_records_the_device_of_each_image(fake_zoo, tmp_path):
    import csv

    from earsr.eval.infer import evaluate_on_bench, make_predictor
    from earsr.io import imwrite_rgb

    rng = np.random.default_rng(1)
    bench = tmp_path / "bench"
    rows = []
    for i in range(4):
        lr = rng.integers(0, 256, (12, 10, 3), dtype=np.uint8)
        imwrite_rgb(bench / "lr" / "hr40_x4_bic" / f"{i:03d}_front.png", lr)
        imwrite_rgb(bench / "hr40" / f"{i:03d}_front.png", np.repeat(np.repeat(lr, 4, 0), 4, 1))
        rows.append({"dataset": "ami", "subject": f"{i:03d}", "view": "front", "tier": 40, "file": f"hr40/{i:03d}_front.png"})
    with open(bench / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = make_predictor("fake", 4, GPU)
    p.runner.log, p.runner.clock = None, Clock()            # đồng hồ đứng yên: sau khi hết bộ nhớ thì ở lại CPU
    _FakeSR.schedule = [False, True]
    out = evaluate_on_bench("fake", bench, 40, 4, "bic", None, tmp_path / "o.csv", device=GPU, predictor=p)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["device"] for r in got] == [GPU, "cpu", "cpu", "cpu"]
    assert all(float(r["psnr_y"]) > 60 for r in got)        # đầu ra đúng trên cả hai thiết bị


def test_latency_benchmark_waits_and_retries_but_never_measures_on_cpu(monkeypatch, tmp_path):
    """Đo độ trễ: hết bộ nhớ thì đo lại trên cùng thiết bị; hết lượt thử thì ghi dòng lỗi và đi tiếp."""
    import csv
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench_local", root / "scripts" / "bench_local.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = {"bicubic": 2, "spanvar_zero": 99}            # số lần hết bộ nhớ của từng mô hình
    seen = []

    def fake_measure(model, hw, device, warmup, runs, half):
        name = "bicubic" if isinstance(model, mod.Bicubic) else f"spanvar_{model.padding_mode[:4]}".replace("zeros", "zero")
        seen.append((name, device))
        if plan.get(name, 0) > 0:
            plan[name] -= 1
            raise _oom()
        return {"median_ms": 1.0, "p95_ms": 2.0, "device": device}

    monkeypatch.setattr(mod, "measure_latency", fake_measure)
    monkeypatch.setattr(mod, "all_variants", lambda: ["zero", "replicate"])
    out = tmp_path / "lat.csv"
    monkeypatch.setattr(sys, "argv", ["bench_local.py", "--out", str(out), "--device", GPU, "--groups", "khong-co",
                                      "--oom-retries", "2", "--oom-wait", "0"])
    mod.main()
    with open(out, newline="") as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    assert rows["bicubic"]["median_ms"] == "1.0" and not rows["bicubic"]["error"]      # đo được ở lần thử thứ ba
    assert "hết bộ nhớ" in rows["spanvar_zero"]["error"] and rows["spanvar_zero"]["median_ms"] == ""
    assert rows["spanvar_replicate"]["median_ms"] == "1.0"                              # mô hình sau vẫn được đo
    assert {d for _, d in seen} == {GPU} and [n for n, _ in seen].count("spanvar_zero") == 3


def test_perceptual_metric_falls_back_instead_of_nan():
    from earsr.eval.metrics import PerceptualMetrics

    pm = PerceptualMetrics(GPU, want=())
    state = {"fail": 1, "built": []}

    def factory(device):
        state["built"].append(device)
        return device

    def score(net, x, y):
        if net == GPU and state["fail"]:
            state["fail"] -= 1
            raise _oom()
        return float((x - y).abs().mean()) + (0.0 if net == GPU else 0.0)

    pm._add("fake", factory, score)
    pm.fn["fake"].log, pm.fn["fake"].clock = None, Clock()
    a = np.zeros((8, 8, 3), np.uint8)
    b = np.full((8, 8, 3), 255, np.uint8)
    assert pm.available == ["fake"]
    assert pm(a, b) == {"fake": 1.0}                        # lượt đầu hết bộ nhớ trên GPU: vẫn ra số, không NaN
    assert state["built"] == [GPU, "cpu"] and pm.fn["fake"].last_device == "cpu"
    pm.fn["fake"].clock.t = 1000.0
    assert pm(a, b) == {"fake": 1.0} and pm.fn["fake"].last_device == GPU
    # lỗi khác hết bộ nhớ vẫn cho NaN như trước
    pm._add("bad", lambda d: d, lambda net, x, y: 1 / 0)
    pm.fn["bad"].log = None
    assert np.isnan(pm(a, b)["bad"])
