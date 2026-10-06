# Dữ liệu đặt ở đâu, chạy script nào, kết quả nằm ở file nào

File này là bản đối chiếu duy nhất giữa **việc trong kế hoạch**, **script**, và **file kết quả**. Thứ tự chạy
chi tiết kèm lệnh đầy đủ: `docs/RUNBOOK.md`.

## 1. Đặt dữ liệu

Bố cục mặc định, tính từ thư mục gốc của project (thư mục `data/` không đưa lên git):

```
data/raw/
  AMI/                 BẮT BUỘC. 700 file, tên đúng mẫu NNN_<góc>_ear.jpg
                       NNN = 000..099 (mã người); góc = back, down, front, left, right, up, zoom
                       mọi ảnh cỡ 492×702 (rộng × cao). Để phẳng hoặc trong thư mục con đều được;
                       file không đúng mẫu tên bị bỏ qua.
  DIV2K_train_HR/      800 ảnh PNG, để phẳng (0001.png ... 0800.png). Cần cho tiền huấn luyện (T6 iv).
  DIV2K_valid_HR/      100 ảnh PNG, để phẳng. Validation của tiền huấn luyện; phép thử ngoài ảnh tai (T6 v).
  EarVN1.0/            <mã người>/<ảnh>   (ví dụ EarVN1.0/001.ALI_HD/001 (1).jpg). Cần cho N2 và S5.
  awex/                <mã người>/<ảnh>   (ví dụ awex/001/01.png). Chỉ dùng để test.
  landmarks/           ảnh và file .pts cùng tên, cùng thư mục (tùy chọn; chỉ khi làm N1 hoặc đo điểm mốc).
  Set5/                5 ảnh HR dạng PNG (tùy chọn; chỉ cho kiểm thử kho mô hình: EARSR_SET5=data/raw/Set5 pytest -q).
weights/               trọng số công bố; tạo bằng: bash scripts/get_weights.sh
```

Với EarVN1.0 và AWEx, **mã người là tên thư mục chứa ảnh**. Nếu bản tải về có thêm một tầng thư mục (ví dụ
`EarVN1.0/Images/001.ALI_HD/...`) thì trỏ `--root` vào tầng chứa các thư mục người. File chú thích (json, txt)
nằm lẫn trong đó không sao; script chỉ đọc `.jpg .jpeg .png .bmp`.

Dữ liệu ở nơi khác cũng được: mọi script nhận đường dẫn qua tham số (`--ami-raw`, `--root`, `--train-dir`...).

Kiểm tra trước khi chạy:

```bash
python scripts/check_data.py            # hoặc: --root /nơi/khác
```

## 2. Thư mục do project tạo ra

| Thư mục | Nội dung | Tạo bởi |
|---|---|---|
| `data/bench/ami/` | Benchmark AMI: `hr96/ hr144/ hr192/ hr244/` (ảnh đáp án PNG), `lr/hr<cỡ>_x<hệ số>_<suy giảm>/` (ảnh vào), `manifest.csv` | `build_benchmark.py`; ảnh LR do `evaluate.py`, `train.py` tự sinh khi cần |
| `data/bench/earvn/`, `awex_s20/`, `awex_s15/`, `div2k/` | Benchmark ngoài thực tế, cùng dạng; thêm `scan.csv`, `dropped_hr<cỡ>.csv` (ảnh bị loại và lý do) | `build_wild.py` |
| `runs/<mã lần chạy>/` | Một lần huấn luyện: `config.json`, `log.csv`, `summary.json`, `ckpt/best.pt`, `ckpt/last.pt`, `test_*.csv` | `train.py`, `pretrain.py` |
| `results/` | Mọi bảng kết quả (mục 3 dưới đây). Thư mục này được đưa lên git | các script đánh giá và tổng hợp |
| `deploy_jetson/onnx/` | File ONNX và `index.json` để chép sang thiết bị | `export_for_device.py` |
| `jobs/` | Danh sách lệnh và trạng thái hàng đợi | `make_jobs.py`, `run_queue.py` |
| `paper/tables/`, `paper/figures/` | Bảng LaTeX; hình PDF | `make_tables.py`, `compare_table.py`, `make_figures.py` |
| `results/tables/` | Bảng so sánh dạng CSV (kèm khoảng tin cậy và chênh lệch ghép cặp) | `compare_table.py` |

## 3. Việc nào, script nào, file kết quả nào

Hai loại file: **theo từng ảnh** (một dòng mỗi ảnh; là dữ liệu gốc, mọi thống kê tính lại được từ đây) và
**tổng hợp** (trung bình, khoảng tin cậy, kết luận).

### Giai đoạn 1, không huấn luyện (`bash scripts/run_stage1.sh data/raw/AMI data/raw/DIV2K_valid_HR`)

| Việc trong kế hoạch | Script | File theo từng ảnh | File tổng hợp (đọc file này) |
|---|---|---|---|
| P1 dựng benchmark | `build_benchmark.py` | `data/bench/ami/manifest.csv` | |
| **T2** mô hình có sẵn, bicubic và JPEG 75 | `evaluate.py` (tên cũ `run_t2.py`) rồi `summarize_t2.py` | `results/t2/<mô hình>__hr<cỡ>_x<hệ số>_<suy giảm>.csv` | `results/t2_summary/T2_summary.md` (đọc trước), `quality.csv` (bảng chất lượng), `ranking.json` (thứ hạng có đảo không, τ-b), `mde.csv` (mức chênh nhỏ nhất phát hiện được, để chỉnh ngưỡng) |
| **T6 (i), (v)** phép thử ngữ cảnh | `run_t6_context.py` rồi `summarize_context.py` | `results/t6_context/<mô hình>__<bộ ảnh>__<cấu hình>.csv` | `results/t6_summary/T6_context_summary.md`, `context.csv` |
| **T3** độ trễ trên máy này (so tương đối) | `bench_local.py` | | `results/latency_local.csv` |
| **T3** độ trễ trên Jetson | `export_for_device.py`, rồi trên Jetson `deploy_jetson/bench_trtexec.py` | `deploy_jetson/logs/` (log thô) | `latency_trt.csv` trên Jetson; chép về thành `results/latency_jetson.csv` |
| Bảng cho bài (tự động) | `make_tables.py` | | `paper/tables/t2_hr<cỡ>_x<hệ số>.tex`, `latency_*.tex`, `t6_context.tex`, `n2_realism.tex`, `viewer_<phần>.tex` |
| Bảng so sánh giữa các phương pháp (bảng 4, 6, 7, 8, 11) | `compare_table.py` | các file theo ảnh của từng phương pháp | `results/tables/<tên>.csv`, `paper/tables/<tên>.tex` |
| Hình cho bài | `make_figures.py` (tự động: hình 2, 3, 4; lệnh con `cost`: hình 5; `grid`: hình 1, 6) | | `paper/figures/*.pdf` và `*.png` |

Cột của file theo từng ảnh: `model, dataset, subject, view, key, fold, scale, tier, degrade, psnr_y, ssim_y,
psnr_rgb, psnr_y_c, ssim_y_c` (hậu tố `_c` = vùng giữa ảnh), `ms_ssim_y, ms_ssim_scales, gmsd, grad_psnr, lr_psnr_y`
(luôn có), `lpips, dists` (nếu cài được), `stlpips, topiq_fr, fsim, vif, pieapp` (`--more-metrics`), `host_ms`,
`lr_psnr_y` chặn trên ở 100 dB (ảnh SR thu nhỏ lại trùng khít đáp án thu nhỏ; file chấm trước 06/10/2026 ghi `inf` ở các ảnh đó và bảng tổng hợp đọc `inf` thành 100).
`device` (thiết bị đã chạy ảnh đó: `cuda`, hoặc `cpu` nếu lúc ấy GPU hết bộ nhớ; `host_ms` chỉ so được trong cùng thiết bị); thêm
`ridge_false, ridge_missed, ridge_f1` (`--ridge`), `lm_dev, lm_floor_noise, lm_floor_jpeg` (`--landmark-ckpt`),
`psnr_y_box` (`--boxes`), `niqe, maniqa, musiq, clipiqa, ntire_score` (`--nr`).

### Giai đoạn 2, có huấn luyện

| Việc trong kế hoạch | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| Mọi lần huấn luyện | `train.py` (qua `make_jobs.py` + `run_queue.py`) | `runs/<mã>/test_hr<cỡ>_x<hệ số>_<suy giảm>.csv` (chỉ người test của fold đó) | `runs/<mã>/summary.json`, `runs/<mã>/log.csv` (đường học); **`results/runs.csv`** (sổ ghi: một dòng mỗi lần chạy, trạng thái, PSNR validation và test) |
| **N2 sớm** (khối `n2`): ba kiểu suy giảm lúc huấn luyện, fold 2 | `make_jobs.py n2 --degrade-params ...`, rồi `evaluate.py --runs` trên EarVN1.0 và trên AMI với từng kiểu | `results/n2/<mã>__hr<cỡ>_x4_<suy giảm>.csv` | `check_criteria.py gain --name n2-...` → `results/criteria/` |
| **Kiểu đệm, phép thử rẻ** (khối `pad`): trọng số công bố, ba kiểu đệm | `make_jobs.py pad` | `runs/T6pad_*/test_*.csv` | `check_criteria.py n5b-gain` |
| Dò tốc độ học (khối `lr`) | như trên | không chấm test | cột `best_val_psnr_y` trong `results/runs.csv` |
| **T6 (ii)** ba giao thức → N5a | khối `t6ii`, rồi `check_criteria.py n5a` | `runs/T6ii_*/test_*.csv` | `results/criteria/n5a.json` |
| **T6 (iii)** có cần tiền huấn luyện | khối `t6iii`, rồi `check_criteria.py gain` | `runs/T6iii_*/test_*.csv` | `results/criteria/gain.json` (đặt tên bằng `--name`) |
| Tiền huấn luyện rút gọn (khối `pre`) | `pretrain.py` | | `runs/PRE_*/log.csv`, `summary.json`; checkpoint ở `ckpt/best.pt`, `ckpt/step<N>.pt` (50% và 100% ngân sách) |
| **T6 (iv)** năm biến thể → N5b (b) | khối `t6iv`, rồi `check_criteria.py n5b-gain` | `runs/T6iv_*/test_*.csv` | `results/criteria/n5b-gain.json` |
| **T6 (v)** phép thử tương tác → N5b (c) | `build_wild.py --subject-per-image`, `evaluate.py --runs runs/PRE_*`, `check_criteria.py n5b-interaction` | `results/t6v/<mã>__hr<cỡ>_x4_bic.csv` | `results/criteria/n5b-interaction.json` |
| Ghép độ trễ (mốc nới rộng; hình dạng wide, deep) | `match_latency.py` | | in ra màn hình: variant `zero-cNN` cần dùng |
| **S2** ô chính, 5 fold | khối `s2` | `runs/S2_*/test_*.csv` | `results/runs.csv`; so sánh bằng `check_criteria.py gain` |
| **S4** đường chất lượng theo cỡ ảnh vào | `run_size_sweep.py` | `results/size_sweep/per_image.csv` | `results/size_sweep/summary.csv` |

### N2 và ảnh ngoài thực tế

| Việc | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| Chia vai người trong EarVN1.0 | `fit_degradation.py` (lần đầu tự tạo) | | `splits/earvn_roles.json` (commit) |
| Ước lượng suy giảm (N2) | `fit_degradation.py` | | `configs/degrade_estimated.json` (tham số, commit), `configs/degrade_estimated.diagnostics.json` |
| N2 (a) bộ phân loại "mô phỏng hay thật" | `realism_classifier.py` | | `results/n2_realism.json` |
| **S5** test trên EarVN1.0, AWEx | `build_wild.py`, `evaluate.py --folds none` | `results/s5/<mô hình>__hr<cỡ>_x4_<suy giảm>.csv` | tính bằng `check_criteria.py gain` |

### Tùy chọn (sau điểm kiểm tra 2)

| Việc | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| **T4** cổng oracle → giữ hay bỏ N3 | `run_t4_oracle.py` | `results/t4/t4_per_image.csv` | `results/t4/t4_summary.json` (khóa `decision`) |
| N3 hai giai đoạn | `train.py --objective n3s1A` rồi `n3s2` | `runs/<mã>/test_*_opF.csv`, `_opP.csv`, `_headS.csv`, `_headT.csv` | `runs/<mã>/operating_points.json`; `check_criteria.py n3` → `results/criteria/n3.json` |
| Bộ dò điểm mốc | `train_landmarks.py` | | `weights/lm_*.pt` |
| Nhãn giả, hộp bao, sàn nhiễu | `landmark_tools.py pseudo / boxes / floor` | | `data/ami_lm.npz`, `data/bench/ami/boxes.csv`, sàn nhiễu in ra màn hình |
| **T5** đầu phụ N1 | `train.py --objective aux`, `evaluate.py --landmark-ckpt --ridge`, `check_criteria.py n1` | `runs/<mã>/test_*.csv`, file của `evaluate.py` | `results/criteria/n1.json` |
| **S7** khảo sát người xem | `evaluate.py --save-sr`, `viewer_study.py make / analyze` | `results/viewer/<phần>/answers_long.csv` | `results/viewer/<phần>/analysis.csv`; `study.html` gửi người xem, `key.csv` giữ lại |

## 4. Mã lần chạy

`{thí nghiệm}_{thân}-{biến thể}_{tiền huấn luyện}_{giao thức}_x{hệ số}_hr{cỡ}_{suy giảm}_f{fold}`

Ví dụ `T6iv_span-replicate_ps20_rand_x4_hrall_bic_f3`: phép thử T6 (iv), thân SPAN đệm lặp viền, tiền huấn
luyện rút gọn 20%, giao thức tỉ lệ ngẫu nhiên, ×4, một mô hình cho mọi cỡ, suy giảm bicubic, fold 3.
Phần sau dấu `+` trong biến thể là nhãn: `+gan`, `+ldl`, `+aux`, `+xearvn` (có ảnh thêm), `+lr1e-4` (`--tag`).

## 5. File nào quyết định điều gì

| Quyết định trong kế hoạch | Đọc file |
|---|---|
| Điểm kiểm tra 1: SR có dư địa không; thứ hạng có đảo không | `results/t2_summary/T2_summary.md`, `ranking.json` |
| Điểm kiểm tra 1: hiệu ứng viền có thật không | `results/t6_summary/T6_context_summary.md` |
| Điểm kiểm tra 1: mô hình nào real-time; kiểu đệm nào chạy được | `results/latency_jetson.csv` |
| Chỉnh ngưỡng một lần sau T2 | `results/t2_summary/mde.csv` → sửa `configs/criteria.yaml`, ghi mục `history` |
| Điểm kiểm tra 2: giao thức, N5b, N2 | `results/criteria/n5a.json`, `n5b-gain.json`, `n5b-interaction.json`, `results/n2_realism.json` |
| Giữ hay bỏ N3, N1 | `results/t4/t4_summary.json`, `results/criteria/n1.json`, `n3.json` |

Mỗi file trong `results/criteria/` có khóa `pass` (true, false) và ghi kèm phiên bản `criteria.yaml` đã dùng.
