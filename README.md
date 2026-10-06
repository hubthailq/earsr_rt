# earsr_rt

Mã nguồn cho bài "SR real-time cho ảnh tai độ phân giải thấp". Project này mới hoàn toàn, tách khỏi
`earsr_project_span`. Kế hoạch đi kèm: `claude/paper-plan-ridgesr-2026-10-05.md` (bản 22).

## Trạng thái (05/10/2026)

Mã cho mọi giai đoạn của kế hoạch đã viết xong. **Chưa có lần chạy nào trên GPU và chưa có lần huấn luyện
thật nào**: máy dựng project chỉ có CPU. Phần nào đã chạy trên dữ liệu thật, phần nào chỉ qua kiểm thử trên
dữ liệu giả: xem `docs/STATUS.md`.

**Đọc theo thứ tự:** `docs/DATA_AND_OUTPUTS.md` (dữ liệu đặt ở đâu, script nào, kết quả ở file nào), rồi
`docs/RUNBOOK.md` (lệnh đầy đủ theo thứ tự chạy).

| Phần | Trạng thái |
|---|---|
| Giai đoạn 1: benchmark AMI, đánh giá mô hình có sẵn (T2), phép thử ngữ cảnh (T6 i, v), độ trễ trên máy, xuất ONNX | Đã chạy trên AMI thật (CPU); kết quả sơ bộ ở `results/`, chưa có LPIPS và DISTS |
| Giai đoạn 2: tiền huấn luyện, tinh chỉnh ba giao thức, biến thể thân, phép thử tương tác, dò tốc độ học, hàng đợi | Kiểm thử đầu cuối trên dữ liệu giả, vài bước trên CPU |
| N2: ước lượng suy giảm, bộ phân loại "mô phỏng hay thật", benchmark ảnh ngoài thực tế | Kiểm thử trên dữ liệu giả; chưa chạy trên EarVN1.0 và AWEx |
| N1 (đầu phụ, bộ dò điểm mốc, độ lệch điểm mốc) và N3 (hai đầu, cổng, oracle T4, GAN, LDL) | Kiểm thử trên dữ liệu giả; chưa huấn luyện thật |
| Thước đo: bảng kiểm tra gờ, chỉ số không tham chiếu NTIRE, hộp bao vùng tai | Gờ: đã kiểm. Chỉ số không tham chiếu, LPIPS, DISTS: **chưa từng chạy** (cần `lpips`, `pyiqa` và mạng) |
| Báo cáo: tiêu chí đạt, bảng LaTeX, khảo sát người xem, sổ ghi `runs.csv` | Kiểm thử trên dữ liệu giả |
| Đo trên Jetson (`deploy_jetson/`), Android (`docs/ANDROID.md`) | Đã viết; **chưa chạy trên thiết bị** |

## Cài đặt

```bash
pip install -r requirements.txt
bash scripts/get_weights.sh          # tải trọng số công bố vào ./weights (từ GitHub)
python scripts/check_data.py         # kiểm dữ liệu dưới data/raw (bố cục: docs/DATA_AND_OUTPUTS.md)
```

## Chạy giai đoạn 1 (không cần GPU)

```bash
bash scripts/run_stage1.sh /path/to/AMI [/path/to/DIV2K_valid_HR]
```

hoặc từng bước:

```bash
python scripts/build_benchmark.py --ami-raw /path/to/AMI --out data/bench/ami
python scripts/run_t2.py --bench data/bench/ami --out results/t2
python scripts/summarize_t2.py --in results/t2 --out results/t2_summary
python scripts/run_t6_context.py --ami-raw /path/to/AMI --out results/t6_context
python scripts/summarize_context.py --in results/t6_context --out results/t6_summary
python scripts/bench_local.py --out results/latency_local.csv
python scripts/export_for_device.py        # rồi chép deploy_jetson/ sang Jetson
```

Thư mục AMI gốc phải có đủ 700 file `NNN_view_ear.jpg` (100 người × 7 góc), cỡ 492×702. File khác bị bỏ qua.

## Chạy trên máy GPU

Làm theo `docs/RUNBOOK.md`. Bước đầu tiên, bắt buộc:

```bash
pytest -q
python scripts/smoke_test.py --ami-raw /path/to/AMI
```

Ví dụ một lần tinh chỉnh (mốc SPAN công bố, giao thức tỉ lệ ngẫu nhiên, fold 2; seed = số fold):

```bash
python scripts/train.py --exp T6ii --backbone span --variant zero --pretrain pub --protocol rand \
    --fold 2 --ami-raw /path/to/AMI --bench data/bench/ami
```

Quy tắc không được phá:

- Checkpoint, tốc độ học và mọi lựa chọn khác chỉ được chỉnh trên 10 người validation của fold.
- Fold 1 và 5 là hai fold giữ kín: không chạy trước lần chạy cuối.
- Các phép thử quyết định (T4, T5, T6) chỉ chạy trên fold 2, 3, 4.
- `splits/ami_5fold.json` và `configs/criteria.yaml` không sửa tay. `criteria.yaml` chỉ được chỉnh một lần sau T2.

## Mã lần chạy

`{thí nghiệm}_{thân}-{biến thể}_{tiền huấn luyện}_{giao thức}_x{hệ số}_hr{cỡ}_{suy giảm}_f{fold}`,
ví dụ `T6_span-reflect_ps20_rand_x4_hrall_bic_f3`. Xem `earsr/runid.py`.

## Cấu trúc

```
earsr/
  data/      resize.py (hàm thu phóng duy nhất), ami.py, splits.py, build_lr.py, datasets.py, wild.py
  degrade/   ops.py, pipelines.py (bic, bicjpegQ, generic, est), fit_estimated.py (N2)
  models/    span.py, variants.py (kiểu đệm, số kênh, số khối), registry.py, zoo/, optional/ (N1, N3)
  train/     trainer.py, objectives.py (L1, GAN, LDL, N1, N3), losses.py, finetune.py
  eval/      infer.py, metrics.py, metrics_nr.py, ridge.py, context_test.py, receptive_field.py,
             complexity.py, oracle.py (T4), realism.py (N2)
  landmarks/ pts.py, nets.py (hai bộ dò khác kiến trúc), data.py, train.py, metric.py
  stats/     bootstrap.py (theo người; ghép cặp; tương tác), mde.py, multiple.py (Holm), ranking.py (τ-b)
  deploy/    export_onnx.py, latency.py
  report/    t2.py, context.py, t4.py, criteria.py, latex.py, viewer.py
  runid.py, runlog.py (sổ ghi results/runs.csv)
  device.py  ưu tiên GPU; hết bộ nhớ thì chạy trên CPU rồi tự quay lại GPU (chỉ cho suy luận và số đo)
scripts/
  giai đoạn 1   build_benchmark, evaluate (= run_t2), summarize_t2, run_t6_context, summarize_context,
                bench_local, export_for_device, run_stage1.sh, get_weights.sh
  giai đoạn 2   train, pretrain, make_jobs, run_queue, match_latency, check_criteria, interaction, run_size_sweep
  N2            fit_degradation, realism_classifier, build_wild
  tùy chọn      run_t4_oracle, train_landmarks, landmark_tools
  báo cáo       make_tables, compare_table, make_figures, viewer_study
  kiểm tra      check_data, smoke_test
deploy_jetson/  bench_trtexec.py (tự chứa, Python 3.6, chỉ cần trtexec), README.txt
configs/criteria.yaml   tiêu chí đạt ghi trước
splits/ami_5fold.json   chia 5 fold theo người (có mã băm)
docs/        DATA_AND_OUTPUTS.md, RUNBOOK.md, AMI_PERMISSION.md, STATUS.md, ANDROID.md, THIRD_PARTY.md
results/     kết quả sơ bộ của giai đoạn 1 (CPU)
tests/       143 kiểm thử
```

## Kiểm thử

```bash
EARSR_WEIGHTS=weights EARSR_SET5=/path/to/Set5_HR pytest -q
```

Không có trọng số hoặc Set5 thì các kiểm thử cần chúng tự bỏ qua. Các nhóm chính:

- rò rỉ: người train, validation, test không giao nhau; file fold bị sửa thì bị từ chối;
- hàm thu phóng khớp bản tương thích MATLAB;
- công thức khối SPAB khớp bài gốc (bắt được lỗi đảo thứ tự cộng và nhân);
- nhánh huấn luyện và nhánh đã gộp cho cùng kết quả, với cả ba kiểu đệm;
- đổi kiểu đệm chỉ làm đổi vùng sát mép;
- phép thử ngữ cảnh đo đúng thứ nó định đo;
- trọng số công bố tái lập PSNR Set5 ×4 (SwinIR-light 32,44; RRDB 32,73) trong 0,05 dB;
- khoảng tin cậy bootstrap phủ đúng 95% trên dữ liệu giả;
- chạy thử đầu cuối trên dữ liệu giả, kể cả `train.py` với từng mục tiêu huấn luyện;
- toạ độ điểm mốc đi đúng qua thu phóng, cắt, lật; bộ dò học được trên dữ liệu giả;
- từng tiêu chí trong `criteria.yaml` cho đúng kết luận trên dữ liệu dựng sẵn (đạt và không đạt);
- trang khảo sát người xem không lộ tên mô hình.

## Khác với kế hoạch (cần biết khi viết bài)

- **EDSR-baseline** (`edsr_baseline`) dùng trọng số chính thức của tác giả, tải từ máy chủ của họ (không nằm
  trên GitHub); Set5 ×4 đo được 32,10 dB. `msrresnet` (cùng cỡ, 1,5 triệu tham số) vẫn ở trong kho như một mốc
  "CNN cỡ vừa" thứ hai; trong bài phải gọi đúng tên từng mô hình, không dùng mô hình này thay tên mô hình kia.
- **SPAN 48 kênh** (`span_ch48`) từ 06/10/2026 dùng trọng số chính thức của tác giả (Google Drive, tải tay; xem
  `scripts/get_weights.sh`): Set5 ×4 đo được 32,20 dB, đúng số công bố; tiền huấn luyện trên DF2K. Trong bài SPAN,
  bản 48 kênh (426 nghìn tham số) mang tên **SPAN-S**; bản 52 kênh mới mang tên SPAN. Trọng số của đội 44 NTIRE
  2025 (32,13 dB) còn trong kho dưới tên `span_ch48_t44`; kết quả sơ bộ mang tên `span_ch48` là của bộ này.
- **Dải giá trị của SPAN** (`img_range`) nằm trong state_dict: trọng số chính thức học với ảnh vào nhân 255, bản
  của đội 44 thì không. Checkpoint của project mang theo giá trị này; trọng số công bố không có khóa đó thì giữ
  giá trị lúc dựng.
- **Mọi phép chấm chạy ở FP32 đầy đủ, TF32 tắt** (suy luận, validation, LPIPS, DISTS; `earsr.device.full_precision`).
  Trên GPU RTX 30, PyTorch mặc định dùng TF32 cho tích chập; đo trên RTX 3080 với ảnh AMI thật, đầu ra GPU lệch
  CPU 3,4e-4 khi bật và 1,6e-6 khi tắt. Các bước huấn luyện vẫn dùng mặc định của PyTorch. Bài nên ghi điều này
  ở phần giao thức.
- **Tinh chỉnh chỉ trên AMI làm mô hình hỏng trên ảnh có vùng sáng** (AMI không có ảnh nào như vậy): SPAN tinh chỉnh
  với bicubic đạt 22,08 dB trên EarVN1.0, mốc công bố 36,39 dB. Từ 06/10/2026 mọi lần tinh chỉnh có tăng cường độ
  sáng (`train.py --photo-aug`, mặc định 0,75), khối `n2` thêm ảnh EarVN1.0 nhóm train, và sau mỗi lần huấn luyện có
  phép thử ảnh sáng (`earsr/eval/stress.py`; dòng `phép thử ảnh sáng: ...` trong log, cột `stress_*` trong `runs.csv`).
  Tám lần chạy đầu của khối `n2` bị loại; kết quả cũ ở `results/n2_amionly/`. Đây cũng là một dữ kiện cho bài.
- **`span26` trùng `span_ch28`:** baseline chính thức của NTIRE 2026 là đúng bộ trọng số SPAN 28 kênh thắng NTIRE 2024
  (hai file trùng từng byte), nên kho có 24 tên nhưng 23 bộ trọng số khác nhau. Trong bài, nhóm tối ưu PSNR đếm 16 mô
  hình: bỏ `span26`, và `span_ch48_t44` chỉ là bộ trọng số thứ hai của SPAN 48 kênh. `results/t2_summary/` còn tính cả hai tên.
- **MSRResNet** (KAIR): không nguồn nào nêu dữ liệu tiền huấn luyện, nên cột đó để "không rõ". Mốc "CNN cỡ vừa"
  chính là EDSR-baseline.
- **ECBSR** chưa có trong kho.
- **Mô hình NTIRE 2026** (`span26`, `pds26`, `pkdsr26`, `dscf26`, `disp26`, `errn26`): trọng số ở dạng đã gộp nhánh; tinh
  chỉnh chúng là tinh chỉnh tích chập thường. SPANV2 (hạng 1) không vào kho vì cần một nhân CUDA riêng.
- **MS-SSIM** dùng số tầng giảm theo cỡ ảnh (cột `ms_ssim_scales`); không so được với MS-SSIM 5 tầng trong tài liệu.
- **Bộ điểm mốc tai của Imperial College:** theo trí nhớ, phần có 55 điểm mốc là "Collection A" (khoảng 605 ảnh),
  còn con số 2.058 ảnh, 231 người trong kế hoạch là "Collection B" không có điểm mốc. Phải xác nhận ở T1.
- "PSNR không giảm" của N1 được hiện thực là "không có mức giảm có ý nghĩa thống kê" (cận trên khoảng tin cậy ≥ 0).
- Giao thức `rand` rút cỡ ảnh từ lưới 96, 104, ..., 192 px, không phải liên tục.
- Phép thử ngữ cảnh ngoài ảnh tai đã chạy trên **Urban100**, không phải DIV2K (máy dựng project không tải được DIV2K).
- NTIRE Efficient SR tính PSNR trên RGB; NTIRE Image SR tính trên kênh Y. Kết quả ghi cả hai (`psnr_rgb`, `psnr_y`).
- FLOPs đếm tích chập và lớp tuyến tính; với SwinIR con số thiếu phần attention (cột `flops_partial`).
- SAFMN++ và SMFANet không xuất được sang ONNX opset 13, nên chưa đo được bằng TensorRT của JetPack 4.6.
