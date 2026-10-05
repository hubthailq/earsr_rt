# Đã kiểm gì, chưa kiểm gì (05/10/2026)

Mục đích của file này: phân biệt rõ phần mã đã chạy thật với phần mới chỉ được viết ra.

## Đã chạy trên dữ liệu thật (AMI, 700 ảnh)

| Việc | Cách kiểm |
|---|---|
| Dựng benchmark AMI ở 4 cỡ, chia 5 fold, sinh ảnh LR (bic, bicjpeg75; ×4, ×2) | Chạy trên 700 ảnh AMI; file fold nằm trong `splits/` |
| Kho mô hình, 16 bộ trọng số | Nạp chặt; tái lập PSNR Set5 ×4 đã công bố: SwinIR-light 32,45 (công bố 32,44), RRDB 32,73 (32,73), ESRGAN 30,47 |
| T2: đánh giá mô hình có sẵn | Chạy trên AMI; số đo theo từng ảnh ở `results/t2/` |
| T6 (i), (v): phép thử ngữ cảnh | Chạy trên AMI và Urban100; `results/t6_context/` |
| Thống kê: bootstrap theo người, ghép cặp, τ-b, MDE | Dùng để tạo các bảng trong `results/`; có kiểm thử độ phủ |
| Xuất ONNX opset 13 và so với PyTorch bằng onnxruntime | 19 trong 21 mô hình xuất được; SAFMN++ và SMFANet không xuất được |

## Đã chạy thử, nhưng chỉ vài bước trên CPU

| Việc | Mức kiểm |
|---|---|
| Tinh chỉnh (`scripts/train.py`) với ba giao thức, SPAN và RLFN, kể cả từ checkpoint tiền huấn luyện | 2 đến 6 bước; kiểm rằng mô hình sau vài bước cho đúng PSNR của mô hình công bố (lệch 0,001 dB) |
| Tiền huấn luyện (`scripts/pretrain.py`) | 6 bước trên 8 ảnh Urban100 |
| Chạy thử đầu cuối trên dữ liệu giả | `tests/test_pipeline.py` |

Chưa có lần huấn luyện thật nào. Tốc độ học, số bước, cỡ lô trong các script là giá trị khởi đầu, chưa được dò.
Thời gian một lần huấn luyện trên RTX 3080 chưa đo.

## Thêm ở bản 22

| Việc | Mức kiểm |
|---|---|
| Sáu mô hình NTIRE 2026 trong kho | Nạp chặt trọng số; chạy trên Set5 (31,7 đến 31,9 dB, hợp với cỡ 0,13 đến 0,25 triệu tham số); số tham số khớp báo cáo; xuất ONNX opset 13 khớp PyTorch. **Chưa chạy trên AMI, chưa tinh chỉnh** |
| Đổi kiểu đệm trên mọi thân | Kiểm thử trên 7 thân: chỉ vùng sát mép đổi |
| MS-SSIM, GMSD, PSNR của gradient, LR-PSNR | Kiểm thử: giá trị ở ca trùng nhau, đơn điệu theo độ mờ, GMSD khớp tính tay |
| ST-LPIPS, TOPIQ-FR, FSIM, VIF, PieAPP (`--more-metrics`) | **Chưa từng chạy** (cần `pyiqa`); tên số đo ghi theo trí nhớ |

## Thêm ở bản 24

| Việc | Mức kiểm |
|---|---|
| `make_figures.py` (hình 2, 3, 4, 5 và lưới định tính) | Kiểm thử trên dữ liệu giả; hình 3 và 4 đã vẽ thử từ kết quả sơ bộ và xem bằng mắt. Hình 2, 5 và lưới chưa vẽ trên dữ liệu thật |
| `compare_table.py` | Kiểm thử: chênh lệch ghép cặp, dấu †, dòng khác tập ảnh, cột chi phí |
| Bảng N2 và bảng khảo sát trong `make_tables.py` | Chạy thử một lần trên file giả |

## Thêm ngày 06/10/2026

| Việc | Mức kiểm |
|---|---|
| EDSR-baseline (`edsr_baseline`) với trọng số chính thức của tác giả | Mã băm file khớp; nạp chặt (1.517.571 tham số); Set5 ×4 đo được 32,10 dB. **Tái lập số tác giả công bố:** 28,96 dB trên DIV2K 0801 đến 0900 (ảnh LR bicubic chính thức, PSNR trên RGB, bỏ viền 6 px; 28,95 dB nếu bỏ viền 10 px) so với 28,95 dB công bố. Xuất ONNX opset 13 khớp PyTorch (lệch 1e-6). Đường đánh giá trên AMI mới chạy thử 20 ảnh ở cỡ 144: 36,95 dB (bic), 32,96 dB (JPEG 75), sát MSRResNet. **Chưa chạy đủ 700 ảnh AMI** |
| Các kiểm thử Set5 của kho mô hình | Lần đầu chạy ngoài máy dựng project (Set5 đặt ở `data/raw/Set5`): 22 kiểm thử trong `tests/test_zoo.py` đạt |
| Tên số đo của `pyiqa` | Cả 10 tên dùng trong mã (`dists`, `stlpips`, `topiq_fr`, `fsim`, `vif`, `pieapp`, `niqe`, `maniqa`, `musiq`, `clipiqa`) có trong `pyiqa` 0.1.16, kiểu FR hay NR đúng như mã dùng. **Mới kiểm tên, chưa chạy số đo nào** (cần tải trọng số) |
| Mốc SPAN đổi sang trọng số chính thức (`span_ch48`; bản đội 44 thành `span_ch48_t44`) | File trùng từng byte với `spanx4_ch48.pth` trong `span.zip` của tác giả; nạp chặt; Set5 ×4 đo được 32,2005 dB (bài SPAN, bảng 1, dòng SPAN-S ×4: 32,20); xuất ONNX opset 13 khớp PyTorch. Trên 700 ảnh AMI, cỡ 144, bicubic (CPU, chạy thử): 39,04 dB (bản đội 44: 39,00). **Chưa chạy đủ T2 với trọng số mới** |
| Dải giá trị của SPAN đi theo trọng số (`img_range` trong state_dict) | Ba kiểm thử mới trong `tests/test_span.py`; trước khi sửa, thân dựng để tinh chỉnh từ trọng số chính thức lệch 58% so với mô hình công bố mà không báo lỗi. Chạy thử `train.py` 4 bước từ trọng số chính thức (đệm zero và replicate): validation xuất phát ở 39,09 dB; nạp lại lần chạy qua `evaluate.py --runs` cho đúng PSNR của mô hình công bố trên 140 ảnh test của fold 2 (39,397) |
| Dữ liệu tiền huấn luyện của MSRResNet (KAIR) | Đã tra README, file cấu hình và trang release của KAIR: không nêu. Ghi "không rõ" trong `registry.py` |
| Ưu tiên GPU, hết bộ nhớ thì xuống CPU rồi quay lại GPU (`earsr/device.py`; dùng trong `evaluate.py`, `run_t6_context.py`, LPIPS và DISTS; `bench_local.py` thì chờ rồi đo lại) | 12 kiểm thử trong `tests/test_device.py` với lỗi hết bộ nhớ **giả lập** (thiết bị chính giả là `cpu:0`); chạy thử với mô hình thật và lỗi giả lập ở 1 trong 3 lượt gọi: PSNR từng ảnh trùng bản chạy CPU thuần. **Chưa chạy với lỗi hết bộ nhớ thật trên GPU.** Huấn luyện chưa có cơ chế này |
| **Lần đầu chạy trên GPU** (`smoke_test.py` trên labai217: RTX 3080, torch 2.11 + CUDA 13, Python 3.12, `pyiqa` 0.1.15) | Đạt: suy luận trên GPU, LPIPS và DISTS ra số (bicubic 0,294 và 0,282; SPAN 0,204 và 0,222 trên 50 ảnh), huấn luyện L1 ở FP32 và AMP, huấn luyện GAN với VGG19 thật, xuất ONNX, bốn số đo không tham chiếu. Tốc độ: 6,0 bước mỗi giây (0,92 giờ cho 20.000 bước L1; GAN 2,76 giờ); AMP không nhanh hơn FP32. PSNR 50 ảnh trên GPU khớp CPU máy sửa mã (37,238 so với 37,2383) |
| Chấm ở FP32 đầy đủ, TF32 tắt (`earsr.device.full_precision`) | Bước smoke test "GPU và CPU cho cùng kết quả" từng báo lỗi (lệch 2,79) vì dùng nhiễu trắng, thứ làm SPAN khuếch đại sai số làm tròn; đã đổi sang ảnh LR thật. Đo riêng trên labai217 với ảnh AMI thật: GPU lệch CPU 3,4e-4 khi TF32 bật và 1,6e-6 khi tắt (SPAN 48 kênh; EDSR-baseline 8,4e-5 và 4,9e-7). Hai kiểm thử mới xác nhận mọi đường chấm chạy với TF32 tắt. **Bước smoke test đã sửa chưa chạy lại trên GPU** |
| Tính tái lập giữa các máy | 20 ảnh đầu của T2 (bicubic, MSRResNet; bic và JPEG 75; cỡ 144) chạy lại trên máy sửa mã cho đúng PSNR của kết quả sơ bộ tới 3 chữ số thập phân |

## Chỉ qua kiểm thử trên dữ liệu giả (CPU)

| Việc | Kiểm thử cho thấy gì | Chưa biết gì |
|---|---|---|
| `train.py` với L1, GAN, LDL, N1, N3; ảnh thêm; suy giảm ngẫu nhiên; nạp lại mô hình từ thư mục lần chạy | Chạy đầu cuối 4 bước; mỗi mục tiêu chỉ đổi đúng phần tham số nó được đổi; số đo nạp lại khớp | Hội tụ, trọng số loss, tốc độ học. Loss cảm nhận chỉ chạy với VGG19 trọng số ngẫu nhiên |
| Bộ dò điểm mốc, độ lệch điểm mốc, hộp bao | Trên ảnh chấm tròn giả, bộ dò đạt sai số dưới 1% đường chéo; toạ độ đi đúng qua thu phóng, cắt, lật | Chất lượng trên tai thật; bộ dò chạy trên cả ảnh chứ không cắt quanh tai, nên có thể kém nếu khung ảnh khác lúc huấn luyện |
| Bộ phân loại "mô phỏng hay thật" | Tách được suy giảm sai (trên 85%), không tách được suy giảm đúng (dưới 70%) | Kết quả trên EarVN1.0 |
| Cổng oracle và tổng hợp T4 | Đúng định nghĩa; kết luận đúng trên dữ liệu dựng sẵn | Chạy thật cần LPIPS |
| Tiêu chí đạt (`check_criteria.py`) | Mỗi tiêu chí cho đúng kết luận ở ca đạt và ca không đạt | |
| `build_wild.py`, `fit_degradation.py` | Lọc, loại trùng, chia vai, manifest | Cấu trúc thư mục thật của EarVN1.0 và AWEx phải là `root/<người>/<ảnh>` |
| Khảo sát người xem, bảng LaTeX, sổ ghi, hàng đợi, `make_jobs.py`, `run_size_sweep.py`, `match_latency.py` | Chạy đúng trên dữ liệu giả | Trang HTML chưa được mở thử bằng trình duyệt với người thật |

## Chưa từng chạy

| Việc | Vì sao |
|---|---|
| Mọi thứ trên GPU, kể cả AMP | Máy dựng project chỉ có CPU. `scripts/smoke_test.py` kiểm phần này |
| LPIPS, DISTS (`earsr/eval/metrics.py`), NIQE, MANIQA, MUSIQ, CLIP-IQA (`metrics_nr.py`) | Không cài được `lpips`, `pyiqa`, không tải được trọng số. Tên chỉ số của `pyiqa` ghi theo trí nhớ |
| Loss cảm nhận với trọng số VGG19 thật | Không tải được từ download.pytorch.org |
| `deploy_jetson/bench_trtexec.py` | Chưa có Jetson. Biểu thức đọc dòng "Latency" của trtexec viết theo tài liệu |
| Đo trên Android | Chỉ có hướng dẫn (`docs/ANDROID.md`) |
| Kiểu suy giảm `generic` so với Real-ESRGAN gốc | Đây là bản rút gọn một bậc |

## Chưa viết

ECBSR trong kho mô hình; mô hình đầu bảng NTIRE ESR 2026 và AIM 2025; HAT; lượng tử hóa INT8;
phân rã độ trễ theo tầng (E10). Nhãn điểm mốc gắn tay cần công cụ gắn nhãn bên ngoài.

## Một việc cần bạn dọn

Trong lúc kiểm quyền ghi, một file rỗng tên `.w_test` đã bị tạo trong thư mục AMI gốc của bạn
(`~/Downloads/AMI/.w_test`). File này vô hại và project bỏ qua nó (chỉ đọc file đúng mẫu `NNN_view_ear.jpg`),
nhưng nó không thuộc về bộ dữ liệu. Bạn xóa giúp.
