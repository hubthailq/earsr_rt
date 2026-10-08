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
| **Giai đoạn 1 chạy đủ trên GPU** (labai217, 06/10/2026) | 152 file T2 (24 mô hình và hai mốc nội suy; ba cỡ, hai kiểu suy giảm, ×4 và ×2), 90 file phép thử ngữ cảnh, độ trễ trên máy, xuất ONNX (22 trong 24; SAFMN++ và SMFANet không xuất được). Không ảnh nào phải chạy trên CPU; không ô LPIPS, DISTS nào trống. Tóm tắt: `results/t2_summary/T2_summary.md`, `results/t6_summary/T6_context_summary.md`. Độ trễ trên Jetson (T3) chưa có |
| Chuỗi N2 trên dữ liệu thật (chạy thử trên máy sửa mã, ra thư mục tạm; **chưa có lần chạy chính thức**) | `fit_degradation.py`, `realism_classifier.py` (rút ngắn: 300 bước, 1 seed), `build_wild.py` cho EarVN1.0 và AWEx, sáu lệnh của khối `n2` (4 bước mỗi lệnh, kể cả tinh chỉnh `disp26`), `evaluate.py --runs` trên EarVN1.0 và trên AMI với bốn kiểu suy giảm: đều chạy được. Lưới độ mờ đã mở xuống tới "không mờ"; tối ưu ở σ [0,2; 0,6] (KS 0,106), nay nằm trong lòng lưới. File vai của EarVN1.0 cho cùng kết quả ở hai lần chạy |
| Đọc PNG 4 kênh của AWEx (`earsr/data/wild.py`) | Trước khi sửa, 1.801 trên 4.004 ảnh AWEx (phần AWE và CVLE) bị loại vì có kênh alpha; cả 1.801 ảnh đều có alpha đặc. Sau khi sửa: 0 ảnh không đọc được; biên an toàn 2 lần cho 217 ảnh của 138 người ở cỡ 96 và 60 ảnh của 49 người ở cỡ 144 (kế hoạch ghi 226 và 61) |
| LR-PSNR chặn ở 100 dB | Trước đó ảnh trùng khít cho vô cực và làm hỏng trung bình tính tay; bảng tổng hợp đọc `inf` trong file cũ thành 100 |
| Phép thử đầu của N2 (labai217, 06/10/2026; chính thức) | File vai của EarVN1.0 và tham số suy giảm đã chốt và commit; chúng trùng khít lần chạy thử trên máy sửa mã. Bộ phân loại (3.000 bước, 3 seed; chấm trên ảnh của 4 người): bicubic 80,1%, tổng quát 78,0%, ước lượng 58,5%, bicubic rồi JPEG 75 56,7%. Tiêu chí N2 (a) đạt (ước lượng gần 50% hơn tổng quát, khoảng tin cậy không chứa 0); ước lượng không hơn "bicubic rồi JPEG 75". Khối huấn luyện `n2` (8 lần) chưa xong |
| `span26` và `span_ch28` là cùng một bộ trọng số | Hai file trùng từng byte; số đo theo ảnh trùng nhau. Số thứ hạng tính lại không có `span26` và `span_ch48_t44` (15 mô hình nhẹ và vừa): τ-b 0,08; 0,09; −0,06 ở ba cỡ, kết luận "thứ hạng đảo" không đổi. `results/t2_summary/` chưa được tạo lại |
| Kế hoạch bài báo, bản 25 | Phần 2 cập nhật theo số thật (mục 2.0 mới; C2, C3, C5 viết hẹp lại; câu ở mục 2.3 và 2.6 đã điền số) |
| **Khối `n2` lần đầu (8 lần huấn luyện chỉ trên AMI, labai217, 06/10): kết quả trên EarVN1.0 không dùng được** | Mô hình tinh chỉnh chỉ trên AMI hỏng trên ảnh có vùng sáng. SPAN tinh chỉnh với bicubic: 22,08 dB trên 158 ảnh EarVN (mốc công bố 36,39; bicubic 32,33); 35,86 dB trên 49 ảnh không có vùng sáng và 15,89 dB trên 109 ảnh có vùng sáng. DISP nhẹ hơn (36,55 và 32,28). Tái hiện được trên máy sửa mã từ checkpoint (lệch từng ảnh dưới 0,001 dB). Cơ chế đo được: trên ảnh sáng, biên độ kích hoạt tăng dần qua các khối SPAB (khối 2: 45; khối 6: 2.333; mô hình công bố: dưới 4) và đầu ra vọt ra ngoài [0, 1]. AMI không có ảnh nào có vùng sáng (phân vị 99 lớn nhất 199; EarVN test: 109 trên 158 ảnh từ 200 trở lên). Nhân độ sáng ảnh validation AMI lên 1,4 lần là đủ làm 41 trên 70 ảnh hỏng |
| Tăng cường độ sáng lúc huấn luyện (`photometric_jitter`, `train.py --photo-aug`, mặc định 0,75) và phép thử ảnh sáng sau mỗi lần huấn luyện (`earsr/eval/stress.py`, ghi `stress.json` và cột `stress_*` trong `runs.csv`) | Hai kiểm thử mới. Phép thử ảnh sáng bắt đúng checkpoint hỏng (×1,8: 8,45 dB so với bicubic 31,46) và cho mô hình công bố qua (34,06 dB). Tiêu chí của phép thử: khoảng cách tới bicubic không tụt quá 1,5 dB so với ở ảnh gốc và không ảnh nào kém bicubic quá 10 dB (không đòi hơn bicubic, vì mô hình chưa tinh chỉnh vốn kém bicubic trên ảnh nén). So sánh ngắn trên CPU (SPAN, 800 bước, lô 16, chỉ khác tăng cường độ sáng): không tăng cường cho validation 39,22 dB, ảnh sáng ×1,8 còn 27,90 dB (bicubic 31,46; 49 trên 210 ảnh hỏng), EarVN 35,24 dB; có tăng cường cho 39,21 dB, 34,53 dB (ổn định), EarVN 36,19 dB (mốc công bố 36,39). Mười lệnh của khối `n2` chạy lại đã chạy thử 4 bước mỗi lệnh (490 ảnh AMI cộng 601 ảnh EarVN dùng được). **Chưa có lần huấn luyện đủ 20.000 bước nào với tùy chọn mới** |
| **Khối `n2` chạy lại** (labai217, 06/10: 10 lần, fold 2, một seed; có tăng cường độ sáng; 8 lần có ảnh EarVN1.0 nhóm train) | Cả 10 lần qua phép thử ảnh sáng. Chấm trên người test fold 2 của AMI (140 ảnh, bốn kiểu ảnh vào) và trên EarVN1.0 nhóm test (158 ảnh, ảnh sạch): `results/n2/` (65 file). SPAN tinh chỉnh với ảnh sạch nay hơn trọng số công bố trên EarVN 0,07 dB (lần trước: kém 14,3 dB). Số liệu và diễn giải: mục 2.0 của kế hoạch bài báo. **Chưa có:** fold 3 và 4; chấm trên EarVN1.0 và AWEx với ảnh vào có nén; khảo sát người xem |
| **Mô hình có sẵn trên EarVN1.0 và AWEx, và quét mức nén trên AMI** (`scripts/run_wild_t2.sh`, labai217, 06/10) | 96 file trên EarVN1.0 (158 ảnh), 192 file trên AWEx (217 ảnh cỡ 96, 60 ảnh cỡ 144), 144 file mới trên AMI (JPEG 60, 85, 93 ở cỡ 96 và 144). Không ô LPIPS, DISTS nào trống. Tóm tắt: `results/t2_earvn_summary/`, `results/t2_awex_summary/`, `results/t2_summary/` (các file tóm tắt còn tính cả `span26` và `span_ch48_t44`) |
| **Khối `n2` trên ba fold** (labai217, 07/10: thêm 20 lần cho fold 3 và 4; tổng 30 lần) | Cả 30 lần qua phép thử ảnh sáng. Chấm bằng `scripts/score_n2.sh` trên AMI (ba cỡ, năm kiểu ảnh vào; 60 người test), EarVN1.0 nhóm test và AWEx (bốn kiểu ảnh vào): 495, 132 và 264 file, không ô LPIPS nào trống. Kết quả gộp ở mục 2.0 của kế hoạch bài báo. **Chưa có:** khảo sát người xem; fold 1 và 5 (giữ kín) |
| Kiểu suy giảm `jpegmix`, `jpegu` và khối `n2c` (07/10) | Kiểm thử: với cùng mức nén, ảnh ra trùng `bicjpegQ` từng bit; mã băm của kiểu `est` không đổi sau khi sửa. Chạy thử trên máy Mac: hai lần huấn luyện 30 bước (CPU), chấm thử 6 ảnh mỗi bộ bằng `score_n2.sh`. **Chưa chạy đủ trên GPU** |
| Đo nhận dạng trên ảnh nhỏ thật (07/10): `earsr/recog/`, `scripts/train_recognizer.py`, `recog_eval.py`, `summarize_recog.py`, `run_recog.sh` | Năm kiểm thử (số đo trên ca biết trước đáp án, cách chia ảnh, một lượt chạy đủ trên dữ liệu giả, chặn rò người giữa tập học và tập chấm). Chạy thử trên ảnh EarVN1.0 thật ở máy Mac với mạng nhận dạng chưa học (1 epoch, không trọng số ImageNet): cả chuỗi chạy thông. **Chưa biết:** mạng nhận dạng học đủ tốt không (máy Mac không tải được trọng số ImageNet); phải xem `val_acc` và dòng `ref_large` sau khi chạy trên labai217 |
| Bản thảo LaTeX (07/10): `paper/`, `scripts/make_paper.py` | Script sinh 157 macro số, 10 thân bảng và 2 hình từ `results/`; bản thảo biên dịch được bằng tectonic (21 trang). 22 macro còn là ô TBD (khối n2c, nhận dạng). `refs.bib` ghi theo trí nhớ, **chưa đối chiếu bản gốc**. Script chưa có kiểm thử tự động |
| **Khối `n2c` và đo nhận dạng, chạy đủ trên labai217 (07/10)** | 6 lần huấn luyện `jpegmix`, `jpegu` xong, cả 6 qua phép thử ảnh sáng; chấm đủ 702, 195, 390 file. Mạng nhận dạng: 95 người, 6.586 ảnh, độ chính xác theo dõi 86,8%, rank-1 trên ảnh dò lớn 60,2% (vượt mức tối thiểu 50% đặt trước). 47 phương pháp chấm với hai mạng nhận dạng. Kết quả ở mục 3b của `docs/story-imavis.md`. **Giới hạn:** mạng nhận dạng ở mức trung bình; một bộ dữ liệu; ảnh định tính đã lưu đều của một người |
| Script lần chạy cuối `make_final_jobs.sh` (07/10) | Sinh đúng 24 lệnh (12 mỗi fold, đủ các nhánh của fold 2, 3, 4); hai script con từ chối fold 1 và 5 khi gọi trực tiếp. **Chưa chạy huấn luyện nào trên hai fold này** |
| Rà tài liệu và `paper/refs.bib` (07/10) | 21 trên 45 mục đã tra bằng tìm kiếm web (tên bài, tác giả, nơi đăng, năm); 24 mục còn lại ghi theo trí nhớ. Mục Related Work viết từ tóm tắt, **chưa đọc toàn văn bài nào**; không mở được toàn văn bài IWSSIP 2023. Chưa rà có hệ thống trên IEEE Xplore, Scopus |
| Phần nhận dạng mở rộng (07/10, tối) | `recog_eval.py` chạy được không cần file vai (AWEx) và ghi mức nén JPEG của từng ảnh dò; `summarize_recog.py` ghi `by_quality.csv`; `run_recog.sh` chấm hai bộ ảnh × ba mạng nhận dạng và lưu 12 ảnh minh họa của 12 người. Đã thử trên máy Mac với mạng nhận dạng chưa học: đường AWEx (130 ảnh dò thử), phép tách theo mức nén (574 ảnh ở mức 93, khớp số đếm tay), ảnh lưu thuộc 12 người khác nhau. **Chưa chạy trên GPU**; mạng ResNet-50 chưa từng được huấn luyện |
| **Lần chạy cuối trên 5 fold (labai217, 08/10)** | 24 lần huấn luyện trên fold 1 và 5 xong, cả 24 qua phép thử ảnh sáng (tổng 60 lần chạy). Chấm đủ 1.134 / 315 / 630 file. Nhận dạng: hai bộ ảnh × ba mạng nhận dạng, 60 mô hình mỗi lượt; mạng ResNet-50 đạt 88,8% trên ảnh theo dõi. `scripts/probe_blockiness.py` (mới, chưa có kiểm thử tự động; đã kiểm tay trên ảnh JPEG tổng hợp: 2,0 so với 0,95 khi không nén). Bản thảo không còn macro số nào trống |
| Xuất Core ML cho iPhone (08/10): `scripts/export_coreml.py`, `deploy_ios/` | Sáu mô hình (SPAN 48 kênh, DISP, EDSR-baseline, SwinIR-light, RRDB, BSRGAN) chuyển được ở ảnh vào 68×48, FP16. So với PyTorch trên máy Mac: PSNR giữa hai đầu ra 63,1 dB (SPAN), 57,0 dB (DISP), 39,2 dB (BSRGAN, nhạy với FP16). Chạy trong `.venv-coreml` (Python 3.9, torch 2.7.1); trên Python 3.14 coremltools thiếu phần nhị phân. **Chưa đo trên iPhone**; các bước trong Xcode chưa được chạy thử. Script chưa có kiểm thử tự động |
| Đối chứng trên ảnh tự nhiên (08/10): `scripts/run_div2k_control.sh` | Dựng benchmark DIV2K valid ở năm cỡ, chấm 16 mô hình có sẵn với năm kiểu ảnh vào, tổng hợp như T2. Đã thử trên máy Mac: 6 ảnh, 3 mô hình, 2 cỡ, không LPIPS; cả chuỗi chạy thông. **Chưa chạy đủ trên GPU** |
| **Đối chứng trên ảnh tự nhiên, chạy đủ (labai217, 08/10)** | 425 file ở `results/t2_div2k/`. `make_paper.py` có thêm `error_ratio` (ghi `results/error_ratio.csv`, hình `fig_ratio.pdf`); con số ngưỡng được tính lại bằng tay một lần trước khi đưa vào script và khớp. Bản thảo còn 6 macro trống (độ trễ iPhone). Hàm mới chưa có kiểm thử tự động |
| **Độ trễ trên iPhone 12 Pro Max (08/10)** | Đo bằng công cụ Performance của Xcode 26.3 trên máy thật, iOS 18.7.8, ảnh vào 48×68, FP16: SPAN 4,72 / 0,80 ms, DISP 2,33 / 0,45 ms, BSRGAN 101,23 / 15,82 ms (chỉ CPU / mọi đơn vị). Số đo đọc từ ảnh chụp màn hình báo cáo của Xcode. Xcode 27.0 cho ra 0,00 ms (lỗi công cụ). Chỉ có trung vị. Trọng số đo là trọng số công bố (cùng kiến trúc với mô hình của bài) |
| Phân tích tỉ số sai số, bản sửa (08/10, trưa) | `error_ratio` trong `make_paper.py` nay tính ba thứ từ file theo ảnh: tỉ số và phần hơn từng ô; hệ số giảm sai số nội suy và khuếch đại sai số nén của từng mô hình; điểm đổi dấu theo từng bộ ảnh với bootstrap 400 lần. Các số được tính một lần bằng script rời trước khi đưa vào và khớp. Ghi `results/error_ratio.csv`, `results/error_ratio_crossover.csv`. **Bản đầu viết "ngưỡng chung 0,17" là sai và đã gỡ.** Chưa có kiểm thử tự động |
| Bản thảo, bản hoàn chỉnh (08/10, chiều) | 35 trang, 14 bảng, 4 hình, 44 tài liệu; không còn ô TBD; biên dịch bằng tectonic không có tham chiếu thiếu. Danh mục tham khảo đối chiếu với Crossref bằng script (`refcheck.py`, chạy một lần, không lưu trong kho). Văn bản viết lại toàn bộ; **chưa có người đọc soát**. Các câu mô tả công trình liên quan dựa trên tóm tắt |
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
