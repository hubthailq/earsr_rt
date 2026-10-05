# HANDOFF: dự án `earsr_rt` (SR real-time cho ảnh tai độ phân giải thấp)

Ngày viết: 05/10/2026, 21:40 (giờ Việt Nam). File này tóm tắt toàn bộ một phiên làm việc dài trên Claude (phiên đám mây) để một phiên Claude Code khác (trong Cursor, chạy trên máy người dùng) tiếp tục được mà không cần lịch sử trò chuyện.

**Gửi Claude Code đọc file này:** đọc hết file này trước, rồi đọc `README.md`, `docs/STATUS.md`, `docs/DATA_AND_OUTPUTS.md`, `docs/RUNBOOK.md` trong repo, và file kế hoạch `paper-plan-ridgesr-2026-10-05.md` (bản 24). File này ghi bối cảnh, quyết định, lý do và những điều chưa nằm trong các file kia. Nếu file này và mã nguồn nói khác nhau, tin mã nguồn và kiểm thử.

---

## 1. Người dùng và cách làm việc

- Thái Lê Quang, nghiên cứu sinh tiến sĩ ngành thị giác máy tính, làm việc dưới sự hướng dẫn của một giáo sư.
- **Trả lời bằng tiếng Việt.** Chú thích và docstring trong mã cũng bằng tiếng Việt (cố ý).
- Luôn giao **cả project** (trước đây là file zip đầy đủ, không giao file lẻ). Rà lại toàn bộ project trước khi giao bản cập nhật. Người dùng coi trọng việc rà mã kỹ, nhiều vòng.
- Người dùng tự chạy thí nghiệm trên máy GPU của mình (một RTX 3080). **Không chạy thí nghiệm dài thay họ.** Chỉ viết mã và chạy kiểm thử ngắn. (Ở phiên trước, việc chạy 700 ảnh trên CPU hơn một giờ khi chỉ được yêu cầu viết mã đã bị nhắc.)
- Người dùng muốn câu trả lời thẳng: phần nào đã kiểm, phần nào chưa; không nói quá.
- Chưa có Jetson Nano (đời 2019, JetPack 4.6.x, Python 3.6, TensorRT 8.2, ONNX opset 13). Sẽ tìm sau.

## 2. Mục tiêu

Một bài báo tạp chí **Q1, không săn mồi** (giáo sư chọn tạp chí). Đề bài của giáo sư, nguyên văn: cải tiến SR model, một mô hình real-time cho ảnh tai; đánh giá thông qua metric, chất lượng hình ảnh; bỏ qua phần recognition cho id/gender; làm cho ảnh to lên, rõ hơn trước; real-time hoặc near real-time.

Ràng buộc đã chốt: số đo kiểu NTIRE; ×4 (chính) và ×2; chỉ dùng bộ dữ liệu công khai; 5 fold theo người, seed bằng số fold; Jetson chỉ là thiết bị đo.

Project này (`earsr_rt`) là **project mới**, tách khỏi project cũ `earsr_project_span` của người dùng (bài span_tiny).

## 3. Hai sản phẩm

1. **File kế hoạch** `claude/paper-plan-ridgesr-2026-10-05.md` trong Project trên claude.ai, hiện là **bản 24**. Cấu trúc: Phần 1 đề xuất (1.1 đến 1.13, có 1.2b, 1.2c, 1.8b), Phần 2 khung bản thảo (2.1 đến 2.8, có 2.5b), Phần 3 thiết kế project (3.1 đến 3.14), Phụ lục 1 đến 6 (trả lời phản biện vòng 5 đến 9).
2. **Mã nguồn** `earsr_rt` (giao bằng `earsr_rt.zip`, 247 file, khoảng 13.000 dòng Python, **121 kiểm thử đạt trên CPU**).

## 4. Diễn biến và các quyết định (theo thứ tự)

### 4.1 Kế hoạch qua 5 vòng phản biện (bản 15 đến 19)

Một phản biện (do người dùng chuyển lời) đã đọc kế hoạch 5 vòng. Kết luận của phản biện ở vòng 9: không còn ý kiến về ý tưởng; chờ bảng T2, số đo T3 và hình bề rộng ngữ cảnh. Phản biện chấm kịch bản "N5b và N2 đều đạt trên hai thân" là mức "major revision theo hướng nhận" ở tạp chí Q1 phạm vi rộng.

Các điểm mới ứng viên (ký hiệu dùng khắp kế hoạch và mã):

| Ký hiệu | Nội dung | Trạng thái sau phiên này |
|---|---|---|
| N1 | Đầu phụ học cấu trúc tai (điểm mốc), chỉ lúc huấn luyện | Tùy chọn; mã có, chưa huấn luyện |
| N2 | Suy giảm ước lượng từ ảnh tai thật (EarVN1.0) | **Giải pháp chính** (từ bản 22) |
| N3 | Hai đầu ra và cổng điểm ảnh | Ứng viên bị cắt đầu tiên; mã có |
| N4 | Benchmark ảnh tai nhỏ, số đo thiết bị | Chắc chắn có |
| N5a | Giao thức huấn luyện tỉ lệ ngẫu nhiên | Hạ xuống thành giao thức, không tính là điểm mới trừ khi đạt tiêu chí |
| N5b | Chỉnh thân cho ảnh vào nhỏ hơn vùng nhìn của mạng (kiểu đệm, sâu và rộng) | Giải pháp phụ; chỉ giữ nếu đạt 0,1 dB |

Quy tắc không được phá: fold 1 và 5 giữ kín tới lần chạy cuối; phép thử quyết định chỉ trên fold 2, 3, 4; mọi lựa chọn chỉ trên 10 người validation của fold; ngưỡng ghi trước trong `configs/criteria.yaml` (bản 21), không sửa sau khi thấy kết quả.

### 4.2 Viết mã và các thay đổi hướng trong phiên này

1. Viết xong mã mọi giai đoạn (huấn luyện L1, GAN, LDL, N1, N3; đánh giá; N2; T4; tiêu chí; bảng; khảo sát người xem; sổ ghi; hàng đợi).
2. **Bản 20:** thêm `docs/DATA_AND_OUTPUTS.md` và `scripts/check_data.py`; ghi lý do chọn AMI làm benchmark chính (mục 1.6).
3. **Bản 21:** giao thức so sánh công bằng (mục 1.8b): hai nhánh so sánh.
4. **Bản 22:** người dùng yêu cầu không gắn với SPAN, rà mô hình mới, và thêm số đo. Kết quả: sáu mô hình NTIRE 2026 vào kho; trọng tâm bài chuyển sang N2; thêm số đo.
5. **Bản 23:** viết lại Phần 2 (khung bản thảo).
6. **Bản 24:** script dựng bảng, vẽ hình; Bảng 1 công trình liên quan; thư mẫu xin phép AMI.

### 4.3 Câu chuyện của bài (bản 23, mục 2.1)

Một câu: SR hiệu quả hiện nay không dùng được cho ảnh tai nhỏ thật, và lý do là suy giảm và cỡ ảnh vào chứ không phải kiến trúc; sửa đúng hai chỗ đó thì một mô hình real-time cho ảnh tai rõ hơn mô hình đa dụng đã tinh chỉnh.

Ba câu hỏi nghiên cứu: (RQ1) thứ hạng của SR hiệu quả có giữ được ở ảnh tai nhỏ có nén không; (RQ2) phần kém đến từ kiến trúc, suy giảm hay cỡ ảnh vào; (RQ3) sửa suy giảm và cỡ ảnh vào thêm được bao nhiêu so với chỉ tinh chỉnh, ở cùng độ trễ.

Chín luận điểm C1 đến C9 (mục 2.2 của kế hoạch), mỗi cái có bảng, hình, file kết quả, điều kiện bác bỏ, và câu viết sẵn cho nhánh đạt lẫn không đạt (mục 2.6).

Ba kịch bản: đủ (C5 và C6 đạt); chỉ suy giảm (C5 đạt); phân tích và benchmark (C5 không đạt, phải bàn lại với giáo sư).

### 4.4 Đánh giá thẳng về tính mới (đã nói với người dùng)

- Bài **không có khối kiến trúc mới**. Hướng "tự nghĩ khối mới" đã được cân nhắc và không khuyên: ở NTIRE 2026 nhiều đội thử khối mới mà không vượt họ SPAN về tốc độ.
- Mạnh nhất là phát hiện (C2, C3) và benchmark, vì là dữ kiện. N2 và N5b là áp ý đã có vào miền mới; bài chống lại phê bình đó bằng độ chặt của bằng chứng.
- Nếu giáo sư kỳ vọng "mô hình cải tiến" theo nghĩa có khối mới, người dùng nên nói rõ với thầy: mô hình đề xuất là "thân mới nhất + huấn luyện đúng miền + chỉnh cho ảnh nhỏ".
- Phép thử quyết định của N2 khó hơn vẻ ngoài: mô hình nào tinh chỉnh với ảnh có nén cũng vượt lại bicubic. N2 chỉ đạt khi suy giảm **ước lượng** hơn suy giảm **tổng quát** trên ảnh nhỏ thật.

### 4.5 Tạp chí (chưa nằm trong file nào khác)

Chỉ là gợi ý để bàn với giáo sư; xếp hạng Q phải tra lại trên Scimago hoặc JCR.

- Hợp nhất: Engineering Applications of Artificial Intelligence; Expert Systems with Applications; Complex & Intelligent Systems (truy cập mở, có phí).
- Cao hơn, rủi ro: Pattern Recognition; Knowledge-Based Systems; Neurocomputing. Không khuyên TIP, TPAMI.
- Dự phòng: Image and Vision Computing; CVIU; Journal of Real-Time Image Processing.
- Tránh tạp chí sinh trắc học (reviewer sẽ hỏi về nhận dạng, mà đề bài đã bỏ). Các tạp chí MDPI: hỏi quy định của trường trước.
- Chốt tạp chí sau điểm kiểm tra 2.

## 5. Số đo đã có (đều là sơ bộ)

**Tất cả chạy trên CPU của máy đám mây, chưa có LPIPS và DISTS, phải chạy lại trên GPU trước khi dùng trong bài.** Nằm trong `results/` của repo.

Đo trên dữ liệu:

- AMI: 700 ảnh, 100 người × 7 góc (back, down, front, left, right, up, zoom), 492×702, JPEG mức 90.
- EarVN1.0: 28.412 ảnh, 164 người; cạnh ngắn từ 128 px: 6.386 ảnh của 135 người; từ 192 px: 1.086 ảnh của 95 người; từ 288 px: 30 ảnh. Khoảng mức nén 75 ở 90% ảnh. Trung vị cạnh ngắn 77 px.
- AWEx: 4.004 ảnh, 336 người; từ 128 px: 563; từ 192 px: 226 ảnh của 142 người; từ 288 px: 61.
- Cặp "front" và "zoom" của AMI không dùng được làm cặp SR (độ phóng trung vị chỉ 1,18).

T2 sơ bộ (PSNR-Y, ×4, 700 ảnh AMI):

| | Cỡ 96 | Cỡ 144 | Cỡ 192 |
|---|---|---|---|
| bicubic | 34,30 | 36,73 | 38,13 |
| span_ch48 | 37,01 | 39,00 | 39,93 |
| msrresnet | 37,04 | 39,06 | 39,97 |
| swinir_light | 37,13 | 39,10 | |
| rrdb_psnr | | 39,28 | |

- Với ảnh vào JPEG 75, cỡ 144: bicubic 34,36; các mô hình khoảng 33,9 (thua bicubic khoảng 0,4 dB). Thứ hạng đảo: τ-b 0,15; 14 cặp đổi chiều có ý nghĩa.
- Mức chênh nhỏ nhất phát hiện được (span_ch48 so với rlfn, 60 người): 0,016 dB.

Phép thử ngữ cảnh sơ bộ: thiếu ngữ cảnh mất 0,125 đến 0,20 dB (AMI, cấu hình wide), 0,29 đến 0,35 dB (cửa sổ nhỏ 24×34); trên Urban100 mạng mất 0,09 đến 0,26 dB, bicubic 0,02; hồi phục khi có 4 px ngữ cảnh; chỉ vành ngoài cùng bị ảnh hưởng; đệm lặp viền hoặc phản chiếu **ở ngoài mạng** làm kém đi 0,5 đến 1,9 dB. Vùng nhìn của SPAN: bán kính 21, r95 = 5.

Set5 ×4 (kiểm kho mô hình): SwinIR-light 32,45 (công bố 32,44); RRDB 32,73 (32,73); ESRGAN 30,47; MSRResNet 32,22; span_ch48 32,13 (bài SPAN: 32,20; trọng số đang dùng là của đội 44 NTIRE 2025, không phải của tác giả). Sáu mô hình 2026: span26 31,73; pds26 31,76; pkdsr26 31,77; dscf26 31,74; disp26 31,94; errn26 31,89.

**Sáu mô hình NTIRE 2026 chưa được chạy trên AMI.** Câu "kể cả các mô hình đầu bảng 2026 cũng thua bicubic" hiện là giả định, chưa có số.

## 6. Mã nguồn: cái gì ở đâu

```
earsr/
  data/      resize.py (hàm thu phóng duy nhất, tương thích MATLAB), ami.py, splits.py, build_lr.py,
             datasets.py (SRTrainDataset: native|fixed|rand; MixedDataset; điểm mốc; min_downscale), wild.py
  degrade/   ops.py, pipelines.py (bic, bicjpegQ, generic, est), fit_estimated.py
  models/    span.py (SPAN cấu hình được; Conv3XC), variants.py (zero|replicate|reflect, wide|deep, cNN, bNN, +nhãn),
             padding.py (đổi kiểu đệm trên mọi thân), registry.py (24 mô hình), zoo/, optional/ (heads.py, n3.py)
  train/     trainer.py, objectives.py (l1, gan, ldl, aux, n3s1A, n3s1B, n3s2), losses.py, finetune.py
  eval/      infer.py, metrics.py, metrics_nr.py, ridge.py, context_test.py, receptive_field.py,
             complexity.py, oracle.py, realism.py
  landmarks/ pts.py, nets.py (HeatmapNet, RegressNet), data.py, train.py, metric.py
  stats/     bootstrap.py, mde.py, multiple.py, ranking.py
  deploy/    export_onnx.py, latency.py
  report/    t2.py, context.py, t4.py, criteria.py, latex.py, viewer.py, figures.py
  runid.py, runlog.py
scripts/     check_data, smoke_test, build_benchmark, evaluate (= run_t2), summarize_t2, run_t6_context,
             summarize_context, bench_local, export_for_device, run_stage1.sh, get_weights.sh,
             train, pretrain, make_jobs, run_queue, match_latency, check_criteria, interaction, run_size_sweep,
             fit_degradation, realism_classifier, build_wild, run_t4_oracle, train_landmarks, landmark_tools,
             make_tables, compare_table, make_figures, viewer_study
deploy_jetson/bench_trtexec.py   (Python 3.6, chỉ cần trtexec)
configs/criteria.yaml   splits/ami_5fold.json   docs/   results/   tests/
```

Mã lần chạy: `{thí nghiệm}_{thân}-{biến thể}_{tiền huấn luyện}_{giao thức}_x{hệ số}_hr{cỡ}_{suy giảm}_f{fold}`. Mã không chứa mọi siêu tham số; chạy lại cùng mã với siêu tham số khác thì `train.py` dừng và báo (thêm `--tag`).

Bố cục dữ liệu mặc định: `data/raw/{AMI, DIV2K_train_HR, DIV2K_valid_HR, EarVN1.0, awex, landmarks}`. Với EarVN1.0 và AWEx, mã người là **tên thư mục chứa ảnh**; chưa kiểm với bản tải về thật.

## 7. Đã kiểm gì, chưa kiểm gì

| Mức | Nội dung |
|---|---|
| Đã chạy trên AMI thật (CPU) | Dựng benchmark, T2 với 16 mô hình cũ, phép thử ngữ cảnh, xuất ONNX opset 13 (SAFMN++ và SMFANet không xuất được) |
| Chỉ qua kiểm thử trên dữ liệu giả | Mọi thứ của giai đoạn 2 trở đi: `train.py` với mọi mục tiêu (4 bước), bộ dò điểm mốc, bộ phân loại N2, T4, tiêu chí, bảng, hình, khảo sát người xem, hàng đợi, `build_wild.py` |
| **Chưa từng chạy** | Mọi thứ trên GPU và AMP; LPIPS, DISTS; NIQE, MANIQA, MUSIQ, CLIP-IQA; ST-LPIPS, TOPIQ-FR, FSIM, VIF, PieAPP (tên số đo `pyiqa` ghi theo trí nhớ); loss cảm nhận với VGG19 thật; script Jetson; Android (chỉ có hướng dẫn) |
| Chưa viết | ECBSR, HAT trong kho; lượng tử hóa INT8; phân rã độ trễ theo tầng. (EDSR-baseline đã thêm ngày 06/10/2026 với trọng số chính thức; xem `docs/STATUS.md`) |

Hai lượt rà mã độc lập bằng agent phụ đã định làm nhưng bị dừng vì hết hạn mức phiên; lượt rà cuối là tự rà. **Một lượt rà mã độc lập vẫn là việc đáng làm ở phiên mới**, nhất là các đường mã GPU, tiếp tục từ checkpoint, và thống kê trong `earsr/report/criteria.py`.

## 8. Việc tiếp theo của người dùng (theo thứ tự)

1. Giải nén zip vào repo, `git init`, commit. `pip install -r requirements.txt`; `bash scripts/get_weights.sh`.
2. `python scripts/check_data.py`; `pytest -q`; **`python scripts/smoke_test.py --ami-raw data/raw/AMI`** (5 đến 10 phút; kiểm GPU, AMP, LPIPS, DISTS, VGG19; in tốc độ huấn luyện). Có bước LỖI thì sửa trước khi đi tiếp.
3. Giai đoạn 1: `bash scripts/run_stage1.sh data/raw/AMI data/raw/DIV2K_valid_HR`, rồi `make_tables.py`, `make_figures.py`. Đây là lúc có số thật cho C1, C2, C3, kể cả cho sáu mô hình 2026.
4. Điểm kiểm tra 1. Chỉnh `criteria.yaml` một lần nếu `mde.csv` đòi, rồi commit.
5. **N2 sớm:** `fit_degradation.py` → `realism_classifier.py` → `make_jobs.py n2` → chấm trên EarVN1.0 và AMI.
6. **Kiểu đệm rẻ:** `make_jobs.py pad` (trọng số công bố, đổi kiểu đệm, tinh chỉnh; không tiền huấn luyện).
7. Dò tốc độ học, T6 (ii), (iii); chỉ khi bước 6 hoặc phép thử ngữ cảnh ủng hộ thì mới tiền huấn luyện và chạy T6 (iv), (v).
8. Điểm kiểm tra 2, rồi khối `s2` (lần đầu dùng fold 1 và 5).
9. Điền bài theo mục 2.7 của kế hoạch.

Lệnh đầy đủ: `docs/RUNBOOK.md`.

## 9. Việc còn mở và chỗ hở đã biết

- **(Đã xử lý ngày 06/10/2026) Trọng số SPAN 48 kênh:** `span_ch48` nay dùng bản chính thức của tác giả (Set5 32,20; cần `img_range=255`, nay nằm trong state_dict); bản đội 44 (32,13) thành `span_ch48_t44`. Mọi con số `span_ch48` ở mục 5 và trong `results/` sơ bộ là của bản đội 44. Trong bài SPAN, bản 48 kênh tên là SPAN-S. Xem `docs/STATUS.md`. Repo NTIRE 2026 có SPAN 28 kênh chính thức của ban tổ chức (`span26`).
- **Trọng số sáu mô hình 2026 ở dạng đã gộp nhánh**; tinh chỉnh chúng là tinh chỉnh tích chập thường. Bài phải nói rõ.
- **Bộ điểm mốc tai của Imperial College:** theo trí nhớ, phần có 55 điểm mốc là "Collection A" khoảng 605 ảnh; con số 2.058 ảnh, 231 người trong kế hoạch có thể là "Collection B" không có điểm mốc. Phải xác nhận trước khi tính đến N1. Điều khoản sử dụng chưa kiểm.
- **Bảng 1 (công trình liên quan)** mới điền tới mức đã xác minh (mục 2.5b). Chỉ tìm được một bài SR cho ảnh tai ("Improving Ear Recognition with Super-resolution", IEEE 2023, chưa đọc nội dung). Chưa được viết "chưa ai làm" cho tới khi tìm có hệ thống.
- **Hình dùng ảnh AMI** cần thư đồng ý của tác giả (CC BY-NC-ND). Thư mẫu ở `docs/AMI_PERMISSION.md`; người dùng phải gửi.
- **MS-SSIM** dùng số tầng giảm theo cỡ ảnh (4 tầng ở 144 px); không so được với số trong tài liệu.
- **"PSNR không giảm" của N1** được hiện thực là "không có mức giảm có ý nghĩa thống kê".
- **Chi phí nhánh so sánh có kiểm soát:** 7 lần tiền huấn luyện rút gọn và 4 lần đủ; thời gian mỗi lần chưa đo.
- **File rỗng `.w_test`** đã lỡ bị tạo trong thư mục AMI gốc của người dùng (`~/Downloads/AMI/.w_test`). Vô hại, project bỏ qua, nhưng người dùng cần xóa.
- Số bước, tốc độ học, trọng số loss trong các script đều là giá trị khởi đầu, chưa dò.

## 10. Sai lầm đã mắc trong các phiên trước (đừng lặp lại)

- Chạy thí nghiệm dài khi chỉ được yêu cầu viết mã.
- Chọn biên an toàn 1,33 lần cho EarVN1.0 (vết nén còn trong đáp án); đã quay về 2 lần.
- So N5a với một mốc yếu (cắt từ ảnh gốc); mốc đúng là "ảnh HR đúng cỡ".
- Hai lỗi mã bắt được ở lượt rà cuối: mô hình SPAN dạng đã gộp gần như không học được khi tinh chỉnh (đã sửa bằng `unfreeze_deploy`); hai cấu hình khác siêu tham số trùng mã lần chạy (đã chặn).
- Ghi một file thử vào thư mục dữ liệu gốc của người dùng.

## 11. Ghi chú kỹ thuật dễ vấp

- `Conv3XC`: `eval_conv` chỉ được gộp lại từ các nhánh huấn luyện khi gọi `.eval()`. Mọi chỗ dùng, lưu hoặc chấm mô hình ở chế độ eval phải gọi `.eval()` trước.
- Công thức khối SPAB: `out = (H + x) * (sigmoid(H) - 0.5)`; giá trị thứ hai trả về là đầu ra **sau kích hoạt** của tầng đầu.
- `span_ch28`, `span_ch26`, `span26`: `img_range=255`. `rlfn`: `data_range=255`. `swinir_light`: bội số 8.
- Hai mô hình PDS và PKDSR bản gốc ép lên cuda trong `__init__`; bản trong repo đã bỏ lệnh đó.
- Trên máy ít lõi, torch đa luồng với mô hình rất nhỏ chậm hơn một luồng hàng chục lần; `tests/conftest.py` đặt một luồng.
- `torch>=2.3` (dùng `torch.amp.GradScaler`).
- Mẫu huấn luyện được đánh số toàn cục (bước k dùng chỉ số [k·B, (k+1)·B)), nên tiếp tục từ checkpoint cho đúng dòng dữ liệu.

## 12. Cách dùng file này trong Cursor

Đặt file này ở gốc repo (ví dụ `docs/HANDOFF.md`) và thêm một dòng vào `CLAUDE.md` của repo: "Đọc `docs/HANDOFF.md` trước khi làm việc trên project này." Khi các việc ở mục 8 và 9 xong, cập nhật `docs/STATUS.md` và xóa các mục đã cũ ở đây.
