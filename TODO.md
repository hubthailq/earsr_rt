# TODO: việc cần làm của project earsr_rt

Cập nhật lần cuối: 06/10/2026. File này là danh sách việc đang mở. Xong việc nào thì đánh dấu `[x]` và ghi ngày;
trạng thái kiểm của mã nằm ở `docs/STATUS.md`, lệnh đầy đủ ở `docs/RUNBOOK.md`, số liệu và luận điểm ở mục 2.0 của
`docs/paper-plan-ridgesr-2026-10-05.md` (bản 25).

Ký hiệu người làm: **[Bạn]** việc chỉ bạn làm được; **[labai217]** lệnh chạy trên máy GPU; **[Claude]** việc sửa mã hoặc
phân tích trên máy Mac.

## 0. LÀM GÌ TIẾP (đọc mục này trước; cập nhật 07/10/2026)

**Quyết định của người dùng (06/10, đêm):** làm trước **một mô hình real-time tốt hơn các mô hình có sẵn** (hướng A). Hướng
kiến trúc (B, C ở mục 2b) để sau; câu hỏi cho thầy vẫn còn mở.

**Trạng thái:** khối `n2` chạy lại trên fold 2 đã xong và đã phân tích (mục 2.0 của kế hoạch bài báo). Còn thiếu hai thứ, đang
làm: (a) các mô hình `n2` chưa được chấm trên EarVN1.0 và AWEx với ảnh vào có nén; (b) mới một fold, tiêu chí cần ba.

**Khi người dùng hỏi "làm gì tiếp", trả lời theo đúng thứ tự dưới đây.**

1. **[Bạn] Commit và push** (script `score_n2.sh`, `make_n2b_jobs.sh` có thêm FOLDS, kết quả EarVN dời sang `results/n2_earvn/`).
2. **[labai217] Chấm 10 mô hình fold 2 trên cả ba bộ ảnh** (khoảng 20 đến 30 phút), rồi đẩy về:
   ```bash
   git pull
   bash scripts/score_n2.sh          # cuối cùng in số file của results/n2, n2_earvn, n2_awex rồi "Xong."
   git add results && git commit -m "N2 fold 2 scored on AMI, EarVN, AWEx" && git push
   ```
   Số file mong đợi với 10 lần chạy: `n2` 195, `n2_earvn` 52, `n2_awex` 104.
3. **[labai217] Huấn luyện fold 3 và 4: 20 lần, khoảng 9 đến 10 giờ** (để qua đêm):
   ```bash
   FOLDS="3 4" bash scripts/make_n2b_jobs.sh jobs/n2b_f34.txt     # phải in: 20 lệnh, fold: 3 4
   nohup python scripts/run_queue.py jobs/n2b_f34.txt > n2b_f34.log 2>&1 &
   ```
   Kiểm sau khi xong: `tail -3 n2b_f34.log` có `{'done': 20, 'failed': 0, ...}`, và
   `grep -c "phép thử ảnh sáng: ỔN ĐỊNH" jobs/n2b_f34.txt.logs/*.log | grep -c ":1"` ra 20.
4. **[labai217] Chấm lại sau khi fold 3 và 4 xong** (script bỏ qua file đã có, chỉ chấm lần chạy mới), rồi đẩy về:
   ```bash
   bash scripts/score_n2.sh
   git add results && git commit -m "N2 folds 3 and 4 trained and scored" && git push
   ```
   Số file mong đợi với 30 lần chạy: `n2` 495, `n2_earvn` 132, `n2_awex` 264.
5. **[Claude] Phân tích.** Sau bước 2: mô hình của bài so với mô hình có sẵn và bicubic trên EarVN1.0 và AWEx với ảnh nén; ước
   lượng so với JPEG 75 ở JPEG 93. Sau bước 4: gộp ba fold (60 người test của AMI), so sánh ghép cặp, bootstrap theo người,
   áp tiêu chí. Cập nhật mục 2.0 của kế hoạch, `docs/STATUS.md` và file này.
6. **Tiếp theo của hướng A:** khảo sát người xem trên ảnh nhỏ thật (nửa sau của tiêu chí N2) → dò tốc độ học (khối `lr`) → ba
   giao thức huấn luyện (T6 ii) → điểm kiểm tra 2 → chạy chính trên 5 fold (S2) → S4, S5 → đo trên Jetson → dựng bảng, hình và
   viết. Khối `pad` bỏ hoặc để cuối. Mọi khối huấn luyện đều thêm ảnh EarVN (kết quả `n2` cho thấy bắt buộc).
7. **[Bạn] Hỏi thầy** khi tiện: bài có cần một kiến trúc mới không. Nếu có: phép thử rẻ của hướng B hoặc C (mục 2b) trước các
   khối dài.
8. **Các quyết định nhỏ còn treo** (mục 2): loại `span26` (trùng `span_ch28`) rồi chạy lại `summarize_t2.py`; có thêm SPAN 52
   kênh không; bộ dữ liệu thứ ba (EarVN2.0, Imperial College, UERC).

Việc song song không phụ thuộc gì: rà tài liệu có hệ thống (mục 4), tìm Jetson Nano (mục 3).

## 1. Nhật ký các việc gần đây

> **Phát hiện ngày 06/10:** mô hình tinh chỉnh **chỉ trên AMI** hỏng trên ảnh có vùng sáng (AMI không có ảnh nào như vậy).
> SPAN tinh chỉnh với bicubic: 22,08 dB trên EarVN1.0 (mốc công bố 36,39); cho ra nhiễu trên 82 trên 158 ảnh. Đã tái hiện
> trên máy Mac và tìm ra cơ chế (xem `docs/STATUS.md`). Tám lần chạy đầu của khối `n2` bị loại (`results/n2_amionly/`).
> Đã sửa: tăng cường độ sáng lúc huấn luyện (mặc định), ảnh EarVN nhóm train trong khối `n2`, phép thử ảnh sáng sau mỗi lần
> huấn luyện.

- [x] 06/10 **[labai217]** Hàng đợi `n2b`: 10 lần xong, cả 10 qua phép thử ảnh sáng; đã chấm trên EarVN (ảnh sạch) và AMI
      (bốn kiểu ảnh vào). `run_wild_t2.sh` xong: mọi mô hình có sẵn trên EarVN1.0, AWEx, và quét mức nén trên AMI.
- [x] 06/10 **[Claude]** Đọc kết quả `n2` và kết quả ngoài AMI; ghi vào mục 2.0 của kế hoạch bài báo (C2, C4, C5 cập nhật).
- [x] 06/10 **[Bạn]** Commit và push bản sửa (`0466f7c`); pull trên labai217; khởi động hàng đợi `n2b`.
- [x] 06/10 **[Claude]** Script chấm trên EarVN1.0 và AWEx (`scripts/run_wild_t2.sh`), đã chạy thử 12 ảnh mỗi bộ.
- [x] 06/10 **[Claude]** Tái hiện lỗi từ checkpoint, tìm cơ chế, thêm tăng cường độ sáng và phép thử ảnh sáng, chạy thử
      10 lệnh mới vài bước trên CPU, đánh dấu 8 lần chạy cũ là `excluded`.
- [x] 06/10 **[labai217]** Khối `n2` lần đầu: 8 lần huấn luyện xong (24 đến 36 phút mỗi lần), kết quả bị loại vì lỗi trên.

## 2. Quyết định đang chờ bạn

- [ ] **Nói với thầy về định hướng bài, mang theo các hướng ở mục 2b.** Hiện bài không có khối kiến trúc mới; đóng góp là
      phát hiện (SR có sẵn kém bicubic về độ trung thực khi ảnh bị nén, thứ hạng đảo) và suy giảm đo từ ảnh tai thật.
      Nên nói trước khi tốn thêm thời gian huấn luyện, và sau khi có kết quả khối `n2`.
- [ ] **`span26` trùng `span_ch28`** (cùng một file trọng số): có cho Claude loại `span26` rồi chạy lại `summarize_t2.py`
      không. Việc này ghi đè `results/t2_summary/` đã commit. `span_ch48_t44` giữ làm dòng phụ hay bỏ khỏi bảng.
- [ ] **SPAN 52 kênh:** bài SPAN gọi bản 48 kênh là SPAN-S, bản 52 kênh mới là "SPAN". Có thêm bản 52 kênh vào kho không
      (`spanx4_ch52.pth` có sẵn trong `~/Downloads/span.zip`).
- [ ] **Có chạy khối `pad` (27 lần, khoảng 25 giờ) không.** Triển vọng của N5b thấp: phần mất do thiếu ngữ cảnh mà riêng mạng
      gây ra chỉ 0,03 đến 0,06 dB với SPAN 48 kênh, dưới ngưỡng 0,10 dB. Kế hoạch vẫn ghi chạy `pad` sau N2.
- [ ] **Tiêu đề bài:** số liệu hợp với tiêu đề 2 ("Degradation Matters More Than Architecture"); chốt sau điểm kiểm tra 2.
- [ ] **Tạp chí:** chốt sau điểm kiểm tra 2 (gợi ý ở `docs/HANDOFF.md` mục 4.5; phải tra lại xếp hạng Q).

## 2b. Các hướng đi để bàn với thầy

Mục tiêu của đề bài: ảnh tai to hơn, rõ hơn, real-time, và hơn các mô hình SR khác. SR có giúp ảnh tai (ô chính, ảnh sạch:
bicubic 36,73 dB lên 39,04 dB; LPIPS 0,244 xuống 0,163). Câu hỏi còn mở là "hơn các mô hình khác" bằng cách nào.
Các hướng không loại trừ nhau. Quyết định sau khi có kết quả khối `n2` và ý kiến của thầy.

**Hướng A: suy giảm theo miền (hướng đang làm, N2).**

- Ý: giữ thân có sẵn (SPAN-S, DISP), huấn luyện với suy giảm đo từ ảnh tai thật.
- Đã có: phát hiện giai đoạn 1 (16/16 mô hình kém bicubic về độ trung thực khi nén, thứ hạng đảo); bộ phân loại
  (ước lượng 58,5%, tổng quát 78,0%, bicubic 80,1%; càng gần 50% càng giống ảnh thật).
- Đang chờ: kết quả khối `n2`; khảo sát người xem.
- Đạt khi: mô hình học với suy giảm ước lượng hơn mô hình học với suy giảm tổng quát trên ảnh nhỏ thật.
- Điểm yếu: không có kiến trúc mới; phần "ước lượng" không hơn "bicubic rồi JPEG 75" ở bộ phân loại (58,5% so với 56,7%).

**Hướng B: cỡ mạng theo ngân sách độ trễ cho ảnh vào cực nhỏ (mới, chưa có thí nghiệm nào).**

- Ý: ảnh tai chỉ 24 đến 48 px nên trong 33 ms chạy được mạng lớn hơn các mô hình "hiệu quả" nhiều lần; chọn cỡ thân lấp đầy
  ngân sách thay vì nhỏ nhất có thể. Khớp với mục 1.5 của kế hoạch ("mốc phải vượt là mô hình tinh chỉnh tốt nhất trong ngân sách").
- Căn cứ (suy giảm bicubic, 700 ảnh): RRDB (16,7 triệu tham số) hơn SPAN 48 kênh 0,14 / 0,23 / 0,49 dB ở ảnh vào
  48 / 36 / 24 px. Ảnh càng nhỏ, dung lượng càng có giá trị.
- Ba điều chưa biết, theo thứ tự quan trọng:
  1. Đường cong không đều. Mạng 1,5 triệu tham số (EDSR-baseline, MSRResNet) **không hơn** SPAN 48 kênh ở ô chính
     (39,03 và 39,06 so với 39,04 dB); phần hơn chỉ thấy ở RRDB, lớn gấp 39 lần (39,28 dB). Chưa có điểm nào ở giữa.
  2. Khi suy giảm lệch (JPEG 75) thì dung lượng không giúp gì (−0,04 dB). Chưa biết sau khi huấn luyện đúng suy giảm thì dung
     lượng có giúp không; nếu có thì hai hướng cộng được.
  3. Ngân sách thật trên Jetson (chưa có thiết bị). Trên RTX 3080: SPAN 48 kênh 0,50 ms, EDSR-baseline 1,18 ms, RRDB 13,39 ms.
- Phép thử rẻ đầu tiên (chỉ khi quyết định theo hướng này): tinh chỉnh RRDB và SPAN 48 kênh với cùng suy giảm ước lượng, fold 2,
  rồi so ở ba cỡ ảnh vào. Cần sửa `make_jobs.py` hoặc gọi `train.py` tay; chưa ước lượng thời gian huấn luyện RRDB.
- Rủi ro: "dùng mạng lớn hơn" dễ bị coi là hiển nhiên. Phải phát biểu thành nguyên tắc chọn cỡ theo ngân sách, kèm đường chất
  lượng theo độ trễ và so ở cùng độ trễ (mục 1.8b của kế hoạch).

**Hướng C: khối cơ bản ổn định hơn SPAN khi tinh chỉnh trên dữ liệu nhỏ (manh mối từ lỗi ngày 06/10; chưa có thí nghiệm nào).**

- Căn cứ: khi tinh chỉnh chỉ trên AMI, SPAN cho ra nhiễu trên ảnh sáng. Đo được: biên độ kích hoạt phình dần qua các khối
  SPAB (khối 2: 45, khối 6: 2.333; mô hình công bố: dưới 4). DISP (họ khác) bị nhẹ hơn nhiều (−2,6 dB so với −14,3 dB).
- Ý: thiết kế lại khối để biên độ tự bị chặn, sao cho mô hình bền khi tinh chỉnh trên một bộ ảnh nhỏ và đồng nhất. Đây sẽ là
  đóng góp kiến trúc xuất phát từ đúng bài toán của bài.
- Chưa biết: thủ phạm có đúng là phép attention không tham số của SPAN không (mới thấy sự phình to qua các khối, chưa tách
  nguyên nhân); và tăng cường độ sáng đã chữa được triệu chứng, nên khối mới phải cho thấy lợi ích vượt hơn thế (bền mà
  không cần tăng cường, hoặc PSNR cao hơn ở cùng độ trễ).
- Phép thử rẻ đầu tiên (chỉ khi được chọn): tinh chỉnh chỉ trên AMI, không tăng cường, với vài biến thể chặn biên độ, rồi chạy
  phép thử ảnh sáng. Mỗi lần khoảng 25 phút trên RTX 3080.

## 3. Thí nghiệm còn lại, theo thứ tự của kế hoạch

- [x] 06/10 **[labai217] Kiểm độ tổng quát của phát hiện giai đoạn 1 ngoài AMI** (xong; kết quả ở mục 0 và mục 2.0 của kế hoạch) (chỉ chấm, không huấn luyện; khoảng 1 đến 2 giờ):
      `nohup bash scripts/run_wild_t2.sh > wild_t2.log 2>&1 &`, chạy sau khi hàng đợi `n2b` xong, rồi commit `results/`.
      Script chấm mọi mô hình có trọng số công bố trên EarVN1.0 (nhóm test: 158 ảnh, cỡ 96) và AWEx (217 ảnh cỡ 96, 60 ảnh
      cỡ 144) với ảnh vào bicubic, JPEG 75, JPEG 93 và suy giảm ước lượng; và quét mức nén 60, 85, 93 trên AMI.
      Đã chạy thử 12 ảnh mỗi bộ trên máy Mac (06/10). Tín hiệu sơ bộ, **chưa phải kết luận**: ở JPEG 75 mô hình kém bicubic
      trên cả ba bộ; ở JPEG 93 mô hình lại hơn bicubic; với suy giảm ước lượng thì xấp xỉ hòa. Nếu lần chạy đủ xác nhận,
      phát hiện phải viết là "lợi thế so với bicubic biến mất, và thành âm khi nén từ khoảng mức 85 trở xuống".
- [ ] **[Bạn] Bộ dữ liệu thứ ba ngoài EarVN1.0 và AWEx.** Ứng viên: EarVN2.0 (có sẵn trên máy Mac: 716 người, 134.024 ảnh,
      1.696 ảnh có cạnh ngắn từ 192 px; phải bỏ những người trùng với EarVN1.0, và bộ này chưa công bố); bộ ảnh tai ngoài
      thực tế của Imperial College; bộ UERC. Mọi bộ ngoài thực tế đều ít ảnh lớn, nên đáp án chỉ có ở cỡ 96.
- [ ] Hướng B, phép thử đầu (**chỉ khi được chọn**): xem mục 2b.

- [ ] **N2 (b): khảo sát người xem trên ảnh nhỏ thật** của nhóm `viewer` (17 người EarVN1.0). Đây là nửa sau của tiêu chí N2.
      Công cụ: `evaluate.py --save-sr`, `viewer_study.py make / analyze`. Trang HTML chưa từng mở thử với người thật.
- [ ] Khối `pad` (nếu quyết định chạy): `make_jobs.py pad`, rồi `check_criteria.py n5b-gain`.
- [ ] Dò tốc độ học (khối `lr`, fold 2): ba mức cho mỗi mốc, chọn trên validation.
- [ ] T6 (ii): ba giao thức huấn luyện, rồi `check_criteria.py n5a`.
- [ ] T6 (iii): có cần tiền huấn luyện không.
- [ ] T6 (iv), (v): **chỉ khi** `pad` hoặc phép thử ngữ cảnh ủng hộ. Cần tiền huấn luyện trên DIV2K (đắt).
- [ ] **Điểm kiểm tra 2:** chốt giao thức, thân, kiểu suy giảm chính; viết lại mục 1.4 và Phần 2 của kế hoạch.
- [ ] S2: ô chính trên cả 5 fold. **Lần đầu được dùng fold 1 và 5; không chạy trước điểm kiểm tra 2.**
- [ ] S4: đường chất lượng theo cỡ ảnh vào (`run_size_sweep.py`), cho hình 2.
- [ ] S5: test ngoài thực tế trên EarVN1.0 và AWEx. Dựng lại benchmark AWEx bằng mã đã sửa (đọc được PNG 4 kênh):
      217 ảnh ở cỡ 96 và 60 ảnh ở cỡ 144 với biên an toàn 2 lần.
- [ ] T3: đo độ trễ trên Jetson Nano (`deploy_jetson/`). **Chưa có thiết bị.** Thiếu số này thì C8 (real-time) không viết được.
      SAFMN++ và SMFANet không xuất được ONNX opset 13 nên sẽ không có số trên Jetson.
- [ ] Đo trên Android (`docs/ANDROID.md`), nếu cần số thứ hai.
- [ ] Tùy chọn, sau điểm kiểm tra 2: T4 (cổng oracle, cho N3) và T5 (đầu phụ cấu trúc, cho N1).
      N1 cần xác nhận bộ điểm mốc tai của Imperial College (phần nào có 55 điểm mốc, điều khoản sử dụng).

## 4. Chuẩn bị bài báo

- [x] 06/10 **[Bạn] Giấy phép dùng ảnh AMI cho hình minh họa: đã có** (bạn báo ngày 06/10). Lưu thư đồng ý cùng hồ sơ nộp bài.
- [ ] **T1: rà tài liệu có hệ thống** về SR cho ảnh tai và sinh trắc học vùng nhỏ (IEEE Xplore, Scopus, Google Scholar);
      đọc toàn văn các dòng "chỉ tồn tại" trong Bảng 1 (mục 2.5b). Chưa xong thì không được viết "chưa ai làm".
- [ ] Sinh lại mọi con số ghép cặp ở mục 2.0 của kế hoạch bằng `compare_table.py` (hiện vài số được tính trực tiếp từ file
      theo ảnh). Không gõ tay con số vào bản thảo.
- [ ] Dựng bảng và hình bằng script: `make_tables.py`, `compare_table.py`, `make_figures.py`. Bảng 3 phải có cả cột độ trung thực
      lẫn LPIPS, DISTS; hình 4 phải có đường của bicubic.
- [ ] Những điều bài phải nói rõ (đã ghi trong README, mục "Khác với kế hoạch"):
      chấm ở FP32 với TF32 tắt; bản 48 kênh của SPAN tên là SPAN-S; trọng số các mô hình 2026 ở dạng đã gộp nhánh;
      MS-SSIM dùng số tầng giảm theo cỡ ảnh; dữ liệu tiền huấn luyện của MSRResNet không rõ; bộ phân loại N2 chấm trên ảnh
      của 4 người; nhân mờ không ước lượng được từ ảnh nhỏ đã nén; AMI chụp trong nhà, 100 người.
- [ ] Làm theo quy trình mục 2.7 của kế hoạch khi có kết quả cuối: xác định kịch bản, điền `[X]`, chọn nhánh câu, bình luận
      từng bảng, đối chiếu chéo mọi con số trong abstract.
- [ ] **[Bạn] Đồng bộ bản 25 của kế hoạch lên claude.ai** (bản ở đó còn là bản 24).

## 5. Mã và hạ tầng

- [ ] Huấn luyện chưa có cơ chế chống hết bộ nhớ GPU. Nếu labai217 hay bị người khác dùng GPU: thêm kiểu "chờ GPU trống
      rồi chạy tiếp" (không cho rơi xuống CPU). Hiện tại: lần chạy dừng, gọi lại cùng lệnh thì tiếp tục từ checkpoint.
- [ ] Các khối huấn luyện khác (`pad`, `lr`, T6, S2) hiện chỉ dùng AMI. Tăng cường độ sáng đã bật mặc định cho chúng,
      nhưng chưa quyết có thêm ảnh EarVN cho mọi khối hay không; quyết sau khi có kết quả khối `n2` chạy lại.
- [ ] Kiểm xem các thân khác (ERRN, RLFN) có cùng kiểu mất ổn định khi tinh chỉnh không; DISP bị nhẹ hơn SPAN nhưng cũng bị.
- [ ] Rà mã độc lập một lượt: đường GPU, tiếp tục từ checkpoint, thống kê trong `earsr/report/criteria.py`.
- [ ] `results/README.md` còn mô tả kết quả sơ bộ trên CPU; nay `results/` đã là kết quả GPU. Viết lại.
- [ ] `docs/HANDOFF.md` mục 5 (số sơ bộ) và mục 8 (việc tiếp theo) đã cũ; cập nhật hoặc trỏ sang `TODO.md` và mục 2.0 của kế hoạch.
- [ ] Huấn luyện có vẻ nghẽn ở khâu nạp dữ liệu (AMP không nhanh hơn FP32: 5,8 so với 6,0 bước mỗi giây). Thử tăng
      `--workers` trước khi chạy các khối lớn; 20.000 bước L1 hiện mất khoảng 0,9 giờ.
- [ ] `grad_psnr` vẫn trả về vô cực khi hai ảnh trùng khít (LR-PSNR đã được chặn ở 100 dB). Chặn tương tự nếu đưa cột này vào bảng.
- [ ] Chưa tính mức chênh nhỏ nhất phát hiện được cho LPIPS (chỉ cần nếu làm N3, tiêu chí theo LPIPS).
- [ ] Chưa có trong kho: ECBSR, HAT; lượng tử hóa INT8; phân rã độ trễ theo tầng.
- [ ] Máy Mac không tính được LPIPS, DISTS (lỗi chứng chỉ SSL của bản Python này). Không chặn việc gì; sửa nếu muốn thử
      số đo cảm nhận tại chỗ.
- [ ] **[Bạn]** Xóa file rỗng `~/Downloads/AMI/.w_test` (phiên trước lỡ tạo; bản chép trong project không có file này).

## 6. Điều dễ quên khi đồng bộ hai máy

- `git` chỉ mang mã, `results/`, `configs/`, `splits/`. **`data/`, `weights/`, `runs/`, `jobs/` không đi theo git**: thêm dữ
  liệu hoặc trọng số mới thì phải chép riêng (rsync), và checkpoint cùng log huấn luyện chỉ nằm trên labai217.
- Sau mỗi lần chạy trên labai217: commit và push `results/` ở đó, rồi `git pull` trên máy Mac. Kiểm `git log` có commit kết quả
  chưa trước khi phân tích.
- Trên máy Mac dùng `python3`, `python3 -m pytest`; trên labai217 dùng `python` trong môi trường ảo.
- Ba quy tắc không được phá: fold 1 và 5 giữ kín tới S2; mọi lựa chọn chỉ trên 10 người validation của fold;
  `configs/criteria.yaml`, `splits/ami_5fold.json`, `splits/earvn_roles.json`, `configs/degrade_estimated.json` không sửa tay.

## 7. Đã xong (để khỏi làm lại)

- [x] 06/10: dữ liệu và 24 bộ trọng số vào đúng chỗ trên cả hai máy; EDSR-baseline và SPAN 48 kênh dùng trọng số chính thức.
- [x] 06/10: smoke test trên RTX 3080 đạt hết; chấm ở FP32 đầy đủ (TF32 tắt); cơ chế ưu tiên GPU, hết bộ nhớ thì xuống CPU.
- [x] 06/10: **giai đoạn 1** chạy đủ trên GPU (T2 với 24 tên mô hình, phép thử ngữ cảnh, độ trễ trên máy, xuất ONNX).
- [x] 06/10: **điểm kiểm tra 1.** Ngưỡng trong `criteria.yaml` đã rà sau T2, không đổi (ghi ở mục `history`).
- [x] 06/10: N2, phần đầu: file vai của EarVN1.0 và tham số suy giảm đã chốt và commit; bộ phân loại "mô phỏng hay thật" cho
      tiêu chí N2 (a) đạt (ước lượng 58,5%, tổng quát 78,0%, bicubic 80,1%, bicubic rồi JPEG 75 56,7%).
- [x] 06/10: kế hoạch bài báo lên bản 25 (mục 2.0 với số thật; C2, C3, C5 viết hẹp lại).
