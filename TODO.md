# TODO: việc cần làm của project earsr_rt

Cập nhật lần cuối: 08/10/2026. File này là danh sách việc đang mở. Xong việc nào thì đánh dấu `[x]` và ghi ngày;
trạng thái kiểm của mã nằm ở `docs/STATUS.md`, lệnh đầy đủ ở `docs/RUNBOOK.md`, số liệu và luận điểm ở mục 2.0 của
`docs/paper-plan-ridgesr-2026-10-05.md` (bản 25).

Ký hiệu người làm: **[Bạn]** việc chỉ bạn làm được; **[labai217]** lệnh chạy trên máy GPU; **[Claude]** việc sửa mã hoặc
phân tích trên máy Mac.

## Cách làm việc với thầy (người dùng dặn ngày 07/10/2026; áp cho cả project)

**Thầy chỉ đọc khi có một bài hoàn chỉnh.** Hỏi ý thầy giữa chừng thì thầy không có căn cứ để trả lời. Vì vậy:

- Không đặt việc nào ở trạng thái "chờ thầy" hay "hỏi thầy". Người dùng và Claude tự quyết, ghi lý do vào file này.
- Mục tiêu là một **bản thảo hoàn chỉnh**: câu chuyện, mục tiêu, số liệu chứng minh đủ, bài viết xong. Khi đó mới đưa thầy
  đọc, thầy review và quyết định.
- Các dòng cũ bên dưới còn ghi "hỏi thầy", "chờ thầy đồng ý" (mục 2, 2b, 2c, và mục 1.1, E7 của kế hoạch) đọc là: quyết
  định thuộc về người dùng.

## 0. LÀM GÌ TIẾP (đọc mục này trước; cập nhật 08/10/2026)

**Bài này:** phát hiện cộng benchmark cộng cách huấn luyện, không có kiến trúc mới (kiến trúc để bài sau); đích IMAVIS; thầy
đọc khi bản thảo hoàn chỉnh. Phạm vi và kết quả: `docs/story-imavis.md`. Bản thảo: `paper/` (dựng theo `paper/README.md`).
**Không còn fold giữ kín:** không chạy lại huấn luyện với cấu hình khác. **Không đề xuất thêm thí nghiệm ngoài danh sách dưới.**

**Đã xong:** tách "có nén" khỏi "đo từ dữ liệu"; 5 fold (60 lần huấn luyện); nhận dạng trên hai bộ ảnh với ba mạng nhận
dạng; rà tài liệu lượt đầu; chuyển mô hình sang Core ML; viết lại câu đóng góp về nhận dạng theo "hai điều kiện".

### Danh sách đang dở (làm theo thứ tự; xong việc nào đánh dấu việc đó)

- [x] 08/10 **A. [Bạn, máy Mac] Commit và push** (xong):
  ```bash
  cd /Users/tql3308/Documents/ncs/paper_source_code/earsr_rt
  git add .gitignore TODO.md docs paper scripts deploy_ios/README.md results
  git commit -m "Identification reframed as two conditions; natural-image control script; Core ML export" && git push
  ```
- [x] 08/10 **B. Đo độ trễ trên iPhone 12 Pro Max: XONG** (Xcode 26.3; bản 27.0 cho ra 0,00 ms nên không dùng). Trung vị thời
  gian dự đoán, ảnh vào 48×68: SPAN 4,72 ms (chỉ CPU) và 0,80 ms (mọi đơn vị); DISP 2,33 và 0,45 ms; BSRGAN 101,23 và
  15,82 ms. Ở chế độ mọi đơn vị, cả ba mô hình chạy trọn trên Neural Engine. Số nằm ở `results/latency_ios.csv`.
  **Chữ "Real-Time" trong tiêu đề giữ được.** Điều phải viết thẳng (đã viết ở mục 6.7): trên Neural Engine cả BSRGAN cũng
  dưới 33 ms, nên ở cỡ ảnh này ngưỡng real-time không loại được mô hình lớn; lý do chọn mạng nhỏ là độ trung thực và phần
  dư thời gian, không phải tính khả thi. Chỉ có trung vị, không có phân vị 95. 14 mô hình còn lại chưa đo (tùy chọn).
- [x] 08/10 **C. [labai217] Đối chứng trên ảnh tự nhiên: XONG** (425 file, commit `300de40`). Kết quả: mục 3b4 của `docs/story-imavis.md`. Khi xong phải commit và push kết quả
  (hai dòng cuối của khối lệnh dưới). Đây là Phần 2 của góp ý ghi ở mục "Góp ý ngày 08/10" ngay bên dưới. Chỉ chấm, không huấn luyện, khoảng 30 đến 60 phút (ước lượng):
  ```bash
  git pull
  bash scripts/run_div2k_control.sh       # dòng cuối: "results/t2_div2k: 425 file" rồi "Xong."
  git add results && git commit -m "Natural-image control: published models on DIV2K, five input sizes" && git push
  ```
  Nội dung: 16 mô hình có sẵn, cùng năm mức JPEG, trên 100 ảnh DIV2K ở ảnh vào 24, 36, 48, 96, 192 px. Câu hỏi: ngưỡng "kém
  bicubic khi nén" là của ảnh tai, của cỡ ảnh nhỏ, hay của nén nói chung. Tín hiệu từ lần thử 6 ảnh, 3 mô hình trên máy Mac
  (**chưa phải kết luận**): trên ảnh tự nhiên ở JPEG 75 mô hình xấp xỉ hòa bicubic ở cả ảnh 36 px lẫn 192 px, chỉ kém ở
  JPEG 60; tức có thể không có hiệu ứng theo cỡ ảnh, nhưng ngưỡng của ảnh tai (quanh mức 85) cao hơn của ảnh tự nhiên.
  **Không làm:** chạy mạng mới, hay thêm bước khử nén để tìm số đẹp.
- [x] 08/10 **D. `git pull` phần đối chứng và báo số đo iPhone: xong.**
- [x] 08/10 **E. [Claude] Điền bài và viết phần diễn giải: XONG.** Mục 4.3 (đối chứng, quy tắc tỉ số sai số) và mục 6.7 (iPhone) đã
  viết; bản thảo không còn macro số nào trống. Còn ba việc "dè dặt" của quy tắc tỉ số (ngay dưới danh sách này).
- [ ] **A2. [Bạn, máy Mac] Commit và push** phần vừa làm:
  ```bash
  git add TODO.md docs paper scripts deploy_ios/README.md results
  git commit -m "Natural-image control interpreted; error-ratio rule; iPhone latency measured" && git push
  ```
- [ ] **F. [Bạn] Phần còn lại của rà tài liệu:** (0) **mới:** tìm xem đã có ai nêu quy tắc "SR chỉ thắng nội suy khi sai số nén
  nhỏ so với sai số nội suy" chưa (từ khóa: compressed image super-resolution, interpolation error, crossover, JPEG quality);
  bài hiện viết đây là quy tắc thực nghiệm của mình, chưa kiểm tính mới; (a) đọc toàn văn bài IWSSIP 2023 (IEEE Xplore 10180250) rồi sửa hai ô TBD ở
  mục 1 và 2 của bản thảo; (b) tìm có hệ thống trên IEEE Xplore và Scopus; (c) đối chiếu 24 mục "chưa tra" ở đầu
  `paper/refs.bib`; (d) đọc Li và cộng sự (TIFS 2019), Bulat và cộng sự (ECCV 2018).
- [ ] **G. [Claude, khi bạn bảo] Hoàn thiện bản thảo:** hình phổ sai số cho mục 4.1 (ô TBD); chọn lại ảnh cho hình định tính
  (đã có 12 ảnh của 12 người ở `results/recog/sr2`); rút abstract xuống giới hạn của tạp chí (hiện dài); highlights và các
  tuyên bố Elsevier yêu cầu; số trang cho `refs.bib`; tên khoa của đơn vị; quyền dùng ảnh EarVN1.0 và AMI trong hình; mục 6.6
  (khảo sát người xem) đang là ô TBD: không làm khảo sát thì xóa mục đó.
- [ ] **H. Đưa thầy đọc** `paper/main.pdf` khi E và G xong.

### Ba điểm dè dặt của quy tắc tỉ số sai số: việc phải làm (người dùng dặn ghi lại ngày 08/10)

Quy tắc: mô hình có sẵn hơn bicubic khi sai số do nén thêm vào nhỏ hơn khoảng 0,17 lần sai số nội suy (mục 4.3 của bản thảo,
mục 3b4 của `docs/story-imavis.md`). Ba chỗ chưa chắc, mỗi chỗ một việc:

- [ ] **Dè dặt 1: ngưỡng là thực nghiệm, mới kiểm với mô hình học bằng bicubic, ×4, nén JPEG.**
  Việc [Claude, không tốn GPU]: thêm các ô ×2 đã có sẵn trong `results/t2/` (SwinIR-light ×2 và các mô hình ×2 khác trên AMI,
  ảnh sạch và JPEG 75) vào phép tính tỉ số, xem chúng có nằm đúng phía của ngưỡng không; báo kết quả dù thuận hay nghịch.
  Phần không kiểm được bằng số liệu có sẵn (codec khác, mô hình học có nén) giữ ở mục giới hạn của bài, không chạy thêm.
- [ ] **Dè dặt 2: ranh giới 0,168 / 0,170 sắc một phần do may** (hai ô sát ngưỡng thuộc hai bộ ảnh khác nhau, phần hơn khác nhau).
  Việc [Claude, không tốn GPU]: (i) báo khoảng đổi dấu của **từng bộ ảnh** thay cho một con số chung (DIV2K: giữa 0,12 và
  0,17; AMI: giữa 0,17 và 0,36; AWEx: giữa 0,08 và 0,38; EarVN1.0: giữa 0,06 và 0,39) và viết ngưỡng là "khoảng một phần sáu";
  (ii) bootstrap theo ảnh cho tỉ số và cho dấu của phần hơn ở các ô sát ngưỡng, để biết ranh giới chắc tới đâu;
  (iii) sửa câu trong abstract và mục 4.3 nếu khoảng tin cậy cho thấy con số 0,17 viết quá chính xác.
- [ ] **Dè dặt 3: chưa rà xem quy tắc này đã có ai nêu chưa.**
  Việc [Claude, khi bạn bảo]: một lượt tìm kiếm web như lượt rà tài liệu ngày 07/10 (từ khóa: compressed image
  super-resolution, interpolation error, JPEG quality crossover, SR worse than bicubic).
  Việc [Bạn]: tìm có hệ thống trên IEEE Xplore và Scopus (đã ghi ở việc F, mục 0). Nếu đã có người nêu: bài trích dẫn và
  viết đóng góp là "kiểm quy tắc đó trên ảnh tai và chỉ ra ảnh tai nằm bên kia ngưỡng"; nếu chưa: giữ cách viết hiện tại.

### Góp ý ngày 08/10 (hai phần) và trạng thái: ĐỌC LẠI SAU KHI ĐO IPHONE XONG

Người dùng dặn ghi lại để sau khi đo iPhone còn nhớ và làm tiếp. Nguyên văn góp ý nhận được:

> Có một cách làm bài mạnh hơn mà không cần chạy thêm. Đưa kết quả AWEx vào câu đóng góp, đừng viết nó như một thất bại.
> Câu đó là: phóng ảnh chỉ giúp nhận dạng tai khi hai điều cùng đúng. Ảnh dò đã bị nén, và ảnh lớn của cùng những người đó
> được nhận tốt hơn ảnh nhỏ. EarVN1.0 thỏa cả hai: ảnh lớn 60%, ảnh nhỏ 13%, mạng học với dải JPEG thêm khoảng 10 đến 12
> điểm rank-1. AWEx không thỏa điều thứ hai: ảnh lớn 32%, ảnh nhỏ 32%, và không phương pháp nào đổi rank-1. Phần độ trung
> thực giữ nguyên: từ khoảng JPEG 85 trở xuống, mạng công bố kém bicubic, và dải JPEG lúc huấn luyện là thứ sửa được.
> Một thí nghiệm còn có thể làm câu này sắc hơn. Chấm các mạng công bố trên ảnh thường, cùng mức JPEG, ở cỡ nhỏ và cỡ lớn.
> Nếu chỉ ảnh nhỏ rơi xuống dưới bicubic, bài có thêm một phát hiện về cỡ ảnh. Nếu ảnh lớn cũng rơi như vậy, viết rằng
> ngưỡng là của nén, và phần riêng của tai là nhận dạng. Không chạy mạng mới. Không chạy khử nén để tìm số đẹp.

**Phần 1: viết AWEx thành câu đóng góp ("hai điều kiện").** Trạng thái: **đã viết vào bản thảo** (abstract, mở bài, mục 6.5,
thảo luận, kết luận), đã commit ở việc A. Không cần chạy gì thêm.

- Câu trong bài: phóng ảnh giúp nhận dạng ở nơi ảnh dò đã bị nén **và** mạng nhận dạng mất độ chính xác ở độ phân giải
  thấp ("có dư địa"); EarVN1.0 thỏa cả hai, AWEx không thỏa cái nào; "không tuyên bố SR cải thiện nhận dạng tai nói chung,
  chỉ ra khi nào thì có".
- **Chỗ Claude sửa so với góp ý:** AWEx không thỏa **cả hai** điều kiện, không chỉ điều kiện thứ hai (ảnh nhỏ của AWEx không
  có lưới khối JPEG: tỉ số biên khối 0,98; EarVN1.0 là 1,19 và 1,08). Vì vậy hai bộ ảnh không tách được điều kiện nào là
  điều kiện cần; bài nói rõ điều này và gọi đó là "điều kiện do số liệu gợi ý", hình thành sau khi thấy kết quả AWEx.
- Bằng chứng trong từng bộ ảnh, tính từ kết quả có sẵn (không chạy mới): trên EarVN1.0 (30 người), nửa ít dư địa (+25,4
  điểm) tăng +7,5, nửa nhiều dư địa (+55,9) tăng +16,4; với ResNet-50 là +5,3 và +14,6. Tương quan hạng dương nhưng chưa
  đạt ý nghĩa (p = 0,14 và 0,06). Trên AWEx người có dư địa vẫn không tăng (−1,1 và +3,4), số rất nhiễu.
- **Việc còn lại của Phần 1:** (i) bạn đọc lại các đoạn đã sửa trong `paper/main.pdf` (abstract; đoạn "When does upscaling
  help identification?" ở mục 6.5) xem có đúng ý góp ý không; (ii) sau khi có kết quả Phần 2, Claude rà lại câu chữ của
  Phần 1 cho khớp (ví dụ câu "phần riêng của ảnh tai là gì").

**Phần 2: đối chứng trên ảnh tự nhiên.** Trạng thái: **XONG ngày 08/10**, đã viết vào bản thảo. Kết quả rơi vào nhánh thứ ba:
cỡ ảnh không dời ngưỡng (ngưỡng là của nén), nhưng ảnh tai đổi dấu ở mức nén nhẹ hơn ảnh tự nhiên nhiều (93 đến 85 so với
75 đến 60). Cả hai được giải thích bằng một đại lượng, tỉ số sai số nén trên sai số nội suy, ngưỡng quanh 0,17 trên 35 ô của
bốn bộ ảnh (mục 3b4 của `docs/story-imavis.md`). Câu "phần riêng của ảnh tai" đã viết lại: là vị trí của ngưỡng, không
phải hiện tượng. Phần ghi chú bên dưới giữ để đối chiếu.

- Khi chạy xong: commit và push kết quả trên labai217, `git pull` trên máy Mac, báo Claude (việc D).
- **Claude sẽ làm (việc E):** `make_paper.py`; viết diễn giải mục 4.3 theo đúng hai nhánh của góp ý:
  - nếu chỉ ảnh nhỏ rơi dưới bicubic → bài có thêm một phát hiện về cỡ ảnh;
  - nếu ảnh lớn cũng rơi → viết rằng ngưỡng là của nén, và phần riêng của ảnh tai là benchmark và kết quả nhận dạng;
  - nhánh thứ ba mà lần thử 6 ảnh gợi ra (chưa phải kết luận): trên ảnh tự nhiên ngưỡng thấp hơn (quanh JPEG 60 đến 75) so
    với ảnh tai (quanh 85), không phụ thuộc cỡ ảnh → khi đó phần riêng của ảnh tai gồm cả "ngưỡng cao hơn".
  Sau đó sửa câu đóng góp ở abstract và mở bài cho khớp nhánh thực tế, và cập nhật mục 3b3 của `docs/story-imavis.md`.
- Giữ đúng hai điều góp ý dặn: không chạy mạng mới; không thêm bước khử nén để tìm số đẹp.

**Để dành, không làm cho bài này trừ khi bạn hoặc thầy yêu cầu:** khảo sát người xem; các mục 2b, 2c, 3, 3b bên dưới.

**Ghi chú kỹ thuật cho việc B:** môi trường chuyển mô hình là `.venv-coreml` (Python 3.9 của Xcode, torch 2.7; coremltools
không chạy đủ trên Python 3.14). Số tham khảo đo trên chính máy Mac (không dùng cho bài): SPAN 2,5 ms chỉ CPU, 0,4 ms mọi đơn
vị; BSRGAN 56 ms và 14 ms. Tùy chọn: chép một lần chạy `est` từ labai217 về để đo bằng chính trọng số của bài (README mục 1).
Xcode chỉ cho trung vị, không có phân vị 95.

## 1. Nhật ký các việc gần đây

> **Phát hiện ngày 06/10:** mô hình tinh chỉnh **chỉ trên AMI** hỏng trên ảnh có vùng sáng (AMI không có ảnh nào như vậy).
> SPAN tinh chỉnh với bicubic: 22,08 dB trên EarVN1.0 (mốc công bố 36,39); cho ra nhiễu trên 82 trên 158 ảnh. Đã tái hiện
> trên máy Mac và tìm ra cơ chế (xem `docs/STATUS.md`). Tám lần chạy đầu của khối `n2` bị loại (`results/n2_amionly/`).
> Đã sửa: tăng cường độ sáng lúc huấn luyện (mặc định), ảnh EarVN nhóm train trong khối `n2`, phép thử ảnh sáng sau mỗi lần
> huấn luyện.

- [x] 08/10 **[Bạn + Claude]** Đo iPhone bằng Xcode 26.3 (6 lần đo, ba mô hình); Claude đọc số từ ảnh chụp màn hình, điền
      `results/latency_ios.csv`, viết kết luận ở mục 6.7. Trước đó: chuyển đủ 17 mô hình sang Core ML.
- [x] 08/10 **[Claude]** Đọc kết quả đối chứng trên DIV2K; tìm ra quy tắc tỉ số sai số (35 ô, bốn bộ ảnh, ngưỡng quanh 0,17,
      tương quan hạng −0,91) từ kết quả có sẵn; thêm `error_ratio` và hình vào `make_paper.py`; viết mục 4.3, viết lại abstract,
      sửa mở bài, thảo luận, kết luận.
- [x] 08/10 **[Claude]** Xét góp ý "hai điều kiện" cho phần nhận dạng. Viết lại câu đóng góp (abstract, mở bài, mục 6.5, thảo
      luận) theo đó, kèm một phép kiểm trong từng bộ ảnh tính từ kết quả có sẵn: phần tăng theo "dư địa" của từng người.
      Ghi rõ hai điều kiện bị lẫn nhau giữa hai bộ ảnh (EarVN1.0 thỏa cả hai, AWEx không thỏa cái nào). Thêm
      `scripts/run_div2k_control.sh` và mục 4.3 chờ số.
- [x] 08/10 **[Claude]** Việc 4, phần chuẩn bị: `scripts/export_coreml.py`, `deploy_ios/README.md`, `results/latency_ios.csv`
      (chờ điền), bảng độ trễ và mục 6.7 trong bản thảo. Sáu mô hình chuyển được hết; lệch so với PyTorch do làm tròn 16 bit.
- [x] 08/10 **[labai217]** Lần chạy cuối: 24 lần huấn luyện trên fold 1 và 5, chấm trên ba bộ ảnh, nhận dạng trên hai bộ
      ảnh với ba mạng nhận dạng. Ba commit "Final run: ...".
- [x] 08/10 **[Claude]** Đọc kết quả 5 fold (kết luận về độ trung thực không đổi, mức đổi lớn nhất 0,07 dB) và nhận dạng
      (EarVN1.0 được xác nhận bằng ResNet-50; AWEx không lặp lại). Thêm `scripts/probe_blockiness.py`. Viết phần diễn
      giải vào bản thảo, sửa abstract, mở bài, thảo luận, kết luận; bỏ ghi chú nháp về ba fold.
- [x] 07/10 **[Claude]** Mã đào sâu phần nhận dạng: tách theo mức nén của file ảnh dò, AWEx làm bộ thứ hai, ResNet-50 làm
      mạng thứ hai, ảnh minh họa rải trên nhiều người; `make_paper.py` và bản thảo có thêm hai bảng. Kiểm thử bắt được một
      lỗi (dòng `ref_large` lọt vào bảng theo mức nén) và đã sửa. 150 kiểm thử qua; chạy thử trên ảnh AWEx và EarVN1.0 thật.
- [x] 07/10 **[Claude]** Rà tài liệu lượt đầu: tra 21 mục của `refs.bib`, thêm 11 bài (UERC 2017 và 2019, Rathgeb 2016 về nén và
      nhận dạng tai, Bulat 2018, CISRDCNN, AIM 2022, Ma 2024, Ohtani 2024, Li 2019, Menezes 2021, Ignatov 2021), viết lại mục
      Related Work. Thêm đồng tác giả Vinh Truong Hoang (tác giả liên hệ, HCMC Open University) vào `paper/main.tex`.
- [x] 07/10 **[Claude]** Script lần chạy cuối `scripts/make_final_jobs.sh` (24 lệnh, fold 1 và 5); `make_n2b_jobs.sh` và
      `make_n2c_jobs.sh` chỉ mở fold giữ kín khi được gọi từ script đó. Đã thử: sinh danh sách, đếm nhánh, chặn gọi trực tiếp.
- [x] 07/10 **[labai217]** Khối `n2c` (6 lần) xong và đã chấm; đo nhận dạng xong với hai mạng nhận dạng.
- [x] 07/10 **[Claude]** Đọc kết quả `n2c` và nhận dạng; chốt câu chữ P3 (mục 3b của `docs/story-imavis.md`); thêm 35 macro
      và hình định tính vào `make_paper.py`; viết phần diễn giải ở mục 6.4, 6.5 và sửa abstract, mở bài, thảo luận của bản thảo.
- [x] 07/10 **[Claude]** Bản thảo LaTeX ở `paper/` và `scripts/make_paper.py` (157 macro số, 10 bảng, 2 hình sinh từ
      `results/`; 22 macro còn TBD). Biên dịch được bằng tectonic.
- [x] 07/10 **[Claude]** Mã đo nhận dạng tai trên ảnh nhỏ thật (`earsr/recog/`, bốn script). Người dùng quyết định làm (07/10).
      Đã thử trên ảnh EarVN1.0 thật với mạng nhận dạng chưa học; 150 kiểm thử qua.
- [x] 07/10 **[Claude]** Việc 1 của bài, phần mã: kiểu suy giảm `jpegmix` và `jpegu` (`earsr/degrade/pipelines.py`), danh sách
      lệnh `scripts/make_n2c_jobs.sh`, kiểu ảnh vào `jpegmix` trong `score_n2.sh`. Đã thử: kiểm thử mới (thất bại trước khi
      sửa, qua sau khi sửa), mã băm của kiểu `est` không đổi, hai lần huấn luyện 30 bước trên CPU, chấm thử 6 ảnh mỗi bộ.
- [x] 07/10 **[Claude]** Phác câu chuyện bài báo cho IMAVIS (`docs/story-imavis.md`): một câu, vấn đề, ba luận điểm với bằng
      chứng, danh sách đóng năm việc còn thiếu, phần để dành. Mục 0 viết lại theo danh sách đó.
- [x] 07/10 **[Claude]** Xét ba góp ý nâng độ mạnh của bài, ghi vào mục 2c; viết lại thứ tự ở mục 0.
- [x] 07/10 **[Claude]** Gộp ba fold của khối `n2` trên AMI (60 người test), EarVN1.0, AWEx; so sánh ghép cặp, bootstrap theo
      người; ghi vào mục 2.0 của kế hoạch bài báo và `docs/STATUS.md`. Các số đã được tính lại một lần từ file theo ảnh.
- [x] 07/10 **[labai217]** Fold 3 và 4: 20 lần huấn luyện xong, cả 20 qua phép thử ảnh sáng; `score_n2.sh` chấm 30 mô hình
      trên ba bộ ảnh (495, 132, 264 file), có cả JPEG 93.
- [x] 06/10 **[labai217]** Hàng đợi `n2b`: 10 lần xong, cả 10 qua phép thử ảnh sáng; đã chấm trên EarVN (ảnh sạch) và AMI
      (bốn kiểu ảnh vào). `run_wild_t2.sh` xong: mọi mô hình có sẵn trên EarVN1.0, AWEx, và quét mức nén trên AMI.
- [x] 06/10 **[Claude]** Đọc kết quả `n2` và kết quả ngoài AMI; ghi vào mục 2.0 của kế hoạch bài báo (C2, C4, C5 cập nhật).
- [x] 06/10 **[Bạn]** Commit và push bản sửa (`0466f7c`); pull trên labai217; khởi động hàng đợi `n2b`.
- [x] 06/10 **[Claude]** Script chấm trên EarVN1.0 và AWEx (`scripts/run_wild_t2.sh`), đã chạy thử 12 ảnh mỗi bộ.
- [x] 06/10 **[Claude]** Tái hiện lỗi từ checkpoint, tìm cơ chế, thêm tăng cường độ sáng và phép thử ảnh sáng, chạy thử
      10 lệnh mới vài bước trên CPU, đánh dấu 8 lần chạy cũ là `excluded`.
- [x] 06/10 **[labai217]** Khối `n2` lần đầu: 8 lần huấn luyện xong (24 đến 36 phút mỗi lần), kết quả bị loại vì lỗi trên.

## 2. Quyết định đang chờ bạn

- [x] 07/10 ~~Nói với thầy về định hướng bài~~ **Thay bằng:** hoàn thành bản thảo rồi mới đưa thầy (xem đầu file). Ghi chú cũ: Hiện bài không có khối kiến trúc mới; đóng góp là
      phát hiện (SR có sẵn kém bicubic về độ trung thực khi ảnh bị nén, thứ hạng đảo) và suy giảm đo từ ảnh tai thật.
      Nên nói trước khi tốn thêm thời gian huấn luyện, và sau khi có kết quả khối `n2`.
- [ ] **`span26` trùng `span_ch28`** (cùng một file trọng số): có cho Claude loại `span26` rồi chạy lại `summarize_t2.py`
      không. Việc này ghi đè `results/t2_summary/` đã commit. `span_ch48_t44` giữ làm dòng phụ hay bỏ khỏi bảng.
- [ ] **SPAN 52 kênh:** bài SPAN gọi bản 48 kênh là SPAN-S, bản 52 kênh mới là "SPAN". Có thêm bản 52 kênh vào kho không
      (`spanx4_ch52.pth` có sẵn trong `~/Downloads/span.zip`).
- [ ] **Có chạy khối `pad` (27 lần, khoảng 25 giờ) không.** Triển vọng của N5b thấp: phần mất do thiếu ngữ cảnh mà riêng mạng
      gây ra chỉ 0,03 đến 0,06 dB với SPAN 48 kênh, dưới ngưỡng 0,10 dB. Kế hoạch vẫn ghi chạy `pad` sau N2.
- [x] 07/10 **Tách bài:** kiến trúc mới thành một bài khác, bài này là bài phát hiện cộng benchmark (làm theo hướng này; thầy
      quyết khi đọc bản thảo). Còn nợ: sửa mục 1.1, 1.2 và 2.1 của kế hoạch cho khớp `docs/story-imavis.md` khi viết bản thảo.
- [x] 07/10 **Có đo nhận dạng tai không:** có (người dùng quyết định). Mã đã viết; chờ chạy trên labai217.
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
- Đã có (07/10): khối `n2` trên ba fold và ba bộ ảnh; nhánh ước lượng hơn nhánh tổng quát 1,1 đến 2,1 dB (mục 0).
- Đang chờ: khảo sát người xem; mốc "JPEG ngẫu nhiên" (mục 0, bước 3).
- Đạt khi: mô hình học với suy giảm ước lượng hơn mô hình học với suy giảm tổng quát trên ảnh nhỏ thật.
- Điểm yếu: không có kiến trúc mới; phần "ước lượng" không hơn "bicubic rồi JPEG 75" ở bộ phân loại (58,5% so với 56,7%)
  và chỉ hơn 0,14 đến 0,26 dB khi huấn luyện; trên ảnh sạch mô hình kém trọng số công bố 1,2 đến 1,8 dB.

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

## 2c. Ba hướng nâng độ mạnh của bài (góp ý nhận ngày 07/10/2026, để tham khảo)

Nguồn: một bản góp ý kiểu phản biện do người dùng gửi. Phần "Đánh giá" là của Claude, dựa trên số liệu ba fold; xếp hạng Q của
các tạp chí nêu trong góp ý **chưa được tra lại**.

**Hướng 1: cơ chế mà "JPEG đúng mức" không giải thích được.** Giữ bài là bài ảnh tai.

- Góp ý: phải có một yếu tố ngoài phân bố mức nén (mờ, nhiễu, biên độ sáng, cấu trúc gờ tai) thắng "JPEG đúng mức" trên
  EarVN1.0 và AWEx, khoảng tin cậy không chứa 0, cỡ có ý nghĩa, và người xem chọn trên 50%. Tạp chí mở thêm: CVIU; Image and
  Vision Computing, Neurocomputing thành chỗ nên nộp. Kết quả âm vẫn là một câu hữu ích, không nâng tạp chí.
- Đánh giá: **đúng, và phải làm dù chọn hướng nào.** Số ba fold: nhánh ước lượng hơn JPEG 75 cố định 0,14 đến 0,26 dB dưới
  suy giảm ước lượng, 0,4 đến 0,6 dB ở JPEG 93, thua 0,06 đến 0,14 dB ở JPEG 75, và hơn khoảng 1 dB trên ảnh sạch (AMI: 37,86
  so với 36,88). Mẫu này khớp với "chỉ là phân bố mức nén". Chưa có nhánh nào tách mờ và nhiễu khỏi phân bố mức nén.
  Phép so trên ảnh nén tổng hợp còn có tính vòng: ảnh chấm sinh từ chính phân bố mà nhánh ước lượng học.
- Khả thi: cao, rẻ (mục 0, bước 3: 6 lần huấn luyện). Xác suất tìm ra "cơ chế": thấp (mờ đo được chỉ 0,2 đến 0,6 px).
- Ứng viên ngoài JPEG **đã có số**: mô hình tinh chỉnh chỉ trên AMI hỏng trên ảnh sáng (22,08 dB so với 36,39 dB trên
  EarVN1.0), cơ chế đã biết (biên độ kích hoạt phình qua các khối SPAB), sửa bằng tăng cường độ sáng; thêm ảnh EarVN cho
  +0,62 dB trên EarVN1.0, +0,30 và +0,19 dB trên AWEx. Đây là hiện tượng của dữ liệu phòng thí nghiệm hẹp dải sáng, chưa
  chắc riêng cho tai. Liên quan hướng C ở mục 2b.

**Hướng 2: "ảnh vào vài chục pixel đã nén", nhiều miền (tự nhiên, mặt, tai).** Đổi đề tài.

- Góp ý: phần mới phải là tương tác với cỡ ảnh (dưới một ngưỡng, thứ hạng, dung lượng mạng hoặc độ trung thực đổi theo cách
  không thấy ở ảnh lớn). Nếu các miền khác chỉ lặp lại hiệu ứng JPEG thì bài yếu đi. Trần: CVIU, có cửa ở Pattern Recognition.
- Đánh giá: **rủi ro nêu trong góp ý là thật, và số AMI hiện có nghiêng về phía rủi ro.** Dưới JPEG 75, SPAN 48 kênh kém
  bicubic 0,32 / 0,43 / 0,42 dB ở ảnh vào 24 / 36 / 48 px: không có ngưỡng theo cỡ, ảnh nhỏ nhất còn bị ít hơn. Tương tác
  với cỡ ảnh chỉ thấy ở ảnh sạch: RRDB hơn SPAN 0,49 / 0,23 / 0,15 dB ở 24 / 36 / 48 px, và mất hẳn khi nén (−0,05 / −0,04 /
  −0,02 dB). Tức ứng viên duy nhất cho "quy luật theo cỡ" là dung lượng theo cỡ ảnh trên ảnh sạch (hướng B ở mục 2b).
- Khả thi: phần chấm rẻ (DIV2K có sẵn; ảnh mặt phải tải và xem giấy phép). Nhưng tai thành một miền trong vài miền, khó làm
  chương chính của luận án SR ảnh tai. **Khuyến nghị: không chọn làm hướng chính; chỉ làm phép đối chứng trên DIV2K** (mục 0,
  bước 6).

**Hướng 3: SR làm gì với nhận dạng tai.** Khớp đề tài nhất.

- Góp ý: tạp chí mở thêm là IEEE T-BIOM. Đã có bài hội nghị cùng hướng (Markičević, Peer, Emeršič, "Improving Ear Recognition
  with Super-resolution", IWSSIP 2023: EDSR và SwinIR, ×2 và ×4, trên UERC, đo rank-1). Câu hỏi phải khác: trên ảnh tai nhỏ
  đã nén, SR hiệu quả làm gì với danh tính so với bicubic, so với JPEG đúng mức, so với suy giảm tổng quát, trên người không
  dùng để huấn luyện SR. "Sắc hơn nhưng nhận dạng không hơn hoặc kém hơn" là kết quả đáng nộp.
- Đánh giá: **đúng; đây là hướng đáng thêm nhất.** Bài IWSSIP 2023 có thật (đã tra: IEEE Xplore 10180250); câu "EDSR hơn
  bicubic về rank-1" chưa kiểm, phải đọc bài (bảng tài liệu liên quan của kế hoạch còn ô "[đọc bài]"). Lợi ích riêng cho bài
  này: nhãn danh tính là **thước đo khách quan trên ảnh nhỏ thật**, thứ kế hoạch đang ghi là thiếu (hiện chỉ có khảo sát
  người xem). Nó cũng phá tính vòng của hướng 1: so các nhánh `est`, `jpegmix`, `generic` trên ảnh thật, không cần đáp án.
- Dữ liệu có sẵn (đếm 07/10): nhóm `test` của EarVN1.0 có 41 người chưa từng dùng để huấn luyện SR, 2.060 ảnh nhỏ thật (cạnh
  ngắn 24 đến 48 px), 2.141 ảnh cỡ 96 đến 191 px và 158 ảnh từ 192 px để làm ảnh đăng ký. Nhóm `viewer` thêm 17 người, 526
  ảnh nhỏ. AWEx và người test của AMI dùng được với ảnh nén tổng hợp.
- Việc phải làm (chưa có mã): một mạng nhận dạng tai đóng băng, huấn luyện trên người không thuộc nhóm chấm, cộng một mạng
  đặc trưng tổng quát làm đối chứng (để kết luận không phụ thuộc mạng nhận dạng); ghép ảnh nhỏ đã phóng to với ảnh đăng ký
  lớn; rank-1, EER, bootstrap theo người. Ước lượng: 1 đến 2 ngày viết mã và kiểm thử, vài giờ GPU.
- Rủi ro: (1) **cần thầy đồng ý** (đề bài bỏ phần nhận dạng; E7 của kế hoạch chờ thầy); (2) kết quả "hơn rank-1 một chút"
  thì trùng bài 2023; (3) mạng nhận dạng có thể thiên về ảnh kiểu bicubic nếu huấn luyện trên ảnh thu nhỏ, phải kiểm bằng
  mạng đối chứng; (4) chưa đoán được chiều của kết quả.

**Khuyến nghị của Claude:** làm phép thử của hướng 1 ngay (rẻ, bắt buộc); chọn hướng 3 làm phần nâng bài nếu thầy đồng ý;
hướng 2 chỉ giữ ở mức một phép đối chứng.

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
- [ ] **Tăng cỡ mẫu của phần ngoài thực tế** (để trả lời trước câu "dữ liệu ít"; đều chỉ chấm, không huấn luyện):
      (a) với phân tích mô hình có sẵn, dùng mọi ảnh lớn của EarVN1.0 (1.086 ảnh, 95 người) thay vì chỉ nhóm test (158 ảnh);
      mô hình của bài thì vẫn chỉ chấm trên nhóm test; (b) AWEx thêm mức biên an toàn 1,5 lần làm phân tích phụ (409 ảnh cỡ 96,
      153 ảnh cỡ 144; kế hoạch vốn đã ghi báo cả hai mức); (c) EarVN2.0, chỉ những người không có trong EarVN1.0, làm bộ
      thứ ba (cần bạn quyết vì bộ này chưa công bố); (d) bộ phân loại N2 hiện chấm trên ảnh của 4 người: đổi sang chia chéo
      trên cả nhóm `clf` để dùng hết 13 người.
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

## 3b. Bổ sung để đủ một bài Q1, ngoài các việc ở trên (đề xuất của Claude, 07/10/2026; chưa việc nào được duyệt)

> **Trạng thái 07/10:** người dùng không duyệt việc bổ sung mở rộng. Theo `docs/story-imavis.md`: mục 3 và 9 thuộc năm việc
> của bài; mục 5 thành một hình, không chạy gì mới; mục 1, 2, 4, 6 để dành cho vòng phản biện; mục 7 bỏ; mục 8 làm lúc nộp.

Giả định: bài này là bài phát hiện cộng benchmark cộng cách huấn luyện đúng; kiến trúc mới để bài sau (người dùng đang cân
nhắc, chưa chốt). Xếp theo mức quan trọng. Các mục 1, 2, 3 chạm thẳng vào câu đóng góp chính.

- [ ] **1. Mốc có sẵn loại "SR ngoài thực tế, bản tối ưu độ trung thực"**: BSRNet, Real-ESRNet, SwinIR real-SR bản PSNR.
      Mười sáu mô hình PSNR đang có đều học bằng bicubic; các mô hình học bằng suy giảm thực tế trong kho đều là bản GAN. Chưa
      có mô hình nào vừa học suy giảm thực tế vừa tối ưu PSNR. Nếu chúng hơn bicubic khi ảnh bị nén thì câu "mọi mô hình có sẵn
      kém bicubic" phải sửa thành "mọi mô hình học bằng bicubic". Chỉ chấm; BSRNet và Real-ESRNet dùng kiến trúc RRDB đã có
      trong kho. Phải kiểm lại link trọng số. Khoảng nửa ngày mã, 1 giờ GPU.
- [ ] **2. Dao động do huấn luyện (nhiều seed).** Kế hoạch có E9 (3 seed trên một fold) nhưng chưa vào danh sách chạy. Các
      phép so đang quyết định câu chữ chỉ chênh 0,14 đến 0,26 dB; bootstrap theo người không bắt được dao động giữa các lần
      huấn luyện. Chạy SPAN, fold 2, thêm 2 seed cho mỗi nhánh `est`, `bicjpeg75`, `jpegmix`: 6 lần, khoảng 3 giờ. Cần thêm
      tham số seed tách khỏi số fold trong mã lần chạy.
- [ ] **3. Định vị so với tài liệu ngoài ảnh tai.** T1 (mục 4) mới rà SR cho tai và sinh trắc. Phải rà thêm: SR cho ảnh đã
      nén (theo trí nhớ của Claude có cuộc thi AIM 2022 về SR ảnh và video đã nén; **cần kiểm**), suy giảm ngoài thực tế
      (Real-ESRGAN, BSRGAN), ước lượng suy giảm theo miền, SR ảnh mặt nhỏ ngoài thực tế. Người phản biện sẽ dẫn các bài này để
      nói hiện tượng đã biết; bài phải nói rõ phần nào mới (ảnh vài chục pixel, ngưỡng mức nén, miền tai, mô hình real-time).
- [ ] **4. Mốc "khử nén rồi mới SR"**: một mạng khử vết JPEG có sẵn (ví dụ FBCNN) đặt trước SPAN công bố. Câu hỏi hiển nhiên
      của người phản biện: sao không khử nén trước. Chỉ chấm; phải thêm kiến trúc vào kho. Mốc này không real-time, dùng làm
      tham chiếu chất lượng.
- [ ] **5. Giải thích vì sao mô hình kém bicubic**, không chỉ báo số: phổ sai số, sai số theo vị trí trên lưới khối 8×8,
      độ nhất quán với ảnh vào (LR-PSNR đã có). Một bài phân tích cần ít nhất một hình cơ chế. Tính từ ảnh SR lưu bằng
      `evaluate.py --save-sr`; khoảng một ngày mã.
- [ ] **6. Độ bền với thứ tự suy giảm**: nén trước rồi thu nhỏ, nén hai lần, lấy mẫu màu 4:2:0. Ảnh tai thật được cắt từ
      ảnh lớn đã nén, nên chuỗi "thu nhỏ rồi nén một lần" chỉ là một giả định. Chỉ chấm trên các kiểu ảnh vào mới.
- [ ] **7. Cái giá trên ảnh sạch**: thử một nhánh trộn thêm ảnh không nén (ví dụ 20%) xem lấy lại được bao nhiêu trong 1,2
      đến 1,8 dB mà không mất trên ảnh nén. Nếu không làm thì bài phải có một hình đánh đổi. Ghép được vào đợt chạy của mục 0,
      bước 3.
- [ ] **8. Phát hành mã và benchmark**: repo công khai, trọng số mô hình của bài, script dựng lại benchmark từ dữ liệu gốc
      (không phát hành lại ảnh). Kiểm điều khoản của AMI, EarVN1.0, AWEx. Với bài benchmark, đây là một phần của đóng góp.
- [ ] **9. Phương án dự phòng cho chữ "real-time"** nếu không tìm được Jetson: đo trên điện thoại Android (đã có hướng dẫn)
      hoặc CPU qua ONNX Runtime, và sửa mục 1.5 của kế hoạch cho khớp thiết bị đo được.

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
