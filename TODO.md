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
  dư thời gian, không phải tính khả thi. Chỉ có trung vị, không có phân vị 95. **Đo thêm ba mốc (08/10):** EDSR-baseline 12,75 / 1,76 ms; RRDB 107,91 / 15,53 ms;
  SwinIR-light 61,47 / 40,13 ms (không đạt ngưỡng ở cả hai chế độ; 240 trên 1.280 phép tính vẫn nằm trên CPU). Tổng cộng 6
  mô hình đã đo; 11 mô hình còn lại không đo.
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
- [x] 08/10 **F (phần Claude làm được): đối chiếu danh mục tham khảo.** Cả 44 mục của `paper/refs.bib` đã được đối chiếu với
  Crossref (tên bài, tác giả, nơi đăng, năm, trang, DOI) hoặc trang arXiv, NeurIPS, CVF. Sửa: chính tả tên Markićević, trang và
  DOI của mọi mục, năm của ESRGAN và AIM 2022 theo kỷ yếu. Bỏ 2 mục không kiểm đủ (Bogatyrev BMVC 2024, Menezes 2021).
- [x] 08/10 **G. Hoàn thiện bản thảo: XONG.** Viết lại toàn bộ văn bản theo văn phong chừng mực (người dùng yêu cầu: bỏ dấu gạch
  dài, bỏ lời khẳng định mạnh); gỡ mọi ô TBD và ghi chú nháp; bỏ mục khảo sát người xem; đổi tiêu đề; hình định tính lấy ảnh
  của bốn người; soạn `paper/highlights.txt`. (Mục khai dùng AI do Claude tự thêm đã được gỡ theo yêu cầu của người dùng ngày 08/10; không thêm lại.) Bản thảo 35 trang, 14 bảng, 4 hình.
- [x] 08/10 **G2. Sắp lại phần mở bài để đưa điểm riêng lên trước** (người dùng hỏi): thêm một đoạn nói thẳng điều đã biết và
  ba câu hỏi bài trả lời; ba đoạn kết quả theo ba câu hỏi đó; danh sách đóng góp thành bốn mục, phần ảnh thật thành mục riêng.
- [x] 08/10 **G3. Hình minh họa (Hình 4)** (người dùng hỏi vì hàng 3 trông "ảo"): thêm cột `Input` là ảnh gốc lặp điểm ảnh
  4×4 (đọc từ `data/raw/EarVN1.0`, tham số `--earvn-raw` của `make_paper.py`); đổi ảnh hàng 1 (ảnh 40×146 px cắt giữa
  không còn thấy vành tai) sang ảnh số 1 của `results/recog/sr2`. Hình vẫn là PNG do mã sinh. Hai câu nhận xét bằng mắt tôi tự thêm vào 6.5
  (ảnh mịn hơn; BSRGAN vẽ ra cạnh không có trong ảnh vào) đã gỡ theo yêu cầu người dùng: không thêm lại.
- [x] 08/10 **G4. Ba chỗ sửa theo nhận xét phản biện thử** (người dùng dán nhận xét, đồng ý sửa): (1) câu cuối đoạn "Relation to
  previous findings" không còn so mạng nhỏ đã huấn luyện có nén với mạng lớn học bằng bicubic; viết rõ là chưa huấn luyện mạng
  lớn có nén; (2) gộp đóng góp 3 và 4 thành một, bỏ số mili-giây khỏi danh sách đóng góp (vẫn ở tóm tắt và mục 6.6);
  (3) thêm một câu giới hạn: chưa so với cách khử nén bằng mạng riêng rồi mới phóng. Bài thành 35 trang.
  (Ghi chú lúc đó: mốc khử nén rồi phóng để dành cho vòng sửa. Cùng ngày người dùng đổi ý và duyệt làm trước khi nộp: việc J.)
- [x] 08/10 **G5. Hai câu về AWEx viết cùng mức với đoạn "hai điều kiện"** (nhận xét phản biện thử, người dùng dán): câu kết
  không còn giải thích AWEx bằng riêng "không có cấu trúc nén"; câu tổng kết cuối 6.5 bỏ chữ "depended on". Cả hai nay nêu
  hai khác biệt (không thấy khối JPEG; khoảng trống ảnh lớn/ảnh nhỏ nhỏ hơn nhiều so với EarVN1.0) và nói dữ liệu không tách được.
  Viết "much smaller", không viết "gần bằng không": trên AWEx khoảng trống là 0,4 điểm (ResNet-18) và 3,3 điểm (ResNet-50).
- [x] 08/10 **G6. Phần đầu bài viết cùng mức với câu kết** (nhận xét phản biện thử; người dùng bảo sửa chỗ nào nhận xét đúng):
  tóm tắt (242 từ), đoạn trả lời câu hỏi thứ hai và đóng góp thứ hai không còn viết "suggest two conditions"; nay nêu: tăng trên
  EarVN1.0, không đổi trên AWEx, hai khác biệt xét sau khi có kết quả và không tách được. Dòng highlights thứ năm sửa theo.
  Mở bài: "As a result" đổi thành "Consistent with this". Không hạ thêm phần nhận dạng nữa: ngưỡng tỉ số của ảnh tai cao hơn
  ảnh tự nhiên khoảng hai lần (0,26 đến 0,28 so với 0,13 đến 0,17) trong khi tỉ số ở JPEG 75 chênh bảy lần (0,69 so với 0,10),
  nên độ mịn giải thích đúng chiều; đây là lý lẽ trả lời nếu phản biện thật hỏi.
- [ ] **J. Mốc cho vòng phản biện, đưa vào bài trước khi nộp** (người dùng duyệt ngày 08/10 sau hai vòng phản biện thử;
  danh sách đóng của `docs/story-imavis.md` mở lại ĐÚNG MỘT LẦN cho ba việc này rồi đóng). Mỗi việc một cấu hình, chốt trước
  khi chạy, ra số nào báo số đó; không chạy lại với cấu hình khác để tìm số đẹp.
  - **J1. Khử nén rồi mới phóng** (chỉ chấm): FBCNN bản màu, chế độ mù, rồi bicubic (`fbcnn_bicubic`) và rồi SPAN công bố
    (`fbcnn_span_ch48`). Kèm đo độ trễ của chuỗi, vì bài là bài real-time. FBCNN có 71,9 triệu tham số (đã đếm).
  - **J2a. RRDB học có nén, trọng số có sẵn** (chỉ chấm): `bsrnet`, `realesrnet` (cùng kiến trúc RRDB 16,7 triệu tham số).
  - **J2b. Tinh chỉnh RRDB với đúng công thức của SPAN nhánh est** (huấn luyện mới, 5 lần): `scripts/make_rrdb_jobs.sh`;
    mã lần chạy `REV_rrdb-zero+xearvn_..._est_f<fold>`, gộp thành nhánh `rrdb/est`. Khác SPAN đúng một chỗ: bật `--amp`, vì
    RRDB ở lô 64 cần khoảng 12 GB ở FP32 (ước từ phép thử trên CPU), quá RTX 3080. Fold 2, 3, 4 trước; chỉ khi cả ba qua
    phép thử ảnh sáng mới chạy fold 1 và 5, một lần. Nếu phải chỉnh (ví dụ tốc độ học) thì chỉ chỉnh trên fold 2, 3, 4.
  - [x] 08/10 Mã đã viết và thử trên Mac: `earsr/models/zoo/fbcnn.py` (rút từ kho của tác giả, Apache-2.0), kho mô hình có
    `bsrnet`, `realesrnet`, `RESTORERS`, `CHAINS`; `scripts/run_review.sh`, `scripts/summarize_review.py`,
    `scripts/make_rrdb_jobs.sh`; `bench_local.py --models`; 7 kiểm thử mới (`tests/test_review.py`), tổng 157 qua.
    Ba bộ trọng số nạp chặt được; FBCNN tăng PSNR trên Set5 nén (JPEG 10: 26,58 lên 29,23 dB) và đoán đúng mức nén.
    Phép thử nối dây trên 10 ảnh AMI ở 36 px, JPEG 75 (KHÔNG phải kết quả): bicubic 33,96; SPAN công bố 33,52; BSRNet 33,67;
    Real-ESRNet 32,49; FBCNN rồi bicubic 34,39; FBCNN rồi SPAN 34,72 dB. Chuỗi FBCNN rồi SPAN chậm hơn SPAN khoảng 6,5 lần
    trên CPU máy Mac (35 so với 5,5 ms).
  - [x] 08/10 **[Bạn, máy Mac]** commit và push (commit `8b91bbe`).
  - [x] 08/10 **[Bạn, labai217] Bước 1** (commit `fb4e1b7`, đã pull về Mac)**:** tải ba bộ trọng số mới, rồi `nohup bash scripts/run_review.sh > review.log 2>&1 &`
    (J1 và J2a; chấm độ trung thực trên ba bộ, nhận dạng với ba mạng, độ trễ trên GPU). Xong thì commit `results/` và push.
    Lệnh đầy đủ: `docs/RUNBOOK.md` mục 6.
  - [x] 08/10 **Kết quả J1 và J2a, đã đưa vào bài** (mục 6.6 mới "Compression removal before upscaling and large models
    trained with compression", bảng 14 `tab_review.tex`, 52 macro `Rev*`; sửa câu ở mở bài, thảo luận, kết luận; gỡ câu giới
    hạn "We did not compare with a pipeline..." và thay bằng kết quả). Bài thành 37 trang.
    - FBCNN rồi SPAN công bố so với SPAN nhánh est, PSNR: AMI có nén −0,93 đến +0,16 dB; EarVN1.0 và AWEx chỉ nén JPEG
      +0,09 đến +0,48; có nhiễu và mờ (est) −0,24 đến −0,10; ảnh sạch ngoài thực tế +0,81 đến +0,93. Hạng 1 trên EarVN1.0:
      21,1% (ResNet-18) và 24,5% (ResNet-50), kém ta 3,9 [1,7; 6,9] và 2,6 [0,8; 4,5] điểm, hơn bicubic 7,9 và 6,9 điểm.
      Chi phí: 72,3 triệu tham số (170 lần SPAN), 3,87 so với 0,57 ms trên RTX 3080. Chưa đo trên iPhone.
    - BSRNet: PSNR trong khoảng −0,72 đến +0,76 dB so với bicubic khi có nén, kém ta 0,90 đến 2,32 dB. Hạng 1: 27,1% và
      32,9%; so với ta +2,1 [−0,5; 5,2] (không có ý nghĩa) và +5,9 [2,6; 9,5] (có ý nghĩa). Real-ESRNet: PSNR kém bicubic ở mọi
      ô có nén; hạng 1 25,0% và 30,5% (+0,0 và +3,4 [0,2; 7,0]). Cả hai: 39 lần tham số, 24 lần độ trễ.
    - AWEx: không mốc mới nào đổi hạng 1 có ý nghĩa so với bicubic.
    - Hệ quả cho câu chữ (đã viết): bài KHÔNG nói mô hình của ta cho nhận dạng cao nhất; nói nó có chi phí thấp nhất trong các
      phương pháp cải thiện cả PSNR lẫn nhận dạng. PSNR và nhận dạng xếp hạng các phương pháp khác nhau (BSRNet).
    - Chưa làm: đo FBCNN rồi SPAN trên iPhone (cần xuất Core ML và đo bằng Xcode; chỉ làm nếu người dùng muốn).
  - [x] 04/11 **Kết quả J2b (RRDB tinh chỉnh, 5 fold, commit `e547752`), đã đưa vào bài:** cả 5 lần qua phép thử ảnh sáng
    (khoảng cách tới bicubic tụt nhiều nhất 0,04 dB, 0 ảnh hỏng). JPEG 75, ba ô như DegEff: tinh chỉnh tăng PSNR của RRDB
    1,79 đến 2,15 dB so với trọng số công bố (SPAN: 1,62 đến 1,90); RRDB tinh chỉnh hơn SPAN tinh chỉnh 0,14 đến 0,22 dB
    (mọi ô có nén: 0,08 đến 0,34; đều có ý nghĩa). Hạng 1 trên EarVN1.0: 25,4% và 29,5%; so với SPAN của ta +0,3 [−0,9; 1,7]
    và +2,4 [1,1; 3,9]; vẫn thấp hơn BSRNet với ResNet-50 (32,9%). Câu "suy giảm quan trọng hơn kiến trúc" nay có đối chứng
    cùng điều kiện. Đã viết: đoạn mới ở 6.6, hàng mới ở bảng 14, câu ở mở bài, thảo luận, kết luận; gỡ câu "We did not train
    a larger network with compression". Bài 38 trang. Độ trễ trên GPU đo lại ở lượt chấm này (SPAN 0,50 ms), nên các tỉ số
    độ trễ trong 6.6 đổi nhẹ (FBCNN rồi SPAN 7,6 lần, RRDB 27 lần).
  - [x] (xong) **[Bạn, labai217] Bước 2:** `bash scripts/make_rrdb_jobs.sh` rồi chạy hàng đợi (fold 2, 3, 4). Xem dòng
    "phép thử ảnh sáng" của ba lần chạy. Cả ba ỔN ĐỊNH thì `FINAL_RUN=1 FOLDS="1 5" bash scripts/make_rrdb_jobs.sh jobs/rrdb_final.txt`
    và chạy tiếp. Sau đó gọi lại `bash scripts/run_review.sh` (file đã có được bỏ qua), commit và push.
  - [x] 04/11 **[Claude, máy Mac] Bước 3 (J1, J2a ngày 08/10; J2b ngày 04/11), sau khi pull kết quả:** đọc `results/rev_summary/summary.md`; thêm vào `make_paper.py`
    một bảng (các mốc mới cạnh bicubic, SPAN công bố, SPAN nhánh est: PSNR ở ảnh vào sạch, JPEG 93, 75, suy giảm ước lượng;
    hạng 1 trên EarVN1.0; tham số; độ trễ) và các macro; viết một mục con ở mục 6 và sửa các câu liên quan. Quy tắc câu chữ:
    - FBCNN rồi SPAN ngang hoặc hơn SPAN nhánh est: phần huấn luyện viết thành "lựa chọn rẻ nhất lúc chạy", bỏ ý "tốt nhất";
      sửa tóm tắt nếu cần. Kém hơn: ghi số, không bình luận thêm.
    - BSRNet hoặc Real-ESRNet hơn bicubic dưới nén: câu "mô hình công bố kém bicubic" giữ phạm vi "học bằng bicubic" (đã viết
      như vậy), thêm một câu về hai mô hình này.
    - RRDB nhánh est hơn SPAN nhánh est: ghi mức chênh cạnh chi phí (CPU điện thoại 108 so với 4,7 ms); câu "suy giảm có tác
      dụng lớn hơn kiến trúc" chỉ giữ nếu mức chênh nhỏ hơn tác dụng của suy giảm trên cùng thân. Gỡ câu "We did not train a
      larger network with compression" ở phần thảo luận và câu giới hạn về khử nén (thêm ở G4), thay bằng kết quả.
    - Sau đó đóng lại danh sách trong `docs/story-imavis.md`.
- [x] 08/10 **G7. Tóm tắt có hai mốc mới** (góp ý phản biện thử, người dùng dán): thêm câu cuối "Compression removal before
  upscaling gives a comparable PSNR with 170 times the parameters, and a large published model trained with synthetic
  degradations gives a higher rank-1 rate with one of two recognisers." Để giữ dưới 250 từ, rút gọn câu về AWEx thành một
  mệnh đề và sửa bốn cụm khác, không bỏ ý nào. Tóm tắt nay 248 từ (đếm từ `abstract.tex` sau khi thay macro).
- [x] 05/11 **G8. Rà văn phong (người dùng hỏi có dấu hiệu máy viết không):** bỏ hai trong ba câu mở "It is known that"
  (giữ ở mở bài); đổi khuôn "For the first/second/third question" và "The first/second is"; bớt "therefore" (12 còn 5) và
  "in our experiments"; xóa chú thích tiếng Việt và lệnh `\draftnote` trong `main.tex` (giữ `\todo` vì `make_paper.py` dùng
  khi thiếu số). Sửa nội dung một câu ở 6.7 đã sai sau khi có RRDB tinh chỉnh ("advantage of the small network ... lies
  in fidelity"). Gỡ một chỗ ghi tên trợ lý trong `paper/SUBMISSION_CHECKLIST.md`.
- [x] 05/11 **G9. Đọc soát câu chữ toàn bài** (người dùng yêu cầu: tránh câu cụt, lủng củng): sửa khoảng 25 câu ở tóm tắt,
  mở bài, tài liệu liên quan, benchmark, mục 4, 5, 6, thảo luận. Sinh chữ thay số khi cần ("all 16", "six", "both reach
  27.7%", "75 and 85"); bỏ dấu + ở các khoảng luôn dương. Thêm tài liệu Luo, Mu, Zhang, CCBR 2014 (đã kiểm trên Crossref),
  vì câu "the one study of SR for ear images that we found" không còn đúng. Tóm tắt 248 từ.
- [x] 05/11 **G10. Phụ lục** (người dùng đồng ý): file mới `paper/sections/appendix.tex`, sáu mục A đến F (bảng DIV2K đầy đủ;
  bộ phân loại độ thật; lỗi tinh chỉnh chỉ trên AMI và phép thử ảnh sáng; bảng mức JPEG cố định; khoảng trống theo người;
  độ chính xác khi chuyển FBCNN). Phần chính trỏ xuống bằng \ref, giữ các số chính. Phần chính nay kết thúc ở trang 31
  (trước: 33), phụ lục trang 33 đến 35, tài liệu từ trang 36; tổng 41 trang; phần chính còn 11 bảng. Rà lại cùng ngày:
  bảng phụ lục đánh số A.1 đến D.1 và giữ đúng chỗ (gói `float`); thêm câu dẫn cho mục C, D, E vốn mở đầu đột ngột.
- [ ] **L. Chưa đo trên điện thoại** (ghi ngày 08/10; iPhone 12 Pro Max, Xcode 26.3, Core ML FP16, ảnh vào 48×68, đo CPU và
- [x] 06/11 **G11. Đọc hai bài SR cho tai** (người dùng tải PDF): Markićević 2023 đọc đủ, không nhắc nén, EDSR ×2 +1 điểm hạng 1;
  Luo 2014 đọc 2 trang đầu (từ điển LC-KSVD, nhắm vào nhận dạng). Thêm Zarachoff, Sheikh-Akbari, Monekosso, IST 2018
  (doi 10.1109/IST.2018.8577134; đã đọc đủ 06/11: lọc thông thấp rồi thu nhỏ, không có nén, nhận dạng bằng trị riêng PCA). Sửa câu ở mở bài và tài liệu liên quan, có rào
  "as far as we could determine" vì hai bài chưa đọc đủ. Danh mục 46 mục.
- [x] 06/11 **G12. Viết gọn toàn bài** (người dùng: thầy chê dài dòng; Q1 cần súc tích, có câu chuyện, điểm mạnh lên trước):
  phần chính từ khoảng 10.100 xuống 5.800 từ (mở bài 1.380 xuống 710; thực nghiệm 3.640 xuống 1.840; thảo luận và kết luận
  1.240 xuống 610). Mở bài nay đi theo mạch: vấn đề, khoảng trống, phát hiện chính (phân rã sai số), hai hệ quả, bốn đóng
  góp. Không bỏ số liệu hay bảng nào; giữ mọi câu rào đón đã thống nhất với phản biện thử (AWEx, hai điều kiện, BSRNet).
  Phần chính kết thúc trang 22 (trước: 31); tổng 32 trang. Bản trước khi cắt lưu ở scratchpad `sections_before_trim/`.
- [x] 06/11 **G13. Sửa theo phản biện IMAVIS (người dùng dán, đồng ý làm bước 1 đến 5):**
  (1) phân rã sai số: thêm phương trình; mô hình kém bicubic khi phần dư (b−1)E_c vượt phần tiết kiệm (1−a)E_i; phần dư tăng
  theo mức nén ở 10/10 nhóm, nên câu "khuếch đại mạnh nhất khi nén nhẹ" chỉ là hiệu ứng mẫu số (đã bỏ); hình mới
  `fig_excess.png`. (2) tóm tắt viết lại theo một câu dẫn (243 từ); đóng góp 1 là quy tắc tỉ số; đóng góp 4 là hệ quả kèm
  quy tắc chọn theo ngân sách. (3) câu dùng chung: dưới nén kiến trúc gần như không đổi PSNR; nhận dạng xếp hạng khác và
  PSNR không dự đoán được. (4) từ khóa đổi thứ tự; AWEx "không thấy mức tăng cỡ EarVN1.0"; giới hạn mới: nén rồi cắt,
  4:2:0, đáp án chỉ sạch tương đối (EarVN1.0 nguồn JPEG 71 đến 93, trung vị 75; AMI 90), bộ nhận dạng chưa học trên ảnh
  đã phóng; bỏ câu "tìm không hệ thống, chưa thấy"; cách đọc mức JPEG; một seed mỗi fold; GPU chậm hơn CPU ở FP32; bảng
  theo bộ ảnh xuống phụ lục B. Bài 34 trang.
  (5) KẾT QUẢ, chưa đưa vào bài (chờ người dùng): trên 23 phương pháp, LPIPS trên ảnh EarVN1.0 mô phỏng đi cùng hạng 1 trên
  ảnh dò thật (Spearman −0,81 đến −0,87, p < 0,001, cả hai bộ nhận dạng), PSNR thì không (+0,15 đến +0,18, không ý nghĩa);
  DISTS ở giữa (−0,54 đến −0,60). Hình đánh đổi nháp ở scratchpad `tradeoff.png`: dưới ngưỡng 33 ms trên CPU điện thoại chỉ
  có SPAN và DISP học có nén là tăng nhận dạng; mọi phương pháp cao hơn đều quanh 100 ms.
  Để dành nếu phản biện đòi: học lại bộ nhận dạng trên ảnh bicubic-phóng (một ResNet-18, chấm lại mọi cách phóng).
  - [x] 06/11 Người dùng đồng ý đưa (5) vào bài: đoạn "Fidelity and identification" và bảng `tab_fidrecog` ở 6.6 (thêm kiểm theo
    họ: chỉ mô hình L1 do bài tinh chỉnh thì PSNR cũng đi cùng hạng 1, +0,82 và +0,76; nên câu viết hẹp: PSNR xếp đúng trong một
    họ mục tiêu, LPIPS xếp đúng qua các họ); hình `fig_tradeoff.png` ở 6.7; sửa câu PSNR ở tóm tắt (249 từ), mở bài, thảo luận,
    kết luận. Gọi là mô tả, làm sau khi có kết quả. Bài 35 trang, phần chính hết trang 26.
  mọi bộ xử lý như `results/latency_ios.csv`). Đã đo 6 mô hình: SPAN, DISP, EDSR-baseline, SwinIR-light, RRDB, BSRGAN.
  - [x] 04/11 **L1 xong, đã vào bài (mục 6.6).** FP32 trên iPhone 12 Pro Max: FBCNN rồi SPAN 93,59 ms CPU, 57,81 ms mọi bộ
    xử lý (280 trên 284 phép tính chạy trên GPU, không phép nào trên Neural Engine): vượt 33 ms ở cả hai. SPAN FP32: 10,33 ms
    CPU, 19,43 ms mọi bộ xử lý (GPU chậm hơn CPU ở mô hình nhỏ này). Số ở `results/latency_ios.csv`, ghi chú FP32.
    Lập luận real-time của bài nay có số: chuỗi khử nén rồi phóng không đạt ngưỡng trên điện thoại này.
  - [x] **L1. FBCNN rồi SPAN công bố (`fbcnn_span_ch48`)**: người dùng duyệt 04/11. Mã xong 04/11: `export_coreml.py` xuất
    được tên trong `CHAINS`. FBCNN KHÔNG chuyển đúng sang FP16 (đầu ra lệch, PSNR giữa Core ML và PyTorch 30,0 dB; bản FP32
    khớp, 120 dB), nên đo bằng FP32: `deploy_ios/models/fbcnn_span_ch48_fp32.mlpackage` (276 MB) và, để so cùng điều kiện,
    `span_ch48_fp32.mlpackage`. Bốn dòng trống đã thêm vào `results/latency_ios.csv`. Còn: bạn đo bằng Xcode.
    Ghi chú cũ: CHƯA ĐO. Đây là số còn thiếu quan trọng nhất, vì bài đang nói
    chuỗi này "đắt hơn" chỉ dựa trên GPU máy bàn (3,87 ms, vẫn dưới 33 ms). Việc phải làm trước: `scripts/export_coreml.py`
    hiện chỉ xuất mô hình trong `SPECS`; cần thêm nhánh cho tên trong `CHAINS` (dùng `build_chain`) [Claude]. Sau đó bạn xuất
    bằng `.venv-coreml/bin/python`, đo bằng Xcode, chụp hai màn hình (CPU Only và All) như lần trước.
  - [ ] **L2. FBCNN rồi bicubic (`fbcnn_bicubic`)**: không xuất được (coremltools chưa có phép `upsample_bicubic2d`); bỏ, không cần cho lập luận.
  - **BSRNet và Real-ESRNet**: không cần đo riêng. Cùng kiến trúc RRDB đã đo (107,91 ms CPU; 15,53 ms mọi bộ xử lý); bài
    phải ghi rõ là số của kiến trúc, không phải của từng bộ trọng số.
  - **RRDB tinh chỉnh (việc J2b)**: không cần đo riêng, cùng kiến trúc RRDB.
  - 10 mô hình hiệu quả còn lại (MSRResNet, RLFN, EFDN, SAFMN++, SMFANet, SPAN 28 và 26 kênh, PDS, PKDSR, DSCF, ERRN): đã
    xuất Core ML nhưng chưa đo; không cần cho lập luận của bài, chỉ đo nếu phản biện yêu cầu.
- [x] 06/11 **G14. Sửa theo phản biện vòng 2 (minor revision; người dùng đồng ý):** (1) tóm tắt, mở bài, đóng góp 1 dẫn bằng
  phần đo được (phần dư luôn dương và tăng theo nén; phần tiết kiệm tính trên ảnh sạch nên chỉ phụ thuộc ảnh và cỡ ảnh), bỏ
  chữ "governed" và dải 0,13 đến 0,36 khỏi câu dẫn; mục 4.2 nói rõ bất đẳng thức chỉ là cách viết lại phép so MSE (PSNR).
  (2) câu LPIPS: "PSNR mất liên hệ khi có mô hình GAN" (không còn "chỉ trong một mục tiêu"); bảng thêm cột DISTS; ghi tên
  LPIPS thay "a perceptual metric". (3) benchmark: AMI "near-clean" (gốc JPEG 90, thu 2,6 đến 5 lần), EarVN1.0 đáp án còn
  nén (71 đến 93, trung vị 75), AWEx PNG không rõ lịch sử nén; AMI là bộ gánh các câu độ trung thực. (4) mở bài khớp phạm vi
  phần tài liệu liên quan; câu Markićević trỏ sang nhóm ảnh dò JPEG ≥ 90 (+3,0 điểm). Highlights viết lại. Tóm tắt 247 từ.
- [x] 06/11 **G15. Sửa theo phản biện vòng 3 (minor):** thảo luận và kết luận không còn dẫn bằng bất đẳng thức (chỉ một mệnh đề);
  dẫn bằng phần đo được và tỉ số 0,69 so với 0,10 ở JPEG 75. Bỏ chữ "only" (phần tiết kiệm phụ thuộc mô hình qua a; chỉ không
  đổi theo mức nén). Kiểm thêm: phần dư tăng theo nén ở 160/160 cặp mô hình × nhóm, nên "every model" đúng (macro
  ExcessMonoModels). "Near-clean" chỉ gắn với AMI; câu AMI hạ thành "we treat ... as near-clean". Bỏ ước lượng tỉ số khỏi
  đóng góp 1. Câu LPIPS gắn "on EarVN1.0" và mệnh đề đáp án còn nén. Sửa câu đứt "; Across". Sửa "BSRNet ... the highest
  rank-1 rate" (chỉ đúng với ResNet-50; với ResNet-18 BSRGAN và Real-ESRGAN cao hơn). Highlights dòng 5 có điều kiện.
  Tóm tắt 248 từ; bài 36 trang.
- [x] 06/11 **G16. Hai câu cuối theo phản biện (trước khi lên khuôn):** mục 4.2 bỏ mệnh đề "loses its advantage when this excess
  exceeds..." (phần đo được dừng ở b > 1 và phần dư tăng); thảo luận bỏ "LPIPS is the better proxy...", dừng ở phạm vi EarVN1.0.
- [x] 04/11 **O. Phản biện thứ tư (04/11; người dùng: "làm hết đi").** Xong cả O1 đến O5. Thứ tự B (tính ngắn trên Mac), C (labai217), rồi A (chữ),
  vì câu chữ phụ thuộc kết quả. Phát hiện khi đối chiếu: mã nén (`cv2.imencode`, mặc định) đã lấy mẫu màu 4:2:0 (Y 2×2,
  màu 1×1), nên câu "chưa kiểm 4:2:0" ở phần giới hạn là SAI, phải sửa.
  - [x] 04/11 O1 Hình cơ chế (`scripts/analyze_blocks.py`, chạy trên Mac, `results/mechanism/`; hình `fig_mechanism.png`,
    macro `Mech*`, đoạn "Where the excess lies" ở mục 4.2). Kết quả: ở JPEG 75 trên AMI phần dư mỗi điểm ảnh sát biên khối
    gấp 1,45 lần chỗ khác (52% phần dư trên 42% điểm ảnh); ở JPEG 93 và trên DIV2K gần đều (1,07; 1,08). Theo tần số: 84%
    phần dư nằm trong 0,6 đến 1,4 lần Nyquist của ảnh vào, 57% ở trên Nyquist (sai số nén của bicubic: 15%). Tức phần dư
    chủ yếu là chi tiết mạng dựng ở dải quanh Nyquist, biên khối chỉ góp một phần. Mô tả, không phải phép thử nhân quả.
  - [x] 04/11 O2 Thứ tự suy giảm (`scripts/run_order_control.sh`, `results/order/`, 189 file; bảng B.1, macro `Ord*`).
    Lưới lệch: 16/16 vẫn dưới bicubic ở JPEG 75 (0,43 đến 0,56 dB; khác bản thẳng lưới tối đa 0,03 dB); nén hai lần 93 rồi
    75: 16/16 dưới (0,48 đến 0,61); 4:4:4: đổi tối đa 0,18 dB; nén trước thu nhỏ sau: ảnh gần sạch, mọi mô hình trên
    bicubic. SPAN tinh chỉnh hơn bicubic 1,19 đến 1,33 dB ở mọi biến thể nén. Câu ở 4.1, 6.2, giới hạn; câu fidelity giữ phạm vi.
  - [x] 04/11 O3 Độ khối của đáp án AMI: ảnh gốc 1,34; đáp án 144 px 1,00, 96 px 1,01 (`results/gt_quality.json`, macro
    `BlockAmi*`; câu ở mục 3.1 của benchmark). Thu nhỏ hệ số không nguyên đã xóa lưới.
  - [x] 04/11 O4 Bộ nhận dạng học thêm ảnh đã phóng (`results/recog_up/`; bảng H.1, macro `RecUp*`, đoạn ở 6.4). EarVN1.0:
    bicubic 13,2% lên 31,3%; mức tăng của SPAN tinh chỉnh so với bicubic co từ 11,8 xuống 5,5 điểm [1,7; 9,3], vẫn dương;
    SPAN công bố ngang bicubic (32,3%). AWEx: ba phương pháp ngang nhau (36,7 / 36,7 / 36,6%). Đúng mẫu của SR mặt.
  - [x] 04/11 O5 Chữ (xong; tóm tắt 250 từ, bài 38 trang, 0 macro thiếu, thêm câu UERC ở giới hạn): câu 4:2:0; viết lại mục 3 (lõi là mức JPEG đọc từ file; noise, blur chuyển phụ lục); còn 3 đóng
    góp, nhận dạng và LPIPS thành kết quả có điều kiện, rời khỏi tóm tắt và kết luận; cắt lặp số ở thảo luận và kết luận;
    câu nội suy và bootstrap của điểm đổi dấu (in một số lẻ); câu về một seed; hình đánh đổi tách FP32; tiêu đề.
- [x] 05/11 **P. Kiểm toán bảng biểu (phản biện thứ năm, 04/11; người dùng: "làm").** Đã đối chiếu từng điểm với mã: đúng hết.
  - [x] 04/11 P1 Độ trễ RRDB lệch giữa Bảng 1/11 (13,39, `latency_local.csv`) và Bảng 9 (13,21, `latency_review.csv`, lượt đo
    sau). Sửa: mọi mô hình kiến trúc RRDB ở Bảng 9 dùng số của Bảng 1 (như Hình 7); FBCNN ghi rõ lượt đo riêng; điền ô Real-ESRNet. Xong: Bảng 9 in 13,39 cho cả bốn hàng RRDB; chú thích ghi lượt đo riêng của FBCNN.
  - [x] 04/11 P2 Khoảng tin cậy Bảng 8 và H.1 lệch 0,1: seed vẫn cố định, nhưng `summarize_recog.py` lấy 1000 mẫu còn
    `recog_upscaled` lấy 2000. Sửa: hàng "large images only" đọc từ `paired.csv`; hàng còn lại cũng 1000 mẫu (khoảng thành [1,6; 9,3]).
  - [x] 04/11 P3 Trùng lắp: hàng Measured của G.1 = hàng JPEG 75 của Bảng 4 (giữ Bảng 4, chú thích G.1); chú thích Bảng 9, 11 về
    hàng/cột lặp; gộp A.1 với C.1 (C.1 đã có sẵn hàng Ear (AMI)). Xong: phụ lục C cũ bỏ, A.1 có 6 cột, DIV2K ở nửa dưới.
  - [x] 04/11 P4 EER: một câu ở 6.4 (bicubic-trained tăng EER 0,7 đến 1,2 điểm, khoảng tin cậy chứa 0); hàng bicubic ở F.1;
    chú thích Bảng 7 về 10 ảnh dò ngoài hai nhóm (kiểm lại: 3 ảnh mức 65, 7 ảnh mức 81 đến 87, không phải toàn 81 đến 89).
  - [x] 04/11 P5 Đặt tên: Ear-R18, Ear-R50 (bộ nhận dạng tai), ImageNet-R50 (đối chứng đông lạnh), đồng loạt bảng, thân bài và Hình 7.
  - [x] 05/11 P6 [labai217] Đã chạy (156 file: thêm BSRNet, Real-ESRNet, không vào bảng). KẾT QUẢ ĐỔI MỘT KẾT LUẬN:
    điểm đổi dấu trên AMI dời theo cỡ ảnh, 86,8 / 89,0 / 91,3 ở 24 / 36 / 48 px, khoảng không chồng; tỉ số E_c/E_i lúc đổi
    0,28 / 0,26 / 0,24. Đã sửa câu "cỡ ảnh ít ảnh hưởng" ở 4.2, phần mở đầu, đóng góp 1 (nội dung là chính, cỡ ảnh ít hơn;
    DIV2K vẫn phẳng). JPEG 93 ở 48 px: ERRN âm 0,01 dB, không ý nghĩa, câu 4.1 sửa thành "không mô hình nào kém có ý nghĩa".
    Measured trên AMI: 16/16 kém bicubic có ý nghĩa ở cả ba cỡ (0,09 đến 0,59 dB), thêm một câu ở 4.1. Ghi chú cũ: AMI 48 px chỉ có sạch và JPEG 75; cột Measured của AMI chưa đo cho 16 mô hình công bố. Chạy suy luận
    rồi thêm hàng AMI 48 vào Bảng 2; trong lúc chờ, câu ở 4.2 nói rõ bằng chứng về cỡ ảnh đến từ DIV2K (đã sửa 04/11).
    Lệnh: `bash scripts/run_ami48.sh` (đã thử trên Mac với 2 ảnh). Bảng 2 và A.1 tự thêm số; `WildEstGain*` đã giới hạn
    cho EarVN1.0 và AWEx. Có số rồi: đọc lại câu 4.2 (cỡ 48 có cùng điểm đổi dấu không) và tóm tắt (\CrossQAmi*).
- [ ] **Q. Làm mạnh bài sau kiểm toán (05/11; người dùng: "làm 3 cái này trước").** Ba việc, cố định trước khi chạy:
  - [ ] Q1 Chuẩn nén khác JPEG: 16 mô hình công bố trên AMI 36 px, WebP (và AVIF nếu OpenCV có), vài mức chất lượng.
    Hỏi: phần dư (b > 1) và điểm đổi dấu có xuất hiện không. Chỉ suy luận.
  - [ ] Q2 Tách điều kiện nén của phần nhận dạng: EarVN1.0, ảnh lớn của người test thu nhỏ thành ảnh dò, không nén / JPEG 93 /
    JPEG 75; cùng người, cùng Ear-R18, Ear-R50. Hỏi: SPAN tinh chỉnh chỉ hơn bicubic khi ảnh dò bị nén? Chỉ suy luận.
  - [ ] Q3 Thêm seed: fold 1, nhánh est, bicjpeg75, bic, seed 1 và 2 (6 lần huấn luyện). Hỏi: dao động do seed so với các
    chênh lệch ở ablation.
  - 05/11 Mã xong, đã thử trên Mac (165 kiểm thử qua): `ops.webp`, `ops.avif`, kiểu `bicwebpQ`, `bicavifQ`
    (`scripts/run_codec.sh`, mức chọn theo E_c/E_i trên 100 ảnh AMI); `recog_eval.py --synth` (`scripts/run_recog_synth.sh`,
    ảnh dò 36 px từ nửa ảnh lớn); `train.py --seed`, nhãn `+sN`, nhánh `[seed N]` (`scripts/run_seeds.sh`). Chờ chạy trên
    labai217; phần phân tích trong `make_paper.py` viết sau khi có số.
- [ ] **M. Việc cần làm cho bốn điểm yếu sau khi có kết quả J1, J2a** (ghi ngày 08/10 theo yêu cầu người dùng; mục nào ghi
  "đề xuất" là chưa duyệt).
  - **M1. Mô hình của ta không thắng rõ ở đâu** (BSRNet hơn ta 5,9 điểm hạng 1 với ResNet-50; FBCNN rồi SPAN ngang ta về PSNR
    trên ảnh ngoài thực tế và giữ được chất lượng trên ảnh sạch).
    - [x] 04/11 Chạy xong J2b (RRDB tinh chỉnh với công thức của ta); kết quả ở việc J. Nó tách "do mạng lớn" khỏi "do kiểu suy giảm" trong kết quả
      của BSRNet: RRDB nhánh est hơn BSRNet thì công thức của ta có giá trị cả trên mạng lớn; ngang hoặc kém thì ghi thẳng.
    - [ ] Sau J2b: viết lại đoạn cuối mục 6.6 và câu khuyến nghị ở phần thảo luận thành một bảng đánh đổi rõ ràng (ngân sách
      CPU thì SPAN tinh chỉnh; có bộ xử lý thần kinh và cần nhận dạng cao nhất thì mạng lớn học có nén).
    - Câu chữ hiện tại đã đúng mức ("lowest cost among the methods that improved both PSNR and identification"); không hạ thêm.
  - **M2. Phản biện có thể nói "dùng đồ có sẵn là đủ"** (BSRNet, FBCNN là trọng số công bố từ 2021).
    - [x] 04/11 (số trên điện thoại đã vào văn bản 6.6, không thêm cột vì bảng 14 trộn FP16 và FP32) Làm L1 để lập luận chi phí có số trên điện thoại, rồi thêm cột "CPU điện thoại (ms)" vào bảng 14.
    - [ ] Đề xuất, chưa duyệt: một hình đánh đổi (trục ngang độ trễ CPU điện thoại, trục dọc hạng 1 trên EarVN1.0, mỗi điểm
      một phương pháp), sinh PNG bằng `make_paper.py`. Không cần chạy gì mới ngoài L1.
    - [x] 04/11 Viết một câu ở 6.6 nói rõ điều bài thêm so với đồ có sẵn: với cùng thân SPAN, tinh chỉnh có nén cho hạng 1 cao hơn
      khử nén rồi phóng (25,0% so với 21,1%) mà không thêm mạng thứ hai. Số đã có trong bảng 14.
  - **M3. Lập luận real-time có chỗ hở** (RRDB chạy 15,53 ms trên iPhone khi dùng mọi bộ xử lý, dưới ngưỡng 33 ms của bài;
    chỉ tách được ở CPU: 4,72 so với 107,91 ms).
    - [x] 04/11 (không dưới: 93,6 ms) Làm L1. Nếu chuỗi FBCNN rồi SPAN cũng dưới 33 ms trên CPU thì bài phải nói thẳng, và lợi thế còn lại là số tham
      số và bộ nhớ.
    - [x] 04/11 Mục 6.7 (Efficiency): thêm một câu nói rõ kiến trúc RRDB (BSRNet, Real-ESRNet) đạt ngưỡng khi có bộ xử lý thần kinh
      và không đạt trên CPU, và rằng định nghĩa real-time của bài tính trên CPU. Hiện mục này chỉ nói vậy cho BSRGAN.
    - [ ] Sau L1, quyết định có giữ chữ "Real-Time" trong tiêu đề hay đổi thành cách nói hẹp hơn (việc này cũng nằm trong
      `paper/SUBMISSION_CHECKLIST.md`, mục quyết định về tiêu đề).
  - **M4. PSNR bị đặt dấu hỏi** (BSRNet: PSNR ngang bicubic nhưng hạng 1 cao nhất).
    - [x] 04/11 Thêm một câu giới hạn ở phần thảo luận: phần phân tích tỉ số sai số nói về độ trung thực; PSNR không dự đoán được
      thứ hạng nhận dạng giữa các phương pháp học bằng suy giảm khác nhau.
    - [ ] Đề xuất, chưa duyệt, chỉ tính từ số đã có: tương quan hạng giữa PSNR (ảnh vào `est` của EarVN1.0) và hạng 1 trên
      mọi phương pháp của bảng 10 và bảng 14; và xem LPIPS, DISTS (đã có trong `results/rev*`) có đi cùng hạng 1 hơn PSNR
      không. Nếu có thì thêm một câu và một cột; nếu không thì chỉ ghi câu giới hạn ở trên.
    - Không làm: đổi số đo chính của bài sang nhận dạng. Phần lõi (16 mô hình, tỉ số sai số, DIV2K) là kết quả về độ trung
      thực và vẫn đứng; với mô hình học bằng bicubic, PSNR và nhận dạng cho cùng kết luận.
- [ ] **N. Luật chọn lúc chạy (góp ý ngày 04/11; người dùng bảo chạy phép thử khả thi).** Chọn giữa bicubic và mô hình
  công bố cho từng ảnh từ mức JPEG và sai số tự nội suy của ảnh nhỏ (không dùng ảnh gốc). Mã: `scripts/analyze_selector.py`,
  chỉ phân tích số đã có (`results/t2*`), ảnh nhỏ dựng lại ở `data/bench` trên Mac (khớp 0,0 dB với labai217). Kết quả:
  `results/selector/summary_{span_ch48,rrdb_psnr}.md`.
  - [x] 04/11 Phép thử xong. Sai số tự nội suy tương quan với E_i thật (Spearman 0,70 đến 0,81 trong từng bộ); tỉ số ước
    lượng tương quan với tỉ số thật 0,81 đến 0,91.
    - Hiệu chỉnh và thử trên cùng loại ảnh (AMI với AMI): luật tỉ số KHÔNG hơn luật chỉ dùng mức JPEG (−0,002 dB). Đúng như
      dự đoán: trong một bộ, mức JPEG đã đủ.
    - Hiệu chỉnh trên ảnh tự nhiên (DIV2K), thử trên ảnh tai: luật chỉ dùng JPEG chọn sai (dùng mô hình ở JPEG 75 và 85),
      luật tỉ số hơn nó +0,099 [0,091; 0,105] dB trên AMI, +0,065 trên AWEx, +0,034 trên EarVN1.0 (đều có ý nghĩa; trên AMI
      lấy lại khoảng 83% phần mà chọn đúng tuyệt đối lấy được). Chiều ngược lại (AMI sang DIV2K): +0,110. Lặp lại với RRDB
      công bố: cùng kết quả.
    - Hiệu chỉnh trên cả hai loại: lẫn lộn (hơn trên DIV2K và EarVN1.0, kém 0,017 dB trên AWEx).
    - Mức lợi tuyệt đối nhỏ (0,03 đến 0,2 dB tính trên mọi mức nén), và không có ý nghĩa nếu đã có mô hình học dải JPEG.
  - [x] 04/11 Người dùng đồng ý: đã thêm đoạn "Estimating the ratio without ground truth" ở mục 4.3 (8 macro `Sel*`
    sinh từ `results/selector/comparisons_span_ch48.json`; có kiểm tự động). Không thành đóng góp riêng. Ghi chú cũ: Quyết định (người dùng): đưa vào bài thành một đoạn ngắn ở mục 4.3 (bằng chứng tỉ số có sức dự đoán ngoài miền
    hiệu chỉnh) hay chỉ một câu ở phần thảo luận, hay bỏ. Đề xuất của Claude: một đoạn ngắn ở 4.3, không thành đóng góp riêng.
- [ ] **K. Ý để dành, CHƯA duyệt, không làm trước khi phản biện thật yêu cầu** (ghi ngày 08/10 để nhớ):
  - **Tách hai điều kiện của phần nhận dạng.** Không huấn luyện lại bộ nhận dạng trên AWEx (gallery trung vị 2 ảnh mỗi người
    nên khoảng trống khó xuất hiện; chỉ thử được một chiều). Cách rẻ hơn, chỉ chấm: trên EarVN1.0, thu nhỏ ảnh lớn của người
    test thành ảnh dò nhỏ không nén, rồi nén ở JPEG 75 và 93; cùng người, cùng bộ nhận dạng, chỉ đổi mức nén. Điểm yếu: ảnh dò
    là ảnh tổng hợp. Ghi chú thêm: SPAN công bố không đổi hạng 1 trên AWEx dù ảnh ở đó không thấy khối JPEG, nên khoảng trống
    gần bằng không có thể là lý do chính; đây là phỏng đoán, phép thử trên sẽ kiểm được.
  - **Chuẩn nén khác JPEG** (WebP, HEIC, khung hình video): hiện chỉ trả lời bằng phần giới hạn.
  - **Ý tăng novelty từ khảo sát ngày 04/11 (chưa duyệt; khảo sát bằng tìm kiếm web, không có hệ thống):**
    (a) SPAN điều kiện theo mức JPEG đọc từ file (ý từ FBCNN, DASR); (b) loss giữ danh tính cho SR thời gian thực của tai
    (ý từ SR mặt: SICNN, MERL 2019); (c) gộp điểm nhận dạng của nhiều cách phóng theo mức JPEG (ý từ mống mắt). (a) và (b) cần
    huấn luyện mới; (c) chỉ phân tích. Bài SR cho tai đã kiểm trên Crossref: Markićević, Peer, Emeršič, IWSSIP 2023 (đã trích);
    Luo, Mu, Zhang, CCBR 2014, LNCS, doi 10.1007/978-3-319-12484-1_50 (SR phân biệt cho nhận dạng tai, chưa trích).
  - **So với phương pháp phóng ảnh dành riêng cho tai hoặc mặt:** hiện chỉ trả lời bằng phần tài liệu liên quan.
  - Các mục còn lại ở `docs/story-imavis.md` mục 5 (thêm seed, thứ tự suy giảm, hệ số ×2, bộ dữ liệu thứ ba).
- [ ] **Đang chờ: người dùng đọc soát `paper/main.pdf`**, ổn thì gửi thầy; sau khi thầy đồng ý mới dọn mã nguồn (việc I).
- [ ] **F2. [Bạn] Việc chỉ tác giả làm được:** theo `paper/SUBMISSION_CHECKLIST.md` (đơn vị công tác, tuyên bố xung đột lợi ích,
  CRediT, tài trợ, quyết định về tiêu đề, quyền dùng ảnh ở hình 4, đọc toàn văn bài IWSSIP 2023, tìm tài
  liệu có hệ thống).
- [ ] **H. Đưa thầy đọc** `paper/main.pdf`.
- [ ] **I. TRƯỚC KHI NỘP hoặc công khai mã (người dùng dặn ngày 08/10): không để lại dấu vết nào của AI trong bài và trong mã.**
  Trạng thái: thư mục `paper/` (bản thảo, PDF, checklist) và mọi file mã, README đã sạch (đã rà bằng grep ngày 08/10).
  Còn chứa: các ghi chú làm việc nội bộ (`TODO.md`, `CLAUDE.md`, `docs/HANDOFF.md`, `docs/STATUS.md`, `docs/story-imavis.md`,
  `docs/paper-plan-*.md`) và **lịch sử git** của kho làm việc. Cách xử lý: không công khai kho làm việc; xuất bản sạch bằng
  ```bash
  bash scripts/make_release.sh ~/earsr_rt_release      # bỏ ghi chú nội bộ, rà từ cấm, dừng nếu còn
  ```
  rồi tạo kho công khai mới từ thư mục đó (không kèm lịch sử). Chạy lại script này mỗi lần trước khi công khai.
  Bản thảo gửi tạp chí chỉ lấy từ `paper/`.

### Ba điểm dè dặt của quy tắc tỉ số sai số: ĐÃ LÀM ngày 08/10, kết quả làm bài phải viết hẹp lại

- [x] 08/10 **Dè dặt 2 (ranh giới sắc do may): ĐÚNG LÀ DO MAY. Không có ngưỡng chung 0,17.** Nội suy điểm đổi dấu cho từng bộ
  ảnh và cỡ ảnh, bootstrap theo người: ảnh tự nhiên (DIV2K) đổi dấu ở tỉ số 0,13 đến 0,17 (mức JPEG 64 đến 66); AMI ở 0,26
  đến 0,28 (mức 87 đến 89); ảnh tai ngoài thực tế ở 0,23 đến 0,36 (mức 76 đến 81; chỉ có hai mức nén nên ước lượng thô).
  Khoảng tin cậy của DIV2K và AMI không chồng nhau. **Bài đã sửa:** bỏ con số 0,17; viết "một giá trị phụ thuộc bộ ảnh, giữa
  0,13 và 0,36"; thêm bảng điểm đổi dấu (mục 4.3); hình đổi vạch đứng thành dải.
  Phần còn đứng vững, và nay được đo chứ không chỉ lập luận: trong cả 35 ô, **mọi** mô hình trong 16 mô hình đều giảm sai
  số nội suy (còn 0,38 đến 0,76 lần) và khuếch đại sai số nén (1,6 đến 15,8 lần). Tỉ số xếp thứ tự các ô tốt hơn mức JPEG
  (tương quan hạng −0,91 so với 0,79). Hệ số khuếch đại không phải hằng số (lớn nhất khi nén nhẹ), nên không có mô hình
  hai tham số gọn.
- [x] 08/10 **Dè dặt 1 (mới kiểm ×4):** kết quả cũ chỉ có **một** mô hình PSNR ở ×2 (SwinIR-light ×2 trên AMI) và một ô: tỉ số
  1,21, phần hơn −0,53 dB (ảnh sạch +2,41). Khớp chiều nhưng là một điểm nằm rất xa điểm đổi dấu, không kiểm được gì
  nhiều. Bài ghi đúng như vậy và giữ "×4, JPEG, mô hình học bằng bicubic" ở mục giới hạn. Không chạy thêm.
- [x] 08/10 **Dè dặt 3 (tính mới): lượt tìm kiếm web (4 truy vấn) không thấy bài nào nêu quan hệ này theo tỉ số sai số.** Thấy:
  các bài SR cho ảnh nén có lúc báo một phương pháp SR chung kém bicubic trên ảnh JPEG (ví dụ trong so sánh của CISRDCNN);
  một benchmark SR sau codec video (Bogatyrev và cộng sự, BMVC 2024; mới thấy tên bài, chưa mở). Bài viết: "không tìm thấy
  phát biểu nào về quan hệ này trong phần tài liệu đã tìm, nhưng lượt tìm chưa có hệ thống". **Vẫn là việc của bạn** ở việc
  F: tìm có hệ thống, và mở bài BMVC 2024 để kiểm trích dẫn (mục này trong `refs.bib` còn nhãn VERIFY).

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

- [x] 08/10 **[Claude]** Ba điểm dè dặt của quy tắc tỉ số: tính điểm đổi dấu theo từng bộ ảnh với khoảng tin cậy; đo hệ số
      giảm sai số nội suy và khuếch đại sai số nén của từng mô hình ở từng ô; kiểm ô ×2; tìm kiếm web. Kết quả: không có
      ngưỡng chung, bài sửa lại (abstract, mở bài, mục 4.3, thảo luận, related work). `make_paper.py`: hàm `error_ratio` viết lại.
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
      **08/10: đã duyệt và viết mã cho BSRNet, Real-ESRNet (mục 0, việc J2a); chờ chạy.**
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
- [ ] **4. Mốc "khử nén rồi mới SR"** (**08/10: đã duyệt và viết mã, mục 0 việc J1; chờ chạy**): một mạng khử vết JPEG có sẵn (ví dụ FBCNN) đặt trước SPAN công bố. Câu hỏi hiển nhiên
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
