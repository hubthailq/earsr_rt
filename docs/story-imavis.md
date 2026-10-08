# Phác thảo câu chuyện bài báo (bản nháp 07/10/2026)

Đích: Image and Vision Computing (IMAVIS). Loại bài: phát hiện, benchmark và một cách huấn luyện; không có kiến trúc mới
(kiến trúc để bài sau). Thầy chỉ đọc bài hoàn chỉnh, nên file này là căn cứ để làm tới bản thảo, không phải để xin ý kiến
giữa chừng. Số liệu lấy từ mục 2.0 của `paper-plan-ridgesr-2026-10-05.md`. Xếp hạng Q của IMAVIS chưa tra lại.

File này chốt **phạm vi**: bài cần đúng những bằng chứng liệt kê ở mục 4. Việc nào không nằm trong mục 4 thì không làm cho
bài này.

## 1. Một câu

Trên ảnh tai nhỏ đã nén, thứ quyết định chất lượng phóng ảnh là suy giảm dùng lúc huấn luyện, không phải kiến trúc: các mô
hình SR hiệu quả có sẵn kém cả nội suy bicubic, còn một mô hình real-time có sẵn, huấn luyện với suy giảm có nén khớp ảnh
tai thật, hơn tất cả.

Tiêu đề dự kiến: *Real-Time Super-Resolution for Low-Resolution Ear Images: Degradation Matters More Than Architecture*.

## 2. Vấn đề và động cơ

1. **Ảnh tai ngoài thực tế thì nhỏ và đã nén.** Trong EarVN1.0, trung vị cạnh ngắn là 77 px; ảnh tai nhỏ (cạnh ngắn 24 đến
   48 px) có 52% ở mức JPEG 75 và 44% ở mức 93. Muốn nhìn rõ thì phải phóng lên, và phải nhanh.
2. **SR hiệu quả được thiết kế và xếp hạng ở điều kiện khác hẳn:** ảnh tự nhiên lớn, thu nhỏ bằng bicubic, không nén (giao
   thức NTIRE). Bài SR cho ảnh tai đã có (IWSSIP 2023, EDSR và SwinIR trên UERC) chưa được đọc toàn văn; phải kiểm nó
   dùng suy giảm nào trước khi viết câu so sánh.
3. **Câu hỏi chưa ai trả lời:** người làm ảnh tai lấy một mô hình SR hiệu quả có sẵn về dùng thì có được gì không, và nếu
   không thì phải sửa ở đâu, kiến trúc hay dữ liệu huấn luyện.

## 3. Ba luận điểm và bằng chứng

| # | Luận điểm | Bằng chứng | Trạng thái |
|---|---|---|---|
| P1 | **Phát hiện.** Mô hình SR hiệu quả có sẵn (học bằng bicubic) mất hết lợi thế khi ảnh tai vào đã nén, và kém bicubic về độ trung thực từ khoảng mức 85 trở xuống | 16 mô hình trên ba bộ ảnh. Ở JPEG 75: cả 16 kém bicubic trên AMI (0,40 đến 0,53 dB ở ảnh vào 36 px) và trên AWEx; trên EarVN1.0 xấp xỉ hòa. Ở JPEG 93: cả 16 lại hơn. Trên ảnh sạch chúng hơn bicubic 2,0 đến 2,6 dB | Đã có |
| P2 | **Chẩn đoán.** Nút thắt là suy giảm, không phải kiến trúc | Dưới JPEG 75, 16 kiến trúc chỉ cách nhau 0,12 dB, và mạng lớn gấp 39 lần không hơn (−0,04 dB). Đổi suy giảm huấn luyện trên cùng một thân thì được 1,3 đến 1,9 dB (ba bộ ảnh). Đúng trên hai thân khác họ | Đã có (ba fold) |
| P3 | **Cách sửa.** Huấn luyện một thân real-time có sẵn với suy giảm có nén đo từ ảnh tai thật, kèm hai điều kiện để tinh chỉnh không hỏng (tăng cường độ sáng, thêm ảnh tai ngoài thực tế) | Dưới suy giảm ước lượng, trên AMI, EarVN1.0, AWEx: hơn bicubic 1,2 đến 2,2 dB, hơn trọng số công bố 1,3 đến 1,9 dB, hơn suy giảm tổng quát kiểu Real-ESRGAN 1,1 đến 2,1 dB; LPIPS cũng tốt hơn cả ba. Tinh chỉnh chỉ trên AMI làm mô hình hỏng trên ảnh sáng (22 so với 36 dB) | Đã có trên ảnh nén tổng hợp; **thiếu trên ảnh nhỏ thật** |
| P4 | **Real-time.** Cách sửa không thêm chi phí lúc chạy | SPAN 48 kênh: 0,5 ms trên RTX 3080 ở ảnh vào 48×68, nhanh hơn BSRGAN 27 lần | Có trên GPU; **thiếu trên thiết bị** |

**Đóng góp ghi trong bài (ba dòng):** (1) benchmark ảnh tai nhỏ có đáp án sạch ở ba cỡ, ba bộ dữ liệu, và phát hiện P1, P2;
(2) suy giảm đo từ ảnh tai thật và công thức tinh chỉnh ổn định; (3) một mô hình real-time hơn mọi mô hình có sẵn về độ
trung thực trên ảnh tai có nén.

**Bài không tuyên bố** (viết thẳng ở phần giới hạn): kiến trúc mới; hơn trên ảnh sạch (kém trọng số công bố 1,2 đến 1,8 dB);
LPIPS tốt hơn các mô hình GAN lớn (BSRGAN 0,173 so với 0,246, nhưng PSNR kém bicubic 2 dB và chậm hơn 27 lần); cải thiện
nhận dạng.

## 3b. Kết quả của việc 1 và việc 2a (07/10/2026) và câu chữ đã chốt cho P3

**Việc 1 (khối `n2c`, SPAN × ba fold):** tách "có nén" khỏi "đo từ dữ liệu".

- Trên ảnh vào theo suy giảm ước lượng, nhánh ước lượng hơn nhánh chỉ có mức nén đã đo 0,04 đến 0,17 dB. Trên ảnh vào cùng
  mức nén nhưng không mờ, không nhiễu thì ngược lại (kém 0,09 đến 0,22 dB); trên ảnh sạch kém 0,43 đến 0,61 dB. Nhánh nào
  cũng thắng trên đúng kiểu ảnh nó học, nên ảnh mô phỏng không phân xử được.
- Mức nén đã đo so với mức nén rút đều 60 đến 95: chênh 0,00 đến 0,06 dB. Không cần đo phân bố, chỉ cần phủ đúng dải.

**Việc 2a (nhận dạng trên ảnh nhỏ thật; 51 người đăng ký, 36 người có ảnh nhỏ, 2.015 ảnh dò):** rank-1.

| Nhóm | Rank-1 | So với bicubic |
|---|---|---|
| Ảnh dò lớn (mức tham chiếu của mạng nhận dạng) | 60,2% | |
| Bicubic ×4 | 13,2% | |
| Mô hình học bằng bicubic: SPAN công bố; SPAN tinh chỉnh trên ảnh tai | 14,0%; 13,2% | +0,8 [−0,1; 1,9]; +0,0 [−1,2; 1,3] |
| Mô hình học có nén: ước lượng; JPEG 75; mức nén rút đều; chỉ mức nén đã đo; tổng quát | 25,0%; 25,4%; 24,4%; 23,9%; 23,9% | ước lượng: +11,8 [7,3; 16,8] |
| Mô hình GAN lớn: BSRGAN, Real-ESRGAN | 27,7% | so với ước lượng: +2,7 [−1,7; 7,6], không có ý nghĩa |

Mạng đối chứng (ImageNet, chưa từng thấy tai) cho cùng thứ tự ở mức thấp hơn nhiều: 2,9% → 7,2% (+4,3 [1,9; 7,1]).

**Câu chữ đã chốt cho P3:** "phải huấn luyện với nén JPEG phủ đúng dải mức nén của ảnh thật". Việc đo từ ảnh tai thật dùng
để **xác định dải đó** (hai đỉnh ở mức 75 và 93); ước lượng nhiễu và độ mờ **không cần**. Mô hình tham chiếu của bài vẫn là
nhánh ước lượng, vì nó được chốt trước khi có các phép so này (không chọn mô hình theo kết quả test).

**Hai điều phải viết hẹp lại so với bản phác đầu:**

- Suy giảm tổng quát: kém 1,1 đến 2,1 dB về độ trung thực và kém bicubic trên ảnh sạch, nhưng **không kém về nhận dạng** trên
  ảnh thật (−1,0 [−2,4; 0,2]). Không viết "suy giảm tổng quát không giải quyết được".
- Mô hình GAN lớn có rank-1 cao hơn 2,7 điểm (không có ý nghĩa thống kê), chậm hơn 27 lần. Không viết "tốt nhất về nhận dạng".

**Điểm mạnh mới của bài:** phát hiện P1 nay có bằng chứng trên ảnh thật, không chỉ trên ảnh mô phỏng: mô hình học bằng
bicubic không giúp gì cho nhận dạng (kể cả sau khi tinh chỉnh trên ảnh tai), mô hình học có nén tăng rank-1 gần gấp đôi.

## 3b2. Kết quả lần chạy cuối trên 5 fold và phần nhận dạng mở rộng (08/10/2026)

**Hai fold giữ kín (1 và 5) cho cùng kết luận.** 60 lần huấn luyện, cả 60 qua phép thử ảnh sáng. Trên AMI nay là 700 ảnh
của 100 người. Trong 149 con số của phần độ trung thực, 41 số đổi, mức đổi lớn nhất 0,07 dB. Theo từng fold (AMI, 36 px,
ảnh vào ước lượng): ước lượng trừ bicubic +1,46 / +1,48 / +1,48 / +1,40 / +1,30 dB ở fold 1 đến 5; trừ tổng quát
+0,93 / +1,09 / +0,96 / +1,17 / +0,90 dB.

**Nhận dạng trên EarVN1.0 được xác nhận bằng mạng thứ hai (ResNet-50):** bicubic 17,5% lên 27,0% (+9,5 [6,5; 12,7]); SPAN công
bố +1,1 [−0,5; 2,6]. Với 5 fold, nhánh tổng quát kém nhánh ước lượng một chút nhưng có ý nghĩa (−1,3 [−2,6; −0,1] với
ResNet-18; −2,6 [−4,9; −0,6] với ResNet-50). JPEG 75 và mức nén rút đều vẫn ngang nhánh ước lượng.

**Tách theo mức nén của file ảnh dò (EarVN1.0):** ở mức 70 đến 80 (1.431 ảnh, 33 người) SPAN công bố −0,1 [−1,3; 1,0], mô
hình của bài +9,3 [4,3; 14,8], BSRGAN +6,3 [−1,3; 13,8]. Ở mức từ 90 (574 ảnh, 12 người) SPAN công bố +3,0 [1,6; 4,6], mô
hình của bài +18,5 [10,6; 26,7], BSRGAN +35,2 [22,5; 48,0]. Ngưỡng của ảnh mô phỏng lặp lại trên ảnh thật.

**AWEx: phần tăng KHÔNG lặp lại.** 112 người đăng ký, 81 người có ảnh nhỏ thật, 428 ảnh dò. Không phương pháp nào đổi rank-1
có ý nghĩa: mô hình của bài −3,8 [−7,9; 0,5] (ResNet-18) và +1,9 [−2,7; 6,4] (ResNet-50); SPAN công bố +0,5 và −0,5; BSRGAN
+0,5 và +1,6. Hai quan sát rút ra **sau khi thấy kết quả** (chưa phải lời giải thích đã kiểm):
- ảnh nhỏ của AWEx không có lưới khối JPEG (tỉ số biên khối 0,98; EarVN1.0 là 1,19 ở mức 75 và 1,08 ở mức 93); AWEx phát
  hành dạng PNG, không rõ lịch sử nén;
- phép đo yếu trên AWEx: trung vị 2 ảnh đăng ký mỗi người (EarVN1.0: 24), và ảnh dò lớn không được nhận tốt hơn ảnh dò nhỏ
  (32,2% so với 31,8%), tức độ phân giải không phải thứ giới hạn mạng nhận dạng ở bộ này.

**Câu chữ trong bài đã sửa theo:** phần tăng nhận dạng "gắn với việc ảnh vào bị nén"; bài viết rõ "không tuyên bố SR cải thiện
nhận dạng tai nói chung". Kết quả AWEx được báo nguyên trong bảng 10 và trong abstract.

## 3b3. Câu đóng góp về nhận dạng, viết lại theo góp ý ngày 08/10

**Câu:** phóng ảnh chỉ giúp nhận dạng tai khi hai điều cùng đúng: ảnh dò đã bị nén, và mạng nhận dạng nhận ảnh lớn tốt hơn
ảnh nhỏ của cùng những người đó ("có dư địa"). EarVN1.0 thỏa cả hai (ảnh lớn 60,2%, ảnh nhỏ 13,2%; mô hình của bài thêm
11,8 điểm). AWEx thì ảnh lớn 32,2%, ảnh nhỏ 31,8%, và không phương pháp nào đổi rank-1.

**Chỗ góp ý chưa nói, và bài đã ghi rõ:** AWEx không thỏa **cả hai** điều kiện (ảnh nhỏ của nó cũng không có lưới khối
JPEG), nên hai bộ ảnh không tách được điều kiện nào là điều kiện cần. Bằng chứng trong từng bộ ảnh, tính từ kết quả có sẵn:
- điều kiện nén: phép tách theo mức nén của file trên EarVN1.0 (mục 3b2);
- điều kiện dư địa: trên EarVN1.0 (30 người có từ 5 ảnh dò), nửa số người có ít dư địa (+25,4 điểm) tăng +7,5 điểm, nửa có
  nhiều dư địa (+55,9) tăng +16,4; với ResNet-50 là +5,3 và +14,6. Tương quan hạng dương nhưng **chưa đạt ý nghĩa thống kê**
  (ρ = 0,28, p = 0,14; ρ = 0,35, p = 0,06);
- trên AWEx, những người có dư địa (+26,8) vẫn không tăng (−1,1 và +3,4): dư địa không đủ nếu ảnh không nén. Số theo người
  ở AWEx dựa trên trung vị 2 ảnh dò, rất nhiễu.

Bài viết hai điều kiện này là "điều kiện do số liệu gợi ý", nêu rõ là hình thành sau khi thấy kết quả AWEx.

**Đối chứng trên ảnh tự nhiên** (`scripts/run_div2k_control.sh`; chờ chạy): xem mục 0 của `TODO.md`.

## 3b4. Đối chứng trên ảnh tự nhiên và quy tắc "tỉ số sai số" (08/10/2026)

16 mô hình có sẵn trên 100 ảnh DIV2K, ảnh vào 24, 36, 48, 96, 192 px, cùng năm kiểu ảnh vào (`results/t2_div2k/`, 425 file).

- **Cỡ ảnh không dời ngưỡng.** Trên ảnh tự nhiên, ở JPEG 75 cả 16 mô hình vẫn hơn bicubic ở mọi cỡ (trung vị +0,11 đến
  +0,20 dB); ở JPEG 60 cả 16 đều kém ở mọi cỡ (−0,06 đến −0,08 dB).
- **Nội dung thì có.** Ảnh tự nhiên đổi dấu giữa mức 75 và 60; ảnh tai (AMI) đổi dấu giữa mức 93 và 85.
- **Kiến trúc mất khác biệt khi nén cũng là hiện tượng chung:** trên ảnh tự nhiên 16 mô hình cách nhau 0,47 dB khi sạch và
  0,07 dB ở JPEG 75.

**Một đại lượng giải thích cả hai** (tính từ kết quả có sẵn, không chạy thêm): tỉ số giữa phần sai số do nén thêm vào và
sai số nội suy của bicubic (E_c / E_i). Trên 35 ô (bốn bộ ảnh × cỡ ảnh vào 24 đến 192 px × mức JPEG 60 đến 93): mọi ô có tỉ
số từ 0,168 trở xuống thì cả 16 mô hình hơn bicubic; mọi ô từ 0,170 trở lên thì cả 16 kém (riêng EarVN1.0 ở mức 75 là 14
trên 16). Tương quan hạng −0,91. Ảnh tai trơn nên sai số nội suy nhỏ (bicubic đạt 36,7 dB trên AMI, 23,5 dB trên DIV2K),
cùng một mức JPEG chiếm phần sai số lớn hơn nhiều: ở mức 75 tỉ số là 0,69 với AMI và 0,10 với DIV2K.

**Phần riêng của ảnh tai, viết lại:** không phải hiện tượng, mà là vị trí của nó: các mức nén mà ảnh tai nhỏ thật mang
nằm bên kia ngưỡng, còn ảnh tự nhiên ở cùng mức nén thì chưa.

**Giới hạn phải nêu:** ngưỡng 0,17 là thực nghiệm; chỉ kiểm với 16 mô hình học bằng bicubic, ×4, nén JPEG; hai ô sát ngưỡng
nằm hai phía ở tỉ số gần bằng nhau với phần hơn khác nhau, nên phần hơn không phải hàm của riêng tỉ số. Chưa rà tài liệu
xem quy tắc này đã có ai nêu chưa.

Bản thảo: mục 4.3 mới (bảng, hình `fig_ratio.pdf`), abstract viết lại mở đầu bằng quy tắc này, mở bài và thảo luận sửa theo.

## 3c. Kết quả rà tài liệu lượt đầu (07/10/2026; tìm kiếm web, chưa phải rà có hệ thống)

- **Về ảnh tai:** chỉ tìm thấy một bài SR cho ảnh tai (Markičević, Peer, Emeršič, IWSSIP 2023: EDSR và SwinIR, ×2 và ×4, trên
  UERC, đo rank-1 bằng ResNet, mốc là ảnh gốc và ảnh nội suy bicubic). Chưa đọc được toàn văn, nên chưa biết họ có xét nén
  không. Có hai bài nền cho động cơ: UERC 2019 (độ phân giải là một nguồn dao động chính của nhận dạng tai) và Rathgeb và
  cộng sự 2016 (ảnh hưởng của nén ảnh tới nhận dạng tai).
- **Về SR cho ảnh đã nén:** hướng này **đã có** trên ảnh tự nhiên: CISRDCNN (2018), cuộc thi AIM 2022, và một mạng hiệu quả
  cho ảnh nén (Ma và cộng sự, WACVW 2024). Vì vậy bài **không được** viết "huấn luyện có nén là ý mới".
- **Về học suy giảm từ ảnh thật của một đặc trưng sinh trắc:** Bulat và cộng sự (ECCV 2018) đã làm cho ảnh mặt, bằng GAN.
- **Chưa tìm thấy:** bài nào đo ngưỡng mức nén mà từ đó các mô hình SR hiệu quả có sẵn kém nội suy, ở ảnh vào vài chục
  pixel, có kiểm trên ảnh thật của miền đích. Đây là phần mới mà bài đứng trên đó.
- **Hệ quả cho định vị:** bài là một nghiên cứu thực nghiệm trên một miền và cỡ ảnh chưa ai đo, không phải một phương pháp
  mới. Khớp với đánh giá trước đó; không có gì trong lượt rà này làm bài mạnh lên hay yếu đi rõ rệt.

## 4. Bằng chứng còn thiếu: danh sách đóng

Năm việc. **Việc 1, nửa đầu việc 2 (nhận dạng) và việc 3 đã xong (08/10)**; việc 4 xong (iPhone 12 Pro Max: SPAN 4,7 ms chỉ CPU, 0,8 ms trên Neural Engine); việc 5 xong lượt đầu; khảo sát người xem để dành.

| # | Việc | Phục vụ | Chi phí | Kết quả đổi gì trong bài |
|---|---|---|---|---|
| 1 | Tách phần "có nén" khỏi phần "đo từ dữ liệu": hai nhánh huấn luyện mới (JPEG với đúng phân bố mức nén đã đo; JPEG rút đều 60 đến 95) | P3 | 6 lần huấn luyện, khoảng 3 giờ | Chỉ đổi câu chữ của P3: "đo từ ảnh tai thật" hay "có nén ở dải mức phù hợp". Bài đứng được ở cả hai nhánh |
| 2 | Một thước đo trên ảnh tai nhỏ thật: khảo sát người xem. Thêm nhận dạng tai (người dùng đã quyết làm, 07/10): 51 người được đăng ký, 36 người trong đó có ảnh nhỏ thật, 2.015 ảnh dò | P3 | Người xem: vài ngày. Nhận dạng: 1 đến 2 ngày mã, vài giờ GPU | Không có việc này thì P3 chỉ đúng trên ảnh mô phỏng; đây là điểm người phản biện bắt đầu tiên |
| 3 | Chạy cuối trên đủ 5 fold (thêm fold 1 và 5 đang giữ kín), các nhánh của P2, P3 trên hai thân | P2, P3 | Khoảng 20 lần huấn luyện, 10 giờ | Số chính của bài: 700 ảnh, 100 người, năm lần huấn luyện độc lập cho mỗi nhánh |
| 4 | Độ trễ trên iPhone 12 Pro Max qua Core ML (người dùng chốt 07/10; không dùng Jetson hay Android) | P4 | Một buổi | Không đạt 33 ms thì hạ xuống "near real-time" hoặc bỏ chữ "Real-Time" khỏi tiêu đề |
| 5 | Rà tài liệu: SR ảnh tai, SR ảnh đã nén, suy giảm ngoài thực tế | P1 | Vài ngày đọc | Xác định P1 mới đến đâu; quyết định câu "first" có được viết không |

Kèm theo, không cần chạy gì mới: một hình cơ chế cho P1 (giả thuyết cần kiểm: sai số của mô hình có sẵn bám theo lưới khối 8×8 của JPEG), dựng
bảng và hình bằng script.

## 5. Những việc để dành, không làm cho bài này

Chỉ làm nếu người phản biện yêu cầu, hoặc thuộc bài sau.

- **Để dành cho vòng phản biện** (đều rẻ, chỉ chấm hoặc vài lần huấn luyện): mốc BSRNet và Real-ESRNet; mốc khử nén rồi SR;
  thêm seed; độ bền với thứ tự suy giảm; đối chứng trên DIV2K; hệ số ×2; AWEx biên an toàn 1,5; bộ dữ liệu thứ ba.
- **Bỏ khỏi bài này:** dò tốc độ học, ba giao thức huấn luyện (T6), khối `pad` và N5b, N1, N3, T4, T5, nhánh trộn ảnh sạch.
- **Bài sau:** kiến trúc mới (hướng B, C ở `TODO.md` mục 2b), dùng benchmark và mốc của bài này.

Hệ quả cho câu chữ: vì không có mốc BSRNet và Real-ESRNet, P1 phải viết là "mô hình học bằng bicubic", không viết "mọi mô
hình có sẵn". Suy giảm tổng quát đã có mặt trong bài qua nhánh huấn luyện có kiểm soát (cùng thân, khác suy giảm).

## 6. Rủi ro còn lại, nói thẳng

- Hiện tượng "mô hình học bằng bicubic hỏng khi ảnh bị nén" đã được biết với ảnh tự nhiên. Phần mới của bài là: đo trên ảnh
  vài chục pixel của một miền sinh trắc, chỉ ra ngưỡng mức nén nằm đúng giữa hai mức mà ảnh tai thật hay gặp, và chỉ ra suy
  giảm tổng quát không giải quyết được. Đủ cho IMAVIS hay không phụ thuộc nhiều vào việc 2 và việc 5.
- Mỗi fold một seed. Năm fold cho năm lần huấn luyện độc lập; nếu dấu của mọi phép so giống nhau ở cả năm thì đủ.
- AMI chụp trong phòng, 100 người; ảnh ngoài thực tế có đáp án chỉ ở cỡ nhỏ nhất.
