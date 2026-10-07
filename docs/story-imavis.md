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

## 4. Bằng chứng còn thiếu: danh sách đóng

Năm việc. **Việc 1 và nửa đầu việc 2 (nhận dạng) đã xong ngày 07/10**; còn khảo sát người xem, việc 3, 4, 5.

| # | Việc | Phục vụ | Chi phí | Kết quả đổi gì trong bài |
|---|---|---|---|---|
| 1 | Tách phần "có nén" khỏi phần "đo từ dữ liệu": hai nhánh huấn luyện mới (JPEG với đúng phân bố mức nén đã đo; JPEG rút đều 60 đến 95) | P3 | 6 lần huấn luyện, khoảng 3 giờ | Chỉ đổi câu chữ của P3: "đo từ ảnh tai thật" hay "có nén ở dải mức phù hợp". Bài đứng được ở cả hai nhánh |
| 2 | Một thước đo trên ảnh tai nhỏ thật: khảo sát người xem. Thêm nhận dạng tai (người dùng đã quyết làm, 07/10): 51 người được đăng ký, 36 người trong đó có ảnh nhỏ thật, 2.015 ảnh dò | P3 | Người xem: vài ngày. Nhận dạng: 1 đến 2 ngày mã, vài giờ GPU | Không có việc này thì P3 chỉ đúng trên ảnh mô phỏng; đây là điểm người phản biện bắt đầu tiên |
| 3 | Chạy cuối trên đủ 5 fold (thêm fold 1 và 5 đang giữ kín), các nhánh của P2, P3 trên hai thân | P2, P3 | Khoảng 20 lần huấn luyện, 10 giờ | Số chính của bài: 700 ảnh, 100 người, năm lần huấn luyện độc lập cho mỗi nhánh |
| 4 | Độ trễ trên một thiết bị (Jetson; không có thì điện thoại hoặc CPU) | P4 | Một buổi khi có thiết bị | Không có thì bỏ chữ "Real-Time" khỏi tiêu đề, giữ số GPU |
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
