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

## 4. Bằng chứng còn thiếu: danh sách đóng

Năm việc. Xong năm việc này thì viết.

| # | Việc | Phục vụ | Chi phí | Kết quả đổi gì trong bài |
|---|---|---|---|---|
| 1 | Tách phần "có nén" khỏi phần "đo từ dữ liệu": hai nhánh huấn luyện mới (JPEG với đúng phân bố mức nén đã đo; JPEG rút đều 60 đến 95) | P3 | 6 lần huấn luyện, khoảng 3 giờ | Chỉ đổi câu chữ của P3: "đo từ ảnh tai thật" hay "có nén ở dải mức phù hợp". Bài đứng được ở cả hai nhánh |
| 2 | Một thước đo trên ảnh tai nhỏ thật: khảo sát người xem. Thêm nhận dạng tai nếu người dùng quyết định làm (41 người, 2.060 ảnh nhỏ thật) | P3 | Người xem: vài ngày. Nhận dạng: 1 đến 2 ngày mã, vài giờ GPU | Không có việc này thì P3 chỉ đúng trên ảnh mô phỏng; đây là điểm người phản biện bắt đầu tiên |
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
