# SR real-time cho ảnh tai độ phân giải thấp: đề xuất và bản phác thảo bài báo

Ngày lập: 05/10/2026. **Bản 25 (06/10/2026; bổ sung tối cùng ngày: kết quả trên EarVN1.0 và AWEx, và khối `n2` chạy lại, ở mục 2.0):** giai đoạn 1 đã chạy đủ trên GPU và phép thử đầu của N2 (bộ phân loại) đã có kết quả. Phần 2 được cập nhật theo số thật: thêm mục 2.0 (dữ kiện đã có và bảng theo dõi luận điểm); viết hẹp lại C2 (chỉ đúng trên số đo độ trung thực), C3 (chỉ đúng khi suy giảm lệch) và C5 (suy giảm đơn giản đo từ dữ liệu là đủ, hơn suy giảm tổng quát); điền các chỗ `[X]` đã có số ở mục 2.3 và 2.6. Phần 1 không đổi; số sơ bộ ở mục 1.2c được thay bằng số ở mục 2.0. **Bản 24:** bổ sung ba chỗ còn thiếu của Phần 2: script dựng bảng so sánh và vẽ hình (mục 2.5), Bảng 1 công trình liên quan điền tới mức đã xác minh (mục 2.5b), và thư mẫu xin phép dùng ảnh AMI; mục 3.14 cập nhật theo. **Bản 23:** viết lại Phần 2 (khung bản thảo): câu chuyện, ba câu hỏi nghiên cứu, chín luận điểm kèm bảng, hình và file kết quả tương ứng, câu viết sẵn cho nhánh đạt và không đạt, và quy trình điền bài sau khi có kết quả. **Bản 22:** thêm mục 1.2c: trọng tâm chuyển sang N2 theo kết quả sơ bộ của giai đoạn 1; thân mô hình chọn trong nhóm NTIRE 2026 (sáu mô hình đã vào kho) thay vì gắn với SPAN gốc; chạy phép thử N2 và phép thử kiểu đệm rẻ trước khi tiền huấn luyện; thêm số đo MS-SSIM, GMSD, PSNR của gradient, LR-PSNR và nhóm số đo qua pyiqa. **Bản 21:** thêm mục 1.8b, giao thức so sánh công bằng: hai nhánh so sánh (có kiểm soát; trọng số công bố), quy tắc "chỉ khác một yếu tố", và việc nhóm tự tiền huấn luyện lại các mốc chủ chốt trở thành bắt buộc. **Bản 20:** thêm mục 3.14 (dữ liệu đặt ở đâu, script nào, file kết quả nào cho từng việc) và một đoạn ở mục 1.6 giải thích vì sao benchmark chính là AMI; các quyết định khoa học không đổi so với bản 19. **Bản 19, sửa theo phản biện vòng 9.** Phản biện không còn ý kiến nào về ý tưởng. Bản này đưa Phần 3 (pipeline) theo kịp Phần 1: nhánh chính của mô hình là thân N5b huấn luyện với suy giảm N2; thêm giai đoạn tiền huấn luyện, công cụ cho hai phép thử lõi, ba trường trong mã lần chạy, một nhóm kiểm thử; và đổi thứ tự: tháng đầu chỉ chạy các phép thử không cần huấn luyện, T4 và T5 dời ra sau điểm kiểm tra 1. Chi tiết ở **Phụ lục 6**. Các bản trước: bản 18 (vòng 8, Phụ lục 5), bản 17 (vòng 7, Phụ lục 4), bản 16 (Phụ lục 3).

Cấu trúc: Phần 1 đề xuất; Phần 2 khung bản thảo; Phần 3 thiết kế project; Phụ lục 1 và 2 trả lời phản biện vòng 5 và 6; Phụ lục 3 thay đổi của bản 16; Phụ lục 4, 5 và 6 trả lời phản biện vòng 7, 8 và 9.

**Trạng thái.** Chưa có kết quả huấn luyện nào. Số liệu ở mục 1.3 và 1.6 là số đo thật trên dữ liệu. Mọi chỗ `[X]` là số chưa có. Mọi ngưỡng số ghi "đề xuất" là do nhóm đặt và có thể chỉnh trước khi chạy, không chỉnh sau khi chạy.

**Đề bài của giáo sư, nguyên văn:** cải tiến SR model, một mô hình real-time cho ảnh tai; đánh giá thông qua metric, chất lượng hình ảnh; bỏ qua phần recognition cho id/gender; làm cho ảnh to lên, rõ hơn trước; real-time hoặc near real-time.

---

# PHẦN 1. ĐỀ XUẤT

## 1.1 Mục tiêu và loại tạp chí

Một mô hình SR real-time cho ảnh tai nhỏ (ảnh vào có cạnh ngắn 24 đến 48 px), ở ×4 (chính) và ×2, đánh giá bằng metric chất lượng ảnh theo giao thức NTIRE. Dữ liệu công khai, không tự chụp. Bài độc lập với bài span_tiny (chưa nộp).

**Tạp chí:** do giáo sư quyết định. Điều kiện duy nhất: xếp Q1 và không phải tạp chí săn mồi; không gắn với một tạp chí cụ thể nào. Việc của kế hoạch này là làm bài mạnh nhất có thể, với mọi thí nghiệm làm được trên một RTX 3080 và dữ liệu công khai, và viết bản thảo ở dạng nộp được cho cả tạp chí xử lý ảnh lẫn tạp chí sinh trắc.

Một điểm phụ thuộc lựa chọn đó: nếu nộp tạp chí sinh trắc, reviewer gần như chắc sẽ hỏi SR giúp hay hại nhận dạng, trong khi đề bài bỏ phần nhận dạng. Phép kiểm tra "không gây hại" (một mạng nhận dạng tai có sẵn, đóng băng, cho thấy ảnh SR không làm xấu đặc trưng danh tính so với ảnh nội suy) vì vậy được chuẩn bị sẵn như một thí nghiệm phụ lục (mục 1.10, E7), và chỉ chạy khi giáo sư đồng ý.

## 1.2 Câu chuyện và chuỗi luận điểm

*Từ bản 23, câu chuyện và chuỗi luận điểm hiện hành nằm ở Phần 2 (mục 2.1, 2.2), viết lại theo trọng tâm mới của mục 1.2c. Bảng L1 đến L7 dưới đây giữ lại để theo dõi lịch sử; chỗ nào khác Phần 2 thì theo Phần 2.*

**Câu chuyện, trong bốn câu.** Ảnh tai chụp ngoài thực tế thì nhỏ. Các mô hình SR real-time lại được thiết kế và xếp hạng trên ảnh tự nhiên lớn, với suy giảm bicubic, nên lệch với ảnh tai ở hai chỗ chính: cỡ ảnh vào và kiểu suy giảm. Bài sửa hai chỗ lệch đó trong một mô hình real-time, bằng một thân chỉnh cho ảnh vào cực nhỏ và một mô hình suy giảm ước lượng từ ảnh tai thật. Kết quả là ảnh tai rõ hơn, ở cùng tốc độ, so với việc lấy một mô hình đa dụng rồi tinh chỉnh trên ảnh tai.

**Chuỗi luận điểm.** Mỗi luận điểm là một mắt xích; bài chỉ đứng vững khi các mắt xích được chứng minh theo thứ tự.

| # | Luận điểm | Bằng chứng | Bị bác nếu | Vị trí trong bài |
|---|---|---|---|---|
| L1 | Bài toán có thật và có chỗ để làm: ở ảnh tai nhỏ, SR còn dư địa so với nội suy, và dư địa đó phụ thuộc cỡ ảnh vào | Bicubic và các mô hình có sẵn theo ba cỡ ảnh, trên đáp án sạch của AMI | Mọi mô hình gần bằng bicubic ở cả ba cỡ | Mục 3.1 |
| L2 | Mô hình đa dụng lệch với chế độ này: ở ảnh tai nhỏ, tập mô hình real-time và thứ hạng của chúng khác với bảng NTIRE; thêm nén và mờ thì thứ hạng đổi tiếp | Độ trễ đo được; thứ hạng dưới bicubic và dưới suy giảm có nén | Thứ hạng giữ nguyên như NTIRE trong mọi thiết lập | Mục 3.4, 5.5 |
| L3 | Tinh chỉnh mô hình đa dụng trên ảnh tai là mốc mạnh, và là mốc đúng để so | Mô hình nguyên bản so với mô hình tinh chỉnh | (không cần bác; đây là cách đặt mốc) | Mục 5.1 |
| L4 | Sửa từng chỗ lệch đều thêm được trên mốc đó: cỡ ảnh vào (N5b) và suy giảm (N2) là hai chỗ chính; thêm tối đa một trong cấu trúc (N1) và nơi được phép thêm chi tiết (N3) | Ablation từng thành phần theo tiêu chí ở mục 1.2b | Thành phần không đạt tiêu chí của nó | Mục 5.3 |
| L5 | Ảnh rõ hơn thật, không phải do bịa cấu trúc | Độ lệch điểm mốc; bảng kiểm tra gờ; khảo sát người xem có đáp án ("ảnh nào giống đáp án hơn") | Điểm mốc lệch nhiều hơn mốc; người xem không chọn nhiều hơn 50% | Mục 5.2, 5.6 |
| L6 | Vẫn real-time, và hơn mốc dọc theo cả đường chất lượng theo độ trễ | Độ trễ trên Jetson Nano và điện thoại; mô hình đề xuất ở hai đến ba cỡ so với đường của các mốc | Không đạt ngưỡng ở mục 1.5, hoặc nằm dưới đường của các mốc | Mục 5.7 |
| L7 | Kết quả không do riêng một kiến trúc hay một bộ dữ liệu | Thân thứ hai; nhóm người giữ riêng của EarVN1.0; AWEx không dùng khi huấn luyện (chỉ đủ ảnh ở cỡ đáp án 96 px) | Chỉ có trên một thân hoặc chỉ trên AMI | Mục 5.3, 5.5 |

**Nếu một mắt xích gãy.**

- L1 gãy: không có bài toán để giải; dừng.
- L2 gãy: câu chuyện mất phần "mô hình đa dụng lệch chế độ"; bài rút về "thiết kế riêng hơn tinh chỉnh" (L3 đến L7), yếu hơn nhưng vẫn viết được.
- L4 chỉ đạt một phần: bài chỉ tuyên bố những thành phần đạt.
- L5 gãy: không được dùng chữ "rõ hơn"; chỉ còn tuyên bố theo metric.

L1, L2 và cơ sở của N5b (hiệu ứng viền) được kiểm ngay ở giai đoạn 1, không cần huấn luyện.

## 1.2b Các điểm mới và tiêu chí đạt

**Luận điểm trung tâm (một phép so sánh, không dùng chữ "đầu tiên").** Ở cùng tốc độ đo được, một mô hình thiết kế riêng cho ảnh tai nhỏ cho ảnh rõ hơn một mô hình SR hiệu quả đa dụng đã được tinh chỉnh trên ảnh tai.

Mốc phải vượt là **mô hình đa dụng đã tinh chỉnh trên ảnh tai** (ví dụ SPAN tinh chỉnh), không phải mô hình đa dụng nguyên bản. Việc tinh chỉnh đã lấy gần hết phần hơn do dữ liệu; các điểm mới dưới đây phải thêm được gì trên mức đó.

| # | Điểm mới ứng viên | Công trình gần nhất | Khác ở đâu | Tiêu chí đạt (đề xuất) |
|---|---|---|---|---|
| N1 | Đầu phụ học cấu trúc tai (điểm mốc, đường viền), chỉ dùng lúc huấn luyện | SR ảnh mặt dẫn hướng bằng điểm mốc (FSRNet, DIC) | Áp cho tai; không tốn gì lúc chạy | So với cùng thân không có đầu phụ, so sánh ghép cặp theo từng ảnh, bootstrap theo người, gộp các seed: (1) độ lệch điểm mốc giảm ít nhất 5% và khoảng tin cậy 95% của mức giảm không chứa 0; (2) mức giảm cùng chiều ở từng seed; (3) thước đo có dải động (mục 1.9); (4) bảng kiểm tra gờ và LPIPS không xấu đi quá 2% tương đối (cận xấu của khoảng tin cậy); (5) PSNR không giảm; (6) vẫn hơn nhánh đối chứng "thân + ảnh thêm, không nhãn" |
| N2 | Suy giảm ước lượng từ ảnh tai thật | Real-ESRGAN, BSRGAN; Ji và cộng sự 2020 (ước lượng nhân mờ và nhiễu từ ảnh thật) | Tham số lấy từ thống kê của EarVN1.0 | (a) Bộ phân loại "mô phỏng hay thật" có độ chính xác gần 50% hơn so với suy giảm tổng quát; (b) trên ảnh nhỏ thật, người xem chọn ảnh của mô hình này nhiều hơn 50%, khoảng tin cậy không chứa 50% |
| N3 | Cổng điểm ảnh chạy một lượt | LDL, DeSRA; SROOE (bản đồ mục tiêu theo điểm ảnh) | Quyết định trong mô hình, lúc suy luận | Bản thiên sắc nét được tụt PSNR không quá 0,5 dB so với thân tinh chỉnh (ngân sách ghi trước). **Điều kiện chính:** tại điểm vận hành đó, LPIPS thấp hơn đường trộn ở cùng PSNR ít nhất 3% tương đối, khoảng tin cậy 95% của chênh lệch ghép cặp không chứa 0; cùng điều kiện so với thân + LDL. Mức giảm LPIPS so với thân tinh chỉnh chỉ báo cáo, không là tiêu chí |
| N4 | Benchmark ảnh tai nhỏ: đáp án sạch ở ba cỡ, phân tích dư địa của SR theo cỡ ảnh vào, số đo thiết bị | Bài EDSR và SwinIR trên ảnh tai của UERC | Đáp án sạch; nhiều cỡ; giao thức NTIRE | Làm đủ các bảng |
| N5a | **Giao thức huấn luyện một mô hình cho mọi cỡ** (tỉ lệ ngẫu nhiên từ 96 đến 192 px). **Mặc định không tính là điểm mới** | Tăng cường đa tỉ lệ thông thường; FixRes; TLC | Một bộ trọng số dùng cho mọi cỡ ảnh vào, thay cho một mô hình mỗi cỡ | Ba nhánh trên hai đến ba mốc: (1) cắt từ ảnh gốc 492×702, sai tỉ lệ; (2) huấn luyện trên ảnh HR đúng cỡ của ô đang xét, là cách hiển nhiên; (3) tỉ lệ ngẫu nhiên. N5a chỉ được nêu là đóng góp nếu nhánh 3 hơn nhánh 2 ít nhất 0,1 dB (khoảng tin cậy không chứa 0), hoặc một mô hình duy nhất của nhánh 3 không kém các mô hình chuyên từng cỡ của nhánh 2 quá 0,05 dB ở cả ba cỡ. Nếu không, nó là một dòng ablation |
| N5b | **Kiến trúc cho ảnh vào nhỏ hơn vùng nhìn của mạng:** kiểu đệm viền, tỉ lệ sâu và rộng ở cùng độ trễ | Mind the Pad; Kayhan và van Gemert; partial convolution padding; SR ảnh mặt 16×16 | Làm cho SR hiệu quả, dưới ràng buộc độ trễ; phản biện không biết bài nào đã làm | (a) Phép thử ngữ cảnh: tại cùng điểm ảnh, sai số khi chỉ có đệm cao hơn khi có ngữ cảnh thật, khoảng tin cậy không chứa 0. (b) Trong nhóm cùng ngân sách tiền huấn luyện rút gọn, biến thể hơn thân tham chiếu ít nhất 0,1 dB; sau đó biến thể thắng được tiền huấn luyện đủ và phải hơn mốc dùng trọng số công bố, đã tinh chỉnh bằng giao thức tốt hơn, ở cùng độ trễ. (c) Phép thử tương tác: phần hơn ở ảnh vào 36 px lớn hơn phần hơn ở ảnh vào lớn, khoảng tin cậy của hiệu hai phần hơn không chứa 0. Thiếu (c) thì chỉ được nói "thân tốt hơn" |

- N5b là giả thuyết, chưa có số liệu nào. N5a được hạ xuống thành giao thức huấn luyện theo phản biện vòng 8: thắng nhánh "cắt từ ảnh gốc" không chứng minh được gì, vì người làm SR bình thường vốn tinh chỉnh trên ảnh HR đúng cỡ của benchmark. Mốc phải vượt luôn dùng giao thức tốt hơn trong nhánh 2 và nhánh 3, chọn trên validation; mô hình đề xuất dùng cùng giao thức đó.
- **Lõi của bài là N5b và N2:** hai chỗ "lệch chế độ" là cỡ ảnh vào và suy giảm. Bài giữ thêm **tối đa một** trong N1, N3. N3 là ứng viên cắt đầu tiên: mức mới thấp nhất và kéo theo nhiều thứ nhất (teacher, nhánh GAN, LDL, đường trộn, hai giai đoạn). N1 được ưu tiên nếu đạt, vì là thành phần duy nhất gắn với giải phẫu tai. **Ngoại lệ ghi trước:** nếu phần hơn của N1 nhỏ hơn 1,5 lần ngưỡng của nó, trong khi phần hơn của N3 so với đường trộn từ 2 lần ngưỡng trở lên, thì giữ N3. Thành phần còn lại chuyển xuống phụ lục của bài.
- N4 chắc chắn có. N1, N2, N3, N5b mỗi điểm chỉ được giữ khi đạt tiêu chí của nó. Bài có benchmark và tối đa ba đóng góp phương pháp: N5b, N2, và một trong N1, N3.
- **Công bằng cho mốc so sánh.** Mỗi phép so sánh chỉ khác nhau đúng một yếu tố. Khi thử N2, thân tinh chỉnh cũng được huấn luyện với cùng kiểu suy giảm. Đường trộn và thân + LDL được chỉnh tham số trên cùng tập validation, với cùng số lần thử, như mô hình đề xuất.
- **N1 dùng thêm dữ liệu có nhãn mà mốc không có** (bộ điểm mốc tai). Bài nói rõ điều này ở phần phương pháp và trong bảng kết quả. Để tách phần hơn do nhãn cấu trúc khỏi phần hơn do có thêm ảnh, có một nhánh đối chứng: thân tinh chỉnh được huấn luyện thêm trên chính các ảnh đó nhưng không dùng nhãn.
- Real-time là ràng buộc phải thỏa, không phải điểm mới. Jetson Nano và điện thoại chỉ là thiết bị để đo.
- Chưa điểm nào có số liệu. Các công trình gần nhất trong bảng được ghi theo trí nhớ và phải xác nhận ở T1; cũng chưa rà để biết N1 đã có ai làm cho ảnh tai chưa.
- **Đánh giá thẳng về tính mới.** N1, N2, N3 đều là phiên bản áp dụng của ý đã có. Nếu chỉ có ba điểm này, một reviewer khó tính sẽ gọi bài là "tổ hợp kỹ thuật đã biết cho một miền mới". N5b là điểm duy nhất xuất phát từ chính bài toán, và mức mới của nó cũng chỉ vừa phải: ảnh hưởng của đệm viền và lệch tỉ lệ đã được nghiên cứu ở các bài toán khác. Nếu N5b đạt, nó là lõi kỹ thuật của bài; nếu không, bài ở kịch bản A, mức ranh giới. Việc này chốt ở điểm kiểm tra 1.

**Kịch bản.**

| Kịch bản | Điều kiện | Kết quả |
|---|---|---|
| A+ | **N5b và N2 đều đạt**, trên cả hai thân | Bài mô hình, lõi là thiết kế cho ảnh vào cực nhỏ. Phản biện vòng 7 và 8: mức "major revision theo hướng nhận" ở tạp chí Q1 phạm vi rộng |
| A | Ít nhất hai điểm đạt trên cả hai thân, nhưng thiếu N5b (ví dụ N2 cộng N1) | Phản biện: vẫn là mức ranh giới, vì bài không còn thành phần nào xuất phát từ chính bài toán ảnh nhỏ |
| B | Chỉ một điểm đạt | Bài yếu; cân nhắc lại với giáo sư trước khi viết |
| C | Không điểm nào đạt | Không có bài theo hướng này |

## 1.2c Điều chỉnh của bản 22: trọng tâm, thân mô hình, thứ tự chạy, số đo

Mục này ghi bốn thay đổi và có hiệu lực cao hơn các mục cũ ở chỗ nào hai bên khác nhau. Ngưỡng trong `criteria.yaml` không đổi.

**1. Trọng tâm chuyển từ N5b sang N2.** Lý do là kết quả sơ bộ của giai đoạn 1 (700 ảnh AMI, ×4, chạy trên CPU, chưa có LPIPS; phải chạy lại trên GPU trước khi dùng trong bài):

| Quan sát | Số đo sơ bộ | Hệ quả |
|---|---|---|
| Với ảnh vào có nén JPEG mức 75, mọi mô hình SR hiệu quả có trọng số công bố đều thua bicubic | Cỡ 144 px: bicubic 34,36 dB; các mô hình khoảng 33,9 dB. Thứ hạng giữa các mô hình đảo (τ-b 0,15) | Lệch suy giảm là nút thắt lớn nhất. Đây là phát hiện chính của bài |
| Ở suy giảm bicubic, các mô hình cách nhau rất ít | SPAN 39,00; SwinIR-light 39,10; RRDB (lớn gấp 40 lần) 39,28 dB | Kiến trúc không phải nút thắt; đổi khối kiến trúc khó cho phần hơn đáng kể |
| Hiệu ứng viền có thật nhưng nhỏ, và đệm đơn giản trước mạng không sửa được | Mất 0,13 đến 0,35 dB khi thiếu ngữ cảnh; đệm lặp viền hoặc phản chiếu ở ngoài mạng làm kém đi 0,5 đến 1,9 dB | N5b còn là giả thuyết yếu hơn N2; giữ nếu đạt ngưỡng |

Cấu trúc đóng góp của bài, theo thứ tự: (1) phát hiện và phân tích: SR hiệu quả đa dụng thất bại ở ảnh tai nhỏ có nén; (2) N2, suy giảm ước lượng từ ảnh tai thật; (3) N5b, chỉnh thân cho ảnh vào nhỏ, chỉ giữ nếu đạt; (4) N4, benchmark và số đo thiết bị. Kịch bản A+ vẫn là "N5b và N2 đều đạt"; nếu chỉ N2 đạt cùng với phát hiện (1), bài vẫn có lõi, và mức tạp chí phù hợp là Q1 phạm vi rộng hoặc ứng dụng.

**Phép thử quyết định của N2 khó hơn vẻ ngoài.** Mô hình nào được tinh chỉnh với ảnh có nén cũng sẽ vượt lại bicubic; điều đó không chứng minh gì cho N2. N2 chỉ đạt khi suy giảm *ước lượng từ ảnh tai* hơn suy giảm *tổng quát kiểu Real-ESRGAN* trên ảnh nhỏ thật: bộ phân loại "mô phỏng hay thật" và khảo sát người xem (tiêu chí ở mục 1.2b), kèm số đo có tham chiếu trên EarVN1.0 và AWEx. Số đo trên ảnh AMI được suy giảm bằng chính tham số ước lượng thì thiên vị cho N2 và chỉ được báo là phụ.

**2. Thân mô hình không gắn với SPAN gốc.** Rà NTIRE 2026 Efficient SR (báo cáo tháng 4/2026): SPAN là baseline chính thức; ba hạng đầu đều thuộc họ SPAN (SPANV2; hai bản SPANF cắt kênh và chưng cất); hạng 4 thuộc họ TSSR (DISP); hạng 6 là ERRN2. Phần hơn của các đội chủ yếu đến từ cắt kênh, chưng cất và tối ưu phần cứng; PSNR của tất cả quanh 27,0 dB; không đội nào bàn về kiểu đệm hay ảnh vào nhỏ. Sáu mô hình đã vào kho của project, nạp chặt trọng số, xuất được ONNX opset 13:

| Tên trong kho | Mô hình | Họ | Tham số | Set5 ×4 (đo) |
|---|---|---|---|---|
| `span26` | SPAN 28 kênh, baseline chính thức 2026 | SPAN | 0,151 M | 31,73 dB |
| `pds26` | PDS, hạng 2 | SPANF | 0,142 M | 31,76 dB |
| `pkdsr26` | PKDSR, hạng 3 | SPANF | 0,144 M | 31,77 dB |
| `dscf26` | DSCF bản gộp | DSCF | 0,126 M | 31,74 dB |
| `disp26` | DISP, hạng 4 | TSSR | 0,164 M | 31,94 dB |
| `errn26` | ERRN2, hạng 6 | ERRN | 0,246 M | 31,89 dB |

SPANV2 (hạng 1) bị loại vì tốc độ của nó dựa vào một nhân CUDA tự viết, không mang sang TensorRT trên Jetson được; các mô hình Mamba bị loại vì cùng lý do. Hai thân của bài là hai mô hình tốt nhất sau khi tinh chỉnh trên ảnh tai trong nhóm đạt real-time trên thiết bị, khác họ nhau (ứng viên: một từ họ SPAN hoặc SPANF, một từ TSSR hoặc ERRN); chốt bằng T2 và T3, không chốt bằng tên. Trọng số các mô hình 2026 ở dạng đã gộp nhánh; tinh chỉnh chúng là tinh chỉnh tích chập thường, và bài phải nói rõ điều đó.

Kiểu đệm viền nay đổi được trên mọi thân trong kho (`earsr/models/padding.py`; kiểm thử xác nhận trên cả sáu mô hình rằng chỉ vùng sát mép đổi). Vì kiểu đệm không có tham số, có thêm một phép thử rẻ cho N5b: **trọng số công bố, đổi kiểu đệm, rồi tinh chỉnh** (khối `pad`), không cần tiền huấn luyện. Nếu phép thử này không cho thấy gì trên cả hai thân, phần đệm của T6 (iv) bị bỏ trước khi tốn tiền tiền huấn luyện.

**3. Thứ tự chạy mới.** Giai đoạn 1 như cũ, thêm sáu mô hình 2026 → điểm kiểm tra 1 → **N2 sớm** (ước lượng suy giảm; bộ phân loại; khối `n2`: hai mốc × ba kiểu suy giảm, fold 2) → khối `pad` → dò tốc độ học → T6 (ii), (iii) → chỉ khi `pad` hoặc phép thử ngữ cảnh ủng hộ: tiền huấn luyện và T6 (iv), (v) → điểm kiểm tra 2. Lý do: phép thử N2 rẻ và quyết định bài có lõi hay không; tiền huấn luyện đắt và phục vụ giả thuyết yếu hơn.

**4. Số đo bổ sung ngoài PSNR, SSIM, LPIPS, DISTS.** Bốn số đo đầu được tính cho mọi ảnh, mọi mô hình, không cần tải gì:

| Số đo | Đo gì | Chiều tốt | Ghi chú |
|---|---|---|---|
| MS-SSIM (kênh Y) | Cấu trúc ở nhiều tỉ lệ | Cao | Ảnh tai nhỏ hơn mức cần cho 5 tầng chuẩn, nên số tầng giảm theo cỡ ảnh (4 tầng ở 144 px) và trọng số được chuẩn hóa lại; **không so được với MS-SSIM trong tài liệu**, chỉ so giữa các mô hình trên cùng cỡ. Bài ghi số tầng |
| GMSD | Độ giống của bản đồ gradient | Thấp | Theo đúng bài gốc (Xue và cộng sự, 2014) |
| PSNR của gradient | Độ đúng của cạnh và gờ, ít bị vùng phẳng chi phối | Cao | Sobel 3×3 trên kênh Y |
| LR-PSNR | Độ nhất quán với ảnh vào: thu nhỏ ảnh SR rồi so với ảnh đáp án thu nhỏ | Cao | Bắt mô hình bịa chi tiết làm đổi nội dung tần thấp; bổ sung cho LPIPS |
| ST-LPIPS, TOPIQ-FR, FSIM, VIF, PieAPP | Số đo cảm nhận có tham chiếu khác | Tùy số đo | Qua `pyiqa`, bật bằng `evaluate.py --more-metrics`; **chưa chạy thật** |
| NIQE, MANIQA, MUSIQ, CLIP-IQA, điểm NTIRE | Không tham chiếu | Tùy số đo | Chỉ ở cỡ 244 px (mục 1.9); **chưa chạy thật** |
| Gờ giả, gờ mất; độ lệch điểm mốc | Cấu trúc tai | Thấp | Mục 1.9 |

Số đo chính của mọi tiêu chí vẫn là các số đo đã ghi trước ở mục 1.2b (PSNR, LPIPS, độ lệch điểm mốc). Các số đo bổ sung được báo trong bảng để người đọc thấy kết luận không phụ thuộc một số đo; chúng không được dùng để thay tiêu chí sau khi thấy kết quả.

## 1.3 Vì sao trụ "đáp án danh nghĩa" đã bị bỏ

Bản 9 đến 11 có một trụ thứ hai: giả thuyết rằng phần lớn ảnh "lớn" của các bộ dữ liệu tai ngoài thực tế là ảnh nhỏ bị phóng to. Phép thử đối chứng trên 700 ảnh AMI (ảnh máy chuyên dụng, 492×702) ngày 05/10/2026 đã bác giả thuyết ở dạng mạnh. PSNR kênh xám, trung vị, của phép thu nhỏ rồi phóng lại bằng bicubic:

| | Thu nhỏ 2 lần | Thu nhỏ 4 lần |
|---|---|---|
| AMI ở cỡ gốc | 47,5 dB | 41,6 dB |
| AMI thu về cạnh ngắn 192 px | 40,9 dB | 36,4 dB |
| EarVN1.0, cạnh ngắn khoảng 192 px | 45,0 dB | 36,7 dB |
| AWEx, cạnh ngắn khoảng 192 px | 41,0 dB | 35,6 dB |

1. Ở cùng cỡ ảnh, bicubic ×4 trên ảnh AMI nét (36,4 dB) gần bằng EarVN1.0 (36,7 dB): con số đó là đặc điểm của ảnh tai, không phải dấu hiệu phóng to.
2. Phép đo với ngưỡng 40 dB xếp cả ảnh AMI gốc vào loại "không có chi tiết": nó nhầm độ mịn với độ nhòe.
3. Phần còn lại (khoảng 20 đến 30% ảnh EarVN1.0 cỡ 144 đến 192 px có số đo giống ảnh bị phóng 2 lần) là thiểu số và chưa tách được khỏi hai yếu tố khác: khung hình và nén JPEG. Số đo trên AMI ở cỡ 192 px (trung vị, bicubic thu nhỏ rồi phóng lại ×2): ảnh nét 40,9 dB; nén JPEG mức 75 làm tăng lên 42,0 dB; cắt sát tai (70% khung) làm tăng lên 43,5 dB; cả hai cùng lúc cho 44,6 dB, gần bằng EarVN1.0 (45,0 dB). Ảnh bị phóng 2 lần cho 49,4 dB, và chỉ còn 47,2 dB sau khi nén. Nén vừa đẩy ảnh nét lên vừa kéo ảnh bị phóng xuống, nên hai nhóm càng khó tách.

Hệ quả giữ lại cho thiết kế: SR chỉ có dư địa khi ảnh vào nhỏ (mục 1.6), và đáp án lấy từ EarVN1.0, AWEx phải có biên an toàn.

## 1.4 Mô hình

**Thân mô hình chưa chốt.** Thân được chọn sau khi đo độ trễ (mục 1.5): đó là mô hình có chất lượng cao nhất, sau khi tinh chỉnh trên ảnh tai, trong số các mô hình đạt ngưỡng real-time. Nếu một CNN cỡ vừa như EDSR-baseline đạt ngưỡng và hơn SPAN, mô hình đề xuất được xây trên thân đó. Thân thứ hai, dùng để chứng minh phần hơn không do riêng một kiến trúc, là mô hình đứng thứ hai theo cùng tiêu chí và khác họ kiến trúc.

**Thành phần ứng viên trên thân đã chọn.**

- *N5b, thân chỉnh theo chế độ ảnh vào cực nhỏ:* kiểu đệm viền và tỉ lệ sâu và rộng ở cùng độ trễ, chọn theo kết quả T6. Nếu N5b đạt, thân này là phần lõi của mô hình; nếu không, thân là mô hình công bố đã tinh chỉnh. Giao thức huấn luyện (mục 3.10, điểm 2) dùng chung cho cả mốc lẫn mô hình đề xuất.
- *N1, đầu phụ cấu trúc:* một nhánh nhỏ dự đoán bản đồ điểm mốc hoặc đường viền tai từ đặc trưng của thân; chỉ có lúc huấn luyện, gỡ bỏ lúc chạy. Nhãn sinh bằng một bộ dò điểm mốc huấn luyện trên bộ ảnh tai gắn 55 điểm mốc.
- *N3, cổng điểm ảnh:*

```
thân → ├─ đầu trung thực  f_S
       ├─ đầu kết cấu     f_T
       └─ cổng            G = g(đặc trưng, |f_T − f_S|)
x̂ = G ⊙ f_S + (1 − G) ⊙ f_T
```

  Cổng học ở giai đoạn hai với nhãn có biên τ. Hai cấu hình thử: đầu trung thực là thân đã tinh chỉnh rồi đóng băng (A), hoặc hai đầu huấn luyện chung (B). Dịch độ lệch cổng lúc suy luận cho một bản thiên trung thực và một bản thiên sắc nét từ cùng bộ trọng số. Nhãn điểm mốc của N1 có thể dùng thêm để giám sát cổng.
- *N2* không phải thành phần của kiến trúc mà là cách tạo dữ liệu huấn luyện (mục 1.7).
- Trong N1 và N3, mô hình cuối giữ tối đa một (mục 1.2b). Nếu N3 bị cắt, mô hình chỉ có một đầu ra; các mô hình cảm nhận vẫn có trong bảng so sánh, dùng trọng số công bố.

**Tiêu chí chung cho mô hình cuối.** (1) Đạt ngưỡng real-time ở mục 1.5. (2) Bản thiên trung thực có PSNR không thấp hơn thân tinh chỉnh quá 0,05 dB. (3) Bản thiên sắc nét tụt PSNR không quá 0,5 dB so với thân tinh chỉnh, và ở cùng PSNR có LPIPS, DISTS tốt hơn đường trộn (tiêu chí N3, mục 1.2b). (4) Ablation cho thấy từng thành phần được giữ đều đóng góp.

## 1.5 Ngưỡng real-time (đề xuất, ghi trước khi đo)

| Yếu tố | Giá trị đề xuất |
|---|---|
| Thiết bị | Jetson Nano (cần bạn xác nhận đời máy và dung lượng bộ nhớ); thêm một điện thoại Android làm số đo thứ hai |
| Cỡ ảnh vào | 48×68 px (cỡ lớn nhất của ba cỡ chính ở ×4). Ở ×2 ảnh vào lớn nhất là 96×136 px, gấp 4 lần số điểm ảnh, nên độ trễ ×2 được đo và báo riêng; ngưỡng áp cho từng hệ số phóng |
| Độ chính xác số | FP16, TensorRT, lô 1 |
| Phạm vi đo | Chỉ lượt chạy của mô hình; độ trễ cả pipeline (cắt vùng tai, tiền và hậu xử lý) báo riêng |
| Ngưỡng | Trung vị độ trễ không quá 33 ms (30 FPS); "gần real-time" là không quá 66 ms (15 FPS) |
| Chế độ nguồn | Mức cao nhất của máy (MAXN), khóa xung nhịp bằng `jetson_clocks`; ghi lại chế độ, phiên bản JetPack và TensorRT, nhiệt độ lúc đo |
| Số lượt | 50 lượt làm nóng, 500 lượt đo liên tiếp |
| Số báo cáo | Trung vị và phân vị 95. Xếp nhóm theo trung vị; mô hình có phân vị 95 vượt ngưỡng của nhóm thì ghi chú "không ổn định" |

Mô hình nào đạt ngưỡng thì là đối thủ, dù lớn hay nhỏ; không đạt thì là mốc tham chiếu.

**Đường chất lượng theo độ trễ.** Một ngưỡng đơn có thể không ràng buộc gì ở ảnh vào nhỏ. Vì vậy kết quả chính về tốc độ là một hình: chất lượng theo độ trễ đo được, với mô hình đề xuất ở hai đến ba cỡ và mọi mốc trên cùng trục. Đường này là kết quả chính; thêm một số đo trên CPU của điện thoại. Một ngân sách chặt cho SR (phần còn lại của khung 33 ms sau bước dò tai) chỉ được dùng nếu thời gian dò tai là số đo thật của một bộ dò cụ thể trên cùng thiết bị; bộ dò điểm mốc ở P6a sẽ được đo cho việc này.

**Ước lượng thô của phản biện (chưa đo):** ở ảnh vào 48×68, SPAN mất vài ms trên Jetson Nano, EDSR-baseline mất vài chục ms, tức quanh ngưỡng. Nếu phép đo T3 xác nhận, có hai hệ quả. Thứ nhất, mọi mô hình hiệu quả đều đạt ngưỡng với khoảng dư lớn, nên ngưỡng chỉ loại mô hình nặng chứ không xếp hạng được nhóm nhẹ. Thứ hai, ngân sách 33 ms cho phép một mô hình lớn hơn SPAN vài lần; mốc phải vượt khi đó là mô hình tinh chỉnh tốt nhất trong ngân sách, không mặc nhiên là SPAN. Vì vậy so sánh chính được làm **ở độ trễ ngang nhau** (chênh không quá 10%), không chỉ ở chỗ "cùng đạt ngưỡng".

## 1.6 Dữ liệu và benchmark

**Dư địa của SR theo cỡ ảnh (đo trên AMI, bicubic ×4, PSNR kênh xám, trung vị):** đáp án 492 px: 41,6 dB; 192 px: 36,4 dB; 144 px: 35,0 dB; 96 px: 32,8 dB.

**Benchmark chính: AMI thu nhỏ.**

- 700 ảnh, 100 người, 492×702 px. Ba cỡ đáp án: cạnh ngắn 96, 144, 192 px; ảnh vào ở ×4 là 24, 36, 48 px, ở ×2 là 48, 72, 96 px. Thêm cỡ 246×351 px chỉ để tính đủ bộ chỉ số cảm nhận của NTIRE.
- **Cross-validation 5 fold theo người, ghép với seed:** fold thứ i dùng seed thứ i. Mỗi người được test đúng một lần, nên tập test là cả 700 ảnh của 100 người thay vì 280 ảnh của 40 người, với cùng chi phí như 5 seed trên một cách chia. Trong mỗi fold: 70 người train, 10 người validation, 20 người test. Phương sai báo cáo gộp cả seed lẫn cách chia, và bài nói rõ điều đó.
- **Chống rò rỉ.** Mọi lựa chọn trong một fold (checkpoint, biên τ của cổng, hai điểm vận hành, tham số của đường trộn và của LDL) chỉ được chỉnh trên 10 người validation của fold đó, không bao giờ trên người test. Vì 10 người là ít, mỗi lựa chọn dùng một lưới giá trị nhỏ ghi trước.
- **Hai fold giữ kín.** Các phép thử quyết định giữ hay bỏ thành phần (T4, T5, T6) và việc dò tốc độ học chỉ chạy trên fold 2, 3, 4. Người test của fold 1 và fold 5 không được dùng cho quyết định nào trước lần chạy cuối. Bài báo cáo riêng kết quả trên hai fold này bên cạnh kết quả gộp.
- Huấn luyện đúng tỉ lệ, trên AMI cộng ảnh thêm từ EarVN1.0 (mục 3.10, điểm 1 và 2).

**Vì sao benchmark chính là AMI mà không phải EarVN1.0 hay AWEx.** Bài đo bằng số đo có tham chiếu (PSNR, SSIM, LPIPS), nên chất lượng của ảnh đáp án quyết định mọi con số. Một bộ ảnh chỉ làm được benchmark chính nếu đáp án của nó sạch, đủ nhiều ở cỡ cần đo, và chia được theo người. So ba bộ theo các yêu cầu đó (số liệu là số đo của nhóm trên dữ liệu, ngày 05/10/2026):

| Yêu cầu | AMI | EarVN1.0 | AWEx |
|---|---|---|---|
| Ảnh gốc đủ lớn để thu nhỏ thành đáp án | Cả 700 ảnh đều 492×702; thu nhỏ 2,5 đến 5 lần là ra ba cỡ đáp án 96, 144, 192 px | Phần lớn ảnh nhỏ. Cạnh ngắn từ 192 px: 1.086 ảnh của 95 người; từ 288 px: 30 ảnh | Cạnh ngắn từ 192 px: 226 ảnh của 142 người; từ 288 px: 61 ảnh |
| Số ảnh làm được đáp án 144 px (ô chính) với biên an toàn 2 lần | 700 | 30 | 61 |
| Mức nén của ảnh gốc | JPEG mức 90; sau khi thu nhỏ 2,5 đến 5 lần, vết nén gần như mất | JPEG khoảng mức 75 ở 90% ảnh; ở biên thu nhỏ thấp, vết nén còn lại trong đáp án và mô hình sẽ học tái tạo nó | Ảnh thu thập từ web; mức nén chưa đo |
| Cỡ và khung hình đồng nhất | Một cỡ, một cách chụp, 7 góc mỗi người | Cỡ và khung hình thay đổi theo từng ảnh | Cỡ và khung hình thay đổi theo từng ảnh |
| Chia fold theo người, mỗi người đủ ảnh | 100 người × 7 ảnh: 5 fold cân, mỗi người test đúng một lần | Ở cỡ dùng được, số ảnh mỗi người rất lệch và 69 trong 164 người không còn ảnh nào | Ở cỡ dùng được, trung bình dưới 2 ảnh mỗi người |

Ba hệ quả. (1) Ô chính của bài (đáp án 144 px, ảnh vào 36 px) chỉ AMI có đủ ảnh: 30 và 61 ảnh không đủ cho một phép so sánh có ngưỡng 0,1 dB. (2) Phép thử ở mục 1.3 cho thấy nén JPEG và khung hình làm lệch số đo của ảnh ngoài thực tế tới vài dB; đo trên đáp án như vậy thì không tách được phần hơn của mô hình khỏi đặc điểm của ảnh. (3) Đường chất lượng theo cỡ ảnh vào (S4) cần cùng một ảnh ở nhiều cỡ; chỉ ảnh gốc lớn và đồng nhất mới làm được.

EarVN1.0 và AWEx vì vậy không bị bỏ mà đổi vai, đúng với thứ chúng có: EarVN1.0 có hơn 11.000 ảnh nhỏ thật, là nguồn duy nhất cho thống kê suy giảm của N2, cho bộ phân loại "mô phỏng hay thật", cho khảo sát người xem trên ảnh nhỏ thật, và cho ảnh huấn luyện thêm; phần ảnh lớn của nó và của AWEx làm test ngoài thực tế ở cỡ 96 px, có biên an toàn. Cái giá của lựa chọn này là AMI chụp trong nhà, chỉ 100 người, trong điều kiện chụp có kiểm soát; bài nói rõ điều đó ở mục Limitations và dùng hai bộ kia để cho thấy kết luận không chỉ đúng trên AMI.

**Ảnh ngoài thực tế.**

| Bộ | Vai trò |
|---|---|
| EarVN1.0 (28.412 ảnh, 164 người) | Người được chia thành các nhóm tách rời. *Nhóm train:* mặc định chỉ dùng ảnh có cạnh ngắn từ 192 px, thu nhỏ 2 lần. Tập rộng hơn (cạnh ngắn từ 128 px, 6.386 ảnh trên toàn bộ, biên 1,33 lần) chỉ được thêm nếu ablation trên AMI cho thấy có lợi, vì ở biên này vết nén mức 75 còn lại trong đáp án và mô hình sẽ học tái tạo nó. *Nhóm test giữ riêng:* đáp án từ ảnh có cạnh ngắn từ 192 px (1.086 ảnh trên toàn bộ), thu nhỏ ít nhất 2 lần. *Ảnh nhỏ nguyên bản* (hơn 11.000 ảnh): ảnh LR thật và nguồn thống kê suy giảm |
| AWEx (4.004 ảnh, 336 người) | Test chéo, không bao giờ dùng để huấn luyện. Ở biên an toàn 2 lần: 226 ảnh cho đáp án 96 px, 61 ảnh cho 144 px. Kết quả báo ở hai mức biên (2 lần và 1,5 lần) |
| CelebAMask-HQ (tùy chọn, chưa tải) | Ảnh tai cắt theo nhãn vùng tai có sẵn, chỉ để train |
| Bộ ảnh tai gắn 55 điểm mốc của Imperial College (2.058 ảnh, 231 người; chưa kiểm tra điều khoản) | Huấn luyện bộ dò điểm mốc cho N1; bộ test thứ tư nếu dùng được |
| DIV2K, LSDIR | Pretrain; hai dòng theo giao thức NTIRE |

**Giới hạn.** AMI chụp trong nhà. Không có cặp ảnh LR thật và HR. Cặp ảnh "front" và "zoom" của AMI đã được kiểm và không dùng được làm cặp SR: độ phóng giữa hai ảnh chỉ 1,18 lần (mục 3.11). Với đáp án từ 144 px trở lên, nguồn sạch duy nhất là AMI; bằng chứng ngoài thực tế có đáp án chủ yếu ở cỡ 96 px.

**Giấy phép.** AMI là CC BY-NC-ND: hình minh họa dùng AMI cần xin phép tác giả; bài công bố script và danh sách file. Điều khoản của EarVN1.0, AWEx và bộ điểm mốc phải kiểm tra trước khi dùng.

## 1.7 Suy giảm

- **Bicubic:** theo NTIRE.
- **Suy giảm tổng quát:** kiểu Real-ESRGAN với tham số mặc định.
- **Suy giảm ước lượng (N2):** mức nén JPEG lấy từ bảng lượng tử trong file của EarVN1.0 (khoảng 75 ở 90% ảnh); mức nhiễu ước lượng từ vùng phẳng; độ mờ là một dải tham số được chỉnh để thống kê độ nét của ảnh mô phỏng khớp với ảnh nhỏ thật. Nhân mờ không ước lượng được tin cậy từ ảnh nhỏ đã nén, và bài nói rõ điều đó. Các tham số này được chốt **một lần, trước mọi lần huấn luyện, chỉ từ EarVN1.0**; chúng không dùng người nào của AMI nên không chỉnh theo fold. Những người của EarVN1.0 dùng để khớp tham số được tách khỏi những người dùng cho bộ phân loại và cho khảo sát người xem.

**Cách chứng minh N2.**

1. *Bộ phân loại "mô phỏng hay thật".* Để bộ phân loại không dựa vào khác biệt nội dung giữa AMI (trong nhà) và EarVN1.0 (ngoài thực tế), ảnh mô phỏng được tạo từ ảnh lớn của chính EarVN1.0, và bộ phân loại chỉ nhìn patch nhỏ. So độ chính xác giữa ba kiểu suy giảm.
2. *Khảo sát người xem trên ảnh nhỏ thật* (mục 1.9).

**Thiết lập chính** là bicubic nếu phép thử ở mục 1.10 cho thấy thứ hạng mô hình không đổi giữa bicubic và suy giảm có nén; là suy giảm ước lượng nếu thứ hạng đảo. Kiểu còn lại thành bảng phụ.

## 1.8 Mô hình so sánh

| Nhóm | Mô hình | Trả lời câu hỏi |
|---|---|---|
| Mốc dưới | Bicubic; bicubic + lọc làm nét | SR có hơn nội suy và hơn một bộ lọc đơn giản không |
| Nhẹ, tối ưu PSNR | SPAN 48 kênh (trọng số chính thức của tác giả; trong bài SPAN bản này tên là SPAN-S), SPAN 28 và 26 kênh (NTIRE 2024), RLFN, SAFMN++, SMFANet, EFDN, sáu mô hình NTIRE ESR 2026 (mục 1.2c). Chưa có trong kho: ECBSR, SPAN 3 block | Có hơn các mô hình hiệu quả tốt nhất không |
| Cỡ vừa | EDSR-baseline (mốc chính; trọng số chính thức, tiền huấn luyện DIV2K), MSRResNet (mốc phụ cùng cỡ; dữ liệu tiền huấn luyện không rõ), SwinIR-light | Ở ảnh nhỏ, dùng mô hình lớn hơn có tốt hơn không |
| Cảm nhận | ESRGAN (bản PSNR và bản GAN); Real-ESRGAN và bản compact; BSRGAN; một mô hình đầu bảng AIM 2025 Efficient Perceptual SR nếu có mã nguồn | Có sắc nét bằng mô hình chuyên về độ sắc không |
| Cùng thân, khác cách huấn luyện | Thân tinh chỉnh bằng L1 (**mốc phải vượt**); bằng GAN; với loss LDL; đường trộn; **mốc nới rộng cho bằng độ trễ của mô hình đề xuất** | Phần hơn đến từ thiết kế hay từ cách huấn luyện có sẵn |
| Trần trên | HAT hoặc SwinIR bản lớn; cổng oracle | Còn cách mức tốt nhất bao xa |

*Cập nhật ngày 06/10/2026 (chỉ làm bảng khớp với kho mô hình; không đổi quyết định nào).* Hai dòng "Nhẹ" và "Cỡ vừa" ghi theo thứ thực có trong `earsr/models/registry.py`. (1) EDSR-baseline nay có trong kho với trọng số chính thức; project tái lập được số tác giả công bố (28,96 so với 28,95 dB trên DIV2K validation, RGB). MSRResNet của KAIR được giữ làm mốc phụ: không nguồn nào nêu dữ liệu tiền huấn luyện của nó, nên nó không dùng cho tuyên bố nào dựa vào cột "dữ liệu tiền huấn luyện". (2) Mốc SPAN 48 kênh nay dùng trọng số chính thức (Set5 ×4 đo được 32,20 dB, đúng bảng 1 của bài SPAN; tiền huấn luyện DF2K). Bài SPAN gọi bản 48 kênh (426 nghìn tham số) là SPAN-S và bản 52 kênh là SPAN; khi viết bài phải dùng đúng tên, và nếu muốn có dòng "SPAN" theo nghĩa của bài gốc thì thêm bản 52 kênh (có sẵn trong cùng file tải về). Trọng số của đội 44 NTIRE 2025 còn trong kho dưới tên `span_ch48_t44`; số sơ bộ ở mục 1.2c là của bộ này.

**Giao thức huấn luyện của mốc.** Mốc dùng giao thức tốt hơn trong hai cách: huấn luyện trên ảnh HR đúng cỡ của ô đang xét, hoặc tỉ lệ ngẫu nhiên; chọn trên validation ở T6 (ii). Chỉ mô hình thuộc nhóm real-time được tinh chỉnh; mô hình nặng và mô hình cảm nhận dùng trọng số công bố. Dòng "trọng số công bố rồi tinh chỉnh" luôn nằm trong bảng chính và mô hình đề xuất phải hơn nó, kể cả khi các biến thể học thẳng trên ảnh tai.

**Công bằng khi tinh chỉnh các mốc.** Mỗi mốc được thử ba mức tốc độ học trên validation của fold 2, rồi dùng mức tốt nhất cho mọi fold; mô hình đề xuất cũng chỉ được ba lần thử như vậy. Bảng kết quả có một cột ghi dữ liệu tiền huấn luyện của từng mô hình (DIV2K, DF2K, LSDIR), vì trọng số công bố không cùng nguồn. Các mốc chủ chốt được nhóm tiền huấn luyện lại bằng cùng công thức với mô hình đề xuất; việc này là bắt buộc (mục 1.8b).

Mỗi mô hình báo ở hai chế độ (trọng số công bố; tinh chỉnh theo cùng quy trình). Nhóm real-time xếp theo số đo ở mục 1.5. Báo PSNR, SSIM, LPIPS, DISTS và bảng hiệu quả cạnh nhau.

## 1.8b Giao thức so sánh công bằng (chốt ở bản 21, trước mọi lần huấn luyện)

**Vấn đề.** Mô hình đề xuất đổi kiến trúc nên không nạp được trọng số công bố; nhóm phải tự tiền huấn luyện nó. Các mốc thì có trọng số do tác giả huấn luyện, trên dữ liệu và với số bước khác nhau (DIV2K, DF2K, LSDIR). So thẳng hai bên là so hai thứ cùng lúc: kiến trúc và công thức tiền huấn luyện. Reviewer sẽ hỏi phần hơn (hoặc phần kém) đến từ đâu.

**Giải pháp: hai nhánh so sánh, mỗi nhánh trả lời một câu hỏi, và mọi tuyên bố ghi rõ nó dựa trên nhánh nào.**

| | Nhánh có kiểm soát (bảng chính của tuyên bố về kiến trúc) | Nhánh trọng số công bố (bảng thực tế) |
|---|---|---|
| Câu hỏi | Ở cùng điều kiện huấn luyện, kiến trúc nào tốt hơn | Mô hình đề xuất có hơn thứ tốt nhất mà người dùng tải về rồi tinh chỉnh được không |
| Mô hình | Mô hình đề xuất; SPAN gốc; SPAN nới rộng cho bằng độ trễ; một mốc khác họ kiến trúc (RLFN). Thêm mốc nếu còn thời gian | Mọi mốc real-time có trọng số công bố |
| Tiền huấn luyện | **Nhóm tự huấn luyện lại tất cả từ khởi tạo ngẫu nhiên**, cùng một script, cùng dữ liệu (DIV2K train), cùng số bước, cỡ lô, cỡ patch, bộ tối ưu, lịch tốc độ học, tăng cường, EMA | Của tác giả; nguồn ghi ở một cột của bảng |
| Tinh chỉnh trên ảnh tai | Giống hệt nhau (xem bảng quy tắc dưới) | Giống hệt nhau |
| Tuyên bố được phép | "Ở cùng dữ liệu và cùng ngân sách huấn luyện, kiến trúc đề xuất hơn SPAN x dB" | "Mô hình đề xuất hơn (hoặc ngang, hoặc kém) mốc tinh chỉnh tốt nhất ở cùng độ trễ" |

Hai nhánh đều nằm trong bài. Nếu mô hình đề xuất thắng ở nhánh có kiểm soát mà không thắng ở nhánh trọng số công bố, bài nói đúng như vậy và nêu lý do có thể kiểm chứng (dữ liệu tiền huấn luyện ít hơn); không được gộp hai nhánh thành một câu "tốt hơn SPAN".

**Quy tắc "chỉ khác một yếu tố", áp cho mọi cặp được so trong cùng một bảng.**

| Yếu tố | Quy tắc | Hiện thực trong code |
|---|---|---|
| Dữ liệu tiền huấn luyện, số bước, công thức | Giống nhau trong nhánh có kiểm soát | `pretrain.py --backbone ... --budget ...`; `make_jobs.py pre` sinh lệnh cho mọi mô hình của nhánh |
| Dữ liệu tinh chỉnh | Cùng tập người train của fold; nếu mô hình đề xuất dùng ảnh thêm từ EarVN1.0 thì mọi mốc trong bảng cũng dùng | `train.py --extra-dir`; số ảnh ghi trong `config.json` |
| Suy giảm lúc huấn luyện | Cùng kiểu với mô hình đề xuất, và cùng kiểu với bộ test của bảng đó | `train.py --degrade` |
| Giao thức tạo mẫu | Cùng giao thức (chọn ở T6 ii) | `train.py --protocol` |
| Số bước, cỡ lô, cỡ patch, EMA, loss | Giống nhau | Mặc định chung của `train.py`; chạy lại cùng mã với siêu tham số khác thì bị chặn |
| Tốc độ học | Mỗi mô hình, kể cả mô hình đề xuất, được đúng ba mức thử trên validation của fold 2; mức được chọn ghi trong phụ lục | `make_jobs.py lr` |
| Chọn checkpoint | Trên 10 người validation của fold; không bao giờ trên test | `trainer.py` |
| Seed và fold | Cùng fold, seed = số fold; so sánh ghép cặp trên cùng ảnh | `runid.py`, `stats/bootstrap.py` |
| Ảnh test | Cùng một bộ ảnh LR đã lưu đĩa cho mọi mô hình | `build_lr.py` |
| Độ trễ | So ở độ trễ đo trên thiết bị chênh không quá 10%; mọi mô hình ở dạng đã gộp nhánh, FP16, cùng cỡ ảnh vào | `match_latency.py`, `deploy_jetson/` |
| Số lần thử thiết kế | Số biến thể đã thử của mô hình đề xuất được báo đủ (năm thân ở T6 iv), kể cả biến thể thua | `results/runs.csv` giữ mọi lần chạy |

**Ba mức bằng chứng cho N5b, theo thứ tự chạy.** (1) *Chọn biến thể:* năm thân, cùng ngân sách rút gọn, fold 2, 3, 4 (T6 iv). (2) *Xác nhận ở ngân sách đủ:* thân thắng, SPAN gốc, SPAN nới rộng cho bằng độ trễ, và RLFN được tiền huấn luyện lại với ngân sách đủ, rồi tinh chỉnh trên cả 5 fold; tuyên bố về kiến trúc dựa trên bước này, vì biến thể được chọn trên fold 2, 3, 4 phải thắng lại trên fold 1 và 5 chưa từng dùng. (3) *Đối chiếu thực tế:* cùng thân đó so với mốc dùng trọng số công bố đã tinh chỉnh.

**Kiểm tra rằng công thức của nhóm không làm yếu mốc.** SPAN do nhóm tự tiền huấn luyện ở ngân sách đủ được chấm trên Set5 và DIV2K validation cạnh SPAN trọng số công bố. Khoảng cách giữa hai bản được báo trong bài. Nếu bản của nhóm kém quá 0,1 dB, nhánh có kiểm soát vẫn hợp lệ (mọi mô hình chịu cùng công thức) nhưng bài phải nói rõ rằng công thức đó chưa đạt mức của tác giả.

**Những điểm reviewer có thể hỏi, và câu trả lời có sẵn trong thiết kế.**

| Câu hỏi | Trả lời |
|---|---|
| "Mốc được huấn luyện kém hơn mô hình của các anh" | Nhánh có kiểm soát: cùng script, cùng dữ liệu, cùng số bước; cấu hình từng lần chạy được công bố |
| "Các anh chỉnh mô hình của mình kỹ hơn" | Ba mức tốc độ học cho mọi mô hình; mọi thứ khác dùng mặc định chung; số biến thể đã thử được báo đủ |
| "Mô hình của các anh lớn hơn nên tốt hơn" | So ở cùng độ trễ đo được; có mốc SPAN nới rộng cho bằng độ trễ |
| "Chọn mô hình trên tập test" | Chọn trên validation; fold 1 và 5 giữ kín tới lần chạy cuối; ngưỡng ghi trước trong `criteria.yaml` |
| "Chênh lệch nằm trong nhiễu" | 5 fold, so sánh ghép cặp theo ảnh, bootstrap theo người, khoảng tin cậy 95%, hiệu chỉnh Holm |
| "Mốc dùng trọng số công bố có lợi thế dữ liệu, hoặc ngược lại" | Hai nhánh tách riêng; cột dữ liệu tiền huấn luyện trong bảng |
| "Mốc cài đặt sai" | Mỗi mô hình trong kho tái lập PSNR công bố trên Set5 trong 0,05 dB (kiểm thử tự động) |
| "Mô hình cảm nhận bị so không công bằng" | Chúng dùng trọng số công bố, học với suy giảm khác; chỉ là mốc tham chiếu, bài không tuyên bố thắng chúng về PSNR |

**Hai chỗ còn hở, phải xử lý trước khi viết bài.**

- *(Đã xử lý ngày 06/10/2026: `span_ch48` nay dùng trọng số chính thức, Set5 ×4 đo được 32,20 dB; xem ghi chú ở mục 1.8. Phần còn lại của mục này giữ để theo dõi lịch sử.)* *Trọng số SPAN 48 kênh đang dùng không phải của tác giả* mà của đội 44 NTIRE 2025 (Set5 ×4: 32,13 dB; bài SPAN công bố 32,20 dB). Ở nhánh trọng số công bố, reviewer có thể nói mốc SPAN yếu hơn bản chính thức. Cần tải trọng số chính thức (Google Drive của tác giả) và thay vào trước lần chạy S2.
- *Chi phí.* Nhánh có kiểm soát thêm 2 lần tiền huấn luyện rút gọn (SPAN nới rộng, RLFN) và nâng số lần tiền huấn luyện đủ từ 2 lên 4. Thời gian mỗi lần chưa đo. Nếu không đủ thời gian, cắt mốc RLFN ở ngân sách đủ trước; không cắt SPAN gốc và SPAN nới rộng, vì thiếu hai mốc đó thì tuyên bố về kiến trúc không đứng được.

## 1.9 Thước đo

- **Theo NTIRE:** bảng hiệu quả (runtime, tham số, FLOPs ở 256×256, điểm theo công thức NTIRE Efficient SR); PSNR, SSIM; LPIPS, DISTS. NIQE, MANIQA, MUSIQ, CLIP-IQA và điểm cảm nhận tổng chỉ tính ở cỡ 246×351 px. Kênh Y, bỏ viền bằng hệ số phóng.
- **Độ lệch điểm mốc (mới):** chạy một bộ dò điểm mốc trên ảnh SR và ảnh HR, đo độ lệch chuẩn hóa. Đây là chỉ số "giữ cấu trúc tai". Bộ dò dùng để đo phải khác bộ dò đã sinh nhãn huấn luyện cho N1 (khác kiến trúc và khác cách chia dữ liệu). **Sàn nhiễu của bộ dò** được đo trước: độ lệch giữa ảnh HR và chính nó sau một nhiễu nhẹ (nhiễu Gauss σ = 2/255, và nén JPEG mức 90). **Điều kiện dải động:** độ lệch của thân tinh chỉnh phải ít nhất gấp 2 lần sàn này; nếu không, thước đo bị coi là không dùng được ở cỡ ảnh đó và N1 được xét bằng bảng kiểm tra gờ.
- **Số bộ dò:** nếu N1 không được theo đuổi, một bộ dò là đủ, vì khi đó bộ dò không sinh nhãn huấn luyện và thước đo không còn vòng lặp. Hai bộ dò và nhãn gắn tay chỉ cần khi làm N1.
- **Nhãn điểm mốc gắn tay (chỉ khi làm N1; quyết định sau điểm kiểm tra 1):** khoảng 100 ảnh test của AMI, 10 đến 15 điểm mốc, gắn hai lượt. Nếu có, độ lệch điểm mốc được đo so với nhãn người và vòng lặp giữa bộ dò sinh nhãn và bộ dò để đo bị cắt hẳn (mục 3.10, điểm 5).
- **Bảng kiểm tra gờ giả và gờ mất:** giữ làm đối chứng độc lập với độ lệch điểm mốc.
- **Khảo sát người xem (bắt buộc):** 15 đến 20 người, so sánh theo cặp, hai phần. *Phần có đáp án* (ảnh test của AMI): hiện đáp án ở giữa, hỏi "ảnh nào giống đáp án hơn"; đây là bằng chứng cho L5, vì người xem không thể thưởng cho chi tiết bịa. *Phần không có đáp án* (ảnh nhỏ thật): hỏi "ảnh nào rõ hơn"; kết quả phần này chỉ được diễn giải là sở thích, không phải độ trung thực. Cần cho N2 và cho tuyên bố "rõ hơn" nói chung, vì ở cỡ ảnh chính không tính được đủ bộ chỉ số cảm nhận.
- **Phạm vi đo:** mỗi số đo được báo ở ba phạm vi: toàn ảnh (bỏ viền theo NTIRE), trong hộp bao vùng tai, và vùng giữa ảnh (bỏ 8 px ảnh vào mỗi cạnh).
- **Độ tin cậy của LPIPS ở ảnh nhỏ:** sau khảo sát, tính tỉ lệ cặp ảnh mà LPIPS và đa số người xem cùng chọn. Nếu tỉ lệ gần 50%, tiêu chí N3 được chấm lại bằng khảo sát người xem. Phiên bản thư viện số đo và mạng nền của LPIPS được ghi cứng trong cấu hình và trong bài.
- **Thiết bị:** theo mục 1.5; thêm mức giảm chất lượng khi lượng tử hóa.
- **Thống kê:** đơn vị là ảnh, gom cụm theo người; bootstrap theo cụm trên cả 5 fold; kiểm định ghép cặp; hiệu chỉnh Holm trên tập so sánh nêu trước. Mọi tiêu chí đạt đều phát biểu bằng khoảng tin cậy của chênh lệch ghép cặp, không bằng độ lệch chuẩn giữa các seed. Kendall τ luôn báo kèm khoảng tin cậy bootstrap.

## 1.10 Thí nghiệm

**Ô chính:** ×4, đáp án 144 px (ảnh vào 36 px), một kiểu suy giảm (chốt theo mục 1.7). Chỉ ô chính có đủ mọi baseline, đủ 5 fold, đủ ablation và hai thân. Các ô khác (cỡ 96 và 192 px; ×2; kiểu suy giảm còn lại; cỡ 246×351) chỉ chạy mô hình đề xuất và bốn baseline chủ chốt.

**Các phép thử quyết định, kèm ngưỡng đề xuất.** Thứ tự chạy (theo phản biện vòng 9): **giai đoạn 1, không cần huấn luyện:** T1, T2, T3, và bước (i) cùng phần phép thử ngữ cảnh của bước (v) ở T6. **Giai đoạn 2, sau điểm kiểm tra 1:** các bước còn lại của T6. **Tùy chọn, sau điểm kiểm tra 2:** T4 (cho N3) và T5 (cho N1), vì kịch bản A+ chỉ cần N5b và N2.

| Mã | Phép thử | Cần huấn luyện | Quyết định |
|---|---|---|---|
| T1 | Rà tài liệu (SR ảnh tai; SR ảnh mặt dẫn hướng bằng điểm mốc; LDL, DeSRA; NTIRE và AIM 2026) và giấy phép | Không | N1 đã có ai làm cho tai chưa; dùng được bộ điểm mốc không |
| T2 | Dựng benchmark AMI; chạy mọi mô hình có trọng số công bố ở ba cỡ, với bicubic và với JPEG mức 75 | Không | Thứ hạng có đảo không. Cặp mô hình nào có khoảng tin cậy của chênh lệch chứa 0 thì tính là hòa (τ-b). Coi là đảo khi có ít nhất một cặp trong nhóm real-time đổi chiều có ý nghĩa ở cả hai điều kiện, hoặc τ-b dưới 0,7 với cận trên khoảng tin cậy dưới 0,9; khi đó G3 thành bắt buộc |
| T3 | Đo độ trễ theo mục 1.5. Kèm theo: xuất một mô hình chưa huấn luyện với từng kiểu đệm (số 0, lặp viền, phản chiếu) và đo trên từng thiết bị | Không | Mô hình nào thuộc nhóm real-time; chọn thân. Kiểu đệm nào không được hỗ trợ hoặc chậm hơn đệm số 0 quá 10% thì bị loại khỏi T6 (iv), không tốn tiền huấn luyện cho nó |
| T4 | Cổng oracle (hai định nghĩa) trên ô chính, fold 2, 3, 4 | Không | Nếu cả hai oracle không vượt đường trộn ít nhất 10% LPIPS ở cùng PSNR thì bỏ N3. Ngưỡng nâng từ 5% lên 10% theo phản biện vòng 7: vượt sát nút thì không đáng giữ, vì N3 kéo theo nhiều thành phần nhất |
| T5 | Đầu phụ cấu trúc, ba nhánh: thân; thân + ảnh thêm không nhãn; thân + đầu phụ. Ảnh vào 24 px, fold 2, 3, 4 (60 người test). Đo sàn nhiễu của bộ dò trước | Có, ngắn | Áp tiêu chí của N1 ở mục 1.2b |
| T6 | Chế độ ảnh nhỏ, bốn bước. **(i) Phép thử ngữ cảnh, không huấn luyện:** cắt một cửa sổ từ ảnh LR rộng hơn, chạy mô hình hai lần (có ngữ cảnh thật xung quanh; chỉ có đệm), so sai số tại cùng điểm ảnh; quét bề rộng ngữ cảnh 0, 4, 8, 16, 24 px. **(ii) Giao thức huấn luyện, ba nhánh:** cắt từ ảnh gốc; ảnh HR đúng cỡ; tỉ lệ ngẫu nhiên. Hai đến ba mốc, fold 2, 3, 4; nhánh đúng cỡ được huấn luyện ở cả ba cỡ cho hai mốc. **(iii) Có cần tiền huấn luyện không:** một thân học từ đầu chỉ trên ảnh tai, được cho đủ số bước để hội tụ, so với cùng thân có trọng số công bố. **(iv) Kiến trúc:** thân tham chiếu, hai biến thể sâu và rộng, hai kiểu đệm khác; cả năm được tiền huấn luyện lại với cùng một ngân sách rút gọn, rồi tinh chỉnh trên fold 2, 3, 4. Ngân sách khởi đầu là một phần năm số bước gốc và chỉ được coi là đủ khi: 10% số bước cuối của thân tham chiếu thêm dưới 0,02 dB, và thứ hạng các biến thể giống nhau ở 50% và 100% ngân sách. **(v) Ngoài ảnh tai, chỉ suy luận:** phép thử ngữ cảnh trên ảnh DIV2K thu nhỏ với trọng số công bố (làm được ngay ở giai đoạn 1); phép thử tương tác trên cùng bộ ảnh, dùng các biến thể đã tiền huấn luyện ở (iv), chưa tinh chỉnh trên tai (giai đoạn 2) | Một phần | Áp tiêu chí N5b; (ii) chọn giao thức. Kết quả (v) quyết định tuyên bố của N5b dừng ở ảnh tai hay mở ra ảnh vào nhỏ nói chung. Nếu (i) không cho thấy hiệu ứng viền thì bỏ phần đệm của (iv). Nếu (iii) cho thấy tiền huấn luyện thêm dưới 0,05 dB thì các biến thể học thẳng trên ảnh tai, không cần tiền huấn luyện |

**Điểm kiểm tra 1, sau giai đoạn 1 (chưa huấn luyện gì).** Ba câu hỏi: SR có dư địa ở ảnh tai nhỏ không (T2); nhóm real-time gồm những mô hình nào và kiểu đệm nào chạy được trên thiết bị (T3); hiệu ứng viền có thật không, trên ảnh tai và trên DIV2K thu nhỏ (T6 i, v). Nếu không có hiệu ứng viền, N5b mất cơ sở và bài chỉ còn kịch bản A; cân nhắc lại với giáo sư trước khi tốn tiền huấn luyện.

**Điểm kiểm tra 2, sau giai đoạn 2.** Chốt giao thức huấn luyện, chốt thân (N5b đạt hay không), chốt kiểu suy giảm chính; quyết định có làm T4, T5 không; rồi viết lại mục 1.4 và Phần 2. Nếu không còn điểm nào trong N1, N2, N3, N5b có triển vọng thì dừng (kịch bản C).

**Chỉnh ngưỡng một lần sau T2, trước mọi lần huấn luyện.** T2 cho biết phương sai của từng số đo theo ảnh và theo người. Từ đó tính mức chênh nhỏ nhất phát hiện được với 60 người (fold 2, 3, 4) và với 100 người. Ngưỡng nào ở mục 1.2b nhỏ hơn mức đó thì được nâng lên; file ngưỡng được commit lại một lần duy nhất ở thời điểm này.

**Ngân sách số lần huấn luyện (đếm sơ bộ, chưa có thời gian mỗi lần).**

| Khối | Số lần huấn luyện |
|---|---|
| T5: 3 nhánh × 3 fold | 9 |
| T6 (ii): nhánh cắt từ ảnh gốc, 3 mốc × 3 fold; nhánh đúng cỡ, 2 mốc × 3 cỡ × 3 fold, cộng mốc thứ ba ở cỡ chính (nhánh tỉ lệ ngẫu nhiên dùng lại ở S2) | 30 |
| T6 (iii): học từ đầu trên ảnh tai × 3 fold | 3 |
| T6 (iv): 5 biến thể × 3 fold | 15 |
| **Tiền huấn luyện (mục 1.8b):** rút gọn cho 5 biến thể, SPAN nới rộng và RLFN; đủ cho thân thắng, SPAN gốc, SPAN nới rộng và RLFN | 7 ngắn + 4 dài, **thời gian chưa đo** |
| Dò tốc độ học: khoảng 8 mốc real-time × 3 mức, 1 fold | 24 (ngắn) |
| S2, ô chính: khoảng 8 mốc real-time × 5 fold | 40 |
| S2, S3: mô hình đề xuất, mốc cùng độ trễ, đối chứng, khoảng 5 ablation, × 5 fold | 40 (60 nếu giữ N3) |
| S3: thân thứ hai (mốc, đề xuất, 1 đối chứng) × 5 fold | 15 |
| S4: ×2 ở một cỡ và kiểu suy giảm còn lại; 5 mô hình × 3 fold. Hai cỡ còn lại ở ×4 không cần huấn luyện thêm nếu giao thức tỉ lệ ngẫu nhiên được chọn | 30 (60 nếu phải dùng mô hình chuyên từng cỡ) |

Tổng khoảng 205 lần tinh chỉnh (225 nếu giữ N3; thêm 30 nếu phải dùng mô hình chuyên từng cỡ), cộng 7 lần tiền huấn luyện rút gọn và 4 lần đủ. Nhánh có kiểm soát (mục 1.8b) thêm khoảng 16 lần tinh chỉnh: SPAN nới rộng và RLFN ở T6 (iv) trên 3 fold (6), và RLFN học lại ở ngân sách đủ trên 5 fold cùng SPAN gốc học lại trên 5 fold (10). Thời gian mỗi lần tinh chỉnh và mỗi lần tiền huấn luyện đo ở lần chạy thử đầu cuối, rồi cộng vào mục 1.11. Thứ tự cắt nếu không đủ thời gian (theo phản biện vòng 7): S4 trước; rồi ablation riêng của thân thứ hai.

**Sau đó.**

| Mã | Nội dung |
|---|---|
| S1 | Pilot trên ô chính, 1 fold: thân tinh chỉnh; thân + LDL; thân bản GAN; mô hình đề xuất với từng thành phần |
| S2 | Ô chính đầy đủ: mọi baseline, 5 fold |
| S3 | Ablation (kể cả "chỉ AMI" so với "AMI + ảnh thêm") và thân thứ hai trên ô chính |
| S4 | Các ô thu nhỏ (3 fold): hai cỡ còn lại ở ×4; ×2 ở một cỡ; kiểu suy giảm còn lại. **Đường chất lượng theo cỡ ảnh vào, từ 16 đến 64 px** (trước là E1, nay thuộc phần lõi vì là bằng chứng trực tiếp cho N5): mô hình đề xuất và ba mốc, chỉ suy luận; dải cỡ đã thấy khi huấn luyện được đánh dấu trên hình |
| S5 | Ảnh ngoài thực tế: nhóm test giữ riêng của EarVN1.0, AWEx, ở hai mức biên an toàn; bộ phân loại "mô phỏng hay thật" |
| S6 | Thiết bị; lượng tử hóa; đường chất lượng theo độ trễ với mô hình đề xuất ở hai đến ba cỡ |
| S7 | Khảo sát người xem; bảng kiểm tra gờ; hai dòng trên DIV2K và LSDIR (ngưỡng 26,90 dB) |

**Thí nghiệm mở rộng, làm sau phần lõi theo thứ tự ưu tiên.** Phần lõi (T1 đến T6, S1 đến S7) đủ để chứng minh L1 đến L7. Các thí nghiệm dưới đây làm bài chắc hơn và trả lời trước câu hỏi của reviewer; chúng được xếp theo mức đáng làm, và làm được bao nhiêu tùy thời gian còn lại.

| Mã | Thí nghiệm | Củng cố |
|---|---|---|
| E1 | (đã chuyển vào phần lõi, S4) | |
| E2 | Độ bền với mức nén: test ở JPEG 50, 75, 90 và không nén | L2, L4 |
| E3 | Tổng quát chéo suy giảm: huấn luyện một kiểu, test kiểu khác | L4 (N2) |
| E4 | Ablation teacher và ablation biên τ của cổng | L4 (N3) |
| E5 | Phân tích ca thất bại: ảnh có tóc che, hoa tai, tư thế nghiêng; ảnh mà mô hình thua mốc | L5; mục Limitations |
| E6 | Hình hóa bản đồ cổng và bản đồ điểm mốc dự đoán | L4, L5 |
| E7 | Phép kiểm tra "không gây hại" với mạng nhận dạng đóng băng (chỉ khi giáo sư đồng ý) | Trả lời reviewer sinh trắc |
| E8 | Bộ dữ liệu thứ tư (bộ 55 điểm mốc hoặc UBEAR) làm test | L7 |
| E9 | Tách phương sai: thêm 3 seed trên một fold cố định (phụ lục của bài) | Thống kê |
| E10 | Phân rã độ trễ theo tầng; bộ nhớ đỉnh; năng lượng nếu đo được | L6 |

Giới hạn thật: với một RTX 3080, không thể chạy mọi baseline ở mọi ô với đủ 5 fold. Vì vậy phần lõi dồn vào ô chính, và danh sách trên là thứ tự cắt bớt nếu thiếu thời gian (cắt từ dưới lên).

**Điểm kiểm tra 3, sau S1.** Áp tiêu chí ở mục 1.2b và 1.4 cho từng thành phần. Không đạt thì thử các phương án sửa (cổng ở mức đặc trưng; một đầu với loss LDL cộng chưng cất; đầu kết cấu chỉ thêm ở dải tần cao), tối đa hai vòng.

## 1.11 Tiến độ ước lượng (một RTX 3080)

| Giai đoạn | Thời gian |
|---|---|
| Giai đoạn 1 (T1, T2, T3, T6 i và v): dựng code P1 đến P5, không huấn luyện | 3 đến 4 tuần |
| Giai đoạn 2 (T6 ii, iii, iv): tinh chỉnh và tiền huấn luyện rút gọn | 3 đến 5 tuần, **chưa gồm thời gian tiền huấn luyện** (chưa đo) |
| Tùy chọn (T4, T5), kể cả bộ dò điểm mốc thứ hai nếu làm N1 | 2 đến 4 tuần |
| S1, kể cả vòng sửa | 4 đến 6 tuần |
| S2, S3 | 7 đến 9 tuần |
| S4, S5 (thêm 3 đến 4 tuần nếu suy giảm ước lượng là thiết lập chính) | 5 đến 8 tuần |
| S6, S7 | 3 tuần |
| Viết | 4 tuần |

Tổng khoảng 7 đến 10 tháng, chưa gồm tiền huấn luyện. Con số thật đo ở lần chạy thử đầu cuối và ở giai đoạn 2.

## 1.12 Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| Không điểm mới nào đạt tiêu chí | Cao | Ba điểm kiểm tra; giai đoạn 1 không tốn huấn luyện; kịch bản B, C |
| Phần hơn của N1 nằm trong nhiễu, hoặc chỉ do có thêm ảnh | Cao | So sánh ghép cặp, bootstrap theo người; sàn nhiễu của bộ dò; nhánh đối chứng không nhãn; ảnh vào 24 px |
| Lựa chọn thành phần bị khớp vào tập test | Trung bình | Hai fold giữ kín; EarVN1.0 và AWEx không dùng cho quyết định nào |
| Thân phải đổi sau khi đo độ trễ | Trung bình | Mục 1.4 để ngỏ; code viết độc lập với thân |
| Dư địa của SR nhỏ ngay cả ở ảnh vào 24 đến 48 px | Cao | Biết ở T2 |
| N2 không chứng minh được vì không có đáp án | Trung bình | Bộ phân loại và khảo sát người xem |
| Bộ điểm mốc không dùng được (điều khoản, độ phân giải) | Trung bình | Biết ở T1; phương án lùi là đường viền từ mặt nạ tai |
| Reviewer sinh trắc hỏi về nhận dạng | Trung bình | Thí nghiệm E7 chuẩn bị sẵn, chạy nếu giáo sư đồng ý |
| AMI trong nhà | Trung bình | Test chéo; Limitations |
| Real-time không phân loại được mô hình ở cỡ ảnh nhỏ | Trung bình | Đường chất lượng theo độ trễ; ngân sách chặt; so ở độ trễ ngang nhau |
| N5b không đạt: không có hiệu ứng viền, hoặc phép thử tương tác không có ý nghĩa | Cao | Bước (i) của T6 không cần huấn luyện; nếu không đạt thì bài giữ cấu trúc cũ |
| Tiền huấn luyện cho các biến thể quá đắt | Cao | T6 (iii) kiểm trước xem có cần tiền huấn luyện không; ngân sách rút gọn dùng chung cho cả nhóm |
| Cắt N3 làm bài mất phần "sắc nét" | Trung bình | Chữ "rõ hơn" dựa vào PSNR, LPIPS, độ lệch điểm mốc, khảo sát có đáp án; mô hình cảm nhận vẫn có trong bảng |
| Ảnh train thêm từ EarVN1.0 làm đáp án mềm đi | Trung bình | Ablation "chỉ AMI" so với "AMI + ảnh thêm" |
| Khoảng 205 lần tinh chỉnh cộng tiền huấn luyện không chạy nổi trên một GPU | Cao | Đo thời gian ở lần chạy thử; thứ tự cắt ghi ở mục 1.10 |
| Trang tải AMI ngừng hoạt động, làm benchmark khó tái lập | Trung bình | Kiểm ở T1 |
| Không được phép dùng ảnh AMI trong hình | Trung bình | Xin phép sớm |
| Khối lượng tính toán | Trung bình | Ô chính và các ô thu nhỏ; hai thân |

## 1.13 Quyết định

**Đã chốt:** bài mô hình theo đề bài; luận điểm là phép so sánh với mô hình đa dụng đã tinh chỉnh; ảnh vào nhỏ, đáp án từ AMI thu nhỏ; cross-validation 5 fold ghép seed; đánh giá theo NTIRE; ×4 và ×2; hai thân; so sánh chính với mô hình real-time theo độ trễ đo được; khảo sát người xem; độ lệch điểm mốc và bảng kiểm tra gờ; không tự chụp; bài span_tiny tách riêng; tạp chí do giáo sư quyết định; làm phần lõi trước, rồi các thí nghiệm mở rộng theo thứ tự ưu tiên; huấn luyện đúng tỉ lệ; EarVN1.0 thêm vào tập train, AWEx không bao giờ train; mốc cùng độ trễ; khảo sát người xem hai phần; ngưỡng chỉnh một lần sau T2.

**Chờ bạn và giáo sư:**

1. Đời máy Jetson Nano và điện thoại dùng để đo, để chốt mục 1.5.
2. Giáo sư có đồng ý chạy phép kiểm tra "không gây hại" (E7) không.
3. Câu hỏi của các vòng trước đã được phản biện trả lời hết; các ngưỡng số hiện hành nằm ở mục 1.2b, 1.5, 1.10 và chỉ còn được chỉnh một lần sau T2.
4. Có gắn nhãn tay khoảng 100 ảnh AMI không (mục 1.9).
5. Có dùng CelebAMask-HQ làm nguồn train thêm không.
6. Xin phép tác giả AMI để dùng ảnh trong hình; kiểm trang tải AMI.

---

# PHẦN 2. KHUNG BẢN THẢO (viết lại ở bản 23)

Phần này là bản thiết kế của bài báo: kể câu chuyện gì, nêu vấn đề gì, chứng minh luận điểm nào bằng bảng và hình nào, và mỗi con số lấy từ file kết quả nào. Sau khi chạy xong, nhóm đọc kết quả, điền các chỗ `[X]`, chọn câu theo nhánh "đạt" hoặc "không đạt" đã viết sẵn, rồi viết bình luận. Phần chữ tiếng Anh là khung câu; *[ghi chú]* là hướng dẫn điền. Phần này thay cho bảng luận điểm ở mục 1.2 ở chỗ nào hai bên khác nhau.

## 2.0 Kết quả đã có (bản 25, ngày 06/10/2026)

Mọi số ở mục này là số đo thật: 700 ảnh AMI của 100 người, ×4, chạy trên RTX 3080, chấm ở FP32 đầy đủ (TF32 tắt). Khoảng trong ngoặc vuông là khoảng tin cậy 95%, bootstrap theo người. "16 mô hình" là các mô hình tối ưu PSNR khác nhau có trọng số công bố: 12 mô hình nhẹ (trong đó có baseline và năm mô hình đầu bảng của NTIRE 2026), 3 mô hình cỡ vừa, và RRDB. Kho có 18 tên, nhưng hai tên không được đếm trong bài: `span26` trùng từng byte với `span_ch28` (baseline của NTIRE 2026 chính là SPAN 28 kênh của 2024), và `span_ch48_t44` là bộ trọng số thứ hai của SPAN 48 kênh. Các file trong `results/t2_summary/` hiện còn tính cả hai tên đó; số thứ hạng dưới đây đã được tính lại không có chúng.

**Dữ kiện từ giai đoạn 1.**

| Dữ kiện | Ảnh vào 24 px | Ảnh vào 36 px (ô chính) | Ảnh vào 48 px | Nguồn |
|---|---|---|---|---|
| Bicubic, PSNR-Y (suy giảm bicubic) | 34,30 dB | 36,73 dB | 38,13 dB | `results/t2_summary/quality.csv` |
| SPAN 48 kênh hơn bicubic | +2,73 [2,63; 2,84] | +2,32 [2,21; 2,42] | +1,84 [1,75; 1,94] | như trên, cột `gain` |
| Dải phần hơn của 16 mô hình | +2,39 đến +3,22 | +2,03 đến +2,55 | +1,63 đến +1,99 | như trên |
| JPEG 75: 16 mô hình so với bicubic, PSNR-Y | −0,29 đến −0,49 | −0,40 đến −0,53 | −0,40 đến −0,50 | như trên |
| JPEG 75: số mô hình **kém** bicubic có ý nghĩa trên PSNR, SSIM, MS-SSIM, GMSD, LR-PSNR | 16/16 ở cả năm số đo | 16/16 | 16/16 | ghép cặp từ `results/t2/*.csv` |
| JPEG 75: số mô hình **hơn** bicubic có ý nghĩa trên LPIPS và trên DISTS | 16/16 ở cả hai | 16/16 | 16/16 | như trên |
| τ-b giữa thứ hạng ở bicubic và ở JPEG 75 (15 mô hình nhẹ và vừa khác nhau) | 0,08 | 0,09 | −0,06 | tính lại bằng `earsr.report.t2.rank_agreement`; file `ranking.json` (17 tên) ghi 0,13; 0,13; −0,04 |
| Số cặp đổi chiều có ý nghĩa (trên 105 cặp) | 44 | 42 | 48 | như trên |
| RRDB trừ SPAN 48 kênh (39 lần tham số), suy giảm bicubic | +0,49 [0,46; 0,51] | +0,23 [0,22; 0,25] | +0,14 [0,13; 0,15] | ghép cặp từ `results/t2/*.csv` |
| RRDB trừ SPAN 48 kênh, JPEG 75 | −0,05 [−0,06; −0,04] | −0,04 [−0,05; −0,04] | −0,03 [−0,03; −0,02] | như trên |
| RRDB trừ SPAN 26 kênh (128 lần tham số), suy giảm bicubic | +0,82 | +0,52 | +0,36 | như trên |
| Mất khi thiếu ngữ cảnh: SPAN 48 kênh so với bicubic | 0,274 so với 0,214 dB | 0,129 so với 0,096 dB | | `results/t6_summary/context.csv` |

Thêm: ở ×2 (ảnh vào 72 px) SwinIR-light hơn bicubic 2,41 dB khi suy giảm bicubic và kém 0,53 dB khi có JPEG 75. SPAN 48 kênh mất 5,1 dB khi ảnh vào bị nén (39,04 xuống 33,93 dB ở ô chính). Mức chênh nhỏ nhất phát hiện được ở ô chính là 0,013 dB (60 người). Các số ghép cặp ghi "ghép cặp từ `results/t2/*.csv`" được tính trực tiếp từ file theo ảnh; khi dựng bảng cho bài phải sinh lại bằng `compare_table.py`.

**Dữ kiện từ phép thử đầu của N2** (`results/n2_realism.json`; bộ phân loại học 3.000 bước, 3 seed; chấm trên ảnh của 4 người giữ riêng: 218 ảnh nhỏ thật, 512 ảnh mô phỏng). Càng gần 50% thì ảnh mô phỏng càng khó phân biệt với ảnh nhỏ thật của EarVN1.0.

| Kiểu mô phỏng | Độ chính xác cân bằng |
|---|---|
| Bicubic | 80,1% [77,3; 86,6] |
| Tổng quát (kiểu Real-ESRGAN) | 78,0% [76,8; 80,0] |
| Ước lượng từ ảnh tai (N2) | 58,5% [56,8; 59,9] |
| Bicubic rồi JPEG 75 | 56,7% [55,7; 60,5] |

Suy giảm ước lượng gần 50% hơn suy giảm tổng quát 19,5 điểm [17,1; 22,8] và hơn bicubic 21,6 điểm [19,3; 29,9]; so với "bicubic rồi JPEG 75" thì không thấy khác biệt (−1,8 điểm [−2,7; +3,7]). Tham số đã chốt (`configs/degrade_estimated.json`, từ 621 ảnh nhỏ thật của 16 người): độ mờ σ từ 0,2 đến 0,6 (KS 0,106; "không mờ" cho KS 0,178); nhiễu σ từ 0,85 đến 6,3; JPEG mức 75 ở 52% ảnh và mức 93 ở 44% ảnh.

**Dữ kiện bổ sung, tối 06/10/2026: độ tổng quát của phát hiện, và khối `n2` chạy lại.**

*Độ tổng quát ngoài AMI* (`results/t2_earvn_summary/`, `results/t2_awex_summary/`, `results/t2_summary/`; 16 mô hình khác nhau; chỉ chấm, không huấn luyện). Mức hơn bicubic về PSNR-Y, từ mô hình thấp nhất đến cao nhất; trong ngoặc là số mô hình **kém** bicubic có ý nghĩa:

| Bộ ảnh (ảnh vào) | Sạch | JPEG 60 | JPEG 75 | JPEG 85 | JPEG 93 | Suy giảm ước lượng |
|---|---|---|---|---|---|---|
| AMI, 700 ảnh (24 px) | +2,39 đến +3,22 | −0,53 đến −0,37 (16) | −0,49 đến −0,29 (16) | −0,28 đến −0,06 (16) | +0,37 đến +0,67 (0) | chưa chấm |
| AMI, 700 ảnh (36 px) | +2,03 đến +2,55 | −0,62 đến −0,50 (16) | −0,53 đến −0,40 (16) | −0,36 đến −0,22 (16) | +0,18 đến +0,35 (0) | chưa chấm |
| EarVN1.0 nhóm test, 158 ảnh (24 px) | +3,47 đến +4,45 | | −0,22 đến +0,02 (6) | | +1,42 đến +1,87 (0) | +0,43 đến +0,61 (0) |
| AWEx, 217 ảnh (24 px) | +3,04 đến +3,86 | | −0,38 đến −0,13 (16) | | +1,01 đến +1,33 (0) | +0,13 đến +0,31 (0) |
| AWEx, 60 ảnh (36 px) | +2,18 đến +2,65 | | −0,43 đến −0,24 (16) | | +0,60 đến +0,87 (0) | −0,25 đến −0,09 (5) |

Ở mọi ô, cả 16 mô hình đều tốt hơn bicubic về LPIPS và DISTS. Thứ hạng đảo ở cả ba bộ (τ-b giữa ảnh sạch và JPEG 75: 0,08 đến 0,16; giữa ảnh sạch và suy giảm ước lượng: 0,16 đến 0,20). Ba điều rút ra: (1) hiện tượng không riêng của AMI; (2) nó phụ thuộc mức nén: mô hình kém bicubic từ mức 85 trở xuống và lại hơn bicubic ở mức 93; (3) với suy giảm đo từ ảnh thật (trộn hai mức), lợi thế về PSNR còn 0,1 đến 0,6 dB so với 2,2 đến 4,5 dB trên ảnh sạch, tức mất khoảng 85 đến 95%.

*Khối `n2` chạy lại* (`results/n2/`; 10 lần huấn luyện trên fold 2, một seed; SPAN 48 kênh và DISP; có tăng cường độ sáng, 8 lần có thêm ảnh EarVN1.0 nhóm train; cả 10 qua phép thử ảnh sáng). PSNR-Y trên **người test của fold 2** của AMI (140 ảnh, 20 người, ảnh vào 36 px), hàng là kiểu suy giảm lúc huấn luyện, cột là ảnh vào lúc chấm; cột cuối là EarVN1.0 nhóm test (158 ảnh, 24 px, ảnh vào sạch):

| SPAN 48 kênh | Sạch | JPEG 75 | Ước lượng | Tổng quát | EarVN, sạch |
|---|---|---|---|---|---|
| Bicubic (nội suy) | 37,07 | 34,62 | 34,61 | 29,07 | 32,33 |
| Trọng số công bố | 39,40 | 34,18 | 34,18 | 27,50 | 36,39 |
| Tinh chỉnh, sạch | 39,85 | 34,14 | 34,12 | 27,25 | 36,46 |
| Tinh chỉnh, JPEG 75 | 37,27 | 36,06 | 35,97 | 29,17 | 33,67 |
| Tinh chỉnh, ước lượng (N2) | 38,22 | 36,00 | 36,09 | 29,46 | 34,79 |
| Tinh chỉnh, tổng quát | 35,92 | 35,00 | 35,00 | 31,71 | 31,94 |

Chênh lệch ghép cặp (bootstrap theo người; mọi khoảng tin cậy nêu ở đây không chứa 0):

- *Sửa được chỗ lệch.* Trên ảnh nén, SPAN học với suy giảm ước lượng hơn bicubic 1,38 dB (JPEG 75) và 1,48 dB (ước lượng), LPIPS 0,245 so với 0,391; trọng số công bố thì kém bicubic 0,44 dB. Sau khi tinh chỉnh đúng suy giảm, mô hình hơn bicubic trên **cả** số đo độ trung thực lẫn số đo cảm nhận. DISP: +1,10 và +1,20 dB.
- *Ước lượng hơn tổng quát.* Trên ảnh nén, mô hình học với suy giảm ước lượng hơn mô hình học với suy giảm tổng quát 1,00 đến 1,09 dB (SPAN) và 1,22 đến 1,36 dB (DISP); LPIPS thấp hơn 0,036 đến 0,048. DISP học với suy giảm tổng quát còn kém bicubic 0,12 đến 0,16 dB. Trên ảnh sạch của EarVN, mô hình học với suy giảm tổng quát kém cả bicubic (31,94 so với 32,33).
- *Ước lượng so với JPEG 75 cố định.* Trên ảnh nén hai mô hình ngang nhau: mỗi mô hình hơn khoảng 0,05 đến 0,12 dB trên đúng kiểu ảnh của nó, LPIPS như nhau. Khác biệt nằm ở ảnh nén nhẹ hoặc sạch: mô hình học với suy giảm ước lượng hơn 0,95 dB trên ảnh sạch của AMI và 1,12 dB trên EarVN (DISP: 0,45 và 0,54). Tức phân bố mức nén đo từ dữ liệu cho mô hình **bền hơn theo mức nén**, không cho mô hình tốt hơn ở mức 75. Cần chấm thêm ở JPEG 93 để khẳng định.
- *Ảnh ngoài thực tế trong tập huấn luyện là bắt buộc.* Tinh chỉnh với ảnh sạch, có ảnh EarVN: hơn trọng số công bố trên EarVN 0,07 dB (SPAN) và 0,26 dB (DISP). Chỉ AMI (có tăng cường độ sáng): kém trọng số công bố 0,57 và 0,42 dB. Ảnh EarVN thêm 0,64 và 0,67 dB trên EarVN, không tốn gì trên AMI (39,85 so với 39,88).
- *Không có mô hình nào tốt nhất ở mọi kiểu ảnh vào.* Mô hình học với ảnh sạch tốt nhất trên ảnh sạch và kém bicubic trên ảnh nén; mô hình học với suy giảm ước lượng hơn bicubic ở cả bốn cột.

Giới hạn của các số trong khối `n2`: một fold, một seed, và chưa chấm các mô hình này trên EarVN1.0 và AWEx với ảnh vào có nén. Phép thử quyết định theo kế hoạch phải chạy trên fold 2, 3, 4.

**Bảng theo dõi luận điểm** (yêu cầu ở mục 2.7, bước 1; cập nhật tối 06/10).

| # | Trạng thái | Căn cứ | Hệ quả cho bài |
|---|---|---|---|
| C1 | **Đạt** | Mọi mô hình hơn bicubic 1,6 đến 3,2 dB; phần hơn giảm đều khi ảnh vào lớn lên | Giữ nguyên. Hình 2 (đường theo cỡ ảnh vào) còn chờ `run_size_sweep.py` |
| C2 | **Đạt trên số đo độ trung thực, từ mức nén 85 trở xuống, trên cả ba bộ ảnh; không đạt trên số đo cảm nhận, và đảo chiều ở mức nén 93** | AMI và AWEx: 16/16 mô hình kém bicubic ở JPEG 75; EarVN1.0: xấp xỉ hòa (6/16 kém có ý nghĩa, không mô hình nào hơn). Ở JPEG 93 cả 16 mô hình hơn bicubic trên cả ba bộ. Với suy giảm ước lượng, lợi thế còn 0,1 đến 0,6 dB. Thứ hạng đảo ở mọi bộ | Phát biểu lại: "lợi thế về độ trung thực của SR so với nội suy gần như biến mất trên ảnh tai nhỏ đã nén, và thành âm khi nén từ khoảng mức 85 trở xuống". Bài phải có hình PSNR theo mức nén. Không dùng chữ "fails" |
| C3 | **Đạt khi suy giảm lệch; chỉ đạt ở ô chính khi suy giảm khớp** | Dưới JPEG 75, mô hình lớn gấp 39 lần không hơn (−0,04 dB). Dưới bicubic nó hơn 0,23 dB ở ô chính (dưới ngưỡng bác bỏ 0,5 dB) nhưng 0,49 dB ở ảnh vào 24 px, và 0,82 dB so với mô hình 0,13 triệu tham số | Viết lại thành hai vế (mục 2.2). Quan sát "ảnh càng nhỏ, dung lượng càng có giá trị" đưa vào mục 3.5 của bài |
| C4 | **Có số sơ bộ (fold 2)** | Tinh chỉnh với ảnh sạch hơn trọng số công bố 0,45 dB trên AMI (39,85 so với 39,40) và 0,07 dB trên EarVN, **chỉ khi** tập huấn luyện có ảnh ngoài thực tế và tăng cường độ sáng; chỉ AMI thì kém trọng số công bố trên EarVN 0,57 dB, và không tăng cường thì mô hình hỏng trên ảnh sáng | Mốc "tinh chỉnh" trong bài phải là bản có ảnh ngoài thực tế. Sự hỏng khi tinh chỉnh chỉ trên AMI là một dữ kiện cho mục 5.7 |
| C5 | **Tiêu chí (a) đạt; bằng chứng huấn luyện ủng hộ (fold 2); khảo sát người xem chưa làm** | Bộ phân loại: ước lượng 58,5%, tổng quát 78,0%. Huấn luyện: mô hình học với suy giảm ước lượng hơn mô hình học với suy giảm tổng quát 1,0 đến 1,4 dB trên ảnh nén, và hơn bicubic 1,1 đến 1,5 dB. So với JPEG 75 cố định: ngang trên ảnh nén, hơn 0,5 đến 1,1 dB trên ảnh sạch | Đóng góp phát biểu là: suy giảm tổng quát không hợp với miền này; phân bố mức nén đo từ dữ liệu cho mô hình bền theo mức nén. Mọi bảng của N2 có mốc JPEG 75 cố định. Còn nợ: fold 3 và 4, chấm trên EarVN1.0 và AWEx với ảnh nén, JPEG 93, khảo sát người xem |
| C6 | **Cơ sở (a) đạt về hình thức; triển vọng thấp** | Mọi mô hình mất 0,10 đến 0,41 dB khi thiếu ngữ cảnh, nhưng trên ảnh tai bicubic cũng mất 0,10 đến 0,21 dB; phần riêng của SPAN 48 kênh chỉ 0,03 đến 0,06 dB, dưới ngưỡng 0,10 dB của tiêu chí (b). Trên DIV2K phần riêng của mạng rõ hơn (0,10 đến 0,17 so với 0,03 dB) | Chỉ chạy phép thử `pad` rẻ; không tiền huấn luyện trừ khi `pad` cho thấy phần hơn. Nhiều khả năng thành mục phân tích (nhánh "không đạt" ở mục 2.6) |
| C7 | Chưa có | Cần mô hình đã huấn luyện và khảo sát người xem | Lưu ý từ C2: mô hình có sẵn đã hơn bicubic về LPIPS, DISTS dù kém về độ trung thực; bảng 8 phải báo cả hai nhóm số đo |
| C8 | Chưa có | Chưa có Jetson. Trên RTX 3080 mọi mô hình nhẹ chạy dưới 2 ms; đệm lặp viền và phản chiếu chậm hơn đệm số 0 khoảng 13 đến 15% (chỉ để tham khảo) | |
| C9 | Chưa có | | |

**Việc phải làm trước khi dựng bảng cho bài:** loại `span26` (trùng `span_ch28`) khỏi mọi bảng và mọi phép tính thứ hạng, quyết định `span_ch48_t44` vào bảng như một dòng phụ hay bỏ, rồi chạy lại `summarize_t2.py` để `ranking.json` và `quality.csv` khớp với số ở mục này.

**Kịch bản hiện tại** (theo bảng ở mục 2.2; cập nhật tối 06/10): "Chỉ suy giảm", với bằng chứng huấn luyện đã ủng hộ trên một fold. Kịch bản "Đủ" cần C6, mà triển vọng của C6 thấp. Bài theo kịch bản này không có đóng góp kiến trúc; xem ba hướng ở `TODO.md` mục 2b.

## 2.1 Câu chuyện

**Một câu (bản 25, theo số thật).** Trên ảnh tai nhỏ đã nén, SR hiệu quả có sẵn kém nội suy bicubic về độ trung thực và thứ hạng của chúng đảo; nguyên nhân là lệch suy giảm chứ không phải kiến trúc; và thứ sửa được chỗ lệch đó là một suy giảm đơn giản đo từ chính ảnh tai, không phải bộ suy giảm tổng quát, vốn không thật hơn bicubic. *(Vế cuối mới có bằng chứng từ bộ phân loại; phần "mô hình học với nó cho ảnh tốt hơn" còn chờ kết quả huấn luyện và khảo sát người xem.)*

*Câu của bản 23, giữ để đối chiếu:* SR hiệu quả hiện nay không dùng được cho ảnh tai nhỏ thật, và lý do là suy giảm và cỡ ảnh vào chứ không phải kiến trúc; sửa đúng hai chỗ đó thì một mô hình real-time cho ảnh tai rõ hơn mô hình đa dụng đã tinh chỉnh. Hai chỗ không còn đứng: "không dùng được" (về LPIPS và DISTS các mô hình vẫn hơn bicubic) và "cỡ ảnh vào" như một chỗ sửa (dư địa của N5b quá nhỏ).

**Năm bước của câu chuyện.**

1. *Bối cảnh.* Ảnh tai chụp ngoài thực tế thì nhỏ (trung vị cạnh ngắn của EarVN1.0 là 77 px) và đã qua nén. Muốn nhìn rõ hơn thì phải phóng lên, và phải nhanh.
2. *Khoảng trống.* SR hiệu quả được thiết kế và xếp hạng trên ảnh tự nhiên lớn, suy giảm bicubic (giao thức NTIRE). Chưa ai kiểm xem thứ hạng đó có còn đúng ở ảnh vào vài chục pixel đã nén hay không.
3. *Phát hiện.* Không còn đúng. Trên ảnh tai nhỏ có nén JPEG mức 75, cả 16 mô hình (kể cả baseline và năm mô hình đầu bảng NTIRE 2026) kém nội suy bicubic 0,3 đến 0,5 dB và kém trên mọi số đo độ trung thực, ở cả ba cỡ ảnh vào; thứ hạng giữa chúng gần như không liên quan tới thứ hạng ở suy giảm bicubic (τ-b 0,09). Về LPIPS và DISTS thì chúng vẫn hơn bicubic: ảnh sắc hơn nhưng sai hơn. Khi đó mô hình lớn gấp 39 lần cũng không hơn. Nút thắt là suy giảm, không phải kiến trúc.
4. *Giải pháp.* Sửa ở đúng nơi lệch: huấn luyện với suy giảm đo từ ảnh tai thật (mức nén lấy từ bảng lượng tử của file, nhiễu và độ mờ nhẹ). Bộ suy giảm tổng quát kiểu Real-ESRGAN không dùng được ở đây: ảnh nó tạo ra dễ phân biệt với ảnh nhỏ thật ngang bicubic thường (78% so với 80%), trong khi suy giảm đo từ dữ liệu gần mức ngẫu nhiên (58,5%). Áp trên hai thân mới nhất, khác họ, ở cùng độ trễ. Phần chỉnh thân cho ảnh vào nhỏ (N5b) chỉ vào bài nếu phép thử `pad` đạt.
5. *Kết quả.* Rõ hơn mốc "mô hình đa dụng đã tinh chỉnh" theo số đo, theo cấu trúc tai, theo mắt người; vẫn real-time trên thiết bị nhúng; đúng trên hai thân và trên ảnh ngoài thực tế.

**Câu hỏi nghiên cứu ghi trong bài.**

- RQ1. Thứ hạng của SR hiệu quả trên benchmark chuẩn có giữ được ở ảnh tai nhỏ, có nén không?
- RQ2. Phần kém đến từ kiến trúc, từ suy giảm, hay từ cỡ ảnh vào?
- RQ3. Sửa suy giảm và cỡ ảnh vào thêm được bao nhiêu so với việc chỉ tinh chỉnh một mô hình đa dụng, ở cùng độ trễ?

## 2.2 Chuỗi luận điểm và bằng chứng

Mỗi luận điểm ứng với một bảng hoặc hình, một file kết quả, và một điều kiện bác bỏ. Cột cuối ghi bài đổi thế nào nếu luận điểm không đứng.

| # | Luận điểm | Bằng chứng trong bài | Lấy số từ | Bị bác nếu | Nếu bị bác |
|---|---|---|---|---|---|
| C1 | Ở ảnh tai nhỏ, SR còn dư địa so với nội suy, và dư địa giảm theo cỡ ảnh vào | Bảng 2; Hình 2 (đường chất lượng theo cỡ ảnh vào) | `results/t2_summary/quality.csv` (cột `gain`); `results/size_sweep/summary.csv` | Mọi mô hình gần bằng bicubic ở cả ba cỡ | Không có bài toán; dừng |
| C2 | Dưới suy giảm có nén, SR hiệu quả có trọng số công bố kém bicubic **về độ trung thực** (PSNR, SSIM, MS-SSIM, GMSD, LR-PSNR) và thứ hạng đảo; về LPIPS, DISTS chúng vẫn hơn bicubic, và bài phải nói rõ điều đó | Bảng 3 (hai cột suy giảm cạnh nhau); số τ-b | `results/t2_summary/quality.csv`, `ranking.json` | Mô hình vẫn hơn bicubic và thứ hạng giữ nguyên | Bỏ chữ "thất bại"; bài còn "thiết kế riêng hơn tinh chỉnh", yếu hơn |
| C3 | Kiến trúc không phải nút thắt **khi suy giảm lệch**: dưới nén, mô hình lớn gấp 39 lần không hơn. Khi suy giảm khớp (bicubic), dung lượng có giá trị và giá trị đó tăng khi ảnh vào nhỏ đi (0,14, 0,23, 0,49 dB ở 48, 36, 24 px) | Bảng 3; Hình 3 (PSNR theo số tham số hoặc độ trễ) | `results/t2_summary/quality.csv`; `results/latency_jetson.csv` | Mô hình lớn hơn hẳn mô hình nhỏ (trên 0,5 dB) | Bỏ luận điểm; thêm mốc lớn vào so sánh chính |
| C4 | Tinh chỉnh trên ảnh tai là mốc mạnh và là mốc đúng để so | Bảng 4 (nguyên bản so với tinh chỉnh) | `runs/S2_*/test_*.csv`; `results/t2/` | (cách đặt mốc; không cần bác) | |
| C5 | Với ảnh tai nhỏ, một suy giảm đơn giản đo từ dữ liệu (N2) gần ảnh thật hơn hẳn suy giảm tổng quát, vốn không thật hơn bicubic; và mô hình học với nó cho ảnh tốt hơn trên ảnh nhỏ thật. Bài không tuyên bố phép ước lượng hơn "bicubic rồi JPEG ở một mức cố định": bộ phân loại không thấy khác biệt, và mốc đó có mặt trong mọi bảng của N2 | Bảng 5 (bộ phân loại); Bảng 6 (ảnh ngoài thực tế); Bảng 9 (người xem, phần không đáp án) | `results/n2_realism.json`; `results/s5/*.csv`; `results/viewer/noref/analysis.csv` | Bộ phân loại không gần 50% hơn; hoặc người xem không chọn trên 50% | Bài mất giải pháp chính; còn phát hiện và benchmark; bàn lại với giáo sư |
| C6 | Chỉnh thân cho ảnh vào nhỏ thêm được phần hơn, và phần hơn đó lớn hơn ở ảnh nhỏ (N5b) | Hình 4 (sai số theo bề rộng ngữ cảnh); Bảng 7 (biến thể, nhánh có kiểm soát); phép thử tương tác | `results/t6_summary/context.csv`; `results/criteria/n5b-gain.json`, `n5b-interaction.json` | Dưới 0,1 dB hoặc khoảng tin cậy chứa 0; thiếu tương tác | N5b thành một mục phân tích (hiệu ứng viền có thật, sửa chưa được); không nằm trong danh sách đóng góp |
| C7 | Ảnh rõ hơn thật, không do bịa cấu trúc | Bảng 8 (gờ giả, gờ mất, LR-PSNR, GMSD, PSNR của gradient; độ lệch điểm mốc nếu có); Bảng 9 (người xem, phần có đáp án) | cột `ridge_*`, `lr_psnr_y`, `gmsd`, `grad_psnr` trong các file theo ảnh; `results/viewer/ref/analysis.csv` | Gờ giả tăng, LR-PSNR tụt, hoặc người xem không chọn trên 50% | Không dùng chữ "clearer"; chỉ tuyên bố theo số đo |
| C8 | Real-time trên thiết bị nhúng, và hơn mốc dọc theo đường chất lượng theo độ trễ | Bảng 10; Hình 5 | `results/latency_jetson.csv`; `results/criteria/` | Không đạt 33 ms, hoặc nằm dưới đường của các mốc | Hạ xuống "near real-time" nếu đạt 66 ms; nếu không, bỏ tuyên bố |
| C9 | Kết quả không do riêng một thân hay một bộ dữ liệu | Bảng 7 (thân thứ hai); Bảng 6 (EarVN1.0, AWEx); fold 1 và 5 báo riêng | `runs/` của thân thứ hai; `results/s5/` | Chỉ có trên một thân, hoặc chỉ trên AMI | Thu hẹp tuyên bố về đúng thân và bộ dữ liệu đó; nêu ở Limitations |

C1, C2, C3 và nửa đầu của C6 có ngay sau giai đoạn 1, không cần huấn luyện. C5 có sau phép thử N2 sớm. Vì vậy sau hai bước rẻ đầu tiên đã biết bài thuộc kịch bản nào.

**Ba kịch bản của bài, tùy C5 và C6.**

| Kịch bản | Điều kiện | Tên gọi mô hình trong bài | Trọng tâm |
|---|---|---|---|
| Đủ | C5 và C6 đạt | Một mô hình đề xuất, hai thành phần | Phát hiện + hai chỗ sửa |
| Chỉ suy giảm | C5 đạt, C6 không | Mô hình đề xuất = thân mới nhất + suy giảm ước lượng; hiệu ứng viền là mục phân tích | Phát hiện + N2 |
| Phân tích | C5 không đạt | Không có "mô hình đề xuất"; bài là benchmark và phân tích | C1 đến C4, C8; bàn lại tạp chí |

## 2.3 Title, Abstract

**Title** (chốt sau điểm kiểm tra 2). *Bản 25:* số liệu hợp với tiêu đề 2 nhất; tiêu đề 1 chỉ dùng được nếu thêm phạm vi cho chữ "Fails" (về độ trung thực), vì trên LPIPS và DISTS các mô hình không thua bicubic:

1. *Why Efficient Super-Resolution Fails on Small Ear Images, and a Real-Time Fix* (kịch bản đủ hoặc chỉ suy giảm)
2. *Real-Time Super-Resolution for Low-Resolution Ear Images: Degradation Matters More Than Architecture*
3. *A Benchmark and Analysis of Efficient Super-Resolution on Small Ear Images* (kịch bản phân tích)

**Abstract** (dưới 220 từ; mỗi câu một việc):

1. *Bối cảnh:* Ear images captured in unconstrained settings are small and compressed, whereas efficient super-resolution (SR) networks are designed and ranked on large natural images under bicubic downsampling.
2. *Phát hiện:* On a benchmark of ear images with clean ground truth at three scales, we find that all 16 PSNR-oriented SR networks with public weights, including the NTIRE 2026 baseline and five of its top entries, fall below bicubic interpolation by 0.3 to 0.5 dB once the input is JPEG-compressed (quality 75), on every fidelity measure and at every input size, while remaining ahead on LPIPS and DISTS; that their ranking is nearly unrelated to the ranking under bicubic degradation (Kendall τ-b = 0.09); and that a 39× larger network is then no better (−0.04 dB). *[số ở mục 2.0; viết lại nếu bảng cuối khác]*
3. *Chẩn đoán:* The bottleneck is therefore the degradation, not the architecture; a generic real-world degradation pipeline does not close the gap, since a classifier separates its outputs from real small ear images as easily as plain bicubic downsampling (78% vs 80%).
4. *Giải pháp:* We instead measure the degradation from 621 real small ear images (compression level from the files' quantisation tables, mild noise and blur), which brings the classifier close to chance (58.5%) {C6, chỉ khi đạt: , and configure the backbone for inputs smaller than the receptive field}.
5. *Kết quả chính:* Applied to two recent backbones of different families and compared at matched measured latency ([X] ms on a Jetson Nano), the resulting model improves [metric] from [X] to [X] over the same backbones fine-tuned on ear images, on [n] subjects with subject-wise cross-validation.
6. *Rõ hơn thật:* False ridges do not increase ([X] vs [X]), consistency with the input is preserved, and [n] viewers prefer its outputs in [X]% of pairs.
7. *Ngoài thực tế và tài nguyên:* The gains hold on two in-the-wild ear datasets; the benchmark, splits and code are released.

## 2.4 Dàn ý từng mục

### 1. Introduction (khoảng 1 trang)

| Đoạn | Việc của đoạn | Câu mở (khung) | Số cần điền |
|---|---|---|---|
| P1 | Bài toán | Ear images acquired in unconstrained settings are small. | Trung vị cạnh ngắn EarVN1.0 (77 px); tỉ lệ ảnh dưới 100 px |
| P2 | Khoảng trống | Efficient SR is developed and ranked on large natural images under bicubic downsampling; whether its conclusions transfer to inputs of a few dozen compressed pixels has not been examined. | Cỡ ảnh vào của NTIRE so với 24 đến 48 px |
| P3 | Phát hiện (C2, C3) | They do not. | Số dB thua bicubic; τ-b; khoảng cách giữa mô hình nhỏ và lớn |
| P4 | Chẩn đoán và giải pháp | The gap originates in the degradation and the input size, so we act there rather than on the architecture. | |
| P5 | Kết quả trong ba con số | At matched latency, ... | Phần hơn chính; độ trễ; tỉ lệ người xem |
| Đóng góp | Danh sách, mỗi mục một câu có số | (1) a benchmark ...; (2) an analysis showing ...; (3) a degradation model ...; {(4) a backbone configuration ...} | Chỉ liệt kê mục đạt tiêu chí |
| Không tuyên bố | Một đoạn ngắn | We do not propose a new building block, and we do not evaluate recognition. | |

### 2. Related Work (khoảng 0,75 trang)

Bốn nhóm, mỗi nhóm kết bằng một câu "khác ở đâu": (a) SR hiệu quả và NTIRE 2023 đến 2026 (SPAN, SPANF, DSCF, TSSR, ERRN; phần hơn gần đây đến từ cắt kênh và chưng cất); (b) suy giảm thực tế (Real-ESRGAN, BSRGAN, ước lượng nhân mờ và nhiễu từ ảnh thật); (c) hiệu ứng viền và đệm trong CNN (Mind the Pad, partial convolution padding, TLC); (d) SR cho sinh trắc học và ảnh tai. Bảng 1 ở mục 2.5b.

### 3. Benchmark and Analysis (phần mang phát hiện; khoảng 2,5 trang)

| Mục | Nội dung | Luận điểm | Bảng, hình |
|---|---|---|---|
| 3.1 Construction | AMI ở ba cỡ; vì sao AMI (mục 1.6 của kế hoạch, rút gọn); 5 fold theo người; hai fold giữ kín; ×4 và ×2; hàm thu phóng; ảnh ngoài thực tế và vai của chúng | | Bảng số ảnh theo bộ và cỡ |
| 3.2 Protocol | Số đo; phạm vi đo; thống kê ghép cặp theo người; ngưỡng real-time và cách đo | | |
| 3.3 Headroom | Bicubic và mô hình có sẵn theo cỡ ảnh | C1 | Bảng 2, Hình 2 |
| 3.4 Generic efficient SR under compression | Hai cột suy giảm; thứ hạng | C2 | Bảng 3 |
| 3.5 Architecture is not the bottleneck | Khoảng cách giữa các mô hình theo cỡ | C3 | Hình 3 |
| 3.6 Tiny inputs and borders | Phép thử ngữ cảnh trên ảnh tai và ảnh tự nhiên | C6 (cơ sở) | Hình 4 |

### 4. Method (khoảng 1,5 trang)

| Mục | Nội dung | Ghi chú viết |
|---|---|---|
| 4.1 Degradation estimated from real ear images | Ba thứ ước lượng (mức nén từ bảng lượng tử, nhiễu, dải độ mờ khớp thống kê độ nét); tách người dùng để khớp khỏi người dùng để kiểm | Nói rõ giới hạn: nhân mờ không ước lượng được từ ảnh nhỏ đã nén |
| 4.2 {Backbone configuration for tiny inputs} | Kiểu đệm; tỉ lệ sâu và rộng ở cùng độ trễ; áp cho thân bất kỳ | Chỉ có nếu C6 đạt; nếu không, nội dung này nằm ở 3.6 và 6 |
| 4.3 Training protocol and fair comparison | Hai nhánh so sánh; quy tắc chỉ khác một yếu tố (mục 1.8b) | Viết thành một bảng; đây là thứ reviewer đọc kỹ |

### 5. Experiments (khoảng 4 trang)

| Mục | Câu hỏi | Luận điểm | Bảng, hình | Lấy số từ |
|---|---|---|---|---|
| 5.1 Main comparison | Hơn mốc tinh chỉnh bao nhiêu, ở cùng độ trễ | C4, kết quả chính | Bảng 4 | `runs/S2_*/test_hr144_x4_*.csv`; `check_criteria.py gain` |
| 5.2 Does the estimated degradation matter | Ước lượng so với tổng quát | C5 | Bảng 5, 6 | `results/n2_realism.json`; `results/s5/` |
| 5.3 Does the backbone configuration matter | Biến thể so với thân tham chiếu; tương tác theo cỡ | C6 | Bảng 7 | `results/criteria/n5b-*.json` |
| 5.4 Is it truly clearer | Gờ, nhất quán với ảnh vào, người xem | C7 | Bảng 8, 9 | file theo ảnh; `results/viewer/` |
| 5.5 Generality | Thân thứ hai; ảnh ngoài thực tế; fold giữ kín; các cỡ khác và ×2 | C9 | Bảng 6, 7; một bảng phụ | `runs/`, `results/s5/` |
| 5.6 Deployment | Độ trễ; đường chất lượng theo độ trễ | C8 | Bảng 10, Hình 5 | `results/latency_jetson.csv` |
| 5.7 Ablation and negative results | Từng thành phần; thứ đã thử mà không được | | Bảng 11 | `results/runs.csv` |

### 6. Discussion and Limitations (khoảng 0,5 trang)

Viết thẳng, mỗi ý một câu: AMI chụp trong nhà, 100 người; không có cặp LR thật và HR; ảnh ngoài thực tế có đáp án chỉ ở cỡ 96 px; ảnh nhỏ thật chỉ đánh giá được bằng sở thích và số đo không tham chiếu; số đo không tham chiếu không dùng được ở cỡ chính; MS-SSIM dùng số tầng giảm; nhân mờ không ước lượng được; trọng số các mô hình 2026 ở dạng đã gộp nên việc tinh chỉnh chúng khác cách tác giả huấn luyện; không có thí nghiệm nhận dạng; phương sai gộp seed và cách chia.

### 7. Conclusion

Ba câu: phát hiện; giải pháp và mức hơn; việc tiếp theo.

## 2.5 Danh sách bảng và hình, kèm nguồn số

| # | Nội dung | Hàng × cột | Nguồn | Trạng thái |
|---|---|---|---|---|
| Bảng 1 | So với công trình liên quan | công trình × tính chất | mục 2.5b | điền một phần; còn ô phải đọc bài |
| Bảng 2 | Dư địa của SR theo cỡ ảnh | mô hình × ba cỡ (PSNR, phần hơn so với bicubic kèm khoảng tin cậy) | `results/t2_summary/quality.csv` | **có số thật** (06/10/2026); bảng LaTeX: `make_tables.py` |
| Bảng 3 | Mô hình có sẵn, bicubic so với có nén | mô hình (kể cả sáu mô hình 2026) × hai suy giảm × (PSNR, SSIM, LPIPS, GMSD) | như trên; sinh bằng `make_tables.py` | **có số thật** cho 24 tên trong kho, gồm nhóm 2026 (bỏ `span26` và `span_ch48_t44` khi dựng bảng cho bài). Bảng phải có cả cột độ trung thực lẫn LPIPS, DISTS, vì hai nhóm số đo cho kết luận ngược nhau |
| Bảng 4 | So sánh chính ở ô chính | mô hình × (PSNR, SSIM, MS-SSIM, LPIPS, DISTS, GMSD, độ trễ, tham số), hai nhánh so sánh | `runs/S2_*` | chưa có |
| Bảng 5 | Bộ phân loại "mô phỏng hay thật" | kiểu suy giảm × (độ chính xác, khoảng tin cậy) | `results/n2_realism.json` | **có số thật** (bốn kiểu: bicubic, bicubic rồi JPEG 75, tổng quát, ước lượng) |
| Bảng 6 | Ảnh ngoài thực tế | mô hình × (EarVN1.0, AWEx ở hai biên an toàn) | `results/s5/` | chưa có |
| Bảng 7 | Biến thể thân, hai thân, nhánh có kiểm soát | biến thể × thân × (PSNR, phần hơn, khoảng tin cậy, độ trễ) | `runs/T6iv_*`, `runs/T6pad_*`, `results/criteria/` | chưa có |
| Bảng 8 | Cấu trúc và độ trung thực | mô hình × (gờ giả, gờ mất, LR-PSNR, PSNR của gradient, độ lệch điểm mốc) | cột tương ứng trong file theo ảnh | chưa có |
| Bảng 9 | Khảo sát người xem | cặp mô hình × hai phần (tỉ lệ chọn, khoảng tin cậy, số người) | `results/viewer/*/analysis.csv` | chưa có |
| Bảng 10 | Hiệu quả trên thiết bị | mô hình × (trung vị, phân vị 95, tham số, FLOPs, nhóm real-time) | `results/latency_jetson.csv` | chưa có (cần Jetson) |
| Bảng 11 | Ablation và kết quả âm tính | cấu hình × số đo chính | `results/runs.csv`, `runs/` | chưa có |
| Hình 1 | Ví dụ mở bài: ảnh vào, bicubic, mô hình đa dụng, mô hình đề xuất, đáp án | | ảnh từ `evaluate.py --save-sr`; cần xin phép dùng ảnh AMI | chưa có |
| Hình 2 | Chất lượng theo cỡ ảnh vào, 16 đến 64 px | | `results/size_sweep/summary.csv` | chưa có |
| Hình 3 | PSNR theo độ trễ hoặc số tham số, hai suy giảm | | `quality.csv` + bảng độ trễ | có số chất lượng thật; vẽ theo số tham số được ngay, theo độ trễ thì chờ Jetson |
| Hình 4 | Sai số theo bề rộng ngữ cảnh | | `results/t6_summary/context.csv` | **có số thật**; hình phải vẽ cả đường của bicubic, vì trên ảnh tai bicubic cũng mất gần bằng mạng |
| Hình 5 | Đường chất lượng theo độ trễ trên thiết bị | | bảng 4 + bảng 10 | chưa có |
| Hình 6 | So sánh định tính, kèm ca thất bại | | ảnh SR đã lưu | chưa có |

**Script (bản 24).** `make_tables.py` sinh bảng 2, 3, 5, 9, 10 và bảng phép thử ngữ cảnh; `compare_table.py` sinh bảng 4, 6, 7, 8, 11 (trung bình, khoảng tin cậy, chênh lệch ghép cặp so với dòng mốc, dấu † khi khoảng tin cậy không chứa 0); `make_figures.py` vẽ hình 2, 3, 4 tự động, hình 5 bằng lệnh con `cost`, hình 1 và 6 bằng lệnh con `grid`. Lệnh mẫu ở `docs/RUNBOOK.md`, mục 5. Hình 3 và 4 đã vẽ thử từ kết quả sơ bộ; các hình còn lại mới qua kiểm thử trên dữ liệu giả.

**Hình dùng ảnh AMI** (hình 1, 6) cần thư đồng ý của tác giả bộ dữ liệu, do giấy phép CC BY-NC-ND. Thư mẫu và phương án dự phòng ở `docs/AMI_PERMISSION.md`. Việc gửi thư do nghiên cứu sinh làm, nên làm ngay vì có thể mất vài tuần.

## 2.5b Bảng 1: công trình liên quan (điền tới mức đã xác minh, ngày 05/10/2026)

Cột "Đã xác minh" ghi mức kiểm của nhóm: *có nguồn* nghĩa là đã mở được trang của bài và các ô trong dòng khớp với nội dung đọc được; *chỉ tồn tại* nghĩa là đã thấy bài tồn tại nhưng chưa đọc nội dung, các ô còn lại phải đọc bài rồi mới điền; *theo trí nhớ* nghĩa là chưa tra.

| Công trình | Ảnh tai | Cỡ ảnh vào được xét | Suy giảm | Mô hình real-time, đo trên thiết bị | Khác bài này ở đâu | Đã xác minh |
|---|---|---|---|---|---|---|
| NTIRE 2026 Efficient SR (báo cáo, CVPRW 2026) | Không | Ảnh tự nhiên lớn (DIV2K, LSDIR) | Bicubic ×4 | Có, đo trên GPU máy bàn | Không xét ảnh vào vài chục pixel, không xét nén, không bàn kiểu đệm | Có nguồn |
| NTIRE 2025 Efficient SR (báo cáo, CVPRW 2025) | Không | Như trên | Bicubic ×4 | Có, GPU máy bàn | Như trên | Có nguồn |
| SPAN (Wan và cộng sự, CVPRW 2024) | Không | Ảnh tự nhiên | Bicubic | Có | Là baseline của NTIRE 2026; bài này dùng làm một thân | Có nguồn (qua báo cáo NTIRE 2026 và mã nguồn) |
| ESC (Lee và cộng sự, ICCV 2025) | Không | Ảnh tự nhiên | [đọc bài] | [đọc bài] | Kiến trúc mới cho SR hiệu quả; không xét miền ảnh tai | Chỉ tồn tại |
| Real-ESRGAN (Wang và cộng sự, 2021) | Không | Ảnh tự nhiên | Tổng hợp bậc cao | Không phải mục tiêu | Suy giảm tổng quát, không ước lượng theo miền | Chỉ tồn tại |
| BSRGAN (Zhang và cộng sự, 2021) | Không | Ảnh tự nhiên | Tổng hợp, xáo thứ tự | Không | Như trên | Chỉ tồn tại |
| Ji và cộng sự, "Real-World Super-Resolution via Kernel Estimation and Noise Injection" (CVPRW 2020) | Không | Ảnh tự nhiên | Ước lượng nhân mờ và nhiễu từ ảnh thật | Không | Gần N2 nhất về ý; bài này áp cho ảnh tai nhỏ đã nén, nơi nhân mờ không ước lượng được, và kiểm bằng bộ phân loại cùng người xem | Chỉ tồn tại |
| Alsallakh và cộng sự, "Mind the Pad: CNNs Can Develop Blind Spots" (ICLR 2021) | Không | Ảnh tự nhiên (phân loại, phát hiện) | Không áp dụng | Không | Gần N5b nhất về ý; không làm cho SR, không dưới ràng buộc độ trễ | Chỉ tồn tại |
| "Improving Ear Recognition with Super-resolution" (IEEE, 2023) | Có | [đọc bài] | [đọc bài] | [đọc bài] | Theo tiêu đề, mục tiêu là độ chính xác nhận dạng, không phải chất lượng ảnh và tốc độ | Chỉ tồn tại |
| Nguyen và cộng sự, "Super-resolution for biometrics: A comprehensive survey" (Pattern Recognition, 2018) | [đọc bài: có mục về tai không] | | | | Tổng quan; dùng để định vị bài | Chỉ tồn tại |
| Partial convolution padding; TLC (lệch giữa huấn luyện theo patch và suy luận cả ảnh); FixRes | Không | | | | Các công trình về viền và lệch cỡ ảnh ở bài toán khác | Theo trí nhớ |
| SR ảnh mặt dẫn hướng bằng điểm mốc (FSRNet, DIC) | Không (mặt) | Ảnh mặt 16×16 | Bicubic | Không | Chỉ liên quan nếu giữ N1 | Theo trí nhớ |

**Còn phải làm ở T1 trước khi viết mục Related Work:** (1) đọc toàn văn các dòng "chỉ tồn tại" và điền ô `[đọc bài]`; (2) tìm có hệ thống các bài SR cho ảnh tai và cho sinh trắc học vùng nhỏ (mống mắt, vân tay, mặt ở xa) trên IEEE Xplore, Scopus và Google Scholar, vì lần rà này chỉ tìm được một bài về ảnh tai và chưa đủ để nói "không còn bài nào khác"; (3) kiểm lại năm, hội nghị và tên tác giả của mọi dòng. Bài không được viết câu "chưa ai làm" cho tới khi xong bước (2).

Nguồn đã mở trong lần rà này: báo cáo NTIRE 2026 ESR (arXiv 2604.03198) và kho mã `Amazingren/NTIRE2026_ESR`; báo cáo NTIRE 2025 ESR (arXiv 2504.10686); ESC (arXiv 2503.06671); "Mind the Pad" (arXiv 2010.02178); kho mã `jixiaozhong/RealSR`; Real-ESRGAN (arXiv 2107.10833); trang IEEE Xplore của "Improving Ear Recognition with Super-resolution" (mã tài liệu 10180250; không đọc được nội dung); bài tổng quan trên Pattern Recognition 2018.

## 2.6 Câu viết sẵn theo từng nhánh kết quả

Mỗi luận điểm có hai phiên bản câu. Khi có số, chọn phiên bản đúng, không sửa câu cho hợp với mong muốn.

| Luận điểm | Nếu đạt | Nếu không đạt |
|---|---|---|
| C2 | **Chọn nhánh này, có thêm phạm vi (bản 25).** "With JPEG-compressed inputs (quality 75), all 16 networks fall below bicubic interpolation in PSNR (by 0.40 to 0.53 dB at 36-pixel inputs; 0.29 to 0.49 dB at 24 pixels; 0.40 to 0.50 dB at 48 pixels), and likewise in SSIM, MS-SSIM, GMSD and input consistency, while all 16 remain better than bicubic in LPIPS and DISTS: the outputs are sharper but less faithful. Their ranking is nearly unrelated to the ranking under bicubic degradation (τ-b = 0.08, 0.09 and −0.06 at 24, 36 and 48 pixels; 42 to 48 of 105 pairs reverse significantly)." | "Compression lowers all networks by [X] dB but they remain above bicubic, and their ranking is preserved (τ-b = [X])." Bỏ chữ "fails" khỏi tiêu đề |
| C3 | **Chọn nhánh này, viết thành hai vế (bản 25).** "Under compression, a network with 39× more parameters is no better (−0.04 dB, 95% CI [−0.05, −0.04]); the architecture explains none of the gap to bicubic. Under matched (bicubic) degradation it gains 0.23 dB [0.22, 0.25] at 36-pixel inputs, and capacity matters more as the input shrinks (0.14, 0.23 and 0.49 dB at 48, 36 and 24 pixels)." | "Larger networks remain clearly better ([X] dB)." Thêm mốc lớn vào bảng 4 |
| C5 | **Nửa đầu chọn nhánh này (bản 25); nửa sau chờ khảo sát.** "A classifier separates real small ear images from images produced by the measured degradation with 58.5% balanced accuracy (95% CI [56.8, 59.9]), against 78.0% [76.8, 80.0] for the generic pipeline and 80.1% [77.3, 86.6] for plain bicubic downsampling; a fixed bicubic-plus-JPEG-75 degradation is statistically indistinguishable from the measured one (56.7%). Viewers prefer the model trained with it in [X]% of pairs (95% CI [X, X])." Phải nêu giới hạn: bộ phân loại chấm trên ảnh của 4 người | "The estimated degradation is not closer to real images than a generic pipeline ([X]% vs [X]%); a generic pipeline is sufficient for this domain." Viết thành kết quả âm tính ở 5.7 |
| C6 | "The configured backbone improves PSNR by [X] dB (95% CI [X, X]) at matched latency, and the improvement is larger at 36-pixel inputs than at [X]-pixel inputs (difference [X] dB, CI excludes zero)." | "Inputs smaller than the receptive field lose [X] dB at the border, but neither padding mode nor depth-width trade-off recovers it (at most [X] dB)." Chuyển sang mục 3.6 và 6 |
| C7 | "The gain is not obtained by inventing structure: false ridges [X] vs [X], input consistency [X] vs [X] dB, and with the ground truth shown, viewers choose our output in [X]% of pairs." | "The gain is in distortion metrics only; structure measures and viewers do not separate the models." Không dùng "clearer" |
| C8 | "It runs in [X] ms (median; p95 [X] ms) on a Jetson Nano at FP16 for a 48×68 input." | "It runs in [X] ms, which meets the near-real-time threshold of 66 ms but not 33 ms." |
| So sánh chính | "At matched latency (within 10%), the model improves [metric] by [X] over the same backbone fine-tuned with the same data and schedule (paired difference, 95% CI [X, X], [n] subjects, five folds)." | "The model matches the fine-tuned baseline ([X], CI contains zero)." Bài rơi về kịch bản phân tích |

## 2.7 Quy trình sau khi có kết quả

1. **Đọc trước, viết sau.** Với mỗi luận điểm, mở file ở cột "Lấy số từ", ghi kết luận đạt hay không vào một bảng theo dõi (thêm vào cuối kế hoạch). Xác định kịch bản (mục 2.2) trước khi viết chữ nào.
2. **Dựng bảng và hình bằng script**, không gõ tay: `make_tables.py` và các lệnh `check_criteria.py`. Số trong bài chỉ được chép từ file do script sinh ra.
3. **Điền `[X]`** trong abstract, introduction và các câu ở mục 2.6, chọn nhánh câu theo kết quả.
4. **Viết bình luận cho từng bảng theo ba câu:** bảng cho thấy gì (kết luận); con số nào chứng minh (kèm khoảng tin cậy); điều đó có nghĩa gì cho luận điểm. Thêm câu thứ tư nếu có ngoại lệ trong bảng (dòng nào đi ngược xu hướng và vì sao).
5. **Kết quả âm tính** viết thành mục riêng (5.7), không giấu. Biến thể thua trong T6 (iv) đều được báo.
6. **Đối chiếu chéo:** mỗi con số trong abstract phải xuất hiện đúng như vậy trong một bảng; mỗi tuyên bố trong introduction phải trỏ được tới một bảng hoặc hình; fold 1 và 5 được báo riêng.
7. **Rà lại Limitations** theo những gì thực sự xảy ra khi chạy (ví dụ bộ điểm mốc không dùng được, Jetson không có).

Khi bạn gửi kết quả (thư mục `results/` và `results/runs.csv`), tôi sẽ làm bước 1 đến 4 cùng bạn.

## 2.8 Quy tắc viết

Mỗi đoạn mở bằng kết luận rồi đến số. Không dùng "first", "novel", "state-of-the-art", "significantly" khi không kèm bằng chứng. Mọi chênh lệch đi kèm khoảng tin cậy và số người. Tuyên bố về kiến trúc chỉ dựa trên nhánh có kiểm soát; tuyên bố thực tế chỉ dựa trên nhánh trọng số công bố (mục 1.8b). Giữ bản thảo ở dạng dễ ẩn danh. Kiểm tra chéo mọi con số trước khi nộp.

---

# PHẦN 3. THIẾT KẾ PROJECT (PIPELINE)

Phần này mô tả project code hoàn chỉnh cho bài: từ dữ liệu thô tới bảng và hình trong bản thảo. Thiết kế chưa có dòng code nào; các chỗ ghi "giả định" cần kiểm lại khi đọc được code cũ ở `earsr_project_span`.

## 3.1 Nguyên tắc

1. **Repo mới, tách khỏi `earsr_project_span`.** Bài này là bài mới hoàn toàn. Code cũ chỉ được chép sang từng file, sau khi rà lại (ví dụ định nghĩa SPAN trong `sr_models.py`).
2. **Mỗi lần chạy là một file cấu hình.** Không sửa tham số trong code. Hai lần chạy được so với nhau chỉ khác nhau đúng một khóa cấu hình (quy tắc công bằng ở mục 1.2b).
3. **Cách chia dữ liệu và ảnh test là file cố định**, sinh một lần, ghi mã băm, đưa vào git. Không lần chạy nào tự chia lại.
4. **Độc lập với thân, và nhánh tùy chọn tách riêng.** Kiểu đệm và tỉ lệ sâu và rộng là tham số của thân, đổi bằng cấu hình. Các thành phần tùy chọn (đầu phụ N1; hai đầu và cổng N3) nằm trong một thư mục riêng và nhận đặc trưng của thân qua một giao diện chung; nhánh chính chạy được khi không có chúng.
5. **Số đo lưu theo từng ảnh.** Suy luận chỉ ghi ảnh và số đo từng ảnh; mọi trung bình, khoảng tin cậy, kiểm định được tính ở một bước riêng. Nhờ đó đổi cách thống kê không phải chạy lại mô hình.
6. **Mọi con số trong bài truy được về một mã lần chạy.** Bảng và hình trong bản thảo được sinh tự động từ kết quả, không gõ tay.
7. **Lỗi im lặng được chặn bằng kiểm thử** (mục 3.6). Bài span_tiny đã từng mất toàn bộ kết quả vì một lỗi thứ tự phép tính trong SPAB không gây crash.

## 3.2 Sơ đồ tổng

```
 DỮ LIỆU THÔ (chỉ đọc)                    TRỌNG SỐ CÔNG BỐ (bạn tải về)
 AMI · EarVN1.0 · AWEx · bộ điểm mốc      SPAN, RLFN, SAFMN, SMFANet, EDSR, SwinIR-light,
 DIV2K (tiền huấn luyện, kiểm chứng)      ESRGAN, Real-ESRGAN, BSRGAN ...
        │                                          │
        ▼                                          │
 P1  DỰNG DỮ LIỆU ─────────────────────────┐       │
     benchmark AMI 3 cỡ + cỡ 244×348       │       │
     DIV2K: train, và bản thu nhỏ để test  │       │
     5 fold theo người (file cố định)      │       │
     tập EarVN1.0 / AWEx có biên an toàn   │       │
        │                                  │       │
        ▼                                  │       │
 P2  SUY GIẢM                              │       │
     bicubic · tổng quát · ước lượng (N2)  │       │
     train: sinh lúc chạy                  │       │
     test: sinh một lần, lưu đĩa           │       │
        │                                  ▼       ▼
        │                         P3  KHO MÔ HÌNH (registry)
        │                             thân và biến thể (đệm, sâu rộng) · bộ dò điểm mốc
        │                             thành phần tùy chọn (N1, N3)
        │                                  │
        ├──────────────► P4  ĐÁNH GIÁ KHÔNG HUẤN LUYỆN ───────────┐
        │                    T2 · phép thử ngữ cảnh (T6 i, v)     │
        │                                  │                      │
        │                P5  ĐO ĐỘ TRỄ TRÊN THIẾT BỊ (T3, S6) ────┤
        │                    kể cả hỗ trợ từng kiểu đệm           │
        │                    PyTorch → ONNX → TensorRT / Android  │
        │                                  │                      │
        │                          [ĐIỂM KIỂM TRA 1: có dư địa?   │
        │                           có hiệu ứng viền?]            │
        ▼                                  ▼                      │
 P6  HUẤN LUYỆN                                                   │
     6pre tiền huấn luyện trên DIV2K: 5 lần rút gọn (thân tham    │
          chiếu và 4 biến thể), 1 đến 2 lần đủ cho thân thắng     │
     6a bộ dò điểm mốc (1 bộ để đo; bộ thứ 2 chỉ khi làm N1)      │
     6b thân tinh chỉnh trên tai  = MỐC (giao thức chọn ở T6 ii)  │
     6c mốc cùng độ trễ; đối chứng của nhánh tùy chọn             │
     6d MÔ HÌNH ĐỀ XUẤT, nhánh chính: thân N5b + suy giảm N2      │
        nhánh tùy chọn: + đầu phụ N1, hoặc + hai đầu và cổng N3   │
        │                                                         │
        ▼                                                         │
 P7  SUY LUẬN + SỐ ĐO TỪNG ẢNH  ◄─────────────────────────────────┘
     ảnh SR · PSNR/SSIM · LPIPS/DISTS · bộ cảm nhận NTIRE
     độ lệch điểm mốc · bảng gờ · phép thử tương tác
        │
        ▼
 P8  THỐNG KÊ                       P9  KHẢO SÁT NGƯỜI XEM
     bootstrap theo người · ghép cặp    công cụ so cặp, mù, ngẫu nhiên
     Holm · τ-b · tiêu chí đạt          │
        │                               │
        ▼                               ▼
 P10 BẢNG, HÌNH, BẢN THẢO (LaTeX, sinh tự động)  →  gói công bố (code, file danh sách, script)
```

## 3.3 Cấu trúc thư mục

```
earsr_rt/
├── configs/
│   ├── data/          ami_hr96.yaml  ami_hr144.yaml  ami_hr192.yaml  ami_hr244.yaml  earvn.yaml  awex.yaml  div2k.yaml  div2k_tiny.yaml
│   ├── degrade/       bicubic.yaml  generic.yaml  estimated.yaml
│   ├── model/         span.yaml  rlfn.yaml  edsr_base.yaml ...  variants/ (kiểu đệm, sâu và rộng)  heads.yaml  gate.yaml  aux.yaml
│   ├── train/         pretrain_short.yaml  pretrain_full.yaml  finetune.yaml  protocol_{native,fixed,random}.yaml  (tùy chọn: stage1, stage2, ldl, gan)
│   └── exp/           T2.yaml  T4.yaml  T5.yaml  T6.yaml  S1.yaml ... E10.yaml   (mỗi thí nghiệm một file)
├── earsr/
│   ├── data/          build_ami.py  build_wild.py  build_div2k.py  build_div2k_tiny.py  splits.py  datasets.py  resize.py
│   ├── degrade/       bicubic.py  generic.py  estimated.py  fit_estimated.py  realism_clf.py
│   ├── models/        registry.py  backbones/  padding.py  variants.py  reparam.py  landmark/  optional/ (heads.py  gate.py  aux_head.py)
│   ├── losses/        pixel.py  perceptual.py  gan.py  ldl.py  gate_label.py  aux.py
│   ├── train/         trainer.py  pretrain.py  (tùy chọn: stage1.py  stage2.py)
│   ├── eval/          infer.py  metrics_fr.py  metrics_nr.py  context_test.py  receptive_field.py  landmark_dev.py  ridge_check.py  ntire_score.py
│   ├── stats/         bootstrap.py  paired.py  interaction.py  mde.py  holm.py  kendall.py  criteria.py
│   ├── deploy/        export_onnx.py  bench_trt.py  bench_padding.py  bench_android.md  quantize.py
│   └── report/        tables.py  figures.py  viewer_study/
├── splits/            ami_5fold.json  earvn_roles.json  awex_test.txt   (trong git, có mã băm)
├── tests/             (mục 3.6)
├── scripts/           run_exp.sh  queue.sh  make_paper.sh
├── runs/              <mã lần chạy>/{config.yaml, ckpt/, log, per_image.parquet}   (ngoài git)
├── results/           runs.csv  agg/  tables/*.tex  figures/*.pdf
└── paper/             main.tex  sections/  (bảng và hình \input từ results/)
```

## 3.4 Từng giai đoạn

| Giai đoạn | Vào | Ra | Điều phải đúng trước khi đi tiếp |
|---|---|---|---|
| **P1 Dựng dữ liệu** | Ảnh thô | Ảnh HR dạng PNG ở từng cỡ; file fold; file danh sách kèm mã băm; tập DIV2K để tiền huấn luyện và bản DIV2K thu nhỏ (từ phần valid) để test | Không người nào nằm ở hai phần của cùng một fold; mỗi người test đúng một lần; số ảnh khớp (700) |
| **P2 Suy giảm** | Ảnh HR; cấu hình suy giảm | Ảnh LR test lưu đĩa; hàm sinh LR lúc huấn luyện | Sinh hai lần cho cùng kết quả từng bit; bicubic khớp cách thu nhỏ của NTIRE |
| **P3 Kho mô hình** | Định nghĩa kiến trúc; trọng số công bố | Mô hình gọi theo tên; bản deploy đã gộp nhánh | Tái lập PSNR công bố của từng mô hình trên một bộ chuẩn (Set5 hoặc DIV2K valid) trong sai số 0,05 dB |
| **P4 Đánh giá không huấn luyện** | P1, P2, P3 | Bảng bicubic và mô hình có sẵn ở ba cỡ, hai kiểu suy giảm (T2); đường sai số theo bề rộng ngữ cảnh, trên ảnh tai và DIV2K thu nhỏ (T6 i, v); cổng oracle nếu làm N3 (T4) | L1, L2; cơ sở của N5b |
| **P5 Độ trễ** | Mô hình bản deploy; mô hình chưa huấn luyện với từng kiểu đệm | File JSON độ trễ theo giao thức mục 1.5, cho từng mô hình, từng thiết bị; bảng hỗ trợ kiểu đệm trên từng thiết bị | PSNR của đầu ra TensorRT FP16 so với đáp án lệch không quá 0,05 dB so với đầu ra PyTorch |
| **P6 Huấn luyện** | P1, P2; trọng số khởi tạo | Checkpoint chọn trên validation của fold | Mốc tinh chỉnh không thua mô hình nguyên bản trên ảnh tai |
| **P7 Suy luận và số đo** | Checkpoint; ảnh LR test | Ảnh SR; một bảng số đo theo từng ảnh | Số ảnh mỗi lần chạy khớp với file fold |
| **P8 Thống kê** | Các bảng từng ảnh | Trung bình, khoảng tin cậy, kiểm định; file "đạt hay không đạt" cho từng tiêu chí | Tiêu chí đọc từ một file ngưỡng ghi trước, không sửa sau khi chạy |
| **P9 Khảo sát người xem** | Ảnh SR của các mô hình được chọn | File phiếu trả lời; tỉ lệ chọn kèm khoảng tin cậy | Thứ tự và vị trí trái phải ngẫu nhiên; người xem không biết mô hình |
| **P10 Bảng, hình, bản thảo** | Kết quả P8, P9 | File `.tex` của từng bảng; hình PDF; bản thảo | Một lệnh dựng lại toàn bộ bảng và hình từ `results/` |

### Chi tiết kỹ thuật cần chốt ngay ở P1 và P2

- **Cỡ ảnh chia hết cho 4.** Ảnh AMI là 492×702. Sau khi thu nhỏ theo cạnh ngắn, cắt giữa để hai cạnh chia hết cho 4:

| Cạnh ngắn HR | Cỡ HR dùng | Ảnh vào ×4 | Ảnh vào ×2 |
|---|---|---|---|
| 96 | 96×136 | 24×34 | 48×68 |
| 144 | 144×204 | 36×51 | 72×102 |
| 192 | 192×272 | 48×68 | 96×136 |
| 246 (chỉ cho bộ cảm nhận NTIRE) | 244×348 | 61×87 | 122×174 |

- **Hàm thu nhỏ.** Dùng một hàm bicubic có khử răng cưa, tương thích MATLAB `imresize` như NTIRE, cho cả việc tạo HR từ ảnh gốc lẫn tạo LR bicubic. Các thư viện cho kết quả khác nhau, nên chỉ một hàm duy nhất (`earsr/data/resize.py`) được phép dùng trong toàn project.
- **AMI là JPEG.** HR được tạo bằng cách thu nhỏ từ 2,5 đến 5 lần, nên vết nén của ảnh gốc gần như mất; HR lưu PNG, không nén lại.
- **Fold.** Chia 100 người thành 5 nhóm 20 người, hạt giống cố định. Ở fold i: nhóm i là test, 10 người lấy ngẫu nhiên từ 80 người còn lại là validation, 70 người là train. Fold 1 và 5 là hai fold giữ kín (mục 1.6).
- **Vai của người trong EarVN1.0** (`earvn_roles.json`): các nhóm người tách rời: train thêm; test giữ riêng; khớp tham số suy giảm N2; huấn luyện và test bộ phân loại "mô phỏng hay thật"; ảnh nhỏ thật cho khảo sát người xem.
- **Ảnh LR test** của suy giảm có yếu tố ngẫu nhiên (nhiễu, nén) được sinh một lần với hạt giống ghi trong cấu hình rồi lưu đĩa. Mọi mô hình test trên đúng cùng một bộ ảnh.

### Tiền xử lý dữ liệu

Nguyên tắc: không bước nào được làm thay đổi nội dung ảnh đáp án (không khử nhiễu, không làm nét, không cân bằng sáng, không nén lại). Chuẩn hóa giá trị về [0, 1] chỉ làm trong bộ nạp dữ liệu.

| Bộ | Việc phải làm | Ghi chú |
|---|---|---|
| **AMI** (đã kiểm: 700 ảnh, 100 người, mỗi người 7 ảnh, tất cả 492×702, RGB, không có cờ xoay EXIF) | Lấy mã người từ tên file; gắn nhãn góc chụp (back, down, front, left, right, up, zoom); thu nhỏ về từng cỡ bằng hàm chung; cắt giữa cho chia hết cho 4; lưu PNG | Không cắt vùng tai và không căn chỉnh ở benchmark chính, để không thêm bước chủ quan. Kết quả báo theo góc chụp; ảnh "zoom" có khung hình khác nên được báo riêng |
| **Hộp bao vùng tai** (cho AMI và tập ngoài thực tế) | Sinh một lần từ bộ dò điểm mốc, lưu thành file | Mọi số đo được báo ở hai phạm vi: toàn ảnh và trong vùng tai. Bài span_tiny từng bị lệch giữa hai phạm vi này |
| **EarVN1.0, AWEx làm đáp án** | Lọc theo cạnh ngắn; lọc chế độ màu khác RGB và tỉ lệ khung bất thường; bỏ ảnh trùng và gần trùng; thu nhỏ ít nhất 2 lần làm biên an toàn; cắt cho chia hết cho 4 | Ngưỡng cạnh ngắn chốt ở P1, sau khi đếm số ảnh còn lại ở từng mức. Danh sách ảnh giữ và ảnh loại (kèm lý do) được lưu |
| **Ảnh nhỏ thật của EarVN1.0** (cho N2 và khảo sát người xem) | Chọn theo cạnh ngắn 24 đến 48 px; chia vai theo người | Không xử lý gì; đây là ảnh vào thật |
| **Bộ điểm mốc** | Đổi tọa độ điểm mốc theo cùng phép thu nhỏ và cắt của ảnh; sinh nhãn giả trên AMI bằng bộ dò thứ nhất | Chỉ sau khi T1 xác nhận điều khoản |
| **Lúc huấn luyện** | Huấn luyện đúng tỉ lệ (mục 3.10, điểm 2); lật ngang | Lật ngang và xoay phải biến đổi cả tọa độ điểm mốc. Xoay 90 độ tạo hướng tai không có thật; có dùng hay không được quyết bằng một phép thử nhỏ ở S1 |

### Chi tiết P6: thứ tự huấn luyện

```
DIV2K ──► 6pre TIỀN HUẤN LUYỆN (bỏ qua nếu T6 iii cho thấy không cần)
            rút gọn, cùng ngân sách: thân tham chiếu (đệm số 0)
                                     2 biến thể sâu và rộng
                                     2 kiểu đệm khác (chỉ kiểu qua được T3)
            đủ: thân thắng; thân thứ hai
                │
trọng số công bố│
      │         │   tinh chỉnh trên ảnh tai (train của fold), loss điểm ảnh,
      ▼         ▼   giao thức chọn ở T6 (ii), suy giảm theo mục 1.7
  MỐC = thân công bố       THÂN N5b đã tinh chỉnh
  đã tinh chỉnh ──┐              │
                  │              ├─► NHÁNH CHÍNH: thân N5b + suy giảm N2  = MÔ HÌNH ĐỀ XUẤT
                  │              │
                  │              └─► NHÁNH TÙY CHỌN (sau điểm kiểm tra 2, tối đa một):
                  │                    + đầu phụ N1 (gỡ lúc suy luận)
                  │                    + hai đầu và cổng N3 (hai giai đoạn; cần mốc bản GAN,
                  │                      mốc + LDL và đường trộn làm đối chứng)
                  │
                  ├─► mốc huấn luyện với cùng suy giảm N2      (đối chứng N2)
                  ├─► mốc nới rộng cho bằng độ trễ của mô hình đề xuất
                  └─► mốc + ảnh bộ điểm mốc, không nhãn        (đối chứng N1, chỉ khi làm N1)
```

Mọi phép so sánh chỉ khác nhau một khóa cấu hình: cùng giao thức, cùng số bước, cùng dữ liệu. Trong nhóm chọn biến thể (T6 iv), mọi thân có cùng ngân sách tiền huấn luyện rút gọn; phép so sánh cuối là thân thắng (tiền huấn luyện đủ) với mốc dùng trọng số công bố.

## 3.5 Cấu hình, mã lần chạy, sổ ghi

- **Mã lần chạy:** `{thí nghiệm}_{thân}-{biến thể}_{tiền huấn luyện}_{giao thức}_x{hệ số}_hr{cỡ}_{suy giảm}_f{fold}`, ví dụ `T6_span-reflect_ps20_rand_x4_hr144_bic_f3`. Biến thể: kiểu đệm và tỉ lệ sâu rộng (`zero`, `replicate`, `reflect`, `wide`, `deep`). Tiền huấn luyện: `pub` (trọng số công bố), `ps20` (rút gọn, 20% số bước), `pf` (đủ), `none` (học từ đầu). Giao thức: `native` (cắt từ ảnh gốc), `fixed` (ảnh HR đúng cỡ), `rand` (tỉ lệ ngẫu nhiên). Seed bằng số fold. Một kiểm thử bảo đảm hai cấu hình khác nhau không bao giờ cho cùng một mã.
- **Mỗi thư mục lần chạy** giữ bản sao cấu hình đầy đủ, mã commit của git, mã băm file fold, log, checkpoint, và bảng số đo từng ảnh.
- **`results/runs.csv`** là sổ ghi duy nhất: một dòng mỗi lần chạy, kèm trạng thái (xong, lỗi, bị loại và lý do). Lần chạy bị loại không bị xóa.
- **Bảng số đo từng ảnh** có các cột: mã lần chạy, bộ dữ liệu, người, ảnh, fold, hệ số, cỡ, suy giảm, điểm vận hành (F hoặc P, hoặc α), rồi từng số đo.
- **File ngưỡng** `configs/criteria.yaml` chép nguyên các tiêu chí ở mục 1.2b, 1.4, 1.5, 1.10. Bước P8 đọc file này và ghi kết quả đạt hay không; file được commit trước lần chạy đầu tiên.

## 3.6 Kiểm thử tự động

| Nhóm | Kiểm gì | Vì sao |
|---|---|---|
| Rò rỉ | Ảnh DIV2K dùng để test (bản thu nhỏ) không nằm trong tập tiền huấn luyện. Người train, validation, test của mỗi fold không giao nhau; các nhóm vai trong EarVN1.0 không giao nhau | Quy tắc chống rò rỉ, mục 1.6 |
| Công thức kiến trúc | Từng khối của từng thân được so với công thức trong bài gốc trên một tensor nhỏ tính tay | Lỗi thứ tự phép tính trong SPAB ở bài cũ |
| Gộp nhánh | Đầu ra bản huấn luyện và bản deploy lệch nhau không quá 1e-4; tham số và FLOPs đếm trên bản deploy | Số tham số bản huấn luyện của SPAN gây hiểu sai |
| Trọng số công bố | Tái lập PSNR công bố trong 0,05 dB | Nạp sai trọng số hoặc sai tiền xử lý không gây crash |
| Số đo | PSNR, SSIM kênh Y so với một cài đặt tham chiếu trên ảnh mẫu; công thức điểm NTIRE so với ví dụ trong báo cáo NTIRE | Sai lệch về kênh màu, viền, dải giá trị |
| Suy giảm | Tất định theo hạt giống; LR bicubic khớp hàm tham chiếu | Ảnh test phải giống nhau giữa các mô hình |
| Kiểu đệm | Với cùng trọng số, đổi kiểu đệm thì đầu ra ở vùng giữa ảnh (xa viền hơn bán kính vùng nhìn) không đổi, còn vùng gần viền thì đổi. Chỉ áp cho thân không có phép toán toàn cục; bán kính vùng nhìn được đo bằng `receptive_field.py` | Bắt lỗi cài đặt đệm không gây crash |
| Phép thử ngữ cảnh | Với bề rộng ngữ cảnh 0, hai lần chạy cho đầu ra giống hệt nhau; với ngữ cảnh lớn hơn bán kính vùng nhìn, đầu ra của cửa sổ bằng đúng phần tương ứng khi chạy cả ảnh | Phép thử lõi của N5b phải đo đúng thứ nó định đo |
| Mã lần chạy | Hai cấu hình khác nhau không trùng mã | Tránh ghi đè kết quả |
| Thành phần tùy chọn (chỉ khi dùng) | Cổng bằng 1 thì đầu ra đúng bằng đầu trung thực; bằng 0 thì đúng bằng đầu kết cấu; gỡ đầu phụ không đổi đầu ra | Bản F phải thật sự là mốc trung thực |
| Thống kê | Bootstrap trên dữ liệu giả có hiệu ứng biết trước cho khoảng tin cậy phủ đúng | Tiêu chí đạt dựa hoàn toàn vào bước này |
| Chạy thử đầu cuối | Toàn bộ P1 đến P10 trên 10 người, vài trăm bước, trong vài phút | Bắt lỗi nối giữa các giai đoạn trước khi chạy dài |

## 3.7 Chạy trên một GPU

- **Hàng đợi tuần tự** (`scripts/queue.sh`): mỗi mục là một file cấu hình; lần chạy nào cũng tiếp tục được từ checkpoint gần nhất; lần chạy lỗi không chặn mục sau.
- **Thứ tự trong một thí nghiệm:** đi hết một fold cho mọi mô hình trước khi sang fold kế, để có bảng so sánh sơ bộ sớm.
- **Thiết bị:** máy huấn luyện xuất ONNX; một script duy nhất chạy trên Jetson (dựng engine TensorRT, đo, ghi JSON) để bạn chỉ cần chép thư mục sang và chạy một lệnh. Phần Android được viết sau khi biết máy.
- **Chưa biết:** thời gian một lần huấn luyện và dung lượng đĩa. Cả hai được đo ở lần chạy thử đầu cuối và ở T5, rồi cập nhật mục 1.11.

## 3.8 Bản đồ: giai đoạn, thí nghiệm, luận điểm, vị trí trong bài

| Thí nghiệm | Giai đoạn dùng | Luận điểm | Bảng hoặc hình trong bài |
|---|---|---|---|
| T2 | P1, P2, P3, P4, P8 | L1, L2 | Bảng dư địa theo cỡ ảnh; bảng thứ hạng hai kiểu suy giảm |
| T3, S6, E10 | P3, P5 | L2, L6 | Bảng độ trễ và hiệu quả theo NTIRE |
| T4 (tùy chọn) | P4, P6c | L4 (N3) | Hình đường PSNR và LPIPS: oracle, đường trộn |
| T5 (tùy chọn) | P6a, P6b, P6d, P7, P8 | L4 (N1) | Bảng ablation đầu phụ, ba nhánh |
| T6 | P4, P6b, P7, P8 | L2, L4 (N5b) | Hình sai số theo bề rộng ngữ cảnh, trên ảnh tai và trên DIV2K thu nhỏ; bảng kiểu đệm và sâu rộng; dòng ablation ba nhánh giao thức |
| S1, S2 | P6, P7, P8 | L3, L4 | Bảng chính: ô ×4, 144 px |
| S3, E4 | P6, P7, P8 | L4, L7 | Bảng ablation; bảng thân thứ hai |
| S4, E1, E2, E3 | P2, P6, P7, P8 | L1, L4 | Bảng các cỡ và ×2; hình chất lượng theo cỡ ảnh vào |
| S5, E8 | P1, P2, P7 | L7, L4 (N2) | Bảng ảnh ngoài thực tế; bảng bộ phân loại "mô phỏng hay thật" |
| S7, E5, E6 | P7, P9, P10 | L5 | Bảng người xem; bảng gờ; hình so sánh; hình cổng |
| E9 | P6, P8 | Thống kê | Phụ lục của bài |

## 3.9 Thứ tự dựng code

**Giai đoạn 1, không huấn luyện (tới điểm kiểm tra 1).**

1. P1 (AMI ở các cỡ, fold, DIV2K thu nhỏ), P2, kiểm thử rò rỉ và suy giảm.
2. P3 với các mô hình có trọng số công bố; kiểm thử công thức, gộp nhánh, tái lập PSNR.
3. P7, P8 ở mức tối thiểu (PSNR, SSIM, LPIPS, DISTS; bootstrap; mức chênh nhỏ nhất phát hiện được), đủ để ra T2.
4. `context_test.py` và `receptive_field.py`, cùng kiểm thử của chúng, đủ để ra T6 (i) và phần phép thử ngữ cảnh của (v).
5. P5: script đo độ trễ và đo hỗ trợ kiểu đệm, đủ để ra T3.

**Giai đoạn 2 (tới điểm kiểm tra 2).**

6. `padding.py`, `variants.py`, kiểm thử kiểu đệm; `pretrain.py`; ba giao thức huấn luyện; `interaction.py`. Chạy T6 (ii), (iii), (iv), và phép thử tương tác của (v).

**Sau điểm kiểm tra 2.**

7. Suy giảm ước lượng N2 và bộ phân loại "mô phỏng hay thật"; một bộ dò điểm mốc để đo; các số đo còn lại; P9, P10.
8. Tùy chọn: thành phần N1 hoặc N3 cùng đối chứng của nó (T5 hoặc T4).

Giai đoạn 1 trả lời L1, L2 và câu hỏi "N5b có cơ sở không" mà không tốn một lần huấn luyện nào.

## 3.10 Rà soát thiết kế, phần 1: năm lỗ hổng về huấn luyện và so sánh

Năm lỗ hổng dưới đây được tìm ra khi tự rà lại thiết kế dưới góc nhìn reviewer. Mỗi lỗ hổng có cách sửa, chi phí, và phần chưa sửa được.

### 1. Dữ liệu huấn luyện ít và đồng nhất (490 ảnh AMI mỗi fold)

**Sửa:** thêm ảnh tai ngoài thực tế vào tập train, giữ AMI làm tập test sạch.

| Nguồn thêm | Số ảnh (đã đếm trên dữ liệu) | Cách dùng | Điểm yếu |
|---|---|---|---|
| EarVN1.0, cạnh ngắn từ 192 px | 1.086 ảnh, 95 người | **Mặc định.** Thu nhỏ 2 lần, đáp án cỡ 96 px trở lên | Ít; chỉ phủ cỡ nhỏ nhất |
| EarVN1.0, cạnh ngắn từ 128 px | 6.386 ảnh, 135 người | Chỉ thêm nếu ablation trên AMI cho thấy có lợi (biên an toàn chỉ 1,33 lần) | Vết nén mức 75 còn lại trong đáp án |
| AWEx | 563 ảnh từ 128 px | **Không dùng để train**; giữ nguyên làm bộ test chưa từng thấy | |
| Ảnh tai cắt từ CelebAMask-HQ (có sẵn nhãn vùng tai) | Chưa đếm | Chỉ để train, nếu bạn tải về và điều khoản cho phép | Tai nhìn từ mặt trước, khác tai nhìn nghiêng |

- Người của EarVN1.0 được chia thành train và test; người dùng để test (S5) không có trong tập train.
- Mốc tinh chỉnh và mô hình đề xuất dùng cùng tập train. Có một ablation "chỉ AMI" so với "AMI + ảnh thêm" để biết ảnh thêm giúp hay hại.
- Nhánh GAN: khởi tạo từ trọng số ESRGAN công bố thay vì học từ đầu; tăng cường dữ liệu cho bộ phân biệt nếu nó vẫn học thuộc.
- **Chưa sửa được:** số ảnh tai nét và lớn đã công bố là có hạn. Với đáp án từ 144 px trở lên, nguồn sạch duy nhất vẫn là AMI. Bài ghi điều này ở Limitations.

### 2. Cách cắt patch làm lệch tỉ lệ nội dung

**Sửa: huấn luyện đúng tỉ lệ.** Có hai cách đúng tỉ lệ. Cách hiển nhiên: huấn luyện trên ảnh HR đã thu về đúng cỡ của ô đang xét. Cách đề xuất: thu ảnh gốc về cạnh ngắn s lấy ngẫu nhiên trong khoảng 96 đến 192 px, rồi cắt một ô 96×96 làm đáp án (ảnh vào 24×24 ở ×4, 48×48 ở ×2), để một mô hình dùng được cho mọi cỡ. Cách sai tỉ lệ (cắt patch từ ảnh 492×702) chỉ là một nhánh đối chứng. Bản 16 và 17 đã so cách đề xuất với cách sai tỉ lệ; phản biện vòng 8 chỉ ra đó là so với người rơm, nên phép so sánh có nghĩa là giữa hai cách đúng tỉ lệ (T6 ii).

### 3. Viền ảnh chi phối toàn bộ ảnh vào

SPAN có khoảng 20 lớp tích chập 3×3, mỗi điểm ảnh ra nhìn xa khoảng 20 px, trong khi ảnh vào chỉ 36×51. **Sửa bằng ba việc, theo thứ tự chi phí:**

1. *Đo trước, không huấn luyện: phép thử ngữ cảnh.* Bản đồ sai số theo khoảng cách tới viền bị lẫn nội dung (viền là tóc và nền, giữa là da), nên được thay bằng phép thử ở T6 (i): cùng một cửa sổ, chạy với ngữ cảnh thật và chỉ với đệm, so tại cùng điểm ảnh. Ở cỡ benchmark, ảnh vào là cả khung hình nên không có ngữ cảnh bên ngoài; phép thử vì vậy chạy trên cửa sổ con của ảnh LR lớn hơn, với bề rộng ngữ cảnh quét tới 24 px (lớn hơn bán kính vùng nhìn khoảng 20 px của SPAN). Nếu không có hiệu ứng thì dừng và bỏ việc 2.
2. *Ablation kiểu đệm:* số 0, lặp viền, phản chiếu; kiểm tra TensorRT trên thiết bị có hỗ trợ và độ trễ có đổi không.
3. *Sâu và rộng ở cùng độ trễ:* ba biến thể của thân (nông và rộng, vừa, sâu và hẹp) có độ trễ đo được ngang nhau.

Một đối chứng thực dụng: trong triển khai thật, bước dò tai có thể cắt rộng hơn vùng tai để cấp ngữ cảnh thật. Cách "cắt rộng rồi bỏ rìa" tốn thêm độ trễ theo diện tích, và là thứ mà kiểu đệm tốt phải hơn ở cùng độ trễ.

Mọi biến thể ở việc 2 và 3 được tiền huấn luyện lại với cùng ngân sách rút gọn như thân tham chiếu (T6 iv), vì đổi kiểu đệm hoặc kiến trúc trên trọng số đã học với đệm số 0 sẽ tạo lệch.

Kèm theo: số đo được báo thêm ở vùng giữa ảnh (bỏ 8 px ảnh vào mỗi cạnh), bên cạnh cách bỏ viền của NTIRE.

Ba việc này là N5b (mục 1.2b), thử ở T6 (mục 1.10). Điểm 2 là giao thức huấn luyện.

### 4. Thiếu mốc cùng độ trễ

**Sửa:** thêm nhánh "mốc nới rộng": tăng số kênh của thân cho tới khi độ trễ đo được bằng độ trễ của mô hình đề xuất (chênh không quá 10%), rồi tinh chỉnh với cùng lịch huấn luyện. Bảng chính có cả hai mốc: mốc cùng thân và mốc cùng độ trễ. Đường trộn cần chạy hai mạng nên chậm gấp đôi; bài ghi rõ điều này vì nó là một lợi thế của cổng chạy một lượt.

### 5. Thước đo điểm mốc còn vòng lặp

**Sửa chính (chờ bạn quyết): gắn nhãn tay một tập nhỏ.** Khoảng 100 ảnh test của AMI, 10 đến 15 điểm mốc dễ xác định, gắn trên ảnh gốc 492×702 rồi đổi tọa độ. Gắn hai lượt (hai người, hoặc một người hai lần) để biết sai số của người. Ước lượng thô là vài giờ công. Khi đó độ lệch điểm mốc được đo so với nhãn người, và bộ dò chỉ còn là công cụ.

**Sửa phụ, làm trong mọi trường hợp:** hai bộ dò học trên hai nửa người tách rời của bộ điểm mốc và khác kiến trúc; N1 phải đạt cả trên bảng kiểm tra gờ, là thước đo không dùng điểm mốc.

### Điểm nhỏ

- Độ tin cậy của LPIPS ở ảnh nhỏ và phiên bản thư viện số đo: đã đưa vào mục 1.9.
- **Khảo sát người xem:** chuẩn bị mẫu đồng ý tham gia; hỏi giáo sư về yêu cầu đạo đức của trường.

## 3.11 Rà soát thiết kế, phần 2: bảy lỗ hổng ở tầng bằng chứng

| # | Lỗ hổng | Cách sửa | Trạng thái |
|---|---|---|---|
| 1 | Ảnh nhỏ thật không có số đo khách quan; câu hỏi "rõ hơn" thưởng cho chi tiết bịa | Khảo sát hai phần, phần có đáp án hỏi "giống hơn" (mục 1.9). Thước đo khách quan trên ảnh nhỏ thật chỉ có E7 | **Sửa một phần.** E7 chờ giáo sư |
| 2 | Sau khi đưa EarVN1.0 vào train, bằng chứng chéo bộ dữ liệu mỏng: AWEx chỉ có 226 ảnh cho đáp án 96 px và 61 ảnh cho 144 px ở biên an toàn 2 lần | Giữ riêng một nhóm người của EarVN1.0 không dùng để train; báo kết quả ở hai mức biên (2 lần và 1,5 lần); ghi giới hạn trong bài | **Sửa một phần.** Không tạo thêm được ảnh lớn |
| 3 | Ngưỡng real-time có thể không ràng buộc gì | Đường chất lượng theo độ trễ là kết quả chính; số đo CPU điện thoại (mục 1.5) | **Đã sửa trong thiết kế.** Con số chốt sau T3 |
| 4 | Ngưỡng đặt khi chưa biết độ nhiễu của phép đo | Chỉnh ngưỡng một lần sau T2 (mục 1.10) | **Đã sửa** |
| 5 | Mốc so sánh chưa công bằng về tiền huấn luyện và công thức tinh chỉnh | Dò tốc độ học cho từng mốc; cột dữ liệu tiền huấn luyện (mục 1.8) | **Sửa một phần.** Huấn luyện lại mọi mốc từ cùng nguồn là quá đắt |
| 6 | Benchmark phụ thuộc AMI, giấy phép cấm phân phối bản sửa đổi | Kiểm ở T1: trang tải còn hoạt động không; xin phép dùng ảnh trong hình. Bài công bố script và danh sách file | **Chưa sửa.** Chờ T1 |
| 7 | Chưa đếm số lần huấn luyện | Bảng ngân sách ở mục 1.10 (khoảng 205 lần tinh chỉnh, cộng tiền huấn luyện) | **Đã đếm.** Khả thi hay không chờ đo thời gian mỗi lần |

**Cặp ảnh "front" và "zoom" của AMI: đã kiểm, không dùng được.** Khớp đặc trưng SIFT và ước lượng phép đồng dạng trên cả 100 cặp (05/10/2026): độ phóng giữa hai ảnh có trung vị 1,18 lần (khoảng 5% đến 95%: 1,15 đến 1,22), không phải 1,5 lần như tỉ lệ tiêu cự; 79 trong 100 cặp khớp được với ít nhất 15 điểm, sai số khớp khoảng 1 px. Độ phóng 1,18 lần là quá nhỏ để làm cặp ảnh SR ở ×2 hay ×4. Ý này bị bỏ.

## 3.12 Trạng thái của 12 lỗ hổng

| Trạng thái | Lỗ hổng |
|---|---|
| Đã có cách sửa trong thiết kế (7) | Cắt patch lệch tỉ lệ; viền chi phối ảnh vào; thiếu mốc cùng độ trễ; ngưỡng real-time không ràng buộc; ngưỡng đặt khi chưa biết độ nhiễu; chưa đếm số lần huấn luyện; khảo sát thưởng cho chi tiết bịa |
| Sửa một phần, phần còn lại ghi ở Limitations (3) | Dữ liệu train ít; bằng chứng chéo bộ dữ liệu mỏng; công bằng về tiền huấn luyện giữa các mốc |
| Còn mở (2) | Vòng lặp của thước đo điểm mốc (chờ quyết định gắn nhãn tay); phụ thuộc giấy phép và trang tải AMI (chờ T1) |

"Đã có cách sửa" nghĩa là thiết kế đã có câu trả lời, chưa phải đã có kết quả. Ảnh nhỏ thật vẫn chưa có thước đo khách quan nào ngoài E7.

## 3.13 Cần từ bạn để bắt đầu

1. Quyền đọc `earsr_project_span`, để rà và lấy lại phần định nghĩa mô hình.
2. Trọng số công bố của các mô hình ở mục 1.8 (không tải được từ đây), đặt trong một thư mục đã kết nối.
3. Đời máy Jetson Nano, phiên bản JetPack, và điện thoại Android.
4. Bộ ảnh tai gắn 55 điểm mốc, nếu điều khoản cho phép (T1); chưa cần ở giai đoạn 1.
5. Nơi đặt repo mới: thư mục trên máy bạn và, nếu muốn, một repo GitHub riêng.
6. Quyết định có gắn nhãn tay khoảng 100 ảnh AMI hay không (mục 3.10, điểm 5): **dời ra sau điểm kiểm tra 2**, chỉ cần khi làm N1.
7. DIV2K (hoặc bộ tiền huấn luyện tương đương), cho các lần tiền huấn luyện ở T6 và cho kiểm chứng ngoài ảnh tai.
8. CelebAMask-HQ, nếu bạn muốn dùng làm nguồn ảnh train thêm.

---

## 3.14 Dữ liệu, script và file kết quả (bản đối chiếu; đã hiện thực trong code)

Mục này chép từ `docs/DATA_AND_OUTPUTS.md` của project `earsr_rt` và là bản đối chiếu giữa việc trong kế hoạch, script và file kết quả. Mã cho mọi dòng dưới đây đã viết xong và qua kiểm thử trên dữ liệu giả; chưa có lần chạy nào trên GPU (trạng thái từng phần: `docs/STATUS.md`). Lệnh đầy đủ theo thứ tự chạy: `docs/RUNBOOK.md`.


### 1. Đặt dữ liệu

Bố cục mặc định, tính từ thư mục gốc của project (thư mục `data/` không đưa lên git):

```
data/raw/
  AMI/                 BẮT BUỘC. 700 file, tên đúng mẫu NNN_<góc>_ear.jpg
                       NNN = 000..099 (mã người); góc = back, down, front, left, right, up, zoom
                       mọi ảnh cỡ 492×702 (rộng × cao). Để phẳng hoặc trong thư mục con đều được;
                       file không đúng mẫu tên bị bỏ qua.
  DIV2K_train_HR/      800 ảnh PNG, để phẳng (0001.png ... 0800.png). Cần cho tiền huấn luyện (T6 iv).
  DIV2K_valid_HR/      100 ảnh PNG, để phẳng. Validation của tiền huấn luyện; phép thử ngoài ảnh tai (T6 v).
  EarVN1.0/            <mã người>/<ảnh>   (ví dụ EarVN1.0/001.ALI_HD/001 (1).jpg). Cần cho N2 và S5.
  awex/                <mã người>/<ảnh>   (ví dụ awex/001/01.png). Chỉ dùng để test.
  landmarks/           ảnh và file .pts cùng tên, cùng thư mục (tùy chọn; chỉ khi làm N1 hoặc đo điểm mốc).
weights/               trọng số công bố; tạo bằng: bash scripts/get_weights.sh
```

Với EarVN1.0 và AWEx, **mã người là tên thư mục chứa ảnh**. Nếu bản tải về có thêm một tầng thư mục (ví dụ
`EarVN1.0/Images/001.ALI_HD/...`) thì trỏ `--root` vào tầng chứa các thư mục người. File chú thích (json, txt)
nằm lẫn trong đó không sao; script chỉ đọc `.jpg .jpeg .png .bmp`.

Dữ liệu ở nơi khác cũng được: mọi script nhận đường dẫn qua tham số (`--ami-raw`, `--root`, `--train-dir`...).

Kiểm tra trước khi chạy:

```bash
python scripts/check_data.py            # hoặc: --root /nơi/khác
```

### 2. Thư mục do project tạo ra

| Thư mục | Nội dung | Tạo bởi |
|---|---|---|
| `data/bench/ami/` | Benchmark AMI: `hr96/ hr144/ hr192/ hr244/` (ảnh đáp án PNG), `lr/hr<cỡ>_x<hệ số>_<suy giảm>/` (ảnh vào), `manifest.csv` | `build_benchmark.py`; ảnh LR do `evaluate.py`, `train.py` tự sinh khi cần |
| `data/bench/earvn/`, `awex_s20/`, `awex_s15/`, `div2k/` | Benchmark ngoài thực tế, cùng dạng; thêm `scan.csv`, `dropped_hr<cỡ>.csv` (ảnh bị loại và lý do) | `build_wild.py` |
| `runs/<mã lần chạy>/` | Một lần huấn luyện: `config.json`, `log.csv`, `summary.json`, `ckpt/best.pt`, `ckpt/last.pt`, `test_*.csv` | `train.py`, `pretrain.py` |
| `results/` | Mọi bảng kết quả (mục 3 dưới đây). Thư mục này được đưa lên git | các script đánh giá và tổng hợp |
| `deploy_jetson/onnx/` | File ONNX và `index.json` để chép sang thiết bị | `export_for_device.py` |
| `jobs/` | Danh sách lệnh và trạng thái hàng đợi | `make_jobs.py`, `run_queue.py` |
| `paper/tables/`, `paper/figures/` | Bảng LaTeX; hình PDF | `make_tables.py`, `compare_table.py`, `make_figures.py` |
| `results/tables/` | Bảng so sánh dạng CSV (kèm khoảng tin cậy và chênh lệch ghép cặp) | `compare_table.py` |

### 3. Việc nào, script nào, file kết quả nào

Hai loại file: **theo từng ảnh** (một dòng mỗi ảnh; là dữ liệu gốc, mọi thống kê tính lại được từ đây) và
**tổng hợp** (trung bình, khoảng tin cậy, kết luận).

#### Giai đoạn 1, không huấn luyện (`bash scripts/run_stage1.sh data/raw/AMI data/raw/DIV2K_valid_HR`)

| Việc trong kế hoạch | Script | File theo từng ảnh | File tổng hợp (đọc file này) |
|---|---|---|---|
| P1 dựng benchmark | `build_benchmark.py` | `data/bench/ami/manifest.csv` | |
| **T2** mô hình có sẵn, bicubic và JPEG 75 | `evaluate.py` (tên cũ `run_t2.py`) rồi `summarize_t2.py` | `results/t2/<mô hình>__hr<cỡ>_x<hệ số>_<suy giảm>.csv` | `results/t2_summary/T2_summary.md` (đọc trước), `quality.csv` (bảng chất lượng), `ranking.json` (thứ hạng có đảo không, τ-b), `mde.csv` (mức chênh nhỏ nhất phát hiện được, để chỉnh ngưỡng) |
| **T6 (i), (v)** phép thử ngữ cảnh | `run_t6_context.py` rồi `summarize_context.py` | `results/t6_context/<mô hình>__<bộ ảnh>__<cấu hình>.csv` | `results/t6_summary/T6_context_summary.md`, `context.csv` |
| **T3** độ trễ trên máy này (so tương đối) | `bench_local.py` | | `results/latency_local.csv` |
| **T3** độ trễ trên Jetson | `export_for_device.py`, rồi trên Jetson `deploy_jetson/bench_trtexec.py` | `deploy_jetson/logs/` (log thô) | `latency_trt.csv` trên Jetson; chép về thành `results/latency_jetson.csv` |
| Bảng cho bài (tự động) | `make_tables.py` | | `paper/tables/t2_hr<cỡ>_x<hệ số>.tex`, `latency_*.tex`, `t6_context.tex`, `n2_realism.tex`, `viewer_<phần>.tex` |
| Bảng so sánh giữa các phương pháp (bảng 4, 6, 7, 8, 11) | `compare_table.py` | các file theo ảnh của từng phương pháp | `results/tables/<tên>.csv`, `paper/tables/<tên>.tex` |
| Hình cho bài | `make_figures.py` (tự động: hình 2, 3, 4; lệnh con `cost`: hình 5; `grid`: hình 1, 6) | | `paper/figures/*.pdf` và `*.png` |

Cột của file theo từng ảnh: `model, dataset, subject, view, key, fold, scale, tier, degrade, psnr_y, ssim_y,
psnr_rgb, psnr_y_c, ssim_y_c` (hậu tố `_c` = vùng giữa ảnh), `ms_ssim_y, ms_ssim_scales, gmsd, grad_psnr, lr_psnr_y`
(luôn có), `lpips, dists` (nếu cài được), `stlpips, topiq_fr, fsim, vif, pieapp` (`--more-metrics`), `host_ms`; thêm
`ridge_false, ridge_missed, ridge_f1` (`--ridge`), `lm_dev, lm_floor_noise, lm_floor_jpeg` (`--landmark-ckpt`),
`psnr_y_box` (`--boxes`), `niqe, maniqa, musiq, clipiqa, ntire_score` (`--nr`).

#### Giai đoạn 2, có huấn luyện

| Việc trong kế hoạch | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| Mọi lần huấn luyện | `train.py` (qua `make_jobs.py` + `run_queue.py`) | `runs/<mã>/test_hr<cỡ>_x<hệ số>_<suy giảm>.csv` (chỉ người test của fold đó) | `runs/<mã>/summary.json`, `runs/<mã>/log.csv` (đường học); **`results/runs.csv`** (sổ ghi: một dòng mỗi lần chạy, trạng thái, PSNR validation và test) |
| **N2 sớm** (khối `n2`): ba kiểu suy giảm lúc huấn luyện, fold 2 | `make_jobs.py n2 --degrade-params ...`, rồi `evaluate.py --runs` trên EarVN1.0 và trên AMI với từng kiểu | `results/n2/<mã>__hr<cỡ>_x4_<suy giảm>.csv` | `check_criteria.py gain --name n2-...` → `results/criteria/` |
| **Kiểu đệm, phép thử rẻ** (khối `pad`): trọng số công bố, ba kiểu đệm | `make_jobs.py pad` | `runs/T6pad_*/test_*.csv` | `check_criteria.py n5b-gain` |
| Dò tốc độ học (khối `lr`) | như trên | không chấm test | cột `best_val_psnr_y` trong `results/runs.csv` |
| **T6 (ii)** ba giao thức → N5a | khối `t6ii`, rồi `check_criteria.py n5a` | `runs/T6ii_*/test_*.csv` | `results/criteria/n5a.json` |
| **T6 (iii)** có cần tiền huấn luyện | khối `t6iii`, rồi `check_criteria.py gain` | `runs/T6iii_*/test_*.csv` | `results/criteria/gain.json` (đặt tên bằng `--name`) |
| Tiền huấn luyện rút gọn (khối `pre`) | `pretrain.py` | | `runs/PRE_*/log.csv`, `summary.json`; checkpoint ở `ckpt/best.pt`, `ckpt/step<N>.pt` (50% và 100% ngân sách) |
| **T6 (iv)** năm biến thể → N5b (b) | khối `t6iv`, rồi `check_criteria.py n5b-gain` | `runs/T6iv_*/test_*.csv` | `results/criteria/n5b-gain.json` |
| **T6 (v)** phép thử tương tác → N5b (c) | `build_wild.py --subject-per-image`, `evaluate.py --runs runs/PRE_*`, `check_criteria.py n5b-interaction` | `results/t6v/<mã>__hr<cỡ>_x4_bic.csv` | `results/criteria/n5b-interaction.json` |
| Ghép độ trễ (mốc nới rộng; hình dạng wide, deep) | `match_latency.py` | | in ra màn hình: variant `zero-cNN` cần dùng |
| **S2** ô chính, 5 fold | khối `s2` | `runs/S2_*/test_*.csv` | `results/runs.csv`; so sánh bằng `check_criteria.py gain` |
| **S4** đường chất lượng theo cỡ ảnh vào | `run_size_sweep.py` | `results/size_sweep/per_image.csv` | `results/size_sweep/summary.csv` |

#### N2 và ảnh ngoài thực tế

| Việc | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| Chia vai người trong EarVN1.0 | `fit_degradation.py` (lần đầu tự tạo) | | `splits/earvn_roles.json` (commit) |
| Ước lượng suy giảm (N2) | `fit_degradation.py` | | `configs/degrade_estimated.json` (tham số, commit), `configs/degrade_estimated.diagnostics.json` |
| N2 (a) bộ phân loại "mô phỏng hay thật" | `realism_classifier.py` | | `results/n2_realism.json` |
| **S5** test trên EarVN1.0, AWEx | `build_wild.py`, `evaluate.py --folds none` | `results/s5/<mô hình>__hr<cỡ>_x4_<suy giảm>.csv` | tính bằng `check_criteria.py gain` |

#### Tùy chọn (sau điểm kiểm tra 2)

| Việc | Script | File theo từng ảnh | File tổng hợp |
|---|---|---|---|
| **T4** cổng oracle → giữ hay bỏ N3 | `run_t4_oracle.py` | `results/t4/t4_per_image.csv` | `results/t4/t4_summary.json` (khóa `decision`) |
| N3 hai giai đoạn | `train.py --objective n3s1A` rồi `n3s2` | `runs/<mã>/test_*_opF.csv`, `_opP.csv`, `_headS.csv`, `_headT.csv` | `runs/<mã>/operating_points.json`; `check_criteria.py n3` → `results/criteria/n3.json` |
| Bộ dò điểm mốc | `train_landmarks.py` | | `weights/lm_*.pt` |
| Nhãn giả, hộp bao, sàn nhiễu | `landmark_tools.py pseudo / boxes / floor` | | `data/ami_lm.npz`, `data/bench/ami/boxes.csv`, sàn nhiễu in ra màn hình |
| **T5** đầu phụ N1 | `train.py --objective aux`, `evaluate.py --landmark-ckpt --ridge`, `check_criteria.py n1` | `runs/<mã>/test_*.csv`, file của `evaluate.py` | `results/criteria/n1.json` |
| **S7** khảo sát người xem | `evaluate.py --save-sr`, `viewer_study.py make / analyze` | `results/viewer/<phần>/answers_long.csv` | `results/viewer/<phần>/analysis.csv`; `study.html` gửi người xem, `key.csv` giữ lại |

### 4. Mã lần chạy

`{thí nghiệm}_{thân}-{biến thể}_{tiền huấn luyện}_{giao thức}_x{hệ số}_hr{cỡ}_{suy giảm}_f{fold}`

Ví dụ `T6iv_span-replicate_ps20_rand_x4_hrall_bic_f3`: phép thử T6 (iv), thân SPAN đệm lặp viền, tiền huấn
luyện rút gọn 20%, giao thức tỉ lệ ngẫu nhiên, ×4, một mô hình cho mọi cỡ, suy giảm bicubic, fold 3.
Phần sau dấu `+` trong biến thể là nhãn: `+gan`, `+ldl`, `+aux`, `+xearvn` (có ảnh thêm), `+lr1e-4` (`--tag`).

### 5. File nào quyết định điều gì

| Quyết định trong kế hoạch | Đọc file |
|---|---|
| Điểm kiểm tra 1: SR có dư địa không; thứ hạng có đảo không | `results/t2_summary/T2_summary.md`, `ranking.json` |
| Điểm kiểm tra 1: hiệu ứng viền có thật không | `results/t6_summary/T6_context_summary.md` |
| Điểm kiểm tra 1: mô hình nào real-time; kiểu đệm nào chạy được | `results/latency_jetson.csv` |
| Chỉnh ngưỡng một lần sau T2 | `results/t2_summary/mde.csv` → sửa `configs/criteria.yaml`, ghi mục `history` |
| Điểm kiểm tra 2: giao thức, N5b, N2 | `results/criteria/n5a.json`, `n5b-gain.json`, `n5b-interaction.json`, `results/n2_realism.json` |
| Giữ hay bỏ N3, N1 | `results/t4/t4_summary.json`, `results/criteria/n1.json`, `n3.json` |

Mỗi file trong `results/criteria/` có khóa `pass` (true, false) và ghi kèm phiên bản `criteria.yaml` đã dùng.

---

# PHỤ LỤC 1. TRẢ LỜI PHẢN BIỆN VÒNG 5

*Mục C và D của phụ lục này đã được phản biện trả lời ở vòng 6; xem Phụ lục 2.*

## A. Các điểm đồng ý và phần đã sửa

| # | Ý phản biện | Sửa ở |
|---|---|---|
| 1 | Tạp chí sinh trắc mâu thuẫn với việc không làm nhận dạng | 1.1: mặc định là tạp chí xử lý ảnh; phép kiểm tra "không gây hại" chờ giáo sư. Bỏ câu "N1 mạnh nhất cho tạp chí sinh trắc" |
| 2 | "Đầu tiên" là điểm tựa yếu; luận điểm nên là phép so sánh với mô hình đa dụng đã tinh chỉnh | 1.2; Phần 2 (tiêu đề, abstract, mục "do not claim") |
| 3a | Phép thử đầu phụ cần ít nhất 3 seed và ngưỡng ghi trước | 1.2 (tiêu chí N1), 1.10 (T5) |
| 3b | Dùng bộ dò điểm mốc làm thước đo, với bộ dò khác bộ sinh nhãn; giữ bảng gờ làm đối chứng | 1.9 |
| 4a | N2 chưa có cách chứng minh; bộ phân loại "mô phỏng hay thật" | 1.7, có một điều chỉnh ở mục B |
| 4b | Khảo sát người xem thành bắt buộc | 1.9, 1.13 |
| 5a | Quy tắc độ trễ có thể buộc đổi thân; chưa nên viết thân là SPAN | 1.4 |
| 5b | Ngưỡng real-time cần con số ghi trước | 1.5 |
| 6 | Phần 2 vẫn là bài về cổng điểm ảnh | Phần 2 viết lại, trung lập với thành phần |
| 7 | N1 và N2 chưa có tiêu chí đạt | 1.2 |
| 8 | Mục rủi ro còn ghi "ba thân" | 1.12 |
| 9 | Tiến độ chưa cộng G2, G3; cần một ô chính | 1.10, 1.11 |
| 10 | Cross-validation 5 fold ghép với seed | 1.6 |
| 11 | Tiêu chí 3 phát biểu "ở cùng mức PSNR" | 1.2, 1.4 |
| 12 | Ngưỡng số cho hai điểm kiểm tra | 1.2, 1.10 |

## B. Các điểm đề nghị điều chỉnh

| # | Ý phản biện | Đề nghị của chúng tôi |
|---|---|---|
| R1 | Bộ phân loại phân biệt ảnh LR mô phỏng (từ AMI) với ảnh LR thật của EarVN1.0 | AMI là ảnh trong nhà, EarVN1.0 là ảnh ngoài thực tế, nên bộ phân loại có thể tách hai nhóm bằng nội dung (nền, ánh sáng, tóc) thay vì bằng suy giảm. Chúng tôi tạo ảnh mô phỏng từ ảnh lớn của chính EarVN1.0 và chỉ cho bộ phân loại nhìn patch nhỏ |
| R2 | Phép kiểm tra "không gây hại" bằng mạng nhận dạng đóng băng | Đồng ý là rẻ và nhiều khả năng cho kết quả mong muốn. Chưa đưa vào kế hoạch vì đề bài của giáo sư bỏ phần nhận dạng; đây là quyết định của giáo sư |

## C. Những gì chúng tôi chưa nhận được

Bạn viết rằng đã trả lời bốn câu hỏi ở mục E của bản trước "ở lượt trước". Chúng tôi chỉ nhận được phần gợi ý sáu hướng làm bài mạnh lên, không nhận được các câu trả lời đó. Ba đề nghị bạn nhắc tên (cross-validation, "ở cùng mức PSNR", ngưỡng số) đã được đưa vào theo cách hiểu của chúng tôi; **các ngưỡng số trong bản này là do chúng tôi tự đề xuất**, có thể khác con số bạn đã nêu. Nếu bạn gửi lại, chúng tôi sẽ đối chiếu.

## D. Câu hỏi gửi lại

1. Các ngưỡng đề xuất có hợp lý không: N1 giảm độ lệch điểm mốc ít nhất 5% và hơn 2 lần độ lệch chuẩn giữa seed; N3 giảm LPIPS ít nhất 10% tương đối ở PSNR chênh không quá 0,1 dB; thứ hạng coi là đảo khi Kendall τ dưới 0,7.
2. Ngưỡng real-time ở mục 1.5 (33 ms, FP16, ảnh vào 48×68, chỉ tính lượt chạy của mô hình) có đủ chặt không.
3. Với cross-validation ghép seed, phương sai gộp cả seed lẫn cách chia. Bạn có yêu cầu tách hai nguồn này không (ví dụ thêm vài seed trên một fold)?

---

# PHỤ LỤC 2. TRẢ LỜI PHẢN BIỆN VÒNG 6

## A. Các điểm đồng ý và phần đã sửa

| # | Ý phản biện | Sửa ở |
|---|---|---|
| 1 | N1: bỏ "hơn 2 lần độ lệch chuẩn giữa seed"; dùng so sánh ghép cặp theo ảnh, bootstrap theo người, gộp seed | 1.2b, 1.9 |
| 2 | N1: đo sàn nhiễu của bộ dò điểm mốc trước | 1.9, T5; có một điều chỉnh ở mục B |
| 3 | N1 không được làm xấu bảng kiểm tra gờ và LPIPS | 1.2b, với biên 2% tương đối |
| 4 | N3: ngân sách tụt PSNR 0,5 dB; điều kiện có nghĩa là hơn đường trộn ở cùng PSNR | 1.2b, 1.4 |
| 5 | Kendall τ: không để thứ hạng đảo do nhiễu | 1.10 (T2), 1.9; dùng cả hai cách bạn nêu |
| 6 | Độ trễ: chế độ nguồn, số lượt làm nóng, số lượt đo, phân vị 95 | 1.5 |
| 7 | Tách phương sai (E9) đặt ở phụ lục | 1.10 |
| 8 | Chống rò rỉ trong cross-validation | 1.6; có một điều chỉnh ở mục B |
| 9 | Công bằng cho mốc: cùng kiểu suy giảm khi thử N2; nói rõ N1 dùng thêm dữ liệu có nhãn | 1.2b |
| 10 | Mục 1.3 điểm 3: ghi cả nén JPEG bên cạnh khung hình | 1.3, kèm số đo (mục B, R6) |

## B. Các điểm đề nghị điều chỉnh hoặc bổ sung

| # | Ý phản biện | Đề nghị của chúng tôi |
|---|---|---|
| R3 | "Mức 5% phải lớn hơn sàn nhiễu của bộ dò" | Sàn nhiễu là dao động trên từng ảnh, còn mức giảm là trung bình trên hàng trăm ảnh; một mức giảm trung bình nhỏ hơn dao động từng ảnh vẫn có thể là thật, và khoảng tin cậy ghép cặp đã xét việc đó. Điều sàn nhiễu thật sự cho biết là thước đo có dải động hay không. Chúng tôi đổi thành: độ lệch của thân tinh chỉnh phải ít nhất gấp 2 lần sàn; nếu không, bỏ thước đo này ở cỡ ảnh đó và xét N1 bằng bảng gờ |
| R4 | Bootstrap theo người, gộp seed | Bootstrap theo người chỉ bắt được dao động do chọn ảnh, không bắt được dao động do huấn luyện khi chỉ có 3 seed. Chúng tôi thêm điều kiện mức giảm cùng chiều ở từng seed (ở T5 là từng cặp fold và seed) |
| R5 | Tham số độ mờ của N2 chỉnh trên validation của fold | Các tham số này khớp với thống kê của EarVN1.0, không dùng người nào của AMI. Chúng tôi chốt chúng một lần trước mọi lần huấn luyện, và tách người của EarVN1.0 dùng để khớp khỏi người dùng để kiểm (bộ phân loại, khảo sát người xem) |
| R6 | "Nén JPEG mức 75 tạo ra đúng dạng chênh lệch đó" | Đúng về chiều, nhỏ hơn về độ lớn. Trên AMI ở 192 px, nén mức 75 làm tăng số đo ×2 thêm 1,1 dB; khung hình sát tai thêm 2,6 dB; phóng 2 lần thêm 8,5 dB. Nén cộng khung hình cho 44,6 dB, gần bằng EarVN1.0 (45,0 dB), nên hai yếu tố này đủ giải thích trung vị của EarVN1.0 mà không cần giả thuyết phóng to |
| R7 | N3: "khoảng tin cậy không chồng" | Thiết kế là ghép cặp, nên chúng tôi dùng khoảng tin cậy của chênh lệch, thống nhất với N1. Thêm mức tối thiểu 3% LPIPS để tránh "có ý nghĩa nhưng không đáng kể" (oracle ở T4 phải hơn 5%) |

**Ba điểm chúng tôi tự thêm.**

1. *Nhánh đối chứng cho N1.* Nói rõ là dùng thêm dữ liệu thì chưa đủ; reviewer sẽ hỏi phần hơn đến từ nhãn hay từ ảnh. Thêm nhánh "thân + cùng các ảnh đó, không nhãn".
2. *Hai fold giữ kín.* T4 và T5 quyết định giữ hay bỏ thành phần dựa trên người test, tức là một dạng chọn mô hình trên tập test mà quy tắc chống rò rỉ theo fold không chặn được. Các phép thử này chỉ chạy trên fold 2, 3, 4; fold 1 và 5 báo cáo riêng.
3. *So sánh ở độ trễ ngang nhau.* Nếu ước lượng của bạn đúng (SPAN vài ms, ngưỡng 33 ms), mọi mô hình nhẹ đều đạt ngưỡng với khoảng dư lớn, và mốc mạnh nhất trong ngân sách có thể không phải SPAN. So sánh chính vì vậy được ghép theo độ trễ đo được (chênh không quá 10%).

## C. Câu hỏi gửi lại

1. Điều kiện dải động (độ lệch của thân ít nhất gấp 2 lần sàn nhiễu) có thay được điều kiện "5% lớn hơn sàn" không.
2. Ngân sách 0,5 dB và mức tối thiểu 3% LPIPS so với đường trộn: chấp nhận được không.
3. Hai fold giữ kín có đủ để trả lời câu hỏi "chọn thành phần trên tập test" không, hay cần một tập giữ riêng hẳn (sẽ làm giảm số người test).
4. Bạn đánh giá bài mô hình thuần ở mức ranh giới cho Q1, lên trung bình nếu thắng mốc trên hai thân và có nhánh suy giảm kiểu EarVN1.0. Ngoài hai điều kiện đó, còn thí nghiệm nào trong danh sách E1 đến E10 mà theo bạn làm thay đổi mức đánh giá này?

---

# PHỤ LỤC 3. THAY ĐỔI CỦA BẢN 16 (SAU PHẢN BIỆN VÒNG 6)

*Phụ lục này giữ nguyên như đã gửi ở bản 16. Phản biện vòng 7 đã trả lời các câu hỏi ở mục C; một số nội dung ở mục A đã đổi theo đó (biên an toàn của EarVN1.0, ngân sách 10 ms, số lần huấn luyện, tiêu chí của N5). Bản hiện hành là Phần 1 đến 3 và Phụ lục 4.*

Phản biện vòng 6 đề nghị sửa thẳng vào bản 15. Sau khi sửa, chúng tôi thêm phần thiết kế project và tự rà soát lại dưới góc nhìn reviewer. Lượt rà soát đó tìm ra 12 lỗ hổng mà chưa vòng phản biện nào nêu. Phụ lục này liệt kê những gì đã đổi để bạn không phải đọc lại toàn bộ.

## A. Thay đổi so với bản 15

| # | Thay đổi | Lý do | Ở đâu |
|---|---|---|---|
| 1 | Chỉnh ngưỡng N1, N3, thứ hạng; giao thức đo độ trễ; chống rò rỉ; công bằng cho mốc | Phản biện vòng 6 | Phụ lục 2 |
| 2 | Thêm Phần 3: pipeline 10 giai đoạn, cấu trúc repo, tiền xử lý, kiểm thử, bản đồ thí nghiệm | Cần thiết kế trước khi viết code | Phần 3 |
| 3 | **Điểm mới ứng viên N5** và phép thử T6: thiết kế theo chế độ ảnh vào cực nhỏ (kiểu đệm viền, sâu và rộng ở cùng độ trễ, huấn luyện đúng tỉ lệ) | Ảnh vào 24 đến 48 px nhỏ hơn vùng nhìn của mạng; N1, N2, N3 đều là ý vay từ lĩnh vực khác | 1.2b, 1.4, 1.10, 3.10 |
| 4 | Huấn luyện đúng tỉ lệ thay cho cắt patch từ ảnh gốc | Cắt từ ảnh 492×702 dạy mô hình vân da ở độ phóng lớn, không phải cái tai ở cỡ 144 px | 3.10, điểm 2 |
| 5 | EarVN1.0 thêm vào tập train (6.386 ảnh, biên 1,33 lần); AWEx không bao giờ train | 490 ảnh AMI mỗi fold là quá ít và đồng nhất | 1.6, 3.10 điểm 1 |
| 6 | Mốc nới rộng cho bằng độ trễ của mô hình đề xuất | Mô hình đề xuất có thêm hai đầu và cổng | 1.8, 3.10 điểm 4 |
| 7 | Đường chất lượng theo độ trễ và một ngân sách chặt | Ngưỡng 33 ms có thể không ràng buộc gì ở ảnh nhỏ | 1.5 |
| 8 | Khảo sát người xem hai phần; phần có đáp án hỏi "ảnh nào giống đáp án hơn" | Câu hỏi "rõ hơn" thưởng cho chi tiết bịa, không chứng minh được L5 | 1.9 |
| 9 | Chỉnh ngưỡng một lần sau T2, dựa trên mức chênh nhỏ nhất phát hiện được | Các ngưỡng hiện đặt khi chưa biết độ nhiễu của phép đo | 1.10 |
| 10 | Dò tốc độ học cho từng mốc; cột dữ liệu tiền huấn luyện | Cùng một công thức tinh chỉnh có thể thiên vị | 1.8 |
| 11 | Số đo báo ở ba phạm vi; kiểm độ tin cậy của LPIPS ở ảnh nhỏ | Tiêu chí N3 dựa vào LPIPS ở ảnh 144×204 | 1.9 |
| 12 | Đếm số lần huấn luyện: khoảng 330 | Chưa từng đếm | 1.10 |
| 13 | Kịch bản A+ và đánh giá thẳng về tính mới | N5 có thể thành lõi của bài | 1.2b |

## B. Những gì chưa sửa được

1. Với đáp án từ 144 px trở lên, nguồn ảnh tai sạch đã công bố duy nhất vẫn là AMI. Bằng chứng ngoài thực tế có đáp án chủ yếu ở cỡ 96 px.
2. Ảnh nhỏ thật không có thước đo khách quan nào ngoài phép kiểm tra nhận dạng (E7), đang chờ giáo sư.
3. Các mốc dùng trọng số công bố học trên dữ liệu khác nhau; huấn luyện lại tất cả từ cùng nguồn là quá đắt.
4. Thước đo điểm mốc còn vòng lặp nếu không gắn nhãn tay.
5. N5 và mọi điểm mới khác vẫn là giả thuyết.

## C. Câu hỏi gửi phản biện

1. **N5 có đủ để đổi mức đánh giá "ranh giới cho Q1" không**, nếu cả ba tiêu chí của nó đạt? Bạn có biết công trình nào đã nghiên cứu kiểu đệm viền hoặc tỉ lệ sâu và rộng của mạng SR ở ảnh vào nhỏ hơn vùng nhìn không?
2. Tiêu chí (c) của N5 (thứ hạng biến thể ở ảnh vào 36 px khác thứ hạng ở ảnh vào lớn) có phải cách đúng để chứng minh "riêng cho ảnh nhỏ" không, hay có phép thử tốt hơn?
3. Thêm EarVN1.0 vào tập train với biên an toàn 1,33 lần: rủi ro đáp án mềm có đáng không, hay nên chỉ dùng 1.086 ảnh ở biên 2 lần?
4. Khảo sát người xem hai phần có đủ cho L5 không. Nhãn điểm mốc gắn tay trên 100 ảnh có đáng công không.
5. Ngân sách chặt 10 ms cho SR (phần còn lại của khung 33 ms sau bước dò tai) có phải cách đặt ngưỡng thuyết phục không.
6. Khoảng 330 lần huấn luyện: nếu phải cắt, bạn cắt khối nào trước (bảng ở mục 1.10)?
7. Cặp ảnh "front" và "zoom" của AMI (cùng một tai, hai tiêu cự) có đáng thử làm cặp ảnh thật không, hay sai số căn chỉnh sẽ làm nó vô dụng?

---

# PHỤ LỤC 4. TRẢ LỜI PHẢN BIỆN VÒNG 7

*Phụ lục này giữ nguyên như đã gửi ở bản 17. Phản biện vòng 8 đã trả lời các câu hỏi ở mục E; vai trò của N5a đã đổi theo đó (Phụ lục 5).*

## A. Các điểm đồng ý và phần đã sửa

| # | Ý phản biện | Sửa ở |
|---|---|---|
| 1 | Tiền huấn luyện của các biến thể N5 chưa được tính; so một thân tự tiền huấn luyện ngắn với thân công bố là không công bằng; đổi kiểu đệm trên trọng số cũ tạo lệch | T6 (iv): cả thân tham chiếu lẫn biến thể được tiền huấn luyện lại với cùng ngân sách rút gọn. Bảng ngân sách có dòng tiền huấn luyện. Có hai bổ sung ở mục B |
| 2 | Công thức huấn luyện nằm ở bên nào của phép so sánh | N5 tách thành N5a (công thức, chứng minh trên các mốc) và N5b (kiến trúc, so với mốc đã dùng công thức tốt). Mốc báo ở hai công thức. Có một điều chỉnh ở mục B |
| 3 | Tiêu chí (a) lẫn nội dung; phép thử ngữ cảnh sạch hơn | T6 (i), mục 3.10 điểm 3. Có một lưu ý ở mục B |
| 4 | Năm đóng góp nhỏ thay vì hai đóng góp sâu; lấy "lệch chế độ" làm xương sống; N3 cắt đầu tiên | 1.2b: xương sống là N5a, N5b, N2; tối đa một trong N1, N3. Ngưỡng oracle ở T4 nâng từ 5% lên 10% |
| 5 | Mức 3% LPIPS phải lớn hơn mức chênh nhỏ nhất phát hiện được | Đã nằm trong quy tắc chỉnh ngưỡng sau T2 (1.10) |
| 6 | E1 phải vào phần lõi | Chuyển vào S4 |
| 7 | Tiêu chí (c): thay "thứ hạng khác" bằng phép thử tương tác | 1.2b |
| 8 | EarVN1.0: mặc định chỉ dùng 1.086 ảnh ở biên 2 lần | 1.6, 3.10 điểm 1. Lựa chọn 1,33 lần ở bản 16 là sai: chúng tôi đã đo mức nén 75 ở 90% ảnh EarVN1.0 nhưng không tính tới việc nó còn lại trong đáp án |
| 9 | Ngân sách 10 ms chỉ thuyết phục khi có số đo thật của một bộ dò | 1.5: đường chất lượng theo độ trễ là kết quả chính |
| 10 | Thứ tự cắt: S4 (3 fold, ×2 một cỡ); chỉ tinh chỉnh nhóm real-time; ablation của thân thứ hai | 1.8, 1.10. Áp luôn hai mục đầu vào ngân sách mặc định |
| 11 | Cặp "front" và "zoom": nửa ngày kiểm tra | Đã kiểm, kết quả ở mục C |

## B. Các điểm bổ sung hoặc điều chỉnh

| # | Ý phản biện | Đề nghị của chúng tôi |
|---|---|---|
| R8 | So các biến thể trong nhóm cùng ngân sách tiền huấn luyện rút gọn | Đồng ý cho bước chọn biến thể. Nhưng nếu dừng ở đó, phép so sánh cuối của bài là với một mốc bị làm yếu đi. Biến thể thắng vì vậy được tiền huấn luyện đủ, và phải hơn mốc dùng trọng số công bố. Chi phí: một đến hai lần tiền huấn luyện dài |
| R9 | (cùng vấn đề 1) | Trước khi tốn tiền huấn luyện, kiểm xem nó có cần không ở chế độ này: một thân học từ đầu chỉ trên ảnh tai so với cùng thân có trọng số công bố (T6 iii). Nếu chênh dưới 0,05 dB, các biến thể học thẳng trên ảnh tai và vấn đề 1 biến mất. Kết quả này tự nó cũng đáng báo |
| R10 | Báo mốc ở cả hai công thức | Làm cho ba đến bốn mốc real-time chủ chốt, không phải mọi mốc, vì chi phí gấp đôi. Các mốc còn lại chỉ dùng công thức tốt |
| R11 | "AMI đủ rộng để làm phép thử ngữ cảnh" | Ở cỡ benchmark, ảnh vào là cả khung hình, không có gì bên ngoài. Phép thử phải chạy trên cửa sổ con của ảnh LR lớn hơn, tức nội dung ở độ phóng khác benchmark. Chúng tôi quét bề rộng ngữ cảnh từ 0 đến 24 px và báo đường sai số theo bề rộng; đường này cũng cho biết cần bao nhiêu ngữ cảnh. Thêm đối chứng thực dụng "cắt rộng rồi bỏ rìa", là thứ kiểu đệm tốt phải hơn ở cùng độ trễ |
| R12 | N3 cắt đầu tiên; giữ tối đa một trong N1, N3 | Đồng ý, và chúng tôi ưu tiên N1 nếu nó đạt. Lý do: sau khi cắt, N5 là về ảnh nhỏ nói chung và N2 là về dữ liệu; N1 là thành phần duy nhất gắn với giải phẫu tai, trong khi tên bài là về ảnh tai. Hệ quả cần nói rõ: nếu N3 bị cắt, mô hình chỉ có một đầu ra tối ưu độ trung thực, và chữ "rõ hơn" dựa vào PSNR, LPIPS, độ lệch điểm mốc và khảo sát có đáp án |

## C. Kết quả kiểm tra cặp "front" và "zoom"

Khớp SIFT và ước lượng phép đồng dạng trên 100 cặp: độ phóng trung vị 1,18 lần (5% đến 95%: 1,15 đến 1,22); 79 cặp khớp được với ít nhất 15 điểm; sai số khớp khoảng 1 px. Độ phóng thấp hơn nhiều so với ước lượng 1,5 lần, nên cặp này không dùng được cho SR ở ×2 hay ×4. Ý này bị bỏ.

## D. Tài liệu bạn nêu

Đã tra và xác nhận tồn tại: "Mind the Pad" (ICLR 2021); partial convolution based padding (2018); TLC, "Improving Image Restoration by Revisiting Global Information Aggregation" (ECCV 2022), bài này nói về lệch cỡ patch giữa huấn luyện và test trong phục hồi ảnh và gần với N5a. Chưa tra: Kayhan và van Gemert (CVPR 2020), FixRes, các bài SR ảnh mặt 16×16. Tất cả vào danh sách của T1.

## E. Câu hỏi gửi lại

1. Ngân sách tiền huấn luyện rút gọn: bạn coi mức nào là đủ để so sánh trong nhóm có nghĩa (ví dụ một phần năm số bước của công thức gốc, trên DIV2K)?
2. Nếu T6 (iii) cho thấy tiền huấn luyện gần như không giúp gì ở ảnh tai nhỏ, bạn có chấp nhận cả mốc lẫn mô hình đề xuất đều học thẳng trên ảnh tai, với mốc dùng trọng số công bố chỉ là một dòng tham chiếu không?
3. Ưu tiên N1 hơn N3 khi cả hai đạt: bạn đồng ý không, hay xét theo độ lớn của phần hơn?
4. Với N5 đã tách: kịch bản A+ cần cả N5a lẫn N5b, hay N5a cộng N2 đã đủ cho mức "major revision theo hướng nhận"?
5. Bạn viết "một tạp chí Q1 ứng dụng". Nhận định đó có áp dụng cho các tạp chí Q1 phạm vi rộng về AI, xử lý ảnh hoặc sinh trắc (ví dụ Complex & Intelligent Systems) không? Nếu không, bài còn thiếu gì?

---

# PHỤ LỤC 5. TRẢ LỜI PHẢN BIỆN VÒNG 8

*Phụ lục này giữ nguyên như đã gửi ở bản 18.*

## A. Các điểm đồng ý và phần đã sửa

| # | Ý phản biện | Sửa ở |
|---|---|---|
| 1 | N5a là so với người rơm: cách hiển nhiên (tinh chỉnh trên ảnh HR đúng cỡ) vốn đã đúng tỉ lệ | 1.2b: N5a mặc định không tính là điểm mới. T6 (ii) thành ba nhánh. Mục 3.10 điểm 2 ghi rõ phép so sánh cũ là sai đối tượng |
| 2 | Lõi của bài là N5b và N2; không đếm N5a vào A+ | 1.2b: A+ cần N5b và N2 trên hai thân; thiếu N5b thì là kịch bản A, mức ranh giới |
| 3 | Ngân sách tiền huấn luyện rút gọn phải đạt hai điều kiện (đường học phẳng; thứ hạng ổn định ở 50% và 100%) | T6 (iv) |
| 4 | Nếu tiền huấn luyện không giúp: dòng "trọng số công bố rồi tinh chỉnh" vẫn ở bảng chính và phải bị vượt; nhánh học từ đầu được đủ số bước | 1.8, T6 (iii) |
| 5 | Ngoại lệ cho quy tắc ưu tiên N1 hơn N3 | 1.2b, viết thành con số (mục B) |
| 6 | Tạp chí chuyên thị giác sẽ đòi kiểm chứng N5b ngoài ảnh tai | T6 (v), mục B |
| 7 | Ba lỗi nhỏ ở mục 3.9, 1.13, 3.13 | Đã sửa |

## B. Các điểm bổ sung

| # | Ý phản biện | Bổ sung của chúng tôi |
|---|---|---|
| R13 | Ba nhánh giao thức | Giao thức tỉ lệ ngẫu nhiên có một giá trị thực dụng ngoài chuyện điểm mới: nếu một mô hình không kém mô hình chuyên từng cỡ, hai cỡ còn lại ở ×4 không cần huấn luyện thêm, và ngân sách S4 giảm một nửa. Mốc và mô hình đề xuất luôn dùng cùng giao thức, là giao thức tốt hơn trên validation |
| R14 | Ngoại lệ "N1 sát ngưỡng, N3 dư lớn" | Viết thành con số để không phải phán đoán sau khi có kết quả: giữ N3 khi phần hơn của N1 dưới 1,5 lần ngưỡng của nó và phần hơn của N3 so với đường trộn từ 2 lần ngưỡng trở lên |
| R15 | Kiểm chứng N5b trên một bộ ảnh nhỏ khác | Làm được gần như không tốn huấn luyện: các biến thể đã được tiền huấn luyện trên DIV2K ở T6 (iv), nên phép thử ngữ cảnh và phép thử tương tác chạy thẳng trên ảnh DIV2K thu nhỏ, chỉ suy luận. Vì rẻ, chúng tôi đưa nó vào T6 thay vì để ở phần mở rộng. Tuyên bố của N5b sẽ theo kết quả: mở ra "ảnh vào nhỏ nói chung" nếu lặp lại được, giới hạn ở ảnh tai nếu không |

## C. Trạng thái

Mọi câu hỏi của các vòng trước đã có trả lời. Chúng tôi không còn câu hỏi nào về thiết kế. Các quyết định còn lại phụ thuộc số liệu: T2 (dư địa), T3 (độ trễ), T6 (i) (hiệu ứng viền có thật không). Lượt gửi tiếp theo sẽ kèm kết quả của ba phép thử này.

---

# PHỤ LỤC 6. TRẢ LỜI PHẢN BIỆN VÒNG 9

## A. Các điểm đồng ý và phần đã sửa

| # | Ý phản biện | Sửa ở |
|---|---|---|
| 1 | Thiếu giai đoạn tiền huấn luyện; thiếu dựng dữ liệu DIV2K | Sơ đồ 3.2 và chi tiết P6: bước 6pre; P1 có DIV2K và bản thu nhỏ |
| 2 | Sơ đồ mô hình đề xuất còn là N3 | Chi tiết P6 viết lại: nhánh chính là thân N5b với suy giảm N2; N1 và N3 là nhánh tùy chọn, nằm ở thư mục riêng |
| 3 | Thiếu công cụ cho hai phép thử lõi và bộ dựng DIV2K thu nhỏ | 3.3: `context_test.py`, `receptive_field.py`, `interaction.py`, `mde.py`, `build_div2k_tiny.py`, `padding.py`, `variants.py`, `pretrain.py`, `bench_padding.py` |
| 4 | Mã lần chạy thiếu ba trường | 3.5: thêm biến thể thân, tiền huấn luyện, giao thức; có kiểm thử chống trùng mã |
| 5 | Kiểm tra hỗ trợ kiểu đệm trên thiết bị ở T3 | 1.10 (T3), P5. Kiểu đệm không được hỗ trợ hoặc chậm hơn đệm số 0 quá 10% bị loại trước khi tiền huấn luyện |
| 6 | Kiểm thử cho kiểu đệm | 3.6, có một lưu ý ở mục B |
| 7 | Thứ tự: tháng đầu chỉ T1, T2, T3, T6 (i) và (v); T4, T5 để sau | 1.10, 1.11, 3.9: ba giai đoạn và ba điểm kiểm tra |
| 8 | Một bộ dò điểm mốc là đủ nếu bỏ N1 | 1.9 |

## B. Ba lưu ý

| # | Ý phản biện | Lưu ý của chúng tôi |
|---|---|---|
| R16 | Đổi kiểu đệm thì vùng giữa ảnh không đổi | Chỉ đúng với thân chỉ có phép toán cục bộ. Thân có gộp toàn cục hoặc chuẩn hóa theo thống kê cả ảnh sẽ đổi ở mọi nơi. Kiểm thử vì vậy chỉ áp cho thân không có phép toán toàn cục, và bán kính vùng nhìn được đo chứ không giả định. Chúng tôi thêm một kiểm thử tương tự cho chính phép thử ngữ cảnh |
| R17 | Bước (v) của T6 chạy ngay trong tháng đầu | Chỉ phần phép thử ngữ cảnh chạy được ngay, với trọng số công bố. Phép thử tương tác cần các biến thể đã tiền huấn luyện, nên thuộc giai đoạn 2 |
| R18 | Một bộ dò là đủ nếu bỏ N1 | Đồng ý, và có một hệ quả: khi không có N1, bộ dò không sinh nhãn huấn luyện, nên thước đo điểm mốc không còn vòng lặp. Nhãn gắn tay 100 ảnh vì vậy chỉ cần khi làm N1, và quyết định được dời ra sau điểm kiểm tra 2 |

## C. Trạng thái

Thiết kế đã chốt. Lượt gửi tiếp theo gồm ba thứ bạn đang chờ: bảng T2, số đo T3 (kèm bảng hỗ trợ kiểu đệm), và hình sai số theo bề rộng ngữ cảnh trên ảnh tai và trên DIV2K thu nhỏ.

---

## Nguồn tham khảo đã dùng

- NTIRE 2025 Efficient SR: https://arxiv.org/abs/2504.10686
- NTIRE 2025 Image SR ×4: https://arxiv.org/abs/2504.14582
- NTIRE 2026 Efficient SR: https://arxiv.org/html/2604.03198v1
- NTIRE 2026 Mobile Real-World Image SR: https://arxiv.org/html/2604.17306v1
- AIM 2025 Efficient Perceptual SR: https://arxiv.org/abs/2510.12765
- Mind the Pad (ICLR 2021): https://openreview.net/forum?id=m1CD7tPubNy
- Partial Convolution based Padding: https://arxiv.org/pdf/1811.11718
- TLC, Improving Image Restoration by Revisiting Global Information Aggregation (ECCV 2022): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670053.pdf
- AMI Ear Database: https://ctim.ulpgc.es/research_works/ami_ear_database/
- Danh mục bộ dữ liệu tai (IAPR TC4): https://iapr-tc4.org/ear-datasets/
- Chưa tra lại, trích theo trí nhớ hoặc theo người phản biện: LDL, DeSRA, trọng số ESRGAN bản PSNR và bản GAN, SR ảnh mặt dẫn hướng bằng điểm mốc (FSRNet, DIC), Ji và cộng sự 2020, SROOE, nhãn vùng tai của CelebAMask-HQ, điều khoản của EarVN1.0, AWEx và bộ điểm mốc
