# Đo độ trễ trên iPhone bằng Xcode (việc 4 của docs/story-imavis.md)

Thiết bị: iPhone 12 Pro Max (chip A14), iOS 18.7.8. Máy Mac: Xcode 26.3. Ảnh vào 68×48 (cao × rộng), mỗi lần một ảnh, ×4.
Ngưỡng real-time của bài: trung vị không quá 33 ms.

**Trạng thái (08/10/2026):** 17 mô hình (16 mô hình PSNR của bài và BSRGAN) đã chuyển sang Core ML và khớp với PyTorch trên máy
Mac; `span_ch48` đã biên dịch thử được cho iOS 17; iPhone đã bật Developer Mode. Các bước đo trong Xcode
dưới đây viết theo hiểu biết về Xcode, **chưa được chạy thử trên máy này**; tên nút có thể khác đôi chút.

## 1. Chuyển mô hình (làm trên máy Mac; đã làm xong cho sáu mô hình mặc định)

```bash
# một lần: môi trường riêng trên Python 3.9 của Xcode (coremltools không chạy đủ trên Python 3.14)
/usr/bin/python3 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install "torch==2.7.*" coremltools numpy opencv-python-headless
# chuyển: ghi deploy_ios/models/<tên>.mlpackage và index.json (kèm phép so với PyTorch)
.venv-coreml/bin/python scripts/export_coreml.py
```

Mô hình của bài dùng đúng kiến trúc của `span_ch48` và `disp26`, chỉ khác giá trị trọng số, nên độ trễ như nhau. Muốn đo
bằng chính trọng số của bài thì chép một lần chạy từ labai217 về rồi chuyển (sửa địa chỉ máy cho đúng):

```bash
R=N2_span-zero+xearvn_pub_rand_x4_hrall_est_f2
rsync -av --include='config.json' --include='ckpt/' --include='ckpt/best.pt' --exclude='*' \
      <user>@labai217:~/ThaiLe/new_exprerient/earsr_rt/runs/$R/ runs/$R/
.venv-coreml/bin/python scripts/export_coreml.py --models --runs runs/$R
```

## 2. Chuẩn bị iPhone

- Cắm cáp vào máy Mac, mở khóa, bấm "Trust" nếu được hỏi.
- Bật Developer Mode: Settings → Privacy & Security → Developer Mode (máy sẽ khởi động lại).
- Tắt Low Power Mode, sạc trên 50%, để máy nguội (không vừa chơi game hay quay phim xong).

## 3. Đo từng mô hình trong Xcode

**Bắt buộc đo ba mô hình:** `span_ch48` và `disp26` (hai thân của bài) và `bsrgan` (mốc lớn, chậm). Sáu lần chạy, khoảng 15 phút.
14 mô hình còn lại là tùy chọn: bài chỉ cần chứng minh mô hình của mình chạy real-time, không cần độ trễ trên điện thoại của
cả 16 mô hình (bảng 1 của bài đã có độ trễ GPU cho cả 16). Nếu muốn thêm thì ưu tiên `edsr_baseline`, `swinir_light`,
`rrdb_psnr`; đo mô hình nào thì bảng độ trễ của bài tự có dòng mô hình đó.

1. Trong Finder, mở thư mục `deploy_ios/models/`, bấm đúp `span_ch48.mlpackage`. Xcode mở mô hình.
2. Chọn thẻ **Performance**.
3. Bấm dấu **+** (tạo báo cáo hiệu năng). Chọn thiết bị là iPhone (không chọn "My Mac"). Bấm Next.
4. Ở **Compute Units** chọn **CPU Only**. Bấm **Run Test** và đợi tới khi xong.
5. Ghi lại hai số: **Prediction** (trung vị, ms) và **Load** (trung vị, ms).
6. Tạo một báo cáo nữa cho cùng mô hình, lần này Compute Units chọn **All**. Ghi lại hai số như trên, và xem báo cáo cho
   biết các phép tính chạy trên đâu (Neural Engine, GPU hay CPU).
7. Đợi khoảng một phút cho máy nguội, rồi sang mô hình kế tiếp.

## 4. Ghi số vào results/latency_ios.csv

File đã có sẵn các dòng; điền cột `median_ms` (và `load_ms`, `note` nếu có). Dòng nào không đo thì để trống. Ví dụ:

```
span_ch48,cpu,7.42,35.1,iPhone 12 Pro Max,A14 Bionic,18.7.8,26.3,2026-10-08,
span_ch48,all,1.95,120.4,iPhone 12 Pro Max,A14 Bionic,18.7.8,26.3,2026-10-08,toàn bộ trên Neural Engine
```

(Hai dòng trên là ví dụ về định dạng, không phải số đo.) Sau đó commit file này và báo Claude; `scripts/make_paper.py` sẽ
đưa số vào bảng độ trễ và mục 6.7 của bản thảo.

## 5. Nếu Xcode không cho đo

- Không thấy iPhone trong danh sách thiết bị: mở Xcode → Window → Devices and Simulators, xem máy đã "Connected" chưa.
- Xcode đòi tài khoản: Xcode → Settings → Accounts, thêm Apple ID (miễn phí cũng được).
- Báo cáo hiệu năng chỉ cho trung vị, không có phân vị 95. Bài báo trung vị và ghi rõ điều đó.
- Vẫn không được thì báo Claude: phương án dự phòng là một app đo nhỏ, hoặc đo trên Android theo `docs/ANDROID.md`.
