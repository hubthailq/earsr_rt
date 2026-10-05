# Đo độ trễ trên điện thoại Android (số đo thứ hai, mục 1.5)

**Trạng thái: hướng dẫn, chưa chạy trên thiết bị nào.** Các lệnh dưới đây theo tài liệu của ncnn; hãy đối
chiếu với phiên bản bạn tải về và ghi lại phiên bản đó cùng số đo.

Mục đích: một số đo CPU của điện thoại, cùng giao thức với Jetson (ảnh vào 48×68 ở ×4, lô 1, 50 lượt làm
nóng, 500 lượt đo, báo trung vị và phân vị 95). Đây là số đo phụ; số đo chính vẫn là Jetson.

## Cách đề xuất: ncnn + `benchncnn` qua adb

1. Xuất ONNX trên máy tính (đã có): `python scripts/export_for_device.py` → `deploy_jetson/onnx/*.onnx`.
2. Chuyển ONNX sang ncnn bằng `pnnx` (đi kèm ncnn):
   `pnnx model.onnx inputshape=[1,3,68,48]` → `model.ncnn.param`, `model.ncnn.bin`.
   Mô hình nào `pnnx` báo toán tử không hỗ trợ thì ghi vào bảng là "không chuyển được", như cách làm với
   TensorRT. Kiểu đệm `reflect` và `replicate` cần kiểm riêng: ncnn có lớp `Padding` với ba kiểu, nhưng
   phải xem file `.param` sinh ra có dùng đúng kiểu không.
3. Tải bản dựng sẵn `ncnn-android` (có `benchncnn`), đẩy sang máy:
   `adb push benchncnn /data/local/tmp/ && adb push *.param /data/local/tmp/`
4. Chạy, ví dụ 4 luồng, lõi lớn, CPU:
   `adb shell "cd /data/local/tmp && ./benchncnn 500 4 2 -1 0 param=model.ncnn.param shape=[48,68,3]"`
   (thứ tự tham số: số lượt, số luồng, chế độ lõi, GPU (−1 là CPU), thời gian nghỉ; kiểm lại bằng
   `./benchncnn` không tham số).
5. `benchncnn` in min, max, avg. Giao thức của bài cần trung vị và phân vị 95: nếu bản bạn dùng không in,
   chạy 5 lần × 100 lượt và báo trung vị của các giá trị avg, kèm ghi chú rằng đây không phải phân vị 95.

## Ghi lại cùng số đo

Tên máy và chip; phiên bản Android; phiên bản ncnn; số luồng và chế độ lõi; máy có cắm sạc không; nhiệt độ
(đo sau khi để máy nghỉ 5 phút); FP32 hay FP16 (`use_fp16_*` trong ncnn mặc định bật trên ARM).

## Kiểm đầu ra

Trước khi tin số độ trễ, so đầu ra ncnn với PyTorch trên một ảnh: PSNR so với đáp án lệch không quá
0,05 dB (cùng điều kiện như với TensorRT, mục 3.4 P5 của kế hoạch).
