# Kết quả bị loại: khối n2 lần đầu (chỉ AMI, không tăng cường độ sáng)

Tám mô hình này được tinh chỉnh chỉ trên AMI ngày 06/10/2026 và **hỏng trên ảnh có vùng sáng**: AMI không có ảnh
nào như vậy. Các file ở đây là kết quả chấm trên EarVN1.0 (nhóm test, 158 ảnh, đáp án 96 px, ảnh vào bicubic).
SPAN tinh chỉnh với bicubic đạt 22,08 dB (mốc công bố `span_ch48`: 36,39 dB; bicubic: 32,33 dB); trên 109 ảnh có vùng
sáng nó còn 15,89 dB.

Giữ lại làm bằng chứng của lỗi; **không dùng để kết luận gì về N2**. Trong `results/runs.csv` tám lần chạy này mang
trạng thái `excluded`. Lần chạy lại (nhãn `+xearvn`, `+pa`) ghi kết quả vào `results/n2/`. Chi tiết: `docs/STATUS.md`.
