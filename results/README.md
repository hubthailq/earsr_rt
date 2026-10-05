# Kết quả sơ bộ của giai đoạn 1

Các file ở đây được tạo lúc dựng project, trên CPU, để kiểm rằng mã chạy đúng trên AMI thật. Chúng **chưa đủ**
và nên được chạy lại trên máy của bạn bằng `scripts/run_stage1.sh` (script không ghi đè file đã có: dời `t2/`,
`t2_summary/`, `t6_context/`, `t6_summary/` đi trước, xem `docs/RUNBOOK.md` mục 1):

- chưa có LPIPS, DISTS (máy dựng project không tải được trọng số ImageNet);
- T2 mới xong ×4 ở ba cỡ cho nhóm mô hình nhẹ và vừa; thiếu RRDB (trần trên), nhóm mô hình cảm nhận,
  ×2, và cỡ 192 px với JPEG;
- phép thử ngữ cảnh mới xong một phần các cấu hình; phần ngoài ảnh tai dùng Urban100 (100 ảnh), không phải DIV2K;
- chưa có số đo độ trễ nào.

`t2/` và `t6_context/`: số đo theo từng ảnh. `t2_summary/`, `t6_summary/`: bảng tổng hợp sinh bằng
`scripts/summarize_t2.py` và `scripts/summarize_context.py`.
