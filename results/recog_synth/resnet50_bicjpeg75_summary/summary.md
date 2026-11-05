# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 1887 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet50/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 64.92 [60.99; 68.97] | 91.26 | 10.40 [9.30; 11.32] | 56.12 |
| span/est | 5 | 56.95 [52.35; 61.69] | 84.85 | 12.63 [11.36; 14.08] | 46.36 |
| bicubic | 1 | 54.85 [48.09; 60.46] | 82.94 | 13.66 [11.99; 15.87] | 45.95 |
| direct | 1 | 54.58 [47.60; 60.17] | 82.94 | 13.71 [11.99; 15.89] | 45.68 |
| span_ch48 (published) | 1 | 53.58 [47.07; 59.34] | 81.51 | 14.50 [12.73; 16.75] | 43.67 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span/est | +2.10 [-1.47; +6.64] | -1.03 [-2.58; +0.35] |
| direct | -0.26 [-0.69; +0.11] | +0.05 [-0.13; +0.21] |
| span_ch48 (published) | -1.27 [-3.17; +0.57] | +0.84 [+0.24; +1.46]* |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span/est | +2.36 [-1.36; +6.92] | -1.07 [-2.68; +0.39] |
| bicubic | +0.26 [-0.11; +0.69] | -0.05 [-0.21; +0.13] |
| span_ch48 (published) | -1.01 [-2.78; +0.85] | +0.79 [+0.20; +1.46]* |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span/est | +3.37 [+0.16; +7.67]* | -1.87 [-3.33; -0.59]* |
| bicubic | +1.27 [-0.57; +3.17] | -0.84 [-1.46; -0.24]* |
| direct | +1.01 [-0.85; +2.78] | -0.79 [-1.46; -0.20]* |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | -2.10 [-6.64; +1.47] | +1.03 [-0.35; +2.58] |
| direct | -2.36 [-6.92; +1.36] | +1.07 [-0.39; +2.68] |
| span_ch48 (published) | -3.37 [-7.67; -0.16]* | +1.87 [+0.59; +3.33]* |
