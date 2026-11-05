# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 1887 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet18/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 60.20 [56.79; 63.76] | 87.33 | 11.81 [10.78; 12.95] | 50.29 |
| span_ch48 (published) | 1 | 54.85 [50.97; 58.43] | 85.48 | 12.90 [11.66; 14.50] | 46.37 |
| bicubic | 1 | 54.69 [49.44; 59.15] | 84.31 | 12.98 [11.39; 14.77] | 45.57 |
| direct | 1 | 54.64 [49.37; 59.08] | 84.63 | 12.87 [11.40; 14.69] | 45.79 |
| span/est | 5 | 51.30 [46.33; 56.88] | 81.58 | 14.33 [12.62; 16.03] | 42.49 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +0.16 [-2.21; +2.90] | -0.08 [-0.87; +0.82] |
| direct | -0.05 [-0.38; +0.23] | -0.11 [-0.24; +0.14] |
| span/est | -3.39 [-7.96; +1.95] | +1.35 [-0.47; +3.03] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +0.21 [-2.21; +2.98] | +0.03 [-0.81; +0.83] |
| bicubic | +0.05 [-0.23; +0.38] | +0.11 [-0.14; +0.24] |
| span/est | -3.34 [-7.76; +2.06] | +1.46 [-0.44; +3.09] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | -0.16 [-2.90; +2.21] | +0.08 [-0.82; +0.87] |
| direct | -0.21 [-2.98; +2.21] | -0.03 [-0.83; +0.81] |
| span/est | -3.55 [-7.00; +0.05] | +1.43 [-0.16; +2.74] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +3.55 [-0.05; +7.00] | -1.43 [-2.74; +0.16] |
| bicubic | +3.39 [-1.95; +7.96] | -1.35 [-3.03; +0.47] |
| direct | +3.34 [-2.06; +7.76] | -1.46 [-3.09; +0.44] |
