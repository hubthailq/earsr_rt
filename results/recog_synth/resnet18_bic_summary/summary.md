# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 1887 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet18/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 60.20 [56.79; 63.76] | 87.33 | 11.81 [10.78; 12.95] | 50.29 |
| span_ch48 (published) | 1 | 56.97 [53.26; 61.14] | 86.54 | 12.20 [10.91; 13.63] | 49.81 |
| direct | 1 | 56.76 [51.79; 61.35] | 86.54 | 12.36 [10.97; 13.64] | 47.59 |
| bicubic | 1 | 56.76 [51.71; 61.54] | 86.49 | 12.39 [10.95; 13.68] | 47.75 |
| span/est | 5 | 51.22 [46.05; 56.99] | 81.28 | 14.38 [12.46; 16.03] | 42.75 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +0.21 [-2.58; +3.65] | -0.19 [-0.88; +0.77] |
| direct | +0.00 [-0.28; +0.26] | -0.03 [-0.16; +0.13] |
| span/est | -5.53 [-9.72; -0.07]* | +1.99 [+0.38; +3.47]* |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +0.21 [-2.65; +3.65] | -0.16 [-0.87; +0.76] |
| bicubic | +0.00 [-0.26; +0.28] | +0.03 [-0.13; +0.16] |
| span/est | -5.53 [-9.82; -0.15]* | +2.02 [+0.41; +3.50]* |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | -0.21 [-3.65; +2.58] | +0.19 [-0.77; +0.88] |
| direct | -0.21 [-3.65; +2.65] | +0.16 [-0.76; +0.87] |
| span/est | -5.74 [-8.60; -2.39]* | +2.18 [+0.90; +3.18]* |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +5.74 [+2.39; +8.60]* | -2.18 [-3.18; -0.90]* |
| bicubic | +5.53 [+0.07; +9.72]* | -1.99 [-3.47; -0.38]* |
| direct | +5.53 [+0.15; +9.82]* | -2.02 [-3.50; -0.41]* |
