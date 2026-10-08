# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 2015 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet18/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 60.20 [56.79; 63.76] | 87.33 | 11.81 [10.78; 12.95] | 50.29 |
| bsrnet (published) | 1 | 27.15 [19.98; 35.55] | 53.85 | 25.13 [21.38; 29.16] | 21.59 |
| span/est | 5 | 25.03 [19.27; 31.10] | 50.91 | 26.82 [23.03; 30.51] | 19.41 |
| realesrnet (published) | 1 | 25.01 [18.53; 32.69] | 52.61 | 26.58 [22.55; 30.18] | 21.29 |
| fbcnn_span_ch48 (published) | 1 | 21.09 [15.69; 26.39] | 47.94 | 27.86 [25.13; 30.68] | 16.03 |
| fbcnn_bicubic (published) | 1 | 18.71 [13.31; 23.54] | 45.16 | 28.29 [25.78; 30.51] | 13.45 |
| rrdb_psnr (published) | 1 | 14.44 [11.03; 18.01] | 36.87 | 31.91 [29.44; 34.23] | 9.98 |
| span_ch48 (published) | 1 | 14.00 [10.62; 17.63] | 36.92 | 32.12 [29.61; 33.89] | 10.02 |
| bicubic | 1 | 13.20 [9.66; 17.00] | 36.03 | 31.22 [28.86; 33.65] | 8.83 |
| direct | 1 | 13.05 [9.65; 16.81] | 36.18 | 31.29 [28.82; 33.72] | 8.68 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +13.95 [+7.66; +21.31]* | -6.09 [-9.59; -2.79]* |
| span/est | +11.83 [+7.34; +16.89]* | -4.41 [-7.68; -1.58]* |
| realesrnet (published) | +11.81 [+5.94; +18.70]* | -4.64 [-8.43; -1.58]* |
| fbcnn_span_ch48 (published) | +7.89 [+4.30; +11.71]* | -3.36 [-5.40; -1.63]* |
| fbcnn_bicubic (published) | +5.51 [+2.60; +8.68]* | -2.93 [-4.55; -1.66]* |
| rrdb_psnr (published) | +1.24 [+0.29; +2.41]* | +0.69 [-0.63; +1.68] |
| span_ch48 (published) | +0.79 [-0.11; +1.92] | +0.90 [-0.44; +1.72] |
| direct | -0.15 [-0.37; +0.05] | +0.07 [-0.21; +0.23] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +14.09 [+7.81; +21.45]* | -6.15 [-9.68; -2.73]* |
| span/est | +11.98 [+7.39; +16.99]* | -4.47 [-7.74; -1.50]* |
| realesrnet (published) | +11.96 [+6.19; +18.90]* | -4.70 [-8.49; -1.62]* |
| fbcnn_span_ch48 (published) | +8.04 [+4.42; +11.88]* | -3.43 [-5.47; -1.63]* |
| fbcnn_bicubic (published) | +5.66 [+2.71; +8.85]* | -3.00 [-4.62; -1.69]* |
| rrdb_psnr (published) | +1.39 [+0.44; +2.47]* | +0.62 [-0.67; +1.66] |
| span_ch48 (published) | +0.94 [+0.09; +2.01]* | +0.83 [-0.57; +1.80] |
| bicubic | +0.15 [-0.05; +0.37] | -0.07 [-0.23; +0.21] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +13.15 [+7.32; +19.85]* | -6.99 [-9.54; -3.79]* |
| span/est | +11.04 [+6.76; +15.73]* | -5.30 [-7.80; -2.70]* |
| realesrnet (published) | +11.02 [+5.61; +17.38]* | -5.54 [-8.61; -2.71]* |
| fbcnn_span_ch48 (published) | +7.10 [+3.67; +10.59]* | -4.26 [-5.52; -2.61]* |
| fbcnn_bicubic (published) | +4.71 [+1.77; +7.72]* | -3.83 [-4.83; -2.67]* |
| rrdb_psnr (published) | +0.45 [-0.10; +0.99] | -0.21 [-0.57; +0.59] |
| bicubic | -0.79 [-1.92; +0.11] | -0.90 [-1.72; +0.44] |
| direct | -0.94 [-2.01; -0.09]* | -0.83 [-1.80; +0.57] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +2.11 [-0.45; +5.15] | -1.68 [-2.52; -0.52]* |
| realesrnet (published) | -0.02 [-2.38; +2.74] | -0.23 [-1.57; +0.83] |
| fbcnn_span_ch48 (published) | -3.94 [-6.90; -1.69]* | +1.05 [-0.12; +2.50] |
| fbcnn_bicubic (published) | -6.32 [-10.01; -3.42]* | +1.47 [-0.38; +3.42] |
| rrdb_psnr (published) | -10.59 [-15.32; -6.43]* | +5.09 [+2.72; +7.66]* |
| span_ch48 (published) | -11.04 [-15.73; -6.76]* | +5.30 [+2.70; +7.80]* |
| bicubic | -11.83 [-16.89; -7.34]* | +4.41 [+1.58; +7.68]* |
| direct | -11.98 [-16.99; -7.39]* | +4.47 [+1.50; +7.74]* |

## Ảnh dò có mức nén JPEG của file từ 70 đến 80: 1431 ảnh, 33 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| span/est | 23.58 | +9.25 [+4.30; +14.79]* |
| bsrnet (published) | 23.06 | +8.74 [+2.41; +16.04]* |
| fbcnn_span_ch48 (published) | 22.71 | +8.39 [+3.61; +13.64]* |
| realesrnet (published) | 21.38 | +7.06 [+1.10; +14.42]* |
| fbcnn_bicubic (published) | 20.48 | +6.15 [+2.34; +10.26]* |
| rrdb_psnr (published) | 15.09 | +0.77 [-0.44; +2.10] |
| bicubic | 14.33 | +0.00 [+0.00; +0.00] |
| span_ch48 (published) | 14.26 | -0.07 [-1.27; +0.97] |
| direct | 14.12 | -0.21 [-0.51; +0.00] |

## Ảnh dò có mức nén JPEG của file từ 90 đến 100: 574 ảnh, 12 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| bsrnet (published) | 37.80 | +27.18 [+15.95; +38.53]* |
| realesrnet (published) | 34.49 | +23.87 [+13.89; +34.54]* |
| span/est | 29.09 | +18.47 [+10.63; +26.69]* |
| fbcnn_span_ch48 (published) | 17.42 | +6.79 [+3.91; +9.91]* |
| fbcnn_bicubic (published) | 14.63 | +4.01 [+0.82; +7.72]* |
| span_ch48 (published) | 13.59 | +2.96 [+1.57; +4.60]* |
| rrdb_psnr (published) | 13.07 | +2.44 [+0.65; +4.21]* |
| bicubic | 10.63 | +0.00 [+0.00; +0.00] |
| direct | 10.63 | +0.00 [-0.53; +0.50] |
