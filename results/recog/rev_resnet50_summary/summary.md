# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 2015 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet50/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 64.92 [60.99; 68.97] | 91.26 | 10.40 [9.30; 11.32] | 56.12 |
| bsrnet (published) | 1 | 32.90 [26.39; 40.05] | 61.99 | 21.37 [18.44; 23.91] | 23.82 |
| realesrnet (published) | 1 | 30.47 [24.47; 37.30] | 60.99 | 22.11 [19.40; 24.50] | 22.78 |
| span/est | 5 | 27.04 [20.78; 33.21] | 56.49 | 23.36 [21.08; 25.75] | 18.61 |
| fbcnn_span_ch48 (published) | 1 | 24.47 [18.63; 30.46] | 54.09 | 24.98 [22.30; 27.33] | 16.77 |
| fbcnn_bicubic (published) | 1 | 23.77 [18.13; 29.79] | 52.26 | 25.22 [22.48; 27.98] | 15.33 |
| span_ch48 (published) | 1 | 18.61 [13.94; 23.32] | 42.68 | 28.40 [26.16; 30.67] | 11.41 |
| rrdb_psnr (published) | 1 | 18.46 [13.73; 23.21] | 41.89 | 28.63 [26.34; 30.97] | 11.76 |
| bicubic | 1 | 17.52 [12.69; 22.70] | 43.23 | 28.24 [25.57; 31.12] | 10.92 |
| direct | 1 | 17.42 [12.64; 22.54] | 43.08 | 28.33 [25.59; 31.16] | 10.72 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +15.38 [+10.39; +21.16]* | -6.87 [-9.94; -4.46]* |
| realesrnet (published) | +12.95 [+7.94; +18.55]* | -6.12 [-9.03; -3.86]* |
| span/est | +9.52 [+6.49; +12.75]* | -4.87 [-6.45; -3.25]* |
| fbcnn_span_ch48 (published) | +6.95 [+4.51; +9.38]* | -3.26 [-4.49; -2.06]* |
| fbcnn_bicubic (published) | +6.25 [+4.36; +8.22]* | -3.02 [-3.91; -1.93]* |
| span_ch48 (published) | +1.09 [-0.47; +2.64] | +0.16 [-0.65; +1.01] |
| rrdb_psnr (published) | +0.94 [-0.74; +2.65] | +0.40 [-0.42; +1.36] |
| direct | -0.10 [-0.34; +0.15] | +0.09 [-0.20; +0.23] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +15.48 [+10.48; +21.22]* | -6.95 [-9.92; -4.45]* |
| realesrnet (published) | +13.05 [+7.99; +18.66]* | -6.21 [-9.00; -3.87]* |
| span/est | +9.62 [+6.47; +12.85]* | -4.96 [-6.49; -3.31]* |
| fbcnn_span_ch48 (published) | +7.05 [+4.54; +9.55]* | -3.34 [-4.57; -2.11]* |
| fbcnn_bicubic (published) | +6.35 [+4.50; +8.34]* | -3.11 [-3.97; -1.89]* |
| span_ch48 (published) | +1.19 [-0.40; +2.78] | +0.07 [-0.71; +0.98] |
| rrdb_psnr (published) | +1.04 [-0.68; +2.75] | +0.31 [-0.48; +1.33] |
| bicubic | +0.10 [-0.15; +0.34] | -0.09 [-0.23; +0.20] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +14.29 [+9.14; +19.98]* | -7.03 [-9.80; -4.69]* |
| realesrnet (published) | +11.86 [+6.98; +17.52]* | -6.28 [-8.93; -4.05]* |
| span/est | +8.43 [+5.22; +11.98]* | -5.03 [-6.46; -3.55]* |
| fbcnn_span_ch48 (published) | +5.86 [+3.14; +8.58]* | -3.42 [-4.78; -2.26]* |
| fbcnn_bicubic (published) | +5.16 [+2.63; +7.84]* | -3.18 [-4.55; -1.81]* |
| rrdb_psnr (published) | -0.15 [-0.76; +0.45] | +0.23 [-0.17; +0.74] |
| bicubic | -1.09 [-2.64; +0.47] | -0.16 [-1.01; +0.65] |
| direct | -1.19 [-2.78; +0.40] | -0.07 [-0.98; +0.71] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +5.87 [+2.64; +9.48]* | -1.99 [-3.62; -0.82]* |
| realesrnet (published) | +3.43 [+0.23; +6.99]* | -1.25 [-2.71; -0.17]* |
| fbcnn_span_ch48 (published) | -2.57 [-4.47; -0.79]* | +1.62 [+0.66; +2.42]* |
| fbcnn_bicubic (published) | -3.27 [-5.61; -1.21]* | +1.85 [+0.56; +3.15]* |
| span_ch48 (published) | -8.43 [-11.98; -5.22]* | +5.03 [+3.55; +6.46]* |
| rrdb_psnr (published) | -8.58 [-11.85; -5.44]* | +5.27 [+3.63; +6.82]* |
| bicubic | -9.52 [-12.75; -6.49]* | +4.87 [+3.25; +6.45]* |
| direct | -9.62 [-12.85; -6.47]* | +4.96 [+3.31; +6.49]* |

## Ảnh dò có mức nén JPEG của file từ 70 đến 80: 1431 ảnh, 33 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| bsrnet (published) | 30.75 | +12.16 [+6.84; +18.37]* |
| span/est | 28.57 | +9.98 [+6.46; +14.52]* |
| realesrnet (published) | 28.16 | +9.57 [+4.08; +16.05]* |
| fbcnn_span_ch48 (published) | 26.90 | +8.32 [+5.54; +11.46]* |
| fbcnn_bicubic (published) | 25.93 | +7.34 [+5.14; +9.92]* |
| span_ch48 (published) | 19.71 | +1.12 [-0.58; +2.89] |
| rrdb_psnr (published) | 19.43 | +0.84 [-0.90; +2.48] |
| bicubic | 18.59 | +0.00 [+0.00; +0.00] |
| direct | 18.45 | -0.14 [-0.55; +0.19] |

## Ảnh dò có mức nén JPEG của file từ 90 đến 100: 574 ảnh, 12 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| bsrnet (published) | 38.50 | +23.34 [+14.72; +32.34]* |
| realesrnet (published) | 36.59 | +21.43 [+13.90; +29.49]* |
| span/est | 23.52 | +8.36 [+4.39; +12.76]* |
| fbcnn_bicubic (published) | 18.82 | +3.66 [+1.28; +6.07]* |
| fbcnn_span_ch48 (published) | 18.82 | +3.66 [+1.03; +6.64]* |
| rrdb_psnr (published) | 16.38 | +1.22 [-2.84; +5.05] |
| span_ch48 (published) | 16.20 | +1.05 [-3.05; +5.07] |
| bicubic | 15.16 | +0.00 [+0.00; +0.00] |
| direct | 15.16 | +0.00 [+0.00; +0.00] |
