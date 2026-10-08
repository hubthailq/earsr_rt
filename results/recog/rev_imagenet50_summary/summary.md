# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 2015 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `imagenet-resnet50`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 24.59 [20.93; 28.52] | 59.67 | 33.25 [31.56; 34.60] | 7.84 |
| bsrnet (published) | 1 | 8.39 [5.15; 12.23] | 23.23 | 46.64 [42.32; 50.95] | 2.28 |
| realesrnet (published) | 1 | 8.19 [4.86; 12.39] | 21.94 | 48.09 [43.94; 52.09] | 2.13 |
| span/est | 5 | 7.16 [4.35; 10.29] | 20.45 | 48.90 [45.49; 52.19] | 2.01 |
| fbcnn_span_ch48 (published) | 1 | 5.56 [3.02; 8.96] | 15.14 | 50.10 [46.86; 53.16] | 1.44 |
| fbcnn_bicubic (published) | 1 | 4.67 [2.52; 7.78] | 11.66 | 51.42 [47.90; 54.44] | 0.74 |
| rrdb_psnr (published) | 1 | 3.97 [1.73; 7.66] | 9.33 | 52.14 [48.50; 55.00] | 0.74 |
| span_ch48 (published) | 1 | 3.92 [1.70; 7.52] | 9.48 | 51.87 [48.68; 54.93] | 0.69 |
| direct | 1 | 3.03 [1.12; 6.32] | 7.59 | 52.42 [49.15; 55.23] | 0.35 |
| bicubic | 1 | 2.93 [1.07; 6.13] | 7.84 | 52.50 [49.13; 55.20] | 0.35 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +5.46 [+1.43; +9.38]* | -5.87 [-9.14; -2.58]* |
| realesrnet (published) | +5.26 [+1.03; +9.57]* | -4.41 [-7.66; -1.31]* |
| span/est | +4.23 [+1.79; +6.92]* | -3.61 [-6.22; -0.67]* |
| fbcnn_span_ch48 (published) | +2.63 [+0.96; +4.59]* | -2.41 [-4.02; -0.49]* |
| fbcnn_bicubic (published) | +1.74 [+0.40; +3.26]* | -1.09 [-2.75; +0.81] |
| rrdb_psnr (published) | +1.04 [+0.25; +2.20]* | -0.36 [-1.41; +0.80] |
| span_ch48 (published) | +0.99 [+0.22; +2.06]* | -0.63 [-1.39; +0.64] |
| direct | +0.10 [+0.00; +0.27] | -0.09 [-0.18; +0.27] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +5.36 [+1.27; +9.27]* | -5.78 [-9.19; -2.66]* |
| realesrnet (published) | +5.16 [+0.82; +9.47]* | -4.33 [-7.70; -1.40]* |
| span/est | +4.13 [+1.67; +6.84]* | -3.52 [-6.28; -0.63]* |
| fbcnn_span_ch48 (published) | +2.53 [+0.84; +4.52]* | -2.32 [-4.06; -0.55]* |
| fbcnn_bicubic (published) | +1.64 [+0.25; +3.13]* | -1.00 [-2.75; +0.81] |
| rrdb_psnr (published) | +0.94 [+0.20; +2.03]* | -0.28 [-1.40; +0.77] |
| span_ch48 (published) | +0.89 [+0.15; +1.92]* | -0.55 [-1.43; +0.60] |
| bicubic | -0.10 [-0.27; +0.00] | +0.09 [-0.27; +0.18] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +4.47 [+0.51; +8.57]* | -5.24 [-9.33; -1.72]* |
| realesrnet (published) | +4.27 [+0.00; +8.50] | -3.78 [-7.69; -0.58]* |
| span/est | +3.24 [+0.45; +5.98]* | -2.97 [-6.31; +0.42] |
| fbcnn_span_ch48 (published) | +1.64 [-0.27; +3.64] | -1.77 [-4.18; +0.40] |
| fbcnn_bicubic (published) | +0.74 [-1.12; +2.37] | -0.45 [-2.93; +1.89] |
| rrdb_psnr (published) | +0.05 [-0.23; +0.41] | +0.27 [-0.44; +0.74] |
| direct | -0.89 [-1.92; -0.15]* | +0.55 [-0.60; +1.43] |
| bicubic | -0.99 [-2.06; -0.22]* | +0.63 [-0.64; +1.39] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +1.23 [-1.17; +3.78] | -2.26 [-5.09; +0.94] |
| realesrnet (published) | +1.03 [-1.57; +3.80] | -0.81 [-3.71; +2.24] |
| fbcnn_span_ch48 (published) | -1.60 [-2.84; -0.38]* | +1.20 [-0.30; +2.68] |
| fbcnn_bicubic (published) | -2.49 [-4.06; -0.99]* | +2.52 [+0.84; +3.91]* |
| rrdb_psnr (published) | -3.19 [-5.90; -0.39]* | +3.24 [-0.36; +6.49] |
| span_ch48 (published) | -3.24 [-5.98; -0.45]* | +2.97 [-0.42; +6.31] |
| direct | -4.13 [-6.84; -1.67]* | +3.52 [+0.63; +6.28]* |
| bicubic | -4.23 [-6.92; -1.79]* | +3.61 [+0.67; +6.22]* |

## Ảnh dò có mức nén JPEG của file từ 70 đến 80: 1431 ảnh, 33 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| realesrnet (published) | 9.71 | +6.43 [+0.76; +12.04]* |
| bsrnet (published) | 9.50 | +6.22 [+0.88; +11.48]* |
| span/est | 8.39 | +5.10 [+1.73; +9.25]* |
| fbcnn_span_ch48 (published) | 7.13 | +3.84 [+1.70; +6.63]* |
| fbcnn_bicubic (published) | 6.01 | +2.73 [+0.99; +4.91]* |
| rrdb_psnr (published) | 4.54 | +1.26 [+0.24; +2.57]* |
| span_ch48 (published) | 4.54 | +1.26 [+0.29; +2.40]* |
| direct | 3.42 | +0.14 [+0.00; +0.38] |
| bicubic | 3.28 | +0.00 [+0.00; +0.00] |

## Ảnh dò có mức nén JPEG của file từ 90 đến 100: 574 ảnh, 12 người

| Nhánh | Rank-1 | So với bicubic |
|---|---|---|
| bsrnet (published) | 5.40 | +3.31 [+0.54; +6.03]* |
| span/est | 4.22 | +2.13 [+0.47; +4.15]* |
| realesrnet (published) | 4.01 | +1.92 [+0.15; +3.98]* |
| rrdb_psnr (published) | 2.61 | +0.52 [-0.26; +1.67] |
| span_ch48 (published) | 2.44 | +0.35 [-0.56; +1.34] |
| bicubic | 2.09 | +0.00 [+0.00; +0.00] |
| direct | 2.09 | +0.00 [+0.00; +0.00] |
| fbcnn_span_ch48 (published) | 1.74 | -0.35 [-1.51; +0.94] |
| fbcnn_bicubic (published) | 1.39 | -0.70 [-2.03; +0.37] |
