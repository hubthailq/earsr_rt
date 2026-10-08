# Nhận dạng tai trên ảnh nhỏ thật (data/raw/awex; all): 112 người, 428 ảnh dò, 291 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet50/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 41.42 [34.97; 47.15] | 68.34 | 15.96 [13.71; 19.06] | 41.42 |
| bsrnet (published) | 1 | 41.36 [30.16; 50.96] | 71.96 | 14.95 [12.33; 18.83] | 43.22 |
| realesrnet (published) | 1 | 40.42 [30.18; 48.84] | 70.33 | 15.18 [12.56; 19.27] | 43.93 |
| span/est | 5 | 39.95 [30.25; 48.57] | 70.51 | 15.04 [12.75; 18.59] | 41.59 |
| fbcnn_span_ch48 (published) | 1 | 39.49 [28.92; 48.96] | 70.79 | 15.55 [12.83; 18.36] | 39.49 |
| fbcnn_bicubic (published) | 1 | 39.02 [29.82; 47.04] | 68.93 | 15.12 [12.86; 17.33] | 37.38 |
| bicubic | 1 | 38.08 [28.89; 46.72] | 68.93 | 14.88 [12.66; 18.01] | 37.85 |
| direct | 1 | 37.85 [28.76; 46.53] | 68.93 | 15.17 [12.60; 18.18] | 38.08 |
| span_ch48 (published) | 1 | 37.62 [27.89; 46.37] | 65.65 | 15.37 [13.49; 18.49] | 38.32 |
| rrdb_psnr (published) | 1 | 36.45 [26.81; 44.76] | 65.42 | 15.41 [13.56; 18.28] | 36.92 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +3.27 [-4.05; +11.16] | +0.07 [-2.14; +2.49] |
| realesrnet (published) | +2.34 [-4.70; +9.01] | +0.30 [-1.81; +2.94] |
| span/est | +1.87 [-2.70; +6.42] | +0.16 [-1.49; +1.97] |
| fbcnn_span_ch48 (published) | +1.40 [-2.90; +6.07] | +0.67 [-1.10; +1.68] |
| fbcnn_bicubic (published) | +0.93 [-2.14; +4.20] | +0.24 [-1.59; +0.96] |
| direct | -0.23 [-0.69; +0.00] | +0.29 [-0.35; +0.63] |
| span_ch48 (published) | -0.47 [-4.00; +3.18] | +0.49 [-0.89; +1.89] |
| rrdb_psnr (published) | -1.64 [-5.20; +1.98] | +0.53 [-0.78; +1.95] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +3.50 [-3.74; +11.40] | -0.22 [-2.22; +2.46] |
| realesrnet (published) | +2.57 [-4.56; +9.24] | +0.01 [-1.88; +2.75] |
| span/est | +2.10 [-2.46; +6.56] | -0.13 [-1.56; +1.86] |
| fbcnn_span_ch48 (published) | +1.64 [-2.64; +6.13] | +0.38 [-1.15; +1.62] |
| fbcnn_bicubic (published) | +1.17 [-1.79; +4.20] | -0.05 [-1.74; +0.96] |
| bicubic | +0.23 [+0.00; +0.69] | -0.29 [-0.63; +0.35] |
| span_ch48 (published) | -0.23 [-3.54; +3.18] | +0.20 [-0.99; +1.90] |
| rrdb_psnr (published) | -1.40 [-4.68; +2.05] | +0.24 [-0.86; +1.88] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +3.74 [-3.00; +10.58] | -0.42 [-2.80; +1.94] |
| realesrnet (published) | +2.80 [-4.07; +8.83] | -0.19 [-2.31; +2.32] |
| span/est | +2.34 [-2.75; +6.63] | -0.33 [-2.18; +1.62] |
| fbcnn_span_ch48 (published) | +1.87 [-3.17; +5.76] | +0.18 [-1.53; +1.18] |
| fbcnn_bicubic (published) | +1.40 [-2.54; +4.99] | -0.25 [-2.30; +0.68] |
| bicubic | +0.47 [-3.18; +4.00] | -0.49 [-1.89; +0.89] |
| direct | +0.23 [-3.18; +3.54] | -0.20 [-1.90; +0.99] |
| rrdb_psnr (published) | -1.17 [-3.33; +0.91] | +0.04 [-0.68; +0.70] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bsrnet (published) | +1.40 [-3.14; +5.60] | -0.09 [-1.39; +1.30] |
| realesrnet (published) | +0.47 [-3.49; +3.87] | +0.14 [-0.99; +1.62] |
| fbcnn_span_ch48 (published) | -0.47 [-2.64; +1.58] | +0.51 [-1.18; +1.51] |
| fbcnn_bicubic (published) | -0.93 [-4.14; +2.57] | +0.08 [-1.93; +0.99] |
| bicubic | -1.87 [-6.42; +2.70] | -0.16 [-1.97; +1.49] |
| direct | -2.10 [-6.56; +2.46] | +0.13 [-1.86; +1.56] |
| span_ch48 (published) | -2.34 [-6.63; +2.75] | +0.33 [-1.62; +2.18] |
| rrdb_psnr (published) | -3.50 [-7.48; +1.75] | +0.37 [-1.65; +2.14] |
