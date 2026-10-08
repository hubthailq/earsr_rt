# Nhận dạng tai trên ảnh nhỏ thật (data/raw/awex; all): 112 người, 428 ảnh dò, 291 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet18/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| direct | 1 | 32.48 [24.63; 39.39] | 59.11 | 18.91 [16.21; 21.46] | 30.84 |
| ref_large | 1 | 32.25 [26.58; 38.03] | 57.99 | 19.81 [17.40; 23.41] | 33.43 |
| span_ch48 (published) | 1 | 32.24 [24.64; 39.91] | 55.84 | 20.06 [17.01; 23.51] | 29.21 |
| bicubic | 1 | 31.78 [23.74; 39.16] | 59.35 | 18.91 [16.32; 21.51] | 30.84 |
| rrdb_psnr (published) | 1 | 31.31 [22.78; 39.88] | 56.07 | 19.89 [17.32; 22.58] | 28.50 |
| fbcnn_bicubic (published) | 1 | 31.31 [21.68; 41.11] | 62.38 | 17.78 [15.19; 20.34] | 29.21 |
| fbcnn_span_ch48 (published) | 1 | 29.91 [21.58; 38.06] | 58.18 | 18.01 [15.23; 20.87] | 28.74 |
| span/est | 5 | 27.99 [19.63; 35.89] | 58.93 | 17.81 [14.89; 21.18] | 29.02 |
| realesrnet (published) | 1 | 27.10 [18.85; 34.61] | 54.44 | 18.32 [15.30; 22.60] | 29.67 |
| bsrnet (published) | 1 | 26.64 [18.18; 34.65] | 55.37 | 18.94 [13.92; 22.77] | 28.97 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| direct | +0.70 [-0.36; +1.75] | +0.00 [-0.36; +0.46] |
| span_ch48 (published) | +0.47 [-3.00; +3.62] | +1.15 [-0.59; +3.06] |
| fbcnn_bicubic (published) | -0.47 [-4.03; +3.09] | -1.13 [-3.12; +0.69] |
| rrdb_psnr (published) | -0.47 [-4.38; +3.20] | +0.98 [-0.51; +2.39] |
| fbcnn_span_ch48 (published) | -1.87 [-5.24; +1.48] | -0.90 [-3.01; +1.17] |
| span/est | -3.79 [-7.91; +0.48] | -1.10 [-3.50; +1.70] |
| realesrnet (published) | -4.67 [-10.31; +0.84] | -0.59 [-3.27; +2.87] |
| bsrnet (published) | -5.14 [-10.94; +0.30] | +0.03 [-4.89; +3.41] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | -0.23 [-3.64; +2.85] | +1.15 [-0.67; +2.98] |
| bicubic | -0.70 [-1.75; +0.36] | -0.00 [-0.46; +0.36] |
| fbcnn_bicubic (published) | -1.17 [-4.51; +2.37] | -1.13 [-3.19; +0.68] |
| rrdb_psnr (published) | -1.17 [-4.76; +2.38] | +0.97 [-0.52; +2.34] |
| fbcnn_span_ch48 (published) | -2.57 [-5.69; +0.44] | -0.90 [-3.15; +1.10] |
| span/est | -4.49 [-8.15; -0.78]* | -1.10 [-3.69; +1.64] |
| realesrnet (published) | -5.37 [-10.91; +0.20] | -0.59 [-3.44; +2.86] |
| bsrnet (published) | -5.84 [-11.55; -0.28]* | +0.03 [-5.01; +3.37] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| direct | +0.23 [-2.85; +3.64] | -1.15 [-2.98; +0.67] |
| bicubic | -0.47 [-3.62; +3.00] | -1.15 [-3.06; +0.59] |
| fbcnn_bicubic (published) | -0.93 [-5.25; +3.14] | -2.28 [-4.50; -0.34]* |
| rrdb_psnr (published) | -0.93 [-3.08; +0.99] | -0.17 [-1.49; +1.00] |
| fbcnn_span_ch48 (published) | -2.34 [-5.25; +0.41] | -2.05 [-4.30; +0.33] |
| span/est | -4.25 [-8.21; -0.55]* | -2.25 [-4.94; +0.85] |
| realesrnet (published) | -5.14 [-9.79; -0.45]* | -1.74 [-4.38; +1.72] |
| bsrnet (published) | -5.61 [-10.62; -0.63]* | -1.12 [-5.65; +2.79] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| direct | +4.49 [+0.78; +8.15]* | +1.10 [-1.64; +3.69] |
| span_ch48 (published) | +4.25 [+0.55; +8.21]* | +2.25 [-0.85; +4.94] |
| bicubic | +3.79 [-0.48; +7.91] | +1.10 [-1.70; +3.50] |
| rrdb_psnr (published) | +3.32 [-0.71; +7.69] | +2.08 [-0.77; +4.45] |
| fbcnn_bicubic (published) | +3.32 [-0.33; +7.98] | -0.03 [-1.88; +1.29] |
| fbcnn_span_ch48 (published) | +1.92 [-0.34; +4.32] | +0.20 [-1.34; +1.34] |
| realesrnet (published) | -0.89 [-3.83; +2.35] | +0.51 [-0.57; +2.31] |
| bsrnet (published) | -1.36 [-5.44; +2.49] | +1.13 [-1.94; +2.69] |
