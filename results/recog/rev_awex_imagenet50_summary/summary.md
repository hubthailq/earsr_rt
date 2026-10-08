# Nhận dạng tai trên ảnh nhỏ thật (data/raw/awex; all): 112 người, 428 ảnh dò, 291 ảnh đăng ký

Mạng nhận dạng: `imagenet-resnet50`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 16.57 [12.16; 21.24] | 34.02 | 34.97 [31.79; 37.81] | 10.06 |
| realesrnet (published) | 1 | 13.32 [8.13; 19.24] | 32.71 | 32.23 [27.61; 37.37] | 9.58 |
| bsrnet (published) | 1 | 11.68 [5.83; 18.28] | 33.88 | 32.53 [27.55; 37.82] | 11.45 |
| fbcnn_span_ch48 (published) | 1 | 11.68 [7.94; 15.20] | 32.71 | 32.47 [28.05; 36.93] | 8.41 |
| fbcnn_bicubic (published) | 1 | 11.45 [6.34; 15.71] | 29.21 | 31.45 [27.72; 35.70] | 7.24 |
| span/est | 5 | 11.26 [8.29; 13.87] | 33.93 | 31.49 [26.41; 36.92] | 9.02 |
| span_ch48 (published) | 1 | 10.98 [5.01; 16.81] | 30.84 | 33.42 [28.69; 38.28] | 7.24 |
| rrdb_psnr (published) | 1 | 10.05 [4.04; 15.59] | 31.54 | 33.28 [28.49; 37.89] | 7.94 |
| direct | 1 | 9.35 [4.53; 13.60] | 31.07 | 31.01 [26.29; 35.93] | 7.48 |
| bicubic | 1 | 9.11 [4.72; 13.21] | 30.84 | 31.10 [26.42; 36.18] | 7.48 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| realesrnet (published) | +4.21 [-2.49; +11.11] | +1.13 [-2.52; +4.56] |
| bsrnet (published) | +2.57 [-4.78; +9.86] | +1.42 [-2.44; +4.87] |
| fbcnn_span_ch48 (published) | +2.57 [-0.26; +5.64] | +1.37 [-1.86; +3.72] |
| fbcnn_bicubic (published) | +2.34 [+0.00; +5.01] | +0.35 [-2.00; +2.43] |
| span/est | +2.15 [-2.88; +6.70] | +0.39 [-4.12; +3.86] |
| span_ch48 (published) | +1.87 [-1.06; +5.14] | +2.31 [+0.33; +3.94]* |
| rrdb_psnr (published) | +0.93 [-2.27; +4.20] | +2.17 [+0.13; +4.05]* |
| direct | +0.23 [-0.87; +1.10] | -0.10 [-0.68; +0.41] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| realesrnet (published) | +3.97 [-2.72; +11.02] | +1.22 [-2.20; +4.51] |
| bsrnet (published) | +2.34 [-4.96; +9.62] | +1.52 [-2.20; +5.04] |
| fbcnn_span_ch48 (published) | +2.34 [-0.70; +5.96] | +1.47 [-1.53; +3.88] |
| fbcnn_bicubic (published) | +2.10 [-0.25; +5.23] | +0.44 [-1.88; +2.49] |
| span/est | +1.92 [-3.11; +6.65] | +0.48 [-3.86; +3.85] |
| span_ch48 (published) | +1.64 [-1.27; +4.86] | +2.41 [+0.52; +4.04]* |
| rrdb_psnr (published) | +0.70 [-2.22; +3.72] | +2.27 [+0.35; +4.11]* |
| bicubic | -0.23 [-1.10; +0.87] | +0.10 [-0.41; +0.68] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| realesrnet (published) | +2.34 [-5.67; +11.11] | -1.19 [-4.51; +2.50] |
| bsrnet (published) | +0.70 [-8.05; +9.81] | -0.89 [-4.44; +2.71] |
| fbcnn_span_ch48 (published) | +0.70 [-2.91; +5.02] | -0.94 [-3.49; +1.43] |
| fbcnn_bicubic (published) | +0.47 [-3.91; +4.95] | -1.97 [-4.17; +0.45] |
| span/est | +0.28 [-5.82; +6.02] | -1.93 [-5.68; +1.76] |
| rrdb_psnr (published) | -0.93 [-3.09; +1.62] | -0.14 [-1.45; +1.04] |
| direct | -1.64 [-4.86; +1.27] | -2.41 [-4.04; -0.52]* |
| bicubic | -1.87 [-5.14; +1.06] | -2.31 [-3.94; -0.33]* |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| realesrnet (published) | +2.06 [-2.32; +7.58] | +0.74 [-1.65; +3.13] |
| bsrnet (published) | +0.42 [-4.64; +7.09] | +1.04 [-1.50; +3.42] |
| fbcnn_span_ch48 (published) | +0.42 [-3.25; +3.75] | +0.98 [-1.36; +2.89] |
| fbcnn_bicubic (published) | +0.19 [-4.56; +4.44] | -0.04 [-2.91; +2.77] |
| span_ch48 (published) | -0.28 [-6.02; +5.82] | +1.93 [-1.76; +5.68] |
| rrdb_psnr (published) | -1.21 [-6.92; +4.24] | +1.79 [-1.61; +4.91] |
| direct | -1.92 [-6.65; +3.11] | -0.48 [-3.85; +3.86] |
| bicubic | -2.15 [-6.70; +2.88] | -0.39 [-3.86; +4.12] |
