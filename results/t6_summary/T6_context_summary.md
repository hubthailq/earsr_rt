# T6 (i), (v): phép thử ngữ cảnh (không huấn luyện)

Mất khi thiếu ngữ cảnh = PSNR-Y của cửa sổ khi mạng thấy ngữ cảnh thật đầy đủ, trừ PSNR-Y khi mạng chỉ thấy cửa sổ. Cùng cửa sổ, cùng điểm ảnh; so ghép cặp, bootstrap theo người (hoặc theo ảnh).

## ami, cấu hình small (700 ảnh)

| Mô hình | PSNR đủ ngữ cảnh | Mất khi thiếu ngữ cảnh (dB) [KTC 95%] | Hồi phục hết ở | MSE vành sát mép (lần) | MSE vành trong (lần) | Đệm trước lặp viền (dB) | Đệm trước phản chiếu (dB) |
|---|---|---|---|---|---|---|---|
| rlfn | 40.28 | 0.347 [0.328, 0.368] | 8 px | 1.33 | 1.015 | -1.62 | -1.76 |
| span_ch28 | 39.95 | 0.325 [0.306, 0.346] | 4 px | 1.35 | 1.000 | -1.54 | -1.71 |
| span_ch48 | 40.27 | 0.286 [0.272, 0.303] | 4 px | 1.32 | 1.000 | -1.94 | -1.90 |
| bicubic | 36.97 | 0.214 [0.203, 0.226] | 2 px | 1.35 | 1.000 | -0.03 | -0.49 |

## ami, cấu hình wide (700 ảnh)

| Mô hình | PSNR đủ ngữ cảnh | Mất khi thiếu ngữ cảnh (dB) [KTC 95%] | Hồi phục hết ở | MSE vành sát mép (lần) | MSE vành trong (lần) | Đệm trước lặp viền (dB) | Đệm trước phản chiếu (dB) |
|---|---|---|---|---|---|---|---|
| safmnpp | 43.47 | 0.202 [0.190, 0.214] | 8 px | 1.36 | 1.003 | -0.58 | -0.87 |
| bicubic | 42.08 | 0.166 [0.154, 0.180] | 4 px | 1.38 | 1.000 | -0.03 | -0.45 |
| span_ch28 | 43.50 | 0.155 [0.145, 0.166] | 4 px | 1.31 | 1.000 | -0.71 | -0.88 |
| rlfn | 43.61 | 0.142 [0.131, 0.156] | 4 px | 1.27 | 1.001 | -0.79 | -0.93 |
| efdn | 43.58 | 0.141 [0.132, 0.152] | 4 px | 1.27 | 1.001 | -0.80 | -0.96 |
| span_ch48 | 43.58 | 0.125 [0.116, 0.135] | 4 px | 1.25 | 1.000 | -0.98 | -0.97 |

## urban100, cấu hình small (100 ảnh)

| Mô hình | PSNR đủ ngữ cảnh | Mất khi thiếu ngữ cảnh (dB) [KTC 95%] | Hồi phục hết ở | MSE vành sát mép (lần) | MSE vành trong (lần) | Đệm trước lặp viền (dB) | Đệm trước phản chiếu (dB) |
|---|---|---|---|---|---|---|---|
| rlfn | 23.17 | 0.262 [0.196, 0.344] | 8 px | 1.21 | 1.036 | -0.33 | -0.34 |
| efdn | 23.03 | 0.203 [0.164, 0.251] | 8 px | 1.18 | 1.011 | -0.33 | -0.36 |
| span_ch48 | 23.18 | 0.180 [0.143, 0.224] | 4 px | 1.19 | 0.999 | -0.45 | -0.47 |
| msrresnet | 23.13 | 0.162 [0.129, 0.200] | 4 px | 1.17 | 1.000 | -0.40 | -0.52 |
| span_ch28 | 22.82 | 0.134 [0.101, 0.174] | 2 px | 1.13 | 1.000 | -0.37 | -0.39 |
| bicubic | 21.27 | 0.025 [0.018, 0.032] | 2 px | 1.03 | 1.000 | -0.00 | -0.06 |

## urban100, cấu hình wide (100 ảnh)

| Mô hình | PSNR đủ ngữ cảnh | Mất khi thiếu ngữ cảnh (dB) [KTC 95%] | Hồi phục hết ở | MSE vành sát mép (lần) | MSE vành trong (lần) | Đệm trước lặp viền (dB) | Đệm trước phản chiếu (dB) |
|---|---|---|---|---|---|---|---|
| span_ch48 | 24.57 | 0.152 [0.114, 0.198] | 4 px | 1.19 | 1.000 | -0.52 | -0.61 |
| efdn | 24.44 | 0.142 [0.083, 0.192] | 4 px | 1.19 | 1.000 | -0.44 | -0.55 |
| rlfn | 24.55 | 0.136 [0.062, 0.190] | 4 px | 1.19 | 0.999 | -0.47 | -0.60 |
| msrresnet | 24.57 | 0.135 [0.091, 0.177] | 4 px | 1.18 | 1.000 | -0.49 | -0.67 |
| span_ch28 | 24.08 | 0.090 [0.048, 0.128] | 4 px | 1.14 | 1.000 | -0.43 | -0.50 |
| bicubic | 21.89 | 0.021 [0.016, 0.027] | 4 px | 1.04 | 1.000 | -0.00 | -0.05 |
