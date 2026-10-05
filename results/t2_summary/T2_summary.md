# T2: mô hình có sẵn trên benchmark AMI (không huấn luyện)

## ×4, đáp án 96 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| swinir_light | mid | 37.13 [36.91, 37.34] | +2.83 [+2.71, +2.94] | 0.9240 | 35.41 | 35.85 |
| msrresnet | mid | 37.04 [36.82, 37.26] | +2.74 [+2.63, +2.85] | 0.9229 | 35.31 | 35.76 |
| span_ch48 | light | 37.01 [36.79, 37.23] | +2.71 [+2.60, +2.82] | 0.9226 | 35.28 | 35.76 |
| rlfn | light | 36.92 [36.70, 37.13] | +2.62 [+2.51, +2.73] | 0.9214 | 35.18 | 35.57 |
| smfan | light | 36.89 [36.68, 37.11] | +2.59 [+2.49, +2.70] | 0.9207 | 35.14 | 35.52 |
| efdn | light | 36.86 [36.66, 37.07] | +2.56 [+2.46, +2.66] | 0.9206 | 35.12 | 35.48 |
| safmnpp | light | 36.77 [36.56, 36.99] | +2.47 [+2.37, +2.57] | 0.9192 | 35.04 | 35.35 |
| span_ch28 | light | 36.71 [36.50, 36.92] | +2.41 [+2.31, +2.50] | 0.9183 | 34.98 | 35.32 |
| span_ch26 | light | 36.71 [36.50, 36.92] | +2.40 [+2.31, +2.50] | 0.9180 | 34.97 | 35.34 |
| bicubic_sharp | mốc | 34.51 [34.35, 34.69] | +0.21 [+0.21, +0.22] | 0.8787 | 32.84 | 32.68 |
| bicubic | mốc | 34.30 [34.14, 34.47] |  | 0.8749 | 32.65 | 32.41 |

## ×4, đáp án 96 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 32.15 [32.02, 32.28] |  | 0.8121 | 29.52 | 30.22 |
| bicubic_sharp | mốc | 32.14 [32.01, 32.27] | -0.01 [-0.01, -0.01] | 0.8107 | 29.50 | 30.22 |
| smfan | light | 31.84 [31.72, 31.95] | -0.31 [-0.33, -0.29] | 0.8020 | 29.27 | 29.86 |
| span_ch48 | light | 31.83 [31.71, 31.95] | -0.32 [-0.34, -0.30] | 0.8019 | 29.26 | 29.87 |
| msrresnet | mid | 31.83 [31.71, 31.94] | -0.32 [-0.34, -0.30] | 0.8018 | 29.25 | 29.87 |
| span_ch26 | light | 31.81 [31.69, 31.93] | -0.34 [-0.36, -0.32] | 0.8012 | 29.26 | 29.85 |
| swinir_light | mid | 31.80 [31.69, 31.92] | -0.35 [-0.37, -0.32] | 0.8010 | 29.24 | 29.83 |
| span_ch28 | light | 31.79 [31.67, 31.91] | -0.36 [-0.38, -0.33] | 0.8006 | 29.24 | 29.83 |
| rlfn | light | 31.74 [31.62, 31.85] | -0.41 [-0.44, -0.39] | 0.7992 | 29.19 | 29.78 |
| safmnpp | light | 31.73 [31.61, 31.84] | -0.42 [-0.45, -0.40] | 0.7981 | 29.19 | 29.78 |
| efdn | light | 31.67 [31.55, 31.78] | -0.49 [-0.51, -0.46] | 0.7968 | 29.13 | 29.73 |

## ×4, đáp án 144 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| rrdb_psnr | upper | 39.28 [38.97, 39.56] | +2.55 [+2.44, +2.66] | 0.9380 | 37.51 | 40.12 |
| swinir_light | mid | 39.10 [38.79, 39.37] | +2.37 [+2.26, +2.48] | 0.9362 | 37.34 | 39.77 |
| msrresnet | mid | 39.06 [38.76, 39.33] | +2.33 [+2.22, +2.44] | 0.9358 | 37.29 | 39.71 |
| span_ch48 | light | 39.00 [38.70, 39.28] | +2.28 [+2.17, +2.38] | 0.9354 | 37.24 | 39.62 |
| rlfn | light | 38.99 [38.69, 39.27] | +2.27 [+2.16, +2.37] | 0.9353 | 37.22 | 39.62 |
| efdn | light | 38.95 [38.65, 39.22] | +2.22 [+2.12, +2.32] | 0.9349 | 37.17 | 39.52 |
| smfan | light | 38.89 [38.60, 39.16] | +2.17 [+2.07, +2.26] | 0.9342 | 37.11 | 39.42 |
| safmnpp | light | 38.83 [38.54, 39.10] | +2.11 [+2.01, +2.20] | 0.9337 | 37.07 | 39.28 |
| span_ch28 | light | 38.79 [38.50, 39.06] | +2.06 [+1.97, +2.16] | 0.9332 | 37.02 | 39.25 |
| span_ch26 | light | 38.76 [38.47, 39.03] | +2.03 [+1.94, +2.12] | 0.9328 | 37.00 | 39.20 |
| bicubic_sharp | mốc | 36.91 [36.68, 37.13] | +0.18 [+0.18, +0.19] | 0.9103 | 35.19 | 36.37 |
| bicubic | mốc | 36.73 [36.51, 36.95] |  | 0.9084 | 35.03 | 36.11 |

## ×4, đáp án 144 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 34.36 [34.19, 34.53] |  | 0.8630 | 31.49 | 33.39 |
| bicubic_sharp | mốc | 34.32 [34.16, 34.49] | -0.03 [-0.04, -0.03] | 0.8612 | 31.45 | 33.35 |
| msrresnet | mid | 33.94 [33.79, 34.10] | -0.42 [-0.43, -0.40] | 0.8510 | 31.17 | 32.89 |
| smfan | light | 33.94 [33.78, 34.10] | -0.42 [-0.44, -0.40] | 0.8509 | 31.17 | 32.89 |
| span_ch26 | light | 33.93 [33.78, 34.09] | -0.43 [-0.44, -0.41] | 0.8509 | 31.18 | 32.87 |
| span_ch48 | light | 33.93 [33.77, 34.08] | -0.43 [-0.45, -0.41] | 0.8505 | 31.16 | 32.87 |
| swinir_light | mid | 33.92 [33.77, 34.08] | -0.43 [-0.45, -0.41] | 0.8506 | 31.16 | 32.87 |
| span_ch28 | light | 33.91 [33.76, 34.07] | -0.44 [-0.46, -0.42] | 0.8504 | 31.16 | 32.84 |
| rlfn | light | 33.87 [33.72, 34.03] | -0.49 [-0.50, -0.47] | 0.8494 | 31.13 | 32.80 |
| safmnpp | light | 33.84 [33.69, 33.99] | -0.52 [-0.54, -0.50] | 0.8482 | 31.10 | 32.77 |
| efdn | light | 33.83 [33.69, 33.99] | -0.52 [-0.54, -0.50] | 0.8484 | 31.09 | 32.77 |

## ×4, đáp án 192 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| msrresnet | mid | 39.97 [39.64, 40.28] | +1.84 [+1.75, +1.93] | 0.9398 | 38.23 | 40.83 |
| rlfn | light | 39.94 [39.61, 40.25] | +1.81 [+1.72, +1.90] | 0.9396 | 38.20 | 40.80 |
| span_ch48 | light | 39.93 [39.60, 40.23] | +1.80 [+1.71, +1.89] | 0.9395 | 38.19 | 40.77 |
| efdn | light | 39.91 [39.58, 40.21] | +1.78 [+1.69, +1.87] | 0.9394 | 38.17 | 40.74 |
| smfan | light | 39.87 [39.55, 40.17] | +1.74 [+1.65, +1.83] | 0.9389 | 38.13 | 40.68 |
| safmnpp | light | 39.83 [39.51, 40.13] | +1.70 [+1.62, +1.79] | 0.9386 | 38.09 | 40.61 |
| span_ch28 | light | 39.78 [39.46, 40.08] | +1.65 [+1.57, +1.73] | 0.9380 | 38.05 | 40.56 |
| span_ch26 | light | 39.76 [39.44, 40.06] | +1.63 [+1.55, +1.71] | 0.9378 | 38.03 | 40.53 |
| bicubic_sharp | mốc | 38.28 [38.00, 38.55] | +0.15 [+0.15, +0.16] | 0.9222 | 36.57 | 38.48 |
| bicubic | mốc | 38.13 [37.86, 38.40] |  | 0.9209 | 36.45 | 38.28 |

## Thứ hạng: bicubic so với bicubic + JPEG 75 (mô hình nhẹ và vừa, tối ưu PSNR)

**×4, 96 px.** τ-b (có hòa) = 0.20; cặp cùng chiều 20, đổi chiều có ý nghĩa 13; τ trên thứ hạng trung bình = 0.11 [0.11, 0.22]. Kết luận theo quy tắc ghi trước: **thứ hạng đảo**.

- thứ tự ở bic: swinir_light, msrresnet, span_ch48, rlfn, smfan, efdn, safmnpp, span_ch28, span_ch26
- thứ tự ở bicjpeg75: smfan, span_ch48, msrresnet, span_ch26, swinir_light, span_ch28, rlfn, safmnpp, efdn
- cặp đổi chiều: efdn / safmnpp, msrresnet / swinir_light, rlfn / smfan, smfan / msrresnet, smfan / swinir_light, span_ch26 / efdn, span_ch26 / rlfn, span_ch26 / safmnpp, span_ch28 / efdn, span_ch28 / rlfn, span_ch28 / safmnpp, span_ch48 / smfan, span_ch48 / swinir_light

**×4, 144 px.** τ-b (có hòa) = 0.15; cặp cùng chiều 19, đổi chiều có ý nghĩa 14; τ trên thứ hạng trung bình = 0.11 [0.06, 0.17]. Kết luận theo quy tắc ghi trước: **thứ hạng đảo**.

- thứ tự ở bic: swinir_light, msrresnet, span_ch48, rlfn, efdn, smfan, safmnpp, span_ch28, span_ch26
- thứ tự ở bicjpeg75: msrresnet, smfan, span_ch26, span_ch48, swinir_light, span_ch28, rlfn, safmnpp, efdn
- cặp đổi chiều: efdn / smfan, msrresnet / swinir_light, rlfn / smfan, smfan / swinir_light, span_ch26 / efdn, span_ch26 / rlfn, span_ch26 / safmnpp, span_ch26 / swinir_light, span_ch28 / efdn, span_ch28 / rlfn, span_ch28 / safmnpp, span_ch28 / span_ch26, span_ch48 / smfan, span_ch48 / span_ch26

## Mức chênh nhỏ nhất phát hiện được (ô chính, cặp span_ch48 và rlfn)

| Số đo | Số người | Độ lệch chuẩn giữa người | MDE (α = 0,05; lực 0,8) |
|---|---|---|---|
| psnr_y | 60 | 0.0452 | 0.0163 |
| psnr_y | 100 | 0.0452 | 0.0126 |
| ssim_y | 60 | 0.0003 | 0.0001 |
| ssim_y | 100 | 0.0003 | 0.0001 |

MDE này tính từ độ biến thiên của chênh lệch giữa hai mô hình có sẵn. Chênh lệch giữa hai mô hình cùng thân, cùng fold thường biến thiên ít hơn, nên đây là ước lượng thận trọng.
