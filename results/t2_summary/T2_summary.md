# T2: mô hình có sẵn trên benchmark AMI (không huấn luyện)

## ×2, đáp án 144 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| swinir_light_x2 | mid | 44.33 [43.99, 44.64] | +2.41 [+2.31, +2.50] | 0.9785 | 42.53 | 44.95 |
| bicubic_sharp | mốc | 42.36 [42.05, 42.66] | +0.44 [+0.43, +0.45] | 0.9694 | 40.58 | 42.73 |
| bicubic | mốc | 41.92 [41.61, 42.22] |  | 0.9661 | 40.23 | 42.27 |
| realesrgan_x2 | perceptual | 35.21 [34.97, 35.43] | -6.72 [-6.94, -6.49] | 0.9115 | 33.33 | 35.18 |

## ×2, đáp án 144 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 38.30 [38.08, 38.52] |  | 0.9261 | 35.12 | 38.42 |
| bicubic_sharp | mốc | 38.03 [37.82, 38.24] | -0.27 [-0.28, -0.26] | 0.9220 | 34.90 | 38.11 |
| swinir_light_x2 | mid | 37.77 [37.57, 37.98] | -0.53 [-0.55, -0.50] | 0.9175 | 34.75 | 37.79 |
| realesrgan_x2 | perceptual | 34.35 [34.14, 34.56] | -3.95 [-4.11, -3.78] | 0.8909 | 32.47 | 34.35 |

## ×4, đáp án 96 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| rrdb_psnr | upper | 37.52 [37.30, 37.74] | +3.22 [+3.09, +3.34] | 0.9292 | 35.78 | 36.49 |
| swinir_light | mid | 37.13 [36.92, 37.34] | +2.83 [+2.72, +2.94] | 0.9240 | 35.41 | 35.85 |
| msrresnet | mid | 37.04 [36.84, 37.25] | +2.74 [+2.64, +2.85] | 0.9229 | 35.31 | 35.76 |
| span_ch48 | light | 37.04 [36.83, 37.25] | +2.73 [+2.63, +2.84] | 0.9228 | 35.29 | 35.73 |
| edsr_baseline | mid | 37.02 [36.82, 37.23] | +2.72 [+2.62, +2.83] | 0.9225 | 35.28 | 35.73 |
| span_ch48_t44 | light | 37.01 [36.81, 37.22] | +2.71 [+2.61, +2.81] | 0.9226 | 35.28 | 35.76 |
| rlfn | light | 36.92 [36.71, 37.12] | +2.62 [+2.52, +2.72] | 0.9214 | 35.18 | 35.57 |
| disp26 | light | 36.90 [36.69, 37.10] | +2.60 [+2.49, +2.70] | 0.9211 | 35.16 | 35.54 |
| smfan | light | 36.89 [36.69, 37.10] | +2.59 [+2.49, +2.69] | 0.9207 | 35.14 | 35.52 |
| efdn | light | 36.86 [36.66, 37.06] | +2.56 [+2.46, +2.66] | 0.9206 | 35.12 | 35.48 |
| errn26 | light | 36.83 [36.63, 37.03] | +2.53 [+2.43, +2.63] | 0.9204 | 35.10 | 35.50 |
| safmnpp | light | 36.77 [36.58, 36.97] | +2.47 [+2.38, +2.57] | 0.9192 | 35.04 | 35.35 |
| dscf26 | light | 36.71 [36.51, 36.90] | +2.41 [+2.32, +2.50] | 0.9180 | 34.98 | 35.34 |
| span26 | light | 36.71 [36.51, 36.90] | +2.41 [+2.31, +2.50] | 0.9183 | 34.98 | 35.32 |
| span_ch28 | light | 36.71 [36.51, 36.90] | +2.41 [+2.31, +2.50] | 0.9183 | 34.98 | 35.32 |
| span_ch26 | light | 36.71 [36.51, 36.90] | +2.40 [+2.31, +2.50] | 0.9180 | 34.97 | 35.34 |
| pds26 | light | 36.70 [36.49, 36.89] | +2.39 [+2.30, +2.49] | 0.9183 | 34.92 | 35.30 |
| pkdsr26 | light | 36.69 [36.49, 36.89] | +2.39 [+2.30, +2.48] | 0.9182 | 34.92 | 35.30 |
| esrgan | perceptual | 34.95 [34.74, 35.16] | +0.65 [+0.51, +0.78] | 0.8805 | 33.14 | 33.78 |
| bicubic_sharp | mốc | 34.51 [34.35, 34.68] | +0.21 [+0.21, +0.22] | 0.8787 | 32.84 | 32.68 |
| bicubic | mốc | 34.30 [34.14, 34.47] |  | 0.8749 | 32.65 | 32.41 |
| bsrgan | perceptual | 31.46 [31.34, 31.60] | -2.84 [-2.97, -2.71] | 0.8261 | 29.31 | 29.48 |
| realesr_compact | perceptual | 31.22 [31.06, 31.37] | -3.08 [-3.23, -2.93] | 0.8508 | 29.23 | 29.71 |
| realesrgan | perceptual | 29.17 [28.99, 29.35] | -5.13 [-5.32, -4.94] | 0.7938 | 27.32 | 27.16 |

## ×4, đáp án 96 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 32.15 [32.02, 32.28] |  | 0.8121 | 29.52 | 30.22 |
| bicubic_sharp | mốc | 32.14 [32.01, 32.27] | -0.01 [-0.01, -0.01] | 0.8107 | 29.50 | 30.22 |
| edsr_baseline | mid | 31.86 [31.75, 31.97] | -0.29 [-0.31, -0.27] | 0.8027 | 29.29 | 29.90 |
| smfan | light | 31.84 [31.72, 31.95] | -0.31 [-0.33, -0.29] | 0.8020 | 29.27 | 29.86 |
| span_ch48_t44 | light | 31.83 [31.71, 31.95] | -0.32 [-0.34, -0.30] | 0.8019 | 29.26 | 29.87 |
| span_ch48 | light | 31.83 [31.71, 31.95] | -0.32 [-0.34, -0.30] | 0.8018 | 29.26 | 29.87 |
| msrresnet | mid | 31.83 [31.71, 31.94] | -0.32 [-0.34, -0.30] | 0.8018 | 29.25 | 29.87 |
| pkdsr26 | light | 31.82 [31.70, 31.94] | -0.33 [-0.35, -0.31] | 0.8016 | 29.28 | 29.87 |
| span_ch26 | light | 31.81 [31.69, 31.93] | -0.34 [-0.36, -0.32] | 0.8012 | 29.26 | 29.85 |
| swinir_light | mid | 31.80 [31.69, 31.92] | -0.35 [-0.37, -0.32] | 0.8010 | 29.24 | 29.83 |
| dscf26 | light | 31.80 [31.68, 31.91] | -0.35 [-0.38, -0.33] | 0.8010 | 29.25 | 29.85 |
| pds26 | light | 31.79 [31.68, 31.91] | -0.36 [-0.38, -0.33] | 0.8006 | 29.25 | 29.84 |
| span26 | light | 31.79 [31.67, 31.91] | -0.36 [-0.38, -0.33] | 0.8006 | 29.24 | 29.83 |
| span_ch28 | light | 31.79 [31.67, 31.91] | -0.36 [-0.38, -0.33] | 0.8006 | 29.24 | 29.83 |
| rrdb_psnr | upper | 31.78 [31.67, 31.89] | -0.37 [-0.39, -0.34] | 0.8006 | 29.22 | 29.83 |
| disp26 | light | 31.75 [31.63, 31.86] | -0.40 [-0.43, -0.38] | 0.7995 | 29.20 | 29.81 |
| rlfn | light | 31.74 [31.62, 31.85] | -0.41 [-0.44, -0.39] | 0.7992 | 29.19 | 29.78 |
| safmnpp | light | 31.73 [31.61, 31.84] | -0.42 [-0.45, -0.40] | 0.7981 | 29.19 | 29.78 |
| errn26 | light | 31.71 [31.59, 31.83] | -0.44 [-0.47, -0.42] | 0.7981 | 29.17 | 29.79 |
| efdn | light | 31.67 [31.55, 31.78] | -0.49 [-0.51, -0.46] | 0.7968 | 29.13 | 29.73 |
| bsrgan | perceptual | 30.97 [30.85, 31.10] | -1.18 [-1.26, -1.09] | 0.7943 | 29.07 | 28.70 |
| esrgan | perceptual | 30.65 [30.52, 30.77] | -1.50 [-1.56, -1.44] | 0.7570 | 28.27 | 28.67 |
| realesr_compact | perceptual | 29.97 [29.81, 30.12] | -2.18 [-2.31, -2.06] | 0.7985 | 28.18 | 28.11 |
| realesrgan | perceptual | 29.04 [28.87, 29.20] | -3.11 [-3.24, -2.98] | 0.7720 | 27.26 | 27.18 |

## ×4, đáp án 144 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| rrdb_psnr | upper | 39.28 [38.99, 39.55] | +2.55 [+2.44, +2.67] | 0.9380 | 37.51 | 40.12 |
| swinir_light | mid | 39.10 [38.81, 39.37] | +2.37 [+2.26, +2.48] | 0.9362 | 37.34 | 39.77 |
| msrresnet | mid | 39.06 [38.77, 39.33] | +2.33 [+2.22, +2.44] | 0.9358 | 37.29 | 39.71 |
| span_ch48 | light | 39.04 [38.76, 39.32] | +2.32 [+2.21, +2.42] | 0.9357 | 37.27 | 39.68 |
| edsr_baseline | mid | 39.03 [38.75, 39.31] | +2.31 [+2.20, +2.41] | 0.9355 | 37.26 | 39.67 |
| span_ch48_t44 | light | 39.00 [38.72, 39.27] | +2.28 [+2.17, +2.38] | 0.9354 | 37.24 | 39.62 |
| rlfn | light | 38.99 [38.71, 39.27] | +2.27 [+2.16, +2.37] | 0.9353 | 37.22 | 39.62 |
| efdn | light | 38.95 [38.66, 39.22] | +2.22 [+2.12, +2.32] | 0.9349 | 37.17 | 39.52 |
| disp26 | light | 38.92 [38.64, 39.19] | +2.19 [+2.09, +2.29] | 0.9346 | 37.15 | 39.46 |
| smfan | light | 38.89 [38.61, 39.16] | +2.17 [+2.07, +2.26] | 0.9342 | 37.11 | 39.42 |
| errn26 | light | 38.87 [38.59, 39.13] | +2.14 [+2.04, +2.24] | 0.9342 | 37.10 | 39.38 |
| safmnpp | light | 38.83 [38.55, 39.10] | +2.11 [+2.01, +2.20] | 0.9337 | 37.07 | 39.28 |
| span26 | light | 38.79 [38.52, 39.05] | +2.06 [+1.97, +2.16] | 0.9332 | 37.02 | 39.25 |
| span_ch28 | light | 38.79 [38.52, 39.05] | +2.06 [+1.97, +2.16] | 0.9332 | 37.02 | 39.25 |
| pkdsr26 | light | 38.76 [38.49, 39.03] | +2.04 [+1.94, +2.13] | 0.9330 | 36.96 | 39.20 |
| dscf26 | light | 38.76 [38.49, 39.03] | +2.03 [+1.94, +2.13] | 0.9328 | 37.00 | 39.21 |
| span_ch26 | light | 38.76 [38.49, 39.03] | +2.03 [+1.94, +2.13] | 0.9328 | 37.00 | 39.20 |
| pds26 | light | 38.76 [38.49, 39.03] | +2.03 [+1.94, +2.12] | 0.9330 | 36.96 | 39.19 |
| bicubic_sharp | mốc | 36.91 [36.69, 37.13] | +0.18 [+0.18, +0.19] | 0.9103 | 35.19 | 36.37 |
| esrgan | perceptual | 36.81 [36.52, 37.08] | +0.08 [-0.05, +0.21] | 0.8983 | 34.96 | 37.47 |
| bicubic | mốc | 36.73 [36.51, 36.94] |  | 0.9084 | 35.03 | 36.11 |
| realesr_compact | perceptual | 33.26 [33.11, 33.42] | -3.46 [-3.62, -3.29] | 0.8876 | 31.18 | 32.03 |
| bsrgan | perceptual | 32.97 [32.84, 33.11] | -3.75 [-3.91, -3.58] | 0.8559 | 30.77 | 31.74 |
| realesrgan | perceptual | 30.99 [30.77, 31.20] | -5.74 [-5.96, -5.53] | 0.8434 | 29.13 | 29.69 |

## ×4, đáp án 144 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 34.36 [34.19, 34.52] |  | 0.8630 | 31.49 | 33.39 |
| bicubic_sharp | mốc | 34.32 [34.16, 34.48] | -0.03 [-0.04, -0.03] | 0.8612 | 31.45 | 33.35 |
| edsr_baseline | mid | 33.95 [33.80, 34.11] | -0.40 [-0.42, -0.38] | 0.8513 | 31.18 | 32.91 |
| msrresnet | mid | 33.94 [33.79, 34.09] | -0.42 [-0.44, -0.40] | 0.8510 | 31.17 | 32.89 |
| smfan | light | 33.94 [33.78, 34.09] | -0.42 [-0.44, -0.40] | 0.8509 | 31.17 | 32.89 |
| span_ch26 | light | 33.93 [33.78, 34.09] | -0.43 [-0.45, -0.41] | 0.8509 | 31.18 | 32.87 |
| span_ch48 | light | 33.93 [33.78, 34.09] | -0.43 [-0.45, -0.41] | 0.8508 | 31.17 | 32.88 |
| pkdsr26 | light | 33.93 [33.78, 34.08] | -0.43 [-0.45, -0.41] | 0.8508 | 31.19 | 32.87 |
| span_ch48_t44 | light | 33.93 [33.77, 34.08] | -0.43 [-0.45, -0.41] | 0.8505 | 31.16 | 32.87 |
| dscf26 | light | 33.92 [33.77, 34.08] | -0.43 [-0.45, -0.41] | 0.8508 | 31.17 | 32.86 |
| swinir_light | mid | 33.92 [33.77, 34.08] | -0.43 [-0.45, -0.41] | 0.8506 | 31.16 | 32.87 |
| span26 | light | 33.91 [33.76, 34.07] | -0.44 [-0.46, -0.42] | 0.8504 | 31.16 | 32.84 |
| span_ch28 | light | 33.91 [33.76, 34.07] | -0.44 [-0.46, -0.42] | 0.8504 | 31.16 | 32.84 |
| pds26 | light | 33.90 [33.75, 34.06] | -0.46 [-0.48, -0.43] | 0.8499 | 31.16 | 32.84 |
| rrdb_psnr | upper | 33.89 [33.73, 34.04] | -0.47 [-0.49, -0.45] | 0.8498 | 31.13 | 32.83 |
| disp26 | light | 33.87 [33.72, 34.02] | -0.49 [-0.51, -0.46] | 0.8493 | 31.12 | 32.81 |
| rlfn | light | 33.87 [33.72, 34.03] | -0.49 [-0.51, -0.47] | 0.8494 | 31.13 | 32.80 |
| safmnpp | light | 33.84 [33.69, 33.99] | -0.52 [-0.54, -0.50] | 0.8482 | 31.10 | 32.77 |
| efdn | light | 33.83 [33.69, 33.99] | -0.52 [-0.54, -0.50] | 0.8484 | 31.09 | 32.77 |
| errn26 | light | 33.83 [33.68, 33.98] | -0.53 [-0.55, -0.50] | 0.8482 | 31.09 | 32.77 |
| esrgan | perceptual | 32.75 [32.60, 32.91] | -1.60 [-1.65, -1.56] | 0.8136 | 30.20 | 31.50 |
| bsrgan | perceptual | 32.37 [32.25, 32.50] | -1.98 [-2.07, -1.89] | 0.8325 | 30.42 | 31.18 |
| realesr_compact | perceptual | 31.50 [31.35, 31.65] | -2.86 [-2.99, -2.72] | 0.8434 | 29.63 | 30.08 |
| realesrgan | perceptual | 30.16 [29.99, 30.33] | -4.19 [-4.35, -4.03] | 0.8124 | 28.38 | 28.87 |

## ×4, đáp án 192 px, suy giảm bic (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| rrdb_psnr | upper | 40.12 [39.79, 40.43] | +1.99 [+1.89, +2.09] | 0.9412 | 38.38 | 41.04 |
| swinir_light | mid | 40.03 [39.70, 40.33] | +1.89 [+1.80, +1.99] | 0.9403 | 38.29 | 40.90 |
| span_ch48 | light | 39.98 [39.65, 40.28] | +1.84 [+1.75, +1.94] | 0.9398 | 38.24 | 40.83 |
| msrresnet | mid | 39.97 [39.65, 40.27] | +1.84 [+1.75, +1.93] | 0.9398 | 38.23 | 40.83 |
| edsr_baseline | mid | 39.96 [39.64, 40.26] | +1.83 [+1.73, +1.91] | 0.9397 | 38.22 | 40.81 |
| rlfn | light | 39.94 [39.61, 40.25] | +1.81 [+1.71, +1.90] | 0.9396 | 38.20 | 40.80 |
| span_ch48_t44 | light | 39.93 [39.61, 40.23] | +1.80 [+1.70, +1.88] | 0.9395 | 38.19 | 40.77 |
| efdn | light | 39.91 [39.59, 40.21] | +1.78 [+1.69, +1.86] | 0.9394 | 38.17 | 40.74 |
| disp26 | light | 39.89 [39.57, 40.19] | +1.75 [+1.67, +1.84] | 0.9391 | 38.15 | 40.71 |
| smfan | light | 39.87 [39.55, 40.17] | +1.74 [+1.65, +1.82] | 0.9389 | 38.13 | 40.68 |
| safmnpp | light | 39.83 [39.52, 40.13] | +1.70 [+1.62, +1.78] | 0.9386 | 38.09 | 40.61 |
| errn26 | light | 39.83 [39.51, 40.13] | +1.70 [+1.61, +1.78] | 0.9387 | 38.09 | 40.62 |
| pkdsr26 | light | 39.78 [39.47, 40.08] | +1.65 [+1.57, +1.73] | 0.9381 | 38.02 | 40.57 |
| pds26 | light | 39.78 [39.47, 40.08] | +1.65 [+1.57, +1.73] | 0.9381 | 38.02 | 40.57 |
| span26 | light | 39.78 [39.46, 40.08] | +1.65 [+1.57, +1.73] | 0.9380 | 38.05 | 40.56 |
| span_ch28 | light | 39.78 [39.46, 40.08] | +1.65 [+1.57, +1.73] | 0.9380 | 38.05 | 40.56 |
| dscf26 | light | 39.76 [39.45, 40.06] | +1.63 [+1.55, +1.71] | 0.9378 | 38.03 | 40.53 |
| span_ch26 | light | 39.76 [39.44, 40.06] | +1.63 [+1.55, +1.71] | 0.9378 | 38.03 | 40.53 |
| bicubic_sharp | mốc | 38.28 [38.01, 38.56] | +0.15 [+0.15, +0.16] | 0.9222 | 36.57 | 38.48 |
| bicubic | mốc | 38.13 [37.86, 38.40] |  | 0.9209 | 36.45 | 38.28 |
| esrgan | perceptual | 37.52 [37.19, 37.83] | -0.61 [-0.74, -0.50] | 0.9013 | 35.69 | 38.37 |
| realesr_compact | perceptual | 34.77 [34.60, 34.94] | -3.37 [-3.54, -3.19] | 0.9047 | 32.66 | 34.18 |
| bsrgan | perceptual | 33.99 [33.85, 34.14] | -4.14 [-4.33, -3.94] | 0.8680 | 31.90 | 33.49 |
| realesrgan | perceptual | 32.31 [32.04, 32.56] | -5.82 [-6.08, -5.58] | 0.8662 | 30.48 | 31.79 |

## ×4, đáp án 192 px, suy giảm bicjpeg75 (700 ảnh)

| Mô hình | Nhóm | PSNR-Y (dB) [KTC 95%] | Hơn bicubic (dB) [KTC 95%] | SSIM-Y | PSNR-RGB | PSNR-Y vùng giữa |
|---|---|---|---|---|---|---|
| bicubic | mốc | 35.65 [35.46, 35.85] |  | 0.8833 | 32.77 | 35.42 |
| bicubic_sharp | mốc | 35.60 [35.41, 35.80] | -0.05 [-0.05, -0.05] | 0.8817 | 32.72 | 35.36 |
| edsr_baseline | mid | 35.26 [35.07, 35.44] | -0.40 [-0.42, -0.37] | 0.8739 | 32.47 | 34.90 |
| span_ch26 | light | 35.25 [35.07, 35.43] | -0.40 [-0.42, -0.38] | 0.8739 | 32.48 | 34.89 |
| dscf26 | light | 35.25 [35.07, 35.43] | -0.41 [-0.42, -0.38] | 0.8738 | 32.48 | 34.88 |
| msrresnet | mid | 35.25 [35.06, 35.43] | -0.41 [-0.43, -0.38] | 0.8737 | 32.46 | 34.89 |
| pkdsr26 | light | 35.24 [35.06, 35.42] | -0.41 [-0.43, -0.39] | 0.8735 | 32.48 | 34.87 |
| span_ch48 | light | 35.24 [35.05, 35.42] | -0.42 [-0.43, -0.39] | 0.8734 | 32.46 | 34.87 |
| smfan | light | 35.24 [35.06, 35.42] | -0.42 [-0.43, -0.39] | 0.8734 | 32.46 | 34.87 |
| swinir_light | mid | 35.24 [35.05, 35.42] | -0.42 [-0.44, -0.39] | 0.8735 | 32.46 | 34.87 |
| span26 | light | 35.23 [35.05, 35.42] | -0.42 [-0.44, -0.40] | 0.8734 | 32.46 | 34.86 |
| span_ch28 | light | 35.23 [35.05, 35.42] | -0.42 [-0.44, -0.40] | 0.8734 | 32.46 | 34.86 |
| span_ch48_t44 | light | 35.23 [35.05, 35.41] | -0.42 [-0.44, -0.40] | 0.8732 | 32.46 | 34.87 |
| pds26 | light | 35.21 [35.03, 35.39] | -0.44 [-0.46, -0.42] | 0.8728 | 32.45 | 34.84 |
| rrdb_psnr | upper | 35.21 [35.03, 35.39] | -0.44 [-0.46, -0.42] | 0.8730 | 32.44 | 34.84 |
| disp26 | light | 35.19 [35.01, 35.37] | -0.46 [-0.48, -0.44] | 0.8725 | 32.42 | 34.81 |
| rlfn | light | 35.19 [35.01, 35.37] | -0.47 [-0.49, -0.44] | 0.8725 | 32.43 | 34.80 |
| efdn | light | 35.16 [34.99, 35.35] | -0.49 [-0.51, -0.46] | 0.8719 | 32.40 | 34.78 |
| safmnpp | light | 35.16 [34.98, 35.34] | -0.49 [-0.51, -0.47] | 0.8715 | 32.40 | 34.78 |
| errn26 | light | 35.15 [34.97, 35.33] | -0.50 [-0.53, -0.48] | 0.8715 | 32.39 | 34.77 |
| esrgan | perceptual | 34.20 [34.02, 34.37] | -1.46 [-1.50, -1.41] | 0.8457 | 31.59 | 33.61 |
| bsrgan | perceptual | 33.40 [33.26, 33.55] | -2.25 [-2.35, -2.14] | 0.8500 | 31.43 | 33.00 |
| realesr_compact | perceptual | 33.12 [32.96, 33.27] | -2.53 [-2.66, -2.40] | 0.8714 | 31.15 | 32.43 |
| realesrgan | perceptual | 31.53 [31.34, 31.74] | -4.12 [-4.31, -3.93] | 0.8429 | 29.72 | 31.05 |

## Thứ hạng: bicubic so với bicubic + JPEG 75 (mô hình nhẹ và vừa, tối ưu PSNR)

**×4, 96 px.** τ-b (có hòa) = 0.13; cặp cùng chiều 68, đổi chiều có ý nghĩa 52; τ trên thứ hạng trung bình = 0.10 [0.05, 0.19]. Kết luận theo quy tắc ghi trước: **thứ hạng đảo**.

- thứ tự ở bic: swinir_light, msrresnet, span_ch48, edsr_baseline, span_ch48_t44, rlfn, disp26, smfan, efdn, errn26, safmnpp, dscf26, span_ch28, span26, span_ch26, pds26, pkdsr26
- thứ tự ở bicjpeg75: edsr_baseline, smfan, span_ch48_t44, span_ch48, msrresnet, pkdsr26, span_ch26, swinir_light, dscf26, pds26, span26, span_ch28, disp26, rlfn, safmnpp, errn26, efdn
- cặp đổi chiều: dscf26 / disp26, dscf26 / errn26, edsr_baseline / swinir_light, efdn / dscf26, efdn / errn26, efdn / pds26, efdn / pkdsr26, efdn / safmnpp, efdn / span26, msrresnet / edsr_baseline, msrresnet / swinir_light, pds26 / disp26, pds26 / errn26, pds26 / pkdsr26, pkdsr26 / disp26, pkdsr26 / dscf26, pkdsr26 / errn26, rlfn / disp26, rlfn / dscf26, rlfn / pds26, rlfn / pkdsr26, rlfn / smfan, rlfn / span26, safmnpp / dscf26, safmnpp / errn26, safmnpp / pds26, safmnpp / pkdsr26, safmnpp / span26, smfan / msrresnet, smfan / swinir_light, span26 / disp26, span26 / errn26, span26 / pkdsr26, span_ch26 / disp26, span_ch26 / dscf26, span_ch26 / efdn, span_ch26 / errn26, span_ch26 / pkdsr26, span_ch26 / rlfn, span_ch26 / safmnpp, span_ch28 / disp26, span_ch28 / efdn, span_ch28 / errn26, span_ch28 / pkdsr26, span_ch28 / rlfn, span_ch28 / safmnpp, span_ch48 / edsr_baseline, span_ch48 / smfan, span_ch48 / swinir_light, span_ch48_t44 / smfan, span_ch48_t44 / swinir_light, swinir_light / pkdsr26

**×4, 144 px.** τ-b (có hòa) = 0.13; cặp cùng chiều 68, đổi chiều có ý nghĩa 52; τ trên thứ hạng trung bình = 0.11 [0.07, 0.16]. Kết luận theo quy tắc ghi trước: **thứ hạng đảo**.

- thứ tự ở bic: swinir_light, msrresnet, span_ch48, edsr_baseline, span_ch48_t44, rlfn, efdn, disp26, smfan, errn26, safmnpp, span26, span_ch28, pkdsr26, dscf26, span_ch26, pds26
- thứ tự ở bicjpeg75: edsr_baseline, msrresnet, smfan, span_ch26, span_ch48, pkdsr26, span_ch48_t44, dscf26, swinir_light, span26, span_ch28, pds26, disp26, rlfn, safmnpp, efdn, errn26
- cặp đổi chiều: dscf26 / disp26, dscf26 / errn26, edsr_baseline / swinir_light, efdn / disp26, efdn / dscf26, efdn / pds26, efdn / pkdsr26, efdn / smfan, efdn / span26, msrresnet / edsr_baseline, msrresnet / swinir_light, pds26 / disp26, pds26 / errn26, pkdsr26 / disp26, pkdsr26 / errn26, rlfn / dscf26, rlfn / pds26, rlfn / pkdsr26, rlfn / smfan, rlfn / span26, safmnpp / dscf26, safmnpp / errn26, safmnpp / pds26, safmnpp / pkdsr26, safmnpp / span26, smfan / disp26, smfan / swinir_light, span26 / disp26, span26 / dscf26, span26 / errn26, span26 / pkdsr26, span_ch26 / disp26, span_ch26 / efdn, span_ch26 / errn26, span_ch26 / rlfn, span_ch26 / safmnpp, span_ch26 / span26, span_ch26 / swinir_light, span_ch28 / disp26, span_ch28 / dscf26, span_ch28 / efdn, span_ch28 / errn26, span_ch28 / pkdsr26, span_ch28 / rlfn, span_ch28 / safmnpp, span_ch28 / span_ch26, span_ch48 / edsr_baseline, span_ch48 / smfan, span_ch48 / swinir_light, span_ch48_t44 / smfan, span_ch48_t44 / span_ch26, swinir_light / pkdsr26

**×4, 192 px.** τ-b (có hòa) = -0.04; cặp cùng chiều 55, đổi chiều có ý nghĩa 60; τ trên thứ hạng trung bình = -0.05 [-0.08, 0.04]. Kết luận theo quy tắc ghi trước: **thứ hạng đảo**.

- thứ tự ở bic: swinir_light, span_ch48, msrresnet, edsr_baseline, rlfn, span_ch48_t44, efdn, disp26, smfan, safmnpp, errn26, pkdsr26, pds26, span26, span_ch28, dscf26, span_ch26
- thứ tự ở bicjpeg75: edsr_baseline, span_ch26, dscf26, msrresnet, pkdsr26, span_ch48, smfan, swinir_light, span_ch28, span26, span_ch48_t44, pds26, disp26, rlfn, efdn, safmnpp, errn26
- cặp đổi chiều: dscf26 / disp26, dscf26 / errn26, edsr_baseline / swinir_light, efdn / disp26, efdn / dscf26, efdn / pds26, efdn / pkdsr26, efdn / smfan, efdn / span26, msrresnet / edsr_baseline, msrresnet / swinir_light, pds26 / disp26, pds26 / dscf26, pds26 / errn26, pkdsr26 / disp26, pkdsr26 / dscf26, pkdsr26 / errn26, rlfn / disp26, rlfn / dscf26, rlfn / pds26, rlfn / pkdsr26, rlfn / smfan, rlfn / span26, safmnpp / dscf26, safmnpp / pds26, safmnpp / pkdsr26, safmnpp / span26, smfan / disp26, smfan / dscf26, span26 / disp26, span26 / dscf26, span26 / errn26, span_ch26 / disp26, span_ch26 / dscf26, span_ch26 / efdn, span_ch26 / errn26, span_ch26 / msrresnet, span_ch26 / pds26, span_ch26 / pkdsr26, span_ch26 / rlfn, span_ch26 / safmnpp, span_ch26 / smfan, span_ch26 / span26, span_ch26 / swinir_light, span_ch28 / disp26, span_ch28 / dscf26, span_ch28 / efdn, span_ch28 / errn26, span_ch28 / rlfn, span_ch28 / safmnpp, span_ch28 / span_ch26, span_ch48 / dscf26, span_ch48 / edsr_baseline, span_ch48 / span_ch26, span_ch48_t44 / dscf26, span_ch48_t44 / pkdsr26, span_ch48_t44 / rlfn, span_ch48_t44 / smfan, span_ch48_t44 / span_ch26, swinir_light / dscf26

## Mức chênh nhỏ nhất phát hiện được (ô chính, cặp span_ch48 và rlfn)

| Số đo | Số người | Độ lệch chuẩn giữa người | MDE (α = 0,05; lực 0,8) |
|---|---|---|---|
| psnr_y | 60 | 0.0363 | 0.0131 |
| psnr_y | 100 | 0.0363 | 0.0102 |
| ssim_y | 60 | 0.0003 | 0.0001 |
| ssim_y | 100 | 0.0003 | 0.0001 |

MDE này tính từ độ biến thiên của chênh lệch giữa hai mô hình có sẵn. Chênh lệch giữa hai mô hình cùng thân, cùng fold thường biến thiên ít hơn, nên đây là ước lượng thận trọng.
