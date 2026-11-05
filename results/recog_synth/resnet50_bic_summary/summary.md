# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 1887 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet50/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| span_ch48 (published) | 1 | 65.34 [61.54; 69.09] | 90.83 | 10.38 [9.44; 11.52] | 54.32 |
| ref_large | 1 | 64.92 [60.99; 68.97] | 91.26 | 10.40 [9.30; 11.32] | 56.12 |
| bicubic | 1 | 63.17 [59.18; 67.01] | 89.40 | 11.20 [10.06; 12.24] | 52.62 |
| direct | 1 | 63.06 [59.10; 66.92] | 89.35 | 11.22 [10.06; 12.29] | 52.68 |
| span/est | 5 | 59.84 [55.58; 64.13] | 87.84 | 11.81 [10.53; 13.15] | 48.18 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +2.17 [+0.72; +3.95]* | -0.82 [-1.42; -0.02]* |
| direct | -0.11 [-0.55; +0.23] | +0.02 [-0.15; +0.17] |
| span/est | -3.33 [-6.05; -0.58]* | +0.61 [-0.30; +1.72] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +2.28 [+0.79; +4.22]* | -0.84 [-1.45; -0.06]* |
| bicubic | +0.11 [-0.23; +0.55] | -0.02 [-0.17; +0.15] |
| span/est | -3.22 [-5.86; -0.44]* | +0.60 [-0.35; +1.77] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | -2.17 [-3.95; -0.72]* | +0.82 [+0.02; +1.42]* |
| direct | -2.28 [-4.22; -0.79]* | +0.84 [+0.06; +1.45]* |
| span/est | -5.50 [-7.54; -3.44]* | +1.43 [+0.55; +2.26]* |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | +5.50 [+3.44; +7.54]* | -1.43 [-2.26; -0.55]* |
| bicubic | +3.33 [+0.58; +6.05]* | -0.61 [-1.72; +0.30] |
| direct | +3.22 [+0.44; +5.86]* | -0.60 [-1.77; +0.35] |
