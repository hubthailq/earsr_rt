# Nhận dạng tai trên ảnh nhỏ thật (data/raw/EarVN1.0; test, viewer): 51 người, 1887 ảnh dò, 1862 ảnh đăng ký

Mạng nhận dạng: `runs/RECOG_resnet50/ckpt.pt`. Khoảng tin cậy 95%, bootstrap theo người. Đơn vị: %.

| Nhánh | Số mô hình | Rank-1 | Rank-5 | EER | TAR ở FAR 1% |
|---|---|---|---|---|---|
| ref_large | 1 | 64.92 [60.99; 68.97] | 91.26 | 10.40 [9.30; 11.32] | 56.12 |
| bicubic | 1 | 61.63 [56.86; 65.86] | 87.33 | 11.87 [10.58; 13.27] | 50.77 |
| span_ch48 (published) | 1 | 61.58 [57.25; 65.80] | 88.29 | 11.47 [10.49; 13.00] | 51.93 |
| direct | 1 | 61.37 [56.43; 65.82] | 87.33 | 11.77 [10.56; 13.30] | 50.56 |
| span/est | 5 | 59.62 [54.74; 64.33] | 87.37 | 11.99 [10.79; 13.38] | 47.79 |

## Chênh lệch ghép cặp so với `bicubic` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| span_ch48 (published) | -0.05 [-1.91; +1.71] | -0.40 [-0.93; +0.36] |
| direct | -0.26 [-0.79; +0.19] | -0.10 [-0.18; +0.17] |
| span/est | -2.01 [-5.42; +1.64] | +0.12 [-0.83; +1.21] |

## Chênh lệch ghép cặp so với `direct` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | +0.26 [-0.19; +0.79] | +0.10 [-0.17; +0.18] |
| span_ch48 (published) | +0.21 [-1.48; +1.97] | -0.30 [-0.90; +0.36] |
| span/est | -1.75 [-5.13; +1.96] | +0.22 [-0.83; +1.27] |

## Chênh lệch ghép cặp so với `span_ch48 (published)` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | +0.05 [-1.71; +1.91] | +0.40 [-0.36; +0.93] |
| direct | -0.21 [-1.97; +1.48] | +0.30 [-0.36; +0.90] |
| span/est | -1.96 [-4.42; +0.72] | +0.52 [-0.45; +1.38] |

## Chênh lệch ghép cặp so với `span/est` (điểm phần trăm; dấu * là khoảng tin cậy không chứa 0)

| Nhánh | Rank-1 | EER |
|---|---|---|
| bicubic | +2.01 [-1.64; +5.42] | -0.12 [-1.21; +0.83] |
| span_ch48 (published) | +1.96 [-0.72; +4.42] | -0.52 [-1.38; +0.45] |
| direct | +1.75 [-1.96; +5.13] | -0.22 [-1.27; +0.83] |
