#!/usr/bin/env bash
# Chuẩn nén khác JPEG (kiểm toán 04/11; TODO.md, việc Q1). Chỉ chấm, không huấn luyện.
# Câu hỏi, cố định trước khi chạy: với WebP và AVIF, mô hình công bố có thêm phần dư vào sai số nén (b > 1) và có đổi dấu
# so với bicubic như với JPEG không; nếu có, ở tỉ số E_c/E_i nào (so với JPEG trên cùng ảnh: Bảng 2, AMI 36 px).
# Mức chất lượng chọn trên Mac (100 ảnh AMI) để E_c/E_i của bicubic phủ cùng dải với JPEG 93 đến 60:
#   WebP 95, 90, 85, 75, 60 (khoảng 0,11 / 0,26 / 0,43 / 0,89 / 1,18); AVIF 90, 85, 80, 70, 60 (0,10 / 0,18 / 0,30 / 0,51 / 0,79).
# AMI 36 px (đáp án 144), mọi mô hình có trọng số công bố; ảnh sạch lấy từ results/t2.
# Dùng (từ gốc repo):  bash scripts/run_codec.sh
# Biến môi trường: PYTHON; RES (mặc định results/codec); BENCH (mặc định data/bench); LIMIT (N ảnh đầu, để thử).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/codec}"; BENCH="${BENCH:-data/bench}"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
"$PY" - <<'EOF'
import cv2, numpy as np
x = np.full((24, 36, 3), 128, np.uint8)
for ext, f in ((".webp", cv2.IMWRITE_WEBP_QUALITY), (".avif", cv2.IMWRITE_AVIF_QUALITY)):
    ok, _ = cv2.imencode(ext, x, [f, 50])
    assert ok, f"OpenCV {cv2.__version__} không nén được {ext}"
print("OpenCV", cv2.__version__, "nén được WebP và AVIF")
EOF
KINDS=(bicwebp95 bicwebp90 bicwebp85 bicwebp75 bicwebp60 bicavif90 bicavif85 bicavif80 bicavif70 bicavif60)
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 144 --kinds "${KINDS[@]}" --out "$RES" --no-perceptual \
      ${LIM[@]+"${LIM[@]}"}
echo "$RES: $(ls "$RES"/*.csv | wc -l | tr -d ' ') file"
echo "Xong."
