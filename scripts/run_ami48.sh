#!/usr/bin/env bash
# Lấp hai ô trống của Bảng A.1 trên AMI (kiểm toán bảng ngày 04/11; TODO.md, việc P6). Chỉ chấm, không huấn luyện.
#   1. Ảnh vào 48 px (đáp án 192): JPEG 93, 85, 60 (mức 75 và ảnh sạch đã có), để tính được điểm đổi dấu ở Bảng 2.
#   2. Suy giảm đo được (est) ở cả ba cỡ 24, 36, 48 px, cho cột Measured.
# Mọi mô hình có trọng số công bố, như scripts/run_wild_t2.sh; ghi thêm vào results/t2 (file đã có thì bỏ qua).
# Dùng (từ gốc repo):  bash scripts/run_ami48.sh
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); LIMIT (N ảnh đầu, để thử);
#                  PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"
P="configs/degrade_estimated.json"
[ -f "$P" ] || { echo "LỖI: thiếu $P" >&2; exit 1; }
[ -d "$BENCH/ami/hr192" ] || { echo "LỖI: thiếu $BENCH/ami/hr192" >&2; exit 1; }
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")

"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 192 --kinds bicjpeg93 bicjpeg85 bicjpeg60 \
      --out "$RES/t2" $PERC ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 96 144 192 --kinds est --degrade-params "$P" \
      --out "$RES/t2" $PERC ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/summarize_t2.py --in "$RES/t2" --out "$RES/t2_summary"
echo "$RES/t2: $(ls "$RES"/t2/*__hr192_x4_bicjpeg{93,85,60}.csv "$RES"/t2/*_x4_est.csv 2>/dev/null | wc -l | tr -d ' ') file mới"
echo "Xong."
