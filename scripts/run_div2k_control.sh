#!/usr/bin/env bash
# Phép đối chứng trên ảnh tự nhiên (08/10/2026): ngưỡng "kém bicubic khi ảnh vào bị nén" là của ảnh tai, của cỡ ảnh
# nhỏ, hay của nén nói chung? Chỉ chấm các mô hình có trọng số công bố; không huấn luyện gì, không thêm mô hình nào.
#   Ảnh: DIV2K valid (100 ảnh), thu cả ảnh về cạnh ngắn 96, 144, 192, 384, 768 px, tức ảnh vào 24, 36, 48, 96, 192 px ở ×4.
#   Ảnh vào: bicubic, JPEG 93, 85, 75, 60 (cùng các mức đã dùng trên AMI).
# Kết quả: results/t2_div2k/ và results/t2_div2k_summary/ (cùng định dạng với results/t2_summary/ của AMI).
# Đọc kết quả: nếu chỉ ảnh nhỏ rơi dưới bicubic thì bài có thêm một phát hiện về cỡ ảnh; nếu ảnh lớn cũng rơi thì
# ngưỡng là của nén, và phần riêng của ảnh tai là phép đo nhận dạng. Cả hai trường hợp đều được báo.
# Dùng (từ gốc repo):  bash scripts/run_div2k_control.sh
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); LIMIT; PERCEPTUAL=0; MODELS; TIERS.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
read -r -a T <<< "${TIERS:-96 144 192 384 768}"
# 16 mô hình tối ưu PSNR có trọng số khác nhau (cùng danh sách với bảng của bài) và mốc bicubic
read -r -a M <<< "${MODELS:-bicubic span_ch48 span_ch28 span_ch26 rlfn efdn safmnpp smfan msrresnet edsr_baseline swinir_light rrdb_psnr pds26 pkdsr26 dscf26 disp26 errn26}"
[ -d data/raw/DIV2K_valid_HR ] || { echo "LỖI: thiếu data/raw/DIV2K_valid_HR" >&2; exit 1; }
# biên an toàn 1,05: ảnh DIV2K valid nhỏ nhất có cạnh ngắn 816 px, vừa đủ cho cỡ 768; ảnh gốc là PNG không nén
"$PY" scripts/build_wild.py --root data/raw/DIV2K_valid_HR --name div2k --out "$BENCH/div2k" --tiers "${T[@]}" \
      --safety 1.05 --subject-per-image
"$PY" scripts/evaluate.py --bench "$BENCH/div2k" --folds none --tiers "${T[@]}" \
      --kinds bic bicjpeg93 bicjpeg85 bicjpeg75 bicjpeg60 --models "${M[@]}" --out "$RES/t2_div2k" $PERC "${LIM[@]}"
"$PY" scripts/summarize_t2.py --in "$RES/t2_div2k" --out "$RES/t2_div2k_summary" > /dev/null
echo "$RES/t2_div2k: $(ls "$RES/t2_div2k"/*.csv | wc -l | tr -d ' ') file"
echo "Xong. Đọc: $RES/t2_div2k_summary/T2_summary.md"
