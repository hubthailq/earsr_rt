#!/usr/bin/env bash
# Chấm các mô hình của khối n2 chạy lại (mọi lần chạy có dấu + trong mã, mọi fold) cùng ba mốc (bicubic, SPAN 48 kênh
# và DISP với trọng số công bố), trên:
#   AMI      : ba cỡ, ảnh vào sạch, JPEG 75, JPEG 93, suy giảm ước lượng, suy giảm tổng quát.
#              Mỗi mô hình tự huấn luyện chỉ được chấm trên người test của fold của nó (evaluate.py tự giới hạn).
#   EarVN1.0 : nhóm test, cỡ 96, ảnh vào sạch, JPEG 75, JPEG 93, suy giảm ước lượng.
#   AWEx     : cỡ 96 và 144, cùng bốn kiểu ảnh vào. AWEx không bao giờ dùng để huấn luyện.
# Từ 07/10/2026 thêm kiểu ảnh vào jpegmix (cùng phân bố mức nén với suy giảm ước lượng, không mờ, không nhiễu) trên cả ba bộ,
# để phép so est với jpegmix được chấm ở cả hai phía chứ không chỉ trên phân bố mà est học.
# File đã có được bỏ qua, nên gọi lại sau khi có thêm lần chạy (fold 3, 4) là an toàn.
# Dùng (từ gốc repo):  bash scripts/score_n2.sh
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); RUNS (mặc định runs);
#                  LIMIT (chỉ chấm N ảnh đầu, để thử); PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"; RUNS="${RUNS:-runs}"
P="configs/degrade_estimated.json"
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
shopt -s nullglob
R=("$RUNS"/N2_*+*)
[ "${#R[@]}" -gt 0 ] || { echo "LỖI: không thấy lần chạy nào khớp $RUNS/N2_*+*" >&2; exit 1; }
echo "chấm ${#R[@]} lần chạy"
M=(--models bicubic span_ch48 disp26)
[ -f "$BENCH/earvn/manifest.csv" ] || "$PY" scripts/build_wild.py --root data/raw/EarVN1.0 --name earvn --out "$BENCH/earvn" \
      --tiers 96 --roles splits/earvn_roles.json --role test
[ -f "$BENCH/awex_s20/manifest.csv" ] || "$PY" scripts/build_wild.py --root data/raw/awex --name awex --out "$BENCH/awex_s20" \
      --tiers 96 144 --safety 2.0
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 96 144 192 --kinds bic bicjpeg75 bicjpeg93 est generic jpegmix \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/n2" $PERC "${LIM[@]}"
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds bic bicjpeg75 bicjpeg93 est jpegmix \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/n2_earvn" $PERC "${LIM[@]}"
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds bic bicjpeg75 bicjpeg93 est jpegmix \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/n2_awex" $PERC "${LIM[@]}"
for d in n2 n2_earvn n2_awex; do echo "$RES/$d: $(ls "$RES/$d"/*.csv | wc -l | tr -d ' ') file"; done
echo "Xong."
