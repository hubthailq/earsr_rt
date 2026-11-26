#!/usr/bin/env bash
# Chấm các lần chạy SwinIR-light tinh chỉnh cùng bicubic và
# SwinIR-light trọng số công bố, trên cùng ảnh vào như SPAN và RRDB:
#   AMI      : ba cỡ; sạch, JPEG 93, 85, 75, suy giảm đo được (mỗi lần chạy chỉ trên người test của fold nó) -> results/swin
#   EarVN1.0 : nhóm test, cỡ 96; sạch, JPEG 93, 75, suy giảm đo được                                     -> results/swin_earvn
#   AWEx     : cỡ 96 và 144; cùng bốn kiểu ảnh vào                                                       -> results/swin_awex
# Chạy sau khi jobs/swinir.txt xong. File đã có được bỏ qua.
# Dùng (từ gốc repo):  nohup bash scripts/score_swinir.sh > swinir_score.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); RUNS (mặc định runs);
#                  LIMIT (chỉ chấm N ảnh đầu, để thử); PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"; RUNS="${RUNS:-runs}"
P="configs/degrade_estimated.json"
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
shopt -s nullglob
R=("$RUNS"/SWIN_swinir-*)
[ "${#R[@]}" -gt 0 ] || { echo "LỖI: không thấy lần chạy nào khớp $RUNS/SWIN_swinir-*" >&2; exit 1; }
echo "chấm ${#R[@]} lần chạy: $(for r in "${R[@]}"; do basename "$r"; done | tr '\n' ' ')"
M=(--models bicubic swinir_light)
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 96 144 192 --kinds bic bicjpeg93 bicjpeg85 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/swin" $PERC ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/swin_earvn" $PERC ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/swin_awex" $PERC ${LIM[@]+"${LIM[@]}"}
for d in swin swin_earvn swin_awex; do echo "$RES/$d: $(ls "$RES/$d"/*.csv | wc -l | tr -d ' ') file"; done
echo "Xong."
