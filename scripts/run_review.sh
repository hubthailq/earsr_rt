#!/usr/bin/env bash
# Các mốc thêm cho vòng phản biện (chốt ngày 08/10/2026, trước khi chạy; xem mục "Mốc cho vòng phản biện" của TODO.md):
#   1.  Khử nén rồi mới phóng: FBCNN (chế độ mù) rồi bicubic, và FBCNN rồi SPAN công bố   fbcnn_bicubic, fbcnn_span_ch48
#   2a. RRDB học bằng suy giảm ngoài thực tế, bản tối ưu độ trung thực                    bsrnet, realesrnet
#   2b. RRDB tinh chỉnh với công thức của SPAN nhánh est (nếu đã có lần chạy runs/REV_*)  scripts/make_rrdb_jobs.sh
# Chỉ chấm điểm, không huấn luyện. Mỗi mốc một cấu hình; ra số nào báo số đó, không chạy lại với cấu hình khác.
# Chấm cùng các mốc đã có trong bài (bicubic, SPAN công bố, RRDB công bố, SPAN nhánh est) để so ghép cặp trên cùng ảnh:
#   AMI      : ba cỡ; ảnh vào sạch, JPEG 93, 85, 75, suy giảm ước lượng                 -> results/rev
#   EarVN1.0 : nhóm test, cỡ 96; ảnh vào sạch, JPEG 93, 75, suy giảm ước lượng          -> results/rev_earvn
#   AWEx     : cỡ 96 và 144, cùng bốn kiểu ảnh vào                                       -> results/rev_awex
#   Nhận dạng trên ảnh nhỏ thật của EarVN1.0 và AWEx, ba mạng nhận dạng                  -> results/recog/rev_*
#   Độ trễ trên máy này (GPU), ảnh vào 48×68                                             -> results/latency_review.csv
# File đã có được bỏ qua, nên gọi lại sau khi có thêm lần chạy REV_* (fold 1, 5) là an toàn.
# Dùng (từ gốc repo):  bash scripts/get_weights.sh          # tải thêm BSRNet, Real-ESRNet, FBCNN (lần đầu)
#                      nohup bash scripts/run_review.sh > review.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); RUNS (mặc định runs);
#                  LIMIT (chỉ chấm N ảnh đầu, để thử); PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS;
#                  RECOG=0 bỏ phần nhận dạng; LATENCY=0 bỏ phần độ trễ.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"; RUNS="${RUNS:-runs}"
P="configs/degrade_estimated.json"
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
for w in BSRNet.pth RealESRNet_x4plus.pth fbcnn_color.pth; do
  [ -f "${EARSR_WEIGHTS:-weights}/$w" ] || { echo "LỖI: thiếu trọng số $w; chạy: bash scripts/get_weights.sh" >&2; exit 1; }
done
shopt -s nullglob
OURS=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f*)
[ "${#OURS[@]}" -gt 0 ] || { echo "LỖI: không thấy lần chạy SPAN nhánh est ($RUNS/N2_span-zero+xearvn_*_est_f*)" >&2; exit 1; }
REV=("$RUNS"/REV_*+*)
R=("${OURS[@]}" "${REV[@]}")
echo "mốc đã có: ${#OURS[@]} lần chạy SPAN nhánh est; lần chạy mới REV_*: ${#REV[@]}"
M=(--models bicubic span_ch48 rrdb_psnr bsrnet realesrnet fbcnn_bicubic fbcnn_span_ch48)

"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 96 144 192 --kinds bic bicjpeg93 bicjpeg85 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/rev" $PERC "${LIM[@]}"
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/rev_earvn" $PERC "${LIM[@]}"
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" "${M[@]}" --runs "${R[@]}" --out "$RES/rev_awex" $PERC "${LIM[@]}"
for d in rev rev_earvn rev_awex; do echo "$RES/$d: $(ls "$RES/$d"/*.csv | wc -l | tr -d ' ') file"; done

if [ "${RECOG:-1}" != "0" ]; then
  for tag in resnet18 resnet50 imagenet50; do
    case "$tag" in imagenet50) REC="imagenet-resnet50";; *) REC="$RUNS/RECOG_$tag/ckpt.pt";; esac
    "$PY" scripts/recog_eval.py --root data/raw/EarVN1.0 --roles splits/earvn_roles.json --recognizer "$REC" "${M[@]}" \
          --runs "${R[@]}" --out "$RES/recog/rev_$tag"
    "$PY" scripts/recog_eval.py --root data/raw/awex --recognizer "$REC" "${M[@]}" --runs "${R[@]}" --out "$RES/recog/rev_awex_$tag"
    for d in "rev_$tag" "rev_awex_$tag"; do
      "$PY" scripts/summarize_recog.py --in "$RES/recog/$d" --out "$RES/recog/${d}_summary" > /dev/null
    done
  done
fi

if [ "${LATENCY:-1}" != "0" ]; then
  "$PY" scripts/bench_local.py --models span_ch48 rrdb_psnr bsrnet fbcnn_bicubic fbcnn_span_ch48 --out "$RES/latency_review.csv"
fi

"$PY" scripts/summarize_review.py --results "$RES"
echo "Xong. Đọc: $RES/rev_summary/summary.md"
