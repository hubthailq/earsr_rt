#!/usr/bin/env bash
# Dao động do seed huấn luyện (kiểm toán 04/11; TODO.md, việc Q3).
# Câu hỏi, cố định trước khi chạy: trên fold 1, ba nhánh SPAN (bic, bicjpeg75, est) huấn luyện lại với seed 101 và 102,
# cùng mọi siêu tham số của lần chạy cuối (seed mặc định = số fold = 1). PSNR của một nhánh dao động bao nhiêu giữa ba seed,
# và các chênh lệch giữa nhánh (est trừ bicjpeg75, est trừ bic) có đổi dấu không. Không chọn gì theo kết quả này.
# 6 lần huấn luyện (khoảng 3 giờ trên một RTX 3080), rồi chấm cả 9 lần chạy của fold 1 trên AMI, EarVN1.0, AWEx.
# Dùng (từ gốc repo):  nohup bash scripts/run_seeds.sh > seeds.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results/seeds); BENCH (mặc định data/bench); RUNS (mặc định runs);
#                  ITERS (chỉ để thử, mặc định như lần chạy cuối); LIMIT (N ảnh đầu khi chấm, để thử).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/seeds}"; BENCH="${BENCH:-data/bench}"; RUNS="${RUNS:-runs}"
P="configs/degrade_estimated.json"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
JOBS=jobs/seeds.txt
mkdir -p jobs "$RES"
T="$(mktemp)"
FINAL_RUN=1 FOLDS="1" bash scripts/make_n2b_jobs.sh "$T" > /dev/null
: > "$JOBS"
for k in bic bicjpeg75 est; do
  line=$(grep -- "--backbone span " "$T" | grep -- "--degrade $k " | grep -v -- "--tag pa" || true)
  [ "$(printf '%s\n' "$line" | grep -c .)" = "1" ] || { echo "LỖI: không tìm được đúng một lệnh SPAN $k fold 1" >&2; exit 1; }
  for s in 101 102; do
    cmd="$line --seed $s"
    [ -n "${ITERS:-}" ] && cmd="$(printf '%s' "$cmd" | sed "s/--iters [0-9]*/--iters $ITERS/")"
    [ "$PY" != "python" ] && cmd="$(printf '%s' "$cmd" | sed "s/^python /$PY /")"
    echo "$cmd" >> "$JOBS"
  done
done
rm -f "$T"
[ "$(grep -c . "$JOBS")" = "6" ] || { echo "LỖI: mong đợi 6 lệnh" >&2; exit 1; }
echo "đã ghi $JOBS (6 lệnh)"
"$PY" scripts/run_queue.py "$JOBS"

shopt -s nullglob
R=()
for k in bic bicjpeg75 est; do
  R+=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_${k}_f1 "$RUNS"/N2_span-zero+xearvn+s10[12]_pub_rand_x4_hrall_${k}_f1)
done
[ "${#R[@]}" = "9" ] || { echo "LỖI: cần 9 lần chạy fold 1, thấy ${#R[@]}" >&2; exit 1; }
for r in "${R[@]}"; do [ -f "$r/ckpt/best.pt" ] || { echo "LỖI: $r chưa xong" >&2; exit 1; }; done
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 144 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" --models bicubic --runs "${R[@]}" --out "$RES/ami" --no-perceptual ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" --models bicubic --runs "${R[@]}" --out "$RES/earvn" --no-perceptual ${LIM[@]+"${LIM[@]}"}
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds bic bicjpeg93 bicjpeg75 est \
      --degrade-params "$P" --models bicubic --runs "${R[@]}" --out "$RES/awex" --no-perceptual ${LIM[@]+"${LIM[@]}"}
for d in ami earvn awex; do echo "$RES/$d: $(ls "$RES/$d"/*.csv | wc -l | tr -d ' ') file"; done
echo "Xong."
