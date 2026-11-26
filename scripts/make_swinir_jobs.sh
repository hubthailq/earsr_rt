#!/usr/bin/env bash
# Tinh chỉnh một mô hình transformer (SwinIR-light, 0,93 triệu tham số, trọng số công bố ×4) với ĐÚNG công thức của SPAN
# nhánh est: suy giảm đo được, ảnh EarVN1.0 nhóm train 30%, tăng
# cường độ sáng, 20.000 bước, lô 64, tốc độ học 2e-4, seed bằng số fold. Thêm nhánh bicubic cùng fold để so cùng điều kiện.
# Câu hỏi: với một transformer, kiểu suy giảm khi huấn luyện có còn quan trọng hơn kiến trúc như với SPAN và RRDB không.
# Mặc định fold 2 và 3 (fold phát triển; fold 1 và 5 giữ kín, không dùng ở đây).
# Bật --amp như RRDB để vừa bộ nhớ RTX 3080; đặt AMP=0 nếu đủ bộ nhớ. Mã lần chạy mang tiền tố SWIN_.
# Công thức chốt trước khi chạy, không dò; nếu phép thử ảnh sáng báo KHÔNG ỔN ĐỊNH thì ghi lại, không chỉnh để "đẹp".
# Dùng (từ gốc repo):  bash scripts/make_swinir_jobs.sh                       # -> jobs/swinir.txt (4 lệnh)
#                      nohup python scripts/run_queue.py jobs/swinir.txt > swinir.log 2>&1 &
# Biến môi trường: FOLDS (mặc định "2 3"); KINDS (mặc định "est bic"); AMP; PYTHON.
set -euo pipefail
OUT="${1:-jobs/swinir.txt}"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
P="configs/degrade_estimated.json"
[ -f "$P" ] || { echo "LỖI: thiếu $P" >&2; exit 1; }
A=(--ami-raw data/raw/AMI --bench data/bench/ami --degrade-params "$P")
X="--extra-dir data/raw/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn --extra-prob 0.3"
FOLDS="${FOLDS:-2 3}"; KINDS="${KINDS:-est bic}"
FLAG=" --amp"; [ "${AMP:-1}" = "0" ] && FLAG=""
mkdir -p "$(dirname "$OUT")"
ALL="$("$PY" scripts/make_jobs.py n2 "${A[@]}" --baselines swinir --extra="$X")"
: > "$OUT"
NE=0
for k in $KINDS; do
  SEL="$(printf '%s\n' "$ALL" | grep -- "--backbone swinir " | grep -- "--degrade $k " || true)"
  [ "$(printf '%s\n' "$SEL" | grep -c .)" = "1" ] || { echo "LỖI: make_jobs.py không sinh đúng một lệnh $k cho swinir" >&2; exit 1; }
  for f in $FOLDS; do
    case "$f" in 2|3|4) ;; *) echo "LỖI: chỉ dùng fold phát triển 2, 3, 4 (fold $f giữ kín)" >&2; exit 1;; esac
    printf '%s%s\n' "$SEL" "$FLAG" | sed -e "s/--exp N2 /--exp SWIN /" -e "s/--fold 2 /--fold $f /" >> "$OUT"
    NE=$((NE + 1))
  done
done
N=$(grep -c . "$OUT")
[ "$N" = "$NE" ] || { echo "LỖI: mong đợi $NE lệnh, có $N" >&2; exit 1; }
grep -q -- "--exp SWIN --backbone swinir " "$OUT" || { echo "LỖI: lệnh không mang --exp SWIN --backbone swinir_light" >&2; exit 1; }
echo "đã ghi $OUT ($N lệnh; kiểu: $KINDS; fold: $FOLDS; amp:${FLAG:- tắt})"
