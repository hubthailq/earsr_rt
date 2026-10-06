#!/usr/bin/env bash
# Sinh danh sách lệnh của khối n2 chạy lại (06/10/2026), sau khi phát hiện mô hình tinh chỉnh chỉ trên AMI
# hỏng trên ảnh có vùng sáng. Mười lần huấn luyện mỗi fold, đều có tăng cường độ sáng (mặc định của train.py):
#   8 lần: 2 thân × 4 kiểu suy giảm (bic, generic, est, bicjpeg75), thêm ảnh EarVN1.0 nhóm train (30% số mẫu);
#   2 lần: chỉ AMI, suy giảm bic, hai thân (nhãn +pa), để tách tác dụng của ảnh thêm khỏi tác dụng của tăng cường.
# Dùng (từ gốc repo):  bash scripts/make_n2b_jobs.sh [jobs/n2b.txt]                      # fold 2
#                      FOLDS="3 4" bash scripts/make_n2b_jobs.sh jobs/n2b_f34.txt        # hai fold phát triển còn lại
#                      nohup python scripts/run_queue.py jobs/n2b.txt > n2b.log 2>&1 &
# FOLDS chỉ được chứa 2, 3, 4 (fold 1 và 5 giữ kín tới lần chạy cuối). Seed của mỗi lần chạy bằng số fold.
set -euo pipefail
OUT="${1:-jobs/n2b.txt}"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
A=(--ami-raw data/raw/AMI --bench data/bench/ami --degrade-params configs/degrade_estimated.json)
X="--extra-dir data/raw/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn --extra-prob 0.3"
mkdir -p "$(dirname "$OUT")"
FOLDS="${FOLDS:-2}"
TMP="$(mktemp)"
"$PY" scripts/make_jobs.py n2 "${A[@]}" --extra="$X" > "$TMP"
grep -- "--degrade generic" "$TMP" | sed 's/--degrade generic/--degrade bicjpeg75/' >> "$TMP"
"$PY" scripts/make_jobs.py n2 "${A[@]}" --extra="--tag pa" | grep -- "--degrade bic " >> "$TMP"
[ "$(grep -c -- "--fold 2 " "$TMP")" = "10" ] || { echo "LỖI: mong đợi 10 lệnh fold 2 từ make_jobs.py" >&2; exit 1; }
: > "$OUT"
NF=0
for f in $FOLDS; do
  case "$f" in 2|3|4) ;; *) echo "LỖI: fold $f không được phép (chỉ 2, 3, 4)" >&2; exit 1;; esac
  sed "s/--fold 2 /--fold $f /" "$TMP" >> "$OUT"
  NF=$((NF + 1))
done
rm -f "$TMP"
N=$(grep -c . "$OUT")
[ "$N" = "$((10 * NF))" ] || { echo "LỖI: mong đợi $((10 * NF)) lệnh, có $N" >&2; exit 1; }
echo "đã ghi $OUT ($N lệnh, fold: $FOLDS)"
