#!/usr/bin/env bash
# Sinh danh sách lệnh của khối n2 chạy lại (06/10/2026), sau khi phát hiện mô hình tinh chỉnh chỉ trên AMI
# hỏng trên ảnh có vùng sáng. Mười lần huấn luyện, fold 2, đều có tăng cường độ sáng (mặc định của train.py):
#   8 lần: 2 thân × 4 kiểu suy giảm (bic, generic, est, bicjpeg75), thêm ảnh EarVN1.0 nhóm train (30% số mẫu);
#   2 lần: chỉ AMI, suy giảm bic, hai thân (nhãn +pa), để tách tác dụng của ảnh thêm khỏi tác dụng của tăng cường.
# Dùng (từ gốc repo):  bash scripts/make_n2b_jobs.sh [jobs/n2b.txt]
#                      nohup python scripts/run_queue.py jobs/n2b.txt > n2b.log 2>&1 &
set -euo pipefail
OUT="${1:-jobs/n2b.txt}"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
A=(--ami-raw data/raw/AMI --bench data/bench/ami --degrade-params configs/degrade_estimated.json)
X="--extra-dir data/raw/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn --extra-prob 0.3"
mkdir -p "$(dirname "$OUT")"
"$PY" scripts/make_jobs.py n2 "${A[@]}" --extra="$X" > "$OUT"
grep -- "--degrade generic" "$OUT" | sed 's/--degrade generic/--degrade bicjpeg75/' >> "$OUT"
"$PY" scripts/make_jobs.py n2 "${A[@]}" --extra="--tag pa" | grep -- "--degrade bic " >> "$OUT"
N=$(grep -c . "$OUT")
[ "$N" = "10" ] || { echo "LỖI: mong đợi 10 lệnh, có $N" >&2; exit 1; }
echo "đã ghi $OUT ($N lệnh)"
