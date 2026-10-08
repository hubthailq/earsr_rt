#!/usr/bin/env bash
# Mốc cho vòng phản biện (08/10/2026): tinh chỉnh một mạng lớn (RRDB, 16,7 triệu tham số, trọng số công bố bản PSNR)
# với ĐÚNG công thức của SPAN nhánh est: suy giảm đo được, ảnh EarVN1.0 nhóm train 30%, tăng cường độ sáng, 20.000 bước,
# lô 64, tốc độ học 2e-4, seed bằng số fold. Câu hỏi: khi mạng lớn cũng học có nén thì nó hơn SPAN bao nhiêu.
# Khác duy nhất so với SPAN: bật --amp (độ chính xác hỗn hợp), vì RRDB ở lô 64 cần khoảng 12 GB ở FP32, quá RTX 3080.
# Đặt AMP=0 nếu máy đủ bộ nhớ. Mã lần chạy mang tiền tố REV_ để không lẫn với khối N2 của bài.
# Thứ tự bắt buộc: fold 2, 3, 4 trước. Công thức chốt trước khi chạy; nếu phép thử ảnh sáng báo KHÔNG ỔN ĐỊNH hoặc
# huấn luyện phân kỳ thì chỉ được chỉnh trên ba fold này. Fold 1 và 5 chạy đúng một lần, sau cùng, với FINAL_RUN=1.
# Dùng (từ gốc repo):  bash scripts/make_rrdb_jobs.sh                                   # fold 2 3 4 -> jobs/rrdb.txt
#                      FINAL_RUN=1 FOLDS="1 5" bash scripts/make_rrdb_jobs.sh jobs/rrdb_final.txt
#                      nohup python scripts/run_queue.py jobs/rrdb.txt > rrdb.log 2>&1 &
set -euo pipefail
OUT="${1:-jobs/rrdb.txt}"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
P="configs/degrade_estimated.json"
[ -f "$P" ] || { echo "LỖI: thiếu $P" >&2; exit 1; }
A=(--ami-raw data/raw/AMI --bench data/bench/ami --degrade-params "$P")
X="--extra-dir data/raw/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn --extra-prob 0.3"
FOLDS="${FOLDS:-2 3 4}"
FLAG=" --amp"; [ "${AMP:-1}" = "0" ] && FLAG=""
mkdir -p "$(dirname "$OUT")"
SEL="$("$PY" scripts/make_jobs.py n2 "${A[@]}" --baselines rrdb --extra="$X" | grep -- "--degrade est ")"
[ "$(printf '%s\n' "$SEL" | grep -c .)" = "1" ] || { echo "LỖI: make_jobs.py n2 không sinh đúng một lệnh est cho thân rrdb" >&2; exit 1; }
: > "$OUT"
NF=0
for f in $FOLDS; do
  case "$f" in
    2|3|4) ;;
    1|5) [ "${FINAL_RUN:-0}" = "1" ] || { echo "LỖI: fold $f chỉ chạy sau cùng, một lần, với FINAL_RUN=1" >&2; exit 1; } ;;
    *) echo "LỖI: fold $f không tồn tại" >&2; exit 1;;
  esac
  printf '%s%s\n' "$SEL" "$FLAG" | sed -e "s/--exp N2 /--exp REV /" -e "s/--fold 2 /--fold $f /" >> "$OUT"
  NF=$((NF + 1))
done
N=$(grep -c . "$OUT")
[ "$N" = "$NF" ] || { echo "LỖI: mong đợi $NF lệnh, có $N" >&2; exit 1; }
grep -q -- "--exp REV --backbone rrdb " "$OUT" || { echo "LỖI: lệnh không mang --exp REV --backbone rrdb" >&2; exit 1; }
echo "đã ghi $OUT ($N lệnh; thân: rrdb; fold: $FOLDS; amp:${FLAG:- tắt})"
