#!/usr/bin/env bash
# Khối n2c (07/10/2026): tách phần "có nén" khỏi phần "đo từ dữ liệu" của suy giảm ước lượng (việc 1 của
# docs/story-imavis.md). Hai nhánh huấn luyện mới, cùng công thức với khối n2 chạy lại (ảnh EarVN1.0 nhóm train 30%,
# tăng cường độ sáng), chỉ khác kiểu suy giảm:
#   jpegmix : bicubic rồi JPEG với đúng phân bố mức nén đã đo (configs/degrade_estimated.json); không mờ, không nhiễu;
#   jpegu   : bicubic rồi JPEG, mức nén rút đều 60 đến 95; không dùng gì đo từ dữ liệu.
# Đọc kết quả: est hơn jpegmix -> mờ và nhiễu có đóng góp; est ngang jpegmix -> phần hơn chỉ là phân bố mức nén;
#              jpegu ngang jpegmix -> không cần đo phân bố.
# Dùng (từ gốc repo):  bash scripts/make_n2c_jobs.sh [jobs/n2c.txt]             # SPAN, fold 2 3 4: 6 lệnh
#                      BACKBONES="span disp26" bash scripts/make_n2c_jobs.sh    # thêm thân thứ hai: 12 lệnh
#                      nohup python scripts/run_queue.py jobs/n2c.txt > n2c.log 2>&1 &
# FOLDS chỉ được chứa 2, 3, 4. Fold 1 và 5 giữ kín tới lần chạy cuối và chỉ mở qua scripts/make_final_jobs.sh. Seed của mỗi lần chạy bằng số fold.
set -euo pipefail
OUT="${1:-jobs/n2c.txt}"
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
P="configs/degrade_estimated.json"
[ -f "$P" ] || { echo "LỖI: thiếu $P" >&2; exit 1; }
A=(--ami-raw data/raw/AMI --bench data/bench/ami --degrade-params "$P")
X="--extra-dir data/raw/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn --extra-prob 0.3"
FOLDS="${FOLDS:-2 3 4}"; BACKBONES="${BACKBONES:-span}"
mkdir -p "$(dirname "$OUT")"
TMP="$(mktemp)"; SEL="$(mktemp)"
"$PY" scripts/make_jobs.py n2 "${A[@]}" --extra="$X" | grep -- "--degrade generic " > "$TMP"
NB=0
for b in $BACKBONES; do
  grep -- "--backbone $b " "$TMP" >> "$SEL" || { echo "LỖI: make_jobs.py n2 không sinh lệnh cho thân $b" >&2; exit 1; }
  NB=$((NB + 1))
done
: > "$OUT"
NF=0
for f in $FOLDS; do
  case "$f" in
    2|3|4) ;;
    1|5) [ "${FINAL_RUN:-0}" = "1" ] || { echo "LỖI: fold $f đang giữ kín; chỉ scripts/make_final_jobs.sh được mở" >&2; exit 1; } ;;
    *) echo "LỖI: fold $f không tồn tại" >&2; exit 1;;
  esac
  sed -e "s/--fold 2 /--fold $f /" -e "s#--degrade generic #--degrade jpegmix --degrade-params $P #" "$SEL" >> "$OUT"
  sed -e "s/--fold 2 /--fold $f /" -e "s/--degrade generic /--degrade jpegu /" "$SEL" >> "$OUT"
  NF=$((NF + 1))
done
rm -f "$TMP" "$SEL"
N=$(grep -c . "$OUT")
[ "$N" = "$((2 * NB * NF))" ] || { echo "LỖI: mong đợi $((2 * NB * NF)) lệnh, có $N" >&2; exit 1; }
echo "đã ghi $OUT ($N lệnh; thân: $BACKBONES; fold: $FOLDS)"
