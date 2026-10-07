#!/usr/bin/env bash
# LẦN CHẠY CUỐI (việc 3 của docs/story-imavis.md): huấn luyện trên hai fold giữ kín, 1 và 5.
# Đây là lần đầu và lần duy nhất dùng hai fold này. Danh sách nhánh được chốt ngày 07/10/2026, trước khi chạy, và giống
# hệt các fold 2, 3, 4 (không thêm, không bớt, không đổi siêu tham số):
#   SPAN và DISP × (bic, generic, bicjpeg75, est), có ảnh EarVN1.0 nhóm train          8 lần mỗi fold
#   SPAN và DISP, bic, chỉ AMI (nhãn +pa)                                                2 lần mỗi fold
#   SPAN × (jpegmix, jpegu)                                                              2 lần mỗi fold
# Tổng 24 lần, khoảng 12 giờ trên một RTX 3080. Sau khi xong: score_n2.sh, run_recog.sh, rồi make_paper.py trên máy Mac;
# mọi bảng của bài tự chuyển sang 5 fold (100 người của AMI).
# KHÔNG được: xem kết quả rồi chạy lại với cấu hình khác, hay bỏ một nhánh khỏi bài vì kết quả của nó.
# Dùng (từ gốc repo):  bash scripts/make_final_jobs.sh [jobs/final.txt]
#                      nohup python scripts/run_queue.py jobs/final.txt > final.log 2>&1 &
set -euo pipefail
OUT="${1:-jobs/final.txt}"
A="$(mktemp)"; B="$(mktemp)"
FINAL_RUN=1 FOLDS="1 5" bash scripts/make_n2b_jobs.sh "$A" > /dev/null
FINAL_RUN=1 FOLDS="1 5" BACKBONES="span" bash scripts/make_n2c_jobs.sh "$B" > /dev/null
mkdir -p "$(dirname "$OUT")"
cat "$A" "$B" > "$OUT"; rm -f "$A" "$B"
N=$(grep -c . "$OUT")
[ "$N" = "24" ] || { echo "LỖI: mong đợi 24 lệnh, có $N" >&2; exit 1; }
[ "$(grep -c -- '--fold 1 ' "$OUT")" = "12" ] && [ "$(grep -c -- '--fold 5 ' "$OUT")" = "12" ] || { echo "LỖI: số lệnh mỗi fold không phải 12" >&2; exit 1; }
if grep -q -- '--fold [234] ' "$OUT"; then echo "LỖI: danh sách chứa fold phát triển" >&2; exit 1; fi
echo "đã ghi $OUT ($N lệnh; fold giữ kín: 1 5; commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Lần chạy cuối: không sửa cấu hình và không chạy lại theo kết quả."
