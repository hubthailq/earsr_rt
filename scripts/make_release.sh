#!/usr/bin/env bash
# Xuất bản phát hành công khai của mã nguồn: bản sao sạch của commit hiện tại, không kèm ghi chú làm việc nội bộ, không
# kèm bản thảo và không kèm lịch sử git (lịch sử của kho làm việc chứa các ghi chú đó).
#   bash scripts/make_release.sh /đường/dẫn/thư_mục_mã [/đường/dẫn/results.tar.gz]
# - Thư mục mã: để tạo kho GitHub MỚI (git init), không đẩy kho làm việc lên. README.md là bản tiếng Anh docs/README_release.md.
# - results.tar.gz (tùy chọn): kết quả theo ảnh (results/) để đưa lên Zenodo; không nằm trong kho mã cho nhẹ.
# Sau khi xuất, script rà toàn bộ bản xuất tìm từ không được xuất hiện (DENY: tên công cụ AI; INTERNAL_WORDS: dấu vết
# quy trình nội bộ), kiểm cú pháp mọi file Python và shell, và dừng nếu có lỗi.
set -euo pipefail
OUT="${1:?cần đường dẫn thư mục xuất}"
RES_TAR="${2:-}"
[ -e "$OUT" ] && { echo "LỖI: $OUT đã tồn tại" >&2; exit 1; }
[ -n "$RES_TAR" ] && [ -e "$RES_TAR" ] && { echo "LỖI: $RES_TAR đã tồn tại" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "LỖI: còn thay đổi chưa commit; bản xuất lấy từ commit hiện tại" >&2; exit 1; }
[ -f docs/README_release.md ] || { echo "LỖI: thiếu docs/README_release.md" >&2; exit 1; }
# ghi chú nội bộ: danh sách việc, kế hoạch, nhật ký trạng thái, hướng dẫn phiên làm việc, việc trước khi nộp
INTERNAL=(CLAUDE.md TODO.md docs/HANDOFF.md docs/STATUS.md docs/story-imavis.md docs/AMI_PERMISSION.md
          scripts/make_release.sh)
DENY='claude|anthropic|chatgpt|openai|copilot|gemini|generative ai|ai assistant|ai-assisted|trợ lý ai|language model|\bllm\b'
INTERNAL_WORDS='thầy|TODO\.md|phản biện|kiểm toán|người dùng|story-imavis|STATUS\.md|HANDOFF|paper-plan|reviewer'
mkdir -p "$OUT"
git archive HEAD | tar -x -C "$OUT"
for f in "${INTERNAL[@]}"; do rm -f "$OUT/$f"; done
rm -f "$OUT"/docs/paper-plan-*.md "$OUT"/*.log
rm -rf "$OUT/paper" "$OUT/docs/refs"           # bản thảo chưa công bố; PDF tài liệu tham khảo
if [ -n "$RES_TAR" ]; then
  tar -czf "$RES_TAR" -C "$OUT" results
  echo "đã đóng gói kết quả: $RES_TAR ($(du -h "$RES_TAR" | cut -f1))"
fi
rm -rf "$OUT/results"
mkdir -p "$OUT/results" && cp results/runs.csv "$OUT/results/runs.csv"   # sổ các lần chạy: nhỏ, giữ trong kho mã
mv "$OUT/docs/README_release.md" "$OUT/README.md"
HITS="$(grep -rIilE "$DENY" "$OUT" || true)"
IHITS="$(grep -rIilE "$INTERNAL_WORDS" "$OUT" || true)"
if [ -n "$HITS$IHITS" ]; then
  echo "LỖI: bản xuất còn file chứa từ không được xuất hiện:" >&2; echo "$HITS $IHITS" >&2
  grep -rInIE -i "$DENY|$INTERNAL_WORDS" "$OUT" | cut -c1-200 | head -20 >&2
  exit 1
fi
python3 -m compileall -q "$OUT/earsr" "$OUT/scripts" "$OUT/tests" >/dev/null || { echo "LỖI: Python không biên dịch được" >&2; exit 1; }
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
for f in "$OUT"/scripts/*.sh; do bash -n "$f" || { echo "LỖI: cú pháp shell $f" >&2; exit 1; }; done
echo "đã xuất $(find "$OUT" -type f | wc -l | tr -d ' ') file ($(du -sh "$OUT" | cut -f1)) vào $OUT; không còn từ nào trong danh sách cấm."
echo "Tạo kho công khai: cd \"$OUT\" && git init && git add -A && git commit -m \"Initial release\""
