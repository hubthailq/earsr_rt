#!/usr/bin/env bash
# Xuất bản phát hành công khai của mã nguồn: bản sao sạch của commit hiện tại, không kèm ghi chú làm việc nội bộ và không
# kèm lịch sử git (lịch sử của kho làm việc chứa các ghi chú đó).
#   bash scripts/make_release.sh /đường/dẫn/thư_mục_mới
# Sau khi xuất: script rà toàn bộ bản xuất tìm các từ không được xuất hiện (danh sách ở biến DENY) và dừng nếu còn.
# Kho công khai nên được tạo MỚI từ thư mục này (git init), không đẩy kho làm việc lên.
set -euo pipefail
OUT="${1:?cần đường dẫn thư mục xuất}"
[ -e "$OUT" ] && { echo "LỖI: $OUT đã tồn tại" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "LỖI: còn thay đổi chưa commit; bản xuất lấy từ commit hiện tại" >&2; exit 1; }
# ghi chú nội bộ: danh sách việc, kế hoạch, nhật ký trạng thái, hướng dẫn phiên làm việc, việc trước khi nộp
INTERNAL=(CLAUDE.md TODO.md docs/HANDOFF.md docs/STATUS.md docs/story-imavis.md docs/AMI_PERMISSION.md
          paper/SUBMISSION_CHECKLIST.md scripts/make_release.sh)
DENY='claude|anthropic|chatgpt|openai|copilot|gemini|generative ai|ai assistant|ai-assisted|trợ lý ai|language model|\bllm\b'
mkdir -p "$OUT"
git archive HEAD | tar -x -C "$OUT"
for f in "${INTERNAL[@]}"; do rm -f "$OUT/$f"; done
rm -f "$OUT"/docs/paper-plan-*.md
HITS="$(grep -rIilE "$DENY" "$OUT" || true)"
if [ -n "$HITS" ]; then
  echo "LỖI: bản xuất còn file chứa từ không được xuất hiện:" >&2; echo "$HITS" >&2
  echo "Các dòng:" >&2; grep -rInIE -i "$DENY" "$OUT" | cut -c1-200 | head -20 >&2
  exit 1
fi
echo "đã xuất $(find "$OUT" -type f | wc -l | tr -d ' ') file vào $OUT; không còn từ nào trong danh sách cấm."
echo "Tạo kho công khai: cd \"$OUT\" && git init && git add -A && git commit -m \"Initial release\""
