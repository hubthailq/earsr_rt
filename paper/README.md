# Bản thảo bài báo

Đích: Image and Vision Computing. Câu chuyện và phạm vi: `docs/story-imavis.md`.

## Dựng

```bash
python3 scripts/make_paper.py        # từ gốc repo: sinh paper/generated/*.tex và paper/figures/*.png từ results/
cd paper && tectonic main.tex        # tectonic tự tải lớp elsarticle; hoặc: latexmk -pdf main.tex (cần tlmgr install elsarticle)
```

## Quy tắc

- **Không gõ tay con số nào vào các file `sections/*.tex`.** Mọi số là macro trong `generated/numbers.tex`, mọi thân bảng
  nằm trong `generated/tab_*.tex`; cả hai do `scripts/make_paper.py` sinh từ `results/`. Có kết quả mới thì chạy lại script.
- Bản thảo hiện không còn ô `[TBD]` nào. Nếu thiếu kết quả, script sẽ sinh lại ô đỏ và in danh sách macro còn thiếu.
- `refs.bib` (44 mục) đã đối chiếu với Crossref hoặc trang của nhà xuất bản ngày 08/10/2026; ghi chú ở đầu file.
- Văn phong: câu trần thuật, không dùng gạch dài, không dùng câu khẩu hiệu hay chữ nghiêng để nhấn; kết luận viết kèm phạm vi
  ("in our experiments", "for the models we tested").

## Việc còn lại trước khi nộp

Xem `SUBMISSION_CHECKLIST.md`: các tuyên bố Elsevier yêu cầu, đơn vị công tác, quyền dùng ảnh, rà tài liệu có hệ thống.
`highlights.txt` chứa 5 dòng highlights.
