# Bản thảo bài báo

Đích: Image and Vision Computing. Câu chuyện và phạm vi: `docs/story-imavis.md`.

## Dựng

```bash
python3 scripts/make_paper.py        # từ gốc repo: sinh paper/generated/*.tex và paper/figures/*.pdf từ results/
cd paper && tectonic main.tex        # tectonic tự tải lớp elsarticle; hoặc: latexmk -pdf main.tex (cần tlmgr install elsarticle)
```

## Quy tắc

- **Không gõ tay con số nào vào các file `sections/*.tex`.** Mọi số là macro trong `generated/numbers.tex`, mọi thân bảng
  nằm trong `generated/tab_*.tex`; cả hai do `scripts/make_paper.py` sinh từ `results/`. Có kết quả mới thì chạy lại script.
- Ô đỏ `[TBD: ...]` là chỗ chưa có số hoặc chưa viết được. Script in danh sách macro còn thiếu ở cuối.
- Chữ xanh "Draft note" là ghi chú cho người đọc bản nháp; xóa trước khi nộp.
- `refs.bib` được ghi **theo trí nhớ, chưa đối chiếu với bản gốc**. Phải kiểm từng mục (tác giả, năm, nơi đăng) trước khi nộp.

## Còn thiếu (07/10/2026)

| Chỗ trong bài | Cần gì | Lệnh |
|---|---|---|
| Bảng `tab:ablation`, mục 6.4 | Khối n2c (6 lần huấn luyện) | `scripts/make_n2c_jobs.sh`, `scripts/score_n2.sh` |
| Bảng `tab:recog`, mục 6.5, câu cuối abstract | Đo nhận dạng trên ảnh nhỏ thật | `scripts/run_recog.sh` |
| Hình định tính | Ảnh do `run_recog.sh` lưu ở `results/recog/sr` | dựng hình sau |
| Mục 6.6 | Khảo sát người xem | `evaluate.py --save-sr`, `viewer_study.py` |
| Mục 6.7, tiêu đề | Độ trễ trên thiết bị | `deploy_jetson/` |
| Mọi bảng của mục 6 | Fold 1 và 5 (chạy cuối) | chưa có script; chỉ chạy sau khi chốt danh sách nhánh |
| Mục 2, `refs.bib` | Rà tài liệu, kiểm trích dẫn | việc của người viết |
| Mục 4.1 | Hình phổ sai số (giải thích vì sao kém bicubic) | chưa có mã |
