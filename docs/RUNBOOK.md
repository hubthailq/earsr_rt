# Thứ tự chạy trên máy GPU

Mọi lệnh chạy từ thư mục gốc của project. Thay `/data/...` bằng đường dẫn thật (bố cục mặc định là
`data/raw/...`; xem `docs/DATA_AND_OUTPUTS.md`, file đó cũng ghi kết quả của từng việc nằm ở file nào).

## 0. Cài đặt và kiểm tra nhanh (bắt buộc, 5 đến 10 phút)

```bash
pip install -r requirements.txt
bash scripts/get_weights.sh
python scripts/check_data.py --root /data    # dữ liệu đúng chỗ, đúng cấu trúc chưa
pytest -q                                   # khoảng 3 phút; cần đạt hết (vài kiểm thử tự bỏ qua nếu thiếu Set5)
python scripts/smoke_test.py --ami-raw /data/AMI
```

`smoke_test.py` chạy những đường mã chưa từng chạy ở máy dựng project: GPU, AMP, LPIPS, DISTS, VGG19. Nó
cũng in tốc độ huấn luyện và ước lượng số giờ cho 20.000 bước. Có bước LỖI thì dừng ở đây và gửi lại
`results/smoke_test.json`.

## 1. Giai đoạn 1, không huấn luyện

```bash
# chỉ ở lần chạy đầu trên máy GPU: dời kết quả sơ bộ của CPU đi (xem giải thích bên dưới)
mkdir -p results_prelim_cpu
mv results/t2 results/t2_summary results/t6_context results/t6_summary results_prelim_cpu/

bash scripts/run_stage1.sh /data/AMI /data/DIV2K_valid_HR
python scripts/make_tables.py
```

Ra: `results/t2_summary/`, `results/t6_summary/`, `results/latency_local.csv`, `deploy_jetson/onnx/`.
Thư mục `results/` đi kèm project chứa kết quả sơ bộ chạy trên CPU (chưa có LPIPS, DISTS; `span_ch48` trong đó là
trọng số đội 44). `run_stage1.sh` **không ghi đè** chúng: `evaluate.py` và `run_t6_context.py` bỏ qua file kết quả
đã có (nhờ vậy chạy tiếp được sau khi bị ngắt). Vì thế phải dời bốn thư mục trên đi trước lần chạy đầu. Nếu quên,
`run_stage1.sh` tự dừng ở dòng đầu và in đúng lệnh cần chạy (nó nhận ra thư mục của chính nó qua file dấu
`.stage1_run`). Trong lúc bốn thư mục đó vắng mặt, hai kiểm thử đọc `results/` (`tests/test_stage2_tools.py`) sẽ
lỗi cho tới khi giai đoạn 1 xong.

`run_stage1.sh` cũng **dừng ngay nếu LPIPS hoặc DISTS không nạp được** (thiếu gói, hoặc máy không tải được trọng
số), để không mất vài giờ cho một bảng thiếu cột. Muốn chạy thiếu hai số đo đó: `PERCEPTUAL=0 bash scripts/run_stage1.sh ...`.
Script dùng lệnh `python`, không có thì `python3`; chỉ định lệnh khác bằng `PYTHON=/đường/dẫn/python`.
Sau bước này là **điểm kiểm tra 1** (mục 1.10 của kế hoạch). Chỉnh `configs/criteria.yaml` một lần nếu
`results/t2_summary/mde.csv` cho thấy ngưỡng nào nhỏ hơn mức chênh phát hiện được, rồi commit.

**Mô hình hỏng trên ảnh sáng.** Tinh chỉnh chỉ trên AMI cho mô hình ra nhiễu trên ảnh có vùng sáng (AMI không có ảnh
nào như vậy), trong khi PSNR validation vẫn đẹp. Vì thế `train.py` mặc định đổi độ sáng, tông và màu của ảnh huấn luyện
(`--photo-aug 0.75`; `0` để tắt), và sau mỗi lần huấn luyện in một dòng `phép thử ảnh sáng: ỔN ĐỊNH ...` hoặc
`CẢNH BÁO: phép thử ảnh sáng: KHÔNG ỔN ĐỊNH ...` (chi tiết ở `runs/<mã>/stress.json`, cột `stress_*` trong
`results/runs.csv`). Lần chạy nào báo KHÔNG ỔN ĐỊNH thì không dùng kết quả của nó.

**Độ chính xác khi chấm.** Mọi phép chấm (suy luận, validation, LPIPS, DISTS) chạy ở FP32 đầy đủ với TF32 tắt, để
số trên GPU khớp số trên CPU (lệch cỡ 1e-6 thay vì 3e-4). Các bước huấn luyện dùng mặc định của PyTorch. Độ trễ
trong `bench_local.py` cũng đo theo mặc định; cột `host` ghi trạng thái TF32 lúc đo.

**Khi GPU dùng chung bị đầy.** `evaluate.py` và `run_t6_context.py` in thiết bị ở dòng đầu. Chúng ưu tiên GPU;
ảnh nào gặp lỗi hết bộ nhớ GPU thì chạy trên CPU, và GPU được thử lại sau 15 giây (quãng chờ nhân đôi tới 240 giây
nếu vẫn đầy). Mỗi lần chuyển in một dòng `[thiết bị] ...`; cột `device` trong file theo ảnh ghi thiết bị của từng
ảnh. LPIPS và DISTS cũng vậy. `bench_local.py` thì khác: độ trễ không được đo thay trên CPU, nên nó chờ 30 giây rồi
đo lại (3 lần), vẫn không được thì ghi dòng lỗi cho mô hình đó và đi tiếp. Huấn luyện (`train.py`, `pretrain.py`)
**chưa** có cơ chế này: hết bộ nhớ thì lần chạy dừng, chạy lại cùng lệnh sẽ tiếp tục từ checkpoint.

Khi có Jetson: chép `deploy_jetson/` sang máy đó, chạy `python3 bench_trtexec.py` (xem `deploy_jetson/README.txt`).

## 2. Giai đoạn 2

```bash
mkdir -p jobs
A="--ami-raw /data/AMI --bench data/bench/ami"

# 2.0 (bản 22) Hai phép thử rẻ, chạy TRƯỚC mọi thứ khác của giai đoạn 2.
#   N2 sớm (cần EarVN1.0; xem mục 3 để tạo configs/degrade_estimated.json và data/bench/earvn).
#   Mười lần huấn luyện: 2 thân × 4 kiểu suy giảm, có ảnh EarVN nhóm train; cộng 2 lần chỉ AMI (nhãn +pa).
bash scripts/make_n2b_jobs.sh jobs/n2b.txt && python scripts/run_queue.py jobs/n2b.txt
#   chỉ chấm các lần chạy mới (có dấu + trong mã); 8 lần chạy đầu, không có dấu +, đã bị loại
python scripts/evaluate.py --bench data/bench/earvn --folds none --tiers 96 --kinds bic \
       --models bicubic span_ch48 disp26 --runs runs/N2_*+* --out results/n2
python scripts/evaluate.py --bench data/bench/ami --tiers 144 --kinds bic bicjpeg75 generic est \
       --degrade-params configs/degrade_estimated.json --models bicubic span_ch48 disp26 --runs runs/N2_*+* --out results/n2
#   Khối n2c (07/10): hai nhánh chỉ nén, để tách phần "có nén" khỏi phần "đo từ dữ liệu". jpegmix = JPEG với đúng phân bố
#   mức nén đã đo, không mờ, không nhiễu; jpegu = mức nén rút đều 60 đến 95. SPAN × fold 2, 3, 4 = 6 lần.
bash scripts/make_n2c_jobs.sh jobs/n2c.txt && python scripts/run_queue.py jobs/n2c.txt
#   chấm mọi lần chạy N2 có dấu + trên AMI, EarVN1.0, AWEx (bỏ qua file đã có); ảnh vào gồm cả kiểu jpegmix
bash scripts/score_n2.sh
#   LẦN CHẠY CUỐI trên hai fold giữ kín 1 và 5 (24 lần; chỉ chạy một lần, không sửa cấu hình theo kết quả):
bash scripts/make_final_jobs.sh jobs/final.txt && python scripts/run_queue.py jobs/final.txt
#   Đối chứng trên ảnh tự nhiên (08/10): 16 mô hình có sẵn trên DIV2K valid, ảnh vào 24 đến 192 px, năm kiểu ảnh vào
bash scripts/run_div2k_control.sh
#   Đo nhận dạng tai trên ảnh nhỏ thật (07/10): huấn luyện hai mạng nhận dạng (ResNet-18, ResNet-50; chỉ ảnh lớn của người
#   có vai train, fit, clf), rồi chấm mọi phương pháp phóng ảnh trên ảnh nhỏ thật của EarVN1.0 (nhóm test, viewer) và AWEx,
#   mỗi bộ với ba mạng nhận dạng; kết quả EarVN1.0 được tách thêm theo mức nén JPEG của file ảnh dò.
#   Cần torchvision và tải được trọng số ImageNet ở lần đầu. Kết quả: results/recog/*_summary/summary.md
bash scripts/run_recog.sh
#   Bản thảo: sinh số, bảng, hình từ results/ rồi biên dịch (xem paper/README.md)
python scripts/make_paper.py && (cd paper && tectonic main.tex)
#   Kiểu đệm trên trọng số công bố (không tiền huấn luyện), ba mốc mặc định: span, disp26, errn26
python scripts/make_jobs.py pad $A > jobs/pad.txt && python scripts/run_queue.py jobs/pad.txt

# 2a. dò tốc độ học (fold 2, không chấm test)
python scripts/make_jobs.py lr $A > jobs/lr.txt && python scripts/run_queue.py jobs/lr.txt
#     xem cột best_val_psnr_y trong results/runs.csv, chọn một mức cho mỗi mốc, ví dụ:
LR="--lr-of span=2e-4 disp26=1e-4 errn26=2e-4"

# 2b. T6 (ii): ba giao thức
python scripts/make_jobs.py t6ii $A $LR > jobs/t6ii.txt && python scripts/run_queue.py jobs/t6ii.txt
python scripts/check_criteria.py n5a --tier 96 --a <rand>/test_hr96_x4_bic.csv ... --b <fixed 96>/... \
       --tier 144 --a ... --b ... --tier 192 --a ... --b ...

# 2c. T6 (iii): có cần tiền huấn luyện không
python scripts/make_jobs.py t6iii $A $LR --protocol <giao thức đã chọn> > jobs/t6iii.txt

# 2d. T6 (iv): tiền huấn luyện rút gọn năm biến thể, rồi tinh chỉnh
#     trước đó, nếu T3 cho thấy hai hình dạng wide, deep lệch độ trễ quá 10%:
python scripts/match_latency.py --target-variant zero --base zero-b3      # rồi dùng --variants ở hai lệnh dưới
python scripts/make_jobs.py pre --div2k-train /data/DIV2K_train_HR --div2k-val /data/DIV2K_valid_HR > jobs/pre.txt
python scripts/make_jobs.py t6iv $A $LR --protocol <...> > jobs/t6iv.txt
python scripts/check_criteria.py n5b-gain --a runs/T6iv_span-replicate_*/test_hr144_x4_bic.csv \
                                         --b runs/T6iv_span-zero_ps20_*/test_hr144_x4_bic.csv

# 2d'. Nhánh so sánh có kiểm soát (mục 1.8b của kế hoạch): các lệnh 'pre' và 't6iv' ở trên đã gồm RLFN học lại
#      từ đầu. Thêm mốc SPAN nới rộng cho bằng độ trễ của thân thắng, rồi chạy lại ở ngân sách đủ:
python scripts/match_latency.py --target-variant <thân thắng> --base zero          # -> ví dụ zero-c56
python scripts/make_jobs.py pre  --budget 100 --variants zero <thân thắng> --ctl-variants zero-c56 ... > jobs/pre_full.txt
python scripts/make_jobs.py t6iv --budget 100 --variants zero <thân thắng> --ctl-variants zero-c56 $A $LR > jobs/ctl_dev.txt
#      (fold 1 và 5 của nhánh này chạy sau điểm kiểm tra 2, bằng train.py --exp S2 --pretrain pf --init-ckpt ...)

# 2e. T6 (v): phép thử tương tác trên ảnh tự nhiên, dùng checkpoint tiền huấn luyện (chưa tinh chỉnh trên tai)
python scripts/build_wild.py --root /data/DIV2K_valid_HR --name div2k --out data/bench/div2k \
       --tiers 144 576 --safety 2.0 --subject-per-image
python scripts/evaluate.py --bench data/bench/div2k --folds none --tiers 144 576 --kinds bic \
       --runs runs/PRE_span-zero_ps20_* runs/PRE_span-replicate_ps20_* --out results/t6v
python scripts/check_criteria.py n5b-interaction --a <biến thể, 144> --b <tham chiếu, 144> --a2 <biến thể, 576> --b2 <tham chiếu, 576>
```

Sau đó là **điểm kiểm tra 2**. Chỉ khi đó mới chạy khối `s2` (có fold 1 và 5).

`run_queue.py` bỏ qua lệnh đã xong và `train.py` tự tiếp tục từ checkpoint gần nhất, nên tắt máy giữa chừng
rồi chạy lại cùng lệnh là được.

## 3. N2 (cần EarVN1.0, cấu trúc `root/<người>/<ảnh>`)

Thứ tự nên theo: hai lệnh đầu (ước lượng, bộ phân loại) chỉ mất vài phút và là thứ quyết định N2 theo tiêu chí ghi
trước; xem `results/n2_realism.json` rồi mới chạy khối huấn luyện `n2` (6 lần, khoảng 6 giờ). Sáu lần huấn luyện đó
không tự quyết định N2: mô hình nào cũng thắng ở đúng kiểu suy giảm nó được huấn luyện. Hai file do lệnh đầu tạo
(`splits/earvn_roles.json`, `configs/degrade_estimated.json`) chỉ được tạo một lần, trước mọi lần huấn luyện; commit ngay.

```bash
python scripts/fit_degradation.py --root /data/EarVN1.0 --roles splits/earvn_roles.json   # tạo file vai; commit
python scripts/realism_classifier.py --root /data/EarVN1.0 --degrade-params configs/degrade_estimated.json \
       --kinds bic bicjpeg75 generic est      # bicjpeg75: mốc đơn giản; est phải thật hơn cả nó, không chỉ hơn generic
python scripts/build_wild.py --root /data/EarVN1.0 --name earvn --out data/bench/earvn --tiers 96 \
       --roles splits/earvn_roles.json --role test
python scripts/build_wild.py --root /data/awex --name awex --out data/bench/awex_s20 --tiers 96 144 --safety 2.0
# mô hình đề xuất và đối chứng N2 (mốc huấn luyện với cùng suy giảm)
python scripts/train.py --exp S3 ... --degrade est --degrade-params configs/degrade_estimated.json \
       --extra-dir /data/EarVN1.0 --extra-roles splits/earvn_roles.json --extra-name earvn
python scripts/evaluate.py --bench data/bench/earvn --folds none --tiers 96 --kinds bic est \
       --degrade-params configs/degrade_estimated.json --models bicubic --runs runs/S3_... --out results/s5
```

## 4. Tùy chọn, sau điểm kiểm tra 2

- **T4 (N3):** `python scripts/run_t4_oracle.py --bench data/bench/ami --fs rrdb_psnr --ft esrgan`.
  Nếu đạt: `train.py --objective n3s1A --init-ckpt <thân L1>/ckpt/best.pt --pretrain pf`, rồi
  `--objective n3s2 --init-ckpt <giai đoạn 1>/ckpt/best.pt`; đối chứng `--objective gan`, `--objective ldl`.
- **T5 (N1):** `train_landmarks.py` (hai bộ dò) → `landmark_tools.py pseudo` → `train.py --objective aux --landmarks ...`;
  nhánh đối chứng không nhãn: `train.py --extra-dir <ảnh bộ điểm mốc> --extra-name lm`; chấm bằng
  `evaluate.py --landmark-ckpt <bộ dò B> --ridge`, rồi `check_criteria.py n1`.
- **Đường chất lượng theo cỡ ảnh vào:** `run_size_sweep.py`.
- **Khảo sát người xem:** `evaluate.py --save-sr` → `viewer_study.py make` → `viewer_study.py analyze`.
- **Android:** `docs/ANDROID.md`.

## 5. Dựng bảng và hình cho bài (sau mỗi mốc có kết quả mới)

Không gõ tay con số nào vào bản thảo. Mọi bảng và hình sinh từ `results/` và `runs/`.

```bash
# bảng tự động: T2 theo từng cỡ, độ trễ, phép thử ngữ cảnh, bộ phân loại N2, khảo sát người xem
python scripts/make_tables.py                      # -> paper/tables/*.tex
# hình tự động: hình 2 (cỡ ảnh vào), hình 3 (chất lượng theo độ trễ và theo tham số), hình 4 (ngữ cảnh)
python scripts/make_figures.py                     # -> paper/figures/*.pdf và *.png

# bảng so sánh giữa các phương pháp (bảng 4, 6, 7, 8, 11): tự chọn dòng, dòng mốc và số đo
python scripts/compare_table.py --name main \
    --row "Bicubic=results/s2/bicubic__hr144_x4_bic.csv" \
    --row "SPAN (fine-tuned)=runs/S2_span-zero_pub_*/test_hr144_x4_bic.csv" \
    --row "Ours=runs/S2_<mã mô hình đề xuất>_*/test_hr144_x4_bic.csv" \
    --ref "SPAN (fine-tuned)" --metrics psnr_y ssim_y ms_ssim_y lpips dists gmsd \
    --cost results/latency_jetson.csv --cost-key "SPAN (fine-tuned)=<tên trong file độ trễ>" --cost-key "Ours=<...>"
#   -> results/tables/main.csv (trung bình, khoảng tin cậy, chênh lệch ghép cặp) và paper/tables/main.tex
#   Bảng 8 (cấu trúc): cùng lệnh với --metrics ridge_false ridge_missed lr_psnr_y grad_psnr lm_dev
#   (cần chấm bằng evaluate.py --ridge [--landmark-ckpt ...])

# hình 5: đường chất lượng theo độ trễ của các mô hình đã huấn luyện, từ bảng vừa dựng
python scripts/make_figures.py cost --quality results/tables/main.csv --cost results/latency_jetson.csv \
    --out paper/figures/fig5_quality_latency.pdf --threshold 33
#   (tên ở cột method của bảng phải trùng tên ở cột name của file độ trễ; đặt tên dòng cho khớp)

# hình 1 và 6: lưới so sánh định tính (cần ảnh SR đã lưu bằng evaluate.py --save-sr)
python scripts/make_figures.py grid --col LR=data/bench/ami/lr/hr144_x4_bic --col Bicubic=results/sr/hr144_x4_bic/bicubic \
    --col Ours=results/sr/hr144_x4_bic/<mã> --col HR=data/bench/ami/hr144 --keys 012_front 047_left --zoom-lr 4 \
    --out paper/figures/fig1.pdf
```

Hình dùng ảnh AMI cần thư đồng ý của tác giả bộ dữ liệu: xem `docs/AMI_PERMISSION.md` (có thư mẫu). Gửi thư sớm.

Sau đó làm theo mục 2.7 của kế hoạch: xác định kịch bản, điền `[X]`, chọn nhánh câu, viết bình luận từng bảng.

## 6. Mốc thêm cho vòng phản biện (08/10/2026; `TODO.md` mục 0, việc J)

Ba việc, mỗi việc một cấu hình chốt trước: khử nén rồi mới phóng (FBCNN), RRDB học có nén với trọng số có sẵn (BSRNet,
Real-ESRNet), và RRDB tinh chỉnh với đúng công thức của SPAN nhánh est.

```bash
# Bước 1: chỉ chấm (độ trung thực trên ba bộ, nhận dạng, độ trễ). Không chạy song song với việc khác trên cùng GPU,
# vì script có đo độ trễ.
cd weights
curl -L --fail -o BSRNet.pth            https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth
curl -L --fail -o RealESRNet_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
curl -L --fail -o fbcnn_color.pth       https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth
cd ..
nohup bash scripts/run_review.sh > review.log 2>&1 &
tail -f review.log                       # xong khi thấy dòng "Xong. Đọc: results/rev_summary/summary.md"

# Bước 2: tinh chỉnh RRDB, fold 2, 3, 4 trước
bash scripts/make_rrdb_jobs.sh
nohup python scripts/run_queue.py jobs/rrdb.txt > rrdb.log 2>&1 &
grep -h "phép thử ảnh sáng" jobs/rrdb.txt.logs/*.log     # phải là ba dòng ỔN ĐỊNH, không có CẢNH BÁO

# Bước 3: chỉ khi cả ba fold trên ỔN ĐỊNH. Fold 1 và 5 chạy đúng một lần.
FINAL_RUN=1 FOLDS="1 5" bash scripts/make_rrdb_jobs.sh jobs/rrdb_final.txt
nohup python scripts/run_queue.py jobs/rrdb_final.txt > rrdb_final.log 2>&1 &

# Bước 4: chấm thêm các lần chạy RRDB (file đã có được bỏ qua) và tóm tắt lại
nohup bash scripts/run_review.sh > review2.log 2>&1 &
```

Hết bộ nhớ GPU ở bước 2 (lệnh lỗi ngay trong phút đầu, xem `jobs/rrdb.txt.logs/`): dừng, không tự giảm lô; lô nhỏ hơn là một
công thức khác và phải ghi lại trước khi chạy. Kết quả: `results/rev*`, `results/recog/rev_*`, `results/latency_review.csv`,
`results/rev_summary/summary.md`.

## Quy tắc về mã lần chạy

Mã lần chạy không chứa mọi siêu tham số (tốc độ học, số bước, trọng số loss...). Chạy lại cùng mã với siêu
tham số khác thì `train.py` dừng và báo; thêm `--tag` để có mã mới. Lần chạy bị loại: ghi tay trạng thái
`excluded` và lý do vào `results/runs.csv`, không xóa thư mục.
