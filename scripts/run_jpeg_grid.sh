#!/usr/bin/env bash
# Lưới mức JPEG dày (góp ý của thầy 24/11/2026, mục 2; TODO.md, việc W6). Chỉ chấm, không huấn luyện.
# Câu hỏi: điểm đổi dấu của mức hơn bicubic nằm ở đâu khi đo trực tiếp, thay vì nội suy giữa 85 và 93.
#   AMI      : thêm JPEG 80, 83, 87, 89, 91, 95, 97 ở ba cỡ (96, 144, 192), 16 mô hình công bố và bicubic   -> results/t2
#              cùng SPAN tinh chỉnh nhánh est (5 fold, mỗi mô hình trên người test của fold nó)          -> results/n2
#   EarVN1.0 : thêm JPEG 85 và 60 (cỡ 96)                                                            -> results/t2_earvn
#   AWEx     : thêm JPEG 85 và 60 (cỡ 96 và 144)                                                     -> results/t2_awex
# Cùng ảnh, cùng hạt giống như các mức đã có, nên ghép cặp được với kết quả cũ. File đã có được bỏ qua.
# Không tính LPIPS, DISTS (phần phân tích chỉ dùng PSNR); đặt PERCEPTUAL=1 nếu muốn tính.
# Dùng (từ gốc repo):  nohup bash scripts/run_jpeg_grid.sh > jpeg_grid.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results); BENCH (mặc định data/bench); RUNS (mặc định runs);
#                  LIMIT (chỉ chấm N ảnh đầu, để thử); AMI_Q, WILD_Q (đổi danh sách mức, để thử); FT=0 bỏ SPAN tinh chỉnh;
#                  AMI_TIERS (mặc định "96 144 192"; máy thiếu cỡ 192 thì đặt "96 144" để thử).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"; RUNS="${RUNS:-runs}"
PERC="--no-perceptual"; [ "${PERCEPTUAL:-0}" = "1" ] && PERC="--require-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
read -r -a AQ <<< "${AMI_Q:-80 83 87 89 91 95 97}"
read -r -a WQ <<< "${WILD_Q:-85 60}"
read -r -a AT <<< "${AMI_TIERS:-96 144 192}"
AK=(); for q in "${AQ[@]}"; do AK+=("bicjpeg$q"); done
WK=(); for q in "${WQ[@]}"; do WK+=("bicjpeg$q"); done
M=(--models bicubic span_ch48 span_ch28 span_ch26 rlfn efdn safmnpp smfan msrresnet edsr_baseline swinir_light rrdb_psnr \
            pds26 pkdsr26 dscf26 disp26 errn26)
for d in "$BENCH/ami" "$BENCH/earvn" "$BENCH/awex_s20"; do
  [ -f "$d/manifest.csv" ] || { echo "LỖI: thiếu $d/manifest.csv (chạy lại các script dựng benchmark trước)" >&2; exit 1; }
done

echo "== AMI, mức ${AQ[*]}"
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers "${AT[@]}" --kinds "${AK[@]}" "${M[@]}" \
      --out "$RES/t2" $PERC ${LIM[@]+"${LIM[@]}"}
if [ "${FT:-1}" = "1" ]; then
  shopt -s nullglob
  R=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f?)
  [ "${#R[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy SPAN est, thấy ${#R[@]}" >&2; exit 1; }
  echo "== AMI, SPAN tinh chỉnh (est), mức ${AQ[*]}"
  "$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers "${AT[@]}" --kinds "${AK[@]}" --models bicubic span_ch48 \
        --runs "${R[@]}" --out "$RES/n2" $PERC ${LIM[@]+"${LIM[@]}"}
fi
echo "== EarVN1.0, mức ${WQ[*]}"
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds "${WK[@]}" "${M[@]}" \
      --out "$RES/t2_earvn" $PERC ${LIM[@]+"${LIM[@]}"}
echo "== AWEx, mức ${WQ[*]}"
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds "${WK[@]}" "${M[@]}" \
      --out "$RES/t2_awex" $PERC ${LIM[@]+"${LIM[@]}"}
for d in t2 t2_earvn t2_awex; do
  echo "$RES/$d: $(ls "$RES/$d" | grep -cE "bicjpeg($(IFS='|'; echo "${AQ[*]}|${WQ[*]}"))\.csv$") file ở các mức mới"
done
echo "Xong."
