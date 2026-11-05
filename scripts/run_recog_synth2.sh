#!/usr/bin/env bash
# Ảnh dò tổng hợp gần ảnh thật hơn (TODO.md, việc Q2b, 05/11). Chỉ chấm, không huấn luyện.
# Lần đầu (run_recog_synth.sh): ảnh dò 36 px thu từ ảnh lớn gần như không mất độ chính xác (bicubic 56,8% so với 60,2% của
# ảnh lớn; ảnh nhỏ thật 13,2%), nên không còn dư địa để so phương pháp.
# Câu hỏi, cố định trước khi chạy:
#   1. Ảnh dò 24 px (cận dưới của ảnh thật) và suy giảm đo được (est: mờ, nhiễu, hai mức JPEG) có hạ rank-1 của bicubic
#      về gần ảnh thật không; trên các ảnh dò đó SPAN tinh chỉnh (est) có lấy lại được phần lớn mức tăng của ảnh thật không.
#   2. Việc SPAN tinh chỉnh kém bicubic trên ảnh dò không nén là do tinh chỉnh trên ảnh tai nói chung (nhánh bic cũng kém)
#      hay do huấn luyện có nén (chỉ nhánh bicjpeg75 và est kém).
# Lưới cố định: cạnh ngắn 24 và 36 × kiểu bic, bicjpeg75, est × Ear-R18, Ear-R50; phương pháp: bicubic, SPAN công bố,
# SPAN tinh chỉnh với bic, bicjpeg75, est (mỗi nhánh 5 fold, gộp như bài). Không chọn hay bỏ ô nào theo kết quả.
# Dùng (từ gốc repo):  nohup bash scripts/run_recog_synth2.sh > synth2.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results/recog_synth2); RUNS (mặc định runs); LIMIT (N ảnh dò mỗi người, để thử);
#                  SHORTS, KINDS, TAGS (để thử; mặc định như lưới trên).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog_synth2}"; RUNS="${RUNS:-runs}"
P="configs/degrade_estimated.json"
read -r -a SH <<< "${SHORTS:-24 36}"
read -r -a KS <<< "${KINDS:-bic bicjpeg75 est}"
read -r -a TG <<< "${TAGS:-resnet18 resnet50}"
EARVN=(--root data/raw/EarVN1.0 --roles splits/earvn_roles.json)
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
shopt -s nullglob
R=()
for k in bic bicjpeg75 est; do
  x=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_${k}_f?)
  [ "${#x[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy SPAN $k, thấy ${#x[@]}" >&2; exit 1; }
  R+=("${x[@]}")
done
mkdir -p "$RES"
for tag in "${TG[@]}"; do
  REC="$RUNS/RECOG_$tag/ckpt.pt"
  [ -f "$REC" ] || { echo "LỖI: thiếu $REC" >&2; exit 1; }
  for s in "${SH[@]}"; do
    for k in "${KS[@]}"; do
      o="$RES/${tag}_${s}px_$k"
      "$PY" scripts/recog_eval.py "${EARVN[@]}" --recognizer "$REC" --models bicubic span_ch48 --runs "${R[@]}" \
            --synth "$k" --synth-short "$s" --degrade-params "$P" --out "$o" ${LIM[@]+"${LIM[@]}"}
      "$PY" scripts/summarize_recog.py --in "$o" --out "${o}_summary" > /dev/null
    done
  done
done
echo "$RES: $(ls "$RES"/*/*.csv | wc -l | tr -d ' ') file"
echo "Xong."
