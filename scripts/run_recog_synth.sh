#!/usr/bin/env bash
# Tách điều kiện nén của phần nhận dạng (kiểm toán 04/11; TODO.md, việc Q2). Chỉ chấm, không huấn luyện.
# Câu hỏi, cố định trước khi chạy: trên cùng người và cùng ảnh của EarVN1.0, mức tăng hạng 1 của SPAN tinh chỉnh (suy giảm
# đo được) so với bicubic có xuất hiện khi ảnh dò nhỏ KHÔNG nén không, hay chỉ khi ảnh dò bị nén (JPEG 93, 75)?
# Ảnh dò: nửa ảnh lớn không dùng làm ảnh đăng ký (như ref_large), thu về cạnh ngắn 144 rồi suy giảm ×4 thành ảnh 36 px
# (trung vị cạnh ngắn của ảnh nhỏ thật là 37). Ảnh đăng ký, bộ nhận dạng (Ear-R18, Ear-R50) giữ như bài.
# Phương pháp: bicubic, SPAN công bố, SPAN tinh chỉnh (5 fold, gộp như bài).
# Dùng (từ gốc repo):  bash scripts/run_recog_synth.sh
# Biến môi trường: PYTHON; RES (mặc định results/recog_synth); RUNS (mặc định runs); LIMIT (N ảnh dò mỗi người, để thử).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog_synth}"; RUNS="${RUNS:-runs}"
EARVN=(--root data/raw/EarVN1.0 --roles splits/earvn_roles.json)
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
shopt -s nullglob
EST=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f?)
[ "${#EST[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy SPAN est, thấy ${#EST[@]}" >&2; exit 1; }
mkdir -p "$RES"
for tag in resnet18 resnet50; do
  REC="$RUNS/RECOG_$tag/ckpt.pt"
  [ -f "$REC" ] || { echo "LỖI: thiếu $REC" >&2; exit 1; }
  for k in bic bicjpeg93 bicjpeg75; do
    "$PY" scripts/recog_eval.py "${EARVN[@]}" --recognizer "$REC" --models bicubic span_ch48 --runs "${EST[@]}" \
          --synth "$k" --synth-short 36 --out "$RES/${tag}_$k" ${LIM[@]+"${LIM[@]}"}
    "$PY" scripts/summarize_recog.py --in "$RES/${tag}_$k" --out "$RES/${tag}_${k}_summary" > /dev/null
  done
done
echo "$RES: $(ls "$RES"/*/*.csv | wc -l | tr -d ' ') file"
echo "Xong."
