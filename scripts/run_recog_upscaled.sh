#!/usr/bin/env bash
# Bộ nhận dạng học trên ảnh đã phóng (phản biện ngày 04/11; TODO.md, việc O4).
# Câu hỏi: mức tăng hạng 1 của SPAN tinh chỉnh so với bicubic có còn khi mỗi phương pháp có một bộ nhận dạng học trên
# chính ảnh do nó phóng không (trong SR mặt, mức tăng kiểu này thường co lại)?
# Công thức cố định trước, không dò: giống hệt ResNet-18 của bài (30 epoch, lr 3e-4, seed 0, cùng người học, cùng ảnh
# theo dõi), chỉ thêm ảnh NHỎ THẬT (cạnh ngắn 24..48) của chính những người học, phóng ×4 bằng phương pháp đó.
# Ba phương pháp: bicubic, SPAN công bố, SPAN tinh chỉnh (est; ảnh học chia đều cho 5 mô hình fold, chấm cả 5 fold rồi
# gộp như bài). Mỗi bộ nhận dạng chỉ chấm phương pháp của nó, trên
# EarVN1.0 (test, viewer) và AWEx. So với bộ nhận dạng chỉ học ảnh lớn: results/recog/resnet18 và awex_resnet18.
# Dùng (từ gốc repo):  bash scripts/run_recog_upscaled.sh        (khoảng 1 đến 2 giờ trên GPU, ước lượng)
# Thêm ngày 06/11 (phản biện thứ tám; TODO.md, việc V): cùng công thức cho bốn cách phóng có tính đến nén:
#   METHODS="bsrnet realesrnet fbcnn_span_ch48 rrdb_est" bash scripts/run_recog_upscaled.sh
#   (rrdb_est = 5 lần chạy REV_rrdb-*_est_f?, gộp như SPAN est). Công thức cố định, không dò; ra số nào báo số đó.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog_up}"; RUNS="${RUNS:-runs}"
EARVN=(--root data/raw/EarVN1.0 --roles splits/earvn_roles.json)
AWEX=(--root data/raw/awex)
shopt -s nullglob
read -r -a MS <<< "${METHODS:-bicubic span_ch48 est}"
EST=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f?)
RRDB=("$RUNS"/REV_rrdb-zero+xearvn_pub_rand_x4_hrall_est_f?)
for m in "${MS[@]}"; do
  case "$m" in
    est) [ "${#EST[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy SPAN est, thấy ${#EST[@]}" >&2; exit 1; };;
    rrdb_est) [ "${#RRDB[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy RRDB est, thấy ${#RRDB[@]}" >&2; exit 1; };;
  esac
done
mkdir -p "$RES"
for m in "${MS[@]}"; do
  case "$m" in
    est) SRC=(--sr-run "${EST[@]}"); EV=(--models --runs "${EST[@]}");;
    rrdb_est) SRC=(--sr-run "${RRDB[@]}"); EV=(--models --runs "${RRDB[@]}");;
    *) SRC=(--sr-model "$m"); EV=(--models "$m");;
  esac
  CK="$RUNS/RECOG_resnet18_up-$m"
  [ -f "$CK/ckpt.pt" ] || "$PY" scripts/train_recognizer.py "${EARVN[@]}" --arch resnet18 "${SRC[@]}" --out "$CK"
  cp "$CK/train_log.json" "$RES/train_log_$m.json"
  "$PY" scripts/recog_eval.py "${EARVN[@]}" --recognizer "$CK/ckpt.pt" "${EV[@]}" --out "$RES/earvn_$m"
  "$PY" scripts/recog_eval.py "${AWEX[@]}" --recognizer "$CK/ckpt.pt" "${EV[@]}" --out "$RES/awex_$m"
done
echo "$RES: $(ls "$RES"/*/*.csv | wc -l | tr -d ' ') file"
echo "Xong."
