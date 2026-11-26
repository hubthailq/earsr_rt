#!/usr/bin/env bash
# AWEx nén lại ở JPEG 75 (góp ý của thầy 24/11/2026, mục 8; TODO.md, việc W7). Chỉ chấm, không huấn luyện.
# Giả thuyết trong bài (mục 7): AWEx không có mức tăng vì ảnh dò không mang lỗi nén JPEG. Phép kiểm: giải nén ảnh dò thật
# của AWEx rồi nén lại ở JPEG 75 (libjpeg 4:2:0, như benchmark), rồi chấm lại theo giao thức chính, tức mỗi cách phóng
# dùng đúng bộ nhận dạng đã học trên ảnh phóng bằng chính nó (runs/RECOG_resnet18_up-<cách>, từ run_recog_upscaled.sh).
# Bảy cách phóng như Bảng 5. Ra số nào báo số đó: nếu xuất hiện mức tăng thì giả thuyết được củng cố, nếu không thì ghi đúng.
# Dùng (từ gốc repo):  nohup bash scripts/run_awex_recompress.sh > awex_q75.log 2>&1 &
# Biến môi trường: PYTHON; RES (mặc định results/recog_up); RUNS (mặc định runs); Q (mặc định 75);
#                  METHODS (mặc định bảy cách); LIMIT (N ảnh dò mỗi người, để thử).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog_up}"; RUNS="${RUNS:-runs}"; Q="${Q:-75}"
read -r -a MS <<< "${METHODS:-bicubic span_ch48 est rrdb_est bsrnet realesrnet fbcnn_span_ch48}"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
shopt -s nullglob
EST=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f?)
RRDB=("$RUNS"/REV_rrdb-zero+xearvn_pub_rand_x4_hrall_est_f?)
for m in "${MS[@]}"; do
  [ -f "$RUNS/RECOG_resnet18_up-$m/ckpt.pt" ] || { echo "LỖI: thiếu $RUNS/RECOG_resnet18_up-$m/ckpt.pt" >&2; exit 1; }
  case "$m" in
    est) [ "${#EST[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy SPAN est, thấy ${#EST[@]}" >&2; exit 1; };;
    rrdb_est) [ "${#RRDB[@]}" -eq 5 ] || { echo "LỖI: cần 5 lần chạy RRDB est, thấy ${#RRDB[@]}" >&2; exit 1; };;
  esac
done
mkdir -p "$RES"
for m in "${MS[@]}"; do
  case "$m" in
    est) EV=(--models --runs "${EST[@]}");;
    rrdb_est) EV=(--models --runs "${RRDB[@]}");;
    *) EV=(--models "$m");;
  esac
  echo "== $m, ảnh dò AWEx nén lại ở JPEG $Q"
  "$PY" scripts/recog_eval.py --root data/raw/awex --recognizer "$RUNS/RECOG_resnet18_up-$m/ckpt.pt" "${EV[@]}" \
        --probe-jpeg "$Q" --out "$RES/awex_q${Q}_$m" ${LIM[@]+"${LIM[@]}"}
done
echo "$RES: $(ls -d "$RES"/awex_q${Q}_* | wc -l | tr -d ' ') thư mục awex_q${Q}_*"
echo "Xong."
