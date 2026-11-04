#!/usr/bin/env bash
# Đối chứng thứ tự suy giảm (phản biện ngày 04/11; TODO.md, việc O2). Chỉ chấm, không huấn luyện.
# AMI, ảnh vào 36 px (đáp án 144), mọi người; 16 mô hình công bố, bicubic, và SPAN tinh chỉnh (est, JPEG 75; 5 fold,
# mỗi mô hình chỉ chấm người test của fold của nó). Các kiểu ảnh vào (earsr/degrade/pipelines.py):
#   bicjpeg75, bicjpeg93      : như bài (thu nhỏ rồi nén, lưới khối thẳng với ảnh, màu 4:2:0)
#   bicjpeg75c444             : như trên nhưng không lấy mẫu màu
#   shiftjpeg75, shiftjpeg93  : lưới khối lệch 1..7 px so với ảnh (ảnh cắt từ ảnh đã nén)
#   dbljpeg93to75             : nén 93 lệch lưới, cắt, rồi lưu lại ở 75
#   jpeg75bic                 : nén ảnh lớn trước, thu nhỏ sau
# Dùng (từ gốc repo):  bash scripts/run_order_control.sh        (khoảng 20 đến 40 phút trên GPU, ước lượng)
# Biến môi trường: PYTHON; RES (mặc định results); RUNS (mặc định runs); LIMIT (thử N ảnh); NORUNS=1 bỏ mô hình tinh chỉnh.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; RUNS="${RUNS:-runs}"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
K=(bicjpeg75 bicjpeg93 bicjpeg75c444 shiftjpeg75 shiftjpeg93 dbljpeg93to75 jpeg75bic)
M=(bicubic span_ch48 span_ch28 span_ch26 rlfn efdn safmnpp smfan msrresnet edsr_baseline swinir_light rrdb_psnr
   pds26 pkdsr26 dscf26 disp26 errn26)
"$PY" scripts/evaluate.py --bench data/bench/ami --tiers 144 --kinds "${K[@]}" --models "${M[@]}" \
      --out "$RES/order" --no-perceptual ${LIM[@]+"${LIM[@]}"}
if [ "${NORUNS:-0}" != "1" ]; then
  shopt -s nullglob
  R=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f? "$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_bicjpeg75_f?)
  [ "${#R[@]}" -eq 10 ] || { echo "LỖI: cần 10 lần chạy SPAN (est và bicjpeg75, 5 fold), thấy ${#R[@]}" >&2; exit 1; }
  "$PY" scripts/evaluate.py --bench data/bench/ami --tiers 144 --kinds "${K[@]}" --models --runs "${R[@]}" \
        --out "$RES/order" --no-perceptual ${LIM[@]+"${LIM[@]}"}
fi
echo "$RES/order: $(ls "$RES/order"/*.csv | wc -l | tr -d ' ') file (đủ là 189)"
echo "Xong."
