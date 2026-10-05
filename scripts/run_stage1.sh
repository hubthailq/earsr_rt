#!/usr/bin/env bash
# Giai đoạn 1 (không huấn luyện): P1, P2, T2, T6 (i) và (v), độ trễ trên máy, xuất ONNX cho thiết bị.
# Dùng: bash scripts/run_stage1.sh /path/AMI [/path/ảnh_tự_nhiên_HR]
# Biến môi trường: PYTHON=lệnh python cần dùng (mặc định python, không có thì python3);
#                  PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS (mặc định: thiếu thì dừng).
set -euo pipefail
AMI="${1:?cần đường dẫn AMI gốc}"; GEN="${2:-}"
B=data/bench/ami

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
  command -v "$PY" >/dev/null 2>&1 || { echo "LỖI: PYTHON='$PY' không chạy được" >&2; exit 1; }
else
  PY=python
  command -v "$PY" >/dev/null 2>&1 || PY=python3
  command -v "$PY" >/dev/null 2>&1 || { echo "LỖI: không thấy lệnh python hay python3 (đặt PYTHON=/đường/dẫn/python)" >&2; exit 1; }
fi
echo "dùng: $PY ($("$PY" --version 2>&1))"

# Các script chấm bỏ qua file kết quả đã có (để chạy tiếp được sau khi bị ngắt). Vì vậy kết quả của một lần
# chạy khác (ví dụ kết quả sơ bộ trên CPU đi kèm project, không có LPIPS, DISTS, và 'span_ch48' trong đó là
# trọng số đội 44) phải được dời đi trước, nếu không chúng lẫn vào bảng. File dấu .stage1_run đánh dấu thư mục
# do chính script này tạo ra.
STALE=""
for d in results/t2 results/t6_context; do
  if [ ! -f "$d/.stage1_run" ] && ls "$d"/*.csv >/dev/null 2>&1; then STALE="$STALE $d"; fi
done
if [ -n "$STALE" ]; then
  {
    echo "DỪNG: có kết quả cũ không do script này tạo ra trong:$STALE"
    echo "Dời chúng đi rồi chạy lại:"
    echo "  mkdir -p results_prelim_cpu"
    echo "  for d in t2 t2_summary t6_context t6_summary; do [ -e results/\$d ] && mv results/\$d results_prelim_cpu/; done"
  } >&2
  exit 1
fi
mkdir -p results/t2 results/t6_context
touch results/t2/.stage1_run results/t6_context/.stage1_run

PERC="--require-perceptual"
if [ "${PERCEPTUAL:-1}" = "0" ]; then PERC="--no-perceptual"; echo "CHÚ Ý: PERCEPTUAL=0, kết quả sẽ không có LPIPS, DISTS"; fi

"$PY" scripts/build_benchmark.py --ami-raw "$AMI" --out $B
"$PY" scripts/run_t2.py --bench $B --out results/t2 --tiers 96 144 192 --scale 4 --kinds bic bicjpeg75 $PERC
"$PY" scripts/run_t2.py --bench $B --out results/t2 --tiers 144 --scale 2 --kinds bic bicjpeg75 $PERC
"$PY" scripts/summarize_t2.py --in results/t2 --out results/t2_summary
"$PY" scripts/run_t6_context.py --ami-raw "$AMI" --out results/t6_context --configs wide near small
if [ -n "$GEN" ]; then
  "$PY" scripts/run_t6_context.py --generic "$GEN" --generic-name generic --out results/t6_context --configs wide small
fi
"$PY" scripts/summarize_context.py --in results/t6_context --out results/t6_summary
"$PY" scripts/bench_local.py --out results/latency_local.csv
"$PY" scripts/export_for_device.py --out deploy_jetson/onnx
echo "Xong giai đoạn 1. Chép deploy_jetson sang thiết bị để đo T3."
