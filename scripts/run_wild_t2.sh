#!/usr/bin/env bash
# Kiểm độ tổng quát của phát hiện giai đoạn 1 ngoài AMI (chỉ chấm, không huấn luyện):
#   1. EarVN1.0 (nhóm test) và AWEx: mọi mô hình có trọng số công bố, ảnh vào bicubic, JPEG 75, JPEG 93 và
#      suy giảm ước lượng. Đáp án là ảnh lớn thu nhỏ ít nhất 2 lần, nên chỉ có ở cỡ 96 (AWEx có thêm cỡ 144).
#   2. AMI: quét mức nén JPEG (60, 85, 93; mức 75 đã có từ run_stage1.sh) ở cỡ 96 và 144, để biết từ mức nào
#      mô hình bắt đầu kém bicubic. Ảnh nhỏ thật của EarVN1.0 có 52% ở mức 75 và 44% ở mức 93.
# Dùng (từ gốc repo):  bash scripts/run_wild_t2.sh
# Biến môi trường: PYTHON; RES (thư mục kết quả, mặc định results); BENCH (mặc định data/bench);
#                  LIMIT (chỉ chấm N ảnh đầu, để thử); PERCEPTUAL=0 để chạy khi không có LPIPS, DISTS.
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results}"; BENCH="${BENCH:-data/bench}"
P="configs/degrade_estimated.json"
[ -f "$P" ] || { echo "LỖI: thiếu $P (chạy scripts/fit_degradation.py trước)" >&2; exit 1; }
[ -f splits/earvn_roles.json ] || { echo "LỖI: thiếu splits/earvn_roles.json" >&2; exit 1; }
PERC="--require-perceptual"; [ "${PERCEPTUAL:-1}" = "0" ] && PERC="--no-perceptual"
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
KINDS=(bic bicjpeg75 bicjpeg93 est)

# 1a. EarVN1.0, nhóm test (những người này không bao giờ dùng để huấn luyện hay khớp tham số)
"$PY" scripts/build_wild.py --root data/raw/EarVN1.0 --name earvn --out "$BENCH/earvn" --tiers 96 \
      --roles splits/earvn_roles.json --role test
"$PY" scripts/evaluate.py --bench "$BENCH/earvn" --folds none --tiers 96 --kinds "${KINDS[@]}" \
      --degrade-params "$P" --out "$RES/t2_earvn" $PERC "${LIM[@]}"
"$PY" scripts/summarize_t2.py --in "$RES/t2_earvn" --out "$RES/t2_earvn_summary"

# 1b. AWEx (không bao giờ dùng để huấn luyện), biên an toàn 2 lần
"$PY" scripts/build_wild.py --root data/raw/awex --name awex --out "$BENCH/awex_s20" --tiers 96 144 --safety 2.0
"$PY" scripts/evaluate.py --bench "$BENCH/awex_s20" --folds none --tiers 96 144 --kinds "${KINDS[@]}" \
      --degrade-params "$P" --out "$RES/t2_awex" $PERC "${LIM[@]}"
"$PY" scripts/summarize_t2.py --in "$RES/t2_awex" --out "$RES/t2_awex_summary"

# 2. AMI: quét mức nén (ghi thêm vào results/t2; các file đã có được bỏ qua)
"$PY" scripts/evaluate.py --bench "$BENCH/ami" --tiers 96 144 --kinds bicjpeg60 bicjpeg85 bicjpeg93 \
      --out "$RES/t2" $PERC "${LIM[@]}"
"$PY" scripts/summarize_t2.py --in "$RES/t2" --out "$RES/t2_summary"
echo "Xong. Đọc: $RES/t2_earvn_summary/T2_summary.md, $RES/t2_awex_summary/T2_summary.md, $RES/t2_summary/T2_summary.md"
