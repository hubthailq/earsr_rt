#!/usr/bin/env bash
# Đo nhận dạng tai trên ảnh nhỏ thật (việc 2 của docs/story-imavis.md). Ba bước:
#   1. huấn luyện mạng nhận dạng (ResNet-18, chỉ ảnh lớn của người có vai train, fit, clf; khoảng 10 đến 20 phút);
#   2. chấm mọi phương pháp phóng ảnh trên ảnh nhỏ thật của nhóm test và viewer, với hai mạng nhận dạng:
#      mạng vừa huấn luyện và một mạng ImageNet đóng băng chưa từng thấy ảnh tai (đối chứng);
#   3. tổng hợp: results/recog/<mạng>_summary/summary.md.
# File đã có được bỏ qua, nên gọi lại sau khi có thêm lần chạy là an toàn.
# Dùng (từ gốc repo):  bash scripts/run_recog.sh
# Biến môi trường: PYTHON; RES (mặc định results/recog); RUNS (mặc định runs); REC (thư mục mạng nhận dạng).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog}"; RUNS="${RUNS:-runs}"; REC="${REC:-$RUNS/RECOG_resnet18}"
D=(--root data/raw/EarVN1.0 --roles splits/earvn_roles.json)
shopt -s nullglob
R=("$RUNS"/N2_*+*)
[ "${#R[@]}" -gt 0 ] || { echo "LỖI: không thấy lần chạy nào khớp $RUNS/N2_*+*" >&2; exit 1; }
[ -f "$REC/ckpt.pt" ] || "$PY" scripts/train_recognizer.py "${D[@]}" --out "$REC"
mkdir -p "$RES"; cp "$REC/train_log.json" "$RES/recognizer_train_log.json"
M=(--models bicubic span_ch48 disp26 rrdb_psnr bsrgan realesrgan)
SAVE='^(bicubic|span_ch48|bsrgan|N2_span-zero\+xearvn_pub_rand_x4_hrall_(est|generic|bicjpeg75)_f2)$'
"$PY" scripts/recog_eval.py "${D[@]}" --recognizer "$REC/ckpt.pt" "${M[@]}" --runs "${R[@]}" --out "$RES/resnet18" \
      --save-sr "$RES/sr" --save-match "$SAVE"
"$PY" scripts/recog_eval.py "${D[@]}" --recognizer imagenet-resnet50 "${M[@]}" --runs "${R[@]}" --out "$RES/imagenet50"
for t in resnet18 imagenet50; do "$PY" scripts/summarize_recog.py --in "$RES/$t" --out "$RES/${t}_summary" > /dev/null; done
echo "Xong. Đọc: $RES/resnet18_summary/summary.md và $RES/imagenet50_summary/summary.md"
