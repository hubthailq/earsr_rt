#!/usr/bin/env bash
# Đo nhận dạng tai trên ảnh nhỏ thật (việc 2 của docs/story-imavis.md).
#   1. Huấn luyện hai mạng nhận dạng (ResNet-18 và ResNet-50), chỉ trên ảnh lớn của người EarVN1.0 có vai train, fit, clf.
#   2. Chấm mọi phương pháp phóng ảnh trên ảnh nhỏ thật của hai bộ:
#        EarVN1.0, nhóm test và viewer (những người không dùng để huấn luyện SR hay mạng nhận dạng);
#        AWEx, mọi người (bộ này chưa từng dùng để huấn luyện gì);
#      mỗi bộ với ba mạng nhận dạng: ResNet-18, ResNet-50, và một mạng ImageNet đóng băng chưa từng thấy ảnh tai.
#   3. Tổng hợp: results/recog/<tên>_summary/summary.md (có bảng tách theo mức nén JPEG của file ảnh dò, nếu là ảnh JPEG).
# File đã có được bỏ qua, nên gọi lại sau khi có thêm lần chạy là an toàn.
# Dùng (từ gốc repo):  bash scripts/run_recog.sh
# Biến môi trường: PYTHON; RES (mặc định results/recog); RUNS (mặc định runs).
set -euo pipefail
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3
RES="${RES:-results/recog}"; RUNS="${RUNS:-runs}"
EARVN=(--root data/raw/EarVN1.0 --roles splits/earvn_roles.json)
AWEX=(--root data/raw/awex)
shopt -s nullglob
R=("$RUNS"/N2_*+*)
[ "${#R[@]}" -gt 0 ] || { echo "LỖI: không thấy lần chạy nào khớp $RUNS/N2_*+*" >&2; exit 1; }
mkdir -p "$RES"
for arch in resnet18 resnet50; do
  [ -f "$RUNS/RECOG_$arch/ckpt.pt" ] || "$PY" scripts/train_recognizer.py "${EARVN[@]}" --arch "$arch" --out "$RUNS/RECOG_$arch"
done
cp "$RUNS/RECOG_resnet18/train_log.json" "$RES/recognizer_train_log.json"
cp "$RUNS/RECOG_resnet50/train_log.json" "$RES/recognizer_resnet50_train_log.json"
M=(--models bicubic span_ch48 disp26 rrdb_psnr bsrgan realesrgan)
for tag in resnet18 resnet50 imagenet50; do
  case "$tag" in imagenet50) REC="imagenet-resnet50";; *) REC="$RUNS/RECOG_$tag/ckpt.pt";; esac
  "$PY" scripts/recog_eval.py "${EARVN[@]}" --recognizer "$REC" "${M[@]}" --runs "${R[@]}" --out "$RES/$tag"
  "$PY" scripts/recog_eval.py "${AWEX[@]}" --recognizer "$REC" "${M[@]}" --runs "${R[@]}" --out "$RES/awex_$tag"
  for d in "$tag" "awex_$tag"; do "$PY" scripts/summarize_recog.py --in "$RES/$d" --out "$RES/${d}_summary" > /dev/null; done
done
# ảnh minh họa cho hình định tính: 12 ảnh dò rải trên nhiều người, sáu phương pháp (thư mục _qual không đưa vào git)
Q=("$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_generic_f2 "$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_bicjpeg75_f2
   "$RUNS"/N2_span-zero+xearvn_pub_rand_x4_hrall_est_f2)
[ -d "$RES/sr2/bicubic" ] || "$PY" scripts/recog_eval.py "${EARVN[@]}" --recognizer "$RUNS/RECOG_resnet18/ckpt.pt" \
      --models bicubic span_ch48 bsrgan --runs "${Q[@]}" --out "$RES/_qual" --save-sr "$RES/sr2" --save-n 12 > /dev/null
echo "Xong. Đọc: $RES/resnet18_summary/summary.md (EarVN1.0) và $RES/awex_resnet18_summary/summary.md (AWEx)"
