#!/usr/bin/env bash
# Tải trọng số công bố của các mô hình so sánh vào ./weights (tất cả từ GitHub).
# Dùng: bash scripts/get_weights.sh [thư_mục_đích]
set -euo pipefail
W="${1:-weights}"; mkdir -p "$W"; TMP="$(mktemp -d)"
# Tải vào file .part rồi mới đổi tên (file tải dở không bị coi là xong); thử lại tối đa 3 lần.
dl() {
  [ -f "$W/$2" ] && return 0
  for _ in 1 2 3; do
    curl -L --fail --retry 3 -o "$W/$2.part" "$1" && mv "$W/$2.part" "$W/$2" && return 0
    sleep 3
  done
  echo "LỖI: không tải được $1" >&2; return 1
}

# NTIRE Efficient SR 2024 và 2025: trọng số nằm trong repo (giấy phép MIT)
git clone --depth 1 https://github.com/Amazingren/NTIRE2024_ESR "$TMP/n24"
git clone --depth 1 https://github.com/Amazingren/NTIRE2025_ESR "$TMP/n25"
cp "$TMP/n24/model_zoo/"{team00_rlfn.pth,team23_safmnpp.pth,team24_smfan.pth,team38_span_ch28_slim.pth,team39_spantiny_ch26_slim.pth} "$W/"
cp "$TMP/n25/model_zoo/"{team00_EFDN.pth,team44_SPANx4.pth} "$W/"
# NTIRE Efficient SR 2026: baseline SPAN chính thức và năm đội đầu bảng chạy được bằng tích chập thường
git clone --depth 1 https://github.com/Amazingren/NTIRE2026_ESR "$TMP/n26"
for f in team00_SPAN team01_PDS team15_DSCF_Fused team16_PKDSR team18_DISP team20_ERRN2; do
  cp "$TMP/n26/model_zoo/$f.pth" "$W/nt26_$f.pth"
done

K=https://github.com/cszn/KAIR/releases/download/v1.0
dl $K/RRDB.pth              kair_RRDB_psnr_x4.pth
dl $K/ESRGAN.pth            kair_ESRGAN_x4.pth
dl $K/BSRGAN.pth            BSRGAN.pth
dl $K/msrresnet_x4_psnr.pth msrresnet_x4_psnr.pth
R=https://github.com/xinntao/Real-ESRGAN/releases/download
dl $R/v0.1.0/RealESRGAN_x4plus.pth        RealESRGAN_x4plus.pth
dl $R/v0.2.1/RealESRGAN_x2plus.pth        RealESRGAN_x2plus.pth
dl $R/v0.2.5.0/realesr-general-x4v3.pth   realesr-general-x4v3.pth
S=https://github.com/JingyunLiang/SwinIR/releases/download/v0.0
dl $S/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth swinir_light_x4.pth
dl $S/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth swinir_light_x2.pth
# EDSR-baseline: máy chủ của tác giả (không nằm trên GitHub). Tên file gốc mang 8 ký tự đầu của SHA-256.
dl https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt edsr_baseline_x4.pth
SHA="$( (sha256sum "$W/edsr_baseline_x4.pth" 2>/dev/null || shasum -a 256 "$W/edsr_baseline_x4.pth") | cut -c1-8)"
if [ "$SHA" != "6b446fab" ]; then
  echo "LỖI: $W/edsr_baseline_x4.pth sai mã băm ($SHA, cần 6b446fab); xóa file rồi chạy lại." >&2; exit 1
fi
rm -rf "$TMP"; ls -la "$W"
echo "Xong các bộ tải tự động được."
if [ ! -f "$W/span_ch48_x4_official.pth" ]; then
  echo "CÒN THIẾU: trọng số SPAN 48 kênh chính thức (mốc 'span_ch48'). File nằm trên Google Drive, phải tải tay:"
  echo "  https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view   (span.zip, 1,2 GB)"
  echo "  giải nén lấy spanx4_ch48.pth (18029841 byte) rồi chép thành $W/span_ch48_x4_official.pth"
fi
