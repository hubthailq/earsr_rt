# Mã nguồn và trọng số của bên thứ ba

## Mã nguồn chép vào project

| File | Nguồn | Giấy phép | Thay đổi |
|---|---|---|---|
| `earsr/models/zoo/rlfn.py` | github.com/Amazingren/NTIRE2024_ESR, `models/team00_RLFN.py` (RLFN, ByteDance) | MIT | không |
| `earsr/models/zoo/efdn.py` | github.com/Amazingren/NTIRE2025_ESR, `models/team00_EFDN.py` | MIT | không |
| `earsr/models/zoo/safmnpp.py` | github.com/Amazingren/NTIRE2024_ESR, `models/team23_safmnpp.py` | MIT | bỏ đoạn chạy thử cuối file |
| `earsr/models/zoo/smfan.py` | github.com/Amazingren/NTIRE2024_ESR, `models/team24_smfan.py` | MIT | bỏ đoạn chạy thử cuối file |
| `earsr/models/zoo/swinir.py` | github.com/JingyunLiang/SwinIR, `models/network_swinir.py` | Apache-2.0 | thay import `timm` bằng `_timm_shim.py`; bỏ đoạn chạy thử |

Bản sao giấy phép: `docs/LICENSE_NTIRE_ESR_MIT.txt`, `docs/LICENSE_SwinIR_Apache2.txt`.

## Mã nguồn viết lại theo bài gốc

| File | Theo |
|---|---|
| `earsr/models/span.py` | SPAN, github.com/hongyuanyu/SPAN; giữ tên tham số, thêm kiểu đệm và số khối |
| `earsr/models/zoo/rrdbnet.py` | RRDBNet của ESRGAN (tên tham số theo BasicSR) |
| `earsr/models/zoo/msrresnet.py` | MSRResNet của KAIR |
| `earsr/models/zoo/edsr.py` | EDSR-baseline, github.com/sanghyun-son/EDSR-PyTorch; giữ tên tham số |
| `earsr/models/zoo/srvgg.py` | SRVGGNetCompact của Real-ESRGAN |
| `earsr/data/resize.py` | thuật toán `imresize` của MATLAB; đã so với bản của BasicSR |

## Trọng số (không nằm trong repo; tải bằng `scripts/get_weights.sh`)

| Tên trong project | File | Nguồn |
|---|---|---|
| span_ch48 | span_ch48_x4_official.pth (`spanx4_ch48.pth` trong `span.zip`; tải tay từ Google Drive) | github.com/hongyuanyu/SPAN, Apache-2.0 |
| span_ch48_t44 | team44_SPANx4.pth | NTIRE2025_ESR, đội 44 |
| edsr_baseline | edsr_baseline_x4.pth (`edsr_baseline_x4-6b446fab.pt`) | cv.snu.ac.kr/research/EDSR, kho github.com/sanghyun-son/EDSR-PyTorch (MIT) |
| span_ch28, span_ch26 | team38_span_ch28_slim.pth, team39_spantiny_ch26_slim.pth | NTIRE2024_ESR |
| rlfn, safmnpp, smfan | team00_rlfn.pth, team23_safmnpp.pth, team24_smfan.pth | NTIRE2024_ESR |
| efdn | team00_EFDN.pth | NTIRE2025_ESR |
| msrresnet, rrdb_psnr, esrgan, bsrgan | KAIR release v1.0 | github.com/cszn/KAIR |
| realesrgan, realesrgan_x2, realesr_compact | Real-ESRGAN releases | github.com/xinntao/Real-ESRGAN |
| swinir_light, swinir_light_x2 | SwinIR release v0.0 | github.com/JingyunLiang/SwinIR |

Trước khi công bố bài, kiểm lại điều khoản sử dụng của từng bộ trọng số.

## Dữ liệu

AMI Ear Database: CC BY-NC-ND. Repo không chứa ảnh AMI hay ảnh suy ra từ AMI; chỉ có script dựng lại,
danh sách file và file chia fold. Hình minh họa dùng ảnh AMI trong bài cần xin phép tác giả.

## NTIRE 2026 Efficient SR

`earsr/models/zoo/nt26_pds.py`, `nt26_pkdsr.py`, `nt26_dscf.py`, `nt26_disp.py`, `nt26_errn2.py` chép từ
https://github.com/Amazingren/NTIRE2026_ESR (giấy phép MIT, `docs/LICENSE_NTIRE2026_ESR_MIT.txt`). Thay đổi duy nhất:
bỏ lệnh làm nóng GPU trong `__init__` của PDS và PKDSR, và bỏ một dòng `import tqdm` không dùng.
