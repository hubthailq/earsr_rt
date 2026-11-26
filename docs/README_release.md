# Efficient super-resolution of small compressed ear images

Code for the paper *Efficient Super-Resolution of Small Compressed Ear Images: Compression Error, Image Content and
Training Degradation* (Thai Le Quang, Vinh Truong Hoang).

The code measures where published efficient super-resolution (SR) models fall below bicubic interpolation on small
JPEG-compressed ear images, decomposes their error into an interpolation part and a compression part, fine-tunes a
real-time network over the JPEG range read from the target files, and evaluates identification on real small ear
images.

The per-image result files and the trained weights are archived separately (see *Results and weights* below), so that
every table and figure of the paper can be regenerated without a GPU.

## Installation

```bash
pip install -r requirements.txt
bash scripts/get_weights.sh        # published weights of the evaluated models, into ./weights
python scripts/check_data.py       # checks the dataset layout under data/raw
```

Python 3.10 or later and PyTorch are required. LPIPS and DISTS are computed when their packages are installed.

## Data

The datasets are not redistributed; obtain them from their providers and place them under `data/raw/`:

| Dataset | Use | Layout |
|---|---|---|
| AMI ear database | benchmark with near-clean ground truth, fine-tuning | `data/raw/AMI/` |
| EarVN1.0 | in-the-wild benchmark, degradation measurement, identification | `data/raw/EarVN1.0/<subject>/<image>` |
| AWEx (extended Annotated Web Ears) | in-the-wild benchmark and identification, never used for training | `data/raw/awex/<subject>/<image>` |
| DIV2K validation | natural-image control | `data/raw/DIV2K_valid_HR/` |

The subject-wise splits used in the paper are in `splits/` (five AMI folds and the EarVN1.0 subject roles), and the
measured degradation parameters are in `configs/degrade_estimated.json`. `docs/DATA_AND_OUTPUTS.md` describes the
layout of inputs and outputs.

## Reproducing the paper

The commands below are run from the repository root on a machine with a GPU. Each script documents its options at the
top of the file; files that already exist are skipped, so a script can be resumed.

1. **Published models on small ear images** (no training): benchmark construction, JPEG levels, input sizes,
   in-the-wild sets, natural-image control, order of degradations and other codecs.
   ```bash
   bash scripts/run_stage1.sh data/raw/AMI data/raw/DIV2K_valid_HR
   bash scripts/run_wild_t2.sh
   bash scripts/run_ami48.sh
   bash scripts/run_div2k_control.sh
   bash scripts/run_jpeg_grid.sh
   bash scripts/run_order_control.sh
   bash scripts/run_codec.sh
   ```
2. **Measured compression range and fine-tuning** (SPAN and DISP, five folds; RRDB and SwinIR-light under the same
   recipe; extra seeds).
   ```bash
   python scripts/fit_degradation.py --help        # reads JPEG levels from the files of real small ear images
   bash scripts/make_n2b_jobs.sh jobs/n2b.txt && python scripts/run_queue.py jobs/n2b.txt
   bash scripts/make_n2c_jobs.sh jobs/n2c.txt && python scripts/run_queue.py jobs/n2c.txt
   bash scripts/make_final_jobs.sh jobs/final.txt && python scripts/run_queue.py jobs/final.txt
   bash scripts/make_rrdb_jobs.sh && python scripts/run_queue.py jobs/rrdb.txt
   bash scripts/make_swinir_jobs.sh && python scripts/run_queue.py jobs/swinir.txt
   bash scripts/run_seeds.sh
   ```
3. **Scoring the fine-tuned models and published alternatives.**
   ```bash
   bash scripts/score_n2.sh
   bash scripts/run_review.sh
   bash scripts/score_swinir.sh
   ```
4. **Identification on real small ear images.**
   ```bash
   bash scripts/run_recog.sh                 # recognisers trained on large images
   bash scripts/run_recog_upscaled.sh        # main protocol: one recogniser per upscaling method
   METHODS="bsrnet realesrnet fbcnn_span_ch48 rrdb_est" bash scripts/run_recog_upscaled.sh
   bash scripts/run_recog_synth2.sh          # synthesised probes
   bash scripts/run_awex_recompress.sh       # AWEx probes recompressed at JPEG 75
   ```
5. **Tables, figures and numbers.** `scripts/make_paper.py` reads `results/` and writes every number, table and figure
   of the paper; `scripts/recog_curves.py` computes CMC and TAR@FAR from the stored similarity scores.
6. **Latency on a smartphone.** `scripts/export_coreml.py` converts the models to Core ML; `deploy_ios/README.md`
   describes the measurement with the Xcode performance report.

Every training run passes a brightness stress test at the end (`earsr/eval/stress.py`); runs are logged in
`results/runs.csv`.

## Results and weights

The per-image result files (`results/`) and the trained weights are archived at:
**[archive DOI, to be added]**. Unpacking the results archive at the repository root is enough to rerun step 5
without a GPU.

## Third-party code and licences

The model definitions of published networks come from their authors' repositories; their licences are listed in
`docs/THIRD_PARTY.md` and the `docs/LICENSE_*` files.

## Citation

If you use this code, please cite the paper (reference to be added on publication).
