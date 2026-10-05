#!/usr/bin/env python3
"""T2: đánh giá mô hình có sẵn trên benchmark AMI, không huấn luyện.

Tên cũ của ``scripts/evaluate.py``; mọi tham số như nhau. Ví dụ:
  python scripts/run_t2.py --bench data/bench/ami --out results/t2 --tiers 96 144 192 --kinds bic bicjpeg75
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
