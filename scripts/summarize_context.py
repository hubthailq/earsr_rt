#!/usr/bin/env python3
"""Tổng hợp phép thử ngữ cảnh.  python scripts/summarize_context.py --in results/t6_context --out results/t6_summary"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earsr.report.context import load_dir, summarize, to_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(load_dir(a.inp), a.n_boot)
    s.to_csv(out / "context.csv", index=False)
    (out / "T6_context_summary.md").write_text(to_markdown(s), encoding="utf-8")
    print(to_markdown(s))


if __name__ == "__main__":
    main()
