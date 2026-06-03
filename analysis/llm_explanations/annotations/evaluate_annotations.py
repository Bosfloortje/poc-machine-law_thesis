#!/usr/bin/env python3
"""
Full evaluation pipeline — run once to get all metrics over all annotation sheets.

Steps:
  1. parse_annotations  -> annotations_long.csv, annotations_scores.csv, annotations_model.csv
  2. inter_rater        -> interrater_kappa.csv, interrater_spearman.csv,
                           auto_metrics.csv, auto_vs_human_correlation.csv

Usage (run from project root):
    uv run python analysis/llm_explanations/annotations/evaluate_annotations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "web"))

import parse_annotations
import inter_rater

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 -- Parse annotation sheets")
    print("=" * 60)
    parse_annotations.main()

    print()
    print("=" * 60)
    print("STEP 2 -- Inter-rater + auto metrics + correlations")
    print("=" * 60)
    inter_rater.main()

    print()
    print("=" * 60)
    print("Done. All output in:")
    print(str(parse_annotations.OUTPUT_DIR))
    print("=" * 60)
