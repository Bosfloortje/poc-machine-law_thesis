#!/usr/bin/env python3
"""
Inter-rater reliability and correlation analysis for annotation data.

Computes:
  Burgers:
    1. Cohen's Kappa       - inter-rater agreement on actionability (Ja/Deels/Nee)
    2. Spearman + Kendall  - inter-rater correlation on duidelijkheid + leesbaarheid
  Juristen:
    3. Cohen's Kappa       - inter-rater agreement on ontbrekende_informatie (Ja/Deels/Nee)
    4. Spearman + Kendall  - inter-rater correlation on juridische_correctheid
  All:
    5. Automatic metrics   - text length, sentence count per explanation
    6. Pearson/Spearman/Kendall - automatic metrics vs. mean human scores

Output:
  results/interrater_kappa.csv
  results/interrater_spearman.csv
  results/auto_metrics.csv
  results/auto_vs_human_correlation.csv

Usage (run from project root):
    uv run python analysis/llm_explanations/annotations/inter_rater.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR   = Path(__file__).parent / "results"
THESIS_DIR   = Path(__file__).parent.parent / "output" / "thesis_20260428_091358"
LONG_CSV     = OUTPUT_DIR / "annotations_long.csv"


LAW_SLUGS = {
    "zorgtoeslag":  "zorgtoeslag",
    "bijstand":     "participatiewet_bijstand",
    "alcoholwet":   "alcoholwet_vergunning",
}


# ---------------------------------------------------------------------------
# 1. Inter-rater reliability
# ---------------------------------------------------------------------------

def compute_interrater(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kappa on categorical fields, Spearman+Kendall on ordinal fields."""
    results_kappa = []
    results_spearman = []

    for ann_type, group in df.groupby("annotator_type"):
        respondents = sorted(group["respondent"].unique())
        if len(respondents) < 2:
            print(f"  [{ann_type}] Need at least 2 respondents — skipping.")
            continue

        for i in range(len(respondents)):
            for j in range(i + 1, len(respondents)):
                r1, r2 = respondents[i], respondents[j]
                d1 = group[group["respondent"] == r1].sort_values(["wet", "uitleg_nr"])
                d2 = group[group["respondent"] == r2].sort_values(["wet", "uitleg_nr"])
                merged = d1.merge(d2, on=["wet", "uitleg_nr"], suffixes=("_r1", "_r2"))
                if len(merged) < 3:
                    continue

                # Categorical field for kappa
                kappa_field = "actionability" if ann_type == "burger" else "ontbrekende_informatie"
                c1 = f"{kappa_field}_r1"
                c2 = f"{kappa_field}_r2"
                if c1 in merged.columns and c2 in merged.columns:
                    valid = merged[c1].notna() & merged[c2].notna()
                    if valid.sum() >= 2:
                        cats_r1 = _float_to_cat(merged.loc[valid, c1])
                        cats_r2 = _float_to_cat(merged.loc[valid, c2])
                        try:
                            kappa = cohen_kappa_score(cats_r1, cats_r2)
                        except Exception:
                            kappa = None
                        results_kappa.append({
                            "annotator_type": ann_type,
                            "rater_1": r1, "rater_2": r2,
                            "metric": kappa_field,
                            "n_pairs": int(valid.sum()),
                            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
                            "interpretation": _kappa_label(kappa),
                        })

                # Ordinal fields for Spearman + Kendall
                if ann_type == "burger":
                    ordinal_fields = ["duidelijkheid", "leesbaarheid"]
                else:
                    ordinal_fields = ["juridische_correctheid"]

                for metric in ordinal_fields:
                    col1 = f"{metric}_r1"
                    col2 = f"{metric}_r2"
                    if col1 not in merged.columns or col2 not in merged.columns:
                        continue
                    valid_m = merged[col1].notna() & merged[col2].notna()
                    if valid_m.sum() < 3:
                        continue
                    rho, pval = stats.spearmanr(merged.loc[valid_m, col1], merged.loc[valid_m, col2])
                    tau, tpval = stats.kendalltau(merged.loc[valid_m, col1], merged.loc[valid_m, col2])
                    results_spearman.append({
                        "annotator_type": ann_type,
                        "rater_1": r1, "rater_2": r2,
                        "metric": metric,
                        "n_pairs": int(valid_m.sum()),
                        "kendall_tau": round(tau, 4),
                        "kendall_p": round(tpval, 4),
                        "kendall_sig_p05": tpval < 0.05,
                        "spearman_rho": round(rho, 4),
                        "spearman_p": round(pval, 4),
                        "spearman_sig_p05": pval < 0.05,
                    })

    return pd.DataFrame(results_kappa), pd.DataFrame(results_spearman)


def _float_to_cat(s: pd.Series) -> list[str]:
    mapping = {0.0: "Nee", 0.5: "Deels", 1.0: "Ja"}
    return [mapping.get(v, str(v)) for v in s]


def _kappa_label(k) -> str:
    if k is None:
        return "n/a"
    if k < 0:      return "No agreement"
    if k < 0.20:   return "Slight"
    if k < 0.40:   return "Fair"
    if k < 0.60:   return "Moderate"
    if k < 0.80:   return "Substantial"
    return "Almost perfect"


# ---------------------------------------------------------------------------
# 2. Automatic text metrics
# ---------------------------------------------------------------------------

def extract_auto_metrics(wet: str) -> pd.DataFrame:
    """Extract text-level metrics from explanation JSONL files."""
    models    = ["haiku", "llama3.1", "mistral", "deepseek", "gpt4"]
    approaches = ["graph", "open"]
    wet_slug  = LAW_SLUGS.get(wet, wet)
    rows = []

    idx = 1
    for approach in approaches:
        for model in models:
            fname = THESIS_DIR / model / f"{approach}_{model}_{wet_slug}.jsonl"
            if not fname.exists():
                continue
            with open(fname, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("record_type") != "explanation":
                        continue
                    text = r.get("explanation") or ""
                    rows.append({
                        "wet":            wet,
                        "uitleg_nr":      idx,
                        "approach":       approach,
                        "model":          model,
                        "profile_name":   r.get("profile_name", ""),
                        "requirements_met": r.get("requirements_met"),
                        **_text_metrics(text),
                    })
                    idx += 1
    return pd.DataFrame(rows)


def _text_metrics(text: str) -> dict:
    if not text:
        return {
            "char_count": 0, "word_count": 0, "sentence_count": 0,
            "avg_sentence_len": 0.0,
        }
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = len(words)
    sent_count = max(len(sentences), 1)
    return {
        "char_count":       len(text),
        "word_count":       word_count,
        "sentence_count":   sent_count,
        "avg_sentence_len": round(word_count / sent_count, 2),
    }


# ---------------------------------------------------------------------------
# 3. Automatic vs. human correlation
# ---------------------------------------------------------------------------

def auto_vs_human(auto_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    """Correlate automatic metrics with mean human scores (Pearson + Spearman + Kendall)."""
    # Burger: duidelijkheid_mean, actionability_mean, leesbaarheid_mean
    # Jurist: juridische_correctheid_mean, ontbrekende_informatie_mean
    burger_scores = scores_df[scores_df["annotator_type"] == "burger"]
    jurist_scores = scores_df[scores_df["annotator_type"] == "jurist"]

    rows = []
    auto_cols = ["char_count", "word_count", "sentence_count", "avg_sentence_len"]

    for human_type, hdf, human_cols in [
        ("burger", burger_scores, ["duidelijkheid_mean", "actionability_mean", "leesbaarheid_mean"]),
        ("jurist", jurist_scores, ["juridische_correctheid_mean", "ontbrekende_informatie_mean"]),
    ]:
        # scores_df has multi-level columns from groupby agg — flatten
        # Pick only the _mean columns
        mean_cols = [c for c in hdf.columns if c.endswith("_mean")]
        merge_on = ["wet", "uitleg_nr"]
        subset_cols = [c for c in ["wet", "uitleg_nr", "annotator_type"] + mean_cols if c in hdf.columns]
        merged = auto_df.merge(hdf[subset_cols], on=merge_on, how="inner")

        for ac in auto_cols:
            for hc in [c for c in human_cols if c in merged.columns]:
                valid = merged[[ac, hc]].dropna()
                if len(valid) < 3:
                    continue
                pr, pp = stats.pearsonr(valid[ac], valid[hc])
                sr, sp = stats.spearmanr(valid[ac], valid[hc])
                tr, tp = stats.kendalltau(valid[ac], valid[hc])
                rows.append({
                    "annotator_type": human_type,
                    "auto_metric":   ac,
                    "human_metric":  hc,
                    "n":             len(valid),
                    "pearson_r":     round(pr, 4),
                    "pearson_p":     round(pp, 4),
                    "pearson_sig":   pp < 0.05,
                    "spearman_rho":  round(sr, 4),
                    "spearman_p":    round(sp, 4),
                    "spearman_sig":  sp < 0.05,
                    "kendall_tau":   round(tr, 4),
                    "kendall_p":     round(tp, 4),
                    "kendall_sig":   tp < 0.05,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LONG_CSV.exists():
        print(f"Run parse_annotations.py first — {LONG_CSV} not found.")
        return

    df_long = pd.read_csv(LONG_CSV)
    df_scores = pd.read_csv(OUTPUT_DIR / "annotations_scores.csv")

    # 1. Inter-rater
    print("1. Inter-rater reliability")
    kappa_df, spearman_df = compute_interrater(df_long)
    if not kappa_df.empty:
        kappa_df.to_csv(OUTPUT_DIR / "interrater_kappa.csv", index=False)
        print(kappa_df.to_string(index=False))
    if not spearman_df.empty:
        spearman_df.to_csv(OUTPUT_DIR / "interrater_spearman.csv", index=False)
        print(spearman_df.to_string(index=False))

    # 2. Auto metrics
    print("\n2. Automatic text metrics")
    wetten = df_long["wet"].unique()
    all_auto = pd.concat([extract_auto_metrics(wet) for wet in wetten], ignore_index=True)
    all_auto.to_csv(OUTPUT_DIR / "auto_metrics.csv", index=False)
    print(all_auto[["approach", "model", "profile_name",
                    "word_count", "avg_sentence_len"]].to_string(index=False))

    # 3. Auto vs human
    print("\n3. Auto metrics vs. human scores (Pearson + Spearman + Kendall)")
    corr_df = auto_vs_human(all_auto, df_scores)
    if not corr_df.empty:
        corr_df.to_csv(OUTPUT_DIR / "auto_vs_human_correlation.csv", index=False)
        notable = corr_df[(corr_df["pearson_p"] < 0.15) | (corr_df["spearman_p"] < 0.15)]
        if notable.empty:
            print("  No notable correlations (p < 0.15).")
            print(corr_df[["annotator_type", "auto_metric", "human_metric",
                           "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                           "kendall_tau", "kendall_p"]].to_string(index=False))
        else:
            print(notable[["annotator_type", "auto_metric", "human_metric",
                            "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                            "kendall_tau", "kendall_p"]].to_string(index=False))

    print("\nDone. Output in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
