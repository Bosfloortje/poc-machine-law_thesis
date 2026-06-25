"""
correlate.py — Koppel automatische evaluatiescores aan menselijke annotaties.

Joins eval_results.jsonl (from evaluate.py) with an annotation CSV on record_id,
then computes Pearson/Spearman correlations between:

  Automatic Dim3 scores  ↔  "makkelijkheid_burgers"  (human citizen score)
  Automatic Dim2 scores  ↔  "juridische_aantoonbaarheid"  (human legal score)

Annotation CSV format (one row per annotated explanation):
    record_id,makkelijkheid_burgers,juridische_aantoonbaarheid
    zorgtoeslag__312847291__haiku__graph,4,3
    ...

  record_id must match: {law}__{profile}__{model}__{approach}
  Scores are expected as integers (e.g. 1–5 Likert scale).

Usage:
    uv run python analysis/llm_explanations/scripts/evaluation/correlate.py \\
        --eval  analysis/llm_explanations/output/thesis_.../eval_results.jsonl \\
        --annot annotation_results.csv \\
        --output correlation_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score as _sklearn_roc_auc


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return round(float(_sklearn_roc_auc(labels, scores)), 3)

# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 3)


def _rank(values: list[float]) -> list[float]:
    sorted_vals = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    return _pearson(_rank(x), _rank(y))


def _kendall(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    nc = nd = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                nc += 1
            elif prod < 0:
                nd += 1
    denom = n * (n - 1) / 2
    return round((nc - nd) / denom, 3) if denom else None


def _corr_stats(auto: list[float], human: list[float], label: str) -> dict:
    return {
        "label": label,
        "n": len(auto),
        "pearson": _pearson(auto, human),
        "spearman": _spearman(auto, human),
        "kendall_tau": _kendall(auto, human),
        "auto_avg": round(sum(auto) / len(auto), 3) if auto else None,
        "human_avg": round(sum(human) / len(human), 3) if human else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_EVAL_OUTPUT = Path(__file__).parent.parent.parent / "output" / "evaluation_output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate automatic scores with human annotations.")
    parser.add_argument("--eval", default=str(_EVAL_OUTPUT / "eval_results.jsonl"),
                        help="eval_results.jsonl from evaluate.py (default: evaluation_output/eval_results.jsonl)")
    parser.add_argument("--annot", required=True, help="Annotation CSV (record_id, makkelijkheid_burgers, juridische_aantoonbaarheid)")
    parser.add_argument("--output", default=str(_EVAL_OUTPUT / "correlation_report.json"),
                        help="Write correlation report to JSON file (default: evaluation_output/correlation_report.json)")
    parser.add_argument("--by-model", action="store_true", help="Also break down correlations per model")
    parser.add_argument("--by-law", action="store_true", help="Also break down correlations per law")
    parser.add_argument("--by-approach", action="store_true", help="Also break down correlations per approach")
    args = parser.parse_args()
    _EVAL_OUTPUT.mkdir(exist_ok=True)

    # Load evaluation results
    eval_by_id: dict[str, dict] = {}
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("record_id")
            if rid:
                eval_by_id[rid] = r

    print(f"Loaded {len(eval_by_id)} eval records from {args.eval}")

    # Load annotation CSV
    annot_by_id: dict[str, dict] = {}
    with open(args.annot, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("record_id", "").strip()
            if rid:
                annot_by_id[rid] = row

    print(f"Loaded {len(annot_by_id)} annotation rows from {args.annot}")

    # Join
    joined: list[dict] = []
    n_missing_eval = 0
    for rid, annot in annot_by_id.items():
        if rid not in eval_by_id:
            n_missing_eval += 1
            continue
        ev = eval_by_id[rid]
        try:
            citizen_score = float(annot["makkelijkheid_burgers"])
            legal_score_str = annot.get("juridische_aantoonbaarheid", "").strip()
            legal_score = float(legal_score_str) if legal_score_str else None
        except (ValueError, KeyError):
            continue
        joined.append({
            "record_id": rid,
            "law": ev.get("law", ""),
            "model": ev.get("model", ""),
            "approach": ev.get("approach", ""),
            "human_citizen": citizen_score,
            "human_legal": legal_score,
            # Dim3 (citizen)
            "d3_flesch": ev.get("d3_flesch"),
            "d3_contestability": ev.get("d3_contestability"),
            "d3_word_count": ev.get("d3_word_count"),
            # Dim2 (faithfulness — only graph approach)
            "d2_faithfulness": ev.get("d2_faithfulness"),
            "d2_nli_claim_scores": ev.get("d2_nli_claim_scores"),
            "d2_string_labels": ev.get("d2_string_labels"),
        })

    if n_missing_eval:
        print(f"Warning: {n_missing_eval} annotation rows had no matching eval record", file=sys.stderr)

    n = len(joined)
    if n == 0:
        print("No matched records found — check that record_id values align.")
        return

    print(f"Joined {n} records for correlation analysis\n")

    # ---------------------------------------------------------------------------
    # Compute correlations: automatic → human_citizen (Dim3 → makkelijkheid)
    # ---------------------------------------------------------------------------
    def _citizen_corrs(rows: list[dict]) -> list[dict]:
        corrs = []
        for auto_field, label in [
            ("d3_flesch", "Flesch ↔ makkelijkheid_burgers"),
            ("d3_contestability", "Contestability ↔ makkelijkheid_burgers"),
        ]:
            pairs = [(r[auto_field], r["human_citizen"]) for r in rows
                     if r.get(auto_field) is not None]
            if pairs:
                auto_vals, hum_vals = zip(*pairs)
                corrs.append(_corr_stats(list(auto_vals), list(hum_vals), label))
        return corrs

    def _legal_corrs(rows: list[dict]) -> list[dict]:
        corrs: list[dict] = []
        legal_rows = [r for r in rows if r.get("d2_faithfulness") is not None and r.get("human_legal") is not None]
        if legal_rows:
            auto_vals = [r["d2_faithfulness"] for r in legal_rows]
            hum_vals = [r["human_legal"] for r in legal_rows]
            corrs.append(_corr_stats(auto_vals, hum_vals, "Faithfulness ↔ juridische_aantoonbaarheid"))
        # ROC AUC: NLI entailment prob vs string-match binary label (no human annotations needed)
        all_nli: list[float] = []
        all_lbls: list[int] = []
        for r in rows:
            nli = r.get("d2_nli_claim_scores") or []
            lbls = r.get("d2_string_labels") or []
            if len(nli) == len(lbls):
                all_nli.extend(nli)
                all_lbls.extend(lbls)
        auc = _roc_auc(all_lbls, all_nli)
        if auc is not None:
            corrs.append({
                "label": "NLI ROC AUC (NLI prob vs string-match ground truth)",
                "n": len(all_nli),
                "auc": auc,
                "pearson": None,
                "spearman": None,
                "kendall_tau": None,
                "auto_avg": round(sum(all_nli) / len(all_nli), 3) if all_nli else None,
                "human_avg": None,
            })
        return corrs

    report: dict[str, Any] = {
        "n_joined": n,
        "overall": {
            "citizen": _citizen_corrs(joined),
            "legal": _legal_corrs(joined),
        },
    }

    # ---------------------------------------------------------------------------
    # Optional breakdowns
    # ---------------------------------------------------------------------------
    def _breakdown(group_key: str) -> dict:
        groups: dict[str, list] = {}
        for r in joined:
            groups.setdefault(r.get(group_key, "unknown"), []).append(r)
        return {
            key: {"citizen": _citizen_corrs(rows), "legal": _legal_corrs(rows)}
            for key, rows in groups.items()
        }

    if args.by_model:
        report["by_model"] = _breakdown("model")
    if args.by_law:
        report["by_law"] = _breakdown("law")
    if args.by_approach:
        report["by_approach"] = _breakdown("approach")

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print("=" * 65)
    print("Overall correlations")
    print("-" * 65)
    for c in report["overall"]["citizen"] + report["overall"]["legal"]:
        print(f"  {c['label']}")
        if c.get("auc") is not None:
            print(f"    n={c['n']}  ROC AUC={c['auc']:.3f}")
        else:
            p = f"{c['pearson']:.3f}" if c["pearson"] is not None else "n/a"
            s = f"{c['spearman']:.3f}" if c["spearman"] is not None else "n/a"
            t = f"{c['kendall_tau']:.3f}" if c.get("kendall_tau") is not None else "n/a"
            print(f"    n={c['n']}  Pearson={p}  Spearman={s}  Kendall={t}")

    for group_key in ("by_model", "by_law", "by_approach"):
        if group_key in report:
            print(f"\n{group_key.replace('_', ' ').title()}:")
            for key, corrs in report[group_key].items():
                all_corrs = corrs["citizen"] + corrs["legal"]
                if all_corrs:
                    print(f"  [{key}]")
                    for c in all_corrs:
                        if c.get("auc") is not None:
                            print(f"    {c['label'][:50]:<50}  AUC={c['auc']:.3f}")
                        else:
                            p = f"{c['pearson']:.3f}" if c["pearson"] is not None else "n/a"
                            s = f"{c['spearman']:.3f}" if c["spearman"] is not None else "n/a"
                            t = f"{c['kendall_tau']:.3f}" if c.get("kendall_tau") is not None else "n/a"
                            print(f"    {c['label'][:50]:<50}  P={p}  S={s}  K={t}")

    print("=" * 65)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
