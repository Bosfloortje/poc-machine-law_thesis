#!/usr/bin/env python3
"""
evaluate.py — Main evaluation orchestrator.

Reads JSONL output from extract.py / extract_graphrag.py and runs all
evaluation dimensions:

  Dim 3: Citizen-focused  (readability, jargon, contestability)
  Dim 2: Faithfulness     (NLI-based sentence-level fact matching vs trace)
                          — requires 'transformers' + mDeBERTa model

Optionally compares against human gold annotations (YAML templates generated
by generate_gold_templates.py).

Usage:
    # Quick run — Dim3 only, no gold:
    uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
        --input analysis/llm_explanations/output/20260408_.../*.jsonl

    # Full run with gold annotations:
    uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
        --input analysis/llm_explanations/output/20260408_.../*.jsonl \
        --gold-dir analysis/llm_explanations/scripts/evaluation/gold \
        --dim2 \
        --output analysis/llm_explanations/output/eval_results.jsonl

    # Filter:
    uv run ... --law zorgtoeslag --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from dim1_legal import score_legal  # noqa: E402
from dim3_citizen import score_citizen  # noqa: E402

# ---------------------------------------------------------------------------
# Dim2: NLI faithfulness (optional — soft dependency)
# ---------------------------------------------------------------------------

def _try_load_dim2():
    """Import dim2_faithfulness if available."""
    try:
        from dim2_faithfulness import score_faithfulness  # noqa: E402
        return score_faithfulness
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Gold annotation loading
# ---------------------------------------------------------------------------

def load_gold_cache(gold_dir: str | Path) -> dict[str, dict]:
    """Load all gold YAML files into a dict keyed by '{law}_{bsn}'."""
    gold_cache: dict[str, dict] = {}
    try:
        import yaml
    except ImportError:
        print("Warning: pyyaml not installed, gold annotations disabled", file=sys.stderr)
        return gold_cache

    gold_path = Path(gold_dir)
    if not gold_path.exists():
        print(f"Warning: gold-dir does not exist: {gold_dir}", file=sys.stderr)
        return gold_cache

    for p in gold_path.glob("*.yaml"):
        try:
            with open(p, encoding="utf-8") as f:
                g = yaml.safe_load(f)
            key = f"{g.get('law', '')}_{g.get('profile', '')}"
            gold_cache[key] = g
        except Exception as e:
            print(f"Warning: could not load {p.name}: {e}", file=sys.stderr)

    return gold_cache


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------

def evaluate_record(
    record: dict,
    gold: dict | None = None,
    score_faithfulness=None,
) -> dict:
    """
    Evaluate a single explanation record.

    Returns a flat result dict containing all scores.
    """
    explanation = record.get("explanation") or ""
    trace = record.get("evaluation_trace") or {}
    decisive = trace.get("decisive_condition", {}).get("label", "")

    # --- Dim 1: legal grounding ---
    dim1 = score_legal(explanation, trace)

    # --- Dim 3: citizen ---
    dim3 = score_citizen(explanation, decisive, gold)

    result: dict[str, Any] = {
        # Identity
        "law": record.get("law", ""),
        "profile": str(record.get("profile", "")),
        "profile_name": record.get("profile_name", ""),
        "model": record.get("model", ""),
        "approach": record.get("approach", ""),

        # Engine ground truth (from trace)
        "outcome": trace.get("outcome", ""),
        "amount_euro": trace.get("amount_euro"),
        "decisive_condition": decisive,

        # Dim1
        "d1_article_coverage": dim1.get("article_coverage"),
        "d1_n_available": dim1.get("n_available"),
        "d1_n_cited": dim1.get("n_cited"),

        # Dim3
        "d3_flesch": dim3.get("flesch"),
        "d3_avg_sentence_length": dim3.get("avg_sentence_length"),
        "d3_jargon_density": dim3.get("jargon_density"),
        "d3_word_count": dim3.get("word_count"),
        "d3_contestability": dim3["contestability"]["contestability_score"],
        "d3_has_decisive": dim3["contestability"]["has_decisive_condition"],
        "d3_has_counterfactual": dim3["contestability"]["has_counterfactual"],
        "d3_has_action": dim3["contestability"]["has_action_mention"],
    }

    # Gold comparison (Dim3)
    if gold and dim3.get("gold_comparison"):
        for field, comp in dim3["gold_comparison"].items():
            result[f"d3_gold_{field}_auto"] = comp.get("auto")
            result[f"d3_gold_{field}_gold"] = comp.get("gold")
            result[f"d3_gold_{field}_match"] = comp.get("match")

    # --- Dim 2: faithfulness (optional) ---
    if score_faithfulness is not None and explanation:
        try:
            dim2 = score_faithfulness(explanation, trace)
            result["d2_faithfulness"] = dim2.get("faithfulness_score")
            result["d2_n_claims"] = dim2.get("n_claims")
            result["d2_n_supported"] = dim2.get("n_supported")
        except Exception as e:
            result["d2_faithfulness"] = None
            result["d2_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _avg(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(results: list[dict]) -> dict:
    """Compute aggregate statistics across all evaluation results."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    flesch_vals = [r["d3_flesch"] for r in results if r.get("d3_flesch") is not None]
    jargon_vals = [r["d3_jargon_density"] for r in results if r.get("d3_jargon_density") is not None]
    contest_vals = [r["d3_contestability"] for r in results if r.get("d3_contestability") is not None]
    decisive_n = sum(1 for r in results if r.get("d3_has_decisive"))
    counter_n = sum(1 for r in results if r.get("d3_has_counterfactual"))
    action_n = sum(1 for r in results if r.get("d3_has_action"))

    cov_vals = [r["d1_article_coverage"] for r in results if r.get("d1_article_coverage") is not None]
    full_cov_n = sum(1 for v in cov_vals if v >= 1.0)

    summary: dict[str, Any] = {
        "n": n,
        "d1": {
            "article_coverage_avg": _avg(cov_vals),
            "n": len(cov_vals),
            "full_coverage_n": full_cov_n,
            "full_coverage_pct": round(full_cov_n / len(cov_vals), 3) if cov_vals else None,
        },
        "d3": {
            "flesch_avg": _avg(flesch_vals),
            "flesch_n": len(flesch_vals),
            "flesch_readable_n": sum(1 for v in flesch_vals if v >= 60),
            "jargon_avg": _avg(jargon_vals),
            "contestability_avg": _avg(contest_vals),
            "decisive_condition_n": decisive_n,
            "decisive_condition_pct": round(decisive_n / n, 3),
            "counterfactual_n": counter_n,
            "counterfactual_pct": round(counter_n / n, 3),
            "action_mention_n": action_n,
            "action_mention_pct": round(action_n / n, 3),
        },
    }

    # Dim2 if available
    d2_vals = [r["d2_faithfulness"] for r in results if r.get("d2_faithfulness") is not None]
    if d2_vals:
        summary["d2"] = {"faithfulness_avg": _avg(d2_vals), "n": len(d2_vals)}

    # Gold agreement
    gold_fields = set()
    for r in results:
        for k in r:
            if k.startswith("d3_gold_") and k.endswith("_match"):
                gold_fields.add(k[len("d3_gold_"):-len("_match")])

    if gold_fields:
        gold_agreement: dict[str, Any] = {}
        for field in sorted(gold_fields):
            matches = [r[f"d3_gold_{field}_match"] for r in results if f"d3_gold_{field}_match" in r]
            if matches:
                gold_agreement[field] = {
                    "n": len(matches),
                    "match_n": sum(matches),
                    "match_pct": round(sum(matches) / len(matches), 3),
                }
        summary["gold_agreement"] = gold_agreement

    # Per-model breakdown
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r.get("model", "unknown")].append(r)

    if len(by_model) > 1:
        model_summary: dict[str, Any] = {}
        for model, recs in by_model.items():
            f_vals = [r["d3_flesch"] for r in recs if r.get("d3_flesch") is not None]
            c_vals = [r["d3_contestability"] for r in recs if r.get("d3_contestability") is not None]
            j_vals = [r["d3_jargon_density"] for r in recs if r.get("d3_jargon_density") is not None]
            model_summary[model] = {
                "n": len(recs),
                "flesch_avg": _avg(f_vals),
                "contestability_avg": _avg(c_vals),
                "jargon_avg": _avg(j_vals),
            }
        summary["by_model"] = model_summary

    # Per-law breakdown
    by_law: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_law[r.get("law", "unknown")].append(r)

    if len(by_law) > 1:
        law_summary: dict[str, Any] = {}
        for law, recs in by_law.items():
            f_vals = [r["d3_flesch"] for r in recs if r.get("d3_flesch") is not None]
            c_vals = [r["d3_contestability"] for r in recs if r.get("d3_contestability") is not None]
            law_summary[law] = {
                "n": len(recs),
                "flesch_avg": _avg(f_vals),
                "contestability_avg": _avg(c_vals),
            }
        summary["by_law"] = law_summary

    return summary


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    n = summary["n"]
    if n == 0:
        print("No explanation records found.")
        return

    print(f"\n{'='*65}")
    print(f"Total explanations evaluated: {n}")

    d1 = summary.get("d1", {})
    if d1 and d1.get("n", 0) > 0:
        print(f"\nDimension 1 — Legal grounding")
        cov_avg = d1.get("article_coverage_avg")
        if cov_avg is not None:
            print(f"  Article coverage (avg): {cov_avg:.2f}  (0–1, higher = better)")
        full_n = d1.get("full_coverage_n", 0)
        print(f"  Full coverage (1.0):    {full_n}/{d1['n']}")

    d3 = summary.get("d3", {})
    if d3:
        print(f"\nDimension 3 — Citizen-focused")
        flesch_avg = d3.get("flesch_avg")
        if flesch_avg is not None:
            readable_n = d3.get("flesch_readable_n", 0)
            flesch_n = d3.get("flesch_n", 0)
            print(f"  Flesch (avg):          {flesch_avg:.1f}  (target >= 60)")
            print(f"  Readable (>= 60):      {readable_n}/{flesch_n}")
        jargon_avg = d3.get("jargon_avg")
        if jargon_avg is not None:
            print(f"  Jargon density (avg):  {jargon_avg:.4f}  (lower = better)")
        c_avg = d3.get("contestability_avg")
        if c_avg is not None:
            print(f"  Contestability (avg):  {c_avg:.2f}  (0–1, higher = better)")
        dec_n = d3.get("decisive_condition_n", 0)
        dec_pct = d3.get("decisive_condition_pct", 0)
        ctr_n = d3.get("counterfactual_n", 0)
        ctr_pct = d3.get("counterfactual_pct", 0)
        act_n = d3.get("action_mention_n", 0)
        act_pct = d3.get("action_mention_pct", 0)
        print(f"  Decisive condition:    {dec_n}/{n} ({dec_pct:.0%})")
        print(f"  Counterfactual:        {ctr_n}/{n} ({ctr_pct:.0%})")
        print(f"  Action mention:        {act_n}/{n} ({act_pct:.0%})")

    d2 = summary.get("d2")
    if d2:
        print(f"\nDimension 2 — Faithfulness (NLI)")
        print(f"  Faithfulness (avg):    {d2['faithfulness_avg']:.2f}  (0–1, higher = better)")
        print(f"  Records scored:        {d2['n']}/{n}")

    gold = summary.get("gold_agreement")
    if gold:
        print(f"\nGold annotation agreement:")
        for field, stats in gold.items():
            print(f"  {field:<28} {stats['match_n']}/{stats['n']} ({stats['match_pct']:.0%})")

    by_model = summary.get("by_model")
    if by_model:
        print(f"\nPer-model breakdown:")
        for model, stats in by_model.items():
            f = f"{stats['flesch_avg']:.1f}" if stats.get("flesch_avg") is not None else "n/a"
            c = f"{stats['contestability_avg']:.2f}" if stats.get("contestability_avg") is not None else "n/a"
            j = f"{stats['jargon_avg']:.4f}" if stats.get("jargon_avg") is not None else "n/a"
            print(f"  {model:<20} n={stats['n']:<4}  flesch={f:<6}  contest={c}  jargon={j}")

    by_law = summary.get("by_law")
    if by_law:
        print(f"\nPer-law breakdown:")
        for law, stats in by_law.items():
            f = f"{stats['flesch_avg']:.1f}" if stats.get("flesch_avg") is not None else "n/a"
            c = f"{stats['contestability_avg']:.2f}" if stats.get("contestability_avg") is not None else "n/a"
            print(f"  {law:<28} n={stats['n']:<4}  flesch={f:<6}  contest={c}")

    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM explanations across all dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="JSONL file(s) from extract.py / extract_graphrag.py")
    parser.add_argument("--gold-dir", default=None,
                        help="Directory with gold YAML templates")
    parser.add_argument("--dim2", action="store_true",
                        help="Enable Dim2 NLI faithfulness scoring (requires transformers)")
    parser.add_argument("--output", default=None,
                        help="Write per-record results to this JSONL file")
    parser.add_argument("--summary-json", default=None,
                        help="Write summary statistics to this JSON file")
    parser.add_argument("--law", nargs="+", default=None,
                        help="Filter to specific law(s)")
    parser.add_argument("--model", nargs="+", default=None,
                        help="Filter to specific model(s)")
    parser.add_argument("--approach", nargs="+", default=None,
                        help="Filter to specific approach(es)")
    parser.add_argument("--no-trace-skip", action="store_true",
                        help="Include records without evaluation_trace (limited scoring)")
    args = parser.parse_args()

    # Default output paths: next to the first input file
    first_input_dir = Path(args.input[0]).parent
    if not args.output:
        args.output = str(first_input_dir / "eval_results.jsonl")
    if not args.summary_json:
        args.summary_json = str(first_input_dir / "eval_summary.json")

    # Load optional gold cache
    gold_cache: dict[str, dict] = {}
    if args.gold_dir:
        gold_cache = load_gold_cache(args.gold_dir)
        if gold_cache:
            print(f"Loaded {len(gold_cache)} gold templates from {args.gold_dir}")

    # Load optional dim2
    score_faithfulness = None
    if args.dim2:
        score_faithfulness = _try_load_dim2()
        if score_faithfulness is None:
            print("Warning: dim2_faithfulness not available — Dim2 scoring skipped", file=sys.stderr)

    # Filters
    law_filter = set(args.law) if args.law else None
    model_filter = set(args.model) if args.model else None
    approach_filter = set(args.approach) if args.approach else None

    all_results: list[dict] = []
    n_skipped_no_trace = 0
    n_skipped_error = 0
    n_skipped_filter = 0

    output_fh = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fh = open(output_path, "w", encoding="utf-8")

    try:
        for input_path_str in args.input:
            input_path = Path(input_path_str)
            print(f"\nProcessing: {input_path.name}")
            print("-" * 60)

            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if record.get("record_type") != "explanation":
                        continue

                    # Skip error records (no explanation)
                    if record.get("error") or not record.get("explanation"):
                        n_skipped_error += 1
                        continue

                    # Skip records without trace (unless --no-trace-skip)
                    if not record.get("evaluation_trace") and not args.no_trace_skip:
                        n_skipped_no_trace += 1
                        continue

                    # Apply filters
                    law = record.get("law", "")
                    model = record.get("model", "")
                    approach = record.get("approach", "")

                    if law_filter and law not in law_filter:
                        n_skipped_filter += 1
                        continue
                    if model_filter and model not in model_filter:
                        n_skipped_filter += 1
                        continue
                    if approach_filter and approach not in approach_filter:
                        n_skipped_filter += 1
                        continue

                    # Gold lookup
                    bsn = str(record.get("profile", ""))
                    gold = gold_cache.get(f"{law}_{bsn}")

                    # Evaluate
                    try:
                        result = evaluate_record(record, gold=gold, score_faithfulness=score_faithfulness)
                    except Exception as e:
                        print(f"  Error evaluating {law}/{bsn}: {e}", file=sys.stderr)
                        continue

                    all_results.append(result)

                    # Live per-record output
                    name = result.get("profile_name") or bsn
                    f_str = f"{result['d3_flesch']:.0f}" if result.get("d3_flesch") is not None else "n/a"
                    c_str = f"{result['d3_contestability']:.2f}"
                    j_str = f"{result['d3_jargon_density']:.3f}"
                    print(
                        f"  {name:<22} [{result['model']:<14}]  "
                        f"flesch={f_str:<5}  jargon={j_str}  contest={c_str}"
                    )

                    if output_fh:
                        output_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    finally:
        if output_fh:
            output_fh.close()

    # Diagnostics
    if n_skipped_no_trace > 0:
        print(f"\nNote: skipped {n_skipped_no_trace} records without evaluation_trace")
        print("  (re-run extract.py to add traces, or use --no-trace-skip)")
    if n_skipped_error > 0:
        print(f"Note: skipped {n_skipped_error} error/empty records")
    if n_skipped_filter > 0:
        print(f"Note: filtered out {n_skipped_filter} records")

    if not all_results:
        print("\nNo records to evaluate.")
        return

    # Summary
    summary = summarize(all_results)
    print_summary(summary)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSummary written to: {summary_path}")

    if args.output:
        print(f"Per-record results written to: {args.output}")


if __name__ == "__main__":
    main()
