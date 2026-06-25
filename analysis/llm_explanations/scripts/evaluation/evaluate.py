#!/usr/bin/env python3
"""
evaluate.py — Main evaluation orchestrator.

Reads JSONL output from extract.py / extract_graph.py and runs:

  Dim 3: Citizen-focused  (readability, jargon, contestability)
          — always computed; maps to "makkelijkheid burgers" in annotation
  Dim 2: Faithfulness     (NLI-based sentence-level fact matching vs trace)
          — only for records with evaluation_trace (graph approach)
          — maps to "juridische aantoonbaarheid" in annotation
          — requires 'transformers' + mDeBERTa model

Each output record includes a `record_id` ({law}__{profile}__{model}__{approach})
for joining with external annotation results via correlate.py.

Open approach records (no evaluation_trace) are included and scored on Dim3;
Dim2 fields will be None for those records.

Usage:
    # Dim3 only (fast):
    uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
        --input analysis/llm_explanations/output/thesis_.../*.jsonl

    # Include Dim2 NLI faithfulness (slow, requires transformers):
    uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
        --input analysis/llm_explanations/output/thesis_.../*.jsonl \
        --dim2 \
        --output analysis/llm_explanations/output/eval_results.jsonl

    # Filter by law/model/approach:
    uv run ... --law zorgtoeslag --model gpt-4o --approach graph
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
EVAL_DIR     = Path(__file__).parent
EVAL_OUTPUT  = Path(__file__).parent.parent.parent / "output" / "evaluation_output"
EVAL_OUTPUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from dim3_citizen import score_citizen  # noqa: E402
from sklearn.metrics import roc_auc_score as _sklearn_roc_auc


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return round(float(_sklearn_roc_auc(labels, scores)), 3)

# ---------------------------------------------------------------------------
# Dim2: NLI faithfulness (optional — soft dependency)
# ---------------------------------------------------------------------------

def _try_load_dim2():
    """Import dim2_faithfulness if available. Returns (score_faithfulness, _get_nli_pipeline) or (None, None)."""
    try:
        from dim2_faithfulness import _get_nli_pipeline, score_faithfulness  # noqa: E402
        return score_faithfulness, _get_nli_pipeline
    except ImportError:
        return None, None


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
# Law name normalization (graph approach stores full internal paths)
# ---------------------------------------------------------------------------

_LAW_NORMALIZE: dict[str, str] = {
    "alcoholwet/vergunning": "alcoholwet",
    "participatiewet/bijstand": "bijstand",
    "zorgtoeslagwet": "zorgtoeslag",
}


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------

def evaluate_record(
    record: dict,
    gold: dict | None = None,
    score_faithfulness=None,
    nli_pipe=None,
) -> dict:
    """
    Evaluate a single explanation record.

    Returns a flat result dict containing all scores.
    record_id = {law}__{profile}__{model}__{approach} — use to join with annotation data.
    """
    explanation = record.get("explanation") or ""
    trace = record.get("evaluation_trace") or {}
    decisive = (trace.get("decisive_condition") or {}).get("label", "")

    law = _LAW_NORMALIZE.get(record.get("law", ""), record.get("law", ""))
    profile = str(record.get("profile", ""))
    model = record.get("model", "")
    approach = record.get("approach", "")

    # --- Dim 3: citizen (always) ---
    dim3 = score_citizen(explanation, decisive, gold)

    result: dict[str, Any] = {
        # Identity + join key for annotation
        "record_id": f"{law}__{profile}__{model}__{approach}",
        "law": law,
        "profile": profile,
        "profile_name": record.get("profile_name", ""),
        "model": model,
        "approach": approach,

        # Engine ground truth (from trace; None for open approach)
        "outcome": trace.get("outcome", ""),
        "amount_euro": trace.get("amount_euro"),
        "decisive_condition": decisive,

        # Dim3 — makkelijkheid burgers
        "d3_flesch": dim3.get("flesch"),
        "d3_avg_sentence_length": dim3.get("avg_sentence_length"),
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

    # --- Dim 2: faithfulness (optional; skipped when no trace, e.g. open approach) ---
    if score_faithfulness is not None and explanation and trace:
        try:
            dim2 = score_faithfulness(explanation, trace, nli_pipe=nli_pipe)
            result["d2_faithfulness"] = dim2.get("faithfulness_score")
            result["d2_n_claims"] = dim2.get("n_claims")
            result["d2_n_supported"] = dim2.get("n_supported")
            result["d2_nli_claim_scores"] = dim2.get("nli_claim_scores")
            result["d2_string_labels"] = dim2.get("string_labels")
        except Exception as e:
            result["d2_faithfulness"] = None
            result["d2_nli_claim_scores"] = None
            result["d2_string_labels"] = None
            result["d2_error"] = str(e)
    else:
        result["d2_faithfulness"] = None
        result["d2_n_claims"] = None
        result["d2_n_supported"] = None
        result["d2_nli_claim_scores"] = None
        result["d2_string_labels"] = None

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
    contest_vals = [r["d3_contestability"] for r in results if r.get("d3_contestability") is not None]
    decisive_n = sum(1 for r in results if r.get("d3_has_decisive"))
    counter_n = sum(1 for r in results if r.get("d3_has_counterfactual"))
    action_n = sum(1 for r in results if r.get("d3_has_action"))

    summary: dict[str, Any] = {
        "n": n,
        "d3": {
            "flesch_avg": _avg(flesch_vals),
            "flesch_n": len(flesch_vals),
            "flesch_readable_n": sum(1 for v in flesch_vals if v >= 60),
            "contestability_avg": _avg(contest_vals),
            "decisive_condition_n": decisive_n,
            "decisive_condition_pct": round(decisive_n / n, 3),
            "counterfactual_n": counter_n,
            "counterfactual_pct": round(counter_n / n, 3),
            "action_mention_n": action_n,
            "action_mention_pct": round(action_n / n, 3),
        },
    }

    # Dim2 faithfulness (graph only — open approach has no trace)
    d2_vals = [r["d2_faithfulness"] for r in results if r.get("d2_faithfulness") is not None]
    all_nli_scores: list[float] = []
    all_string_labels: list[int] = []
    for r in results:
        nli = r.get("d2_nli_claim_scores") or []
        lbls = r.get("d2_string_labels") or []
        if len(nli) == len(lbls):
            all_nli_scores.extend(nli)
            all_string_labels.extend(lbls)
    if d2_vals:
        summary["d2"] = {
            "faithfulness_avg": _avg(d2_vals),
            "n": len(d2_vals),
            "nli_auc": _roc_auc(all_string_labels, all_nli_scores),
            "note": "graph approach only (open approach has no evaluation_trace)",
        }

    # Gold agreement (kept for backwards compatibility)
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

    def _breakdown(group_key: str) -> dict[str, Any]:
        by_group: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_group[r.get(group_key, "unknown")].append(r)
        out: dict[str, Any] = {}
        for key, recs in by_group.items():
            f_vals = [r["d3_flesch"] for r in recs if r.get("d3_flesch") is not None]
            c_vals = [r["d3_contestability"] for r in recs if r.get("d3_contestability") is not None]
            faith_vals = [r["d2_faithfulness"] for r in recs if r.get("d2_faithfulness") is not None]
            grp_nli: list[float] = []
            grp_lbls: list[int] = []
            for r in recs:
                nli = r.get("d2_nli_claim_scores") or []
                lbls = r.get("d2_string_labels") or []
                if len(nli) == len(lbls):
                    grp_nli.extend(nli)
                    grp_lbls.extend(lbls)
            out[key] = {
                "n": len(recs),
                "flesch_avg": _avg(f_vals),
                "contestability_avg": _avg(c_vals),
                "faithfulness_avg": _avg(faith_vals),
                "nli_auc": _roc_auc(grp_lbls, grp_nli),
            }
        return out

    if len({r.get("model") for r in results}) > 1:
        summary["by_model"] = _breakdown("model")
    if len({r.get("law") for r in results}) > 1:
        summary["by_law"] = _breakdown("law")
    if len({r.get("approach") for r in results}) > 1:
        summary["by_approach"] = _breakdown("approach")

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

    d3 = summary.get("d3", {})
    if d3:
        print("\nDimension 3 — Citizen-focused")
        flesch_avg = d3.get("flesch_avg")
        if flesch_avg is not None:
            readable_n = d3.get("flesch_readable_n", 0)
            flesch_n = d3.get("flesch_n", 0)
            print(f"  Flesch (avg):          {flesch_avg:.1f}  (target >= 60)")
            print(f"  Readable (>= 60):      {readable_n}/{flesch_n}")
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
        print("\nDimension 2 — Faithfulness (NLI)")
        print(f"  Faithfulness (avg):    {d2['faithfulness_avg']:.2f}  (0–1, higher = better)")
        print(f"  Records scored:        {d2['n']}/{n}")
        if d2.get("nli_auc") is not None:
            print(f"  NLI ROC AUC:           {d2['nli_auc']:.3f}  (meta-eval: NLI prob vs string-match label)")

    gold = summary.get("gold_agreement")
    if gold:
        print("\nGold annotation agreement:")
        for field, stats in gold.items():
            print(f"  {field:<28} {stats['match_n']}/{stats['n']} ({stats['match_pct']:.0%})")

    def _print_breakdown(title: str, breakdown: dict) -> None:
        print(f"\n{title}")
        for key, stats in breakdown.items():
            f = f"{stats['flesch_avg']:.1f}" if stats.get("flesch_avg") is not None else "n/a"
            c = f"{stats['contestability_avg']:.2f}" if stats.get("contestability_avg") is not None else "n/a"
            faith = f"{stats['faithfulness_avg']:.2f}" if stats.get("faithfulness_avg") is not None else "n/a"
            auc = f"{stats['nli_auc']:.3f}" if stats.get("nli_auc") is not None else "n/a"
            print(f"  {key:<22} n={stats['n']:<5} flesch={f:<6} contest={c} faith={faith} auc={auc}")

    by_approach = summary.get("by_approach")
    if by_approach:
        _print_breakdown("Per-approach breakdown (open=no trace, faith=n/a):", by_approach)

    by_model = summary.get("by_model")
    if by_model:
        _print_breakdown("Per-model breakdown:", by_model)

    by_law = summary.get("by_law")
    if by_law:
        _print_breakdown("Per-law breakdown:", by_law)

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
                        help="JSONL file(s) from extract.py / extract_graph.py")
    parser.add_argument("--gold-dir", default=None,
                        help="Directory with gold YAML templates")
    parser.add_argument("--dim2", action="store_true",
                        help="Enable Dim2 NLI faithfulness scoring (requires transformers)")
    parser.add_argument("--dim2-string", action="store_true",
                        help="Enable Dim2 string-based faithfulness scoring (fast, no model required)")
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
    parser.add_argument("--enrich-open", action="store_true",
                        help="Inject evaluation_trace from matching graph record into open-approach records "
                             "(matched on law+profile). Enables Dim2 scoring for open approach.")
    args = parser.parse_args()

    # Default output paths: dedicated evaluation_output folder
    if not args.output:
        args.output = str(EVAL_OUTPUT / "eval_results.jsonl")
    if not args.summary_json:
        args.summary_json = str(EVAL_OUTPUT / "eval_summary.json")

    # Load optional gold cache
    gold_cache: dict[str, dict] = {}
    if args.gold_dir:
        gold_cache = load_gold_cache(args.gold_dir)
        if gold_cache:
            print(f"Loaded {len(gold_cache)} gold templates from {args.gold_dir}")

    # Load optional dim2
    score_faithfulness = None
    nli_pipe = None
    if args.dim2:
        score_faithfulness, get_nli_pipeline = _try_load_dim2()
        if score_faithfulness is None:
            print("Warning: dim2_faithfulness not available — Dim2 scoring skipped", file=sys.stderr)
        elif get_nli_pipeline is not None:
            nli_pipe = get_nli_pipeline()
    elif getattr(args, "dim2_string", False):
        score_faithfulness, _ = _try_load_dim2()
        if score_faithfulness is None:
            print("Warning: dim2_faithfulness not available — Dim2 scoring skipped", file=sys.stderr)
        # nli_pipe stays None → string-based only

    # Filters
    law_filter = set(args.law) if args.law else None
    model_filter = set(args.model) if args.model else None
    approach_filter = set(args.approach) if args.approach else None

    # Build graph trace index for open-approach enrichment
    # Key: (law, profile) — any graph record for that law+profile works since engine output is deterministic
    graph_trace_index: dict[tuple[str, str], dict] = {}
    if args.enrich_open:
        for input_path_str in args.input:
            with open(Path(input_path_str), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("record_type") != "explanation":
                        continue
                    if r.get("approach") != "graph":
                        continue
                    trace = r.get("evaluation_trace")
                    if not trace:
                        continue
                    law = _LAW_NORMALIZE.get(r.get("law", ""), r.get("law", ""))
                    profile = str(r.get("profile", ""))
                    key = (law, profile)
                    if key not in graph_trace_index:
                        graph_trace_index[key] = trace
        print(f"Enrich-open: indexed {len(graph_trace_index)} graph traces for open-approach injection")

    all_results: list[dict] = []
    n_skipped_no_trace = 0
    n_skipped_error = 0
    n_skipped_filter = 0

    output_fh = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fh = open(output_path, "w", encoding="utf-8")  # noqa: SIM115

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

                    # Inject graph trace into open records if --enrich-open
                    if args.enrich_open and record.get("approach") == "open" and not record.get("evaluation_trace"):
                        law_norm = _LAW_NORMALIZE.get(record.get("law", ""), record.get("law", ""))
                        profile_key = str(record.get("profile", ""))
                        injected = graph_trace_index.get((law_norm, profile_key))
                        if injected:
                            record = dict(record)
                            record["evaluation_trace"] = injected

                    # Records without trace (open approach) are kept — Dim3 still scores,
                    # Dim2 faithfulness will be None for those records.
                    if not record.get("evaluation_trace"):
                        n_skipped_no_trace += 1  # counted but not skipped

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
                        result = evaluate_record(record, gold=gold, score_faithfulness=score_faithfulness, nli_pipe=nli_pipe)
                    except Exception as e:
                        print(f"  Error evaluating {law}/{bsn}: {e}", file=sys.stderr)
                        continue

                    all_results.append(result)

                    # Live per-record output
                    name = result.get("profile_name") or bsn
                    f_str = f"{result['d3_flesch']:.0f}" if result.get("d3_flesch") is not None else "n/a"
                    c_str = f"{result['d3_contestability']:.2f}"
                    print(
                        f"  {name:<22} [{result['model']:<14}]  "
                        f"flesch={f_str:<5}  contest={c_str}"
                    )

                    if output_fh:
                        output_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    finally:
        if output_fh:
            output_fh.close()

    # Diagnostics
    if n_skipped_no_trace > 0:
        print(f"\nNote: {n_skipped_no_trace} records without evaluation_trace (open approach) — Dim2 is None for these")
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
