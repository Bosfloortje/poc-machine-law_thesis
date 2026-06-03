"""
Dimension 3: Citizen-focused evaluation.

Measures whether an LLM-generated explanation is understandable and actionable
for a non-expert citizen. Two sub-dimensions:

  1. Readability     — Flesch Reading Ease (Dutch), target >= 60
  2. Contestability  — decisive condition + counterfactual presence

Optionally compares against a gold annotation when provided.

This module is importable (for evaluate.py) and runnable standalone:

    uv run python analysis/llm_explanations/scripts/evaluation/dim3_citizen.py \
        --input output/20260408_.../*.jsonl
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------

def _flesch_nl(text: str) -> float | None:
    """Compute Flesch Reading Ease for Dutch text via textstat.

    Returns None if textstat is not installed or text is too short.
    Dutch formula: 206.835 - 0.93 * (syllables/words) * 100 - 1.015 * (words/sentences) * 100
    textstat handles this with set_lang('nl').
    """
    try:
        import textstat
        textstat.set_lang("nl")
        if len(text.split()) < 10:
            return None
        return round(textstat.flesch_reading_ease(text), 1)
    except ImportError:
        return None


def _avg_sentence_length(text: str) -> float:
    """Average words per sentence — proxy for readability when textstat unavailable."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return round(sum(word_counts) / len(word_counts), 1)


# ---------------------------------------------------------------------------
# Contestability (extended from explain/contestability.py)
# ---------------------------------------------------------------------------

def _contestability(text: str, decisive_condition: str = "") -> dict:
    """Score contestability with a slightly richer regex than the chat version."""
    has_decisive = (
        decisive_condition.lower() in text.lower()
    ) if decisive_condition else False

    # Counterfactual: als/indien/wanneer ... dan / zou u / heeft u recht
    has_counterfactual = bool(re.search(
        r"(als|indien|wanneer|zou|tenzij|behoudens).{5,120}(dan|zou u|heeft u recht|kunt u|in aanmerking)",
        text, re.IGNORECASE | re.DOTALL,
    ))

    # Action mention: does the text tell the citizen what to do?
    has_action = bool(re.search(
        r"(kunt u|kunt u contact|kunt u bezwaar|kunt u meer informatie|ga naar|bezoek|www\.|toeslagen\.nl|rijksoverheid\.nl)",
        text, re.IGNORECASE,
    ))

    score = sum([has_decisive, has_counterfactual, has_action]) / 3
    return {
        "has_decisive_condition": has_decisive,
        "has_counterfactual": has_counterfactual,
        "has_action_mention": has_action,
        "contestability_score": round(score, 2),
        "decisive_condition_checked": decisive_condition,
    }


# ---------------------------------------------------------------------------
# Gold comparison
# ---------------------------------------------------------------------------

def _compare_gold(scores: dict, gold: dict) -> dict:
    """
    Compare automatic scores against human gold annotations.

    Returns a dict of {field: {"auto": x, "gold": y, "match": bool}}.
    Only fields where gold has a non-None value are included.
    """
    comparisons: dict[str, Any] = {}
    human = gold.get("human_scores", {})

    # Counterfactual
    g_counterfactual = human.get("counterfactual_present")
    if g_counterfactual is not None:
        auto = scores["contestability"]["has_counterfactual"]
        comparisons["counterfactual"] = {
            "auto": auto,
            "gold": g_counterfactual,
            "match": auto == g_counterfactual,
        }

    # Overall quality proxy: flesch >= 60 → readable
    g_understandable = human.get("citizen_understandable")
    if g_understandable is not None and scores.get("flesch") is not None:
        auto_readable = scores["flesch"] >= 55
        comparisons["citizen_understandable"] = {
            "auto": auto_readable,
            "auto_flesch": scores["flesch"],
            "gold": g_understandable,
            "match": auto_readable == g_understandable,
        }

    return comparisons


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_citizen(
    explanation: str,
    decisive_condition: str = "",
    gold: dict | None = None,
) -> dict:
    """
    Compute all Dimension 3 scores for a single explanation.

    Args:
        explanation:        The LLM-generated explanation text.
        decisive_condition: The engine's decisive condition label (from evaluation_trace).
        gold:               Optional gold annotation dict (from YAML template).

    Returns:
        Dict with keys: flesch, avg_sentence_length, contestability,
        gold_comparison (if gold provided).
    """
    text = (explanation or "").strip()

    flesch = _flesch_nl(text)
    avg_sent = _avg_sentence_length(text)
    contestability = _contestability(text, decisive_condition)

    result: dict = {
        "flesch": flesch,                          # None if textstat unavailable
        "avg_sentence_length": avg_sent,
        "word_count": len(text.split()),
        "contestability": contestability,
    }

    if gold is not None:
        result["gold_comparison"] = _compare_gold(result, gold)

    return result


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dimension 3 citizen-focused evaluation.")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL file(s) from extract.py")
    parser.add_argument("--gold-dir", default=None,
                        help="Directory with gold YAML templates (evaluation/gold/)")
    parser.add_argument("--law", nargs="+", default=None, help="Filter to specific law(s)")
    args = parser.parse_args()

    gold_cache: dict[str, dict] = {}
    if args.gold_dir:
        try:
            import yaml
            for p in Path(args.gold_dir).glob("*.yaml"):
                with open(p, encoding="utf-8") as f:
                    g = yaml.safe_load(f)
                key = f"{g.get('law', '')}_{g.get('profile', '')}"
                gold_cache[key] = g
            print(f"Loaded {len(gold_cache)} gold templates from {args.gold_dir}")
        except ImportError:
            print("Warning: pyyaml not installed, gold annotations skipped")

    law_filter = set(args.law) if args.law else None

    all_scores: list[dict] = []

    for input_path in args.input:
        print(f"\n{Path(input_path).name}")
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
                if record.get("error") or not record.get("explanation"):
                    continue

                law = record.get("law", "")
                bsn = str(record.get("profile", ""))
                if law_filter and law not in law_filter:
                    continue

                trace = record.get("evaluation_trace", {})
                decisive = trace.get("decisive_condition", {}).get("label", "")
                gold = gold_cache.get(f"{law}_{bsn}")

                scores = score_citizen(record["explanation"], decisive, gold)

                name = record.get("profile_name", bsn)
                model = record.get("model", "?")
                f_str = f"{scores['flesch']:.0f}" if scores["flesch"] is not None else "n/a"
                c_score = scores["contestability"]["contestability_score"]
                decisive_y = "Y" if scores["contestability"]["has_decisive_condition"] else "N"
                counter_y  = "Y" if scores["contestability"]["has_counterfactual"] else "N"

                print(
                    f"  {name:<22} [{model:<12}]  "
                    f"flesch={f_str:<5}  contest={c_score:.2f}  decisive={decisive_y}  counter={counter_y}"
                )
                if gold and scores.get("gold_comparison"):
                    for field, comp in scores["gold_comparison"].items():
                        match_str = "OK" if comp["match"] else "MISMATCH"
                        print(f"    gold {field}: auto={comp['auto']} gold={comp['gold']} [{match_str}]")

                all_scores.append({
                    "law": law, "profile": bsn, "profile_name": name,
                    "model": model, "approach": record.get("approach", ""),
                    **scores,
                })

    if not all_scores:
        print("No explanation records found.")
        return

    # Summary
    n = len(all_scores)
    flesch_vals = [s["flesch"] for s in all_scores if s["flesch"] is not None]
    contest_vals = [s["contestability"]["contestability_score"] for s in all_scores]

    print(f"\n{'='*60}")
    print(f"Total explanations: {n}")
    if flesch_vals:
        print(f"Flesch (avg):       {sum(flesch_vals)/len(flesch_vals):.1f}  (target >= 60)")
        print(f"  >= 60 (readable): {sum(1 for v in flesch_vals if v >= 60)}/{len(flesch_vals)}")
    print(f"Contestability (avg): {sum(contest_vals)/n:.2f}")
    decisive_n = sum(1 for s in all_scores if s["contestability"]["has_decisive_condition"])
    counter_n  = sum(1 for s in all_scores if s["contestability"]["has_counterfactual"])
    print(f"  decisive condition: {decisive_n}/{n} ({decisive_n/n:.0%})")
    print(f"  counterfactual:     {counter_n}/{n} ({counter_n/n:.0%})")

    if gold_cache:
        # Gold agreement summary
        all_comps = [s["gold_comparison"] for s in all_scores if s.get("gold_comparison")]
        if all_comps:
            print(f"\nGold agreement ({len(all_comps)} records with gold):")
            for field in ["counterfactual", "citizen_understandable"]:
                matches = [c[field]["match"] for c in all_comps if field in c]
                if matches:
                    print(f"  {field}: {sum(matches)}/{len(matches)} match ({sum(matches)/len(matches):.0%})")


if __name__ == "__main__":
    main()
