#!/usr/bin/env python3
"""
Run contestability scoring on existing chat batch JSONL output.

Usage:
    uv run python analysis/llm_explanations/scripts/evaluation/score_contestability.py \
        analysis/llm_explanations/output/chat/20260408_143652_zorgtoeslag_llama3.2_graphrag_batch.jsonl

    # Multiple files:
    uv run python analysis/llm_explanations/scripts/evaluation/score_contestability.py \
        analysis/llm_explanations/output/chat/*.jsonl

    # Save annotated output:
    uv run python analysis/llm_explanations/scripts/evaluation/score_contestability.py \
        output.jsonl --save
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from explain.contestability import contestability_score


def score_file(path: Path, save: bool = False) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    conversations = [r for r in records if r.get("record_type") == "conversation"]
    results = []

    for conv in conversations:
        bsn = conv.get("bsn", "?")
        name = conv.get("profile_name", bsn)
        turns = conv.get("turns", [])
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]

        per_turn = []
        for i, turn in enumerate(assistant_turns):
            msg = turn.get("message", "")
            # Use saved decisive_condition from server if available, else empty
            saved = turn.get("contestability", {})
            decisive_condition = saved.get("decisive_condition_checked", "")
            score = contestability_score(msg, rac_trace={"decisive_condition": decisive_condition})
            per_turn.append({"turn": i + 1, **score})

        if not per_turn:
            continue

        # Conversation-level: criterion passes if met AT LEAST ONCE across all turns
        # Works regardless of number of turns (4, 8, 10, ...)
        checks = ["has_decisive_condition", "has_contestable_path"]
        conv_checks = {c: any(t[c] for t in per_turn) for c in checks}
        conv_score = round(sum(conv_checks.values()) / len(checks), 2)

        result = {
            "bsn": bsn,
            "profile_name": name,
            "conversation_score": conv_score,
            "conversation_checks": conv_checks,
            "turns": per_turn,
        }
        results.append(result)

        print(f"  {name:<25} score={conv_score:.2f}  "
              f"decisive={'Y' if conv_checks['has_decisive_condition'] else 'N'}  "
              f"counterfactual={'Y' if conv_checks['has_contestable_path'] else 'N'}")

    if save and results:
        out_path = path.with_suffix(".scored.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  → Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Score contestability on chat batch JSONL output")
    parser.add_argument("files", nargs="+", help="One or more .jsonl batch output files")
    parser.add_argument("--save", action="store_true", help="Save annotated scores as .scored.json next to input")
    args = parser.parse_args()

    all_results = []
    for pattern in args.files:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        for path in paths:
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            print(f"\n{path.name}")
            print("-" * 60)
            results = score_file(path, save=args.save)
            all_results.extend(results)

    if not all_results:
        return

    # Overall summary — conversation-level (pass = met at least once per conversation)
    n = len(all_results)
    all_scores = [r["conversation_score"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"Total conversations scored: {n}")
    print(f"Overall avg contestability: {sum(all_scores)/n:.2f}")
    checks = ["has_decisive_condition", "has_contestable_path"]
    for check in checks:
        passed = sum(1 for r in all_results if r["conversation_checks"][check])
        print(f"  {check}: {passed}/{n} conversations ({passed/n:.0%})")


if __name__ == "__main__":
    main()
