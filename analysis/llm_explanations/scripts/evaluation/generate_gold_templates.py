#!/usr/bin/env python3
"""
Generate gold annotation YAML templates from extract.py / extract_graphrag.py output.

For each explanation record that has an evaluation_trace, this script writes a
YAML template to evaluation/gold/{law}_{bsn}.yaml.  The template contains all
engine-derived facts pre-filled; you only need to fill in human_scores and any
corrections.

Usage:
    uv run python analysis/llm_explanations/scripts/evaluation/generate_gold_templates.py \
        --input analysis/llm_explanations/output/20260408_.../*.jsonl \
        --gold-dir analysis/llm_explanations/scripts/evaluation/gold \
        --overwrite          # re-generate existing templates

    # Filter to specific laws or profiles:
        --law zorgtoeslag
        --bsn 140592532 403987006
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _euro_str(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_template(record: dict) -> dict:
    """Build a gold annotation template from one JSONL explanation record."""
    trace = record.get("evaluation_trace", {})
    law = record.get("law", "")
    bsn = str(record.get("profile", ""))
    profile_name = record.get("profile_name", bsn)
    explanation = record.get("explanation", "") or ""

    # Required claims derived from trace
    required_claims = []

    # Outcome claim
    outcome = trace.get("outcome", "ONBEKEND")
    if outcome == "RECHT":
        required_claims.append({
            "id": "c_outcome",
            "type": "outcome",
            "description": "Uitleg vermeldt dat burger RECHT heeft",
            "pattern": "recht",  # case-insensitive substring check
        })
    elif outcome == "GEEN_RECHT":
        required_claims.append({
            "id": "c_outcome",
            "type": "outcome",
            "description": "Uitleg vermeldt dat burger GEEN RECHT heeft",
            "pattern": "geen recht",
        })

    # Amount claim
    amount_euro = trace.get("amount_euro")
    if amount_euro is not None and amount_euro > 0:
        required_claims.append({
            "id": "c_amount",
            "type": "amount",
            "description": f"Uitleg vermeldt het correcte bedrag",
            "value_euro": amount_euro,
            "formatted": _euro_str(amount_euro),
            "tolerance_euro": 0.02,  # rounding tolerance
        })

    # Key facts claims (income, vermogen, etc.)
    for field, info in trace.get("key_facts", {}).items():
        v_euro = info.get("value_euro")
        if v_euro is None:
            continue
        required_claims.append({
            "id": f"c_fact_{field.lower()}",
            "type": "fact",
            "description": f"Uitleg vermeldt {info.get('label', field)}",
            "field": field,
            "value_euro": v_euro,
            "formatted": _euro_str(v_euro),
            "tolerance_euro": 0.02,
        })

    # Decisive condition claim
    decisive = trace.get("decisive_condition", {})
    if decisive.get("label"):
        required_claims.append({
            "id": "c_decisive",
            "type": "condition",
            "description": "Uitleg noemt de doorslaggevende voorwaarde",
            "label": decisive["label"],
        })

    # Counterfactual
    required_claims.append({
        "id": "c_counterfactual",
        "type": "counterfactual",
        "description": "Uitleg bevat een 'als ... dan ...' zin over wat anders zou zijn",
        "required": True,
    })

    # Forbidden patterns — common error patterns
    forbidden_claims: list[str] = []
    if outcome == "RECHT":
        forbidden_claims.append("geen recht")
    if outcome == "GEEN_RECHT":
        forbidden_claims.append("heeft recht op")
    # Eurocent division error: if amount ~= X, forbid X/100
    if amount_euro and amount_euro > 100:
        wrong = round(amount_euro / 100, 2)
        forbidden_claims.append(_euro_str(wrong))

    return {
        "profile": bsn,
        "profile_name": profile_name,
        "law": law,
        "approach": record.get("approach", ""),
        "model": record.get("model", ""),
        "annotator": None,            # fill in: your name
        "date": None,                 # fill in: annotation date

        "engine_ground_truth": {
            "outcome": outcome,
            "requirements_met": trace.get("requirements_met"),
            "amount_euro": amount_euro,
            "decisive_condition": decisive.get("label", ""),
            "satisfied_conditions": trace.get("satisfied_conditions", []),
            "failed_conditions": trace.get("failed_conditions", []),
        },

        "required_claims": required_claims,
        "forbidden_claims": forbidden_claims,

        "human_scores": {
            # Fill these in after reading the explanation
            "overall_quality": None,           # 1–5
            "legally_correct": None,           # true / false
            "citizen_understandable": None,    # true / false
            "amount_correct": None,            # true / false / null (if no amount)
            "counterfactual_present": None,    # true / false
            "notes": "",
        },

        # The explanation text is included for convenience during annotation
        "_explanation_for_annotation": explanation[:800] + ("..." if len(explanation) > 800 else ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gold annotation YAML templates from extract output.")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL file(s) from extract.py / extract_graphrag.py")
    parser.add_argument("--gold-dir", default=str(Path(__file__).parent / "gold"),
                        help="Output directory for YAML templates")
    parser.add_argument("--law", nargs="+", default=None, help="Filter to specific law(s)")
    parser.add_argument("--bsn", nargs="+", default=None, help="Filter to specific BSN(s)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing templates")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    law_filter = set(args.law) if args.law else None
    bsn_filter = set(args.bsn) if args.bsn else None

    generated = skipped = no_trace = 0

    for input_path in args.input:
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
                if record.get("error"):
                    continue

                law = record.get("law", "")
                bsn = str(record.get("profile", ""))

                if law_filter and law not in law_filter:
                    continue
                if bsn_filter and bsn not in bsn_filter:
                    continue

                if not record.get("evaluation_trace"):
                    no_trace += 1
                    continue

                out_path = gold_dir / f"{law}_{bsn}.yaml"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                template = build_template(record)
                with open(out_path, "w", encoding="utf-8") as out_f:
                    yaml.dump(template, out_f, allow_unicode=True, default_flow_style=False,
                              sort_keys=False, width=100)

                generated += 1
                print(f"  Generated: {out_path.name}")

    print(f"\nDone. Generated: {generated} | Skipped (exists): {skipped} | No trace: {no_trace}")
    if no_trace > 0:
        print(f"  Tip: records without evaluation_trace were generated with an older extract.py — re-run extract to add traces.")


if __name__ == "__main__":
    main()
