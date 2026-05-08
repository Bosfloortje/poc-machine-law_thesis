"""
Dimension 2: Faithfulness evaluation.

Scores whether an LLM-generated explanation faithfully reflects the
evaluation_trace (engine ground truth) using sentence-level NLI.

Approach:
  For each factual claim derived from the trace (outcome, amount, decisive
  condition, key facts), the model's explanation is checked sentence-by-
  sentence for entailment.  The faithfulness score = supported_claims /
  total_claims.

Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual, Dutch-capable)
Requires: pip install transformers torch  (or: uv add transformers torch)

This module is importable (for evaluate.py) and runnable standalone:

    uv run python analysis/llm_explanations/scripts/evaluation/dim2_faithfulness.py \
        --input output/20260408_.../*.jsonl
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
_nli_pipeline = None   # lazy-loaded singleton


def _get_nli_pipeline():
    """Lazy-load the NLI pipeline (downloads model on first call)."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline

    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required for Dim2 scoring.\n"
            "Install with: uv add transformers torch"
        ) from e

    print(f"Loading NLI model: {_NLI_MODEL_NAME} (first call may download ~500 MB)")
    _nli_pipeline = pipeline(
        "zero-shot-classification",
        model=_NLI_MODEL_NAME,
        device=-1,  # CPU; change to 0 for GPU
    )
    return _nli_pipeline


# ---------------------------------------------------------------------------
# Claim extraction from evaluation_trace
# ---------------------------------------------------------------------------

def _euro_variants(amount_euro: float) -> list[str]:
    """Generate plausible string variants of a euro amount for matching."""
    variants = []
    # e.g. 1234.56 → "1.234,56", "1234,56", "1234.56"
    s_dot = f"{amount_euro:.2f}"
    s_comma = s_dot.replace(".", ",")
    s_dutch = f"{amount_euro:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    variants.extend([s_dot, s_comma, s_dutch])
    # Also integer variant if .00
    if amount_euro == int(amount_euro):
        variants.append(str(int(amount_euro)))
    return list(dict.fromkeys(variants))  # deduplicate, preserve order


def extract_claims(trace: dict) -> list[dict]:
    """
    Derive a list of verifiable claims from the evaluation_trace.

    Each claim is a dict with:
        id:          unique identifier
        type:        outcome | amount | fact | condition
        premise:     short declarative statement (Dutch) to check against explanation
        required:    bool — if True, absence counts as a faithfulness failure
    """
    claims: list[dict] = []

    outcome = trace.get("outcome", "")
    if outcome:
        up = outcome.upper()
        is_positive = ("RECHT" in up or "VERLEEND" in up or "TOEGEKEND" in up) and "GEEN" not in up
        is_negative = "GEEN" in up or "WEIGER" in up or "AFGEWEZEN" in up
        if is_positive:
            claims.append({
                "id": "outcome",
                "type": "outcome",
                "premise": "De aanvraag is gehonoreerd.",
                "required": True,
            })
        elif is_negative:
            claims.append({
                "id": "outcome",
                "type": "outcome",
                "premise": "De aanvraag is afgewezen.",
                "required": True,
            })

    amount = trace.get("amount_euro")
    if amount is not None and amount > 0:
        variants = _euro_variants(amount)
        # Pick the Dutch-formatted variant as premise text
        dutch = variants[2] if len(variants) > 2 else variants[0]
        claims.append({
            "id": "amount",
            "type": "amount",
            "premise": f"De burger ontvangt een bedrag van €{dutch}.",
            "amount_variants": variants,
            "required": True,
        })

    decisive = trace.get("decisive_condition", {})
    decisive_label = decisive.get("label", "")
    if decisive_label:
        claims.append({
            "id": "decisive",
            "type": "condition",
            "premise": f"De doorslaggevende voorwaarde is: {decisive_label}.",
            "required": False,
        })

    for field, info in trace.get("key_facts", {}).items():
        v_euro = info.get("value_euro")
        label = info.get("label", field)
        if v_euro is not None:
            variants = _euro_variants(v_euro)
            dutch = variants[2] if len(variants) > 2 else variants[0]
            claims.append({
                "id": f"fact_{field.lower()}",
                "type": "fact",
                "premise": f"{label} bedraagt €{dutch}.",
                "amount_variants": variants,
                "required": False,
            })

    return claims


# ---------------------------------------------------------------------------
# NLI-based sentence scoring
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _amount_present(text: str, variants: list[str]) -> bool:
    """Check if any of the amount variants appears in the text."""
    text_lower = text.lower()
    return any(v in text_lower for v in variants)


def _claim_supported_nli(
    explanation: str,
    claim: dict,
    pipeline,
    entailment_threshold: float = 0.5,
) -> dict:
    """
    Check if a claim is supported by any sentence in the explanation.

    For amount claims: first do a fast string check; NLI only if present.
    For other claims: NLI on up to 5 most relevant sentences.

    Returns a dict with: supported (bool), method, score (float).
    """
    premise = claim["premise"]

    # Fast path for amount claims: check string presence
    if claim["type"] == "amount" and claim.get("amount_variants"):
        if not _amount_present(explanation, claim["amount_variants"]):
            return {"supported": False, "method": "string", "score": 0.0}
        # Amount string found — mark as supported without NLI
        return {"supported": True, "method": "string", "score": 1.0}

    # Fast path for fact claims: check string presence first
    if claim["type"] == "fact" and claim.get("amount_variants"):
        if _amount_present(explanation, claim["amount_variants"]):
            return {"supported": True, "method": "string", "score": 1.0}
        return {"supported": False, "method": "string", "score": 0.0}

    # NLI path for outcome / condition claims
    sentences = _split_sentences(explanation)
    if not sentences:
        return {"supported": False, "method": "nli", "score": 0.0}

    # Run NLI: hypothesis is the premise, each explanation sentence is the "sequence"
    # We use zero-shot-classification: labels = ["entailment", "neutral", "contradiction"]
    best_score = 0.0
    for sentence in sentences[:10]:  # cap at 10 sentences for performance
        try:
            result = pipeline(
                sentence,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template=f"{{}} {premise}",
                multi_label=False,
            )
            labels = result["labels"]
            scores = result["scores"]
            entail_idx = labels.index("entailment") if "entailment" in labels else -1
            if entail_idx >= 0:
                best_score = max(best_score, scores[entail_idx])
        except Exception:
            continue

    return {
        "supported": best_score >= entailment_threshold,
        "method": "nli",
        "score": round(best_score, 3),
    }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_faithfulness(
    explanation: str,
    trace: dict,
    entailment_threshold: float = 0.5,
) -> dict:
    """
    Compute Dimension 2 faithfulness score for a single explanation.

    Args:
        explanation:           The LLM-generated explanation text.
        trace:                 The evaluation_trace dict from extract.py output.
        entailment_threshold:  Min entailment score to count as "supported".

    Returns:
        Dict with:
            faithfulness_score: float 0–1 (supported_required / total_required)
            n_claims:           total claims checked
            n_supported:        number of supported claims
            n_required:         required claims count
            n_required_supported: required supported count
            claims:             list of per-claim results
    """
    text = (explanation or "").strip()
    claims = extract_claims(trace)

    if not claims:
        return {
            "faithfulness_score": None,
            "n_claims": 0,
            "n_supported": 0,
            "n_required": 0,
            "n_required_supported": 0,
            "claims": [],
        }

    pipeline = _get_nli_pipeline()
    claim_results: list[dict] = []

    for claim in claims:
        check = _claim_supported_nli(text, claim, pipeline, entailment_threshold)
        claim_results.append({
            "id": claim["id"],
            "type": claim["type"],
            "required": claim["required"],
            "premise": claim["premise"],
            "supported": check["supported"],
            "method": check["method"],
            "score": check["score"],
        })

    n_claims = len(claim_results)
    n_supported = sum(1 for c in claim_results if c["supported"])
    required = [c for c in claim_results if c["required"]]
    n_required = len(required)
    n_required_supported = sum(1 for c in required if c["supported"])

    # Primary score: required claims only (if any), else all claims
    if n_required > 0:
        faithfulness_score = round(n_required_supported / n_required, 3)
    elif n_claims > 0:
        faithfulness_score = round(n_supported / n_claims, 3)
    else:
        faithfulness_score = None

    return {
        "faithfulness_score": faithfulness_score,
        "n_claims": n_claims,
        "n_supported": n_supported,
        "n_required": n_required,
        "n_required_supported": n_required_supported,
        "claims": claim_results,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dimension 2 faithfulness evaluation (NLI).")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL file(s) from extract.py")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment threshold (default: 0.5)")
    parser.add_argument("--law", nargs="+", default=None, help="Filter to specific law(s)")
    parser.add_argument("--verbose", action="store_true", help="Show per-claim details")
    args = parser.parse_args()

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
                if not record.get("evaluation_trace"):
                    continue

                law = record.get("law", "")
                if law_filter and law not in law_filter:
                    continue

                bsn = str(record.get("profile", ""))
                name = record.get("profile_name", bsn)
                model = record.get("model", "?")

                scores = score_faithfulness(
                    record["explanation"],
                    record["evaluation_trace"],
                    args.threshold,
                )

                f_score = scores["faithfulness_score"]
                f_str = f"{f_score:.2f}" if f_score is not None else "n/a"
                print(
                    f"  {name:<22} [{model:<12}]  "
                    f"faithfulness={f_str}  "
                    f"({scores['n_required_supported']}/{scores['n_required']} required, "
                    f"{scores['n_supported']}/{scores['n_claims']} total)"
                )
                if args.verbose:
                    for claim in scores["claims"]:
                        mark = "✓" if claim["supported"] else "✗"
                        req = "*" if claim["required"] else " "
                        print(f"    {mark}{req} [{claim['type']:<10}] {claim['premise'][:70]}")

                all_scores.append({
                    "law": law, "profile": bsn,
                    "model": model,
                    **{k: v for k, v in scores.items() if k != "claims"},
                })

    if not all_scores:
        print("No records found.")
        return

    n = len(all_scores)
    faith_vals = [s["faithfulness_score"] for s in all_scores if s.get("faithfulness_score") is not None]
    print(f"\n{'='*60}")
    print(f"Total: {n}")
    if faith_vals:
        print(f"Faithfulness (avg): {sum(faith_vals)/len(faith_vals):.3f}")
        print(f"Fully faithful:     {sum(1 for v in faith_vals if v >= 1.0)}/{len(faith_vals)}")


if __name__ == "__main__":
    main()
