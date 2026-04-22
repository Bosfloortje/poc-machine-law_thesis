"""
Contestability scoring for LLM-generated explanations.

Measures whether an explanation gives a citizen enough information to
verify and challenge a decision. Only used in the chat + GraphRAG flow.
"""

import re


def contestability_score(explanation_text: str, rac_trace: dict) -> dict:
    """Score how contestable an explanation is (0.0 – 1.0).

    Three binary checks, each worth 1/3:
    1. Verifiable source  — cites a law name or URL the citizen can look up
    2. Decisive condition — names the specific condition that determined the outcome
    3. Contestable path   — includes a counterfactual (what would need to change)

    Args:
        explanation_text: The LLM-generated explanation to score.
        rac_trace: Dict with at least "decisive_condition" (str) and "info_url" (str).

    Returns:
        Dict with contestability_score (float) and three bool flags.
    """
    # Check 1: cites a verifiable source (law name, article, or known URL)
    has_verifiable_source = bool(re.search(
        r"(artikel\s?\d+|wetten\.overheid\.nl|www\.\w+\.nl|wet\s+op\s+de|besluit\s+\w+)",
        explanation_text, re.IGNORECASE,
    ))

    # Check 2: names the decisive condition that determined the outcome
    decisive_condition = rac_trace.get("decisive_condition", "")
    has_decisive_condition = (
        decisive_condition.lower() in explanation_text.lower()
    ) if decisive_condition else False

    # Check 3: includes a counterfactual (what would need to change)
    has_contestable_path = bool(re.search(
        r"(als|indien|wanneer|zou|tenzij|behoudens).{5,80}(dan|zou u|heeft u recht|kunt u)",
        explanation_text, re.IGNORECASE,
    ))

    score = sum([has_verifiable_source, has_decisive_condition, has_contestable_path]) / 3

    return {
        "contestability_score": round(score, 3),
        "has_verifiable_source": has_verifiable_source,
        "has_decisive_condition": has_decisive_condition,
        "has_contestable_path": has_contestable_path,
    }
