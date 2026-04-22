"""
Contestability scoring for LLM-generated explanations.

Measures whether a chat explanation contains the three elements
a citizen needs to contest a government decision:
1. A verifiable source (law article or official URL)
2. The decisive condition that determined the outcome
3. A counterfactual path (what would need to change)
"""

import re


def contestability_score(explanation_text: str, rac_trace: dict) -> dict:
    """Score an explanation on contestability (0.0 – 1.0).

    Args:
        explanation_text: The LLM-generated explanation shown to the citizen.
        rac_trace: Dict with at least 'decisive_condition' key, built from the
                   decision graph for the relevant service.

    Returns:
        Dict with overall score and individual boolean checks.
    """
    # Check 1: decisive condition named — the specific condition that decided the outcome
    decisive_condition = rac_trace.get("decisive_condition", "")
    has_decisive_condition = (
        decisive_condition.lower() in explanation_text.lower()
    ) if decisive_condition else False

    # Check 2: counterfactual path — what would need to change for a different outcome
    has_contestable_path = bool(re.search(
        r"(als|indien|wanneer|zou|tenzij|behoudens).{5,80}(dan|zou u|heeft u recht|kunt u)",
        explanation_text, re.IGNORECASE
    ))

    score = sum([has_decisive_condition, has_contestable_path]) / 2

    return {
        "contestability_score": round(score, 2),
        "has_decisive_condition": has_decisive_condition,
        "has_contestable_path": has_contestable_path,
        "decisive_condition_checked": decisive_condition,
    }
