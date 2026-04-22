"""
Dimension 1: Legal grounding evaluation.

Measures whether an LLM-generated explanation correctly cites the articles
from the law's legal_basis and references sections (as stored in
evaluation_trace).

Approach: string matching — does the explanation mention each article number
together with (part of) the law name?  No external API required.

Score = correctly cited articles / total available articles in trace.

This module is importable (for evaluate.py) and runnable standalone:

    uv run python analysis/llm_explanations/scripts/evaluation/dim1_legal.py \
        --input output/20260415_.../*.jsonl
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Article extraction from evaluation_trace
# ---------------------------------------------------------------------------

def _extract_expected_articles(trace: dict) -> list[dict]:
    """
    Build deduplicated list of expected article citations from evaluation_trace.

    Deduplicates by article number only — "artikel 2 Zorgtoeslagwet" and
    "artikel 2 Wet op de zorgtoeslag" are the same article and count once.
    All law name variants are merged into a single keyword pool so any
    variant in the explanation counts as a match.

    Each entry: {article, law, law_keywords, patterns}
    """
    # article_number -> {canonical_law, all_keywords}
    by_article: dict[str, dict] = {}

    def _add(article: str, law_name: str) -> None:
        if not article or not law_name:
            return
        key = article.lower()
        # Include abbreviations (zvw, wet, etc.) — min length 2
        keywords = {w for w in law_name.lower().split() if len(w) >= 2}
        if key not in by_article:
            by_article[key] = {"article": article, "law": law_name, "keywords": keywords}
        else:
            # Merge keywords from all name variants
            by_article[key]["keywords"] |= keywords

    # Top-level legal_basis
    lb = trace.get("legal_basis") or {}
    if lb.get("article") and lb.get("law"):
        _add(lb["article"], lb["law"])

    # References list
    for ref in trace.get("references", []):
        if ref.get("article") and ref.get("law"):
            _add(ref["article"], ref["law"])

    # Key facts legal_basis
    for _field, info in trace.get("key_facts", {}).items():
        fact_lb = info.get("legal_basis") or {}
        if fact_lb.get("article") and fact_lb.get("law"):
            _add(fact_lb["article"], fact_lb["law"])

    expected: list[dict] = []
    for entry in by_article.values():
        article = entry["article"]
        expected.append({
            "article": article,
            "law": entry["law"],
            "law_keywords": list(entry["keywords"]),
            "patterns": [
                rf"artikel\s+{re.escape(article)}",   # "artikel 2 Zorgtoeslagwet"
                rf"art\.?\s+{re.escape(article)}",    # "art. 2 ..."
                # law name BEFORE article number: "zvw artikel 1", "ZVW art 1"
                *[rf"{re.escape(kw)}\s+(?:artikel|art\.?)\s+{re.escape(article)}"
                  for kw in entry["keywords"]],
            ],
        })

    return expected


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _article_cited(text: str, entry: dict) -> bool:
    """
    Check if an article is cited in the explanation text.

    Requires:
    1. The article number pattern matches somewhere in the text
    2. At least one law keyword appears within 80 characters of the match
    """
    text_lower = text.lower()

    for pattern in entry["patterns"]:
        for m in re.finditer(pattern, text_lower, re.IGNORECASE):
            # Check law keyword proximity
            start = max(0, m.start() - 80)
            end = min(len(text_lower), m.end() + 80)
            context = text_lower[start:end]
            if any(kw in context for kw in entry["law_keywords"]):
                return True
            # Also accept if the full law name shortened appears anywhere in text
            if any(kw in text_lower for kw in entry["law_keywords"]):
                return True

    return False


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_legal(
    explanation: str,
    trace: dict,
) -> dict:
    """
    Compute Dimension 1 legal grounding score for a single explanation.

    Args:
        explanation:  The LLM-generated explanation text.
        trace:        The evaluation_trace dict from extract.py output.

    Returns:
        Dict with keys:
            article_coverage:     float 0–1 (cited / available)
            n_available:          total articles in trace
            n_cited:              articles found in explanation
            articles:             list of per-article results
    """
    text = (explanation or "").strip()
    expected = _extract_expected_articles(trace)

    if not expected:
        return {
            "article_coverage": None,
            "n_available": 0,
            "n_cited": 0,
            "articles": [],
        }

    article_results: list[dict] = []
    for entry in expected:
        cited = _article_cited(text, entry)
        article_results.append({
            "article": entry["article"],
            "law": entry["law"],
            "cited": cited,
        })

    n_available = len(article_results)
    n_cited = sum(1 for a in article_results if a["cited"])
    coverage = round(n_cited / n_available, 3) if n_available else None

    return {
        "article_coverage": coverage,
        "n_available": n_available,
        "n_cited": n_cited,
        "articles": article_results,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Dimension 1 legal grounding evaluation.")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL file(s) from extract.py")
    parser.add_argument("--law", nargs="+", default=None, help="Filter to specific law(s)")
    parser.add_argument("--verbose", action="store_true", help="Show per-article details")
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

                scores = score_legal(record["explanation"], record["evaluation_trace"])

                cov = scores["article_coverage"]
                cov_str = f"{cov:.2f}" if cov is not None else "n/a"
                print(
                    f"  {name:<22} [{model:<12}]  "
                    f"coverage={cov_str}  "
                    f"({scores['n_cited']}/{scores['n_available']} articles)"
                )
                if args.verbose:
                    for art in scores["articles"]:
                        mark = "✓" if art["cited"] else "✗"
                        print(f"    {mark} artikel {art['article']} {art['law']}")

                all_scores.append({
                    "law": law, "profile": bsn, "model": model,
                    **{k: v for k, v in scores.items() if k != "articles"},
                })

    if not all_scores:
        print("No records found.")
        return

    n = len(all_scores)
    cov_vals = [s["article_coverage"] for s in all_scores if s.get("article_coverage") is not None]
    print(f"\n{'='*60}")
    print(f"Total: {n}")
    if cov_vals:
        print(f"Article coverage (avg): {sum(cov_vals)/len(cov_vals):.2f}")
        print(f"Full coverage (1.0):    {sum(1 for v in cov_vals if v >= 1.0)}/{len(cov_vals)}")


if __name__ == "__main__":
    main()
