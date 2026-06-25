"""Quick one-off: ROC AUC per (approach, model), pooled across laws."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

PATH = Path(__file__).parent.parent.parent / "output" / "evaluation_output" / "nli_results.jsonl"

MODEL_SHORT = {
    "gpt-4o": "gpt4",
    "gpt-4o-mini": "gpt4",
    "claude-haiku-4-5-20251001": "haiku",
    "llama3.1:8b": "llama3.1",
    "mistral:7b": "mistral",
}


def main() -> None:
    groups: dict[tuple[str, str], tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))

    with open(PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cs = r.get("nli_claim_scores") or []
            sl = r.get("string_labels") or []
            if len(cs) != len(sl) or not cs:
                continue
            model = MODEL_SHORT.get(r["model"], r["model"])
            key = (r["approach"], model)
            groups[key][0].extend(cs)
            groups[key][1].extend(sl)

    approaches = ["graph", "flat", "open"]
    models = ["gpt4", "haiku", "llama3.1", "mistral"]

    print(f"{'Approach':<8} {'Model':<10} {'N claims':>9} {'Neg':>6} {'ROC AUC':>8}")
    print("-" * 46)
    for approach in approaches:
        for model in models:
            scores, labels = groups.get((approach, model), ([], []))
            if not scores:
                continue
            n = len(scores)
            n_neg = sum(1 for x in labels if x == 0)
            if len(set(labels)) < 2:
                auc_str = "n/a"
            else:
                auc_str = f"{roc_auc_score(labels, scores):.3f}"
            print(f"{approach:<8} {model:<10} {n:>9} {n_neg:>6} {auc_str:>8}")
        print()


if __name__ == "__main__":
    main()
