"""
run_nli_eval.py — mDeBERTa NLI faithfulness evaluation with live progress.

Runs dim2 NLI scoring (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) for all
non-deepseek models across all approaches/laws. Traces for open approach
are injected from matching graph files.

Output: analysis/llm_explanations/output/evaluation_output/nli_results.jsonl
Log:    analysis/llm_explanations/output/evaluation_output/nli_progress.log
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score as _sklearn_roc_auc

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
EVAL_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "evaluation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from dim2_faithfulness import _get_nli_pipeline, build_trace_index, score_faithfulness  # noqa: E402

_LAW_NORMALIZE = {
    "alcoholwet/vergunning": "alcoholwet",
    "participatiewet/bijstand": "bijstand",
    "zorgtoeslagwet": "zorgtoeslag",
}

INPUT_FILES = [
    "analysis/llm_explanations/output/final_output_complete/gpt4/flat/flat_gpt4_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/flat/flat_gpt4_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/flat/flat_gpt4_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/graph/graph_gpt4_alcoholwet/vergunning.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/graph/graph_gpt4_participatiewet/bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/graph/graph_gpt4_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/open/open_gpt4_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/open/open_gpt4_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/gpt4/open/open_gpt4_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/flat/flat_haiku_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/flat/flat_haiku_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/flat/flat_haiku_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/graph/graph_haiku_alcoholwet/vergunning.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/graph/graph_haiku_participatiewet/bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/graph/graph_haiku_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/open/open_haiku_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/open/open_haiku_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/haiku/open/open_haiku_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/flat/flat_llama3.1_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/flat/flat_llama3.1_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/flat/flat_llama3.1_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/graph/graph_llama3.1_alcoholwet/vergunning.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/graph/graph_llama3.1_participatiewet/bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/graph/graph_llama3.1_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/open/open_llama3.1_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/open/open_llama3.1_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/llama3.1/open/open_llama3.1_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/flat/flat_mistral_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/flat/flat_mistral_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/flat/flat_mistral_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/graph/graph_mistral_alcoholwet/vergunning.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/graph/graph_mistral_participatiewet/bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/graph/graph_mistral_zorgtoeslag.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/open/open_mistral_alcoholwet.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/open/open_mistral_bijstand.jsonl",
    "analysis/llm_explanations/output/final_output_complete/mistral/open/open_mistral_zorgtoeslag.jsonl",
]

GRAPH_FILES = [f for f in INPUT_FILES if "/graph/" in f]

LOG_PATH = OUTPUT_DIR / "nli_progress.log"
OUT_PATH = OUTPUT_DIR / "nli_results.jsonl"


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return round(float(_sklearn_roc_auc(labels, scores)), 3)


def log(msg: str, log_fh=None) -> None:
    print(msg, flush=True)
    if log_fh:
        log_fh.write(msg + "\n")
        log_fh.flush()


def avg(vals: list) -> float | None:
    return round(sum(vals) / len(vals), 3) if vals else None


def _record_key(row: dict) -> tuple:
    return (row["law"], row["profile"], row["model"], row["approach"])


def load_existing_results(path: Path) -> list[dict]:
    results: list[dict] = []
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def print_table(results: list[dict], log_fh=None) -> None:
    MODEL_SHORT = {
        "gpt-4o": "gpt4",
        "gpt-4o-mini": "gpt4",
        "claude-haiku-4-5-20251001": "haiku",
        "llama3.1:8b": "llama3.1",
        "mistral:7b": "mistral",
    }
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["approach"], MODEL_SHORT.get(r["model"], r["model"]), r["law"])].append(r)

    header = f"\n{'Approach':<7} {'Model':<12} {'Wet':<12} {'N':>5} | {'NLI faith':>9} {'String':>7} {'ROC AUC':>8}"
    sep = "-" * 70
    log(header, log_fh)
    log(sep, log_fh)

    for approach in ("graph", "flat", "open"):
        for model in ("gpt4", "haiku", "llama3.1", "mistral"):
            for law in ("alcoholwet", "bijstand", "zorgtoeslag"):
                recs = groups.get((approach, model, law), [])
                if not recs:
                    continue
                nli_vals = [r["nli_faith"] for r in recs if r.get("nli_faith") is not None]
                str_vals = [r["string_faith"] for r in recs if r.get("string_faith") is not None]
                all_nli_scores: list[float] = []
                all_string_labels: list[int] = []
                for r in recs:
                    cs = r.get("nli_claim_scores") or []
                    sl = r.get("string_labels") or []
                    if len(cs) == len(sl):
                        all_nli_scores.extend(cs)
                        all_string_labels.extend(sl)
                nli_avg = avg(nli_vals)
                str_avg = avg(str_vals)
                auc = _roc_auc(all_string_labels, all_nli_scores)
                nli_str = f"{nli_avg:.3f}" if nli_avg is not None else "n/a"
                str_str = f"{str_avg:.3f}" if str_avg is not None else "n/a"
                auc_str = f"{auc:.3f}" if auc is not None else "n/a"
                log(f"{approach:<7} {model:<12} {law:<12} {len(recs):>5} | {nli_str:>9} {str_str:>7} {auc_str:>8}", log_fh)
        log("", log_fh)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run mDeBERTa NLI faithfulness evaluation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing nli_results.jsonl, skipping already-scored records")
    args = parser.parse_args()

    existing_results = load_existing_results(OUT_PATH) if args.resume else []
    done_keys = {_record_key(r) for r in existing_results}

    file_mode = "a" if args.resume else "w"

    with open(LOG_PATH, file_mode, encoding="utf-8") as log_fh, \
         open(OUT_PATH, file_mode, encoding="utf-8") as out_fh:

        if args.resume:
            log(f"\n=== RESUMING: {len(existing_results)} records already scored, skipping those ===\n", log_fh)

        log(f"Loading mDeBERTa...", log_fh)
        nli_pipe = _get_nli_pipeline()
        log("Model loaded. Building trace index for open approach...", log_fh)

        trace_index = build_trace_index(GRAPH_FILES)
        log(f"Trace index: {len(trace_index)} entries\n", log_fh)

        all_results: list[dict] = list(existing_results)
        total_done = len(existing_results)
        new_done = 0
        t_start = time.time()

        for file_path in INPUT_FILES:
            path = Path(file_path)
            log(f"\n=== {path.parent.parent.name}/{path.parent.name}/{path.name} ===", log_fh)

            with open(path, encoding="utf-8") as f:
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

                    trace = record.get("evaluation_trace")
                    if not trace:
                        bsn = str(record.get("profile", ""))
                        law_key = record.get("law", "").split("/")[-1]
                        trace = trace_index.get((law_key, bsn))
                    if not trace:
                        continue

                    law = _LAW_NORMALIZE.get(record.get("law", ""), record.get("law", ""))
                    bsn = str(record.get("profile", ""))
                    name = record.get("profile_name", bsn)
                    model = record.get("model", "?")
                    approach = record.get("approach", "?")

                    if (law, bsn, model, approach) in done_keys:
                        continue

                    # String-based (fast reference)
                    str_scores = score_faithfulness(record["explanation"], trace, nli_pipe=None)
                    # NLI-based
                    nli_scores = score_faithfulness(record["explanation"], trace, nli_pipe=nli_pipe)

                    total_done += 1
                    new_done += 1
                    elapsed = time.time() - t_start
                    rate = new_done / elapsed if elapsed > 0 else 0

                    str_f = str_scores["faithfulness_score"]
                    nli_f = nli_scores["faithfulness_score"]
                    log(
                        f"  [{total_done:>5}] {name:<22} | string={str_f:.2f} nli={nli_f:.2f}"
                        f"  ({rate:.1f} rec/s)",
                        log_fh,
                    )

                    row = {
                        "law": law, "profile": bsn, "profile_name": name,
                        "model": model, "approach": approach,
                        "string_faith": str_f,
                        "nli_faith": nli_f,
                        "nli_n_claims": nli_scores["n_claims"],
                        "nli_n_supported": nli_scores["n_supported"],
                        "nli_claim_scores": nli_scores.get("nli_claim_scores", []),
                        "string_labels": nli_scores.get("string_labels", []),
                    }
                    all_results.append(row)
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()

        elapsed_total = time.time() - t_start
        log(f"\n\nDone. {total_done} records in {elapsed_total/60:.1f} min.", log_fh)

        # Overall ROC AUC across all records
        all_nli: list[float] = []
        all_lbl: list[int] = []
        for r in all_results:
            cs = r.get("nli_claim_scores") or []
            sl = r.get("string_labels") or []
            if len(cs) == len(sl):
                all_nli.extend(cs)
                all_lbl.extend(sl)
        overall_auc = _roc_auc(all_lbl, all_nli)
        auc_str = f"{overall_auc:.3f}" if overall_auc is not None else "n/a (no label variation)"
        log(f"Overall ROC AUC (NLI prob vs string-match label): {auc_str}", log_fh)
        log(f"  Total claims evaluated: {len(all_lbl)}", log_fh)

        log("\n" + "=" * 60, log_fh)
        log("BREAKDOWN: approach × model × law (NLI faith | string faith | ROC AUC)", log_fh)
        print_table(all_results, log_fh)


if __name__ == "__main__":
    main()
