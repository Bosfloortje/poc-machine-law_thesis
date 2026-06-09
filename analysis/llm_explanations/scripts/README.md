# Scripts

All runnable scripts for the LLM explanation pipeline.

## Extraction

### `extract.py` — main entry point
Generates LLM explanations for all three approaches (open, flat, graph).

```bash
# Graph approach — structured decision skeleton
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach graph --law zorgtoeslag --models haiku mistral

# Flat approach — decisive condition + key facts as plain text (ablation baseline)
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach flat --law zorgtoeslag --profiles 105886512 --models haiku

# Open approach — raw engine output only, no explicit decisive condition
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach open --law zorgtoeslag --models haiku

# All profiles, quiet mode
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach graph --law zorgtoeslag --models haiku --quiet
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--approach` | required | `open`, `flat`, or `graph` |
| `--law` | required | `zorgtoeslag`, `participatiewet/bijstand`, `alcoholwet/vergunning` |
| `--models` | required | One or more of `haiku`, `mistral`, `llama3.1`, `gpt4`, `deepseek` |
| `--profiles` | all | Filter to specific BSN numbers |
| `--quiet` | off | Suppress verbose output |

Final thesis results are in `output/final_output_complete/`.

### `extraction_generic.py` — core engine
Shared infrastructure imported by `extract.py`. Contains the `DecisionGraphExtractor`, graph builders, and YAML law loaders. Not run directly.

### `extract_graphrag.py` — graph serializer
Serializes the decision graph to the structured text format used by the graph approach prompt. Imported by `extract.py`.

---

## Chat

### `chat_client.py` — interactive single-turn chat
Interactive CLI to ask a single question about a citizen profile and law.

```bash
uv run python analysis/llm_explanations/scripts/chat_client.py
uv run python analysis/llm_explanations/scripts/chat_client.py --bsn 403987006 --law zorgtoeslag
```

### `chat_batch.py` — multi-turn batch chat
Runs a multi-turn conversation pipeline over a set of profiles, using a WebSocket connection to the running web server.

```bash
# Requires the web server to be running:
uv run web/main.py

uv run python analysis/llm_explanations/scripts/chat_batch.py \
    --law zorgtoeslag --model mistral
```

---

## Evaluation (`evaluation/`)

### `evaluate.py` — main evaluation orchestrator
Runs Dim2 (faithfulness) and Dim3 (citizen quality) metrics over a JSONL output file.

```bash
uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
    --input analysis/llm_explanations/output/final_output_complete/haiku/graph/graph_haiku_zorgtoeslag.jsonl
```

### `run_nli_eval.py` — full NLI faithfulness run
Runs mDeBERTa NLI scoring across all models, approaches, and laws. Saves per-record results (including per-claim NLI scores and string labels for ROC AUC) to `output/evaluation_output/nli_results.jsonl`.

```bash
uv run python analysis/llm_explanations/scripts/evaluation/run_nli_eval.py
```

### `dim2_faithfulness.py` — faithfulness scoring
Hybrid scorer: outcome and amount claims use string matching; condition claims use mDeBERTa NLI.
Score = supported required claims / total required claims (outcome + amount).
Imported by `evaluate.py` and `run_nli_eval.py`.

### `dim3_citizen.py` — citizen quality metrics
Computes Flesch reading ease (NL) and contestability (3 binary checks / 3).
Imported by `evaluate.py`.

### `correlate.py` — auto-metric vs human correlation
Computes Pearson and Spearman correlations between automated metrics (Dim2, Dim3) and human annotation scores. Reads from `annotations/results/`.

```bash
uv run python analysis/llm_explanations/scripts/evaluation/correlate.py
```

### `evaluation_output/`
Automated evaluation results:
- `eval_results_complete.jsonl` — full Dim2 + Dim3 results for all models/approaches/laws
- `nli_results.jsonl` — per-record NLI scores with per-claim breakdowns
- `eval_summary.json` — aggregated summary statistics
