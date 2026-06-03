# LLM Explanations — Thesis Pipeline

This directory contains the full pipeline for generating and evaluating LLM-generated explanations of machine-law decisions, as used in the thesis research.

## Directory Structure

```
llm_explanations/
├── scripts/            # All runnable scripts (extraction, evaluation, chat)
│   ├── evaluation/     # Evaluation metrics and correlation analysis
│   └── ...
├── annotations/        # Human annotation input and parsed results
│   ├── input/          # Raw Excel survey responses (6 sheets, 3 laws × 2 rater groups)
│   └── results/        # Parsed CSVs: scores, inter-rater, auto-metrics, correlations
└── output/             # Generated output (not committed except final_output)
    ├── final_output/   # Final thesis explanations per model and approach
    └── graph_visualizations/  # PNG/PDF visualizations of decision graphs
```

## Three Extraction Approaches

The pipeline compares three conditions for generating citizen explanations:

| Approach | What the LLM receives | Purpose |
|---|---|---|
| `open` | Raw engine output only (outcome + conditions met/not met) | Baseline — no explicit decisive condition |
| `flat` | Decisive condition + key profile facts as plain text | Ablation — same info as graph, no structure |
| `graph` | Full decision graph (nodes, edges, condition values, relations) | Full GraphRAG condition |

The `flat` condition isolates whether quality gains come from *having the decisive condition available* versus from *graph structure itself*.

## Quick Start

### Generate explanations
```bash
# Graph approach, zorgtoeslag, all 200 profiles, haiku model
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach graph --law zorgtoeslag --models haiku

# Flat approach, single profile, multiple models
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach flat --law zorgtoeslag --profiles 105886512 --models haiku mistral llama3.1

# Open approach
uv run python analysis/llm_explanations/scripts/extract.py \
    --approach open --law zorgtoeslag --models haiku
```

### Run evaluation
```bash
# Evaluate a JSONL output file (computes Dim2 faithfulness + Dim3 citizen metrics)
uv run python analysis/llm_explanations/scripts/evaluation/evaluate.py \
    --input analysis/llm_explanations/output/final_output/haiku/graph_haiku_zorgtoeslag.jsonl

# Compute correlations between auto-metrics and human annotations
uv run python analysis/llm_explanations/scripts/evaluation/correlate.py
```

### Run annotation analysis
```bash
# Parse Excel annotation sheets + compute inter-rater agreement
uv run python analysis/llm_explanations/annotations/evaluate_annotations.py
```

## Evaluation Dimensions

| Dimension | Metric | Method |
|---|---|---|
| Dim2 — Faithfulness | Required claims supported | String matching (outcome, amount) + mDeBERTa NLI (conditions) |
| Dim3 — Readability | Flesch reading ease (NL) | `textstat` |
| Dim3 — Contestability | 3 binary checks / 3 | Decisive condition present, counterfactual phrasing, action mention |

## Laws Supported

| Law slug | Description |
|---|---|
| `zorgtoeslag` | Zorgtoeslag (health insurance subsidy) |
| `participatiewet/bijstand` | Bijstand (social assistance) |
| `alcoholwet/vergunning` | Alcoholwetvergunning (liquor licence) |

## Models Used

| Slug | Full model name |
|---|---|
| `haiku` | `claude-haiku-4-5-20251001` |
| `gpt4` | `GPT-4o` |
| `mistral` | `mistral:7b` |
| `llama3.1` | `llama3.1:8b` |
| `deepseek` | `deepseek-r1:8b` |

For local open-source models (mistral, llama3.1, deepseek), Ollama must be running:
```bash
ollama serve
```

## Output Format

Each run produces a timestamped folder under `output/`:
```
output/<timestamp>_<law>_<profiles>_<approach>/
└── <model>/
    └── <approach>_<model>_<law>.jsonl
```

Each JSONL file has one header record followed by one explanation record per profile. Records include the full `evaluation_trace` (outcome, decisive condition, key facts, amounts) needed for automated evaluation.
