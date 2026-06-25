# LLM Explanations — Thesis Pipeline

This directory contains the full pipeline for generating and evaluating LLM-generated explanations of machine-law decisions, as used in the thesis research.

## Directory Structure

```
llm_explanations/
├── scripts/            # All runnable scripts (extraction, evaluation, chat, profile generation)
│   ├── evaluation/     # Evaluation metrics and correlation analysis
│   └── profiles/       # Rule-based synthetic profile generator (no LLM)
├── annotations/        # Human annotation input and parsed results
│   ├── first_initial_look/  # Exploratory pre-study annotations — informed the final methodology
│   ├── input/          # Raw Excel survey responses (6 sheets, 3 laws × 2 rater groups)
│   └── results/        # Parsed CSVs: scores, inter-rater, auto-metrics, correlations
└── output/
    ├── final_output_complete/  # All thesis LLM results — 5 models × 3 approaches × 3 laws
    └── evaluation_output/      # Automated evaluation results (eval_results_complete.jsonl)
```

### `final_output_complete/` structure

```
final_output_complete/
└── <model>/
    └── <approach>/
        └── <approach>_<model>_<law>.jsonl
```

Models: `gpt4`, `haiku`, `llama3.1`, `mistral`, `deepseek`  
Approaches: `graph`, `flat`, `open`  
Laws: `alcoholwet`, `bijstand`, `zorgtoeslag`

Each JSONL file contains one header record followed by one explanation record per profile (200 profiles per file). Each record includes the `evaluation_trace` (outcome, decisive condition, key facts, amounts) needed for automated evaluation.

## Three Extraction Approaches

| Approach | What the LLM receives | Purpose |
|---|---|---|
| `open` | Raw engine output (outcome + conditions met/not met) + profile description | Baseline — no explicit decisive condition, free-form prompt |
| `flat` | Decisive condition + key profile facts as plain text bullets | Ablation — same info as graph, no structure |
| `graph` | Structured decision skeleton (outcome, citizen facts, conditions with [JA]/[NEE], legal articles) | Full structured condition — strict "translate literally" prompt |

The `flat` condition isolates whether quality gains come from *having the decisive condition available* versus from *graph structure itself*.

## Quick Start

### Generate citizen profiles
```bash
# Generate 200 CBS-weighted synthetic profiles (no LLM required)
uv run python analysis/llm_explanations/scripts/profiles/generate_profiles.py --count 200
```

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
    --input analysis/llm_explanations/output/final_output_complete/haiku/graph/graph_haiku_zorgtoeslag.jsonl

# Run full NLI faithfulness evaluation across all models/approaches (saves to evaluation_output/)
uv run python analysis/llm_explanations/scripts/evaluation/run_nli_eval.py
```

## Evaluation Dimensions

| Dimension | Metric | Method |
|---|---|---|
| Dim2 — Faithfulness | Required claims supported | String matching (outcome, amount) + mDeBERTa NLI (conditions) |
| Dim3 — Readability | Flesch reading ease (NL) | `textstat` |
| Dim3 — Contestability | 3 binary checks / 3 | Decisive condition present, counterfactual phrasing, action mention |

Results are aggregated in `output/evaluation_output/eval_results_complete.jsonl`.

## Annotation Methodology

Human evaluation was set up in two phases:

**Phase 1 — `annotations/first_initial_look/`**  
Exploratory annotations collected before the formal evaluation survey was designed. A small set of explanations (20 records + cross-law comparison by a second annotator) was assessed using open coding across different models and approaches (llama3.1 and haiku, graph approach, zorgtoeslag). This first look at the data informed which evaluation dimensions to use, which models/approaches to include in the formal study, and the wording and scale anchors of the final survey questions.

**Phase 2 — `annotations/input/`**  
Formal Google Forms survey with fixed rubrics, distributed to two rater groups (citizens and jurists) across all three laws. Responses parsed with `parse_annotations.py`; inter-rater agreement computed with `inter_rater.py`.

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
| `gpt4` | `gpt-4o` |
| `mistral` | `mistral:7b` |
| `llama3.1` | `llama3.1:8b` |
| `deepseek` | `deepseek-r1:8b` |

For local open-source models (mistral, llama3.1, deepseek), Ollama must be running:
```bash
ollama serve
```
