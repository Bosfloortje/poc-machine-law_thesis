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

### `extract_graph.py` — graph serializer
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

---

## Profiles (`profiles/`)

### `profiles/generate_profiles.py` — rule-based profile generator
Generates synthetic citizen profiles in `profiles.yaml` format. Uses no LLM — all variation is produced by rule-based sampling weighted to match CBS (Statistics Netherlands) population statistics for 2023.

**Covers all three laws** by including the data sources required by each:
- Zorgtoeslag: RvIG, Belastingdienst, RVZ, SVB
- Bijstand/Participatiewet: UWV, SZW, gemeente (`werk_en_re_integratie`)
- Alcoholwet/vergunning: KVK (incl. `inrichtingen`, `vergunningen`), SVH (Register Sociale Hygiene), LBB (Bibob advies), RECHTSPRAAK (curatele)

**Demographic distributions (CBS 2023):**
- Herkomst: NL 81.5%, AR 6%, TR 5.5%, SR 4%, AS 2%, EE 1%
- Werkstatus: loondienst 59.6%, gepensioneerd 26.7%, ZZP 8.6%, werkloos 2.9%
- Inkomen: laag/midden/hoog (~33% elk), CBS-gemiddeld per leeftijdsklasse
- Leeftijd: ZZP-specifieke verdeling (45-75j 60%), gepensioneerden 65+, overig 18-64

```bash
# Generate 200 new profiles (auto-saves to data/profielen/profiles_200_<timestamp>.yaml)
uv run python analysis/llm_explanations/scripts/profiles/generate_profiles.py --count 200

# Use a specific input and output file
uv run python analysis/llm_explanations/scripts/profiles/generate_profiles.py \
    --input data/profielen/profiles.yaml \
    --count 100 \
    --output data/profielen/my_profiles.yaml
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--input` | `data/profielen/profiles.yaml` | Existing profiles.yaml to read `globalServices` from |
| `--count` | `10` | Number of profiles to generate |
| `--output` | auto | Output path; auto-generates `profiles_<count>_<timestamp>.yaml` in `data/profielen/` |
| `--start-bsn` | `100000100` | Starting BSN seed (unused BSNs are picked randomly above this) |
