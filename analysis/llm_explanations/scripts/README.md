# LLM Explanation Scripts

Scripts voor het genereren en evalueren van LLM-uitleg bij machine law beslissingen.

---

## Vereisten

- Web server actief voor `chat_client.py`
- Ollama actief voor lokale modellen (`llama3.1`, `mistral`, `deepseek`)
- `ANTHROPIC_API_KEY` in omgeving voor Claude-modellen
- `OPENAI_API_KEY` in omgeving voor OpenAI-modellen

---

## chat_client.py — Interactieve chat met de web interface

Start eerst de server:
```powershell
$env:FEATURE_CHAT='1'; uv run web/main.py
```

### Interactief chatten (met LLM guard)
```bash
uv run python analysis/llm_explanations/scripts/chat_client.py --bsn 403987006
```

### Interactief chatten (zonder LLM guard)
```bash
uv run python analysis/llm_explanations/scripts/chat_client.py --bsn 403987006 --no-guard
```

### Guard beslissingen zichtbaar maken
```bash
uv run python analysis/llm_explanations/scripts/chat_client.py --bsn 403987006 --verbose
```

### Vergelijking nulmeting vs GraphRAG
```powershell
# Terminal 1 — nulmeting (zonder guard, ruwe engine output)
uv run python analysis/llm_explanations/scripts/chat_client.py --bsn 403987006 --no-guard

# Terminal 2 — GraphRAG (knowledge graph als LLM context)
uv run python analysis/llm_explanations/scripts/chat_client_graphrag.py --bsn 403987006
```

### Opties
| Optie | Beschrijving | Default |
|---|---|---|
| `--bsn` | BSN van het profiel | `403987006` (Roos van Leeuwen) |
| `--provider` | LLM provider: `claude` of `vlam` | `claude` |
| `--host` | Host:port van de webserver | `localhost:8000` |
| `--verbose` | Toont guard-beslissingen na elk antwoord | uit |
| `--no-guard` | Schakelt de LLM guard uit | aan |

---

## extract.py — Skeleton-aanpak (kleine/lokale modellen)

Genereert uitleg via een Markdown skeleton → LLM.

```bash
# Alle profielen, één wet, lokaal model
uv run python analysis/llm_explanations/scripts/extract.py \
    --law zorgtoeslag --model llama3.1

# Specifieke profielen
uv run python analysis/llm_explanations/scripts/extract.py \
    --law zorgtoeslag bijstand alcoholwet \
    --model llama3.1 \
    --profiles 403987006 548339668 318140003

# Met graph visualisaties opslaan
uv run python analysis/llm_explanations/scripts/extract.py \
    --law zorgtoeslag --model llama3.1 --save-graphs
```

---

## extract_graphrag.py — GraphRAG-aanpak (grote modellen)

Geeft de volledige beslissingsgraph als JSON aan de LLM.

```bash
# Claude Sonnet (cloud)
uv run python analysis/llm_explanations/scripts/extract_graphrag.py \
    --law zorgtoeslag bijstand alcoholwet \
    --model sonnet \
    --profiles 403987006 548339668 318140003

# Lokaal via Ollama
uv run python analysis/llm_explanations/scripts/extract_graphrag.py \
    --law zorgtoeslag \
    --model llama3.1 \
    --profiles 403987006 909990066

# Met graph visualisaties opslaan
uv run python analysis/llm_explanations/scripts/extract_graphrag.py \
    --law zorgtoeslag --model sonnet --save-graphs

# Vorige run hervatten (cache hergebruiken)
uv run python analysis/llm_explanations/scripts/extract_graphrag.py \
    --law zorgtoeslag --model sonnet \
    --output analysis/llm_explanations/output/20260401_XXXXXX_... \
    --resume
```

### Beschikbare modellen
| Key | Model | Provider |
|---|---|---|
| `sonnet` | claude-sonnet-4-6 | Anthropic |
| `opus` | claude-opus-4-6 | Anthropic |
| `haiku` | claude-haiku-4-5 | Anthropic |
| `llama3.1` | llama3.1:8b | Ollama (lokaal) |
| `llama3.2` | llama3.2:3b | Ollama (lokaal) |
| `mistral` | mistral:7b | Ollama (lokaal) |
| `deepseek` | deepseek-r1:8b | Ollama (lokaal) |
| `gemma2` | gemma2:9b | Ollama (lokaal) |

---

## Profielen voor testen (3 wetten gedekt)

| BSN | Naam | Zorgtoeslag | Bijstand | Alcoholwet |
|---|---|---|---|---|
| `403987006` | Roos van Leeuwen | ✅ | ✅ | ❌ |
| `548339668` | Lisa de Wit | ✅ | ❌ | ✅ |
| `318140003` | Emma Hendriks | ✅ | ❌ | ✅ |

---

## Output locatie

Alle runs worden opgeslagen in:
```
analysis/llm_explanations/output/YYYYMMDD_HHMMSS_{n}laws_{n}profiles_{approach}/{model}/
```

Bestanden per run:
- `graphrag_{model}_{wet}.jsonl` — uitleg-records (JSONL)
- `cache_{wet}.json` — engine-berekeningen cache
- `graph_{wet}_{bsn}.png` — graph visualisaties (met `--save-graphs`)
