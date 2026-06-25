#!/usr/bin/env python3
"""
Graph extraction: pass the full decision knowledge graph directly to a large LLM.

Unlike the skeleton approach (extract.py / extraction_generic.py), this script
serialises the knowledge graph as structured JSON and sends it to a large LLM
(Claude Sonnet/Opus, GPT-4) that can reason over relational data.

Difference vs skeleton approach
--------------------------------
  Skeleton approach →  graph is flattened to Markdown → small/local LLM
  Graph approach    →  graph serialised as JSON → large LLM reasons over structure

The LLM receives:
  - All DECISION nodes  (final outcome + label)
  - All RULE nodes      (each condition, its status and subject value)
  - All FACT nodes      (profile values used in reasoning)
  - Edge relationships  (which facts feed which rules, which rules feed the decision)

This lets the LLM traverse the reasoning chain itself and produce a richer,
more contextual explanation — without a hand-crafted skeleton as intermediary.

Usage (run from project root):
    uv run python analysis/llm_explanations/scripts/extract_graph.py \\
        --law zorgtoeslag --model claude-sonnet-4-6

    uv run python analysis/llm_explanations/scripts/extract_graph.py \\
        --law zorgtoeslag bijstand alcoholwet \\
        --model claude-opus-4-6 \\
        --profiles 403987006 909990066

Output format is identical to the skeleton approach (JSONL) so results can be
compared side-by-side.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — mirrors extract.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent.parent / "output"

from extraction_generic import (  # noqa: E402
    DecisionGraphExtractor,
    KnowledgeGraph,
    get_git_info,
    load_law_yaml,
    load_profiles,
    run_calculation,
)

# ---------------------------------------------------------------------------
# Graph serialisation
# ---------------------------------------------------------------------------

def _serialize_legal_basis(law: dict) -> list[dict]:
    """Extract deduplicated article citations from law YAML legal_basis + references."""
    articles: list[dict] = []
    seen: set[tuple] = set()

    def _add(article: str, law_name: str) -> None:
        key = (article, law_name[:20])
        if key not in seen and article and law_name:
            seen.add(key)
            articles.append({"artikel": article, "wet": law_name})

    lb = law.get("legal_basis") or {}
    if lb.get("article") and lb.get("law"):
        _add(lb["article"], lb["law"])
    for ref in law.get("references", []):
        if ref.get("article") and ref.get("law"):
            _add(ref["article"], ref["law"])
    return articles


def serialize_graph(
    graph: KnowledgeGraph,
    decision_extractor: DecisionGraphExtractor,
    person_name: str,
) -> dict[str, Any]:
    """
    Convert the KnowledgeGraph to a structured dict for LLM consumption.

    The structure preserves the relational information (which facts feed which
    rules, which rules determine the decision) so the LLM can follow the
    reasoning chain rather than just reading a flat list.
    """
    law_name = decision_extractor.law.get("name", "Regeling")
    calc = decision_extractor.calc_result or {}
    requirements_met = calc.get("requirements_met")
    output = calc.get("result", {})

    # --- Index nodes by id for edge resolution ---
    node_by_id: dict[str, Any] = {n.id: n for n in graph.nodes}

    # --- Partition RULE nodes by status ---
    STATUS_SATISFIED = DecisionGraphExtractor.STATUS_SATISFIED
    STATUS_FAILED = DecisionGraphExtractor.STATUS_FAILED
    STATUS_NOT_APPLICABLE = DecisionGraphExtractor.STATUS_NOT_APPLICABLE

    rule_nodes = [n for n in graph.nodes if n.type == "RULE"]
    satisfied = [n for n in rule_nodes if n.properties.get("status") == STATUS_SATISFIED]
    failed = [n for n in rule_nodes if n.properties.get("status") == STATUS_FAILED]
    unknown = [n for n in rule_nodes if n.properties.get("status") == STATUS_NOT_APPLICABLE]

    # --- FACT nodes (profile values) ---
    fact_nodes = [n for n in graph.nodes if n.type == "FACT"]

    # --- DECISION node ---
    decision_nodes = [n for n in graph.nodes if n.type == "DECISION"]
    decision_label = decision_nodes[0].label if decision_nodes else "ONBEKEND"

    # --- Output amounts with period ---
    output_meta = decision_extractor._output_meta
    output_lines: dict[str, str] = {}
    for field_name, value in output.items():
        meta = output_meta.get(field_name, {})
        if meta.get("unit") != "eurocent" and meta.get("type") != "amount":
            continue
        description = meta.get("description", field_name)
        unit = meta.get("unit", "")
        period = meta.get("period", "")
        formatted = decision_extractor._format_value(value, unit)
        output_lines[description] = f"{formatted}{(' ' + period) if period else ''}"

    # --- Edge map: which fact_ids connect to which rule_ids ---
    fact_to_rules: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relation == "USED_IN":
            fact_to_rules.setdefault(edge.source, []).append(edge.target)

    # --- Split facts: used in a rule vs. context-only ---
    used_fact_ids = set(fact_to_rules.keys())

    # --- Serialise conditions with linked fact values ---
    def _serialize_rule(node: Any) -> dict[str, Any]:
        props = node.properties
        subject = props.get("subject", "")
        # Find fact node for this subject
        fact_value = None
        fact_id = f"fact_{subject}"
        if fact_id in node_by_id:
            fn = node_by_id[fact_id]
            fact_value = fn.label.split(": ", 1)[-1] if ": " in fn.label else fn.label
        return {
            "beschrijving": node.label,
            "status": props.get("status", ""),
            "subject": subject,
            "profielwaarde": fact_value,
            "is_or_groep": props.get("is_or_group", False),
        }

    # --- Determine which unknown conditions to surface ---
    # Only show onbekend_gegevens_ontbreken when it is the sole reason for a
    # negative outcome (requirements_voldaan=False AND niet_voldaan is empty).
    # When the decision is positive, or when there are clear failed conditions,
    # missing data is irrelevant and would mislead the LLM into mentioning it.
    show_unknown = bool(unknown) and not requirements_met and not failed

    # --- Assemble final structure ---
    return {
        "regeling": law_name,
        "burger": person_name,
        "beslissing": {
            "uitkomst": decision_label,
            "requirements_voldaan": requirements_met,
        },
        "voorwaarden": {
            "voldaan": [_serialize_rule(n) for n in satisfied],
            "niet_voldaan": [_serialize_rule(n) for n in failed],
            "onbekend_gegevens_ontbreken": [_serialize_rule(n) for n in unknown] if show_unknown else [],
        },
        "feiten_gebruikt": {
            n.properties.get("description", n.id): n.label.split(": ", 1)[-1]
            for n in fact_nodes
            if ": " in n.label and n.id in used_fact_ids
        },
        "feiten_context": {
            n.properties.get("description", n.id): n.label.split(": ", 1)[-1]
            for n in fact_nodes
            if ": " in n.label and n.id not in used_fact_ids
        },
        "berekend_bedrag": output_lines or None,
        "wettelijke_grondslag": _serialize_legal_basis(decision_extractor.law),
        "relaties": [
            {
                "van": node_by_id[e.source].properties.get("description", e.source)
                       if e.source in node_by_id else e.source,
                "naar": node_by_id[e.target].label[:60]
                        if e.target in node_by_id else e.target,
                "relatie": e.relation,
            }
            for e in graph.edges
            if e.relation in ("USED_IN", "DETERMINES")
        ],
    }


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

GRAPHRAG_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse burgers uitleg geeft over overheidsbeslissingen.

Je krijgt een beslissingsgraph als JSON. De graph beschrijft volledig hoe een beslissing tot stand is gekomen:
- `beslissing.uitkomst`: de finale uitkomst
- `beslissing.requirements_voldaan`: true = burger voldoet aan de eisen
- `voorwaarden.voldaan`: voorwaarden waaraan de burger WEL voldoet
- `voorwaarden.niet_voldaan`: voorwaarden waaraan de burger NIET voldoet
- `voorwaarden.onbekend_gegevens_ontbreken`: voorwaarden die niet beoordeeld konden worden
- `feiten_gebruikt`: profielwaarden die direct gebruikt zijn bij het beoordelen van een voorwaarde — gebruik ALLEEN deze feiten in je uitleg
- `feiten_context`: overige profielwaarden, aanwezig als achtergrondinformatie — noem deze NIET in je uitleg
- `berekend_bedrag`: het berekende bedrag (indien van toepassing)
- `wettelijke_grondslag`: lijst van artikelen waarop de beslissing gebaseerd is
- `relaties`: welke feiten welke voorwaarden beïnvloeden

Jouw taak: schrijf een korte uitleg voor de burger op basis van UITSLUITEND de informatie in de graph. Gebruik taalniveau B1: korte zinnen, gewone woorden, geen vakjargon.

ABSOLUTE REGELS:
- Gebruik UITSLUITEND `feiten_gebruikt` voor concrete waarden — noem NOOIT iets uit `feiten_context`
- Schrijf ALTIJD in de u-vorm — gebruik NOOIT "hij/zij/men" of de naam als onderwerp
- Als `requirements_voldaan` true is maar `berekend_bedrag` 0 euro: schrijf dat u aan de basisvoorwaarden voldoet (noem welke), maar dat het berekende bedrag uitkomt op 0 euro. Leg de reden uit op basis van de feiten in `feiten_gebruikt` (bijv. inkomen boven drempel, vermogen boven grens). Schrijf NOOIT "U heeft recht op X" en zaai NOOIT twijfel over de voldane voorwaarden.
- Voorwaarden in `voldaan` zijn bevestigd — zaai NOOIT twijfel over een voldane voorwaarde
- Als `onbekend_gegevens_ontbreken` leeg is: voeg GEEN zin toe over ontbrekende gegevens
- Gebruik het woord "euro", nooit het euro-teken
- Gebruik de `relaties` om causaliteit uit te leggen (welk feit leidde tot welke conclusie)

AFWIJZING DOOR ONTBREKENDE GEGEVENS:
- Als `requirements_voldaan` false is EN `niet_voldaan` leeg is EN `onbekend_gegevens_ontbreken` niet leeg is: de aanvraag kon NIET beoordeeld worden omdat vereiste gegevens ontbreken
- Leg in dat geval uit welke gegevens ontbreken (de condities in `onbekend_gegevens_ontbreken`) — schrijf NIET dat de burger iets heeft overtreden of niet voldoet
- Noem in dat geval de condities in `voldaan` NIET in de uitleg

VERBODEN:
- GEEN briefopmaak (aanhef, ondertekening)
- GEEN vragen aan de lezer
- GEEN technische termen of JSON-sleutels
- GEEN informatie buiten de graph
- GEEN feiten uit `feiten_context` noemen"""


def create_graph_prompt(graph_json: dict, person_name: str) -> str:
    contact = ""
    law_name = graph_json.get("regeling", "")
    if "toeslagen" in law_name.lower() or "toeslag" in law_name.lower():
        contact = "\n\nVoor meer informatie: www.toeslagen.nl"
    elif "gemeente" in law_name.lower():
        contact = "\n\nVoor meer informatie: neem contact op met uw gemeente."
    elif "svb" in law_name.lower() or "aow" in law_name.lower():
        contact = "\n\nVoor meer informatie: www.svb.nl"

    graph_str = json.dumps(graph_json, ensure_ascii=False, indent=2)

    return (
        f"Schrijf een korte uitleg voor {person_name} op basis van de volgende beslissingsgraph.\n\n"
        f"```json\n{graph_str}\n```"
        f"{contact}"
    )


# ---------------------------------------------------------------------------
# LLM call (Anthropic only — large models)
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, tuple[str, str]] = {
    # Anthropic cloud models
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus-4-6":   ("anthropic", "claude-opus-4-6"),
    "claude-haiku-4-5":  ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet":            ("anthropic", "claude-sonnet-4-6"),
    "opus":              ("anthropic", "claude-opus-4-6"),
    "haiku":             ("anthropic", "claude-haiku-4-5-20251001"),
    # Local Ollama models
    "llama3.2":  ("ollama", "llama3.2:3b"),
    "llama3.1":  ("ollama", "llama3.1:8b"),
    "llama3.3":  ("ollama", "llama3.3:70b"),
    "mistral":   ("ollama", "mistral:7b"),
    "deepseek":  ("ollama", "deepseek-r1:8b"),
    "gemma2":    ("ollama", "gemma2:9b"),
}


def call_llm(model_key: str, system_prompt: str, user_prompt: str, api_key: str | None = None) -> tuple[str, dict]:
    """Call an Anthropic or Ollama model and return (text, usage_dict)."""
    provider, model_id = AVAILABLE_MODELS.get(model_key, ("anthropic", model_key))

    if provider == "ollama":
        import ollama
        response = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.3, "num_predict": 1500},
        )
        text = response["message"]["content"].replace("\ufffd", "")
        return text, {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_id,
        max_tokens=1500,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def generate_graph_explanation(
    decision_extractor: DecisionGraphExtractor,
    person_name: str,
    model: str,
    api_key: str | None = None,
) -> tuple[str, dict, dict, str]:
    """
    Serialise the graph, call the LLM, return (explanation, graph_json, usage, prompt_used).
    """
    graph = decision_extractor.graph  # already extracted by caller
    graph_json = serialize_graph(graph, decision_extractor, person_name)
    prompt = create_graph_prompt(graph_json, person_name)

    explanation, usage = call_llm(model, GRAPHRAG_SYSTEM_PROMPT, prompt, api_key)
    return explanation, graph_json, usage, prompt


def run_graph_for_law(
    law: str,
    model: str,
    profiles_filter: list[str] | None,
    output_file: Path,
    api_key: str | None = None,
    verbose: bool = True,
    save_graphs: bool = False,
    graphs_dir: Path | None = None,
    resume: bool = False,
) -> None:
    """Run the graph pipeline for one law, write results to JSONL."""
    all_profiles = load_profiles()
    profiles_to_process = profiles_filter or list(all_profiles.keys())
    law_yaml = load_law_yaml(law)
    _, model_id = AVAILABLE_MODELS.get(model, ("anthropic", model))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file = output_file.parent / f"cache_{law}.json"

    # Load calculation cache (persisted after each profile so partial runs survive)
    cached: dict[str, dict] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        if verbose:
            print(f"  Loaded calculation cache: {len(cached)} profiles from {cache_file.name}", file=sys.stderr)

    # Resume: load already-done profiles from existing JSONL output
    done_bsns: set[str] = set()
    if resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("record_type") == "explanation" and "profile" in r:
                        done_bsns.add(str(r["profile"]))
                except json.JSONDecodeError:
                    pass
        if verbose:
            print(f"  Resuming: {len(done_bsns)} profiles already done, skipping them.", file=sys.stderr)

    file_mode = "a" if (resume and done_bsns) else "w"

    total = len(profiles_to_process)
    processed = 0

    with open(output_file, file_mode, encoding="utf-8") as out_f:
        if file_mode == "w":
            metadata = {
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_id,
                "provider": "anthropic",
                "law": law,
                "profiles_count": total,
                "graph_type": "decision",
                "approach": "graph",
                "git_info": get_git_info(),
            }
            out_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for i, bsn in enumerate(profiles_to_process, 1):
            if bsn not in all_profiles:
                if verbose:
                    print(f"  [{i}/{total}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
                continue

            if bsn in done_bsns:
                if verbose:
                    print(f"  [{i}/{total}] {bsn} — already done (resume), skipping", file=sys.stderr)
                continue

            profile_data = all_profiles[bsn]
            person_name = profile_data.get("name", f"Burger {bsn}")

            if verbose:
                print(f"  [{i}/{total}] {bsn} ({person_name})...", file=sys.stderr)

            # Use cached calculation if available, otherwise run engine
            if bsn in cached:
                calc_result = cached[bsn]
                if verbose:
                    print("    Calculation: from cache (skipping engine)", file=sys.stderr)
            else:
                calc_result = run_calculation(law, bsn, law_yaml, profile_data)
                if calc_result is not None:
                    cached[bsn] = calc_result
                    with open(cache_file, "w", encoding="utf-8") as cf:
                        json.dump(cached, cf, ensure_ascii=False)

            if calc_result is None:
                if verbose:
                    print("    Calculation failed, skipping", file=sys.stderr)
                continue

            req_met = calc_result.get("requirements_met")
            if verbose:
                print(f"    requirements_met={req_met}", file=sys.stderr)

            decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
            graph = decision_extractor.extract()

            # Optional graph visualisation (same as extract.py)
            if save_graphs and graphs_dir:
                graphs_dir.mkdir(parents=True, exist_ok=True)
                graph_png = graphs_dir / f"{law}_{bsn}.png"
                try:
                    graph.visualize(output_path=str(graph_png), title=f"{law} – {person_name} ({bsn})")
                    if verbose:
                        print(f"    Graph saved: {graph_png.name}", file=sys.stderr)
                except Exception as viz_exc:
                    if verbose:
                        print(f"    Graph visualization skipped: {viz_exc}", file=sys.stderr)

            try:
                explanation, graph_json, usage, prompt_used = generate_graph_explanation(
                    decision_extractor, person_name, model, api_key
                )
            except Exception as exc:
                if verbose:
                    print(f"    LLM error: {exc}", file=sys.stderr)
                explanation = ""
                graph_json = {}
                usage = {}
                prompt_used = ""

            # Mirror the output structure of extract.py (graph approach)
            entry = {
                "record_type": "explanation",
                "graph_type": "decision",
                "approach": "graph",
                "law": law,
                "profile": bsn,
                "profile_name": person_name,
                "requirements_met": req_met,
                "law_output": calc_result.get("result", {}),
                "law_input": {
                    k.lstrip("$"): v
                    for k, v in calc_result.get("input_data", {}).items()
                },
                "explanation": explanation,
                "graph_json": graph_json,       # full serialised graph (replaces skeleton_used)
                "prompt_used": prompt_used,
                "model": model_id,
                "usage": usage,
                "evaluation_trace": decision_extractor.to_evaluation_trace(),
                "graph_stats": {
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                },
                "calculation_result": {
                    "requirements_met": req_met,
                    "output": calc_result.get("result", {}),
                },
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1

    if verbose:
        print(f"Completed graph approach! {processed} profiles.", file=sys.stderr)
        print(f"Output saved to: {output_file}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph: pass decision graph directly to large LLM (Claude/GPT-4)."
    )
    parser.add_argument(
        "--law", nargs="+", required=True,
        help="Law service name(s), e.g. zorgtoeslag bijstand alcoholwet",
    )
    parser.add_argument(
        "--model", default="sonnet",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use (default: sonnet). Cloud: sonnet/opus/haiku. Local: llama3.1/llama3.2/mistral/deepseek/gemma2",
    )
    parser.add_argument(
        "--profiles", nargs="+", default=None,
        help="Specific BSNs to process (default: all profiles)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (default: from ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--save-graphs", action="store_true",
        help="Save graph visualisations as PNG (requires networkx + matplotlib)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run: skip already-processed profiles and append to existing output file(s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to existing run directory to resume into (use together with --resume).",
    )
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    provider, _ = AVAILABLE_MODELS.get(args.model, ("anthropic", args.model))
    if provider == "anthropic" and not api_key:
        print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    laws = args.law
    model_key = args.model
    profiles_filter = args.profiles
    verbose = not args.quiet
    save_graphs = args.save_graphs
    do_resume = args.resume

    # Build output folder: use --output if resuming, else create new timestamped folder
    if do_resume and args.output:
        run_dir = Path(args.output) / model_key
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_laws = len(laws)
        n_profiles = len(profiles_filter) if profiles_filter else "all"
        folder_name = f"{timestamp}_{n_laws}laws_{n_profiles}profiles_graph"
        base_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
        run_dir = base_dir / folder_name / model_key

    if verbose:
        _, model_id = AVAILABLE_MODELS[model_key]
        print(f"Graph run: {n_laws} laws, {n_profiles} profiles, model={model_id}", file=sys.stderr)
        print(f"Output dir: {run_dir}", file=sys.stderr)

    for law in laws:
        if verbose:
            print(f"\n=== {law.upper()} ===", file=sys.stderr)
        output_file = run_dir / f"graph_{model_key}_{law}.jsonl"
        graphs_dir = run_dir / "graphs" / law if save_graphs else None
        run_graph_for_law(
            law=law,
            model=model_key,
            profiles_filter=profiles_filter,
            output_file=output_file,
            api_key=api_key,
            verbose=verbose,
            save_graphs=save_graphs,
            graphs_dir=graphs_dir,
            resume=do_resume,
        )


if __name__ == "__main__":
    main()
