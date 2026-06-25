#!/usr/bin/env python3
"""
Combined LLM explanation extraction script for machine law.

Supports two approaches selectable via --approach:
  graph  — Decision graph / skeleton approach (constrained, law-agnostic)
           The LLM receives a pre-filled skeleton derived from a focused decision
           subgraph. Hallucination is minimised; output is short and precise.
  open   — Open prompt approach (MCPLawConnector, law-agnostic)
           The LLM receives the raw calculation result and a free-form prompt.
           More verbose output, works for any law.
  both   — Runs both approaches and writes both files into a shared timestamped folder.

Original scripts are preserved in scripts/archive/:
  scripts/archive/extraction_graph.py       — archive of original graph approach
  scripts/archive/extract_explanations.py   — standalone open approach

Usage (run from project root):
    # Graph approach for all profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1

    # Open approach, specific law:
    uv run python analysis/llm_explanations/scripts/extract.py --approach open --model llama3.1 --laws zorgtoeslag

    # Both approaches, specific profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --model haiku --profiles 174760992

    # Multiple models at once (creates a shared folder):
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --models llama3.1 haiku mistral --profiles 174760992

    # Ollama models (no API key needed):
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model mistral
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
# scripts/extract.py → scripts/ → llm_explanations/ → analysis/ → root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling imports within scripts/

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Per-law extraction module dispatch — all laws use extraction_generic
# ---------------------------------------------------------------------------

_DEFAULT_EXTRACTOR = "extraction_generic"


def _get_extractor(law: str):
    """Import the extraction module for a given law (always extraction_generic)."""
    import importlib
    return importlib.import_module(_DEFAULT_EXTRACTOR)


# ---------------------------------------------------------------------------
# Open-approach helpers (inlined from extract_explanations.py)
# ---------------------------------------------------------------------------
import json as _json
import os as _os

from extraction_generic import (  # noqa: E402
    AVAILABLE_MODELS,
    get_git_info,
    load_profiles,
)

SYSTEM_PROMPT = "Je bent een behulpzame assistent die Nederlandse burgers helpt met vragen over overheidsregelingen. Geef duidelijke, begrijpelijke uitleg in eenvoudig Nederlands (B1-niveau)."
DEFAULT_MODEL = "haiku"


def _call_llm(
    model_id: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Call LLM (Ollama, Anthropic, or OpenAI) and return (text, usage_dict)."""
    if provider == "ollama":
        import ollama
        response = ollama.chat(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            options={"temperature": 0.3, "num_predict": 1500},
        )
        return response["message"]["content"], {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }
    if provider == "openai":
        import openai as _openai
        _oai_key = api_key or _os.environ.get("OPENAI_API_KEY")
        oai_client = _openai.OpenAI(api_key=_oai_key)
        oai_resp = oai_client.chat.completions.create(
            model=model_id, max_tokens=1500, temperature=0.3,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        text = oai_resp.choices[0].message.content or ""
        return text, {
            "input_tokens": oai_resp.usage.prompt_tokens if oai_resp.usage else 0,
            "output_tokens": oai_resp.usage.completion_tokens if oai_resp.usage else 0,
        }
    actual_key = api_key or _os.environ.get("ANTHROPIC_API_KEY")
    import anthropic
    client = anthropic.Anthropic(api_key=actual_key)
    response = client.messages.create(
        model=model_id, max_tokens=1500, temperature=0.3,
        system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _create_flat_prompt(decision_extractor, person_name: str, law_name: str) -> str:
    """
    Build a plain-text prompt with the same engine output as the graph approach,
    but without any graph structure. Isolates graph structure contribution from
    mere information availability as an ablation baseline.
    """
    trace = decision_extractor.to_evaluation_trace()
    outcome = trace.get("outcome", "onbekend")
    amount_euro = trace.get("amount_euro")
    decisive_label = (trace.get("decisive_condition") or {}).get("label", "")
    key_facts = trace.get("key_facts", {})

    lines = [
        f"Ik heb een berekening uitgevoerd voor de regeling '{law_name}'.",
        "",
        f"Naam: {person_name}",
        f"Uitkomst: {outcome}",
        "",
    ]

    if decisive_label:
        lines += ["Doorslaggevende voorwaarde:", f"  {decisive_label}", ""]

    # Filter internal calculation constants — only show citizen attributes
    _SKIP_PREFIXES = ("prev ", "minimum ", "maximum ", "percentage ", "perc ")
    _SKIP_CONTAINS = ("drempelinkomen", "vermogensgrens", "normpremie", "standaardpremie",
                      "basispremie", "grens alleenstaande", "grens partner")

    if key_facts:
        lines.append("Relevante feiten uit het profiel:")
        for _field, info in key_facts.items():
            label = info.get("label", _field)
            label_lower = label.lower()
            if any(label_lower.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if any(s in label_lower for s in _SKIP_CONTAINS):
                continue
            # Use euro value if available, otherwise fall back to raw value
            value = info.get("value_euro")
            if value is not None:
                dutch = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                lines.append(f"  - {label}: €{dutch}")
            else:
                raw = info.get("display_value") or info.get("value", "")
                # skip bare ratios / internal floats < 1
                if isinstance(raw, float) and 0 < raw < 1:
                    continue
                if raw not in (None, ""):
                    lines.append(f"  - {label}: {raw}")
        lines.append("")

    if amount_euro is not None and amount_euro > 0:
        dutch = f"{amount_euro:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        lines += [f"Berekend bedrag: €{dutch} per jaar", ""]

    lines += [
        "Geef een duidelijke uitleg in eenvoudig Nederlands (B1-niveau) over:",
        "1. WAAROM deze burger wel of niet in aanmerking komt voor deze regeling",
        "2. Welke factoren uit het profiel hebben geleid tot dit resultaat",
        "3. Wat de burger eventueel kan doen als ze niet in aanmerking komen",
    ]
    return "\n".join(lines)


def _create_open_prompt(service_name: str, result: dict, profile: dict, bsn: str) -> str:
    requirements_met = result.get("requirements_met", False)
    missing_required = result.get("missing_required", False)
    output = result.get("result", {})
    explanation = result.get("explanation", "")
    return (
        f"Ik heb zojuist een berekening uitgevoerd voor de regeling '{service_name}'.\n\n"
        f"Burgerprofiel:\n"
        f"- Naam: {profile.get('name', 'Onbekend')}\n"
        f"- Beschrijving: {profile.get('description', 'Geen beschrijving')}\n"
        f"- BSN: {bsn}\n\n"
        f"Resultaat van de berekening:\n"
        f"- Voldoet aan voorwaarden: {'Ja' if requirements_met else 'Nee'}\n"
        f"- Ontbrekende essentiële gegevens: {'Ja' if missing_required else 'Nee'}\n"
        f"- Uitkomst: {_json.dumps(output, indent=2, ensure_ascii=False)}\n\n"
        f"Korte uitleg van het systeem: {explanation}\n\n"
        f"Geef een duidelijke uitleg in eenvoudig Nederlands (B1-niveau) over:\n"
        f"1. WAAROM deze burger wel of niet in aanmerking komt voor deze regeling\n"
        f"2. Welke factoren uit het profiel van de burger hebben geleid tot dit resultaat\n"
        f"3. Wat de burger eventueel kan doen als ze niet in aanmerking komen\n\n"
        f"Let op: bedragen in de uitkomst zijn in eurocenten, deel door 100 voor euros."
    )


def _load_profiles_raw(profiles_path: str = "data/profiles.yaml") -> tuple[dict, dict]:
    import yaml as _yaml
    with open(profiles_path) as f:
        raw_data = _yaml.safe_load(f)
    return raw_data.get("profiles", {}), raw_data


def precompute_open_entries(
    laws_filter: list[str] | None = None,
    profiles_filter: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[str], dict]:
    """Compute law calculations for all profile × law combinations once."""
    from explain.mcp_connector import MCPLawConnector
    from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

    services = get_machine_service()
    connector = MCPLawConnector(services, get_case_manager(), get_claim_manager())
    profiles, raw_profiles_data = _load_profiles_raw()

    available_laws = connector.registry.get_service_names()
    if laws_filter:
        available_laws = [law for law in available_laws if law in laws_filter]
    if profiles_filter:
        profiles = {bsn: p for bsn, p in profiles.items() if bsn in profiles_filter}

    if verbose:
        print(f"Loaded {len(profiles)} profiles, {len(available_laws)} laws", file=sys.stderr)

    total = len(profiles) * len(available_laws)
    entries: list[dict] = []

    for current, (bsn, profile) in enumerate(profiles.items(), 1):
        full_profile = raw_profiles_data.get("profiles", {}).get(bsn, profile)
        for law_name in available_laws:
            if verbose:
                print(f"[{current}/{total}] {law_name} / {profile.get('name', bsn)}...", file=sys.stderr)

            entry: dict = {
                "bsn": bsn, "profile": profile, "law_name": law_name,
                "profile_name": profile.get("name", "Unknown"),
                "calc_result": None, "prompt": None, "error": None,
                "requirements_met": None, "calculation_result": None,
            }
            try:
                service = connector.registry.get_service(law_name)
                if not service:
                    entry["error"] = "Service not found"
                    entries.append(entry)
                    continue
                extra_params: dict = {}
                for svc_name in ["KVK", "GEMEENTE_ROTTERDAM", "GEMEENTE_AMSTERDAM", "GEMEENTE_DEN_HAAG",
                                  "GEMEENTE_EINDHOVEN", "GEMEENTE_GRONINGEN", "GEMEENTE_MAASTRICHT", "GEMEENTE_UTRECHT"]:
                    rows = full_profile.get("sources", {}).get(svc_name, {}).get("leidinggevenden", [])
                    if isinstance(rows, list) and rows and rows[0].get("kvk_nummer"):
                        extra_params["KVK_NUMMER"] = str(rows[0]["kvk_nummer"])
                        break
                calc_result = service.execute(bsn, extra_params)
                if "error" in calc_result:
                    entry["error"] = calc_result["error"]
                    entries.append(entry)
                    continue
                entry["calc_result"] = calc_result
                entry["requirements_met"] = calc_result.get("requirements_met")
                entry["calculation_result"] = {
                    "requirements_met": calc_result.get("requirements_met"),
                    "missing_required": calc_result.get("missing_required"),
                    "missing_fields": calc_result.get("missing_fields", []),
                    "output": calc_result.get("result", {}),
                    "input_data": calc_result.get("input_data", {}),
                    "system_explanation": calc_result.get("explanation", ""),
                }
                entry["prompt"] = _create_open_prompt(law_name, calc_result, profile, bsn)
            except Exception as e:
                entry["error"] = str(e)
            entries.append(entry)

    return entries, available_laws, raw_profiles_data


def extract_explanations(
    api_key: str | None = None,
    laws_filter: list[str] | None = None,
    profiles_filter: list[str] | None = None,
    output_file: str = "explanations_output.jsonl",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    precomputed: list[dict] | None = None,
    available_laws: list[str] | None = None,
    raw_profiles_data: dict | None = None,
) -> list[dict]:
    """Extract LLM explanations for all profile × law combinations (open approach)."""
    model_info = AVAILABLE_MODELS[model]
    model_id = model_info["id"]
    provider = model_info["provider"]
    actual_api_key = api_key or _os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else None

    from web.dependencies import TODAY

    entries, laws_used, raw_data = (
        (precomputed, available_laws or [], raw_profiles_data or {})
        if precomputed is not None
        else precompute_open_entries(laws_filter=laws_filter, profiles_filter=profiles_filter, verbose=verbose)
    )

    results = []
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_json.dumps({
            "record_type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "model": model_id, "provider": provider,
            "law": laws_filter[0] if laws_filter and len(laws_filter) == 1 else None,
            "profiles_count": len({e["bsn"] for e in entries}),
            "approach": "open_prompt",
            "git_info": get_git_info(),
            "reference_date": TODAY,
            "filters": {"laws_filter": laws_filter, "profiles_filter": profiles_filter},
        }, ensure_ascii=False) + "\n")

        for i, entry in enumerate(entries, 1):
            bsn, law_name = entry["bsn"], entry["law_name"]
            record: dict = {
                "record_type": "explanation", "approach": "open", "graph_type": None,
                "law": law_name, "profile": bsn, "profile_name": entry["profile_name"],
                "requirements_met": entry["requirements_met"],
                "explanation": None, "skeleton_used": None,
                "prompt_used": entry["prompt"], "model": model_id,
                "usage": None, "graph_stats": None,
                "calculation_result": entry["calculation_result"],
            }
            if entry.get("error"):
                record["error"] = entry["error"]
            elif entry.get("prompt"):
                try:
                    text, usage = _call_llm(model_id, provider, SYSTEM_PROMPT, entry["prompt"], actual_api_key)
                    record["explanation"] = text
                    record["usage"] = usage
                except Exception as e:
                    record["error"] = str(e)
            else:
                record["error"] = "No prompt (calculation failed)"
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

    if verbose:
        print(f"\nOpen approach complete: {len(results)} records → {output_path}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _run_label(model: str, law: str | None, profiles: list[str] | None) -> str:
    """Shared label used in filenames and folder names."""
    law_part = law or "all-laws"
    if not profiles:
        profile_part = "all-profiles"
    elif len(profiles) == 1:
        profile_part = profiles[0]
    else:
        profile_part = f"{len(profiles)}profiles"
    return f"{model}_{law_part}_{profile_part}"


def generate_output_filename(approach: str, model: str, law: str | None, profiles: list[str] | None) -> str:
    """Generate a timestamped output filename (used for single-approach runs)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{_run_label(model, law, profiles)}_{approach}.jsonl"
    return str(OUTPUT_DIR / filename)


def generate_both_output_dir(model: str, law: str | None, profiles: list[str] | None) -> Path:
    """Create and return a timestamped folder for single-model --approach both runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{_run_label(model, law, profiles)}_both"
    folder = OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def generate_multi_output_dir(approach: str, law: str | None, profiles: list[str] | None) -> Path:
    """Create and return a timestamped top-level folder for multi-model runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    law_part = law or "all-laws"
    profile_part = "all-profiles" if not profiles else (profiles[0] if len(profiles) == 1 else f"{len(profiles)}profiles")
    folder_name = f"{timestamp}_multi_{law_part}_{profile_part}_{approach}"
    folder = OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# Graph approach runner
# ---------------------------------------------------------------------------

def precompute_graph_entries(
    law: str,
    profiles_filter: list[str] | None,
    save_graphs: bool = False,
    graphs_dir: Path | None = None,
    verbose: bool = True,
    cache_file: Path | None = None,
    profiles_file: str | None = None,
) -> list[dict]:
    """Compute calc_result + decision graph + skeleton for every profile once.

    Returns a list of entry dicts (one per profile) that can be reused across
    multiple models without re-running the calculation or graph extraction.

    If cache_file is given:
    - On first run: saves calc_results to JSON after computing each profile.
    - On resume: loads cached calc_results and skips the engine call entirely;
      only rebuilds the in-memory extractor/graph objects (fast, no API calls).
    """
    extractor_mod = _get_extractor(law)
    DecisionGraphExtractor = extractor_mod.DecisionGraphExtractor
    load_law_yaml = extractor_mod.load_law_yaml
    run_calculation = extractor_mod.run_calculation

    all_profiles = load_profiles(profiles_file) if profiles_file else load_profiles()
    profiles_to_process = profiles_filter or list(all_profiles.keys())
    law_yaml = load_law_yaml(law)

    # Load existing cache if present
    cached: dict[str, dict] = {}  # bsn → calc_result
    if cache_file and cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        if verbose:
            print(f"  Loaded calculation cache: {len(cached)} profiles from {cache_file.name}", file=sys.stderr)

    total = len(profiles_to_process)
    entries: list[dict] = []

    for i, bsn in enumerate(profiles_to_process, 1):
        if bsn not in all_profiles:
            if verbose:
                print(f"  [{i}/{total}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
            continue

        profile_data = all_profiles[bsn]
        person_name = profile_data.get("name", f"Burger {bsn}")

        if bsn in cached:
            calc_result = cached[bsn]
            if verbose:
                print(f"  [{i}/{total}] {bsn} — from cache (skipping engine)", file=sys.stderr)
        else:
            if verbose:
                print(f"  [{i}/{total}] Processing {bsn}...", file=sys.stderr)
            calc_result = run_calculation(law, bsn, law_yaml, profile_data)
            if calc_result and verbose:
                req_met = calc_result.get("requirements_met", False)
                print(f"    Calculation: requirements_met={req_met}", file=sys.stderr)
            # Save to cache immediately (so partial runs are also cached)
            if cache_file:
                cached[bsn] = calc_result
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False)

        decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
        graph = decision_extractor.extract()

        if save_graphs and graphs_dir:
            graphs_dir.mkdir(parents=True, exist_ok=True)
            graph_png = graphs_dir / f"{law}_{bsn}.png"
            try:
                graph.visualize(output_path=str(graph_png), title=f"{law} – {person_name} ({bsn})")
                if verbose:
                    print(f"    Graph saved: {graph_png.name}", file=sys.stderr)
            except ImportError:
                if verbose:
                    print("    Graph visualization skipped (networkx/matplotlib not available)", file=sys.stderr)
            except Exception as viz_exc:
                if verbose:
                    print(f"    Graph visualization failed: {viz_exc}", file=sys.stderr)

        calc_output = calc_result.get("result", {}) if calc_result else {}
        profile_vals = decision_extractor.profile_values

        entries.append({
            "bsn": bsn,
            "person_name": person_name,
            "decision_extractor": decision_extractor,
            "graph": graph,
            "calc_result": calc_result,
            "calc_output": calc_output,
            "profile_vals": profile_vals,
        })

    return entries


def run_graph_approach(
    model: str,
    law: str,
    profiles_filter: list[str] | None,
    output_file: str,
    api_key: str | None = None,
    verbose: bool = True,
    resume: bool = False,
    save_graphs: bool = False,
    precomputed: list[dict] | None = None,
) -> list[dict]:
    """Run the graph approach LLM step for all (or selected) profiles.

    If precomputed is provided, skips calculation/graph and goes straight to LLM.
    If resume=True and the output file already exists, already-processed profiles
    are skipped and new results are appended to the existing file.
    """
    extractor_mod = _get_extractor(law)
    generate_decision_explanation = extractor_mod.generate_decision_explanation

    model_config = AVAILABLE_MODELS[model]
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build entries: use precomputed if available, otherwise compute now
    if precomputed is not None:
        entries = precomputed
    else:
        entries = precompute_graph_entries(
            law=law,
            profiles_filter=profiles_filter,
            save_graphs=save_graphs,
            graphs_dir=OUTPUT_DIR / "graphs" if save_graphs else None,
            verbose=verbose,
        )

    # --- Resume: skip already-completed profiles ---
    already_done: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("record_type") == "explanation" and "profile" in r:
                        already_done.add(r["profile"])
                except json.JSONDecodeError:
                    pass
        if verbose:
            print(f"  Resuming: {len(already_done)} profiles already done, skipping them.", file=sys.stderr)
        entries = [e for e in entries if e["bsn"] not in already_done]

    total = len(entries)
    file_mode = "a" if (resume and already_done) else "w"
    results: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    with open(output_path, file_mode, encoding="utf-8") as f:
        if file_mode == "w":
            metadata = {
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_config["id"],
                "provider": model_config.get("provider", "anthropic"),
                "law": law,
                "profiles_count": total,
                "graph_type": "decision",
                "approach": "graph",
                "git_info": get_git_info(),
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        for i, entry in enumerate(entries, 1):
            bsn = entry["bsn"]
            person_name = entry["person_name"]
            decision_extractor = entry["decision_extractor"]
            graph = entry["graph"]
            calc_result = entry["calc_result"]
            calc_output = entry["calc_output"]
            profile_vals = entry["profile_vals"]

            if verbose:
                print(f"  [{i}/{total}] LLM for {bsn} ({model})...", file=sys.stderr)

            try:
                result = generate_decision_explanation(
                    decision_extractor=decision_extractor,
                    person_name=person_name,
                    api_key=api_key,
                    model=model,
                )

                total_input_tokens += result["usage"]["input_tokens"]
                total_output_tokens += result["usage"]["output_tokens"]

                record = {
                    "record_type": "explanation",
                    "graph_type": "decision",
                    "approach": "graph",
                    "law": law,
                    "profile": bsn,
                    "profile_name": person_name,
                    "requirements_met": decision_extractor.effective_requirements_met,
                    "law_output": calc_output,
                    "law_input": {k: v["value"] for k, v in profile_vals.items()},
                    "explanation": result["explanation"],
                    "skeleton_used": result["skeleton_used"],
                    "prompt_used": result["prompt_used"],
                    "model": result["model"],
                    "usage": result["usage"],
                    "evaluation_trace": decision_extractor.to_evaluation_trace(),
                    "graph_stats": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
                    "calculation_result": {
                        "requirements_met": calc_result.get("requirements_met") if calc_result else None,
                        "output": calc_output,
                    } if calc_result else None,
                }

            except Exception as e:
                if verbose:
                    print(f"  Exception for {bsn}: {e}", file=sys.stderr)
                record = {
                    "record_type": "explanation",
                    "graph_type": "decision",
                    "approach": "graph",
                    "law": law,
                    "profile": bsn,
                    "profile_name": person_name,
                    "error": str(e),
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

    if verbose:
        print(f"\nCompleted graph approach! {len(results)} profiles.", file=sys.stderr)
        print(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output", file=sys.stderr)
        print(f"Output saved to: {output_path.absolute()}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Flat-context approach runner
# ---------------------------------------------------------------------------

def run_flat_approach(
    model: str,
    law: str,
    profiles_filter: list[str] | None,
    output_file: str,
    api_key: str | None = None,
    verbose: bool = True,
    resume: bool = False,
    precomputed: list[dict] | None = None,
) -> list[dict]:
    """Run the flat-context ablation baseline.

    Same engine output as the graph approach (outcome, decisive condition, key facts,
    amount) but formatted as plain text without graph structure. Isolates whether
    the graph approach wins because of *structure* or merely because the decisive
    condition is made available to the LLM.

    Comparison:
        open  — raw calc output, decisive condition NOT explicitly labelled
        flat  — decisive condition + key facts as plain text, no graph structure
        graph — same info as flat PLUS graph structure, relations, legal articles
    """
    model_config = AVAILABLE_MODELS[model]
    model_id = model_config["id"]
    provider = model_config.get("provider", "anthropic")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = precomputed if precomputed is not None else precompute_graph_entries(
        law=law, profiles_filter=profiles_filter, verbose=verbose,
    )

    already_done: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("record_type") == "explanation" and "profile" in r:
                        already_done.add(r["profile"])
                except json.JSONDecodeError:
                    pass
        if verbose:
            print(f"  Resuming: {len(already_done)} profiles already done, skipping.", file=sys.stderr)
        entries = [e for e in entries if e["bsn"] not in already_done]

    total = len(entries)
    file_mode = "a" if (resume and already_done) else "w"
    results: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    with open(output_path, file_mode, encoding="utf-8") as f:
        if file_mode == "w":
            f.write(json.dumps({
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_id,
                "provider": provider,
                "law": law,
                "profiles_count": total,
                "approach": "flat",
                "git_info": get_git_info(),
            }, ensure_ascii=False) + "\n")

        for i, entry in enumerate(entries, 1):
            bsn = entry["bsn"]
            person_name = entry["person_name"]
            decision_extractor = entry["decision_extractor"]
            graph = entry["graph"]
            calc_output = entry["calc_output"]

            if verbose:
                print(f"  [{i}/{total}] LLM (flat) for {bsn} ({model})...", file=sys.stderr)

            flat_prompt = _create_flat_prompt(decision_extractor, person_name, law)

            record: dict = {
                "record_type": "explanation",
                "graph_type": None,
                "approach": "flat",
                "law": law,
                "profile": bsn,
                "profile_name": person_name,
                "requirements_met": decision_extractor.effective_requirements_met,
                "law_output": calc_output,
                "explanation": None,
                "skeleton_used": None,
                "prompt_used": flat_prompt,
                "model": model_id,
                "usage": None,
                "evaluation_trace": decision_extractor.to_evaluation_trace(),
                "graph_stats": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
            }

            try:
                text, usage = _call_llm(model_id, provider, SYSTEM_PROMPT, flat_prompt, api_key)
                record["explanation"] = text
                record["usage"] = usage
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
            except Exception as e:
                if verbose:
                    print(f"  Exception for {bsn}: {e}", file=sys.stderr)
                record["error"] = str(e)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

    if verbose:
        print(f"\nCompleted flat approach! {len(results)} profiles.", file=sys.stderr)
        print(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output", file=sys.stderr)
        print(f"Output saved to: {output_path.absolute()}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined LLM explanation extraction (open prompt + decision graph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Approaches:
  graph  Constrained skeleton from decision subgraph. Short, precise output.
         Runs for a single --law (default: zorgtoeslag). Works for any law.
  open   Free-form prompt with raw calculation result. Verbose, law-agnostic.
         Use --laws to filter (default: all laws).
  both   Runs graph then open; saves both files inside a shared timestamped folder.

Multiple models:
  Pass --models m1 m2 ... to run several models in sequence.
  All output goes into a single timestamped folder with one subfolder per model.

Resuming an interrupted run:
  Pass --resume with the same --output (or same folder) to skip already-done profiles
  and append new results. Works for graph approach.

Examples:
    # Graph approach, all profiles, llama:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1

    # Graph approach, specific profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model haiku --profiles 174760992 311508199

    # Open approach, zorgtoeslag only:
    uv run python analysis/llm_explanations/scripts/extract.py --approach open --model llama3.1 --laws zorgtoeslag

    # Both approaches:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --model llama3.1 --profiles 174760992

    # Multiple models:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --models llama3.1 haiku mistral --profiles 174760992
""",
    )

    parser.add_argument(
        "--approach",
        choices=["open", "graph", "flat", "both"],
        default="open",
        help=(
            "Extraction approach (default: open):\n"
            "  open  — free prompt with raw calc output; decisive condition NOT explicit\n"
            "  graph — constrained skeleton from decision graph\n"
            "  flat  — ablation baseline: same info as graph (decisive condition, key facts,\n"
            "           amount) as plain text, without graph structure\n"
            "  both  — runs graph + open (flat can be added with a separate run)"
        ),
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=["haiku"],
        metavar="MODEL",
        dest="models",
        help="One or more models to run (default: haiku). Multiple models create a shared folder.",
    )
    parser.add_argument(
        "--law",
        "--graph-laws",
        nargs="+",
        default=["zorgtoeslag"],
        metavar="LAW",
        dest="laws_graph",
        help="Law(s) for the graph approach (default: zorgtoeslag). Works for any law.",
    )
    parser.add_argument(
        "--laws",
        nargs="+",
        help="Law(s) for the open approach (default: all). Defaults to the same as --law if not set.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Specific BSN(s) to process (default: all profiles)",
    )
    parser.add_argument(
        "--profiles-file",
        default=None,
        help="Path to a profiles YAML file (default: data/profiles.yaml). Use a smaller file for faster startup.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (single model + single approach only).",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var). Not required for Ollama models.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run: skip already-processed profiles and append to existing output file(s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Resume into an existing run folder instead of creating a new timestamped one.",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Save decision graph visualizations as PNG files in output/graphs/.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    # Determine which models to run
    models_to_run: list[str] = args.models
    multi_model = len(models_to_run) > 1

    # Laws for each approach
    graph_laws: list[str] = args.laws_graph
    open_laws: list[str] | None = args.laws  # None = all laws

    do_graph = args.approach in ("graph", "both")
    do_open = args.approach in ("open", "both")
    do_flat = args.approach == "flat"

    # Label for folder names
    label_law = graph_laws[0] if len(graph_laws) == 1 else f"{len(graph_laws)}laws"
    label_profiles = "all-profiles" if not args.profiles else (args.profiles[0] if len(args.profiles) == 1 else f"{len(args.profiles)}profiles")

    # Use existing folder if --output-dir given, otherwise create a new timestamped one
    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"{timestamp}_{label_law}_{label_profiles}_{args.approach}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {run_dir}")

    # -------------------------------------------------------------------
    # Precompute ONCE (outside model loop) for both approaches
    # -------------------------------------------------------------------
    graph_precomputed: dict[str, list[dict]] = {}
    if do_graph or do_flat:
        for law in graph_laws:
            cache_path = run_dir / f"cache_{law}.json"
            if cache_path.exists():
                print(f"\nLoading cached calculations for law: {law} ({cache_path.name})")
            else:
                print(f"\nPrecomputing graph for law: {law}...")
            graph_precomputed[law] = precompute_graph_entries(
                law=law,
                profiles_filter=args.profiles,
                save_graphs=args.graphs,
                graphs_dir=run_dir / "graphs" / law if args.graphs else None,
                verbose=verbose,
                cache_file=cache_path,
                profiles_file=args.profiles_file,
            )

    open_entries: list[dict] = []
    open_laws_used: list[str] = []
    open_raw_data: dict = {}
    if do_open:
        open_filter = open_laws or graph_laws
        print(f"\nPrecomputing open calculations for laws: {open_filter}...")
        open_entries, open_laws_used, open_raw_data = precompute_open_entries(
            laws_filter=open_filter,
            profiles_filter=args.profiles,
            verbose=verbose,
        )

    for model in models_to_run:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")
        model_dir = run_dir / model
        model_dir.mkdir(exist_ok=True)

        # -------------------------------------------------------------------
        # Graph approach — LLM only (calc+graph already done above)
        # -------------------------------------------------------------------
        if do_graph:
            for law in graph_laws:
                output_graph = str(model_dir / f"graph_{model}_{law}.jsonl")
                print(f"Output file (graph, {law}): {output_graph}")
                run_graph_approach(
                    model=model,
                    law=law,
                    profiles_filter=args.profiles,
                    output_file=output_graph,
                    api_key=args.api_key,
                    verbose=verbose,
                    resume=args.resume,
                    save_graphs=False,  # already saved during precompute
                    precomputed=graph_precomputed[law],
                )

        # -------------------------------------------------------------------
        # Open approach — LLM only (calculations already done above)
        # -------------------------------------------------------------------
        if do_open:
            output_open = str(model_dir / f"open_{model}_{label_law}.jsonl")
            print(f"Output file (open): {output_open}")
            extract_explanations(
                api_key=args.api_key,
                laws_filter=open_laws or graph_laws,
                profiles_filter=args.profiles,
                output_file=output_open,
                model=model,
                verbose=verbose,
                precomputed=open_entries,
                available_laws=open_laws_used,
                raw_profiles_data=open_raw_data,
            )

        # -------------------------------------------------------------------
        # Flat approach — LLM only (uses same precomputed graph entries)
        # -------------------------------------------------------------------
        if do_flat:
            for law in graph_laws:
                output_flat = str(model_dir / f"flat_{model}_{law}.jsonl")
                print(f"Output file (flat, {law}): {output_flat}")
                run_flat_approach(
                    model=model,
                    law=law,
                    profiles_filter=args.profiles,
                    output_file=output_flat,
                    api_key=args.api_key,
                    verbose=verbose,
                    resume=args.resume,
                    precomputed=graph_precomputed[law],
                )


if __name__ == "__main__":
    main()
