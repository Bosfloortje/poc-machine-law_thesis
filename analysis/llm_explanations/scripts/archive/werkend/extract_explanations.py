#!/usr/bin/env python3
"""
Script to extract LLM-generated "waarom" explanations for machine law decisions.

This script iterates over profiles and laws, executes each combination,
and generates LLM explanations for the results. Output is saved to JSONL
(JSON Lines) format for analysis (e.g., for thesis research on LLM explanations).

Output format:
- First line: metadata record with version info for reproducibility
- Subsequent lines: one record per profile × law combination

Output columns match extraction_zorgtoeslag.py for unified analysis:
  record_type, graph_type, law, profile (BSN), profile_name,
  requirements_met, hoogte_toeslag, hoogte_toeslag_per_maand,
  inkomen, partner_inkomen, vermogen, leeftijd, heeft_partner, is_verzekerde,
  explanation, skeleton_used, prompt_used, model, usage, graph_stats,
  calculation_result

Usage (run from project root):
    # With Anthropic API key:
    export ANTHROPIC_API_KEY="your-key-here"
    uv run python analysis/llm_explanations/scripts/extract_explanations.py

    # Ollama models (no API key needed):
    uv run python analysis/llm_explanations/scripts/extract_explanations.py --model llama3.1
    uv run python analysis/llm_explanations/scripts/extract_explanations.py --model mistral

    # Limit to specific laws or profiles:
    uv run python analysis/llm_explanations/scripts/extract_explanations.py --laws zorgtoeslag huurtoeslag --profiles 100000001

    # Save to specific output file:
    uv run python analysis/llm_explanations/extract_explanations.py --output my_explanations.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root and web directory to path for proper imports
# scripts/extract_explanations.py → scripts/ → llm_explanations/ → analysis/ → root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def generate_output_filename(model: str, laws_filter: list[str] | None, profiles_filter: list[str] | None) -> str:
    """Generate a descriptive output filename, matching extraction_zorgtoeslag.py naming convention."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    law_part = laws_filter[0] if laws_filter and len(laws_filter) == 1 else "all-laws"

    if profiles_filter is None or len(profiles_filter) == 0:
        profile_part = "all-profiles"
    elif len(profiles_filter) == 1:
        profile_part = profiles_filter[0]
    else:
        profile_part = f"{len(profiles_filter)}profiles"

    filename = f"{timestamp}_{model}_{law_part}_{profile_part}_open.jsonl"
    return str(OUTPUT_DIR / filename)


def get_git_info() -> dict:
    """Get git commit information for reproducibility."""
    info = {}

    try:
        # Main repo commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_commit"] = result.stdout.strip()

        # Main repo branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_dirty"] = len(result.stdout.strip()) > 0

        # Law submodule commit
        law_path = PROJECT_ROOT / "submodules" / "regelrecht-laws"
        if law_path.exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=law_path,
            )
            if result.returncode == 0:
                info["regelrecht_laws_commit"] = result.stdout.strip()

            # Check for uncommitted changes in submodule
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=law_path,
            )
            if result.returncode == 0:
                info["regelrecht_laws_dirty"] = len(result.stdout.strip()) > 0

    except Exception as e:
        info["git_error"] = str(e)

    return info


def load_profiles_raw(profiles_path: str = "data/profiles.yaml") -> tuple[dict, dict]:
    """Load all profiles from the profiles.yaml file.

    Returns:
        Tuple of (profiles_dict, raw_yaml_data)
    """
    with open(profiles_path) as f:
        raw_data = yaml.safe_load(f)
    return raw_data.get("profiles", {}), raw_data


def get_discoverable_laws(services) -> dict[str, list[str]]:
    """Get all discoverable laws from the engine."""
    return services.get_discoverable_service_laws("CITIZEN")


def create_explanation_prompt(service_name: str, result: dict, profile: dict, bsn: str) -> str:
    """Create a prompt asking for an explanation of the result."""
    requirements_met = result.get("requirements_met", False)
    missing_required = result.get("missing_required", False)
    output = result.get("result", {})
    explanation = result.get("explanation", "")

    prompt = f"""Ik heb zojuist een berekening uitgevoerd voor de regeling '{service_name}'.

Burgerprofiel:
- Naam: {profile.get("name", "Onbekend")}
- Beschrijving: {profile.get("description", "Geen beschrijving")}
- BSN: {bsn}

Resultaat van de berekening:
- Voldoet aan voorwaarden: {"Ja" if requirements_met else "Nee"}
- Ontbrekende essentiële gegevens: {"Ja" if missing_required else "Nee"}
- Uitkomst: {json.dumps(output, indent=2, ensure_ascii=False)}

Korte uitleg van het systeem: {explanation}

Geef een duidelijke uitleg in eenvoudig Nederlands (B1-niveau) over:
1. WAAROM deze burger wel of niet in aanmerking komt voor deze regeling
2. Welke factoren uit het profiel van de burger hebben geleid tot dit resultaat
3. Wat de burger eventueel kan doen als ze niet in aanmerking komen

Let op: bedragen in de uitkomst zijn in eurocenten, deel door 100 voor euros."""

    return prompt


# Available models — matches extraction_zorgtoeslag.py for unified analysis
AVAILABLE_MODELS = {
    "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "description": "Fast and cheap, good for batch processing",
    },
    "sonnet": {
        "id": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "description": "Balanced performance and cost",
    },
    "opus": {
        "id": "claude-opus-4-6",
        "provider": "anthropic",
        "description": "Most capable, highest quality output",
    },
    "llama3.2": {
        "id": "llama3.2:3b",
        "provider": "ollama",
        "description": "Llama 3.2 3B via local Ollama - small, runs on most hardware (~2GB RAM)",
    },
    "llama3.1": {
        "id": "llama3.1:8b",
        "provider": "ollama",
        "description": "Llama 3.1 8B via local Ollama - balanced (~5GB RAM)",
    },
    "llama3.3": {
        "id": "llama3.3:70b",
        "provider": "ollama",
        "description": "Llama 3.3 70B via local Ollama - high quality, needs GPU (~40GB RAM)",
    },
    "mistral": {
        "id": "mistral:7b",
        "provider": "ollama",
        "description": "Mistral 7B via local Ollama - fast and capable (~4GB RAM)",
    },
    "deepseek": {
        "id": "deepseek-r1:8b",
        "provider": "ollama",
        "description": "DeepSeek R1 8B via local Ollama - reasoning model (~5GB RAM)",
    },
    "gemma2": {
        "id": "gemma2:9b",
        "provider": "ollama",
        "description": "Gemma 2 9B via local Ollama - Google model (~6GB RAM)",
    },
}

DEFAULT_MODEL = "haiku"

SYSTEM_PROMPT = "Je bent een behulpzame assistent die Nederlandse burgers helpt met vragen over overheidsregelingen. Geef duidelijke, begrijpelijke uitleg in eenvoudig Nederlands (B1-niveau)."


def call_llm(
    model_id: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Call the LLM and return (text, usage_dict).

    Returns:
        Tuple of (generated_text, usage_dict with input_tokens/output_tokens)
    """
    if provider == "ollama":
        import ollama

        response = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.3,
                "num_predict": 1500,
            },
        )
        text = response["message"]["content"]
        usage = {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }
        return text, usage

    else:  # anthropic
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=1500,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage


def _get_input_value(input_data: dict, key: str) -> int | float | bool | None:
    """Extract a resolved input value from calc_result['input_data'].

    input_data keys are prefixed with '$', e.g. '$INKOMEN'.
    """
    return input_data.get(f"${key}")


def precompute_open_entries(
    laws_filter: list[str] | None = None,
    profiles_filter: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[str], dict]:
    """Compute law calculations for all profile × law combinations once.

    Returns (entries, available_laws, raw_profiles_data).
    Each entry has everything needed except the LLM call.
    """
    from explain.mcp_connector import MCPLawConnector
    from web.dependencies import TODAY, get_case_manager, get_claim_manager, get_machine_service

    if verbose:
        print("Initializing services...")

    services = get_machine_service()
    case_manager = get_case_manager()
    claim_manager = get_claim_manager()
    mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

    profiles, raw_profiles_data = load_profiles_raw()
    if verbose:
        print(f"Loaded {len(profiles)} profiles")

    available_laws = mcp_connector.registry.get_service_names()
    if laws_filter:
        available_laws = [law for law in available_laws if law in laws_filter]
        if verbose:
            print(f"Filtered to laws: {available_laws}")

    if profiles_filter:
        profiles = {bsn: profile for bsn, profile in profiles.items() if bsn in profiles_filter}
        if verbose:
            print(f"Filtered to {len(profiles)} profiles")

    total = len(profiles) * len(available_laws)
    current = 0
    entries: list[dict] = []

    for bsn, profile in profiles.items():
        full_profile = raw_profiles_data.get("profiles", {}).get(bsn, profile)

        for law_name in available_laws:
            current += 1
            if verbose:
                print(f"\n[{current}/{total}] Calculating {law_name} for {profile.get('name', bsn)}...")

            entry: dict = {
                "bsn": bsn,
                "profile": profile,
                "law_name": law_name,
                "profile_name": profile.get("name", "Unknown"),
                "calc_result": None,
                "prompt": None,
                "error": None,
                # Pre-filled record fields
                "requirements_met": None,
                "hoogte_toeslag": None,
                "hoogte_toeslag_per_maand": None,
                "inkomen": None,
                "partner_inkomen": None,
                "vermogen": None,
                "leeftijd": None,
                "heeft_partner": None,
                "is_verzekerde": None,
                "calculation_result": None,
            }

            try:
                service = mcp_connector.registry.get_service(law_name)
                if not service:
                    entry["error"] = "Service not found"
                    entries.append(entry)
                    continue

                # Build extra params (e.g. KVK_NUMMER for business laws)
                extra_params: dict = {}
                profile_sources = full_profile.get("sources", {})
                for svc_name in ["KVK", "GEMEENTE_ROTTERDAM", "GEMEENTE_AMSTERDAM", "GEMEENTE_DEN_HAAG",
                                 "GEMEENTE_EINDHOVEN", "GEMEENTE_GRONINGEN", "GEMEENTE_MAASTRICHT", "GEMEENTE_UTRECHT"]:
                    rows = profile_sources.get(svc_name, {}).get("leidinggevenden", [])
                    if isinstance(rows, list) and rows and rows[0].get("kvk_nummer"):
                        extra_params["KVK_NUMMER"] = str(rows[0]["kvk_nummer"])
                        break

                calc_result = service.execute(bsn, extra_params)

                if "error" in calc_result:
                    entry["error"] = calc_result["error"]
                    entries.append(entry)
                    continue

                calc_output = calc_result.get("result", {})
                input_data = calc_result.get("input_data", {})

                entry["calc_result"] = calc_result
                entry["requirements_met"] = calc_result.get("requirements_met")
                entry["hoogte_toeslag"] = calc_output.get("hoogte_toeslag", 0) / 100 if calc_output.get("hoogte_toeslag") else 0
                entry["hoogte_toeslag_per_maand"] = entry["hoogte_toeslag"] / 12
                entry["inkomen"] = _get_input_value(input_data, "INKOMEN")
                entry["partner_inkomen"] = _get_input_value(input_data, "PARTNER_INKOMEN")
                entry["vermogen"] = _get_input_value(input_data, "VERMOGEN")
                entry["leeftijd"] = _get_input_value(input_data, "LEEFTIJD")
                entry["heeft_partner"] = _get_input_value(input_data, "HEEFT_PARTNER")
                entry["is_verzekerde"] = _get_input_value(input_data, "IS_VERZEKERDE")
                entry["calculation_result"] = {
                    "requirements_met": calc_result.get("requirements_met"),
                    "missing_required": calc_result.get("missing_required"),
                    "missing_fields": calc_result.get("missing_fields", []),
                    "output": calc_output,
                    "input_data": input_data,
                    "system_explanation": calc_result.get("explanation", ""),
                }
                entry["prompt"] = create_explanation_prompt(law_name, calc_result, profile, bsn)

            except Exception as e:
                if verbose:
                    print(f"  Exception: {e}")
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
    """
    Extract LLM explanations for all profile × law combinations.

    If precomputed is provided, skips calculation and uses those entries directly.
    """
    if model not in AVAILABLE_MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    model_info = AVAILABLE_MODELS[model]
    model_id = model_info["id"]
    provider = model_info["provider"]

    if provider == "anthropic":
        actual_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not actual_api_key:
            print("ERROR: No Anthropic API key found!")
            sys.exit(1)
    else:
        actual_api_key = None

    from web.dependencies import TODAY

    # Use precomputed entries or compute now
    if precomputed is not None:
        entries = precomputed
        laws_used = available_laws or []
        raw_data = raw_profiles_data or {}
    else:
        entries, laws_used, raw_data = precompute_open_entries(
            laws_filter=laws_filter,
            profiles_filter=profiles_filter,
            verbose=verbose,
        )

    if verbose:
        print(f"Using model: {model_id} ({model_info['description']})")

    git_info = get_git_info()
    results = []
    output_path = Path(output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        metadata = {
            "record_type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "provider": provider,
            "law": laws_filter[0] if laws_filter and len(laws_filter) == 1 else None,
            "profiles_count": len({e["bsn"] for e in entries}),
            "graph_type": None,
            "approach": "open_prompt",
            "git_info": git_info,
            "extraction_date": datetime.now().isoformat(),
            "reference_date": TODAY,
            "filters": {"laws_filter": laws_filter, "profiles_filter": profiles_filter},
            "total_laws": len(laws_used),
            "expected_records": len(entries),
            "global_services": raw_data.get("globalServices", {}),
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        for i, entry in enumerate(entries, 1):
            bsn = entry["bsn"]
            law_name = entry["law_name"]
            profile = entry["profile"]

            if verbose:
                print(f"\n[{i}/{len(entries)}] LLM for {law_name} / {profile.get('name', bsn)} ({model})...")

            record: dict = {
                "record_type": "explanation",
                "approach": "open",
                "graph_type": None,
                "law": law_name,
                "profile": bsn,
                "profile_name": entry["profile_name"],
                "requirements_met": entry["requirements_met"],
                "hoogte_toeslag": entry["hoogte_toeslag"],
                "hoogte_toeslag_per_maand": entry["hoogte_toeslag_per_maand"],
                "inkomen": entry["inkomen"],
                "partner_inkomen": entry["partner_inkomen"],
                "vermogen": entry["vermogen"],
                "leeftijd": entry["leeftijd"],
                "heeft_partner": entry["heeft_partner"],
                "is_verzekerde": entry["is_verzekerde"],
                "explanation": None,
                "skeleton_used": None,
                "prompt_used": entry["prompt"],
                "model": model_id,
                "usage": None,
                "graph_stats": None,
                "calculation_result": entry["calculation_result"],
            }

            if entry.get("error"):
                record["error"] = entry["error"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)
                continue

            if not entry.get("prompt"):
                record["error"] = "No prompt (calculation failed)"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)
                continue

            try:
                explanation, usage = call_llm(
                    model_id=model_id,
                    provider=provider,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=entry["prompt"],
                    api_key=actual_api_key,
                )
                record["explanation"] = explanation
                record["usage"] = usage
                if verbose:
                    print(f"  Requirements met: {entry['requirements_met']}")
            except Exception as e:
                if verbose:
                    print(f"  Exception: {e}")
                record["error"] = str(e)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Extraction complete! {len(results)} records")
        print(f"Output saved to: {output_path.absolute()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM explanations for machine law decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all combinations (uses haiku by default):
    uv run python analysis/llm_explanations/extract_explanations.py

    # Use a different Anthropic model:
    uv run python analysis/llm_explanations/extract_explanations.py --model sonnet
    uv run python analysis/llm_explanations/extract_explanations.py --model opus

    # Use a local Ollama model (no API key needed):
    uv run python analysis/llm_explanations/extract_explanations.py --model llama3.1
    uv run python analysis/llm_explanations/extract_explanations.py --model mistral

    # Extract specific laws only:
    uv run python analysis/llm_explanations/extract_explanations.py --laws zorgtoeslag huurtoeslag

    # Extract for specific profiles:
    uv run python analysis/llm_explanations/extract_explanations.py --profiles 100000001 100000002

    # Save to specific file:
    uv run python analysis/llm_explanations/extract_explanations.py --output my_results.jsonl

    # Quiet mode (less output):
    uv run python analysis/llm_explanations/extract_explanations.py --quiet
""",
    )

    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var). Not required for Ollama models.")
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}). Ollama models run locally without API key.",
    )
    parser.add_argument("--laws", nargs="+", help="Filter to specific laws (e.g., zorgtoeslag huurtoeslag)")
    parser.add_argument("--profiles", nargs="+", help="Filter to specific BSNs")
    parser.add_argument("--output", default=None, help="Output JSONL file path (default: auto-generated in analysis/llm_explanations/output/)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--list-laws", action="store_true", help="List available laws and exit")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")

    args = parser.parse_args()

    # List mode
    if args.list_laws or args.list_profiles:
        if args.list_profiles:
            profiles, _ = load_profiles_raw()
            print("Available profiles:")
            for bsn, profile in profiles.items():
                print(f"  {bsn}: {profile.get('name', 'Unknown')} - {profile.get('description', '')[:60]}...")
            print(f"\nTotal: {len(profiles)} profiles")

        if args.list_laws:
            from explain.mcp_connector import MCPLawConnector
            from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

            services = get_machine_service()
            case_manager = get_case_manager()
            claim_manager = get_claim_manager()
            mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

            print("\nAvailable laws:")
            for law_name in mcp_connector.registry.get_service_names():
                service = mcp_connector.registry.get_service(law_name)
                if service:
                    print(f"  {law_name}: {service.description} ({service.service_type})")

        return

    # Auto-generate output filename if not specified
    output_file = args.output or generate_output_filename(args.model, args.laws, args.profiles)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output file: {output_file}")

    # Run extraction
    extract_explanations(
        api_key=args.api_key,
        laws_filter=args.laws,
        profiles_filter=args.profiles,
        output_file=output_file,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
