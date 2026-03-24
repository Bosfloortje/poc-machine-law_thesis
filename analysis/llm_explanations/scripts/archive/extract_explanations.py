#!/usr/bin/env python3
"""
Script to extract LLM-generated "waarom" explanations for machine law decisions.

This script iterates over profiles and laws, executes each combination,
and generates LLM explanations for the results. Output is saved to JSONL
(JSON Lines) format for analysis (e.g., for thesis research on LLM explanations).

Output format:
- First line: metadata record with version info for reproducibility
- Subsequent lines: one record per profile × law combination

Output columns match extraction_graph.py for unified analysis:
  record_type, graph_type, law, profile (BSN), profile_name,
  requirements_met, hoogte_toeslag, hoogte_toeslag_per_maand,
  inkomen, partner_inkomen, vermogen, leeftijd, heeft_partner, is_verzekerde,
  explanation, skeleton_used, prompt_used, model, usage, graph_stats,
  calculation_result

Usage (run from project root):
    # With Anthropic API key:
    export ANTHROPIC_API_KEY="your-key-here"
    uv run python analysis/llm_explanations/extract_explanations.py

    # Ollama models (no API key needed):
    uv run python analysis/llm_explanations/extract_explanations.py --model llama3.1
    uv run python analysis/llm_explanations/extract_explanations.py --model mistral

    # Limit to specific laws or profiles:
    uv run python analysis/llm_explanations/extract_explanations.py --laws zorgtoeslag huurtoeslag --profiles 100000001

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
# From analysis/llm_explanations/ we need to go up two levels
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

OUTPUT_DIR = Path(__file__).parent / "output"


def generate_output_filename(model: str, laws_filter: list[str] | None, profiles_filter: list[str] | None) -> str:
    """Generate a descriptive output filename, matching extraction_graph.py naming convention."""
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


# Available models — matches extraction_graph.py for unified analysis
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


def extract_explanations(
    api_key: str | None = None,
    laws_filter: list[str] | None = None,
    profiles_filter: list[str] | None = None,
    output_file: str = "explanations_output.jsonl",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> list[dict]:
    """
    Extract LLM explanations for all profile × law combinations.

    Output columns match extraction_graph.py for unified analysis.

    Args:
        api_key: Anthropic API key (uses env var if not provided; not needed for Ollama)
        laws_filter: List of law names to include (None = all)
        profiles_filter: List of BSNs to include (None = all)
        output_file: Path to output JSONL file
        model: Model to use (haiku, sonnet, opus, llama3.1, mistral, ...)
        verbose: Print progress information

    Returns:
        List of explanation records
    """
    # Validate model choice
    if model not in AVAILABLE_MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    model_info = AVAILABLE_MODELS[model]
    model_id = model_info["id"]
    provider = model_info["provider"]

    # Only require API key for Anthropic models
    if provider == "anthropic":
        actual_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not actual_api_key:
            print("ERROR: No Anthropic API key found!")
            print("Set ANTHROPIC_API_KEY environment variable or use --api-key argument")
            sys.exit(1)
    else:
        actual_api_key = None  # Not needed for Ollama

    # Import after setting up environment
    from explain.mcp_connector import MCPLawConnector
    from web.dependencies import TODAY, get_case_manager, get_claim_manager, get_machine_service

    # Initialize services
    if verbose:
        print("Initializing services...")

    services = get_machine_service()
    case_manager = get_case_manager()
    claim_manager = get_claim_manager()

    # Initialize MCP connector
    mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

    # Load profiles with raw data for reproducibility
    profiles, raw_profiles_data = load_profiles_raw()
    if verbose:
        print(f"Loaded {len(profiles)} profiles")

    # Get available laws
    available_laws = mcp_connector.registry.get_service_names()
    if verbose:
        print(f"Available laws: {available_laws}")

    # Apply filters
    if laws_filter:
        available_laws = [law for law in available_laws if law in laws_filter]
        if verbose:
            print(f"Filtered to laws: {available_laws}")

    if profiles_filter:
        profiles = {bsn: profile for bsn, profile in profiles.items() if bsn in profiles_filter}
        if verbose:
            print(f"Filtered to {len(profiles)} profiles")

    if verbose:
        print(f"Using model: {model_id} ({model_info['description']})")

    # Collect git info
    git_info = get_git_info()

    # Results storage
    results = []
    total_combinations = len(profiles) * len(available_laws)
    current = 0

    # Open output file and write metadata first
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata record first — matches extraction_graph.py metadata format
        metadata = {
            "record_type": "metadata",
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "provider": provider,
            "law": laws_filter[0] if laws_filter and len(laws_filter) == 1 else None,
            "profiles_count": len(profiles),
            "graph_type": None,  # Not using graph approach
            "approach": "open_prompt",
            "git_info": git_info,
            # Legacy fields kept for compatibility
            "extraction_date": datetime.now().isoformat(),
            "reference_date": TODAY,
            "filters": {
                "laws_filter": laws_filter,
                "profiles_filter": profiles_filter,
            },
            "total_profiles": len(profiles),
            "total_laws": len(available_laws),
            "expected_records": total_combinations,
            "global_services": raw_profiles_data.get("globalServices", {}),
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # Iterate over all combinations
        for bsn, profile in profiles.items():
            full_profile = raw_profiles_data.get("profiles", {}).get(bsn, profile)

            for law_name in available_laws:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total_combinations}] Processing {law_name} for {profile.get('name', bsn)}...")

                # Base record — column names match extraction_graph.py
                record: dict = {
                    "record_type": "explanation",
                    "graph_type": None,
                    "law": law_name,
                    "profile": bsn,
                    "profile_name": profile.get("name", "Unknown"),
                    # Top-level calculated values (filled in after calculation)
                    "requirements_met": None,
                    "hoogte_toeslag": None,
                    "hoogte_toeslag_per_maand": None,
                    "inkomen": None,
                    "partner_inkomen": None,
                    "vermogen": None,
                    "leeftijd": None,
                    "heeft_partner": None,
                    "is_verzekerde": None,
                    # LLM output
                    "explanation": None,
                    "skeleton_used": None,  # Not applicable — open prompt approach
                    "prompt_used": None,
                    "model": model_id,
                    "usage": None,
                    "graph_stats": None,  # Not applicable
                    "calculation_result": None,
                }

                try:
                    # Get the service
                    service = mcp_connector.registry.get_service(law_name)
                    if not service:
                        if verbose:
                            print("  Skipping: service not found")
                        record["error"] = "Service not found"
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        results.append(record)
                        continue

                    # Execute the law calculation
                    calc_result = service.execute(bsn, {})

                    if "error" in calc_result:
                        if verbose:
                            print(f"  Error in calculation: {calc_result['error']}")
                        record["error"] = calc_result["error"]
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        results.append(record)
                        continue

                    # Fill top-level calculated values
                    calc_output = calc_result.get("result", {})
                    input_data = calc_result.get("input_data", {})
                    record["requirements_met"] = calc_result.get("requirements_met")
                    if calc_output.get("hoogte_toeslag"):
                        record["hoogte_toeslag"] = calc_output["hoogte_toeslag"] / 100
                        record["hoogte_toeslag_per_maand"] = calc_output["hoogte_toeslag"] / 100 / 12
                    else:
                        record["hoogte_toeslag"] = 0
                        record["hoogte_toeslag_per_maand"] = 0

                    # Profile input values from resolved calculation inputs
                    record["inkomen"] = _get_input_value(input_data, "INKOMEN")
                    record["partner_inkomen"] = _get_input_value(input_data, "PARTNER_INKOMEN")
                    record["vermogen"] = _get_input_value(input_data, "VERMOGEN")
                    record["leeftijd"] = _get_input_value(input_data, "LEEFTIJD")
                    record["heeft_partner"] = _get_input_value(input_data, "HEEFT_PARTNER")
                    record["is_verzekerde"] = _get_input_value(input_data, "IS_VERZEKERDE")

                    record["calculation_result"] = {
                        "requirements_met": calc_result.get("requirements_met"),
                        "missing_required": calc_result.get("missing_required"),
                        "missing_fields": calc_result.get("missing_fields", []),
                        "output": calc_output,
                        "input_data": calc_result.get("input_data", {}),
                        "system_explanation": calc_result.get("explanation", ""),
                    }

                    # Create prompt for explanation
                    prompt = create_explanation_prompt(law_name, calc_result, profile, bsn)
                    record["prompt_used"] = prompt

                    # Get LLM explanation
                    if verbose:
                        print("  Requesting LLM explanation...")

                    explanation, usage = call_llm(
                        model_id=model_id,
                        provider=provider,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt,
                        api_key=actual_api_key,
                    )

                    record["explanation"] = explanation
                    record["usage"] = usage

                    if verbose:
                        print(f"  Requirements met: {calc_result.get('requirements_met', 'N/A')}")
                        print(f"  Explanation length: {len(explanation)} chars")

                except Exception as e:
                    if verbose:
                        print(f"  Exception: {str(e)}")
                    record["error"] = str(e)

                # Write record immediately (streaming to file)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Extraction complete!")
        print(f"Total records: {len(results)}")
        print(f"Output saved to: {output_path.absolute()}")

        # Summary statistics
        successful = sum(1 for r in results if r.get("explanation"))
        errors = sum(1 for r in results if r.get("error"))
        met_requirements = sum(1 for r in results if r.get("requirements_met") is True)

        print("\nStatistics:")
        print(f"  Successful explanations: {successful}")
        print(f"  Errors: {errors}")
        print(f"  Requirements met: {met_requirements}/{successful}")

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
