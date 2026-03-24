#!/usr/bin/env python3
"""
Script to extract LLM-generated explanations with self-correction/validation.

This script adds a verification loop:
1. LLM generates explanation
2. Validator checks if numbers/amounts in explanation match actual calculation results
3. If errors found: LLM gets feedback and must correct
4. Repeat until correct or max attempts reached

Usage:
    uv run python analysis/llm_explanations/extract_with_validation.py --laws zorgtoeslag --profiles 100000001 --output results.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def generate_output_filename(
    model: str,
    laws_filter: list[str] | None,
    profiles_filter: list[str] | None,
    script_type: str,
) -> str:
    """Generate descriptive output filename based on run metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add law filter info
    if laws_filter:
        law_part = "-".join(laws_filter)
    else:
        law_part = "all-laws"

    # Add profile filter info
    if profiles_filter:
        profile_part = f"{len(profiles_filter)}profiles"
    else:
        profile_part = "all-profiles"

    filename = f"{timestamp}_{model}_{law_part}_{profile_part}_{script_type}.jsonl"
    return f"analysis/llm_explanations/output/{filename}"


# Add project root and web directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml


def get_git_info() -> dict:
    """Get git commit information for reproducibility."""
    info = {}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["machine_law_dirty"] = len(result.stdout.strip()) > 0

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
    """Load all profiles from the profiles.yaml file."""
    with open(profiles_path) as f:
        raw_data = yaml.safe_load(f)
    return raw_data.get("profiles", {}), raw_data


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from text (including currency amounts)."""
    # Match numbers including euros, decimals, and percentages
    # Examples: €500, 500,00, 1.234,56, 50%, 1234
    pattern = r"€?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+(?:[.,]\d+)?)\s*%?"
    matches = re.findall(pattern, text)

    numbers = []
    for match in matches:
        # Convert Dutch/European format (1.234,56) to float
        # Remove thousand separators and convert decimal comma to dot
        num_str = match.replace(".", "").replace(",", ".")
        try:
            numbers.append(float(num_str))
        except ValueError:
            continue

    return numbers


def validate_explanation(explanation: str, calculation_result: dict) -> tuple[bool, list[str], dict]:
    """
    Strictly validate that the LLM explanation contains correct numbers from calculation.

    Returns:
        (is_valid, list_of_errors, validation_details)
    """
    errors = []
    validation_details = {
        "expected_values": {},
        "found_values": {},
        "missing_values": [],
        "incorrect_values": [],
    }

    # Extract numbers from explanation
    explanation_numbers = extract_numbers_from_text(explanation)
    explanation_lower = explanation.lower()

    # Get expected values from calculation result (convert from eurocents to euros)
    output = calculation_result.get("output", {})

    # Build expected values with better categorization
    for key, value in output.items():
        if isinstance(value, (int, float)):
            # Determine if this is a currency value (in eurocents) or not
            is_currency = any(
                word in key.lower()
                for word in [
                    "bedrag",
                    "toeslag",
                    "inkomen",
                    "vermogen",
                    "huur",
                    "premie",
                    "grens",
                    "eigenwoningforfait",
                    "aftrek",
                    "belasting",
                    "eigenwoningwaardering",
                ]
            )

            if is_currency:
                expected_euro_value = round(value / 100, 2)  # Convert eurocents to euros
            else:
                expected_euro_value = round(value, 2)

            # Only validate significant non-zero values
            if abs(expected_euro_value) > 0.01:
                validation_details["expected_values"][key] = {
                    "original_value": value,
                    "euro_value": expected_euro_value,
                    "is_currency": is_currency,
                }

    # Check each expected value
    for key, value_info in validation_details["expected_values"].items():
        expected_value = value_info["euro_value"]

        # Find if this exact value appears in the explanation
        found = False
        matched_number = None

        for num in explanation_numbers:
            # Allow tiny rounding differences (max 1 cent)
            if abs(num - expected_value) < 0.01:
                found = True
                matched_number = num
                break

        if found:
            validation_details["found_values"][key] = {
                "expected": expected_value,
                "found": matched_number,
                "match": True,
            }
        else:
            # Value not found - check if it's critical
            # Critical values are main results (toeslag amounts, income, etc.)
            is_critical = any(
                critical_word in key.lower()
                for critical_word in ["toeslag", "bedrag", "uitkering", "inkomen", "vermogen"]
            )

            if is_critical or abs(expected_value) > 100:  # Critical or large amounts must be mentioned
                validation_details["missing_values"].append(key)
                errors.append(
                    f"ONTBREEKT: '{key}' moet €{expected_value:.2f} zijn, maar dit bedrag staat niet in de uitleg"
                )

    # Check if any numbers in explanation DON'T match expected values
    # This catches cases where LLM makes up wrong numbers
    for num in explanation_numbers:
        # Skip very small numbers (likely percentages or years)
        if num < 1:
            continue

        # Check if this number matches any expected value
        matches_expected = False
        matches_monthly_instead_of_yearly = False

        for value_info in validation_details["expected_values"].values():
            expected = value_info["euro_value"]

            # Check exact match
            if abs(num - expected) < 0.01:
                matches_expected = True
                break

            # Check if this is the monthly version (yearly / 12)
            monthly_version = round(expected / 12, 2)
            if abs(num - monthly_version) < 0.01:
                # Check if the yearly amount is ALSO mentioned in the explanation
                yearly_also_mentioned = False
                for other_num in explanation_numbers:
                    if abs(other_num - expected) < 0.01:
                        yearly_also_mentioned = True
                        break

                if yearly_also_mentioned:
                    # OK: both yearly and monthly are mentioned
                    matches_expected = True
                else:
                    # ERROR: only monthly mentioned, not yearly
                    matches_monthly_instead_of_yearly = True
                    validation_details["incorrect_values"].append(num)
                    errors.append(
                        f"FOUT: €{num:.2f} is het maandbedrag, maar jaarbedrag €{expected:.2f} per jaar ontbreekt"
                    )
                break

        if not matches_expected and not matches_monthly_instead_of_yearly and num > 10:  # Only flag significant amounts
            # This number appears in explanation but doesn't match calculation
            validation_details["incorrect_values"].append(num)
            errors.append(f"FOUT BEDRAG: €{num:.2f} staat in de uitleg maar komt niet uit de berekening")

    # Check if requirements_met status is correctly mentioned
    requirements_met = calculation_result.get("requirements_met")
    validation_details["requirements_met_check"] = {
        "expected": requirements_met,
        "mentioned_correctly": False,
    }

    if requirements_met is True:
        # Should mention that person DOES qualify
        positive_phrases = [
            "komt in aanmerking",
            "voldoet",
            "recht op",
            "komt wel",
            "heeft recht",
            "krijgt",
            "ontvang",
        ]
        if any(phrase in explanation_lower for phrase in positive_phrases):
            validation_details["requirements_met_check"]["mentioned_correctly"] = True
        else:
            errors.append(
                "STATUS FOUT: Burger voldoet WEL (requirements_met=True), maar dit wordt niet duidelijk vermeld"
            )

    elif requirements_met is False:
        # Should mention that person does NOT qualify
        # More flexible matching to catch natural language variations
        negative_phrases = [
            "komt niet in aanmerking",
            "voldoet niet",
            "geen recht",
            "komt niet",
            "heeft geen recht",
            "krijgt geen",
            "ontvang geen",
            "kan geen",  # "kan geen huurtoeslag krijgen"
            "niet krijgen",  # "niet in aanmerking komen", "geen toeslag krijgen"
            "geen.*krijgen",  # Pattern: "geen [iets] krijgen"
        ]

        # Check both exact matches and regex patterns
        found_negative = False
        for phrase in negative_phrases:
            if phrase in explanation_lower:
                found_negative = True
                break

        # Also check with regex for more complex patterns
        if not found_negative:
            if re.search(r"geen\s+\w+\s+krijgen", explanation_lower):
                found_negative = True

        if found_negative:
            validation_details["requirements_met_check"]["mentioned_correctly"] = True
        else:
            errors.append(
                "STATUS FOUT: Burger voldoet NIET (requirements_met=False), maar dit wordt niet duidelijk vermeld"
            )

    is_valid = len(errors) == 0
    return is_valid, errors, validation_details


def create_explanation_prompt(service_name: str, result: dict, profile: dict, bsn: str) -> str:
    """Create initial prompt for explanation."""
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

BELANGRIJK:
- Alle bedragen in de uitkomst zijn in eurocenten, deel door 100 voor euros
- Vermeld concrete bedragen uit de berekening in je uitleg (in euros, niet eurocenten)
- Wees specifiek en gebruik de exacte getallen uit het resultaat
- Vermeld ALTIJD bedragen als jaarbedragen (per jaar), NOOIT per maand
- Bijvoorbeeld: zeg "€1.654,12 per jaar" en NIET "€137,84 per maand"
- Als je een maandbedrag wilt noemen voor verduidelijking, bereken dan: jaarbedrag / 12 en vermeld beide

BELANGRIJK - Formaat:
- Dit is een standalone informatieve tekst, GEEN chatgesprek
- Eindig NIET met vragen zoals "Heb je nog vragen?" of "Kan ik je ergens mee helpen?"
- Eindig NIET met aanbiedingen zoals "Ik help je graag verder" of "Neem contact op als je vragen hebt"
- Geef gewoon de complete uitleg en stop daar"""

    return prompt


def create_correction_prompt(
    original_explanation: str, validation_errors: list[str], calculation_result: dict
) -> str:
    """Create prompt to ask LLM to correct errors."""
    output = calculation_result.get("output", {})

    prompt = f"""Je vorige uitleg bevatte enkele fouten of ontbrekende informatie.

Je vorige uitleg:
{original_explanation}

Gevonden fouten:
{chr(10).join(f"- {error}" for error in validation_errors)}

Correcte berekening resultaten (in eurocenten, deel door 100 voor euros):
{json.dumps(output, indent=2, ensure_ascii=False)}

Corrigeer je uitleg zodat:
1. Alle bedragen correct zijn (converteer van eurocenten naar euros)
2. De status (wel/niet in aanmerking komen) correct wordt vermeld
3. De uitleg duidelijk en begrijpelijk blijft in B1-niveau Nederlands
4. Alle bedragen worden vermeld als jaarbedragen (per jaar), NOOIT per maand
5. Als je een maandbedrag wilt noemen, bereken dan: jaarbedrag / 12 en vermeld beide

Geef alleen de gecorrigeerde uitleg, zonder extra commentaar."""

    return prompt


# Model configuration
AVAILABLE_MODELS = {
    "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "description": "Fast and cheap, good for batch processing",
        "input_cost_per_mtok": 1.00,
        "output_cost_per_mtok": 5.00,
    },
    "sonnet": {
        "id": "claude-sonnet-4-5-20250929",
        "description": "Balanced performance and cost",
        "input_cost_per_mtok": 3.00,
        "output_cost_per_mtok": 15.00,
    },
    "opus": {
        "id": "claude-opus-4-5-20251101",
        "description": "Most capable, highest cost",
        "input_cost_per_mtok": 5.00,
        "output_cost_per_mtok": 25.00,
    },
}

DEFAULT_MODEL = "haiku"
MAX_CORRECTION_ATTEMPTS = 3  # Maximum number of correction loops


def extract_explanations_with_validation(
    api_key: str | None = None,
    laws_filter: list[str] | None = None,
    profiles_filter: list[str] | None = None,
    output_file: str = "explanations_validated.jsonl",
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_CORRECTION_ATTEMPTS,
    verbose: bool = True,
) -> list[dict]:
    """Extract LLM explanations with validation and self-correction."""
    import anthropic

    # Set API key
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    actual_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not actual_api_key:
        print("ERROR: No Anthropic API key found!")
        sys.exit(1)

    # Validate model
    if model not in AVAILABLE_MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    model_info = AVAILABLE_MODELS[model]
    model_id = model_info["id"]

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=actual_api_key)

    # Import after setting up environment
    from explain.mcp_connector import MCPLawConnector
    from web.dependencies import TODAY, get_case_manager, get_claim_manager, get_machine_service

    # Initialize services
    if verbose:
        print("Initializing services...")

    services = get_machine_service()
    case_manager = get_case_manager()
    claim_manager = get_claim_manager()

    mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

    # Load profiles
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
        print(f"Max correction attempts: {max_attempts}")

    # LLM parameters
    llm_params = {
        "model_id": model_id,
        "model_name": model,
        "max_tokens": 1500,
        "temperature": 0.3,
        "system_prompt": "Je bent een informatiesysteem dat Nederlandse burgers objectieve uitleg geeft over overheidsregelingen. Schrijf standalone informatieve teksten in eenvoudig Nederlands (B1-niveau). Wees nauwkeurig met bedragen en getallen. Dit is GEEN chatgesprek - geef alleen de gevraagde informatie zonder vragen te stellen of hulp aan te bieden.",
        "max_correction_attempts": max_attempts,
    }

    # Collect git info
    git_info = get_git_info()

    # Results storage
    results = []
    total_combinations = len(profiles) * len(available_laws)
    current = 0

    # Statistics
    total_corrections = 0
    total_validation_passes = 0
    total_validation_failures = 0

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file
    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata
        metadata = {
            "record_type": "metadata",
            "extraction_date": datetime.now().isoformat(),
            "reference_date": TODAY,
            "git_info": git_info,
            "llm_params": llm_params,
            "filters": {
                "laws_filter": laws_filter,
                "profiles_filter": profiles_filter,
            },
            "total_profiles": len(profiles),
            "total_laws": len(available_laws),
            "expected_records": total_combinations,
            "validation_enabled": True,
            "global_services": raw_profiles_data.get("globalServices", {}),
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # Process all combinations
        for bsn, profile in profiles.items():
            full_profile = raw_profiles_data.get("profiles", {}).get(bsn, profile)

            for law_name in available_laws:
                current += 1
                if verbose:
                    print(
                        f"\n[{current}/{total_combinations}] Processing {law_name} for {profile.get('name', bsn)}..."
                    )

                record = {
                    "record_type": "explanation",
                    "record_number": current,
                    "timestamp": datetime.now().isoformat(),
                    "bsn": bsn,
                    "profile": {
                        "name": profile.get("name", "Unknown"),
                        "description": profile.get("description", ""),
                        "sources": full_profile.get("sources", {}),
                    },
                    "validation_attempts": [],
                }

                try:
                    # Get service
                    service = mcp_connector.registry.get_service(law_name)
                    if not service:
                        if verbose:
                            print("  Skipping: service not found")
                        record["error"] = "Service not found"
                        record["law"] = {"name": law_name}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        results.append(record)
                        continue

                    # Get rule spec
                    rule_spec = services.get_rule_spec(service.law_path, TODAY, service.service_type)

                    record["law"] = {
                        "name": law_name,
                        "description": service.description,
                        "service_type": service.service_type,
                        "law_path": service.law_path,
                        "rule_spec_name": rule_spec.get("name") if rule_spec else None,
                        "rule_spec_version": rule_spec.get("version") if rule_spec else None,
                    }

                    # Execute calculation
                    calc_result = service.execute(bsn, {})

                    if "error" in calc_result:
                        if verbose:
                            print(f"  Error in calculation: {calc_result['error']}")
                        record["error"] = calc_result["error"]
                        record["calculation_result"] = None
                        record["llm_explanation"] = None
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        results.append(record)
                        continue

                    record["calculation_result"] = {
                        "requirements_met": calc_result.get("requirements_met"),
                        "missing_required": calc_result.get("missing_required"),
                        "missing_fields": calc_result.get("missing_fields", []),
                        "output": calc_result.get("result", {}),
                        "input_data": calc_result.get("input_data", {}),
                        "system_explanation": calc_result.get("explanation", ""),
                    }

                    # Validation loop with self-correction
                    current_explanation = None
                    is_valid = False
                    attempt = 0

                    while attempt < max_attempts and not is_valid:
                        attempt += 1

                        if attempt == 1:
                            # First attempt: generate initial explanation
                            prompt = create_explanation_prompt(law_name, calc_result, profile, bsn)
                            if verbose:
                                print(f"  Attempt {attempt}/{max_attempts}: Generating explanation...")
                        else:
                            # Correction attempt
                            prompt = create_correction_prompt(
                                current_explanation, validation_errors, record["calculation_result"]
                            )
                            if verbose:
                                print(f"  Attempt {attempt}/{max_attempts}: Correcting explanation...")
                            total_corrections += 1

                        # Call LLM
                        response = client.messages.create(
                            model=model_id,
                            max_tokens=llm_params["max_tokens"],
                            temperature=llm_params["temperature"],
                            system=llm_params["system_prompt"],
                            messages=[{"role": "user", "content": prompt}],
                        )

                        current_explanation = response.content[0].text

                        # Validate explanation
                        is_valid, validation_errors, validation_details = validate_explanation(
                            current_explanation, record["calculation_result"]
                        )

                        # Record attempt
                        attempt_record = {
                            "attempt_number": attempt,
                            "prompt_used": prompt,
                            "explanation": current_explanation,
                            "is_valid": is_valid,
                            "validation_errors": validation_errors if not is_valid else [],
                            "validation_details": validation_details,
                            "tokens_used": {
                                "input": response.usage.input_tokens if hasattr(response, "usage") else None,
                                "output": response.usage.output_tokens if hasattr(response, "usage") else None,
                            },
                        }
                        record["validation_attempts"].append(attempt_record)

                        if is_valid:
                            if verbose:
                                print(f"  [OK] Validation passed on attempt {attempt}")
                            total_validation_passes += 1
                            break
                        else:
                            if verbose:
                                print(f"  [FAIL] Validation failed: {len(validation_errors)} errors")
                                for error in validation_errors:
                                    print(f"    - {error}")

                    # Store final result
                    record["llm_explanation"] = current_explanation
                    record["final_validation_status"] = is_valid
                    record["total_attempts"] = attempt

                    if not is_valid:
                        total_validation_failures += 1
                        if verbose:
                            print(f"  [FAIL] Failed to validate after {max_attempts} attempts")

                    # Calculate total token usage
                    total_input_tokens = sum(
                        a["tokens_used"]["input"] for a in record["validation_attempts"] if a["tokens_used"]["input"]
                    )
                    total_output_tokens = sum(
                        a["tokens_used"]["output"]
                        for a in record["validation_attempts"]
                        if a["tokens_used"]["output"]
                    )
                    record["total_tokens_used"] = {
                        "input": total_input_tokens,
                        "output": total_output_tokens,
                    }

                    if verbose:
                        print(f"  Requirements met: {calc_result.get('requirements_met', 'N/A')}")
                        print(f"  Total tokens: {total_input_tokens} input, {total_output_tokens} output")

                except Exception as e:
                    if verbose:
                        print(f"  Exception: {str(e)}")
                    record["error"] = str(e)
                    record["calculation_result"] = None
                    record["llm_explanation"] = None

                # Write record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

    if verbose:
        print(f"\n{'=' * 60}")
        print("Extraction complete!")
        print(f"Total records: {len(results)}")
        print(f"Output saved to: {output_path.absolute()}")

        # Summary statistics
        successful = sum(1 for r in results if r.get("llm_explanation"))
        errors = sum(1 for r in results if r.get("error"))
        met_requirements = sum(
            1 for r in results if (r.get("calculation_result") or {}).get("requirements_met") is True
        )

        print("\nStatistics:")
        print(f"  Successful explanations: {successful}")
        print(f"  Errors: {errors}")
        print(f"  Requirements met: {met_requirements}/{successful}")
        print(f"\nValidation statistics:")
        print(f"  Validated correctly: {total_validation_passes}")
        print(f"  Failed validation: {total_validation_failures}")
        print(f"  Total corrections made: {total_corrections}")
        if successful > 0:
            avg_attempts = sum(r.get("total_attempts", 1) for r in results if r.get("llm_explanation")) / successful
            print(f"  Average attempts per explanation: {avg_attempts:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM explanations with validation and self-correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--laws", nargs="+", help="Filter to specific laws")
    parser.add_argument("--profiles", nargs="+", help="Filter to specific BSNs")
    parser.add_argument("--output", default="analysis/llm_explanations/output/explanations_validated.jsonl", help="Output JSONL file (default: analysis/llm_explanations/output/explanations_validated.jsonl)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_CORRECTION_ATTEMPTS,
        help=f"Max correction attempts (default: {MAX_CORRECTION_ATTEMPTS})",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Generate output filename if using default
    if args.output == "analysis/llm_explanations/output/explanations_validated.jsonl":
        args.output = generate_output_filename(
            model=args.model,
            laws_filter=args.laws,
            profiles_filter=args.profiles,
            script_type="validated",
        )
        if not args.quiet:
            print(f"Output file: {args.output}")

    extract_explanations_with_validation(
        api_key=args.api_key,
        laws_filter=args.laws,
        profiles_filter=args.profiles,
        output_file=args.output,
        model=args.model,
        max_attempts=args.max_attempts,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
