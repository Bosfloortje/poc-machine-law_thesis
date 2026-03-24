"""
Post-hoc validation of LLM explanation JSONL output files.

Reads an existing JSONL file produced by extraction_graph.py and validates
each explanation against the calculation results stored in the same record.

Checks:
  1. Number validation: are the amounts in the explanation correct?
  2. Status validation: does the explanation correctly say yes/no for eligibility?
  3. B1 language level: sentence length, jargon, complex words

Usage:
  uv run python analysis/llm_explanations/validate_output.py <jsonl_file>
  uv run python analysis/llm_explanations/validate_output.py <jsonl_file> --verbose
  uv run python analysis/llm_explanations/validate_output.py <jsonl_file> --output validated.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Number extraction (Dutch + American format)
# ---------------------------------------------------------------------------

def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from text, handling both Dutch and American formats.

    Dutch:    104.819,50  (dot=thousands, comma=decimal)
    American: 104,819.50  (comma=thousands, dot=decimal)
    Simple:   104819.50 or 104819,50 or 104819
    """
    # Collect matches as (start, end, value) tuples
    # Order matters: try specific patterns first, then generic
    spans = []

    def _add(m: re.Match, val: float) -> None:
        spans.append((m.start(), m.end(), val))

    # 1. Dutch format: 104.819,50 (dot=thousands, comma=decimal)
    for m in re.finditer(r"€?\s*(\d{1,3}(?:\.\d{3})+,\d{1,2})", text):
        num_str = m.group(1).replace(".", "").replace(",", ".")
        try:
            _add(m, float(num_str))
        except ValueError:
            pass

    # 2. American format: 104,819.50 (comma=thousands, dot=decimal)
    for m in re.finditer(r"€?\s*(\d{1,3}(?:,\d{3})+\.\d{1,2})", text):
        num_str = m.group(1).replace(",", "")
        try:
            _add(m, float(num_str))
        except ValueError:
            pass

    # 3. Dutch thousands without decimal: 104.819
    for m in re.finditer(r"€?\s*(\d{1,3}(?:\.\d{3})+)(?![,.\d])", text):
        num_str = m.group(1).replace(".", "")
        try:
            _add(m, float(num_str))
        except ValueError:
            pass

    # 4. American thousands without decimal: 104,819
    for m in re.finditer(r"€?\s*(\d{1,3}(?:,\d{3})+)(?![.,\d])", text):
        num_str = m.group(1).replace(",", "")
        try:
            _add(m, float(num_str))
        except ValueError:
            pass

    def _overlaps(m: re.Match) -> bool:
        """Check if match overlaps with any already-captured span."""
        return any(s <= m.start() < e or s < m.end() <= e for s, e, _ in spans)

    # 5. Simple decimal with comma: 138,58
    for m in re.finditer(r"(\d+,\d{1,2})(?!\d)", text):
        if _overlaps(m):
            continue
        num_str = m.group(1).replace(",", ".")
        try:
            _add(m, float(num_str))
        except ValueError:
            pass

    # 6. Simple decimal with dot: 138.58
    for m in re.finditer(r"(\d+\.\d{1,2})(?!\d)", text):
        if _overlaps(m):
            continue
        try:
            _add(m, float(m.group(1)))
        except ValueError:
            pass

    # 7. Plain integer (only if not part of an already-matched span)
    for m in re.finditer(r"(\d+)", text):
        if _overlaps(m):
            continue
        try:
            _add(m, float(m.group(1)))
        except ValueError:
            pass

    # Return values in text order, deduplicated by position
    return [val for _, _, val in sorted(spans, key=lambda x: x[0])]


# ---------------------------------------------------------------------------
# Number / amount validation
# ---------------------------------------------------------------------------

def validate_numbers(explanation: str, record: dict) -> tuple[bool, list[str], dict]:
    """Validate that amounts in the explanation match the calculation results."""
    errors = []
    details = {
        "expected_values": {},
        "found_values": {},
        "missing_values": [],
        "incorrect_values": [],
    }

    explanation_numbers = extract_numbers_from_text(explanation)

    # Build expected values from top-level record fields
    expected = {}

    hoogte = record.get("hoogte_toeslag")
    if hoogte is not None and hoogte > 0:
        expected["hoogte_toeslag"] = {"value": round(hoogte, 2), "critical": True}

    hoogte_maand = record.get("hoogte_toeslag_per_maand")
    if hoogte_maand is not None and hoogte_maand > 0:
        expected["hoogte_toeslag_per_maand"] = {"value": round(hoogte_maand, 2), "critical": False}

    inkomen = record.get("inkomen")
    if inkomen is not None and isinstance(inkomen, (int, float)):
        # inkomen can be in eurocents in the profile
        val = inkomen / 100 if inkomen > 100000 else inkomen
        if val > 0:
            expected["inkomen"] = {"value": round(val, 2), "critical": True}

    partner_inkomen = record.get("partner_inkomen")
    if partner_inkomen is not None and isinstance(partner_inkomen, (int, float)):
        val = partner_inkomen / 100 if partner_inkomen > 100000 else partner_inkomen
        if val > 0:
            expected["partner_inkomen"] = {"value": round(val, 2), "critical": False}

    vermogen = record.get("vermogen")
    if vermogen is not None and isinstance(vermogen, (int, float)):
        val = vermogen / 100 if vermogen > 100000 else vermogen
        if val > 0:
            expected["vermogen"] = {"value": round(val, 2), "critical": False}

    details["expected_values"] = expected

    # Check each expected value against explanation
    for key, info in expected.items():
        val = info["value"]
        found = any(abs(num - val) < 0.02 for num in explanation_numbers)

        if found:
            details["found_values"][key] = val
        elif info["critical"]:
            details["missing_values"].append(key)
            errors.append(f"ONTBREEKT: '{key}' moet {val:.2f} euro zijn maar staat niet in de uitleg")

    # Check for made-up numbers
    all_expected_vals = [info["value"] for info in expected.values()]
    # Also accept monthly variants
    all_expected_vals += [round(v / 12, 2) for v in all_expected_vals if v > 0]
    # Accept age and common small numbers
    leeftijd = record.get("leeftijd")
    if leeftijd:
        all_expected_vals.append(float(leeftijd))
    all_expected_vals.append(18.0)  # minimum age threshold

    for num in explanation_numbers:
        if num < 10:
            continue
        if any(abs(num - ev) < 0.02 for ev in all_expected_vals):
            continue
        details["incorrect_values"].append(num)
        errors.append(f"ONBEKEND BEDRAG: {num:.2f} euro in uitleg komt niet uit de berekening")

    return len(errors) == 0, errors, details


# ---------------------------------------------------------------------------
# Status validation (recht / geen recht)
# ---------------------------------------------------------------------------

def validate_status(explanation: str, record: dict) -> tuple[bool, list[str], dict]:
    """Validate that the explanation correctly states eligibility status."""
    errors = []
    explanation_lower = explanation.lower()

    requirements_met = record.get("requirements_met")
    hoogte = record.get("hoogte_toeslag", 0) or 0
    actually_gets_money = requirements_met and hoogte > 0

    details = {
        "requirements_met": requirements_met,
        "hoogte_toeslag": hoogte,
        "actually_gets_money": actually_gets_money,
        "status_correct": False,
    }

    positive_phrases = [
        "heeft recht", "recht op", "komt in aanmerking",
        "voldoet", "krijgt", "ontvangt",
    ]
    negative_phrases = [
        "geen recht", "heeft geen recht", "niet in aanmerking",
        "voldoet niet", "krijgt geen", "ontvangt geen",
        "geen toeslag", "krijgt u geen",
    ]
    zero_toeslag_phrases = [
        "geen toeslag", "krijgt u geen", "ontvangt u geen",
        "0 euro", "0,00 euro", "geen zorgtoeslag",
        "inkomen te hoog", "niet in aanmerking",
    ]

    has_positive = any(p in explanation_lower for p in positive_phrases)
    has_negative = any(p in explanation_lower for p in negative_phrases)
    has_zero_indication = any(p in explanation_lower for p in zero_toeslag_phrases)

    if actually_gets_money:
        # Should be positive, should NOT say "geen recht"
        if has_positive and not has_negative:
            details["status_correct"] = True
        elif has_negative:
            errors.append(f"STATUS FOUT: Burger krijgt {hoogte:.2f} euro maar uitleg zegt 'geen recht'")
        else:
            errors.append("STATUS FOUT: Uitleg vermeldt niet duidelijk dat burger recht heeft")
    elif requirements_met and hoogte == 0:
        # Meets requirements but gets 0 - should indicate no money
        if has_zero_indication or has_negative:
            details["status_correct"] = True
        elif has_positive and not has_zero_indication:
            errors.append("STATUS FOUT: Burger krijgt 0 euro maar uitleg zegt alleen 'heeft recht' zonder dit te nuanceren")
    elif not requirements_met:
        # Should be negative
        if has_negative:
            details["status_correct"] = True
        elif has_positive and not has_negative:
            errors.append("STATUS FOUT: Burger voldoet NIET maar uitleg zegt 'heeft recht'")
        else:
            errors.append("STATUS FOUT: Uitleg vermeldt niet duidelijk dat burger geen recht heeft")

    return len(errors) == 0, errors, details


# ---------------------------------------------------------------------------
# B1 language level validation
# ---------------------------------------------------------------------------

LEGAL_JARGON = [
    "belastingplichtige", "vermogenstoets", "eigenwoningforfait", "draagkracht",
    "toetsingsinkomen", "norminkomen", "drempelbedrag", "maximale grondslag",
    "afbouwpercentage", "afbouwtraject", "normpremie", "standaardpremie",
    "peildatum", "verzamelinkomen", "heffingskorting", "inkomensafhankelijk",
]

CHAT_ENDINGS = [
    "heb je vragen", "heeft u vragen", "neem gerust contact",
    "ik help je graag", "laat het me weten", "kan ik je helpen",
    "met vriendelijke groet", "mvg", "hartelijke groet",
]


def count_syllables_dutch(word: str) -> int:
    """Heuristic Dutch syllable count."""
    word = word.lower().strip(".,!?;:()")
    vowels = "aeiouáéíóúàèìòùäëïöü"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def validate_b1(explanation: str) -> tuple[bool, list[str], dict]:
    """Validate B1 language level."""
    warnings = []

    sentences = re.split(r"[.!?]+", explanation)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Sentence length
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    # Complex words
    all_words = explanation.split()
    complex_words = []
    for w in all_words:
        clean = re.sub(r"[^a-zA-Zéèëïöüà]", "", w)
        if clean and count_syllables_dutch(clean) > 4:
            complex_words.append({"word": clean, "syllables": count_syllables_dutch(clean)})

    # Jargon
    explanation_lower = explanation.lower()
    jargon_found = [j for j in LEGAL_JARGON if j in explanation_lower]

    # Chat endings
    chat_found = [c for c in CHAT_ENDINGS if c in explanation_lower]

    details = {
        "avg_sentence_length": round(avg_length, 1),
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
        "complex_words": complex_words[:10],
        "jargon_found": jargon_found,
        "chat_endings": chat_found,
    }

    if avg_length > 25:
        warnings.append(f"TAAL: Gemiddelde zinlengte te lang ({avg_length:.0f} woorden, max 25)")
    if jargon_found:
        warnings.append(f"TAAL: Juridisch jargon gevonden: {', '.join(jargon_found)}")
    if chat_found:
        warnings.append(f"TAAL: Ongewenste afsluiting: {', '.join(chat_found)}")

    # B1 warnings are non-blocking
    return len(warnings) == 0, warnings, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_jsonl(input_path: str, output_path: str | None = None, verbose: bool = False) -> None:
    """Validate all explanation records in a JSONL file."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    explanations = [r for r in records if r.get("record_type") == "explanation"]
    metadata = [r for r in records if r.get("record_type") == "metadata"]

    if metadata:
        m = metadata[0]
        print(f"Model: {m.get('model')} ({m.get('provider')})")
        print(f"Law: {m.get('law')}")
        print(f"Profiles: {m.get('profiles_count')}")
        print()

    # Counters
    total = len(explanations)
    num_valid = 0
    num_invalid = 0
    all_errors = []
    all_warnings = []
    status_errors = 0
    number_errors = 0
    b1_warnings_count = 0

    validated_records = list(metadata)  # Keep metadata as-is

    for record in explanations:
        explanation = record.get("explanation", "")
        profile = record.get("profile", "?")
        name = record.get("profile_name", "?")

        # Run validations
        num_ok, num_errs, num_details = validate_numbers(explanation, record)
        status_ok, status_errs, status_details = validate_status(explanation, record)
        b1_ok, b1_warns, b1_details = validate_b1(explanation)

        is_valid = num_ok and status_ok
        errors = num_errs + status_errs
        warnings = b1_warns

        if is_valid:
            num_valid += 1
        else:
            num_invalid += 1

        if num_errs:
            number_errors += 1
        if status_errs:
            status_errors += 1
        if b1_warns:
            b1_warnings_count += 1

        all_errors.extend(errors)
        all_warnings.extend(warnings)

        # Build validated record
        validated = {**record}
        validated["validation"] = {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "number_validation": num_details,
            "status_validation": status_details,
            "b1_validation": b1_details,
        }
        validated_records.append(validated)

        if verbose and (errors or warnings):
            print(f"[{profile}] {name}")
            for e in errors:
                print(f"  ERROR: {e}")
            for w in warnings:
                print(f"  WARN:  {w}")
            print()

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total explanations:    {total}")
    print(f"Valid:                 {num_valid} ({num_valid/total*100:.1f}%)")
    print(f"Invalid:               {num_invalid} ({num_invalid/total*100:.1f}%)")
    print()
    print(f"Number errors:         {number_errors} profiles")
    print(f"Status errors:         {status_errors} profiles")
    print(f"B1 warnings:           {b1_warnings_count} profiles")
    print()

    # Error frequency
    if all_errors:
        error_types = {}
        for e in all_errors:
            key = e.split(":")[0]
            error_types[key] = error_types.get(key, 0) + 1
        print("Error breakdown:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")
        print()

    if all_warnings:
        warn_types = {}
        for w in all_warnings:
            key = w.split(":")[0]
            warn_types[key] = warn_types.get(key, 0) + 1
        print("Warning breakdown:")
        for wtype, count in sorted(warn_types.items(), key=lambda x: -x[1]):
            print(f"  {wtype}: {count}")
        print()

    # Write validated output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in validated_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Validated output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLM explanation JSONL output")
    parser.add_argument("input", help="Path to JSONL file to validate")
    parser.add_argument("--output", "-o", help="Write validated JSONL to this path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show errors per profile")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    validate_jsonl(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
