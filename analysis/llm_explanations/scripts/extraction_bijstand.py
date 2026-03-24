#!/usr/bin/env python3
"""
Generic GraphRAG extraction script for Bijstand (Participatiewet) — alle gemeenten.

Detecteert automatisch de gemeente van een profiel op basis van woonplaats en laadt
het bijbehorende gemeentelijke YAML-bestand. Werkt voor alle gemeenten waarvoor een
YAML-bestand beschikbaar is in laws/participatiewet/bijstand/gemeenten/.

Ondersteunde gemeenten:
  Amsterdam, Rotterdam, Den Haag, Utrecht, Eindhoven, Groningen, Maastricht

Usage:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1 --law bijstand
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1 --law bijstand --profiles 159428317
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# scripts/ → llm_explanations/ → analysis/ → root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import os

from extraction_zorgtoeslag import (  # noqa: E402
    AVAILABLE_MODELS,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    LawGraphExtractor,
    ProfileGraphExtractor,
    get_git_info,
    load_profiles,
    DecisionGraphExtractor as _BaseDecisionGraphExtractor,
    _build_expected_values,
    _fix_rounded_amounts,
)


# ---------------------------------------------------------------------------
# Bijstand-specific LLM prompt
# ---------------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse burgers uitleg geeft over overheidsbeslissingen.

Je taak is om een beslissingsskeleton om te zetten naar een korte, begrijpelijke uitleg.

VERPLICHT:
- Gebruik ALLEEN informatie uit het skeleton - voeg NIETS toe
- Schrijf in eenvoudig Nederlands (B1-niveau) - korte zinnen, gewone woorden
- Dit is een informatieve tekst, GEEN brief of gesprek
- Eindig altijd met de tekst over de website van de gemeente uit het skeleton

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN verwijzingen naar andere instanties dan die in het skeleton staan
- GEEN aanbiedingen voor hulp of vragen aan de lezer
- GEEN technische termen of codes behouden - alles moet in normale taal"""


def create_decision_prompt(skeleton: str, person_name: str) -> str:
    """Create a bijstand-specific LLM prompt using the decision skeleton."""
    import re
    url_match = re.search(r'www\.\S+', skeleton)
    info_url = url_match.group(0).rstrip(".,)") if url_match else "www.gemeenten.nl"

    return f"""# Informatie over de beslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau).

WAT JE MOET DOEN:
- Begin met de conclusie (wel of geen recht op bijstand)
- Noem alleen de voorwaarden waaraan NIET voldaan wordt als reden
- Noem het uitkeringsbedrag ALLEEN als het expliciet in de "Berekening"-sectie van het skeleton staat
- Eindig met de zin over {info_url}
- Praat altijd in de u-vorm tegen {person_name}

WAT JE NIET MAG DOEN:
- GEEN briefopmaak gebruiken (geen "Geachte heer/mevrouw", geen "Met vriendelijke groet")
- GEEN verwijzingen naar andere instanties dan de gemeente
- GEEN extra voorwaarden of regels verzinnen die niet in het skeleton staan
- GEEN bedragen noemen als er geen "Berekening"-sectie in het skeleton staat
- GEEN uitleg geven over voorwaarden waaraan WEL voldaan wordt
- GEEN vragen stellen aan de lezer of aanbod om te helpen
- GEEN derde persoon aanspreken - schrijf direct tegen {person_name}

VOORBEELD GOEDE UITLEG BIJ GEEN RECHT (structuurvoorbeeld):
"U heeft geen recht op bijstand.
U voldoet niet aan de arbeidsvoorwaarde: u heeft volledig arbeidsvermogen en geen re-integratietraject.
Voor meer informatie kunt u terecht op www.gemeenten.nl"

VOORBEELD GOEDE UITLEG BIJ WEL RECHT (structuurvoorbeeld):
"U heeft recht op bijstand van 900 euro per maand.
U voldoet aan alle voorwaarden.
Voor meer informatie kunt u terecht op www.gemeenten.nl"

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"Geachte heer Jansen, Hierbij informeren wij u... Met vriendelijke groet" <- FOUT: briefopmaak
"Neem contact op met het UWV" <- FOUT: verkeerde instantie
"U heeft geen recht op bijstand van 1.200 euro" <- FOUT: bedrag noemen als er geen recht is"""


def generate_decision_explanation(
    decision_extractor: "DecisionGraphExtractor",
    person_name: str,
    api_key: str | None = None,
    model: str = "haiku",
) -> dict:
    """Generate a bijstand LLM explanation using the constrained decision skeleton."""
    model_config = AVAILABLE_MODELS[model]
    model_id = model_config["id"]
    provider = model_config.get("provider", "anthropic")

    skeleton = decision_extractor.to_explanation_skeleton()
    prompt = create_decision_prompt(skeleton, person_name)
    expected_values = _build_expected_values(decision_extractor)

    if provider == "ollama":
        import ollama
        response = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "num_predict": 1000},
        )
        explanation = _fix_rounded_amounts(response["message"]["content"], expected_values)
        return {
            "explanation": explanation,
            "skeleton_used": skeleton,
            "prompt_used": prompt,
            "model": model_id,
            "provider": provider,
            "usage": {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            },
        }
    else:
        import anthropic
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=1000,
            temperature=0.2,
            system=DECISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        explanation = _fix_rounded_amounts(response.content[0].text, expected_values)
        return {
            "explanation": explanation,
            "skeleton_used": skeleton,
            "prompt_used": prompt,
            "model": model_id,
            "provider": provider,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


# ---------------------------------------------------------------------------
# Gemeente mappings
# ---------------------------------------------------------------------------

# Woonplaats (lowercase) → service/tabel-naam in profiles
WOONPLAATS_TO_SERVICE: dict[str, str] = {
    "amsterdam":  "GEMEENTE_AMSTERDAM",
    "rotterdam":  "GEMEENTE_ROTTERDAM",
    "den haag":   "GEMEENTE_DEN_HAAG",
    "the hague":  "GEMEENTE_DEN_HAAG",
    "utrecht":    "GEMEENTE_UTRECHT",
    "eindhoven":  "GEMEENTE_EINDHOVEN",
    "groningen":  "GEMEENTE_GRONINGEN",
    "maastricht": "GEMEENTE_MAASTRICHT",
}

# Service-naam → gemeente-website
GEMEENTE_URLS: dict[str, str] = {
    "GEMEENTE_AMSTERDAM":  "www.amsterdam.nl",
    "GEMEENTE_ROTTERDAM":  "www.rotterdam.nl",
    "GEMEENTE_DEN_HAAG":   "www.denhaag.nl",
    "GEMEENTE_UTRECHT":    "www.utrecht.nl",
    "GEMEENTE_EINDHOVEN":  "www.eindhoven.nl",
    "GEMEENTE_GRONINGEN":  "www.groningen.nl",
    "GEMEENTE_MAASTRICHT": "www.maastricht.nl",
}

# Service-naam → YAML-bestandsnaam
GEMEENTE_YAML_FILES: dict[str, str] = {
    "GEMEENTE_AMSTERDAM":  "GEMEENTE_AMSTERDAM-2023-01-01.yaml",
    "GEMEENTE_ROTTERDAM":  "GEMEENTE_ROTTERDAM-2023-01-01.yaml",
    "GEMEENTE_DEN_HAAG":   "GEMEENTE_DEN_HAAG-2023-01-01.yaml",
    "GEMEENTE_UTRECHT":    "GEMEENTE_UTRECHT-2023-01-01.yaml",
    "GEMEENTE_EINDHOVEN":  "GEMEENTE_EINDHOVEN-2023-01-01.yaml",
    "GEMEENTE_GRONINGEN":  "GEMEENTE_GRONINGEN-2023-01-01.yaml",
    "GEMEENTE_MAASTRICHT": "GEMEENTE_MAASTRICHT-2023-01-01.yaml",
}

_BIJSTAND_GEMEENTEN_DIR = PROJECT_ROOT / "laws" / "participatiewet" / "bijstand" / "gemeenten"
_DEFAULT_GEMEENTE = "GEMEENTE_AMSTERDAM"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_gemeente(profile: dict) -> str:
    """Detecteer de gemeente van een profiel via woonplaats in RvIG verblijfplaats."""
    verblijf = profile.get("sources", {}).get("RvIG", {}).get("verblijfplaats", [])
    if isinstance(verblijf, list) and verblijf:
        woonplaats = verblijf[0].get("woonplaats", "")
    else:
        woonplaats = ""
    return WOONPLAATS_TO_SERVICE.get(woonplaats.lower(), _DEFAULT_GEMEENTE)


def _load_gemeente_yaml(gemeente: str) -> dict | None:
    """Laad het gemeentelijke bijstand YAML-bestand. Geeft None als het niet bestaat."""
    filename = GEMEENTE_YAML_FILES.get(gemeente)
    if not filename:
        return None
    yaml_path = _BIJSTAND_GEMEENTEN_DIR / filename
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        return yaml.load(f, Loader=Loader)


# ---------------------------------------------------------------------------
# Law YAML loader (voor extract.py — laadt Amsterdam als default)
# ---------------------------------------------------------------------------

def load_law_yaml(law_name: str = "bijstand") -> dict:
    """Laad het bijstand YAML. De DecisionGraphExtractor vervangt dit per profiel."""
    result = _load_gemeente_yaml(_DEFAULT_GEMEENTE)
    if result is None:
        raise FileNotFoundError(
            f"Bijstand YAML niet gevonden voor {_DEFAULT_GEMEENTE} in {_BIJSTAND_GEMEENTEN_DIR}"
        )
    return result


# ---------------------------------------------------------------------------
# Custom run_calculation — detecteert gemeente en roept evaluate aan
# ---------------------------------------------------------------------------

def run_calculation(law_name: str, bsn: str) -> dict | None:
    """Voer de bijstandsberekening uit voor de juiste gemeente."""
    try:
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        all_profiles = load_profiles()
        profile = all_profiles.get(bsn, {})
        gemeente = _detect_gemeente(profile)

        services = get_machine_service()
        today = datetime.today().strftime("%Y-%m-%d")

        underlying = getattr(services, "services", None)
        if underlying is None:
            return None

        result = underlying.evaluate(
            service=gemeente,
            law="participatiewet/bijstand",
            parameters={"BSN": str(bsn)},
            reference_date=today,
            approved=False,
        )

        return {
            "eligibility": result.requirements_met,
            "requirements_met": result.requirements_met,
            "result": result.output,
            "input_data": result.input,
            "missing_requirements": result.missing_required,
            "missing_required": result.missing_required,
        }

    except Exception as e:
        print(f"Warning: Could not run bijstand calculation for {bsn}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Generic bijstand DecisionGraphExtractor
# ---------------------------------------------------------------------------

class DecisionGraphExtractor(_BaseDecisionGraphExtractor):
    """
    Decision graph extractor voor Bijstand (Participatiewet) — alle gemeenten.

    Detecteert automatisch de gemeente van het profiel en laadt het bijbehorende
    gemeente-YAML. Leest arbeid/re-integratie data uit de juiste gemeentetabel.
    """

    FIELD_TRANSLATIONS = {
        "LEEFTIJD":                                 "Uw leeftijd",
        "HEEFT_VAST_ADRES":                         "U heeft een vast woonadres",
        "POSTADRES":                                "Uw postadres (briefadres)",
        "IS_ONDERNEMER":                            "U bent ondernemer / ZZP-er",
        "BEDRIJFSINKOMEN":                          "Uw jaarlijkse bedrijfsinkomen",
        "VOLDOET_AAN_LANDELIJKE_VOORWAARDEN":       "U voldoet aan de landelijke bijstandsvoorwaarden",
        "LANDELIJK_BASISBEDRAG":                    "Landelijke bijstandsnorm",
        "KOSTENDELERSNORM":                         "Kostendelersnorm",
        "ARBEIDSVERMOGEN.arbeidsvermogen":          "Uw arbeidsvermogensstatus",
        "ARBEIDSVERMOGEN.re_integratie_traject":    "Uw re-integratietraject",
    }

    def __init__(self, law_yaml: dict, profile: dict, bsn: str, calc_result: dict | None = None):
        # Detecteer gemeente en laad het juiste YAML voor dit profiel
        self._gemeente = _detect_gemeente(profile)
        gemeente_yaml = _load_gemeente_yaml(self._gemeente)
        if gemeente_yaml is not None:
            law_yaml = gemeente_yaml
        super().__init__(law_yaml, profile, bsn, calc_result)

    # ------------------------------------------------------------------ #
    # Profile value extraction                                            #
    # ------------------------------------------------------------------ #

    def _extract_profile_values(self) -> dict:
        """Extract bijstand-relevante waarden uit het profiel en calc_result."""
        sources = self.profile.get("sources", {})
        values: dict[str, dict] = {}

        def get_nested(service: str, table: str, field: str) -> Any:
            try:
                rows = sources.get(service, {}).get(table, [])
                if isinstance(rows, list) and rows:
                    return rows[0].get(field)
                if isinstance(rows, dict):
                    return rows.get(field)
            except (KeyError, IndexError, TypeError):
                pass
            return None

        # --- Van RvIG ---
        age = get_nested("RvIG", "personen", "age")
        if age is not None:
            values["LEEFTIJD"] = {"value": age, "source": "RvIG", "unit": "jaar"}

        has_fixed_address = get_nested("RvIG", "personen", "has_fixed_address")
        if has_fixed_address is not None:
            values["HEEFT_VAST_ADRES"] = {"value": has_fixed_address, "source": "RvIG", "unit": ""}

        input_data = (self.calc_result or {}).get("input_data", {})

        postadres = input_data.get("$POSTADRES")
        if postadres is not None:
            values["POSTADRES"] = {"value": postadres, "source": "RvIG", "unit": ""}
        else:
            verblijf_rows = sources.get("RvIG", {}).get("verblijfplaats", [])
            if isinstance(verblijf_rows, list) and verblijf_rows:
                vb = verblijf_rows[0]
                straat = vb.get("straat", "")
                huisnr = vb.get("huisnummer", "")
                postcode = vb.get("postcode", "")
                woonplaats = vb.get("woonplaats", "")
                if straat:
                    formatted = f"{straat} {huisnr}, {postcode} {woonplaats}".strip(", ")
                    values["POSTADRES"] = {"value": formatted, "source": "RvIG (verblijfplaats)", "unit": ""}

        # --- Van de gemeente (dynamisch) ---
        arbeid_rows = sources.get(self._gemeente, {}).get("werk_en_re_integratie", [])
        if isinstance(arbeid_rows, list) and arbeid_rows:
            arbeid = arbeid_rows[0]
            arbeidsvermogen = arbeid.get("arbeidsvermogen")
            if arbeidsvermogen is not None:
                values["ARBEIDSVERMOGEN.arbeidsvermogen"] = {
                    "value": arbeidsvermogen, "source": self._gemeente, "unit": "",
                }
            re_traject = arbeid.get("re_integratie_traject")
            values["ARBEIDSVERMOGEN.re_integratie_traject"] = {
                "value": re_traject, "source": self._gemeente, "unit": "",
            }

        # --- Van BELASTINGDIENST ---
        bedrijfsinkomen = get_nested("BELASTINGDIENST", "bedrijfsinkomen", "bedrag")
        if bedrijfsinkomen is not None:
            values["BEDRIJFSINKOMEN"] = {
                "value": bedrijfsinkomen,
                "source": "BELASTINGDIENST",
                "unit": "eurocent",
            }
            values["IS_ONDERNEMER"] = {
                "value": bedrijfsinkomen > 0,
                "source": "BELASTINGDIENST (afgeleid)",
                "unit": "",
            }

        landelijk_voldoet = input_data.get("$VOLDOET_AAN_LANDELIJKE_VOORWAARDEN")
        if landelijk_voldoet is not None:
            values["VOLDOET_AAN_LANDELIJKE_VOORWAARDEN"] = {
                "value": landelijk_voldoet, "source": "SZW", "unit": "",
            }

        basisbedrag = input_data.get("$LANDELIJK_BASISBEDRAG")
        if basisbedrag is not None:
            values["LANDELIJK_BASISBEDRAG"] = {
                "value": basisbedrag, "source": "SZW", "unit": "eurocent",
            }

        kostendelersnorm = input_data.get("$KOSTENDELERSNORM")
        if kostendelersnorm is not None:
            values["KOSTENDELERSNORM"] = {
                "value": kostendelersnorm, "source": "SZW", "unit": "",
            }

        return values

    # ------------------------------------------------------------------ #
    # Extended condition evaluation                                        #
    # ------------------------------------------------------------------ #

    def _evaluate_extended_condition(self, cond: dict) -> tuple[bool | None, str]:
        subject = cond.get("subject", "")
        operation = cond.get("operation", "")
        value = cond.get("value", "")
        values_ref = cond.get("values")

        if not subject or not operation:
            return None, self.STATUS_NOT_APPLICABLE

        name = subject[1:] if subject.startswith("$") else subject

        if operation == "NOT_NULL":
            if name in self.profile_values:
                actual = self.profile_values[name]["value"]
                result = actual is not None
                return result, self.STATUS_SATISFIED if result else self.STATUS_FAILED
            return None, self.STATUS_NOT_APPLICABLE

        if operation == "IN":
            if name in self.profile_values:
                actual = self.profile_values[name]["value"]
                if actual is None:
                    return None, self.STATUS_NOT_APPLICABLE
                if isinstance(values_ref, str) and values_ref.startswith("$"):
                    expected_list = self._get_value(values_ref)
                else:
                    expected_list = values_ref
                if isinstance(expected_list, list):
                    result = actual in expected_list
                    return result, self.STATUS_SATISFIED if result else self.STATUS_FAILED
            return None, self.STATUS_NOT_APPLICABLE

        return self._evaluate_condition(subject, operation, value)

    def _add_or_group_node(self, or_conditions: list, group_id: str, used_definitions: set) -> None:
        sub_results: list[tuple[bool | None, str]] = []
        sub_labels: list[str] = []

        for sub_cond in or_conditions:
            result, status = self._evaluate_extended_condition(sub_cond)
            sub_results.append((result, status))

            subject = sub_cond.get("subject", "")
            operation = sub_cond.get("operation", "")
            values_ref = sub_cond.get("values")
            value = sub_cond.get("value", "")
            subject_name = subject[1:] if subject.startswith("$") else subject

            ref = values_ref if values_ref else (value if isinstance(value, str) else "")
            if isinstance(ref, str) and ref.startswith("$"):
                val_name = ref[1:]
                if val_name in self.definitions:
                    used_definitions.add(val_name)

            field_label = self.FIELD_TRANSLATIONS.get(subject_name, subject_name)
            if name_in_pv := self.profile_values.get(subject_name):
                actual_display = self._format_value(name_in_pv["value"], name_in_pv.get("unit", ""))
            else:
                actual_display = "?"

            if operation == "NOT_NULL":
                pv_entry = self.profile_values.get(subject_name)
                if pv_entry is None:
                    sub_labels.append(f"{field_label}: onbekend")
                elif pv_entry["value"] is not None:
                    sub_labels.append(f"{field_label}: aanwezig ({pv_entry['value']})")
                else:
                    sub_labels.append(f"{field_label}: afwezig (leeg)")
            elif operation == "IN":
                ref_label = ref[1:] if isinstance(ref, str) and ref.startswith("$") else str(ref)
                sub_labels.append(f"{field_label}: {actual_display} (in {ref_label})")
            else:
                op_sym = {"EQUALS": "==", "GREATER_OR_EQUAL": ">=", "LESS_OR_EQUAL": "<="}
                op_display = op_sym.get(operation, operation)
                sub_labels.append(f"{field_label} {op_display} {value}")

        any_true = any(r is True for r, _ in sub_results)
        any_unknown = any(r is None for r, _ in sub_results)

        if any_true:
            overall_status = self.STATUS_SATISFIED
            overall_result: bool | None = True
        elif not any_unknown:
            overall_status = self.STATUS_FAILED
            overall_result = False
        else:
            overall_status = self.STATUS_NOT_APPLICABLE
            overall_result = None

        combined_label = " OF ".join(sub_labels)
        rule_node = GraphNode(
            id=group_id,
            type="RULE",
            label=combined_label,
            properties={
                "subject": group_id,
                "operation": "OR",
                "actual_value": overall_result,
                "expected_value": True,
                "status": overall_status,
                "result": overall_result,
                "is_or_group": True,
            },
        )
        self.graph.add_node(rule_node)
        self.graph.add_edge(GraphEdge(
            source=group_id, target="decision",
            relation=overall_status, properties={"evaluated": True},
        ))

        for sub_cond in or_conditions:
            subject = sub_cond.get("subject", "")
            if not subject:
                continue
            subject_name = subject[1:] if subject.startswith("$") else subject
            pv = self.profile_values.get(subject_name)
            actual_val = pv["value"] if pv else None
            actual_display = self._format_value(actual_val)
            fact_id = f"fact_{subject_name.replace('.', '_')}"
            fact_node = GraphNode(
                id=fact_id, type="FACT",
                label=f"{subject_name} = {actual_display}",
                properties={
                    "field": subject_name,
                    "value": actual_val,
                    "source": pv.get("source", "Niet beschikbaar") if pv else "Niet beschikbaar",
                },
            )
            self.graph.add_node(fact_node)
            self.graph.add_edge(GraphEdge(source=f"person_{self.bsn}", target=fact_id, relation="HAS_FACT"))
            self.graph.add_edge(GraphEdge(source=fact_id, target=group_id, relation="USED_IN"))

    # ------------------------------------------------------------------ #
    # Extract                                                              #
    # ------------------------------------------------------------------ #

    def extract(self) -> KnowledgeGraph:
        requirements_met = (self.calc_result or {}).get("requirements_met", False)
        output = (self.calc_result or {}).get("result", {})
        uitkering = output.get("uitkeringsbedrag", 0) or 0

        if requirements_met and uitkering > 0:
            decision_label = f"RECHT: {uitkering / 100:,.2f} euro per maand"
        elif requirements_met:
            decision_label = "RECHT OP BIJSTAND"
        else:
            decision_label = "GEEN RECHT OP BIJSTAND"

        decision_node = GraphNode(
            id="decision", type="DECISION", label=decision_label,
            properties={
                "requirements_met": requirements_met,
                "law": self.law.get("name", ""),
                "output": output,
            },
        )
        self.graph.add_node(decision_node)

        person_node = GraphNode(
            id=f"person_{self.bsn}", type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn},
        )
        self.graph.add_node(person_node)
        self.graph.add_edge(GraphEdge(
            source=f"person_{self.bsn}", target="decision", relation="KRIJGT_BESLISSING",
        ))

        requirements = self.law.get("requirements", [])
        used_definitions: set[str] = set()

        for req in requirements:
            conditions = req.get("all") or req.get("any") or [req]
            for i, cond in enumerate(conditions):
                if "or" in cond:
                    self._add_or_group_node(cond["or"], f"rule_or_{i}", used_definitions)
                    continue

                subject = cond.get("subject", "")
                operation = cond.get("operation", "")
                value = cond.get("value", "")

                if not subject or not operation:
                    continue

                result, status = self._evaluate_condition(subject, operation, value)

                subject_name = subject[1:] if subject.startswith("$") else subject
                actual_value = self._get_value(subject)
                actual_unit = self.profile_values.get(subject_name, {}).get("unit", "")
                value_name = value[1:] if isinstance(value, str) and value.startswith("$") else None
                expected_value = self._get_value(value)

                if value_name and value_name in self.definitions:
                    used_definitions.add(value_name)

                op_labels = {
                    "GREATER_OR_EQUAL": ">=", "GREATER_THAN": ">",
                    "LESS_OR_EQUAL": "<=", "LESS_THAN": "<",
                    "EQUALS": "==", "NOT_EQUALS": "!=",
                }
                op_label = op_labels.get(operation, operation)
                actual_display = self._format_value(actual_value, actual_unit)
                expected_display = self._format_value(expected_value)

                if actual_value is None:
                    rule_label = f"{subject_name}: ? {op_label} {expected_display}"
                else:
                    rule_label = f"{subject_name}: {actual_display} {op_label} {expected_display}"

                rule_node = GraphNode(
                    id=f"rule_{subject_name}", type="RULE", label=rule_label,
                    properties={
                        "subject": subject_name, "operation": operation,
                        "actual_value": actual_value, "expected_value": expected_value,
                        "status": status, "result": result,
                    },
                )
                self.graph.add_node(rule_node)
                self.graph.add_edge(GraphEdge(
                    source=f"rule_{subject_name}", target="decision",
                    relation=status, properties={"evaluated": True},
                ))

                fact_node = GraphNode(
                    id=f"fact_{subject_name}", type="FACT",
                    label=f"{subject_name} = {actual_display}",
                    properties={
                        "field": subject_name, "value": actual_value,
                        "source": self.profile_values.get(subject_name, {}).get("source", "Niet beschikbaar"),
                    },
                )
                self.graph.add_node(fact_node)
                self.graph.add_edge(GraphEdge(source=f"person_{self.bsn}", target=f"fact_{subject_name}", relation="HAS_FACT"))
                self.graph.add_edge(GraphEdge(source=f"fact_{subject_name}", target=f"rule_{subject_name}", relation="USED_IN"))

        # Definitie-drempelwaarden
        for def_name in used_definitions:
            def_value = self.definitions[def_name]
            def_display = self._format_value(def_value)
            def_node = GraphNode(
                id=f"def_{def_name}", type="THRESHOLD",
                label=f"{def_name} = {def_display}",
                properties={"name": def_name, "value": def_value},
            )
            self.graph.add_node(def_node)
            for node in self.graph.nodes:
                if node.type == "RULE" and node.properties.get("expected_value") == def_value:
                    self.graph.add_edge(GraphEdge(
                        source=f"def_{def_name}", target=node.id, relation="DEFINES_THRESHOLD",
                    ))

        # Berekeningsnode
        if requirements_met and uitkering > 0:
            calc_node = GraphNode(
                id="amount_calc", type="CALCULATION",
                label=f"Berekend uitkeringsbedrag: {uitkering / 100:,.2f} euro per maand",
                properties={"output_field": "uitkeringsbedrag", "output_amount": uitkering},
            )
            self.graph.add_node(calc_node)
            if "fact_LANDELIJK_BASISBEDRAG" in [n.id for n in self.graph.nodes]:
                self.graph.add_edge(GraphEdge(
                    source="fact_LANDELIJK_BASISBEDRAG", target="amount_calc",
                    relation=self.STATUS_AFFECTS_AMOUNT,
                ))
            self.graph.add_edge(GraphEdge(source="amount_calc", target="decision", relation="DETERMINES"))

        return self.graph

    # ------------------------------------------------------------------ #
    # Human-readable rule formatting                                       #
    # ------------------------------------------------------------------ #

    def _format_rule_human_readable(self, rule_node: GraphNode) -> str:
        if rule_node.properties.get("is_or_group"):
            label = rule_node.label
            # Vertaal technische codes naar leesbare tekst
            arbeid = self.profile_values.get("ARBEIDSVERMOGEN.arbeidsvermogen", {}).get("value")
            re_traject = self.profile_values.get("ARBEIDSVERMOGEN.re_integratie_traject", {}).get("value")
            if "VOLLEDIGE_ONTHEFFING_REDENEN" in label or "re_integratie_traject" in label:
                if arbeid and re_traject is None:
                    return f"U bent arbeidsgeschikt ({arbeid.lower()}) en heeft geen re-integratietraject"
                if arbeid and re_traject:
                    return f"U bent arbeidsgeschikt ({arbeid.lower()}) maar heeft wel een re-integratietraject: {re_traject}"
                if arbeid is None:
                    return "Uw arbeidsgeschiktheid is niet bekend bij de gemeente"
            return label

        subject = rule_node.properties.get("subject", "")
        operation = rule_node.properties.get("operation", "")
        actual_value = rule_node.properties.get("actual_value")
        expected_value = rule_node.properties.get("expected_value")

        field_name = self.FIELD_TRANSLATIONS.get(subject, subject)
        op_text = self.OPERATION_TRANSLATIONS.get(operation, operation)
        unit = self.profile_values.get(subject, {}).get("unit", "")
        actual_display = self._format_value(actual_value, unit)
        expected_display = self._format_value(expected_value, unit if unit else "")

        if isinstance(actual_value, bool):
            if subject == "HEEFT_VAST_ADRES":
                return "Ja, u heeft een vast woonadres" if actual_value else "Nee, u heeft geen vast woonadres"
            if subject == "IS_ONDERNEMER":
                return "Ja, u bent ondernemer/ZZP-er" if actual_value else "Nee, u bent geen ondernemer"
            if subject == "VOLDOET_AAN_LANDELIJKE_VOORWAARDEN":
                return (
                    "Ja, u voldoet aan de landelijke bijstandsvoorwaarden"
                    if actual_value
                    else "Nee, u voldoet niet aan de landelijke bijstandsvoorwaarden"
                )
            return f"{field_name}: {'Ja' if actual_value else 'Nee'}"

        if subject == "LEEFTIJD":
            return f"U bent {actual_display} jaar oud (vereist: {op_text} {expected_display} jaar)"

        if subject == "BEDRIJFSINKOMEN":
            return f"Uw bedrijfsinkomen: {actual_display} ({op_text} {expected_display})"

        return f"{field_name}: {actual_display} ({op_text} {expected_display})"

    # ------------------------------------------------------------------ #
    # Explanation skeleton                                                  #
    # ------------------------------------------------------------------ #

    def to_explanation_skeleton(self) -> str:
        lines = []

        decision_node = next((n for n in self.graph.nodes if n.type == "DECISION"), None)
        if not decision_node:
            return "Geen beslissing gevonden."

        requirements_met = decision_node.properties.get("requirements_met", False)
        law_name = decision_node.properties.get("law", "")
        output = decision_node.properties.get("output", {})

        gemeente_label = self._gemeente.replace("GEMEENTE_", "").capitalize()
        info_url = GEMEENTE_URLS.get(self._gemeente, "www.gemeenten.nl")

        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        lines.append("## Persoonlijke situatie:")

        if "LEEFTIJD" in self.profile_values:
            lines.append(f"- Leeftijd: {self.profile_values['LEEFTIJD']['value']} jaar")
        else:
            lines.append("- Leeftijd: Onbekend")

        if "HEEFT_VAST_ADRES" in self.profile_values:
            vast = self.profile_values["HEEFT_VAST_ADRES"]["value"]
            lines.append(f"- Vast woonadres: {'Ja' if vast else 'Nee'}")
        else:
            lines.append("- Vast woonadres: Onbekend")

        if "POSTADRES" in self.profile_values and not self.profile_values.get("HEEFT_VAST_ADRES", {}).get("value"):
            postadres = self.profile_values["POSTADRES"]["value"]
            lines.append(f"- Postadres (briefadres): {postadres or 'Onbekend'}")

        if "ARBEIDSVERMOGEN.arbeidsvermogen" in self.profile_values:
            arbeid = self.profile_values["ARBEIDSVERMOGEN.arbeidsvermogen"]["value"]
            lines.append(f"- Arbeidsvermogen: {arbeid or 'Onbekend'}")
            re_traject = self.profile_values.get("ARBEIDSVERMOGEN.re_integratie_traject", {}).get("value")
            if re_traject:
                lines.append(f"- Re-integratietraject: {re_traject}")
        else:
            lines.append(f"- Arbeidsvermogen: Onbekend (geen gegevens van gemeente {gemeente_label})")

        if "IS_ONDERNEMER" in self.profile_values:
            zzp = self.profile_values["IS_ONDERNEMER"]["value"]
            lines.append(f"- Ondernemer/ZZP-er: {'Ja' if zzp else 'Nee'}")
            if zzp and "BEDRIJFSINKOMEN" in self.profile_values:
                bi = self.profile_values["BEDRIJFSINKOMEN"]["value"]
                lines.append(f"- Jaarlijks bedrijfsinkomen: {self._format_value(bi, 'eurocent')}")

        if "VOLDOET_AAN_LANDELIJKE_VOORWAARDEN" in self.profile_values:
            ln = self.profile_values["VOLDOET_AAN_LANDELIJKE_VOORWAARDEN"]["value"]
            lines.append(f"- Voldoet aan landelijke voorwaarden: {'Ja' if ln else 'Nee'}")

        lines.append("")

        satisfied_rules, failed_rules, unknown_rules = [], [], []
        for node in self.graph.nodes:
            if node.type == "RULE":
                status = node.properties.get("status", "")
                if status == self.STATUS_SATISFIED:
                    satisfied_rules.append(node)
                elif status == self.STATUS_FAILED:
                    failed_rules.append(node)
                elif status == self.STATUS_NOT_APPLICABLE:
                    unknown_rules.append(node)

        if unknown_rules:
            lines.append("## Voorwaarden die we niet kunnen beoordelen:")
            lines.append("(Er ontbreken gegevens)")
            for rule in unknown_rules:
                if rule.properties.get("is_or_group"):
                    lines.append(f"- {rule.label}: gegevens ontbreken")
                else:
                    subject = rule.properties.get("subject", "")
                    field_name = self.FIELD_TRANSLATIONS.get(subject, subject)
                    lines.append(f"- {field_name}: gegevens ontbreken")
            lines.append("")

        if failed_rules:
            lines.append("## Voorwaarden waar u NIET aan voldoet:")
            for rule in failed_rules:
                lines.append(f"- [NEE] {self._format_rule_human_readable(rule)}")
            lines.append("")

        if satisfied_rules:
            lines.append("## Voorwaarden waar u WEL aan voldoet:")
            for rule in satisfied_rules:
                lines.append(f"- [JA] {self._format_rule_human_readable(rule)}")
            lines.append("")

        calc_node = next((n for n in self.graph.nodes if n.type == "CALCULATION"), None)
        uitkering = output.get("uitkeringsbedrag", 0) or 0
        if calc_node and requirements_met and uitkering > 0:
            lines.append("## Berekening uitkeringsbedrag:")

            if "LANDELIJK_BASISBEDRAG" in self.profile_values:
                basis = self.profile_values["LANDELIJK_BASISBEDRAG"]["value"]
                lines.append(f"- Landelijke bijstandsnorm: {self._format_value(basis, 'eurocent')}")

            if "KOSTENDELERSNORM" in self.profile_values:
                kdn = self.profile_values["KOSTENDELERSNORM"]["value"]
                lines.append(f"- Kostendelersnorm: {kdn}")

            lines.append(f"- Uitkeringsbedrag: {self._format_value(uitkering, 'eurocent')} per maand")

            woonkostentoeslag = output.get("woonkostentoeslag", 0) or 0
            if woonkostentoeslag > 0:
                lines.append(f"- Woonkostentoeslag (briefadres): {self._format_value(woonkostentoeslag, 'eurocent')} per maand")

            startkapitaal = output.get("startkapitaal", 0) or 0
            if startkapitaal > 0:
                lines.append(f"- Startkapitaal ZZP: {self._format_value(startkapitaal, 'eurocent')}")

            lines.append("")

        lines.append("## Conclusie:")
        if requirements_met and uitkering > 0:
            lines.append(f"U heeft recht op {law_name}.")
        elif requirements_met:
            lines.append(f"U voldoet aan de voorwaarden voor {law_name}, maar er wordt geen bedrag uitbetaald.")
        else:
            lines.append(f"U heeft geen recht op {law_name}.")

        lines.append("")
        lines.append("## Meer informatie:")
        lines.append(f"Voor meer informatie over bijstand in {gemeente_label} kunt u terecht op {info_url}")

        return "\n".join(lines)
