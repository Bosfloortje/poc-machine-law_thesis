#!/usr/bin/env python3
"""
GraphRAG extraction script for Alcoholwetvergunning Horeca (GEMEENTE_ROTTERDAM).

Imports shared graph infrastructure from extraction_zorgtoeslag and overrides
the alcoholwet-specific parts: field translations, profile value extraction,
decision graph logic, and the explanation skeleton.

This law applies to businesses (horecabedrijven), not individual citizens.
The main parameter is KVK_NUMMER (from the leidinggevende's profile data).

Law YAML: laws/alcoholwet/vergunning/gemeenten/GEMEENTE_ROTTERDAM-2024-01-01.yaml
Service name: "GEMEENTE_ROTTERDAM", law: "alcoholwet/vergunning"
Info URL: www.rotterdam.nl

Usage:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --law alcoholwet --profiles 999999990
"""

import sys
from pathlib import Path
from typing import Any

# scripts/ -> llm_explanations/ -> analysis/ -> root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# ---------------------------------------------------------------------------
# Import shared infrastructure from extraction_zorgtoeslag
# ---------------------------------------------------------------------------
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
# Alcoholwet-specific LLM prompt
# ---------------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse ondernemers uitleg geeft over vergunningsbeslissingen.

Je taak is om een beslissingsskeleton om te zetten naar een korte, begrijpelijke uitleg.

VERPLICHT:
- Gebruik ALLEEN informatie uit het skeleton - voeg NIETS toe
- Schrijf in eenvoudig Nederlands (B1-niveau) - korte zinnen, gewone woorden
- Dit is een informatieve tekst, GEEN brief of gesprek
- Eindig altijd met de tekst over de gemeente-website uit het skeleton

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN verwijzingen naar instanties die niet in het skeleton staan
- GEEN aanbiedingen voor hulp of vragen aan de lezer
- GEEN technische termen of codes behouden - alles moet in normale taal"""


def create_decision_prompt(skeleton: str, person_name: str) -> str:
    """Create an alcoholwet-specific LLM prompt using the decision skeleton."""
    import re
    url_match = re.search(r'www\.\S+', skeleton)
    info_url = url_match.group(0).rstrip(".,)") if url_match else "www.gemeenten.nl"

    return f"""# Informatie over de vergunningsbeslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau).

WAT JE MOET DOEN:
- Schrijf de tekst uit de "Conclusie"-sectie van het skeleton in eigen woorden
- Als het skeleton een "Reden"-sectie heeft, gebruik die als uitleg
- Als het skeleton voorwaarden heeft, noem alleen de voorwaarden waaraan NIET voldaan wordt
- Eindig met de zin over {info_url}
- Praat altijd in de u-vorm tegen {person_name}

WAT JE NIET MAG DOEN:
- GEEN briefopmaak gebruiken (geen "Geachte heer/mevrouw", geen "Met vriendelijke groet")
- GEEN verwijzingen naar andere instanties dan die in het skeleton staan
- GEEN extra voorwaarden of regels verzinnen die niet in het skeleton staan
- GEEN verwijzing naar "uw horecabedrijf" als het skeleton zegt dat er geen onderneming is
- GEEN herhaling - zeg het één keer, niet twee keer op een andere manier
- GEEN vragen stellen aan de lezer of aanbod om te helpen
- GEEN derde persoon aanspreken - schrijf direct tegen {person_name}

VOORBEELD BIJ GEEN ONDERNEMING (structuurvoorbeeld):
"U heeft geen onderneming. U komt daarom niet in aanmerking voor een alcoholvergunning.
Voor meer informatie kunt u terecht op {info_url}"

VOORBEELD BIJ WEL RECHT (structuurvoorbeeld):
"U heeft recht op een alcoholvergunning.
U voldoet aan alle voorwaarden: uw leidinggevende heeft een SVH-diploma en er is geen Bibob-bezwaar.
Voor meer informatie kunt u terecht op {info_url}"

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"U heeft geen alcoholvergunning voor uw horecabedrijf. U heeft geen onderneming." <- FOUT: tegenstrijdig en dubbel
"Neem contact op met de politie" <- FOUT: instantie niet in skeleton"""


def generate_decision_explanation(
    decision_extractor: "DecisionGraphExtractor",
    person_name: str,
    api_key: str | None = None,
    model: str = "haiku",
) -> dict:
    """Generate an alcoholwet LLM explanation using the constrained decision skeleton."""
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
# Alcoholwet-specific YAML loader
# ---------------------------------------------------------------------------

def load_law_yaml(law_name: str = "alcoholwet") -> dict:
    """Load the Alcoholwet Rotterdam YAML."""
    yaml_path = (
        PROJECT_ROOT
        / "laws"
        / "alcoholwet"
        / "vergunning"
        / "gemeenten"
        / "GEMEENTE_ROTTERDAM-2024-01-01.yaml"
    )
    if not yaml_path.exists():
        raise FileNotFoundError(f"Alcoholwet Rotterdam YAML not found at: {yaml_path}")
    with open(yaml_path) as f:
        return yaml.load(f, Loader=Loader)


# ---------------------------------------------------------------------------
# Custom run_calculation — uses KVK_NUMMER instead of BSN
# ---------------------------------------------------------------------------

def run_calculation(law_name: str, bsn: str) -> dict | None:
    """Run the alcoholwet calculation using the KVK_NUMMER found in the profile."""
    try:
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service
        from datetime import datetime

        all_profiles = load_profiles()
        profile = all_profiles.get(bsn, {})
        sources = profile.get("sources", {})

        # Find KVK_NUMMER from leidinggevenden table (KVK or any municipality service)
        kvk_nummer = None
        for svc_name in [
            "KVK",
            "GEMEENTE_ROTTERDAM", "GEMEENTE_AMSTERDAM", "GEMEENTE_DEN_HAAG",
            "GEMEENTE_EINDHOVEN", "GEMEENTE_GRONINGEN", "GEMEENTE_MAASTRICHT", "GEMEENTE_UTRECHT",
        ]:
            rows = sources.get(svc_name, {}).get("leidinggevenden", [])
            if isinstance(rows, list) and rows:
                kvk_nummer = rows[0].get("kvk_nummer")
                if kvk_nummer:
                    break

        if not kvk_nummer:
            return None

        services = get_machine_service()
        today = datetime.today().strftime("%Y-%m-%d")

        # Call evaluate directly on the underlying Services object
        underlying = getattr(services, "services", None)
        if underlying is None:
            return None

        result = underlying.evaluate(
            service="GEMEENTE_ROTTERDAM",
            law="alcoholwet/vergunning",
            parameters={"KVK_NUMMER": str(kvk_nummer)},
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
        print(f"Warning: Could not run alcoholwet calculation for {bsn}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Alcoholwet-specific DecisionGraphExtractor
# ---------------------------------------------------------------------------

class DecisionGraphExtractor(_BaseDecisionGraphExtractor):
    """
    Decision graph extractor for Alcoholwetvergunning Horeca Rotterdam.

    Overrides field translations, profile value extraction, decision label
    logic and the explanation skeleton. Everything else is inherited.
    """

    FIELD_TRANSLATIONS = {
        "BEDRIJF_STATUS":               "Status onderneming in handelsregister",
        "SVH_REGISTRATIE_GELDIG":       "SVH-registratie geldig (Register Sociale Hygiene)",
        "VOLDOET_AAN_NATIONALE_EISEN":  "Voldoet aan nationale Alcoholwet-eisen (art. 8 en 10)",
        "BIBOB_MATE_VAN_GEVAAR":        "Bibob-advies mate van gevaar",
        "LEEFTIJD_EXPLOITANT":          "Leeftijd leidinggevende",
        "IS_ONDER_CURATELE_EXPLOITANT": "Leidinggevende staat onder curatele",
        "VLOEROPPERVLAKTE_HORECALOKALITEIT": "Vloeroppervlakte horecalokaliteit (m2)",
        "TYPE_BEDRIJF":                 "Type bedrijf",
        "HEEFT_ACTIEVE_ALCOHOLWETVERGUNNING": "Heeft actieve Alcoholwetvergunning",
        "SVH_REGISTRATIENUMMER":        "SVH-registratienummer",
        "WEIGERINGSGROND_NATIONAAL":    "Weigeringsgrond (nationaal)",
    }

    # ------------------------------------------------------------------ #
    # Profile value extraction                                             #
    # ------------------------------------------------------------------ #

    def _extract_profile_values(self) -> dict:
        """Extract alcoholwet-relevant values from profile sources and calc_result."""
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

        # Find which service's tables have this profile's data (KVK or municipality)
        gemeente = None
        for svc_name in [
            "KVK",
            "GEMEENTE_ROTTERDAM", "GEMEENTE_AMSTERDAM", "GEMEENTE_DEN_HAAG",
            "GEMEENTE_EINDHOVEN", "GEMEENTE_GRONINGEN", "GEMEENTE_MAASTRICHT", "GEMEENTE_UTRECHT",
        ]:
            if sources.get(svc_name, {}).get("leidinggevenden"):
                gemeente = svc_name
                break

        if gemeente:
            # --- Leidinggevende data ---
            leiding_rows = sources.get(gemeente, {}).get("leidinggevenden", [])
            leiding = leiding_rows[0] if leiding_rows else {}

            if leiding.get("leeftijd") is not None:
                values["LEEFTIJD_EXPLOITANT"] = {
                    "value": leiding["leeftijd"], "source": gemeente, "unit": "jaar",
                }

            curatele = leiding.get("is_onder_curatele")
            if curatele is not None:
                values["IS_ONDER_CURATELE_EXPLOITANT"] = {
                    "value": curatele, "source": gemeente, "unit": "",
                }

            svh_reg = leiding.get("is_ingeschreven_svh_register")
            if svh_reg is not None:
                values["SVH_REGISTRATIE_GELDIG"] = {
                    "value": svh_reg, "source": gemeente, "unit": "",
                }

            svh_nr = leiding.get("svh_registratienummer")
            if svh_nr is not None:
                values["SVH_REGISTRATIENUMMER"] = {
                    "value": svh_nr, "source": gemeente, "unit": "",
                }

            kvk_nummer = leiding.get("kvk_nummer")
            if kvk_nummer is not None:
                values["KVK_NUMMER"] = {
                    "value": str(kvk_nummer), "source": gemeente, "unit": "",
                }

            # --- Inrichting data ---
            inr_rows = sources.get(gemeente, {}).get("inrichtingen", [])
            inr = inr_rows[0] if inr_rows else {}

            vloer = inr.get("vloeroppervlakte_horecalokaliteit")
            if vloer is not None:
                values["VLOEROPPERVLAKTE_HORECALOKALITEIT"] = {
                    "value": vloer, "source": gemeente, "unit": "m2",
                }

            type_bedrijf = inr.get("type_bedrijf")
            if type_bedrijf is not None:
                values["TYPE_BEDRIJF"] = {
                    "value": type_bedrijf, "source": gemeente, "unit": "",
                }

            # --- Vergunning data ---
            verg_rows = sources.get(gemeente, {}).get("vergunningen", [])
            verg = verg_rows[0] if verg_rows else {}

            heeft_alcohol = verg.get("heeft_alcoholvergunning")
            if heeft_alcohol is not None:
                values["HEEFT_ACTIEVE_ALCOHOLWETVERGUNNING"] = {
                    "value": heeft_alcohol, "source": gemeente, "unit": "",
                }

        # --- From calc_result input_data (resolved by engine) ---
        input_data = (self.calc_result or {}).get("input_data", {})

        bedrijf_status = input_data.get("$BEDRIJF_STATUS")
        if bedrijf_status is not None:
            values["BEDRIJF_STATUS"] = {
                "value": bedrijf_status, "source": "KVK", "unit": "",
            }

        # SVH_REGISTRATIE_GELDIG from engine overrides profile if available
        svh_geldig = input_data.get("$SVH_REGISTRATIE_GELDIG")
        if svh_geldig is not None:
            values["SVH_REGISTRATIE_GELDIG"] = {
                "value": svh_geldig, "source": "SVH", "unit": "",
            }

        voldoet_nationaal = input_data.get("$VOLDOET_AAN_NATIONALE_EISEN")
        if voldoet_nationaal is not None:
            values["VOLDOET_AAN_NATIONALE_EISEN"] = {
                "value": voldoet_nationaal, "source": "VWS", "unit": "",
            }

        bibob = input_data.get("$BIBOB_MATE_VAN_GEVAAR")
        # Store even if None so IS_NULL check works
        values["BIBOB_MATE_VAN_GEVAAR"] = {
            "value": bibob, "source": "LBB (Bibob)", "unit": "",
        }

        weigeringsgrond = input_data.get("$WEIGERINGSGROND_NATIONAAL")
        if weigeringsgrond is not None:
            values["WEIGERINGSGROND_NATIONAAL"] = {
                "value": weigeringsgrond, "source": "VWS", "unit": "",
            }

        return values

    # ------------------------------------------------------------------ #
    # Extended condition evaluation (IS_NULL, NOT_EQUALS, IN, NOT_NULL)   #
    # ------------------------------------------------------------------ #

    def _evaluate_extended_condition(self, cond: dict) -> tuple[bool | None, str]:
        """Evaluate conditions including IS_NULL, NOT_NULL, IN, and standard ops."""
        subject = cond.get("subject", "")
        operation = cond.get("operation", "")
        value = cond.get("value", "")
        values_ref = cond.get("values")

        if not subject or not operation:
            return None, self.STATUS_NOT_APPLICABLE

        name = subject[1:] if subject.startswith("$") else subject

        if operation == "IS_NULL":
            pv = self.profile_values.get(name)
            if pv is None:
                return True, self.STATUS_SATISFIED  # field absent = treated as null
            result = pv["value"] is None
            return result, self.STATUS_SATISFIED if result else self.STATUS_FAILED

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

        # Standard operations
        return self._evaluate_condition(subject, operation, value)

    def _add_or_group_node(
        self,
        or_conditions: list,
        group_id: str,
        used_definitions: set,
    ) -> None:
        """Evaluate an OR group and add a single combined RULE node + edges."""
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

            # Track definitions used
            ref = values_ref if values_ref else (value if isinstance(value, str) else "")
            if isinstance(ref, str) and ref.startswith("$"):
                val_name = ref[1:]
                if val_name in self.definitions:
                    used_definitions.add(val_name)

            field_label = self.FIELD_TRANSLATIONS.get(subject_name, subject_name)
            pv_entry = self.profile_values.get(subject_name)
            actual_display = self._format_value(pv_entry["value"] if pv_entry else None)

            if operation == "IS_NULL":
                is_null = (pv_entry is None) or (pv_entry["value"] is None)
                sub_labels.append(
                    f"{field_label}: {'niet aanwezig (geen Bibob-advies)' if is_null else actual_display}"
                )
            elif operation == "NOT_NULL":
                sub_labels.append(
                    f"{field_label}: {'aanwezig' if (pv_entry and pv_entry['value'] is not None) else 'afwezig'}"
                )
            elif operation == "NOT_EQUALS":
                ref_val = self._get_value(value) if isinstance(value, str) and value.startswith("$") else value
                sub_labels.append(f"{field_label}: {actual_display} != {ref_val}")
            elif operation == "IN":
                ref_label = ref[1:] if isinstance(ref, str) and ref.startswith("$") else str(ref)
                sub_labels.append(f"{field_label}: {actual_display} (in {ref_label})")
            else:
                op_sym = {"EQUALS": "==", "GREATER_OR_EQUAL": ">=", "LESS_OR_EQUAL": "<="}
                op_display = op_sym.get(operation, operation)
                ref_val = self._get_value(value) if isinstance(value, str) and value.startswith("$") else value
                sub_labels.append(f"{field_label} {op_display} {ref_val}")

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
        self.graph.add_edge(
            GraphEdge(
                source=group_id,
                target="decision",
                relation=overall_status,
                properties={"evaluated": True},
            )
        )

        for sub_cond in or_conditions:
            subject = sub_cond.get("subject", "")
            if not subject:
                continue
            subject_name = subject[1:] if subject.startswith("$") else subject
            pv = self.profile_values.get(subject_name)
            actual_val = pv["value"] if pv else None
            fact_id = f"fact_{subject_name.replace('.', '_')}"
            fact_node = GraphNode(
                id=fact_id,
                type="FACT",
                label=f"{subject_name} = {self._format_value(actual_val)}",
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
    # Main extract method                                                  #
    # ------------------------------------------------------------------ #

    def extract(self) -> KnowledgeGraph:
        """Extract a focused decision subgraph for the alcoholwet vergunning."""
        requirements_met = (self.calc_result or {}).get("requirements_met", False)
        output = (self.calc_result or {}).get("result", {})

        if requirements_met:
            decision_label = "RECHT OP ALCOHOLWETVERGUNNING"
        else:
            weigeringsgrond = output.get("weigeringsgrond") or ""
            decision_label = f"GEEN VERGUNNING: {weigeringsgrond}" if weigeringsgrond else "GEEN VERGUNNING"

        kvk = self.profile_values.get("KVK_NUMMER", {}).get("value", self.bsn)
        decision_node = GraphNode(
            id="decision",
            type="DECISION",
            label=decision_label,
            properties={
                "requirements_met": requirements_met,
                "law": self.law.get("name", ""),
                "output": output,
            },
        )
        self.graph.add_node(decision_node)

        person_node = GraphNode(
            id=f"person_{self.bsn}",
            type="PERSON",
            label=self.profile.get("name", f"Ondernemer {kvk}"),
            properties={"bsn": self.bsn, "kvk_nummer": kvk},
        )
        self.graph.add_node(person_node)
        self.graph.add_edge(
            GraphEdge(
                source=f"person_{self.bsn}",
                target="decision",
                relation="KRIJGT_BESLISSING",
            )
        )

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

                result, status = self._evaluate_extended_condition(cond)

                subject_name = subject[1:] if subject.startswith("$") else subject
                actual_value = self._get_value(subject)
                actual_unit = self.profile_values.get(subject_name, {}).get("unit", "")
                value_name = value[1:] if isinstance(value, str) and value.startswith("$") else None
                expected_value = self._get_value(value) if isinstance(value, str) and value.startswith("$") else value

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
                    id=f"rule_{subject_name}",
                    type="RULE",
                    label=rule_label,
                    properties={
                        "subject": subject_name,
                        "operation": operation,
                        "actual_value": actual_value,
                        "expected_value": expected_value,
                        "status": status,
                        "result": result,
                    },
                )
                self.graph.add_node(rule_node)
                self.graph.add_edge(
                    GraphEdge(
                        source=f"rule_{subject_name}",
                        target="decision",
                        relation=status,
                        properties={"evaluated": True},
                    )
                )

                fact_node = GraphNode(
                    id=f"fact_{subject_name}",
                    type="FACT",
                    label=f"{subject_name} = {actual_display}",
                    properties={
                        "field": subject_name,
                        "value": actual_value,
                        "source": self.profile_values.get(subject_name, {}).get("source", "Niet beschikbaar"),
                    },
                )
                self.graph.add_node(fact_node)
                self.graph.add_edge(GraphEdge(source=f"person_{self.bsn}", target=f"fact_{subject_name}", relation="HAS_FACT"))
                self.graph.add_edge(GraphEdge(source=f"fact_{subject_name}", target=f"rule_{subject_name}", relation="USED_IN"))

        # Add used definitions as THRESHOLD nodes
        for def_name in used_definitions:
            def_value = self.definitions[def_name]
            def_node = GraphNode(
                id=f"def_{def_name}",
                type="THRESHOLD",
                label=f"{def_name} = {self._format_value(def_value)}",
                properties={"name": def_name, "value": def_value},
            )
            self.graph.add_node(def_node)
            for node in self.graph.nodes:
                if node.type == "RULE" and node.properties.get("expected_value") == def_value:
                    self.graph.add_edge(GraphEdge(source=f"def_{def_name}", target=node.id, relation="DEFINES_THRESHOLD"))

        return self.graph

    # ------------------------------------------------------------------ #
    # Human-readable rule formatting                                       #
    # ------------------------------------------------------------------ #

    def _format_rule_human_readable(self, rule_node: GraphNode) -> str:
        if rule_node.properties.get("is_or_group"):
            return rule_node.label

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
            if subject == "SVH_REGISTRATIE_GELDIG":
                return (
                    "Ja, leidinggevende staat ingeschreven in het Register Sociale Hygiene"
                    if actual_value
                    else "Nee, leidinggevende staat NIET ingeschreven in het Register Sociale Hygiene"
                )
            if subject == "VOLDOET_AAN_NATIONALE_EISEN":
                return (
                    "Ja, voldoet aan alle nationale Alcoholwet-eisen (artikel 8 en 10)"
                    if actual_value
                    else "Nee, voldoet NIET aan nationale Alcoholwet-eisen"
                )
            if subject == "IS_ONDER_CURATELE_EXPLOITANT":
                return (
                    "Leidinggevende staat onder curatele (niet toegestaan)"
                    if actual_value
                    else "Leidinggevende staat niet onder curatele"
                )
            return f"{field_name}: {'Ja' if actual_value else 'Nee'}"

        if subject == "BEDRIJF_STATUS":
            return f"Onderneming status in KVK: {actual_display} (vereist: Actief)"

        return f"{field_name}: {actual_display} ({op_text} {expected_display})"

    # ------------------------------------------------------------------ #
    # Explanation skeleton                                                 #
    # ------------------------------------------------------------------ #

    def to_explanation_skeleton(self) -> str:
        lines = []

        decision_node = next((n for n in self.graph.nodes if n.type == "DECISION"), None)
        if not decision_node:
            return "Geen beslissing gevonden."

        requirements_met = decision_node.properties.get("requirements_met", False)
        law_name = decision_node.properties.get("law", "")
        output = decision_node.properties.get("output", {})

        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        # Vroeg uitstappen als er geen bedrijfsdata is
        kvk = self.profile_values.get("KVK_NUMMER", {}).get("value")
        if not kvk and self.calc_result is None:
            lines.append("## Reden:")
            lines.append("Er is geen onderneming gevonden voor dit profiel.")
            lines.append("Een Alcoholwetvergunning is alleen mogelijk voor ondernemers met een KVK-inschrijving.")
            lines.append("")
            lines.append("## Conclusie:")
            lines.append(f"U heeft geen onderneming en komt daarom niet in aanmerking voor een {law_name}.")
            lines.append("")
            lines.append("## Meer informatie:")
            lines.append("Voor meer informatie over de Alcoholwetvergunning in Rotterdam: www.rotterdam.nl")
            return "\n".join(lines)

        # Bedrijfssituatie
        lines.append("## Bedrijfssituatie:")

        kvk = self.profile_values.get("KVK_NUMMER", {}).get("value")
        if kvk:
            lines.append(f"- KVK-nummer: {kvk}")

        type_bedrijf = self.profile_values.get("TYPE_BEDRIJF", {}).get("value")
        if type_bedrijf:
            lines.append(f"- Type bedrijf: {type_bedrijf}")

        vloer = self.profile_values.get("VLOEROPPERVLAKTE_HORECALOKALITEIT", {}).get("value")
        if vloer is not None:
            lines.append(f"- Vloeroppervlakte: {vloer} m2")

        bedrijf_status = self.profile_values.get("BEDRIJF_STATUS", {}).get("value")
        if bedrijf_status:
            lines.append(f"- Status KVK: {bedrijf_status}")

        leeftijd = self.profile_values.get("LEEFTIJD_EXPLOITANT", {}).get("value")
        if leeftijd is not None:
            lines.append(f"- Leeftijd leidinggevende: {leeftijd} jaar")

        curatele = self.profile_values.get("IS_ONDER_CURATELE_EXPLOITANT", {}).get("value")
        if curatele is not None:
            lines.append(f"- Leidinggevende onder curatele: {'Ja' if curatele else 'Nee'}")

        svh_nr = self.profile_values.get("SVH_REGISTRATIENUMMER", {}).get("value")
        svh_geldig = self.profile_values.get("SVH_REGISTRATIE_GELDIG", {}).get("value")
        if svh_nr:
            lines.append(f"- SVH-registratienummer: {svh_nr}")
        elif svh_geldig is not None:
            lines.append(f"- SVH-registratie geldig: {'Ja' if svh_geldig else 'Nee'}")

        bibob = self.profile_values.get("BIBOB_MATE_VAN_GEVAAR", {}).get("value")
        lines.append(f"- Bibob-advies: {bibob if bibob else 'Geen advies aanwezig'}")

        heeft_actief = self.profile_values.get("HEEFT_ACTIEVE_ALCOHOLWETVERGUNNING", {}).get("value")
        if heeft_actief is not None:
            lines.append(f"- Actieve Alcoholwetvergunning: {'Ja' if heeft_actief else 'Nee (nog niet verleend)'}")

        lines.append("")

        # Rules grouped by status
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
            lines.append("## Voorwaarden die niet beoordeeld konden worden:")
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
            lines.append("## Voorwaarden waaraan NIET wordt voldaan:")
            for rule in failed_rules:
                lines.append(f"- [NEE] {self._format_rule_human_readable(rule)}")
            lines.append("")

        if satisfied_rules:
            lines.append("## Voorwaarden waaraan WEL wordt voldaan:")
            for rule in satisfied_rules:
                lines.append(f"- [JA] {self._format_rule_human_readable(rule)}")
            lines.append("")

        # Weigeringsgrond (if denied)
        weigeringsgrond = output.get("weigeringsgrond")
        if not requirements_met and weigeringsgrond:
            lines.append("## Reden weigering:")
            lines.append(f"- {weigeringsgrond}")
            lines.append("")

        # Conclusie
        lines.append("## Conclusie:")
        if requirements_met:
            lines.append(f"Het bedrijf voldoet aan alle voorwaarden voor een {law_name}.")
        else:
            lines.append(f"Het bedrijf heeft geen recht op een {law_name}.")
            if weigeringsgrond:
                lines.append(f"Weigeringsgrond: {weigeringsgrond}")

        lines.append("")
        lines.append("## Meer informatie:")
        lines.append("Voor meer informatie over de Alcoholwetvergunning in Rotterdam: www.rotterdam.nl")

        return "\n".join(lines)
