#!/usr/bin/env python3
"""
GraphRAG extraction script for Bijstand Gemeente Amsterdam (Participatiewet).

Imports shared graph infrastructure from extraction_zorgtoeslag and overrides
only the bijstand-specific parts: field translations, profile value extraction,
decision graph logic, and the explanation skeleton.

Law YAML: laws/participatiewet/bijstand/gemeenten/GEMEENTE_AMSTERDAM-2023-01-01.yaml
Service name (for run_calculation): "bijstand"
Info URL: www.amsterdam.nl

Usage:
    uv run python analysis/llm_explanations/scripts/extraction_bijstand_amsterdam.py --law bijstand --profiles 159428317
"""

import sys
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

# ---------------------------------------------------------------------------
# Import shared infrastructure from extraction_zorgtoeslag
# ---------------------------------------------------------------------------
from extraction_zorgtoeslag import (  # noqa: E402
    AVAILABLE_MODELS,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    LawGraphExtractor,
    ProfileGraphExtractor,
    generate_decision_explanation,
    get_git_info,
    load_profiles,
    run_calculation,
    DecisionGraphExtractor as _BaseDecisionGraphExtractor,
    _build_expected_values,  # noqa: F401  (re-exported for extract.py)
    _fix_rounded_amounts,    # noqa: F401
    create_decision_prompt,  # noqa: F401
    DECISION_SYSTEM_PROMPT,  # noqa: F401
)


# ---------------------------------------------------------------------------
# Bijstand-specific law YAML loader
# ---------------------------------------------------------------------------

def load_law_yaml(law_name: str = "bijstand") -> dict:
    """Load the Bijstand Amsterdam YAML (ignores law_name, always loads Amsterdam)."""
    yaml_path = (
        PROJECT_ROOT
        / "laws"
        / "participatiewet"
        / "bijstand"
        / "gemeenten"
        / "GEMEENTE_AMSTERDAM-2023-01-01.yaml"
    )
    if not yaml_path.exists():
        raise FileNotFoundError(f"Bijstand Amsterdam YAML not found at: {yaml_path}")
    with open(yaml_path) as f:
        return yaml.load(f, Loader=Loader)


# ---------------------------------------------------------------------------
# Bijstand-specific DecisionGraphExtractor
# ---------------------------------------------------------------------------

class DecisionGraphExtractor(_BaseDecisionGraphExtractor):
    """
    Decision graph extractor for Bijstand Gemeente Amsterdam.

    Overrides field translations, profile value extraction, decision label
    logic and the explanation skeleton. Everything else (graph construction,
    rule evaluation, formatting helpers) is inherited from the base class.
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

    # ------------------------------------------------------------------ #
    # Profile value extraction — reads from profile sources + calc_result #
    # ------------------------------------------------------------------ #

    def _extract_profile_values(self) -> dict:
        """Extract bijstand-relevant values from the profile and calc_result."""
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

        # --- From RvIG ---
        age = get_nested("RvIG", "personen", "age")
        if age is not None:
            values["LEEFTIJD"] = {"value": age, "source": "RvIG", "unit": "jaar"}

        has_fixed_address = get_nested("RvIG", "personen", "has_fixed_address")
        if has_fixed_address is not None:
            values["HEEFT_VAST_ADRES"] = {"value": has_fixed_address, "source": "RvIG", "unit": ""}

        # --- From calc_result input_data (fields resolved by the MCP service) ---
        # (Defined early so POSTADRES can fall back to verblijfplaats if not in input_data)
        input_data = (self.calc_result or {}).get("input_data", {})

        # POSTADRES: try input_data first, then format from verblijfplaats
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

        # --- From GEMEENTE_AMSTERDAM ---
        arbeid_rows = sources.get("GEMEENTE_AMSTERDAM", {}).get("werk_en_re_integratie", [])
        if isinstance(arbeid_rows, list) and arbeid_rows:
            arbeid = arbeid_rows[0]
            arbeidsvermogen = arbeid.get("arbeidsvermogen")
            if arbeidsvermogen is not None:
                values["ARBEIDSVERMOGEN.arbeidsvermogen"] = {
                    "value": arbeidsvermogen, "source": "GEMEENTE_AMSTERDAM", "unit": "",
                }
            # Store re_integratie_traject even if None so NOT_NULL check returns FAILED (not NOT_APPLICABLE)
            re_traject = arbeid.get("re_integratie_traject")
            values["ARBEIDSVERMOGEN.re_integratie_traject"] = {
                "value": re_traject, "source": "GEMEENTE_AMSTERDAM", "unit": "",
            }

        # --- From BELASTINGDIENST ---
        bedrijfsinkomen = get_nested("BELASTINGDIENST", "bedrijfsinkomen", "bedrag")
        if bedrijfsinkomen is not None:
            values["BEDRIJFSINKOMEN"] = {
                "value": bedrijfsinkomen,
                "source": "BELASTINGDIENST",
                "unit": "eurocent",
            }
            # Derive IS_ONDERNEMER from whether there is any bedrijfsinkomen
            values["IS_ONDERNEMER"] = {
                "value": bedrijfsinkomen > 0,
                "source": "BELASTINGDIENST (afgeleid)",
                "unit": "",
            }

        landelijk_voldoet = input_data.get("$VOLDOET_AAN_LANDELIJKE_VOORWAARDEN")
        if landelijk_voldoet is not None:
            values["VOLDOET_AAN_LANDELIJKE_VOORWAARDEN"] = {
                "value": landelijk_voldoet,
                "source": "SZW",
                "unit": "",
            }

        basisbedrag = input_data.get("$LANDELIJK_BASISBEDRAG")
        if basisbedrag is not None:
            values["LANDELIJK_BASISBEDRAG"] = {
                "value": basisbedrag,
                "source": "SZW",
                "unit": "eurocent",
            }

        kostendelersnorm = input_data.get("$KOSTENDELERSNORM")
        if kostendelersnorm is not None:
            values["KOSTENDELERSNORM"] = {
                "value": kostendelersnorm,
                "source": "SZW",
                "unit": "",
            }

        return values

    # ------------------------------------------------------------------ #
    # Extended condition evaluation (IN, NOT_NULL, dotted paths)         #
    # ------------------------------------------------------------------ #

    def _evaluate_extended_condition(self, cond: dict) -> tuple[bool | None, str]:
        """Evaluate a single condition with support for IN, NOT_NULL, and dotted field paths."""
        subject = cond.get("subject", "")
        operation = cond.get("operation", "")
        value = cond.get("value", "")
        values_ref = cond.get("values")  # plural key used by IN operation

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

        # Standard operations — fall back to base class
        return self._evaluate_condition(subject, operation, value)

    def _add_or_group_node(
        self,
        or_conditions: list,
        group_id: str,
        used_definitions: set,
    ) -> None:
        """Evaluate an OR group and add a single combined RULE node + edges to the graph."""
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

            # Track any definitions used
            ref = values_ref if values_ref else (value if isinstance(value, str) else "")
            if isinstance(ref, str) and ref.startswith("$"):
                val_name = ref[1:]
                if val_name in self.definitions:
                    used_definitions.add(val_name)

            # Build human-readable sub-label
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

        # Determine overall OR status
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

        # Add fact nodes for each subject in the OR group
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
                id=fact_id,
                type="FACT",
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
    # Decision label — uses uitkeringsbedrag instead of hoogte_toeslag    #
    # ------------------------------------------------------------------ #

    def extract(self) -> KnowledgeGraph:
        """Extract a focused decision subgraph for bijstand."""
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
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn},
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
                # Handle nested OR groups (e.g. adres check, arbeidsverplichting check)
                if "or" in cond:
                    self._add_or_group_node(cond["or"], f"rule_or_{i}", used_definitions)
                    continue

                subject = cond.get("subject", "")
                operation = cond.get("operation", "")
                value = cond.get("value", "")

                # Skip any remaining complex nested conditions
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
                self.graph.add_edge(
                    GraphEdge(
                        source=f"person_{self.bsn}",
                        target=f"fact_{subject_name}",
                        relation="HAS_FACT",
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source=f"fact_{subject_name}",
                        target=f"rule_{subject_name}",
                        relation="USED_IN",
                    )
                )

        # Add used definitions as THRESHOLD nodes
        for def_name in used_definitions:
            def_value = self.definitions[def_name]
            def_display = self._format_value(def_value)
            def_node = GraphNode(
                id=f"def_{def_name}",
                type="THRESHOLD",
                label=f"{def_name} = {def_display}",
                properties={"name": def_name, "value": def_value},
            )
            self.graph.add_node(def_node)
            for node in self.graph.nodes:
                if node.type == "RULE" and node.properties.get("expected_value") == def_value:
                    self.graph.add_edge(
                        GraphEdge(
                            source=f"def_{def_name}",
                            target=node.id,
                            relation="DEFINES_THRESHOLD",
                        )
                    )

        # Add CALCULATION node if uitkeringsbedrag > 0
        if requirements_met and uitkering > 0:
            calc_node = GraphNode(
                id="amount_calc",
                type="CALCULATION",
                label=f"Berekend uitkeringsbedrag: {uitkering / 100:,.2f} euro per maand",
                properties={"output_field": "uitkeringsbedrag", "output_amount": uitkering},
            )
            self.graph.add_node(calc_node)
            if "fact_LANDELIJK_BASISBEDRAG" in [n.id for n in self.graph.nodes]:
                self.graph.add_edge(
                    GraphEdge(
                        source="fact_LANDELIJK_BASISBEDRAG",
                        target="amount_calc",
                        relation=self.STATUS_AFFECTS_AMOUNT,
                    )
                )
            self.graph.add_edge(
                GraphEdge(source="amount_calc", target="decision", relation="DETERMINES")
            )

        return self.graph

    # ------------------------------------------------------------------ #
    # Human-readable rule formatting                                       #
    # ------------------------------------------------------------------ #

    def _format_rule_human_readable(self, rule_node: GraphNode) -> str:
        # OR groups have a pre-built combined label
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

        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        # Persoonlijke situatie
        lines.append("## Persoonlijke situatie:")

        if "LEEFTIJD" in self.profile_values:
            age = self.profile_values["LEEFTIJD"]["value"]
            lines.append(f"- Leeftijd: {age} jaar")
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
            lines.append("- Arbeidsvermogen: Onbekend (geen gegevens van gemeente Amsterdam)")

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

        # Calculation detail
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

            # Extra toeslagen
            woonkostentoeslag = output.get("woonkostentoeslag", 0) or 0
            if woonkostentoeslag > 0:
                lines.append(f"- Woonkostentoeslag (briefadres): {self._format_value(woonkostentoeslag, 'eurocent')} per maand")

            startkapitaal = output.get("startkapitaal", 0) or 0
            if startkapitaal > 0:
                lines.append(f"- Startkapitaal ZZP: {self._format_value(startkapitaal, 'eurocent')}")

            lines.append("")

        # Conclusie
        lines.append("## Conclusie:")
        if requirements_met and uitkering > 0:
            lines.append(f"U heeft recht op {law_name}.")
        elif requirements_met:
            lines.append(f"U voldoet aan de voorwaarden voor {law_name}, maar er wordt geen bedrag uitbetaald.")
        else:
            lines.append(f"U heeft geen recht op {law_name}.")

        lines.append("")
        lines.append("## Meer informatie:")
        lines.append("Voor meer informatie over bijstand in Amsterdam kunt u terecht op www.amsterdam.nl")

        return "\n".join(lines)
