#!/usr/bin/env python3
"""
Generic GraphRAG extraction script for machine law — works with any law YAML.

This script extracts a knowledge graph from law YAML definitions and citizen profiles,
creating a structured representation that can be used as context for LLM explanations.
It is law-agnostic: field names, labels, and output fields are all derived from the
YAML definition rather than being hardcoded.

Difference from extraction_zorgtoeslag.py:
- _extract_profile_values() reads from calc_result["input_data"] instead of hardcoded profile paths
- FIELD_TRANSLATIONS is derived from YAML property descriptions
- to_explanation_skeleton() is fully dynamic based on YAML types
- _build_expected_values() uses YAML `type: amount` instead of a hardcoded MONETARY_KEYS set
- SERVICE_URLS maps service name → info URL (no hardcoded toeslagen.nl)

Usage:
    uv run python analysis/llm_explanations/scripts/extraction_law.py --law zorgtoeslag --profiles 174760992
    uv run python analysis/llm_explanations/scripts/extraction_law.py --law werkloosheidswet --profiles 174760992
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
# scripts/extraction_law.py → scripts/ → llm_explanations/ → analysis/ → root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Service → info URL mapping (no hardcoded toeslagen.nl)
# ---------------------------------------------------------------------------
SERVICE_URLS: dict[str, str] = {
    "TOESLAGEN": "www.toeslagen.nl",
    "UWV": "www.uwv.nl",
    "SVB": "www.svb.nl",
    "BELASTINGDIENST": "www.belastingdienst.nl",
    "GEMEENTE_AMSTERDAM": "www.amsterdam.nl",
    "KIESRAAD": "www.kiesraad.nl",
    "SZW": "www.rijksoverheid.nl",
}


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str  # LAW, REQUIREMENT, INPUT, OUTPUT, DEFINITION, PERSON, VALUE, OPERATION
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relation: str  # HAS_REQUIREMENT, DEPENDS_ON, COMPUTED_BY, HAS_VALUE, etc.
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Knowledge graph for law and profile data."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """Add a node if it doesn't exist."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge."""
        self.edges.append(edge)

    def to_triples(self) -> list[tuple[str, str, str]]:
        """Convert graph to list of (subject, predicate, object) triples."""
        triples = []

        # Node properties as triples
        for node in self.nodes:
            triples.append((node.id, "TYPE", node.type))
            triples.append((node.id, "LABEL", node.label))
            for key, value in node.properties.items():
                if value is not None:
                    triples.append((node.id, key.upper(), str(value)))

        # Edges as triples
        for edge in self.edges:
            triples.append((edge.source, edge.relation, edge.target))
            for key, value in edge.properties.items():
                if value is not None:
                    triples.append((f"{edge.source}->{edge.target}", key.upper(), str(value)))

        return triples

    def to_structured_text(self) -> str:
        """Convert graph to structured text for LLM context."""
        lines = []

        # Group nodes by type
        nodes_by_type: dict[str, list[GraphNode]] = {}
        for node in self.nodes:
            nodes_by_type.setdefault(node.type, []).append(node)

        # Law information
        if "LAW" in nodes_by_type:
            lines.append("## Wet informatie")
            for node in nodes_by_type["LAW"]:
                lines.append(f"- Naam: {node.label}")
                if "description" in node.properties:
                    lines.append(f"- Beschrijving: {node.properties['description']}")
                if "legal_basis" in node.properties:
                    lines.append(f"- Wettelijke basis: {node.properties['legal_basis']}")
            lines.append("")

        # Definitions (constants)
        if "DEFINITION" in nodes_by_type:
            lines.append("## Definities en constanten")
            for node in nodes_by_type["DEFINITION"]:
                value = node.properties.get("value", "")
                if isinstance(value, (int, float)) and value > 10000:
                    euro_value = value / 100
                    lines.append(f"- {node.label}: €{euro_value:,.2f}")
                elif isinstance(value, float) and value < 1:
                    lines.append(f"- {node.label}: {value * 100:.3f}%")
                else:
                    lines.append(f"- {node.label}: {value}")
            lines.append("")

        # Input fields
        if "INPUT" in nodes_by_type:
            lines.append("## Invoervelden (data van burger)")
            for node in nodes_by_type["INPUT"]:
                desc = node.properties.get("description", "")
                source = node.properties.get("source_service", "")
                lines.append(f"- {node.label}: {desc}")
                if source:
                    lines.append(f"  (bron: {source})")
            lines.append("")

        # Output fields
        if "OUTPUT" in nodes_by_type:
            lines.append("## Uitvoervelden (resultaat)")
            for node in nodes_by_type["OUTPUT"]:
                desc = node.properties.get("description", "")
                lines.append(f"- {node.label}: {desc}")
            lines.append("")

        # Requirements
        if "REQUIREMENT" in nodes_by_type:
            lines.append("## Vereisten (voorwaarden)")
            for node in nodes_by_type["REQUIREMENT"]:
                lines.append(f"- {node.label}")
                if "explanation" in node.properties:
                    lines.append(f"  Uitleg: {node.properties['explanation']}")
            lines.append("")

        # Person data with extracted values
        if "PERSON" in nodes_by_type:
            lines.append("## Persoonsgegevens")
            for node in nodes_by_type["PERSON"]:
                lines.append(f"- Naam: {node.label}")
                if "description" in node.properties:
                    lines.append(f"- Beschrijving: {node.properties['description']}")
            lines.append("")

        # Actual values from profile (matched to inputs)
        if "VALUE" in nodes_by_type:
            lines.append("## Actuele waarden van deze burger")
            lines.append("(Dit zijn de waarden die worden gebruikt voor de berekening)")
            lines.append("")
            for node in nodes_by_type["VALUE"]:
                field_name = node.properties.get("field", "")
                display_value = node.properties.get("display_value", str(node.properties.get("value", "")))
                source = node.properties.get("source_service", "")
                lines.append(f"- {field_name}: {display_value} (bron: {source})")
            lines.append("")

        # Logical structure with evaluation
        lines.append("## Logische structuur en evaluatie")

        value_nodes = {n.properties.get("field"): n.properties.get("value")
                       for n in nodes_by_type.get("VALUE", [])}
        def_values = {n.label: n.properties.get("value")
                      for n in nodes_by_type.get("DEFINITION", [])}

        req_edges = [e for e in self.edges if e.relation == "HAS_REQUIREMENT"]
        if req_edges:
            lines.append("### Voorwaarden (alle moeten waar zijn):")
            for edge in req_edges:
                req_node = next((n for n in self.nodes if n.id == edge.target), None)
                if req_node:
                    subject = req_node.properties.get("subject", "").replace("$", "")
                    operation = req_node.properties.get("operation", "")
                    value_ref = req_node.properties.get("value", "")

                    actual_value = value_nodes.get(subject)
                    if isinstance(value_ref, str) and value_ref.startswith("$"):
                        compare_value = def_values.get(value_ref[1:], value_ref)
                    else:
                        compare_value = value_ref

                    if actual_value is not None:
                        if isinstance(actual_value, bool):
                            actual_display = "Ja" if actual_value else "Nee"
                        else:
                            actual_display = str(actual_value)

                        if isinstance(compare_value, bool):
                            compare_display = "Ja" if compare_value else "Nee"
                        else:
                            compare_display = str(compare_value)

                        result = None
                        if operation == "GREATER_OR_EQUAL" and isinstance(actual_value, (int, float)) and isinstance(compare_value, (int, float)):
                            result = actual_value >= compare_value
                        elif operation == "EQUALS":
                            result = actual_value == compare_value

                        result_str = ""
                        if result is not None:
                            result_str = " => VOLDOET" if result else " => VOLDOET NIET"

                        lines.append(f"  - {req_node.label}")
                        lines.append(f"    Actuele waarde: {subject} = {actual_display}")
                        lines.append(f"    Vereiste: {compare_display}{result_str}")
                    else:
                        lines.append(f"  - {req_node.label}")
            lines.append("")

        comp_edges = [e for e in self.edges if e.relation == "COMPUTED_BY"]
        if comp_edges:
            lines.append("### Berekeningen:")
            for edge in comp_edges:
                output_node = next((n for n in self.nodes if n.id == edge.source), None)
                op_node = next((n for n in self.nodes if n.id == edge.target), None)
                if output_node and op_node:
                    lines.append(f"  - {output_node.label} wordt berekend via: {op_node.label}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Convert graph to JSON-serializable dict."""

        def serialize_value(v: Any) -> Any:
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return v

        def serialize_props(props: dict) -> dict:
            return {k: serialize_value(v) for k, v in props.items()}

        return {
            "nodes": [
                {"id": n.id, "type": n.type, "label": n.label, "properties": serialize_props(n.properties)}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "properties": serialize_props(e.properties)}
                for e in self.edges
            ]
        }

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX directed graph."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib are required for visualization. Install with: uv add networkx matplotlib")

        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id, label=node.label, node_type=node.type, **node.properties)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, relation=edge.relation, **edge.properties)
        return G

    def visualize(self, output_path: str, title: str = "Knowledge Graph", figsize: tuple = (20, 16)) -> None:
        """Generate a visual representation of the graph and save to file."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib are required for visualization. Install with: uv add networkx matplotlib")

        G = self.to_networkx()

        node_colors = {
            "LAW": "#4A90D9",
            "REQUIREMENT": "#E74C3C",
            "INPUT": "#27AE60",
            "OUTPUT": "#F39C12",
            "DEFINITION": "#9B59B6",
            "PERSON": "#3498DB",
            "VALUE": "#1ABC9C",
            "OPERATION": "#95A5A6",
            "DECISION": "#2ECC71",
            "RULE": "#E74C3C",
            "FACT": "#3498DB",
            "THRESHOLD": "#9B59B6",
            "CALCULATION": "#F39C12",
        }

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        colors = [node_colors.get(G.nodes[n].get("node_type", ""), "#CCCCCC") for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=2000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666666", arrows=True, arrowsize=20, alpha=0.6, connectionstyle="arc3,rad=0.1")

        labels = {}
        for n in G.nodes():
            label = G.nodes[n].get("label", n)
            if len(label) > 25:
                label = label[:22] + "..."
            labels[n] = label

        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8, font_weight="bold")
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, font_color="#444444")

        legend_elements = []
        for node_type, color in node_colors.items():
            if any(G.nodes[n].get("node_type") == node_type for n in G.nodes()):
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, label=node_type))

        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Graph visualization saved to: {output_path}", file=sys.stderr)


class LawGraphExtractor:
    """Extracts a knowledge graph from a law YAML definition."""

    def __init__(self, law_yaml: dict):
        self.law = law_yaml
        self.graph = KnowledgeGraph()
        self.node_counter = 0

    def _gen_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def extract(self) -> KnowledgeGraph:
        self._extract_law_metadata()
        self._extract_definitions()
        self._extract_inputs()
        self._extract_outputs()
        self._extract_requirements()
        self._extract_actions()
        return self.graph

    def _extract_law_metadata(self) -> None:
        law_id = f"law_{self.law.get('law', 'unknown')}"
        legal_basis = self.law.get("legal_basis", {})
        legal_basis_str = ""
        if legal_basis:
            legal_basis_str = f"{legal_basis.get('law', '')} artikel {legal_basis.get('article', '')}"

        node = GraphNode(
            id=law_id,
            type="LAW",
            label=self.law.get("name", "Onbekende wet"),
            properties={
                "description": self.law.get("description", "").strip(),
                "service": self.law.get("service", ""),
                "valid_from": self.law.get("valid_from", ""),
                "legal_basis": legal_basis_str,
                "law_type": self.law.get("law_type", ""),
                "decision_type": self.law.get("decision_type", ""),
            }
        )
        self.graph.add_node(node)
        self.law_node_id = law_id

    def _extract_definitions(self) -> None:
        definitions = self.law.get("properties", {}).get("definitions", {})
        for name, value in definitions.items():
            node = GraphNode(
                id=f"def_{name}", type="DEFINITION", label=name,
                properties={"value": value, "value_type": type(value).__name__}
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(source=self.law_node_id, target=node.id, relation="HAS_DEFINITION"))

    def _extract_inputs(self) -> None:
        for inp in self.law.get("properties", {}).get("input", []):
            name = inp.get("name", "")
            service_ref = inp.get("service_reference", {})
            legal_basis = inp.get("legal_basis", {})
            node = GraphNode(
                id=f"input_{name}", type="INPUT", label=name,
                properties={
                    "description": inp.get("description", ""),
                    "data_type": inp.get("type", ""),
                    "source_service": service_ref.get("service", ""),
                    "source_field": service_ref.get("field", ""),
                    "source_law": service_ref.get("law", ""),
                    "legal_explanation": legal_basis.get("explanation", ""),
                }
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(source=self.law_node_id, target=node.id, relation="HAS_INPUT"))

    def _extract_outputs(self) -> None:
        for out in self.law.get("properties", {}).get("output", []):
            name = out.get("name", "")
            legal_basis = out.get("legal_basis", {})
            node = GraphNode(
                id=f"output_{name}", type="OUTPUT", label=name,
                properties={
                    "description": out.get("description", ""),
                    "data_type": out.get("type", ""),
                    "citizen_relevance": out.get("citizen_relevance", ""),
                    "legal_explanation": legal_basis.get("explanation", ""),
                }
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(source=self.law_node_id, target=node.id, relation="HAS_OUTPUT"))

    def _extract_requirements(self) -> None:
        for req in self.law.get("requirements", []):
            if "all" in req:
                for cond in req["all"]:
                    self._extract_condition(cond, "AND")
            elif "any" in req:
                for cond in req["any"]:
                    self._extract_condition(cond, "OR")
            else:
                self._extract_condition(req, "SINGLE")

    def _extract_condition(self, condition: dict, logic_type: str) -> str:
        subject = condition.get("subject", "")
        operation = condition.get("operation", "")
        value = condition.get("value", "")

        op_labels = {
            "GREATER_OR_EQUAL": ">=", "GREATER_THAN": ">",
            "LESS_OR_EQUAL": "<=", "LESS_THAN": "<",
            "EQUALS": "==", "NOT_EQUALS": "!=",
        }
        op_label = op_labels.get(operation, operation)
        subject_clean = subject.replace("$", "")
        value_clean = str(value).replace("$", "") if isinstance(value, str) else str(value)
        label = f"{subject_clean} {op_label} {value_clean}"

        node_id = self._gen_id("req")
        node = GraphNode(
            id=node_id, type="REQUIREMENT", label=label,
            properties={"subject": subject, "operation": operation, "value": value, "logic_type": logic_type}
        )
        self.graph.add_node(node)
        self.graph.add_edge(GraphEdge(source=self.law_node_id, target=node_id, relation="HAS_REQUIREMENT"))

        if subject.startswith("$"):
            input_id = f"input_{subject[1:]}"
            if any(n.id == input_id for n in self.graph.nodes):
                self.graph.add_edge(GraphEdge(source=node_id, target=input_id, relation="DEPENDS_ON"))

        return node_id

    def _extract_actions(self) -> None:
        for action in self.law.get("actions", []):
            output_name = action.get("output", "")
            output_id = f"output_{output_name}"

            if "value" in action and "operation" not in action:
                node_id = self._gen_id("action")
                node = GraphNode(
                    id=node_id, type="OPERATION",
                    label=f"Stel {output_name} = {action['value']}",
                    properties={"operation_type": "ASSIGN", "value": action["value"]}
                )
                self.graph.add_node(node)
                self.graph.add_edge(GraphEdge(source=output_id, target=node_id, relation="COMPUTED_BY"))
            elif "operation" in action:
                self._extract_operation(action, output_id)

    def _extract_operation(self, action: dict, output_id: str, depth: int = 0) -> str:
        operation = action.get("operation", "")
        if operation == "IF":
            return self._extract_if_operation(action, output_id, depth)
        else:
            node_id = self._gen_id("op")
            legal_basis = action.get("legal_basis", {})
            node = GraphNode(
                id=node_id, type="OPERATION",
                label=f"{operation} berekening",
                properties={"operation_type": operation, "legal_explanation": legal_basis.get("explanation", "")}
            )
            self.graph.add_node(node)
            if depth == 0:
                self.graph.add_edge(GraphEdge(source=output_id, target=node_id, relation="COMPUTED_BY"))
            return node_id

    def _extract_if_operation(self, action: dict, output_id: str, depth: int) -> str:
        conditions = action.get("conditions", [])
        node_id = self._gen_id("if")
        condition_descriptions = []
        for cond in conditions:
            if "test" in cond:
                test = cond["test"]
                op_labels = {"GREATER_THAN": ">", "GREATER_OR_EQUAL": ">=", "LESS_THAN": "<", "LESS_OR_EQUAL": "<=", "EQUALS": "=="}
                op_label = op_labels.get(test.get("operation", ""), test.get("operation", ""))
                subject_clean = str(test.get("subject", "")).replace("$", "")
                value_clean = str(test.get("value", "")).replace("$", "")
                condition_descriptions.append(f"ALS {subject_clean} {op_label} {value_clean}")
            elif "else" in cond:
                condition_descriptions.append("ANDERS")

        node = GraphNode(
            id=node_id, type="OPERATION",
            label="Conditionele berekening",
            properties={"operation_type": "IF", "conditions": "; ".join(condition_descriptions)}
        )
        self.graph.add_node(node)
        if depth == 0:
            self.graph.add_edge(GraphEdge(source=output_id, target=node_id, relation="COMPUTED_BY"))
        return node_id


class ProfileGraphExtractor:
    """Extracts graph data from a citizen profile (zorgtoeslag-specific field paths)."""

    def __init__(self, profile: dict, bsn: str):
        self.profile = profile
        self.bsn = bsn

    def _get_nested_value(self, sources: dict, service: str, table: str, field_name: str) -> Any:
        try:
            service_data = sources.get(service, {})
            table_data = service_data.get(table, [])
            if isinstance(table_data, list) and len(table_data) > 0:
                return table_data[0].get(field_name)
            elif isinstance(table_data, dict):
                return table_data.get(field_name)
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def extract(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Add profile data to the knowledge graph (uses zorgtoeslag-specific field paths)."""
        person_id = f"person_{self.bsn}"
        sources = self.profile.get("sources", {})
        extracted_values = {}

        age = self._get_nested_value(sources, "RvIG", "personen", "age")
        if age is not None:
            extracted_values["LEEFTIJD"] = {"value": age, "source": "RvIG", "unit": "jaar"}

        has_partner = self._get_nested_value(sources, "RvIG", "personen", "has_partner")
        if has_partner is not None:
            extracted_values["HEEFT_PARTNER"] = {"value": has_partner, "source": "RvIG", "unit": ""}

        polis_status = self._get_nested_value(sources, "RVZ", "verzekeringen", "polis_status")
        if polis_status is not None:
            extracted_values["IS_VERZEKERDE"] = {"value": polis_status == "ACTIEF", "source": "RVZ", "unit": ""}

        inkomen = self._get_nested_value(sources, "UWV", "uwv_toetsingsinkomen", "toetsingsinkomen")
        if inkomen is not None:
            extracted_values["INKOMEN"] = {"value": inkomen, "source": "UWV", "unit": "eurocent"}

        vermogen = self._get_nested_value(sources, "BELASTINGDIENST", "belastingdienst_vermogen", "vermogen")
        if vermogen is not None:
            extracted_values["VERMOGEN"] = {"value": vermogen, "source": "BELASTINGDIENST", "unit": "eurocent"}

        partner_inkomen = self._get_nested_value(sources, "UWV", "uwv_toetsingsinkomen_partner", "toetsingsinkomen")
        if partner_inkomen is not None:
            extracted_values["PARTNER_INKOMEN"] = {"value": partner_inkomen, "source": "UWV", "unit": "eurocent"}
        elif has_partner is False:
            extracted_values["PARTNER_INKOMEN"] = {"value": 0, "source": "N/A (geen partner)", "unit": "eurocent"}

        node = GraphNode(
            id=person_id, type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn, "description": self.profile.get("description", ""), "extracted_values": extracted_values}
        )
        graph.add_node(node)

        for field_name, field_data in extracted_values.items():
            value = field_data["value"]
            source = field_data["source"]
            unit = field_data["unit"]
            value_id = f"value_{self.bsn}_{field_name}"

            if unit == "eurocent" and isinstance(value, (int, float)):
                display_value = f"€{value / 100:,.2f}"
            elif isinstance(value, bool):
                display_value = "Ja" if value else "Nee"
            else:
                display_value = str(value)

            value_node = GraphNode(
                id=value_id, type="VALUE",
                label=f"{field_name} = {display_value}",
                properties={"field": field_name, "value": value, "display_value": display_value, "source_service": source, "unit": unit}
            )
            graph.add_node(value_node)
            graph.add_edge(GraphEdge(source=person_id, target=value_id, relation="HAS_VALUE"))

            input_id = f"input_{field_name}"
            if any(n.id == input_id for n in graph.nodes):
                graph.add_edge(GraphEdge(source=value_id, target=input_id, relation="PROVIDES_VALUE_FOR"))

        return graph


class DecisionGraphExtractor:
    """
    Extracts a focused decision subgraph for GraphRAG — law-agnostic version.

    Reads field names, labels, and monetary output fields from the law YAML
    instead of using hardcoded zorgtoeslag values.
    """

    STATUS_SATISFIED = "SATISFIED"
    STATUS_FAILED = "FAILED"
    STATUS_AFFECTS_AMOUNT = "AFFECTS_AMOUNT"
    STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

    # Human-readable operation translations (generic, not law-specific)
    OPERATION_TRANSLATIONS = {
        "GREATER_OR_EQUAL": "is minimaal",
        "GREATER_THAN": "is meer dan",
        "LESS_OR_EQUAL": "is maximaal",
        "LESS_THAN": "is minder dan",
        "EQUALS": "is",
        "NOT_EQUALS": "is niet",
    }

    def __init__(self, law_yaml: dict, profile: dict, bsn: str, calc_result: dict | None = None):
        self.law = law_yaml
        self.profile = profile
        self.bsn = bsn
        self.calc_result = calc_result
        self.graph = KnowledgeGraph()

        # Build lookup dicts from YAML properties once
        self._input_props: dict[str, dict] = {
            p["name"]: p for p in self.law.get("properties", {}).get("input", []) if "name" in p
        }
        self._output_props: dict[str, dict] = {
            p["name"]: p for p in self.law.get("properties", {}).get("output", []) if "name" in p
        }

        self.profile_values = self._extract_profile_values()
        self.definitions = self.law.get("properties", {}).get("definitions", {})

    @property
    def field_translations(self) -> dict[str, str]:
        """Generate human-readable field labels from YAML property descriptions."""
        return {name: prop.get("description", name) for name, prop in self._input_props.items()}

    def _amount_output_fields(self) -> list[str]:
        """Return output field names that have type: amount."""
        return [name for name, prop in self._output_props.items() if prop.get("type") == "amount"]

    def _amount_input_fields(self) -> list[str]:
        """Return input field names that have type: amount."""
        return [name for name, prop in self._input_props.items() if prop.get("type") == "amount"]

    def _extract_profile_values(self) -> dict:
        """Extract all relevant input values from calc_result["input_data"] using YAML field types."""
        input_data = (self.calc_result or {}).get("input_data", {})
        values = {}
        for name, prop in self._input_props.items():
            value = input_data.get(f"${name}")
            if value is None:
                continue
            unit = "eurocent" if prop.get("type") == "amount" else ""
            values[name] = {
                "value": value,
                "source": "calc_result",
                "unit": unit,
                "description": prop.get("description", name),
            }
        return values

    def _get_value(self, ref: str, return_none_if_missing: bool = False) -> Any:
        """Get a value from profile or definitions, resolving $ references."""
        if isinstance(ref, str) and ref.startswith("$"):
            name = ref[1:]
            if name in self.profile_values:
                return self.profile_values[name]["value"]
            if name in self.definitions:
                return self.definitions[name]
            return None
        return ref

    def _format_value(self, value: Any, unit: str = "") -> str:
        """Format a value for display."""
        if value is None:
            return "Onbekend"
        if unit == "eurocent" and isinstance(value, (int, float)):
            return f"{value / 100:,.2f} euro"
        elif isinstance(value, bool):
            return "Ja" if value else "Nee"
        elif isinstance(value, float):
            if value < 1:
                return f"{value * 100:.3f}%"
            return f"{value:,.2f}"
        elif isinstance(value, int) and value > 10000:
            return f"{value / 100:,.2f} euro"
        return str(value)

    def _evaluate_condition(self, subject: str, operation: str, value: Any) -> tuple[bool | None, str]:
        """Evaluate a single condition and return (result, status)."""
        actual = self._get_value(subject)
        expected = self._get_value(value)

        if actual is None:
            return None, self.STATUS_NOT_APPLICABLE

        op_map = {
            "GREATER_OR_EQUAL": lambda a, b: a >= b,
            "GREATER_THAN": lambda a, b: a > b,
            "LESS_OR_EQUAL": lambda a, b: a <= b,
            "LESS_THAN": lambda a, b: a < b,
            "EQUALS": lambda a, b: a == b,
            "NOT_EQUALS": lambda a, b: a != b,
        }

        if operation in op_map:
            try:
                result = op_map[operation](actual, expected)
                return result, self.STATUS_SATISFIED if result else self.STATUS_FAILED
            except (TypeError, ValueError):
                return None, self.STATUS_NOT_APPLICABLE

        return None, self.STATUS_NOT_APPLICABLE

    def extract(self) -> KnowledgeGraph:
        """Extract a focused decision subgraph."""
        requirements_met = self.calc_result.get("requirements_met", False) if self.calc_result else None
        output = self.calc_result.get("result", {}) if self.calc_result else {}

        # Build decision label from first amount output field (generic)
        amount_fields = self._amount_output_fields()
        primary_amount_field = next((f for f in amount_fields if output.get(f)), None)
        if requirements_met and primary_amount_field:
            amount = output[primary_amount_field] / 100
            if amount > 0:
                desc = self._output_props[primary_amount_field].get("description", primary_amount_field)
                decision_label = f"RECHT: {amount:,.2f} euro ({desc})"
            else:
                decision_label = "RECHT (bedrag €0)"
        elif requirements_met:
            decision_label = "RECHT"
        else:
            decision_label = "GEEN RECHT"

        decision_node = GraphNode(
            id="decision", type="DECISION", label=decision_label,
            properties={"requirements_met": requirements_met, "law": self.law.get("name", ""), "output": output}
        )
        self.graph.add_node(decision_node)

        person_node = GraphNode(
            id=f"person_{self.bsn}", type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn}
        )
        self.graph.add_node(person_node)
        self.graph.add_edge(GraphEdge(source=f"person_{self.bsn}", target="decision", relation="KRIJGT_BESLISSING"))

        # Process requirements
        requirements = self.law.get("requirements", [])
        used_definitions = set()

        for req in requirements:
            conditions = req.get("all") or req.get("any") or [req]
            for cond in conditions:
                subject = cond.get("subject", "")
                operation = cond.get("operation", "")
                value = cond.get("value", "")

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
                    }
                )
                self.graph.add_node(rule_node)
                self.graph.add_edge(GraphEdge(
                    source=f"rule_{subject_name}", target="decision",
                    relation=status, properties={"evaluated": True}
                ))

                fact_node = GraphNode(
                    id=f"fact_{subject_name}", type="FACT",
                    label=f"{subject_name} = {actual_display}",
                    properties={
                        "field": subject_name, "value": actual_value,
                        "source": self.profile_values.get(subject_name, {}).get("source", "Niet beschikbaar"),
                    }
                )
                self.graph.add_node(fact_node)
                self.graph.add_edge(GraphEdge(source=f"person_{self.bsn}", target=f"fact_{subject_name}", relation="HAS_FACT"))
                self.graph.add_edge(GraphEdge(source=f"fact_{subject_name}", target=f"rule_{subject_name}", relation="USED_IN"))

        # Add used definitions
        for def_name in used_definitions:
            def_value = self.definitions[def_name]
            def_display = self._format_value(def_value)
            def_node = GraphNode(
                id=f"def_{def_name}", type="THRESHOLD",
                label=f"{def_name} = {def_display}",
                properties={"name": def_name, "value": def_value}
            )
            self.graph.add_node(def_node)
            for node in self.graph.nodes:
                if node.type == "RULE" and node.properties.get("expected_value") == def_value:
                    self.graph.add_edge(GraphEdge(source=f"def_{def_name}", target=node.id, relation="DEFINES_THRESHOLD"))

        # Add CALCULATION node for the first amount output (generic)
        if requirements_met and primary_amount_field:
            amount_val = output[primary_amount_field]
            desc = self._output_props[primary_amount_field].get("description", primary_amount_field)
            amount_node = GraphNode(
                id="amount_calc", type="CALCULATION",
                label=f"Berekend bedrag: {amount_val / 100:,.2f} euro ({desc})",
                properties={"output_field": primary_amount_field, "output_amount": amount_val}
            )
            self.graph.add_node(amount_node)

            # Connect the first amount input to calculation (if any)
            first_amount_input = next(
                (name for name in self._amount_input_fields() if f"fact_{name}" in [n.id for n in self.graph.nodes]),
                None
            )
            if first_amount_input:
                self.graph.add_edge(GraphEdge(source=f"fact_{first_amount_input}", target="amount_calc", relation=self.STATUS_AFFECTS_AMOUNT))

            self.graph.add_edge(GraphEdge(source="amount_calc", target="decision", relation="DETERMINES"))

        return self.graph

    def _format_rule_human_readable(self, rule_node: GraphNode) -> str:
        """Format a rule in human-readable Dutch instead of technical notation."""
        subject = rule_node.properties.get("subject", "")
        operation = rule_node.properties.get("operation", "")
        actual_value = rule_node.properties.get("actual_value")
        expected_value = rule_node.properties.get("expected_value")

        field_name = self.field_translations.get(subject, subject)
        op_text = self.OPERATION_TRANSLATIONS.get(operation, operation)
        unit = self.profile_values.get(subject, {}).get("unit", "")
        actual_display = self._format_value(actual_value, unit)
        expected_display = self._format_value(expected_value, unit if unit else "")

        # Generic boolean handling: use field description from YAML
        if isinstance(actual_value, bool):
            if actual_value:
                return f"Ja: {field_name}"
            else:
                return f"Nee: {field_name} (niet van toepassing)"

        # Age / numeric: natural language
        if subject == "LEEFTIJD":
            return f"U bent {actual_display} jaar oud (vereist: {op_text} {expected_display} jaar)"

        return f"{field_name}: {actual_display} ({op_text} {expected_display})"

    def to_explanation_skeleton(self) -> str:
        """
        Convert the decision graph to a structured explanation skeleton.
        Fully generic: derives all sections from YAML types and calc_result.
        """
        lines = []

        decision_node = next((n for n in self.graph.nodes if n.type == "DECISION"), None)
        if not decision_node:
            return "Geen beslissing gevonden."

        requirements_met = decision_node.properties.get("requirements_met", False)
        law_name = decision_node.properties.get("law", "")
        output = decision_node.properties.get("output", {})
        amount_output_fields = self._amount_output_fields()

        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        # Persoonlijke situatie: show ALL profile values (not a hardcoded subset)
        if self.profile_values:
            lines.append("## Persoonlijke situatie:")
            for key, info in self.profile_values.items():
                label = self.field_translations.get(key, key)
                value = info["value"]
                unit = info.get("unit", "")
                lines.append(f"- {label}: {self._format_value(value, unit)}")
            lines.append("")

        # Group rules by status
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
                subject = rule.properties.get("subject", "")
                field_label = self.field_translations.get(subject, subject)
                lines.append(f"- {field_label}: gegevens ontbreken")
            lines.append("")

        if failed_rules:
            lines.append("## Voorwaarden waar u NIET aan voldoet:")
            for rule in failed_rules:
                lines.append(f"- ✗ {self._format_rule_human_readable(rule)}")
            lines.append("")

        if satisfied_rules:
            lines.append("## Voorwaarden waar u WEL aan voldoet:")
            for rule in satisfied_rules:
                lines.append(f"- ✓ {self._format_rule_human_readable(rule)}")
            lines.append("")

        # Calculation: show all amount inputs + amount outputs
        calc_node = next((n for n in self.graph.nodes if n.type == "CALCULATION"), None)
        if calc_node and requirements_met and amount_output_fields:
            lines.append("## Berekening:")

            for key, info in self.profile_values.items():
                if self._input_props.get(key, {}).get("type") == "amount":
                    label = self.field_translations.get(key, key)
                    lines.append(f"- {label}: {self._format_value(info['value'], 'eurocent')}")

            for field_name in amount_output_fields:
                if field_name in output:
                    amount = output[field_name]
                    desc = self._output_props[field_name].get("description", field_name)
                    lines.append(f"- {desc}: {self._format_value(amount, 'eurocent')}")
                    # Add per-month breakdown if this looks like a yearly amount
                    if "maand" not in field_name.lower() and "dag" not in field_name.lower():
                        monthly = amount / 12
                        lines.append(f"- Per maand: {self._format_value(monthly, 'eurocent')}")

            lines.append("")

        # Conclusie: generic
        primary_amount = next((output.get(f, 0) for f in amount_output_fields if output.get(f)), 0) or 0
        lines.append("## Conclusie:")
        if requirements_met and primary_amount > 0:
            lines.append(f"U heeft recht op {law_name}.")
        elif requirements_met:
            lines.append(f"U voldoet aan de basisvoorwaarden voor {law_name}.")
        else:
            lines.append(f"U heeft geen recht op {law_name}.")

        # Meer informatie: derive URL from law service
        service = self.law.get("service", "")
        url = SERVICE_URLS.get(service)
        if url:
            lines.append("")
            lines.append("## Meer informatie:")
            lines.append(f"Voor meer informatie kunt u terecht op {url}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def run_calculation(law_name: str, bsn: str) -> dict | None:
    """Run the actual law calculation using the MCP service."""
    try:
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        services = get_machine_service()
        case_manager = get_case_manager()
        claim_manager = get_claim_manager()
        mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

        service = mcp_connector.registry.get_service(law_name)
        if not service:
            return None

        calc_result = service.execute(bsn, {})
        if "error" in calc_result:
            return None

        return calc_result

    except Exception as e:
        print(f"Warning: Could not run calculation for {law_name}/{bsn}: {e}", file=sys.stderr)
        return None


def load_law_yaml(law_name: str) -> dict:
    """Load a law YAML file by name."""
    law_paths = [
        PROJECT_ROOT / "laws" / law_name,
        PROJECT_ROOT / "laws" / f"{law_name}wet",
    ]
    for law_path in law_paths:
        if law_path.exists():
            yaml_files = list(law_path.glob("*.yaml"))
            if yaml_files:
                yaml_files.sort(reverse=True)
                with open(yaml_files[0]) as f:
                    return yaml.load(f, Loader=Loader)
    raise FileNotFoundError(f"Law not found: {law_name}")


# ============================================================================
# LLM Integration
# ============================================================================

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
        "description": "Llama 3.3 70B via local Ollama - best quality (~38GB RAM)",
    },
    "mistral": {
        "id": "mistral:7b",
        "provider": "ollama",
        "description": "Mistral 7B via local Ollama (~4GB RAM)",
    },
    "deepseek": {
        "id": "deepseek-r1:8b",
        "provider": "ollama",
        "description": "DeepSeek R1 8B via local Ollama (~5GB RAM)",
    },
    "gemma2": {
        "id": "gemma2:9b",
        "provider": "ollama",
        "description": "Gemma 2 9B via local Ollama (~6GB RAM)",
    },
}


def create_decision_prompt(skeleton: str, person_name: str) -> str:
    """Create a constrained LLM prompt using the decision graph skeleton."""
    prompt = f"""# Informatie over de beslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau).

WAT JE MOET DOEN:
- Begin met de conclusie (wel of geen recht)
- Leg uit waarom, gebruik de voorwaarden uit het skeleton
- Noem het bedrag als dat er staat
- Eindig met de 'Meer informatie' zin uit het skeleton (als aanwezig)
- Praat altijd in de u-vorm tegen {person_name}
- Schrijf bedragen altijd in de vorm van "1.500 euro per jaar" of "125 euro per maand"

WAT JE NIET MAG DOEN:
- GEEN briefopmaak gebruiken (geen "Geachte heer/mevrouw", geen "Met vriendelijke groet")
- GEEN verwijzingen naar instanties die niet in het skeleton staan
- GEEN extra voorwaarden of regels verzinnen die niet in het skeleton staan
- GEEN vragen stellen aan de lezer
- GEEN aanbod om te helpen
- GEEN derde persoon aanspreken - schrijf direct tegen {person_name}
- GEEN bedragen zonder details te noemen, per jaar en per maand als dat in het skeleton staat

VOORBEELD GOEDE UITLEG:
"U heeft recht op zorgtoeslag.
U voldoet aan alle voorwaarden: u bent 25 jaar (de minimumleeftijd is 18 jaar) en u bent verzekerd voor ziektekosten.
Aangezien uw inkomen 10.000 euro per jaar bedraagt, ontvangt u 1.500 euro zorgtoeslag per jaar. Dit komt neer op 125 euro per maand.

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"Geachte heer Jansen, Hierbij informeren wij u... Met vriendelijke groet" <- FOUT: briefopmaak
"Neem contact op met de gemeente" <- FOUT: instantie niet in skeleton
"Uw vermogen is laag genoeg" <- FOUT: als vermogen niet in skeleton staat"""

    return prompt


def _to_dutch_format(value: float) -> str:
    """Format a float as Dutch number notation: 17068.17 -> 17.068,17"""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")


def _fix_rounded_amounts(text: str, expected_values: dict[str, float]) -> str:
    """Replace rounded amounts in LLM output with exact values from calculation."""
    result = text
    for exact_value in expected_values.values():
        if exact_value < 1:
            continue
        exact_dutch = _to_dutch_format(exact_value)
        if exact_dutch in result:
            continue
        exact_int = int(exact_value)
        cents = round((exact_value - exact_int) * 100)
        cents_str = f"{cents:02d}"
        dutch_thousands = f"{exact_int:,}".replace(",", ".")
        american_thousands = f"{exact_int:,}"
        plain = str(exact_int)

        patterns = []
        if cents > 0:
            patterns += [
                (r"(?<!\d)" + re.escape(plain) + r"," + re.escape(cents_str) + r"(?!\d)", exact_dutch),
                (r"(?<!\d)" + re.escape(plain) + r"\." + re.escape(cents_str) + r"(?!\d)", exact_dutch),
            ]
        patterns += [
            (re.escape(dutch_thousands) + r",00\b", exact_dutch),
            (re.escape(american_thousands) + r"\.00\b", exact_dutch),
            (r"(?<![,.\d])" + re.escape(dutch_thousands) + r"(?![,.\d])", exact_dutch),
            (r"(?<![,.\d])" + re.escape(american_thousands) + r"(?![,.\d])", exact_dutch),
            (r"(?<!\d)" + re.escape(plain) + r"(?![,.\d])", exact_dutch),
        ]

        for pattern, replacement in patterns:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                result = new_result
                break
    return result


def _build_expected_values(decision_extractor: DecisionGraphExtractor) -> dict[str, float]:
    """Extract exact euro values that should appear in the explanation.

    Uses YAML `type: amount` to detect monetary fields — no hardcoded MONETARY_KEYS.
    """
    expected: dict[str, float] = {}
    calc_output = (decision_extractor.calc_result or {}).get("result", {})

    # Amount outputs from calc result
    for name, prop in decision_extractor._output_props.items():
        if prop.get("type") == "amount" and calc_output.get(name):
            expected[name] = calc_output[name] / 100

    # Amount inputs from profile values
    for key, info in decision_extractor.profile_values.items():
        if decision_extractor._input_props.get(key, {}).get("type") == "amount":
            val = info.get("value")
            if isinstance(val, (int, float)) and val > 0:
                unit = info.get("unit", "")
                expected[key] = val / 100 if (unit == "eurocent" or val > 100000) else float(val)

    return expected


DECISION_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse burgers uitleg geeft over overheidsbeslissingen.

Je taak is om een beslissingsskeleton om te zetten naar een korte, begrijpelijke uitleg.

VERPLICHT:
- Gebruik ALLEEN informatie uit het skeleton - voeg NIETS toe
- Schrijf in eenvoudig Nederlands (B1-niveau) - korte zinnen, gewone woorden
- Dit is een informatieve tekst, GEEN brief of gesprek
- Eindig altijd met de 'Meer informatie' sectie uit het skeleton (als aanwezig)

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN verwijzingen naar instanties die niet in het skeleton staan
- GEEN aanbiedingen voor hulp of vragen aan de lezer
- GEEN technische termen of codes behouden - alles moet in normale taal"""


def generate_decision_explanation(
    decision_extractor: DecisionGraphExtractor,
    person_name: str,
    api_key: str | None = None,
    model: str = "haiku",
) -> dict:
    """Generate an LLM explanation using the constrained decision skeleton."""
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
            }
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
            }
        }


def load_profiles(profiles_path: str = "data/profiles.yaml") -> dict:
    """Load profiles from YAML."""
    with open(PROJECT_ROOT / profiles_path) as f:
        data = yaml.load(f, Loader=Loader)
    return data.get("profiles", {})


def generate_output_filename(model: str, law: str, profiles: list[str] | None) -> str:
    """Generate descriptive output filename based on run metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not profiles:
        profile_part = "all-profiles"
    elif len(profiles) == 1:
        profile_part = profiles[0]
    else:
        profile_part = f"{len(profiles)}profiles"
    filename = f"{timestamp}_{model}_{law}_{profile_part}_graph.jsonl"
    return str(PROJECT_ROOT / "analysis" / "llm_explanations" / "output" / filename)


def generate_graph_filename(law: str, profile: str, extension: str = "png") -> str:
    """Generate descriptive filename for graph visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{law}_{profile}_graph.{extension}"
    return str(PROJECT_ROOT / "analysis" / "llm_explanations" / "output" / "graphs" / filename)


def get_git_info() -> dict:
    """Get git commit information for reproducibility."""
    import subprocess

    info = {}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def _first_amount_output(law_yaml: dict, calc_output: dict) -> tuple[str | None, float]:
    """Return (field_name, value_in_euros) for the first non-zero amount output, or (None, 0)."""
    for prop in law_yaml.get("properties", {}).get("output", []):
        if prop.get("type") == "amount":
            name = prop["name"]
            if calc_output.get(name):
                return name, calc_output[name] / 100
    return None, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic GraphRAG extraction for any law YAML")
    parser.add_argument("--law", default="zorgtoeslag", help="Law name (e.g., zorgtoeslag, werkloosheidswet)")
    parser.add_argument("--profiles", nargs="*", help="BSN(s) of profile(s) to include (omit for all profiles)")
    parser.add_argument("--output", help="Output file (default: auto-generated)")
    parser.add_argument("--format", choices=["text", "json", "triples"], default="text")
    parser.add_argument("--llm", action="store_true", help="Generate LLM explanation using graph context")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var). Not required for Ollama models.")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="haiku")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Save graph visualization to output/graphs/")
    parser.add_argument("--viz-format", choices=["png", "pdf", "svg", "jpg"], default="png")
    parser.add_argument("--decision-graph", action="store_true", help="Use focused decision subgraph instead of full law graph")
    parser.add_argument("--all-profiles", action="store_true", help="Process all available profiles")

    args = parser.parse_args()

    all_profiles = load_profiles()

    if args.profiles is not None and len(args.profiles) > 0:
        profiles_to_process = args.profiles
    elif args.all_profiles or args.llm:
        profiles_to_process = list(all_profiles.keys())
    else:
        profiles_to_process = []

    if args.llm:
        if not args.output:
            args.output = generate_output_filename(model=args.model, law=args.law, profiles=args.profiles)
        print(f"Output file: {args.output}", file=sys.stderr)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading law: {args.law}", file=sys.stderr)
        law_yaml = load_law_yaml(args.law)

        total_profiles = len(profiles_to_process)
        print(f"Processing {total_profiles} profiles with {args.model}...", file=sys.stderr)

        total_input_tokens = 0
        total_output_tokens = 0

        with open(output_path, "w", encoding="utf-8") as f:
            model_config = AVAILABLE_MODELS[args.model]
            metadata = {
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_config["id"],
                "provider": model_config.get("provider", "anthropic"),
                "law": args.law,
                "profiles_count": total_profiles,
                "graph_type": "decision",
                "approach": "graph",
                "git_info": get_git_info(),
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            for i, bsn in enumerate(profiles_to_process, 1):
                if bsn not in all_profiles:
                    print(f"  [{i}/{total_profiles}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
                    continue

                if not args.quiet:
                    print(f"  [{i}/{total_profiles}] Processing {bsn}...", file=sys.stderr)

                profile_data = all_profiles[bsn]
                person_name = profile_data.get("name", f"Burger {bsn}")

                calc_result = run_calculation(args.law, bsn)
                if calc_result and not args.quiet:
                    req_met = calc_result.get("requirements_met", False)
                    calc_output = calc_result.get("result", {})
                    field_name, amount = _first_amount_output(law_yaml, calc_output)
                    if field_name:
                        print(f"    Calculation: requirements_met={req_met}, {field_name}={amount:.2f} euro", file=sys.stderr)
                    else:
                        print(f"    Calculation: requirements_met={req_met}", file=sys.stderr)

                decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
                graph = decision_extractor.extract()

                if args.visualize:
                    if not VISUALIZATION_AVAILABLE:
                        print("Error: Install with: uv add networkx matplotlib", file=sys.stderr)
                    else:
                        viz_filename = generate_graph_filename(args.law, f"{bsn}_decision", args.viz_format)
                        viz_path = Path(viz_filename)
                        viz_path.parent.mkdir(parents=True, exist_ok=True)
                        graph.visualize(str(viz_path), title=f"Beslissingsgraph: {args.law} - {person_name}")
                        if not args.quiet:
                            print(f"    Graph saved to: {viz_filename}", file=sys.stderr)

                try:
                    result = generate_decision_explanation(
                        decision_extractor=decision_extractor,
                        person_name=person_name,
                        api_key=args.api_key,
                        model=args.model,
                    )

                    total_input_tokens += result["usage"]["input_tokens"]
                    total_output_tokens += result["usage"]["output_tokens"]

                    calc_output = calc_result.get("result", {}) if calc_result else {}
                    profile_vals = decision_extractor.profile_values

                    record = {
                        "record_type": "explanation",
                        "graph_type": "decision",
                        "approach": "graph",
                        "law": args.law,
                        "profile": bsn,
                        "profile_name": person_name,
                        "requirements_met": calc_result.get("requirements_met") if calc_result else None,
                        # Generic: all law outputs and inputs as dicts
                        "law_output": calc_output,
                        "law_input": {k: v["value"] for k, v in profile_vals.items()},
                        "explanation": result["explanation"],
                        "skeleton_used": result["skeleton_used"],
                        "prompt_used": result["prompt_used"],
                        "model": result["model"],
                        "usage": result["usage"],
                        "graph_stats": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
                        "calculation_result": {
                            "requirements_met": calc_result.get("requirements_met") if calc_result else None,
                            "output": calc_output,
                        } if calc_result else None,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if not args.quiet:
                        print(f"    Tokens: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out", file=sys.stderr)

                except Exception as e:
                    print(f"  [{i}/{total_profiles}] Error processing {bsn}: {e}", file=sys.stderr)
                    f.write(json.dumps({"record_type": "error", "law": args.law, "profile": bsn, "error": str(e)}, ensure_ascii=False) + "\n")

        print(f"\nCompleted! Processed {total_profiles} profiles.", file=sys.stderr)
        print(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output", file=sys.stderr)
        print(f"Output saved to: {args.output}", file=sys.stderr)
        return

    # Non-LLM mode
    print(f"Loading law: {args.law}", file=sys.stderr)
    law_yaml = load_law_yaml(args.law)

    if args.decision_graph:
        if args.visualize and len(profiles_to_process) > 1:
            if not VISUALIZATION_AVAILABLE:
                print("Error: Install with: uv add networkx matplotlib", file=sys.stderr)
                sys.exit(1)

            for i, bsn in enumerate(profiles_to_process, 1):
                profile_data = all_profiles.get(bsn)
                if not profile_data:
                    continue
                person_name = profile_data.get("name", f"Burger {bsn}")
                if not args.quiet:
                    print(f"  [{i}/{len(profiles_to_process)}] Processing {bsn} ({person_name})...", file=sys.stderr)

                calc_result = run_calculation(args.law, bsn)
                decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
                graph = decision_extractor.extract()

                viz_filename = generate_graph_filename(args.law, f"{bsn}_decision", args.viz_format)
                viz_path = Path(viz_filename)
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                graph.visualize(str(viz_path), title=f"Beslissingsgraph: {args.law} - {person_name}")
                if not args.quiet:
                    print(f"    Graph saved to: {viz_filename}", file=sys.stderr)

            print(f"\nCompleted! Generated {len(profiles_to_process)} decision graphs.", file=sys.stderr)
            return

        bsn = profiles_to_process[0] if profiles_to_process else None
        profile_data = all_profiles.get(bsn) if bsn else None
        if not bsn or not profile_data:
            print("Error: --decision-graph requires a profile. Use --profiles <BSN>", file=sys.stderr)
            sys.exit(1)

        calc_result = run_calculation(args.law, bsn)
        decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
        graph = decision_extractor.extract()

        if args.format == "text":
            output = decision_extractor.to_explanation_skeleton()
        elif args.format == "json":
            output = json.dumps(graph.to_json(), indent=2, ensure_ascii=False)
        elif args.format == "triples":
            output = "\n".join(f"({s}, {p}, {o})" for s, p, o in graph.to_triples())
    else:
        extractor = LawGraphExtractor(law_yaml)
        graph = extractor.extract()

        bsn = profiles_to_process[0] if profiles_to_process else None
        profile_data = all_profiles.get(bsn) if bsn else None
        if bsn and profile_data:
            profile_extractor = ProfileGraphExtractor(profile_data, bsn)
            graph = profile_extractor.extract(graph)

        if args.format == "text":
            output = graph.to_structured_text()
        elif args.format == "json":
            output = json.dumps(graph.to_json(), indent=2, ensure_ascii=False)
        elif args.format == "triples":
            output = "\n".join(f"({s}, {p}, {o})" for s, p, o in graph.to_triples())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    print(f"\nGraph statistics:", file=sys.stderr)
    print(f"  Nodes: {len(graph.nodes)}", file=sys.stderr)
    print(f"  Edges: {len(graph.edges)}", file=sys.stderr)

    if args.visualize:
        if not VISUALIZATION_AVAILABLE:
            print("Error: Install with: uv add networkx matplotlib", file=sys.stderr)
            sys.exit(1)

        profile_str = profiles_to_process[0] if profiles_to_process else "geen-profiel"
        graph_type = "decision" if args.decision_graph else "full"
        viz_filename = generate_graph_filename(args.law, f"{profile_str}_{graph_type}", args.viz_format)
        viz_path = Path(viz_filename)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        title = f"Beslissingsgraph: {args.law} - {profile_str}" if args.decision_graph else f"Kennisgraaf: {args.law} - {profile_str}"
        graph.visualize(str(viz_path), title=title)


if __name__ == "__main__":
    main()
