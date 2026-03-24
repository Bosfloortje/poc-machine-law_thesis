#!/usr/bin/env python3
"""
GraphRAG extraction script for machine law.

This script extracts a knowledge graph from law YAML definitions and citizen profiles,
creating a structured representation that can be used as context for LLM explanations.

The graph captures:
- Law structure (requirements, inputs, outputs, definitions)
- Logical operations (AND, OR, comparisons, calculations)
- Citizen profile data
- Relationships between entities

Output formats:
- Triples (subject, predicate, object) for RAG
- Structured text for LLM context
- NetworkX graph for visualization

Usage:
    uv run python analysis/llm_explanations/scripts/extraction_zorgtoeslag.py --law zorgtoeslag --profile 999993653
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
# scripts/extraction_zorgtoeslag.py → scripts/ → llm_explanations/ → analysis/ → root
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
                    # Convert eurocents to euros for readability
                    euro_value = value / 100
                    lines.append(f"- {node.label}: €{euro_value:,.2f}")
                elif isinstance(value, float) and value < 1:
                    # Percentage
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
                field = node.properties.get("field", "")
                display_value = node.properties.get("display_value", str(node.properties.get("value", "")))
                source = node.properties.get("source_service", "")
                lines.append(f"- {field}: {display_value} (bron: {source})")
            lines.append("")

        # Logical structure with evaluation
        lines.append("## Logische structuur en evaluatie")

        # Get value nodes for evaluation
        value_nodes = {n.properties.get("field"): n.properties.get("value")
                       for n in nodes_by_type.get("VALUE", [])}

        # Get definition values for comparison
        def_values = {n.label: n.properties.get("value")
                      for n in nodes_by_type.get("DEFINITION", [])}

        # Find requirement edges and evaluate
        req_edges = [e for e in self.edges if e.relation == "HAS_REQUIREMENT"]
        if req_edges:
            lines.append("### Voorwaarden (alle moeten waar zijn):")
            for edge in req_edges:
                req_node = next((n for n in self.nodes if n.id == edge.target), None)
                if req_node:
                    subject = req_node.properties.get("subject", "").replace("$", "")
                    operation = req_node.properties.get("operation", "")
                    value_ref = req_node.properties.get("value", "")

                    # Get actual value
                    actual_value = value_nodes.get(subject)

                    # Get comparison value (could be a definition reference)
                    if isinstance(value_ref, str) and value_ref.startswith("$"):
                        compare_value = def_values.get(value_ref[1:], value_ref)
                    else:
                        compare_value = value_ref

                    # Format the evaluation
                    if actual_value is not None:
                        # Format values for display
                        if isinstance(actual_value, bool):
                            actual_display = "Ja" if actual_value else "Nee"
                        else:
                            actual_display = str(actual_value)

                        if isinstance(compare_value, bool):
                            compare_display = "Ja" if compare_value else "Nee"
                        else:
                            compare_display = str(compare_value)

                        # Evaluate the condition
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

        # Find computation edges
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
            """Make value JSON serializable."""
            if hasattr(v, "isoformat"):  # date/datetime
                return v.isoformat()
            return v

        def serialize_props(props: dict) -> dict:
            """Serialize all properties."""
            return {k: serialize_value(v) for k, v in props.items()}

        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "label": n.label,
                    "properties": serialize_props(n.properties)
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "properties": serialize_props(e.properties)
                }
                for e in self.edges
            ]
        }

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX directed graph."""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib are required for visualization. Install with: uv add networkx matplotlib")

        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(
                node.id,
                label=node.label,
                node_type=node.type,
                **node.properties
            )

        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                relation=edge.relation,
                **edge.properties
            )

        return G

    def visualize(self, output_path: str, title: str = "Knowledge Graph", figsize: tuple = (20, 16)) -> None:
        """Generate a visual representation of the graph and save to file.

        Args:
            output_path: Path to save the image (supports .png, .jpg, .jpeg, .pdf, .svg)
            title: Title for the graph
            figsize: Figure size in inches (width, height)
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("NetworkX and matplotlib are required for visualization. Install with: uv add networkx matplotlib")

        G = self.to_networkx()

        # Define colors for different node types
        node_colors = {
            # Original graph types
            "LAW": "#4A90D9",        # Blue
            "REQUIREMENT": "#E74C3C", # Red
            "INPUT": "#27AE60",       # Green
            "OUTPUT": "#F39C12",      # Orange
            "DEFINITION": "#9B59B6",  # Purple
            "PERSON": "#3498DB",      # Light blue
            "VALUE": "#1ABC9C",       # Teal
            "OPERATION": "#95A5A6",   # Gray
            # Decision graph types
            "DECISION": "#2ECC71",    # Bright green - the central outcome
            "RULE": "#E74C3C",        # Red - rules/requirements
            "FACT": "#3498DB",        # Light blue - actual values
            "THRESHOLD": "#9B59B6",   # Purple - definition thresholds
            "CALCULATION": "#F39C12", # Orange - amount calculations
        }

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Compute layout - hierarchical layout works well for law graphs
        try:
            # Try graphviz layout if available (best for hierarchical data)
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            # Fall back to spring layout
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Get node colors based on type
        colors = [node_colors.get(G.nodes[n].get("node_type", ""), "#CCCCCC") for n in G.nodes()]

        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=colors,
            node_size=2000,
            alpha=0.9
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#666666",
            arrows=True,
            arrowsize=20,
            alpha=0.6,
            connectionstyle="arc3,rad=0.1"
        )

        # Create shortened labels for display
        labels = {}
        for n in G.nodes():
            label = G.nodes[n].get("label", n)
            # Truncate long labels
            if len(label) > 25:
                label = label[:22] + "..."
            labels[n] = label

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels=labels,
            font_size=8,
            font_weight="bold"
        )

        # Draw edge labels (relations)
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=edge_labels,
            font_size=6,
            font_color="#444444"
        )

        # Create legend
        legend_elements = []
        for node_type, color in node_colors.items():
            # Only include types that are in the graph
            if any(G.nodes[n].get("node_type") == node_type for n in G.nodes()):
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, label=node_type))

        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Tight layout
        plt.tight_layout()

        # Save figure
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
        """Generate a unique node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def extract(self) -> KnowledgeGraph:
        """Extract the full knowledge graph from the law YAML."""
        self._extract_law_metadata()
        self._extract_definitions()
        self._extract_inputs()
        self._extract_outputs()
        self._extract_requirements()
        self._extract_actions()
        return self.graph

    def _extract_law_metadata(self) -> None:
        """Extract law-level metadata."""
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
        """Extract constant definitions."""
        definitions = self.law.get("properties", {}).get("definitions", {})

        for name, value in definitions.items():
            node = GraphNode(
                id=f"def_{name}",
                type="DEFINITION",
                label=name,
                properties={
                    "value": value,
                    "value_type": type(value).__name__
                }
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(
                source=self.law_node_id,
                target=node.id,
                relation="HAS_DEFINITION"
            ))

    def _extract_inputs(self) -> None:
        """Extract input field definitions."""
        inputs = self.law.get("properties", {}).get("input", [])

        for inp in inputs:
            name = inp.get("name", "")
            service_ref = inp.get("service_reference", {})
            legal_basis = inp.get("legal_basis", {})

            node = GraphNode(
                id=f"input_{name}",
                type="INPUT",
                label=name,
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
            self.graph.add_edge(GraphEdge(
                source=self.law_node_id,
                target=node.id,
                relation="HAS_INPUT"
            ))

    def _extract_outputs(self) -> None:
        """Extract output field definitions."""
        outputs = self.law.get("properties", {}).get("output", [])

        for out in outputs:
            name = out.get("name", "")
            legal_basis = out.get("legal_basis", {})

            node = GraphNode(
                id=f"output_{name}",
                type="OUTPUT",
                label=name,
                properties={
                    "description": out.get("description", ""),
                    "data_type": out.get("type", ""),
                    "citizen_relevance": out.get("citizen_relevance", ""),
                    "legal_explanation": legal_basis.get("explanation", ""),
                }
            )
            self.graph.add_node(node)
            self.graph.add_edge(GraphEdge(
                source=self.law_node_id,
                target=node.id,
                relation="HAS_OUTPUT"
            ))

    def _extract_requirements(self) -> None:
        """Extract requirement conditions."""
        requirements = self.law.get("requirements", [])

        for req in requirements:
            if "all" in req:
                # AND condition - all must be true
                all_conditions = req["all"]
                for cond in all_conditions:
                    self._extract_condition(cond, "AND")
            elif "any" in req:
                # OR condition - at least one must be true
                any_conditions = req["any"]
                for cond in any_conditions:
                    self._extract_condition(cond, "OR")
            else:
                # Single condition
                self._extract_condition(req, "SINGLE")

    def _extract_condition(self, condition: dict, logic_type: str) -> str:
        """Extract a single condition and return its node ID."""
        subject = condition.get("subject", "")
        operation = condition.get("operation", "")
        value = condition.get("value", "")

        # Create human-readable label
        op_labels = {
            "GREATER_OR_EQUAL": ">=",
            "GREATER_THAN": ">",
            "LESS_OR_EQUAL": "<=",
            "LESS_THAN": "<",
            "EQUALS": "==",
            "NOT_EQUALS": "!=",
        }
        op_label = op_labels.get(operation, operation)

        # Clean up variable references
        subject_clean = subject.replace("$", "")
        value_clean = str(value).replace("$", "") if isinstance(value, str) else str(value)

        label = f"{subject_clean} {op_label} {value_clean}"

        node_id = self._gen_id("req")
        node = GraphNode(
            id=node_id,
            type="REQUIREMENT",
            label=label,
            properties={
                "subject": subject,
                "operation": operation,
                "value": value,
                "logic_type": logic_type,
            }
        )
        self.graph.add_node(node)
        self.graph.add_edge(GraphEdge(
            source=self.law_node_id,
            target=node_id,
            relation="HAS_REQUIREMENT"
        ))

        # Link to input if subject is an input field
        if subject.startswith("$"):
            input_name = subject[1:]  # Remove $
            input_id = f"input_{input_name}"
            if any(n.id == input_id for n in self.graph.nodes):
                self.graph.add_edge(GraphEdge(
                    source=node_id,
                    target=input_id,
                    relation="DEPENDS_ON"
                ))

        return node_id

    def _extract_actions(self) -> None:
        """Extract action/computation logic."""
        actions = self.law.get("actions", [])

        for action in actions:
            output_name = action.get("output", "")
            output_id = f"output_{output_name}"

            if "value" in action and "operation" not in action:
                # Simple value assignment
                value = action["value"]
                node_id = self._gen_id("action")
                node = GraphNode(
                    id=node_id,
                    type="OPERATION",
                    label=f"Stel {output_name} = {value}",
                    properties={
                        "operation_type": "ASSIGN",
                        "value": value
                    }
                )
                self.graph.add_node(node)
                self.graph.add_edge(GraphEdge(
                    source=output_id,
                    target=node_id,
                    relation="COMPUTED_BY"
                ))

            elif "operation" in action:
                # Complex operation
                self._extract_operation(action, output_id)

    def _extract_operation(self, action: dict, output_id: str, depth: int = 0) -> str:
        """Extract a computation operation recursively."""
        operation = action.get("operation", "")

        if operation == "IF":
            return self._extract_if_operation(action, output_id, depth)
        else:
            # Arithmetic or other operation
            node_id = self._gen_id("op")

            legal_basis = action.get("legal_basis", {})
            explanation = legal_basis.get("explanation", "")

            node = GraphNode(
                id=node_id,
                type="OPERATION",
                label=f"{operation} berekening",
                properties={
                    "operation_type": operation,
                    "legal_explanation": explanation,
                }
            )
            self.graph.add_node(node)

            if depth == 0:
                self.graph.add_edge(GraphEdge(
                    source=output_id,
                    target=node_id,
                    relation="COMPUTED_BY"
                ))

            return node_id

    def _extract_if_operation(self, action: dict, output_id: str, depth: int) -> str:
        """Extract IF/THEN/ELSE conditional logic."""
        conditions = action.get("conditions", [])

        node_id = self._gen_id("if")

        # Build description of conditions
        condition_descriptions = []
        for cond in conditions:
            if "test" in cond:
                test = cond["test"]
                subject = test.get("subject", "")
                operation = test.get("operation", "")
                value = test.get("value", "")

                op_labels = {
                    "GREATER_THAN": ">",
                    "GREATER_OR_EQUAL": ">=",
                    "LESS_THAN": "<",
                    "LESS_OR_EQUAL": "<=",
                    "EQUALS": "==",
                }
                op_label = op_labels.get(operation, operation)

                subject_clean = str(subject).replace("$", "")
                value_clean = str(value).replace("$", "")

                condition_descriptions.append(f"ALS {subject_clean} {op_label} {value_clean}")
            elif "else" in cond:
                condition_descriptions.append("ANDERS")

        node = GraphNode(
            id=node_id,
            type="OPERATION",
            label="Conditionele berekening",
            properties={
                "operation_type": "IF",
                "conditions": "; ".join(condition_descriptions),
            }
        )
        self.graph.add_node(node)

        if depth == 0:
            self.graph.add_edge(GraphEdge(
                source=output_id,
                target=node_id,
                relation="COMPUTED_BY"
            ))

        return node_id


class ProfileGraphExtractor:
    """Extracts graph data from a citizen profile."""

    def __init__(self, profile: dict, bsn: str):
        self.profile = profile
        self.bsn = bsn

    def _get_nested_value(self, sources: dict, service: str, table: str, field: str) -> Any:
        """Get a value from nested profile sources structure."""
        try:
            service_data = sources.get(service, {})
            table_data = service_data.get(table, [])
            if isinstance(table_data, list) and len(table_data) > 0:
                return table_data[0].get(field)
            elif isinstance(table_data, dict):
                return table_data.get(field)
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def extract(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Add profile data to the knowledge graph."""
        person_id = f"person_{self.bsn}"
        sources = self.profile.get("sources", {})

        # Extract key values for toeslagen
        extracted_values = {}

        # LEEFTIJD from RvIG.personen.age
        age = self._get_nested_value(sources, "RvIG", "personen", "age")
        if age is not None:
            extracted_values["LEEFTIJD"] = {"value": age, "source": "RvIG", "unit": "jaar"}

        # HEEFT_PARTNER from RvIG.personen.has_partner
        has_partner = self._get_nested_value(sources, "RvIG", "personen", "has_partner")
        if has_partner is not None:
            extracted_values["HEEFT_PARTNER"] = {"value": has_partner, "source": "RvIG", "unit": ""}

        # IS_VERZEKERDE from RVZ.verzekeringen.polis_status
        polis_status = self._get_nested_value(sources, "RVZ", "verzekeringen", "polis_status")
        if polis_status is not None:
            is_verzekerde = polis_status == "ACTIEF"
            extracted_values["IS_VERZEKERDE"] = {"value": is_verzekerde, "source": "RVZ", "unit": ""}

        # INKOMEN from UWV.uwv_toetsingsinkomen.toetsingsinkomen (in eurocents)
        inkomen = self._get_nested_value(sources, "UWV", "uwv_toetsingsinkomen", "toetsingsinkomen")
        if inkomen is not None:
            extracted_values["INKOMEN"] = {"value": inkomen, "source": "UWV", "unit": "eurocent"}

        # VERMOGEN from BELASTINGDIENST.belastingdienst_vermogen.vermogen
        vermogen = self._get_nested_value(sources, "BELASTINGDIENST", "belastingdienst_vermogen", "vermogen")
        if vermogen is not None:
            extracted_values["VERMOGEN"] = {"value": vermogen, "source": "BELASTINGDIENST", "unit": "eurocent"}

        # PARTNER_INKOMEN - check if partner exists
        partner_inkomen = self._get_nested_value(sources, "UWV", "uwv_toetsingsinkomen_partner", "toetsingsinkomen")
        if partner_inkomen is not None:
            extracted_values["PARTNER_INKOMEN"] = {"value": partner_inkomen, "source": "UWV", "unit": "eurocent"}
        elif has_partner is False:
            extracted_values["PARTNER_INKOMEN"] = {"value": 0, "source": "N/A (geen partner)", "unit": "eurocent"}

        # Create person node with extracted values
        node = GraphNode(
            id=person_id,
            type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={
                "bsn": self.bsn,
                "description": self.profile.get("description", ""),
                "extracted_values": extracted_values,
            }
        )
        graph.add_node(node)

        # Create value nodes for each extracted value
        for field_name, field_data in extracted_values.items():
            value = field_data["value"]
            source = field_data["source"]
            unit = field_data["unit"]

            value_id = f"value_{self.bsn}_{field_name}"

            # Format display value
            if unit == "eurocent" and isinstance(value, (int, float)):
                display_value = f"€{value / 100:,.2f}"
            elif isinstance(value, bool):
                display_value = "Ja" if value else "Nee"
            else:
                display_value = str(value)

            value_node = GraphNode(
                id=value_id,
                type="VALUE",
                label=f"{field_name} = {display_value}",
                properties={
                    "field": field_name,
                    "value": value,
                    "display_value": display_value,
                    "source_service": source,
                    "unit": unit,
                }
            )
            graph.add_node(value_node)
            graph.add_edge(GraphEdge(
                source=person_id,
                target=value_id,
                relation="HAS_VALUE"
            ))

            # Link to input field if exists
            input_id = f"input_{field_name}"
            if any(n.id == input_id for n in graph.nodes):
                graph.add_edge(GraphEdge(
                    source=value_id,
                    target=input_id,
                    relation="PROVIDES_VALUE_FOR"
                ))

        return graph


class DecisionGraphExtractor:
    """
    Extracts a focused decision subgraph for GraphRAG.

    Instead of showing the entire law structure, this creates a minimal graph
    showing only the decision path for a specific person:
    - Facts relevant to this person
    - Rules that fired or failed (with SATISFIED/FAILED status)
    - Definitions used by those rules
    - The final decision outcome

    This makes it impossible for the LLM to hallucinate - it can only:
    - Arrange the order of explanation
    - Simplify the language
    - Formulate at B1 level
    """

    # Status labels for normative evaluation
    STATUS_SATISFIED = "SATISFIED"
    STATUS_FAILED = "FAILED"
    STATUS_AFFECTS_AMOUNT = "AFFECTS_AMOUNT"
    STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

    # Human-readable field name translations (technical -> Dutch)
    FIELD_TRANSLATIONS = {
        "LEEFTIJD": "Uw leeftijd",
        "IS_VERZEKERDE": "U bent verzekerd voor ziektekosten",
        "HEEFT_PARTNER": "U heeft een toeslagpartner",
        "INKOMEN": "Uw toetsingsinkomen",
        "PARTNER_INKOMEN": "Het inkomen van uw partner",
        "VERMOGEN": "Uw vermogen",
    }

    # Human-readable operation translations
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

        # Extract profile values first
        self.profile_values = self._extract_profile_values()
        # Extract law definitions
        self.definitions = self.law.get("properties", {}).get("definitions", {})

    def _extract_profile_values(self) -> dict:
        """Extract all relevant values from the profile."""
        sources = self.profile.get("sources", {})
        values = {}

        def get_nested(service: str, table: str, field: str) -> Any:
            try:
                service_data = sources.get(service, {})
                table_data = service_data.get(table, [])
                if isinstance(table_data, list) and len(table_data) > 0:
                    return table_data[0].get(field)
                elif isinstance(table_data, dict):
                    return table_data.get(field)
            except (KeyError, IndexError, TypeError):
                pass
            return None

        # Extract all relevant values
        age = get_nested("RvIG", "personen", "age")
        if age is not None:
            values["LEEFTIJD"] = {"value": age, "source": "RvIG", "unit": "jaar"}

        has_partner = get_nested("RvIG", "personen", "has_partner")
        if has_partner is not None:
            values["HEEFT_PARTNER"] = {"value": has_partner, "source": "RvIG", "unit": ""}

        polis_status = get_nested("RVZ", "verzekeringen", "polis_status")
        if polis_status is not None:
            is_verzekerde = polis_status == "ACTIEF"
            values["IS_VERZEKERDE"] = {"value": is_verzekerde, "source": "RVZ", "unit": ""}

        inkomen = get_nested("UWV", "uwv_toetsingsinkomen", "toetsingsinkomen")
        if inkomen is not None:
            values["INKOMEN"] = {"value": inkomen, "source": "UWV", "unit": "eurocent"}

        vermogen = get_nested("BELASTINGDIENST", "belastingdienst_vermogen", "vermogen")
        if vermogen is not None:
            values["VERMOGEN"] = {"value": vermogen, "source": "BELASTINGDIENST", "unit": "eurocent"}

        partner_inkomen = get_nested("UWV", "uwv_toetsingsinkomen_partner", "toetsingsinkomen")
        if partner_inkomen is not None:
            values["PARTNER_INKOMEN"] = {"value": partner_inkomen, "source": "UWV", "unit": "eurocent"}
        elif has_partner is False:
            values["PARTNER_INKOMEN"] = {"value": 0, "source": "N/A", "unit": "eurocent"}

        return values

    def _get_value(self, ref: str, return_none_if_missing: bool = False) -> Any:
        """Get a value from profile or definitions, resolving $ references."""
        if isinstance(ref, str) and ref.startswith("$"):
            name = ref[1:]
            # First check profile values
            if name in self.profile_values:
                return self.profile_values[name]["value"]
            # Then check definitions
            if name in self.definitions:
                return self.definitions[name]
            # Not found - return None for profile values (they should exist)
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
        """
        Evaluate a single condition and return (result, status).
        Returns (None, NOT_APPLICABLE) if we can't evaluate.
        """
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
        # 1. Create DECISION node (central node showing the outcome)
        requirements_met = self.calc_result.get("requirements_met", False) if self.calc_result else None
        output = self.calc_result.get("result", {}) if self.calc_result else {}

        if requirements_met and "hoogte_toeslag" in output:
            amount = output["hoogte_toeslag"] / 100
            if amount > 0:
                decision_label = f"RECHT: {amount:,.2f} euro"
            else:
                decision_label = "GEEN RECHT (inkomen te hoog)"
        elif requirements_met:
            decision_label = "RECHT OP TOESLAG"
        else:
            decision_label = "GEEN RECHT"

        decision_node = GraphNode(
            id="decision",
            type="DECISION",
            label=decision_label,
            properties={
                "requirements_met": requirements_met,
                "law": self.law.get("name", ""),
                "output": output,
            }
        )
        self.graph.add_node(decision_node)

        # 2. Create PERSON node
        person_node = GraphNode(
            id=f"person_{self.bsn}",
            type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn}
        )
        self.graph.add_node(person_node)

        # Connect person to decision
        self.graph.add_edge(GraphEdge(
            source=f"person_{self.bsn}",
            target="decision",
            relation="KRIJGT_BESLISSING"
        ))

        # 3. Process requirements and create RULE nodes with status
        requirements = self.law.get("requirements", [])
        used_definitions = set()

        for req in requirements:
            conditions = []
            if "all" in req:
                conditions = req["all"]
            elif "any" in req:
                conditions = req["any"]
            else:
                conditions = [req]

            for cond in conditions:
                subject = cond.get("subject", "")
                operation = cond.get("operation", "")
                value = cond.get("value", "")

                # Evaluate the condition
                result, status = self._evaluate_condition(subject, operation, value)

                # Get actual value for display
                subject_name = subject[1:] if subject.startswith("$") else subject
                actual_value = self._get_value(subject)
                actual_unit = self.profile_values.get(subject_name, {}).get("unit", "")

                # Get expected value for display
                value_name = value[1:] if isinstance(value, str) and value.startswith("$") else None
                expected_value = self._get_value(value)

                # Track used definitions
                if value_name and value_name in self.definitions:
                    used_definitions.add(value_name)

                # Create human-readable label
                op_labels = {
                    "GREATER_OR_EQUAL": ">=",
                    "GREATER_THAN": ">",
                    "LESS_OR_EQUAL": "<=",
                    "LESS_THAN": "<",
                    "EQUALS": "==",
                    "NOT_EQUALS": "!=",
                }
                op_label = op_labels.get(operation, operation)

                # Format the rule label with actual values
                actual_display = self._format_value(actual_value, actual_unit)
                expected_display = self._format_value(expected_value)

                # Create rule label - show status indicator for clarity
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
                    }
                )
                self.graph.add_node(rule_node)

                # Connect rule to decision with status
                self.graph.add_edge(GraphEdge(
                    source=f"rule_{subject_name}",
                    target="decision",
                    relation=status,
                    properties={"evaluated": True}
                ))

                # Create FACT node for the actual value
                fact_label = f"{subject_name} = {actual_display}"
                fact_node = GraphNode(
                    id=f"fact_{subject_name}",
                    type="FACT",
                    label=fact_label,
                    properties={
                        "field": subject_name,
                        "value": actual_value,
                        "source": self.profile_values.get(subject_name, {}).get("source", "Niet beschikbaar"),
                    }
                )
                self.graph.add_node(fact_node)

                # Connect person to fact
                self.graph.add_edge(GraphEdge(
                    source=f"person_{self.bsn}",
                    target=f"fact_{subject_name}",
                    relation="HAS_FACT"
                ))

                # Connect fact to rule
                self.graph.add_edge(GraphEdge(
                    source=f"fact_{subject_name}",
                    target=f"rule_{subject_name}",
                    relation="USED_IN"
                ))

        # 4. Add only the definitions that are actually used
        for def_name in used_definitions:
            def_value = self.definitions[def_name]
            def_display = self._format_value(def_value)

            def_node = GraphNode(
                id=f"def_{def_name}",
                type="THRESHOLD",
                label=f"{def_name} = {def_display}",
                properties={
                    "name": def_name,
                    "value": def_value,
                }
            )
            self.graph.add_node(def_node)

            # Connect definition to the rule that uses it
            rule_id = f"rule_{def_name.replace('MINIMUM_', '').replace('GRENS_', '')}"
            # Find the rule that uses this definition
            for node in self.graph.nodes:
                if node.type == "RULE":
                    expected = node.properties.get("expected_value")
                    if expected == def_value:
                        self.graph.add_edge(GraphEdge(
                            source=f"def_{def_name}",
                            target=node.id,
                            relation="DEFINES_THRESHOLD"
                        ))

        # 5. Add income-affects-amount relationship if applicable
        if requirements_met and "INKOMEN" in self.profile_values:
            inkomen = self.profile_values["INKOMEN"]["value"]
            # Check if income affects the amount (not just eligibility)
            if "hoogte_toeslag" in output:
                amount_node = GraphNode(
                    id="amount_calc",
                    type="CALCULATION",
                    label=f"Berekend bedrag: {output['hoogte_toeslag'] / 100:,.2f} euro",
                    properties={
                        "input_income": inkomen,
                        "output_amount": output["hoogte_toeslag"],
                    }
                )
                self.graph.add_node(amount_node)

                # Connect income fact to calculation
                if any(n.id == "fact_INKOMEN" for n in self.graph.nodes):
                    self.graph.add_edge(GraphEdge(
                        source="fact_INKOMEN",
                        target="amount_calc",
                        relation=self.STATUS_AFFECTS_AMOUNT
                    ))

                # Connect calculation to decision
                self.graph.add_edge(GraphEdge(
                    source="amount_calc",
                    target="decision",
                    relation="DETERMINES"
                ))

        return self.graph

    def _format_rule_human_readable(self, rule_node: GraphNode) -> str:
        """Format a rule in human-readable Dutch instead of technical notation."""
        subject = rule_node.properties.get("subject", "")
        operation = rule_node.properties.get("operation", "")
        actual_value = rule_node.properties.get("actual_value")
        expected_value = rule_node.properties.get("expected_value")

        # Get human-readable field name
        field_name = self.FIELD_TRANSLATIONS.get(subject, subject)

        # Get human-readable operation
        op_text = self.OPERATION_TRANSLATIONS.get(operation, operation)

        # Format values
        unit = self.profile_values.get(subject, {}).get("unit", "")
        actual_display = self._format_value(actual_value, unit)
        expected_display = self._format_value(expected_value, unit if unit else "")

        # Special handling for boolean fields
        if isinstance(actual_value, bool):
            if subject == "IS_VERZEKERDE":
                return "Ja, u bent verzekerd voor ziektekosten" if actual_value else "Nee, u bent niet verzekerd voor ziektekosten"
            elif subject == "HEEFT_PARTNER":
                return "Ja, u heeft een toeslagpartner" if actual_value else "Nee, u heeft geen toeslagpartner"
            else:
                return f"{field_name}: {'Ja' if actual_value else 'Nee'}"

        # For numeric comparisons, make it natural
        if subject == "LEEFTIJD":
            return f"U bent {actual_display} jaar oud (vereist: {op_text} {expected_display} jaar)"

        # Default format
        return f"{field_name}: {actual_display} ({op_text} {expected_display})"

    def to_explanation_skeleton(self) -> str:
        """
        Convert the decision graph to a structured explanation skeleton.
        This is the constrained context that the LLM must follow.
        """
        lines = []

        # Find decision
        decision_node = next((n for n in self.graph.nodes if n.type == "DECISION"), None)
        if not decision_node:
            return "Geen beslissing gevonden."

        requirements_met = decision_node.properties.get("requirements_met", False)
        law_name = decision_node.properties.get("law", "")

        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        # Show personal situation (relevant facts for calculation)
        # Always show key fields, even when unknown - this makes it clear what data is missing
        lines.append("## Persoonlijke situatie:")

        # Partner status
        if "HEEFT_PARTNER" in self.profile_values:
            has_partner = self.profile_values["HEEFT_PARTNER"]["value"]
            lines.append(f"- Heeft partner: {'Ja' if has_partner else 'Nee'}")
        else:
            lines.append("- Heeft partner: Onbekend")

        # Income
        if "INKOMEN" in self.profile_values:
            inkomen = self.profile_values["INKOMEN"]["value"]
            lines.append(f"- Toetsingsinkomen: {self._format_value(inkomen, 'eurocent')}")
        else:
            lines.append("- Toetsingsinkomen: Onbekend")

        # Partner income (only show if has partner or unknown)
        has_partner = self.profile_values.get("HEEFT_PARTNER", {}).get("value")
        if has_partner is True or has_partner is None:
            if "PARTNER_INKOMEN" in self.profile_values:
                partner_inkomen = self.profile_values["PARTNER_INKOMEN"]["value"]
                if partner_inkomen > 0:
                    lines.append(f"- Partner inkomen: {self._format_value(partner_inkomen, 'eurocent')}")
            elif has_partner is True:
                lines.append("- Partner inkomen: Onbekend")

        # Assets/wealth
        if "VERMOGEN" in self.profile_values:
            vermogen = self.profile_values["VERMOGEN"]["value"]
            lines.append(f"- Vermogen: {self._format_value(vermogen, 'eurocent')}")
        else:
            lines.append("- Vermogen: Onbekend")

        lines.append("")

        # Group rules by status
        satisfied_rules = []
        failed_rules = []
        unknown_rules = []

        for node in self.graph.nodes:
            if node.type == "RULE":
                status = node.properties.get("status", "")
                if status == self.STATUS_SATISFIED:
                    satisfied_rules.append(node)
                elif status == self.STATUS_FAILED:
                    failed_rules.append(node)
                elif status == self.STATUS_NOT_APPLICABLE:
                    unknown_rules.append(node)

        # Show unknown/missing data rules
        if unknown_rules:
            lines.append("## Voorwaarden die we niet kunnen beoordelen:")
            lines.append("(Er ontbreken gegevens)")
            for rule in unknown_rules:
                subject = rule.properties.get("subject", "")
                field_name = self.FIELD_TRANSLATIONS.get(subject, subject)
                lines.append(f"- {field_name}: gegevens ontbreken")
            lines.append("")

        # Show failed rules
        if failed_rules:
            lines.append("## Voorwaarden waar u NIET aan voldoet:")
            for rule in failed_rules:
                human_text = self._format_rule_human_readable(rule)
                lines.append(f"- ✗ {human_text}")
            lines.append("")

        # Show satisfied rules
        if satisfied_rules:
            lines.append("## Voorwaarden waar u WEL aan voldoet:")
            for rule in satisfied_rules:
                human_text = self._format_rule_human_readable(rule)
                lines.append(f"- ✓ {human_text}")
            lines.append("")

        # Show amount calculation if applicable - now with more detail
        output = decision_node.properties.get("output", {})
        calc_node = next((n for n in self.graph.nodes if n.type == "CALCULATION"), None)
        if calc_node and requirements_met:
            lines.append("## Berekening hoogte toeslag:")

            # Show inputs that affect the amount
            if "INKOMEN" in self.profile_values:
                inkomen = self.profile_values["INKOMEN"]["value"]
                lines.append(f"- Uw toetsingsinkomen: {self._format_value(inkomen, 'eurocent')}")

            if "HEEFT_PARTNER" in self.profile_values:
                has_partner = self.profile_values["HEEFT_PARTNER"]["value"]
                if has_partner:
                    lines.append("- Type huishouden: met toeslagpartner")
                    if "PARTNER_INKOMEN" in self.profile_values:
                        partner_inkomen = self.profile_values["PARTNER_INKOMEN"]["value"]
                        lines.append(f"- Inkomen partner: {self._format_value(partner_inkomen, 'eurocent')}")
                        gezamenlijk = inkomen + partner_inkomen
                        lines.append(f"- Gezamenlijk inkomen: {self._format_value(gezamenlijk, 'eurocent')}")
                else:
                    lines.append("- Type huishouden: alleenstaand")

            # Show the output amount
            if "hoogte_toeslag" in output:
                amount = output["hoogte_toeslag"]
                lines.append(f"- Berekend jaarbedrag: {self._format_value(amount, 'eurocent')}")
                monthly = amount / 12
                lines.append(f"- Berekend maandbedrag: {self._format_value(monthly, 'eurocent')}")

            lines.append("")

        # Final statement - check if amount is actually > 0
        toeslag_amount = output.get("hoogte_toeslag", 0) if output else 0
        lines.append("## Conclusie:")
        if requirements_met and toeslag_amount > 0:
            lines.append(f"U heeft recht op {law_name}.")
        elif requirements_met and toeslag_amount == 0:
            lines.append(f"U voldoet aan de basisvoorwaarden voor {law_name}, maar op basis van uw inkomen krijgt u geen toeslag.")
        else:
            lines.append(f"U heeft geen recht op {law_name}.")

        # Standard closing - prevents hallucination of incorrect advice
        lines.append("")
        lines.append("## Meer informatie:")
        lines.append("Voor meer informatie over toeslagen kunt u terecht op www.toeslagen.nl")

        return "\n".join(lines)


def run_calculation(law_name: str, bsn: str) -> dict | None:
    """Run the actual law calculation using the MCP service.

    Returns the calculation result or None if calculation failed.
    """
    try:
        # Import the same way as extract_explanations.py
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        # Initialize services
        services = get_machine_service()
        case_manager = get_case_manager()
        claim_manager = get_claim_manager()

        # Initialize MCP connector
        mcp_connector = MCPLawConnector(services, case_manager, claim_manager)

        # Get the service for this law
        service = mcp_connector.registry.get_service(law_name)
        if not service:
            return None

        # Execute the calculation
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
            # Find the most recent YAML file
            yaml_files = list(law_path.glob("*.yaml"))
            if yaml_files:
                # Sort by name (usually includes date)
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


def create_graph_prompt(graph: KnowledgeGraph, calc_result: dict | None = None) -> str:
    """Create an LLM prompt with graph context."""
    graph_context = graph.to_structured_text()

    prompt = f"""# Kenniscontext over de regeling

{graph_context}

"""
    if calc_result:
        requirements_met = calc_result.get("requirements_met", False)
        output = calc_result.get("result", {})

        # Format output with euro amounts
        output_lines = []
        for key, value in output.items():
            if isinstance(value, (int, float)) and (key.endswith("_toeslag") or key.endswith("_bedrag") or "hoogte" in key):
                # Convert eurocents to euros
                euro_value = value / 100
                output_lines.append(f"- {key}: {euro_value:,.2f} euro")
            elif isinstance(value, bool):
                output_lines.append(f"- {key}: {'Ja' if value else 'Nee'}")
            else:
                output_lines.append(f"- {key}: {value}")

        output_formatted = "\n".join(output_lines) if output_lines else "Geen output"

        prompt += f"""# Berekeningsresultaat

- Voldoet aan voorwaarden: {"Ja" if requirements_met else "Nee"}

Berekende uitkomst:
{output_formatted}

"""

    prompt += """# Opdracht

Geef een duidelijke uitleg in eenvoudig Nederlands (B1-niveau) over:
1. WAAROM deze burger wel of niet in aanmerking komt voor deze regeling
2. Welke specifieke voorwaarden wel/niet zijn voldaan op basis van de logische structuur
3. WAT het berekende bedrag is en hoe dit tot stand komt (indien van toepassing)
4. Welke factoren uit het profiel van de burger hebben geleid tot dit resultaat

BELANGRIJK:
- Gebruik de kenniscontext hierboven om nauwkeurig uit te leggen welke regels zijn toegepast
- Verwijs naar de specifieke vereisten en definities uit de wet
- Vermeld het exacte berekende bedrag in euros
- Dit is een standalone informatieve tekst, GEEN chatgesprek
- Eindig NIET met vragen of aanbiedingen voor hulp"""

    return prompt


def create_decision_prompt(skeleton: str, person_name: str, info_url: str | None = None) -> str:
    """
    Create a CONSTRAINED LLM prompt using the decision graph skeleton.

    This prompt gives the LLM a strict skeleton of what to explain,
    preventing hallucination. The LLM can only:
    - Rephrase in B1 Dutch
    - Add connecting words
    - Structure the explanation nicely

    It CANNOT:
    - Add new rules
    - Invent comparisons not in the skeleton
    - Change the evaluation status
    """
    if info_url is None:
        import re
        url_match = re.search(r'www\.\S+', skeleton)
        info_url = url_match.group(0).rstrip(".,)") if url_match else "www.toeslagen.nl"

    prompt = f"""# Informatie over de beslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau).

WAT JE MOET DOEN:
- Begin met de conclusie (wel of geen recht)
- Leg bij elke voorwaarde ook uit WAT het vereiste is: bijv. "u bent 25 jaar oud, de minimumleeftijd is 18 jaar" of "uw inkomen is te hoog voor deze toeslag"
- Noem het bedrag als dat er staat
- Eindig met de zin over {info_url}
- Praat altijd in de u-vorm tegen {person_name}
- Schrijf bedragen altijd in de vorm van "1.500 euro per jaar" of "125 euro per maand"

WAT JE NIET MAG DOEN:
- GEEN briefopmaak gebruiken (geen "Geachte heer/mevrouw", geen "Met vriendelijke groet")
- GEEN verwijzingen naar gemeente, UWV, Belastingdienst of andere instanties
- GEEN extra voorwaarden of regels verzinnen die niet in het skeleton staan
- GEEN vragen stellen aan de lezer
- GEEN aanbod om te helpen
- GEEN derde persoon aanspreken - schrijf direct tegen {person_name}
- GEEN bedragen zonder details te noemen, per jaar en per maand als dat in het skeleton staat
- NIET zomaar zeggen "u bent 25 jaar oud" zonder te vermelden waarom dat relevant is

VOORBEELD GOEDE UITLEG:
"U heeft recht op zorgtoeslag.
U bent 25 jaar oud en voldoet daarmee aan de minimumleeftijd van 18 jaar. Daarnaast bent u verzekerd voor ziektekosten, wat ook vereist is.
Uw inkomen bedraagt 10.000 euro per jaar. Op basis daarvan ontvangt u 1.500 euro zorgtoeslag per jaar. Dit komt neer op 125 euro per maand.

Voor meer informatie kunt u terecht op www.toeslagen.nl"

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"U voldoet aan alle voorwaarden: u bent 25 jaar en u bent verzekerd." <- FOUT: uitlegt niet waarom 25 jaar relevant is
"Geachte heer Jansen, Hierbij informeren wij u... Met vriendelijke groet" <- FOUT: briefopmaak
"Neem contact op met de gemeente" <- FOUT: verkeerde instantie
"Uw vermogen is laag genoeg" <- FOUT: als vermogen niet in skeleton staat"""

    return prompt


def _to_dutch_format(value: float) -> str:
    """Format a float as Dutch number notation: 17068.17 -> 17.068,17"""
    formatted = f"{value:,.2f}"  # American: "17,068.17"
    return formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")


def _fix_rounded_amounts(text: str, expected_values: dict[str, float]) -> str:
    """Replace rounded amounts in LLM output with exact values from calculation.

    Handles all common rounded forms an LLM might produce:
      - "17.068"      (Dutch thousands, no cents)
      - "17,068"      (American thousands, no cents)
      - "17068"       (plain integer)
      - "17068,17"    (no thousands sep, comma decimal)
      - "17068.17"    (no thousands sep, dot decimal)
      - "17.068,00"   (Dutch with zero cents)
      - "17,068.00"   (American with zero cents)
    All are replaced with the exact Dutch notation, e.g. "17.068,17".
    """
    result = text
    for exact_value in expected_values.values():
        if exact_value < 1:
            continue
        exact_dutch = _to_dutch_format(exact_value)
        # Skip if the exact Dutch notation is already present in text
        if exact_dutch in result:
            continue
        exact_int = int(exact_value)
        cents = round((exact_value - exact_int) * 100)
        cents_str = f"{cents:02d}"
        dutch_thousands = f"{exact_int:,}".replace(",", ".")  # "17.068"
        american_thousands = f"{exact_int:,}"  # "17,068"
        plain = str(exact_int)  # "17068"

        # Build patterns from most specific to least specific.
        # The plain-integer patterns use (?![,.\d]) so a number like "1537"
        # inside "1537,82" is never matched (the comma blocks it).
        patterns = []
        if cents > 0:
            # Most specific: full number with wrong decimal separator but no thousands sep
            patterns += [
                (r"(?<!\d)" + re.escape(plain) + r"," + re.escape(cents_str) + r"(?!\d)", exact_dutch),
                (r"(?<!\d)" + re.escape(plain) + r"\." + re.escape(cents_str) + r"(?!\d)", exact_dutch),
            ]
        patterns += [
            # Dutch/American thousands with ,00 / .00 suffix
            (re.escape(dutch_thousands) + r",00\b", exact_dutch),
            (re.escape(american_thousands) + r"\.00\b", exact_dutch),
            # Dutch/American thousands with no decimal part at all
            (r"(?<![,.\d])" + re.escape(dutch_thousands) + r"(?![,.\d])", exact_dutch),
            (r"(?<![,.\d])" + re.escape(american_thousands) + r"(?![,.\d])", exact_dutch),
            # Bare integer — only when NOT followed by comma, dot, or digit
            (r"(?<!\d)" + re.escape(plain) + r"(?![,.\d])", exact_dutch),
        ]

        for pattern, replacement in patterns:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                result = new_result
                break
    return result


def _build_expected_values(decision_extractor: "DecisionGraphExtractor") -> dict[str, float]:
    """Extract the exact euro values that should appear in the explanation.

    Only includes monetary amounts (hoogte_toeslag and euro-denominated income/assets)
    to avoid false positives on ages, boolean flags, or other non-monetary values.
    """
    expected: dict[str, float] = {}
    calc_result = decision_extractor.calc_result or {}
    calc_output = calc_result.get("result", {})

    # Toeslag amount (stored in eurocents)
    if calc_output.get("hoogte_toeslag"):
        expected["hoogte_toeslag"] = calc_output["hoogte_toeslag"] / 100

    # Only include profile values that are monetary (eurocent unit or clearly large money amounts)
    MONETARY_KEYS = {"INKOMEN", "PARTNER_INKOMEN", "VERMOGEN", "TOETSINGSINKOMEN"}
    for key, info in decision_extractor.profile_values.items():
        if key not in MONETARY_KEYS:
            continue
        val = info.get("value")
        if not isinstance(val, (int, float)) or val <= 0:
            continue
        unit = info.get("unit", "")
        # Eurocent values are stored as large integers; convert to euros
        expected[key] = val / 100 if (unit == "eurocent" or val > 100000) else float(val)

    return expected


DECISION_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse burgers uitleg geeft over overheidsbeslissingen.

Je taak is om een beslissingsskeleton om te zetten naar een korte, begrijpelijke uitleg.

VERPLICHT:
- Gebruik ALLEEN informatie uit het skeleton - voeg NIETS toe
- Schrijf in eenvoudig Nederlands (B1-niveau) - korte zinnen, gewone woorden
- Dit is een informatieve tekst, GEEN brief of gesprek
- Eindig altijd met de tekst over www.toeslagen.nl uit het skeleton

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN verwijzingen naar gemeente, UWV of andere instanties (tenzij in skeleton)
- GEEN aanbiedingen voor hulp of vragen aan de lezer
- GEEN technische termen of codes behouden - alles moet in normale taal"""


def generate_decision_explanation(
    decision_extractor: "DecisionGraphExtractor",
    person_name: str,
    api_key: str | None = None,
    model: str = "haiku",
) -> dict:
    """Generate an LLM explanation using the constrained decision skeleton."""
    model_config = AVAILABLE_MODELS[model]
    model_id = model_config["id"]
    provider = model_config.get("provider", "anthropic")

    # Get the skeleton - this is the ONLY context the LLM gets
    skeleton = decision_extractor.to_explanation_skeleton()
    prompt = create_decision_prompt(skeleton, person_name)

    # Pre-compute the exact expected values for post-processing
    expected_values = _build_expected_values(decision_extractor)

    if provider == "ollama":
        # Use Ollama for local Llama models
        import ollama

        response = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.2,
                "num_predict": 1000,
            },
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
        # Use Anthropic
        import anthropic

        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model_id,
            max_tokens=1000,  # Shorter - skeleton constrains output
            temperature=0.2,  # Lower temperature for more faithful reproduction
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


def generate_llm_explanation(
    graph: KnowledgeGraph,
    calc_result: dict | None = None,
    api_key: str | None = None,
    model: str = "haiku",
) -> dict:
    """Generate an LLM explanation using graph context."""
    model_config = AVAILABLE_MODELS[model]
    model_id = model_config["id"]
    provider = model_config.get("provider", "anthropic")

    prompt = create_graph_prompt(graph, calc_result)
    system_prompt = "Je bent een informatiesysteem dat Nederlandse burgers objectieve uitleg geeft over overheidsregelingen. Schrijf standalone informatieve teksten in eenvoudig Nederlands (B1-niveau). Gebruik de kenniscontext om nauwkeurige uitleg te geven over welke regels zijn toegepast."

    if provider == "ollama":
        # Use Ollama for local Llama models
        import ollama

        response = ollama.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.3,
                "num_predict": 1500,
            },
        )

        return {
            "explanation": response["message"]["content"],
            "prompt_used": prompt,
            "model": model_id,
            "provider": provider,
            "usage": {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            }
        }
    else:
        # Use Anthropic
        import anthropic

        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model_id,
            max_tokens=1500,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "explanation": response.content[0].text,
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

    if profiles is None or len(profiles) == 0:
        profile_part = "all-profiles"
    elif len(profiles) == 1:
        profile_part = profiles[0]
    else:
        profile_part = f"{len(profiles)}profiles"

    filename = f"{timestamp}_{model}_{law}_{profile_part}_graphrag.jsonl"
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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception as e:
        info["error"] = str(e)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph from law YAML for GraphRAG"
    )
    parser.add_argument("--law", default="zorgtoeslag", help="Law name (e.g., zorgtoeslag)")
    parser.add_argument("--profiles", nargs="*", help="BSN(s) of profile(s) to include (omit for all profiles)")
    parser.add_argument("--output", help="Output file (default: auto-generated)")
    parser.add_argument("--format", choices=["text", "json", "triples"], default="text",
                        help="Output format")
    parser.add_argument("--llm", action="store_true", help="Generate LLM explanation using graph context")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var). Not required for Ollama models.")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="haiku",
                        help="LLM model to use")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--visualize", action="store_true", help="Save graph visualization to output/graphs/ folder")
    parser.add_argument("--viz-format", choices=["png", "pdf", "svg", "jpg"], default="png",
                        help="Image format for visualization (default: png)")
    parser.add_argument("--decision-graph", action="store_true",
                        help="Use focused decision subgraph (GraphRAG style) instead of full law graph")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Process all available profiles (for batch visualization)")

    args = parser.parse_args()

    # Load all available profiles
    all_profiles = load_profiles()

    # Determine which profiles to process
    if args.profiles is not None and len(args.profiles) > 0:
        # Explicit profiles specified
        profiles_to_process = args.profiles
    elif args.all_profiles or args.llm:
        # --all-profiles flag or LLM mode = process all profiles
        profiles_to_process = list(all_profiles.keys())
    else:
        # Default: no profiles (just show law structure)
        profiles_to_process = []

    # LLM mode - process all profiles
    if args.llm:
        # Generate output filename if not specified
        if not args.output:
            args.output = generate_output_filename(
                model=args.model,
                law=args.law,
                profiles=args.profiles,
            )
        print(f"Output file: {args.output}", file=sys.stderr)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load law once
        print(f"Loading law: {args.law}", file=sys.stderr)
        law_yaml = load_law_yaml(args.law)

        total_profiles = len(profiles_to_process)
        print(f"Processing {total_profiles} profiles with {args.model}...", file=sys.stderr)

        # Track totals
        total_input_tokens = 0
        total_output_tokens = 0

        # Open output file and write metadata first
        with open(output_path, "w", encoding="utf-8") as f:
            # Write metadata record first
            model_config = AVAILABLE_MODELS[args.model]
            metadata = {
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_config["id"],
                "provider": model_config.get("provider", "anthropic"),
                "law": args.law,
                "profiles_count": total_profiles,
                "graph_type": "decision",  # Using focused decision graph (GraphRAG)
                "approach": "constrained_skeleton",  # LLM gets skeleton, can't hallucinate
                "git_info": get_git_info(),
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            # Process each profile
            for i, bsn in enumerate(profiles_to_process, 1):
                if bsn not in all_profiles:
                    print(f"  [{i}/{total_profiles}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
                    continue

                if not args.quiet:
                    print(f"  [{i}/{total_profiles}] Processing {bsn}...", file=sys.stderr)

                profile_data = all_profiles[bsn]
                person_name = profile_data.get("name", f"Burger {bsn}")

                # Run the actual calculation to get the result
                calc_result = run_calculation(args.law, bsn)
                if calc_result and not args.quiet:
                    req_met = calc_result.get("requirements_met", False)
                    output = calc_result.get("result", {})
                    # Show key output value
                    if "hoogte_toeslag" in output:
                        amount = output["hoogte_toeslag"] / 100
                        print(f"    Calculation: requirements_met={req_met}, hoogte_toeslag={amount:.2f} euro", file=sys.stderr)
                    else:
                        print(f"    Calculation: requirements_met={req_met}", file=sys.stderr)

                # Create DECISION GRAPH (focused subgraph, not full law graph)
                decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
                graph = decision_extractor.extract()

                # Visualize decision graph if requested
                if args.visualize:
                    if not VISUALIZATION_AVAILABLE:
                        print("Error: NetworkX and matplotlib are required for visualization.", file=sys.stderr)
                        print("Install with: uv add networkx matplotlib", file=sys.stderr)
                    else:
                        viz_filename = generate_graph_filename(args.law, f"{bsn}_decision", args.viz_format)
                        viz_path = Path(viz_filename)
                        viz_path.parent.mkdir(parents=True, exist_ok=True)
                        title = f"Beslissingsgraph: {args.law} - {person_name}"
                        graph.visualize(str(viz_path), title=title)
                        if not args.quiet:
                            print(f"    Graph saved to: {viz_filename}", file=sys.stderr)

                # Generate explanation using CONSTRAINED decision prompt
                try:
                    result = generate_decision_explanation(
                        decision_extractor=decision_extractor,
                        person_name=person_name,
                        api_key=args.api_key,
                        model=args.model,
                    )

                    total_input_tokens += result["usage"]["input_tokens"]
                    total_output_tokens += result["usage"]["output_tokens"]

                    # Extract key values for easy analysis
                    calc_output = calc_result.get("result", {}) if calc_result else {}
                    profile_vals = decision_extractor.profile_values

                    # Write result record
                    record = {
                        "record_type": "explanation",
                        "graph_type": "decision",  # New: using focused decision graph
                        "law": args.law,
                        "profile": bsn,
                        "profile_name": person_name,
                        # Key calculated values as top-level fields for easy analysis
                        "requirements_met": calc_result.get("requirements_met") if calc_result else None,
                        "hoogte_toeslag": calc_output.get("hoogte_toeslag", 0) / 100 if calc_output.get("hoogte_toeslag") else 0,
                        "hoogte_toeslag_per_maand": (calc_output.get("hoogte_toeslag", 0) / 100 / 12) if calc_output.get("hoogte_toeslag") else 0,
                        # Profile input values
                        "inkomen": profile_vals.get("INKOMEN", {}).get("value"),
                        "partner_inkomen": profile_vals.get("PARTNER_INKOMEN", {}).get("value"),
                        "vermogen": profile_vals.get("VERMOGEN", {}).get("value"),
                        "leeftijd": profile_vals.get("LEEFTIJD", {}).get("value"),
                        "heeft_partner": profile_vals.get("HEEFT_PARTNER", {}).get("value"),
                        "is_verzekerde": profile_vals.get("IS_VERZEKERDE", {}).get("value"),
                        # LLM output
                        "explanation": result["explanation"],
                        "skeleton_used": result["skeleton_used"],  # The constrained skeleton
                        "prompt_used": result["prompt_used"],
                        "model": result["model"],
                        "usage": result["usage"],
                        "graph_stats": {
                            "nodes": len(graph.nodes),
                            "edges": len(graph.edges),
                        },
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
                    # Write error record
                    error_record = {
                        "record_type": "error",
                        "law": args.law,
                        "profile": bsn,
                        "error": str(e),
                    }
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

        print(f"\nCompleted! Processed {total_profiles} profiles.", file=sys.stderr)
        print(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output", file=sys.stderr)
        print(f"Output saved to: {args.output}", file=sys.stderr)

        return

    # Non-LLM mode
    # Load law
    print(f"Loading law: {args.law}", file=sys.stderr)
    law_yaml = load_law_yaml(args.law)

    # Use decision graph or full graph based on flag
    if args.decision_graph:
        # Decision graph mode - can process multiple profiles for visualization
        if args.visualize and len(profiles_to_process) > 1:
            # Batch visualization mode - process all profiles
            if not VISUALIZATION_AVAILABLE:
                print("Error: NetworkX and matplotlib are required for visualization.", file=sys.stderr)
                print("Install with: uv add networkx matplotlib", file=sys.stderr)
                sys.exit(1)

            print(f"Generating decision graphs for {len(profiles_to_process)} profiles...", file=sys.stderr)

            for i, bsn in enumerate(profiles_to_process, 1):
                profile_data = all_profiles.get(bsn)
                if not profile_data:
                    print(f"  [{i}/{len(profiles_to_process)}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
                    continue

                person_name = profile_data.get("name", f"Burger {bsn}")
                print(f"  [{i}/{len(profiles_to_process)}] Processing {bsn} ({person_name})...", file=sys.stderr)

                # Run calculation
                calc_result = run_calculation(args.law, bsn)
                if calc_result and not args.quiet:
                    req_met = calc_result.get("requirements_met", False)
                    output_result = calc_result.get("result", {})
                    if "hoogte_toeslag" in output_result:
                        amount = output_result["hoogte_toeslag"] / 100
                        print(f"    Calculation: requirements_met={req_met}, hoogte_toeslag={amount:.2f} euro", file=sys.stderr)
                    else:
                        print(f"    Calculation: requirements_met={req_met}", file=sys.stderr)

                # Create decision graph
                decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
                graph = decision_extractor.extract()

                # Visualize
                viz_filename = generate_graph_filename(args.law, f"{bsn}_decision", args.viz_format)
                viz_path = Path(viz_filename)
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                title = f"Beslissingsgraph: {args.law} - {person_name}"
                graph.visualize(str(viz_path), title=title)
                if not args.quiet:
                    print(f"    Graph saved to: {viz_filename}", file=sys.stderr)

            print(f"\nCompleted! Generated {len(profiles_to_process)} decision graphs.", file=sys.stderr)
            return

        # Single profile mode
        bsn = profiles_to_process[0] if profiles_to_process else None
        profile_data = all_profiles.get(bsn) if bsn else None

        if not bsn or not profile_data:
            print("Error: --decision-graph requires a profile. Use --profiles <BSN>", file=sys.stderr)
            sys.exit(1)

        print(f"Loading profile: {bsn}", file=sys.stderr)
        print("Building decision subgraph (GraphRAG style)...", file=sys.stderr)

        # Run calculation first to get the result
        calc_result = run_calculation(args.law, bsn)
        if calc_result:
            req_met = calc_result.get("requirements_met", False)
            output_result = calc_result.get("result", {})
            if "hoogte_toeslag" in output_result:
                amount = output_result["hoogte_toeslag"] / 100
                print(f"  Calculation: requirements_met={req_met}, hoogte_toeslag={amount:.2f} euro", file=sys.stderr)
            else:
                print(f"  Calculation: requirements_met={req_met}", file=sys.stderr)

        # Create decision graph
        decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
        graph = decision_extractor.extract()

        # For decision graph, also show the explanation skeleton
        if args.format == "text":
            output = decision_extractor.to_explanation_skeleton()
        elif args.format == "json":
            output = json.dumps(graph.to_json(), indent=2, ensure_ascii=False)
        elif args.format == "triples":
            triples = graph.to_triples()
            output = "\n".join(f"({s}, {p}, {o})" for s, p, o in triples)
    else:
        # Original full graph extraction (law structure without person-specific decision)
        extractor = LawGraphExtractor(law_yaml)
        graph = extractor.extract()

        # Get profile if specified
        bsn = profiles_to_process[0] if profiles_to_process else None
        profile_data = all_profiles.get(bsn) if bsn else None

        # Add profile if specified (first one only for non-LLM mode)
        if bsn and profile_data:
            print(f"Loading profile: {bsn}", file=sys.stderr)
            profile_extractor = ProfileGraphExtractor(profile_data, bsn)
            graph = profile_extractor.extract(graph)
        elif bsn:
            print(f"Warning: Profile {bsn} not found", file=sys.stderr)

        # Regular graph output mode
        if args.format == "text":
            output = graph.to_structured_text()
        elif args.format == "json":
            output = json.dumps(graph.to_json(), indent=2, ensure_ascii=False)
        elif args.format == "triples":
            triples = graph.to_triples()
            output = "\n".join(f"({s}, {p}, {o})" for s, p, o in triples)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    # Stats
    print(f"\nGraph statistics:", file=sys.stderr)
    print(f"  Nodes: {len(graph.nodes)}", file=sys.stderr)
    print(f"  Edges: {len(graph.edges)}", file=sys.stderr)

    node_types = {}
    for node in graph.nodes:
        node_types[node.type] = node_types.get(node.type, 0) + 1
    print(f"  Node types: {node_types}", file=sys.stderr)

    # Visualize if requested
    if args.visualize:
        if not VISUALIZATION_AVAILABLE:
            print("Error: NetworkX and matplotlib are required for visualization.", file=sys.stderr)
            print("Install with: uv add networkx matplotlib", file=sys.stderr)
            sys.exit(1)

        # Generate auto filename
        profile_str = profiles_to_process[0] if profiles_to_process else "geen-profiel"
        graph_type = "decision" if args.decision_graph else "full"
        viz_filename = generate_graph_filename(args.law, f"{profile_str}_{graph_type}", args.viz_format)
        viz_path = Path(viz_filename)

        # Ensure output directory exists
        viz_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate title
        if args.decision_graph:
            title = f"Beslissingsgraph: {args.law} - {profile_str}"
        else:
            title = f"Kennisgraaf: {args.law} - {profile_str}"

        graph.visualize(str(viz_path), title=title)


if __name__ == "__main__":
    main()
