#!/usr/bin/env python3
"""
Generic decision graph extractor for any machine-law YAML.

Uses calc_result["input_data"] (already resolved by the rule engine) instead of
parsing raw profile sources, so it works for every law without a per-law template.

The law YAML's properties.input[].description and properties.output[].description
provide the Dutch field labels that appear in the explanation skeleton.

This module is self-contained: no imports from extraction_zorgtoeslag or other
law-specific modules.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# Optional imports for graph visualization
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents a directed edge in the knowledge graph."""
    source: str
    target: str
    relation: str
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Knowledge graph for law and profile data."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def to_networkx(self) -> "nx.DiGraph":
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("networkx and matplotlib are required: uv add networkx matplotlib")
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id, label=node.label, node_type=node.type, **node.properties)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, relation=edge.relation, **edge.properties)
        return G

    def visualize(self, output_path: str, title: str = "Knowledge Graph", figsize: tuple = (20, 16)) -> None:
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("networkx and matplotlib are required: uv add networkx matplotlib")
        G = self.to_networkx()
        node_colors = {
            "DECISION": "#2ECC71", "RULE": "#E74C3C", "FACT": "#3498DB",
            "THRESHOLD": "#9B59B6", "CALCULATION": "#F39C12",
            "LAW": "#4A90D9", "REQUIREMENT": "#E74C3C", "INPUT": "#27AE60",
            "OUTPUT": "#F39C12", "DEFINITION": "#9B59B6", "PERSON": "#3498DB",
            "VALUE": "#1ABC9C", "OPERATION": "#95A5A6",
        }
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        colors = [node_colors.get(G.nodes[n].get("node_type", ""), "#CCCCCC") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=2000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666666", arrows=True,
                               arrowsize=20, alpha=0.6, connectionstyle="arc3,rad=0.1")
        labels = {n: (G.nodes[n].get("label", n)[:22] + "..." if len(G.nodes[n].get("label", n)) > 25
                      else G.nodes[n].get("label", n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8, font_weight="bold")
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, font_color="#444444")
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=nt)
                           for nt, color in node_colors.items()
                           if any(G.nodes[n].get("node_type") == nt for n in G.nodes())]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Graph visualization saved to: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Available LLM models
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, dict] = {
    "haiku":    {"id": "claude-haiku-4-5-20251001",  "provider": "anthropic", "description": "Fast and cheap, good for batch processing"},
    "sonnet":   {"id": "claude-sonnet-4-5-20250929", "provider": "anthropic", "description": "Balanced performance and cost"},
    "opus":     {"id": "claude-opus-4-6",             "provider": "anthropic", "description": "Most capable, highest quality output"},
    "llama3.2": {"id": "llama3.2:3b",   "provider": "ollama", "description": "Llama 3.2 3B via local Ollama (~2GB RAM)"},
    "llama3.1": {"id": "llama3.1:8b",   "provider": "ollama", "description": "Llama 3.1 8B via local Ollama (~5GB RAM)"},
    "llama3.3": {"id": "llama3.3:70b",  "provider": "ollama", "description": "Llama 3.3 70B via local Ollama (~38GB RAM)"},
    "mistral":  {"id": "mistral:7b",    "provider": "ollama", "description": "Mistral 7B via local Ollama (~4GB RAM)"},
    "deepseek": {"id": "deepseek-r1:8b","provider": "ollama", "description": "DeepSeek R1 8B via local Ollama (~5GB RAM)"},
    "gemma2":   {"id": "gemma2:9b",     "provider": "ollama", "description": "Gemma 2 9B via local Ollama (~6GB RAM)"},
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_git_info() -> dict:
    """Get git commit/branch info for reproducibility (includes submodule if present)."""
    info: dict = {}
    try:
        for cmd, key in [
            (["git", "rev-parse", "HEAD"], "commit"),
            (["git", "rev-parse", "--abbrev-ref", "HEAD"], "branch"),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if r.returncode == 0:
                info[key] = r.stdout.strip()
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if r.returncode == 0:
            info["dirty"] = len(r.stdout.strip()) > 0
        law_path = PROJECT_ROOT / "submodules" / "regelrecht-laws"
        if law_path.exists():
            r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=law_path)
            if r.returncode == 0:
                info["regelrecht_laws_commit"] = r.stdout.strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def load_profiles(profiles_path: str = "data/profiles.yaml") -> dict:
    """Load profiles from YAML; returns the profiles dict keyed by BSN."""
    with open(PROJECT_ROOT / profiles_path) as f:
        data = yaml.load(f, Loader=Loader)
    return data.get("profiles", {})


def _to_dutch_format(value: float) -> str:
    """Format a float as Dutch notation: 17068.17 → '17.068,17'."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")


def _fix_rounded_amounts(text: str, expected_values: dict[str, float]) -> str:
    """Replace rounded amounts in LLM output with exact Dutch-formatted values."""
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


def _build_expected_values(decision_extractor: "DecisionGraphExtractor") -> dict[str, float]:
    """Extract exact euro values (from calc output + inputs) for post-processing."""
    expected: dict[str, float] = {}
    calc_output = (decision_extractor.calc_result or {}).get("result", {})
    # Amount outputs (eurocent → euro)
    for name, meta in decision_extractor._output_meta.items():
        if meta.get("unit") == "eurocent" or meta.get("type") == "amount":
            val = calc_output.get(name)
            if isinstance(val, (int, float)) and val > 0:
                expected[name] = val / 100
    # Amount inputs
    for key, info in decision_extractor.profile_values.items():
        meta = decision_extractor._input_meta.get(key, {})
        if meta.get("unit") == "eurocent" or meta.get("type") == "amount":
            val = info.get("value")
            if isinstance(val, (int, float)) and val > 0:
                expected[key] = val / 100
    return expected


# ---------------------------------------------------------------------------
# Service → info URL mapping
# ---------------------------------------------------------------------------

# Curated URLs per service provider.
SERVICE_URLS: dict[str, str] = {
    "TOESLAGEN": "www.toeslagen.nl",
    "UWV": "www.uwv.nl",
    "SVB": "www.svb.nl",
    "BELASTINGDIENST": "www.belastingdienst.nl",
    "RVO": "www.rvo.nl",
    "KIESRAAD": "www.kiesraad.nl",
    "NVWA": "www.nvwa.nl",
    "KVK": "www.kvk.nl",
    "VWS": "www.rijksoverheid.nl",
    "SZW": "www.rijksoverheid.nl",
}


def _get_service_url(law: dict) -> str | None:
    """Return the canonical info URL for the service that issued this law.

    Returns None for gemeente services — use _get_service_contact() instead.
    """
    return SERVICE_URLS.get(law.get("service", ""))


def _get_service_contact(law: dict) -> str | None:
    """Return the 'meer informatie' line for the skeleton.

    For national services: a URL. For gemeente services: a generic message
    telling the citizen to contact their municipality (no URL, since e.g.
    www.groningen.nl is a tourism site, not the municipal services portal).
    """
    url = _get_service_url(law)
    if url:
        return f"Voor meer informatie kunt u terecht op {url}"
    if law.get("service", "").startswith("GEMEENTE_"):
        return "Voor meer informatie kunt u contact opnemen met uw gemeente."
    return None


# ---------------------------------------------------------------------------
# Shared LLM prompt
# ---------------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """Je bent een informatiesysteem dat Nederlandse burgers uitleg geeft over overheidsbeslissingen.

Je taak is om een beslissingsskeleton om te zetten naar een korte, begrijpelijke uitleg.

VERPLICHT:
- Gebruik ALLEEN informatie uit het skeleton - voeg NIETS toe
- Schrijf in eenvoudig Nederlands (B1-niveau) - korte zinnen, gewone woorden
- Dit is een informatieve tekst, GEEN brief of gesprek
- Noem de naam van de regeling expliciet in de conclusie

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN aanbiedingen voor hulp of vragen aan de lezer
- GEEN technische termen of codes - alles moet in normale taal"""


def create_decision_prompt(skeleton: str, person_name: str) -> str:
    import re
    url_match = re.search(r"www\.\S+", skeleton)
    info_url = url_match.group(0).rstrip(".,)") if url_match else None  # already curated via SERVICE_URLS

    ending = f"Eindig met de zin over {info_url}." if info_url else "Voeg geen website-URL toe."

    return f"""# Informatie over de beslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau).

WAT JE MOET DOEN:
- Begin met de conclusie (wel of geen recht)
- Leg bij elke voorwaarde ook uit WAT het vereiste is: bijv. "u bent 25 jaar oud, de minimumleeftijd is 18 jaar" of "uw inkomen is te hoog voor deze regeling"
- Noem het bedrag als dat er staat, zowel per jaar als per maand
- Praat altijd in de u-vorm tegen {person_name}
- Schrijf bedragen in de vorm van "1.500 euro per jaar" of "125 euro per maand"
- {ending}

WAT JE NIET MAG DOEN:
- GEEN briefopmaak ("Geachte heer/mevrouw", "Met vriendelijke groet")
- GEEN extra regels verzinnen die niet in het skeleton staan
- GEEN vragen stellen aan de lezer, GEEN aanbod om te helpen
- NIET zomaar zeggen "u bent 25 jaar oud" zonder te vermelden waarom dat relevant is
- Als een voorwaarde is gemarkeerd met [JA]: dit betekent dat de burger eraan VOLDOET — noem dit NOOIT als reden voor afwijzing
- Als het bedrag €0 is terwijl alle voorwaarden [JA] zijn: de reden is het inkomen of vermogen boven de grens, NIET leeftijd of andere [JA]-voorwaarden

VOORBEELD GOEDE UITLEG:
"U heeft recht op deze regeling.
U bent 25 jaar oud en voldoet daarmee aan de minimumleeftijd van 18 jaar. Daarnaast bent u verzekerd voor ziektekosten, wat ook vereist is.
Uw inkomen bedraagt 10.000 euro per jaar. Op basis daarvan ontvangt u 1.500 euro per jaar, ofwel 125 euro per maand."

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"U voldoet aan alle voorwaarden: u bent 25 jaar en u bent verzekerd." <- FOUT: legt niet uit waarom 25 jaar relevant is"""


# ---------------------------------------------------------------------------
# Generic DecisionGraphExtractor
# ---------------------------------------------------------------------------

class DecisionGraphExtractor:
    """
    Builds a focused decision subgraph from any law YAML + calc_result.

    profile_values is populated from calc_result["input_data"] — the rule engine
    already resolved every input field from its source service/table/column.
    Field labels come from the law YAML's properties.input[].description.
    """

    STATUS_SATISFIED = "SATISFIED"
    STATUS_FAILED = "FAILED"
    STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

    def __init__(
        self,
        law: dict,
        profile: dict,
        bsn: str,
        calc_result: dict | None,
    ) -> None:
        self.law = law
        self.profile = profile
        self.bsn = bsn
        self.calc_result = calc_result or {}
        self.graph = KnowledgeGraph()

        # Build metadata lookups from law YAML
        self._input_meta: dict[str, dict] = self._build_input_meta()
        self._output_meta: dict[str, dict] = self._build_output_meta()

        # Definitions are constants declared in the law YAML
        self.definitions: dict[str, Any] = self._extract_definitions()

        # profile_values: {field_name: {value, unit, description}}
        self.profile_values: dict[str, dict] = self._extract_profile_values()

    # ------------------------------------------------------------------
    # Effective entitlement (requirements + actions combined)
    # ------------------------------------------------------------------

    @property
    def effective_requirements_met(self) -> bool | None:
        """
        True only when the citizen actually receives an entitlement.

        The engine's `requirements_met` only covers the `requirements` section of the
        YAML.  Many laws (e.g. zorgtoeslag, kindgebonden budget, huurtoeslag) put the
        income/means test inside `actions` as a conditional calculation that returns 0
        when the threshold is exceeded.  In those cases `requirements_met=True` but the
        citizen receives nothing.

        Generic rule: if the primary output field is monetary (unit=eurocent or
        type=amount) AND its computed value is 0 (or absent), we treat the citizen as
        not entitled regardless of what `requirements` returned.

        For non-monetary laws (boolean outputs like kieswet) the engine value is used
        as-is because there is no amount to check.
        """
        base = self.calc_result.get("requirements_met") if self.calc_result else None
        if not base:
            return base  # already False/None — no override needed

        output = (self.calc_result or {}).get("result", {})

        # Find the primary output field in the law YAML
        for out in self.law.get("properties", {}).get("output", []):
            if out.get("citizen_relevance") == "primary":
                unit = (out.get("type_spec") or {}).get("unit", "")
                typ = out.get("type", "")
                if unit == "eurocent" or typ == "amount":
                    val = output.get(out.get("name", ""), 0) or 0
                    return val > 0
                # Primary output is not monetary → keep engine value
                return base

        # No primary output declared: fall back to engine value
        return base

    # ------------------------------------------------------------------
    # Metadata helpers (from law YAML)
    # ------------------------------------------------------------------

    def _build_input_meta(self) -> dict[str, dict]:
        """Build {field_name: {description, unit, type}} from law YAML input definitions."""
        meta: dict[str, dict] = {}
        for inp in self.law.get("properties", {}).get("input", []):
            name = inp.get("name", "")
            if name:
                meta[name] = {
                    "description": inp.get("description", name),
                    "unit": (inp.get("type_spec") or {}).get("unit", ""),
                    "type": inp.get("type", ""),
                }
        return meta

    def _build_output_meta(self) -> dict[str, dict]:
        """Build {field_name: {description, unit, type, citizen_relevance}} from law YAML output."""
        meta: dict[str, dict] = {}
        for out in self.law.get("properties", {}).get("output", []):
            name = out.get("name", "")
            if name:
                meta[name] = {
                    "description": out.get("description", name),
                    "unit": (out.get("type_spec") or {}).get("unit", ""),
                    "type": out.get("type", ""),
                    "citizen_relevance": out.get("citizen_relevance", "secondary"),
                }
        return meta

    def _extract_definitions(self) -> dict[str, Any]:
        """Extract constants from law YAML definitions section."""
        raw = self.law.get("properties", {}).get("definitions", {})
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    def _extract_profile_values(self) -> dict[str, dict]:
        """
        Populate profile_values from calc_result["input_data"].

        input_data keys are prefixed with '$', e.g. '$INKOMEN'.
        We strip the prefix and look up unit/description from the law YAML.

        Filters:
        - Keys not in law YAML input definitions are skipped (they are law
          constants or internal engine variables, not citizen data).
        """
        input_data = self.calc_result.get("input_data", {})
        values: dict[str, dict] = {}
        for key, value in input_data.items():
            field_name = key.lstrip("$") if isinstance(key, str) else key
            # Only include fields explicitly declared as inputs in the law YAML
            if field_name not in self._input_meta:
                continue
            meta = self._input_meta[field_name]
            values[field_name] = {
                "value": value,
                "unit": meta.get("unit", ""),
                "description": meta.get("description", field_name),
                "source": "engine",
            }
        return values

    # ------------------------------------------------------------------
    # Value resolution
    # ------------------------------------------------------------------

    def _get_value(self, ref: Any, return_none_if_missing: bool = False) -> Any:
        """Resolve a $VARIABLE reference against profile_values or definitions."""
        if isinstance(ref, str) and ref.startswith("$"):
            name = ref[1:]
            if name in self.profile_values:
                return self.profile_values[name]["value"]
            if name in self.definitions:
                return self.definitions[name]
            return None
        return ref

    def _format_value(self, value: Any, unit: str = "") -> str:
        """Format a value for human display."""
        if value is None:
            return "Onbekend"
        if unit == "eurocent" and isinstance(value, (int, float)):
            return f"{value / 100:,.2f} euro"
        if isinstance(value, bool):
            return "Ja" if value else "Nee"
        if isinstance(value, float):
            if 0 < value < 1:
                return f"{value * 100:.3f}%"
            return f"{value:,.2f}"
        if isinstance(value, int) and value > 10000:
            return f"{value / 100:,.2f} euro"
        return str(value)

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_condition(self, subject: str, operation: str, value: Any) -> tuple[bool | None, str]:
        """Evaluate a single condition; returns (result, status_label)."""
        actual = self._get_value(subject)
        expected = self._get_value(value)

        if actual is None and operation not in ("IS_NULL", "NOT_NULL"):
            return None, self.STATUS_NOT_APPLICABLE

        try:
            if operation == "IS_NULL":
                result = actual is None
            elif operation == "NOT_NULL":
                result = actual is not None
            elif operation == "IN":
                lst = expected if isinstance(expected, list) else [expected]
                result = actual in lst
            elif operation == "NOT_IN":
                lst = expected if isinstance(expected, list) else [expected]
                result = actual not in lst
            else:
                op_map = {
                    "GREATER_OR_EQUAL": lambda a, b: a >= b,
                    "GREATER_THAN":     lambda a, b: a > b,
                    "LESS_OR_EQUAL":    lambda a, b: a <= b,
                    "LESS_THAN":        lambda a, b: a < b,
                    "EQUALS":           lambda a, b: a == b,
                    "NOT_EQUALS":       lambda a, b: a != b,
                }
                fn = op_map.get(operation)
                if fn is None:
                    return None, self.STATUS_NOT_APPLICABLE
                result = fn(actual, expected)
        except (TypeError, ValueError):
            return None, self.STATUS_NOT_APPLICABLE

        return result, self.STATUS_SATISFIED if result else self.STATUS_FAILED

    def _condition_to_human(self, cond: dict) -> str:
        """Return a readable Dutch description of one condition."""
        subject = cond.get("subject", "")
        operation = cond.get("operation", "")
        raw_value = cond.get("value")
        resolved = self._get_value(raw_value)

        field_name = subject.lstrip("$") if isinstance(subject, str) else subject
        label = self._input_meta.get(field_name, {}).get("description", field_name)
        unit = self._input_meta.get(field_name, {}).get("unit", "")

        op_nl = {
            "GREATER_OR_EQUAL": "moet minimaal",
            "GREATER_THAN":     "moet meer dan",
            "LESS_OR_EQUAL":    "mag maximaal",
            "LESS_THAN":        "moet minder dan",
            "EQUALS":           "moet gelijk zijn aan",
            "NOT_EQUALS":       "mag niet gelijk zijn aan",
            "IS_NULL":          "mag niet aanwezig zijn",
            "NOT_NULL":         "moet aanwezig zijn",
            "IN":               "moet een van de volgende zijn:",
            "NOT_IN":           "mag geen van de volgende zijn:",
        }
        op_text = op_nl.get(operation, operation)

        if operation in ("IS_NULL", "NOT_NULL"):
            return f"{label} {op_text}"
        return f"{label} {op_text} {self._format_value(resolved, unit)}"

    # ------------------------------------------------------------------
    # Requirements traversal
    # ------------------------------------------------------------------

    def _process_conditions(self, conditions: list, prefix: str) -> list[dict]:
        """
        Recursively traverse a list of conditions (from an 'all' or 'any' block).
        Returns a flat list of rule-info dicts for graph nodes.
        """
        nodes: list[dict] = []
        for idx, cond in enumerate(conditions):
            node_id = f"{prefix}_{idx}"

            if "subject" in cond:
                # Leaf condition
                status, _ = self._evaluate_condition(
                    cond["subject"], cond["operation"], cond.get("value")
                )
                nodes.append({
                    "id": node_id,
                    "label": self._condition_to_human(cond),
                    "status": self.STATUS_NOT_APPLICABLE if status is None
                              else (self.STATUS_SATISFIED if status else self.STATUS_FAILED),
                    "is_or_group": False,
                    "subject": cond.get("subject", "").lstrip("$"),
                    "operation": cond.get("operation", ""),
                    "value": cond.get("value"),
                })

            elif "any" in cond:
                # Nested OR group
                sub = self._process_conditions(cond["any"], f"{node_id}_any")
                statuses = [s["status"] for s in sub]
                if any(s == self.STATUS_SATISFIED for s in statuses):
                    group_status = self.STATUS_SATISFIED
                elif all(s == self.STATUS_NOT_APPLICABLE for s in statuses):
                    group_status = self.STATUS_NOT_APPLICABLE
                else:
                    group_status = self.STATUS_FAILED
                labels = [s["label"] for s in sub]
                nodes.append({
                    "id": node_id,
                    "label": " OF ".join(labels),
                    "status": group_status,
                    "is_or_group": True,
                    "subject": "",
                    "operation": "OR",
                    "value": None,
                })

            elif "all" in cond:
                # Nested AND group
                nodes.extend(self._process_conditions(cond["all"], f"{node_id}_all"))

        return nodes

    def _collect_rule_infos(self) -> list[dict]:
        """Collect rule node info from the top-level requirements list."""
        all_infos: list[dict] = []
        for req_idx, req in enumerate(self.law.get("requirements", [])):
            prefix = f"rule_{req_idx}"
            if "all" in req:
                all_infos.extend(self._process_conditions(req["all"], f"{prefix}_all"))
            elif "any" in req:
                sub = self._process_conditions(req["any"], f"{prefix}_any")
                statuses = [s["status"] for s in sub]
                if any(s == self.STATUS_SATISFIED for s in statuses):
                    group_status = self.STATUS_SATISFIED
                elif all(s == self.STATUS_NOT_APPLICABLE for s in statuses):
                    group_status = self.STATUS_NOT_APPLICABLE
                else:
                    group_status = self.STATUS_FAILED
                labels = [s["label"] for s in sub]
                all_infos.append({
                    "id": f"{prefix}_any",
                    "label": " OF ".join(labels),
                    "status": group_status,
                    "is_or_group": True,
                    "subject": "",
                    "operation": "OR",
                    "value": None,
                })
        return all_infos

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def extract(self) -> KnowledgeGraph:
        """Build and return the focused decision subgraph."""
        requirements_met = self.calc_result.get("requirements_met", False) if self.calc_result else None
        output = self.calc_result.get("result", {}) if self.calc_result else {}

        law_name = self.law.get("name", "Regeling")

        # Find the primary output field (citizen_relevance: primary, else first output)
        primary_out: dict = {}
        for out in self.law.get("properties", {}).get("output", []):
            if out.get("citizen_relevance") == "primary":
                primary_out = out
                break
        if not primary_out:
            outputs = self.law.get("properties", {}).get("output", [])
            primary_out = outputs[0] if outputs else {}

        primary_field = primary_out.get("name", "")
        primary_unit = (primary_out.get("type_spec") or {}).get("unit", "")
        primary_type = primary_out.get("type", "")
        primary_value = output.get(primary_field)

        # Build decision label
        if requirements_met and primary_value is not None:
            if primary_unit == "eurocent" or primary_type == "amount":
                amount = primary_value / 100 if isinstance(primary_value, (int, float)) else primary_value
                decision_label = (
                    f"RECHT: {amount:,.2f} euro" if amount > 0
                    else "GEEN RECHT (geen bedrag)"
                )
            elif primary_type == "boolean":
                decision_label = (
                    f"RECHT OP {law_name.upper()}"
                    if primary_value
                    else f"GEEN RECHT OP {law_name.upper()}"
                )
            else:
                decision_label = f"BESLISSING: {primary_value}"
        elif requirements_met:
            decision_label = f"RECHT OP {law_name.upper()}"
        else:
            decision_label = f"GEEN RECHT OP {law_name.upper()}"

        # DECISION node
        decision_node = GraphNode(
            id="decision",
            type="DECISION",
            label=decision_label,
            properties={
                "requirements_met": requirements_met,
                "law": law_name,
                "output": output,
            },
        )
        self.graph.add_node(decision_node)

        # PERSON node
        person_node = GraphNode(
            id=f"person_{self.bsn}",
            type="PERSON",
            label=self.profile.get("name", f"Burger {self.bsn}"),
            properties={"bsn": self.bsn},
        )
        self.graph.add_node(person_node)
        self.graph.add_edge(GraphEdge(
            source=f"person_{self.bsn}",
            target="decision",
            relation="KRIJGT_BESLISSING",
        ))

        # RULE nodes
        for rule_info in self._collect_rule_infos():
            rule_node = GraphNode(
                id=rule_info["id"],
                type="RULE",
                label=rule_info["label"],
                properties={
                    "status": rule_info["status"],
                    "subject": rule_info["subject"],
                    "operation": rule_info["operation"],
                    "is_or_group": rule_info["is_or_group"],
                },
            )
            self.graph.add_node(rule_node)
            self.graph.add_edge(GraphEdge(
                source="decision",
                target=rule_info["id"],
                relation=rule_info["status"],
            ))

        # CALCULATION node (only if there is a non-zero amount output)
        if (
            requirements_met
            and primary_value is not None
            and (primary_unit == "eurocent" or primary_type == "amount")
        ):
            calc_node = GraphNode(
                id="calculation",
                type="CALCULATION",
                label=f"Berekening: {self._format_value(primary_value, primary_unit)}",
                properties={"output_field": primary_field, "output_amount": primary_value},
            )
            self.graph.add_node(calc_node)
            self.graph.add_edge(GraphEdge(
                source="decision",
                target="calculation",
                relation="BEREKEND_ALS",
            ))

        return self.graph

    # ------------------------------------------------------------------
    # Explanation skeleton
    # ------------------------------------------------------------------

    def to_explanation_skeleton(self) -> str:
        """Generate a structured Dutch text skeleton for LLM input."""
        law_name = self.law.get("name", "Regeling")

        decision_node = next((n for n in self.graph.nodes if n.type == "DECISION"), None)
        if not decision_node:
            return "Geen beslissing gevonden."

        requirements_met = decision_node.properties.get("requirements_met", False)
        output = decision_node.properties.get("output", {})

        lines: list[str] = []
        lines.append(f"# Beslissing: {law_name}")
        lines.append(f"## Uitkomst: {decision_node.label}")
        lines.append("")

        # Personal situation — all known input values with Dutch labels
        lines.append("## Uw gegevens:")
        for field_name, info in self.profile_values.items():
            value = info.get("value")
            if value is None:
                continue
            unit = info.get("unit", "")
            description = info.get("description", field_name)
            lines.append(f"- {description}: {self._format_value(value, unit)}")
        lines.append("")

        # Group RULE nodes by status
        rule_nodes = [n for n in self.graph.nodes if n.type == "RULE"]
        satisfied = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_SATISFIED]
        failed    = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_FAILED]
        unknown   = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_NOT_APPLICABLE]

        if unknown:
            lines.append("## Voorwaarden die we niet kunnen beoordelen:")
            lines.append("(Er ontbreken gegevens)")
            for n in unknown:
                lines.append(f"- {n.label}: gegevens ontbreken")
            lines.append("")

        if failed:
            lines.append("## Voorwaarden waar u NIET aan voldoet:")
            for n in failed:
                lines.append(f"- [NEE] {n.label}")
            lines.append("")

        if satisfied:
            lines.append("## Voorwaarden waar u WEL aan voldoet:")
            for n in satisfied:
                lines.append(f"- [JA] {n.label}")
            lines.append("")

        # Output amounts — only show fields that are monetary and > 0
        calc_node = next((n for n in self.graph.nodes if n.type == "CALCULATION"), None)
        if calc_node and requirements_met:
            amount_lines: list[str] = []
            for field_name, value in output.items():
                if value is None or value == 0:
                    continue
                meta = self._output_meta.get(field_name, {})
                if meta.get("unit") != "eurocent" and meta.get("type") != "amount":
                    continue
                unit = meta.get("unit", "")
                description = meta.get("description", field_name)
                amount_lines.append(f"- {description}: {self._format_value(value, unit)}")
            if amount_lines:
                lines.append("## Berekend bedrag:")
                lines.extend(amount_lines)
                lines.append("")

        # Determine effective primary output amount (for conclusion)
        primary_amount: int | float | None = None
        for out_meta in self.law.get("properties", {}).get("output", []):
            if out_meta.get("citizen_relevance") == "primary":
                fname = out_meta.get("name", "")
                unit = (out_meta.get("type_spec") or {}).get("unit", "")
                if unit == "eurocent" or out_meta.get("type") == "amount":
                    primary_amount = output.get(fname, 0) or 0
                break

        # Threshold info — shown when requirements are met but amount is 0.
        # Identifies definitions that look like income/asset limits so the LLM
        # can explain *why* the citizen receives nothing (e.g. income too high).
        if requirements_met and primary_amount is not None and primary_amount == 0:
            threshold_keywords = ("DREMPEL", "GRENS", "MAXIMUM", "LIMIET", "PLAFOND")
            threshold_lines: list[str] = []
            for def_name, def_value in self.definitions.items():
                if not any(kw in def_name.upper() for kw in threshold_keywords):
                    continue
                if not isinstance(def_value, (int, float)):
                    continue
                # Heuristic: values > 100 are likely eurocent amounts
                if def_value > 100:
                    label = def_name.replace("_", " ").capitalize()
                    threshold_lines.append(f"- {label}: {self._format_value(def_value, 'eurocent')}")
            if threshold_lines:
                lines.append("## Toepasselijke grenzen:")
                lines.extend(threshold_lines)
                lines.append("")

        # Conclusion
        lines.append("## Conclusie:")
        if requirements_met and primary_amount is not None and primary_amount == 0:
            lines.append(
                f"U voldoet aan de basisvoorwaarden voor {law_name} "
                f"(leeftijd en verzekering zijn in orde). "
                f"U ontvangt echter geen {law_name} omdat uw inkomen of vermogen "
                f"boven de toepasselijke grens ligt (zie 'Toepasselijke grenzen' hierboven). "
                f"De voorwaarden gemarkeerd met [JA] zijn WEL vervuld — zij zijn NIET de reden voor afwijzing."
            )
        elif requirements_met:
            lines.append(f"U heeft recht op {law_name}.")
        else:
            lines.append(f"U heeft geen recht op {law_name}.")

        # More info — curated per service; gemeente gets generic contact message (no URL)
        contact = _get_service_contact(self.law)
        if contact:
            lines.append("")
            lines.append("## Meer informatie:")
            lines.append(contact)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM integration — same interface as extraction_zorgtoeslag/bijstand/alcoholwet
# ---------------------------------------------------------------------------

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
            },
        }

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
# Law loading + calculation (generic, same pattern as other modules)
# ---------------------------------------------------------------------------

def run_calculation(law_name: str, bsn: str) -> dict | None:
    """Run the law calculation via MCP service. Returns calc_result dict or None."""
    try:
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        services = get_machine_service()
        case_manager = get_case_manager()
        claim_manager = get_claim_manager()
        connector = MCPLawConnector(services, case_manager, claim_manager)

        service = connector.registry.get_service(law_name)
        if not service:
            return None

        calc_result = service.execute(bsn, {})
        if "error" in calc_result:
            return None
        return calc_result
    except Exception as e:
        print(f"Warning: Could not run calculation for {law_name}/{bsn}: {e}", file=sys.stderr)
        return None


def _find_yaml_in_dir(base: Path) -> dict | None:
    """Find and load the most recent YAML in a directory (or its gemeenten/ subdir)."""
    if not base.is_dir():
        return None
    for search_dir in [base, base / "gemeenten"]:
        yaml_files = sorted(search_dir.glob("*.yaml"), reverse=True)
        if yaml_files:
            with open(yaml_files[0]) as f:
                return yaml.load(f, Loader=Loader)
    return None


def load_law_yaml(law_name: str) -> dict:
    """Load the most recent YAML file for a given law/service name.

    Resolution order:
    1. Direct path: laws/<law_name>/
    2. Suffixed path: laws/<law_name>wet/
    3. Via MCPServiceRegistry: look up service.law_path for the given name,
       then resolve that path under laws/.
    """
    for base in [PROJECT_ROOT / "laws" / law_name, PROJECT_ROOT / "laws" / f"{law_name}wet"]:
        result = _find_yaml_in_dir(base)
        if result is not None:
            return result

    # Fall back: ask the MCPServiceRegistry for the registered law_path
    try:
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        services = get_machine_service()
        connector = MCPLawConnector(services, get_case_manager(), get_claim_manager())
        service = connector.registry.get_service(law_name)
        if service and hasattr(service, "law_path"):
            # law_path is e.g. "algemene_ouderdomswet" or "participatiewet/bijstand"
            base = PROJECT_ROOT / "laws" / Path(service.law_path)
            result = _find_yaml_in_dir(base)
            if result is not None:
                return result
    except Exception:
        pass

    raise FileNotFoundError(f"Law YAML not found: {law_name}")
