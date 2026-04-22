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
# Dutch abbreviation expansion
# ---------------------------------------------------------------------------

_DUTCH_ABBREVIATIONS: dict[str, str] = {
    r"\bzvw\b": "Zorgverzekeringswet",
    r"\baow\b": "Algemene Ouderdomswet",
    r"\bww\b": "Werkloosheidswet",
    r"\bbsn\b": "Burgerservicenummer",
    r"\bbrp\b": "Basisregistratie Personen",
    r"\buwv\b": "UWV",
    r"\bsvb\b": "SVB",
    r"\bszw\b": "Ministerie van Sociale Zaken en Werkgelegenheid",
    r"\bvws\b": "Ministerie van Volksgezondheid, Welzijn en Sport",
    r"\bkvk\b": "Kamer van Koophandel",
    r"\bbibob\b": "Wet bevordering integriteitsbeoordelingen",
    r"\bsvh\b": "Register Sociale Hygiëne",
}


def _expand_abbreviations(text: str) -> str:
    """Expand common Dutch government abbreviations in a label string."""
    for pattern, expansion in _DUTCH_ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


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
- Bespreek ELKE voorwaarde uit het skeleton — zowel de voorwaarden die u WEL haalt ([JA]) als die u NIET haalt ([NEE]) als die we niet konden beoordelen
- Leg bij elke voorwaarde uit WAT het vereiste is én hoe de situatie van de burger zich verhoudt tot dat vereiste:
  bijv. "U bent 45 jaar oud en voldoet daarmee aan de minimumleeftijd van 18 jaar."
  bijv. "U bent niet verzekerd voor ziektekosten, terwijl dit wel vereist is."
- Noem het bedrag als dat er staat, zowel per jaar als per maand
- Praat altijd in de u-vorm tegen {person_name}
- Schrijf bedragen in de vorm van "1.500 euro per jaar" of "125 euro per maand"
- Als er geen voorwaarden of gegevens in het skeleton staan: schrijf alleen de conclusie en de meer-informatie-zin
- {ending}

WAT JE NIET MAG DOEN:
- GEEN briefopmaak ("Geachte heer/mevrouw", "Met vriendelijke groet")
- GEEN extra regels verzinnen die niet in het skeleton staan
- GEEN vragen stellen aan de lezer, GEEN aanbod om te helpen
- Een voorwaarde met [JA] NOOIT overslaan — ook vervulde voorwaarden moeten worden uitgelegd
- Als een voorwaarde is gemarkeerd met [JA]: dit betekent dat de burger eraan VOLDOET — noem dit NOOIT als reden voor afwijzing
- Als het bedrag €0 is terwijl alle voorwaarden [JA] zijn: de reden is het inkomen boven de grens, NIET leeftijd of andere [JA]-voorwaarden
- Verzin GEEN bedragen, drempels of grenzen die niet letterlijk in het skeleton staan
- Voeg GEEN commentaar toe over zaken die het skeleton niet noemt (bijv. vermogen, andere uitkeringen, leeftijdsgrenzen, curatele)
- Noem NIET dat iets "niet relevant is" of "geen rol speelt" tenzij het skeleton dit expliciet vermeldt
- Als een voorwaarde beschrijft dat iemand "voldoet aan voorwaarden" (bijv. "Voldoet aan landelijke voorwaarden"): zeg alleen dat de burger hier WEL of NIET aan voldoet — verzin NOOIT welke sub-voorwaarden daarvoor gelden
- Haal GEEN informatie uit andere wetten of regelingen (bijv. zorgtoeslag, AOW, Zorgverzekeringswet) — elke wet heeft zijn eigen skeleton en u mag ALLEEN het skeleton van deze specifieke beslissing gebruiken
- Sluit NOOIT af met een sectietitel of kopje zoals "Conclusie:" of "Meer informatie:" — schrijf gewone tekst

VOORBEELD GOEDE UITLEG (geen recht, twee voorwaarden waarvan één faalt):
"U heeft geen recht op Zorgtoeslag.
U bent 45 jaar oud en voldoet daarmee aan de minimumleeftijd van 18 jaar. U bent echter niet verzekerd voor ziektekosten via de Zorgverzekeringswet, terwijl dit wel verplicht is.
Voor meer informatie kunt u terecht op www.toeslagen.nl."

VOORBEELD SLECHTE UITLEG (DOE DIT NIET):
"U heeft geen recht omdat u niet verzekerd bent." <- FOUT: noemt de leeftijdsvoorwaarde niet, legt niet uit waarom de verzekering relevant is"""


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

        # If the subject field is completely absent from our resolved input data
        # (not in input_data from the engine, not a known definition constant),
        # we cannot evaluate it regardless of the operation.  The engine may have
        # resolved it from an external service (GBA, BRP) without recording it in
        # input_data, so treating it as NULL/FAILED would be incorrect.
        if actual is None and isinstance(subject, str) and subject.startswith("$"):
            field_name = subject[1:]
            if field_name not in self.profile_values and field_name not in self.definitions:
                return None, self.STATUS_NOT_APPLICABLE

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
        """Return a readable Dutch description of one condition.

        Falls back to legal_basis.explanation when the subject field has no
        description in _input_meta (field resolved by engine from external service).
        Abbreviations in labels are expanded to plain Dutch.
        """
        subject = cond.get("subject", "")
        operation = cond.get("operation", "")
        raw_value = cond.get("value")
        resolved = self._get_value(raw_value)

        field_name = subject.lstrip("$") if isinstance(subject, str) else subject
        meta = self._input_meta.get(field_name, {})
        unit = meta.get("unit", "")

        # Label: prefer YAML description, fall back to legal_basis.explanation,
        # last resort is the raw field name (will be filtered by _node_has_human_label)
        if meta.get("description"):
            label = _expand_abbreviations(meta["description"])
        else:
            lb_expl = cond.get("legal_basis", {}).get("explanation", "")
            if lb_expl:
                # Strip leading "Artikel X lid Y:" type prefix for readability
                label = re.sub(r"^Artikel\s+\d+[^\:]*:\s*", "", lb_expl)
                label = label[0].upper() + label[1:] if label else field_name
            else:
                label = field_name  # will be hidden by _node_has_human_label

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

    def _extract_variables(self, expr: Any) -> list[str]:
        """Recursively collect all $VARIABLE names from a values-expression."""
        if isinstance(expr, str) and expr.startswith("$"):
            return [expr[1:]]
        if isinstance(expr, dict):
            result: list[str] = []
            for v in expr.get("values", []):
                result.extend(self._extract_variables(v))
            return result
        return []

    def _evaluate_values_expr(self, expr: Any) -> Any:
        """Evaluate a nested values-expression recursively. Returns None on failure."""
        if isinstance(expr, str) and expr.startswith("$"):
            return self._get_value(expr)
        if isinstance(expr, (int, float, bool)):
            return expr
        if not isinstance(expr, dict):
            return None
        operation = expr.get("operation", "")
        values = expr.get("values", [])
        operands = [self._evaluate_values_expr(v) for v in values]
        if any(o is None for o in operands):
            return None
        try:
            if operation == "ADD":
                return sum(operands)
            if operation == "SUBTRACT":
                return operands[0] - operands[1]
            if operation == "MULTIPLY":
                result = operands[0]
                for o in operands[1:]:
                    result *= o
                return result
            if operation == "DIVIDE":
                return operands[0] / operands[1] if operands[1] != 0 else None
        except (TypeError, ZeroDivisionError):
            return None
        return None

    def _label_for_values_condition(self, cond: dict) -> str:
        """Generate a human label for a values-format (computed expression) condition."""
        # Prefer legal_basis.explanation
        lb_expl = cond.get("legal_basis", {}).get("explanation", "")
        if lb_expl:
            label = re.sub(r"^Artikel\s+\d+[^\:]*:\s*", "", lb_expl)
            return label[0].upper() + label[1:] if label else "Berekende voorwaarde"

        # Fall back: describe from the variables in the expression (only those with descriptions)
        variables = list(dict.fromkeys(self._extract_variables(cond)))  # deduplicated
        described = []
        for var in variables:
            meta = self._input_meta.get(var, {})
            if meta.get("description"):
                described.append(_expand_abbreviations(meta["description"]))

        operation = cond.get("operation", "GREATER_THAN")
        values = cond.get("values", [])
        compare_value = values[-1] if values else 0
        compare_resolved = self._evaluate_values_expr(compare_value) if isinstance(compare_value, dict) else compare_value

        op_nl = {
            "GREATER_THAN": "moet meer zijn dan",
            "GREATER_OR_EQUAL": "moet minimaal zijn",
            "LESS_THAN": "moet minder zijn dan",
            "LESS_OR_EQUAL": "mag maximaal zijn",
        }
        op_text = op_nl.get(operation, operation)

        base = " en ".join(described) if described else "Berekende waarde"
        return f"{base} (totaal) {op_text} {compare_resolved}"

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
                lb_expl = cond.get("legal_basis", {}).get("explanation", "")
                nodes.append({
                    "id": node_id,
                    "label": self._condition_to_human(cond),
                    "status": self.STATUS_NOT_APPLICABLE if status is None
                              else (self.STATUS_SATISFIED if status else self.STATUS_FAILED),
                    "is_or_group": False,
                    "subject": cond.get("subject", "").lstrip("$"),
                    "operation": cond.get("operation", ""),
                    "value": cond.get("value"),
                    "legal_basis_explanation": lb_expl,
                })

            elif "values" in cond and "operation" in cond:
                # Computed expression condition (e.g. ADD($A, $B) > 0)
                label = self._label_for_values_condition(cond)
                # Try to evaluate the left side of the comparison
                values_list = cond.get("values", [])
                if len(values_list) >= 2:
                    lhs = self._evaluate_values_expr(values_list[0])
                    rhs = self._evaluate_values_expr(values_list[1])
                    if lhs is not None and rhs is not None:
                        try:
                            op = cond["operation"]
                            ops = {
                                "GREATER_THAN": lambda a, b: a > b,
                                "GREATER_OR_EQUAL": lambda a, b: a >= b,
                                "LESS_THAN": lambda a, b: a < b,
                                "LESS_OR_EQUAL": lambda a, b: a <= b,
                                "EQUALS": lambda a, b: a == b,
                            }
                            fn = ops.get(op)
                            result = fn(lhs, rhs) if fn is not None else None
                            status = (self.STATUS_SATISFIED if result
                                      else self.STATUS_FAILED) if result is not None else self.STATUS_NOT_APPLICABLE
                        except (TypeError, ValueError):
                            status = self.STATUS_NOT_APPLICABLE
                    else:
                        status = self.STATUS_NOT_APPLICABLE
                else:
                    status = self.STATUS_NOT_APPLICABLE
                lb_expl = cond.get("legal_basis", {}).get("explanation", "")
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "status": status,
                    "is_or_group": False,
                    "subject": "",
                    "operation": cond.get("operation", ""),
                    "value": None,
                    "legal_basis_explanation": lb_expl,
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
                    "legal_basis_explanation": "",
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
                    "legal_basis_explanation": "",
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
                    "legal_basis_explanation": rule_info.get("legal_basis_explanation", ""),
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

    def _node_has_human_label(self, node: GraphNode) -> bool:
        """Return True if the rule node can be described in human-readable Dutch.

        A node is considered human-readable when:
        - It is an OR group (label is composed from child labels)
        - It has no specific subject (computed expression)
        - Its subject field is in _input_meta (has a Dutch description in the YAML)
        - It has a legal_basis_explanation set (fallback label from condition YAML)

        Nodes that are only described by a raw ALL_CAPS field name are hidden so
        the LLM never sees technical variable names.
        """
        if node.properties.get("is_or_group", False):
            return True
        if node.properties.get("legal_basis_explanation"):
            return True
        subject = node.properties.get("subject", "")
        if not subject:
            return True
        return subject in self._input_meta

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

        # Group RULE nodes by status — only keep nodes with human-readable labels
        rule_nodes = [n for n in self.graph.nodes if n.type == "RULE"]
        rule_nodes_human = [n for n in rule_nodes if self._node_has_human_label(n)]
        satisfied = [n for n in rule_nodes_human if n.properties.get("status") == self.STATUS_SATISFIED]
        failed    = [n for n in rule_nodes_human if n.properties.get("status") == self.STATUS_FAILED]
        unknown   = [n for n in rule_nodes_human if n.properties.get("status") == self.STATUS_NOT_APPLICABLE]

        # Collect condition subjects so we only show citizen-relevant profile data
        condition_subjects: set[str] = {
            n.properties.get("subject", "")
            for n in rule_nodes_human
            if n.properties.get("subject")
        }

        # Personal situation — only fields that correspond to actual conditions
        profile_lines: list[str] = []
        for field_name, info in self.profile_values.items():
            if field_name not in condition_subjects:
                continue  # skip normative constants and calculation-only inputs
            value = info.get("value")
            if value is None:
                continue
            unit = info.get("unit", "")
            description = _expand_abbreviations(info.get("description", field_name))
            profile_lines.append(f"- {description}: {self._format_value(value, unit)}")
        if profile_lines:
            lines.append("## Uw gegevens:")
            lines.extend(profile_lines)
            lines.append("")

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
            # Be specific about what caused the €0: check if income fields are present
            income_keys = [k for k in self.profile_values if "INKOMEN" in k.upper()]
            if income_keys:
                reason = "uw inkomen boven de toepasselijke inkomensgrens ligt"
            else:
                reason = "een berekeningsgrens niet gehaald wordt (zie 'Toepasselijke grenzen' hierboven)"
            lines.append(
                f"U voldoet aan de basisvoorwaarden voor {law_name}. "
                f"U ontvangt echter geen {law_name} omdat {reason}. "
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


def _find_yaml_in_dir(base: Path, org_prefix: str | None = None) -> dict | None:
    """Find and load the most recent YAML in a directory (or its gemeenten/ subdir).

    If org_prefix is given (e.g. "GEMEENTE_AMSTERDAM"), the gemeenten/ subdir is
    searched first and only files whose name starts with that prefix are considered.
    This ensures the skeleton uses the same municipality YAML that the engine ran,
    so its conditions and input fields align with the engine's input_data.
    """
    if not base.is_dir():
        return None

    gemeenten_dir = base / "gemeenten"
    if org_prefix and gemeenten_dir.is_dir():
        # Try org-specific YAML first
        org_yamls = sorted(gemeenten_dir.glob(f"{org_prefix}*.yaml"), reverse=True)
        if org_yamls:
            with open(org_yamls[0]) as f:
                return yaml.load(f, Loader=Loader)

    # When no org_prefix is available, prefer a gemeente YAML over the national
    # YAML that lives in base/.  All gemeente YAMLs for the same law have the
    # same properties.input structure, so any one of them is equally valid for
    # building the _input_meta lookup.
    if gemeenten_dir.is_dir():
        gemeente_yamls = sorted(gemeenten_dir.glob("*.yaml"), reverse=True)
        if gemeente_yamls:
            with open(gemeente_yamls[0]) as f:
                return yaml.load(f, Loader=Loader)

    # Last resort: national/generic YAML in base directory
    yaml_files = sorted(base.glob("*.yaml"), reverse=True)
    if yaml_files:
        with open(yaml_files[0]) as f:
            return yaml.load(f, Loader=Loader)
    return None


def load_law_yaml(law_name: str) -> dict:
    """Load the most recent YAML file for a given law/service name.

    Resolution order:
    1. MCPServiceRegistry (preferred): gets the exact law_path + organisation
       so that gemeente-specific YAMLs are loaded instead of national overrides.
    2. Direct path: laws/<law_name>/
    3. Suffixed path: laws/<law_name>wet/

    MCPServiceRegistry is tried first so that a law like 'alcoholwet' loads
    the gemeente YAML (via org_prefix) rather than the national VWS YAML that
    lives at laws/alcoholwet/VWS-2024-01-01.yaml.
    """
    # 1. Preferred: MCPServiceRegistry with org context
    try:
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        services = get_machine_service()
        connector = MCPLawConnector(services, get_case_manager(), get_claim_manager())
        service = connector.registry.get_service(law_name)
        if service and hasattr(service, "law_path"):
            base = PROJECT_ROOT / "laws" / Path(service.law_path)
            # Find which organisation runs this law (for gemeente-specific YAMLs)
            org_prefix: str | None = None
            try:
                for org, law_list in services.get_discoverable_service_laws().items():
                    if service.law_path in law_list:
                        org_prefix = org
                        break
            except Exception:
                pass
            result = _find_yaml_in_dir(base, org_prefix=org_prefix)
            if result is not None:
                return result
    except Exception:
        pass

    # 2. Fallback: direct path (no org context)
    for base in [PROJECT_ROOT / "laws" / law_name, PROJECT_ROOT / "laws" / f"{law_name}wet"]:
        result = _find_yaml_in_dir(base)
        if result is not None:
            return result

    raise FileNotFoundError(f"Law YAML not found: {law_name}")
