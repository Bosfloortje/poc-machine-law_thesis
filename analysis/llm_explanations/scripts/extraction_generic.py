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
    "gpt4":     {"id": "gpt-4o",        "provider": "openai",     "description": "GPT-4o via OpenAI API"},
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


def _humanize_code_value(value: str) -> str:
    """Convert ALL_CAPS_CODE or snake_case_code values to readable lowercase Dutch text.

    Matches YAML enum codes: all-uppercase (MEDISCH_VOLLEDIG) and lowercase snake_case
    (ernstig_gevaar, geen_gevaar). Leaves normal words like 'Actief', 'Ja', 'Nee' untouched.
    """
    # Any identifier containing at least one underscore (snake_case or UPPER_CASE)
    if re.match(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$", value):
        return value.replace("_", " ").lower()
    # ALL_CAPS without underscore (e.g. ACTIEF)
    if re.match(r"^[A-Z]{2,}$", value):
        return value.lower()
    return value


def _normalize_unicode_for_llm(text: str) -> str:
    """Replace accented Dutch characters with ASCII equivalents.

    Prevents Ollama models from emitting replacement characters (U+FFFD) when
    they encounter multi-byte UTF-8 sequences like ë (U+00EB).
    The original skeleton (with correct unicode) is preserved separately.
    """
    replacements = {
        "ë": "e", "é": "e", "è": "e", "ê": "e",
        "ï": "i", "í": "i", "î": "i",
        "ü": "u", "ú": "u", "û": "u",
        "ö": "o", "ó": "o", "ô": "o",
        "ä": "a", "á": "a", "â": "a",
        "ñ": "n",
        "Ë": "E", "É": "E", "È": "E",
        "Ï": "I", "Ü": "U", "Ö": "O", "Ä": "A",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _fix_rounded_amounts(text: str, expected_values: dict[str, float]) -> str:
    """Replace rounded amounts in LLM output with exact Dutch-formatted values."""
    # Remove euro signs — the prompt instructs "euro" not "€", but models often ignore that.
    # U+20AC (€) and U+FFFD (replacement char from encoding issues) before digits.
    result = re.sub(r"[€\ufffd](?=\s*\d)", "", text)
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
                # LLM mixed Dutch/English: "1,907,98" (American thousands + Dutch decimal comma)
                (re.escape(american_thousands) + r"," + re.escape(cents_str) + r"(?!\d)", exact_dutch),
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
    """Expand common Dutch government abbreviations to full names."""
    for pattern, expansion in _DUTCH_ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Service → info URL mapping
# ---------------------------------------------------------------------------

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
    """Return the canonical info URL for the service that issued this law."""
    return SERVICE_URLS.get(law.get("service", ""))


def _get_service_contact(law: dict) -> str | None:
    """Return the 'meer informatie' line for the skeleton."""
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

Je taak is om een beslissingsskeleton LETTERLIJK om te zetten naar een korte uitleg. Het skeleton bevat de volledige, feitelijke beslissing. Jij schrijft die om naar begrijpelijk Nederlands — je voegt NIETS toe en verzint NIETS.

ABSOLUTE REGELS:
- Gebruik UITSLUITEND informatie die letterlijk in het skeleton staat
- Verzin NOOIT bedragen, inkomens, leeftijden of andere gegevens die niet in het skeleton staan
- Als het skeleton zegt "RECHT: X euro": schrijf dat de burger recht heeft en vermeld het exacte bedrag
- Als het skeleton zegt "RECHT OP ... MAAR BEDRAG IS 0 EURO": schrijf dat de burger formeel aan de voorwaarden voldoet, maar dat het berekende bedrag 0 euro is vanwege het inkomen of vermogen. Eindig NOOIT met "U heeft recht op X" — de uitkomst is immers 0 euro.
- Als het skeleton zegt "GEEN RECHT" of "U heeft geen recht": schrijf dan dat de burger GEEN RECHT heeft
- [JA] bij een voorwaarde betekent dat de burger VOLDOET — noem dit NOOIT als reden voor afwijzing of twijfel
- [NEE] bij een voorwaarde betekent dat de burger NIET voldoet — dit is een reden voor afwijzing
- "gegevens ontbreken" betekent dat de voorwaarde niet beoordeeld kon worden — beschrijf dit neutraal, de burger hoeft niets te doen
- Gebruik het woord "euro", nooit het euro-teken
- Schrijf ALTIJD in de u-vorm — gebruik NOOIT de naam van de persoon als onderwerp, NOOIT "hij/zij/men"

VERBODEN:
- GEEN briefopmaak ("Geachte", "Met vriendelijke groet", aanhef, ondertekening)
- GEEN vragen aan de lezer, GEEN aanbod om te helpen
- GEEN technische termen of codes — alles in normale taal
- GEEN informatie uit andere regelingen dan die in het skeleton
- NOOIT twijfel zaaien over een [JA]-voorwaarde ("we weten niet of", "onduidelijk of")"""


def create_decision_prompt(skeleton: str, person_name: str) -> str:
    url_match = re.search(r"www\.\S+", skeleton)
    info_url = url_match.group(0).rstrip(".,)") if url_match else None

    ending = f"Eindig met de zin over {info_url}." if info_url else "Voeg geen website-URL toe."

    return f"""# Informatie over de beslissing

{skeleton}

# Opdracht

Schrijf een korte uitleg voor {person_name} in eenvoudig Nederlands (B1-niveau). Gebruik ALLEEN wat hierboven staat.

WAT JE MOET DOEN:
- Begin direct met de uitkomst uit "## Uitkomst" (recht of geen recht)
- Bespreek elke voorwaarde: voor [JA] beschrijf je dat de burger eraan voldoet, voor [NEE] dat dit de reden is voor afwijzing, voor "gegevens ontbreken" dat er onvoldoende informatie was
- Noem het berekende bedrag als dat in "## Berekend bedrag" staat, in de vorm "X euro per jaar"
- Praat in de u-vorm
- Gebruik het woord "euro", nooit het €-teken
- {ending}

WAT JE NIET MAG DOEN:
- Verzin NOOIT gegevens (bedragen, inkomens, leeftijden) die niet in het skeleton staan
- Noem een [JA]-voorwaarde NOOIT als reden voor afwijzing
- Voeg GEEN informatie toe uit andere regelingen
- Bij samengestelde voorwaarden (bijv. "Voldoet aan landelijke voorwaarden"): noem alleen het resultaat, verzin NOOIT de onderliggende details"""


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
    # Requirements met — directly from engine, never overridden
    # ------------------------------------------------------------------

    @property
    def effective_requirements_met(self) -> bool | None:
        """Return the engine's requirements_met directly.

        The engine is authoritative: if it says True, the citizen meets the conditions.
        The amount calculated by actions is separate and may be 0 (e.g. income over
        threshold) — that does not change whether the requirements were met.
        """
        return self.calc_result.get("requirements_met") if self.calc_result else None

    # ------------------------------------------------------------------
    # Metadata helpers (from law YAML)
    # ------------------------------------------------------------------

    def _build_input_meta(self) -> dict[str, dict]:
        """Build {field_name: {description, unit, type, legal_basis}} from law YAML input + sources."""
        meta: dict[str, dict] = {}
        props = self.law.get("properties", {})
        # Both "input" and "sources" sections contain field definitions
        for section in ("input", "sources"):
            for inp in props.get(section, []):
                name = inp.get("name", "")
                if name:
                    entry: dict = {
                        "description": inp.get("description", name),
                        "unit": (inp.get("type_spec") or {}).get("unit", ""),
                        "type": inp.get("type", ""),
                    }
                    if inp.get("legal_basis"):
                        entry["legal_basis"] = inp["legal_basis"]
                    meta[name] = entry
        return meta

    def _build_output_meta(self) -> dict[str, dict]:
        """Build {field_name: {description, unit, type, citizen_relevance, period}} from law YAML output."""
        _period_nl = {"month": "per maand", "year": "per jaar", "week": "per week", "day": "per dag"}
        meta: dict[str, dict] = {}
        for out in self.law.get("properties", {}).get("output", []):
            name = out.get("name", "")
            if name:
                period_type = (out.get("temporal") or {}).get("period_type", "")
                meta[name] = {
                    "description": out.get("description", name),
                    "unit": (out.get("type_spec") or {}).get("unit", ""),
                    "type": out.get("type", ""),
                    "citizen_relevance": out.get("citizen_relevance", "secondary"),
                    "period": _period_nl.get(period_type, ""),
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

        Only fields explicitly declared as inputs in the law YAML are kept.
        Descriptions are abbreviation-expanded for readability.
        """
        input_data = self.calc_result.get("input_data", {})

        # Normalize input_data keys: strip leading "$"
        normalized: dict[str, Any] = {}
        for key, value in input_data.items():
            field_name = key.lstrip("$") if isinstance(key, str) else key
            normalized[field_name] = value

        values: dict[str, dict] = {}
        # Include ALL fields declared in input+sources (engine may return None for some)
        for field_name, meta in self._input_meta.items():
            value = normalized.get(field_name)
            values[field_name] = {
                "value": value,
                "unit": meta.get("unit", ""),
                "description": _expand_abbreviations(meta.get("description", field_name)),
                "source": "engine",
            }
        # Also add any extra fields the engine returned that aren't in the YAML metadata
        for field_name, value in normalized.items():
            if field_name not in values and field_name not in ("BSN",):
                values[field_name] = {
                    "value": value,
                    "unit": "",
                    "description": _expand_abbreviations(field_name.replace("_", " ").capitalize()),
                    "source": "engine",
                }
        return values

    # ------------------------------------------------------------------
    # Value resolution
    # ------------------------------------------------------------------

    def _get_value(self, ref: Any) -> Any:
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
            return f"{_to_dutch_format(value / 100)} euro"
        if isinstance(value, bool):
            return "Ja" if value else "Nee"
        if isinstance(value, float):
            if 0 < value < 1:
                return f"{value * 100:.3f}%"
            if value == int(value):  # whole number like 66.0 → "66"
                return str(int(value))
            return f"{value:,.2f}".replace(",", ".")  # keep plain floats readable
        if isinstance(value, int) and value > 10000:
            return f"{_to_dutch_format(value / 100)} euro"
        if isinstance(value, list):
            return ", ".join(
                _humanize_code_value(str(v)) if isinstance(v, str) else str(v)
                for v in value
            )
        if isinstance(value, str):
            return _humanize_code_value(value)
        return str(value)

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_expression(self, expr: Any) -> Any:
        """Recursively evaluate an expression node (e.g. {operation: ADD, values: [...]})."""
        if isinstance(expr, dict):
            op = expr.get("operation", "")
            vals = expr.get("values", [])
            resolved = [self._evaluate_expression(v) for v in vals]
            if any(v is None for v in resolved):
                return None
            try:
                if op == "ADD":
                    return sum(resolved)
                if op == "SUBTRACT":
                    return resolved[0] - resolved[1] if len(resolved) >= 2 else None
                if op == "MULTIPLY":
                    result = 1
                    for v in resolved:
                        result *= v
                    return result
                if op == "DIVIDE":
                    return resolved[0] / resolved[1] if len(resolved) >= 2 and resolved[1] != 0 else None
            except (TypeError, ValueError):
                return None
            return None
        return self._get_value(expr)

    def _expression_to_human(self, expr: Any) -> str:
        """Convert an expression node to a readable Dutch label fragment.

        Descriptions are truncated to 4 words to keep expression labels short.
        """
        if isinstance(expr, dict):
            op = expr.get("operation", "")
            vals = expr.get("values", [])
            parts = [self._expression_to_human(v) for v in vals]
            op_sym = {"ADD": "+", "SUBTRACT": "-", "MULTIPLY": "×", "DIVIDE": "÷"}.get(op, op)
            return f" {op_sym} ".join(parts)
        if isinstance(expr, str) and expr.startswith("$"):
            name = expr[1:]
            meta = self._input_meta.get(name, {})
            desc = _expand_abbreviations(meta.get("description", name))
            return desc
        return str(expr)

    def _evaluate_expression_condition(self, cond: dict) -> tuple[str, bool | None]:
        """Evaluate a condition with no subject (expression on both sides)."""
        operation = cond.get("operation", "")
        vals = cond.get("values", [])
        if len(vals) < 2:
            return self.STATUS_NOT_APPLICABLE, None
        lhs = self._evaluate_expression(vals[0])
        rhs = self._evaluate_expression(vals[1])
        if lhs is None:
            return self.STATUS_NOT_APPLICABLE, None
        try:
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
                return self.STATUS_NOT_APPLICABLE, None
            result = fn(lhs, rhs)
        except (TypeError, ValueError):
            return self.STATUS_NOT_APPLICABLE, None
        return (self.STATUS_SATISFIED if result else self.STATUS_FAILED), result

    def _expression_condition_to_human(self, cond: dict) -> str:
        """Return a readable Dutch label for an expression condition (no subject)."""
        operation = cond.get("operation", "")
        vals = cond.get("values", [])
        op_nl = {
            "GREATER_OR_EQUAL": "moet minimaal",
            "GREATER_THAN":     "moet meer dan",
            "LESS_OR_EQUAL":    "mag maximaal",
            "LESS_THAN":        "moet minder dan",
            "EQUALS":           "moet gelijk zijn aan",
            "NOT_EQUALS":       "mag niet gelijk zijn aan",
        }
        op_text = op_nl.get(operation, operation)
        lhs_label = self._expression_to_human(vals[0]) if len(vals) > 0 else "?"
        rhs_label = self._expression_to_human(vals[1]) if len(vals) > 1 else "?"
        return f"{lhs_label} {op_text} {rhs_label}"

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
        # Handle both "value" and "values" keys
        raw_value = cond.get("value") if "value" in cond else cond.get("values")
        resolved = self._get_value(raw_value)

        field_name = subject.lstrip("$") if isinstance(subject, str) else subject
        # Handle dot-notation sub-fields (e.g. "ARBEIDSVERMOGEN.arbeidsvermogen")
        parent_field = field_name.split(".")[0] if "." in field_name else field_name
        meta = self._input_meta.get(field_name) or self._input_meta.get(parent_field, {})
        if "." in field_name and meta:
            sub = field_name.split(".", 1)[1]
            # Try sub-field description first; fall back to parent description (no code in parentheses)
            sub_meta = self._input_meta.get(sub, {})
            if sub_meta.get("description"):
                label = _expand_abbreviations(sub_meta["description"])
            else:
                label = _expand_abbreviations(meta.get("description", parent_field))
        else:
            label = _expand_abbreviations(meta.get("description", field_name))
        # Strip parenthetical enum lists / field codes — not useful for citizens.
        # Matches: (UPPER_CASE_CODE), (snake_case_code), (lowercase_identifier)
        label = re.sub(r"\s*\([^)]*[A-Z_]{3,}[^)]*\)", "", label).strip()   # UPPERCASE codes
        label = re.sub(r"\s*\([^)]*[a-z][a-z0-9]*_[a-z0-9_]+[^)]*\)", "", label).strip()  # snake_case
        label = re.sub(r"\s*\([a-z][a-z0-9]+\)", "", label).strip()          # single lowercase identifier
        unit = meta.get("unit", "")

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
        # For boolean EQUALS/NOT_EQUALS the label already implies the yes/no meaning;
        # appending "moet gelijk zijn aan Ja" is redundant and confuses the LLM.
        _bool_true  = {True, "Ja", "ja", "true", "True", 1}
        _bool_false = {False, "Nee", "nee", "false", "False", 0}
        if operation == "EQUALS" and resolved in _bool_true:
            return label
        if operation == "EQUALS" and resolved in _bool_false:
            return f"{label}: Nee"
        if operation == "NOT_EQUALS" and resolved in _bool_true:
            return f"{label}: mag niet Ja zijn"
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
                # Handle both "value" and "values" keys in conditions
                raw_val = cond.get("value") if "value" in cond else cond.get("values")
                status, _ = self._evaluate_condition(
                    cond["subject"], cond["operation"], raw_val
                )
                nodes.append({
                    "id": node_id,
                    "label": self._condition_to_human(cond),
                    "status": self.STATUS_NOT_APPLICABLE if status is None
                              else (self.STATUS_SATISFIED if status else self.STATUS_FAILED),
                    "is_or_group": False,
                    "subject": cond.get("subject", "").lstrip("$"),
                    "operation": cond.get("operation", ""),
                    "value": raw_val,
                })

            elif "any" in cond or "or" in cond:
                or_items = cond.get("any") or cond.get("or") or []
                sub = self._process_conditions(or_items, f"{node_id}_any")
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
                nodes.extend(self._process_conditions(cond["all"], f"{node_id}_all"))

            elif "operation" in cond and "values" in cond:
                # Expression condition: no subject, e.g. ADD(X, Y) > 0
                status, _ = self._evaluate_expression_condition(cond)
                nodes.append({
                    "id": node_id,
                    "label": self._expression_condition_to_human(cond),
                    "status": status,
                    "is_or_group": False,
                    "subject": "",
                    "operation": cond.get("operation", ""),
                    "value": None,
                })

        return nodes

    def _collect_action_threshold_conditions(self) -> list[dict]:
        """
        Extract threshold-check conditions from the actions section.

        Action conditions like "$INKOMEN GREATER_THAN $DREMPELINKOMEN → 0" represent
        financial disqualification tests. We invert them into citizen-facing rules:
        e.g. "Toetsingsinkomen mag maximaal drempelinkomen zijn" (SATISFIED if not exceeded).

        These are NOT in requirements but are just as decisive for the amount outcome.
        """
        _op_inverse = {
            "GREATER_THAN":     "LESS_OR_EQUAL",
            "GREATER_OR_EQUAL": "LESS_THAN",
            "LESS_THAN":        "GREATER_OR_EQUAL",
            "LESS_OR_EQUAL":    "GREATER_THAN",
        }
        results: list[dict] = []
        seen: set[tuple] = set()

        def _traverse(node: Any, depth: int = 0) -> None:
            if depth > 15 or not isinstance(node, (dict, list)):
                return
            if isinstance(node, list):
                for item in node:
                    _traverse(item, depth)
                return
            # IF-conditions block
            for cond in node.get("conditions", []):
                if not isinstance(cond, dict):
                    continue
                test = cond.get("test")
                then = cond.get("then")
                # Only threshold tests that yield a zero/false disqualification
                if test and isinstance(test, dict) and "subject" in test and then in (0, False):
                    subj = test.get("subject", "")
                    op = test.get("operation", "")
                    val = test.get("value")
                    key = (subj, op, str(val))
                    if key not in seen and op in _op_inverse:
                        seen.add(key)
                        op_inv = _op_inverse[op]
                        actual, _ = self._evaluate_condition(subj, op_inv, val)
                        cond_dict = {"subject": subj, "operation": op_inv, "value": val}
                        results.append({
                            "id": f"action_rule_{len(results)}",
                            "label": self._condition_to_human(cond_dict),
                            "status": self.STATUS_NOT_APPLICABLE if actual is None
                                      else (self.STATUS_SATISFIED if actual else self.STATUS_FAILED),
                            "is_or_group": False,
                            "subject": subj.lstrip("$") if isinstance(subj, str) else subj,
                            "operation": op_inv,
                            "value": val,
                        })
                # Recurse into nested then/else/conditions
                for sub_key in ("then", "else"):
                    child = cond.get(sub_key)
                    if isinstance(child, dict):
                        _traverse(child, depth + 1)
            # Recurse into other nested dicts/lists (excluding leaf fields)
            _skip = {"subject", "operation", "value", "legal_basis", "explanation",
                     "output", "bwb_id", "article", "paragraph", "url", "juriconnect"}
            for k, v in node.items():
                if k not in _skip and isinstance(v, (dict, list)):
                    _traverse(v, depth + 1)

        for action in self.law.get("actions", []):
            _traverse(action)
        return results

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

        # Build decision label — engine's requirements_met is authoritative
        if requirements_met and primary_value is not None:
            if primary_unit == "eurocent" or primary_type == "amount":
                amount = primary_value / 100 if isinstance(primary_value, (int, float)) else primary_value
                if amount == 0:
                    decision_label = f"RECHT OP {law_name.upper()}, MAAR BEDRAG IS 0 EURO"
                else:
                    decision_label = f"RECHT: {_to_dutch_format(amount)} euro"
            elif primary_type == "boolean":
                # requirements_met is authoritative — don't contradict it with a False boolean
                if requirements_met:
                    decision_label = f"RECHT OP {law_name.upper()}"
                else:
                    decision_label = f"GEEN RECHT OP {law_name.upper()}"
            else:
                decision_label = f"BESLISSING: {primary_value}"
        elif requirements_met:
            decision_label = f"RECHT OP {law_name.upper()}"
        elif requirements_met is False:
            decision_label = f"GEEN RECHT OP {law_name.upper()}"
        else:
            decision_label = f"GEEN BESLISSING MOGELIJK: {law_name.upper()}"

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

        # Collect rule infos: action threshold conditions first (financially decisive),
        # then eligibility requirements. This ordering ensures that the income/vermogen
        # threshold check surfaces as decisive_condition in graphrag_context.py.
        action_infos = self._collect_action_threshold_conditions()
        rule_infos = action_infos + self._collect_rule_infos()

        # RULE nodes
        for rule_info in rule_infos:
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

        # FACT nodes — one per profile value, connected person→fact and fact→relevant rules
        for field_name, info in self.profile_values.items():
            value = info.get("value")
            if value is None:
                continue
            unit = info.get("unit", "")
            description = info.get("description", field_name)
            fact_id = f"fact_{field_name}"
            fact_node = GraphNode(
                id=fact_id,
                type="FACT",
                label=f"{description}: {self._format_value(value, unit)}",
                properties={
                    "field": field_name,
                    "value": value,
                    "unit": unit,
                    "description": description,
                },
            )
            self.graph.add_node(fact_node)
            self.graph.add_edge(GraphEdge(
                source=f"person_{self.bsn}",
                target=fact_id,
                relation="HAS_FACT",
            ))
            # Connect fact to rules that use this field as subject
            for rule_info in rule_infos:
                if rule_info.get("subject") == field_name:
                    self.graph.add_edge(GraphEdge(
                        source=fact_id,
                        target=rule_info["id"],
                        relation="USED_IN",
                    ))

        # THRESHOLD nodes — definitions used in condition comparisons
        seen_thresholds: set[str] = set()
        for rule_info in rule_infos:
            raw_val = rule_info.get("value")
            if not isinstance(raw_val, str) or not raw_val.startswith("$"):
                continue
            def_name = raw_val[1:]
            if def_name not in self.definitions or def_name in seen_thresholds:
                continue
            seen_thresholds.add(def_name)
            def_value = self.definitions[def_name]
            if not isinstance(def_value, (int, float)):
                continue
            threshold_id = f"threshold_{def_name}"
            label = def_name.replace("_", " ").capitalize()
            formatted = self._format_value(def_value, "eurocent" if def_value > 100 else "")
            threshold_node = GraphNode(
                id=threshold_id,
                type="THRESHOLD",
                label=f"{label}: {formatted}",
                properties={"definition": def_name, "value": def_value},
            )
            self.graph.add_node(threshold_node)
            self.graph.add_edge(GraphEdge(
                source=threshold_id,
                target=rule_info["id"],
                relation="DEFINES_THRESHOLD",
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

    def to_evaluation_trace(self) -> dict:
        """
        Return a machine-readable, normalised trace for the evaluation framework.

        All amounts are in euro (float), not eurocent strings.
        Conditions are structured dicts, not formatted labels.
        Purely additive — existing extract pipelines are unaffected.
        """
        calc = self.calc_result or {}
        output = calc.get("result", {})
        requirements_met = calc.get("requirements_met")

        # Primary amount (eurocent → euro)
        amount_euro: float | None = None
        primary_field = ""
        for out in self.law.get("properties", {}).get("output", []):
            if out.get("citizen_relevance") == "primary":
                primary_field = out.get("name", "")
                break
        if not primary_field:
            outputs = self.law.get("properties", {}).get("output", [])
            primary_field = outputs[0].get("name", "") if outputs else ""
        if primary_field:
            raw = output.get(primary_field)
            meta = self._output_meta.get(primary_field, {})
            if isinstance(raw, (int, float)) and (meta.get("unit") == "eurocent" or meta.get("type") == "amount"):
                amount_euro = round(raw / 100, 2)

        # Key facts — normalised numeric values where possible
        key_facts: dict[str, dict] = {}
        for field_name, info in self.profile_values.items():
            value = info.get("value")
            if value is None:
                continue
            unit = info.get("unit", "")
            label = info.get("description", field_name)
            entry: dict = {"label": label, "raw": value}
            if unit == "eurocent" and isinstance(value, (int, float)) or isinstance(value, (int, float)) and value > 10000 and unit == "":
                entry["value_euro"] = round(value / 100, 2)
            else:
                entry["value"] = value
            # Add legal_basis from input meta if available
            lb = self._input_meta.get(field_name, {}).get("legal_basis")
            if lb:
                entry["legal_basis"] = lb
            key_facts[field_name] = entry

        # Conditions from graph nodes
        rule_nodes = [n for n in self.graph.nodes if n.type == "RULE"]

        def _cond(node: "GraphNode") -> dict:
            props = node.properties
            return {
                "label": node.label,
                "status": props.get("status"),
                "subject": props.get("subject", ""),
                "operation": props.get("operation", ""),
                "is_action_rule": node.id.startswith("action_rule_"),
            }

        satisfied = [_cond(n) for n in rule_nodes if n.properties.get("status") == self.STATUS_SATISFIED]
        failed    = [_cond(n) for n in rule_nodes if n.properties.get("status") == self.STATUS_FAILED]
        unknown   = [_cond(n) for n in rule_nodes if n.properties.get("status") == self.STATUS_NOT_APPLICABLE]

        # Decisive condition — action rules (financial thresholds) take priority
        if not requirements_met:
            decisive = ([c for c in failed if c["is_action_rule"]] or failed or [{}])[0]
        else:
            decisive = ([c for c in satisfied if c["is_action_rule"]] or satisfied or [{}])[0]

        return {
            "outcome": "RECHT" if requirements_met else "GEEN_RECHT" if requirements_met is False else "ONBEKEND",
            "requirements_met": requirements_met,
            "amount_euro": amount_euro,
            "decisive_condition": {
                "label": decisive.get("label", ""),
                "subject": decisive.get("subject", ""),
                "operation": decisive.get("operation", ""),
                "is_action_rule": decisive.get("is_action_rule", False),
            },
            "key_facts": key_facts,
            "satisfied_conditions": [c["label"] for c in satisfied],
            "failed_conditions":    [c["label"] for c in failed],
            "unknown_conditions":   [c["label"] for c in unknown],
            "legal_basis": self.law.get("legal_basis"),
            "references": self.law.get("references", []),
        }

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

        # Group RULE nodes by status
        rule_nodes = [n for n in self.graph.nodes if n.type == "RULE"]
        satisfied = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_SATISFIED]
        failed    = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_FAILED]
        unknown   = [n for n in rule_nodes if n.properties.get("status") == self.STATUS_NOT_APPLICABLE]

        # Collect condition subjects so we only show citizen-relevant profile data
        # (filters out normative constants like LANDELIJK_BASISBEDRAG)
        condition_subjects: set[str] = {
            n.properties.get("subject", "")
            for n in rule_nodes
            if n.properties.get("subject")
        }

        # Personal situation — only fields used directly in conditions
        profile_lines: list[str] = []
        for field_name, info in self.profile_values.items():
            if field_name not in condition_subjects:
                continue
            value = info.get("value")
            if value is None:
                continue
            unit = info.get("unit", "")
            description = info.get("description", field_name)
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

        # Output amounts
        has_zero_amount = False
        if requirements_met:
            amount_lines: list[str] = []
            zero_amount_fields: list[str] = []
            for field_name, value in output.items():
                meta = self._output_meta.get(field_name, {})
                if meta.get("unit") != "eurocent" and meta.get("type") != "amount":
                    continue
                unit = meta.get("unit", "")
                description = meta.get("description", field_name)
                period = meta.get("period", "")
                period_suffix = f" {period}" if period else ""
                if value is None or value == 0:
                    zero_amount_fields.append(description)
                else:
                    amount_lines.append(f"- {description}: {self._format_value(value, unit)}{period_suffix}")
            if amount_lines:
                lines.append("## Berekend bedrag:")
                lines.extend(amount_lines)
                lines.append("")
            elif zero_amount_fields:
                # req_met=True but all amounts are 0 — include actual income/threshold values
                has_zero_amount = True
                lines.append("## Berekend bedrag:")
                for desc in zero_amount_fields:
                    lines.append(f"- {desc}: 0,00 euro")
                # Find relevant income/threshold FACT nodes to show concrete values
                income_keywords = {"inkomen", "drempel", "vermogen", "toetsing", "grens"}
                income_facts = [
                    n for n in self.graph.nodes
                    if n.type == "FACT"
                    and any(kw in n.properties.get("description", "").lower() for kw in income_keywords)
                ]
                if income_facts:
                    lines.append("- Reden: Uw inkomen overschrijdt de grens voor deze regeling:")
                    for fn in income_facts:
                        lines.append(f"  - {fn.label}")
                else:
                    lines.append("- Reden: Uw inkomen of vermogen overschrijdt de grens voor deze regeling.")
                lines.append("")

        # Conclusion — engine's requirements_met is authoritative
        lines.append("## Conclusie:")
        if requirements_met and has_zero_amount:
            lines.append(f"U voldoet aan de formele voorwaarden voor {law_name}, maar het berekende bedrag is 0 euro vanwege uw inkomen of vermogen.")
        elif requirements_met:
            lines.append(f"U voldoet aan de voorwaarden voor {law_name}.")
        elif requirements_met is False:
            lines.append(f"U heeft geen recht op {law_name}.")
        else:
            lines.append(f"Er zijn onvoldoende gegevens om een beslissing te nemen over {law_name}.")

        # Legal basis — articles from law YAML
        legal_basis = self.law.get("legal_basis")
        references = self.law.get("references", [])
        if legal_basis or references:
            lines.append("")
            lines.append("## Wettelijke grondslag:")
            if legal_basis:
                art = legal_basis.get("article", "")
                law_name_ref = legal_basis.get("law", "")
                if art and law_name_ref:
                    lines.append(f"- Artikel {art} {law_name_ref}")
            for ref in references:
                art = ref.get("article", "")
                law_name_ref = ref.get("law", "")
                desc = ref.get("description", "")
                if art and law_name_ref:
                    ref_line = f"- Artikel {art} {law_name_ref}"
                    if desc:
                        ref_line += f" ({desc})"
                    lines.append(ref_line)

        # More info
        contact = _get_service_contact(self.law)
        if contact:
            lines.append("")
            lines.append("## Meer informatie:")
            lines.append(contact)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM integration
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
    # Normalize unicode for LLM (keeps original skeleton intact for skeleton_used field)
    prompt = create_decision_prompt(_normalize_unicode_for_llm(skeleton), person_name)
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
        raw = response["message"]["content"].replace("\ufffd", "")
        explanation = _fix_rounded_amounts(raw, expected_values)
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

    if provider == "openai":
        import openai as _openai
        _oai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not _oai_key:
            raise ValueError("No OPENAI_API_KEY provided for gpt4 model")
        oai_client = _openai.OpenAI(api_key=_oai_key)
        oai_resp = oai_client.chat.completions.create(
            model=model_id,
            max_tokens=1000,
            temperature=0.2,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = oai_resp.choices[0].message.content or ""
        explanation = _fix_rounded_amounts(raw, expected_values)
        return {
            "explanation": explanation,
            "skeleton_used": skeleton,
            "prompt_used": prompt,
            "model": model_id,
            "provider": provider,
            "usage": {
                "input_tokens": oai_resp.usage.prompt_tokens if oai_resp.usage else 0,
                "output_tokens": oai_resp.usage.completion_tokens if oai_resp.usage else 0,
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
# Law loading + calculation (generic)
# ---------------------------------------------------------------------------

def _resolve_law_parameters(law: dict, profile: dict, bsn: str) -> dict:
    """Generically resolve required law parameters from the profile sources.

    Looks at the law YAML's `properties.parameters` list. For each required parameter
    (other than BSN), searches all profile source tables for a field with a matching name
    (case-insensitive). Returns the first match found per parameter.

    This handles cases like KVK_NUMMER for alcoholwet: the profile's KVK.inschrijvingen
    table has a `kvk_nummer` field that corresponds to the law parameter.
    """
    params: dict[str, Any] = {}
    sources = profile.get("sources", {})

    for param in law.get("properties", {}).get("parameters", []):
        pname = param.get("name", "")
        if not pname or pname.upper() in ("BSN",):
            continue  # BSN is always handled by the service
        pname_lower = pname.lower()

        # Search all source service tables for a matching field
        found = False
        for svc_data in sources.values():
            if not isinstance(svc_data, dict) or found:
                break
            for table_rows in svc_data.values():
                if not isinstance(table_rows, list) or found:
                    continue
                for row in table_rows:
                    if not isinstance(row, dict):
                        continue
                    for field, val in row.items():
                        if field.lower() == pname_lower and val is not None:
                            params[pname] = val
                            found = True
                            break
                    if found:
                        break

    return params


def run_calculation(law_name: str, bsn: str, law: dict | None = None, profile: dict | None = None) -> dict | None:
    """Run the law calculation via MCP service. Returns calc_result dict or None.

    If `law` and `profile` are provided, required law parameters (e.g. KVK_NUMMER) are
    auto-resolved from the profile sources so the engine can perform complete lookups.
    """
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

        # Auto-resolve required law parameters from the profile (e.g. KVK_NUMMER)
        params: dict = {}
        if law and profile:
            params = _resolve_law_parameters(law, profile, bsn)

        # For gemeente-specific laws, detect the right gemeente from the profile
        # so we don't always evaluate against GEMEENTE_AMSTERDAM (the first registered)
        sources = (profile or {}).get("sources", {})
        gemeente = next((k for k in sources if k.startswith("GEMEENTE_")), None)
        if gemeente and gemeente != service.service_type and service.service_type.startswith("GEMEENTE_"):
            from web.dependencies import TODAY
            params["BSN"] = bsn
            result = services.evaluate(
                service=gemeente,
                law=service.law_path,
                parameters=params,
                reference_date=TODAY,
                approved=True,
            )
            if result is None:
                return None
            return {
                "requirements_met": result.requirements_met,
                "missing_required": result.missing_required,
                "result": result.output or {},
                "input_data": result.input or {},
                "explanation": (
                    "U voldoet aan alle voorwaarden."
                    if result.requirements_met
                    else "U voldoet niet aan alle voorwaarden."
                ),
            }

        calc_result = service.execute(bsn, params)
        if "error" in calc_result:
            return None
        return calc_result
    except Exception as e:
        print(f"Warning: Could not run calculation for {law_name}/{bsn}: {e}", file=sys.stderr)
        return None


def _find_yaml_in_dir(base: Path, org_prefix: str | None = None) -> dict | None:
    """Find and load the most recent YAML for a law directory.

    If org_prefix is given (e.g. "GEMEENTE_AMSTERDAM"), looks for a matching
    file in gemeenten/ first.  Without a prefix, gemeenten/ is preferred over
    the national file so the skeleton matches the service used in the calculation.
    """
    if not base.is_dir():
        return None

    gemeenten_dir = base / "gemeenten"

    # 1. Org-specific gemeente YAML
    if org_prefix and gemeenten_dir.is_dir():
        org_yamls = sorted(gemeenten_dir.glob(f"{org_prefix}*.yaml"), reverse=True)
        if org_yamls:
            with open(org_yamls[0], encoding="utf-8") as f:
                return yaml.load(f, Loader=Loader)

    # 2. Any gemeente YAML — preferred over national file
    if gemeenten_dir.is_dir():
        gemeente_yamls = sorted(gemeenten_dir.glob("*.yaml"), reverse=True)
        if gemeente_yamls:
            with open(gemeente_yamls[0], encoding="utf-8") as f:
                return yaml.load(f, Loader=Loader)

    # 3. National/generic YAML in base directory
    yaml_files = sorted(base.glob("*.yaml"), reverse=True)
    if yaml_files:
        with open(yaml_files[0], encoding="utf-8") as f:
            return yaml.load(f, Loader=Loader)

    return None


def load_law_yaml(law_name: str) -> dict:
    """Load the most recent YAML file for a given law/service name.

    Resolution order:
    1. MCPServiceRegistry (preferred) — knows the exact service + org prefix
    2. Direct path: laws/<law_name>/  (with gemeente preference)
    3. Suffixed path: laws/<law_name>wet/
    """
    # 1. Preferred: ask MCPServiceRegistry so we get the right gemeente YAML
    try:
        from explain.mcp_connector import MCPLawConnector
        from web.dependencies import get_case_manager, get_claim_manager, get_machine_service

        services = get_machine_service()
        connector = MCPLawConnector(services, get_case_manager(), get_claim_manager())
        service = connector.registry.get_service(law_name)
        if service and hasattr(service, "law_path"):
            base = PROJECT_ROOT / "laws" / Path(service.law_path)
            # org_prefix comes from the service attribute (e.g. "GEMEENTE_AMSTERDAM")
            org_prefix: str | None = getattr(service, "service", None)
            result = _find_yaml_in_dir(base, org_prefix=org_prefix)
            if result is not None:
                return result
    except Exception:
        pass

    # 2. Direct path fallback
    for base in [PROJECT_ROOT / "laws" / law_name, PROJECT_ROOT / "laws" / f"{law_name}wet"]:
        result = _find_yaml_in_dir(base)
        if result is not None:
            return result

    raise FileNotFoundError(f"Law YAML not found: {law_name}")
