#!/usr/bin/env python3
"""
Generate two visualizations:
1. Law structure graph — Zorgtoeslagwet as itself (no profile data)
2. Profile decision graph — Bram de Groot (BSN 326889935, gets zorgtoeslag)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

try:
    from yaml import CLoader as Loader
except ImportError:
    pass

from extraction_generic import DecisionGraphExtractor, load_law_yaml, load_profiles, run_calculation

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "graph_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_COLORS = {
    "LAW":         "#4A90D9",
    "DECISION":    "#2ECC71",
    "RULE":        "#E74C3C",
    "FACT":        "#3498DB",
    "INPUT":       "#27AE60",
    "OUTPUT":      "#F39C12",
    "THRESHOLD":   "#9B59B6",
    "CALCULATION": "#F39C12",
}


def draw_graph(G: nx.DiGraph, title: str, output_path: Path, figsize=(22, 16)) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    node_types = [G.nodes[n].get("node_type", "") for n in G.nodes()]
    colors = [NODE_COLORS.get(t, "#CCCCCC") for t in node_types]
    sizes = [3500 if t in ("DECISION", "LAW") else 2200 for t in node_types]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.92)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#555555", arrows=True, arrowsize=18,
        alpha=0.65, connectionstyle="arc3,rad=0.08",
        min_source_margin=20, min_target_margin=20,
    )

    labels = {}
    for n in G.nodes():
        lbl = G.nodes[n].get("label", n)
        words = lbl.split()
        lines, cur = [], []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) > 20:
                lines.append(" ".join(cur[:-1]))
                cur = [w]
        lines.append(" ".join(cur))
        labels[n] = "\n".join(lines)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=7.5, font_weight="bold")

    edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, font_color="#333333")

    present_types = set(node_types)
    legend = [
        mpatches.Patch(facecolor=NODE_COLORS.get(t, "#CCCCCC"), label=t)
        for t in NODE_COLORS if t in present_types
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}", file=sys.stderr)
    print(f"Saved: {pdf_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. LAW STRUCTURE GRAPH (no profile)
# ---------------------------------------------------------------------------

def build_law_structure_graph(law_yaml: dict) -> nx.DiGraph:
    """Build an abstract graph of the law's structure from YAML alone."""
    G = nx.DiGraph()
    law_name = law_yaml.get("name", "Wet")

    # Translations for YAML-derived Dutch labels
    LABEL_EN = {
        "Zorgtoeslag": "Health Care Benefit Law (Zorgtoeslag)",
        "Geeft aan of iemand recht heeft op zorgtoeslag": "Entitled to health care benefit",
        "De hoogte van de zorgtoeslag": "Amount of healthcare allowance",
        "Leeftijd van de aanvrager": "Age of applicant",
        "Aanvrager is verzekerde in de zin van de zvw": "Applicant is insured (ZVW)",
        "Toetsingsinkomen": "Assessed income",
        "Vermogen": "Assets",
        "Gezamenlijk vermogen": "Joint assets",
        "Partnerinkomen": "Partner income",
        "Partnerschap status": "Partnership status",
        "Standaardpremie": "Standard premium",
        "Alle voorwaarden": "All conditions",
        "Een van de voorwaarden": "One of the conditions",
    }

    def _en(label: str) -> str:
        return LABEL_EN.get(label, label)

    law_name = _en(law_yaml.get("name", "Law"))

    # Law node
    G.add_node("law", label=law_name, node_type="LAW")

    # Output nodes
    for out in law_yaml.get("properties", {}).get("output", []):
        nid = f"out_{out['name']}"
        G.add_node(nid, label=_en(out.get("description", out["name"])), node_type="OUTPUT")
        G.add_edge("law", nid, relation="PRODUCES")

    # Input nodes
    for inp in law_yaml.get("properties", {}).get("input", []):
        nid = f"in_{inp['name']}"
        svc = inp.get("service_reference", {}).get("service", "")
        label = _en(inp.get("description", inp["name"]))
        if svc:
            label = f"{label}\n[{svc}]"
        G.add_node(nid, label=label, node_type="INPUT")

    # Requirement nodes + edges
    def process_requirements(reqs, parent_id=None):
        for i, req in enumerate(reqs):
            if "all" in req:
                rid = f"req_all_{i}"
                G.add_node(rid, label="All conditions", node_type="RULE")
                if parent_id:
                    G.add_edge(parent_id, rid, relation="CONTAINS")
                else:
                    for out in law_yaml.get("properties", {}).get("output", []):
                        G.add_edge(rid, f"out_{out['name']}", relation="DETERMINES")
                process_requirements(req["all"], rid)
            elif "any" in req:
                rid = f"req_any_{i}"
                G.add_node(rid, label="One of the conditions", node_type="RULE")
                if parent_id:
                    G.add_edge(parent_id, rid, relation="CONTAINS")
                process_requirements(req["any"], rid)
            elif "subject" in req:
                subj = req["subject"].lstrip("$")
                val = req.get("value", "")
                if isinstance(val, str):
                    val = val.lstrip("$")
                op_map = {
                    "GREATER_OR_EQUAL": "≥", "GREATER_THAN": ">",
                    "LESS_OR_EQUAL": "≤", "LESS_THAN": "<", "EQUALS": "=",
                }
                op = op_map.get(req.get("operation", ""), req.get("operation", ""))
                rid = f"rule_{subj}_{i}"
                label = f"{subj} {op} {val}"
                G.add_node(rid, label=label, node_type="RULE")
                if parent_id:
                    G.add_edge(parent_id, rid, relation="CONTAINS")
                if f"in_{subj}" in G.nodes:
                    G.add_edge(f"in_{subj}", rid, relation="USED_IN")

    reqs = law_yaml.get("requirements", [])
    if reqs:
        process_requirements(reqs)

    # Action nodes: connect inputs to outputs through conditions
    for inp in law_yaml.get("properties", {}).get("input", []):
        nid = f"in_{inp['name']}"
        for out in law_yaml.get("properties", {}).get("output", []):
            out_nid = f"out_{out['name']}"
            if not G.has_edge(nid, out_nid):
                G.add_edge(nid, out_nid, relation="INFLUENCES")

    return G


# ---------------------------------------------------------------------------
# 2. PROFILE DECISION GRAPH (uses existing extractor)
# ---------------------------------------------------------------------------

PROFILE_LABEL_EN = {
    # Decision outcomes
    "RECHT": "ENTITLED",
    "GEEN RECHT": "NOT ENTITLED",
    "ONBEKEND": "UNKNOWN",
    # Condition descriptions
    "Toetsingsinkomen mag maximaal 39.719,00 euro": "Assessed income ≤ €39,719",
    "Vermogen mag maximaal 141.896,00 euro": "Assets ≤ €141,896",
    "Gezamenlijk vermogen mag maximaal 179.429,00 euro": "Joint assets ≤ €179,429",
    "Leeftijd van de aanvrager moet minimaal 18": "Age ≥ 18 years",
    "Aanvrager is verzekerde in de zin van de Zorgverzekeringswet": "Insured under Health Insurance Act",
    # Fact labels
    "Leeftijd van de aanvrager": "Age of applicant",
    "Aanvrager is verzekerde in de zin van de Zorgverzekeringswet": "Insured (ZVW)",
    "Toetsingsinkomen": "Assessed income",
    "Vermogen": "Assets",
    "Gezamenlijk vermogen": "Joint assets",
    "Partnerschap status": "Partnership status",
    "Standaardpremie": "Standard premium",
    "Partnerinkomen": "Partner income",
    "Drempelinkomen alleenstaande": "Income threshold (single)",
    "Drempelinkomen toeslagpartner": "Income threshold (partner)",
    "Vermogensgrens alleenstaande": "Asset threshold (single)",
    "Vermogensgrens toeslagpartner": "Asset threshold (partner)",
    "Percentage drempelinkomen alleenstaand": "Income threshold % (single)",
    "Percentage drempelinkomen met partner": "Income threshold % (partner)",
    "Minimum leeftijd": "Minimum age",
    "Prev january first": "Reference date",
    "De hoogte van de zorgtoeslag": "Healthcare allowance amount",
    "Geeft aan of iemand recht heeft op zorgtoeslag": "Entitled to health care benefit",
}


def _translate_node_labels(G: nx.DiGraph) -> nx.DiGraph:
    """Translate Dutch node labels to English in-place."""
    for node, data in G.nodes(data=True):
        lbl = data.get("label", "")
        # Direct lookup
        if lbl in PROFILE_LABEL_EN:
            data["label"] = PROFILE_LABEL_EN[lbl]
            continue
        # Partial match for "KEY: value" fact nodes (e.g. "Leeftijd van de aanvrager: 68")
        for nl, en in PROFILE_LABEL_EN.items():
            if lbl.startswith(nl + ":"):
                data["label"] = en + lbl[len(nl):]
                break
        # Translate RECHT/GEEN RECHT in decision node labels
        for nl, en in PROFILE_LABEL_EN.items():
            if nl in lbl and nl in ("RECHT", "GEEN RECHT", "ONBEKEND"):
                data["label"] = lbl.replace(nl, en)
    return G


def build_profile_graph(bsn: str, law: str) -> nx.DiGraph:
    all_profiles = load_profiles()
    profile_data = all_profiles[bsn]
    law_yaml = load_law_yaml(law)
    calc_result = run_calculation(law, bsn, law_yaml, profile_data)
    extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
    kg = extractor.extract()
    G = kg.to_networkx()
    return _translate_node_labels(G)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building law structure graph...", file=sys.stderr)
    law_yaml = load_law_yaml("zorgtoeslag")
    G_law = build_law_structure_graph(law_yaml)
    draw_graph(
        G_law,
        title="Health Care Benefit Law (Zorgtoeslagwet) — Law Structure (no profile)",
        output_path=OUTPUT_DIR / "zorgtoeslagwet_law_structure.png",
        figsize=(24, 16),
    )

    print("Building profile decision graph (Bram de Groot, BSN 326889935)...", file=sys.stderr)
    G_profile = build_profile_graph("326889935", "zorgtoeslag")
    draw_graph(
        G_profile,
        title="Health Care Benefit Law (Zorgtoeslagwet) — Decision Graph: Bram de Groot (€1,630.66/year)",
        output_path=OUTPUT_DIR / "zorgtoeslagwet_bram_de_groot.png",
        figsize=(22, 14),
    )

    print("Done.", file=sys.stderr)
