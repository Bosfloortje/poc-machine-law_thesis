"""
GraphRAG context builder for the chat interface.

Bridges the DecisionGraphExtractor (analysis scripts) into the web chat,
so the LLM receives a structured knowledge graph instead of raw service output.

Imports are lazy so a missing dependency never crashes the web server.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent.parent / "analysis" / "llm_explanations" / "scripts"

# Placeholder — resolved on first call
GRAPHRAG_SYSTEM_PROMPT: str = ""


def _ensure_imports() -> tuple:
    """Lazy-load heavy analysis deps. Returns (DecisionGraphExtractor, serialize_graph, system_prompt)."""
    global GRAPHRAG_SYSTEM_PROMPT

    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))

    from extraction_generic import DecisionGraphExtractor  # noqa: E402
    from extract_graphrag import GRAPHRAG_SYSTEM_PROMPT as _SP, serialize_graph  # noqa: E402

    GRAPHRAG_SYSTEM_PROMPT = _SP
    return DecisionGraphExtractor, serialize_graph, _SP


def build_graph_context(
    service_results: dict,
    services,
    registry,
    profile: dict,
    bsn: str,
    person_name: str,
    reference_date: str,
    return_traces: bool = False,
) -> "str | tuple[str, dict]":
    """
    For each executed service result, build a knowledge graph via DecisionGraphExtractor
    and serialize it as JSON. Returns a combined context string for the LLM.

    Falls back to empty string if graph building fails for any reason.

    Args:
        return_traces: If True, returns (context_str, rac_traces) instead of just context_str.
                       rac_traces maps service_name → {decisive_condition, law_name, info_url}.
                       Default False keeps the original single-return behaviour (extract scripts unaffected).
    """
    try:
        DecisionGraphExtractor, serialize_graph, _ = _ensure_imports()
        from extraction_generic import _get_service_url  # noqa: E402
    except Exception as e:
        logger.warning("GraphRAG imports failed, falling back to raw context: %s", e)
        return ("", {}) if return_traces else ""

    graph_blocks: list[str] = []
    rac_traces: dict = {}

    for service_name, result in service_results.items():
        try:
            service_obj = registry.get_service(service_name)
            if not service_obj:
                continue

            law = services.get_rule_spec(service_obj.law_path, reference_date, service_obj.service_type)
            if not law:
                continue

            calc_result = {
                "requirements_met": result.get("requirements_met"),
                "result": result.get("result", {}),
                "input_data": result.get("input_data", {}),
            }

            extractor = DecisionGraphExtractor(law=law, profile=profile, bsn=bsn, calc_result=calc_result)
            graph = extractor.extract()
            graph_json = serialize_graph(graph, extractor, person_name)

            # Extract rac_trace for this service
            requirements_met = graph_json["beslissing"]["requirements_voldaan"]
            candidates = graph_json["voorwaarden"]["niet_voldaan" if not requirements_met else "voldaan"]
            decisive_condition = candidates[0]["beschrijving"] if candidates else ""
            info_url = _get_service_url(law) or "www.rijksoverheid.nl"
            law_name = graph_json.get("regeling", service_name)
            rac_traces[service_name] = {
                "decisive_condition": decisive_condition,
                "law_name": law_name,
                "info_url": info_url,
            }

            # Append mandatory citation instructions to the graph block
            graph_str = json.dumps(graph_json, ensure_ascii=False, indent=2)
            instructions = (
                f"\nINSTRUCTIE voor uitleg:\n"
                f"- Noem altijd: {info_url}\n"
                f"- Noem de doorslaggevende voorwaarde letterlijk: \"{decisive_condition}\"\n"
                f"- Voeg een zin toe: \"Als [voorwaarde anders], dan zou u...\"\n"
            )
            graph_blocks.append(graph_str + instructions)

        except Exception as e:
            logger.warning("GraphRAG context failed for %s: %s", service_name, e)
            continue

    if not graph_blocks:
        return ("", {}) if return_traces else ""

    context_str = "Beslissingsgraph(en):\n\n```json\n" + "\n\n---\n\n".join(graph_blocks) + "\n```"

    if return_traces:
        return context_str, rac_traces
    return context_str


def build_graph_html(
    service_results: dict,
    services,
    registry,
    profile: dict,
    bsn: str,
    person_name: str,
    reference_date: str,
) -> str:
    """
    Build an HTML card for each executed service result, showing the decision graph.
    Returns a combined HTML string suitable for embedding in the chat interface.
    Returns empty string if graph building fails.
    """
    try:
        DecisionGraphExtractor, serialize_graph, _ = _ensure_imports()
    except Exception as e:
        logger.warning("GraphRAG HTML build failed: %s", e)
        return ""

    html_blocks: list[str] = []

    for service_name, result in service_results.items():
        try:
            service_obj = registry.get_service(service_name)
            if not service_obj:
                continue

            law = services.get_rule_spec(service_obj.law_path, reference_date, service_obj.service_type)
            if not law:
                continue

            calc_result = {
                "requirements_met": result.get("requirements_met"),
                "result": result.get("result", {}),
                "input_data": result.get("input_data", {}),
            }

            extractor = DecisionGraphExtractor(law=law, profile=profile, bsn=bsn, calc_result=calc_result)
            graph = extractor.extract()
            graph_json = serialize_graph(graph, extractor, person_name)

            html_blocks.append(_graph_json_to_html(graph_json, bsn=bsn, service_name=service_name))

        except Exception as e:
            logger.warning("GraphRAG HTML failed for %s: %s", service_name, e)
            continue

    if not html_blocks:
        return ""

    return "\n".join(html_blocks)


def _graph_json_to_html(graph_json: dict, bsn: str = "", service_name: str = "") -> str:
    """Convert a serialised graph dict to a collapsible HTML card."""
    import html as html_lib

    law_name = html_lib.escape(graph_json.get("regeling", "Regeling"))
    beslissing = graph_json.get("beslissing", {})
    outcome = html_lib.escape(str(beslissing.get("uitkomst", "")))
    req_met = beslissing.get("requirements_voldaan")

    # Header colour
    header_cls = "bg-green-100 text-green-800" if req_met else "bg-red-100 text-red-800"
    outcome_icon = "✓" if req_met else "✗"

    # Conditions
    voorwaarden = graph_json.get("voorwaarden", {})
    cond_rows = ""
    for cond in voorwaarden.get("voldaan", []):
        desc = html_lib.escape(cond.get("beschrijving", ""))
        val = html_lib.escape(str(cond.get("profielwaarde") or ""))
        val_cell = f'<span class="text-gray-500 ml-1">({val})</span>' if val else ""
        cond_rows += (
            f'<tr class="border-t border-gray-100">'
            f'<td class="py-1 pr-2 text-green-600 font-bold">✓</td>'
            f'<td class="py-1 text-gray-800">{desc}{val_cell}</td></tr>'
        )
    for cond in voorwaarden.get("niet_voldaan", []):
        desc = html_lib.escape(cond.get("beschrijving", ""))
        val = html_lib.escape(str(cond.get("profielwaarde") or ""))
        val_cell = f'<span class="text-gray-500 ml-1">({val})</span>' if val else ""
        cond_rows += (
            f'<tr class="border-t border-gray-100">'
            f'<td class="py-1 pr-2 text-red-600 font-bold">✗</td>'
            f'<td class="py-1 text-gray-800">{desc}{val_cell}</td></tr>'
        )
    for cond in voorwaarden.get("onbekend_gegevens_ontbreken", []):
        desc = html_lib.escape(cond.get("beschrijving", ""))
        cond_rows += (
            f'<tr class="border-t border-gray-100">'
            f'<td class="py-1 pr-2 text-gray-400">?</td>'
            f'<td class="py-1 text-gray-500 italic">{desc}</td></tr>'
        )

    # Facts used
    feiten = graph_json.get("feiten_gebruikt", {})
    fact_rows = ""
    for key, val in feiten.items():
        k = html_lib.escape(str(key))
        v = html_lib.escape(str(val))
        fact_rows += (
            f'<tr class="border-t border-gray-100">'
            f'<td class="py-1 pr-3 text-gray-500 whitespace-nowrap">{k}</td>'
            f'<td class="py-1 font-medium text-gray-800">{v}</td></tr>'
        )

    # Amount
    bedrag = graph_json.get("berekend_bedrag") or {}
    amount_html = ""
    if bedrag:
        amounts = "; ".join(
            f'<strong>{html_lib.escape(str(v))}</strong> ({html_lib.escape(str(k))})'
            for k, v in bedrag.items()
        )
        amount_html = f'<p class="mt-2 text-sm text-gray-700">Berekend bedrag: {amounts}</p>'

    conditions_section = (
        f'<table class="w-full text-sm mt-2"><tbody>{cond_rows}</tbody></table>'
        if cond_rows else ""
    )
    facts_section = (
        f'<p class="text-xs font-semibold text-gray-500 uppercase mt-3 mb-1">Gebruikte gegevens</p>'
        f'<table class="w-full text-sm"><tbody>{fact_rows}</tbody></table>'
        if fact_rows else ""
    )

    graph_btn = ""
    if bsn and service_name:
        graph_url = f"/chat/graph/{html_lib.escape(bsn)}/{html_lib.escape(service_name)}"
        graph_btn = (
            f'<div class="mt-3">'
            f'<a href="{graph_url}" target="_blank" rel="noopener noreferrer" '
            f'class="inline-flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 '
            f'border border-indigo-200 rounded px-2 py-1 hover:bg-indigo-50">'
            f'<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" '
            f'viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">'
            f'<path stroke-linecap="round" stroke-linejoin="round" '
            f'd="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>'
            f'</svg>Toon grafiek</a></div>'
        )

    return (
        f'<details class="mt-2 rounded-lg border border-gray-200 overflow-hidden text-sm">'
        f'<summary class="cursor-pointer px-3 py-2 {header_cls} font-semibold select-none">'
        f'{outcome_icon} Beslissingsgraph: {law_name}</summary>'
        f'<div class="px-3 pb-3">'
        f'<p class="mt-2 font-semibold text-gray-800">{outcome}</p>'
        f'{conditions_section}'
        f'{facts_section}'
        f'{amount_html}'
        f'{graph_btn}'
        f'</div></details>'
    )


__all__ = ["build_graph_context", "build_graph_html", "GRAPHRAG_SYSTEM_PROMPT"]
