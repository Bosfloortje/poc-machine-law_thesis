#!/usr/bin/env python3
"""
Thesis benchmark run: 3 laws × 2 profiles × 5 models × 2 approaches (graph + open).

Laws and profiles (1 positive + 1 negative per law):
  zorgtoeslag:              312847291 (✓ laag inkomen), 591847362 (✗ inkomen boven grens)
  participatiewet/bijstand: 748291634 (✓ geen inkomen), 914827361 (✗ looninkomen)
  alcoholwet/vergunning:    263948172 (✓ café SVH-diploma), 481927364 (✗ 19jr + geen SVH)

Models: haiku, llama3.1, mistral, deepseek, gpt4

Output: analysis/llm_explanations/output/thesis_<timestamp>/
  <model>/
    graph_<model>_<law_slug>.jsonl
    open_<model>_<law_slug>.jsonl

Usage (run from project root):
    uv run python analysis/llm_explanations/scripts/thesis_run.py
    uv run python analysis/llm_explanations/scripts/thesis_run.py --models haiku gpt4
    uv run python analysis/llm_explanations/scripts/thesis_run.py --resume --output-dir analysis/llm_explanations/output/thesis_20260427_120000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent.parent / "output"
PROFILES_FILE = str(PROJECT_ROOT / "data" / "profiles.yaml")

# ---------------------------------------------------------------------------
# Thesis configuration
# ---------------------------------------------------------------------------

THESIS_LAWS: dict[str, dict] = {
    "zorgtoeslag": {
        "profiles": ["312847291", "591847362"],
        "labels": ["✓ laag inkomen verzekerd", "✗ inkomen boven grens"],
    },
    "participatiewet/bijstand": {
        "profiles": ["748291634", "914827361"],
        "labels": ["✓ geen inkomen arbeidsvermogen", "✗ looninkomen €2200/mnd"],
    },
    "alcoholwet/vergunning": {
        "profiles": ["263948172", "481927364"],
        "labels": ["✓ café SVH-diploma geen curatele", "✗ 19jr + geen SVH"],
    },
}

DEFAULT_MODELS = ["haiku", "llama3.1", "mistral", "deepseek", "gpt4"]

SYSTEM_PROMPT = (
    "Je bent een behulpzame assistent die Nederlandse burgers helpt met vragen over "
    "overheidsregelingen. Geef duidelijke, begrijpelijke uitleg in eenvoudig Nederlands (B1-niveau)."
)


# ---------------------------------------------------------------------------
# Custom open-approach precompute using the graph engine (not MCPLawConnector)
# This ensures the correct gemeente data is used for each profile.
# ---------------------------------------------------------------------------


def _make_open_prompt(law: str, calc_result: dict, profile: dict, bsn: str) -> str:
    requirements_met = calc_result.get("requirements_met", False)
    missing_required = calc_result.get("missing_required", False)
    output = calc_result.get("result", {})
    explanation = calc_result.get("explanation", "")
    name = profile.get("name", "Onbekend")
    description = profile.get("description", "Geen beschrijving")
    return (
        f"Ik heb zojuist een berekening uitgevoerd voor de regeling '{law}'.\n\n"
        f"Burgerprofiel:\n"
        f"- Naam: {name}\n"
        f"- Beschrijving: {description}\n"
        f"- BSN: {bsn}\n\n"
        f"Resultaat van de berekening:\n"
        f"- Voldoet aan voorwaarden: {'Ja' if requirements_met else 'Nee'}\n"
        f"- Ontbrekende essentiële gegevens: {'Ja' if missing_required else 'Nee'}\n"
        f"- Uitkomst: {json.dumps(output, indent=2, ensure_ascii=False)}\n\n"
        f"Korte uitleg van het systeem: {explanation}\n\n"
        f"Geef een duidelijke uitleg in eenvoudig Nederlands (B1-niveau) over:\n"
        f"1. WAAROM deze burger wel of niet in aanmerking komt voor deze regeling\n"
        f"2. Welke factoren uit het profiel van de burger hebben geleid tot dit resultaat\n"
        f"3. Wat de burger eventueel kan doen als ze niet in aanmerking komen\n\n"
        f"Let op: bedragen in de uitkomst zijn in eurocenten, deel door 100 voor euros."
    )


def _eval_via_engine(law: str, bsn: str, profile: dict) -> dict | None:
    """Evaluate a law directly via the machine engine, bypassing MCPLawConnector.

    Determines the right service (gemeente) from the profile's available data,
    then calls engine.evaluate with BSN + any extra parameters (e.g. KVK_NUMMER).
    """
    from web.dependencies import get_machine_service

    engine = get_machine_service()
    sources = profile.get("sources", profile)

    # Determine which service to use for gemeente-specific laws
    # (e.g. GEMEENTE_UTRECHT for bijstand, GEMEENTE_ROTTERDAM for alcoholwet)
    service: str | None = None

    # For gemeente laws: find the first GEMEENTE_X key in the profile's sources
    gemeente_keys = [k for k in sources if k.startswith("GEMEENTE_")]
    if gemeente_keys:
        service = gemeente_keys[0]

    # For laws without a gemeente service in the profile, derive from KVK vestigingsadres
    if service is None:
        for svc_name in ("KVK",):
            svc_data = sources.get(svc_name, {})
            if not isinstance(svc_data, dict):
                continue
            for rows in svc_data.values():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    addr = str(row.get("vestigingsadres", "")).strip()
                    if addr:
                        # Map city name to service key
                        _city_map = {
                            "rotterdam": "GEMEENTE_ROTTERDAM",
                            "amsterdam": "GEMEENTE_AMSTERDAM",
                            "den haag": "GEMEENTE_DEN_HAAG",
                            "eindhoven": "GEMEENTE_EINDHOVEN",
                            "groningen": "GEMEENTE_GRONINGEN",
                            "maastricht": "GEMEENTE_MAASTRICHT",
                            "utrecht": "GEMEENTE_UTRECHT",
                        }
                        for city, svc in _city_map.items():
                            if city in addr.lower():
                                service = svc
                                break
                    if service:
                        break
                if service:
                    break

    # Engine law name mapping: thesis key → actual engine law name + service
    _engine_law: str = law
    _law_service_overrides: dict[str, tuple[str, str]] = {
        "zorgtoeslag": ("TOESLAGEN", "zorgtoeslagwet"),
    }
    if law in _law_service_overrides:
        service, _engine_law = _law_service_overrides[law]
    elif service is None:
        service = "TOESLAGEN"

    # Build extra params (KVK_NUMMER for business laws)
    params: dict = {"BSN": bsn}
    for svc_name in ("KVK",):
        svc_data = sources.get(svc_name, {})
        if not isinstance(svc_data, dict):
            continue
        for rows in svc_data.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                kvk = row.get("kvk_nummer")
                if kvk:
                    params["KVK_NUMMER"] = str(kvk)
                    break
            if "KVK_NUMMER" in params:
                break

    try:
        from web.dependencies import TODAY

        result = engine.evaluate(
            service=service,
            law=_engine_law,
            parameters=params,
            reference_date=TODAY,
            approved=True,
        )
        if result is not None:
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
    except Exception as e:
        print(f"  Engine evaluate failed for {bsn}/{law}: {e}", file=sys.stderr)
    return None


def thesis_precompute_open(
    law: str,
    profiles_filter: list[str] | None = None,
    verbose: bool = True,
    cache_file: Path | None = None,
) -> list[dict]:
    """Compute open-approach entries by evaluating directly via the machine engine.

    If cache_file exists, loads calc_results from it instead of calling the engine.
    profiles_filter=None processes all profiles from profiles.yaml.
    """
    from extraction_generic import load_profiles

    all_profiles = load_profiles(PROFILES_FILE)
    bsns = profiles_filter if profiles_filter is not None else list(all_profiles.keys())

    # Load existing cache if available (avoids re-running engine on resume)
    cached: dict[str, dict] = {}
    if cache_file and cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        if verbose:
            print(f"  Open precompute: loaded {len(cached)} cached results from {cache_file.name}", file=sys.stderr)

    entries: list[dict] = []
    total = len(bsns)

    for i, bsn in enumerate(bsns, 1):
        if bsn not in all_profiles:
            if verbose:
                print(f"  [{i}/{total}] Warning: profile {bsn} not found, skipping", file=sys.stderr)
            continue

        profile = all_profiles[bsn]
        name = profile.get("name", bsn)

        if bsn in cached:
            calc_result = cached[bsn]
        else:
            if verbose:
                print(f"  [{i}/{total}] Open precompute: {bsn} ({name}) / {law}...", file=sys.stderr)
            calc_result = _eval_via_engine(law, bsn, profile)
            if cache_file and calc_result:
                cached[bsn] = calc_result
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False)

        entry: dict = {
            "bsn": bsn,
            "profile": profile,
            "profile_name": name,
            "law_name": law,
            "calc_result": calc_result,
            "requirements_met": calc_result.get("requirements_met") if calc_result else None,
            "calculation_result": {
                "requirements_met": calc_result.get("requirements_met"),
                "missing_required": calc_result.get("missing_required"),
                "output": calc_result.get("result", {}),
                "input_data": calc_result.get("input_data", {}),
                "system_explanation": calc_result.get("explanation", ""),
            }
            if calc_result
            else None,
            "prompt": _make_open_prompt(law, calc_result, profile, bsn) if calc_result else None,
            "error": None if calc_result else f"run_calculation returned None for {bsn}/{law}",
        }
        entries.append(entry)

    return entries


def run_open_approach(
    model: str,
    law: str,
    precomputed: list[dict],
    output_file: str,
    verbose: bool = True,
    resume: bool = False,
) -> list[dict]:
    """Write open-approach JSONL for a single law + model."""
    from extract import _call_llm
    from extraction_generic import AVAILABLE_MODELS, get_git_info

    from web.dependencies import TODAY

    model_info = AVAILABLE_MODELS[model]
    model_id = model_info["id"]
    provider = model_info["provider"]
    api_key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else None

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-done profiles
    already_done: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("record_type") == "explanation" and "profile" in r:
                        already_done.add(r["profile"])
                except json.JSONDecodeError:
                    pass
        if verbose and already_done:
            print(f"  Resuming open: {len(already_done)} already done", file=sys.stderr)

    entries = [e for e in precomputed if e["bsn"] not in already_done]
    file_mode = "a" if (resume and already_done) else "w"
    results: list[dict] = []

    with open(output_path, file_mode, encoding="utf-8") as f:
        if file_mode == "w":
            f.write(
                json.dumps(
                    {
                        "record_type": "metadata",
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id,
                        "provider": provider,
                        "law": law,
                        "profiles_count": len(entries),
                        "approach": "open_prompt",
                        "git_info": get_git_info(),
                        "reference_date": TODAY,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        for entry in entries:
            bsn = entry["bsn"]
            record: dict = {
                "record_type": "explanation",
                "approach": "open",
                "graph_type": None,
                "law": law,
                "profile": bsn,
                "profile_name": entry["profile_name"],
                "requirements_met": entry["requirements_met"],
                "explanation": None,
                "skeleton_used": None,
                "prompt_used": entry["prompt"],
                "model": model_id,
                "usage": None,
                "graph_stats": None,
                "calculation_result": entry["calculation_result"],
            }
            if entry.get("error"):
                record["error"] = entry["error"]
            elif entry.get("prompt"):
                if verbose:
                    print(f"    LLM open {bsn} ({model})...", file=sys.stderr)
                try:
                    text, usage = _call_llm(model_id, provider, SYSTEM_PROMPT, entry["prompt"], api_key)
                    record["explanation"] = text
                    record["usage"] = usage
                except Exception as e:
                    record["error"] = str(e)
            else:
                record["error"] = "No prompt (calculation failed)"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            if not record.get("explanation"):
                print(
                    f"\n⚠️  WARNING: geen explanation voor {bsn} ({entry['profile_name']}) [{law} / {model}]",
                    file=sys.stderr,
                )
                if record.get("error"):
                    print(f"   Fout: {record['error']}", file=sys.stderr)
                try:
                    input("   Druk Enter om door te gaan, of Ctrl+C om te stoppen... ")
                except KeyboardInterrupt:
                    print("\nAfgebroken door gebruiker.", file=sys.stderr)
                    return results

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from extract import precompute_graph_entries, run_graph_approach
    from extraction_generic import AVAILABLE_MODELS

    parser = argparse.ArgumentParser(
        description="Thesis benchmark: 3 laws × 2 profiles × 5 models × graph+open",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(AVAILABLE_MODELS.keys()), default=DEFAULT_MODELS, metavar="MODEL"
    )
    parser.add_argument(
        "--laws", nargs="+", choices=list(THESIS_LAWS.keys()), default=list(THESIS_LAWS.keys()), metavar="LAW"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"thesis_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nThesis run output: {run_dir}")
    print(f"Laws   : {', '.join(args.laws)}")
    print(f"Models : {', '.join(args.models)}")

    laws_to_run = {k: v for k, v in THESIS_LAWS.items() if k in args.laws}

    # -------------------------------------------------------------------
    # Precompute calculations ONCE per law (reused across all models)
    # -------------------------------------------------------------------
    graph_precomputed: dict[str, list[dict]] = {}
    open_precomputed: dict[str, list[dict]] = {}

    for law, cfg in laws_to_run.items():
        law_slug = law.replace("/", "_")
        cache_path = run_dir / f"cache_{law_slug}.json"
        print(f"\n{'─' * 60}")
        print(f"Precomputing: {law}  (all profiles)")

        # Open precompute: loads from cache if available (fast on resume), else calls engine
        open_precomputed[law] = thesis_precompute_open(
            law=law,
            profiles_filter=None,
            verbose=verbose,
            cache_file=cache_path,
        )

        # Seed graph cache with open precomputed results (shares the same cache file)
        if not cache_path.exists():
            seed_cache = {e["bsn"]: e["calc_result"] for e in open_precomputed[law] if e.get("calc_result") is not None}
            if seed_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as _cf:
                    json.dump(seed_cache, _cf, ensure_ascii=False)

        # Graph precompute — loads from same cache, skips re-evaluation
        graph_precomputed[law] = precompute_graph_entries(
            law=law,
            profiles_filter=None,
            verbose=verbose,
            cache_file=cache_path,
            profiles_file=PROFILES_FILE,
        )

    # -------------------------------------------------------------------
    # LLM pass: iterate models, then laws
    # -------------------------------------------------------------------
    for model in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")

        model_dir = run_dir / model
        model_dir.mkdir(exist_ok=True)

        for law, cfg in laws_to_run.items():
            profiles = cfg["profiles"]
            law_slug = law.replace("/", "_")

            # Graph approach
            graph_out = str(model_dir / f"graph_{model}_{law_slug}.jsonl")
            print(f"\n  Graph → {Path(graph_out).name}")
            run_graph_approach(
                model=model,
                law=law,
                profiles_filter=profiles,
                output_file=graph_out,
                verbose=verbose,
                resume=args.resume,
                precomputed=graph_precomputed[law],
            )

            # Open approach
            open_out = str(model_dir / f"open_{model}_{law_slug}.jsonl")
            print(f"  Open  → {Path(open_out).name}")
            run_open_approach(
                model=model,
                law=law,
                precomputed=open_precomputed[law],
                output_file=open_out,
                verbose=verbose,
                resume=args.resume,
            )

    print(f"\n{'=' * 60}")
    print(f"Done. All output in: {run_dir.absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
