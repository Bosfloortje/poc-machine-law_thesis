#!/usr/bin/env python3
"""
Combined LLM explanation extraction script for machine law.

Supports two approaches selectable via --approach:
  graph  — Decision graph / skeleton approach (constrained, law-agnostic)
           The LLM receives a pre-filled skeleton derived from a focused decision
           subgraph. Hallucination is minimised; output is short and precise.
  open   — Open prompt approach (MCPLawConnector, law-agnostic)
           The LLM receives the raw calculation result and a free-form prompt.
           More verbose output, works for any law.
  both   — Runs both approaches and writes both files into a shared timestamped folder.

Original scripts are preserved in scripts/archive/:
  scripts/archive/extraction_graph.py       — archive of original graph approach
  scripts/archive/extract_explanations.py   — standalone open approach

Usage (run from project root):
    # Graph approach for all profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1

    # Open approach, specific law:
    uv run python analysis/llm_explanations/scripts/extract.py --approach open --model llama3.1 --laws zorgtoeslag

    # Both approaches, specific profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --model haiku --profiles 174760992

    # Multiple models at once (creates a shared folder):
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --models llama3.1 haiku mistral --profiles 174760992

    # Ollama models (no API key needed):
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model mistral
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
# scripts/extract.py → scripts/ → llm_explanations/ → analysis/ → root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling imports within scripts/

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Imports from the two original scripts
# ---------------------------------------------------------------------------
from extraction_law import (  # noqa: E402
    AVAILABLE_MODELS,
    DecisionGraphExtractor,
    generate_decision_explanation,
    get_git_info,
    load_law_yaml,
    load_profiles,
    run_calculation,
)
from extract_explanations import extract_explanations  # noqa: E402


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _run_label(model: str, law: str | None, profiles: list[str] | None) -> str:
    """Shared label used in filenames and folder names."""
    law_part = law or "all-laws"
    if not profiles:
        profile_part = "all-profiles"
    elif len(profiles) == 1:
        profile_part = profiles[0]
    else:
        profile_part = f"{len(profiles)}profiles"
    return f"{model}_{law_part}_{profile_part}"


def generate_output_filename(approach: str, model: str, law: str | None, profiles: list[str] | None) -> str:
    """Generate a timestamped output filename (used for single-approach runs)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{_run_label(model, law, profiles)}_{approach}.jsonl"
    return str(OUTPUT_DIR / filename)


def generate_both_output_dir(model: str, law: str | None, profiles: list[str] | None) -> Path:
    """Create and return a timestamped folder for single-model --approach both runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{_run_label(model, law, profiles)}_both"
    folder = OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def generate_multi_output_dir(approach: str, law: str | None, profiles: list[str] | None) -> Path:
    """Create and return a timestamped top-level folder for multi-model runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    law_part = law or "all-laws"
    profile_part = "all-profiles" if not profiles else (profiles[0] if len(profiles) == 1 else f"{len(profiles)}profiles")
    folder_name = f"{timestamp}_multi_{law_part}_{profile_part}_{approach}"
    folder = OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# Graph approach runner
# ---------------------------------------------------------------------------

def run_graph_approach(
    model: str,
    law: str,
    profiles_filter: list[str] | None,
    output_file: str,
    api_key: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> list[dict]:
    """Run the decision graph / skeleton approach for all (or selected) profiles.

    If resume=True and the output file already exists, already-processed profiles
    are skipped and new results are appended to the existing file.
    """
    all_profiles = load_profiles()
    profiles_to_process = profiles_filter or list(all_profiles.keys())

    model_config = AVAILABLE_MODELS[model]
    law_yaml = load_law_yaml(law)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume: find already-completed profiles ---
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
        if verbose:
            print(f"  Resuming: {len(already_done)} profiles already done, skipping them.", file=sys.stderr)
        profiles_to_process = [bsn for bsn in profiles_to_process if bsn not in already_done]

    total = len(profiles_to_process)
    file_mode = "a" if (resume and already_done) else "w"

    results: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    with open(output_path, file_mode, encoding="utf-8") as f:
        if file_mode == "w":
            metadata = {
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "model": model_config["id"],
                "provider": model_config.get("provider", "anthropic"),
                "law": law,
                "profiles_count": total,
                "graph_type": "decision",
                "approach": "graph",
                "git_info": get_git_info(),
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        for i, bsn in enumerate(profiles_to_process, 1):
            if bsn not in all_profiles:
                if verbose:
                    print(f"  [{i}/{total}] Warning: Profile {bsn} not found, skipping", file=sys.stderr)
                continue

            if verbose:
                print(f"  [{i}/{total}] Processing {bsn}...", file=sys.stderr)

            profile_data = all_profiles[bsn]
            person_name = profile_data.get("name", f"Burger {bsn}")

            calc_result = run_calculation(law, bsn)
            if calc_result and verbose:
                req_met = calc_result.get("requirements_met", False)
                print(f"    Calculation: requirements_met={req_met}", file=sys.stderr)

            decision_extractor = DecisionGraphExtractor(law_yaml, profile_data, bsn, calc_result)
            graph = decision_extractor.extract()

            try:
                result = generate_decision_explanation(
                    decision_extractor=decision_extractor,
                    person_name=person_name,
                    api_key=api_key,
                    model=model,
                )

                total_input_tokens += result["usage"]["input_tokens"]
                total_output_tokens += result["usage"]["output_tokens"]

                calc_output = calc_result.get("result", {}) if calc_result else {}
                profile_vals = decision_extractor.profile_values

                record = {
                    "record_type": "explanation",
                    "graph_type": "decision",
                    "approach": "graph",
                    "law": law,
                    "profile": bsn,
                    "profile_name": person_name,
                    "requirements_met": calc_result.get("requirements_met") if calc_result else None,
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

            except Exception as e:
                if verbose:
                    print(f"  Exception for {bsn}: {e}", file=sys.stderr)
                record = {
                    "record_type": "explanation",
                    "graph_type": "decision",
                    "approach": "graph",
                    "law": law,
                    "profile": bsn,
                    "profile_name": person_name,
                    "error": str(e),
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

    if verbose:
        print(f"\nCompleted graph approach! {len(results)} profiles.", file=sys.stderr)
        print(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output", file=sys.stderr)
        print(f"Output saved to: {output_path.absolute()}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined LLM explanation extraction (open prompt + decision graph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Approaches:
  graph  Constrained skeleton from decision subgraph. Short, precise output.
         Runs for a single --law (default: zorgtoeslag). Works for any law.
  open   Free-form prompt with raw calculation result. Verbose, law-agnostic.
         Use --laws to filter (default: all laws).
  both   Runs graph then open; saves both files inside a shared timestamped folder.

Multiple models:
  Pass --models m1 m2 ... to run several models in sequence.
  All output goes into a single timestamped folder with one subfolder per model.

Resuming an interrupted run:
  Pass --resume with the same --output (or same folder) to skip already-done profiles
  and append new results. Works for graph approach.

Examples:
    # Graph approach, all profiles, llama:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model llama3.1

    # Graph approach, specific profiles:
    uv run python analysis/llm_explanations/scripts/extract.py --approach graph --model haiku --profiles 174760992 311508199

    # Open approach, zorgtoeslag only:
    uv run python analysis/llm_explanations/scripts/extract.py --approach open --model llama3.1 --laws zorgtoeslag

    # Both approaches:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --model llama3.1 --profiles 174760992

    # Multiple models:
    uv run python analysis/llm_explanations/scripts/extract.py --approach both --models llama3.1 haiku mistral --profiles 174760992
""",
    )

    parser.add_argument(
        "--approach",
        choices=["open", "graph", "both"],
        default="graph",
        help="Extraction approach: graph (skeleton), open (free prompt), or both (default: graph)",
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=["haiku"],
        metavar="MODEL",
        dest="models",
        help="One or more models to run (default: haiku). Multiple models create a shared folder.",
    )
    parser.add_argument(
        "--law",
        default="zorgtoeslag",
        help="Law for the graph approach (default: zorgtoeslag)",
    )
    parser.add_argument(
        "--laws",
        nargs="+",
        help="Law(s) for the open approach (default: all). If a single law is given and --law is not set, it is also used for the graph approach.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Specific BSN(s) to process (default: all profiles)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (single model + single approach only).",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var). Not required for Ollama models.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run: skip already-processed profiles and append to existing output file(s).",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    # Determine which models to run
    models_to_run: list[str] = args.models
    multi_model = len(models_to_run) > 1

    # If a single --laws value was given without --law, treat it as the graph law too
    effective_law = args.law
    if args.laws and len(args.laws) == 1 and args.law == "zorgtoeslag":
        effective_law = args.laws[0]

    do_graph = args.approach in ("graph", "both")
    do_open = args.approach in ("open", "both")

    # For multi-model runs, create one top-level folder; each model gets a subfolder
    multi_dir: Path | None = None
    if multi_model:
        multi_dir = generate_multi_output_dir(args.approach, effective_law, args.profiles)
        print(f"Output folder (multi): {multi_dir}")

    for model in models_to_run:
        if multi_model:
            print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

        # Determine the base directory for this model's output
        if multi_dir is not None:
            model_dir = multi_dir / model
            model_dir.mkdir(exist_ok=True)
        elif args.approach == "both":
            model_dir = generate_both_output_dir(model, effective_law, args.profiles)
            print(f"Output folder (both): {model_dir}")
        else:
            model_dir = None  # flat file output

        # -------------------------------------------------------------------
        # Graph approach
        # -------------------------------------------------------------------
        if do_graph:
            if model_dir is not None:
                output_graph = str(model_dir / f"graph_{model}_{effective_law}.jsonl")
            elif args.output is not None:
                output_graph = args.output
            else:
                output_graph = generate_output_filename("graph", model, effective_law, args.profiles)
            print(f"Output file (graph): {output_graph}")
            run_graph_approach(
                model=model,
                law=effective_law,
                profiles_filter=args.profiles,
                output_file=output_graph,
                api_key=args.api_key,
                verbose=verbose,
                resume=args.resume,
            )

        # -------------------------------------------------------------------
        # Open approach
        # -------------------------------------------------------------------
        if do_open:
            if model_dir is not None:
                output_open = str(model_dir / f"open_{model}_{effective_law}.jsonl")
            elif args.output is not None:
                output_open = args.output
            else:
                output_open = generate_output_filename("open", model, effective_law, args.profiles)
            print(f"Output file (open): {output_open}")
            extract_explanations(
                api_key=args.api_key,
                laws_filter=args.laws or [effective_law],
                profiles_filter=args.profiles,
                output_file=output_open,
                model=model,
                verbose=verbose,
            )


if __name__ == "__main__":
    main()
