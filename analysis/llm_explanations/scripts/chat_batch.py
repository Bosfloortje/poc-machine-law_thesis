#!/usr/bin/env python3
"""
Automated batch chat runner for machine-law.

Simulates multi-turn conversations with the chat LLM for a set of profiles,
using a predefined conversation script per law. Records all turns to JSONL.

Unlike extract.py (which calls the engine directly), this goes through the
full WebSocket chat — so the LLM, MCP connector, guard and graph context
are all active, exactly as a real user would experience.

Usage:
    # All profiles in default profiles.yaml, zorgtoeslag, claude:
    uv run python analysis/llm_explanations/scripts/chat_batch.py --law zorgtoeslag

    # Custom profiles file, specific law, specific provider:
    uv run python analysis/llm_explanations/scripts/chat_batch.py \\
        --law bijstand \\
        --profiles-file data/profiles_500_chat_20260402.yaml \\
        --provider haiku \\
        --limit 10

    # Specific BSNs only:
    uv run python analysis/llm_explanations/scripts/chat_batch.py \\
        --law zorgtoeslag --bsn 403987006 174760992

Requires the web server:
    $env:FEATURE_CHAT='1'; uv run web/main.py
"""

import argparse
import asyncio
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import websockets
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Conversation scripts per law
# Ordered: opening question first, then follow-ups.
# The LLM's answer to turn N is in context when turn N+1 is sent.
# ---------------------------------------------------------------------------

SCRIPTS: dict[str, list[str]] = {
    "zorgtoeslag": [
        "Kom ik in aanmerking voor zorgtoeslag?",
        "Wat als mijn jaarinkomen €20.000 zou zijn? Heb ik dan recht op zorgtoeslag?",
        "Wat als mijn jaarinkomen €60.000 zou zijn? Heb ik dan recht op zorgtoeslag?",
    ],
    "kinderbijslag": [
        "Heb ik recht op kinderbijslag?",
        "Waarom wel of niet, en welke voorwaarden gelden?",
        "Hoeveel kinderbijslag ontvang ik per kwartaal?",
    ],
    "kinderopvangtoeslag": [
        "Kom ik in aanmerking voor kinderopvangtoeslag?",
        "Hoeveel kinderopvangtoeslag kan ik verwachten?",
        "Wat zijn de belangrijkste voorwaarden die voor mij gelden?",
    ],
    "huurtoeslag": [
        "Heb ik recht op huurtoeslag?",
        "Welke factoren bepalen de hoogte van mijn huurtoeslag?",
        "Hoeveel huurtoeslag zou ik maandelijks ontvangen?",
        "Wat kan ik doen als ik geen recht heb?",
    ],
    "bijstand": [
        "Heb ik recht op bijstand?",
        "Wat als ik een deeltijdbaan heb met een inkomen van €500 per maand? Heb ik dan nog recht op bijstand?",
        "Wat als ik €20.000 spaargeld heb? Heb ik dan nog recht op bijstand?",
    ],
    "kindgebonden_budget": [
        "Heb ik recht op kindgebonden budget?",
        "Hoeveel kindgebonden budget ontvang ik?",
        "Welke factoren bepalen de hoogte?",
    ],
    "werkloosheidswet": [
        "Heb ik recht op een WW-uitkering?",
        "Wat zijn de voorwaarden en voldoe ik eraan?",
        "Hoeveel WW zou ik ontvangen en voor hoe lang?",
    ],
    "alcoholwet": [
        "Kom ik in aanmerking voor een vergunning op basis van de alcoholwet?",
        "Wat als mijn bedrijf niet actief staat ingeschreven bij de KVK? Krijg ik dan nog een vergunning?",
        "Wat als ik geen geldige SVH-registratie voor sociale hygiëne heb? Kom ik dan nog in aanmerking?",
    ],
    "kieswet": [
        "Heb ik stemrecht op basis van de kieswet?",
        "Waarom wel of niet? Wat zijn de voorwaarden?",
        "Wat zijn mijn opties als ik geen stemrecht heb?",
    ],
}

# Generic fallback for laws without a specific script
_GENERIC_SCRIPT = [
    "Kom ik in aanmerking voor {law}?",
    "Waarom wel of niet? Leg de belangrijkste voorwaarden uit.",
    "Wat krijg ik als ik in aanmerking kom — een bedrag, een vergunning of een ander recht?",
    "Wat zijn mijn opties als ik niet in aanmerking kom?",
]


_OUTCOME_SIGNALS_POS = [
    "recht op", "heeft recht", "komt in aanmerking", "kunt u aanspraak",
    "toegekend", "u ontvangt", "u krijgt", "vergunning verleend",
    "in aanmerking", "wel recht",
]
_OUTCOME_SIGNALS_NEG = [
    "geen recht", "niet in aanmerking", "afgewezen", "geen vergunning",
    "voldoet niet", "niet gehonoreerd", "helaas niet", "komt u niet",
]


def _check_cf_used(question: str, answer: str) -> dict:
    """
    Check whether the LLM actually used the changed input from the counterfactual question.

    Returns a dict with:
        cf_value_in_response: bool  — the hypothetical value from the question appears in the answer
        cf_has_outcome:       bool  — the answer contains a clear yes/no outcome signal
        cf_outcome_positive:  bool | None — True=positive, False=negative, None=unclear
        cf_used_input:        bool  — best-effort overall: value present AND outcome stated
    """
    answer_lower = answer.lower()

    # Extract euro amounts from the question (e.g. "€20.000", "€ 20.000", "€500")
    euro_amounts = re.findall(r"€\s?[\d.,]+", question)
    # Also extract plain numbers followed by context words
    plain_amounts = re.findall(r"\b(\d[\d.,]*)\s*(euro|per maand|spaargeld|inkomen|jaarinkomen)", question, re.I)

    value_found = False
    for amt in euro_amounts:
        # Normalise: strip €, spaces, try both . and , as thousands separator
        digits = re.sub(r"[€\s]", "", amt)
        variants = {digits, digits.replace(".", ""), digits.replace(",", ""), digits.replace(".", ","), digits.replace(",", ".")}
        if any(v in answer_lower for v in variants):
            value_found = True
            break
    if not value_found:
        for digits, _ in plain_amounts:
            variants = {digits, digits.replace(".", ""), digits.replace(",", "")}
            if any(v in answer_lower for v in variants):
                value_found = True
                break

    # Boolean conditions in question (KVK, SVH) — check if response addresses them
    boolean_keywords = re.findall(r"\b(kvk|svh|ingeschreven|registratie|sociale hygiëne)\b", question, re.I)
    for kw in boolean_keywords:
        if kw.lower() in answer_lower:
            value_found = True
            break

    pos = any(s in answer_lower for s in _OUTCOME_SIGNALS_POS)
    neg = any(s in answer_lower for s in _OUTCOME_SIGNALS_NEG)
    has_outcome = pos or neg
    outcome_positive = True if (pos and not neg) else (False if (neg and not pos) else None)

    return {
        "cf_value_in_response": value_found,
        "cf_has_outcome": has_outcome,
        "cf_outcome_positive": outcome_positive,
        "cf_used_input": value_found and has_outcome,
    }


def get_script(law: str) -> list[str]:
    if law in SCRIPTS:
        return SCRIPTS[law]
    return [q.format(law=law) for q in _GENERIC_SCRIPT]


# ---------------------------------------------------------------------------
# Profiles loader
# ---------------------------------------------------------------------------

def load_profiles(profiles_file: str) -> dict[str, dict]:
    with open(profiles_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {})


def load_done_bsns(output_path: Path) -> set[str]:
    """Return the set of BSNs that already have a completed conversation record."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") == "conversation" and rec.get("bsn"):
                done.add(str(rec["bsn"]))
    return done


def save_graph(law: str, bsn: str, profile: dict, graphs_dir: Path) -> None:
    """Save the decision graph PNG for a single profile + law."""
    try:
        from extraction_generic import DecisionGraphExtractor, load_law_yaml, run_calculation
    except ImportError as e:
        print(f"  [graph] Kan grafiek niet opslaan (dependencies ontbreken): {e}", file=sys.stderr)
        return

    try:
        law_yaml = load_law_yaml(law)
        if not law_yaml:
            print(f"  [graph] Law YAML niet gevonden voor {law}", file=sys.stderr)
            return

        calc_result = run_calculation(law, bsn, law_yaml, profile)
        if not calc_result:
            print(f"  [graph] Berekening mislukt voor {bsn}", file=sys.stderr)
            return

        extractor = DecisionGraphExtractor(law=law_yaml, profile=profile, bsn=bsn, calc_result=calc_result)
        graph = extractor.extract()

        graphs_dir.mkdir(parents=True, exist_ok=True)
        out_path = graphs_dir / f"{law}_{bsn}.png"
        person_name = profile.get("name", bsn)
        law_name = law_yaml.get("name", law)
        graph.visualize(output_path=str(out_path), title=f"{law_name} — {person_name}")
        print(f"  [graph] Opgeslagen: {out_path.name}")
    except Exception as e:
        print(f"  [graph] Fout bij opslaan grafiek voor {bsn}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Single-profile conversation
# ---------------------------------------------------------------------------

async def run_conversation(
    bsn: str,
    profile_name: str,
    law: str,
    provider: str,
    script: list[str],
    base_url: str,
    graph: bool,
    no_guard: bool,
    timeout: float,
    verbose: bool,
) -> dict:
    """
    Connects to the WebSocket, sends each message in the script, collects
    all assistant turns. Returns a conversation record dict.
    """
    client_id = f"batch_{bsn}_{uuid.uuid4().hex[:8]}"
    uri = f"{base_url}/chat/ws/{client_id}"
    turns: list[dict] = []
    error: str | None = None

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=180) as ws:
            # Handshake
            await ws.send(json.dumps({
                "bsn": bsn,
                "provider": provider,
                "graph": graph,
                "no_guard": no_guard,
            }))
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw)

            if data.get("error"):
                return {"error": data["error"], "turns": []}
            if data.get("feature_disabled"):
                return {"error": "Chat feature disabled — start server with FEATURE_CHAT=1", "turns": []}

            model = data.get("model", "unknown")

            if verbose:
                print(f"  [{bsn}] {profile_name} — model: {model}")

            for i, message in enumerate(script, 1):
                if verbose:
                    print(f"    turn {i}/{len(script)}: {message[:60]}...")

                turns.append({"role": "user", "message": message})
                assistant_parts: list[str] = []
                current_contestability: dict | None = None

                await ws.send(json.dumps({"message": message}))

                # Collect all messages for this turn (processing + final + chained)
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        msg_data = json.loads(raw)

                        if msg_data.get("error"):
                            error = msg_data["error"]
                            break

                        if msg_data.get("isProcessing"):
                            continue

                        if msg_data.get("applicationPanel") or msg_data.get("graphPanel"):
                            # Skip UI panels in batch mode
                            continue

                        text = msg_data.get("message", "")
                        if text:
                            assistant_parts.append(text)
                            guard = msg_data.get("guard")
                            contestability = msg_data.get("contestability")
                            if contestability:
                                # Store latest contestability score for this turn
                                current_contestability = contestability
                            if verbose and guard:
                                status = "✓" if guard.get("valid") else "✗"
                                print(f"      guard: {status} {guard.get('explanation', '')[:60]}")
                            if verbose and contestability:
                                print(f"      contestability: {contestability.get('contestability_score', '?'):.2f}")

                        # Wait for chained follow-up messages (tool call may take time)
                        try:
                            follow_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                            follow_data = json.loads(follow_raw)
                            if follow_data.get("isProcessing"):
                                continue
                            if follow_data.get("applicationPanel") or follow_data.get("graphPanel"):
                                continue
                            follow_text = follow_data.get("message", "")
                            if follow_text:
                                assistant_parts.append(follow_text)
                        except TimeoutError:
                            break

                    except TimeoutError:
                        if not assistant_parts:
                            error = f"Timeout waiting for response on turn {i}"
                        break

                if assistant_parts:
                    full_answer = "\n\n".join(assistant_parts)
                    turn_record: dict = {
                        "role": "assistant",
                        "message": full_answer,
                    }
                    if current_contestability:
                        turn_record["contestability"] = current_contestability
                    # For counterfactual turns (2+), check if LLM used the changed input
                    if i > 1:
                        cf = _check_cf_used(message, full_answer)
                        turn_record["cf_check"] = cf
                        if verbose:
                            used = "Y" if cf["cf_used_input"] else "N"
                            val = "Y" if cf["cf_value_in_response"] else "N"
                            out = "Y" if cf["cf_has_outcome"] else "N"
                            print(f"      cf_used={used}  value_in_resp={val}  has_outcome={out}")
                    turns.append(turn_record)


                if error:
                    break

    except ConnectionRefusedError:
        error = "Connection refused — is the server running?"
    except websockets.exceptions.ConnectionClosedError as e:
        error = f"WebSocket closed unexpectedly (code {e.code})"
    except TimeoutError:
        error = "Timeout during WebSocket handshake"
    except Exception as e:
        error = str(e)

    return {
        "bsn": bsn,
        "profile_name": profile_name,
        "law": law,
        "provider": provider,
        "model": model if "model" in dir() else "unknown",
        "turns": turns,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    law: str,
    profiles: dict[str, dict],
    provider: str,
    output_path: Path,
    base_url: str,
    graph: bool,
    no_guard: bool,
    timeout: float,
    verbose: bool,
    limit: int | None,
    save_graphs: bool = False,
    resume: bool = False,
) -> None:
    script = get_script(law)
    bsns = list(profiles.keys())
    if limit:
        bsns = bsns[:limit]

    done_bsns: set[str] = set()
    if resume:
        done_bsns = load_done_bsns(output_path)
        remaining = [b for b in bsns if b not in done_bsns]
        print(f"\nResume: {len(done_bsns)} al gedaan, {len(remaining)} resterend -> {output_path.name}")
        bsns = remaining
    else:
        print(f"\nBatch chat: {len(bsns)} profiles x {len(script)} turns -> {output_path.name}")

    total = len(bsns)
    print(f"Law: {law} | Provider: {provider} | Graph: {graph} | Guard: {not no_guard}")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume else "w"

    with open(output_path, file_mode, encoding="utf-8") as f:
        if not resume:
            f.write(json.dumps({
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "law": law,
                "provider": provider,
                "graph": graph,
                "guard_enabled": not no_guard,
                "profiles_count": total,
                "script_turns": len(script),
                "script": script,
            }, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({
                "record_type": "metadata",
                "timestamp": datetime.now().isoformat(),
                "resumed": True,
                "skipped_bsns": len(done_bsns),
                "remaining": total,
            }, ensure_ascii=False) + "\n")

        for i, bsn in enumerate(bsns, 1):
            profile_data = profiles[bsn]
            profile_name = profile_data.get("name", bsn)

            print(f"[{i}/{total}] {bsn} — {profile_name}")

            record = await run_conversation(
                bsn=bsn,
                profile_name=profile_name,
                law=law,
                provider=provider,
                script=script,
                base_url=base_url,
                graph=graph,
                no_guard=no_guard,
                timeout=timeout,
                verbose=verbose,
            )
            record["record_type"] = "conversation"

            if record.get("error"):
                print(f"  [FOUT] {record['error']}")
            else:
                assistant_turns = [t for t in record["turns"] if t["role"] == "assistant"]
                # Conversation-level contestability: criterion passes if met in any turn
                all_c = [t.get("contestability", {}) for t in assistant_turns if t.get("contestability")]
                if all_c:
                    decisive = any(c.get("has_decisive_condition") for c in all_c)
                    counterfact = any(c.get("has_contestable_path") for c in all_c)
                    score = sum([decisive, counterfact]) / 2
                    print(f"  contestability: {score:.2f}  decisive={'Y' if decisive else 'N'}  counterfactual={'Y' if counterfact else 'N'}")
                elif verbose:
                    print(f"  {len(assistant_turns)} turns (no contestability data)")
                # Counterfactual input-use: did the LLM use the changed input in CF turns?
                cf_turns = [t for t in assistant_turns if t.get("cf_check")]
                if cf_turns:
                    cf_used_all = [t["cf_check"]["cf_used_input"] for t in cf_turns]
                    cf_used_pct = sum(cf_used_all) / len(cf_used_all)
                    record["cf_used_input_pct"] = round(cf_used_pct, 2)
                    print(f"  cf_used_input: {sum(cf_used_all)}/{len(cf_used_all)} turns ({cf_used_pct:.0%})")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if save_graphs:
                graphs_dir = output_path.parent / "graphs"
                save_graph(law, bsn, profile_data, graphs_dir)

    print(f"\nDone. Output: {output_path.absolute()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated batch chat runner — simulates multi-turn conversations per profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 10 profiles, zorgtoeslag, haiku:
    uv run python analysis/llm_explanations/scripts/chat_batch.py \\
        --law zorgtoeslag --provider haiku --limit 10

    # 500-profile file, bijstand, claude:
    uv run python analysis/llm_explanations/scripts/chat_batch.py \\
        --law bijstand \\
        --profiles-file data/profiles_500_chat_20260402.yaml \\
        --provider claude

    # Graph mode (full knowledge graph context):
    uv run python analysis/llm_explanations/scripts/chat_batch.py \\
        --law zorgtoeslag --provider claude --graph

Available scripts:
""" + "\n".join(f"  {k}: {len(v)} turns" for k, v in SCRIPTS.items()),
    )

    parser.add_argument("--law", required=True, help="Law to run (e.g. zorgtoeslag, bijstand)")
    parser.add_argument(
        "--profiles-file",
        default="data/profiles.yaml",
        help="Path to profiles YAML (default: data/profiles.yaml)",
    )
    parser.add_argument(
        "--provider",
        default="claude",
        choices=["claude", "haiku", "vlam", "gpt-4o", "gpt-4o-mini", "llama3.1", "llama3.2", "llama3.3", "mistral", "deepseek", "gemma2"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument("--bsn", nargs="+", help="Filter to specific BSN(s)")
    parser.add_argument("--limit", type=int, help="Max number of profiles to process")
    parser.add_argument("--host", default="localhost:8000", help="Server host:port (default: localhost:8000)")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: auto-timestamped)")
    parser.add_argument("--graph", action="store_true", help="Use graph knowledge graph context")
    parser.add_argument("--no-guard", action="store_true", dest="no_guard", help="Disable LLM guard")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds to wait per LLM response (default: 120)")
    parser.add_argument("--verbose", action="store_true", help="Show turn details and guard decisions")
    parser.add_argument("--save-graphs", action="store_true", dest="save_graphs",
                        help="Save decision graph PNG per profile to output/chat/graphs/ (requires networkx + matplotlib)")
    parser.add_argument("--resume", metavar="FILE",
                        help="Resume an interrupted batch run: append to FILE, skipping already-completed BSNs")

    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    profiles_path = project_root / args.profiles_file

    profiles = load_profiles(str(profiles_path))

    if args.bsn:
        profiles = {bsn: p for bsn, p in profiles.items() if bsn in args.bsn}

    if not profiles:
        print(f"[FOUT] Geen profielen gevonden in {profiles_path}", file=sys.stderr)
        sys.exit(1)

    # Output path
    output_dir = script_dir.parent / "output" / "chat"
    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            print(f"[FOUT] Resume-bestand niet gevonden: {output_path}", file=sys.stderr)
            sys.exit(1)
        resume = True
    elif args.output:
        output_path = Path(args.output)
        resume = False
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "graph" if args.graph else "chat"
        output_path = output_dir / f"{timestamp}_{args.law}_{args.provider}_{mode}_batch.jsonl"
        resume = False

    base_url = f"ws://{args.host}"

    asyncio.run(run_batch(
        law=args.law,
        profiles=profiles,
        provider=args.provider,
        output_path=output_path,
        base_url=base_url,
        graph=args.graph,
        no_guard=args.no_guard,
        timeout=args.timeout,
        verbose=args.verbose,
        limit=args.limit,
        save_graphs=args.save_graphs,
        resume=resume,
    ))


if __name__ == "__main__":
    main()
