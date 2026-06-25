"""
Graph chat client — identiek aan chat_client.py maar altijd in graph modus.

In graph modus krijgt de LLM een geserialiseerde knowledge graph als JSON
in plaats van de ruwe engine-output. Bedoeld als vergelijking met de nulmeting.

Usage:
    uv run python analysis/llm_explanations/scripts/chat_client_graph.py
    uv run python analysis/llm_explanations/scripts/chat_client_graph.py --bsn 403987006
    uv run python analysis/llm_explanations/scripts/chat_client_graph.py --bsn 403987006 --verbose

Requires the web server:
    $env:FEATURE_CHAT='1'; uv run web/main.py
"""

import argparse
import asyncio
import json
import sys
import uuid

import websockets


BASE_URL = "ws://localhost:8000"
VERBOSE = False


async def chat_session(bsn: str, provider: str) -> None:
    client_id = f"graph_{bsn}_{uuid.uuid4().hex[:8]}"
    uri = f"{BASE_URL}/chat/ws/{client_id}"

    print(f"[Graph] Verbinding maken met {uri} (BSN: {bsn}, provider: {provider})...")

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=120) as ws:
            await ws.send(json.dumps({"bsn": bsn, "provider": provider, "graph": True}))

            raw = await ws.recv()
            data = json.loads(raw)

            if data.get("error"):
                print(f"[FOUT] {data['error']}")
                return

            if data.get("feature_disabled"):
                print("[FOUT] Chat feature is uitgeschakeld. Start server met: $env:FEATURE_CHAT='1'")
                return

            model = data.get("model", "onbekend")
            print(f"Verbonden. Model: {model} | Modus: Graph")
            print("Type uw vraag en druk op Enter. Typ 'exit' om te stoppen.\n")
            print("=" * 60)

            loop = asyncio.get_running_loop()
            while True:
                try:
                    user_input = (await loop.run_in_executor(None, input, "\n[U] ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nSessie beëindigd.")
                    break

                if user_input.lower() in ("exit", "quit", "stop"):
                    print("Sessie beëindigd.")
                    break

                if not user_input:
                    continue

                await _send_and_receive(ws, user_input)

    except ConnectionRefusedError:
        print("[FOUT] Kan geen verbinding maken. Is de webserver actief?")
        print("Start de server met:  $env:FEATURE_CHAT='1'; uv run web/main.py")
        sys.exit(1)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"\n[SESSIE BEËINDIGD] Server sloot de verbinding (code {e.code}). Controleer de server logs.")
    except websockets.exceptions.WebSocketException as e:
        print(f"[FOUT] WebSocket fout: {e}")
        sys.exit(1)


async def _send_and_receive(ws, message: str) -> None:
    await ws.send(json.dumps({"message": message}))

    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
            data = json.loads(raw)

            if data.get("error"):
                print(f"\n[FOUT] {data['error']}")
                break

            if data.get("isProcessing"):
                print(f"  ... {data.get('message', '')}")
                continue

            if data.get("applicationPanel"):
                print("\n[FORMULIER] Aanvraagformulier beschikbaar in de web interface.")

            msg = data.get("message", "")
            if msg:
                print(f"\n[Assistent]\n{msg}")

                if VERBOSE and "guard" in data:
                    guard = data["guard"]
                    status = "✓ in-scope" if guard.get("valid") else "✗ out-of-scope (redirect)"
                    print(f"\n  [guard] {status} — {guard.get('explanation', '')}")

                print("-" * 60)

            try:
                follow_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                follow_data = json.loads(follow_raw)
                if follow_data.get("isProcessing"):
                    print(f"  ... {follow_data.get('message', '')}")
                    continue
                follow_msg = follow_data.get("message", "")
                if follow_msg:
                    print(f"\n[Assistent vervolg]\n{follow_msg}")
                    print("-" * 60)
            except asyncio.TimeoutError:
                break

        except asyncio.TimeoutError:
            print("[TIMEOUT] Geen antwoord ontvangen binnen 120 seconden.")
            break


def main() -> None:
    global BASE_URL, VERBOSE

    parser = argparse.ArgumentParser(description="Graph chat client voor machine-law")
    parser.add_argument("--bsn", default="403987006", help="BSN van het profiel (default: 403987006)")
    parser.add_argument("--provider", default="claude",
        choices=["claude", "vlam", "gpt-4o", "gpt-4o-mini", "llama3.1", "llama3.2", "llama3.3", "mistral", "deepseek", "gemma2"],
        help="LLM provider (default: claude)")
    parser.add_argument("--host", default="localhost:8000", help="Host:port van de webserver")
    parser.add_argument("--verbose", action="store_true", help="Toon LLM guard beslissingen")
    args = parser.parse_args()

    BASE_URL = f"ws://{args.host}"
    VERBOSE = args.verbose

    asyncio.run(chat_session(args.bsn, args.provider))


if __name__ == "__main__":
    main()
