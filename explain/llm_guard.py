"""
LLM-as-judge guard for the chat interface.

After the chatbot generates a response, this module evaluates whether the
response is in-scope (Dutch government regulations, benefits, permits) using
a separate LLM call. Out-of-scope responses are replaced with a redirect.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

GUARD_SYSTEM_PROMPT = """Je bent een QA-expert voor een overheids-chatbot.
De chatbot mag UITSLUITEND helpen met vragen over Nederlandse overheidsregelingen,
toeslagen, uitkeringen, vergunningen en aanvragen.

Beoordeel of het antwoord van de chatbot past bij de verwachte intentie.
Reageer ALLEEN met één regel JSON: { "valid": true/false, "explanation": "..." }
"""

REDIRECT_MESSAGE = (
    "Ik kan u alleen helpen met vragen over Nederlandse overheidsregelingen, "
    "toeslagen, uitkeringen en vergunningen. "
    "Heeft u een vraag hierover?"
)


def validate_response(user_prompt: str, bot_answer: str) -> tuple[bool, str]:
    """
    Ask a judge LLM whether the bot's response is in-scope.

    Returns (valid, explanation). Falls back to valid=True on any error
    so the guard never blocks legitimate responses due to API issues.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return True, "No API key — guard skipped"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        user_content = (
            f"User Prompt: {user_prompt}\n"
            f"Chatbot Answer: {bot_answer}\n"
            "Expected Intent: De chatbot helpt burgers uitsluitend met vragen over "
            "Nederlandse overheidsregelingen, toeslagen, uitkeringen en vergunningen."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheap + fast for guard calls
            max_tokens=128,
            temperature=0,
            system=GUARD_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text.strip()
        verdict = json.loads(raw)
        return bool(verdict.get("valid", True)), verdict.get("explanation", "")

    except Exception as e:
        logger.warning("LLM guard failed, defaulting to valid=True: %s", e)
        return True, f"guard error: {e}"
