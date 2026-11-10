"""
LLM as API — analyze_customer_feedback
Uses OpenRouter (chat.completions) with a free model.
Install: pip install openai
Set env:  OPENROUTER_API_KEY=<your_key>
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads variables into the environment

# ── Client setup ──────────────────────────────────────────────────────────────

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY"),
# )
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


MODEL = "openai/gpt-4o-mini"        
# MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# ── System prompt (the API contract) ─────────────────────────────────────────

SYSTEM_PROMPT = """
You are a customer feedback analysis API.

FUNCTION: analyze_customer_feedback

INPUT: A JSON object with:
  - feedback_text  : string  — raw customer feedback
  - product_name   : string  — product being reviewed
  - language       : string  — "en" | "hi" | "auto"

OUTPUT: Return ONLY a valid JSON object. No explanation, no markdown, no preamble.

OUTPUT SCHEMA:
{
  "sentiment"        : "positive" | "neutral" | "negative",
  "sentiment_score"  : float,        // -1.0 (very negative) to 1.0 (very positive)
  "intent"           : "complaint" | "praise" | "query" | "suggestion",
  "topics"           : string[],     // key topics/issues mentioned (max 5)
  "urgency"          : "low" | "medium" | "high",
  "suggested_action" : string        // one-line recommended next step for support team
}

RULES:
- Return ONLY valid JSON. No extra keys. No trailing text.
- If input is missing required fields return: {"error": "invalid_input", "message": "<reason>"}
- urgency is "high" if there is financial impact, data loss, or repeated failure.
- sentiment_score must be a float rounded to 2 decimal places.
""".strip()

# ── Core function ─────────────────────────────────────────────────────────────

def analyze_feedback(feedback_text: str, product_name: str, language: str = "en") -> dict:
    """
    Calls the LLM as a structured feedback analysis API.
    Returns a parsed dict matching the output schema.
    """
    payload = {
        "feedback_text": feedback_text,
        "product_name": product_name,
        "language": language,
    }

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(payload)},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model ignores the instruction
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)

# ── Sample queries ─────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    {
        "feedback_text": "The app keeps crashing every time I try to checkout. "
                         "I've lost my cart twice now. This is unacceptable for a paid subscription.",
        "product_name": "ShopEase Pro",
        "language": "en",
    },
    {
        "feedback_text": "Absolutely love the new dashboard! The filters are super intuitive "
                         "and the export feature saves me hours every week.",
        "product_name": "DataFlow Analytics",
        "language": "en",
    },
    {
        "feedback_text": "Can you add dark mode? It would make the app much easier to use at night.",
        "product_name": "NoteKeeper",
        "language": "en",
    },
]

# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i} — {query['product_name']}")
        print(f"Input: {query['feedback_text'][:80]}...")
        print("-" * 60)
        result = analyze_feedback(**query)
        print(json.dumps(result, indent=2))
