"""
LLM as API — classify_support_ticket
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

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4o-mini"
# MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# ── System prompt (the API contract) ─────────────────────────────────────────

SYSTEM_PROMPT = """
You are a support ticket classification API.

FUNCTION: classify_support_ticket

INPUT: A plain-text string containing the support ticket body (submitted by a customer).

OUTPUT: Return ONLY a valid JSON object. No explanation, no markdown, no preamble.

OUTPUT SCHEMA:
{
  "department"        : "billing" | "technical" | "account" | "shipping" | "general",
  "priority"          : "P1-critical" | "P2-high" | "P3-medium" | "P4-low",
  "category"          : string,           // short label e.g. "payment failure", "login issue"
  "summary"           : string,           // 1-sentence summary of the issue
  "keywords"          : string[],         // 3–5 keywords from the ticket
  "requires_human"    : boolean,          // true if too complex/sensitive for a bot reply
  "suggested_response": string            // one-paragraph draft response to send the customer
}

PRIORITY RULES:
- P1-critical : system down, data loss, security breach, financial fraud, service fully unusable
- P2-high     : core feature broken, payment issue, repeated failures, SLA at risk
- P3-medium   : partial feature issue, workaround exists, general complaint
- P4-low      : how-to question, cosmetic issue, feature request, general inquiry

DEPARTMENT RULES:
- billing    : payments, invoices, refunds, subscriptions, charges
- technical  : bugs, crashes, errors, integrations, API, performance
- account    : login, password, profile, permissions, 2FA, account deletion
- shipping   : delivery, tracking, returns, damaged goods (for e-commerce)
- general    : anything that doesn't fit the above

RULES:
- Return ONLY valid JSON. No extra keys. No trailing text.
- suggested_response must be empathetic, professional, and include a next step.
- If the input is empty or gibberish return: {"error": "invalid_input", "message": "<reason>"}
""".strip()

# ── Core function ─────────────────────────────────────────────────────────────

def classify_support_ticket(ticket_text: str) -> dict:
    """
    Calls the LLM as a structured support ticket classification API.
    Returns a parsed dict matching the output schema.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ticket_text},
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
    # Sample 1 — billing / high priority
    """
    Subject: Charged twice for the same order!

    Hi, I placed order #ORD-88231 on Sunday and I can see two charges of ₹2,499
    on my credit card statement dated March 10th. I only placed the order once.
    I need an immediate refund for the duplicate charge. This is really frustrating
    and I'm considering disputing the charge with my bank if this isn't resolved today.
    """,

    # Sample 2 — technical / critical
    """
    Our entire team is locked out of the platform since 9 AM this morning.
    We're getting a 503 Service Unavailable error on every page. We have a product
    demo with a major client at 2 PM today and this is a disaster. Our account is
    team@rapidgrowth.io. Please escalate this immediately — every minute of downtime
    is costing us money.
    """,

    # Sample 3 — account / low priority
    """
    Hi there, I forgot how to change my profile picture on the app.
    I've looked in settings but can't find it. Can you guide me?
    Thanks, Anjali
    """,

    # Sample 4 — general feature request
    """
    It would be great if you could add a dark mode to the mobile app.
    A lot of us use it late at night and the white background is really harsh on the eyes.
    Not urgent at all, just a suggestion for a future update!
    """,
]

# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, ticket in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print(f"Ticket: {ticket.strip()[:80]}...")
        print("-" * 60)
        result = classify_support_ticket(ticket)
        print(json.dumps(result, indent=2))
