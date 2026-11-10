"""
LLM as API — extract_invoice_fields
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
You are an invoice data extraction API.

FUNCTION: extract_invoice_fields

INPUT: A plain-text string containing raw invoice content (pasted text, OCR output, email body, etc.)

OUTPUT: Return ONLY a valid JSON object. No explanation, no markdown, no preamble.

OUTPUT SCHEMA:
{
  "invoice_number"   : string | null,
  "invoice_date"     : string | null,    // ISO 8601 format: YYYY-MM-DD
  "due_date"         : string | null,    // ISO 8601 format: YYYY-MM-DD
  "vendor"           : {
    "name"           : string | null,
    "email"          : string | null,
    "phone"          : string | null,
    "address"        : string | null
  },
  "bill_to"          : {
    "name"           : string | null,
    "email"          : string | null,
    "address"        : string | null
  },
  "line_items"       : [
    {
      "description"  : string,
      "quantity"     : float | null,
      "unit_price"   : float | null,
      "total"        : float | null
    }
  ],
  "subtotal"         : float | null,
  "tax_amount"       : float | null,
  "tax_rate_percent" : float | null,
  "discount"         : float | null,
  "total_amount"     : float | null,
  "currency"         : string,           // ISO 4217 e.g. "INR", "USD", "EUR"
  "payment_terms"    : string | null,    // e.g. "Net 30", "Due on receipt"
  "notes"            : string | null
}

RULES:
- Return ONLY valid JSON. No extra keys. No trailing text.
- Use null for any field not found in the input.
- All monetary values must be floats (no currency symbols).
- If the text is clearly not an invoice return: {"error": "not_an_invoice", "message": "Input does not appear to be invoice content"}
""".strip()

# ── Core function ─────────────────────────────────────────────────────────────

def extract_invoice_fields(raw_text: str) -> dict:
    """
    Calls the LLM as a structured invoice extraction API.
    Returns a parsed dict matching the output schema.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw_text},
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
    # Sample 1 — clean structured invoice
    """
    INVOICE
    Invoice No: INV-2024-0892
    Date: 15 March 2024
    Due Date: 14 April 2024

    From:
    TechSolutions Pvt Ltd
    tech@techsolutions.in
    +91-98765-43210
    12 Whitefield Main Road, Bangalore 560066

    Bill To:
    Acme Corp
    accounts@acme.com
    45 MG Road, Bangalore 560001

    ITEMS:
    UI/UX Design Services    1    ₹45,000    ₹45,000
    API Integration           2    ₹12,000    ₹24,000
    DevOps Setup              1    ₹8,500     ₹8,500

    Subtotal:  ₹77,500
    GST (18%): ₹13,950
    Total:     ₹91,450

    Payment Terms: Net 30
    Notes: Please include invoice number in bank transfer reference.
    """,

    # Sample 2 — messy email-style invoice
    """
    Hi Ramesh,

    Please find below the invoice for last month's work.

    Invoice #: 007  |  Issued: Jan 5 2024
    Freelancer: Priya Sharma (priya.freelance@gmail.com)

    Work done:
    - Content writing for blog posts: 8 articles x $50 = $400
    - Social media captions: 30 posts x $10 = $300
    - Monthly retainer fee: $200

    Subtotal: $900
    Discount (loyalty): $50
    Tax (GST): $76.50
    TOTAL DUE: $926.50

    Please pay within 15 days. Bank details shared separately.

    Thanks,
    Priya
    """,

    # Sample 3 — minimal / incomplete
    """
    Invoice from Raj Electricals
    Date: 22-02-2024
    Customer: Mr. Ahmed

    Wiring work for 2BHK apartment   ₹15,000
    Materials (wire, switches)        ₹4,200

    Total: ₹19,200
    Cash payment preferred.
    """,
]

# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, raw_text in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print(f"Input preview: {raw_text.strip()[:80]}...")
        print("-" * 60)
        result = extract_invoice_fields(raw_text)
        print(json.dumps(result, indent=2))
