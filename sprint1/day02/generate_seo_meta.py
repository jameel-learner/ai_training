"""
LLM as API — generate_seo_meta
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
You are an SEO metadata generation API.

FUNCTION: generate_seo_meta

INPUT: A plain-text string containing the full body of an article, blog post, or web page.

OUTPUT: Return ONLY a valid JSON object. No explanation, no markdown, no preamble.

OUTPUT SCHEMA:
{
  "title"                : string,     // SEO title tag — 50–60 characters, includes primary keyword
  "meta_description"     : string,     // 150–160 characters, compelling, includes primary keyword
  "primary_keyword"      : string,     // single most important keyword/phrase
  "secondary_keywords"   : string[],   // 4–6 supporting keyword phrases
  "slug"                 : string,     // URL-friendly slug using primary keyword (lowercase, hyphens)
  "og_title"             : string,     // Open Graph title — can be slightly more creative, max 60 chars
  "og_description"       : string,     // Open Graph description — 1–2 sentences, engaging, max 200 chars
  "schema_type"          : "Article" | "BlogPosting" | "HowTo" | "FAQPage" | "Product" | "LocalBusiness",
  "content_grade"        : "thin" | "medium" | "comprehensive",  // based on depth/length
  "readability"          : "beginner" | "intermediate" | "expert",
  "word_count_estimate"  : integer
}

RULES:
- Return ONLY valid JSON. No extra keys. No trailing text.
- title must be between 50–60 characters. Count carefully.
- meta_description must be between 150–160 characters. Count carefully.
- slug must use only lowercase letters, numbers, and hyphens. No special characters.
- secondary_keywords must be phrases (2–4 words each), not single words.
- If input is too short (under 50 words) return: {"error": "content_too_short", "message": "Article must be at least 50 words"}
""".strip()

# ── Core function ─────────────────────────────────────────────────────────────

def generate_seo_meta(article_text: str) -> dict:
    """
    Calls the LLM as a structured SEO metadata generation API.
    Returns a parsed dict matching the output schema.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": article_text},
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
    # Sample 1 — tech/AI article
    """
    What is RAG? A Beginner's Guide to Retrieval-Augmented Generation

    Retrieval-Augmented Generation (RAG) is a technique in AI that combines the
    power of large language models (LLMs) with external knowledge bases. Instead
    of relying purely on information baked into the model during training, RAG
    allows the model to fetch relevant documents at query time and use them to
    generate more accurate, up-to-date answers.

    How does RAG work?
    RAG works in two main steps. First, a retrieval step searches a vector database
    for documents semantically similar to the user's question. Then, a generation
    step passes those retrieved documents as context to the LLM, which uses them
    to produce a grounded answer.

    Why use RAG?
    LLMs have a knowledge cutoff — they don't know about events after their training
    date. RAG solves this by connecting the model to live or updated data sources
    like company wikis, product documentation, or news feeds. It also reduces
    hallucination because the model can cite actual retrieved content rather than
    making things up.

    RAG is widely used in enterprise chatbots, customer support automation,
    legal document search, and medical information retrieval systems.
    """,

    # Sample 2 — health/wellness blog
    """
    10 Science-Backed Benefits of Walking 10,000 Steps a Day

    Walking is one of the simplest and most accessible forms of exercise, yet its
    health benefits are profound. Research consistently shows that walking 10,000
    steps a day — roughly 7 to 8 kilometres — can dramatically improve physical
    and mental health.

    1. Improves cardiovascular health: Regular walking strengthens the heart,
    lowers blood pressure, and reduces LDL cholesterol levels.

    2. Aids weight management: Walking burns calories and boosts metabolism,
    helping to maintain a healthy weight without the strain of high-impact exercise.

    3. Reduces risk of type 2 diabetes: Studies show that people who walk regularly
    have significantly lower fasting blood glucose levels.

    4. Boosts mood and reduces anxiety: Physical activity releases endorphins and
    serotonin, natural chemicals that elevate mood and reduce stress.

    5. Strengthens bones and joints: Weight-bearing exercises like walking improve
    bone density and reduce the risk of osteoporosis.

    Starting a daily walking habit is easy — begin with 5,000 steps and increase
    gradually. Use a fitness tracker or smartphone app to monitor your progress.
    """,

    # Sample 3 — product page / e-commerce
    """
    BrewMaster Pro 3000 — Automatic Coffee Maker with Built-in Grinder

    The BrewMaster Pro 3000 is a premium automatic coffee machine designed for
    coffee lovers who demand café-quality espresso and filter coffee at home.

    Key Features:
    - Built-in conical burr grinder with 5 grind settings
    - 15-bar pressure pump for rich, crema-topped espresso
    - 1.8-litre removable water tank
    - Programmable auto-brew timer
    - Compatible with beans and pre-ground coffee
    - Milk frother for lattes and cappuccinos
    - Easy-clean detachable brew group

    The BrewMaster Pro 3000 is perfect for households that want the flexibility
    of a full grind-and-brew system without the complexity of a professional setup.
    Backed by a 2-year manufacturer warranty and free shipping across India.

    Price: ₹18,999 | Available in: Matte Black, Pearl White
    """,
]

# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i, article in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print(f"Article preview: {article.strip()[:80]}...")
        print("-" * 60)
        result = generate_seo_meta(article)
        print(json.dumps(result, indent=2))
