# Codegen Studio rules
# Applied to every mode (chat, bugbot, agent: planner, coder and reviewer) — like .cursorrules.
# Edit freely; changes apply on the next request (no restart needed). Lines starting with # are ignored.

## Secrets
- Never hardcode API keys or tokens. Read them from environment variables loaded with python-dotenv
  (`load_dotenv()`), and add `python-dotenv` to requirements.txt when used.

## OpenRouter (verified 2026-11-10)
- Base URL: https://openrouter.ai/api/v1 (NOT api.openrouter.ai / openrouter.com)
- Chat endpoint: POST https://openrouter.ai/api/v1/chat/completions (plural "completions")
- Auth header: `Authorization: Bearer $OPENROUTER_API_KEY`; key env var: OPENROUTER_API_KEY
- Body: {"model": ..., "messages": [{"role": "user", "content": ...}]}; reply text: choices[0].message.content
- Model ids are "provider/model", e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4.5 (NOT "gpt-3.5-turbo")
- With the openai SDK: OpenAI(api_key=..., base_url="https://openrouter.ai/api/v1")

## Hugging Face Inference Providers (verified 2026-11-10)
- Base URL: https://router.huggingface.co/v1 — OpenAI-compatible
- The legacy https://api-inference.huggingface.co/models/... endpoint is shut down: never use it.
- Chat endpoint: POST https://router.huggingface.co/v1/chat/completions, same body/response shape as OpenAI
- Auth header: `Authorization: Bearer $HF_TOKEN`; token env var: HF_TOKEN
- Model ids are Hub repo ids, e.g. meta-llama/Llama-3.1-8B-Instruct; list available ones with GET https://router.huggingface.co/v1/models
- With the openai SDK: OpenAI(api_key=os.environ["HF_TOKEN"], base_url="https://router.huggingface.co/v1")

## Prefer the openai SDK for OpenAI-compatible providers
- All providers below accept the OpenAI chat-completions format. Unless the request says otherwise, use
  `from openai import OpenAI; client = OpenAI(api_key=..., base_url=...)` and `client.chat.completions.create(...)`,
  with the base URL and env var from the matching section. Add `openai` to requirements.txt.

## OpenAI (verified 2026-11-10)
- Base URL: https://api.openai.com/v1 (the openai SDK's default — no base_url needed)
- Key env var: OPENAI_API_KEY
- Model ids have NO provider prefix: gpt-4o-mini, gpt-4.1-mini, gpt-5-mini, gpt-5.5 (NOT "openai/gpt-4o-mini" — that is OpenRouter's form)
- Do not use retired models such as gpt-3.5-turbo or text-davinci-*; do not use the legacy /v1/completions endpoint

## Google Gemini (verified 2026-11-10)
- Key env var: GOOGLE_API_KEY
- OpenAI-compatible base URL: https://generativelanguage.googleapis.com/v1beta/openai/ (trailing slash included)
- Native REST: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  with header `x-goog-api-key: $GOOGLE_API_KEY`, body {"contents":[{"parts":[{"text": ...}]}]},
  reply text: candidates[0].content.parts[0].text
- Native SDK: package `google-genai` → `from google import genai; client = genai.Client(api_key=...)`;
  `client.models.generate_content(model=..., contents=...)`, reply `.text`. Do NOT use the deprecated `google-generativeai` package.
- Model ids: gemini-3.5-flash (default), gemini-3.5-flash-lite (cheaper), gemini-flash-latest (alias, sometimes overloaded/503)
- Do NOT use gemini-2.5-* or older for text generation (gemini-2.5-flash returns 404) or gemini-pro / gemini-1.5-*

## Groq (verified 2026-11-10)
- Base URL: https://api.groq.com/openai/v1 (NOT api.groq.ai / groq.com/v1)
- Key env var: GROQ_API_KEY
- Chat model ids: openai/gpt-oss-20b (fast default), openai/gpt-oss-120b, qwen/qwen3.8-27b
- Llama 3 / Mixtral ids such as llama3-8b-8192 or mixtral-8x7b-32768 are retired — do not use them
- Speech-to-text: whisper-large-v3-turbo via POST /audio/transcriptions

## DeepSeek (verified 2026-11-10)
- Base URL: https://api.deepseek.com (also accepts https://api.deepseek.com/v1)
- Key env var: DEEPSEEK_API_KEY
- Model ids: deepseek-flash (fast default), deepseek-v4-pro (stronger) — NOT the old deepseek-chat / deepseek-coder

## Unknown APIs
- For any other external API, state the endpoint URL and model/version you assume in a comment at the top of the
  file and keep them in one constant so they are easy to correct.
