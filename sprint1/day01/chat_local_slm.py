"""
Local SLM (Small Language Model) - no API key, no internet call.

Runs entirely on your machine via Ollama (CPU-friendly, works fine on a
16GB RAM / i7 laptop with no GPU). Two ways to call the same local model,
mirroring the other files in this folder:

1. Native `ollama` SDK        - Ollama's own client library.
2. OpenAI-compatible endpoint - Ollama also exposes a `chat.completions`
                                 endpoint, so the exact same `openai`
                                 client code from chat_openai.py etc.
                                 works against your local model too.

--- One-time setup ---
1. Install Ollama: https://ollama.com/download
2. Pull a small model that fits comfortably in 16GB RAM, CPU-only:
       ollama pull phi3:mini        # Microsoft Phi-3-mini, ~2.3GB, 3.8B params
   Other good CPU-friendly options: qwen2.5:3b, llama3.2:3b, gemma2:2b
3. Ollama runs a local server automatically on http://localhost:11434
   (start it manually with `ollama serve` if it isn't already running).
4. pip install ollama openai

No API key is needed - Ollama runs fully offline once the model is pulled.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
      https://github.com/ollama/ollama/blob/main/docs/openai.md
"""
from openai import OpenAI
import ollama

MODEL = "phi3:mini"
question = "What is the capital of India?"
s_question = "Write 5 sentences about India?"


def call_via_native_sdk() -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"]


def call_via_openai_compatible() -> str:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the SDK, but Ollama ignores its value
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def call_with_streaming() -> str:
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": s_question},
        ],
        stream=True,
    )

    chunks = []
    for event in stream:
        delta = event["message"]["content"]
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()

    return "".join(chunks)


if __name__ == "__main__":
    print("=== Native ollama SDK ===")
    print(call_via_native_sdk())
    print()

    print("=== OpenAI-compatible endpoint (same openai SDK as the cloud demos) ===")
    print(call_via_openai_compatible())
    print()

    print("=== Streaming (stream=True) ===")
    call_with_streaming()
