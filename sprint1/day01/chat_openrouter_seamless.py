"""
OpenRouter - two ways to make the same call, across many providers.

1. OpenAI-compatible endpoint - via the generic `openai` SDK. This is
   the point of the demo: one client, one `chat.completions.create`
   call shape, and switching providers/models is just a string change.
2. Native REST API            - OpenRouter has no dedicated Python SDK;
   its own docs show plain HTTP, so that's shown here via `requests`.
Docs: https://openrouter.ai/docs
"""
import os
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()  # reads .env and loads variables into the environment

question = "What is the capital of India?"
s_question = "Write 3 sentences about India?"

# Same call shape, different backing model each time.
models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-opus-5",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
]


def call_via_openai_compatible(model: str) -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def call_via_native_rest_api(model: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            "max_tokens": 500,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def call_with_streaming(model: str) -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    stream = client.chat.completions.create(
        model=model,
        max_tokens=500,
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": s_question},
        ],
    )

    chunks = []
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()

    return "".join(chunks)



if __name__ == "__main__":
    # print("=== OpenAI-compatible endpoint (openai SDK) ===")
    # for model in models:
    #     print(f"--- {model} ---")
    #     print(call_via_openai_compatible(model))
    #     print()

    # print("=== Native REST API (raw HTTP, no SDK) ===")
    # for model in models:
    #     print(f"--- {model} ---")
    #     print(call_via_native_rest_api(model))
    #     print()

    print("=== Streaming (stream=True) ===")
    for model in models:        #[:1]:
        print(f"--- {model} ---")
        call_with_streaming(model)
        print()
