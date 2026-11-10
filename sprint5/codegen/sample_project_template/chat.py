"""LLM logic for the ChatGPT-clone sample app."""

from collections.abc import Iterator

from openai import OpenAI

import config


def get_client(api_key: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter."""
    return OpenAI(api_key=api_key, base_url=config.BASE_URL)


def stream_response(client: OpenAI, messages: list[dict]) -> Iterator[str]:
    """Stream the assistant's reply token by token."""
    stream = client.chat.completions.create(model=config.MODEL, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
