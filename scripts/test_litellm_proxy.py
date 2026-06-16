#!/usr/bin/env python
"""Test LiteLLM proxy connection for all configured models."""

from openai import OpenAI

# Configuration
BASE_URL = "http://0.0.0.0:4000/v1"
API_KEY = "dummy"  # Replace with the configured master_key if one is set.

MODELS = [
    "openai/gpt-5",
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m2",
]


def test_model(client: OpenAI, model: str) -> None:
    """Test a single model."""
    print(f"=== Testing {model} ===")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=20,
            timeout=30,
        )
        content = resp.choices[0].message.content
        print(f"  ✓ Success: {content}")
    except Exception as e:
        print(f"Error: {e}")
    print()


def main():
    print(f"Testing LiteLLM Proxy at {BASE_URL}\n")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    for model in MODELS:
        test_model(client, model)

    print("Done!")


if __name__ == "__main__":
    main()
