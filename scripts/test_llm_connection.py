#!/usr/bin/env python
"""Test LLM connection from rlaunch environment.

Usage:
    python scripts/test_llm_connection.py
"""

import os
import sys

def main():
    print("=" * 60)
    print("LLM Connection Test")
    print("=" * 60)

    # 1. Check environment variables
    print("\n[1] Environment Variables:")
    base_url = os.getenv("OPENAI_BASE_URL", "NOT SET")
    api_key = os.getenv("OPENAI_API_KEY", "NOT SET")
    print(f"  OPENAI_BASE_URL: {base_url}")
    print(f"  OPENAI_API_KEY: {'***' if api_key != 'NOT SET' else 'NOT SET'}")

    if base_url == "NOT SET":
        print("\n  ERROR: OPENAI_BASE_URL not set!")
        return 1

    # 2. Test network connectivity
    print("\n[2] Network Connectivity:")
    import urllib.request
    import urllib.error

    # Extract host from URL
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.netloc

    try:
        # Try to connect to the health endpoint
        health_url = f"{base_url}/health"
        print(f"  Testing: {health_url}")
        req = urllib.request.Request(health_url, method='GET')
        with urllib.request.urlopen(req, timeout=10) as response:
            print(f"  Status: {response.status}")
            print(f"  Response: {response.read().decode()[:100]}")
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot connect to {base_url}")
        print(f"  Reason: {e.reason}")

        # Try ping
        print(f"\n  Trying to reach host {host}...")
        import subprocess
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{base_url}/health"],
            capture_output=True, text=True, timeout=10
        )
        print(f"  curl status code: {result.stdout}")
        return 1
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return 1

    # 3. Check model_kwargs from registry
    print("\n[3] Check model_kwargs from registry:")
    from miniagenticrouter.models.registry import ModelRegistry
    registry = ModelRegistry.get_instance()

    test_models = [
        "openai/gpt-5",
        "deepseek/deepseek-v3.2",
        "minimax/minimax-m2",
    ]

    for model_name in test_models:
        model_kwargs = registry.get_model_kwargs(model_name)
        print(f"  {model_name}:")
        print(f"    model_kwargs: {model_kwargs}")

    # 4. Test actual LiteLLM completion call for ALL models
    print("\n[4] Test LiteLLM Completion Calls:")

    import litellm
    # litellm._turn_on_debug()  # Uncomment for verbose debug

    proxy_mode = os.getenv("MAR_PROXY_MODE", "NOT SET")
    print(f"  MAR_PROXY_MODE: {proxy_mode}")

    failed_models = []

    for test_model in test_models:
        print(f"\n  --- Testing: {test_model} ---")

        # Get model_kwargs from registry
        model_kwargs = registry.get_model_kwargs(test_model)
        print(f"  Registry model_kwargs: {model_kwargs}")

        # Apply proxy mode logic (same as in LitellmModel._query)
        if os.getenv("MAR_PROXY_MODE"):
            if "custom_llm_provider" not in model_kwargs and "/" in test_model:
                model_kwargs["custom_llm_provider"] = "openai"
                print(f"  Added custom_llm_provider: openai (proxy mode)")

        print(f"  Final model_kwargs: {model_kwargs}")

        try:
            response = litellm.completion(
                model=test_model,
                messages=[{"role": "user", "content": "Say 'hello' in one word"}],
                max_tokens=100,
                **model_kwargs
            )
            print(f"  ✓ SUCCESS! Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed_models.append(test_model)

    print("\n" + "=" * 60)
    if failed_models:
        print(f"FAILED models ({len(failed_models)}/{len(test_models)}): {failed_models}")
        return 1
    else:
        print(f"All {len(test_models)} models passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
