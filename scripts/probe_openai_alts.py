"""Test what image models are actually available on this OpenAI key without
org verification. Tries cheap small calls only."""
from __future__ import annotations

import os
import time

from openai import OpenAI

key = os.environ["OPENAI_API_KEY"]
cli = OpenAI(api_key=key, timeout=120)

models_to_try = [
    "gpt-image-1",
    "gpt-image-2-mini",
    "gpt-image-2",
    "dall-e-3",
    "dall-e-2",
]
for m in models_to_try:
    print(f"\n--- {m} (1024x1024 quality=low) ---")
    t0 = time.time()
    try:
        kw = {"model": m, "prompt": "a blue sky test", "size": "1024x1024", "n": 1}
        if m.startswith("gpt-image"):
            kw["quality"] = "low"
        r = cli.images.generate(**kw)
        print(f"  OK in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  FAIL in {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:250]}")

print("\n--- list available image models ---")
try:
    ml = cli.models.list()
    image_like = [m.id for m in ml.data if "image" in m.id or "dall" in m.id]
    print("  image-like model IDs:")
    for i in sorted(image_like):
        print(f"    {i}")
except Exception as e:
    print(f"  list failed: {e}")
