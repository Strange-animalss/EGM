"""Quick probe for text model availability on OpenAI direct."""
from __future__ import annotations

import os
import time

from openai import OpenAI

c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60)
for m in ["gpt-5.5-pro", "gpt-5.5", "gpt-5", "gpt-4o", "gpt-4.1", "gpt-4o-mini"]:
    t0 = time.time()
    try:
        r = c.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "say OK"}],
            max_tokens=10,
        )
        print(f"{m}: OK in {time.time() - t0:.1f}s -> "
              f"{r.choices[0].message.content!r}")
    except Exception as e:
        print(f"{m}: FAIL in {time.time() - t0:.1f}s: "
              f"{type(e).__name__}: {str(e)[:200]}")
