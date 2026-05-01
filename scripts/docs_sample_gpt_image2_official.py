#!/usr/bin/env python3
"""Generate ONE sample image via OpenAI **official** Images API (no custom base_url).

Writes:
  docs/openai_official_gpt_image2_sample.png
  docs/openai_official_gpt_image2_sample.json

Usage (PowerShell):
  $env:OPENAI_API_KEY = "sk-..."
  python scripts/docs_sample_gpt_image2_official.py

Requires network + a key with gpt-image-2 access on api.openai.com.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT_PNG = DOCS / "openai_official_gpt_image2_sample.png"
OUT_JSON = DOCS / "openai_official_gpt_image2_sample.json"

PROMPT = (
    "Photoreal empty specialty coffee shop interior, late morning natural light, "
    "no people, no animals, no text, wide angle, sharp focus."
)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        OUT_JSON.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "OPENAI_API_KEY is not set",
                    "run": "python scripts/docs_sample_gpt_image2_official.py",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("[docs_sample] OPENAI_API_KEY missing; wrote", OUT_JSON, file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=key)  # official default host only

    kw: dict = dict(
        model="gpt-image-2",
        prompt=PROMPT,
        size="1024x1024",
        n=1,
    )
    try:
        kw["quality"] = "medium"
        resp = client.images.generate(**kw)
    except TypeError:
        kw.pop("quality", None)
        resp = client.images.generate(**kw)

    d0 = resp.data[0]
    b64 = getattr(d0, "b64_json", None)
    url = getattr(d0, "url", None)
    if b64:
        raw = base64.b64decode(b64)
    elif url:
        import urllib.request

        raw = urllib.request.urlopen(url, timeout=120).read()
    else:
        raise RuntimeError(f"no b64_json or url in response: {d0!r}")

    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(OUT_PNG, format="PNG", dpi=(72, 72))

    meta = {
        "ok": True,
        "api": "https://api.openai.com/v1/images/generations",
        "model": kw["model"],
        "size": kw["size"],
        "quality": kw.get("quality"),
        "prompt": PROMPT,
        "output_png": str(OUT_PNG.relative_to(REPO)).replace("\\", "/"),
        "image_size_px": list(img.size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[docs_sample] wrote", OUT_PNG, "and", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
