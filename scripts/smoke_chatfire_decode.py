"""Smoke test: chatfire @ 2048x1024 with the new b64/url-tolerant decoder."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import OpenAI

from erpgen.openai_erp import _decode_image_resp

c = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.chatfire.cn/v1",
    timeout=120,
)
t0 = time.time()
r = c.images.generate(
    model="gpt-image-2",
    prompt=(
        "An empty cafe interior, equirectangular 360 panorama, "
        "late morning, no people."
    ),
    size="2048x1024", n=1, quality="medium",
)
print(f"gen done in {time.time() - t0:.1f}s")
d = r.data[0]
print(f"  has b64: {bool(getattr(d, 'b64_json', None))}, "
      f"has url: {bool(getattr(d, 'url', None))}")
img = _decode_image_resp(r)
print(f"  decoded -> {img.size}")
out = REPO / "outputs" / "_chatfire_smoke.png"
img.save(out)
print(f"  saved -> {out}")
