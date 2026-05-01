"""Probe what shangliu's gpt-image-2 `images.edit` actually accepts.

Uses the cafe_v6 pose_0 (2048x1024 true ERP) as the reference image and
tries a few size variants.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_image2_edit_probe"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("SHANGLIU_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set SHANGLIU_API_KEY or OPENAI_API_KEY")
client = OpenAI(api_key=KEY, base_url="https://super.shangliu.org/v1", timeout=600)

REF_PATH = REPO / "outputs" / "runs" / "cafe_v6_20260501-081138" / "erp" / "rgb" / "pose_0.png"
if not REF_PATH.exists():
    sys.exit(f"FATAL: ref not found {REF_PATH}")

print(f"reference: {REF_PATH}  ({REF_PATH.stat().st_size:,} bytes)")
ref_size = Image.open(REF_PATH).size
print(f"reference dims: {ref_size}")

PROMPT = (
    "Same coffee shop interior viewed from a different camera position "
    "within the same room. Equirectangular 360 panorama, photorealistic, "
    "no people, no text, no watermark."
)


def try_call(label: str, **kwargs) -> dict:
    print(f"\n=== {label} ===")
    print(f"  kwargs: {dict((k, v) for k, v in kwargs.items() if k != 'image')}")
    t0 = time.time()
    try:
        with open(REF_PATH, "rb") as f:
            kwargs["image"] = f
            resp = client.images.edit(**kwargs)
    except Exception as e:
        print(f"  EXC: {type(e).__name__}: {str(e)[:300]}")
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "elapsed_s": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    try:
        d = resp.data[0]
        b64 = getattr(d, "b64_json", None)
        url = getattr(d, "url", None)
        if b64:
            raw = base64.b64decode(b64)
        elif url:
            import requests
            raw = requests.get(url, timeout=60).content
        else:
            return {"label": label, "error": "no b64/url", "elapsed_s": elapsed}
    except Exception as e:
        return {"label": label, "error": f"parse: {e}", "elapsed_s": elapsed}
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    out = OUT / f"{label}.png"
    out.write_bytes(raw)
    print(f"  OK image: {w}x{h}  aspect={round(w/h, 4)}  elapsed={elapsed}s -> {out.name}")
    return {"label": label, "size": [w, h], "aspect": round(w / h, 4),
            "path": str(out), "elapsed_s": elapsed}


tests = [
    # Minimal kwargs, expect 2:1 source -> 2:1 output
    ("E1_2048x1024_minimal", {
        "model": "gpt-image-2", "prompt": PROMPT, "size": "2048x1024", "n": 1,
    }),
    # Smaller landscape
    ("E2_1536x1024_minimal", {
        "model": "gpt-image-2", "prompt": PROMPT, "size": "1536x1024", "n": 1,
    }),
    # square (matches OpenAI api-edit baseline)
    ("E3_1024x1024_minimal", {
        "model": "gpt-image-2", "prompt": PROMPT, "size": "1024x1024", "n": 1,
    }),
    # auto
    ("E4_auto", {
        "model": "gpt-image-2", "prompt": PROMPT, "size": "auto", "n": 1,
    }),
]

results = [try_call(label, **kw) for label, kw in tests]

print("\n\n=== SUMMARY ===")
for r in results:
    if "size" in r:
        print(f"  {r['label']:30s} -> {r['size']}  aspect={r['aspect']}  ({r['elapsed_s']}s)")
    else:
        print(f"  {r['label']:30s} -> FAIL: {r.get('error', '?')[:120]}")
