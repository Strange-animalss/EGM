"""Find an endpoint that actually routes to real gpt-image-2 (which per
OpenAI cookbook supports arbitrary sizes up to ~3840 long-edge, both axes
multiples of 16, ratio <= 3:1, pixels in [655k, 8.3M]).

We test:
  - shangliu's `gpt-image-2-2026-04-21` (dated -> likely real gpt-image-2)
  - shangliu's `gpt-image-2` (alias - may be 1.5)
  - shangliu's `gpt-image-1.5` (control: should reject 2048x1024)
  - OpenRouter's `openai/gpt-image-2` and `openai/gpt-5.4-image-2`
    (chat-image only; size parameter likely ignored but worth confirming)

For each (endpoint, model) pair:
  - try size 2048x1024 (true 2:1 panorama)
  - try size 1536x1024 (3:2 landscape, baseline)
  - try size 1024x512 (alt 2:1)
  - record HTTP status, real returned dimensions, error message
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_image2_real_probe"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHANGLIU_KEY = os.environ.get("SHANGLIU_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not SHANGLIU_KEY:
    sys.exit("Set SHANGLIU_API_KEY or OPENAI_API_KEY (shangliu probes require a key).")

PROMPT = (
    "A clear photorealistic equirectangular 360 degree panorama of a simple "
    "empty cafe interior. Equirectangular projection, 2:1 aspect, full sphere, "
    "seamless horizontal wraparound. No people, no text, no watermark."
)


def probe_images_generate(label: str, base_url: str, key: str,
                          model: str, size: str, quality: str = "medium") -> dict:
    print(f"\n=== {label}  model={model!r}  size={size!r}  quality={quality!r} ===")
    url = f"{base_url}/images/generations"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": PROMPT, "size": size, "n": 1, "quality": quality}
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=300)
    except Exception as e:
        print(f"  REQ EXC: {e}")
        return {"label": label, "model": model, "size": size,
                "error": f"req: {e}", "elapsed_s": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    print(f"  status: {r.status_code}  elapsed={elapsed}s")
    ct = r.headers.get("content-type", "")
    if r.status_code != 200 or "json" not in ct.lower():
        body = r.text[:400]
        print(f"  ct={ct}  body={body}")
        return {"label": label, "model": model, "size": size,
                "status": r.status_code, "ct": ct, "error_body": body,
                "elapsed_s": elapsed}
    j = r.json()
    d0 = (j.get("data") or [{}])[0]
    b64 = d0.get("b64_json")
    url_field = d0.get("url")
    if b64:
        raw = base64.b64decode(b64)
    elif url_field:
        raw = requests.get(url_field, timeout=60).content
    else:
        return {"label": label, "model": model, "size": size,
                "error": f"no b64/url; keys={list(d0.keys())}",
                "elapsed_s": elapsed}
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    safe = label.replace("/", "_")
    fname = f"{safe}__{model.replace('/', '_')}__{size}.png"
    path = OUT / fname
    path.write_bytes(raw)
    print(f"  OK image: {w}x{h}  aspect={round(w/h, 4)}  saved -> {path.name}")
    return {"label": label, "model": model, "size": size,
            "actual": [w, h], "aspect": round(w / h, 4),
            "path": str(path), "elapsed_s": elapsed}


tests = [
    # shangliu — dated gpt-image-2 first (most likely real)
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-2-2026-04-21", "2048x1024"),
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-2-2026-04-21", "1024x512"),
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-2-2026-04-21", "1536x1024"),
    # bare gpt-image-2 (might be alias to 1.5)
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-2", "2048x1024"),
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-2", "1024x512"),
    # control: 1.5 must reject 2048x1024
    ("shangliu", "https://super.shangliu.org/v1", SHANGLIU_KEY, "gpt-image-1.5", "2048x1024"),
]

results = []
for label, base, key, model, size in tests:
    keys_for_label = {"shangliu": SHANGLIU_KEY, "openrouter": OPENROUTER_KEY}
    r = probe_images_generate(label, base, keys_for_label[label], model, size)
    results.append(r)
    # If we already found a working 2:1 with a model id, no need to try more sizes for it.
    if r.get("actual") and r["actual"][0] / max(r["actual"][1], 1) >= 1.95:
        print(f"  >>> WIN: {model} produced {r['actual']} aspect {r['aspect']}, true 2:1")

(OUT / "image2_real_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n\n===== SUMMARY =====")
print(f"{'label':10s}  {'model':30s}  {'requested':12s}  actual          aspect  status")
for r in results:
    if r.get("actual"):
        s = r["actual"]
        sz = f"{s[0]}x{s[1]}"
        print(f"{r['label']:10s}  {r['model']:30s}  {r['size']:12s}  {sz:14s}  {r['aspect']:.4f}  OK")
    else:
        err = (r.get("error_body") or r.get("error") or "?")[:60]
        print(f"{r['label']:10s}  {r['model']:30s}  {r['size']:12s}  FAIL: {err}")

print()
winners = [r for r in results if r.get("actual") and (r["actual"][0] / max(r["actual"][1], 1)) >= 1.95]
if winners:
    print("=== TRUE 2:1 WINNERS ===")
    for w in winners:
        print(f"  {w['label']}/{w['model']}  size={w['size']}  actual={w['actual'][0]}x{w['actual'][1]}  ({w['path']})")
else:
    print("NO TRUE 2:1 WINNERS. The endpoint is alias-ing to gpt-image-1.5 (max landscape 1536x1024).")
