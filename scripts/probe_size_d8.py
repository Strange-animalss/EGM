"""Focused probe: D8 (size='auto' via images.generate) is the highest-priority test.
After D8 we test D7 (size='1536x1024' via images.generate) and a handful of
chat-completions variants. Each call is logged with full payload + result.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests
from omegaconf import OmegaConf
from PIL import Image

# Force UTF-8 output on Windows GBK consoles
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_size_probe"
OUT.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(str(REPO / "config" / "default.yaml"))
KEY = str(cfg.openai.api_key).strip()
MODEL = str(cfg.openai.model)
PROMPT = (
    "Generate a photorealistic 360 degree equirectangular panorama "
    "(2:1 aspect ratio) of an empty specialty coffee shop interior. "
    "Equirectangular projection, full sphere, seamless horizontal wraparound, "
    "no people, no text, no watermark."
)
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/erpgen",
    "X-Title": "EGMOR",
}


def _save(label: str, raw: bytes) -> tuple[int, int, str]:
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


def _sdk_images_generate(label: str, size: str) -> dict:
    """Hit /v1/images/generations via raw requests. Returns whatever response
    we got, including 404 / route-not-found / actual image / error JSON."""
    print(f"\n=== {label} ===")
    print(f"  POST /v1/images/generations  size={size!r}")
    t0 = time.time()
    url = "https://openrouter.ai/api/v1/images/generations"
    payload = {"model": MODEL, "prompt": PROMPT, "size": size, "n": 1}
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=300)
    except Exception as e:
        print(f"  REQUEST EXCEPTION: {e}")
        return {"label": label, "error": f"req: {e}",
                "elapsed_s": round(time.time() - t0, 1)}
    print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
    if r.status_code != 200:
        body = r.text[:600]
        print(f"  body: {body}")
        return {"label": label, "status": r.status_code, "error_body": body,
                "elapsed_s": round(time.time() - t0, 1)}
    try:
        j = r.json()
    except Exception as e:
        return {"label": label, "error": f"json parse: {e}; body={r.text[:300]}",
                "elapsed_s": round(time.time() - t0, 1)}
    data = j.get("data") or []
    if not data:
        return {"label": label, "error": f"no data; body keys={list(j.keys())}",
                "raw_body_preview": str(j)[:400],
                "elapsed_s": round(time.time() - t0, 1)}
    d = data[0]
    b64 = d.get("b64_json")
    url_field = d.get("url")
    if b64:
        raw = base64.b64decode(b64)
    elif url_field:
        raw = requests.get(url_field, timeout=60).content
    else:
        return {"label": label, "error": f"no b64/url in data[0]; keys={list(d.keys())}",
                "elapsed_s": round(time.time() - t0, 1)}
    w, h, path = _save(label, raw)
    print(f"  OK image: {w}x{h}  aspect={round(w/h, 4)}  elapsed={round(time.time()-t0,1)}s")
    return {"label": label, "status": 200, "size": [w, h],
            "aspect": round(w / h, 4), "path": path,
            "elapsed_s": round(time.time() - t0, 1)}


def _chat(label: str, payload: dict) -> dict:
    print(f"\n=== {label} ===")
    extras = {k: v for k, v in payload.items() if k != "messages"}
    print(f"  extras: {json.dumps(extras, ensure_ascii=False)[:200]}")
    t0 = time.time()
    try:
        r = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=300)
    except Exception as e:
        print(f"  REQUEST EXCEPTION: {e}")
        return {"label": label, "error": f"req: {e}",
                "elapsed_s": round(time.time() - t0, 1)}
    print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
    if r.status_code != 200:
        body = r.text[:500]
        print(f"  body: {body}")
        return {"label": label, "status": r.status_code, "error_body": body,
                "elapsed_s": round(time.time() - t0, 1)}
    j = r.json()
    msg = j.get("choices", [{}])[0].get("message", {})
    images = msg.get("images")
    if not images:
        return {"label": label, "status": 200, "no_images": True,
                "content_preview": str(msg.get("content", ""))[:200],
                "elapsed_s": round(time.time() - t0, 1)}
    raw = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    w, h, path = _save(label, raw)
    print(f"  OK image: {w}x{h}  aspect={round(w/h, 4)}")
    return {"label": label, "status": 200, "size": [w, h],
            "aspect": round(w / h, 4), "path": path,
            "elapsed_s": round(time.time() - t0, 1)}


# ---- ordered by priority ----

results = []

# D8 *** HIGHEST PRIORITY ***
results.append(_sdk_images_generate("D8_sdk_images_generate_AUTO", "auto"))

# D7 next-highest
results.append(_sdk_images_generate("D7_sdk_images_generate_1536x1024", "1536x1024"))

# Chat-completions auto variants
results.append(_chat("D9a_chat_extra_size_auto", {
    "model": MODEL,
    "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
    "modalities": ["image", "text"],
    "size": "auto",
}))
results.append(_chat("D9b_chat_image_size_auto", {
    "model": MODEL,
    "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
    "modalities": ["image", "text"],
    "image_size": "auto",
}))
results.append(_chat("D9c_chat_tools_image_generation_auto", {
    "model": MODEL,
    "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
    "modalities": ["image", "text"],
    "tools": [{"type": "image_generation", "size": "auto"}],
}))

(OUT / "size_probe_d8.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n\n===== SUMMARY =====")
print(f"{'label':45s}  size           aspect    elapsed")
for r in results:
    if r.get("status") == 200 and "size" in r:
        s = r["size"]
        sz = f"{s[0]}x{s[1]}"
        print(f"{r['label']:45s}  {sz:14s}  {r.get('aspect'):.4f}  {r.get('elapsed_s')}s")
    else:
        err = r.get("error_body") or r.get("error") or ("no_image" if r.get("no_images") else "?")
        print(f"{r['label']:45s}  FAIL: {str(err)[:80]}  ({r.get('elapsed_s')}s)")
