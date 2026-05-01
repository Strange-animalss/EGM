"""Comprehensive probe of how to pass aspect-ratio / size to gpt-image-2 via
OpenRouter. Tests 7 different parameter shapes (D1..D7) plus dumps the
exact request payload for D0 (current pipeline behavior) for comparison.

Each test reports:
  * which params we sent (for grep-able comparison with web dev-tools)
  * HTTP status
  * actual returned image dimensions (or error body)
  * saved PNG path

D7 is the critical test: it bypasses chat completions entirely and uses
/v1/images/generations through the OpenAI SDK pointed at OpenRouter.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import requests
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_size_probe"
OUT.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(str(REPO / "config" / "default.yaml"))
KEY = str(cfg.openai.api_key).strip()
MODEL = str(cfg.openai.model)
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = (
    "A photorealistic 360 degree equirectangular panorama of an empty "
    "specialty coffee shop interior. Equirectangular projection, 2:1 aspect "
    "ratio, no people, no text, no watermark."
)
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/erpgen",
    "X-Title": "EGMOR",
}


def _probe_chat(label: str, payload: dict) -> dict:
    print(f"\n=== {label} ===")
    extras = {k: v for k, v in payload.items() if k != "messages"}
    print(f"  request extras: {json.dumps(extras, ensure_ascii=False)[:300]}")
    try:
        r = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=300)
    except Exception as e:
        print(f"  REQUEST EXCEPTION: {e}")
        return {"label": label, "error": f"req: {e}"}
    print(f"  status: {r.status_code}")
    if r.status_code != 200:
        body = r.text[:600]
        print(f"  body: {body}")
        return {"label": label, "status": r.status_code, "error_body": body}
    j = r.json()
    msg = j.get("choices", [{}])[0].get("message", {})
    images = msg.get("images")
    if not images:
        cp = str(msg.get("content", ""))[:300]
        print(f"  no images. content={cp}")
        return {"label": label, "status": 200, "no_images": True, "content_preview": cp}
    url0 = images[0]["image_url"]["url"]
    raw = base64.b64decode(url0.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    (OUT / f"{label}.png").write_bytes(raw)
    print(f"  image: {w}x{h}  aspect={round(w/h, 4)}")
    return {"label": label, "status": 200, "size": [w, h], "aspect": round(w / h, 4)}


def _probe_sdk_images_generate(label: str, size: str) -> dict:
    """D7: bypass chat completions, hit /v1/images/generations through the
    OpenAI SDK pointed at OpenRouter."""
    print(f"\n=== {label} ===")
    print(f"  client.images.generate(model={MODEL!r}, size={size!r})")
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(
            api_key=KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/erpgen",
                "X-Title": "EGMOR",
            },
            timeout=300,
        )
        resp = client.images.generate(
            model=MODEL, prompt=PROMPT, size=size, n=1,
        )
    except Exception as e:
        print(f"  SDK EXCEPTION: {type(e).__name__}: {e}")
        return {"label": label, "error": f"{type(e).__name__}: {e}"}
    try:
        b64 = resp.data[0].b64_json
        if not b64:
            url = getattr(resp.data[0], "url", None)
            if url:
                # download
                r = requests.get(url, timeout=60)
                raw = r.content
            else:
                return {"label": label, "error": "no b64 / no url"}
        else:
            raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        (OUT / f"{label}.png").write_bytes(raw)
        print(f"  image: {w}x{h}  aspect={round(w/h, 4)}")
        return {"label": label, "status": 200, "size": [w, h], "aspect": round(w / h, 4)}
    except Exception as e:
        print(f"  parse error: {e}")
        return {"label": label, "error": f"parse: {e}"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

base_msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]

tests_chat: list[tuple[str, dict]] = [
    # D0: current pipeline behavior (size_hint in prompt + modalities, no extras)
    # We dump this so user can compare to the web dev-tools payload.
    ("D0_current_pipeline", {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": PROMPT + "\n\nThe final image must be a 2:1 equirectangular panorama (target 1024x512).",
            }],
        }],
        "modalities": ["image", "text"],
    }),
    # D1: extra_body size at top
    ("D1_extra_size_1536x1024", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "size": "1536x1024",
    }),
    # D2: image_size at top
    ("D2_image_size_1536x1024", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "image_size": "1536x1024",
    }),
    # D3: nested image.size
    ("D3_nested_image_size_1536x1024", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "image": {"size": "1536x1024"},
    }),
    # D4: tools image_generation
    ("D4_tools_image_generation_1536x1024", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "tools": [{"type": "image_generation", "size": "1536x1024", "quality": "high"}],
    }),
    # D5: Responses-API style nested image_size in tool
    ("D5_tools_image_generation_image_size", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "tools": [{"type": "image_generation", "image_size": "1536x1024"}],
    }),
    # D6: provider routing with size override
    ("D6_provider_routing_size", {
        "model": MODEL,
        "messages": base_msg,
        "modalities": ["image", "text"],
        "provider": {"openai": {"size": "1536x1024"}},
    }),
]

results = []

print("===== D0: dump our CURRENT payload =====")
print(json.dumps(tests_chat[0][1], indent=2, ensure_ascii=False)[:1200])
print("(this is what we currently send -- compare to web dev-tools payload)")

for label, payload in tests_chat:
    results.append(_probe_chat(label, payload))

# D7: SDK images.generate route (the most likely-to-work path)
results.append(_probe_sdk_images_generate("D7_sdk_images_generate_1536x1024", "1536x1024"))
results.append(_probe_sdk_images_generate("D7b_sdk_images_generate_2048x1024", "2048x1024"))
results.append(_probe_sdk_images_generate("D7c_sdk_images_generate_1024x512", "1024x512"))


(OUT / "size_probe_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n\n===== SUMMARY =====")
print(f"{'label':45s}  size           aspect    status")
for r in results:
    if r.get("status") == 200 and "size" in r:
        s = r["size"]
        sz = f"{s[0]}x{s[1]}"
        print(f"{r['label']:45s}  {sz:14s}  {r.get('aspect'):.4f}  OK")
    else:
        err = r.get("error_body") or r.get("error") or ("no_image" if r.get("no_images") else "?")
        print(f"{r['label']:45s}  FAIL: {str(err)[:80]}")
