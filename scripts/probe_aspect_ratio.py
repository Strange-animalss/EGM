"""Test which OpenRouter image_config / size parameters actually work for
openai/gpt-5.4-image-2.

We send the same prompt with different parameter shapes and record the
returned raw image dimensions.  6 priority tests only.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import requests
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_aspect_probe"
OUT.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(str(REPO / "config" / "default.yaml"))
KEY = str(cfg.openai.api_key).strip()
MODEL = str(cfg.openai.model)
URL = "https://openrouter.ai/api/v1/chat/completions"

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


def call(label: str, payload: dict) -> dict:
    print(f"\n=== {label} ===")
    extras = {k: v for k, v in payload.items() if k != "messages"}
    print(f"extras: {json.dumps(extras, ensure_ascii=False)[:200]}")
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=300)
    except Exception as e:
        print(f"REQUEST EXCEPTION: {e}")
        return {"label": label, "error": f"request_exception: {e}"}
    print(f"status: {r.status_code}")
    if r.status_code != 200:
        body = r.text[:600]
        print(f"body: {body}")
        return {"label": label, "status": r.status_code, "error_body": body}
    j = r.json()
    msg = j.get("choices", [{}])[0].get("message", {})
    images = msg.get("images")
    if not images:
        print(f"no images. content={str(msg.get('content', ''))[:300]}")
        return {"label": label, "status": 200, "no_images": True,
                "content_preview": str(msg.get("content", ""))[:300]}
    url0 = images[0]["image_url"]["url"]
    raw = base64.b64decode(url0.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    (OUT / f"{label}.png").write_bytes(raw)
    print(f"image size: {w}x{h}  aspect={round(w/h, 4)}")
    return {"label": label, "status": 200, "size": [w, h], "aspect": round(w / h, 4)}


tests = [
    ("P0_baseline_no_extras", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
    }),
    # OpenRouter documented image_config approach
    ("P1_imageconfig_21x9", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "21:9"},
    }),
    ("P2_imageconfig_16x9_2K", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "16:9", "image_size": "2K"},
    }),
    # OpenAI-style top-level size  (used by /v1/images/generations classic)
    ("P3_topsize_1792x1024", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
        "size": "1792x1024",
    }),
    # OpenAI Responses API style: image_generation tool with size
    ("P4_tool_image_generation_2x1", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
        "tools": [{"type": "image_generation", "image_generation": {"size": "2048x1024"}}],
    }),
    # 2:1 native (unsupported by docs but worth a try)
    ("P5_imageconfig_2x1", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "2:1"},
    }),
]

results = []
for label, payload in tests:
    r = call(label, payload)
    results.append(r)

(OUT / "aspect_probe_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\n\n=== SUMMARY ===")
print(f"{'label':35s}  size           aspect")
for r in results:
    if r.get("status") == 200 and "size" in r:
        s = r["size"]
        print(f"{r['label']:35s}  {s[0]}x{s[1]:<6}  {r.get('aspect')}")
    else:
        err = r.get("error_body", r.get("error", "no_image"))
        print(f"{r['label']:35s}  FAIL  ({str(err)[:60]})")
