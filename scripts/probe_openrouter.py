"""Probe OpenRouter for image-capable models and verify the chat-based image API."""
from __future__ import annotations

import base64
import io
import os
import sys

import requests
from PIL import Image

KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/erpgen",
    "X-Title": "ERPGen",
}


def list_image_models() -> None:
    r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=30)
    data = r.json()["data"]
    print(f"Total: {len(data)} models")
    print()
    print("=== Models with image OUTPUT capability ===")
    for m in data:
        arch = m.get("architecture", {})
        out_mods = arch.get("output_modalities", [])
        if "image" in out_mods:
            in_mods = arch.get("input_modalities", [])
            print(f"  {m['id']:50s}  in={in_mods}  out={out_mods}")
    print()
    print("=== id contains 'image' ===")
    for m in data:
        if "image" in m["id"].lower():
            print(f"  {m['id']}")


def gen_image(model: str, prompt: str, ref_b64: str | None = None) -> Image.Image | None:
    msg_content: list = [{"type": "text", "text": prompt}]
    if ref_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
        })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": msg_content}],
        "modalities": ["image", "text"],
    }
    r = requests.post(f"{BASE}/chat/completions", json=payload, headers=HEADERS, timeout=120)
    print(f"  [{model}] status={r.status_code}")
    if r.status_code != 200:
        print(f"    body: {r.text[:300]}")
        return None
    j = r.json()
    msg = j["choices"][0]["message"]
    print(f"    keys: {list(msg.keys())}")
    if "images" not in msg or not msg["images"]:
        print(f"    no images returned. content: {str(msg.get('content'))[:200]}")
        return None
    url = msg["images"][0]["image_url"]["url"]
    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        print(f"    image size: {img.size}")
        return img
    return None


def main() -> int:
    if not KEY:
        print("Set OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_image_models()
        return 0

    print("=== Test image generation (no ref) ===")
    img = gen_image("openai/gpt-5.4-image-2", "A simple red cube on white background.")
    if img is None:
        return 1
    img.save("outputs/_probe_gen.png")
    print("  saved outputs/_probe_gen.png")

    print()
    print("=== Test image generation WITH ref image (edit-style) ===")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ref_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    img2 = gen_image(
        "openai/gpt-5.4-image-2",
        "Take this image and add a small yellow star in the top-right corner. Keep the rest identical.",
        ref_b64=ref_b64,
    )
    if img2:
        img2.save("outputs/_probe_edit.png")
        print("  saved outputs/_probe_edit.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
