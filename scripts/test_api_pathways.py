#!/usr/bin/env python3
"""Probe OpenAI / OpenAI-compatible / OpenRouter API pathways used by EGMOR.

**Security**
  * Never pass API keys on the command line (shell history). Use env vars only.
  * This script does not print secrets. Error strings are scrubbed if they look
    like tokens.

Env vars (set what you need):
  OPENAI_API_KEY          — required for official + relay image/text tests (unless
                            ``openai.api_key`` is set in merged YAML, e.g. secrets.local.yaml)
  OPENAI_COMPAT_BASE_URL    — optional; if set, also tests Images API against
                              this base (e.g. https://api.chatfire.cn/v1)
  OPENROUTER_API_KEY      — optional; for OpenRouter chat-image path (often sk-or-v1-...)
  OPENROUTER_BASE_URL     — optional; default https://openrouter.ai/api/v1

Example (PowerShell):
  $env:OPENAI_API_KEY = "sk-..."
  python scripts/test_api_pathways.py --out-json outputs/_api_pathway_test.json

Example with compat relay in addition to official:
  $env:OPENAI_API_KEY = "sk-..."
  $env:OPENAI_COMPAT_BASE_URL = "https://example.com/v1"
  python scripts/test_api_pathways.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _scrub(s: str | None, max_len: int = 500) -> str:
    if not s:
        return ""
    t = str(s)
    t = re.sub(r"sk-(?:or-v1-|proj-)[A-Za-z0-9_-]{20,}", "[REDACTED_TOKEN]", t)
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def _tiny_png_rgb(w: int = 128, h: int = 128) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), color=(200, 210, 220))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tiny_png_rgba_mask_edit_corner(w: int = 256, h: int = 256) -> tuple[bytes, bytes]:
    """RGB base image + RGBA mask (alpha=0 on a small patch = edit region)."""
    from PIL import Image

    rgb = Image.new("RGB", (w, h), color=(180, 160, 140))
    m = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    # 32x32 transparent square top-left = inpaint region
    for y in range(32):
        for x in range(32):
            m.putpixel((x, y), (0, 0, 0, 0))
    b1 = io.BytesIO()
    rgb.save(b1, format="PNG")
    b2 = io.BytesIO()
    m.save(b2, format="PNG")
    return b1.getvalue(), b2.getvalue()


def _decode_first_image(resp) -> tuple[int, int]:
    from PIL import Image

    d0 = resp.data[0]
    b64 = getattr(d0, "b64_json", None)
    url = getattr(d0, "url", None)
    if b64:
        raw = base64.b64decode(b64)
    elif url:
        import urllib.request

        raw = urllib.request.urlopen(url, timeout=120).read()
    else:
        raise RuntimeError(f"no b64_json or url: {d0!r}")
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return im.size


def _run_case(name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"name": name, "ok": False}
    try:
        detail = fn()
        rec["ok"] = True
        rec["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        if isinstance(detail, dict):
            rec.update(detail)
        elif detail is not None:
            rec["detail"] = detail
    except Exception as exc:
        rec["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        rec["error_type"] = type(exc).__name__
        rec["error"] = _scrub(f"{exc}")
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description="Test API pathways; JSON to stdout.")
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="If set, also write results to this path (relative to repo root ok).",
    )
    p.add_argument(
        "--image-model",
        default=os.environ.get("TEST_IMAGE_MODEL", "gpt-image-2"),
        help="Model id for images.generate / images.edit",
    )
    p.add_argument(
        "--text-model",
        default=os.environ.get("TEST_TEXT_MODEL", "gpt-4o-mini"),
        help="Cheap model for chat.completions smoke test",
    )
    p.add_argument(
        "--with-openrouter",
        action="store_true",
        help="Run OpenRouter chat-image test if OPENROUTER_API_KEY is set.",
    )
    p.add_argument(
        "--openrouter-model",
        default=os.environ.get("TEST_OR_IMAGE_MODEL", "openai/gpt-image-2"),
    )
    args = p.parse_args()

    oa_key = ""
    try:
        from erpgen.config import load_config

        oa_key = str(load_config().openai.get("api_key") or "").strip()
    except Exception:
        pass
    if not oa_key:
        oa_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    compat_url = (os.environ.get("OPENAI_COMPAT_BASE_URL") or "").strip()
    or_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    or_base = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()

    results: list[dict[str, Any]] = []
    meta = {
        "has_OPENAI_API_KEY": bool(oa_key),
        "has_OPENAI_COMPAT_BASE_URL": bool(compat_url),
        "has_OPENROUTER_API_KEY": bool(or_key),
        "image_model": args.image_model,
        "text_model": args.text_model,
    }

    from openai import OpenAI

    # ---- 1) Official host: images.generate ----
    if oa_key:

        def gen_official():
            cli = OpenAI(api_key=oa_key, timeout=600.0)
            kw: dict = {
                "model": args.image_model,
                "prompt": "Minimal flat color abstract square, two pastel tones, no text, no people.",
                "size": "1024x1024",
                "n": 1,
            }
            try:
                kw["quality"] = "low"
                r = cli.images.generate(**kw)
            except TypeError:
                kw.pop("quality", None)
                r = cli.images.generate(**kw)
            w, h = _decode_first_image(r)
            return {"out_size": [w, h], "endpoint": "images.generate"}

        results.append(_run_case("official_openai_images_generate", gen_official))

        # ---- 2) Official: images.edit (whole image i2i) ----
        def edit_i2i():
            cli = OpenAI(api_key=oa_key, timeout=600.0)
            img_bytes = _tiny_png_rgb(256, 256)
            bio = io.BytesIO(img_bytes)
            bio.name = "ref.png"
            kw: dict = {
                "model": args.image_model,
                "image": bio,
                "prompt": "Same composition, slightly warmer tones, still abstract, no text.",
                "size": "1024x1024",
                "n": 1,
            }
            try:
                kw["quality"] = "low"
                r = cli.images.edit(**kw)
            except TypeError:
                kw.pop("quality", None)
                bio.seek(0)
                r = cli.images.edit(**kw)
            w, h = _decode_first_image(r)
            return {"out_size": [w, h], "endpoint": "images.edit_i2i"}

        results.append(_run_case("official_openai_images_edit_i2i", edit_i2i))

        # ---- 3) Official: images.edit + mask ----
        def edit_mask():
            cli = OpenAI(api_key=oa_key, timeout=600.0)
            rgb_b, mask_b = _tiny_png_rgba_mask_edit_corner(256, 256)
            img_io = io.BytesIO(rgb_b)
            img_io.name = "image.png"
            m_io = io.BytesIO(mask_b)
            m_io.name = "mask.png"
            kw: dict = {
                "model": args.image_model,
                "image": img_io,
                "mask": m_io,
                "prompt": "Seamlessly fill the transparent region with a soft wood texture matching the scene.",
                "size": "1024x1024",
                "n": 1,
            }
            try:
                kw["quality"] = "low"
                r = cli.images.edit(**kw)
            except TypeError:
                kw.pop("quality", None)
                img_io.seek(0)
                m_io.seek(0)
                r = cli.images.edit(**kw)
            w, h = _decode_first_image(r)
            return {"out_size": [w, h], "endpoint": "images.edit_mask"}

        results.append(_run_case("official_openai_images_edit_mask", edit_mask))

        # ---- 4) Official: chat.completions text (cheap) ----
        def chat_text():
            cli = OpenAI(api_key=oa_key, timeout=120.0)
            r = cli.chat.completions.create(
                model=args.text_model,
                messages=[{"role": "user", "content": 'Reply with exactly: {"ok": true}'}],
                max_tokens=32,
            )
            txt = (r.choices[0].message.content or "").strip()
            return {"endpoint": "chat.completions", "reply_preview": _scrub(txt, 120)}

        results.append(_run_case("official_openai_chat_completions_text", chat_text))

    else:
        results.append(
            {
                "name": "official_openai_*",
                "ok": False,
                "skipped": True,
                "error": "OPENAI_API_KEY not set",
            }
        )

    # ---- 5) Optional compat relay: same key, custom base_url ----
    if oa_key and compat_url:

        def gen_compat():
            cli = OpenAI(api_key=oa_key, base_url=compat_url, timeout=600.0)
            kw: dict = {
                "model": args.image_model,
                "prompt": "Small abstract gradient, no text.",
                "size": "1024x1024",
                "n": 1,
            }
            try:
                kw["quality"] = "low"
                r = cli.images.generate(**kw)
            except TypeError:
                kw.pop("quality", None)
                r = cli.images.generate(**kw)
            w, h = _decode_first_image(r)
            return {
                "out_size": [w, h],
                "endpoint": "images.generate",
                "base_url": compat_url,
            }

        results.append(_run_case("compat_relay_images_generate", gen_compat))
    elif compat_url and not oa_key:
        results.append(
            {
                "name": "compat_relay_images_generate",
                "ok": False,
                "skipped": True,
                "error": "OPENAI_COMPAT_BASE_URL set but OPENAI_API_KEY missing",
            }
        )

    # ---- 6) OpenRouter: chat + image modality ----
    if args.with_openrouter and or_key:

        def or_chat_image():
            cli = OpenAI(
                api_key=or_key,
                base_url=or_base,
                timeout=600.0,
                default_headers={
                    "HTTP-Referer": os.environ.get(
                        "OPENROUTER_HTTP_REFERER",
                        "https://github.com/Strange-animalss/EGMOR",
                    ),
                    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "EGMOR-pathway-test"),
                },
            )
            r = cli.chat.completions.create(
                model=args.openrouter_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Tiny abstract icon, flat colors,  no text, single centered shape.",
                            }
                        ],
                    }
                ],
                extra_body={"modalities": ["image", "text"]},
            )
            msg = r.choices[0].message
            images = getattr(msg, "images", None)
            if images is None and hasattr(msg, "model_extra"):
                images = (msg.model_extra or {}).get("images")
            if not images:
                raise RuntimeError("no images in chat response")
            first = images[0]
            url = first["image_url"]["url"] if isinstance(first, dict) else first.image_url.url
            if not str(url).startswith("data:"):
                raise RuntimeError(f"unexpected image url prefix: {str(url)[:40]}")
            b64 = str(url).split(",", 1)[1]
            from PIL import Image

            raw = base64.b64decode(b64)
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            return {
                "endpoint": "chat.completions+modalities",
                "base_url": or_base,
                "model": args.openrouter_model,
                "out_size": list(im.size),
            }

        results.append(_run_case("openrouter_chat_image", or_chat_image))
    elif args.with_openrouter and not or_key:
        results.append(
            {
                "name": "openrouter_chat_image",
                "ok": False,
                "skipped": True,
                "error": "--with-openrouter but OPENROUTER_API_KEY not set",
            }
        )

    payload = {"meta": meta, "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)

    if args.out_json:
        out = Path(args.out_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[test_api_pathways] wrote {out}", file=sys.stderr)

    # exit 1 if any non-skipped case failed
    failed = [r for r in results if not r.get("ok") and not r.get("skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
