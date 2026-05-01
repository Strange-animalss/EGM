"""Probe https://api.chatfire.cn/v1 as a candidate replacement for super.shangliu.

T1: GET /v1/models                                  - what's served?
T2: chat.completions on gpt-5.5-pro / gpt-5.5 / gpt-5
T3: images.generate @ 1024x1024 (baseline)
T4: images.generate @ 2048x1024 (real 2:1 ERP — the killer test)
T5: images.edit  @ 2048x1024 using T4's output (corner-pose ref-i2i)

Each call's status + actual returned dimensions + elapsed time + (on
failure) the error body are dumped to ``outputs/_chatfire_probe.log``,
and any returned image PNG is saved into ``outputs/_chatfire_probe/``.

Set ``CHATFIRE_API_KEY`` or ``OPENAI_API_KEY`` in the environment before running.
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
from openai import OpenAI
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_chatfire_probe"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("CHATFIRE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set CHATFIRE_API_KEY or OPENAI_API_KEY")
BASE = "https://api.chatfire.cn/v1"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save_b64(label: str, b64: str) -> tuple[int, int, str]:
    raw = base64.b64decode(b64)
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


def save_url(label: str, url: str) -> tuple[int, int, str]:
    raw = requests.get(url, timeout=120).content
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


# ---------------------------------------------------------------------------
# T1 — GET /v1/models
# ---------------------------------------------------------------------------
header("T1: GET /v1/models")
t0 = time.time()
try:
    r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=30)
    print(f"status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
    print(f"content-type: {r.headers.get('content-type')}")
    if r.status_code == 200 and "json" in str(r.headers.get("content-type", "")).lower():
        j = r.json()
        items = j.get("data") or j
        if isinstance(items, list):
            ids = sorted({m.get("id") if isinstance(m, dict) else str(m) for m in items})
        else:
            ids = sorted(items.keys()) if isinstance(items, dict) else []
        print(f"total models: {len(ids)}")
        image_models = [i for i in ids if any(
            k in (i or "").lower() for k in ["image", "dall", "sd", "flux"]
        )]
        target_text = [i for i in ids if any(
            k in (i or "").lower() for k in ["gpt-5.5", "gpt-5-", "gpt-5_"]
        )]
        print(f"\nImage-relevant model IDs ({len(image_models)}):")
        for m in image_models:
            print(f"  {m}")
        print(f"\nGPT-5.x text model IDs ({len(target_text)}):")
        for m in target_text:
            print(f"  {m}")
        (OUT / "T1_models.json").write_text(
            json.dumps(ids, indent=2), encoding="utf-8"
        )
    else:
        print(f"body preview: {r.text[:600]}")
except Exception as e:
    print(f"EXC: {e}")


# ---------------------------------------------------------------------------
# T2 — chat
# ---------------------------------------------------------------------------
header("T2: chat.completions (find a working text model)")
chat_winner: str | None = None
for m in ["gpt-5.5-pro", "gpt-5.5", "gpt-5-pro", "gpt-5", "gpt-4o"]:
    print(f"\n--- chat model: {m}")
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers=HEADERS,
            json={
                "model": m,
                "messages": [{"role": "user", "content": "Reply with exactly the word: OK"}],
                "max_tokens": 20,
            },
            timeout=60,
        )
        print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
        if r.status_code == 200:
            j = r.json()
            txt = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  reply: {txt[:60]!r}")
            if "ok" in (txt or "").lower():
                chat_winner = m
                break
        else:
            print(f"  body: {r.text[:300]}")
    except Exception as e:
        print(f"  EXC: {e}")
print(f"\nT2 winner: {chat_winner}")


# ---------------------------------------------------------------------------
# Build SDK client for T3-T5
# ---------------------------------------------------------------------------
client = OpenAI(api_key=KEY, base_url=BASE, timeout=300)


def _sdk_generate(label: str, model: str, size: str, prompt: str) -> dict:
    print(f"\n--- {label}  model={model!r}  size={size!r}")
    t0 = time.time()
    try:
        resp = client.images.generate(model=model, prompt=prompt, size=size, n=1)
    except Exception as e:
        print(f"  EXC: {type(e).__name__}: {str(e)[:300]}")
        return {"label": label, "model": model, "size": size,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "elapsed_s": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    url = getattr(d, "url", None)
    if b64:
        w, h, path = save_b64(label, b64)
    elif url:
        w, h, path = save_url(label, url)
    else:
        return {"label": label, "model": model, "size": size,
                "error": "no b64/url", "elapsed_s": elapsed}
    print(f"  OK image: {w}x{h}  aspect={round(w/h,4)}  elapsed={elapsed}s -> {path}")
    return {"label": label, "model": model, "size": size,
            "actual": [w, h], "aspect": round(w / h, 4),
            "path": path, "elapsed_s": elapsed}


def _sdk_edit(label: str, model: str, size: str, ref_path: str, prompt: str) -> dict:
    print(f"\n--- {label}  model={model!r}  size={size!r}  ref={Path(ref_path).name}")
    t0 = time.time()
    try:
        with open(ref_path, "rb") as f:
            resp = client.images.edit(
                model=model, prompt=prompt, image=f, size=size, n=1,
            )
    except Exception as e:
        print(f"  EXC: {type(e).__name__}: {str(e)[:300]}")
        return {"label": label, "model": model, "size": size,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "elapsed_s": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    url = getattr(d, "url", None)
    if b64:
        w, h, path = save_b64(label, b64)
    elif url:
        w, h, path = save_url(label, url)
    else:
        return {"label": label, "model": model, "size": size,
                "error": "no b64/url", "elapsed_s": elapsed}
    print(f"  OK image: {w}x{h}  aspect={round(w/h,4)}  elapsed={elapsed}s -> {path}")
    return {"label": label, "model": model, "size": size,
            "actual": [w, h], "aspect": round(w / h, 4),
            "path": path, "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# T3 — baseline 1024x1024
# ---------------------------------------------------------------------------
header("T3: images.generate @ 1024x1024")
t3 = _sdk_generate(
    "T3_image2_1024x1024", "gpt-image-2", "1024x1024",
    "A simple photographic test image of a clear blue sky.",
)


# ---------------------------------------------------------------------------
# T4 — real 2:1
# ---------------------------------------------------------------------------
header("T4: images.generate @ 2048x1024 (real 2:1 ERP)")
t4_main = _sdk_generate(
    "T4_image2_2048x1024", "gpt-image-2", "2048x1024",
    "A photorealistic equirectangular 360 panorama of a simple empty cafe "
    "interior. Equirectangular projection, 2:1 aspect, full sphere, seamless "
    "horizontal wraparound. No people, no text, no watermark.",
)
# also try the dated variant
t4_dated = _sdk_generate(
    "T4_image2_dated_2048x1024", "gpt-image-2-2026-04-21", "2048x1024",
    "A photorealistic equirectangular 360 panorama of a simple empty cafe "
    "interior. Equirectangular projection, 2:1 aspect, full sphere, seamless "
    "horizontal wraparound. No people, no text, no watermark.",
)


# ---------------------------------------------------------------------------
# T5 — edit @ 2048x1024 using T4 output as ref
# ---------------------------------------------------------------------------
header("T5: images.edit @ 2048x1024 (corner-pose ref-i2i)")
ref_path = None
if isinstance(t4_main, dict) and t4_main.get("path"):
    ref_path = t4_main["path"]
elif isinstance(t4_dated, dict) and t4_dated.get("path"):
    ref_path = t4_dated["path"]

if ref_path is None:
    print("(skipped: no T4 output available)")
    t5 = {"error": "no_ref"}
else:
    t5 = _sdk_edit(
        "T5_image2_edit_2048x1024", "gpt-image-2", "2048x1024", ref_path,
        "Same simple cafe interior viewed from a slightly different angle. "
        "Same materials, lighting, colour palette. Equirectangular 360 "
        "panorama, no people, no text, no watermark.",
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header("SUMMARY")
out = {
    "base_url": BASE,
    "T2_chat_winner": chat_winner,
    "T3": t3,
    "T4_main": t4_main,
    "T4_dated": t4_dated,
    "T5": t5,
}
(OUT / "chatfire_probe_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)

def fmt(r: dict) -> str:
    if r.get("actual"):
        return f"{r['actual'][0]}x{r['actual'][1]}  aspect={r['aspect']}  ({r['elapsed_s']}s)"
    err = (r.get("error") or "?")[:80]
    return f"FAIL: {err}  ({r.get('elapsed_s', '?')}s)"

print(f"  T3 (gen 1024)      : {fmt(t3)}")
print(f"  T4 (gen 2048x1024) : {fmt(t4_main)}")
print(f"  T4 dated 2048x1024 : {fmt(t4_dated)}")
print(f"  T5 (edit 2048x1024): {fmt(t5)}")
print(f"\nT2 chat winner: {chat_winner}")
print(f"\nfull summary -> {OUT / 'chatfire_probe_summary.json'}")
