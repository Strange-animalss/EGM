"""Probe super.shangliu.org as a potential OpenAI-compatible relay that
exposes /v1/images/generations (which OpenRouter does not).

T1: GET /v1/models                          - any image models?
T2: chat.completions on gpt-5.5-pro         - text path sanity
T3: images.generate size=1024x1024          - baseline image endpoint
T4: images.generate at multiple sizes       - which non-1:1 work?
T5: ERP-style cafe prompt at best size      - check pole std + seam score

Set ``SHANGLIU_API_KEY`` (or ``OPENAI_API_KEY`` if you point base_url elsewhere).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Force UTF-8 output on Windows GBK consoles
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("SHANGLIU_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set SHANGLIU_API_KEY or OPENAI_API_KEY")
BASE = "https://super.shangliu.org/v1"
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_shangliu_probe"
OUT.mkdir(parents=True, exist_ok=True)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _save_b64(label: str, b64: str) -> tuple[int, int, str]:
    raw = base64.b64decode(b64)
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


def _save_url(label: str, url: str) -> tuple[int, int, str]:
    raw = requests.get(url, timeout=60).content
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


# ---------------------------------------------------------------------------
# T1: list models
# ---------------------------------------------------------------------------
_print_section("T1: GET /v1/models")
t0 = time.time()
try:
    r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=30)
    print(f"status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
    print(f"content-type: {r.headers.get('content-type')}")
    if r.status_code == 200 and "json" in str(r.headers.get("content-type", "")).lower():
        j = r.json()
        items = j.get("data") or j.get("models") or j
        if isinstance(items, list):
            ids = sorted({m.get("id") if isinstance(m, dict) else str(m) for m in items})
        else:
            ids = sorted(items.keys()) if isinstance(items, dict) else []
        print(f"total models: {len(ids)}")
        # filter image-relevant
        image_models = [i for i in ids if any(
            k in (i or "").lower() for k in ["image", "dall", "sd", "flux", "ideogram"]
        )]
        text_models = [i for i in ids if any(
            k in (i or "").lower() for k in ["gpt-5", "gpt-4", "claude", "o1", "o3", "gemini"]
        )][:30]
        print(f"\nImage-relevant model IDs ({len(image_models)}):")
        for m in image_models:
            print(f"  {m}")
        print(f"\nSample text model IDs ({len(text_models)}):")
        for m in text_models:
            print(f"  {m}")
        (OUT / "T1_models.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    else:
        print(f"body preview: {r.text[:600]}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# T2: chat completion on gpt-5.5-pro
# ---------------------------------------------------------------------------
_print_section("T2: chat.completions on gpt-5.5-pro")
chat_models_to_try = ["gpt-5.5-pro", "gpt-5.5", "gpt-5-pro", "gpt-5", "gpt-4o"]
t2_winner = None
for m in chat_models_to_try:
    print(f"\n--- trying chat model: {m}")
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers=HEADERS,
            json={
                "model": m,
                "messages": [{"role": "user", "content": "Reply with exactly the word: ok"}],
                "max_tokens": 20,
            },
            timeout=60,
        )
        print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
        if r.status_code == 200:
            j = r.json()
            txt = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  reply: {txt[:80]!r}")
            if "ok" in (txt or "").lower():
                t2_winner = m
                break
        else:
            print(f"  body: {r.text[:300]}")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
print(f"\nT2 winner: {t2_winner}")


# ---------------------------------------------------------------------------
# T3: images.generate baseline (1024x1024)
# ---------------------------------------------------------------------------
_print_section("T3: POST /v1/images/generations  size=1024x1024")
image_models_to_try = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1", "dall-e-3", "dall-e-2"]
t3_winner = None
for m in image_models_to_try:
    print(f"\n--- trying image model: {m}")
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/images/generations",
            headers=HEADERS,
            json={
                "model": m,
                "prompt": "A clear blue sky over a calm sea, photographic quality.",
                "size": "1024x1024",
                "n": 1,
            },
            timeout=180,
        )
        print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and "json" in ct.lower():
            j = r.json()
            d0 = (j.get("data") or [{}])[0]
            b64 = d0.get("b64_json")
            url = d0.get("url")
            if b64:
                w, h, path = _save_b64(f"T3_{m}_1024", b64)
                print(f"  OK image: {w}x{h}  saved -> {path}")
            elif url:
                w, h, path = _save_url(f"T3_{m}_1024", url)
                print(f"  OK image: {w}x{h}  saved -> {path}")
            else:
                print(f"  no b64 / no url. data[0] keys: {list(d0.keys())}")
                continue
            t3_winner = m
            break
        else:
            print(f"  ct: {ct}")
            print(f"  body: {r.text[:400]}")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {str(e)[:200]}")
print(f"\nT3 winner: {t3_winner}")


# ---------------------------------------------------------------------------
# T4: explore size variants
# ---------------------------------------------------------------------------
_print_section("T4: explore non-1:1 sizes on the winning image model")
t4_results = []
if t3_winner:
    for size in ["1536x1024", "2048x1024", "1024x512", "1792x1024", "auto"]:
        print(f"\n--- size={size}")
        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/images/generations",
                headers=HEADERS,
                json={
                    "model": t3_winner,
                    "prompt": "A wide landscape photograph of mountains at sunset.",
                    "size": size,
                    "n": 1,
                },
                timeout=300,
            )
            print(f"  status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
            if r.status_code == 200:
                j = r.json()
                d0 = (j.get("data") or [{}])[0]
                if d0.get("b64_json"):
                    w, h, path = _save_b64(f"T4_{size.replace('x','_')}", d0["b64_json"])
                elif d0.get("url"):
                    w, h, path = _save_url(f"T4_{size.replace('x','_')}", d0["url"])
                else:
                    print(f"  no b64/url; keys={list(d0.keys())}")
                    t4_results.append({"size": size, "error": "no_image"})
                    continue
                print(f"  OK image: {w}x{h}  aspect={round(w/h,4)}")
                t4_results.append({
                    "requested": size, "actual": [w, h],
                    "aspect": round(w / h, 4), "path": path,
                    "elapsed_s": round(time.time() - t0, 1),
                })
            else:
                body = r.text[:400]
                print(f"  body: {body}")
                t4_results.append({"size": size, "status": r.status_code,
                                   "error_body": body})
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {str(e)[:200]}")
            t4_results.append({"size": size, "error": str(e)[:200]})
else:
    print("(skipped: no T3 winner)")


# ---------------------------------------------------------------------------
# T5: ERP cafe at best landscape size + pole/seam analysis
# ---------------------------------------------------------------------------
_print_section("T5: ERP cafe prompt at the best non-1:1 size")
ok_t4 = [r for r in t4_results if isinstance(r.get("actual"), list)]
best_size = None
if ok_t4:
    # pick widest aspect (most ERP-like)
    best = max(ok_t4, key=lambda r: r["aspect"])
    best_size = best["requested"]
    print(f"chosen size for T5: {best_size}  (aspect={best['aspect']})")
elif t3_winner:
    # fallback
    best_size = "1024x1024"
    print("no non-1:1 size succeeded; falling back to 1024x1024 for T5 sanity")

if best_size and t3_winner:
    cafe_prompt = (
        "A photorealistic 360 degree equirectangular panorama of an empty "
        "specialty coffee shop interior in late morning. Industrial Scandinavian "
        "style: exposed concrete ceiling, whitewashed brick walls, polished oak "
        "floor, long pale-oak espresso bar with stainless espresso machine, glass "
        "pastry case, communal oak table with bentwood chairs. Equirectangular 360 "
        "panorama, 2:1 aspect ratio, full sphere, seamless horizontal wraparound, "
        "no people, no text, no watermark."
    )
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/images/generations",
            headers=HEADERS,
            json={
                "model": t3_winner,
                "prompt": cafe_prompt,
                "size": best_size,
                "n": 1,
            },
            timeout=300,
        )
        print(f"status: {r.status_code}  elapsed={round(time.time()-t0,1)}s")
        if r.status_code == 200:
            j = r.json()
            d0 = (j.get("data") or [{}])[0]
            if d0.get("b64_json"):
                w, h, path = _save_b64("T5_cafe_erp", d0["b64_json"])
            elif d0.get("url"):
                w, h, path = _save_url("T5_cafe_erp", d0["url"])
            else:
                print("no image data")
                w = h = None
                path = None
            if w is not None:
                print(f"OK image: {w}x{h}  aspect={round(w/h,4)}  saved -> {path}")
                arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
                # pole rows
                row_std = arr.std(axis=(1, 2))
                top5 = [round(float(x), 2) for x in row_std[:5]]
                bot5 = [round(float(x), 2) for x in row_std[-5:]]
                eq = round(float(row_std[h // 2]), 2)
                # seam score
                seam = float(np.sqrt(((arr[:, 0] - arr[:, -1]) ** 2).mean()))
                halfdiff = float(np.sqrt(((arr[:, 0] - arr[:, w // 2]) ** 2).mean()))
                print(f"  pole std (top 5): {top5}    (true ERP would be near 0)")
                print(f"  pole std (bot 5): {bot5}    (true ERP would be near 0)")
                print(f"  equator std: {eq}            (high = good content)")
                print(f"  seam RMS col0 vs col(W-1): {round(seam,2)}  "
                      f"(< 15 = good wrap; baseline col0 vs colW/2 = {round(halfdiff,2)})")
                print(f"  wrap ratio: {round(seam/max(halfdiff,0.001),4)}  (smaller=cleaner)")
        else:
            print(f"body: {r.text[:400]}")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {str(e)[:300]}")


# ---------------------------------------------------------------------------
# Save full results
# ---------------------------------------------------------------------------
summary = {
    "base_url": BASE,
    "T2_chat_winner": t2_winner,
    "T3_image_winner": t3_winner,
    "T4_size_explore": t4_results,
}
(OUT / "shangliu_probe_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
print(f"summary -> {OUT / 'shangliu_probe_summary.json'}")
print(f"images  -> {OUT}")
