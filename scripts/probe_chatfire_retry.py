"""Deeper probe of chatfire: distinguish "group has no gpt-image-2 access"
from "endpoint doesn't exist". Tests user/self + group endpoints, multiple
model-ID variants, and extra_body / header group-routing tricks.
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
OUT = REPO / "outputs" / "_chatfire_probe2"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("CHATFIRE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set CHATFIRE_API_KEY or OPENAI_API_KEY")
BASE = "https://api.chatfire.cn"
V1 = f"{BASE}/v1"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def decode_garbled(text: str) -> str:
    """Try to recover Chinese text mis-encoded as latin1->utf8 by mid-tier."""
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def header(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def get_json(label: str, url: str) -> dict | None:
    print(f"\n--- GET {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"  EXC: {e}")
        return None
    print(f"  status: {r.status_code}  ct={r.headers.get('content-type')}")
    if r.status_code == 200 and "json" in str(r.headers.get("content-type", "")).lower():
        try:
            j = r.json()
            (OUT / f"{label}.json").write_text(
                json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            preview = json.dumps(j, indent=2, ensure_ascii=False)[:1200]
            print(f"  body (first 1200 chars):\n{preview}")
            return j
        except Exception as e:
            print(f"  json parse: {e}")
    else:
        body = r.text[:600]
        body_dec = decode_garbled(body)
        print(f"  body: {body[:300]}")
        if body_dec != body[:len(body_dec)]:
            print(f"  body (decoded): {body_dec[:300]}")
    return None


# ---------------------------------------------------------------------------
# T1: user / token / group endpoints
# ---------------------------------------------------------------------------
header("T1: user / token / group introspection")
candidate_paths = [
    "/api/user/self",
    "/api/user/groups",
    "/api/user/groupinfo",
    "/api/group/list",
    "/api/group/check",
    "/api/channel/test",
    "/api/token/self",
    "/api/token/info",
    "/api/billing/subscription",
    "/api/billing/usage",
    "/dashboard/billing/usage",
    "/dashboard/billing/subscription",
    "/v1/dashboard/billing/usage",
    "/v1/dashboard/billing/subscription",
    "/api/user/info",
    "/api/user/me",
]
endpoint_results: dict = {}
for p in candidate_paths:
    res = get_json(p.replace("/", "_").strip("_"), BASE + p)
    if res:
        endpoint_results[p] = res


# ---------------------------------------------------------------------------
# T2: variant model IDs at 1024x1024 (cheap probe)
# ---------------------------------------------------------------------------
header("T2: variant model-ID routing @ 1024x1024 (cheap probe)")
client = OpenAI(api_key=KEY, base_url=V1, timeout=180)
PROMPT_SHORT = "A simple plain test image of a clear blue sky."

VARIANTS = [
    "gpt-image-2",
    "openai/gpt-image-2",
    "azure/gpt-image-2",
    "official/gpt-image-2",
    "gpt-image-2-2025-04-15",
    "gpt-image-2-2025-04-23",
    "gpt-image-2-2026-04-21",
    "gpt-image-2-preview",
    "gpt-image-2-hd",
    "gpt-image-2-high",
    "image-2",
    "gpt-image",
    "gpt-image-2-mini",
    "gpt-image-2-pro",
]


def try_gen(model: str, size: str, *, label_suffix: str = "",
            client_obj: OpenAI = client) -> dict:
    label = f"gen_{model.replace('/', '_')}_{size}{('_' + label_suffix) if label_suffix else ''}"
    print(f"\n--- {label}")
    t0 = time.time()
    try:
        resp = client_obj.images.generate(model=model, prompt=PROMPT_SHORT, size=size, n=1)
    except Exception as e:
        msg = str(e)[:300]
        if any(c > "\u007f" for c in msg):  # has non-ASCII; might be garbled
            decoded = decode_garbled(msg)
            print(f"  EXC: {type(e).__name__}: {msg}")
            if decoded != msg:
                print(f"  EXC decoded: {decoded[:300]}")
        else:
            print(f"  EXC: {type(e).__name__}: {msg}")
        return {"label": label, "model": model, "size": size,
                "error": f"{type(e).__name__}: {msg}",
                "elapsed_s": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    url = getattr(d, "url", None)
    if b64:
        raw = base64.b64decode(b64)
    elif url:
        raw = requests.get(url, timeout=120).content
    else:
        return {"label": label, "model": model, "size": size,
                "error": "no b64/url", "elapsed_s": elapsed}
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    print(f"  OK: {w}x{h}  aspect={round(w/h, 4)}  saved -> {p.name}")
    return {"label": label, "model": model, "size": size,
            "actual": [w, h], "aspect": round(w / h, 4),
            "path": str(p), "elapsed_s": elapsed}


variant_results: list = []
working_variants: list[str] = []
for v in VARIANTS:
    r = try_gen(v, "1024x1024")
    variant_results.append(r)
    if r.get("actual"):
        working_variants.append(v)


# ---------------------------------------------------------------------------
# T3: try extra_body / header group selectors with the bare model name
# ---------------------------------------------------------------------------
header("T3: group / channel routing tricks (extra_body + header)")
group_attempts: list = []

# extra_body variants
for body_key, body_val in [
    ("group", "openai"),
    ("group", "default"),
    ("group", "vip"),
    ("group", "image"),
    ("channel_group", "openai"),
    ("route", "openai"),
    ("provider", "openai"),
]:
    label = f"eb_{body_key}_{body_val}"
    print(f"\n--- extra_body={{{body_key}: {body_val!r}}}")
    t0 = time.time()
    try:
        resp = client.images.generate(
            model="gpt-image-2", prompt=PROMPT_SHORT, size="1024x1024", n=1,
            extra_body={body_key: body_val},
        )
        d = resp.data[0]
        b64 = getattr(d, "b64_json", None)
        if b64:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = img.size
            p = OUT / f"{label}.png"
            p.write_bytes(raw)
            print(f"  OK: {w}x{h}  -> {p.name}")
            group_attempts.append({"label": label, "actual": [w, h]})
        else:
            print(f"  no b64/url")
            group_attempts.append({"label": label, "error": "no_image"})
    except Exception as e:
        msg = str(e)[:300]
        print(f"  EXC: {type(e).__name__}: {msg}")
        decoded = decode_garbled(msg)
        if decoded != msg:
            print(f"  decoded: {decoded[:300]}")
        group_attempts.append({"label": label, "error": msg,
                               "elapsed_s": round(time.time() - t0, 1)})

# Header variants — different client per attempt
for hdr_key in ["X-Channel-Group", "X-Group", "X-Channel", "Channel-Group"]:
    for grp in ["openai", "default", "vip", "image"]:
        label = f"hdr_{hdr_key}_{grp}"
        print(f"\n--- header {hdr_key}: {grp}")
        t0 = time.time()
        try:
            c = OpenAI(
                api_key=KEY, base_url=V1, timeout=120,
                default_headers={hdr_key: grp},
            )
            resp = c.images.generate(
                model="gpt-image-2", prompt=PROMPT_SHORT, size="1024x1024", n=1,
            )
            d = resp.data[0]
            b64 = getattr(d, "b64_json", None)
            if b64:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                w, h = img.size
                p = OUT / f"{label}.png"
                p.write_bytes(raw)
                print(f"  OK: {w}x{h}  -> {p.name}")
                group_attempts.append({"label": label, "actual": [w, h]})
            else:
                print(f"  no b64/url")
                group_attempts.append({"label": label, "error": "no_image"})
        except Exception as e:
            msg = str(e)[:200]
            print(f"  EXC: {type(e).__name__}: {msg}")
            group_attempts.append({"label": label, "error": msg,
                                   "elapsed_s": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------
# T4: dump available models again with full attributes (any "group" hints?)
# ---------------------------------------------------------------------------
header("T4: full /v1/models with attributes")
try:
    r = requests.get(f"{V1}/models", headers=HEADERS, timeout=30)
    if r.status_code == 200:
        j = r.json()
        items = j.get("data", j) if isinstance(j, dict) else []
        # Look for any model with extra fields suggesting groups
        sample = items[:5] if isinstance(items, list) else []
        print("First 5 model entries (full):")
        for s in sample:
            print(json.dumps(s, indent=2, ensure_ascii=False))
        # Search for "image-2" anywhere in the dump
        text_dump = json.dumps(items, ensure_ascii=False)
        hits = ["gpt-image-2" in text_dump, "image-2" in text_dump,
                "image_2" in text_dump]
        print(f"  'gpt-image-2' literal in dump?: {hits[0]}")
        print(f"  'image-2' (any case) in dump?: {hits[1]}")
        print(f"  'image_2' in dump?: {hits[2]}")
except Exception as e:
    print(f"EXC: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header("SUMMARY")
summary = {
    "endpoint_results_keys": list(endpoint_results.keys()),
    "variant_results": variant_results,
    "working_variants": working_variants,
    "group_attempts": group_attempts,
}
(OUT / "chatfire_retry_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print(f"  T1 working endpoints: {list(endpoint_results.keys())}")
print(f"  T2 working model-ID variants: {working_variants}")
print(f"  T3 working group attempts: {[g['label'] for g in group_attempts if g.get('actual')]}")
print(f"\nfull summary -> {OUT / 'chatfire_retry_summary.json'}")
