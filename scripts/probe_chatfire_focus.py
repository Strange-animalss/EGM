"""Focused chatfire probe after retry: bare 'gpt-image-2' was confirmed
working at 1024x1024 in the previous probe. Now: 2048x1024, 1536x1024,
1024x512 + images.edit for gpt-image-2 and gpt-image-2-high.
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
OUT = REPO / "outputs" / "_chatfire_focus"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("CHATFIRE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set CHATFIRE_API_KEY or OPENAI_API_KEY")
V1 = "https://api.chatfire.cn/v1"
client = OpenAI(api_key=KEY, base_url=V1, timeout=300)

ERP_PROMPT = (
    "An empty specialty coffee shop interior, late morning, soft natural light. "
    "Equirectangular 360-degree panorama, 2:1 aspect ratio, seamless horizontal wrap, "
    "straight horizon, pole compression at top and bottom. No people, no animals."
)
SHORT_PROMPT = "A clean equirectangular 360 panorama photo of an empty cafe interior."


def gen(model: str, size: str, prompt: str = SHORT_PROMPT, **extra) -> dict:
    label = f"gen_{model}_{size}"
    if extra:
        label += "_" + "_".join(f"{k}={v}" for k, v in extra.items())
    print(f"\n--- {label}")
    t0 = time.time()
    try:
        kw = {"model": model, "prompt": prompt, "size": size, "n": 1}
        kw.update(extra)
        resp = client.images.generate(**kw)
        d = resp.data[0]
        b64 = getattr(d, "b64_json", None)
        url = getattr(d, "url", None)
        if b64:
            raw = base64.b64decode(b64)
        elif url:
            raw = requests.get(url, timeout=120).content
        else:
            print(f"  no b64/url; resp={d}")
            return {"label": label, "error": "no_b64_or_url"}
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        p = OUT / f"{label}.png"
        p.write_bytes(raw)
        elapsed = round(time.time() - t0, 1)
        print(f"  OK: requested={size}  actual={w}x{h}  aspect={round(w/h, 4)}  "
              f"elapsed={elapsed}s  -> {p.name}")
        return {"label": label, "requested": size, "actual": [w, h],
                "aspect": round(w / h, 4), "elapsed_s": elapsed}
    except Exception as e:
        msg = str(e)[:300]
        elapsed = round(time.time() - t0, 1)
        print(f"  EXC ({elapsed}s): {type(e).__name__}: {msg}")
        return {"label": label, "error": f"{type(e).__name__}: {msg}",
                "elapsed_s": elapsed}


def edit(ref_path: Path, model: str, size: str, prompt: str = SHORT_PROMPT) -> dict:
    label = f"edit_{model}_{size}_{ref_path.stem}"
    print(f"\n--- {label}")
    t0 = time.time()
    try:
        with ref_path.open("rb") as fh:
            resp = client.images.edit(
                model=model, image=fh, prompt=prompt, size=size, n=1,
            )
        d = resp.data[0]
        b64 = getattr(d, "b64_json", None)
        url = getattr(d, "url", None)
        if b64:
            raw = base64.b64decode(b64)
        elif url:
            raw = requests.get(url, timeout=120).content
        else:
            print(f"  no b64/url; resp={d}")
            return {"label": label, "error": "no_b64_or_url"}
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        p = OUT / f"{label}.png"
        p.write_bytes(raw)
        elapsed = round(time.time() - t0, 1)
        print(f"  OK: requested={size}  actual={w}x{h}  aspect={round(w/h, 4)}  "
              f"elapsed={elapsed}s  -> {p.name}")
        return {"label": label, "requested": size, "actual": [w, h],
                "aspect": round(w / h, 4), "elapsed_s": elapsed}
    except Exception as e:
        msg = str(e)[:300]
        elapsed = round(time.time() - t0, 1)
        print(f"  EXC ({elapsed}s): {type(e).__name__}: {msg}")
        return {"label": label, "error": f"{type(e).__name__}: {msg}",
                "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 1: gpt-image-2 across panoramic sizes")
print("=" * 70)
results: list = []
ref_2x1: Path | None = None
for size in ["2048x1024", "1536x768", "1024x512", "1536x1024", "1024x1024"]:
    r = gen("gpt-image-2", size, ERP_PROMPT)
    results.append(r)
    if r.get("actual") and ref_2x1 is None and r["actual"][0] >= r["actual"][1] * 1.8:
        # first wide-aspect output -> use as edit ref
        ref_2x1 = OUT / f"gen_gpt-image-2_{size}.png"

print("\n" + "=" * 70)
print("PHASE 2: gpt-image-2-high across panoramic sizes")
print("=" * 70)
for size in ["2048x1024", "1536x768", "1024x512"]:
    r = gen("gpt-image-2-high", size, ERP_PROMPT)
    results.append(r)


print("\n" + "=" * 70)
print("PHASE 3: gpt-image-2 with quality variations at 2048x1024")
print("=" * 70)
for q in ["high", "medium", "low", "auto"]:
    r = gen("gpt-image-2", "2048x1024", ERP_PROMPT, quality=q)
    results.append(r)


print("\n" + "=" * 70)
print("PHASE 4: images.edit on chatfire (using best available ref)")
print("=" * 70)
# pick whichever image already saved as ref
edit_results: list = []
candidate_refs: list[Path] = []
for r in results:
    if r.get("actual"):
        # build path from label
        path = OUT / f"{r['label']}.png"
        if path.exists():
            candidate_refs.append(path)
if candidate_refs:
    ref = candidate_refs[0]
    print(f"  using ref: {ref.name}")
    for size in ["2048x1024", "1024x1024", "1024x512"]:
        edit_results.append(edit(ref, "gpt-image-2", size,
                                 "Adjust slightly: keep the same equirectangular panorama, brighten lighting."))
else:
    print("  no successful generation -> skip edit phase")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
summary = {"generations": results, "edits": edit_results}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
ok_gens = [r for r in results if r.get("actual")]
ok_edits = [r for r in edit_results if r.get("actual")]
print(f"  {len(ok_gens)}/{len(results)} generation calls succeeded")
for r in ok_gens:
    print(f"    {r['label']}: req={r['requested']}  actual={r['actual']}  "
          f"aspect={r['aspect']}")
print(f"  {len(ok_edits)}/{len(edit_results)} edit calls succeeded")
for r in ok_edits:
    print(f"    {r['label']}: req={r['requested']}  actual={r['actual']}  "
          f"aspect={r['aspect']}")

# verdict
two_one = [r for r in ok_gens if r.get("actual")
           and r["actual"][0] >= 2 * r["actual"][1] - 50]
print("\nVERDICT")
if two_one:
    print(f"  Native 2:1 ERP available: {two_one[0]['label']}  actual={two_one[0]['actual']}")
else:
    print("  No native 2:1 ERP (chatfire likely upscales/squares). Need outpaint or alt provider.")
if ok_edits:
    print(f"  images.edit works on chatfire: e.g. {ok_edits[0]['label']}")
else:
    print("  images.edit broken on chatfire.")
print(f"\nFull summary -> {OUT / 'summary.json'}")
