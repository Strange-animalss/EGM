"""Probe size + quality + edit support for the image models that ARE
allowed on this unverified OpenAI org."""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import OpenAI

from erpgen.openai_erp import _decode_image_resp

key = os.environ["OPENAI_API_KEY"]
cli = OpenAI(api_key=key, timeout=300)

# 1. Sizes for gpt-image-1, gpt-image-1.5, chatgpt-image-latest
PROMPT = "An empty cafe interior, equirectangular 360 panorama, late morning, no people."

candidates = [
    ("gpt-image-1.5", "3840x1920", "high"),
    ("gpt-image-1.5", "2048x1024", "high"),
    ("gpt-image-1.5", "1536x1024", "high"),
    ("gpt-image-1.5", "1024x1024", "high"),
    ("gpt-image-1",   "2048x1024", "high"),
    ("gpt-image-1",   "1536x1024", "high"),
    ("gpt-image-1",   "1024x1024", "high"),
    ("gpt-image-1-mini", "1024x1024", "high"),
    ("chatgpt-image-latest", "2048x1024", None),
    ("chatgpt-image-latest", "1024x1024", None),
]

results: list[dict] = []
saved_refs: dict[str, Path] = {}
for model, size, quality in candidates:
    label = f"{model}_{size}"
    if quality:
        label += f"_q{quality}"
    print(f"\n--- {label} ---")
    t0 = time.time()
    try:
        kw: dict = {"model": model, "prompt": PROMPT, "size": size, "n": 1}
        if quality:
            kw["quality"] = quality
        r = cli.images.generate(**kw)
        img = _decode_image_resp(r)
        elapsed = time.time() - t0
        print(f"  OK in {elapsed:.1f}s -> requested={size} actual={img.size} aspect={img.size[0]/img.size[1]:.3f}")
        out = REPO / "outputs" / "_oai_alts" / f"{label}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        saved_refs.setdefault(model, out)
        results.append({"label": label, "actual": list(img.size), "elapsed_s": round(elapsed, 1)})
    except Exception as e:
        elapsed = time.time() - t0
        msg = str(e)[:240]
        print(f"  FAIL in {elapsed:.1f}s: {type(e).__name__}: {msg}")
        results.append({"label": label, "error": f"{type(e).__name__}: {msg}", "elapsed_s": round(elapsed, 1)})

# 2. images.edit for each model that successfully generated something at >=1024x1024
print("\n--- images.edit smoke ---")
for model, ref_path in saved_refs.items():
    label = f"edit_{model}"
    print(f"\n--- {label} ---")
    t0 = time.time()
    try:
        with ref_path.open("rb") as fh:
            kw = {
                "model": model, "image": fh,
                "prompt": "Slightly brighten the lighting; keep everything else identical.",
                "size": "1024x1024", "n": 1,
            }
            r = cli.images.edit(**kw)
        img = _decode_image_resp(r)
        elapsed = time.time() - t0
        print(f"  OK in {elapsed:.1f}s -> {img.size}")
        results.append({"label": label, "actual": list(img.size), "elapsed_s": round(elapsed, 1)})
    except Exception as e:
        elapsed = time.time() - t0
        msg = str(e)[:240]
        print(f"  FAIL in {elapsed:.1f}s: {type(e).__name__}: {msg}")
        results.append({"label": label, "error": f"{type(e).__name__}: {msg}", "elapsed_s": round(elapsed, 1)})

import json
out = REPO / "outputs" / "_oai_alts" / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nfull summary -> {out}")
ok = [r for r in results if r.get("actual")]
print(f"\n{len(ok)}/{len(results)} calls succeeded")
for r in ok:
    print(f"  {r['label']}: actual={r['actual']}  elapsed={r['elapsed_s']}s")
