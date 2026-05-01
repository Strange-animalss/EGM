"""Retry gpt-image-2 specifically on super.shangliu.org. Test ERP-aware
prompt at 1536x1024 (max landscape) and see if it closes the seam (true ERP)
or returns a flat panorama photo (like gpt-image-1.5 did).
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

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.environ.get("SHANGLIU_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("Set SHANGLIU_API_KEY or OPENAI_API_KEY")
BASE = "https://super.shangliu.org/v1"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_shangliu_probe"
OUT.mkdir(parents=True, exist_ok=True)


def _save(label: str, raw: bytes) -> tuple[int, int, str]:
    p = OUT / f"{label}.png"
    p.write_bytes(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img.size[0], img.size[1], str(p)


def _gen(label: str, model: str, size: str, prompt: str, max_retry: int = 2) -> dict:
    print(f"\n=== {label}  model={model}  size={size} ===")
    last_err = None
    for attempt in range(max_retry + 1):
        if attempt > 0:
            print(f"  retry {attempt}/{max_retry} ...")
            time.sleep(3)
        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/images/generations",
                headers=HEADERS,
                json={"model": model, "prompt": prompt, "size": size, "n": 1},
                timeout=300,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  {last_err}  elapsed={round(time.time()-t0,1)}s")
            continue
        elapsed = round(time.time() - t0, 1)
        print(f"  status: {r.status_code}  elapsed={elapsed}s")
        if r.status_code != 200:
            body = r.text[:400]
            print(f"  body: {body}")
            return {"label": label, "model": model, "size": size,
                    "status": r.status_code, "error_body": body, "elapsed_s": elapsed}
        j = r.json()
        d0 = (j.get("data") or [{}])[0]
        if d0.get("b64_json"):
            raw = base64.b64decode(d0["b64_json"])
        elif d0.get("url"):
            raw = requests.get(d0["url"], timeout=60).content
        else:
            return {"label": label, "model": model, "size": size,
                    "error": "no b64/url", "elapsed_s": elapsed}
        w, h, path = _save(label, raw)
        print(f"  OK image: {w}x{h}  aspect={round(w/h,4)}  saved -> {path}")
        # ERP analysis
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        row_std = arr.std(axis=(1, 2))
        seam = float(np.sqrt(((arr[:, 0] - arr[:, -1]) ** 2).mean()))
        baseline = float(np.sqrt(((arr[:, 0] - arr[:, w // 2]) ** 2).mean()))
        info = {
            "label": label, "model": model, "size": size,
            "actual": [w, h], "aspect": round(w / h, 4),
            "elapsed_s": elapsed, "path": path,
            "pole_std_top5": [round(float(x), 2) for x in row_std[:5]],
            "pole_std_bot5": [round(float(x), 2) for x in row_std[-5:]],
            "equator_std": round(float(row_std[h // 2]), 2),
            "seam_RMS": round(seam, 2),
            "baseline_RMS_col0_vs_W2": round(baseline, 2),
            "wrap_ratio": round(seam / max(baseline, 0.001), 4),
        }
        print(f"  pole std top5: {info['pole_std_top5']}    (~0 if true ERP poles)")
        print(f"  pole std bot5: {info['pole_std_bot5']}    (~0 if true ERP poles)")
        print(f"  equator std: {info['equator_std']}")
        print(f"  seam RMS: {info['seam_RMS']}    (< 15 = good 360 wrap)")
        print(f"  wrap ratio: {info['wrap_ratio']}    (< 0.3 = clear 360 wrap)")
        return info
    return {"label": label, "model": model, "size": size, "error": last_err}


CAFE_PROMPT = (
    "A photorealistic 360 degree equirectangular panorama of an empty "
    "specialty coffee shop interior in late morning. Industrial Scandinavian "
    "style: exposed concrete ceiling, whitewashed brick walls, polished oak "
    "floor, long pale-oak espresso bar with stainless espresso machine, glass "
    "pastry case, communal oak table with bentwood chairs. Equirectangular "
    "360 panorama, 2:1 aspect ratio, full sphere, seamless horizontal "
    "wraparound (left and right edges identical), no people, no text, no "
    "watermark."
)

results = []

# T6: gpt-image-2 baseline 1024x1024
results.append(_gen("T6_image2_1024x1024", "gpt-image-2", "1024x1024", CAFE_PROMPT))

# T7: gpt-image-2 max landscape 1536x1024
results.append(_gen("T7_image2_1536x1024", "gpt-image-2", "1536x1024", CAFE_PROMPT))

# T8: gpt-image-2 auto
results.append(_gen("T8_image2_auto", "gpt-image-2", "auto", CAFE_PROMPT))

# T9: dated version  
results.append(_gen("T9_image2_dated_1536x1024", "gpt-image-2-2026-04-21",
                    "1536x1024", CAFE_PROMPT))

(OUT / "image2_probe_summary.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n" + "=" * 70)
print("SUMMARY (ERP closure check)")
print("=" * 70)
print(f"{'label':35s}  {'size':10s}  seam   wrap_ratio  poles  verdict")
for r in results:
    if "actual" in r:
        s = r["actual"]
        sz = f"{s[0]}x{s[1]}"
        seam = r["seam_RMS"]
        wr = r["wrap_ratio"]
        # rough verdict: true ERP wraps the seam
        if seam < 15 and wr < 0.3:
            verdict = "TRUE ERP"
        elif seam < 30 and wr < 0.5:
            verdict = "weak ERP"
        else:
            verdict = "NOT ERP"
        # poles
        poles_zero = sum(s < 5 for s in r["pole_std_top5"] + r["pole_std_bot5"])
        print(f"{r['label']:35s}  {sz:10s}  {seam:6.2f}  {wr:6.3f}      {poles_zero:>2d}/10    {verdict}")
    else:
        print(f"{r['label']:35s}  FAIL: {r.get('error_body', r.get('error'))[:60]}")

print("\nDONE")
