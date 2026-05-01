"""Probe whether gpt-image-2 via OpenRouter returns a native ERP when we
stop telling it pixel sizes / stop resizing.

Three conditions, same base scene prompt:

  A "no_size"     prompt mentions ERP but NO target pixel dimensions;
                  raw model output saved as-is (no resize).
  B "current"     prompt has size_hint "(target 1024x512)" + BICUBIC
                  resize 1024x1024 -> 1024x512  <-- existing pipeline
  C "no_resize"   prompt has size_hint as in B, but raw model output
                  saved as-is (no resize).

For each condition we save:
  * raw PNG bytes from the model (no PIL re-encode)
  * a derived `.processed.png`   (PIL-loaded, exactly what the pipeline would store)
  * a small stats line  (width x height, horizontal seam score)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import load_config  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "_erp_native_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same base prompt used by all 3 conditions; only the size hint and
# downstream processing change.
BASE_SCENE = (
    "A photorealistic 360 degree equirectangular panorama of an empty "
    "specialty coffee shop interior in late morning. Industrial Scandinavian "
    "style: exposed concrete ceiling with matte black steel I-beams, "
    "whitewashed brick walls, polished oak floor. Long pale-oak espresso bar "
    "with a stainless-steel three-group espresso machine, glass pastry case, "
    "ceramic cups stacked on the counter, walnut pour-over station with V60 "
    "drippers, communal oak table with bentwood chairs, olive-green leather "
    "banquette along the brick wall, large fiddle-leaf fig in a charcoal "
    "planter, brass pendant lamps hanging in a row over the bar, soft "
    "5600 K daylight from full-height storefront windows on one side. "
    "Equirectangular 360 panorama, 2:1 aspect ratio, full sphere, seamless "
    "horizontal wraparound (left and right edges identical), no people, "
    "no humans, no animals, no portraits, no text, no captions, no watermark, "
    "no logos, no UI overlay."
)

SIZE_HINT_CURRENT = (
    "\n\nThe final image must be a 2:1 equirectangular panorama "
    "(target 1024x512)."
)


def _build_client(cfg):
    from openai import OpenAI  # type: ignore

    api_key = (cfg.openai.get("api_key", "") or "").strip()
    if not api_key:
        api_key = os.environ.get(str(cfg.openai.api_key_env), "").strip()
    if not api_key:
        raise SystemExit(
            "no API key: set OPENAI_API_KEY env or openai.api_key in config"
        )

    base_url = str(cfg.openai.get("base_url", "")).strip() or None
    headers = {}
    if str(cfg.openai.get("provider", "")).lower() == "openrouter":
        headers = {
            "HTTP-Referer": str(cfg.openai.get("http_referer", "https://github.com/erpgen")),
            "X-Title": str(cfg.openai.get("app_title", "EGMOR")),
        }
    kwargs = {"api_key": api_key, "timeout": float(cfg.openai.get("request_timeout_sec", 600))}
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def _chat_image_call(client, model: str, prompt: str) -> bytes:
    """Call OpenRouter chat-image; return the raw PNG bytes from the
    base64 data URL (no PIL re-encode)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        extra_body={"modalities": ["image", "text"]},
    )
    msg = resp.choices[0].message
    images = getattr(msg, "images", None)
    if images is None and hasattr(msg, "model_extra"):
        images = (msg.model_extra or {}).get("images")
    if not images:
        raise RuntimeError(
            f"no images in response. content={getattr(msg, 'content', None)!r}"
        )
    first = images[0]
    url = first["image_url"]["url"] if isinstance(first, dict) else first.image_url.url
    if not url.startswith("data:"):
        raise RuntimeError(f"unexpected url: {url[:80]}")
    return base64.b64decode(url.split(",", 1)[1])


def _seam_score(arr: np.ndarray) -> float:
    """L2 distance between first and last column averaged per pixel."""
    a = arr[:, 0].astype(np.float32)
    b = arr[:, -1].astype(np.float32)
    return float(np.sqrt(((a - b) ** 2).mean()))


def _column_self_similarity(arr: np.ndarray) -> float:
    """L2 distance between column 0 and column W//2 (should be LARGE for any
    well-formed image regardless of whether ERP)."""
    W = arr.shape[1]
    a = arr[:, 0].astype(np.float32)
    b = arr[:, W // 2].astype(np.float32)
    return float(np.sqrt(((a - b) ** 2).mean()))


def _save_and_report(label: str, raw_png: bytes, do_resize: bool) -> dict:
    raw_path = OUT_DIR / f"test_{label}.raw.png"
    raw_path.write_bytes(raw_png)
    img = Image.open(io.BytesIO(raw_png)).convert("RGB")
    raw_w, raw_h = img.size
    if do_resize:
        proc = img.resize((1024, 512), Image.BICUBIC)
    else:
        proc = img
    proc_path = OUT_DIR / f"test_{label}.processed.png"
    proc.save(proc_path)
    arr = np.array(proc, dtype=np.uint8)
    info = {
        "label": label,
        "raw_size": [raw_w, raw_h],
        "raw_aspect": round(raw_w / max(1, raw_h), 4),
        "raw_path": str(raw_path),
        "processed_size": list(proc.size),
        "processed_path": str(proc_path),
        "did_resize": do_resize,
        "seam_score_RMS": round(_seam_score(arr), 3),
        "self_dist_RMS_col0_vs_colW2": round(_column_self_similarity(arr), 3),
    }
    print(f"  [{label}] raw={raw_w}x{raw_h} aspect={info['raw_aspect']} "
          f"seam_RMS={info['seam_score_RMS']} (smaller = more wrap-closed)")
    return info


def main() -> int:
    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    if str(cfg.openai.get("provider", "")).lower() != "openrouter":
        print("WARNING: provider != openrouter; the test still runs but the "
              "size_hint behaviour might not match reality.")
    print(f"provider = {cfg.openai.provider}")
    print(f"model    = {cfg.openai.model}")
    print(f"base_url = {cfg.openai.get('base_url', '')}")
    print(f"output   = {OUT_DIR}")
    print()

    client = _build_client(cfg)
    model = str(cfg.openai.model)

    print("=== A: NO size hint, NO resize ===")
    promptA = BASE_SCENE
    rawA = _chat_image_call(client, model, promptA)
    statsA = _save_and_report("A_no_size", rawA, do_resize=False)
    print()

    print("=== B: size hint + BICUBIC resize to 1024x512 (current pipeline) ===")
    promptB = BASE_SCENE + SIZE_HINT_CURRENT
    rawB = _chat_image_call(client, model, promptB)
    statsB = _save_and_report("B_current", rawB, do_resize=True)
    print()

    print("=== C: size hint kept, NO resize ===")
    promptC = BASE_SCENE + SIZE_HINT_CURRENT
    rawC = _chat_image_call(client, model, promptC)
    statsC = _save_and_report("C_no_resize", rawC, do_resize=False)
    print()

    summary = {
        "model": model,
        "provider": str(cfg.openai.provider),
        "base_url": str(cfg.openai.get("base_url", "")),
        "results": [statsA, statsB, statsC],
        "interpretation_hint": (
            "If A or C produces a panoramic 2:1 image natively (raw aspect > 1.5) "
            "AND has a low seam_score_RMS (<= ~30), the model can produce real ERP "
            "and the bug is our resize/squash to 1024x512.  If all three return "
            "1:1 or 4:3 raw images, OpenRouter's chat-image route never produces "
            "true ERP regardless of prompt and we need a different endpoint."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=== summary saved to", OUT_DIR / "summary.json", "===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
