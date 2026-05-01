"""Smoke test: 3-call OpenRouter outpaint -> 2:1 ERP.

Single pose, 3 OpenRouter chat-image calls. Saves all 4 PNGs (base, left,
right, stitched ERP) and reports seam metrics so the parent agent can
visually confirm before running the full 9-pose cafe pipeline.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from erpgen.config import load_config  # noqa: E402
from erpgen.openai_erp import (  # noqa: E402
    ImageClient,
    OpenAIConfig,
    _png_bytes,
    _stitch_three,
    _ensure_size,
)

OUT = REPO / "outputs" / "_smoke_or_outpaint"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cfg = load_config(str(REPO / "config" / "default.yaml"))
    print(f"provider = {cfg.openai.provider}")
    print(f"model    = {cfg.openai.model}")
    print(f"base_url = {cfg.openai.base_url}")
    print(f"target   = {cfg.openai.size}")
    print()

    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, mock=False, allow_mock_fallback=False, verbose=True)
    if client.mock_mode:
        raise SystemExit("FATAL: client fell back to mock mode (likely missing API key)")

    prompt = (
        "A photorealistic empty specialty coffee shop interior in late morning. "
        "Industrial Scandinavian style: exposed concrete ceiling with matte "
        "black steel I-beams, whitewashed brick walls, polished oak floor, "
        "long pale-oak espresso bar with a stainless-steel three-group espresso "
        "machine, glass pastry case, ceramic cups stacked on the counter, "
        "walnut pour-over station with V60 drippers, communal oak table with "
        "bentwood chairs, olive-green leather banquette along the brick wall, "
        "large fiddle-leaf fig in a charcoal planter, brass pendant lamps "
        "hanging in a row over the bar, soft 5600 K daylight from full-height "
        "storefront windows on one side."
    )

    # Run the 3 calls so we can save the intermediates (left/right concurrent).
    from concurrent.futures import ThreadPoolExecutor

    no_people = (
        " Strict constraints: no people, no humans, no animals, no "
        "portraits, no figures, no text, no captions, no watermark, "
        "no logos, no UI overlay."
    )
    base_prompt = ImageClient._ERP_PROMPT_PREFIX + prompt + no_people
    left_prompt = (
        ImageClient._ERP_PROMPT_PREFIX + prompt + no_people
        + " The reference image is an equirectangular 360 panorama of the "
          "SAME room. Generate another equirectangular 360 panorama of the "
          "same room rotated approximately 120 degrees to the LEFT around "
          "the vertical axis (the camera turned left), preserving identical "
          "materials, lighting, colour palette, and architectural features."
    )
    right_prompt = (
        ImageClient._ERP_PROMPT_PREFIX + prompt + no_people
        + " The reference image is an equirectangular 360 panorama of the "
          "SAME room. Generate another equirectangular 360 panorama of the "
          "same room rotated approximately 120 degrees to the RIGHT around "
          "the vertical axis (the camera turned right), preserving identical "
          "materials, lighting, colour palette, and architectural features."
    )

    t0 = time.time()
    print("=== call 1/3: base (sequential, with strong ERP prompt) ===")
    base = client._call_with_retry(
        client._chat_image_call,
        prompt=base_prompt, size="1024x1024", ref_image_bytes=None,
    )
    base = _ensure_size(base, 1024, 1024)
    base.save(OUT / "base.png")
    print(f"  base saved -> {OUT/'base.png'}  size={base.size}  elapsed={round(time.time()-t0,1)}s")
    base_bytes = _png_bytes(base)

    print("\n=== calls 2 and 3: left + right ref-i2i (PARALLEL) ===")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_fut = pool.submit(
            client._call_with_retry, client._chat_image_call,
            prompt=left_prompt, size="1024x1024", ref_image_bytes=base_bytes,
        )
        right_fut = pool.submit(
            client._call_with_retry, client._chat_image_call,
            prompt=right_prompt, size="1024x1024", ref_image_bytes=base_bytes,
        )
        left = _ensure_size(left_fut.result(), 1024, 1024)
        right = _ensure_size(right_fut.result(), 1024, 1024)
    left.save(OUT / "left.png")
    right.save(OUT / "right.png")
    print(f"  left + right done in parallel  elapsed={round(time.time()-t1,1)}s")

    print("\n=== stitching ===")
    target_w, target_h = client.parse_size()
    erp = _stitch_three(left, base, right, target_size=(target_w, target_h), blend_w=64)
    erp.save(OUT / "erp_2x1.png")
    print(f"  ERP saved -> {OUT/'erp_2x1.png'}  size={erp.size}  total elapsed={round(time.time()-t0,1)}s")

    # Seam metrics: |left's right edge - base's left edge|, etc.
    la = np.array(left, dtype=np.float32)
    ba = np.array(base, dtype=np.float32)
    ra = np.array(right, dtype=np.float32)
    erpa = np.array(erp, dtype=np.float32)
    seam_LB = float(np.sqrt(((la[:, -1] - ba[:, 0]) ** 2).mean()))
    seam_BR = float(np.sqrt(((ba[:, -1] - ra[:, 0]) ** 2).mean()))
    # ERP global wrap: leftmost vs rightmost
    erp_h, erp_w, _ = erpa.shape
    wrap_RMS = float(np.sqrt(((erpa[:, 0] - erpa[:, -1]) ** 2).mean()))
    halfdiff = float(np.sqrt(((erpa[:, 0] - erpa[:, erp_w // 2]) ** 2).mean()))
    print()
    print("=== Seam metrics ===")
    print(f"  left.right vs base.left  RMS: {round(seam_LB, 2)}    (smaller = better continuity)")
    print(f"  base.right vs right.left RMS: {round(seam_BR, 2)}    (smaller = better)")
    print(f"  Final ERP wrap col0 vs col(W-1) RMS: {round(wrap_RMS, 2)}    (< 30 OK, < 15 great)")
    print(f"  Final ERP halfdiff col0 vs col(W/2)  : {round(halfdiff, 2)}")
    print(f"  wrap ratio: {round(wrap_RMS / max(halfdiff, 0.001), 4)}")

    # Pole rows
    rs = erpa.std(axis=(1, 2))
    print(f"  ERP pole std top 5:  {[round(float(x),2) for x in rs[:5]]}")
    print(f"  ERP pole std bot 5:  {[round(float(x),2) for x in rs[-5:]]}")
    print(f"  ERP equator std    : {round(float(rs[erp_h // 2]), 2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
