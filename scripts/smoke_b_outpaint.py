"""Smoke test path B: 1536x1024 base + horizontal outpaint -> true 2:1 ERP.

Single-pose end-to-end: 3 API calls to super.shangliu.org (or whichever
provider is configured). Saves intermediates + final ERP and reports seam
closure metrics.
"""
from __future__ import annotations

import io
import os
import sys
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
    _build_outpaint_canvas,
    _stitch_panorama,
)

OUT = REPO / "outputs" / "_smoke_b"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cfg = load_config(str(REPO / "config" / "default.yaml"))
    print(f"provider = {cfg.openai.provider}")
    print(f"base_url = {cfg.openai.base_url}")
    print(f"model    = {cfg.openai.model}")
    print(f"size     = {cfg.openai.size}")
    print(f"target ERP = {cfg.openai.erp_target_w}x{cfg.openai.erp_target_h}")
    print()

    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, mock=False, allow_mock_fallback=False, verbose=True)
    print(f"mock_mode = {client.mock_mode}  supports_mask = {client.supports_mask}")
    if client.mock_mode:
        raise SystemExit("FATAL: client fell back to mock mode (likely missing API key)")

    prompt = (
        "A photorealistic 360 degree equirectangular panorama of an empty "
        "specialty coffee shop interior in late morning. Industrial Scandinavian "
        "style: exposed concrete ceiling with matte black steel I-beams, "
        "whitewashed brick walls, polished oak floor, long pale-oak espresso bar "
        "with a stainless-steel three-group espresso machine, glass pastry case, "
        "ceramic cups stacked on the counter, walnut pour-over station with V60 "
        "drippers, communal oak table with bentwood chairs, olive-green leather "
        "banquette along the brick wall, large fiddle-leaf fig in a charcoal "
        "planter, brass pendant lamps hanging in a row over the bar, soft "
        "5600 K daylight from full-height storefront windows on one side. "
        "Equirectangular 360 panorama, 2:1 aspect ratio, full sphere, seamless "
        "horizontal wraparound (left and right edges identical), no people, "
        "no humans, no animals, no portraits, no text, no captions, no watermark."
    )

    import time
    t0 = time.time()
    print("=== generate_erp_2x1 ===")
    erp = client.generate_erp_2x1(prompt)
    elapsed = time.time() - t0
    print(f"\nFinal ERP: {erp.size}  elapsed={round(elapsed, 1)}s")

    out_path = OUT / "erp_2x1.png"
    erp.save(out_path)
    print(f"saved -> {out_path}")

    # Save intermediates from cache (last call's intermediates were generated
    # by generate_rgb / edit_with_mask which cache by their own hash; we re-
    # build them here for inspection convenience).
    arr = np.array(erp.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    seam_RMS = float(np.sqrt(((arr[:, 0] - arr[:, -1]) ** 2).mean()))
    halfdiff = float(np.sqrt(((arr[:, 0] - arr[:, w // 2]) ** 2).mean()))
    row_std = arr.std(axis=(1, 2))
    print()
    print("=== ERP geometry analysis ===")
    print(f"  shape: {h}x{w}  aspect={round(w/h, 4)}")
    print(f"  seam RMS col0 vs col(W-1):  {round(seam_RMS, 2)}    (< 30 = passable, < 15 = good)")
    print(f"  baseline   col0 vs col(W/2): {round(halfdiff, 2)}")
    print(f"  wrap ratio: {round(seam_RMS / max(halfdiff, 0.001), 4)}    (< 0.5 = good)")
    print(f"  pole std top 5: {[round(float(x), 2) for x in row_std[:5]]}")
    print(f"  pole std bot 5: {[round(float(x), 2) for x in row_std[-5:]]}")
    print(f"  equator std    : {round(float(row_std[h // 2]), 2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
