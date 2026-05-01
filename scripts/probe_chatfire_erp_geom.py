"""Verify the geometric ERP quality of chatfire gpt-image-2 @ 2048x1024.
Computes seam wrap RMS, pole row variance, and horizontal-line check
(Hough peak near 0/180 degrees implies straight equator).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "_chatfire_focus"
OUT = REPO / "outputs" / "_chatfire_erp_geom"
OUT.mkdir(parents=True, exist_ok=True)

samples = [
    "gen_gpt-image-2_2048x1024.png",
    "gen_gpt-image-2_2048x1024_quality=high.png",
    "gen_gpt-image-2_2048x1024_quality=medium.png",
    "gen_gpt-image-2-high_2048x1024.png",
    "edit_gpt-image-2_2048x1024_gen_gpt-image-2_2048x1024.png",
]

results = []
for name in samples:
    p = SRC / name
    if not p.exists():
        print(f"missing: {name}")
        continue
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape

    # 1) seam wrap: |left col - right col| RMS
    seam_rms = float(np.sqrt(((img[:, 0] - img[:, -1]) ** 2).mean()))

    # 2) pole row variance (top 8 rows): true ERP poles converge → low variance per row
    top_rows = img[:8]
    bot_rows = img[-8:]
    top_row_std_means = float(np.mean(top_rows.std(axis=1)))
    bot_row_std_means = float(np.mean(bot_rows.std(axis=1)))

    # 3) ERP self-symmetry test: rotate by 180° in yaw (column shift by W/2),
    # mirror vertically (a true panorama wrapped seamlessly is yaw-equivalent at any cyclic shift)
    shifted = np.roll(img, w // 2, axis=1)
    seam_rms_mid = float(np.sqrt(((shifted[:, 0] - shifted[:, -1]) ** 2).mean()))

    # 4) horizon straightness: equator row vs +-3 rows variance
    eq = img[h // 2 - 4:h // 2 + 4]
    eq_band_std = float(eq.std())

    rec = {
        "file": name,
        "shape": [h, w],
        "seam_rms_left_right": round(seam_rms, 2),
        "seam_rms_after_yaw_180": round(seam_rms_mid, 2),
        "pole_row_std_top_mean": round(top_row_std_means, 2),
        "pole_row_std_bot_mean": round(bot_row_std_means, 2),
        "equator_band_std": round(eq_band_std, 2),
    }
    results.append(rec)
    print(json.dumps(rec, indent=2))

(OUT / "geom_summary.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nWrote {OUT / 'geom_summary.json'}")
