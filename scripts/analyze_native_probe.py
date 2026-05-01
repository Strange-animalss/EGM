"""Analyse the 3 native-probe images to figure out what 1024x1024 actually means.

Tests:
 1. Are top-256 rows / bottom-256 rows (likely-padding region) different from
    the central 512 rows in mean colour?  If padded, top/bottom should be
    near-uniform.
 2. Compute per-row mean RGB and plot std-dev across rows: a stretched ERP
    with content has high inter-row variance. A padded image has 256 rows of
    uniform colour at top + 256 at bottom + variable middle.
 3. Compute per-column wrap closure (col 0 vs col 1023) and compare to
    a non-ERP control (col 0 vs col 512) to confirm horizontal wrap.
 4. Is the model producing geometric distortion that LOOKS like 2:1 ERP
    that has been vertically stretched?  We can decimate vertically by 2
    and see if the result has plausible aspect-ratio horizons.

Saves a .json report and 3 derived diagnostic images:
  test_<L>.diag.png   = collage of: raw / center 50% strip / decimated 1024x512
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "outputs" / "_erp_native_probe"


def analyse(label: str) -> dict:
    raw_p = PROBE / f"test_{label}.raw.png"
    img = Image.open(raw_p).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    H, W, _ = arr.shape

    # 1. row-mean colour profile
    row_mean = arr.astype(np.float32).mean(axis=(1, 2))  # (H,)
    row_std = arr.astype(np.float32).std(axis=(1, 2))    # (H,)

    # First/last 64 rows mean & std-dev: if padded with uniform colour, std
    # within those bands is near zero.
    top_band_std = float(row_std[:64].mean())
    bot_band_std = float(row_std[-64:].mean())
    mid_band_std = float(row_std[H // 2 - 32 : H // 2 + 32].mean())

    # 2. row-by-row content variability (inter-row gradient magnitude)
    row_grad = float(np.abs(np.diff(row_mean)).mean())

    # 3. horizontal wrap closure
    seam_score = float(np.sqrt(((arr[:, 0].astype(np.float32) - arr[:, -1].astype(np.float32)) ** 2).mean()))
    half_score = float(np.sqrt(((arr[:, 0].astype(np.float32) - arr[:, W // 2].astype(np.float32)) ** 2).mean()))

    # 4. extract content distribution: is the image filling top-to-bottom uniformly,
    # or is it concentrated in the middle band (consistent with a stretched 2:1 ERP
    # the model painted, where original content was mapped into a 1:1 canvas
    # naively i.e. 2x squashed vertically)?
    content_density_top = float((row_std[:H // 4]).mean())          # rows 0..255
    content_density_upper = float((row_std[H // 4 : H // 2]).mean())  # rows 256..511
    content_density_lower = float((row_std[H // 2 : 3 * H // 4]).mean())  # rows 512..767
    content_density_bot = float((row_std[3 * H // 4 :]).mean())     # rows 768..1023

    # 5. Detect "letterbox" pattern (ERP rendered into 2:1 then padded top/bot to 1:1):
    # very low std in top 256 rows AND bottom 256 rows would indicate this.
    # Plain ERP that is only 2:1 internally (squashed) would NOT show this.
    likely_letterbox = (top_band_std < 8 and bot_band_std < 8 and mid_band_std > 25)

    # 6. Likely-vertically-stretched ERP detection:
    # If the image is content all the way top-to-bottom AND content varies
    # smoothly, treating it as 2:1 ERP (decimate by 2 vertically) should
    # still look reasonable. Compute a "compressed entropy" on the decimated.
    deci = arr[::2]                                       # (512, 1024, 3)
    deci_path = PROBE / f"test_{label}.deci_512x1024.png"
    Image.fromarray(deci).save(deci_path)

    return {
        "label": label,
        "shape": [H, W],
        "row_mean_first_5": [round(float(x), 2) for x in row_mean[:5]],
        "row_mean_mid_5": [round(float(x), 2) for x in row_mean[H // 2 - 2 : H // 2 + 3]],
        "row_mean_last_5": [round(float(x), 2) for x in row_mean[-5:]],
        "top_band_std (rows 0..63)": round(top_band_std, 2),
        "bot_band_std (rows -64..-1)": round(bot_band_std, 2),
        "mid_band_std (rows H/2 +- 32)": round(mid_band_std, 2),
        "row_grad_overall_mean_abs_diff": round(row_grad, 2),
        "wrap_closure_RMS_col0_vs_colW-1": round(seam_score, 2),
        "non_wrap_baseline_col0_vs_colW/2": round(half_score, 2),
        "wrap_ratio (smaller=>cleaner wrap)": round(seam_score / max(half_score, 0.001), 4),
        "content_density_quartiles": {
            "top    (rows 0..255)":   round(content_density_top, 2),
            "upper  (rows 256..511)": round(content_density_upper, 2),
            "lower  (rows 512..767)": round(content_density_lower, 2),
            "bot    (rows 768..1023)": round(content_density_bot, 2),
        },
        "likely_letterbox": likely_letterbox,
        "decimated_2x_vertical_path": str(deci_path),
    }


def main() -> int:
    out = {}
    for label in ("A_no_size", "B_current", "C_no_resize"):
        info = analyse(label)
        out[label] = info
        print(f"=== {label} ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    rep = PROBE / "analysis.json"
    rep.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"=== analysis saved to {rep} ===")

    # Verdict
    print()
    print("=== VERDICT ===")
    any_letterbox = any(out[k]["likely_letterbox"] for k in out)
    if any_letterbox:
        print("AT LEAST ONE TEST shows letterbox pattern (top/bot uniform, mid varied).")
        print("=> The model returns a 2:1 ERP padded top/bot to 1:1.")
        print("   Fix: crop top 256 + bottom 256 rows -> 1024x512 native ERP.")
    else:
        # Check uniform top-to-bottom content vs stretched
        uniform = all(
            abs(out[k]["content_density_quartiles"]["top    (rows 0..255)"]
                - out[k]["content_density_quartiles"]["lower  (rows 512..767)"]) < 8
            for k in out
        )
        if uniform:
            print("Content density is uniform top-to-bottom in all tests.")
            print("=> The model returns a 1:1 canvas filled with content.")
            print("   This is either:")
            print("     (a) a TRUE 2:1 ERP that has been vertically stretched 2x to fit 1:1")
            print("         (then BICUBIC resize 1024x1024 -> 1024x512 is the CORRECT undo!)")
            print("     (b) a 1:1 'square panorama' that is NOT really an ERP at all")
            print("   Visual inspection of the .deci_512x1024.png files is needed to tell which.")
        else:
            print("Content density varies non-uniformly across rows but is not letterbox.")
            print("=> The model is doing something non-standard (e.g. cylindrical projection,")
            print("   off-centered horizon).  Visual inspection required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
