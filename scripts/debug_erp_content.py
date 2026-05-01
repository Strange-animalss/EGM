"""Check ERP top/bottom half similarity for both 1x and 4x outputs.

If 1x pose_X already has top ~ bottom, the corruption is at gpt-image-2 (LLM
output is "two stacked panoramas"). If 1x is fine but 4x is corrupted, the
SR module is doing something wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "outputs" / "runs" / "cafe_v3_20260430-121553"


def half_diff(arr: np.ndarray) -> float:
    H = arr.shape[0]
    top = arr[: H // 2].astype(np.int32)
    bot = arr[H // 2 :][::-1].astype(np.int32)
    h = min(top.shape[0], bot.shape[0])
    return float(np.abs(top[:h] - bot[:h]).mean())


def half_diff_aligned(arr: np.ndarray) -> float:
    """Top vs bottom WITHOUT vertical flip; if a buggy SR copied the image
    onto itself this returns ~0, even if naive_half_diff would say "not flipped"."""
    H = arr.shape[0]
    top = arr[: H // 2].astype(np.int32)
    bot = arr[H // 2 :].astype(np.int32)
    h = min(top.shape[0], bot.shape[0])
    return float(np.abs(top[:h] - bot[:h]).mean())


def main() -> int:
    erp_1x_dir = RUN / "erp" / "rgb"
    erp_4x_dir = RUN / "erp" / "rgb_4x"
    print(f"{'pose':<8}{'1x_size':<14}{'4x_size':<14}"
          f"{'1x_t-b':<10}{'1x_t-bf':<10}{'4x_t-b':<10}{'4x_t-bf':<10}")
    for i in range(9):
        p1 = erp_1x_dir / f"pose_{i}.png"
        p4 = erp_4x_dir / f"pose_{i}.png"
        a1 = np.asarray(Image.open(p1).convert("RGB"))
        a4 = np.asarray(Image.open(p4).convert("RGB"))
        d1 = half_diff_aligned(a1)
        d1f = half_diff(a1)
        d4 = half_diff_aligned(a4)
        d4f = half_diff(a4)
        print(
            f"pose_{i:<3}{str(a1.shape[:2]):<14}{str(a4.shape[:2]):<14}"
            f"{d1:<10.2f}{d1f:<10.2f}{d4:<10.2f}{d4f:<10.2f}"
        )
    print()
    print("legend: t-b = mean |top half - bottom half| (NOT flipped)")
    print("        t-bf = mean |top half - VERTICALLY-FLIPPED bottom half|")
    print("        if t-b is ~0 -> top == bottom (image stacked twice)")
    print("        if t-bf is small but t-b is large -> normal ERP (mirrored about equator, e.g. uniform sky/floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
