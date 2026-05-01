"""Tail PSNR/loss out of a nerfstudio tensorboard events file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard not installed; run pip install tensorboard")
    sys.exit(1)


def _scalar_history(ea: EventAccumulator, tag: str) -> list[tuple[int, float]]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(ev.step, float(ev.value)) for ev in ea.Scalars(tag)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("events_dir", type=Path)
    args = p.parse_args()

    cands = list(args.events_dir.rglob("events.out.tfevents.*"))
    if not cands:
        print(f"no events file under {args.events_dir}")
        return 1
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ev = cands[0]
    print(f"reading {ev.relative_to(args.events_dir.parent.parent)}")

    ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"available scalar tags ({len(tags)}):")
    for t in tags:
        print(f"  {t}")

    print()
    interesting = [t for t in tags if any(s in t.lower() for s in (
        "psnr", "ssim", "lpips", "loss", "main_loss", "scale_reg", "num_gauss"
    ))]
    for t in interesting:
        h = _scalar_history(ea, t)
        if not h:
            continue
        first = h[0]
        last = h[-1]
        mid = h[len(h) // 2]
        print(f"{t}:  start@{first[0]}={first[1]:.4f}  mid@{mid[0]}={mid[1]:.4f}  end@{last[0]}={last[1]:.4f}  n={len(h)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
