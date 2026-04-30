"""Smoke test the DAP pipeline on an existing ERP image."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from erpgen.dap import (
    encode_depth_png,
    encode_normal_png,
    estimate_erp_depth,
    normals_from_erp_depth,
)


def main() -> int:
    # Use any existing ERP-ish PNG to test
    # Probe-gen-output from earlier OpenRouter probe = a 1024x1024 image (close to ERP)
    candidate_paths = [
        REPO_ROOT / "outputs" / "_probe_gen.png",
        REPO_ROOT / "outputs" / "_test_openrouter_rgb.png",
    ]
    src = None
    for p in candidate_paths:
        if p.exists():
            src = p
            break
    if src is None:
        print("no candidate test image found")
        return 1

    print(f"--- DAP smoke on {src.name} ---")
    img = Image.open(src).convert("RGB")
    print(f"image size: {img.size}")

    # If it's not 2:1, force 1024x512 to mimic our ERP
    if abs(img.width / img.height - 2.0) > 0.1:
        img = img.resize((1024, 512), Image.BICUBIC)
        print(f"resized to: {img.size}")

    print("\n=== DAP-V2 small model ===")
    t = time.time()
    res = estimate_erp_depth(
        img,
        near_m=0.3, far_m=12.0,
        model_size="small", device="cuda", mode="direct",
    )
    print(f"  total: {time.time()-t:.1f}s, inference: {res.inference_ms:.0f}ms")
    print(f"  model:           {res.model_id}")
    print(f"  depth shape:     {res.depth_m.shape}, dtype: {res.depth_m.dtype}")
    print(f"  depth range:     [{res.depth_m.min():.2f}, {res.depth_m.max():.2f}] m")
    print(f"  depth mean:      {res.depth_m.mean():.2f} m")

    print("\n=== normals from depth ===")
    t = time.time()
    n = normals_from_erp_depth(res.depth_m, pose_R=None, smooth_radius=1)
    print(f"  total: {time.time()-t:.2f}s")
    print(f"  normal shape: {n.shape}, dtype: {n.dtype}")
    norms = np.linalg.norm(n, axis=-1)
    print(f"  unit length check (should be ~1.0): "
          f"min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")

    out_dir = REPO_ROOT / "outputs" / "_dap_smoke"
    out_dir.mkdir(exist_ok=True)
    encode_depth_png(res.depth_m, near_m=0.3, far_m=12.0).save(out_dir / "depth_png.png")
    encode_normal_png(n).save(out_dir / "normal_png.png")
    np.save(out_dir / "depth_m.npy", res.depth_m)
    np.save(out_dir / "normal.npy", n)
    print(f"\nsaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
