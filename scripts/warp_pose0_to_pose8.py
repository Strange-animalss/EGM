"""Warp pose_0 ERP RGB+depth into pose_8's viewpoint as a visual sanity-check
preview before regenerating pose_8.

Steps:
  1. Read pose_0 rgb_4x (4096x2048) + depth_m_4x (4096x2048 metric, radial).
  2. Read pose_0 and pose_8 from poses.json.
  3. Forward-warp pose_0 RGBD into pose_8's ERP plane via z-buffered scatter
     (`forward_warp_erp`, our existing utility).
  4. Save:
       outputs/_pose8_warp_preview/warped_rgb.png            (4096x2048)
       outputs/_pose8_warp_preview/warped_holes.png          (B/W: 255 = hole)
       outputs/_pose8_warp_preview/warped_overlay.png        (RGB with red holes)
       outputs/_pose8_warp_preview/source_pose_0.png         (copy of pose_0 ERP)
       outputs/_pose8_warp_preview/current_pose_8_BAD.png    (copy of current bad pose_8)
       outputs/_pose8_warp_preview/stats.json                (hole %, etc)

This intentionally uses ONLY pose_0 as the source — single-view warping is the
worst case (most holes), so it gives parent agent the most pessimistic
preview. If the warp is acceptable from one view, fusing all 8 non-bad poses
later will be strictly better.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.poses import load_poses_json  # noqa: E402
from erpgen.warp import forward_warp_erp  # noqa: E402

RUN = REPO_ROOT / "outputs" / "runs" / "cafe_v3_20260430-121553"
OUT = REPO_ROOT / "outputs" / "_pose8_warp_preview"
OUT.mkdir(parents=True, exist_ok=True)


def _multi_view_warp(src_indices, target_idx) -> tuple[np.ndarray, np.ndarray]:
    """Warp from multiple source poses into the target ERP, fusing by min-range."""
    poses = load_poses_json(RUN / "poses.json")
    target = poses[target_idx]

    Hd, Wd = 2048, 4096
    rgb_buf = np.zeros((Hd, Wd, 3), dtype=np.uint8)
    range_buf = np.full((Hd, Wd), np.inf, dtype=np.float32)

    for src_idx in src_indices:
        src = poses[src_idx]
        src_rgb = np.asarray(
            Image.open(RUN / "erp" / "rgb_4x" / f"pose_{src_idx}.png").convert("RGB")
        )
        src_dep = np.load(RUN / "erp_decoded" / f"pose_{src_idx}_depth_m_4x.npy").astype(np.float32)
        if src_rgb.shape[:2] != src_dep.shape[:2]:
            raise RuntimeError(f"pose_{src_idx} rgb/depth shape mismatch")

        result = forward_warp_erp(
            src_rgb=src_rgb,
            src_depth=src_dep,
            src_xyz=src.xyz,
            src_R=src.R,
            dst_xyz=target.xyz,
            dst_R=target.R,
            out_size=(Wd, Hd),
        )
        better = result.range_buf < range_buf
        range_buf = np.where(better, result.range_buf, range_buf)
        rgb_buf = np.where(better[..., None], result.rgb, rgb_buf)
        print(
            f"  warped pose_{src_idx} -> pose_{target_idx}: "
            f"hole% before fuse = "
            f"{100 * float((result.hole_mask > 127).mean()):.1f}",
            flush=True,
        )

    holes = (range_buf == np.inf).astype(np.uint8) * 255
    return rgb_buf, holes


def main() -> int:
    print(f"[warp_preview] OUT = {OUT}", flush=True)

    print("\n=== single-view warp (pose_0 -> pose_8) ===")
    rgb_single, holes_single = _multi_view_warp([0], 8)
    pct_single = 100 * float((holes_single > 127).mean())
    Image.fromarray(rgb_single, "RGB").save(OUT / "warped_rgb_single.png")
    Image.fromarray(holes_single, "L").save(OUT / "warped_holes_single.png")
    overlay = rgb_single.copy()
    overlay[holes_single > 127] = (255, 0, 0)
    Image.fromarray(overlay, "RGB").save(OUT / "warped_overlay_single.png")

    print("\n=== fused warp (poses 0..7 -> pose_8) ===")
    rgb_fused, holes_fused = _multi_view_warp(list(range(8)), 8)
    pct_fused = 100 * float((holes_fused > 127).mean())
    Image.fromarray(rgb_fused, "RGB").save(OUT / "warped_rgb_fused.png")
    Image.fromarray(holes_fused, "L").save(OUT / "warped_holes_fused.png")
    overlay = rgb_fused.copy()
    overlay[holes_fused > 127] = (255, 0, 0)
    Image.fromarray(overlay, "RGB").save(OUT / "warped_overlay_fused.png")

    src0 = RUN / "erp" / "rgb_4x" / "pose_0.png"
    cur8_4x = RUN / "erp" / "rgb_4x" / "pose_8.png"
    cur8_1x = RUN / "erp" / "rgb" / "pose_8.png"
    import shutil
    shutil.copy2(src0, OUT / "source_pose_0.png")
    shutil.copy2(cur8_4x, OUT / "current_pose_8_BAD_4x.png")
    shutil.copy2(cur8_1x, OUT / "current_pose_8_BAD_1x.png")

    poses = load_poses_json(RUN / "poses.json")
    delta = float(np.linalg.norm(poses[8].xyz - poses[0].xyz))
    stats = {
        "single_view_warp_hole_pct": round(pct_single, 2),
        "fused_warp_hole_pct": round(pct_fused, 2),
        "pose_0_xyz": poses[0].xyz.tolist(),
        "pose_8_xyz": poses[8].xyz.tolist(),
        "translation_distance_m": round(delta, 3),
        "outputs": {
            "single_warp_rgb": "_pose8_warp_preview/warped_rgb_single.png",
            "single_warp_overlay": "_pose8_warp_preview/warped_overlay_single.png",
            "fused_warp_rgb": "_pose8_warp_preview/warped_rgb_fused.png",
            "fused_warp_overlay": "_pose8_warp_preview/warped_overlay_fused.png",
            "current_bad_pose_8_4x": "_pose8_warp_preview/current_pose_8_BAD_4x.png",
            "source_pose_0": "_pose8_warp_preview/source_pose_0.png",
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[warp_preview] stats: {json.dumps(stats, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
