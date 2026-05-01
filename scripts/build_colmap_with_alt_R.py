"""Rebuild persp48+COLMAP+FastGS-input from an existing ERP run, but with
a different corner_lookat (R matrix) configuration.

Use case: when corner ERPs were produced by a pose-independent route (i2i
ref-image), the same 9 RGB+depth ERPs can be reinterpreted under different
camera-pose R conventions. Only the persp48 split and the COLMAP/init_pcd
output depend on R; the API-generated images do not. This script lets us
amortise one expensive image-generation run across multiple R conventions.

The new run is written to a sibling directory under ``outputs/runs/`` with
ERP arrays hard-linked or copied from the source.

Usage:
    python scripts/build_colmap_with_alt_R.py \
        --source outputs/runs/cafe_v8_nowarp_shared_<ts> \
        --new-run-id cafe_v8c_tilted_<ts> \
        --corner-lookat center
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.colmap_writer import write_colmap_sparse  # noqa: E402
from erpgen.config import load_config, resolve_run_dir, save_resolved_config  # noqa: E402
from erpgen.dap import normals_from_erp_depth  # noqa: E402
from erpgen.erp_to_persp import split_all_to_perspectives  # noqa: E402
from erpgen.init_pcd import build_init_pcd, save_pcd_ply  # noqa: E402
from erpgen.poses import build_pose_set, save_poses_json  # noqa: E402


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # hard link (same filesystem)
    except OSError:
        shutil.copy2(src, dst)


def _copy_erp_dirs(src_run: Path, dst_run: Path) -> None:
    """Bring the per-pose RGB / depth / normal PNGs and the .npy depth/normal
    arrays from the source run into the destination run by hard-link or copy."""
    pairs = [
        ("erp/rgb", "erp/rgb"),
        ("erp/depth", "erp/depth"),
        ("erp/warp", "erp/warp"),
        ("erp/decoder_depth", "erp/decoder_depth"),
        ("erp/decoder_normal", "erp/decoder_normal"),
    ]
    for src_rel, dst_rel in pairs:
        src_dir = src_run / src_rel
        if not src_dir.exists():
            continue
        for f in src_dir.iterdir():
            if f.is_file():
                _link_or_copy(f, dst_run / dst_rel / f.name)
    # depth .npy arrays (pose-independent — depth is in camera-frame
    # ERP coordinates, identical regardless of R).
    decoded_src = src_run / "erp_decoded"
    if decoded_src.exists():
        for f in decoded_src.iterdir():
            if f.suffix == ".npy" and "depth" in f.name:
                _link_or_copy(f, dst_run / "erp_decoded" / f.name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True,
                   help="path to source run dir (must have erp/ + erp_decoded/)")
    p.add_argument("--new-run-id", required=True,
                   help="run id for the rebuilt copy")
    p.add_argument("--corner-lookat", required=True,
                   choices=["center", "outward", "random", "level"],
                   help="new corner pose orientation policy")
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotlist for any extra overrides")
    args = p.parse_args()

    src_run = Path(args.source).resolve()
    if not src_run.is_dir():
        raise SystemExit(f"--source {src_run} is not a directory")

    cfg = load_config(args.config, overrides=args.overrides)
    cfg.cuboid.corner_lookat = args.corner_lookat
    cfg.run.run_id = args.new_run_id
    dst_run = resolve_run_dir(cfg, args.new_run_id)
    save_resolved_config(cfg, dst_run)

    print(f"[alt_R] source: {src_run}")
    print(f"[alt_R] dest:   {dst_run}")
    print(f"[alt_R] corner_lookat = {args.corner_lookat}")

    # ---- 1. rebuild the pose set under the new R policy ----
    poses = build_pose_set(cfg, seed=cfg.prompt.seed)
    save_poses_json(dst_run / "poses.json", poses)
    print(f"[alt_R] wrote {len(poses)} poses ({[p.name for p in poses]})")

    # ---- 2. import the cached source meta + prompts (just for traceability)
    for fname in ("prompts.json", "meta.json"):
        srcp = src_run / fname
        if srcp.exists():
            shutil.copy2(srcp, dst_run / fname.replace(".json", "_source.json"))

    # ---- 3. hard-link / copy ERPs into the new run dir ----
    _copy_erp_dirs(src_run, dst_run)

    # ---- 4. load arrays from disk ----
    rgb_arrs: list[np.ndarray] = []
    depth_arrs: list[np.ndarray] = []
    normal_world_arrs: list[np.ndarray] = []
    Wo: int | None = None
    Ho: int | None = None
    for i, pose in enumerate(poses):
        rgb_path = dst_run / "erp" / "rgb" / f"pose_{i}.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if Wo is None:
            Ho, Wo = rgb.shape[:2]
        depth_npy = dst_run / "erp_decoded" / f"pose_{i}_depth_m.npy"
        depth = np.load(depth_npy).astype(np.float32)
        # Recompute normals under the NEW pose.R since they live in world frame.
        nrm = normals_from_erp_depth(depth, pose_R=pose.R, smooth_radius=1)
        np.save(dst_run / "erp_decoded" / f"pose_{i}_normal_world.npy", nrm)
        rgb_arrs.append(rgb)
        depth_arrs.append(depth)
        normal_world_arrs.append(nrm)

    # ---- 5. perspective split + COLMAP + init_pcd ----
    persp_dir = dst_run / "perspective"
    pose_face_sets, _ = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgb_arrs,
        depth_erps_m=depth_arrs,
        normal_erps_world=normal_world_arrs,
        out_dir=persp_dir,
        scheme=str(cfg.perspective.scheme),
        fov_deg=float(cfg.perspective.fov_deg),
        out_size=int(cfg.perspective.out_size),
    )

    pcd = build_init_pcd(
        poses=poses,
        rgb_erps=rgb_arrs,
        depth_erps_m=depth_arrs,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
        voxel_m=float(cfg.fastgs.init_voxel_m),
        max_points=int(cfg.fastgs.init_max_points),
        stride=4,
    )
    colmap_dir = dst_run / "colmap"
    save_pcd_ply(pcd, colmap_dir / "init_pcd.ply")
    write_colmap_sparse(
        pose_face_sets=pose_face_sets,
        init_pcd=pcd if cfg.fastgs.init_from_points3d else None,
        out_dir=colmap_dir,
        copy_images=True,
    )

    meta = {
        "alt_run_id": args.new_run_id,
        "source_run_id": src_run.name,
        "corner_lookat": args.corner_lookat,
        "num_poses": len(poses),
        "init_pcd_count": int(pcd.xyz.shape[0]),
        "perspective_scheme": str(cfg.perspective.scheme),
        "size": [Wo, Ho],
    }
    (dst_run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[alt_R] done -> {dst_run}")
    print(f"[alt_R] init_pcd points: {pcd.xyz.shape[0]}")
    print(f"[alt_R] persp48 frames written: {sum(len(s.faces) for s in pose_face_sets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
