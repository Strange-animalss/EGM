"""Build COLMAP datasets for FastGS-Instance training.

Two outputs from a single command-line invocation (controlled by --variant):

  --variant=A  : 8 ERPs (poses 0..7, no pose_8) -> colmap_4x_no_pose8/
  --variant=B  : 9 ERPs (poses 0..8, with new pose_8) -> colmap_4x_v2/

Both use the new persp16_4x4 scheme (4 azim x 4 elev = 16 frames per pose,
matching FastGS-Instance/scripts/erp_to_perspective.py defaults but using
OUR ERP camera convention so we don't have to roll the panorama 90 deg).

Output structure (FastGS-Instance / INRIA-3DGS compatible):
  <out_dir>/
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt
    images/<pose>_<frame>.png    perspective RGB at out_size=1024
    init_pcd.ply                 4x DAP back-projected colour cloud
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.colmap_writer import write_colmap_sparse  # noqa: E402
from erpgen.config import load_config, resolve_run_dir  # noqa: E402
from erpgen.erp_to_persp import split_all_to_perspectives  # noqa: E402
from erpgen.init_pcd import build_init_pcd, save_pcd_ply  # noqa: E402
from erpgen.poses import load_poses_json  # noqa: E402

RUN_ID = "cafe_v3_20260430-121553"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("A", "B"), required=True)
    p.add_argument("--persp-size", type=int, default=1024)
    args = p.parse_args()

    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    run_dir = resolve_run_dir(cfg, RUN_ID)
    poses_all = load_poses_json(run_dir / "poses.json")

    if args.variant == "A":
        keep = list(range(8))
        out_persp = run_dir / "perspective_4x_v2_no_pose8"
        out_colmap = run_dir / "colmap_4x_no_pose8"
    else:
        keep = list(range(9))
        out_persp = run_dir / "perspective_4x_v2"
        out_colmap = run_dir / "colmap_4x_v2"

    print(f"[build_colmap] variant={args.variant}  keep_poses={keep}", flush=True)
    print(f"[build_colmap] -> persp={out_persp.name}  colmap={out_colmap.name}", flush=True)

    poses = [poses_all[i] for i in keep]
    rgbs, deps, nrms = [], [], []
    rgb_dir = run_dir / "erp" / "rgb_4x"
    dec_dir = run_dir / "erp_decoded"
    for orig_idx in keep:
        rgb = np.asarray(Image.open(rgb_dir / f"pose_{orig_idx}.png").convert("RGB"))
        d4 = np.load(dec_dir / f"pose_{orig_idx}_depth_m_4x.npy").astype(np.float32)
        n4 = np.load(dec_dir / f"pose_{orig_idx}_normal_world_4x.npy").astype(np.float32)
        if rgb.shape[:2] != d4.shape[:2] or rgb.shape[:2] != n4.shape[:2]:
            raise ValueError(
                f"pose_{orig_idx} shape mismatch: rgb={rgb.shape}, depth={d4.shape}, normal={n4.shape}"
            )
        rgbs.append(rgb); deps.append(d4); nrms.append(n4)

    if out_persp.exists():
        shutil.rmtree(out_persp)
    if out_colmap.exists():
        shutil.rmtree(out_colmap)

    print(f"[build_colmap] splitting persp16_4x4 at out_size={args.persp_size}...", flush=True)
    t0 = time.time()
    pose_face_sets, _ = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgbs,
        depth_erps_m=deps,
        normal_erps_world=nrms,
        out_dir=out_persp,
        scheme="persp16_4x4",
        fov_deg=90.0,
        out_size=int(args.persp_size),
    )
    n_views = sum(len(s.faces) for s in pose_face_sets)
    print(f"[build_colmap] wrote {n_views} views in {time.time()-t0:.1f}s", flush=True)

    for s in pose_face_sets:
        original_idx = keep[s.pose_idx]
        for face in s.faces:
            old_root = out_persp / f"pose_{s.pose_idx}"
            new_root = out_persp / f"pose_{original_idx}"
            for kind, attr in (
                ("rgb", "image_path"),
                ("depth", "depth_path"),
                ("normal", "normal_path"),
            ):
                old_path = Path(getattr(face, attr))
                if not str(old_path):
                    continue
                new_path = new_root / kind / old_path.name
                setattr(face, attr, str(new_path))
        s.pose_idx = original_idx
    for old_idx in range(len(keep)):
        old_dir = out_persp / f"pose_{old_idx}"
        new_idx = keep[old_idx]
        if old_idx != new_idx and old_dir.exists():
            new_dir = out_persp / f"pose_{new_idx}"
            old_dir.rename(new_dir)

    print(f"[build_colmap] building init pcd...", flush=True)
    t0 = time.time()
    pcd = build_init_pcd(
        poses=poses,
        rgb_erps=rgbs,
        depth_erps_m=deps,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
        voxel_m=float(cfg.fastgs.init_voxel_m),
        max_points=int(cfg.fastgs.init_max_points),
        stride=4,
    )
    print(f"[build_colmap] init pcd: {pcd.xyz.shape[0]} pts in {time.time()-t0:.1f}s", flush=True)

    save_pcd_ply(pcd, out_colmap / "init_pcd.ply")
    write_colmap_sparse(
        pose_face_sets=pose_face_sets,
        init_pcd=pcd,
        out_dir=out_colmap,
        copy_images=True,
    )
    n_imgs = len(list((out_colmap / "images").glob("*.png")))
    print(f"[build_colmap] DONE -> {out_colmap}\n"
          f"  images: {n_imgs}\n"
          f"  init_pcd: {pcd.xyz.shape[0]} pts", flush=True)

    summary = {
        "variant": args.variant,
        "kept_poses": keep,
        "n_views": n_views,
        "n_image_files": n_imgs,
        "init_pcd_count": int(pcd.xyz.shape[0]),
        "scheme": "persp16_4x4",
        "persp_size": int(args.persp_size),
        "fov_deg": 90.0,
        "perspective_dir": str(out_persp.relative_to(run_dir.parent.parent)),
        "colmap_dir": str(out_colmap.relative_to(run_dir.parent.parent)),
    }
    (run_dir / f"colmap_build_{args.variant}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
