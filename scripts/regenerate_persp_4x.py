"""Stage 1.6 driver: re-split persp16 + rebuild COLMAP from 4x ERP data.

Inputs (from a run produced by `scripts/sr_erp_4x.py`):
    outputs/runs/<run_id>/erp/rgb_4x/pose_<i>.png            (4096x2048 RGB SR)
    outputs/runs/<run_id>/erp_decoded/pose_<i>_depth_m_4x.npy
    outputs/runs/<run_id>/erp_decoded/pose_<i>_normal_world_4x.npy

Outputs:
    outputs/runs/<run_id>/perspective_4x/pose_<j>/{rgb,depth,normal}/*.png  (out_size=2048 default)
    outputs/runs/<run_id>/perspective_4x/cameras.json
    outputs/runs/<run_id>/colmap_4x/sparse/0/{cameras,images,points3D}.txt
    outputs/runs/<run_id>/colmap_4x/images/*.png
    outputs/runs/<run_id>/colmap_4x/init_pcd_4x.ply

The persp out_size defaults to 2048 (4x of the original 1024). This is
the right ratio for fov=90 deg because the angular resolution of the new
ERP at 4096-wide is exactly 4x the old ERP's 1024-wide, so a 2048 perspective
view samples it without losing detail.

If the GPU can't fit 144 x 2048 x 2048 splat training, pass --persp-size 1024
to re-split the high-res ERP into normal-size views (still gets the SR-ERP
quality bump because the bilinear sampling now reads from 4x denser pixels).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.colmap_writer import write_colmap_sparse  # noqa: E402
from erpgen.config import latest_run_dir, load_config, resolve_run_dir  # noqa: E402
from erpgen.erp_to_persp import split_all_to_perspectives  # noqa: E402
from erpgen.init_pcd import build_init_pcd, save_pcd_ply  # noqa: E402
from erpgen.poses import load_poses_json  # noqa: E402


def _load_4x_arrays(run_dir: Path, n_poses: int):
    rgb_dir = run_dir / "erp" / "rgb_4x"
    dec_dir = run_dir / "erp_decoded"
    rgbs, deps, nrms = [], [], []
    for i in range(n_poses):
        rgb = np.asarray(Image.open(rgb_dir / f"pose_{i}.png").convert("RGB"))
        d4 = np.load(dec_dir / f"pose_{i}_depth_m_4x.npy")
        n4 = np.load(dec_dir / f"pose_{i}_normal_world_4x.npy")
        if rgb.shape[:2] != d4.shape[:2] or rgb.shape[:2] != n4.shape[:2]:
            raise ValueError(
                f"pose_{i}: shape mismatch rgb={rgb.shape}, depth={d4.shape}, normal={n4.shape}"
            )
        rgbs.append(rgb)
        deps.append(d4.astype(np.float32))
        nrms.append(n4.astype(np.float32))
    return rgbs, deps, nrms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="")
    p.add_argument("--persp-size", type=int, default=1024,
                   help="output size for each perspective view (default 1024 "
                        "for persp48_zigzag; raise to 2048 if VRAM allows).")
    p.add_argument("--persp-scheme", default=None,
                   help="override cfg.perspective.scheme (e.g. persp48_zigzag, persp16).")
    p.add_argument("--persp-dir", default="perspective_4x")
    p.add_argument("--colmap-dir", default="colmap_4x")
    p.add_argument("--init-max-points", type=int, default=400000,
                   help="cap for the 4x init point cloud (denser than 1x).")
    p.add_argument("--init-stride", type=int, default=4,
                   help="ERP backproject stride. With 4x ERP, stride=4 gives "
                        "the same point density as stride=1 on 1x ERP.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_id:
        run_dir = resolve_run_dir(cfg, args.run_id)
    else:
        run_dir = latest_run_dir(cfg)
        if run_dir is None:
            raise SystemExit("no runs found and no --run-id given")
    print(f"[regen_4x] run_dir = {run_dir}", flush=True)

    poses = load_poses_json(run_dir / "poses.json")
    print(f"[regen_4x] {len(poses)} poses", flush=True)

    t0 = time.time()
    rgbs, deps, nrms = _load_4x_arrays(run_dir, len(poses))
    H4, W4 = rgbs[0].shape[:2]
    print(f"[regen_4x] loaded 4x ERP arrays {W4}x{H4}  in {time.time()-t0:.1f}s", flush=True)

    # ---- perspective split ----
    scheme = str(args.persp_scheme or cfg.perspective.scheme)
    persp_dir = run_dir / args.persp_dir
    print(f"[regen_4x] splitting scheme={scheme} at out_size={args.persp_size} -> {persp_dir}", flush=True)
    t0 = time.time()
    pose_face_sets, _ = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgbs,
        depth_erps_m=deps,
        normal_erps_world=nrms,
        out_dir=persp_dir,
        scheme=scheme,
        fov_deg=float(cfg.perspective.fov_deg),
        out_size=int(args.persp_size),
    )
    n_views = sum(len(s.faces) for s in pose_face_sets)
    print(f"[regen_4x] wrote {n_views} perspective view(s)  in {time.time()-t0:.1f}s", flush=True)

    # ---- init point cloud at 4x ----
    print(f"[regen_4x] building 4x init pcd (stride={args.init_stride}, "
          f"voxel_m={cfg.fastgs.init_voxel_m})...", flush=True)
    t0 = time.time()
    pcd = build_init_pcd(
        poses=poses,
        rgb_erps=rgbs,
        depth_erps_m=deps,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
        voxel_m=float(cfg.fastgs.init_voxel_m),
        max_points=int(args.init_max_points),
        stride=int(args.init_stride),
    )
    print(f"[regen_4x] init pcd: {pcd.xyz.shape[0]} points in {time.time()-t0:.1f}s", flush=True)

    colmap_dir = run_dir / args.colmap_dir
    save_pcd_ply(pcd, colmap_dir / "init_pcd_4x.ply")
    write_colmap_sparse(
        pose_face_sets=pose_face_sets,
        init_pcd=pcd if cfg.fastgs.init_from_points3d else None,
        out_dir=colmap_dir,
        copy_images=True,
    )
    print(f"[regen_4x] wrote COLMAP -> {colmap_dir}", flush=True)

    summary = {
        "scale": 4,
        "persp_size": int(args.persp_size),
        "persp_scheme": scheme,
        "n_perspective_views": int(n_views),
        "init_pcd_count_4x": int(pcd.xyz.shape[0]),
        "init_max_points": int(args.init_max_points),
        "init_stride": int(args.init_stride),
        "persp_dir": args.persp_dir,
        "colmap_dir": args.colmap_dir,
        "init_pcd_ply": "init_pcd_4x.ply",
    }
    if scheme == "persp48_zigzag":
        from erpgen.erp_to_persp import persp48_zigzag_yaw_pitch
        summary["zigzag_yaw_pitch"] = [
            {"frame": i, "yaw_deg": yp[0], "pitch_deg": yp[1]}
            for i, yp in enumerate(persp48_zigzag_yaw_pitch())
        ]
    (run_dir / "regen_4x_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[regen_4x] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
