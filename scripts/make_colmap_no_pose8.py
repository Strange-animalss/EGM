"""Build a copy of colmap_4x with all pose_8 views (and the corresponding
init point cloud contributions) removed.

Output dir: <run>/colmap_4x_no_pose8/
  sparse/0/cameras.txt   <- copied verbatim
  sparse/0/images.txt    <- pose_8 images dropped (and remaining IDs renumbered to be contiguous)
  sparse/0/points3D.txt  <- copied verbatim from a fresh init_pcd built without pose_8
  images/                <- copied excluding pose_8_*.png
  init_pcd.ply           <- DAP-derived 4x init pcd from poses 0..7 only
"""

from __future__ import annotations

import shutil
import sys
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


def main() -> int:
    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    run_dir = resolve_run_dir(cfg, "cafe_v3_20260430-121553")
    print(f"[no_pose8] run_dir = {run_dir}", flush=True)

    keep_pose_idxs = list(range(8))

    poses_all = load_poses_json(run_dir / "poses.json")
    poses = [poses_all[i] for i in keep_pose_idxs]
    print(f"[no_pose8] keeping {len(poses)} of {len(poses_all)} poses", flush=True)

    rgb_dir = run_dir / "erp" / "rgb_4x"
    dec_dir = run_dir / "erp_decoded"
    rgbs, deps, nrms = [], [], []
    for i in keep_pose_idxs:
        rgbs.append(np.asarray(Image.open(rgb_dir / f"pose_{i}.png").convert("RGB")))
        deps.append(np.load(dec_dir / f"pose_{i}_depth_m_4x.npy").astype(np.float32))
        nrms.append(np.load(dec_dir / f"pose_{i}_normal_world_4x.npy").astype(np.float32))

    out_root = run_dir / "colmap_4x_no_pose8"
    if out_root.exists():
        shutil.rmtree(out_root)

    persp_tmp = run_dir / "_persp_tmp_no_pose8"
    if persp_tmp.exists():
        shutil.rmtree(persp_tmp)

    print(f"[no_pose8] re-running persp48_zigzag split for kept poses (in-memory only)...", flush=True)
    pose_face_sets, _ = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgbs,
        depth_erps_m=deps,
        normal_erps_world=nrms,
        out_dir=persp_tmp,
        scheme="persp48_zigzag",
        fov_deg=float(cfg.perspective.fov_deg),
        out_size=int(cfg.perspective.out_size),
    )
    n_views = sum(len(s.faces) for s in pose_face_sets)
    print(f"[no_pose8] {n_views} views in pose_face_sets", flush=True)

    real_persp = run_dir / "perspective_4x"
    for s in pose_face_sets:
        original_pose_idx = keep_pose_idxs[s.pose_idx]
        for face in s.faces:
            real_face_path = real_persp / f"pose_{original_pose_idx}" / "rgb" / f"{face.face_name}.png"
            face.image_path = str(real_face_path)
            face.depth_path = str(real_persp / f"pose_{original_pose_idx}" / "depth" / f"{face.face_name}.png")
            face.normal_path = str(real_persp / f"pose_{original_pose_idx}" / "normal" / f"{face.face_name}.png")
        s.pose_idx = original_pose_idx

    print(f"[no_pose8] building init pcd from poses 0..7...", flush=True)
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
    print(f"[no_pose8] init pcd: {pcd.xyz.shape[0]} points", flush=True)

    save_pcd_ply(pcd, out_root / "init_pcd.ply")
    write_colmap_sparse(
        pose_face_sets=pose_face_sets,
        init_pcd=pcd,
        out_dir=out_root,
        copy_images=True,
    )

    if persp_tmp.exists():
        shutil.rmtree(persp_tmp)

    sparse_imgs = (out_root / "sparse" / "0" / "images.txt").read_text(encoding="utf-8").splitlines()
    n_image_lines = sum(1 for ln in sparse_imgs if ln and not ln.startswith("#"))
    n_images = (n_image_lines + 1) // 2
    n_image_files = len(list((out_root / "images").glob("*.png")))
    print(
        f"[no_pose8] DONE -> {out_root}\n"
        f"  images.txt entries (after id 0..N renumber): ~{n_images}\n"
        f"  images/*.png count: {n_image_files}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
