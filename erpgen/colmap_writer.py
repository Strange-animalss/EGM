"""Write FastGS-compatible COLMAP sparse reconstruction (text format).

Produces:

    <out_dir>/sparse/0/cameras.txt
    <out_dir>/sparse/0/images.txt
    <out_dir>/sparse/0/points3D.txt
    <out_dir>/images/<pose>_<face>.png      (copied / linked)

Coordinate convention bridge:

    Our cameras use +X forward, +Y left, +Z up.
    COLMAP cameras use +X right, +Y down, +Z forward (OpenCV).
    R_ours_from_col = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]   so that
    v_world = R_world_ours @ v_ours = R_world_ours @ R_ours_from_col @ v_col
    Hence R_world_col = R_world_ours @ R_ours_from_col.

    images.txt stores the camera_from_world transform: qvec is the unit
    quaternion of R_col_from_world = R_world_col^T, and tvec is
    -R_col_from_world @ camera_origin_world.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .erp_to_persp import FaceView, PoseFaceSet
from .init_pcd import InitPcd


# Frame-change matrices (constant)
R_OURS_FROM_COL = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
R_COL_FROM_OURS = R_OURS_FROM_COL.T


def _R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> (qw, qx, qy, qz). Hamilton convention (COLMAP)."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    return (qw / n, qx / n, qy / n, qz / n)


def _ours_world_pose_to_colmap(R_world_ours: np.ndarray, t_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (R_world_ours, t_world) -> (R_col_from_world, t_col_from_world)."""
    R_world_col = np.asarray(R_world_ours, dtype=np.float64) @ R_OURS_FROM_COL
    R_col_world = R_world_col.T
    t_col_world = -R_col_world @ np.asarray(t_world, dtype=np.float64).reshape(3)
    return R_col_world, t_col_world


@dataclass
class CamerasWritten:
    sparse_dir: Path
    images_dir: Path


def write_colmap_sparse(
    *,
    pose_face_sets: Sequence[PoseFaceSet],
    init_pcd: InitPcd | None,
    out_dir: Path,
    image_root: Path | None = None,
    copy_images: bool = True,
) -> CamerasWritten:
    out_dir = Path(out_dir)
    sparse_dir = out_dir / "sparse" / "0"
    images_dir = out_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ---------- cameras.txt (one shared PINHOLE camera) ----------
    if not pose_face_sets or not pose_face_sets[0].faces:
        raise ValueError("no face views to write")
    f0 = pose_face_sets[0].faces[0]
    K = np.asarray(f0.K, dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    width, height = int(f0.width), int(f0.height)
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}",
    ]
    (sparse_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    # ---------- images.txt ----------
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    image_id = 1
    for pset in pose_face_sets:
        for face in pset.faces:
            R_w_ours = np.asarray(face.R, dtype=np.float64)
            t_world = np.asarray(face.t, dtype=np.float64)
            R_col_world, t_col_world = _ours_world_pose_to_colmap(R_w_ours, t_world)
            qw, qx, qy, qz = _R_to_quat(R_col_world)
            tx, ty, tz = (float(v) for v in t_col_world)
            name = f"pose_{face.pose_idx}_{face.face_name}.png"
            img_lines.append(
                f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.6f} {ty:.6f} {tz:.6f} 1 {name}"
            )
            img_lines.append("")  # empty POINTS2D line
            if copy_images:
                src = Path(face.image_path)
                if src.exists():
                    shutil.copy2(src, images_dir / name)
            image_id += 1
    (sparse_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    # ---------- points3D.txt ----------
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    if init_pcd is not None and init_pcd.xyz.shape[0] > 0:
        for i, (xyz, rgb) in enumerate(zip(init_pcd.xyz, init_pcd.rgb), start=1):
            pts_lines.append(
                f"{i} {float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.0"
            )
    (sparse_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n", encoding="utf-8")

    return CamerasWritten(sparse_dir=sparse_dir, images_dir=images_dir)
