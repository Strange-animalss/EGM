"""Build the FastGS initial point cloud by back-projecting per-pose ERP depth.

Each pose contributes its ERP RGB + metric depth as world-frame coloured points.
We voxel-downsample the union to the configured budget so FastGS can use it
as `points3D` initialization (skipping COLMAP SfM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .poses import Pose
from .warp import erp_camera_dirs


@dataclass
class InitPcd:
    xyz: np.ndarray  # (N, 3) float32 world coords
    rgb: np.ndarray  # (N, 3) uint8


def _backproject_one(
    rgb_erp: np.ndarray,
    depth_erp_m: np.ndarray,
    pose: Pose,
    *,
    stride: int,
    near_m: float,
    far_m: float,
) -> InitPcd:
    H, W = depth_erp_m.shape
    if rgb_erp.shape[:2] != (H, W):
        raise ValueError("rgb / depth shape mismatch")
    rgb = rgb_erp[::stride, ::stride]
    dep = depth_erp_m[::stride, ::stride]
    H2, W2 = dep.shape
    dirs = erp_camera_dirs(W2, H2)
    dirs_world = dirs @ pose.R.T.astype(np.float32)
    pts = pose.xyz.astype(np.float32).reshape(1, 1, 3) + dirs_world * dep[..., None]
    valid = (dep > near_m) & (dep < far_m) & np.isfinite(dep)
    return InitPcd(
        xyz=pts[valid].reshape(-1, 3).astype(np.float32),
        rgb=rgb[valid].reshape(-1, 3).astype(np.uint8),
    )


def voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, *, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.size == 0 or voxel_m <= 0:
        return xyz, rgb
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    flat = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(flat, return_index=True)
    return xyz[idx], rgb[idx]


def cap_points(
    xyz: np.ndarray, rgb: np.ndarray, *, max_points: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb
    rng = rng or np.random.default_rng(0)
    sel = rng.choice(n, size=int(max_points), replace=False)
    return xyz[sel], rgb[sel]


def build_init_pcd(
    *,
    poses: Sequence[Pose],
    rgb_erps: Sequence[np.ndarray],
    depth_erps_m: Sequence[np.ndarray],
    near_m: float,
    far_m: float,
    voxel_m: float,
    max_points: int,
    stride: int = 4,
) -> InitPcd:
    """Run back-projection for every pose, merge, voxel-downsample, cap."""
    if not (len(poses) == len(rgb_erps) == len(depth_erps_m)):
        raise ValueError("len mismatch")
    chunks: List[InitPcd] = []
    for pose, rgb, dep in zip(poses, rgb_erps, depth_erps_m):
        chunks.append(
            _backproject_one(
                rgb, dep, pose,
                stride=int(stride),
                near_m=float(near_m),
                far_m=float(far_m),
            )
        )
    xyz = np.concatenate([c.xyz for c in chunks], axis=0) if chunks else np.zeros((0, 3), np.float32)
    rgb = np.concatenate([c.rgb for c in chunks], axis=0) if chunks else np.zeros((0, 3), np.uint8)
    xyz, rgb = voxel_downsample(xyz, rgb, voxel_m=voxel_m)
    xyz, rgb = cap_points(xyz, rgb, max_points=int(max_points))
    return InitPcd(xyz=xyz, rgb=rgb)


def save_pcd_ply(pcd: InitPcd, path: str | Path) -> Path:
    """Write a binary little-endian xyz+rgb PLY for sanity inspection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(pcd.xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.empty(n, dtype=dt)
    arr["x"] = pcd.xyz[:, 0]
    arr["y"] = pcd.xyz[:, 1]
    arr["z"] = pcd.xyz[:, 2]
    arr["red"] = pcd.rgb[:, 0]
    arr["green"] = pcd.rgb[:, 1]
    arr["blue"] = pcd.rgb[:, 2]
    with open(p, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    return p
