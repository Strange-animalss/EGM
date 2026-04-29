"""Pose generation for the cuboid space.

Conventions (consistent across the whole pipeline):

    World coordinates (right-handed, Z-up):
        +X = forward (camera "front" at yaw=0)
        +Y = left
        +Z = up

    Camera coordinates: same as world basis (we just translate + rotate).
        camera looks along its local +X by default.

    A pose stores (xyz, R) with R = world_from_camera. So a unit vector in
    the camera's local frame `v_cam` maps to the world frame as
    `v_world = R @ v_cam`. The camera origin in world is `xyz`.

    ERP image convention:
        u in [0, W): theta = (u/W - 0.5) * 2*pi  (theta=0 at center column = +X)
        v in [0, H): phi   = (0.5 - v/H) * pi    (phi=+pi/2 at top row = +Z)
        camera-frame direction d_cam:
            d_cam.x = cos(phi)*cos(theta)
            d_cam.y = cos(phi)*sin(theta)
            d_cam.z = sin(phi)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


# ---------------------------------------------------------------------------
# Pose dataclass
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """World-frame camera pose. Camera looks along its local +X."""

    xyz: np.ndarray  # (3,) float64
    R: np.ndarray    # (3, 3) world_from_camera, right-handed
    name: str = ""

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)

    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 0]

    @property
    def left(self) -> np.ndarray:
        return self.R[:, 1]

    @property
    def up(self) -> np.ndarray:
        return self.R[:, 2]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "xyz": self.xyz.tolist(),
            "R": self.R.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        return cls(
            xyz=np.asarray(d["xyz"], dtype=np.float64),
            R=np.asarray(d["R"], dtype=np.float64),
            name=str(d.get("name", "")),
        )


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


def euler_xyz_to_R(euler_deg: Sequence[float]) -> np.ndarray:
    """Intrinsic XYZ Euler angles (degrees) -> 3x3 rotation matrix.

    Order: R = Rz @ Ry @ Rx (extrinsic XYZ == intrinsic ZYX). We apply
    intrinsic XYZ which is `Rx @ Ry @ Rz` in matrix multiplication order,
    matching scipy.spatial.transform 'XYZ'.
    """
    rx, ry, rz = (np.deg2rad(float(a)) for a in euler_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def look_at_R(forward: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """Build a world_from_camera rotation given a desired forward direction.

    Camera basis: x = forward, y = left, z = up.
    """
    f = np.asarray(forward, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        raise ValueError("forward direction is zero")
    f = f / n
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    if abs(float(np.dot(up, f))) > 0.999:
        # f nearly colinear with world_up; pick an alternative up.
        up = np.array([1.0, 0.0, 0.0]) if abs(f[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    left = np.cross(up, f)
    left /= float(np.linalg.norm(left))
    z = np.cross(f, left)
    z /= float(np.linalg.norm(z))
    R = np.column_stack([f, left, z])
    return R


# ---------------------------------------------------------------------------
# Pose set construction
# ---------------------------------------------------------------------------


def _to_array(v) -> np.ndarray:
    if isinstance(v, ListConfig):
        v = OmegaConf.to_container(v, resolve=True)
    return np.asarray(v, dtype=np.float64).reshape(-1)


def make_initial_pose(cfg: DictConfig) -> Pose:
    center = _to_array(cfg.cuboid.center_world)
    local = _to_array(cfg.poses.initial.xyz_local)
    xyz = center + local
    R = euler_xyz_to_R(_to_array(cfg.poses.initial.euler_xyz_deg).tolist())
    return Pose(xyz=xyz, R=R, name="pose_0_center")


def _corner_offsets(size_xyz: np.ndarray, inset: float) -> List[np.ndarray]:
    """Return 8 offset vectors from center to inset corners."""
    half = 0.5 * (1.0 - float(inset)) * size_xyz
    offsets: List[np.ndarray] = []
    # Order: (+,+,+), (+,+,-), (+,-,+), (+,-,-), (-,+,+), (-,+,-), (-,-,+), (-,-,-)
    for sx in (+1, -1):
        for sy in (+1, -1):
            for sz in (+1, -1):
                offsets.append(np.array([sx, sy, sz], dtype=np.float64) * half)
    return offsets


def make_eight_corner_poses(cfg: DictConfig, *, seed: int | None = None) -> List[Pose]:
    center = _to_array(cfg.cuboid.center_world)
    size = _to_array(cfg.cuboid.size_xyz)
    inset = float(cfg.cuboid.corner_inset)
    mode = str(cfg.cuboid.corner_lookat).lower()
    rng = random.Random(seed)
    poses: List[Pose] = []
    for k, off in enumerate(_corner_offsets(size, inset)):
        xyz = center + off
        if mode == "center":
            forward = center - xyz
        elif mode == "outward":
            forward = xyz - center
        elif mode == "random":
            yaw = rng.uniform(-np.pi, np.pi)
            pitch = rng.uniform(-np.deg2rad(15), np.deg2rad(15))
            forward = np.array(
                [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
            )
        else:
            raise ValueError(f"unknown corner_lookat: {mode!r}")
        R = look_at_R(forward)
        poses.append(Pose(xyz=xyz, R=R, name=f"pose_{k + 1}_corner"))
    return poses


def _explicit_generation_poses(cfg: DictConfig) -> List[Pose]:
    center = _to_array(cfg.cuboid.center_world)
    out: List[Pose] = []
    raw = OmegaConf.to_container(cfg.poses.generation, resolve=True)
    if not isinstance(raw, list):
        raise ValueError("poses.generation must be 'auto_8_corners' or a list")
    for k, item in enumerate(raw):
        local = np.asarray(item["xyz_local"], dtype=np.float64).reshape(3)
        if "euler_xyz_deg" in item:
            R = euler_xyz_to_R(item["euler_xyz_deg"])
        elif "lookat" in item:
            target = np.asarray(item["lookat"], dtype=np.float64).reshape(3)
            R = look_at_R(target - (center + local))
        else:
            R = look_at_R(center - (center + local))
        out.append(
            Pose(
                xyz=center + local,
                R=R,
                name=str(item.get("name", f"pose_{k + 1}_custom")),
            )
        )
    return out


def build_pose_set(cfg: DictConfig, *, seed: int | None = None) -> List[Pose]:
    """Return [initial_pose, *generation_poses]."""
    initial = make_initial_pose(cfg)
    spec = cfg.poses.generation
    if isinstance(spec, str) and spec == "auto_8_corners":
        gen = make_eight_corner_poses(cfg, seed=seed)
    else:
        gen = _explicit_generation_poses(cfg)
    return [initial, *gen]


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------


def save_poses_json(path: str | Path, poses: Iterable[Pose]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"poses": [pose.to_dict() for pose in poses]}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_poses_json(path: str | Path) -> List[Pose]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [Pose.from_dict(d) for d in payload["poses"]]
