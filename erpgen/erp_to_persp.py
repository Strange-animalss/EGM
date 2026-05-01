"""ERP -> perspective view splitting (cubemap and friends).

For each pose's ERP triplet (RGB / decoded depth / decoded normal), produce:

  * per-face RGB PNG
  * per-face depth PNG (uint16, millimeters; clipped to fastgs depth_far)
  * per-face normal PNG (RGB-encoded world-space unit normals)
  * `cameras.json` listing every (image_path, intrinsics K, world_from_camera R, t).

Cubemap face rotations are expressed in the *ERP camera frame*. To get the
world rotation of a face, multiply by the parent pose: R_world_from_face =
pose.R @ FACE_ROT[name].
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from .poses import Pose, look_at_R
from .warp import erp_dirs_to_uv


# ---------------------------------------------------------------------------
# Face rotations (R_erp_from_face)
# ---------------------------------------------------------------------------


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


CUBEMAP_FACES: dict[str, np.ndarray] = {
    "front": np.eye(3),
    "right": _Rz(-np.pi / 2.0),
    "back": _Rz(np.pi),
    "left": _Rz(np.pi / 2.0),
    "up": _Ry(-np.pi / 2.0),
    "down": _Ry(np.pi / 2.0),
}


def cubemap_face_names() -> list[str]:
    return list(CUBEMAP_FACES.keys())


# ---------------------------------------------------------------------------
# persp48_zigzag: 12 yaw x 4 pitch = 48 views per pose, ordered as a column-
# first zigzag so adjacent frames change in only one axis. This is the format
# SAM3 expects when consuming a perspective sequence as "video".
#
#   yaws    = [0, 30, 60, ..., 330]                       # 12 columns
#   pitches = [+45, +15, -15, -45]                        # 4 rows (deg, +up)
#
# zigzag rule:
#   even columns (yaw=0, 60, 120, ...): pitch goes +45 -> -45 (top to bottom)
#   odd columns  (yaw=30, 90, 150, ...): pitch goes -45 -> +45 (bottom to top)
#
# face name format: "frame_{NNN:03d}_yaw{YAW}_pitch{+/-PITCH}"
# ---------------------------------------------------------------------------


PERSP48_YAWS_DEG: tuple[float, ...] = tuple(float(k * 30.0) for k in range(12))
PERSP48_PITCHES_DEG: tuple[float, ...] = (45.0, 15.0, -15.0, -45.0)


def _format_pitch(p: float) -> str:
    sign = "+" if p >= 0 else "-"
    return f"{sign}{int(round(abs(p)))}"


def _persp48_zigzag_face_rotations() -> "OrderedDict[str, np.ndarray]":
    """48 perspective view rotations (R_erp_from_face) in column-first zigzag.

    Built via `look_at_R(forward, world_up=+Z)`, so:
      - "forward" of every face goes to the (yaw_deg, pitch_deg) direction
        in the ERP camera frame, where ERP convention is
        theta = (u/W - 0.5) * 2*pi  (+Y at azimuth = +90 deg, +X at azimuth=0)
      - "up" of every face is the +Z-aligned cross product, so roll == 0 by
        construction (horizon stays horizontal in the perspective view).

    Frame names follow the SAM3-friendly column-first zigzag pattern:
      col 0  yaw=  0:  pitch +45 -> +15 -> -15 -> -45    (frames  0..3)
      col 1  yaw= 30:  pitch -45 -> -15 -> +15 -> +45    (frames  4..7)
      ...
    Adjacent frames differ in only one axis (yaw or pitch but not both).
    """
    rotations: "OrderedDict[str, np.ndarray]" = OrderedDict()
    world_up = np.array([0.0, 0.0, 1.0])
    frame_idx = 0
    for col, yaw_deg in enumerate(PERSP48_YAWS_DEG):
        pitches = (
            PERSP48_PITCHES_DEG if (col % 2 == 0)
            else tuple(reversed(PERSP48_PITCHES_DEG))
        )
        yaw = np.deg2rad(yaw_deg)
        for pitch_deg in pitches:
            pitch = np.deg2rad(pitch_deg)
            fwd = np.array([
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ])
            R_erp_from_face = look_at_R(fwd, world_up=world_up)
            yaw_token = f"yaw{int(round(yaw_deg))}"
            pitch_token = f"pitch{_format_pitch(pitch_deg)}"
            name = f"frame_{frame_idx:03d}_{yaw_token}_{pitch_token}"
            rotations[name] = R_erp_from_face
            frame_idx += 1
    return rotations


PERSP48_ZIGZAG_FACES: "OrderedDict[str, np.ndarray]" = _persp48_zigzag_face_rotations()


def persp48_zigzag_face_names() -> list[str]:
    return list(PERSP48_ZIGZAG_FACES.keys())


def persp48_zigzag_yaw_pitch() -> list[tuple[float, float]]:
    """Return the (yaw_deg, pitch_deg) sequence in zigzag order, one per frame."""
    out: list[tuple[float, float]] = []
    for col, yaw_deg in enumerate(PERSP48_YAWS_DEG):
        pitches = (
            PERSP48_PITCHES_DEG if (col % 2 == 0)
            else tuple(reversed(PERSP48_PITCHES_DEG))
        )
        for pitch_deg in pitches:
            out.append((float(yaw_deg), float(pitch_deg)))
    return out


def _faces_for_scheme(scheme: str) -> dict[str, np.ndarray]:
    """Lookup table for perspective splitting layouts.

    "persp48_zigzag" is the production default. "cubemap" is kept as a
    geometry primitive used internally by `erpgen.dap` for its
    cubemap-split DAP mode.
    """
    s = (scheme or "persp48_zigzag").lower()
    if s == "persp48_zigzag":
        return PERSP48_ZIGZAG_FACES
    if s == "cubemap":
        return CUBEMAP_FACES
    raise ValueError(
        f"unknown perspective scheme: {scheme!r}; supported: "
        f"persp48_zigzag, cubemap"
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _face_pixel_dirs(out_size: int, fov_deg: float) -> np.ndarray:
    """(out_size, out_size, 3) unit dirs in the FACE camera frame.

    Face camera convention (matches `erpgen.poses`):
        +X forward, +Y left, +Z up.
    Pixel grid: x grows right, y grows down. Principal point at center.
    """
    f = (out_size * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx = out_size * 0.5
    cy = out_size * 0.5
    xs = (np.arange(out_size, dtype=np.float64) + 0.5)
    ys = (np.arange(out_size, dtype=np.float64) + 0.5)
    xv, yv = np.meshgrid(xs, ys)
    dx = np.ones_like(xv)
    dy = -(xv - cx) / f
    dz = -(yv - cy) / f
    d = np.stack([dx, dy, dz], axis=-1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return (d / n).astype(np.float32)


def _erp_sample_uv(dirs_erp: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit dirs (in ERP camera frame) -> ERP pixel coords (u, v) (float)."""
    return erp_dirs_to_uv(dirs_erp, W, H)


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray, *, wrap_u: bool = True) -> np.ndarray:
    """Bilinear sample (H, W[, C]) at float (u, v) coords, with horizontal wrap."""
    H = img.shape[0]
    W = img.shape[1]
    if wrap_u:
        u_mod = np.mod(u, W)
    else:
        u_mod = np.clip(u, 0, W - 1)
    v_mod = np.clip(v, 0, H - 1)
    u0 = np.floor(u_mod).astype(np.int64)
    v0 = np.floor(v_mod).astype(np.int64)
    u1 = (u0 + 1) % W if wrap_u else np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    fu = (u_mod - u0).astype(np.float32)
    fv = (v_mod - v0).astype(np.float32)
    if img.ndim == 2:
        a = img[v0, u0]
        b = img[v0, u1]
        c = img[v1, u0]
        d = img[v1, u1]
        top = a + (b - a) * fu
        bot = c + (d - c) * fu
        return top + (bot - top) * fv
    else:
        a = img[v0, u0]
        b = img[v0, u1]
        c = img[v1, u0]
        d = img[v1, u1]
        fu = fu[..., None]
        fv = fv[..., None]
        top = a + (b - a) * fu
        bot = c + (d - c) * fu
        return top + (bot - top) * fv


def _nearest_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray, *, wrap_u: bool = True) -> np.ndarray:
    H = img.shape[0]
    W = img.shape[1]
    if wrap_u:
        u_int = np.mod(np.floor(u + 0.5).astype(np.int64), W)
    else:
        u_int = np.clip(np.floor(u + 0.5).astype(np.int64), 0, W - 1)
    v_int = np.clip(np.floor(v + 0.5).astype(np.int64), 0, H - 1)
    return img[v_int, u_int]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class FaceView:
    pose_idx: int
    face_name: str
    image_path: str
    depth_path: str
    normal_path: str
    K: list  # 3x3
    R: list  # 3x3 world_from_camera
    t: list  # 3
    width: int
    height: int


@dataclass
class PoseFaceSet:
    pose_idx: int
    pose: Pose
    faces: List[FaceView] = field(default_factory=list)


def split_pose_to_perspectives(
    *,
    pose_idx: int,
    pose: Pose,
    rgb_erp: np.ndarray,
    depth_erp_m: np.ndarray,
    normal_erp_world: np.ndarray | None,
    out_dir: Path,
    scheme: str = "cubemap",
    fov_deg: float = 90.0,
    out_size: int = 1024,
    face_names: list[str] | None = None,
) -> PoseFaceSet:
    """Split one pose's ERP triplet into perspective views and write to disk.

    `scheme` selects the camera layout: "cubemap" (6 faces) or "persp16"
    (8 yaw x 2 pitch = 16 views). `normal_erp_world` is expected in the WORLD
    frame (decode + rotate by pose.R). Pass None to skip writing normal images.
    """
    H, W = depth_erp_m.shape
    if rgb_erp.shape[:2] != (H, W):
        raise ValueError("rgb and depth ERPs must have same H/W")
    face_table = _faces_for_scheme(scheme)
    face_names = face_names or list(face_table.keys())
    f = (out_size * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
    K = np.array([[f, 0.0, out_size * 0.5], [0.0, f, out_size * 0.5], [0.0, 0.0, 1.0]])
    persp_root = out_dir / f"pose_{pose_idx}"
    persp_root.mkdir(parents=True, exist_ok=True)
    face_dirs = {kind: persp_root / kind for kind in ("rgb", "depth", "normal")}
    for d in face_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs_face = _face_pixel_dirs(out_size, fov_deg)
    rgb_f32 = rgb_erp.astype(np.float32)
    depth_f32 = depth_erp_m.astype(np.float32)
    normal_arr = normal_erp_world.astype(np.float32) if normal_erp_world is not None else None
    pose_set = PoseFaceSet(pose_idx=pose_idx, pose=pose)
    for fname in face_names:
        Rerp_from_face = face_table[fname]
        # dirs in ERP frame: (out, out, 3) = dirs_face @ Rerp_from_face^T
        dirs_erp = dirs_face @ Rerp_from_face.T.astype(np.float32)
        u_erp, v_erp = _erp_sample_uv(dirs_erp, W, H)
        face_rgb = _bilinear_sample(rgb_f32, u_erp, v_erp)
        face_rgb = np.clip(face_rgb, 0.0, 255.0).astype(np.uint8)
        # perspective z-depth = range * cos(angle to face forward) = range * d_face.x
        face_range = _bilinear_sample(depth_f32, u_erp, v_erp)
        face_z = face_range * np.clip(dirs_face[..., 0], 1e-6, None)
        depth_mm = np.clip(face_z * 1000.0, 0, 65535).astype(np.uint16)
        rgb_path = face_dirs["rgb"] / f"{fname}.png"
        depth_path = face_dirs["depth"] / f"{fname}.png"
        normal_path = face_dirs["normal"] / f"{fname}.png"
        _SAVE_DPI = (300, 300)
        Image.fromarray(face_rgb, "RGB").save(str(rgb_path), dpi=_SAVE_DPI)
        Image.fromarray(depth_mm, "I;16").save(str(depth_path), dpi=_SAVE_DPI)
        if normal_arr is not None:
            face_n = _bilinear_sample(normal_arr, u_erp, v_erp)
            face_n /= np.maximum(np.linalg.norm(face_n, axis=-1, keepdims=True), 1e-6)
            face_n_rgb = np.clip((face_n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(face_n_rgb, "RGB").save(str(normal_path), dpi=_SAVE_DPI)

        R_world_face = pose.R @ Rerp_from_face
        t_world = pose.xyz.copy()

        pose_set.faces.append(
            FaceView(
                pose_idx=pose_idx,
                face_name=fname,
                image_path=str(rgb_path),
                depth_path=str(depth_path),
                normal_path=str(normal_path) if normal_arr is not None else "",
                K=K.tolist(),
                R=R_world_face.tolist(),
                t=t_world.tolist(),
                width=int(out_size),
                height=int(out_size),
            )
        )
    return pose_set


def split_all_to_perspectives(
    *,
    poses: List[Pose],
    rgb_erps: List[np.ndarray],
    depth_erps_m: List[np.ndarray],
    normal_erps_world: List[np.ndarray] | None,
    out_dir: Path,
    scheme: str = "cubemap",
    fov_deg: float = 90.0,
    out_size: int = 1024,
) -> tuple[List[PoseFaceSet], Path]:
    """Run `split_pose_to_perspectives` for every pose and dump cameras.json."""
    if not (len(poses) == len(rgb_erps) == len(depth_erps_m)):
        raise ValueError("poses / rgb / depth length mismatch")
    if normal_erps_world is not None and len(normal_erps_world) != len(poses):
        raise ValueError("normal length mismatch")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets: List[PoseFaceSet] = []
    for i, pose in enumerate(poses):
        nrm = normal_erps_world[i] if normal_erps_world is not None else None
        s = split_pose_to_perspectives(
            pose_idx=i,
            pose=pose,
            rgb_erp=rgb_erps[i],
            depth_erp_m=depth_erps_m[i],
            normal_erp_world=nrm,
            out_dir=out_dir,
            scheme=scheme,
            fov_deg=fov_deg,
            out_size=out_size,
        )
        sets.append(s)
    cameras_json = out_dir / "cameras.json"
    cameras_json.write_text(
        json.dumps(
            {
                "fov_deg": float(fov_deg),
                "out_size": int(out_size),
                "scheme": str(scheme),
                "views": [face.__dict__ for s in sets for face in s.faces],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sets, cameras_json


# ---------------------------------------------------------------------------
# Helper for the rest of the pipeline
# ---------------------------------------------------------------------------


def rotate_normals_to_world(normal_erp_cam: np.ndarray, pose_R: np.ndarray) -> np.ndarray:
    """Camera-frame normal map -> world-frame normal map.

    Both inputs are (H, W, 3) float; pose_R is world_from_camera.
    """
    flat = normal_erp_cam.reshape(-1, 3)
    out = (pose_R.astype(np.float32) @ flat.T).T
    return out.reshape(normal_erp_cam.shape)
