"""ERP forward warp utilities (vectorized numpy).

Given a source ERP image with metric depth and a source pose, project every
pixel into a target ERP at a target pose using simple z-buffered scatter.
Used by `nvs.py` to build the warped RGB + hole mask for gpt-image-2's edit
endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ERP grid utilities (camera frame)
# ---------------------------------------------------------------------------


def erp_camera_dirs(width: int, height: int) -> np.ndarray:
    """Return (H, W, 3) unit dirs in the camera frame for an ERP grid.

    Convention (matches `erpgen.poses`):
      theta = (u/W - 0.5) * 2*pi  ; theta=0 at center column = +X (forward)
      phi   = (0.5 - v/H) * pi    ; phi=+pi/2 at top row  = +Z (up)
      d.x = cos(phi)*cos(theta)
      d.y = cos(phi)*sin(theta)
      d.z = sin(phi)
    """
    u = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    theta = (u - 0.5) * 2.0 * np.pi  # (W,)
    phi = (0.5 - v) * np.pi  # (H,)
    cphi = np.cos(phi)[:, None]
    sphi = np.sin(phi)[:, None]
    ctheta = np.cos(theta)[None, :]
    stheta = np.sin(theta)[None, :]
    dx = cphi * ctheta
    dy = cphi * stheta
    dz = np.broadcast_to(sphi, dx.shape)
    return np.stack([dx, dy, dz], axis=-1).astype(np.float32)


def erp_dirs_to_uv(dirs: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit dirs in ERP camera frame -> float (u, v) pixel coords."""
    theta = np.arctan2(dirs[..., 1], dirs[..., 0])
    phi = np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0))
    u = (theta / (2.0 * np.pi) + 0.5) * width
    v = (0.5 - phi / np.pi) * height
    return u.astype(np.float32), v.astype(np.float32)


def world_from_erp(
    depth: np.ndarray,
    pose_xyz: np.ndarray,
    pose_R: np.ndarray,
) -> np.ndarray:
    """(H, W) depth + camera pose -> (H, W, 3) world points."""
    H, W = depth.shape
    dirs_cam = erp_camera_dirs(W, H)
    dirs_world = dirs_cam @ pose_R.T.astype(np.float32)
    pts = pose_xyz.astype(np.float32).reshape(1, 1, 3) + dirs_world * depth[..., None]
    return pts


def erp_uv_from_world(
    pts_world: np.ndarray,
    pose_xyz: np.ndarray,
    pose_R: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> ERP (u, v) coords + range, in target camera.

    Returns (u_int, v_int, range), all shape (H_src*W_src,) as int32/float32.
    Out-of-range pixels (v outside [0, H)) get range=+inf.
    """
    pts = pts_world.reshape(-1, 3).astype(np.float32)
    rel = pts - pose_xyz.astype(np.float32).reshape(1, 3)
    cam = rel @ pose_R.astype(np.float32)  # because R is world_from_camera; cam = R^T @ rel = rel @ R
    rng = np.linalg.norm(cam, axis=-1)
    safe = np.maximum(rng, 1e-6)
    cam_n = cam / safe[:, None]
    u, v = erp_dirs_to_uv(cam_n, width, height)
    u_int = np.mod(np.floor(u + 0.5).astype(np.int64), width)
    v_int = np.floor(v + 0.5).astype(np.int64)
    bad = (v_int < 0) | (v_int >= height) | (rng <= 1e-3)
    rng_out = np.where(bad, np.inf, rng).astype(np.float32)
    v_int = np.clip(v_int, 0, height - 1)
    return u_int.astype(np.int32), v_int.astype(np.int32), rng_out


# ---------------------------------------------------------------------------
# Forward warp
# ---------------------------------------------------------------------------


@dataclass
class WarpResult:
    rgb: np.ndarray       # (H, W, 3) uint8
    hole_mask: np.ndarray  # (H, W) uint8, 255 = hole, 0 = filled
    range_buf: np.ndarray  # (H, W) float32, +inf where empty


def forward_warp_erp(
    src_rgb: np.ndarray,
    src_depth: np.ndarray,
    src_xyz: np.ndarray,
    src_R: np.ndarray,
    dst_xyz: np.ndarray,
    dst_R: np.ndarray,
    *,
    out_size: tuple[int, int] | None = None,
) -> WarpResult:
    """Forward-warp source ERP into target ERP via z-buffered scatter.

    Args:
        src_rgb: (Hs, Ws, 3) uint8.
        src_depth: (Hs, Ws) float32 metric.
        src_xyz, src_R: source pose (world_from_camera).
        dst_xyz, dst_R: target pose.
        out_size: optional (W, H) for the target ERP. Defaults to src size.

    Returns:
        WarpResult with rgb (filled where covered, 0 elsewhere) + hole_mask.
    """
    if src_rgb.dtype != np.uint8:
        raise ValueError(f"src_rgb must be uint8, got {src_rgb.dtype}")
    Hs, Ws = src_depth.shape
    if src_rgb.shape[:2] != (Hs, Ws):
        raise ValueError(
            f"shape mismatch: rgb {src_rgb.shape[:2]} vs depth {(Hs, Ws)}"
        )
    if out_size is None:
        Wd, Hd = Ws, Hs
    else:
        Wd, Hd = int(out_size[0]), int(out_size[1])

    pts = world_from_erp(src_depth, src_xyz, src_R)
    valid = np.isfinite(src_depth) & (src_depth > 1e-3)
    if not valid.any():
        return WarpResult(
            rgb=np.zeros((Hd, Wd, 3), dtype=np.uint8),
            hole_mask=np.full((Hd, Wd), 255, dtype=np.uint8),
            range_buf=np.full((Hd, Wd), np.inf, dtype=np.float32),
        )

    pts_flat = pts.reshape(-1, 3)
    rgb_flat = src_rgb.reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    pts_flat = pts_flat[valid_flat]
    rgb_flat = rgb_flat[valid_flat]

    u_int, v_int, rng = erp_uv_from_world(
        pts_flat.reshape(-1, 1, 3), dst_xyz, dst_R, Wd, Hd
    )
    u_int = u_int.reshape(-1)
    v_int = v_int.reshape(-1)
    rng = rng.reshape(-1)
    finite = np.isfinite(rng)
    u_int = u_int[finite]
    v_int = v_int[finite]
    rng = rng[finite]
    rgb_flat = rgb_flat[finite]

    range_buf = np.full((Hd, Wd), np.inf, dtype=np.float32)
    rgb_buf = np.zeros((Hd, Wd, 3), dtype=np.uint8)

    # Sort descending by range so the *smallest* range writes last and wins.
    order = np.argsort(-rng, kind="stable")
    u_o = u_int[order]
    v_o = v_int[order]
    r_o = rng[order]
    rgb_o = rgb_flat[order]

    range_buf[v_o, u_o] = r_o
    rgb_buf[v_o, u_o] = rgb_o

    hole_mask = (range_buf == np.inf).astype(np.uint8) * 255
    return WarpResult(rgb=rgb_buf, hole_mask=hole_mask, range_buf=range_buf)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Binary dilation by `dilate_px` pixels. Uses opencv if available, else
    a numpy max-pool fallback. Input/output are uint8 with {0, 255}."""
    if dilate_px <= 0:
        return mask.copy()
    try:
        import cv2  # type: ignore

        ksize = max(1, int(dilate_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.dilate(mask, kernel, iterations=1)
    except Exception:  # pragma: no cover - fallback path
        out = mask.copy()
        for _ in range(int(dilate_px)):
            shifted = np.maximum.reduce(
                [
                    out,
                    np.roll(out, 1, axis=0),
                    np.roll(out, -1, axis=0),
                    np.roll(out, 1, axis=1),
                    np.roll(out, -1, axis=1),
                ]
            )
            out = shifted
        return out


def hole_mask_to_openai_alpha(hole_mask: np.ndarray, *, base_rgb: np.ndarray | None = None) -> np.ndarray:
    """Build an RGBA mask for OpenAI's `images.edits`.

    OpenAI convention: alpha=0 indicates the area to edit (holes in our case);
    alpha=255 indicates "keep". We optionally embed the source RGB so the
    payload is also a valid stand-alone image.
    """
    H, W = hole_mask.shape
    if base_rgb is None:
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        rgb = base_rgb.astype(np.uint8)
    alpha = np.where(hole_mask > 127, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)
