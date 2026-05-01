"""Depth-Anything-V2 ERP depth + analytic ERP normal estimator.

Replaces the (poor-quality) LLM-painted depth/normal ERPs with a real
monocular-depth estimator that runs locally on GPU. The flow is:

    RGB ERP (PIL, H, W) ---> DAP V2 inference ---> relative depth (H, W)
                          ---> linear scaling to (near_m, far_m)
                          ---> world XYZ via spherical projection
                          ---> world-frame surface normals via depth gradients

We support two ERP-handling modes:

    "direct"         : pass the ERP image straight into DA-V2 (the model is
                       trained on perspective images so the polar regions will
                       be slightly wrong, but it's fast and adequate for the
                       cuboid-room interiors we care about).
    "cubemap_split"  : split the ERP into 6 perspective faces, run DA-V2 on
                       each, then resample the per-face depth back onto the
                       ERP grid. Slower but kinder to the polar regions.
                       (Implemented but `direct` is the default.)

The model is downloaded to `~/.cache/huggingface/` on first use; subsequent
runs are offline. Model size:
    Small  : 99M params,   ~400MB download   (FAST, slightly noisier)
    Base   : 335M params,  ~1.3GB download   (good default)
    Large  : 1.0B params,  ~3.9GB download   (best, slower)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from .warp import erp_camera_dirs


_MODEL_TABLE: dict[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


@dataclass
class DapResult:
    depth_m: np.ndarray            # (H, W) float32, metric meters
    relative_depth: np.ndarray     # (H, W) float32, pre-scaling (relative)
    model_id: str
    inference_ms: float


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


_pipeline_cache: dict[str, "DepthAnythingPipeline"] = {}


class DepthAnythingPipeline:
    """Thin lazy wrapper around HF AutoModelForDepthEstimation."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.model_id = model_id
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        """Returns (H, W) float32 *relative* depth (larger = farther). The
        DAP-V2 raw output is actually inverse depth (larger = closer); we
        invert it here so callers always see "larger = farther"."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # outputs.predicted_depth: (1, H, W) raw inverse depth
        pred = outputs.predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # (1, 1, H, W)
        # Resize back to source resolution
        pred = torch.nn.functional.interpolate(
            pred, size=(image.height, image.width), mode="bicubic", align_corners=False,
        )[0, 0]
        rel_inv = pred.detach().float().cpu().numpy()
        # Convert inverse depth -> proportional depth ("far"=large): take its
        # reciprocal after stabilising at a small floor.
        rel_inv = np.clip(rel_inv, 1e-3, None)
        rel_depth = 1.0 / rel_inv
        return rel_depth.astype(np.float32)


def _get_pipeline(model_size: str = "base", device: str = "cuda") -> DepthAnythingPipeline:
    key = model_size.lower()
    if key not in _MODEL_TABLE:
        raise ValueError(f"Unknown DAP model size: {key!r}. Expected one of {list(_MODEL_TABLE)}")
    model_id = _MODEL_TABLE[key]
    cached = _pipeline_cache.get(model_id)
    if cached is not None:
        return cached
    pipe = DepthAnythingPipeline(model_id=model_id, device=device)
    _pipeline_cache[model_id] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Public ERP depth API
# ---------------------------------------------------------------------------


def estimate_erp_depth(
    erp_rgb: Image.Image,
    *,
    near_m: float,
    far_m: float,
    model_size: str = "base",
    device: str = "cuda",
    mode: Literal["direct", "cubemap_split"] = "direct",
) -> DapResult:
    """Estimate metric ERP depth in meters from a 2:1 ERP RGB image.

    Linearly remaps the model's relative depth into the [near_m, far_m]
    range (5th and 95th percentiles -> near, far) so we get sensible meters
    without any per-scene calibration. This is approximate; for exact metric
    depth you'd run the model with a known-scale reference (DAP V2 metric
    variants exist but require more setup).
    """
    import time as _time

    if mode != "direct":
        # cubemap_split path: split ERP -> 6 cubemap faces -> DAP each -> stitch
        return _estimate_via_cubemap(
            erp_rgb=erp_rgb, near_m=near_m, far_m=far_m,
            model_size=model_size, device=device,
        )

    pipe = _get_pipeline(model_size=model_size, device=device)
    erp_rgb = erp_rgb.convert("RGB")
    t0 = _time.time()
    rel = pipe.predict(erp_rgb)  # (H, W) float32, "far"=large
    inference_ms = (_time.time() - t0) * 1000.0

    metric = _scale_to_metric(rel, near_m=near_m, far_m=far_m)
    return DapResult(
        depth_m=metric, relative_depth=rel,
        model_id=pipe.model_id, inference_ms=inference_ms,
    )


def _scale_to_metric(rel: np.ndarray, *, near_m: float, far_m: float) -> np.ndarray:
    """Robust 5/95 percentile scaling -> [near_m, far_m]."""
    p_lo = float(np.percentile(rel, 5.0))
    p_hi = float(np.percentile(rel, 95.0))
    if p_hi - p_lo < 1e-6:
        return np.full_like(rel, 0.5 * (near_m + far_m), dtype=np.float32)
    norm = np.clip((rel - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    metric = near_m + norm * (far_m - near_m)
    return metric.astype(np.float32)


# ---------------------------------------------------------------------------
# Cubemap-split path (used when the equator-only "direct" mode looks bad
# at the poles). Disabled by default; available via mode="cubemap_split".
# ---------------------------------------------------------------------------


def _estimate_via_cubemap(
    *,
    erp_rgb: Image.Image,
    near_m: float,
    far_m: float,
    model_size: str,
    device: str,
) -> DapResult:
    import time as _time
    from .erp_to_persp import CUBEMAP_FACES, _bilinear_sample, _erp_sample_uv, _face_pixel_dirs

    pipe = _get_pipeline(model_size=model_size, device=device)
    width, height = erp_rgb.size
    rgb_arr = np.array(erp_rgb.convert("RGB"), dtype=np.float32)
    face_size = max(width, height) // 2  # heuristic; e.g. 1024x512 -> 512
    face_dirs = _face_pixel_dirs(face_size, fov_deg=90.0)
    rel_per_pix_sum = np.zeros((height, width), dtype=np.float64)
    rel_per_pix_cnt = np.zeros((height, width), dtype=np.float64)

    t0 = _time.time()
    for fname, R_face in CUBEMAP_FACES.items():
        # sample the ERP at the dirs for this face
        dirs_erp = face_dirs @ R_face.T.astype(np.float32)
        u_erp, v_erp = _erp_sample_uv(dirs_erp, width, height)
        face_rgb = _bilinear_sample(rgb_arr, u_erp, v_erp)
        face_pil = Image.fromarray(np.clip(face_rgb, 0, 255).astype(np.uint8), "RGB")
        rel_face = pipe.predict(face_pil)  # (face_size, face_size)
        # back-project: for each ERP pixel that maps to this face, accumulate
        # we re-use the same u_erp, v_erp grid (face -> ERP) to splat
        u_int = np.clip(np.round(u_erp).astype(np.int32) % width, 0, width - 1)
        v_int = np.clip(np.round(v_erp).astype(np.int32), 0, height - 1)
        np.add.at(rel_per_pix_sum, (v_int, u_int), rel_face.astype(np.float64))
        np.add.at(rel_per_pix_cnt, (v_int, u_int), 1.0)
    inference_ms = (_time.time() - t0) * 1000.0

    cnt = np.maximum(rel_per_pix_cnt, 1.0)
    rel_erp = (rel_per_pix_sum / cnt).astype(np.float32)
    metric = _scale_to_metric(rel_erp, near_m=near_m, far_m=far_m)
    return DapResult(
        depth_m=metric, relative_depth=rel_erp,
        model_id=pipe.model_id, inference_ms=inference_ms,
    )


# ---------------------------------------------------------------------------
# Normal-from-depth (analytic, ERP-aware)
# ---------------------------------------------------------------------------


def normals_from_erp_depth(
    depth_m: np.ndarray,
    *,
    pose_R: np.ndarray | None = None,
    smooth_radius: int = 1,
) -> np.ndarray:
    """Compute world-frame surface normals from an ERP metric depth map.

    Steps:
        1. Project each ERP pixel -> 3D camera-frame point: P_cam = depth * dir_cam
        2. Take 1-pixel finite differences in u and v -> tangent vectors
        3. n_cam = normalize(t_v x t_u) (sign chosen so the normal points
           toward the camera, ie. -dir_cam dot n > 0)
        4. Optionally rotate to world frame with pose_R (world_from_camera).

    Returns (H, W, 3) float32, unit normals.
    """
    H, W = depth_m.shape
    dirs_cam = erp_camera_dirs(W, H).astype(np.float32)  # (H, W, 3)
    P = dirs_cam * depth_m[..., None]                    # (H, W, 3) camera frame XYZ

    # finite differences (with horizontal wrap on u)
    Pu_p = np.roll(P, -1, axis=1)
    Pu_m = np.roll(P, +1, axis=1)
    Pv_p = np.concatenate([P[1:], P[-1:]], axis=0)
    Pv_m = np.concatenate([P[:1], P[:-1]], axis=0)
    t_u = (Pu_p - Pu_m) * 0.5
    t_v = (Pv_p - Pv_m) * 0.5

    n = np.cross(t_v, t_u, axis=-1)  # tangent v x tangent u
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)

    # Make normals face toward the camera (anti-parallel to view dir)
    sign = np.sign(np.sum(-dirs_cam * n, axis=-1, keepdims=True))
    sign = np.where(sign == 0, 1.0, sign)
    n = n * sign

    # smooth slightly to suppress per-pixel speckle from depth noise
    if smooth_radius > 0:
        try:
            from scipy.ndimage import uniform_filter
            n = uniform_filter(n, size=(2 * smooth_radius + 1, 2 * smooth_radius + 1, 1))
            n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        except Exception:
            pass

    if pose_R is not None:
        flat = n.reshape(-1, 3)
        n_world = (np.asarray(pose_R, dtype=np.float32) @ flat.T).T
        n_world = n_world / np.maximum(
            np.linalg.norm(n_world, axis=-1, keepdims=True), 1e-6,
        )
        return n_world.reshape(H, W, 3).astype(np.float32)

    return n.astype(np.float32)


# ---------------------------------------------------------------------------
# PNG encoders for viewer compatibility
# ---------------------------------------------------------------------------


def encode_depth_png(depth_m: np.ndarray, *, near_m: float, far_m: float) -> Image.Image:
    """Encode metric depth as the same RGB PNG format the LLM previously
    produced: white = nearest, black = farthest. (Used by viewer/debug.)"""
    norm = (far_m - depth_m) / max(far_m - near_m, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    g = (norm * 255.0).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    return Image.fromarray(rgb, "RGB")


def encode_normal_png(normal_world: np.ndarray) -> Image.Image:
    """Encode unit normals as RGB PNG: n in [-1,1] -> [0,255]."""
    rgb = np.clip((normal_world * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")
