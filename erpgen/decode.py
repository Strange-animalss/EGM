"""Decode the colorized depth/normal ERPs produced by gpt-image-2.

Convention (must match what `prompts.py` asks the model to render):
  Depth ERP:  grayscale, white(255) = near, black(0) = far. Linear.
              depth_m(gray) = far_m - (gray/255) * (far_m - near_m)
  Normal ERP: world-space normal encoded as RGB:
              n = (rgb / 255) * 2 - 1, then normalized to unit length.

The depth output is metric (meters). For sanity, all-zero or all-constant
inputs are flagged via `decode_depth_png(..., return_meta=True)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def _to_grayscale(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            return arr
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unexpected depth image shape {arr.shape}")
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


@dataclass
class DepthMeta:
    valid_ratio: float
    near_used: float
    far_used: float
    range_m: tuple[float, float]


def decode_depth_png(
    img: Image.Image | np.ndarray,
    *,
    near_m: float = 0.3,
    far_m: float = 12.0,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, DepthMeta]:
    """Decode a colorized depth ERP into metric meters."""
    gray = _to_grayscale(img)
    g01 = np.clip(gray / 255.0, 0.0, 1.0)
    depth = far_m - g01 * (far_m - near_m)
    if return_meta:
        valid = (g01 > 0.005) & (g01 < 0.995)
        meta = DepthMeta(
            valid_ratio=float(valid.mean()),
            near_used=float(near_m),
            far_used=float(far_m),
            range_m=(float(depth.min()), float(depth.max())),
        )
        return depth.astype(np.float32), meta
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


def decode_normal_png(img: Image.Image | np.ndarray) -> np.ndarray:
    """Decode an RGB-encoded normal ERP into unit normals (H, W, 3)."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unexpected normal image shape {arr.shape}")
    n = arr[..., :3] / 255.0 * 2.0 - 1.0
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.where(nrm < 1e-6, 1.0, nrm)
    return (n / nrm).astype(np.float32)


# ---------------------------------------------------------------------------
# Optional DAP-based linear calibration
# ---------------------------------------------------------------------------


def linear_recalibrate_depth(
    decoded_depth: np.ndarray, ref_depth: np.ndarray
) -> tuple[np.ndarray, tuple[float, float]]:
    """Fit `a*x + b` minimizing L2 on valid pixels and apply it.

    Both inputs must be (H, W) float arrays of the same shape. Returns
    `(calibrated, (a, b))`.
    """
    if decoded_depth.shape != ref_depth.shape:
        h = min(decoded_depth.shape[0], ref_depth.shape[0])
        w = min(decoded_depth.shape[1], ref_depth.shape[1])
        decoded_depth = decoded_depth[:h, :w]
        ref_depth = ref_depth[:h, :w]
    valid = np.isfinite(decoded_depth) & np.isfinite(ref_depth) & (ref_depth > 1e-3)
    x = decoded_depth[valid].astype(np.float64)
    y = ref_depth[valid].astype(np.float64)
    if x.size < 1024:
        return decoded_depth.astype(np.float32), (1.0, 0.0)
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    cal = a * decoded_depth + b
    return cal.astype(np.float32), (a, b)


def try_dap_calibrate(
    rgb_img: Image.Image,
    decoded_depth: np.ndarray,
    *,
    weights_dir: str,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Optional: run DAP on the center RGB and linearly recalibrate.

    Falls back to a no-op (returns the input) if DAP / torch / weights are
    unavailable. Returns `(calibrated_depth, info_dict)`.
    """
    info: dict = {"applied": False, "reason": ""}
    try:
        import torch  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - optional path
        info["reason"] = f"torch not importable: {exc}"
        return decoded_depth.astype(np.float32), info
    try:
        import os
        import sys
        from pathlib import Path

        wdir = Path(weights_dir)
        if not wdir.exists():
            info["reason"] = f"weights_dir not found: {wdir}"
            return decoded_depth.astype(np.float32), info
        sys.path.insert(0, str(wdir))
        prev_cwd = os.getcwd()
        os.chdir(str(wdir))
        try:
            import yaml  # type: ignore
            cfg_path = wdir / "config" / "infer.yaml"
            with open(cfg_path, "r") as f:
                dap_cfg = yaml.safe_load(f)
            dap_cfg["load_weights_dir"] = str(wdir)
            from networks.models import make  # type: ignore
            model = make(dap_cfg["model"])
            state = torch.load(str(wdir / "model.pth"), map_location=device)
            m_state = model.state_dict()
            model.load_state_dict(
                {k: v for k, v in state.items() if k in m_state}, strict=False
            )
            model = model.to(device).eval()
        finally:
            os.chdir(prev_cwd)

        rgb = np.array(rgb_img.convert("RGB"), dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        ten = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model(ten)
            pred = out["pred_depth"][0].squeeze().cpu().numpy().astype(np.float32)
        if pred.shape != decoded_depth.shape:
            import cv2  # type: ignore
            pred = cv2.resize(
                pred,
                (decoded_depth.shape[1], decoded_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        valid = pred[pred > 1e-6]
        if valid.size:
            median_val = float(np.median(valid))
            if median_val > 1e-6:
                pred = pred * (5.0 / median_val)
        cal, (a, b) = linear_recalibrate_depth(decoded_depth, pred)
        info.update({"applied": True, "scale": a, "bias": b})
        return cal, info
    except Exception as exc:  # pragma: no cover - optional path
        info["reason"] = f"DAP path failed: {exc}"
        return decoded_depth.astype(np.float32), info
