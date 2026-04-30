"""ERP-aware 4x super-resolution via Real-ESRGAN (RRDBNet x4plus).

Naively feeding a 360 degree ERP RGB image to a SR model gives a visible
seam at the +/-180 deg longitude where the left and right edges meet, because
the convolutional model has no idea that the image wraps around. Two extras
on top of a vanilla `model.forward(rgb)`:

  1. Horizontal wraparound padding. Before SR we tile the input by `wrap_pad`
     pixels on each horizontal side using opposite-edge content (left padding
     gets pulled from the rightmost columns and vice versa). After SR we crop
     the same padding (now `wrap_pad * scale`) back off. The seam is now
     fully inside the model's receptive field on both passes, so the upscaled
     ERP is wraparound-safe.

  2. Vertical reflection at the top and bottom. The poles are still slightly
     stretched but at least the model doesn't see hard zero borders.

This module deliberately depends only on `torch`, `numpy`, `PIL`, and
`spandrel`, so it works against the existing torch 2.7+cu128 install on
Windows without dragging in basicsr/realesrgan (which break against modern
torchvision).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "third_party" / "weights" / "RealESRGAN_x4plus.pth"


@dataclass
class _SrCtx:
    model: torch.nn.Module
    scale: int
    device: torch.device
    half: bool


_CTX: Optional[_SrCtx] = None


def _load_model(weights_path: Path, *, device: torch.device, half: bool) -> _SrCtx:
    from spandrel import ModelLoader

    loader = ModelLoader()
    desc = loader.load_from_file(str(weights_path))
    if desc.scale != 4:
        raise ValueError(f"expected x4 SR model, got scale={desc.scale}")
    model = desc.model.to(device).eval()
    if half and device.type == "cuda":
        model = model.half()
    return _SrCtx(model=model, scale=int(desc.scale), device=device, half=half)


def _ensure_ctx(weights_path: Path | None = None, *, half: bool = True) -> _SrCtx:
    global _CTX
    if _CTX is None:
        wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not wp.is_file():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found at {wp}. Download "
                f"RealESRGAN_x4plus.pth from "
                f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _CTX = _load_model(wp, device=device, half=half and device.type == "cuda")
    return _CTX


def _to_chw(rgb: np.ndarray, *, half: bool, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    if half:
        t = t.half()
    return t


def _to_hwc_u8(t: torch.Tensor) -> np.ndarray:
    arr = t.clamp(0.0, 1.0).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _wrap_pad_h(rgb: np.ndarray, pad: int) -> np.ndarray:
    """Pad horizontally with opposite-edge content (ERP wraparound)."""
    H, W = rgb.shape[:2]
    if pad <= 0:
        return rgb
    if pad >= W:
        raise ValueError(f"wrap_pad={pad} must be < width={W}")
    left = rgb[:, W - pad:]
    right = rgb[:, :pad]
    return np.concatenate([left, rgb, right], axis=1)


def _reflect_pad_v(rgb: np.ndarray, pad: int) -> np.ndarray:
    """Pad top/bottom with reflected rows (cheaper than tile-from-pole)."""
    if pad <= 0:
        return rgb
    top = rgb[:pad][::-1]
    bot = rgb[-pad:][::-1]
    return np.concatenate([top, rgb, bot], axis=0)


def _tiled_forward(ctx: _SrCtx, x: torch.Tensor, *, tile: int, overlap: int) -> torch.Tensor:
    """SR forward in overlapping tiles to bound peak VRAM.

    The 1024x512 ERP gets to 1280x640 after wrap+reflect padding, which is
    small enough to fit in 8 GB VRAM in a single forward, so we treat tile=0
    as "no tiling".
    """
    if tile <= 0:
        with torch.inference_mode():
            return ctx.model(x)
    _, _, H, W = x.shape
    s = ctx.scale
    out = torch.zeros((1, 3, H * s, W * s), dtype=x.dtype, device=x.device)
    weight = torch.zeros((1, 1, H * s, W * s), dtype=x.dtype, device=x.device)
    step = max(tile - overlap, 1)
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch = x[:, :, y0:y1, x0:x1]
            with torch.inference_mode():
                up = ctx.model(patch)
            yo, xo = y0 * s, x0 * s
            yh, xw = up.shape[-2], up.shape[-1]
            out[:, :, yo:yo + yh, xo:xo + xw] += up
            weight[:, :, yo:yo + yh, xo:xo + xw] += 1.0
    return out / weight.clamp(min=1.0)


def upscale_erp_4x(
    img: Image.Image,
    *,
    wrap_pad: int = 128,
    reflect_pad_v: int = 0,
    weights_path: str | Path | None = None,
    half: bool = True,
    tile: int = 0,
    overlap: int = 32,
) -> Image.Image:
    """Run Real-ESRGAN x4 on a 2:1 ERP RGB image, with wraparound-safe seams.

    Args:
        img: PIL RGB image, ideally 2:1 aspect (e.g. 1024x512).
        wrap_pad: pixels of horizontal wrap padding (in input resolution).
        reflect_pad_v: pixels of vertical reflection padding (default 0).
        weights_path: optional override for the .pth file.
        half: run in fp16 on CUDA (~2x speed, no visible quality loss for x4plus).
        tile: tile size in pixels at input resolution; 0 = single forward.
        overlap: tile overlap in pixels (only used when tile > 0).

    Returns:
        PIL RGB image, 4x in each axis (e.g. 4096x2048 from 1024x512).
    """
    ctx = _ensure_ctx(weights_path=weights_path, half=half)
    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    rgb_pad = _wrap_pad_h(rgb, wrap_pad)
    rgb_pad = _reflect_pad_v(rgb_pad, reflect_pad_v)
    x = _to_chw(rgb_pad, half=ctx.half, device=ctx.device)
    y = _tiled_forward(ctx, x, tile=tile, overlap=overlap)
    out = _to_hwc_u8(y)
    s = ctx.scale
    if reflect_pad_v > 0:
        out = out[reflect_pad_v * s:-reflect_pad_v * s, :, :]
    if wrap_pad > 0:
        out = out[:, wrap_pad * s:-wrap_pad * s, :]
    expected = (H * s, W * s, 3)
    if out.shape != expected:
        raise RuntimeError(f"SR shape mismatch: got {out.shape}, expected {expected}")
    return Image.fromarray(out, "RGB")


def upscale_array_bilinear(arr: np.ndarray, scale: int) -> np.ndarray:
    """Cheap 4x upsample for continuous fields (depth, normal) via bilinear.

    Wraps horizontally so the +/-180 deg seam keeps continuous values.
    """
    if scale == 1:
        return arr
    if arr.ndim == 2:
        H, W = arr.shape
        rgb_like = arr[..., None]
    elif arr.ndim == 3:
        H, W, _ = arr.shape
        rgb_like = arr
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    pad = 1
    left = rgb_like[:, W - pad:]
    right = rgb_like[:, :pad]
    padded = np.concatenate([left, rgb_like, right], axis=1)
    t = torch.from_numpy(padded.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    Hu = H * scale
    Wu = (W + 2 * pad) * scale
    up = torch.nn.functional.interpolate(t, size=(Hu, Wu), mode="bilinear", align_corners=False)
    up = up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    up = up[:, pad * scale:pad * scale + W * scale, :]
    if arr.ndim == 2:
        up = up[..., 0]
    return up.astype(arr.dtype)


def horizontal_seam_score(img: Image.Image | np.ndarray, *, n_cols: int = 1) -> float:
    """Mean absolute pixel difference between the leftmost and rightmost columns.

    Used as a quick sanity check that the SR output really wraps. Lower is
    better; a hard seam in pixel-space SR typically gives values > 8 (out of
    255), wrap-aware SR gets it down to 1-3.
    """
    arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    arr = arr.astype(np.float32)
    left = arr[:, :n_cols]
    right = arr[:, -n_cols:]
    return float(np.mean(np.abs(left - right)))
