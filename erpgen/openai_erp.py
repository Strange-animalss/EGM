"""OpenAI gpt-image-2 wrapper used by the ERP pipeline.

We expose a small surface area:
  - generate_rgb(prompt, size)            -> PIL.Image (RGB)
  - edit_with_mask(prompt, image, mask)   -> PIL.Image (RGB)
  - ImageClient.generate_aligned_triplet  -> (rgb, depth_img, normal_img)

The wrapper is fully cache-aware: every request's key is sha256 of
(model, size, quality, prompt, ref_bytes, mask_bytes), so reruns of the
pipeline on the same prompt+pose are free.

If `mock=True` (or `OPENAI_API_KEY` is unset and `allow_mock=True`), the
wrapper falls back to a local procedural renderer that produces synthetic but
self-consistent ERP triplets, so downstream stages can be exercised without
spending API credits.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .warp import erp_camera_dirs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OpenAIConfig:
    model: str = "gpt-image-2"
    size: str = "3840x1920"
    rgb_quality: str = "high"
    edit_quality: str = "high"
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout_sec: float = 180.0
    max_retries: int = 4
    retry_backoff_sec: float = 5.0
    cache_dir: str = "outputs/.openai_cache"
    text_model: str = "gpt-5.5-pro"
    text_model_api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = "high"
    image_reasoning_effort: str | None = None

    @classmethod
    def from_dict(cls, cfg) -> "OpenAIConfig":
        from omegaconf import OmegaConf
        d = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "_content") else dict(cfg)
        image_re = d.get("image_reasoning_effort", None)
        return cls(
            model=str(d.get("model", "gpt-image-2")),
            size=str(d.get("size", "3840x1920")),
            rgb_quality=str(d.get("rgb_quality", "high")),
            edit_quality=str(d.get("edit_quality", "high")),
            api_key_env=str(d.get("api_key_env", "OPENAI_API_KEY")),
            request_timeout_sec=float(d.get("request_timeout_sec", 180.0)),
            max_retries=int(d.get("max_retries", 4)),
            retry_backoff_sec=float(d.get("retry_backoff_sec", 5.0)),
            cache_dir=str(d.get("cache_dir", "outputs/.openai_cache")),
            text_model=str(d.get("text_model", "gpt-5.5-pro")),
            text_model_api_key_env=str(d.get("text_model_api_key_env", "OPENAI_API_KEY")),
            reasoning_effort=str(d.get("reasoning_effort", "high")),
            image_reasoning_effort=str(image_re) if image_re is not None else None,
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGBA" if img.mode == "RGBA" else "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _hash_request(parts: Iterable[bytes | str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            h.update(p.encode("utf-8"))
        else:
            h.update(p)
        h.update(b"\x1f")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Image client
# ---------------------------------------------------------------------------


class ImageClient:
    """Thin wrapper around openai.images.* with caching and retries."""

    def __init__(
        self,
        cfg: OpenAIConfig,
        *,
        mock: bool = False,
        allow_mock_fallback: bool = True,
        verbose: bool = True,
    ) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mock = bool(mock)

        self._client = None
        if not self._mock:
            api_key = os.environ.get(cfg.api_key_env, "").strip()
            if not api_key:
                if allow_mock_fallback:
                    self._mock = True
                    if verbose:
                        print(
                            f"[openai_erp] {cfg.api_key_env} not set; "
                            f"falling back to mock mode."
                        )
                else:
                    raise RuntimeError(
                        f"{cfg.api_key_env} env var is required but not set"
                    )
            else:
                try:
                    from openai import OpenAI  # noqa: WPS433

                    self._client = OpenAI(api_key=api_key, timeout=cfg.request_timeout_sec)
                except Exception as exc:  # pragma: no cover - import error path
                    if allow_mock_fallback:
                        self._mock = True
                        if verbose:
                            print(f"[openai_erp] OpenAI SDK not usable ({exc}); mock mode.")
                    else:
                        raise

    # ------------------------------ caching -------------------------------

    def _load_cached(self, key: str) -> Optional[Image.Image]:
        path = self._cache_dir / f"{key}.png"
        if path.exists():
            try:
                return Image.open(path).copy()
            except Exception:
                return None
        return None

    def _save_cached(self, key: str, img: Image.Image) -> None:
        path = self._cache_dir / f"{key}.png"
        img.save(path, format="PNG")

    # ------------------------------ helpers -------------------------------

    @property
    def mock_mode(self) -> bool:
        return self._mock

    def parse_size(self, size_str: str | None = None) -> tuple[int, int]:
        s = size_str or self.cfg.size
        w, h = s.lower().split("x")
        return int(w), int(h)

    # ------------------------------ generate ------------------------------

    def generate_rgb(self, prompt: str, *, size: str | None = None) -> Image.Image:
        size = size or self.cfg.size
        key = _hash_request([
            "generate", self.cfg.model, size, self.cfg.rgb_quality, prompt,
            self.cfg.image_reasoning_effort or "",
        ])
        cached = self._load_cached(key)
        if cached is not None:
            return cached

        if self._mock:
            img = _mock_rgb_erp(self.parse_size(size), prompt)
            self._save_cached(key, img)
            return img

        img = self._call_with_retry(
            self._call_generate, prompt=prompt, size=size, quality=self.cfg.rgb_quality
        )
        self._save_cached(key, img)
        return img

    def edit_with_mask(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        *,
        size: str | None = None,
        quality: str | None = None,
    ) -> Image.Image:
        size = size or self.cfg.size
        quality = quality or self.cfg.edit_quality
        ref_bytes = _png_bytes(image)
        mask_bytes = _png_bytes(mask)
        key = _hash_request([
            "edits", self.cfg.model, size, quality, prompt, ref_bytes, mask_bytes,
            self.cfg.image_reasoning_effort or "",
        ])
        cached = self._load_cached(key)
        if cached is not None:
            return cached

        if self._mock:
            img = _mock_edit_erp(self.parse_size(size), image, mask, prompt)
            self._save_cached(key, img)
            return img

        img = self._call_with_retry(
            self._call_edit,
            prompt=prompt,
            image_bytes=ref_bytes,
            mask_bytes=mask_bytes,
            size=size,
            quality=quality,
        )
        self._save_cached(key, img)
        return img

    # ------------------------- inner: real API ---------------------------

    def _call_generate(self, *, prompt: str, size: str, quality: str) -> Image.Image:
        kwargs: dict = dict(
            model=self.cfg.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        if self.cfg.image_reasoning_effort:
            kwargs["reasoning_effort"] = self.cfg.image_reasoning_effort
        resp = self._client.images.generate(**kwargs)  # type: ignore[union-attr]
        b64 = resp.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI generate returned empty b64_json")
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def _call_edit(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mask_bytes: bytes,
        size: str,
        quality: str,
    ) -> Image.Image:
        img_io = io.BytesIO(image_bytes)
        img_io.name = "image.png"
        mask_io = io.BytesIO(mask_bytes)
        mask_io.name = "mask.png"
        kwargs = dict(
            model=self.cfg.model,
            prompt=prompt,
            image=img_io,
            mask=mask_io,
            size=size,
            quality=quality,
            n=1,
        )
        if self.cfg.image_reasoning_effort:
            kwargs["reasoning_effort"] = self.cfg.image_reasoning_effort
        resp = self._client.images.edits(**kwargs)  # type: ignore[union-attr]
        b64 = resp.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI edit returned empty b64_json")
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def _call_with_retry(self, fn, **kwargs) -> Image.Image:
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return fn(**kwargs)
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if self.verbose:
                    print(
                        f"[openai_erp] {fn.__name__} attempt {attempt + 1}/"
                        f"{self.cfg.max_retries + 1} failed: {exc}"
                    )
                if attempt == self.cfg.max_retries:
                    break
                time.sleep(self.cfg.retry_backoff_sec * (2**attempt))
        raise RuntimeError(f"OpenAI API failed after retries: {last_exc}")

    # ------------------------- aligned triplet ---------------------------

    def generate_aligned_triplet(
        self,
        prompts: dict[str, str],
        *,
        ref_rgb: Image.Image | None = None,
        ref_mask: Image.Image | None = None,
        size: str | None = None,
    ) -> dict[str, Image.Image]:
        """Generate 3 ERPs (rgb / depth / normal).

        If `ref_rgb` is None: 3 independent generate() calls (used for the
        center pose). Otherwise we route everything through edits() with
        ref_rgb + ref_mask so the depth/normal share the warped layout.
        """
        out: dict[str, Image.Image] = {}
        if ref_rgb is None:
            for k in ("rgb", "depth", "normal"):
                out[k] = self.generate_rgb(prompts[k], size=size)
        else:
            if ref_mask is None:
                raise ValueError("ref_mask is required when ref_rgb is provided")
            out["rgb"] = self.edit_with_mask(
                prompts["rgb"], ref_rgb, ref_mask, size=size
            )
            out["depth"] = self.edit_with_mask(
                prompts["depth"], out["rgb"], ref_mask, size=size
            )
            out["normal"] = self.edit_with_mask(
                prompts["normal"], out["rgb"], ref_mask, size=size
            )
        return out


# ---------------------------------------------------------------------------
# Mock renderer
# ---------------------------------------------------------------------------
#
# These produce synthetic but pose-consistent ERP triplets so the pipeline
# (decode -> cubemap -> COLMAP -> FastGS -> viewer) can be smoke-tested
# without an OpenAI key. The mock encodes:
#   * RGB: a colorful checker-room with hue keyed by prompt hash
#   * Depth: distance to the unit-cube room walls (white near, black far),
#           range mapped into the configured near/far so decode roundtrips
#   * Normal: world-space normal of the dominant cube wall in each pixel
# Mock does NOT account for the camera pose -- that's fine because the mock
# is only for plumbing tests.


def _prompt_hue(prompt: str) -> float:
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return (h[0] / 255.0) % 1.0


def _ray_box_distance(dirs: np.ndarray, half: tuple[float, float, float]) -> np.ndarray:
    """Distance from origin to an axis-aligned cube of half-extents `half`,
    along each ray direction."""
    eps = 1e-6
    safe = np.where(np.abs(dirs) < eps, eps, dirs)
    hx, hy, hz = half
    t1 = hx / np.abs(safe[..., 0])
    t2 = hy / np.abs(safe[..., 1])
    t3 = hz / np.abs(safe[..., 2])
    return np.minimum(np.minimum(t1, t2), t3)


def _mock_rgb_erp(size_wh: tuple[int, int], prompt: str) -> Image.Image:
    width, height = size_wh
    dirs = erp_camera_dirs(width, height)
    box_half = (1.6, 1.6, 1.2)
    t = _ray_box_distance(dirs, box_half)
    hits = dirs * t[..., None]
    base_hue = _prompt_hue(prompt)
    # checker pattern in 2D on the dominant axis face
    abs_hits = np.abs(hits)
    axis = np.argmax(abs_hits, axis=-1)
    coords = np.where(axis[..., None] == 0, hits[..., 1:3],
                      np.where(axis[..., None] == 1, hits[..., 0:3:2], hits[..., 0:2]))
    cell = np.floor(coords * 4.0).astype(np.int32)
    chk = ((cell[..., 0] + cell[..., 1]) & 1).astype(np.float32)
    # hue varies per face
    hue_offset = (axis.astype(np.float32) * 0.17 + base_hue) % 1.0
    sat = 0.55 + 0.25 * chk
    val = 0.6 + 0.3 * chk
    hsv = np.stack([hue_offset, sat, val], axis=-1)
    rgb = _hsv_to_rgb(hsv)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _mock_edit_erp(
    size_wh: tuple[int, int],
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
) -> Image.Image:
    width, height = size_wh
    if "GRAYSCALE DEPTH MAP" in prompt:
        return _mock_depth_erp(size_wh)
    if "SURFACE NORMAL MAP" in prompt:
        return _mock_normal_erp(size_wh)
    base = image.convert("RGB").resize((width, height), Image.BICUBIC)
    fill = _mock_rgb_erp(size_wh, prompt + "_edit")
    m = mask.convert("L").resize((width, height), Image.NEAREST)
    arr_base = np.array(base, dtype=np.uint8)
    arr_fill = np.array(fill, dtype=np.uint8)
    m_arr = (np.array(m, dtype=np.float32) / 255.0)[..., None]
    out = arr_base * (1.0 - m_arr) + arr_fill * m_arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _mock_depth_erp(size_wh: tuple[int, int]) -> Image.Image:
    width, height = size_wh
    dirs = erp_camera_dirs(width, height)
    t = _ray_box_distance(dirs, (1.6, 1.6, 1.2))
    near, far = 0.3, 12.0
    norm = (far - t) / (far - near)
    norm = np.clip(norm, 0.0, 1.0)
    g = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    return Image.fromarray(rgb, "RGB")


def _mock_normal_erp(size_wh: tuple[int, int]) -> Image.Image:
    width, height = size_wh
    dirs = erp_camera_dirs(width, height)
    abs_hits = np.abs(dirs)
    axis = np.argmax(abs_hits, axis=-1)
    sign = np.sign(np.take_along_axis(dirs, axis[..., None], axis=-1))[..., 0]
    nrm = np.zeros_like(dirs)
    idx = np.indices(axis.shape)
    nrm[idx[0], idx[1], axis] = -sign
    rgb = np.clip((nrm * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = (hsv[..., 0] % 1.0) * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)
