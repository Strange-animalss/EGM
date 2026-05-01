"""ImageClient — minimal wrapper around an OpenAI-compatible image API.

Two API paths, selected by ``OpenAIConfig.provider``:

  * ``provider="openai"`` (the **standard** path) — uses
    ``client.images.generate`` / ``client.images.edit`` from the official
    OpenAI Python SDK. The same code works against:
      - OpenAI direct (`base_url=""` or `https://api.openai.com/v1`)
      - any OpenAI-compatible relay (super.shangliu.org / OneAPI / LiteLLM /
        ...) — just point ``base_url`` at it.
    gpt-image-2 native sizes are accepted by the server (e.g. 2048×1024 for
    a real 2:1 ERP, multiples of 16 up to ~3840 long-edge, ratio ≤ 3:1).

  * ``provider="openrouter"`` (the **chat** path) — fallback for OpenRouter,
    which does NOT expose ``/v1/images/*`` for image-output models. We use
    ``client.chat.completions.create`` with ``modalities=["image","text"]``
    and read the result from ``message.images[0].image_url.url``. Output is
    locked to 1024×1024 by the server; no size parameter overrides this.

The class exposes exactly two public generation methods:

  * ``generate_rgb(prompt, size)`` — text-to-image
  * ``generate_with_ref(prompt, ref_image, size)`` — image-to-image (no mask)

plus tiny helpers (``parse_size``, ``supports_native_2x1_ratio``).

All API calls are cached on disk by SHA256 of the request fingerprint.
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

import requests
from PIL import Image


def _decode_image_resp(resp) -> Image.Image:
    """Convert an images.generate / images.edit response to a PIL.Image,
    accepting either ``b64_json`` (OpenAI direct) or ``url`` (chatfire and
    other relays sometimes return a hosted URL instead)."""
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    url = getattr(d, "url", None)
    if b64:
        raw = base64.b64decode(b64)
    elif url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        raw = r.content
    else:
        raise RuntimeError(
            f"image API returned neither b64_json nor url: {d!r}"
        )
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OpenAIConfig:
    # "openai"     -> /v1/images/{generations,edits} (any OpenAI-compatible
    #                 endpoint: openai.com / shangliu / OneAPI / LiteLLM ...)
    # "openrouter" -> /v1/chat/completions with modalities=["image","text"]
    provider: str = "openai"
    model: str = "gpt-image-2"
    size: str = "2048x1024"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""  # do not commit a real key; prefer the env var
    base_url: str = ""  # empty = official OpenAI api.openai.com
    rgb_quality: str = "high"
    request_timeout_sec: float = 600.0
    max_retries: int = 4
    retry_backoff_sec: float = 5.0
    cache_dir: str = "outputs/.openai_cache"
    text_model: str = "gpt-5.5-pro"
    text_model_api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = "high"
    # OpenRouter ranking headers (harmless to leave on for other providers).
    http_referer: str = "https://github.com/Strange-animalss/EGMOR"
    app_title: str = "EGMOR"

    @classmethod
    def from_dict(cls, cfg) -> "OpenAIConfig":
        from omegaconf import OmegaConf
        d = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, "_content") else dict(cfg)
        return cls(
            provider=str(d.get("provider", "openai")).lower(),
            model=str(d.get("model", "gpt-image-2")),
            size=str(d.get("size", "2048x1024")),
            api_key_env=str(d.get("api_key_env", "OPENAI_API_KEY")),
            api_key=str(d.get("api_key", "")),
            base_url=str(d.get("base_url", "")),
            rgb_quality=str(d.get("rgb_quality", "high")),
            request_timeout_sec=float(d.get("request_timeout_sec", 600.0)),
            max_retries=int(d.get("max_retries", 4)),
            retry_backoff_sec=float(d.get("retry_backoff_sec", 5.0)),
            cache_dir=str(d.get("cache_dir", "outputs/.openai_cache")),
            text_model=str(d.get("text_model", "gpt-5.5-pro")),
            text_model_api_key_env=str(d.get("text_model_api_key_env", "OPENAI_API_KEY")),
            reasoning_effort=str(d.get("reasoning_effort", "high")),
            http_referer=str(d.get("http_referer", "https://github.com/Strange-animalss/EGMOR")),
            app_title=str(d.get("app_title", "EGMOR")),
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _hash_request(parts: Iterable[bytes | str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
        h.update(b"\x1f")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Image client
# ---------------------------------------------------------------------------


class ImageClient:
    """Tiny wrapper around the OpenAI image endpoints with caching + retries."""

    def __init__(self, cfg: OpenAIConfig, *, verbose: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        api_key = cfg.api_key.strip() or os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{cfg.api_key_env} env var is required (or set openai.api_key in the config)"
            )

        from openai import OpenAI  # noqa: WPS433
        kwargs: dict = dict(api_key=api_key, timeout=cfg.request_timeout_sec)
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.provider == "openrouter":
            kwargs["default_headers"] = {
                "HTTP-Referer": cfg.http_referer,
                "X-Title": cfg.app_title,
            }
        self._client = OpenAI(**kwargs)

    # ----- caching ------------------------------------------------------

    def _load_cached(self, key: str) -> Optional[Image.Image]:
        path = self._cache_dir / f"{key}.png"
        if path.exists():
            try:
                return Image.open(path).copy()
            except Exception:
                return None
        return None

    def _save_cached(self, key: str, img: Image.Image) -> None:
        img.save(self._cache_dir / f"{key}.png", format="PNG")

    # ----- public helpers ----------------------------------------------

    @property
    def supports_native_2x1_ratio(self) -> bool:
        """OpenRouter chat-image is locked to 1024×1024; the standard path
        accepts whatever the server supports (e.g. 2048×1024 on a relay
        that routes to real gpt-image-2)."""
        return self.cfg.provider != "openrouter"

    def parse_size(self, size_str: str | None = None) -> tuple[int, int]:
        s = size_str or self.cfg.size
        w, h = s.lower().split("x")
        return int(w), int(h)

    # ----- generate -----------------------------------------------------

    def generate_rgb(self, prompt: str, *, size: str | None = None) -> Image.Image:
        size = size or self.cfg.size
        key = _hash_request([
            "generate", self.cfg.provider, self.cfg.model, size,
            (self.cfg.rgb_quality or ""), prompt,
        ])
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        img = self._call_with_retry(
            self._call_generate, prompt=prompt, size=size,
        )
        self._save_cached(key, img)
        return img

    def generate_with_ref(
        self,
        prompt: str,
        ref_image: Image.Image,
        *,
        size: str | None = None,
    ) -> Image.Image:
        """Image-to-image, no mask: the model takes the whole reference and
        produces a new image conditioned on it.

        - standard path (provider="openai"): client.images.edit(image=ref, ...)
        - chat path     (provider="openrouter"): chat.completions with the
          ref encoded as a base64 image_url content part.
        """
        size = size or self.cfg.size
        ref_bytes = _png_bytes(ref_image)
        key = _hash_request([
            "i2i", self.cfg.provider, self.cfg.model, size,
            (self.cfg.rgb_quality or ""), prompt, ref_bytes,
        ])
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        img = self._call_with_retry(
            self._call_generate_with_ref,
            prompt=prompt, ref_bytes=ref_bytes, size=size,
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
    ) -> Image.Image:
        """Mask-aware inpaint: ``client.images.edit`` is told to fill the
        transparent ``alpha=0`` regions of ``mask`` while preserving the rest
        of ``image``. Used by the warp+inpaint NVS path so the model only
        invents the disocclusions revealed by the corner camera move.

        Standard provider only -- OpenRouter does not expose ``images.edits``,
        and the chat path cannot enforce a per-pixel keep mask anyway.
        """
        if self.cfg.provider != "openai":
            raise RuntimeError(
                "edit_with_mask requires provider='openai' (images.edits "
                "is not available on OpenRouter chat path)"
            )
        size = size or self.cfg.size
        ref_bytes = _png_bytes(image)
        # mask must keep its alpha channel — convert to RGBA bytes.
        buf = io.BytesIO()
        mask.convert("RGBA").save(buf, format="PNG")
        mask_bytes = buf.getvalue()
        key = _hash_request([
            "edit_mask", self.cfg.provider, self.cfg.model, size,
            (self.cfg.rgb_quality or ""), prompt, ref_bytes, mask_bytes,
        ])
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        img = self._call_with_retry(
            self._call_edit_with_mask,
            prompt=prompt, ref_bytes=ref_bytes, mask_bytes=mask_bytes, size=size,
        )
        self._save_cached(key, img)
        return img

    # ----- decoder experiment helpers ----------------------------------

    def decode_to_depth(self, rgb: Image.Image, *, size: str | None = None) -> Image.Image:
        """Ask gpt-image-2 to convert an RGB ERP into a grayscale depth map.

        Used purely as an experimental sanity check against DAP-V2 — the main
        pipeline still uses DAP for downstream geometry.
        """
        prompt = (
            "Convert this exact 360-degree equirectangular panorama into a "
            "grayscale depth map. White/bright = nearest surface; "
            "black/dark = farthest surface. Pixel-perfect alignment with the "
            "input image: same layout, same geometry, same pixel positions. "
            "No text, no shading from light, no surface textures or colours, "
            "ONLY a per-pixel monochrome depth representation."
        )
        return self.generate_with_ref(prompt, rgb, size=size)

    def decode_to_normal(self, rgb: Image.Image, *, size: str | None = None) -> Image.Image:
        """Ask gpt-image-2 to convert an RGB ERP into a surface normal map."""
        prompt = (
            "Convert this exact 360-degree equirectangular panorama into a "
            "world-space surface normal map. Each pixel encodes the surface "
            "normal as RGB: R = normal X (right), G = normal Y (up), "
            "B = normal Z (toward camera). Use flat per-surface colours, "
            "no shading, no lighting, no textures. Pixel-perfect alignment "
            "with the input image: same layout, same geometry."
        )
        return self.generate_with_ref(prompt, rgb, size=size)

    # ----- inner: API calls --------------------------------------------

    def _gen_kwargs(self, *, prompt: str, size: str) -> dict:
        kw: dict = dict(model=self.cfg.model, prompt=prompt, size=size, n=1)
        # gpt-image-2 accepts quality=low|medium|high|auto. Some older relays
        # / SDK versions reject the kwarg; we guard against that in the call.
        q = (self.cfg.rgb_quality or "").strip().lower()
        if q:
            kw["quality"] = q
        return kw

    def _call_generate(self, *, prompt: str, size: str) -> Image.Image:
        if self.cfg.provider == "openrouter":
            return self._chat_image_call(prompt=prompt, ref_image_bytes=None)
        # standard OpenAI / OpenAI-compatible relay
        kw = self._gen_kwargs(prompt=prompt, size=size)
        try:
            resp = self._client.images.generate(**kw)  # type: ignore[union-attr]
        except TypeError as te:
            if "quality" in str(te) and "quality" in kw:
                kw.pop("quality", None)
                resp = self._client.images.generate(**kw)  # type: ignore[union-attr]
            else:
                raise
        return _decode_image_resp(resp)

    def _call_edit_with_mask(
        self, *, prompt: str, ref_bytes: bytes, mask_bytes: bytes, size: str,
    ) -> Image.Image:
        """Mask-aware ``images.edit`` call (provider='openai' only)."""
        img_io = io.BytesIO(ref_bytes)
        img_io.name = "image.png"
        mask_io = io.BytesIO(mask_bytes)
        mask_io.name = "mask.png"
        kw: dict = dict(
            model=self.cfg.model, prompt=prompt,
            image=img_io, mask=mask_io, size=size, n=1,
        )
        q = (self.cfg.rgb_quality or "").strip().lower()
        if q:
            kw["quality"] = q
        try:
            resp = self._client.images.edit(**kw)  # type: ignore[union-attr]
        except TypeError as te:
            if "quality" in str(te) and "quality" in kw:
                kw.pop("quality", None)
                img_io.seek(0)
                mask_io.seek(0)
                resp = self._client.images.edit(**kw)  # type: ignore[union-attr]
            else:
                raise
        return _decode_image_resp(resp)

    def _call_generate_with_ref(
        self, *, prompt: str, ref_bytes: bytes, size: str,
    ) -> Image.Image:
        if self.cfg.provider == "openrouter":
            return self._chat_image_call(prompt=prompt, ref_image_bytes=ref_bytes)
        # standard images.edit (no mask = whole-image i2i)
        img_io = io.BytesIO(ref_bytes)
        img_io.name = "image.png"
        kw: dict = dict(
            model=self.cfg.model, prompt=prompt, image=img_io, size=size, n=1,
        )
        # images.edit also supports quality on gpt-image-2; same TypeError guard.
        q = (self.cfg.rgb_quality or "").strip().lower()
        if q:
            kw["quality"] = q
        try:
            resp = self._client.images.edit(**kw)  # type: ignore[union-attr]
        except TypeError as te:
            if "quality" in str(te) and "quality" in kw:
                kw.pop("quality", None)
                # bytes-based image objects can be re-used; reset stream pos
                img_io.seek(0)
                resp = self._client.images.edit(**kw)  # type: ignore[union-attr]
            else:
                raise
        return _decode_image_resp(resp)

    def _chat_image_call(
        self, *, prompt: str, ref_image_bytes: bytes | None,
    ) -> Image.Image:
        """OpenRouter chat-completions image path. Output is always 1024×1024
        on the server side (no size override is honoured), so we just return
        whatever PIL decoded — the caller must accept 1024×1024 ERP."""
        content: list = [{"type": "text", "text": prompt}]
        if ref_image_bytes is not None:
            ref_b64 = base64.b64encode(ref_image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            })
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.cfg.model,
            messages=[{"role": "user", "content": content}],
            extra_body={"modalities": ["image", "text"]},
        )
        msg = resp.choices[0].message
        images = getattr(msg, "images", None)
        if images is None and hasattr(msg, "model_extra"):
            images = (msg.model_extra or {}).get("images")
        if not images:
            raise RuntimeError(
                f"OpenRouter chat returned no images. content={getattr(msg, 'content', None)!r}"
            )
        first = images[0]
        url = first["image_url"]["url"] if isinstance(first, dict) else first.image_url.url
        if not url.startswith("data:"):
            raise RuntimeError(f"Unexpected image url (not data:): {url[:80]}")
        b64 = url.split(",", 1)[1]
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
                        f"{self.cfg.max_retries + 1} failed: {exc}",
                        flush=True,
                    )
                if attempt == self.cfg.max_retries:
                    break
                time.sleep(self.cfg.retry_backoff_sec * (2 ** attempt))
        raise RuntimeError(f"OpenAI API failed after retries: {last_exc}")
