"""Scene expander using GPT-5.5-pro with deep reasoning.

Takes a short user description (e.g. "cyberpunk bar") and expands it into
a structured SceneSpec that can be passed directly into the existing prompt
pipeline.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .prompts import SceneSpec


SYSTEM_PROMPT = """You are a scene description expander for an ERP (equirectangular 360 panorama) 3D scene generation pipeline. Given a short user description of a scene, expand it into a rich, structured scene specification.

Output valid JSON with these exact keys:
- "scene_kind": the type of location (e.g. "coffee shop", "cyberpunk bar", "medieval library")
- "style": the visual aesthetic or architectural style (e.g. "scandinavian", "industrial neon-lit", "gothic with dark wood")
- "light": detailed lighting description (e.g. "soft morning sun streaming through tall windows, warm golden tones")
- "occupancy": the occupancy level with specific subjects (e.g. "empty", "a couple of patrons reading at corner tables")
- "extra_props": a comma-separated list of interior details, props, materials, textures, and decorative elements

Guidelines:
- Be creative and specific. Avoid generic descriptions.
- The scene should be suitable for a photorealistic equirectangular 360 panorama.
- All fields must be coherent with each other and match the user's description.
- Keep each field concise but descriptive.
- Output ONLY valid JSON, no other text."""


class SceneExpander:
    """Wraps OpenAI chat API to expand short descriptions into SceneSpecs."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5-pro",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str = "high",
        request_timeout_sec: float = 180.0,
        max_retries: int = 4,
        retry_backoff_sec: float = 5.0,
        verbose: bool = True,
        mock: bool = False,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.verbose = verbose
        self._mock = bool(mock)
        self._client = None

        if not self._mock:
            key = api_key or os.environ.get(api_key_env, "").strip()
            if not key:
                self._mock = True
                if verbose:
                    print(
                        f"[scene_expander] {api_key_env} not set; "
                        f"falling back to mock mode."
                    )
            else:
                try:
                    from openai import OpenAI  # noqa: WPS433

                    self._client = OpenAI(api_key=key, timeout=request_timeout_sec)
                except Exception as exc:  # pragma: no cover
                    self._mock = True
                    if verbose:
                        print(f"[scene_expander] OpenAI SDK not usable ({exc}); mock mode.")

    @property
    def mock_mode(self) -> bool:
        return self._mock

    def expand(self, user_description: str) -> SceneSpec:
        """Expand a short description into a structured SceneSpec.

        Falls back to a raw SceneSpec on API failure, JSON parse error,
        or when running in mock mode.
        """
        if self._mock:
            return self._mock_expand(user_description)

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_description},
                    ],
                    reasoning_effort=self.reasoning_effort,
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    max_tokens=500,
                )
                content = resp.choices[0].message.content or ""
                data = json.loads(content)
                return SceneSpec(
                    scene_kind=str(data.get("scene_kind", user_description)),
                    style=str(data.get("style", "")),
                    light=str(data.get("light", "")),
                    occupancy=str(data.get("occupancy", "")),
                    extra_props=str(data.get("extra_props", "")),
                    seed=0,
                )
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"[scene_expander] JSON parse error on response, falling back")
                return self._fallback(user_description)
            except Exception as exc:  # pragma: no cover
                if self.verbose:
                    print(
                        f"[scene_expander] attempt {attempt + 1}/"
                        f"{self.max_retries + 1} failed: {exc}"
                    )
                if attempt == self.max_retries:
                    if self.verbose:
                        print("[scene_expander] all retries exhausted, falling back")
                    return self._fallback(user_description)
                time.sleep(self.retry_backoff_sec * (2 ** attempt))

        return self._fallback(user_description)

    def _fallback(self, desc: str) -> SceneSpec:
        return SceneSpec(
            scene_kind=desc, style="", light="",
            occupancy="", extra_props="", seed=0,
        )

    def _mock_expand(self, desc: str) -> SceneSpec:
        """Hardcoded expansion for testing without API keys."""
        return SceneSpec(
            scene_kind=desc,
            style="contemporary",
            light="warm ambient interior lighting, natural daylight",
            occupancy="moderately busy with a few people",
            extra_props="modern furniture, natural materials, decorative plants, "
                        "artwork on walls, ambient pendant lights",
            seed=0,
        )
