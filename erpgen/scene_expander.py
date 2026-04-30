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


SYSTEM_PROMPT = """You are a scene description expander for an ERP (equirectangular 360 panorama) 3D scene generation pipeline. Given a short user description of a scene, expand it into a RICH, HIGHLY DETAILED, structured scene specification optimized for photorealistic 360 panorama generation.

Output valid JSON with these exact keys:
- "scene_kind": the precise type of location (e.g. "specialty pour-over coffee shop", "cyberpunk noodle bar", "medieval university library").
- "style": the visual aesthetic AND architectural style. Mention the building shell (ceiling type, wall material, floor material, window treatment) and the design language (e.g. "industrial Scandinavian: 4m exposed-concrete ceiling with black steel I-beams, whitewashed brick walls, oak-strip floor, tall single-pane windows with thin black frames").
- "light": detailed lighting. Include time of day, natural light direction relative to the room, color temperature, fixture types and placement (pendants, sconces, recessed), shadow character, and any reflective highlights (e.g. "late-morning sun from the south wall, ~5500K daylight cutting across the floor in long warm strips, supplemented by 2700K warm-white pendant cones over each table").
- "occupancy": DEFAULT TO "empty room, no people, no animals, no figures" UNLESS the user input EXPLICITLY mentions people, customers, patrons, animals, pets, or similar. We are training a 3D Gaussian Splatting model and any animate occupants will be reconstructed as ghostly artifacts because they cannot be cross-view consistent. Only when the user explicitly asks for occupants should you describe them, and even then keep the count low and their actions static (e.g. "one person sitting still at a corner table").
- "extra_props": a single comma-separated list, EXTREMELY DETAILED. Aim for 25-50 items covering: the main fixed furniture and its material/finish; counter/bar elements; smaller props on tables and shelves; wall art / signage / books; greenery; floor coverings; ceiling fixtures; visible kitchen / espresso / tap / display equipment; small everyday clutter (mugs, books, magazines, jackets, bags) that makes the room look lived-in. Include explicit colors and materials so a 360-panorama image model can render the room consistently from any angle.

Guidelines:
- Be creative, specific, and grounded. Avoid generic adjectives like "cozy", "modern", "beautiful" without concrete backing.
- The room must read as a coherent, photoreal interior viewable from a 360 sphere. Imagine the camera at standing-eye height in the middle of the room and describe what would be visible in every direction.
- All fields must be mutually consistent.
- Output ONLY valid JSON, no markdown fences, no other text."""


class SceneExpander:
    """Wraps OpenAI chat API to expand short descriptions into SceneSpecs."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5-pro",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "",
        provider: str = "openai",
        http_referer: str = "https://github.com/erpgen",
        app_title: str = "ERPGen",
        reasoning_effort: str = "high",
        request_timeout_sec: float = 180.0,
        max_retries: int = 4,
        retry_backoff_sec: float = 5.0,
        verbose: bool = True,
        mock: bool = False,
    ) -> None:
        self.model = model
        self.provider = (provider or "openai").lower()
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

                    kwargs: dict = dict(api_key=key, timeout=request_timeout_sec)
                    if base_url:
                        kwargs["base_url"] = base_url
                    if self.provider == "openrouter":
                        kwargs["default_headers"] = {
                            "HTTP-Referer": http_referer,
                            "X-Title": app_title,
                        }
                    self._client = OpenAI(**kwargs)
                except Exception as exc:  # pragma: no cover
                    self._mock = True
                    if verbose:
                        print(f"[scene_expander] OpenAI SDK not usable ({exc}); mock mode.")

    @property
    def mock_mode(self) -> bool:
        return self._mock

    def expand(
        self,
        user_description: str,
        *,
        seed: int | None = None,
    ) -> SceneSpec:
        """Expand a short description into a structured SceneSpec.

        Falls back to a raw SceneSpec on API failure, JSON parse error,
        or when running in mock mode. When `seed` is provided we pass it to
        the chat completion (OpenAI-compatible) so the same input + seed
        produces a deterministic expansion.
        """
        if self._mock:
            return self._mock_expand(user_description)

        for attempt in range(self.max_retries + 1):
            try:
                req_kwargs: dict = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_description},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    max_tokens=2000,
                )
                if seed is not None:
                    req_kwargs["seed"] = int(seed)
                if self.provider == "openrouter":
                    # OpenRouter exposes reasoning via a unified `reasoning` extra_body
                    # field rather than the OpenAI-specific `reasoning_effort` kwarg.
                    if self.reasoning_effort:
                        req_kwargs["extra_body"] = {
                            "reasoning": {"effort": self.reasoning_effort}
                        }
                else:
                    req_kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self._client.chat.completions.create(**req_kwargs)
                content = resp.choices[0].message.content or ""
                data = json.loads(content)
                return SceneSpec(
                    scene_kind=str(data.get("scene_kind", user_description)),
                    style=str(data.get("style", "")),
                    light=str(data.get("light", "")),
                    occupancy=str(data.get("occupancy", "")),
                    extra_props=str(data.get("extra_props", "")),
                    seed=int(seed) if seed is not None else 0,
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
