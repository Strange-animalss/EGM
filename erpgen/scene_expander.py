"""Expand a short user description into a rich, structured SceneSpec via a
chat-completions LLM (gpt-5.5-pro on OpenAI direct, ``openai/gpt-5.5-pro``
on OpenRouter, or whatever the relay serves under ``cfg.openai.text_model``).

The reasoning_effort kwarg is OpenAI-specific. Some OpenAI-compatible
relays don't pass it through and respond with a Python ``TypeError`` at
SDK-call time; we catch that one specific case and retry without the
kwarg, so the caller doesn't have to know which relay is in use.
"""

from __future__ import annotations

import json
import os
import time

from .prompts import SceneSpec


SYSTEM_PROMPT = """You are a scene description expander for an ERP (equirectangular 360 panorama) 3D scene generation pipeline. Given a short user description of a scene, expand it into a RICH, structured scene specification optimized for photorealistic 360 panorama generation of a STATIC, EMPTY interior.

The downstream pipeline reconstructs a 3D scene by warping multiple camera viewpoints, so the room MUST be perfectly still and consistent from any angle: NO PEOPLE, NO ANIMALS, NO MOVING OBJECTS, NO ACTIVE LIQUID FLOW (e.g. no steam plumes, no pouring drinks, no running water, no flickering candles, no swaying plants).

Output valid JSON with these exact keys:
- "scene_kind": the precise type of location (e.g. "specialty pour-over coffee shop", "cyberpunk noodle bar", "medieval university library").
- "style": the visual aesthetic AND architectural style. Mention the building shell (ceiling type, wall material, floor material, window treatment) and the design language.
- "light": detailed STATIC lighting. Include time of day, natural light direction relative to the room, color temperature, fixture types and placement, shadow character, reflective highlights. Emphasise that the lighting is constant (no flicker, no rotating spots, no animated displays).
- "occupancy": MUST state explicitly that there are no people, no animals, no figures, no moving objects whatsoever. Describe the room as quiet, tidy, and untouched: chairs neatly tucked, no half-finished drinks, no open laptops, no in-progress tasks. The space should read as a still-life.
- "extra_props": a single comma-separated list, well detailed. Aim for 20-35 items covering main fixed furniture, counter/bar elements, smaller props, wall art, greenery, floor coverings, ceiling fixtures, visible equipment, small clutter. ALL items must be inert and stationary. Include explicit colours and materials so a 360-panorama image model can render the room consistently from any angle.

Guidelines:
- Be specific and grounded; avoid generic adjectives.
- The room must read as a coherent, photoreal STATIC interior viewable from a 360 sphere.
- Output ONLY valid JSON, no markdown fences."""


class SceneExpander:
    """Wraps OpenAI-compatible chat API to expand short descriptions into SceneSpecs."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5-pro",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "",
        provider: str = "openai",
        http_referer: str = "https://github.com/Strange-animalss/EGMOR",
        app_title: str = "EGMOR",
        reasoning_effort: str = "high",
        request_timeout_sec: float = 600.0,
        max_retries: int = 4,
        retry_backoff_sec: float = 5.0,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.provider = (provider or "openai").lower()
        self.reasoning_effort = reasoning_effort
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.verbose = verbose

        key = (api_key or "").strip() or os.environ.get(api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{api_key_env} env var is required for SceneExpander (or pass api_key=...)"
            )

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

    # ------------------------------------------------------------------

    def expand(self, user_description: str, *, seed: int | None = None) -> SceneSpec:
        """Expand into a SceneSpec. Falls back to a raw SceneSpec on any
        non-recoverable error (JSON parse error, retries exhausted)."""
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
                # reasoning_effort routing
                if self.provider == "openrouter":
                    if self.reasoning_effort:
                        req_kwargs["extra_body"] = {
                            "reasoning": {"effort": self.reasoning_effort},
                        }
                elif self.reasoning_effort:
                    # OpenAI direct accepts reasoning_effort. Some relays don't
                    # pass it through and the SDK raises a Python TypeError;
                    # in that one case retry without the kwarg.
                    req_kwargs["reasoning_effort"] = self.reasoning_effort
                try:
                    resp = self._client.chat.completions.create(**req_kwargs)
                except TypeError as te:
                    if "reasoning_effort" in str(te) and "reasoning_effort" in req_kwargs:
                        req_kwargs.pop("reasoning_effort", None)
                        resp = self._client.chat.completions.create(**req_kwargs)
                    else:
                        raise
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
                    print("[scene_expander] JSON parse error, falling back to raw scene")
                return self._fallback(user_description, seed)
            except Exception as exc:  # pragma: no cover - network path
                if self.verbose:
                    print(
                        f"[scene_expander] attempt {attempt + 1}/"
                        f"{self.max_retries + 1} failed: {exc}",
                        flush=True,
                    )
                if attempt == self.max_retries:
                    if self.verbose:
                        print("[scene_expander] retries exhausted, falling back to raw scene")
                    return self._fallback(user_description, seed)
                time.sleep(self.retry_backoff_sec * (2 ** attempt))
        return self._fallback(user_description, seed)

    @staticmethod
    def _fallback(desc: str, seed: int | None) -> SceneSpec:
        return SceneSpec(
            scene_kind=desc,
            style="", light="", occupancy="", extra_props="",
            seed=int(seed) if seed is not None else 0,
        )
