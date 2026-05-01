"""Prompt builder for the ERP RGB call.

A SceneSpec is sampled once per run, then reused across all poses to keep
the scene description consistent. For each pose we add a viewpoint hint
(camera located at..., looking toward...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


ERP_HARD_CONSTRAINT = (
    "equirectangular 360 panorama, 2:1 aspect ratio, full sphere, "
    "seamless horizontal wraparound (left and right edges identical); "
    "STATIC SCENE, NO MOTION, NO MOTION BLUR, no moving objects, "
    "EMPTY ROOM, NO PEOPLE, NO HUMANS, NO ANIMALS, NO FIGURES, NO VEHICLES, "
    "NO PORTRAITS OR PHOTOS OF PEOPLE ON THE WALLS, all objects stationary, "
    "clean still-life interior photography, sharp uniform focus everywhere, "
    "no text, no captions, no watermark, no logos, no UI overlay"
)


@dataclass
class SceneSpec:
    """Structured scene description shared across all poses of a run."""

    scene_kind: str = "coffee shop"
    style: str = ""
    light: str = ""
    occupancy: str = ""
    extra_props: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def scene_from_user_input(scene_description: str) -> SceneSpec:
    """Create a SceneSpec from a plain string without LLM expansion."""
    return SceneSpec(scene_kind=scene_description)


def scene_description(scene: SceneSpec) -> str:
    """Compose the shared scene description string (no ERP / kind suffix).

    Works for both raw SceneSpecs (only `scene_kind` filled) and full
    SceneExpander-produced ones (long sentence-style fields).
    """
    if not (scene.style or scene.light or scene.occupancy or scene.extra_props):
        return (
            f"{scene.scene_kind}. "
            f"Realistic interior photography, sharp focus, well-composed."
        )
    parts = [f"Scene: {scene.scene_kind}."]
    if scene.style:
        parts.append(f"Style and architecture: {scene.style}.")
    if scene.light:
        parts.append(f"Lighting: {scene.light}.")
    if scene.occupancy:
        parts.append(f"Occupancy: {scene.occupancy}.")
    if scene.extra_props:
        parts.append(f"Visible details: {scene.extra_props}.")
    parts.append("Realistic interior photography, sharp focus, well-composed.")
    return " ".join(parts)


def _viewpoint_hint(pose_idx: int, total_poses: int) -> str:
    """Short hint describing where the camera is for this pose."""
    if pose_idx == 0:
        return (
            "the camera is at the center of the room at standing-eye height, "
            "facing forward"
        )
    return (
        f"the camera is near corner #{pose_idx} (of {total_poses - 1}), "
        f"slightly inside the room, facing toward the room center"
    )


# chatfire (and likely most relays) reject or hang on very long prompts.
# Empirically (2026-05-01): 4700 chars -> APIConnectionError after ~6 min;
# 1500 chars -> works in ~60s. Cap below that with margin.
_MAX_PROMPT_CHARS = 1800


def _truncate_extra_props(props: str, budget: int) -> str:
    """Trim a comma-separated list to fit ``budget`` chars, never cutting in
    the middle of an item."""
    if len(props) <= budget:
        return props
    items = [p.strip() for p in props.split(",") if p.strip()]
    out: list[str] = []
    used = 0
    for it in items:
        cost = len(it) + 2  # ", "
        if used + cost > budget:
            break
        out.append(it)
        used += cost
    return ", ".join(out)


def _truncate_at_word(s: str, budget: int) -> str:
    """Cut to budget chars, prefer breaking at a punctuation boundary."""
    if len(s) <= budget:
        return s
    cut = s[:budget]
    for sep in [". ", "; ", ", ", " "]:
        idx = cut.rfind(sep)
        if idx > budget * 0.6:
            return cut[:idx]
    return cut


def build_prompt(
    scene: SceneSpec, *, kind: str, pose_idx: int = 0, total_poses: int,
) -> str:
    """Compose the full prompt for one pose. ``kind`` is kept for backwards
    compatibility but only ``"rgb"`` is meaningful (depth/normal are produced
    locally by DAP-V2, not by the LLM)."""
    if kind.lower() != "rgb":
        raise ValueError(
            f"only kind='rgb' is supported (depth/normal come from DAP); got {kind!r}"
        )
    viewpoint = _viewpoint_hint(pose_idx, total_poses)
    # Suffix is non-negotiable: it carries the ERP hard constraint and pose hint.
    # We always keep it intact and trim the description to fit the budget.
    suffix = f"; {viewpoint}; {ERP_HARD_CONSTRAINT}"
    full = scene_description(scene) + suffix
    if len(full) <= _MAX_PROMPT_CHARS:
        return full

    # Build a budget-respecting description by trimming extra_props first,
    # then by truncating each long field at a word boundary.
    desc_budget = _MAX_PROMPT_CHARS - len(suffix) - 8  # safety
    parts = [f"Scene: {scene.scene_kind}."]
    used = len(parts[0])

    def _try_add(label: str, value: str, max_chars: int) -> None:
        nonlocal used
        if not value:
            return
        chunk = f" {label}: {_truncate_at_word(value, max_chars)}."
        # "+1" for the trailing period if truncated (already added above)
        if used + len(chunk) > desc_budget:
            chunk = chunk[: max(0, desc_budget - used)]
        if chunk:
            parts.append(chunk)
            used += len(chunk)

    # Allocations (chars) tuned so 1.8K-char budget keeps each section short
    # but informative. Style + light + occupancy + props all keep cores.
    _try_add("Style and architecture", scene.style, 380)
    _try_add("Lighting", scene.light, 240)
    _try_add("Occupancy", scene.occupancy, 140)
    if scene.extra_props:
        remaining = max(0, desc_budget - used - 32)
        props_short = _truncate_extra_props(scene.extra_props, remaining)
        if props_short:
            chunk = f" Visible details: {props_short}."
            if used + len(chunk) > desc_budget:
                chunk = chunk[: max(0, desc_budget - used)]
            parts.append(chunk)
            used += len(chunk)
    parts.append(" Realistic interior photography, sharp focus.")
    description = "".join(parts).strip()
    return description + suffix
