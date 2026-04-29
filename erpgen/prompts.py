"""Prompt builder for the gpt-image-2 ERP triplets (RGB / Depth / Normal).

A SceneSpec is sampled once per run, then reused across all 9 poses to keep
the scene description consistent. For each pose we add a viewpoint hint (camera
located at... looking toward...). For each kind (rgb / depth / normal) we
append a kind-specific instruction so the same scene is rendered as colorized
depth or normal map.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


ERP_HARD_CONSTRAINT = (
    "equirectangular 360 panorama, 2:1 aspect ratio, full sphere, "
    "seamless horizontal wraparound (left and right edges identical), "
    "no text, no captions, no watermark, no logos, no UI overlay"
)


DEPTH_KIND_SUFFIX = (
    "RENDER THIS SCENE AS A GRAYSCALE DEPTH MAP (not the photo): "
    "white = nearest surface, black = farthest surface, smooth linear gradient, "
    "uniform monochrome, NO COLORS, no textures, no shading other than depth, "
    "preserve the same geometry and pixel layout as the corresponding RGB ERP"
)


NORMAL_KIND_SUFFIX = (
    "RENDER THIS SCENE AS A WORLD-SPACE SURFACE NORMAL MAP (not the photo): "
    "encode each surface normal as RGB where R=(+X right -> 255), "
    "G=(+Y up -> 255), B=(+Z toward viewer -> 255); flat colors per surface; "
    "no shading, no texture, no lighting, preserve the same geometry and pixel "
    "layout as the corresponding RGB ERP"
)


@dataclass
class SceneSpec:
    """Sampled scene description, reused across all poses of a run."""

    scene_kind: str = "coffee shop"
    style: str = "scandinavian"
    light: str = "soft morning sun"
    occupancy: str = "a couple of patrons reading"
    extra_props: str = "wooden floor, exposed brick"
    seed: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _to_list(v) -> list:
    if isinstance(v, ListConfig):
        return list(OmegaConf.to_container(v, resolve=True))  # type: ignore[arg-type]
    return list(v)


def sample_scene(cfg: DictConfig, *, seed: int | None = None) -> SceneSpec:
    """Pick one combination from the configured pools."""
    pcfg = cfg.prompt
    if seed is None:
        seed = pcfg.seed if pcfg.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    if pcfg.randomize:
        return SceneSpec(
            scene_kind=str(pcfg.scene_default),
            style=rng.choice(_to_list(pcfg.style_pool)),
            light=rng.choice(_to_list(pcfg.light_pool)),
            occupancy=rng.choice(_to_list(pcfg.occupancy_pool)),
            extra_props=rng.choice(_to_list(pcfg.extra_props_pool)),
            seed=int(seed),
        )
    return SceneSpec(
        scene_kind=str(pcfg.scene_default),
        style=_to_list(pcfg.style_pool)[0],
        light=_to_list(pcfg.light_pool)[0],
        occupancy=_to_list(pcfg.occupancy_pool)[0],
        extra_props=_to_list(pcfg.extra_props_pool)[0],
        seed=int(seed),
    )


def scene_from_user_input(scene_description: str) -> SceneSpec:
    """Create a SceneSpec from a plain string without pool sampling.

    When passed to `scene_description()`, the raw text is used as-is
    because all structured fields are empty.
    """
    return SceneSpec(
        scene_kind=scene_description,
        style="",
        light="",
        occupancy="",
        extra_props="",
        seed=0,
    )


def scene_description(scene: SceneSpec) -> str:
    """Compose the shared scene description string (no ERP / kind suffix)."""
    if not scene.style and not scene.light and not scene.occupancy and not scene.extra_props:
        return (
            f"{scene.scene_kind}, "
            f"realistic interior photography, sharp focus, well-composed"
        )
    return (
        f"a {scene.style} {scene.scene_kind}, {scene.light}, "
        f"{scene.occupancy}, {scene.extra_props}, "
        f"realistic interior photography, sharp focus, well-composed"
    )


def _viewpoint_hint(pose_idx: int, total_poses: int) -> str:
    """Short hint describing where the camera is for this pose."""
    if pose_idx == 0:
        return (
            "the camera is at the center of the room at standing-eye height, "
            "facing forward"
        )
    return (
        f"the camera is near corner #{pose_idx} (of {total_poses - 1}), slightly "
        f"inside the room, facing toward the room center"
    )


def build_prompt(
    scene: SceneSpec,
    *,
    kind: str,
    pose_idx: int = 0,
    total_poses: int,
) -> str:
    """Compose the full prompt for one pose+kind.

    kind in {"rgb", "depth", "normal"}.
    """
    kind = kind.lower()
    base = scene_description(scene)
    vp = _viewpoint_hint(pose_idx, total_poses)
    erp = ERP_HARD_CONSTRAINT
    if kind == "rgb":
        return f"{base}; {vp}; {erp}"
    if kind == "depth":
        return f"{base}; {vp}; {DEPTH_KIND_SUFFIX}; {erp}"
    if kind == "normal":
        return f"{base}; {vp}; {NORMAL_KIND_SUFFIX}; {erp}"
    raise ValueError(f"unknown kind: {kind!r}")


def build_prompt_set(
    scene: SceneSpec, *, pose_idx: int, total_poses: int
) -> dict[str, str]:
    """Convenience: return {'rgb': ..., 'depth': ..., 'normal': ...}."""
    return {
        k: build_prompt(scene, kind=k, pose_idx=pose_idx, total_poses=total_poses)
        for k in ("rgb", "depth", "normal")
    }
