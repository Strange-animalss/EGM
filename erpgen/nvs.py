"""Hybrid NVS scheduler.

For pose 0 (center): three independent gpt-image-2 generations (RGB, depth, normal).
For poses 1..N (corners): forward-warp the center RGB to the corner pose using
the center metric depth, build a hole mask, then run `images.edits` with the
warped RGB + (alpha-encoded) mask + the same prompt.

The same edit endpoint is reused for the depth/normal corner ERPs: we condition
on the freshly-edited corner RGB so all three share the same warped layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from .openai_erp import ImageClient
from .poses import Pose
from .prompts import SceneSpec, build_prompt_set
from .warp import (
    dilate_mask,
    forward_warp_erp,
    hole_mask_to_openai_alpha,
)


@dataclass
class PoseTriplet:
    pose_idx: int
    pose: Pose
    rgb: Image.Image
    depth_img: Image.Image
    normal_img: Image.Image
    warped_rgb: Image.Image | None = None  # only for corner poses
    hole_mask: Image.Image | None = None    # only for corner poses


def _save_triplet(
    triplet: PoseTriplet, run_dir: Path, *, save_intermediates: bool = True
) -> dict[str, str]:
    erp = run_dir / "erp"
    (erp / "rgb").mkdir(parents=True, exist_ok=True)
    (erp / "depth").mkdir(parents=True, exist_ok=True)
    (erp / "normal").mkdir(parents=True, exist_ok=True)
    name = f"pose_{triplet.pose_idx}.png"
    paths = {
        "rgb": str(erp / "rgb" / name),
        "depth": str(erp / "depth" / name),
        "normal": str(erp / "normal" / name),
    }
    triplet.rgb.save(paths["rgb"])
    triplet.depth_img.save(paths["depth"])
    triplet.normal_img.save(paths["normal"])
    if save_intermediates and triplet.warped_rgb is not None:
        (erp / "warp").mkdir(parents=True, exist_ok=True)
        triplet.warped_rgb.save(str(erp / "warp" / f"pose_{triplet.pose_idx}_rgb.png"))
        if triplet.hole_mask is not None:
            triplet.hole_mask.save(str(erp / "warp" / f"pose_{triplet.pose_idx}_mask.png"))
    return paths


def run_hybrid_nvs(
    *,
    poses: List[Pose],
    scene: SceneSpec,
    client: ImageClient,
    run_dir: Path,
    hole_dilate_px: int,
    depth_near_m: float,
    depth_far_m: float,
    strategy: str = "hybrid",
) -> List[PoseTriplet]:
    """Generate aligned (rgb, depth, normal) ERPs for every pose."""
    if not poses:
        raise ValueError("poses is empty")
    Wo, Ho = client.parse_size()
    n = len(poses)

    triplets: List[PoseTriplet] = []
    center_rgb_arr: np.ndarray | None = None
    center_depth_m: np.ndarray | None = None
    center_pose: Pose | None = None

    for idx, pose in enumerate(poses):
        prompts = build_prompt_set(scene, pose_idx=idx, total_poses=n)
        do_warp_inpaint = (
            strategy == "warp_inpaint_only"
            or (strategy == "hybrid" and idx > 0)
        )
        if not do_warp_inpaint:
            imgs = client.generate_aligned_triplet(prompts)
            rgb_img = imgs["rgb"].convert("RGB")
            dep_img = imgs["depth"].convert("RGB")
            nrm_img = imgs["normal"].convert("RGB")
            if idx == 0:
                center_rgb_arr = np.array(rgb_img.resize((Wo, Ho), Image.BICUBIC))
                from .decode import decode_depth_png
                center_depth_m = decode_depth_png(
                    dep_img.resize((Wo, Ho), Image.BICUBIC),
                    near_m=depth_near_m,
                    far_m=depth_far_m,
                )
                center_pose = pose
            triplets.append(
                PoseTriplet(
                    pose_idx=idx,
                    pose=pose,
                    rgb=rgb_img,
                    depth_img=dep_img,
                    normal_img=nrm_img,
                )
            )
            _save_triplet(triplets[-1], run_dir)
            continue

        if center_rgb_arr is None or center_depth_m is None or center_pose is None:
            raise RuntimeError(
                "warp+inpaint requires the center pose to have been generated first"
            )

        warp = forward_warp_erp(
            src_rgb=center_rgb_arr,
            src_depth=center_depth_m,
            src_xyz=center_pose.xyz,
            src_R=center_pose.R,
            dst_xyz=pose.xyz,
            dst_R=pose.R,
            out_size=(Wo, Ho),
        )
        hole_mask = dilate_mask(warp.hole_mask, dilate_px=int(hole_dilate_px))
        warped_img = Image.fromarray(warp.rgb, "RGB")
        mask_rgba = hole_mask_to_openai_alpha(hole_mask, base_rgb=warp.rgb)
        mask_img = Image.fromarray(mask_rgba, "RGBA")

        imgs = client.generate_aligned_triplet(
            prompts, ref_rgb=warped_img, ref_mask=mask_img
        )
        triplet = PoseTriplet(
            pose_idx=idx,
            pose=pose,
            rgb=imgs["rgb"].convert("RGB"),
            depth_img=imgs["depth"].convert("RGB"),
            normal_img=imgs["normal"].convert("RGB"),
            warped_rgb=warped_img,
            hole_mask=Image.fromarray(hole_mask, "L"),
        )
        triplets.append(triplet)
        _save_triplet(triplet, run_dir)

    return triplets
