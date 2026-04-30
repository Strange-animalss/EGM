"""ERP triplet scheduler.

For each pose we produce a (rgb, depth_m, normal_world) triplet of ERP arrays:

  * RGB:    one LLM call (gpt-image-2). Center pose uses pure text-to-image;
            corner poses use the center RGB as a reference image so the
            material/lighting/occupancy stays consistent across views (this is
            essential when the provider, e.g. OpenRouter, has no mask edit).

  * Depth:  Depth-Anything-V2 monocular estimator, locally on GPU. We do NOT
            ask the LLM to paint depth maps; LLM-painted depth was unstable
            and not geometrically meaningful.

  * Normal: analytic finite-difference of the metric depth ERP, world-rotated
            by the pose. Mathematically consistent with the depth.

So each pose costs exactly ONE image-2 API call instead of three. Depth and
normal are saved both as float32 .npy (full precision for the GS init pcd)
and as PNG (uint16 mm depth, RGB world normal) for viewer / debug.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .dap import (
    encode_depth_png,
    encode_normal_png,
    estimate_erp_depth,
    normals_from_erp_depth,
)
from .openai_erp import ImageClient
from .poses import Pose
from .prompts import SceneSpec, build_prompt


@dataclass
class PoseTriplet:
    pose_idx: int
    pose: Pose
    rgb: Image.Image
    depth_img: Image.Image                # PNG-encoded depth for compatibility
    normal_img: Image.Image               # PNG-encoded normal for compatibility
    depth_m: np.ndarray | None = None     # raw metric depth (H, W) float32
    normal_world: np.ndarray | None = None  # raw world normals (H, W, 3) float32


def _save_triplet(triplet: PoseTriplet, run_dir: Path) -> dict[str, str]:
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

    npy_dir = run_dir / "erp_decoded"
    npy_dir.mkdir(parents=True, exist_ok=True)
    if triplet.depth_m is not None:
        np.save(npy_dir / f"pose_{triplet.pose_idx}_depth_m.npy", triplet.depth_m)
    if triplet.normal_world is not None:
        np.save(npy_dir / f"pose_{triplet.pose_idx}_normal_world.npy", triplet.normal_world)
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
    verbose: bool = True,
    dap_model_size: str = "base",
    dap_mode: str = "direct",
) -> List[PoseTriplet]:
    """Generate aligned (rgb, depth, normal) ERPs for every pose.

    Pipeline (one LLM call per pose, the rest is local):

        center pose:  text-to-image RGB
                      -> DAP-V2 depth -> normals-from-depth
        corner pose:  i2i RGB conditioned on center-pose RGB
                      -> DAP-V2 depth -> normals-from-depth

    The `strategy` arg is kept for backwards compatibility but is now
    largely informational; the code below picks center-vs-ref-i2i based on
    whether we're at pose 0 or not, and on whether the ImageClient supports
    geometric warp+mask edits.
    """
    if not poses:
        raise ValueError("poses is empty")
    Wo, Ho = client.parse_size()
    n = len(poses)

    use_ref_consistency = not client.supports_mask
    if verbose:
        print(
            f"[nvs] poses={n}  size={Wo}x{Ho}  "
            f"ref_consistency={use_ref_consistency}  "
            f"dap_model={dap_model_size}  dap_mode={dap_mode}",
            flush=True,
        )

    triplets: List[PoseTriplet] = []
    center_rgb_pil: Image.Image | None = None
    run_start = _time.time()

    for idx, pose in enumerate(poses):
        pose_start = _time.time()
        if verbose:
            elapsed = _time.time() - run_start
            print(
                f"[nvs] >>> pose {idx + 1}/{n} (name={pose.name}) "
                f"elapsed_total={elapsed:.0f}s",
                flush=True,
            )

        rgb_prompt = build_prompt(scene, kind="rgb", pose_idx=idx, total_poses=n)

        # --- step 1: RGB ---
        t0 = _time.time()
        is_center = (idx == 0)
        if is_center:
            rgb_img = client.generate_rgb(rgb_prompt).convert("RGB")
        else:
            if center_rgb_pil is None:
                raise RuntimeError("non-center pose needs the center RGB to exist first")
            ref_pose_hint = (
                "This is the SAME interior space as the reference image, viewed "
                "from a DIFFERENT angle within the same room. Preserve the room's "
                "identity exactly: same materials, same wall and floor finishes, "
                "same furniture pieces and color palette, same lighting, same "
                "level of occupancy (if the reference is empty, this view is also "
                "empty -- no people, no animals). Only the camera viewpoint changes. "
            )
            rgb_img = client.generate_with_ref(
                ref_pose_hint + rgb_prompt, center_rgb_pil,
            ).convert("RGB")
        rgb_dt = _time.time() - t0
        if verbose:
            print(f"[nvs]   pose {idx + 1} rgb done ({rgb_dt:.0f}s, {'center' if is_center else 'i2i'})", flush=True)

        # ensure RGB is at the canonical ERP resolution before downstream stages
        if rgb_img.size != (Wo, Ho):
            rgb_img = rgb_img.resize((Wo, Ho), Image.BICUBIC)
        if is_center:
            center_rgb_pil = rgb_img

        # --- step 2: depth via Depth-Anything-V2 ---
        t0 = _time.time()
        dap_res = estimate_erp_depth(
            rgb_img,
            near_m=depth_near_m, far_m=depth_far_m,
            model_size=dap_model_size, device="cuda", mode=dap_mode,
        )
        depth_m = dap_res.depth_m  # (H, W) float32, meters
        if verbose:
            print(
                f"[nvs]   pose {idx + 1} depth done ({(_time.time() - t0):.1f}s, "
                f"DAP={dap_res.model_id.split('/')[-1]} infer={dap_res.inference_ms:.0f}ms)",
                flush=True,
            )

        # --- step 3: normals from depth (world frame) ---
        t0 = _time.time()
        normal_world = normals_from_erp_depth(depth_m, pose_R=pose.R, smooth_radius=1)
        normal_dt = _time.time() - t0
        if verbose:
            print(f"[nvs]   pose {idx + 1} normal done ({normal_dt:.2f}s, analytic from depth)", flush=True)

        depth_png = encode_depth_png(depth_m, near_m=depth_near_m, far_m=depth_far_m)
        normal_png = encode_normal_png(normal_world)

        triplet = PoseTriplet(
            pose_idx=idx, pose=pose,
            rgb=rgb_img, depth_img=depth_png, normal_img=normal_png,
            depth_m=depth_m, normal_world=normal_world,
        )
        triplets.append(triplet)
        _save_triplet(triplet, run_dir)
        if verbose:
            dt = _time.time() - pose_start
            print(
                f"[nvs] <<< pose {idx + 1}/{n} done in {dt:.0f}s "
                f"(rgb={rgb_dt:.0f}s, depth={dap_res.inference_ms / 1000.0:.1f}s, "
                f"normal={normal_dt:.2f}s)",
                flush=True,
            )

    return triplets
