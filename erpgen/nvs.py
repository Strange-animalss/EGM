"""ERP triplet scheduler.

For pose 0 (the centre) we generate the RGB ERP via plain text-to-image and
then estimate depth (DAP-V2) and analytic normals locally. For each corner
pose (1..N) we have three increasingly geometric strategies, selected by
``corner_method``:

* ``"warp_inpaint"`` (default, strongest geometric consistency):
    1. Forward-warp the centre RGB + DAP depth to the corner pose.
    2. Build a hole mask covering disocclusions.
    3. Call ``images.edit(image=warped_rgb, mask=hole_mask, prompt=...)``
       so gpt-image-2 only invents the disoccluded pixels and preserves
       everything the warp filled in. Requires ``provider='openai'``.

* ``"i2i"``: ``images.edit`` with the centre RGB as a whole-image
  reference (no mask). Looser consistency, simpler, still uses
  ``provider='openai'`` standard path.

* ``"generate"``: text-to-image only, no reference. Weakest consistency.

If the chosen mask-aware path keeps failing on a relay we transparently
fall back to plain ``generate_rgb`` for the remaining corner poses to
avoid burning retries.
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
    depth_m: np.ndarray | None = None
    normal_world: np.ndarray | None = None
    warped_rgb: Image.Image | None = None
    hole_mask: Image.Image | None = None


def _pil_save(img: Image.Image, path: Path | str, *, dpi: tuple[int, int]) -> None:
    """Save with the requested DPI metadata (300×300 by default)."""
    img.save(str(path), dpi=dpi)


def _save_triplet(
    triplet: PoseTriplet, run_dir: Path, *, dpi: tuple[int, int],
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
    _pil_save(triplet.rgb, paths["rgb"], dpi=dpi)
    _pil_save(triplet.depth_img, paths["depth"], dpi=dpi)
    _pil_save(triplet.normal_img, paths["normal"], dpi=dpi)

    if triplet.warped_rgb is not None:
        warp_dir = erp / "warp"
        warp_dir.mkdir(parents=True, exist_ok=True)
        _pil_save(triplet.warped_rgb, warp_dir / f"pose_{triplet.pose_idx}_rgb.png", dpi=dpi)
        if triplet.hole_mask is not None:
            _pil_save(triplet.hole_mask, warp_dir / f"pose_{triplet.pose_idx}_mask.png", dpi=dpi)

    npy_dir = run_dir / "erp_decoded"
    npy_dir.mkdir(parents=True, exist_ok=True)
    if triplet.depth_m is not None:
        np.save(npy_dir / f"pose_{triplet.pose_idx}_depth_m.npy", triplet.depth_m)
    if triplet.normal_world is not None:
        np.save(npy_dir / f"pose_{triplet.pose_idx}_normal_world.npy", triplet.normal_world)
    return paths


def _build_warp_inputs(
    *,
    pose: Pose,
    center_pose: Pose,
    center_rgb_arr: np.ndarray,
    center_depth_m: np.ndarray,
    out_w: int,
    out_h: int,
    hole_dilate_px: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Run forward warp + dilate + RGBA mask. Returns (warped_pil_rgb,
    mask_rgba_pil, mask_gray_pil)."""
    warp = forward_warp_erp(
        src_rgb=center_rgb_arr,
        src_depth=center_depth_m,
        src_xyz=center_pose.xyz,
        src_R=center_pose.R,
        dst_xyz=pose.xyz,
        dst_R=pose.R,
        out_size=(out_w, out_h),
    )
    hole_mask = dilate_mask(warp.hole_mask, dilate_px=int(hole_dilate_px))
    warped_pil = Image.fromarray(warp.rgb, "RGB")
    mask_rgba = hole_mask_to_openai_alpha(hole_mask, base_rgb=warp.rgb)
    mask_pil = Image.fromarray(mask_rgba, "RGBA")
    mask_gray = Image.fromarray(hole_mask, "L")
    return warped_pil, mask_pil, mask_gray


def run_hybrid_nvs(
    *,
    poses: List[Pose],
    scene: SceneSpec,
    client: ImageClient,
    run_dir: Path,
    depth_near_m: float,
    depth_far_m: float,
    verbose: bool = True,
    dap_model_size: str = "base",
    corner_method: str = "warp_inpaint",
    hole_dilate_px: int = 4,
    save_dpi: tuple[int, int] = (300, 300),
    # Legacy kwargs kept for callers that still pass them; ignored.
    strategy: str | None = None,
    dap_mode: str | None = None,
) -> List[PoseTriplet]:
    """Generate aligned (rgb, depth, normal) ERPs for every pose."""
    if not poses:
        raise ValueError("poses is empty")
    Wo, Ho = client.parse_size()
    n = len(poses)

    if verbose:
        print(
            f"[nvs] poses={n}  size={Wo}x{Ho}  dap_model={dap_model_size}  "
            f"corner_method={corner_method}  hole_dilate_px={hole_dilate_px}",
            flush=True,
        )

    triplets: List[PoseTriplet] = []
    center_rgb_arr: np.ndarray | None = None
    center_depth_m: np.ndarray | None = None
    center_pose: Pose | None = None
    run_start = _time.time()
    # Memoise relay-side failure of mask-aware edits so we stop burning retries.
    corner_path_works: bool = True
    actual_corner_method = corner_method

    for idx, pose in enumerate(poses):
        pose_start = _time.time()
        if verbose:
            print(
                f"[nvs] >>> pose {idx + 1}/{n} (name={pose.name}) "
                f"elapsed_total={(_time.time() - run_start):.0f}s",
                flush=True,
            )

        rgb_prompt = build_prompt(scene, kind="rgb", pose_idx=idx, total_poses=n)
        is_center = idx == 0

        # ----- step 1: RGB -----
        t0 = _time.time()
        warped_pil: Image.Image | None = None
        mask_gray_pil: Image.Image | None = None
        if is_center:
            rgb_img = client.generate_rgb(rgb_prompt).convert("RGB")
            label = "center"
        else:
            if center_rgb_arr is None or center_depth_m is None or center_pose is None:
                raise RuntimeError("non-center pose needs centre RGBD first")

            method = actual_corner_method if corner_path_works else "generate"
            if method == "warp_inpaint":
                try:
                    warped_pil, mask_rgba_pil, mask_gray_pil = _build_warp_inputs(
                        pose=pose, center_pose=center_pose,
                        center_rgb_arr=center_rgb_arr,
                        center_depth_m=center_depth_m,
                        out_w=Wo, out_h=Ho,
                        hole_dilate_px=hole_dilate_px,
                    )
                    corner_prompt = (
                        "Fill ONLY the transparent regions of this masked "
                        "equirectangular panorama. Keep every visible "
                        "(non-transparent) pixel exactly as given. "
                        "Match the room exactly: same materials, same "
                        "lighting, same colours. Continue the existing "
                        "geometry seamlessly into the holes. " + rgb_prompt
                    )
                    rgb_img = client.edit_with_mask(
                        corner_prompt, warped_pil, mask_rgba_pil,
                    ).convert("RGB")
                    label = "warp_inpaint"
                except Exception as exc:
                    if verbose:
                        print(
                            f"[nvs]   pose {idx + 1} warp_inpaint failed "
                            f"({exc}); falling back to plain generate for this "
                            f"and the remaining corner poses",
                            flush=True,
                        )
                    corner_path_works = False
                    rgb_img = client.generate_rgb(rgb_prompt).convert("RGB")
                    label = "generate-fallback"
            elif method == "i2i":
                try:
                    ref_pose_hint = (
                        "This is the SAME interior space as the reference "
                        "image, viewed from a DIFFERENT angle within the "
                        "same room. Preserve the room's identity exactly: "
                        "same materials, same wall and floor finishes, same "
                        "furniture and color palette, same lighting, "
                        "completely empty (no people, no animals). Only the "
                        "camera viewpoint changes. "
                    )
                    center_rgb_pil = Image.fromarray(center_rgb_arr, "RGB")
                    rgb_img = client.generate_with_ref(
                        ref_pose_hint + rgb_prompt, center_rgb_pil,
                    ).convert("RGB")
                    label = "i2i"
                except Exception as exc:
                    if verbose:
                        print(
                            f"[nvs]   pose {idx + 1} i2i failed ({exc}); "
                            f"falling back to plain generate",
                            flush=True,
                        )
                    corner_path_works = False
                    rgb_img = client.generate_rgb(rgb_prompt).convert("RGB")
                    label = "generate-fallback"
            else:  # "generate" (or fallback)
                rgb_img = client.generate_rgb(rgb_prompt).convert("RGB")
                label = "generate-noref"
        rgb_dt = _time.time() - t0
        if verbose:
            print(
                f"[nvs]   pose {idx + 1} rgb done ({rgb_dt:.0f}s, {label})",
                flush=True,
            )

        # canonicalize size (server may return server-native size; if so,
        # match the configured ERP shape so downstream samplers stay aligned)
        if rgb_img.size != (Wo, Ho):
            rgb_img = rgb_img.resize((Wo, Ho), Image.BICUBIC)

        # ----- step 2: depth via DAP-V2 -----
        t0 = _time.time()
        dap_res = estimate_erp_depth(
            rgb_img,
            near_m=depth_near_m, far_m=depth_far_m,
            model_size=dap_model_size, device="cuda", mode="direct",
        )
        depth_m = dap_res.depth_m
        if verbose:
            print(
                f"[nvs]   pose {idx + 1} depth done "
                f"({(_time.time() - t0):.1f}s, "
                f"DAP={dap_res.model_id.split('/')[-1]} "
                f"infer={dap_res.inference_ms:.0f}ms)",
                flush=True,
            )

        # ----- step 3: analytic normals from depth -----
        t0 = _time.time()
        normal_world = normals_from_erp_depth(depth_m, pose_R=pose.R, smooth_radius=1)
        normal_dt = _time.time() - t0
        if verbose:
            print(
                f"[nvs]   pose {idx + 1} normal done ({normal_dt:.2f}s, analytic)",
                flush=True,
            )

        if is_center:
            center_rgb_arr = np.array(rgb_img, dtype=np.uint8)
            center_depth_m = depth_m.astype(np.float32)
            center_pose = pose

        depth_png = encode_depth_png(depth_m, near_m=depth_near_m, far_m=depth_far_m)
        normal_png = encode_normal_png(normal_world)

        triplet = PoseTriplet(
            pose_idx=idx, pose=pose,
            rgb=rgb_img, depth_img=depth_png, normal_img=normal_png,
            depth_m=depth_m, normal_world=normal_world,
            warped_rgb=warped_pil, hole_mask=mask_gray_pil,
        )
        triplets.append(triplet)
        _save_triplet(triplet, run_dir, dpi=save_dpi)
        if verbose:
            dt = _time.time() - pose_start
            print(
                f"[nvs] <<< pose {idx + 1}/{n} done in {dt:.0f}s "
                f"(rgb={rgb_dt:.0f}s, depth={dap_res.inference_ms / 1000.0:.1f}s, "
                f"normal={normal_dt:.2f}s)",
                flush=True,
            )

    return triplets
