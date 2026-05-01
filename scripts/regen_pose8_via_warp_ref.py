"""Regenerate pose_8 ERP from a fused-warp reference image.

Inputs:
  outputs/_pose8_warp_preview/warped_rgb_fused.png   (4096x2048 with ~52% holes)

Steps:
  1. Downscale fused warp to 1024x512 (OpenRouter native).
  2. Backup current pose_8.png as pose_8_BAD.png (1x and 4x).
  3. Call image-2 i2i with the warp as ref + a strict prompt asking it to
     fill the holes while keeping the same coffee-shop and view direction.
  4. Save new 1024x512 -> erp/rgb/pose_8.png.
  5. Re-run DAP (depth + normal) on the new ERP.
  6. Real-ESRGAN x4 SR -> erp/rgb_4x/pose_8.png (4096x2048).
  7. Bilinear-upsample new depth/normal to *_4x.npy.

Idempotent on cache hits (chat/completions request gets sha256 hashed by
ImageClient so re-runs are free if prompt+ref+model are unchanged).
"""

from __future__ import annotations

import io
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import load_config, resolve_run_dir  # noqa: E402
from erpgen.dap import estimate_erp_depth, normals_from_erp_depth  # noqa: E402
from erpgen.openai_erp import ImageClient, OpenAIConfig, _hash_request, _png_bytes  # noqa: E402
from erpgen.sr import upscale_array_bilinear, upscale_erp_4x  # noqa: E402

RUN_ID = "cafe_v3_20260430-121553"

PROMPT = (
    "This is a partial 360 degree equirectangular panorama (ERP, 2:1) of a "
    "specialty pour-over and espresso coffee shop interior, viewed from a "
    "low corner of the room. The reference image was produced by warping "
    "neighboring viewpoints into this corner pose, so it has irregular "
    "BLACK HOLES where the source views did not see the surface. Your task: "
    "regenerate this exact same view as a clean, sharp, complete 360 degree "
    "equirectangular panorama. Fill in every black hole naturally so the "
    "result is a continuous, hole-free, seamless ERP. Maintain the same "
    "industrial Scandinavian coffee shop interior: 4 meter exposed concrete "
    "ceiling with matte black steel I-beams, limewashed warm-white brick "
    "walls, polished light-gray concrete floor, low oak millwork and pale "
    "ash furniture, full-height southeast-facing storefront windows, an "
    "espresso bar with two-group espresso machine and pour-over drippers, "
    "white pendant lights, ceramics and small greenery. Late morning, "
    "crisp 5600K daylight from the storefront windows, warm 2700K pendants "
    "above the bar. EMPTY ROOM, NO PEOPLE, NO ANIMALS, NO FIGURES. The room "
    "layout, furniture, lighting, and color palette must match the visible "
    "(non-black) parts of the reference image. The bottom of the image is "
    "the floor (this view is from a low corner) and the top is the ceiling. "
    "Final output must be a clean equirectangular panorama at 1024x512."
)


def main() -> int:
    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    run_dir = resolve_run_dir(cfg, RUN_ID)
    print(f"[regen_pose8] run_dir = {run_dir}", flush=True)

    erp_rgb_dir = run_dir / "erp" / "rgb"
    erp_rgb_4x_dir = run_dir / "erp" / "rgb_4x"
    erp_decoded_dir = run_dir / "erp_decoded"
    pose_8_1x = erp_rgb_dir / "pose_8.png"
    pose_8_4x = erp_rgb_4x_dir / "pose_8.png"

    bad_1x = erp_rgb_dir / "pose_8_BAD.png"
    bad_4x = erp_rgb_4x_dir / "pose_8_BAD.png"
    if not bad_1x.exists():
        shutil.copy2(pose_8_1x, bad_1x)
        print(f"[regen_pose8] backed up bad pose_8: {bad_1x}", flush=True)
    if not bad_4x.exists():
        shutil.copy2(pose_8_4x, bad_4x)

    warped_path = REPO_ROOT / "outputs" / "_pose8_warp_preview" / "warped_rgb_fused.png"
    if not warped_path.exists():
        raise SystemExit(f"missing fused warp ref: {warped_path}")
    warp = Image.open(warped_path).convert("RGB").resize((1024, 512), Image.LANCZOS)
    warp_bytes = _png_bytes(warp)
    print(f"[regen_pose8] warp ref shape={warp.size}, bytes={len(warp_bytes)}", flush=True)

    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO_ROOT / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, mock=False, verbose=True)

    cache_dir = Path(oa_cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _hash_request([
        oa_cfg.model, "1024x512", oa_cfg.rgb_quality, "pose8_warp_regen_v1",
        PROMPT, warp_bytes,
    ])
    cache_path = cache_dir / f"{cache_key}.png"

    if cache_path.exists():
        print(f"[regen_pose8] cache hit: {cache_path}", flush=True)
        new_pose8 = Image.open(cache_path).convert("RGB")
    else:
        print(f"[regen_pose8] cache miss; calling chat-image API...", flush=True)
        t0 = time.time()
        new_pose8 = client._chat_image_call(
            prompt=PROMPT,
            size="1024x512",
            ref_image_bytes=warp_bytes,
        )
        print(f"[regen_pose8] api ok in {time.time()-t0:.1f}s, size={new_pose8.size}", flush=True)
        if new_pose8.size != (1024, 512):
            new_pose8 = new_pose8.resize((1024, 512), Image.LANCZOS)
        new_pose8.save(cache_path)

    new_pose8.save(pose_8_1x)
    print(f"[regen_pose8] wrote new {pose_8_1x}", flush=True)

    print(f"\n[regen_pose8] re-running DAP on new pose_8...", flush=True)
    t0 = time.time()
    pose_R = load_pose_R(run_dir, 8)
    dap_res = estimate_erp_depth(
        new_pose8,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
        model_size=str(cfg.nvs.get("dap_model_size", "base")),
        mode=str(cfg.nvs.get("dap_mode", "direct")),
    )
    depth_m = dap_res.depth_m
    normal_world = normals_from_erp_depth(depth_m, pose_R=pose_R, smooth_radius=1)
    print(f"[regen_pose8] DAP done in {time.time()-t0:.1f}s, "
          f"depth={depth_m.shape} normal={normal_world.shape}  "
          f"({dap_res.inference_ms:.0f} ms inference, model={dap_res.model_id})", flush=True)

    np.save(erp_decoded_dir / "pose_8_depth_m.npy", depth_m.astype(np.float32))
    np.save(erp_decoded_dir / "pose_8_normal_world.npy", normal_world.astype(np.float32))

    print(f"\n[regen_pose8] Real-ESRGAN x4 SR on new pose_8...", flush=True)
    t0 = time.time()
    new_pose8_4x = upscale_erp_4x(new_pose8, wrap_pad=128, half=True, tile=0)
    new_pose8_4x.save(pose_8_4x)
    print(f"[regen_pose8] SR done in {time.time()-t0:.1f}s, size={new_pose8_4x.size}", flush=True)

    print(f"\n[regen_pose8] bilinear-upsampling new depth/normal to 4x...", flush=True)
    depth_4x = upscale_array_bilinear(depth_m.astype(np.float32), 4)
    normal_4x = upscale_array_bilinear(normal_world.astype(np.float32), 4)
    nrm = np.linalg.norm(normal_4x, axis=-1, keepdims=True).clip(1e-6)
    normal_4x = (normal_4x / nrm).astype(np.float32)
    np.save(erp_decoded_dir / "pose_8_depth_m_4x.npy", depth_4x)
    np.save(erp_decoded_dir / "pose_8_normal_world_4x.npy", normal_4x)

    print(f"\n[regen_pose8] DONE", flush=True)
    print(f"  new pose_8 1x: {pose_8_1x}")
    print(f"  new pose_8 4x: {pose_8_4x}")
    print(f"  new depth 1x:  {erp_decoded_dir / 'pose_8_depth_m.npy'}")
    print(f"  new depth 4x:  {erp_decoded_dir / 'pose_8_depth_m_4x.npy'}")
    print(f"  new normal 1x: {erp_decoded_dir / 'pose_8_normal_world.npy'}")
    print(f"  new normal 4x: {erp_decoded_dir / 'pose_8_normal_world_4x.npy'}")
    return 0


def load_pose_R(run_dir: Path, idx: int) -> np.ndarray:
    from erpgen.poses import load_poses_json
    poses = load_poses_json(run_dir / "poses.json")
    return poses[idx].R


if __name__ == "__main__":
    sys.exit(main())
