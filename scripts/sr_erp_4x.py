"""Stage 1.5 driver: 4x super-resolution of ERP RGBs (and bilinear 4x of
decoded depth/normal floats) for an existing run.

For each pose i in 0..N-1:
  outputs/runs/<run_id>/erp/rgb/pose_<i>.png  (1024x512)  ->
  outputs/runs/<run_id>/erp/rgb_4x/pose_<i>.png (4096x2048) via Real-ESRGAN

  outputs/runs/<run_id>/erp_decoded/pose_<i>_depth_m.npy (512x1024 float)  ->
  outputs/runs/<run_id>/erp_decoded/pose_<i>_depth_m_4x.npy (2048x4096 float) via bilinear

  same for pose_<i>_normal_world.npy.

This intentionally does *not* re-run DAP at high res. DAP-V2 outputs are
slowly varying, so a 4x bilinear upsample is visually indistinguishable from
re-running DAP at 4096x2048 for our purposes (SfM init / point-cloud back-
projection), and saves a 4-8x slower DAP pass per pose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import latest_run_dir, load_config, resolve_run_dir  # noqa: E402
from erpgen.sr import upscale_erp_4x, upscale_array_bilinear, horizontal_seam_score  # noqa: E402


def _list_poses(rgb_dir: Path) -> list[Path]:
    return sorted(rgb_dir.glob("pose_*.png"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--wrap-pad", type=int, default=128, help="ERP horizontal wrap padding (in 1x px).")
    p.add_argument("--tile", type=int, default=0, help="0 = single forward; otherwise tile size.")
    p.add_argument("--no-half", action="store_true", help="disable fp16 (debug only).")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_id:
        run_dir = resolve_run_dir(cfg, args.run_id)
    else:
        run_dir = latest_run_dir(cfg)
        if run_dir is None:
            raise SystemExit("no runs found and no --run-id given")
    print(f"[sr_erp] run_dir = {run_dir}", flush=True)

    erp_rgb_dir = run_dir / "erp" / "rgb"
    erp_rgb_4x_dir = run_dir / "erp" / "rgb_4x"
    erp_decoded_dir = run_dir / "erp_decoded"
    erp_rgb_4x_dir.mkdir(parents=True, exist_ok=True)

    pose_pngs = _list_poses(erp_rgb_dir)
    if not pose_pngs:
        raise SystemExit(f"no pose_*.png under {erp_rgb_dir}")
    print(f"[sr_erp] found {len(pose_pngs)} ERP RGB(s)", flush=True)

    # ---- 1. RGB SR ----
    sr_meta: list[dict] = []
    for png in pose_pngs:
        t0 = time.time()
        src = Image.open(png).convert("RGB")
        W0, H0 = src.size
        seam_in = horizontal_seam_score(src)
        out = upscale_erp_4x(
            src,
            wrap_pad=args.wrap_pad,
            tile=args.tile,
            half=not args.no_half,
        )
        out_path = erp_rgb_4x_dir / png.name
        out.save(out_path)
        seam_out = horizontal_seam_score(out)
        dt = time.time() - t0
        info = {
            "pose": png.stem,
            "in_size": [W0, H0],
            "out_size": list(out.size),
            "seam_in": round(seam_in, 4),
            "seam_out": round(seam_out, 4),
            "elapsed_sec": round(dt, 3),
        }
        sr_meta.append(info)
        print(
            f"  {png.name}: {W0}x{H0} -> {out.size[0]}x{out.size[1]}  "
            f"seam {seam_in:.2f}->{seam_out:.2f}  {dt:.2f}s",
            flush=True,
        )

    # ---- 2. depth/normal bilinear 4x ----
    print("[sr_erp] bilinear-upsampling depth/normal arrays...", flush=True)
    n_arrays = 0
    for npy in sorted(erp_decoded_dir.glob("pose_*_depth_m.npy")):
        if "_4x" in npy.stem:
            continue
        out_path = npy.with_name(npy.stem + "_4x.npy")
        if out_path.exists():
            continue
        arr = np.load(npy)
        up = upscale_array_bilinear(arr, args.scale)
        np.save(out_path, up.astype(arr.dtype))
        n_arrays += 1
    for npy in sorted(erp_decoded_dir.glob("pose_*_normal_world.npy")):
        if "_4x" in npy.stem:
            continue
        out_path = npy.with_name(npy.stem + "_4x.npy")
        if out_path.exists():
            continue
        arr = np.load(npy)
        up = upscale_array_bilinear(arr, args.scale)
        norm = np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-6)
        up = up / norm
        np.save(out_path, up.astype(arr.dtype))
        n_arrays += 1
    print(f"[sr_erp] upsampled {n_arrays} npy file(s)", flush=True)

    summary_path = run_dir / "sr_meta.json"
    summary = {
        "scale": args.scale,
        "wrap_pad": args.wrap_pad,
        "model": "RealESRGAN_x4plus",
        "weights": "third_party/weights/RealESRGAN_x4plus.pth",
        "per_pose": sr_meta,
        "mean_elapsed_sec": round(float(np.mean([m["elapsed_sec"] for m in sr_meta])), 3),
        "max_seam_out": round(float(max(m["seam_out"] for m in sr_meta)), 4),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sr_erp] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
