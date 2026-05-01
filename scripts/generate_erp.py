"""Stage 1 driver: pose set + prompt -> 9 ERP triplets + COLMAP inputs.

Saves all artifacts under ``outputs/runs/<run_id>/``. Optionally runs the
gpt-image-2 "decoder" experiment (RGB ERP -> depth/normal map via the same
model) for a small subset of poses, and writes a JSON comparing the result
against DAP-V2 / analytic normals.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.colmap_writer import write_colmap_sparse  # noqa: E402
from erpgen.config import (
    REPO_ROOT as _ROOT,  # noqa: F401  -- imported for resolve consistency
    load_config,
    make_run_id,
    resolve_run_dir,
    save_resolved_config,
)
from erpgen.dap import encode_depth_png  # noqa: E402
from erpgen.decode import decode_depth_png, decode_normal_png, try_dap_calibrate  # noqa: E402
from erpgen.erp_to_persp import rotate_normals_to_world, split_all_to_perspectives  # noqa: E402
from erpgen.init_pcd import build_init_pcd, save_pcd_ply  # noqa: E402
from erpgen.nvs import run_hybrid_nvs  # noqa: E402
from erpgen.openai_erp import ImageClient, OpenAIConfig  # noqa: E402
from erpgen.poses import build_pose_set, save_poses_json  # noqa: E402
from erpgen.prompts import scene_from_user_input  # noqa: E402
from erpgen.scene_expander import SceneExpander  # noqa: E402

DEFAULT_DPI = (300, 300)


def _save_meta(run_dir: Path, payload: dict) -> Path:
    p = run_dir / "meta.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _save_with_dpi(img: Image.Image, path: Path | str, *, dpi: tuple[int, int] = DEFAULT_DPI) -> None:
    img.save(str(path), dpi=dpi)


def _normalize_depth_to_unit(depth_m: np.ndarray, *, near_m: float, far_m: float) -> np.ndarray:
    """Clamp metric depth into [near_m, far_m] then linearly map to [0,1].
    The mapping is the same convention encode_depth_png uses, so we can
    compare the LLM-decoded grayscale depth against this directly."""
    d = np.clip(depth_m.astype(np.float32), near_m, far_m)
    d = (d - near_m) / max(1e-6, far_m - near_m)
    return 1.0 - d  # white=near, black=far (matches the prompt + DAP encoding)


def _decoded_depth_gray(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return arr  # white=near (1.0), black=far (0.0) per the decoder prompt


def _decoded_normal_unit(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    n = arr * 2.0 - 1.0  # decode RGB[0,1] -> XYZ[-1,1]
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    return n / norm


def _run_decoder_experiment(
    *,
    client: ImageClient,
    triplets: list,
    run_dir: Path,
    pose_indices: list,
    depth_near_m: float,
    depth_far_m: float,
    dpi: tuple[int, int],
    verbose: bool,
) -> list[dict]:
    """For each pose in ``pose_indices``, ask gpt-image-2 to convert the RGB
    ERP into a depth and a normal map, then compare against DAP-V2 / analytic
    normals. Saves PNGs under ``erp/decoder_*`` and returns a JSON-friendly
    list of per-pose comparison stats."""
    if not pose_indices:
        return []
    erp = run_dir / "erp"
    (erp / "decoder_depth").mkdir(parents=True, exist_ok=True)
    (erp / "decoder_normal").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx in pose_indices:
        if idx < 0 or idx >= len(triplets):
            if verbose:
                print(
                    f"[decoder_exp] skipping out-of-range pose idx={idx}",
                    flush=True,
                )
            continue
        tri = triplets[idx]
        rgb_pil = tri.rgb.convert("RGB")
        if verbose:
            print(
                f"[decoder_exp] pose {idx}: requesting depth + normal "
                f"from gpt-image-2 (this is slow at high res)",
                flush=True,
            )

        rec: dict = {"pose_idx": idx}

        # depth
        try:
            t0_local = __import__("time").time()
            decoded_depth = client.decode_to_depth(rgb_pil).convert("L")
            depth_path = erp / "decoder_depth" / f"pose_{idx}.png"
            _save_with_dpi(decoded_depth, depth_path, dpi=dpi)
            rec["depth_path"] = str(depth_path)
            rec["depth_seconds"] = round(__import__("time").time() - t0_local, 1)

            decoded = _decoded_depth_gray(decoded_depth)
            ground = _normalize_depth_to_unit(
                tri.depth_m, near_m=depth_near_m, far_m=depth_far_m,
            )
            # Resize ground to decoded shape if mismatch (decoded is server-native).
            if decoded.shape != ground.shape:
                from PIL import Image as _PI
                gr = _PI.fromarray((ground * 255).astype(np.uint8), "L").resize(
                    (decoded.shape[1], decoded.shape[0]), _PI.BICUBIC
                )
                ground = np.asarray(gr, dtype=np.float32) / 255.0
            mae = float(np.abs(decoded - ground).mean())
            rec["depth_mae_unit"] = round(mae, 4)
            corr = float(np.corrcoef(decoded.flatten(), ground.flatten())[0, 1])
            rec["depth_pearson"] = round(corr, 4)
        except Exception as exc:
            rec["depth_error"] = f"{type(exc).__name__}: {exc!s:.200}"

        # normal
        try:
            t0_local = __import__("time").time()
            decoded_normal = client.decode_to_normal(rgb_pil).convert("RGB")
            normal_path = erp / "decoder_normal" / f"pose_{idx}.png"
            _save_with_dpi(decoded_normal, normal_path, dpi=dpi)
            rec["normal_path"] = str(normal_path)
            rec["normal_seconds"] = round(__import__("time").time() - t0_local, 1)

            decoded = _decoded_normal_unit(decoded_normal)
            ground = tri.normal_world.astype(np.float32)
            if decoded.shape != ground.shape:
                # resample ground to decoded resolution by nearest-neighbour
                # (analytic normals are smooth so this is fine for stats).
                from PIL import Image as _PI
                tmp = _PI.fromarray(
                    ((ground * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8), "RGB"
                ).resize((decoded.shape[1], decoded.shape[0]), _PI.BICUBIC)
                gr_arr = np.asarray(tmp, dtype=np.float32) / 255.0
                ground = gr_arr * 2.0 - 1.0
                gnorm = np.linalg.norm(ground, axis=-1, keepdims=True)
                ground = ground / np.maximum(gnorm, 1e-6)
            dot = (decoded * ground).sum(axis=-1).clip(-1.0, 1.0)
            ang_deg = float(np.degrees(np.arccos(dot)).mean())
            rec["normal_angle_deg_mean"] = round(ang_deg, 2)
        except Exception as exc:
            rec["normal_error"] = f"{type(exc).__name__}: {exc!s:.200}"

        results.append(rec)
        if verbose:
            print(
                f"[decoder_exp] pose {idx} done: "
                f"depth_mae={rec.get('depth_mae_unit', 'n/a')} "
                f"normal_angle_deg={rec.get('normal_angle_deg_mean', 'n/a')}",
                flush=True,
            )

    out = run_dir / "decoder_vs_dap.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if verbose:
        print(f"[decoder_exp] wrote {out}", flush=True)
    return results


def generate(
    config_path: str,
    *,
    overrides: Sequence[str] | None = None,
    run_id: str | None = None,
    verbose: bool = True,
    scene: str | None = None,
    no_expand: bool = False,
) -> Path:
    cfg = load_config(config_path, overrides=overrides)
    rid = run_id or make_run_id()
    cfg.run.run_id = rid
    run_dir = resolve_run_dir(cfg, rid)
    save_resolved_config(cfg, run_dir)

    # ---- 1. poses ----
    poses = build_pose_set(cfg, seed=cfg.prompt.seed)
    save_poses_json(run_dir / "poses.json", poses)
    if verbose:
        print(f"[generate_erp] poses: {[p.name for p in poses]}", flush=True)

    # ---- 2. prompts (one scene shared across poses) ----
    scene = scene if scene is not None else str(cfg.prompt.scene_default)
    if no_expand:
        scene_spec = scene_from_user_input(scene)
        if verbose:
            print(f"[generate_erp] using raw scene: {scene!r}", flush=True)
    else:
        expander = SceneExpander(
            model=str(cfg.openai.text_model),
            api_key_env=str(cfg.openai.text_model_api_key_env),
            api_key=str(cfg.openai.get("api_key", "")),
            base_url=str(cfg.openai.get("base_url", "")),
            provider=str(cfg.openai.get("provider", "openai")),
            http_referer=str(cfg.openai.get("http_referer", "https://github.com/Strange-animalss/EGMOR")),
            app_title=str(cfg.openai.get("app_title", "EGMOR")),
            reasoning_effort=str(cfg.openai.reasoning_effort),
            verbose=verbose,
        )
        scene_spec = expander.expand(scene, seed=cfg.prompt.seed)
        if verbose:
            print(f"[generate_erp] expanded scene: {scene_spec.to_dict()}", flush=True)

    (run_dir / "prompts.json").write_text(
        json.dumps({"scene": scene_spec.to_dict()}, indent=2), encoding="utf-8"
    )

    # ---- 3. ERP triplet generation ----
    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO_ROOT / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, verbose=verbose)

    nvs_strategy = cfg.get("nvs_strategy", {}) or {}
    triplets = run_hybrid_nvs(
        poses=poses,
        scene=scene_spec,
        client=client,
        run_dir=run_dir,
        depth_near_m=float(cfg.nvs.depth_near_m),
        depth_far_m=float(cfg.nvs.depth_far_m),
        verbose=verbose,
        dap_model_size=str(cfg.nvs.get("dap_model_size", "base")),
        corner_method=str(nvs_strategy.get("corner_method", "warp_inpaint")),
        hole_dilate_px=int(nvs_strategy.get("hole_dilate_px", 4)),
        save_dpi=DEFAULT_DPI,
    )

    # ---- 3b. (optional) decoder experiment: gpt-image-2 RGB->depth/normal -----
    decoder_cfg = cfg.get("decoder_experiment", {}) or {}
    decoder_results: list[dict] = []
    if bool(decoder_cfg.get("enabled", False)):
        exp_poses = list(decoder_cfg.get("experiment_poses", []) or [])
        decoder_results = _run_decoder_experiment(
            client=client, triplets=triplets, run_dir=run_dir,
            pose_indices=exp_poses, depth_near_m=float(cfg.nvs.depth_near_m),
            depth_far_m=float(cfg.nvs.depth_far_m), dpi=DEFAULT_DPI,
            verbose=verbose,
        )

    # ---- 4. collect arrays (depth/normal already estimated by NVS via DAP) ----
    Wo, Ho = client.parse_size()
    rgb_arrs: list[np.ndarray] = []
    depth_arrs: list[np.ndarray] = []
    normal_world_arrs: list[np.ndarray] = []
    for i, tri in enumerate(triplets):
        rgb = np.array(tri.rgb.convert("RGB"))
        if tri.depth_m is None or tri.normal_world is None:
            # Backwards-compat path: triplet came from an old code path that
            # only had PNG depth/normal. Decode the PNGs into floats.
            dep = decode_depth_png(
                tri.depth_img.convert("RGB"),
                near_m=float(cfg.nvs.depth_near_m),
                far_m=float(cfg.nvs.depth_far_m),
            )
            nrm_cam = decode_normal_png(tri.normal_img.convert("RGB"))
            nrm_world = rotate_normals_to_world(nrm_cam, tri.pose.R)
        else:
            dep = tri.depth_m
            nrm_world = tri.normal_world
        rgb_arrs.append(rgb)
        depth_arrs.append(dep)
        normal_world_arrs.append(nrm_world)

    # NVS already saved .npy depth+normal alongside each triplet, but in case
    # the old PNG-decoded path was hit above, ensure all .npy files exist.
    npy_dir = run_dir / "erp_decoded"
    npy_dir.mkdir(parents=True, exist_ok=True)
    for i, (d, n) in enumerate(zip(depth_arrs, normal_world_arrs)):
        dpath = npy_dir / f"pose_{i}_depth_m.npy"
        npath = npy_dir / f"pose_{i}_normal_world.npy"
        if not dpath.exists():
            np.save(dpath, d)
        if not npath.exists():
            np.save(npath, n)

    # ---- 5. perspective split (persp48_zigzag by default) ----
    persp_dir = run_dir / "perspective"
    pose_face_sets, _cam_json = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgb_arrs,
        depth_erps_m=depth_arrs,
        normal_erps_world=normal_world_arrs,
        out_dir=persp_dir,
        scheme=str(cfg.perspective.scheme),
        fov_deg=float(cfg.perspective.fov_deg),
        out_size=int(cfg.perspective.out_size),
    )

    # ---- 6. init point cloud + COLMAP ----
    pcd = build_init_pcd(
        poses=poses,
        rgb_erps=rgb_arrs,
        depth_erps_m=depth_arrs,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
        voxel_m=float(cfg.fastgs.init_voxel_m),
        max_points=int(cfg.fastgs.init_max_points),
        stride=4,
    )
    colmap_dir = run_dir / "colmap"
    save_pcd_ply(pcd, colmap_dir / "init_pcd.ply")
    write_colmap_sparse(
        pose_face_sets=pose_face_sets,
        init_pcd=pcd if cfg.fastgs.init_from_points3d else None,
        out_dir=colmap_dir,
        copy_images=True,
    )

    # ---- 7. meta dump ----
    meta = {
        "run_id": rid,
        "scene": scene_spec.to_dict(),
        "cuboid_size": list(cfg.cuboid.size_xyz),
        "cuboid_center": list(cfg.cuboid.center_world),
        "num_poses": int(len(poses)),
        "perspective_scheme": str(cfg.perspective.scheme),
        "perspective_out_size": int(cfg.perspective.out_size),
        "init_pcd_count": int(pcd.xyz.shape[0]),
        "provider": str(cfg.openai.provider),
        "model": str(cfg.openai.model),
        "size": str(cfg.openai.size),
        "rgb_quality": str(cfg.openai.rgb_quality),
        "corner_method": str((cfg.get("nvs_strategy", {}) or {}).get("corner_method", "warp_inpaint")),
        "decoder_experiment": {
            "enabled": bool((cfg.get("decoder_experiment", {}) or {}).get("enabled", False)),
            "experiment_poses": list((cfg.get("decoder_experiment", {}) or {}).get("experiment_poses", [])),
            "results": decoder_results if 'decoder_results' in dir() else [],
        },
    }
    _save_meta(run_dir, meta)
    if verbose:
        print(f"[generate_erp] done -> {run_dir}", flush=True)
    return run_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="", help="empty -> auto timestamp")
    p.add_argument("--scene", type=str, default=None,
                   help="free-form scene description (e.g. 'a cyberpunk bar at midnight')")
    p.add_argument("--no-expand", action="store_true",
                   help="skip the LLM SceneExpander (use the raw --scene text directly)")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist (e.g. cuboid.size_xyz=[6,6,3])")
    args = p.parse_args()
    generate(
        args.config,
        overrides=args.overrides,
        run_id=args.run_id or None,
        scene=args.scene,
        no_expand=args.no_expand,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
