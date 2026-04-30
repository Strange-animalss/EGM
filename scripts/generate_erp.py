"""Stage 1 driver: pose set + prompt -> 9 ERP triplets + cubemap + COLMAP inputs.

Run separately or via `scripts/e2e_test.py`. Saves all artifacts under
`outputs/runs/<run_id>/`.
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
from erpgen.decode import decode_depth_png, decode_normal_png, try_dap_calibrate  # noqa: E402
from erpgen.erp_to_persp import rotate_normals_to_world, split_all_to_perspectives  # noqa: E402
from erpgen.init_pcd import build_init_pcd, save_pcd_ply  # noqa: E402
from erpgen.nvs import run_hybrid_nvs  # noqa: E402
from erpgen.openai_erp import ImageClient, OpenAIConfig  # noqa: E402
from erpgen.poses import build_pose_set, save_poses_json  # noqa: E402
from erpgen.prompts import sample_scene, scene_description, scene_from_user_input  # noqa: E402
from erpgen.scene_expander import SceneExpander  # noqa: E402


def _save_meta(run_dir: Path, payload: dict) -> Path:
    p = run_dir / "meta.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def generate(
    config_path: str,
    *,
    overrides: Sequence[str] | None = None,
    run_id: str | None = None,
    mock: bool | None = None,
    verbose: bool = True,
    scene: str | None = None,
    no_expand: bool = False,
) -> Path:
    cfg = load_config(config_path, overrides=overrides)
    if mock is not None:
        cfg.mock.enabled = bool(mock)

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
    if scene is not None:
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
                http_referer=str(cfg.openai.get("http_referer", "https://github.com/erpgen")),
                app_title=str(cfg.openai.get("app_title", "ERPGen")),
                reasoning_effort=str(cfg.openai.reasoning_effort),
                verbose=verbose,
                mock=bool(cfg.mock.enabled),
            )
            scene_spec = expander.expand(scene, seed=cfg.prompt.seed)
            if verbose:
                print(f"[generate_erp] expanded scene: {scene_spec.to_dict()}", flush=True)
    else:
        scene_spec = sample_scene(cfg, seed=cfg.prompt.seed)

    (run_dir / "prompts.json").write_text(
        json.dumps({"scene": scene_spec.to_dict()}, indent=2), encoding="utf-8"
    )

    # ---- 3. ERP triplet generation ----
    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO_ROOT / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, mock=bool(cfg.mock.enabled), verbose=verbose)

    triplets = run_hybrid_nvs(
        poses=poses,
        scene=scene_spec,
        client=client,
        run_dir=run_dir,
        hole_dilate_px=int(cfg.nvs.hole_dilate_px),
        depth_near_m=float(cfg.nvs.depth_near_m),
        depth_far_m=float(cfg.nvs.depth_far_m),
        strategy=str(cfg.nvs.strategy),
        verbose=verbose,
        dap_model_size=str(cfg.nvs.get("dap_model_size", "base")),
        dap_mode=str(cfg.nvs.get("dap_mode", "direct")),
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

    # ---- 5. perspective split (cubemap or persp16) ----
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

    # ---- 5b. (optional) per-face image-2 enhancement ----
    enhance = bool(cfg.perspective.get("enhance_with_image2", False))
    only_center = bool(cfg.perspective.get("enhance_only_center_pose", False))
    if enhance and not client.mock_mode:
        scene_text = scene_description(scene_spec)
        target_size = int(cfg.perspective.out_size)
        n_faces = sum(len(s.faces) for s in pose_face_sets)
        if only_center:
            n_faces = len(pose_face_sets[0].faces) if pose_face_sets else 0
        if verbose:
            print(
                f"[generate_erp] enhancing {n_faces} perspective faces with "
                f"{cfg.openai.model} (this is the slow part)...",
                flush=True,
            )
        for s in pose_face_sets:
            if only_center and s.pose_idx != 0:
                continue
            for face in s.faces:
                ref_img = Image.open(face.image_path).convert("RGB")
                refined = client.enhance_perspective(
                    ref_img, scene_text, out_size=target_size,
                )
                refined.save(face.image_path)
                face.width = int(target_size)
                face.height = int(target_size)
                if verbose:
                    print(
                        f"  enhanced pose_{s.pose_idx}/{face.face_name}",
                        flush=True,
                    )
        # Re-write cameras.json with possibly-updated dimensions.
        import json as _json  # local import to avoid shadowing top-level
        (persp_dir / "cameras.json").write_text(
            _json.dumps(
                {
                    "fov_deg": float(cfg.perspective.fov_deg),
                    "out_size": int(target_size),
                    "scheme": str(cfg.perspective.scheme),
                    "enhanced": True,
                    "enhance_only_center_pose": only_center,
                    "views": [face.__dict__ for s in pose_face_sets for face in s.faces],
                },
                indent=2,
            ),
            encoding="utf-8",
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
        "mock_mode": bool(client.mock_mode),
    }
    _save_meta(run_dir, meta)
    if verbose:
        print(f"[generate_erp] done -> {run_dir}", flush=True)
    return run_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="", help="empty -> auto timestamp")
    p.add_argument("--mock", action="store_true", help="force mock mode")
    p.add_argument("--no-mock", action="store_true", help="force real OpenAI API")
    p.add_argument("--scene", type=str, default=None,
                   help="free-form scene description (e.g. 'a cyberpunk bar at midnight')")
    p.add_argument("--no-expand", action="store_true",
                   help="skip GPT-5.5-pro deep reasoning expansion (use raw scene text)")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist (e.g. cuboid.size_xyz=[6,6,3])")
    args = p.parse_args()
    mock: bool | None = None
    if args.mock:
        mock = True
    elif args.no_mock:
        mock = False
    generate(
        args.config,
        overrides=args.overrides,
        run_id=args.run_id or None,
        mock=mock,
        scene=args.scene,
        no_expand=args.no_expand,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
