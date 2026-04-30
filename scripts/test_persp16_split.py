"""Smoke test the persp16 split scheme using a mock ERP."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from erpgen.config import load_config  # noqa: E402
from erpgen.erp_to_persp import (  # noqa: E402
    PERSP16_FACES,
    persp16_face_names,
    split_all_to_perspectives,
)
from erpgen.openai_erp import ImageClient, OpenAIConfig  # noqa: E402
from erpgen.poses import build_pose_set  # noqa: E402


def main() -> int:
    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    cfg.mock.enabled = True
    cfg.perspective.scheme = "persp16"
    cfg.perspective.out_size = 256  # tiny for speed

    print(f"persp16 face names ({len(persp16_face_names())}): {persp16_face_names()}")
    print(f"PERSP16_FACES rotations OK: {len(PERSP16_FACES) == 16}")

    poses = build_pose_set(cfg, seed=cfg.prompt.seed)
    print(f"poses: {len(poses)}")

    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO_ROOT / oa_cfg.cache_dir)
    client = ImageClient(oa_cfg, mock=True, verbose=False)

    rgb_arrs = []
    depth_arrs = []
    normal_arrs = []
    for _ in poses:
        rgb_arrs.append(np.array(client.generate_rgb("test", size="1024x512")))
        depth_arrs.append(np.full((512, 1024), 5.0, dtype=np.float32))
        normal_arrs.append(np.zeros((512, 1024, 3), dtype=np.float32))
        normal_arrs[-1][..., 0] = 1.0

    out_dir = REPO_ROOT / "outputs" / "_persp16_smoke"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    sets, cameras_json = split_all_to_perspectives(
        poses=poses,
        rgb_erps=rgb_arrs,
        depth_erps_m=depth_arrs,
        normal_erps_world=normal_arrs,
        out_dir=out_dir,
        scheme="persp16",
        fov_deg=90.0,
        out_size=int(cfg.perspective.out_size),
    )

    total_faces = sum(len(s.faces) for s in sets)
    print(f"  {len(sets)} pose sets, {total_faces} faces total")
    print(f"  expected: {len(poses) * 16}")
    assert total_faces == len(poses) * 16, "wrong face count"

    sample = sets[0].faces[0]
    print(f"  sample face: pose={sample.pose_idx} name={sample.face_name}")
    print(f"    image: {sample.image_path}")
    print(f"    K diag (focal): {sample.K[0][0]:.2f}  cx={sample.K[0][2]} cy={sample.K[1][2]}")
    print(f"    R rows: {sample.R}")
    print(f"    t: {sample.t}")
    print(f"    size: {sample.width}x{sample.height}")

    print(f"  cameras.json: {cameras_json}")
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
