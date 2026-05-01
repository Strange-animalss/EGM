"""Forensic checks for the persp48_zigzag perspective_4x output.

Validates:
  A. The on-disk frame_000_yaw0_pitch+45.png really looks like a perspective
     view, by re-deriving the ERP UV that its center pixel should sample
     from and comparing the perspective center pixel to the ERP pixel.
  B. The R_erp_from_face rotations are correct (yaw/pitch math).
  C. pose_8 is not literally the same as pose_0 (i.e. corner pose has its
     own ERP source, not a copy).
  D. cameras.json view 0 has K=focal pinhole, R=pose_0.R @ R_erp_face_0.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.erp_to_persp import PERSP48_ZIGZAG_FACES, persp48_zigzag_face_names  # noqa: E402
from erpgen.poses import load_poses_json  # noqa: E402
from erpgen.warp import erp_dirs_to_uv  # noqa: E402

RUN = REPO_ROOT / "outputs" / "runs" / "cafe_v3_20260430-121553"


def hexdig(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_rotation_sanity():
    section("C. Rotation sanity (R_erp_from_face @ +X = expected world dir)")
    names = persp48_zigzag_face_names()

    test_cases = [
        ("frame_000", 0, +45),
        ("frame_004", 30, -45),
        ("frame_024", 180, +45),
        ("frame_028", 210, -45),
        ("frame_044", 330, -45),
    ]
    forward_face = np.array([1.0, 0.0, 0.0])
    all_ok = True
    for prefix, yaw_deg, pitch_deg in test_cases:
        match = [n for n in names if n.startswith(prefix + "_")]
        assert len(match) == 1, f"name lookup failed for {prefix}"
        name = match[0]
        R = PERSP48_ZIGZAG_FACES[name]
        fwd = R @ forward_face
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        expected = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])
        ok = np.allclose(fwd, expected, atol=1e-3)
        all_ok = all_ok and ok
        print(
            f"  {name:42s}  fwd={fwd.round(3)}  expected={expected.round(3)}  ok={ok}"
        )
    print(f"  ALL OK: {all_ok}")
    return all_ok


def _erp_uv_from_world_dir(world_dir: np.ndarray, W: int, H: int) -> tuple[float, float]:
    """For a unit world-frame direction (camera-frame too, since ERP is camera frame),
    compute the (u, v) ERP pixel that corresponds to it. Inverse of erp_dirs_to_uv.
    """
    x, y, z = float(world_dir[0]), float(world_dir[1]), float(world_dir[2])
    theta = np.arctan2(y, x)
    phi = np.arcsin(np.clip(z, -1.0, 1.0))
    u = (theta / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - phi / np.pi) * H
    return u, v


def check_perspective_vs_erp():
    section("A. Perspective frame center vs ERP source pixel")
    poses = load_poses_json(RUN / "poses.json")

    persp_dir = RUN / "perspective_4x"
    erp_4x_dir = RUN / "erp" / "rgb_4x"
    erp_1x_dir = RUN / "erp" / "rgb"

    for pose_idx, frame_prefix, yaw_deg, pitch_deg in [
        (0, "frame_000", 0, +45),
        (0, "frame_004", 30, -45),
        (0, "frame_024", 180, +45),
        (5, "frame_012", 90, -45),
    ]:
        pose = poses[pose_idx]
        names = [n for n in persp48_zigzag_face_names() if n.startswith(frame_prefix + "_")]
        assert len(names) == 1
        face_name = names[0]
        R_erp_face = PERSP48_ZIGZAG_FACES[face_name]
        center_face = np.array([1.0, 0.0, 0.0])
        center_erp = R_erp_face @ center_face

        persp_path = persp_dir / f"pose_{pose_idx}" / "rgb" / f"{face_name}.png"
        erp_4x_path = erp_4x_dir / f"pose_{pose_idx}.png"
        erp_1x_path = erp_1x_dir / f"pose_{pose_idx}.png"
        if not persp_path.exists():
            print(f"  MISSING: {persp_path}")
            continue
        persp = np.asarray(Image.open(persp_path).convert("RGB"))
        erp_4x = np.asarray(Image.open(erp_4x_path).convert("RGB")) if erp_4x_path.exists() else None
        erp_1x = np.asarray(Image.open(erp_1x_path).convert("RGB"))

        H_p, W_p = persp.shape[:2]
        cy, cx = H_p // 2, W_p // 2
        persp_center = persp[cy, cx]

        H_4x, W_4x = (erp_4x.shape[:2] if erp_4x is not None else (None, None))
        H_1x, W_1x = erp_1x.shape[:2]

        u_4x = v_4x = sample_4x = None
        if erp_4x is not None:
            u_4x, v_4x = _erp_uv_from_world_dir(center_erp, W_4x, H_4x)
            sample_4x = erp_4x[
                int(np.clip(v_4x, 0, H_4x - 1)),
                int(np.clip(u_4x, 0, W_4x - 1)) % W_4x,
            ]
        u_1x, v_1x = _erp_uv_from_world_dir(center_erp, W_1x, H_1x)
        sample_1x = erp_1x[
            int(np.clip(v_1x, 0, H_1x - 1)),
            int(np.clip(u_1x, 0, W_1x - 1)) % W_1x,
        ]
        print(
            f"  pose={pose_idx} {face_name}\n"
            f"    persp size={W_p}x{H_p}  center px (RGB)={persp_center}\n"
            f"    ERP 1x  uv=({u_1x:.1f},{v_1x:.1f}) sample={sample_1x}\n"
            f"    ERP 4x  uv=({u_4x},{v_4x}) sample={sample_4x}\n"
            f"    delta(persp-erp_1x)={np.abs(persp_center.astype(int)-sample_1x.astype(int))}"
        )


def check_pose_distinct():
    section("B. pose_0 vs pose_8 frame_000 distinctness")
    p0 = RUN / "perspective_4x" / "pose_0" / "rgb" / "frame_000_yaw0_pitch+45.png"
    p8 = RUN / "perspective_4x" / "pose_8" / "rgb" / "frame_000_yaw0_pitch+45.png"
    if not p0.exists() or not p8.exists():
        print(f"  missing: {p0.exists()=} {p8.exists()=}")
        return
    h0, h8 = hexdig(p0), hexdig(p8)
    print(f"  pose_0/frame_000 sha={h0}")
    print(f"  pose_8/frame_000 sha={h8}")
    print(f"  same image: {h0 == h8}")
    print()
    erp0 = RUN / "erp" / "rgb_4x" / "pose_0.png"
    erp8 = RUN / "erp" / "rgb_4x" / "pose_8.png"
    print(f"  erp_4x/pose_0.png sha={hexdig(erp0)}  size={erp0.stat().st_size}")
    print(f"  erp_4x/pose_8.png sha={hexdig(erp8)}  size={erp8.stat().st_size}")

    print()
    print("  pose_8 directory listing (first 5 + last 5 of rgb/):")
    rgb8 = sorted((RUN / "perspective_4x" / "pose_8" / "rgb").glob("*.png"))
    for p in rgb8[:5] + rgb8[-5:]:
        print(f"    {p.name}")
    print(f"  total RGB frames in pose_8: {len(rgb8)}")
    print(f"  pose_8 unique file shas (first 4): "
          f"{[hexdig(p) for p in rgb8[:4]]}")
    print(f"  any duplicate hashes in pose_8 rgb? "
          f"{len({hexdig(p) for p in rgb8}) != len(rgb8)}")


def check_cameras_json():
    section("D. cameras.json view 0 K, R, t")
    cams_path = RUN / "perspective_4x" / "cameras.json"
    cams = json.loads(cams_path.read_text(encoding="utf-8"))
    print(f"  fov_deg={cams['fov_deg']}  out_size={cams['out_size']}  scheme={cams['scheme']}  n_views={len(cams['views'])}")
    v0 = cams["views"][0]
    print(f"  view 0: pose_idx={v0['pose_idx']} face={v0['face_name']} W={v0['width']} H={v0['height']}")
    K = np.asarray(v0["K"])
    R = np.asarray(v0["R"])
    t = np.asarray(v0["t"])
    print(f"  K=\n{K}")
    print(f"  R=\n{R}")
    print(f"  t={t}")

    poses = load_poses_json(RUN / "poses.json")
    pose0 = poses[0]
    R_erp_face_0 = PERSP48_ZIGZAG_FACES[v0["face_name"]]
    R_expected = pose0.R @ R_erp_face_0
    print(f"  pose_0.R @ R_erp_face_0 =\n{R_expected}")
    print(f"  R matches expected: {np.allclose(R, R_expected, atol=1e-6)}")

    fov_deg = float(cams["fov_deg"])
    out_size = int(cams["out_size"])
    f_expected = (out_size * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
    print(f"  expected focal = {f_expected:.2f}  actual fx = {K[0,0]:.2f}  ok={abs(K[0,0]-f_expected) < 1e-3}")


def check_source_used():
    section("E. Source ERP shape used by re-split")
    rgb_4x = RUN / "erp" / "rgb_4x" / "pose_0.png"
    rgb_1x = RUN / "erp" / "rgb" / "pose_0.png"
    img_4x = Image.open(rgb_4x)
    img_1x = Image.open(rgb_1x)
    print(f"  erp/rgb_4x/pose_0.png  size={img_4x.size}")
    print(f"  erp/rgb/pose_0.png     size={img_1x.size}")

    persp_path = RUN / "perspective_4x" / "pose_0" / "rgb" / "frame_000_yaw0_pitch+45.png"
    pers = Image.open(persp_path)
    print(f"  perspective_4x/.../frame_000.png size={pers.size}")

    sr_meta_path = RUN / "sr_meta.json"
    if sr_meta_path.exists():
        sr_meta = json.loads(sr_meta_path.read_text(encoding="utf-8"))
        print(f"  sr_meta first pose out_size: {sr_meta['per_pose'][0]['out_size']}")

    regen_meta_path = RUN / "regen_4x_meta.json"
    if regen_meta_path.exists():
        regen = json.loads(regen_meta_path.read_text(encoding="utf-8"))
        print(f"  regen_4x_meta: {json.dumps({k:v for k,v in regen.items() if k != 'zigzag_yaw_pitch'}, indent=2)}")


def main() -> int:
    check_rotation_sanity()
    check_source_used()
    check_perspective_vs_erp()
    check_pose_distinct()
    check_cameras_json()
    return 0


if __name__ == "__main__":
    sys.exit(main())
