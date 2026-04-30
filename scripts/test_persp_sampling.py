"""Synthetic-ERP forensics for the persp48_zigzag perspective sampler.

Builds a clean test ERP with known geometry (parallels every 15 deg of
elevation, meridians every 30 deg of azimuth, cardinal-direction colour
blocks at the equator), then runs `split_pose_to_perspectives` against it
and asserts the output really is a gnomonic (pinhole) view:

  GNOMONIC PROJECTION PROPERTY: any great circle through the camera origin
  projects to a STRAIGHT line in the image plane. So if we project the
  azimuth=0 meridian (which passes through the image center for yaw=0 at
  any pitch), it must be a vertical straight line on screen with R^2 > 0.999.

  Likewise the equator (a great circle) is a straight line. Latitude
  parallels (small circles, NOT great circles) are NOT required to be
  straight; they bow into hyperbolas/parabolas. We do NOT test those.

The script writes the synthetic ERP, the two tested perspective frames, and
an annotated overlay (red = expected great-circle line, white pixels = where
we actually sampled it) to outputs/_persp_sanity/.

Pass criteria:
  - mean residual of meridian fit (px) < 1.0
  - mean residual of equator fit (px) < 1.0
  - R^2 of both > 0.9999
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.erp_to_persp import (  # noqa: E402
    PERSP48_ZIGZAG_FACES,
    _bilinear_sample,
    _erp_sample_uv,
    _face_pixel_dirs,
    persp48_zigzag_face_names,
    split_pose_to_perspectives,
)
from erpgen.poses import Pose  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "_persp_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1024, 512
PERSP_SIZE = 512
FOV = 90.0


def make_synthetic_erp() -> np.ndarray:
    """1024x512 RGB. Black background. Gridlines + cardinal colour blocks."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for elev_deg in range(-75, 76, 15):
        v = int(round((0.5 - np.deg2rad(elev_deg) / np.pi) * H))
        if 0 <= v < H:
            img[v - 1:v + 1, :, :] = (0, 200, 0) if elev_deg != 0 else (0, 255, 255)
    for az_deg in range(-180, 181, 30):
        u = int(round((np.deg2rad(az_deg) / (2 * np.pi) + 0.5) * W)) % W
        col = (255, 0, 0) if az_deg in (-180, 0, 180) else (200, 200, 200)
        img[:, u - 1:u + 1, :] = col
    cardinals = {
        0:   (255, 255, 0),
        90:  (255, 128, 0),
        180: (255, 0, 255),
        270: (0, 128, 255),
    }
    block_h = 60
    block_w = 60
    eq_v = H // 2
    for az_deg, color in cardinals.items():
        u = int(round((np.deg2rad(az_deg) / (2 * np.pi) + 0.5) * W)) % W
        u_lo = (u - block_w // 2) % W
        u_hi = (u + block_w // 2) % W
        if u_lo < u_hi:
            img[eq_v - block_h // 2:eq_v + block_h // 2, u_lo:u_hi, :] = color
        else:
            img[eq_v - block_h // 2:eq_v + block_h // 2, u_lo:, :] = color
            img[eq_v - block_h // 2:eq_v + block_h // 2, :u_hi, :] = color
    return img


def project_world_dir_to_face_pixel(R_world_face: np.ndarray, world_dir: np.ndarray) -> tuple[float, float, bool]:
    """Use the face camera intrinsics to project a unit world direction
    into the (x_pix, y_pix) plane of a perspective frame.

    Uses our face camera convention: +X forward, +Y left, +Z up, principal
    point at center, focal f = (S/2)/tan(FOV/2). dy = -(x - cx)/f, etc, see
    `_face_pixel_dirs`.
    """
    R_face_world = R_world_face.T
    d_face = R_face_world @ world_dir
    dx, dy, dz = float(d_face[0]), float(d_face[1]), float(d_face[2])
    if dx <= 1e-6:
        return float("nan"), float("nan"), False
    f = (PERSP_SIZE * 0.5) / np.tan(np.deg2rad(FOV) * 0.5)
    cx = cy = PERSP_SIZE * 0.5
    x_pix = cx - dy * f / dx
    y_pix = cy - dz * f / dx
    return x_pix, y_pix, True


def fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Return mean residual (orthogonal distance) and R^2 of a TLS line fit.

    Robust to vertical lines (no slope blow-up). Used to verify that what
    SHOULD be a straight line in the gnomonic projection actually is one.
    """
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    pts = np.stack([xs, ys], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    resid = centered @ normal
    var_total = np.var(centered.flatten()) * 2.0
    var_resid = float(np.mean(resid ** 2))
    r2 = 1.0 - var_resid / max(var_total, 1e-9)
    return float(np.mean(np.abs(resid))), float(r2)


def run_one(face_name: str, erp_rgb: np.ndarray) -> dict:
    R_erp_face = PERSP48_ZIGZAG_FACES[face_name]
    pose = Pose(xyz=np.zeros(3), R=np.eye(3), name="origin")
    out = split_pose_to_perspectives(
        pose_idx=0,
        pose=pose,
        rgb_erp=erp_rgb,
        depth_erp_m=np.full((H, W), 5.0, dtype=np.float32),
        normal_erp_world=None,
        out_dir=OUT_DIR / face_name,
        scheme="persp48_zigzag",
        fov_deg=FOV,
        out_size=PERSP_SIZE,
        face_names=[face_name],
    )
    persp_path = OUT_DIR / face_name / "pose_0" / "rgb" / f"{face_name}.png"
    persp = np.asarray(Image.open(persp_path).convert("RGB"))

    elevs_deg = np.linspace(-89, 89, 90)
    az_meridian = 0.0
    az_pts_world = []
    for el in elevs_deg:
        e = np.deg2rad(el)
        a = np.deg2rad(az_meridian)
        az_pts_world.append(np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))

    eq_pts_world = []
    for az in np.linspace(-89, 89, 90):
        a = np.deg2rad(az)
        eq_pts_world.append(np.array([np.cos(a), np.sin(a), 0.0]))

    R_world_face = pose.R @ R_erp_face

    def _project_set(world_pts):
        xs, ys, mask = [], [], []
        for d in world_pts:
            x, y, ok = project_world_dir_to_face_pixel(R_world_face, d)
            if ok and 0 <= x < PERSP_SIZE and 0 <= y < PERSP_SIZE:
                xs.append(x); ys.append(y); mask.append(True)
            else:
                mask.append(False)
        return np.array(xs), np.array(ys), np.array(mask)

    mer_x, mer_y, mer_mask = _project_set(az_pts_world)
    eq_x, eq_y, eq_mask = _project_set(eq_pts_world)

    mer_resid = mer_r2 = float("nan")
    if len(mer_x) >= 5:
        mer_resid, mer_r2 = fit_line(mer_x, mer_y)
    eq_resid = eq_r2 = float("nan")
    if len(eq_x) >= 5:
        eq_resid, eq_r2 = fit_line(eq_x, eq_y)

    overlay = persp.copy()
    drawer = ImageDraw.Draw(Image.fromarray(overlay))
    img_pil = Image.fromarray(persp.copy())
    draw = ImageDraw.Draw(img_pil)
    if len(mer_x) >= 2:
        for px, py in zip(mer_x, mer_y):
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], outline=(255, 0, 0), width=1)
    if len(eq_x) >= 2:
        for px, py in zip(eq_x, eq_y):
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], outline=(0, 255, 255), width=1)
    overlay_path = OUT_DIR / f"{face_name}_overlay.png"
    img_pil.save(overlay_path)

    expected_uv_center = None
    yaw_deg = float(face_name.split("yaw")[1].split("_")[0])
    pitch_deg = float(face_name.split("pitch")[1])
    expected_u = (np.deg2rad(yaw_deg) / (2 * np.pi) + 0.5) * W
    expected_v = (0.5 - np.deg2rad(pitch_deg) / np.pi) * H
    expected_uv_center = (float(expected_u % W), float(np.clip(expected_v, 0, H - 1)))
    erp_color = erp_rgb[int(np.clip(expected_v, 0, H - 1)), int(expected_u) % W]
    persp_center = persp[PERSP_SIZE // 2, PERSP_SIZE // 2]
    return {
        "face": face_name,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "persp_path": str(persp_path.relative_to(REPO_ROOT)),
        "overlay_path": str(overlay_path.relative_to(REPO_ROOT)),
        "n_meridian_visible": int(len(mer_x)),
        "meridian_residual_px": round(mer_resid, 4),
        "meridian_r2": round(mer_r2, 6),
        "n_equator_visible": int(len(eq_x)),
        "equator_residual_px": round(eq_resid, 4),
        "equator_r2": round(eq_r2, 6),
        "persp_center_rgb": persp_center.tolist(),
        "erp_center_rgb_at_expected_uv": erp_color.tolist(),
        "expected_erp_uv": expected_uv_center,
    }


def main() -> int:
    erp = make_synthetic_erp()
    Image.fromarray(erp, "RGB").save(OUT_DIR / "synthetic_erp.png")
    print(f"wrote {OUT_DIR / 'synthetic_erp.png'}")

    results = []
    for face_prefix in ("frame_000", "frame_004", "frame_001", "frame_024"):
        names = [n for n in persp48_zigzag_face_names() if n.startswith(face_prefix + "_")]
        if not names:
            continue
        face = names[0]
        r = run_one(face, erp)
        results.append(r)
        print(
            f"\n{face}: yaw={r['yaw_deg']} pitch={r['pitch_deg']}\n"
            f"  meridian (azimuth=0 great circle): "
            f"{r['n_meridian_visible']} samples, "
            f"residual={r['meridian_residual_px']} px, R^2={r['meridian_r2']}\n"
            f"  equator  (elevation=0 great circle): "
            f"{r['n_equator_visible']} samples, "
            f"residual={r['equator_residual_px']} px, R^2={r['equator_r2']}\n"
            f"  persp center RGB: {r['persp_center_rgb']}\n"
            f"  ERP at expected uv {r['expected_erp_uv']}: "
            f"{r['erp_center_rgb_at_expected_uv']}\n"
            f"  -> {r['overlay_path']}"
        )

    pass_count = sum(
        1 for r in results
        if (np.isnan(r["meridian_r2"]) or r["meridian_r2"] > 0.9999)
        and (np.isnan(r["equator_r2"]) or r["equator_r2"] > 0.9999)
    )
    print(f"\n{pass_count}/{len(results)} faces pass gnomonic-projection R^2 > 0.9999")
    return 0


if __name__ == "__main__":
    sys.exit(main())
