"""Stage 3 helper: produce the three .ply deliverables and a SAM3 frames manifest.

Given a run with a trained `<output_dir>/output.ply` (3DGS) and a corresponding
`<colmap_dir>/init_pcd_4x.ply` (DAP-derived back-projected colour cloud), this
script:

  1. Copies the back-projected cloud to `<output_dir>/point_cloud.ply` so the
     "also keep a point cloud" deliverable is co-located with the GS one.
  2. Reads the trained 3DGS PLY, filters out near-transparent gaussians by
     opacity, and writes `<output_dir>/gs_centers.ply` (xyz + rgb from SH DC).
  3. Writes one `frames.txt` per pose under `<persp_dir>/pose_<i>/frames.txt`
     listing the 48 zigzag-ordered frame filenames -- this is what SAM3
     wants when consuming a directory as a video.
  4. Updates `meta.json` with paths, vertex / gaussian counts, and zigzag info.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import latest_run_dir, load_config, resolve_run_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal PLY readers / writers (no plyfile dependency)
# ---------------------------------------------------------------------------


def _parse_ply_header(buf: bytes) -> tuple[dict, int, str]:
    end = buf.find(b"end_header\n")
    if end < 0:
        raise ValueError("ply: no end_header")
    header = buf[: end + len("end_header\n")].decode("ascii", errors="replace")
    body_start = end + len("end_header\n")

    fmt = ""
    n_vertex = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vertex = int(parts[2])
        elif line.startswith("property") and in_vertex:
            toks = line.split()
            ptype = toks[1]
            pname = toks[-1]
            props.append((ptype, pname))
    return {"format": fmt, "n_vertex": n_vertex, "props": props}, body_start, header


_PLY_TYPE_MAP = {
    "float": ("<f4", 4),
    "float32": ("<f4", 4),
    "double": ("<f8", 8),
    "float64": ("<f8", 8),
    "uchar": ("u1", 1),
    "uint8": ("u1", 1),
    "char": ("i1", 1),
    "int8": ("i1", 1),
    "ushort": ("<u2", 2),
    "uint16": ("<u2", 2),
    "short": ("<i2", 2),
    "int16": ("<i2", 2),
    "uint": ("<u4", 4),
    "uint32": ("<u4", 4),
    "int": ("<i4", 4),
    "int32": ("<i4", 4),
}


def _read_3dgs_ply(path: Path) -> dict[str, np.ndarray]:
    """Read a binary little-endian 3DGS PLY (e.g. nerfstudio gaussian-splat export).

    Returns a dict mapping each vertex property name to a (N,) numpy array.
    """
    raw = path.read_bytes()
    info, body_start, _ = _parse_ply_header(raw)
    if info["format"] != "binary_little_endian":
        raise ValueError(f"only binary_little_endian PLY supported, got {info['format']}")
    dt_fields: list[tuple[str, str]] = []
    for ptype, pname in info["props"]:
        if ptype not in _PLY_TYPE_MAP:
            raise ValueError(f"unsupported ply property type: {ptype}")
        np_type, _ = _PLY_TYPE_MAP[ptype]
        dt_fields.append((pname, np_type))
    dt = np.dtype(dt_fields)
    arr = np.frombuffer(raw[body_start:], dtype=dt, count=info["n_vertex"])
    return {name: arr[name].copy() for name, _ in dt_fields}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Map zero-band SH DC coefficients into 8-bit RGB (gsplat / 3DGS convention)."""
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + SH_C0 * f_dc
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _write_xyz_rgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.empty(n, dtype=dt)
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    return path


def extract_gs_centers(
    ply_in: Path,
    ply_out: Path,
    *,
    opacity_threshold: float = 0.005,
    max_points: int = 1_000_000,
) -> dict:
    """Read a trained 3DGS PLY, threshold by opacity, write xyz+rgb point cloud."""
    fields = _read_3dgs_ply(ply_in)
    needed = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"}
    missing = needed - set(fields.keys())
    if missing:
        raise RuntimeError(
            f"trained PLY {ply_in} is missing expected fields: {sorted(missing)} "
            f"(have {sorted(fields.keys())[:12]}...)"
        )
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]], axis=1).astype(np.float32)
    opac = _sigmoid(fields["opacity"].astype(np.float32))
    keep = opac > opacity_threshold
    xyz = xyz[keep]
    f_dc = f_dc[keep]
    opac_kept = opac[keep]
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[sel]
        f_dc = f_dc[sel]
        opac_kept = opac_kept[sel]
    rgb = _sh_dc_to_rgb(f_dc)
    _write_xyz_rgb_ply(ply_out, xyz, rgb)
    return {
        "n_total": int(fields["x"].shape[0]),
        "n_kept": int(xyz.shape[0]),
        "opacity_threshold": float(opacity_threshold),
        "opacity_mean": float(opac_kept.mean()) if xyz.shape[0] else 0.0,
        "out_path": str(ply_out),
    }


def write_sam3_video_manifest(persp_dir: Path) -> list[dict]:
    """For every pose subdir, write a frames.txt listing the RGB frames in
    zigzag order (frame_000.., frame_047..) suitable for SAM3 video input."""
    out: list[dict] = []
    for pose_subdir in sorted(persp_dir.glob("pose_*")):
        if not pose_subdir.is_dir():
            continue
        rgb_dir = pose_subdir / "rgb"
        if not rgb_dir.is_dir():
            continue
        frames = sorted(rgb_dir.glob("frame_*.png"))
        if not frames:
            continue
        manifest = pose_subdir / "frames.txt"
        manifest.write_text(
            "\n".join(str(p.relative_to(pose_subdir).as_posix()) for p in frames) + "\n",
            encoding="utf-8",
        )
        out.append({
            "pose": pose_subdir.name,
            "n_frames": len(frames),
            "manifest": str(manifest.relative_to(persp_dir.parent).as_posix()),
        })
    return out


def _read_xyz_ply_header_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        chunk = f.read(8192)
    info, _, _ = _parse_ply_header(chunk)
    return int(info["n_vertex"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="")
    p.add_argument("--colmap-dir", default="colmap_4x")
    p.add_argument("--output-dir", default="gs_4x")
    p.add_argument("--persp-dir", default="perspective_4x")
    p.add_argument("--init-pcd-name", default="init_pcd_4x.ply")
    p.add_argument("--opacity-threshold", type=float, default=0.005)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_id:
        run_dir = resolve_run_dir(cfg, args.run_id)
    else:
        run_dir = latest_run_dir(cfg)
        if run_dir is None:
            raise SystemExit("no runs found")
    print(f"[finalize] run_dir = {run_dir}", flush=True)

    output_dir = run_dir / args.output_dir
    colmap_dir = run_dir / args.colmap_dir
    persp_dir = run_dir / args.persp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. point_cloud.ply (DAP back-projection) ----
    src_pcd = colmap_dir / args.init_pcd_name
    dst_pcd = output_dir / "point_cloud.ply"
    if src_pcd.exists():
        shutil.copy2(src_pcd, dst_pcd)
        n_pcd = _read_xyz_ply_header_count(dst_pcd)
        print(f"[finalize] point_cloud.ply: {dst_pcd}  ({n_pcd} pts)", flush=True)
    else:
        print(f"[finalize] WARNING: source pcd {src_pcd} missing", flush=True)
        n_pcd = 0

    # ---- 2. gs_centers.ply (sampled gaussian centers) ----
    gs_ply = output_dir / "output.ply"
    centers_info: dict | None = None
    if gs_ply.exists():
        centers_path = output_dir / "gs_centers.ply"
        try:
            centers_info = extract_gs_centers(
                gs_ply, centers_path,
                opacity_threshold=float(args.opacity_threshold),
            )
            print(
                f"[finalize] gs_centers.ply: {centers_path}  "
                f"{centers_info['n_kept']} / {centers_info['n_total']} "
                f"(opac>={args.opacity_threshold})",
                flush=True,
            )
        except Exception as exc:
            print(f"[finalize] gs_centers extraction skipped: {exc}", flush=True)
    else:
        print(f"[finalize] WARNING: trained {gs_ply} missing", flush=True)

    # ---- 3. SAM3 frames.txt manifests ----
    manifests = write_sam3_video_manifest(persp_dir)
    print(f"[finalize] wrote {len(manifests)} per-pose frames.txt manifests", flush=True)

    # ---- 4. update meta.json ----
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    n_gs_total = 0
    if gs_ply.exists():
        n_gs_total = _read_xyz_ply_header_count(gs_ply)
    meta["pipeline"] = {
        "sr_module": "RealESRGAN_x4plus + ERP wrap-pad",
        "perspective_scheme": "persp48_zigzag",
        "perspective_size": int(cfg.perspective.out_size),
        "colmap_dir": args.colmap_dir,
        "gs_dir": args.output_dir,
        "perspective_dir": args.persp_dir,
    }
    meta["gs_artifacts"] = {
        "output_ply": f"{args.output_dir}/output.ply",
        "output_ply_n": n_gs_total,
        "point_cloud_ply": f"{args.output_dir}/point_cloud.ply",
        "point_cloud_n": n_pcd,
        "gs_centers_ply": (
            f"{args.output_dir}/gs_centers.ply" if centers_info else None
        ),
        "gs_centers_n": (centers_info["n_kept"] if centers_info else 0),
    }
    meta["sam3_video_manifests"] = manifests
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[finalize] updated {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
