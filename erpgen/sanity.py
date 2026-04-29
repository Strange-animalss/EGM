"""Per-stage sanity checks used by `scripts/e2e_test.py` to decide whether to retry.

Each check returns a `CheckReport` (passed flag + reasons + metrics dict) so
callers can log them and bail out early on the first hard failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

from .decode import decode_depth_png, decode_normal_png


@dataclass
class CheckReport:
    name: str
    passed: bool = False
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }


def _img_stats(arr: np.ndarray) -> dict[str, float]:
    arr = arr.astype(np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


# ---------------------------------------------------------------------------
# ERP triplet
# ---------------------------------------------------------------------------


def check_rgb_erp(img: Image.Image, *, name: str = "rgb") -> CheckReport:
    arr = np.array(img.convert("RGB"))
    rep = CheckReport(name=name)
    s = _img_stats(arr)
    rep.metrics.update(s)
    if s["std"] < 5.0:
        rep.reasons.append(f"flat image (std={s['std']:.2f} < 5.0)")
    if s["mean"] < 4.0:
        rep.reasons.append(f"too dark (mean={s['mean']:.2f} < 4.0)")
    if s["mean"] > 251.0:
        rep.reasons.append(f"too bright (mean={s['mean']:.2f} > 251)")
    h, w = arr.shape[:2]
    if w / max(h, 1) < 1.7 or w / max(h, 1) > 2.3:
        rep.reasons.append(f"unexpected aspect {w}x{h} (need ~2:1)")
    rep.passed = not rep.reasons
    return rep


def check_depth_erp(
    img: Image.Image,
    *,
    near_m: float,
    far_m: float,
    name: str = "depth",
) -> CheckReport:
    depth, meta = decode_depth_png(img, near_m=near_m, far_m=far_m, return_meta=True)
    rep = CheckReport(name=name)
    rep.metrics.update({
        "valid_ratio": float(meta.valid_ratio),
        "min_m": float(meta.range_m[0]),
        "max_m": float(meta.range_m[1]),
        "std_m": float(depth.std()),
    })
    if meta.valid_ratio < 0.2:
        rep.reasons.append(f"too few valid pixels (ratio={meta.valid_ratio:.3f})")
    if depth.std() < 0.1:
        rep.reasons.append(f"depth too flat (std={depth.std():.3f} m)")
    rep.passed = not rep.reasons
    return rep


def check_normal_erp(img: Image.Image, *, name: str = "normal") -> CheckReport:
    n = decode_normal_png(img)
    lengths = np.linalg.norm(n, axis=-1)
    rep = CheckReport(name=name)
    rep.metrics.update({
        "len_mean": float(lengths.mean()),
        "len_std": float(lengths.std()),
        "channel_std_r": float(n[..., 0].std()),
        "channel_std_g": float(n[..., 1].std()),
        "channel_std_b": float(n[..., 2].std()),
    })
    if abs(lengths.mean() - 1.0) > 0.15:
        rep.reasons.append(f"normal length not unit (mean={lengths.mean():.3f})")
    if (n[..., 0].std() + n[..., 1].std() + n[..., 2].std()) < 0.05:
        rep.reasons.append("normal map appears constant")
    rep.passed = not rep.reasons
    return rep


def check_triplet_alignment(
    rgb: Image.Image, depth: Image.Image, normal: Image.Image
) -> CheckReport:
    rep = CheckReport(name="alignment")
    sizes = {rgb.size, depth.size, normal.size}
    rep.metrics["unique_sizes"] = float(len(sizes))
    if len(sizes) != 1:
        rep.reasons.append(f"size mismatch: {[i.size for i in (rgb, depth, normal)]}")
    rep.passed = not rep.reasons
    return rep


def check_run_erp_dir(
    run_dir: Path, *, near_m: float, far_m: float
) -> tuple[bool, List[CheckReport]]:
    run_dir = Path(run_dir)
    erp = run_dir / "erp"
    rgbs = sorted((erp / "rgb").glob("pose_*.png"))
    deps = sorted((erp / "depth").glob("pose_*.png"))
    nrms = sorted((erp / "normal").glob("pose_*.png"))
    if not rgbs:
        return False, [CheckReport(name="erp_dir", passed=False, reasons=["no RGB ERPs"])]
    if not (len(rgbs) == len(deps) == len(nrms)):
        return False, [CheckReport(
            name="erp_dir",
            passed=False,
            reasons=[f"counts mismatch: rgb={len(rgbs)} dep={len(deps)} nrm={len(nrms)}"],
        )]
    reports: List[CheckReport] = []
    all_pass = True
    for rgb_p, dep_p, nrm_p in zip(rgbs, deps, nrms):
        rgb = Image.open(rgb_p)
        dep = Image.open(dep_p)
        nrm = Image.open(nrm_p)
        for r in (
            check_rgb_erp(rgb, name=f"rgb:{rgb_p.stem}"),
            check_depth_erp(dep, near_m=near_m, far_m=far_m, name=f"dep:{dep_p.stem}"),
            check_normal_erp(nrm, name=f"nrm:{nrm_p.stem}"),
            check_triplet_alignment(rgb, dep, nrm),
        ):
            reports.append(r)
            all_pass = all_pass and r.passed
    return all_pass, reports


# ---------------------------------------------------------------------------
# Output PLY
# ---------------------------------------------------------------------------


def _read_ply_vertex_count(ply_path: Path) -> int:
    try:
        with open(ply_path, "rb") as f:
            head = b""
            while b"end_header" not in head:
                chunk = f.read(4096)
                if not chunk:
                    break
                head += chunk
                if len(head) > 64 * 1024:
                    break
        text = head.decode("ascii", errors="ignore")
        for line in text.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[-1])
    except Exception:
        pass
    return 0


def _read_ply_xyz_bbox(ply_path: Path, max_points: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(ply_path))
        v = ply["vertex"].data
        x = np.asarray(v["x"], dtype=np.float32)
        y = np.asarray(v["y"], dtype=np.float32)
        z = np.asarray(v["z"], dtype=np.float32)
        xyz = np.stack([x, y, z], axis=-1)
        if xyz.shape[0] > max_points:
            idx = np.random.default_rng(0).choice(xyz.shape[0], size=max_points, replace=False)
            xyz = xyz[idx]
        return xyz.min(0), xyz.max(0)
    except Exception:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def check_output_ply(
    ply_path: Path,
    *,
    cuboid_size_xyz: Sequence[float],
    cuboid_center: Sequence[float],
    margin_m: float = 1.5,
    min_vertices: int = 1000,
) -> CheckReport:
    rep = CheckReport(name="ply")
    p = Path(ply_path)
    if not p.exists() or p.stat().st_size < 1024:
        rep.reasons.append(f"PLY missing or too small: {p}")
        rep.passed = False
        return rep
    n = _read_ply_vertex_count(p)
    rep.metrics["vertex_count"] = float(n)
    if n < min_vertices:
        rep.reasons.append(f"vertex_count={n} < {min_vertices}")
    bb_min, bb_max = _read_ply_xyz_bbox(p)
    center = np.asarray(cuboid_center, dtype=np.float32)
    half_room = 0.5 * np.asarray(cuboid_size_xyz, dtype=np.float32) + float(margin_m)
    # Hard cap scales with the cuboid: at least 5 m or 2x the cuboid half-extent,
    # whichever is larger. This is a sanity check, not a tight bound.
    hard_cap = float(max(5.0, 2.0 * float(np.max(half_room))))
    out_min = center - half_room
    out_max = center + half_room
    if (bb_max != 0).any():
        rep.metrics.update({
            "bbox_min_x": float(bb_min[0]),
            "bbox_min_y": float(bb_min[1]),
            "bbox_min_z": float(bb_min[2]),
            "bbox_max_x": float(bb_max[0]),
            "bbox_max_y": float(bb_max[1]),
            "bbox_max_z": float(bb_max[2]),
        })
        if (bb_min < out_min - hard_cap).any() or (bb_max > out_max + hard_cap).any():
            rep.reasons.append(
                f"bbox outside expected cuboid +{hard_cap:.1f} m hard cap "
                f"(min={bb_min.tolist()}, max={bb_max.tolist()})"
            )
    rep.passed = not rep.reasons
    return rep


# ---------------------------------------------------------------------------
# Aggregate dump
# ---------------------------------------------------------------------------


def write_reports(reports: Iterable[CheckReport], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in reports]
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
