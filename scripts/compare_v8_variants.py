"""Aggregate per-variant metrics for the cafe_v8 2x2 comparison
(warp / no-warp × tilted / level)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def _erp_metrics(rgb_path: Path) -> dict:
    img = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape
    seam = float(np.sqrt(((img[:, 0] - img[:, -1]) ** 2).mean()))
    pole_top = float(np.mean(img[:8].std(axis=1)))
    pole_bot = float(np.mean(img[-8:].std(axis=1)))
    return {
        "size": [w, h],
        "seam_rms_left_right": round(seam, 2),
        "pole_top_std": round(pole_top, 2),
        "pole_bot_std": round(pole_bot, 2),
    }


def _gather_run(name: str, run_dir: Path) -> dict:
    out: dict = {"variant": name, "run_id": run_dir.name, "run_dir": str(run_dir)}
    if not run_dir.exists():
        out["error"] = "run dir missing"
        return out

    rgb_dir = run_dir / "erp" / "rgb"
    poses_metrics: list[dict] = []
    if rgb_dir.exists():
        for f in sorted(rgb_dir.iterdir()):
            if f.suffix.lower() == ".png":
                m = _erp_metrics(f)
                m["pose"] = f.stem
                poses_metrics.append(m)
    out["per_pose"] = poses_metrics
    if poses_metrics:
        out["avg_seam_rms"] = round(
            float(np.mean([p["seam_rms_left_right"] for p in poses_metrics])), 2
        )
        out["avg_pole_top_std"] = round(
            float(np.mean([p["pole_top_std"] for p in poses_metrics])), 2
        )
        out["avg_pole_bot_std"] = round(
            float(np.mean([p["pole_bot_std"] for p in poses_metrics])), 2
        )

    # FastGS results
    gs_dir = run_dir / "gs_4x"
    ply = gs_dir / "output.ply"
    if ply.exists():
        out["fastgs_ply"] = str(ply)
        out["fastgs_ply_mb"] = round(ply.stat().st_size / (1024 * 1024), 2)
    train_log = gs_dir / "train.log"
    if train_log.exists():
        out["fastgs_train_log"] = str(train_log)

    # init_pcd count
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            md = json.loads(meta.read_text(encoding="utf-8"))
            out["init_pcd_count"] = md.get("init_pcd_count")
            out["model"] = md.get("model")
            out["size"] = md.get("size")
            out["rgb_quality"] = md.get("rgb_quality")
            out["corner_method"] = md.get("corner_method")
        except Exception:
            pass

    # decoder-vs-DAP, if present
    dvd = run_dir / "decoder_vs_dap.json"
    if dvd.exists():
        out["decoder_vs_dap"] = json.loads(dvd.read_text(encoding="utf-8"))

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "runs" / "cafe_v8_compare.json"))
    p.add_argument("variants", nargs="+",
                   help="space-separated VARIANT_NAME=RUN_DIR pairs")
    args = p.parse_args()

    variants: dict[str, str] = {}
    for spec in args.variants:
        if "=" not in spec:
            raise SystemExit(f"bad spec {spec!r} (need NAME=PATH)")
        name, path = spec.split("=", 1)
        variants[name.strip()] = path.strip()

    results: dict = {"variants": {}}
    for name, path in variants.items():
        run_dir = Path(path).resolve()
        results["variants"][name] = _gather_run(name, run_dir)

    # winner-by-metric (lower is better for these geometric scores)
    by_metric: dict[str, str] = {}
    for metric in ("avg_seam_rms", "avg_pole_top_std", "avg_pole_bot_std"):
        best_name: str | None = None
        best_val: float | None = None
        for name, v in results["variants"].items():
            val = v.get(metric)
            if val is None:
                continue
            if best_val is None or val < best_val:
                best_val = float(val)
                best_name = name
        if best_name is not None:
            by_metric[metric] = f"{best_name} ({best_val:.2f})"
    # FastGS: prefer larger PLY (more detail captured) if no loss reported
    best_ply: tuple[str, float] | None = None
    for name, v in results["variants"].items():
        ply_mb = v.get("fastgs_ply_mb")
        if ply_mb is None:
            continue
        if best_ply is None or float(ply_mb) > best_ply[1]:
            best_ply = (name, float(ply_mb))
    if best_ply is not None:
        by_metric["fastgs_ply_mb"] = f"{best_ply[0]} ({best_ply[1]:.2f} MB)"
    results["winner_by_metric"] = by_metric

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps({
        "variants_keys": list(results["variants"].keys()),
        "winner_by_metric": by_metric,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
