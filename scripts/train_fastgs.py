"""Drive FastGS-Instance/FastGS/train.py against one of our COLMAP datasets,
then export an INRIA-3DGS PLY (compatible with SuperSplat / spark.js).

Why a wrapper:
  * `train.py` uses cwd-relative imports (`from scene import Scene` etc.), so
    we have to cd into `FastGS/` before launching.
  * It saves a `gaussians_final.pt` (state dict) but no PLY by default;
    after training we repack it into INRIA 3DGS PLY layout so the standard
    viewer pipeline picks it up.

Usage:
  python scripts/train_fastgs.py --colmap-dir colmap_4x_no_pose8 \
      --output-dir gs_4x_fastgs_no_pose8 [--iterations 7000]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "cafe_v3_20260430-121553"
FASTGS_ROOT = REPO_ROOT / "third_party" / "FastGS-Instance" / "FastGS"


def _write_inria_ply(state: dict, out_ply: Path) -> dict:
    """Repack a FastGS gaussians_final.pt into an INRIA 3DGS PLY."""
    means = state["_xyz"].numpy().astype(np.float32)
    f_dc = state["_features_dc"].numpy().astype(np.float32)  # (N, 1, 3)
    f_rest = state["_features_rest"].numpy().astype(np.float32)  # (N, 15, 3)
    opac = state["_opacity"].numpy().astype(np.float32)  # (N, 1)
    quats = state["_rotation"].numpy().astype(np.float32)
    scales = state["_scaling"].numpy().astype(np.float32)

    if f_dc.ndim == 3:
        f_dc = f_dc.reshape(f_dc.shape[0], 3)
    if f_rest.ndim == 3 and f_rest.shape[1] == 15:
        n_rest = 45
        f_rest_cm = np.transpose(f_rest, (0, 2, 1)).reshape(f_rest.shape[0], n_rest)
    else:
        n_rest = int(np.prod(f_rest.shape[1:]))
        f_rest_cm = f_rest.reshape(f_rest.shape[0], n_rest)

    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)

    n = means.shape[0]
    fields: list[tuple[str, str]] = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    for i in range(n_rest):
        fields.append((f"f_rest_{i}", "<f4"))
    fields.append(("opacity", "<f4"))
    fields += [("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")]
    fields += [("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
    dt = np.dtype(fields)

    arr = np.empty(n, dtype=dt)
    arr["x"] = means[:, 0]; arr["y"] = means[:, 1]; arr["z"] = means[:, 2]
    arr["nx"] = 0.0; arr["ny"] = 0.0; arr["nz"] = 0.0
    arr["f_dc_0"] = f_dc[:, 0]; arr["f_dc_1"] = f_dc[:, 1]; arr["f_dc_2"] = f_dc[:, 2]
    for i in range(n_rest):
        arr[f"f_rest_{i}"] = f_rest_cm[:, i]
    arr["opacity"] = opac[:, 0]
    arr["scale_0"] = scales[:, 0]; arr["scale_1"] = scales[:, 1]; arr["scale_2"] = scales[:, 2]
    arr["rot_0"] = quats[:, 0]; arr["rot_1"] = quats[:, 1]; arr["rot_2"] = quats[:, 2]; arr["rot_3"] = quats[:, 3]

    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for name, _ in fields:
        header_lines.append(f"property float {name}")
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ply, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    return {
        "n_gaussians": n,
        "n_rest_coefs": n_rest,
        "out_ply": str(out_ply),
        "size_mb": round(out_ply.stat().st_size / 1024 / 1024, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--colmap-dir", required=True,
                   help="run-relative subdir (e.g. colmap_4x_no_pose8)")
    p.add_argument("--output-dir", required=True,
                   help="run-relative subdir for FastGS output (e.g. gs_4x_fastgs_no_pose8)")
    p.add_argument("--iterations", type=int, default=7000,
                   help="iterations (default 7000; FastGS paper uses 30000)")
    p.add_argument("--densification-interval", type=int, default=500)
    p.add_argument("--no-eval", action="store_true",
                   help="omit --eval flag (use all images for training)")
    args = p.parse_args()

    run_dir = REPO_ROOT / "outputs" / "runs" / RUN_ID
    source_path = run_dir / args.colmap_dir
    model_path = run_dir / args.output_dir
    if not source_path.is_dir():
        raise SystemExit(f"COLMAP source missing: {source_path}")
    model_path.mkdir(parents=True, exist_ok=True)

    train_log = model_path / "train.log"

    cmd = [
        sys.executable, "train.py",
        "-s", str(source_path),
        "-i", "images",
        "-m", str(model_path),
        "--iterations", str(args.iterations),
        "--densification_interval", str(args.densification_interval),
        "--optimizer_type", "default",
    ]
    if not args.no_eval:
        cmd.append("--eval")

    print(f"[train_fastgs] cwd      = {FASTGS_ROOT}", flush=True)
    print(f"[train_fastgs] cmd      = {' '.join(cmd)}", flush=True)
    print(f"[train_fastgs] log      = {train_log}", flush=True)

    t0 = time.time()
    with open(train_log, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(FASTGS_ROOT),
            stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    train_secs = time.time() - t0
    print(f"[train_fastgs] returncode={proc.returncode}  wallclock={train_secs:.1f}s", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(train_log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        print(f"[train_fastgs] ---- last 30 log lines ----\n{tail}", flush=True)
        raise SystemExit(f"FastGS train.py exited with {proc.returncode}; see {train_log}")

    ckpt = model_path / "gaussians_final.pt"
    if not ckpt.exists():
        raise SystemExit(f"expected checkpoint missing: {ckpt}")
    print(f"[train_fastgs] loading {ckpt}...", flush=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    out_ply = model_path / "output.ply"
    info = _write_inria_ply(state, out_ply)
    print(f"[train_fastgs] PLY exported: {info}", flush=True)

    summary = {
        "colmap_dir": args.colmap_dir,
        "output_dir": args.output_dir,
        "iterations": int(args.iterations),
        "training_seconds": round(train_secs, 1),
        "n_gaussians": info["n_gaussians"],
        "ply_size_mb": info["size_mb"],
        "output_ply": str(out_ply.relative_to(run_dir)),
    }
    (model_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train_fastgs] DONE", flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
