"""Export a trained nerfstudio splatfacto checkpoint to a standard
INRIA-3DGS PLY (compatible with SuperSplat, spark.js, brush, etc.).

We bypass `ns-export gaussian-splat` because pymeshlab fails to load on
Windows under PYTHONIOENCODING=utf-8 (UnicodeDecodeError when reading its
plugin descriptors). The actual GS parameters live in the .ckpt and only
need a thin re-pack into the INRIA PLY layout:

  property float x, y, z
  property float nx, ny, nz                 (unused, written as 0)
  property float f_dc_0, f_dc_1, f_dc_2     (SH band 0, RGB)
  property float f_rest_0 .. f_rest_44      (SH bands 1-3, channel-major)
  property float opacity                    (logit space, splatfacto default)
  property float scale_0, scale_1, scale_2  (log space, splatfacto default)
  property float rot_0, rot_1, rot_2, rot_3 (quaternion, w-first)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("out_ply", type=Path)
    args = p.parse_args()

    print(f"[export_splat] loading {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pipe = state["pipeline"]

    means = pipe["_model.gauss_params.means"].numpy().astype(np.float32)
    f_dc = pipe["_model.gauss_params.features_dc"].numpy().astype(np.float32)
    f_rest = pipe["_model.gauss_params.features_rest"].numpy().astype(np.float32)
    opac = pipe["_model.gauss_params.opacities"].numpy().astype(np.float32)
    quats = pipe["_model.gauss_params.quats"].numpy().astype(np.float32)
    scales = pipe["_model.gauss_params.scales"].numpy().astype(np.float32)

    n = means.shape[0]
    print(
        f"[export_splat] N={n}  f_dc={f_dc.shape}  f_rest={f_rest.shape}  "
        f"opac={opac.shape}  quats={quats.shape}  scales={scales.shape}",
        flush=True,
    )

    f_rest_cm = np.transpose(f_rest, (0, 2, 1)).reshape(n, -1).astype(np.float32)
    n_rest = f_rest_cm.shape[1]
    if n_rest != 45:
        print(f"[export_splat] WARN: expected 45 f_rest coefs, got {n_rest}", flush=True)

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    quats = quats / norms

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

    args.out_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_ply, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(
        f"[export_splat] wrote {args.out_ply}  "
        f"({args.out_ply.stat().st_size / 1024 / 1024:.2f} MB)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
