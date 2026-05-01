"""Quick reader for STATE.h5 to confirm structure / contents."""
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
with h5py.File(REPO / "STATE.h5", "r") as f:
    print("=== /metadata attrs ===")
    for k, v in f["metadata"].attrs.items():
        print(f"  {k} = {v}")
    print()
    print("=== /variants ===")
    for vname in f["variants"]:
        g = f["variants"][vname]
        print(f"  {vname}:")
        for k, v in g.attrs.items():
            print(f"    @ {k} = {v!s:.100}")
        print(f"    datasets: {list(g.keys())}")
        if "seam_rms" in g:
            seam = g["seam_rms"][:]
            print(f"      seam_rms[{len(seam)}]: mean={float(np.mean(seam)):.2f}")
        if "poses_xyz" in g:
            print(f"      poses_xyz shape: {g['poses_xyz'].shape}")
        if "poses_R" in g:
            print(f"      poses_R shape: {g['poses_R'].shape}")
    print()
    print("=== /comparison ===")
    c = f["comparison"]
    for k, v in c.attrs.items():
        print(f"  @ {k} = {v}")
    for d in c:
        print(f"  /{d} = {list(c[d][:])}")
