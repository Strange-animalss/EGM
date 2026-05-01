"""Stage 2 driver: take a run's COLMAP output and train a Gaussian Splatting model.

Two backends are supported:

  * splatfacto (default): nerfstudio's `ns-train splatfacto` against the COLMAP
    folder, then `ns-export gaussian-splat` to dump a standard 3DGS PLY.
  * fastgs               : invoke the FastGS repo as a subprocess
    (requires `cfg.fastgs.repo_path` and `git clone https://github.com/fastgs/FastGS`).

Both backends ultimately need to JIT-compile a CUDA rasterizer; on Windows
that requires MSVC's "Desktop development with C++" workload (`cl.exe`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import latest_run_dir, load_config, resolve_run_dir  # noqa: E402


def train(
    config_path: str,
    *,
    run_id: str | None = None,
    allow_fallback: bool = True,
    backend: str = "splatfacto",
    iterations: int | None = None,
    verbose: bool = True,
    colmap_dir: str = "colmap",
    output_dir: str = "gs",
):
    cfg = load_config(config_path)
    if run_id:
        run_dir = resolve_run_dir(cfg, run_id)
    else:
        run_dir = latest_run_dir(cfg)
        if run_dir is None:
            raise SystemExit("no runs found and no --run-id provided")
        run_id = run_dir.name
    if verbose:
        print(
            f"[train_gs] run_dir = {run_dir}  backend = {backend}  "
            f"colmap_dir = {colmap_dir}  output_dir = {output_dir}",
            flush=True,
        )

    backend = (backend or "splatfacto").lower()
    if backend == "splatfacto":
        from recon.run_splatfacto import run_splatfacto
        try:
            result = run_splatfacto(
                cfg, run_dir,
                iterations=iterations,
                verbose=verbose,
                colmap_subdir=colmap_dir,
                output_subdir=output_dir,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            print(
                f"[train_gs] splatfacto failed ({type(exc).__name__}: {exc}); "
                f"falling back to FastGS path",
                flush=True,
            )
            from recon.run_fastgs import run_fastgs
            result = run_fastgs(cfg, run_dir, allow_fallback=True)
    elif backend == "fastgs":
        from recon.run_fastgs import run_fastgs
        result = run_fastgs(cfg, run_dir, allow_fallback=allow_fallback)
    else:
        raise SystemExit(f"unknown backend: {backend!r}; pick splatfacto or fastgs")

    if verbose:
        used_label = "real" if result.used_real_fastgs else "FALLBACK"
        print(
            f"[train_gs] {used_label} backend={backend} -> {result.output_ply}",
            flush=True,
        )

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["gs_backend"] = backend
        meta["gs_used_real"] = bool(result.used_real_fastgs)
        meta["gs_output_ply"] = f"{output_dir}/output.ply"
        meta["gs_colmap_dir"] = colmap_dir
        meta["gs_output_dir"] = output_dir
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="")
    p.add_argument(
        "--backend",
        choices=("splatfacto", "fastgs"),
        default="splatfacto",
        help="GS training backend (default: splatfacto; falls back to FastGS if missing).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override cfg.fastgs.iterations for this run.",
    )
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="error out if the chosen backend is missing or fails",
    )
    p.add_argument(
        "--colmap-dir",
        default="colmap",
        help="run-relative subdir to read COLMAP from (default: colmap; "
             "use colmap_4x for the SR pipeline).",
    )
    p.add_argument(
        "--output-dir",
        default="gs",
        help="run-relative subdir to write training output PLY into "
             "(default: gs; use gs_4x for the SR pipeline).",
    )
    args = p.parse_args()
    train(
        args.config,
        run_id=args.run_id or None,
        backend=args.backend,
        iterations=args.iterations,
        allow_fallback=not args.no_fallback,
        colmap_dir=args.colmap_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
