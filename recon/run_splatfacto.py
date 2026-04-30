"""splatfacto (nerfstudio) wrapper.

Drives `ns-train splatfacto` as a subprocess against the run's COLMAP folder
and exports a standard 3DGS PLY to <run_dir>/gs/output.ply.

Why splatfacto: nerfstudio gives us a maintained, Inria-3DGS-compatible
training loop and (importantly) a tested Windows path. We do not write our
own kernels.

Note (Windows-specific): nerfstudio's splatfacto uses gsplat under the hood,
which JIT-compiles a CUDA extension. That requires `cl.exe` (the MSVC
compiler from "Desktop development with C++"). If your VS 2022 install is
missing that workload, splatfacto will error on its first forward pass with
'cannot find cl'.  Open Visual Studio Installer -> Modify -> add
"Desktop development with C++", then retry.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .run_fastgs import FastGsResult


@dataclass
class _ToolPaths:
    ns_train: str | None
    ns_export: str | None


def _find_ns_tools() -> _ToolPaths:
    return _ToolPaths(
        ns_train=shutil.which("ns-train"),
        ns_export=shutil.which("ns-export"),
    )


def _latest_config(model_root: Path) -> Path | None:
    cfgs = list(model_root.rglob("config.yml"))
    if not cfgs:
        return None
    cfgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cfgs[0]


def run_splatfacto(
    cfg,
    run_dir: Path,
    *,
    iterations: int | None = None,
    verbose: bool = True,
    extra_env: dict[str, str] | None = None,
    colmap_subdir: str = "colmap",
    output_subdir: str = "gs",
) -> FastGsResult:
    run_dir = Path(run_dir)
    colmap_dir = run_dir / colmap_subdir
    gs_dir = run_dir / output_subdir
    gs_dir.mkdir(parents=True, exist_ok=True)
    out_ply = gs_dir / "output.ply"
    log_path = gs_dir / "train.log"

    tools = _find_ns_tools()
    if tools.ns_train is None:
        msg = (
            "[run_splatfacto] `ns-train` not found on PATH. Install with:\n"
            "    pip install nerfstudio\n"
            "(Note: splatfacto will then JIT-compile gsplat's CUDA kernel; on Windows "
            "you also need MSVC's 'Desktop development with C++' workload.)"
        )
        log_path.write_text(msg + "\n", encoding="utf-8")
        raise RuntimeError(msg)

    iters = int(iterations if iterations is not None else cfg.fastgs.iterations)

    cmd = [
        tools.ns_train, "splatfacto",
        "--data", str(colmap_dir),
        "--output-dir", str(gs_dir),
        "--max-num-iterations", str(iters),
        "--vis", "tensorboard",
        "--pipeline.model.cull-alpha-thresh", "0.1",
        "colmap",
        "--colmap-path", "sparse/0",
    ]
    if verbose:
        print(f"[run_splatfacto] running: {' '.join(cmd)}", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-train splatfacto exited with code {proc.returncode}; see {log_path}"
        )

    cfg_yml = _latest_config(gs_dir)
    if cfg_yml is None:
        raise RuntimeError(f"no config.yml found under {gs_dir} after training")

    if tools.ns_export is None:
        raise RuntimeError("ns-export not found on PATH; install nerfstudio[full]")

    export_cmd = [
        tools.ns_export, "gaussian-splat",
        "--load-config", str(cfg_yml),
        "--output-dir", str(gs_dir),
    ]
    if verbose:
        print(f"[run_splatfacto] exporting: {' '.join(export_cmd)}", flush=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.run(
            export_cmd, stdout=logf, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ns-export gaussian-splat exited with code {proc.returncode}; see {log_path}"
        )

    candidates = list(gs_dir.rglob("splat.ply")) + list(gs_dir.rglob("point_cloud.ply"))
    if not candidates:
        candidates = [p for p in gs_dir.rglob("*.ply") if "init" not in p.name.lower()]
    if not candidates:
        raise RuntimeError(f"no exported PLY found under {gs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates[0].resolve() != out_ply.resolve():
        shutil.copy2(candidates[0], out_ply)
    return FastGsResult(output_ply=out_ply, used_real_fastgs=True, log_path=log_path)
