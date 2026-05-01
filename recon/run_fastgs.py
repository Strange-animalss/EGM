"""Drive FastGS (https://github.com/fastgs/FastGS) training as a subprocess.

Usage:
    from recon.run_fastgs import run_fastgs
    run_fastgs(cfg, run_dir)

If `cfg.fastgs.repo_path` is missing, this prints a hint and -- when
`allow_fallback=True` -- emits a plain xyz+rgb point cloud PLY built from the
initial point cloud (which Spark.js can still render) so the end-to-end smoke
test still produces a viewable artifact. In production you must install FastGS
to get a real Gaussian Splatting model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Locate FastGS train script
# ---------------------------------------------------------------------------


def _candidate_train_scripts(repo: Path) -> list[Path]:
    return [
        repo / "train.py",
        repo / "fastgs" / "train.py",
        repo / "scripts" / "train.py",
    ]


def find_fastgs_train(repo_path: str | Path) -> Path | None:
    repo = Path(repo_path)
    if not repo.exists():
        return None
    for c in _candidate_train_scripts(repo):
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


@dataclass
class FastGsResult:
    output_ply: Path
    used_real_fastgs: bool
    log_path: Path


def run_fastgs(
    cfg: DictConfig,
    run_dir: Path,
    *,
    allow_fallback: bool = True,
    extra_env: dict[str, str] | None = None,
) -> FastGsResult:
    run_dir = Path(run_dir)
    colmap_dir = run_dir / "colmap"
    gs_dir = run_dir / "gs"
    gs_dir.mkdir(parents=True, exist_ok=True)
    log_path = gs_dir / "train.log"
    out_ply = gs_dir / "output.ply"

    train_script = find_fastgs_train(cfg.fastgs.repo_path)
    if train_script is None:
        msg = (
            f"[run_fastgs] FastGS train script not found under "
            f"'{cfg.fastgs.repo_path}'. Clone with:\n"
            f"  git clone https://github.com/fastgs/FastGS.git --recursive "
            f"{cfg.fastgs.repo_path}\n"
        )
        if not allow_fallback:
            raise RuntimeError(msg)
        print(msg)
        return _fallback_ply(run_dir, out_ply, log_path, reason="fastgs_not_installed")

    cmd = [
        sys.executable,
        str(train_script.resolve()),
        "--source_path", str(colmap_dir.resolve()),
        "--model_path", str(gs_dir.resolve()),
        "--iterations", str(int(cfg.fastgs.iterations)),
    ]
    extra = OmegaConf.to_container(cfg.fastgs.extra_args, resolve=True) or []
    if isinstance(extra, list):
        cmd.extend(str(x) for x in extra)
    print(f"[run_fastgs] running: {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=str(Path(cfg.fastgs.repo_path).resolve()),
            )
        except FileNotFoundError as exc:
            if not allow_fallback:
                raise
            print(f"[run_fastgs] subprocess failed to start: {exc}")
            return _fallback_ply(run_dir, out_ply, log_path, reason=f"start_failed:{exc}")

    if proc.returncode != 0:
        msg = f"[run_fastgs] FastGS exited with code {proc.returncode}; see {log_path}"
        if not allow_fallback:
            raise RuntimeError(msg)
        print(msg)
        return _fallback_ply(run_dir, out_ply, log_path, reason=f"exit_code:{proc.returncode}")

    located = _locate_output_ply(gs_dir)
    if located is None:
        if not allow_fallback:
            raise RuntimeError(f"[run_fastgs] no PLY found under {gs_dir}")
        print(f"[run_fastgs] WARN: no .ply found under {gs_dir}, falling back")
        return _fallback_ply(run_dir, out_ply, log_path, reason="no_ply_found")
    if located.resolve() != out_ply.resolve():
        shutil.copy2(located, out_ply)
    return FastGsResult(output_ply=out_ply, used_real_fastgs=True, log_path=log_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate_output_ply(gs_dir: Path) -> Path | None:
    """Find the highest-iteration point_cloud.ply that FastGS / 3DGS produces."""
    candidates = list(gs_dir.rglob("point_cloud.ply"))
    if not candidates:
        candidates = list(gs_dir.rglob("*.ply"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _fallback_ply(run_dir: Path, out_ply: Path, log_path: Path, *, reason: str) -> FastGsResult:
    init_ply = run_dir / "colmap" / "init_pcd.ply"
    if not init_ply.exists():
        # Try the colmap directory points3D.txt as a last resort -- but
        # it's not a binary PLY. So we just emit an empty PLY and surface
        # the reason in the log.
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        out_ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
                            b"property float x\nproperty float y\nproperty float z\n"
                            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                            b"end_header\n")
        log_path.write_text(f"fallback: empty PLY (reason={reason})\n", encoding="utf-8")
        return FastGsResult(output_ply=out_ply, used_real_fastgs=False, log_path=log_path)
    shutil.copy2(init_ply, out_ply)
    log_path.write_text(f"fallback: copied init point cloud (reason={reason})\n", encoding="utf-8")
    return FastGsResult(output_ply=out_ply, used_real_fastgs=False, log_path=log_path)
