"""Stage 2 driver: take a run's COLMAP output and train FastGS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import latest_run_dir, load_config, resolve_run_dir  # noqa: E402
from recon.run_fastgs import run_fastgs  # noqa: E402


def train(
    config_path: str,
    *,
    run_id: str | None = None,
    allow_fallback: bool = True,
    verbose: bool = True,
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
        print(f"[train_gs] run_dir = {run_dir}", flush=True)
    result = run_fastgs(cfg, run_dir, allow_fallback=allow_fallback)
    if verbose:
        print(
            f"[train_gs] {'FastGS' if result.used_real_fastgs else 'FALLBACK'} "
            f"-> {result.output_ply}",
            flush=True,
        )
    # Append to meta.json so the viewer can show whether it was a real run
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["gs_used_real_fastgs"] = bool(result.used_real_fastgs)
        meta["gs_output_ply"] = "gs/output.ply"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-id", default="")
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="error out if FastGS is missing or fails (default: fall back to xyz-rgb PLY)",
    )
    args = p.parse_args()
    train(
        args.config,
        run_id=args.run_id or None,
        allow_fallback=not args.no_fallback,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
