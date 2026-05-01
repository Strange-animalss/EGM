"""CLI shim around recon.run_fastgs.run_fastgs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import load_config  # noqa: E402
from recon.run_fastgs import run_fastgs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--run-dir", required=True,
                   help="path to the run directory containing colmap/")
    p.add_argument("--iterations", type=int, default=None,
                   help="override fastgs.iterations")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotlist for any extra overrides")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    if args.iterations is not None:
        cfg.fastgs.iterations = int(args.iterations)
    res = run_fastgs(cfg, Path(args.run_dir).resolve(), allow_fallback=True)
    print(f"[run_fastgs_cli] used_real_fastgs={res.used_real_fastgs}")
    print(f"[run_fastgs_cli] output_ply={res.output_ply}")
    print(f"[run_fastgs_cli] log_path={res.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
