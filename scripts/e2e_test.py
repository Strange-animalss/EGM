"""End-to-end smoke test: generate ERPs, train FastGS, sanity-check artifacts.

Retries up to `--budget` times; each retry uses a different prompt seed.
The first run that produces ERPs and a non-empty PLY passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import load_config, make_run_id  # noqa: E402
from erpgen.sanity import (  # noqa: E402
    check_output_ply,
    check_run_erp_dir,
    write_reports,
)
from scripts.generate_erp import generate  # noqa: E402
from scripts.train_gs import train  # noqa: E402


def _print_report(reports, header: str, *, only_failed: bool = True) -> None:
    print(f"--- {header} ---")
    for r in reports:
        if only_failed and r.passed:
            continue
        status = "OK" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {', '.join(r.reasons) or '-'} {r.metrics}")


def run_once(
    *,
    config_path: str,
    overrides: list[str],
    run_id: str,
    seed: int,
    mock: bool | None,
    allow_fallback: bool,
    verbose: bool,
    scene: str | None = None,
    no_expand: bool = False,
) -> tuple[bool, Path, list]:
    base_overrides = list(overrides)
    if scene is None:
        base_overrides.append(f"prompt.seed={seed}")

    cfg = load_config(config_path, overrides=base_overrides)
    if mock is not None:
        cfg.mock.enabled = bool(mock)

    run_dir = generate(
        config_path,
        overrides=base_overrides,
        run_id=run_id,
        mock=mock,
        verbose=verbose,
        scene=scene,
        no_expand=no_expand,
    )

    erp_pass, erp_reports = check_run_erp_dir(
        run_dir,
        near_m=float(cfg.nvs.depth_near_m),
        far_m=float(cfg.nvs.depth_far_m),
    )
    write_reports(erp_reports, run_dir / "sanity_erp.json")
    _print_report(erp_reports, f"sanity erp ({run_id})")
    if not erp_pass:
        return False, run_dir, erp_reports

    train(
        config_path,
        run_id=run_id,
        allow_fallback=allow_fallback,
        verbose=verbose,
    )

    ply_report = check_output_ply(
        run_dir / "gs" / "output.ply",
        cuboid_size_xyz=list(cfg.cuboid.size_xyz),
        cuboid_center=list(cfg.cuboid.center_world),
        min_vertices=int(1000),
    )
    write_reports([ply_report], run_dir / "sanity_ply.json")
    _print_report([ply_report], f"sanity ply ({run_id})", only_failed=False)
    return ply_report.passed, run_dir, erp_reports + [ply_report]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no-mock", action="store_true")
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="treat FastGS missing or training failure as a hard error",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--scene", type=str, default=None,
                   help="free-form scene description (e.g. 'a cyberpunk bar at midnight')")
    p.add_argument("--no-expand", action="store_true",
                   help="skip GPT-5.5-pro deep reasoning expansion")
    p.add_argument("overrides", nargs="*", help="extra OmegaConf dotlist overrides")
    args = p.parse_args()

    mock: bool | None = None
    if args.mock:
        mock = True
    elif args.no_mock:
        mock = False

    base_seed = int(time.time()) & 0x7FFFFFFF
    last_run: Path | None = None
    for attempt in range(int(args.budget)):
        rid = make_run_id(prefix=f"e2e{attempt:02d}")
        seed = (base_seed + attempt * 7919) & 0x7FFFFFFF
        print(f"\n=== attempt {attempt + 1}/{args.budget} run_id={rid} seed={seed} ===")
        ok, run_dir, _reports = run_once(
            config_path=args.config,
            overrides=list(args.overrides),
            run_id=rid,
            seed=seed,
            mock=mock,
            allow_fallback=not args.no_fallback,
            verbose=not args.quiet,
            scene=args.scene,
            no_expand=args.no_expand,
        )
        last_run = run_dir
        if ok:
            print(f"\nPASS: {run_dir}")
            return 0
        print(f"FAIL on attempt {attempt + 1}; retrying with new seed.")
    print(f"\nALL ATTEMPTS FAILED. last run -> {last_run}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
