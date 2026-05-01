"""Config loading and run-id helpers."""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = REPO_ROOT / "config" / "default.yaml"
# Local-only overrides (gitignored). Put `openai.api_key` here instead of default.yaml.
SECRETS_LOCAL_PATH = REPO_ROOT / "config" / "secrets.local.yaml"


def load_config(
    config_path: str | os.PathLike | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load YAML config and apply optional dotlist overrides.

    Merge order: ``default.yaml`` < ``secrets.local.yaml`` (if present) < dotlist.

    Example overrides: ['cuboid.size_xyz=[6,6,3]', 'mock.enabled=true'].
    """
    path = Path(config_path) if config_path else DEFAULT_CFG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(str(path))
    if SECRETS_LOCAL_PATH.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(SECRETS_LOCAL_PATH)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg  # type: ignore[return-value]


def make_run_id(prefix: str = "run") -> str:
    """Timestamped run-id, sortable lexicographically."""
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{ts}"


def resolve_run_dir(cfg: DictConfig, run_id: str | None = None) -> Path:
    """Return outputs/runs/<run_id> as an absolute path; create if missing."""
    rid = run_id or str(cfg.run.run_id) or make_run_id()
    base = Path(cfg.run.outputs_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    run_dir = base / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run.run_id = rid
    return run_dir


def latest_run_dir(cfg: DictConfig) -> Path | None:
    """Find the most recently modified run dir under outputs_dir, if any."""
    base = Path(cfg.run.outputs_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def save_resolved_config(cfg: DictConfig, run_dir: Path) -> Path:
    """Dump the fully resolved config alongside the run for reproducibility."""
    out = run_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, str(out))
    return out
