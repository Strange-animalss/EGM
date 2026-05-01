"""Auto-restart watchdog for cafe_v6.

After the first successful pose, the ERP outputs hash into the cache by
prompt. SceneExpander is non-deterministic across restarts, so to ensure
cache hits across watchdog restarts we capture the FIRST attempt's
prompts.json and force subsequent restarts to use --no-expand with that
captured scene_kind appended.

Usage: python scripts/watchdog_cafe_v6.py <run_id>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else f"cafe_v6_{time.strftime('%Y%m%d-%H%M%S')}"
RUN_DIR = REPO / "outputs" / "runs" / RUN_ID
RGB_DIR = RUN_DIR / "erp" / "rgb"
PROMPTS_FILE = RUN_DIR / "prompts.json"
LOG = REPO / "outputs" / f"_{RUN_ID}_watchdog.log"

DEFAULT_SCENE = "an empty specialty coffee shop, late morning, no people, no animals"

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def count_done() -> int:
    return len(list(RGB_DIR.glob("pose_*.png"))) if RGB_DIR.exists() else 0


def captured_scene_kind() -> str | None:
    """If a prior attempt captured a SceneSpec, reuse its scene_kind."""
    if not PROMPTS_FILE.exists():
        return None
    try:
        j = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
        return j.get("scene", {}).get("scene_kind") or None
    except Exception:
        return None


MAX_ATTEMPTS = 8
attempt = 0
while attempt < MAX_ATTEMPTS:
    n_done = count_done()
    print(f"\n=== watchdog attempt {attempt + 1}/{MAX_ATTEMPTS}  poses on disk: {n_done}/9  ===")
    if n_done >= 9:
        print("All 9 poses already on disk. Done.")
        break
    attempt += 1

    # Always use --no-expand so prompt text is deterministic across restarts;
    # this is the only way `outputs/.openai_cache/` actually hits on resume.
    # Trade-off: loses the SceneExpander rich SceneSpec, kept simple.
    cmd = [
        sys.executable, "-u", "scripts/generate_erp.py",
        "--no-mock", "--no-expand",
        "--scene", DEFAULT_SCENE,
        "--run-id", RUN_ID,
        "prompt.seed=42",
    ]
    print(f"  using --no-expand for cross-attempt cache stability")
    print(f"  scene: {DEFAULT_SCENE}")

    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    with open(LOG, "ab") as logf:
        logf.write(f"\n\n===== attempt {attempt}/{MAX_ATTEMPTS} t={time.strftime('%H:%M:%S')} =====\n".encode())
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  exited rc={proc.returncode}  elapsed={elapsed:.0f}s  poses now: {count_done()}/9")
    if proc.returncode == 0 and count_done() >= 9:
        print("Pipeline reported success.")
        break
    if count_done() < 9 and attempt < MAX_ATTEMPTS:
        print("Sleeping 10s before retry...")
        time.sleep(10)

n = count_done()
print(f"\n=== watchdog FINAL: {n}/9 poses ===")
sys.exit(0 if n >= 9 else 1)
