"""Auto-restart watchdog: keeps re-launching generate_erp.py until all 9
ERP RGB files exist for the run dir. Each restart uses the same run_id +
--no-expand so prompts are deterministic and cache hits skip already-done
poses."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "cafe_v5_20260501-031609"
SCENE = "an empty specialty coffee shop, late morning, no people, no animals"
RUN_DIR = REPO / "outputs" / "runs" / RUN_ID
RGB_DIR = RUN_DIR / "erp" / "rgb"
LOG = REPO / "outputs" / f"_{RUN_ID}_watchdog.log"

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def count_done() -> int:
    return len(list(RGB_DIR.glob("pose_*.png"))) if RGB_DIR.exists() else 0


MAX_ATTEMPTS = 8
attempt = 0
while attempt < MAX_ATTEMPTS:
    n_done = count_done()
    print(f"\n=== watchdog attempt {attempt + 1}/{MAX_ATTEMPTS}  poses on disk: {n_done}/9  ===")
    if n_done >= 9:
        print("All 9 poses already on disk. Done.")
        break
    attempt += 1
    cmd = [
        sys.executable, "-u", "scripts/generate_erp.py",
        "--no-mock", "--no-expand",
        "--scene", SCENE,
        "--run-id", RUN_ID,
        "prompt.seed=42",
    ]
    print(f"launching: {' '.join(cmd)}")
    t0 = time.time()
    with open(LOG, "ab") as logf:
        logf.write(f"\n\n===== attempt {attempt}/{MAX_ATTEMPTS} t={time.strftime('%H:%M:%S')} =====\n".encode())
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"attempt {attempt} exited rc={proc.returncode}  elapsed={elapsed:.0f}s  poses now: {count_done()}/9")
    if proc.returncode == 0 and count_done() >= 9:
        print("Pipeline reported success.")
        break
    # Brief cool-down between attempts
    if count_done() < 9 and attempt < MAX_ATTEMPTS:
        print("Sleeping 10s before retry...")
        time.sleep(10)

n = count_done()
print(f"\n=== watchdog FINAL: {n}/9 poses ===")
sys.exit(0 if n >= 9 else 1)
