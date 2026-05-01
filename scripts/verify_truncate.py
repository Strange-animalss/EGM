"""Quick check that build_prompt fits budget and keeps the ERP constraint."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from erpgen.prompts import SceneSpec, build_prompt

rid = (REPO / "outputs" / "_cafe_v7_run_id.txt").read_text().strip()
prompts_json = REPO / "outputs" / "runs" / rid / "prompts.json"
spec_d = json.loads(prompts_json.read_text(encoding="utf-8"))["scene"]
spec = SceneSpec(**spec_d)
p = build_prompt(spec, kind="rgb", pose_idx=0, total_poses=9)
print(f"len={len(p)}")
print(f"contains ERP constraint?  {'NO PEOPLE' in p}")
print(f"ends with watermark line? {'no watermark' in p}")
print("-" * 60)
print(p)
