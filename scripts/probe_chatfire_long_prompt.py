"""Probe whether chatfire chokes on the actual cafe_v7 long prompt."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import OpenAI

from erpgen.openai_erp import _decode_image_resp
from erpgen.prompts import SceneSpec, build_prompt

# Reuse the exact SceneSpec already saved in cafe_v7 prompts.json
prompts_json = REPO / "outputs" / "runs" / Path(
    (REPO / "outputs" / "_cafe_v7_run_id.txt").read_text().strip()
) / "prompts.json"
if prompts_json.exists():
    spec_d = json.loads(prompts_json.read_text(encoding="utf-8"))["scene"]
else:
    print("(no live cafe_v7 prompts.json; using a hardcoded SceneSpec)")
    spec_d = {
        "scene_kind": "empty specialty pour-over coffee shop",
        "style": "Photorealistic contemporary urban cafe with warm Scandinavian-Japanese design language: white-painted brick walls, polished concrete floor, oak ceiling slats, large floor-to-ceiling windows.",
        "light": "Late morning natural daylight, ~5200K, gentle diffusion. Warm 3000K LED strips and three matte-black pendant lamps over the service counter.",
        "occupancy": "No people and no animals; cafe is clean and quiet.",
        "extra_props": "long white-oak service counter, chrome espresso machine, two coffee grinders, white ceramic cups, glass pastry case, oak shelves with retail coffee bags, round white-oak tables, plywood chairs, leather window bench, jute rug, fiddle-leaf fig in terracotta planter, framed coffee-origin map, brass wall sconces.",
        "seed": 42,
    }
spec = SceneSpec(**spec_d)
prompt = build_prompt(spec, kind="rgb", pose_idx=0, total_poses=9)
print(f"prompt length: {len(prompt)} chars")
print(f"prompt preview: {prompt[:400]}...")

c = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.chatfire.cn/v1",
    timeout=180,
)
t0 = time.time()
try:
    r = c.images.generate(
        model="gpt-image-2", prompt=prompt, size="2048x1024",
        n=1, quality="medium",
    )
    print(f"OK in {time.time() - t0:.1f}s")
    img = _decode_image_resp(r)
    print(f"  decoded -> {img.size}")
    out = REPO / "outputs" / "_chatfire_long_prompt.png"
    img.save(out)
    print(f"  saved -> {out}")
except Exception as e:
    print(f"FAIL in {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:400]}")
