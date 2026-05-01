"""Smoke check for the cafe_v8 codepath: imports, config, and a tiny direct
OpenAI hit (1024x1024 quality=low) to verify the key works."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from erpgen.warp import (
    dilate_mask, erp_camera_dirs, forward_warp_erp, hole_mask_to_openai_alpha,
)
from erpgen.nvs import run_hybrid_nvs  # noqa: F401
from erpgen.openai_erp import ImageClient, OpenAIConfig
from erpgen.prompts import SceneSpec, build_prompt

print("imports OK")

cfg = OpenAIConfig.from_dict({
    "provider": "openai", "base_url": "",
    "model": "gpt-image-2", "size": "3840x1920", "rgb_quality": "high",
    "request_timeout_sec": 900,
})
print(
    f"cfg: provider={cfg.provider} base_url={cfg.base_url!r} "
    f"model={cfg.model} size={cfg.size} q={cfg.rgb_quality}"
)
print(f"edit_with_mask exists: {hasattr(ImageClient, 'edit_with_mask')}")
print(f"decode_to_depth  exists: {hasattr(ImageClient, 'decode_to_depth')}")
print(f"decode_to_normal exists: {hasattr(ImageClient, 'decode_to_normal')}")

spec = SceneSpec(
    scene_kind="empty cafe", style="warm wood", light="late morning",
    occupancy="no people", extra_props="",
)
p = build_prompt(spec, kind="rgb", pose_idx=0, total_poses=9)
print(f"sample prompt length: {len(p)}")
print(f"prompt has STATIC SCENE: {'STATIC SCENE' in p}")
print(f"prompt has NO PEOPLE:    {'NO PEOPLE' in p}")

# Tiny smoke against OpenAI direct with the new key (1024x1024 + low quality
# to keep credits bounded). If this 401s or ratelimits, we'll know.
print("\n--- tiny OpenAI direct smoke (1024x1024, quality=low) ---")
key = os.environ.get("OPENAI_API_KEY", "").strip()
print(f"OPENAI_API_KEY set: {bool(key)}, len={len(key)}")
if key:
    from openai import OpenAI

    from erpgen.openai_erp import _decode_image_resp

    cli = OpenAI(api_key=key, timeout=120)
    t0 = time.time()
    try:
        r = cli.images.generate(
            model="gpt-image-2", prompt="a clear blue sky test",
            size="1024x1024", n=1, quality="low",
        )
        img = _decode_image_resp(r)
        print(f"  OK in {time.time() - t0:.1f}s -> {img.size}")
    except Exception as e:
        print(f"  FAIL in {time.time() - t0:.1f}s: {type(e).__name__}: "
              f"{str(e)[:300]}")
