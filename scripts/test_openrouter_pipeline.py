"""Verify ImageClient + SceneExpander work end-to-end through OpenRouter."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.config import load_config  # noqa: E402
from erpgen.openai_erp import ImageClient, OpenAIConfig  # noqa: E402
from erpgen.scene_expander import SceneExpander  # noqa: E402


def main() -> int:
    cfg = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    print(f"provider = {cfg.openai.get('provider', 'openai')}")
    print(f"base_url = {cfg.openai.get('base_url', '')}")
    print(f"image model = {cfg.openai.model}")
    print(f"text  model = {cfg.openai.text_model}")
    print()

    print("=== 1. SceneExpander chat test ===")
    expander = SceneExpander(
        model=str(cfg.openai.text_model),
        api_key_env=str(cfg.openai.text_model_api_key_env),
        api_key=str(cfg.openai.get("api_key", "")),
        base_url=str(cfg.openai.get("base_url", "")),
        provider=str(cfg.openai.get("provider", "openai")),
        http_referer=str(cfg.openai.get("http_referer", "https://github.com/erpgen")),
        app_title=str(cfg.openai.get("app_title", "ERPGen")),
        reasoning_effort=str(cfg.openai.reasoning_effort),
        request_timeout_sec=float(cfg.openai.request_timeout_sec),
        verbose=True,
        mock=False,
    )
    print(f"  mock_mode: {expander.mock_mode}")
    spec = expander.expand("a cozy reading nook in a sunlit cabin")
    print(f"  scene_kind: {spec.scene_kind}")
    print(f"  style:      {spec.style[:80]}")
    print(f"  light:      {spec.light[:80]}")

    print()
    print("=== 2. ImageClient generate test ===")
    oa_cfg = OpenAIConfig.from_dict(cfg.openai)
    if not Path(oa_cfg.cache_dir).is_absolute():
        oa_cfg.cache_dir = str(REPO_ROOT / oa_cfg.cache_dir)
    # tiny size for the test only -- override the heavy 3840x1920
    oa_cfg.size = "1024x512"
    client = ImageClient(oa_cfg, mock=False, verbose=True)
    print(f"  mock_mode: {client.mock_mode}")
    print(f"  generating ERP RGB ...")
    img = client.generate_rgb(
        "Equirectangular 360 panorama, photorealistic interior of a cozy reading nook "
        "in a sunlit wooden cabin, warm afternoon light, books and a fireplace."
    )
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "_test_openrouter_rgb.png"
    img.save(out_path)
    print(f"  OK: saved {out_path} (size={img.size}, mode={img.mode})")

    print()
    print("=== 3. ImageClient edit test (mask -> magenta paint) ===")
    from PIL import Image, ImageDraw  # noqa: WPS433

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = img.size
    draw.rectangle([w // 4, h // 3, 3 * w // 4, 2 * h // 3], fill=255)
    edited = client.edit_with_mask(
        "Replace the masked region with a large bookshelf full of leather-bound books.",
        img, mask,
    )
    edit_path = out_dir / "_test_openrouter_edit.png"
    edited.save(edit_path)
    print(f"  OK: saved {edit_path} (size={edited.size}, mode={edited.mode})")

    print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
