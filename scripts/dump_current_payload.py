"""Dump the EXACT payload our current pipeline sends to OpenRouter for an
ERP RGB call, so the user can byte-compare with the ChatGPT web dev-tools
network panel.
"""
import json
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/default.yaml")

PROMPT = (
    "Generate a photorealistic 360 degree equirectangular panorama "
    "(2:1 aspect ratio) of an empty specialty coffee shop interior. "
    "Equirectangular projection, full sphere, seamless horizontal wraparound, "
    "no people, no text, no watermark."
)

# Mimic _chat_image_call exactly (see erpgen/openai_erp.py)
target_w, target_h = 1024, 512
size_hint = (
    f"\n\nThe final image must be a 2:1 equirectangular panorama "
    f"(target {target_w}x{target_h})."
)
content = [{"type": "text", "text": PROMPT + size_hint}]
extra_body = {"modalities": ["image", "text"]}

payload = {
    "url": "POST https://openrouter.ai/api/v1/chat/completions",
    "headers": {
        "Authorization": "Bearer sk-or-v1-***REDACTED***",
        "Content-Type": "application/json",
        "HTTP-Referer": str(cfg.openai.get("http_referer", "")),
        "X-Title": str(cfg.openai.get("app_title", "")),
    },
    "body": {
        "model": str(cfg.openai.model),
        "messages": [{"role": "user", "content": content}],
        **extra_body,
    },
}

print("===== CURRENT PIPELINE PAYLOAD (per RGB call) =====")
print(json.dumps(payload, indent=2, ensure_ascii=False))
print("===== END =====")
print()
print("Notable absences (NOT sent):")
print("  - top-level `size` / `image_size`")
print("  - `image_config` / `image_config.aspect_ratio`")
print("  - `tools=[{type:image_generation,...}]`")
print("  - `provider` routing override")
print()
print("And the prompt suffix `(target 1024x512)` IS sent (sabotaging hint).")
