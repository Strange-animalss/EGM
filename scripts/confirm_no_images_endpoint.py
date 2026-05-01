"""Confirm OpenRouter has NO /v1/images/generations endpoint."""
import requests
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/default.yaml")
key = str(cfg.openai.api_key).strip()

r = requests.post(
    "https://openrouter.ai/api/v1/images/generations",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/erpgen",
        "X-Title": "EGMOR",
    },
    json={"model": "openai/gpt-5.4-image-2", "prompt": "test", "size": "auto", "n": 1},
    timeout=30,
)
print(f"status: {r.status_code}")
print(f"content-type: {r.headers.get('content-type')}")
print(f"server: {r.headers.get('server')}")
print(f"first 600 chars of body:")
print(r.text[:600])
