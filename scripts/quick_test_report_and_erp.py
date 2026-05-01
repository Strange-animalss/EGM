#!/usr/bin/env python3
"""One-shot: API pathway checks + optional ERP PNG via official ``gpt-image-2``.

Reads credentials in order:
  * ``--key-file`` (first non-empty line), else
  * ``openai.api_key`` from merged YAML if **non-empty** (``secrets.local.yaml``), else
  * env ``OPENAI_API_KEY``.

(Local YAML intentionally wins over a stale global env var.)

Writes (always):
  * ``outputs/_quick_test/report.json``
  * ``docs/TEST_REPORT_api_and_erp.md``  (human-readable)

On success also writes:
  * ``outputs/_quick_test/erp_rgb_gpt_image2.png``  (best-effort 2:1 ERP-style panorama)

Usage:
  $env:OPENAI_API_KEY = "sk-..."
  python scripts/quick_test_report_and_erp.py

Or:
  python scripts/quick_test_report_and_erp.py --key-file C:\\Users\\you\\.openai_key.txt

ChatFire 一键测（会强制 ``base_url=https://api.chatfire.cn/v1``，需 **ChatFire 控制台** 的 key）：
  python scripts/quick_test_report_and_erp.py --chatfire
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs" / "_quick_test"
DOCS_REPORT = REPO / "docs" / "TEST_REPORT_api_and_erp.md"
JSON_REPORT = OUT_DIR / "report.json"

ERP_PROMPT = (
    "A single seamless equirectangular 360-degree panorama (2:1 aspect) of an empty "
    "specialty coffee shop interior, photoreal, late morning daylight, no people, "
    "no animals, no text, no watermark, no split panels, no before-after comparison."
)


def _scrub(s: str | None, max_len: int = 400) -> str:
    if not s:
        return ""
    t = str(s)
    t = re.sub(r"sk-(?:or-v1-|proj-)[A-Za-z0-9_-]{20,}", "[REDACTED]", t)
    return t[:max_len] + ("..." if len(t) > max_len else "")


def _resolve_key(args: argparse.Namespace, cfg: Any) -> str:
    kf = str(getattr(args, "key_file", "") or "").strip()
    if kf:
        p = Path(kf).expanduser()
        return p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    yaml_k = str(cfg.openai.get("api_key") or "").strip()
    if yaml_k:
        return yaml_k
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _decode_image(resp) -> bytes:
    d0 = resp.data[0]
    b64 = getattr(d0, "b64_json", None)
    url = getattr(d0, "url", None)
    if b64:
        return base64.b64decode(b64)
    if url:
        import urllib.request

        return urllib.request.urlopen(url, timeout=180).read()
    raise RuntimeError(f"no b64_json or url: {d0!r}")


def _try_images_generate(cli, *, model: str, prompt: str, sizes: list[str]) -> dict[str, Any]:
    last_err = ""
    for size in sizes:
        t0 = time.perf_counter()
        kw: dict[str, Any] = dict(model=model, prompt=prompt, size=size, n=1)
        try:
            try:
                r = cli.images.generate(**kw, quality="medium")
            except TypeError:
                r = cli.images.generate(**kw)
        except Exception as e:
            last_err = _scrub(f"{type(e).__name__}: {e}")
            continue
        try:
            raw = _decode_image(r)
        except Exception as e:
            last_err = _scrub(f"{type(e).__name__}: {e}")
            continue
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return {
            "ok": True,
            "size_requested": size,
            "ms": ms,
            "out_wh": list(im.size),
            "png_bytes": raw,
        }
    return {"ok": False, "error": last_err or "all sizes failed"}


def _images_response_meta(resp: Any) -> dict[str, Any]:
    """Best-effort fields from ``images.generate`` response (SDK / provider dependent)."""
    meta: dict[str, Any] = {"response_type": type(resp).__name__}
    m = getattr(resp, "model", None)
    if m is not None:
        meta["response_model"] = str(m)
    cr = getattr(resp, "created", None)
    if cr is not None:
        meta["created"] = int(cr)
    extra = getattr(resp, "model_extra", None)
    if isinstance(extra, dict):
        for k in ("model", "service_tier"):
            if k in extra and extra[k] is not None and k not in meta:
                meta[f"response_{k}"] = str(extra[k])
    return meta


def _try_images_generate_no_size(
    cli, *, model: str, prompt: str,
) -> dict[str, Any]:
    """Call ``images.generate`` **without** ``size`` (provider default aspect)."""
    t0 = time.perf_counter()
    kw: dict[str, Any] = dict(model=model, prompt=prompt, n=1)
    try:
        try:
            r = cli.images.generate(**kw, quality="medium")
        except TypeError:
            r = cli.images.generate(**kw)
    except Exception as e:
        return {
            "ok": False,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": _scrub(f"{type(e).__name__}: {e}"),
        }
    try:
        raw = _decode_image(r)
    except Exception as e:
        return {
            "ok": False,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": _scrub(f"{type(e).__name__}: {e}"),
            "response_meta": _images_response_meta(r),
        }
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return {
        "ok": True,
        "size_requested": None,
        "ms": ms,
        "out_wh": list(im.size),
        "png_bytes": raw,
        "response_meta": _images_response_meta(r),
        "requested_model": model,
    }


def _chat_ping(cli, *, model: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply exactly: OK"}],
            max_tokens=8,
        )
        txt = (r.choices[0].message.content or "").strip()
        return {
            "ok": True,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "reply": _scrub(txt, 80),
            "model": model,
        }
    except Exception as e:
        return {
            "ok": False,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": _scrub(f"{type(e).__name__}: {e}"),
            "model": model,
        }


def _write_md(payload: dict[str, Any]) -> None:
    meta = payload.get("meta", {})
    lines: list[str] = [
        "# API 与 gpt-image-2 ERP 快速测试报告",
        "",
        f"- **生成时间（UTC）**：{meta.get('utc', '')}",
        f"- **机器**：{meta.get('node', '')}  Python {meta.get('python', '')}",
        f"- **凭据**：{'已解析（可调用）' if meta.get('had_key') else '未找到'}；来源 **`{meta.get('key_source', '')}`**（`env` / `yaml_secrets.local` / `key_file` / `none`）",
        "",
        "## 1. 摘要",
        "",
    ]
    erp = payload.get("erp", {})
    if erp.get("ok"):
        lines.append(
            f"- **ERP 输出**：`{erp.get('path', '')}`  请求尺寸 `{erp.get('size_requested')}`  "
            f"实际 `{erp.get('out_wh')}`  耗时 **{erp.get('ms')} ms**"
        )
    else:
        err = erp.get("error") or "未调用（无密钥或上游失败）"
        lines.append(f"- **ERP 输出**：失败 — {_scrub(str(err))}")
    lines.append("")
    lines.append("## 2. 子项结果")
    lines.append("")
    lines.append("| 步骤 | ok | 说明 |")
    lines.append("|------|----|------|")
    for row in payload.get("rows", []):
        ok = "是" if row.get("ok") else "否"
        note = row.get("note") or row.get("error") or ""
        lines.append(f"| {row.get('name','')} | {ok} | {_scrub(note, 200)} |")
    lines.append("")
    lines.append("## 3. 原始 JSON（节选）")
    lines.append("")
    slim = {k: v for k, v in payload.items() if k != "erp" or not erp.get("png_saved_bytes")}
    lines.append("```json")
    lines.append(json.dumps(slim, indent=2, ensure_ascii=False)[:12000])
    lines.append("```")
    lines.append("")
    lines.append("## 4. 复现命令")
    lines.append("")
    lines.append("```powershell")
    lines.append("$env:OPENAI_API_KEY = \"<你的密钥>\"")
    lines.append("python scripts/quick_test_report_and_erp.py")
    lines.append("```")
    lines.append("")
    lines.append("## 5. 多通路细测（可选）")
    lines.append("")
    lines.append(
        "更细的 `images.generate` / `images.edit` / OpenRouter 等分项，请运行 "
        "`python scripts/test_api_pathways.py --out-json outputs/_api_pathway_test.json`。"
    )
    lines.append("")
    DOCS_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", default="", help="First line = API key (do not commit).")
    ap.add_argument(
        "--chatfire",
        action="store_true",
        help="Force base_url to https://api.chatfire.cn/v1 (needs ChatFire-issued key, not OpenAI sk).",
    )
    ap.add_argument(
        "--erp-no-size",
        action="store_true",
        help="Also call images.generate without `size` (same prompt), save erp_rgb_gpt_image2_no_size.png and log response model metadata.",
    )
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument(
        "--text-model",
        default=os.environ.get("TEST_TEXT_MODEL", "gpt-4o-mini"),
        help="Cheap chat ping model.",
    )
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from erpgen.config import load_config

    cfg = load_config()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = _resolve_key(args, cfg)
    had_key = bool(key)

    key_src = "none"
    if str(getattr(args, "key_file", "") or "").strip():
        key_src = "key_file"
    elif str(cfg.openai.get("api_key") or "").strip():
        key_src = "yaml_secrets.local"
    elif (os.environ.get("OPENAI_API_KEY") or "").strip():
        key_src = "env"

    base_url = str(cfg.openai.get("base_url") or "").strip()
    if args.chatfire:
        base_url = "https://api.chatfire.cn/v1"
    image_model = str(cfg.openai.get("model") or args.image_model)
    text_model = str(cfg.openai.get("text_model") or args.text_model)
    if args.chatfire and text_model == "gpt-4o-mini":
        # ChatFire 文档/惯例常用 gpt-5.5 别名；仍可在 secrets.local.yaml 里覆盖 text_model
        text_model = os.environ.get("CHATFIRE_TEXT_MODEL", "gpt-5.5")

    meta = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "node": platform.node(),
        "python": sys.version.split()[0],
        "had_key": had_key,
        "key_source": key_src,
        "chatfire_cli_flag": bool(args.chatfire),
        "erp_no_size_flag": bool(args.erp_no_size),
        "openai_base_url": base_url or "(sdk default → api.openai.com)",
        "image_model": image_model,
        "text_model_ping": text_model,
    }
    rows: list[dict[str, Any]] = []
    erp: dict[str, Any] = {"ok": False}
    erp_no_size: dict[str, Any] | None = None

    if not had_key:
        rows.append(
            {
                "name": "auth",
                "ok": False,
                "error": (
                    "No credential: set OPENAI_API_KEY, or openai.api_key in "
                    "config/secrets.local.yaml, or --key-file"
                ),
            }
        )
        payload = {"meta": meta, "rows": rows, "erp": erp, "erp_no_size": erp_no_size}
        JSON_REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_md(payload)
        print(f"[quick_test] wrote {JSON_REPORT} and {DOCS_REPORT}", file=sys.stderr)
        return 2

    from openai import OpenAI

    cli_kw: dict[str, Any] = dict(api_key=key, timeout=600.0)
    if base_url:
        cli_kw["base_url"] = base_url
    cli = OpenAI(**cli_kw)

    # ping chat
    ping = _chat_ping(cli, model=text_model)
    rows.append(
        {
            "name": "official_chat_completions",
            "ok": ping["ok"],
            "error": ping.get("error"),
            "note": f"model={ping.get('model')} ms={ping.get('ms')} reply={ping.get('reply')}",
        }
    )

    # ERP image
    sizes_try = ["2048x1024", "1536x1024", "1024x512", "1024x1024"]
    gen = _try_images_generate(cli, model=image_model, prompt=ERP_PROMPT, sizes=sizes_try)
    if gen.get("ok"):
        erp_path = OUT_DIR / "erp_rgb_gpt_image2.png"
        erp_path.write_bytes(gen["png_bytes"])
        erp = {
            "ok": True,
            "path": str(erp_path.relative_to(REPO)).replace("\\", "/"),
            "size_requested": gen["size_requested"],
            "out_wh": gen["out_wh"],
            "ms": gen["ms"],
        }
        rows.append(
            {
                "name": "official_images_generate_erp",
                "ok": True,
                "note": f"{gen['size_requested']} -> {gen['out_wh']} in {gen['ms']} ms",
            }
        )
    else:
        erp = {"ok": False, "error": gen.get("error", "unknown")}
        rows.append(
            {
                "name": "official_images_generate_erp",
                "ok": False,
                "error": erp.get("error"),
            }
        )

    if args.erp_no_size:
        ns = _try_images_generate_no_size(cli, model=image_model, prompt=ERP_PROMPT)
        erp_no_size = {k: v for k, v in ns.items() if k != "png_bytes"}
        if ns.get("ok") and ns.get("png_bytes"):
            npath = OUT_DIR / "erp_rgb_gpt_image2_no_size.png"
            npath.write_bytes(ns["png_bytes"])
            erp_no_size["path"] = str(npath.relative_to(REPO)).replace("\\", "/")
        rm = (ns.get("response_meta") or {}).get("response_model", "?")
        rows.append(
            {
                "name": "images_generate_erp_no_size_param",
                "ok": bool(ns.get("ok")),
                "error": ns.get("error"),
                "note": _scrub(
                    f"requested_model={image_model} response_model={rm} "
                    f"out={ns.get('out_wh')} ms={ns.get('ms')}",
                    240,
                ),
            },
        )

    ok_overall = had_key and ping.get("ok") and erp.get("ok")
    payload: dict[str, Any] = {
        "meta": meta, "rows": rows, "erp": erp, "overall_ok": ok_overall,
    }
    if erp_no_size is not None:
        payload["erp_no_size"] = erp_no_size
    JSON_REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(payload)
    print(f"[quick_test] wrote {JSON_REPORT} and {DOCS_REPORT}", file=sys.stderr)
    return 0 if ok_overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
