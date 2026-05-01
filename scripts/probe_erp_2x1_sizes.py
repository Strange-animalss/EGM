#!/usr/bin/env python3
"""Probe ChatFire / OpenAI-compatible ``images.generate`` for **2:1 ERP** sizes.

Uses merged config (``default.yaml`` + ``secrets.local.yaml``) like the main app.
Writes under ``outputs/_erp_2x1_probe/<run_ts>/``:

  * ``img_<W>x<H>.png`` for each successful size
  * ``summary.json`` — per-attempt ok / out_wh / ms / error (scrubbed)

Example:
  python scripts/probe_erp_2x1_sizes.py
  python scripts/probe_erp_2x1_sizes.py --max-attempts 8 --qualities medium,high
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

ERP_PROMPT = (
    "A single seamless equirectangular 360-degree panorama, **exactly 2:1 width:height**, "
    "empty specialty coffee shop interior, photoreal, late morning, no people, no text, "
    "no split panels, no watermark."
)


def _scrub(s: str | None, max_len: int = 400) -> str:
    if not s:
        return ""
    t = str(s)
    t = re.sub(r"sk-(?:or-v1-|proj-)[A-Za-z0-9_-]{20,}", "[REDACTED]", t)
    return t[:max_len] + ("..." if len(t) > max_len else "")


def _decode_image(resp) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image

    d0 = resp.data[0]
    b64 = getattr(d0, "b64_json", None)
    url = getattr(d0, "url", None)
    if b64:
        raw = base64.b64decode(b64)
    elif url:
        import urllib.request

        raw = urllib.request.urlopen(url, timeout=180).read()
    else:
        raise RuntimeError(f"no b64_json or url: {d0!r}")
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return raw, im.size


def _default_sizes_2x1() -> list[str]:
    """Width x height, all 2:1; prefer multiples of 16 for OpenAI-style constraints."""
    return [
        "3840x1920",
        "3584x1792",
        "3328x1664",
        "3072x1536",
        "2816x1408",
        "2560x1280",
        "2304x1152",
        "2048x1024",
        "1920x960",
        "1792x896",
        "1664x832",
        "1536x768",
        "1408x704",
        "1280x640",
        "1152x576",
        "1024x512",
        "896x448",
        "768x384",
        "640x320",
        "512x256",
    ]


def _one_generate(
    cli: Any,
    *,
    model: str,
    prompt: str,
    size: str,
    quality: str | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    kw: dict[str, Any] = dict(model=model, prompt=prompt, size=size, n=1)
    try:
        if quality:
            try:
                r = cli.images.generate(**kw, quality=quality)
            except TypeError:
                r = cli.images.generate(**kw)
        else:
            r = cli.images.generate(**kw)
    except Exception as e:
        return {
            "ok": False,
            "size": size,
            "quality": quality,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": _scrub(f"{type(e).__name__}: {e}"),
        }
    try:
        raw, wh = _decode_image(r)
    except Exception as e:
        return {
            "ok": False,
            "size": size,
            "quality": quality,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": _scrub(f"{type(e).__name__}: {e}"),
        }
    w, h = wh
    ratio = (w / h) if h else 0.0
    return {
        "ok": True,
        "size": size,
        "quality": quality,
        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "out_wh": [w, h],
        "aspect_wh": round(ratio, 4),
        "near_2x1": bool(h > 0 and abs(w / h - 2.0) < 0.02),
        "png_bytes": raw,
        "response_type": type(r).__name__,
        "created": getattr(r, "created", None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        default="",
        help="Comma-separated WxH list; default = built-in 2:1 ladder.",
    )
    ap.add_argument(
        "--qualities",
        default="medium,high,",
        help="Comma-separated: medium,high,low,auto or empty last token for no quality kwarg.",
    )
    ap.add_argument("--max-attempts", type=int, default=25, help="Stop after this many (size×quality) tries.")
    ap.add_argument(
        "--no-size-control",
        action="store_true",
        help="Also try one images.generate **without** size (prompt still demands 2:1).",
    )
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from erpgen.config import load_config

    cfg = load_config()
    key = str(cfg.openai.get("api_key") or "").strip()
    if not key:
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        print("No API key: OPENAI_API_KEY or openai.api_key in secrets.local.yaml", file=sys.stderr)
        return 2

    base_url = str(cfg.openai.get("base_url") or "").strip()
    model = str(cfg.openai.get("model") or "gpt-image-2")

    from openai import OpenAI

    cli_kw: dict[str, Any] = dict(api_key=key, timeout=900.0)
    if base_url:
        cli_kw["base_url"] = base_url
    cli = OpenAI(**cli_kw)

    if args.sizes.strip():
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = _default_sizes_2x1()

    qraw = [q.strip() for q in args.qualities.split(",")]
    qualities: list[str | None] = []
    for q in qraw:
        if q == "":
            qualities.append(None)
        else:
            qualities.append(q)
    if not qualities:
        qualities = ["medium", None]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / "outputs" / "_erp_2x1_probe" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    n_try = 0
    for size in sizes:
        for quality in qualities:
            if n_try >= args.max_attempts:
                break
            n_try += 1
            label = f"{size} q={quality!r}"
            print(f"[probe] {n_try} {label} ...", flush=True)
            rec = _one_generate(cli, model=model, prompt=ERP_PROMPT, size=size, quality=quality)
            rec["label"] = label
            if rec.get("ok") and rec.get("png_bytes"):
                fn = f"img_{size.replace('x', '_')}__q_{quality or 'none'}.png".replace(
                    "/", "-"
                )
                (out_dir / fn).write_bytes(rec["png_bytes"])
                rec["saved"] = str((out_dir / fn).relative_to(REPO)).replace("\\", "/")
            rec.pop("png_bytes", None)
            attempts.append(rec)
            if n_try >= args.max_attempts:
                break
        if n_try >= args.max_attempts:
            break

    if args.no_size_control:
        if n_try < args.max_attempts:
            n_try += 1
            print("[probe] no-size control ...", flush=True)
            t0 = time.perf_counter()
            kw = dict(model=model, prompt=ERP_PROMPT, n=1)
            try:
                try:
                    r = cli.images.generate(**kw, quality="medium")
                except TypeError:
                    r = cli.images.generate(**kw)
            except Exception as e:
                attempts.append(
                    {
                        "label": "no_size q=medium",
                        "ok": False,
                        "size": None,
                        "quality": "medium",
                        "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        "error": _scrub(f"{type(e).__name__}: {e}"),
                    }
                )
            else:
                try:
                    raw, wh = _decode_image(r)
                except Exception as e:
                    attempts.append(
                        {
                            "label": "no_size q=medium",
                            "ok": False,
                            "error": _scrub(f"{type(e).__name__}: {e}"),
                        }
                    )
                else:
                    w, h = wh
                    ratio = (w / h) if h else 0.0
                    (out_dir / "img_no_size__q_medium.png").write_bytes(raw)
                    attempts.append(
                        {
                            "label": "no_size q=medium",
                            "ok": True,
                            "size": None,
                            "quality": "medium",
                            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                            "out_wh": [w, h],
                            "aspect_wh": round(ratio, 4),
                            "near_2x1": bool(h > 0 and abs(w / h - 2.0) < 0.02),
                            "saved": str((out_dir / "img_no_size__q_medium.png").relative_to(REPO)).replace(
                                "\\", "/"
                            ),
                        },
                    )

    ok_list = [a for a in attempts if a.get("ok")]
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url or "(official default)",
        "model": model,
        "out_dir": str(out_dir.relative_to(REPO)).replace("\\", "/"),
        "attempts": len(attempts),
        "successes": len(ok_list),
        "near_2x1_success_labels": [
            a.get("label") for a in ok_list if a.get("near_2x1")
        ],
        "results": attempts,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[probe] wrote {out_dir / 'summary.json'}", flush=True)
    return 0 if ok_list else 1


if __name__ == "__main__":
    raise SystemExit(main())
