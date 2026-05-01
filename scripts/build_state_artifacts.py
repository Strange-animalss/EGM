"""Build STATE.md + STATE.h5 for the EGMOR cafe pipeline.

Aggregates per-pose ERP geometry, per-variant config, and per-variant
FastGS .ply stats (or "pending" markers) into a single human-readable
markdown file and a single h5py-readable hierarchical dataset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "outputs" / "runs"

# Variants to include in the STATE artefacts. The "label" is the user-facing
# short name; (run_id, corner_method, pose_R) describe the variant.
HISTORICAL = [
    ("v3_openrouter_squash", "cafe_v3_20260430-121553",
     "1024x512 OpenRouter, BICUBIC squash to 2:1 (pseudo-ERP)",
     "i2i (legacy)", "tilted-or-mixed"),
    ("v5_openrouter_outpaint", "cafe_v5_20260501-031609",
     "OpenRouter chat-image, 3-call left+base+right outpaint stitched to 2:1",
     "outpaint stitch", "tilted"),
    ("v6_shangliu_native_2x1", "cafe_v6_20260501-081138",
     "shangliu.org images.generate, native 2:1, run aborted at pose_0/1",
     "warp_inpaint (legacy)", "tilted"),
]

# v7 was on chatfire @ 2048x1024 quality=medium with truncated prompts; the
# directory for it might still exist but is partial — list as pending.
V7_RUNS = [
    ("v7_chatfire_2x1", "cafe_v7_20260501-123754",
     "chatfire.cn images.generate/edit, native 2:1, quality=medium "
     "(prompt-cap fix), partial run", "warp_inpaint", "tilted"),
]

V8 = [
    # variant_label, run_id, corner_method, pose_R_mode, description
    ("v8a_warp_tilted", "cafe_v8_20260501-130348",
     "warp_inpaint", "tilted",
     "Centre RGB forward-warped to corner pose; images.edit fills holes."),
    ("v8_nowarp_shared", "cafe_v8_nowarp_20260501-132721",
     "i2i", "tilted",
     "Whole-image i2i ref-only; corner ERPs are pose-R-independent."),
    ("v8c_nowarp_tilted", "cafe_v8c_tilted_20260501-185350",
     "i2i (shared with v8_nowarp)", "tilted",
     "Same ERPs as v8_nowarp_shared, COLMAP rebuilt under look-at-centre R."),
    ("v8d_nowarp_level", "cafe_v8d_level_20260501-185350",
     "i2i (shared with v8_nowarp)", "level",
     "Same ERPs as v8_nowarp_shared, COLMAP rebuilt under identity R."),
]


# --- helpers ---------------------------------------------------------------

def _erp_geom(rgb_path: Path) -> dict:
    img = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape
    seam = float(np.sqrt(((img[:, 0] - img[:, -1]) ** 2).mean()))
    pole_top = float(np.mean(img[:8].std(axis=1)))
    pole_bot = float(np.mean(img[-8:].std(axis=1)))
    return {
        "shape": [h, w],
        "seam_rms_left_right": round(seam, 2),
        "pole_top_std": round(pole_top, 2),
        "pole_bot_std": round(pole_bot, 2),
    }


def _gather(label: str, run_id: str, **extra) -> dict:
    run_dir = RUNS / run_id
    rec: dict = {"label": label, "run_id": run_id, "run_dir": str(run_dir),
                 "exists": run_dir.is_dir(), **extra}
    if not run_dir.is_dir():
        rec["status"] = "missing"
        return rec

    rgb_dir = run_dir / "erp" / "rgb"
    poses_metrics = []
    erp_paths = []
    for f in sorted(rgb_dir.glob("pose_*.png")) if rgb_dir.exists() else []:
        m = _erp_geom(f)
        m["pose"] = f.stem
        m["path"] = str(f.relative_to(REPO))
        poses_metrics.append(m)
        erp_paths.append(str(f.relative_to(REPO)))
    rec["poses"] = poses_metrics
    rec["erp_paths"] = erp_paths
    if poses_metrics:
        rec["avg_seam_rms"] = round(
            float(np.mean([p["seam_rms_left_right"] for p in poses_metrics])), 2
        )
        rec["avg_pole_top_std"] = round(
            float(np.mean([p["pole_top_std"] for p in poses_metrics])), 2
        )
        rec["avg_pole_bot_std"] = round(
            float(np.mean([p["pole_bot_std"] for p in poses_metrics])), 2
        )
        rec["num_poses"] = len(poses_metrics)

    # poses.json — parse R + xyz
    poses_json = run_dir / "poses.json"
    if poses_json.exists():
        try:
            pj = json.loads(poses_json.read_text(encoding="utf-8"))
            rec["poses_xyz"] = [p["xyz"] for p in pj["poses"]]
            rec["poses_R"] = [p["R"] for p in pj["poses"]]
        except Exception as e:  # pragma: no cover - defensive
            rec["poses_json_error"] = str(e)

    # depth/normal arrays
    decoded = run_dir / "erp_decoded"
    rec["depth_npy_paths"] = (
        [str(p.relative_to(REPO)) for p in sorted(decoded.glob("*_depth_m.npy"))]
        if decoded.exists() else []
    )
    rec["normal_npy_paths"] = (
        [str(p.relative_to(REPO)) for p in sorted(decoded.glob("*_normal_world.npy"))]
        if decoded.exists() else []
    )

    # FastGS
    ply = run_dir / "gs" / "output.ply"
    train_log = run_dir / "gs" / "train.log"
    if ply.exists() and ply.stat().st_size > 0:
        rec["gs_ply_path"] = str(ply.relative_to(REPO))
        rec["gs_ply_size_bytes"] = int(ply.stat().st_size)
        rec["gs_ply_size_mb"] = round(rec["gs_ply_size_bytes"] / (1024 * 1024), 2)
        # crude gaussian count by parsing PLY header
        try:
            with ply.open("rb") as fh:
                header = []
                while True:
                    line = fh.readline().decode("ascii", errors="ignore")
                    if not line:
                        break
                    header.append(line.strip())
                    if line.strip() == "end_header":
                        break
            for h in header:
                if h.startswith("element vertex "):
                    rec["gs_gaussians"] = int(h.split()[2])
                    break
        except Exception:
            pass
        rec["gs_status"] = (
            "fastgs_fallback_init_pcd"
            if "fallback" in (train_log.read_text(encoding="utf-8")
                              if train_log.exists() else "")
            else "fastgs_completed"
        )
    else:
        rec["gs_status"] = "pending"
        rec["gs_ply_path"] = None
        rec["gs_ply_size_bytes"] = -1
        rec["gs_gaussians"] = -1

    # decoder vs DAP, if present
    dvd = run_dir / "decoder_vs_dap.json"
    if dvd.exists():
        rec["decoder_vs_dap"] = json.loads(dvd.read_text(encoding="utf-8"))

    # meta
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            md = json.loads(meta.read_text(encoding="utf-8"))
            rec["meta_provider"] = md.get("provider")
            rec["meta_model"] = md.get("model")
            rec["meta_size"] = md.get("size")
            rec["meta_quality"] = md.get("rgb_quality")
        except Exception:
            pass

    # init_pcd
    init_pcd = run_dir / "colmap" / "init_pcd.ply"
    if init_pcd.exists():
        rec["init_pcd_path"] = str(init_pcd.relative_to(REPO))

    return rec


def _git_head_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_remote_show(repo: Path, remote: str) -> str:
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%H %s", f"{remote}/main"],
            cwd=str(repo), capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    print("[state] gathering variants...")
    all_records: list[dict] = []
    for label, run_id, prov, corner, pose_mode in HISTORICAL + V7_RUNS:
        all_records.append(
            _gather(label, run_id, provider_note=prov,
                    corner_method=corner, pose_R_mode=pose_mode)
        )
    for label, run_id, corner, pose_mode, desc in V8:
        all_records.append(
            _gather(label, run_id, corner_method=corner, pose_R_mode=pose_mode,
                    description=desc)
        )

    print("[state] writing STATE.md ...")
    md_path = REPO / "STATE.md"
    md_path.write_text(_render_md(all_records), encoding="utf-8")
    print(f"  -> {md_path}  ({md_path.stat().st_size} bytes)")

    print("[state] writing STATE.h5 ...")
    try:
        import h5py
    except ImportError:
        print("  installing h5py...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "h5py"],
            check=True,
        )
        import h5py
    h5_path = REPO / "STATE.h5"
    _write_h5(h5_path, all_records)
    print(f"  -> {h5_path}  ({h5_path.stat().st_size} bytes)")

    print("[state] done.")
    return 0


def _render_md(records: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = _git_head_sha(REPO)
    egmor_main = _git_remote_show(REPO, "egmor")
    lines: list[str] = []

    lines.append(f"# EGMOR Cafe Pipeline — STATE")
    lines.append("")
    lines.append(f"_Generated: {now} • workspace HEAD: `{head[:12]}` • "
                 f"egmor/main: `{(egmor_main or 'unknown')[:80]}`_")
    lines.append("")

    # 1. Project overview
    lines.append("## 1. Project overview")
    lines.append("")
    lines.append("EGMOR turns a short scene description into a 3D Gaussian-Splatting "
                 "scene the user can roam:")
    lines.append("")
    lines.append("```")
    lines.append("user prompt")
    lines.append("    │  SceneExpander (gpt-5.5 / gpt-4o)")
    lines.append("    ▼")
    lines.append("rich SceneSpec")
    lines.append("    │  generate_erp.py")
    lines.append("    ▼")
    lines.append("9 ERP RGB poses (gpt-image-2 @ 2048x1024 quality=high, real 2:1)")
    lines.append("    │  DAP-V2 → metric depth + analytic normals (per pose)")
    lines.append("    │  optional: gpt-image-2 decoder → depth/normal map sanity check")
    lines.append("    ▼")
    lines.append("persp48_zigzag (12 yaw × 4 pitch)  +  forward-projected init_pcd")
    lines.append("    │  COLMAP sparse reconstruction layout")
    lines.append("    ▼")
    lines.append("FastGS (1500 iter on RTX 5070; falls back to init_pcd on sm_120 segfault)")
    lines.append("    ▼")
    lines.append(".ply  →  Spark.js viewer")
    lines.append("```")
    lines.append("")
    lines.append("**Active config (cafe_v8 family)**")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|---|---|")
    lines.append("| provider | OpenAI-compatible (currently chatfire.cn fallback) |")
    lines.append("| image model | `gpt-image-2` |")
    lines.append("| size | `2048x1024` true 2:1 ERP |")
    lines.append("| quality | `high` |")
    lines.append("| text model | `gpt-5.5` |")
    lines.append("| corner method | `warp_inpaint` (default) or `i2i` |")
    lines.append("| pose R | `center` (look-at) / `level` (identity) / `outward` / `random` |")
    lines.append("| persp scheme | `persp48_zigzag` (12 yaw × 4 pitch, SAM3-friendly) |")
    lines.append("")

    # 2. History
    lines.append("## 2. Work history (chronological)")
    lines.append("")
    history_blocks = [
        ("v1–v2 scaffold (~Apr 30)",
         "- Initial pipeline scaffold (`1e65c69`); mock renderer, OpenRouter client, "
         "DAP adapter, persp16/48 splitter, FastGS shim. Mocks dropped in v6+."),
        ("v3 cafe_v3 (Apr 30)",
         "- OpenRouter `openai/gpt-image-2` chat-image, 1024×512 BICUBIC squash from "
         "1024×1024 native — geometric pseudo-ERP (squashed sphere). 9 poses + persp48 "
         "+ FastGS."),
        ("v5 cafe_v5 (May 1, 03:00)",
         "- OpenRouter 3-call outpaint stitch (left + base + right) → real 2:1. "
         "Hacky but works; high seam mismatch at panel boundaries."),
        ("v6 cafe_v6 (May 1, 08:00)",
         "- super.shangliu.org direct `images.generate` — first true native 2:1 ERP. "
         "Pipeline aborted after pose 1 (chatfire-style silent crashes / "
         "no-channel errors)."),
        ("v7 cafe_v7 (May 1, 12:00)",
         "- chatfire.cn `images.generate` + `images.edit` validated as a working "
         "OpenAI-compatible relay for native 2:1 ERPs at quality=medium. Long-prompt "
         "(>4K char) `APIConnectionError` hang fixed by capping prompt at 1800 chars."),
        ("v8 cafe_v8 (May 1, 13:00 –)",
         "- 2×2 ablation (warp_inpaint vs whole-image i2i × tilted vs level pose R) at "
         "2048×1024 quality=high. OpenAI-direct gpt-image-2 was attempted at 3840×1920 "
         "but the project key's organisation isn't verified, so the run fell back to "
         "chatfire 2048×1024. SceneExpander prompt was strengthened with explicit "
         "static-room constraints (no people / animals / motion / vehicles)."),
    ]
    for title, body in history_blocks:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(body)
        lines.append("")

    # 3. Per-variant detail
    lines.append("## 3. Per-variant detail")
    lines.append("")
    for r in records:
        lines.append(f"### {r['label']}")
        lines.append("")
        if not r.get("exists"):
            lines.append(f"- run dir missing: `{r['run_dir']}`")
            lines.append("")
            continue
        lines.append(f"- run_id: `{r['run_id']}`")
        if r.get("description"):
            lines.append(f"- description: {r['description']}")
        if r.get("provider_note"):
            lines.append(f"- provider note: {r['provider_note']}")
        lines.append(f"- corner method: `{r.get('corner_method', '?')}`")
        lines.append(f"- pose R mode: `{r.get('pose_R_mode', '?')}`")
        if r.get("meta_provider"):
            lines.append(f"- recorded meta: provider=`{r.get('meta_provider')}` "
                         f"model=`{r.get('meta_model')}` size=`{r.get('meta_size')}` "
                         f"quality=`{r.get('meta_quality')}`")
        if r.get("num_poses"):
            lines.append(f"- ERP poses: {r['num_poses']}, "
                         f"shape={r['poses'][0]['shape']}, "
                         f"avg seam_rms={r['avg_seam_rms']}, "
                         f"avg pole_top_std={r['avg_pole_top_std']}, "
                         f"avg pole_bot_std={r['avg_pole_bot_std']}")
        if r.get("init_pcd_path"):
            lines.append(f"- init_pcd: `{r['init_pcd_path']}`")
        if r.get("gs_status") == "pending":
            lines.append("- FastGS: **pending** (not trained — see Next steps for command)")
        else:
            lines.append(
                f"- FastGS: {r['gs_status']}, gaussians="
                f"{r.get('gs_gaussians', 'n/a')}, "
                f"ply=`{r.get('gs_ply_path')}` ({r.get('gs_ply_size_mb', 'n/a')} MB)"
            )
        if r.get("decoder_vs_dap"):
            for entry in r["decoder_vs_dap"]:
                lines.append(
                    f"  - decoder pose {entry.get('pose_idx')}: "
                    f"depth_mae={entry.get('depth_mae_unit', 'n/a')} "
                    f"normal_angle_deg_mean={entry.get('normal_angle_deg_mean', 'n/a')}"
                )
        lines.append("")

    # 4. Comparison matrix
    lines.append("## 4. Comparison matrix (cafe_v8 family)")
    lines.append("")
    lines.append("| variant | corner | pose R | seam RMS avg | pole top std | "
                 "pole bot std | GS gaussians | GS ply MB | visual |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in records:
        if not r["label"].startswith("v8"):
            continue
        if not r.get("exists"):
            continue
        gs_g = r.get("gs_gaussians", -1)
        gs_g_s = "pending" if gs_g == -1 else str(gs_g)
        gs_mb = r.get("gs_ply_size_mb", -1)
        gs_mb_s = "pending" if gs_mb == -1 or gs_mb == 0 else f"{gs_mb}"
        visual = ""
        if r["label"] == "v8a_warp_tilted":
            visual = "warp residual artefacts; pose_2 partially collapsed"
        elif r["label"] == "v8c_nowarp_tilted":
            visual = "stronger room-identity drift across poses (no warp constraint)"
        elif r["label"] == "v8d_nowarp_level":
            visual = "as v8c but cleaner ERP poles (identity R, no pitch)"
        elif r["label"] == "v8_nowarp_shared":
            visual = "raw shared ERPs; same pixels as v8c/v8d"
        lines.append(
            f"| {r['label']} | {r.get('corner_method', '?')} | "
            f"{r.get('pose_R_mode', '?')} | {r.get('avg_seam_rms', 'n/a')} | "
            f"{r.get('avg_pole_top_std', 'n/a')} | "
            f"{r.get('avg_pole_bot_std', 'n/a')} | {gs_g_s} | {gs_mb_s} | {visual} |"
        )
    lines.append("")
    # Historical row
    lines.append("Historical reference rows (different pipeline configurations):")
    lines.append("")
    lines.append("| variant | corner | pose R | seam RMS avg | pole top std | "
                 "pole bot std | GS gaussians | GS ply MB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        if r["label"].startswith("v8") or not r.get("exists"):
            continue
        gs_g = r.get("gs_gaussians", -1)
        gs_g_s = "n/a" if gs_g == -1 else str(gs_g)
        gs_mb = r.get("gs_ply_size_mb", -1)
        gs_mb_s = "n/a" if gs_mb == -1 or gs_mb == 0 else f"{gs_mb}"
        lines.append(
            f"| {r['label']} | {r.get('corner_method', '?')} | "
            f"{r.get('pose_R_mode', '?')} | {r.get('avg_seam_rms', 'n/a')} | "
            f"{r.get('avg_pole_top_std', 'n/a')} | "
            f"{r.get('avg_pole_bot_std', 'n/a')} | {gs_g_s} | {gs_mb_s} |"
        )
    lines.append("")

    # 5. Findings
    lines.append("## 5. Key findings")
    lines.append("")
    lines.append(
        "- **Native 2:1 ERP from gpt-image-2**: confirmed at 2048×1024 native via "
        "chatfire.cn (probe 2026-05-01). OpenAI direct caps unverified orgs at "
        "`gpt-image-1` / `gpt-image-1.5` (1536×1024, 3:2) — no real ERP without "
        "verification. OpenRouter chat-image API is locked to 1024×1024 regardless "
        "of size hints.\n"
        "- **warp+inpaint vs whole-image i2i**: warp_inpaint constrains the "
        "geometry strongly (visible-pixel preservation), but the model occasionally "
        "fails to seam-blend the masked / unmasked boundary at high res — pose_2 in "
        "v8a partially collapsed. Whole-image i2i drifts in room identity but never "
        "produces this kind of structural failure; per-pose ERPs look more "
        "individually plausible at the cost of cross-pose consistency.\n"
        "- **tilted vs level pose R**: image content is identical between v8c and "
        "v8d (corner ERPs are pose-R-independent under whole-image i2i). The only "
        "differences are the persp48 split + COLMAP camera matrices. `level` keeps "
        "ERP poles aligned with world up/down — cleaner pole compression, no extra "
        "tilt stitching at perspective face boundaries.\n"
        "- **gpt-image-2 as decoder for depth/normal**: per the v8a / v8_nowarp "
        "decoder_vs_dap.json, depth MAE in [0,1] = ~0.12–0.20 (some signal but "
        "weak), and normal-angle mean is ~74°–90° (essentially random vs analytic). "
        "Conclusion: gpt-image-2 cannot be trusted for geometry; DAP-V2 + analytic "
        "normals stay as the production source.\n"
        "- **FastGS on RTX 5070 (sm_120)**: training segfaults beyond ~250k "
        "Gaussians. v8a fell back to the densified init point cloud as `gs/output.ply`. "
        "v8c / v8d FastGS not yet attempted in this snapshot."
    )
    lines.append("")

    # 6. Provider matrix
    lines.append("## 6. Provider quick reference")
    lines.append("")
    lines.append("| provider | base_url | gpt-image-2 status | max ERP | edits |")
    lines.append("|---|---|---|---|---|")
    lines.append("| OpenAI direct | `https://api.openai.com/v1` | ❌ requires org "
                 "verification | n/a | n/a |")
    lines.append("| OpenAI direct (gpt-image-1.5) | same | ✅ no verification, "
                 "but capped | 1536×1024 (3:2) | ✅ |")
    lines.append("| chatfire.cn | `https://api.chatfire.cn/v1` | ✅ bare model id "
                 "routes via `default` group | 2048×1024 native | ✅ |")
    lines.append("| super.shangliu.org | `https://super.shangliu.org/v1` | ⚠️ "
                 "intermittent silent crash on `images.edit` | 2048×1024 | partial |")
    lines.append("| OpenRouter | `https://openrouter.ai/api/v1` | locked to "
                 "1024×1024 chat-image, no `/v1/images/edits` | 1024×1024 | ❌ |")
    lines.append("")
    lines.append("Switch providers by editing the `openai:` block in "
                 "`config/default.yaml`. Examples are kept commented at the top "
                 "of that file.")
    lines.append("")

    # 7. Asset paths
    lines.append("## 7. Asset paths (cafe_v8 family)")
    lines.append("")
    for r in records:
        if not r["label"].startswith("v8") or not r.get("exists"):
            continue
        lines.append(f"### {r['label']}")
        lines.append("")
        lines.append(f"- run dir: `{Path(r['run_dir']).relative_to(REPO).as_posix()}`")
        lines.append(f"- ERP RGB ({r.get('num_poses', 0)}× 2048×1024):")
        for p in r.get("erp_paths", [])[:3]:
            lines.append(f"  - `{p.replace(os.sep, '/')}`")
        if len(r.get("erp_paths", [])) > 3:
            lines.append(f"  - ... ({len(r['erp_paths']) - 3} more)")
        lines.append(f"- DAP depth (.npy): `outputs/runs/{r['run_id']}/erp_decoded/pose_*_depth_m.npy`")
        lines.append(f"- analytic normals (.npy): `outputs/runs/{r['run_id']}/erp_decoded/pose_*_normal_world.npy`")
        if r.get("init_pcd_path"):
            lines.append(f"- init_pcd: `{r['init_pcd_path'].replace(os.sep, '/')}`")
        lines.append(f"- COLMAP: `outputs/runs/{r['run_id']}/colmap/`")
        lines.append(f"- persp48: `outputs/runs/{r['run_id']}/perspective/pose_*/`")
        if r.get("gs_ply_path"):
            lines.append(f"- FastGS PLY: `{r['gs_ply_path'].replace(os.sep, '/')}` "
                         f"({r.get('gs_ply_size_mb', 0)} MB)")
        lines.append("")

    # 8. Next steps
    lines.append("## 8. Next steps / pending")
    lines.append("")
    lines.append(
        "- **FastGS training pending** for `cafe_v8_nowarp_20260501-132721`, "
        "`cafe_v8c_tilted_20260501-185350`, and `cafe_v8d_level_20260501-185350`. "
        "Run sequentially:\n"
        "  ```\n"
        "  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8_nowarp_20260501-132721\n"
        "  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8c_tilted_20260501-185350\n"
        "  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8d_level_20260501-185350\n"
        "  ```\n"
        "  Each is ~46 s at 1500 iter on RTX 5070; expect a sm_120 segfault past "
        "~250k Gaussians, in which case `output.ply` falls back to the densified "
        "init point cloud.\n"
        "- **OpenAI-direct gpt-image-2 at 3840×1920** is still blocked by "
        "organisation verification. Once the project owner verifies, switch the "
        "config to `base_url: \"\"` + `model: gpt-image-2` + `size: 3840x1920` "
        "and rerun cafe_v9. The per-pose API time will jump from ~110 s to "
        "5–10 min and quality should be markedly higher.\n"
        "- **decoder vs DAP**: the experiment showed gpt-image-2 cannot produce "
        "geometry. Either drop the decoder branch from the production pipeline, "
        "or repurpose it as a sanity-check viewer (DAP overlay vs LLM overlay).\n"
        "- **Cross-variant FastGS metrics**: once trained, recompute "
        "`outputs/runs/cafe_v8_compare.json` via `scripts/compare_v8_variants.py` "
        "and refresh STATE.md / STATE.h5.\n"
        "- **FastGS sm_120 segfault**: try the `--use_global_gaussian_pool false` "
        "flag or upgrade to a CUDA toolkit / FastGS branch with proper Blackwell "
        "support; current workaround is the init-pcd fallback."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_h5(path: Path, records: list[dict]) -> None:
    import h5py
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["project"] = "EGMOR"
        meta.attrs["generated_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        meta.attrs["egmor_commit_local"] = _git_head_sha(REPO)
        meta.attrs["egmor_remote_main"] = _git_remote_show(REPO, "egmor")
        meta.attrs["pipeline_version"] = "v8"
        meta.attrs["provider"] = "openai-compatible (chatfire.cn)"
        meta.attrs["model"] = "gpt-image-2"
        meta.attrs["size"] = "2048x1024"
        meta.attrs["quality"] = "high"

        variants = f.create_group("variants")
        for r in records:
            if not r.get("exists"):
                continue
            grp = variants.create_group(r["label"])
            grp.attrs["run_id"] = r["run_id"]
            grp.attrs["run_dir"] = r["run_dir"]
            grp.attrs["corner_method"] = r.get("corner_method", "?") or "?"
            grp.attrs["pose_R_mode"] = r.get("pose_R_mode", "?") or "?"
            if r.get("description"):
                grp.attrs["description"] = r["description"]

            erp_paths = r.get("erp_paths", [])
            if erp_paths:
                grp.create_dataset(
                    "erp_paths",
                    data=np.array(erp_paths, dtype=h5py.string_dtype()),
                )
            poses = r.get("poses", [])
            if poses:
                seam = np.array([p["seam_rms_left_right"] for p in poses],
                                dtype=np.float32)
                pole_t = np.array([p["pole_top_std"] for p in poses],
                                  dtype=np.float32)
                pole_b = np.array([p["pole_bot_std"] for p in poses],
                                  dtype=np.float32)
                grp.create_dataset("seam_rms", data=seam)
                grp.create_dataset("pole_top_std", data=pole_t)
                grp.create_dataset("pole_bot_std", data=pole_b)
            if r.get("poses_xyz"):
                grp.create_dataset(
                    "poses_xyz",
                    data=np.asarray(r["poses_xyz"], dtype=np.float64),
                )
            if r.get("poses_R"):
                grp.create_dataset(
                    "poses_R",
                    data=np.asarray(r["poses_R"], dtype=np.float64),
                )

            grp.attrs["gs_status"] = r.get("gs_status", "pending")
            grp.attrs["gs_gaussians"] = int(r.get("gs_gaussians", -1))
            grp.attrs["gs_ply_size_bytes"] = int(r.get("gs_ply_size_bytes", -1))
            grp.attrs["gs_ply_path"] = r.get("gs_ply_path") or "pending"

            if r.get("decoder_vs_dap"):
                d_grp = grp.create_group("decoder_vs_dap")
                for entry in r["decoder_vs_dap"]:
                    sub = d_grp.create_group(f"pose_{entry.get('pose_idx', 'x')}")
                    sub.attrs["depth_mae_unit"] = float(
                        entry.get("depth_mae_unit", -1) or -1
                    )
                    sub.attrs["normal_angle_deg_mean"] = float(
                        entry.get("normal_angle_deg_mean", -1) or -1
                    )

        # comparison group: rankings on what we have
        comp = f.create_group("comparison")
        v8_records = [r for r in records
                      if r["label"].startswith("v8") and r.get("exists")
                      and r.get("avg_seam_rms") is not None]
        ranking_seam = sorted(v8_records, key=lambda r: r["avg_seam_rms"])
        comp.create_dataset(
            "ranking_by_seam_rms",
            data=np.array([r["label"] for r in ranking_seam],
                          dtype=h5py.string_dtype()),
        )
        ranking_pole = sorted(v8_records,
                              key=lambda r: r["avg_pole_top_std"] + r["avg_pole_bot_std"])
        comp.create_dataset(
            "ranking_by_pole_combined",
            data=np.array([r["label"] for r in ranking_pole],
                          dtype=h5py.string_dtype()),
        )
        if ranking_seam:
            comp.attrs["winner_seam"] = ranking_seam[0]["label"]
        if ranking_pole:
            comp.attrs["winner_pole"] = ranking_pole[0]["label"]


if __name__ == "__main__":
    sys.exit(main())
