# EGMOR Cafe Pipeline — STATE

*Generated: 2026-05-01T11:14:54Z • workspace HEAD: `1e65c6994ccf` • egmor/main: `f4e74efbca73274b5ac28f6cbe66b61b0c6bf657 refactor: consolidate API provider path`*

## 1. Project overview

EGMOR turns a short scene description into a 3D Gaussian-Splatting scene the user can roam:

```
user prompt
    │  SceneExpander (gpt-5.5 / gpt-4o)
    ▼
rich SceneSpec
    │  generate_erp.py
    ▼
9 ERP RGB poses (gpt-image-2 @ 2048x1024 quality=high, real 2:1)
    │  DAP-V2 → metric depth + analytic normals (per pose)
    │  optional: gpt-image-2 decoder → depth/normal map sanity check
    ▼
persp48_zigzag (12 yaw × 4 pitch)  +  forward-projected init_pcd
    │  COLMAP sparse reconstruction layout
    ▼
FastGS (1500 iter on RTX 5070; falls back to init_pcd on sm_120 segfault)
    ▼
.ply  →  Spark.js viewer
```

**Active config (cafe_v8 family)**


| key           | value                                                          |
| ------------- | -------------------------------------------------------------- |
| provider      | OpenAI-compatible (currently chatfire.cn fallback)             |
| image model   | `gpt-image-2`                                                  |
| size          | `2048x1024` true 2:1 ERP                                       |
| quality       | `high`                                                         |
| text model    | `gpt-5.5`                                                      |
| corner method | `warp_inpaint` (default) or `i2i`                              |
| pose R        | `center` (look-at) / `level` (identity) / `outward` / `random` |
| persp scheme  | `persp48_zigzag` (12 yaw × 4 pitch, SAM3-friendly)             |


## 2. Work history (chronological)

### v1–v2 scaffold (~Apr 30)

- Initial pipeline scaffold (`1e65c69`); mock renderer, OpenRouter client, DAP adapter, persp16/48 splitter, FastGS shim. Mocks dropped in v6+.

### v3 cafe_v3 (Apr 30)

- OpenRouter `openai/gpt-image-2` chat-image, 1024×512 BICUBIC squash from 1024×1024 native — geometric pseudo-ERP (squashed sphere). 9 poses + persp48 + FastGS.

### v5 cafe_v5 (May 1, 03:00)

- OpenRouter 3-call outpaint stitch (left + base + right) → real 2:1. Hacky but works; high seam mismatch at panel boundaries.

### v6 cafe_v6 (May 1, 08:00)

- super.shangliu.org direct `images.generate` — first true native 2:1 ERP. Pipeline aborted after pose 1 (chatfire-style silent crashes / no-channel errors).

### v7 cafe_v7 (May 1, 12:00)

- chatfire.cn `images.generate` + `images.edit` validated as a working OpenAI-compatible relay for native 2:1 ERPs at quality=medium. Long-prompt (>4K char) `APIConnectionError` hang fixed by capping prompt at 1800 chars.

### v8 cafe_v8 (May 1, 13:00 –)

- 2×2 ablation (warp_inpaint vs whole-image i2i × tilted vs level pose R) at 2048×1024 quality=high. OpenAI-direct gpt-image-2 was attempted at 3840×1920 but the project key's organisation isn't verified, so the run fell back to chatfire 2048×1024. SceneExpander prompt was strengthened with explicit static-room constraints (no people / animals / motion / vehicles).

## 3. Per-variant detail

### v3_openrouter_squash

- run_id: `cafe_v3_20260430-121553`
- provider note: 1024x512 OpenRouter, BICUBIC squash to 2:1 (pseudo-ERP)
- corner method: `i2i (legacy)`
- pose R mode: `tilted-or-mixed`
- ERP poses: 10, shape=[512, 1024], avg seam_rms=33.66, avg pole_top_std=5.19, avg pole_bot_std=8.42
- FastGS: fastgs_fallback_init_pcd, gaussians=155950, ply=`outputs\runs\cafe_v3_20260430-121553\gs\output.ply` (2.23 MB)

### v5_openrouter_outpaint

- run_id: `cafe_v5_20260501-031609`
- provider note: OpenRouter chat-image, 3-call left+base+right outpaint stitched to 2:1
- corner method: `outpaint stitch`
- pose R mode: `tilted`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=47.82, avg pole_top_std=20.43, avg pole_bot_std=10.46
- init_pcd: `outputs\runs\cafe_v5_20260501-031609\colmap\init_pcd.ply`
- FastGS: fastgs_completed, gaussians=238668, ply=`outputs\runs\cafe_v5_20260501-031609\gs\output.ply` (58.27 MB)

### v6_shangliu_native_2x1

- run_id: `cafe_v6_20260501-081138`
- provider note: shangliu.org images.generate, native 2:1, run aborted at pose_0/1
- corner method: `warp_inpaint (legacy)`
- pose R mode: `tilted`
- ERP poses: 2, shape=[1024, 2048], avg seam_rms=8.42, avg pole_top_std=1.83, avg pole_bot_std=4.95
- FastGS: **pending** (not trained — see Next steps for command)

### v7_chatfire_2x1

- run_id: `cafe_v7_20260501-123754`
- provider note: chatfire.cn images.generate/edit, native 2:1, quality=medium (prompt-cap fix), partial run
- corner method: `warp_inpaint`
- pose R mode: `tilted`
- recorded meta: provider=`openai` model=`gpt-image-2` size=`None` quality=`None`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=18.84, avg pole_top_std=12.56, avg pole_bot_std=7.41
- init_pcd: `outputs\runs\cafe_v7_20260501-123754\colmap\init_pcd.ply`
- FastGS: **pending** (not trained — see Next steps for command)

### v8a_warp_tilted

- run_id: `cafe_v8_20260501-130348`
- description: Centre RGB forward-warped to corner pose; images.edit fills holes.
- corner method: `warp_inpaint`
- pose R mode: `tilted`
- recorded meta: provider=`openai` model=`gpt-image-2` size=`2048x1024` quality=`high`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=15.2, avg pole_top_std=8.14, avg pole_bot_std=10.52
- init_pcd: `outputs\runs\cafe_v8_20260501-130348\colmap\init_pcd.ply`
- FastGS: fastgs_fallback_init_pcd, gaussians=200000, ply=`outputs\runs\cafe_v8_20260501-130348\gs\output.ply` (2.86 MB)
  - decoder pose 0: depth_mae=0.1243 normal_angle_deg_mean=85.14
  - decoder pose 4: depth_mae=0.1609 normal_angle_deg_mean=85.55

### v8_nowarp_shared

- run_id: `cafe_v8_nowarp_20260501-132721`
- description: Whole-image i2i ref-only; corner ERPs are pose-R-independent.
- corner method: `i2i`
- pose R mode: `tilted`
- recorded meta: provider=`openai` model=`gpt-image-2` size=`2048x1024` quality=`high`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=27.41, avg pole_top_std=27.57, avg pole_bot_std=12.04
- init_pcd: `outputs\runs\cafe_v8_nowarp_20260501-132721\colmap\init_pcd.ply`
- FastGS: **pending** (not trained — see Next steps for command)
  - decoder pose 0: depth_mae=0.1808 normal_angle_deg_mean=89.79
  - decoder pose 4: depth_mae=0.1994 normal_angle_deg_mean=74.44

### v8c_nowarp_tilted

- run_id: `cafe_v8c_tilted_20260501-185350`
- description: Same ERPs as v8_nowarp_shared, COLMAP rebuilt under look-at-centre R.
- corner method: `i2i (shared with v8_nowarp)`
- pose R mode: `tilted`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=27.41, avg pole_top_std=27.57, avg pole_bot_std=12.04
- init_pcd: `outputs\runs\cafe_v8c_tilted_20260501-185350\colmap\init_pcd.ply`
- FastGS: **pending** (not trained — see Next steps for command)

### v8d_nowarp_level

- run_id: `cafe_v8d_level_20260501-185350`
- description: Same ERPs as v8_nowarp_shared, COLMAP rebuilt under identity R.
- corner method: `i2i (shared with v8_nowarp)`
- pose R mode: `level`
- ERP poses: 9, shape=[1024, 2048], avg seam_rms=27.41, avg pole_top_std=27.57, avg pole_bot_std=12.04
- init_pcd: `outputs\runs\cafe_v8d_level_20260501-185350\colmap\init_pcd.ply`
- FastGS: **pending** (not trained — see Next steps for command)

## 4. Comparison matrix (cafe_v8 family)


| variant           | corner                      | pose R | seam RMS avg | pole top std | pole bot std | GS gaussians | GS ply MB | visual                                                         |
| ----------------- | --------------------------- | ------ | ------------ | ------------ | ------------ | ------------ | --------- | -------------------------------------------------------------- |
| v8a_warp_tilted   | warp_inpaint                | tilted | 15.2         | 8.14         | 10.52        | 200000       | 2.86      | warp residual artefacts; pose_2 partially collapsed            |
| v8_nowarp_shared  | i2i                         | tilted | 27.41        | 27.57        | 12.04        | pending      | pending   | raw shared ERPs; same pixels as v8c/v8d                        |
| v8c_nowarp_tilted | i2i (shared with v8_nowarp) | tilted | 27.41        | 27.57        | 12.04        | pending      | pending   | stronger room-identity drift across poses (no warp constraint) |
| v8d_nowarp_level  | i2i (shared with v8_nowarp) | level  | 27.41        | 27.57        | 12.04        | pending      | pending   | as v8c but cleaner ERP poles (identity R, no pitch)            |


Historical reference rows (different pipeline configurations):


| variant                | corner                | pose R          | seam RMS avg | pole top std | pole bot std | GS gaussians | GS ply MB |
| ---------------------- | --------------------- | --------------- | ------------ | ------------ | ------------ | ------------ | --------- |
| v3_openrouter_squash   | i2i (legacy)          | tilted-or-mixed | 33.66        | 5.19         | 8.42         | 155950       | 2.23      |
| v5_openrouter_outpaint | outpaint stitch       | tilted          | 47.82        | 20.43        | 10.46        | 238668       | 58.27     |
| v6_shangliu_native_2x1 | warp_inpaint (legacy) | tilted          | 8.42         | 1.83         | 4.95         | n/a          | n/a       |
| v7_chatfire_2x1        | warp_inpaint          | tilted          | 18.84        | 12.56        | 7.41         | n/a          | n/a       |


## 5. Key findings

- **Native 2:1 ERP from gpt-image-2**: confirmed at 2048×1024 native via chatfire.cn (probe 2026-05-01). OpenAI direct caps unverified orgs at `gpt-image-1` / `gpt-image-1.5` (1536×1024, 3:2) — no real ERP without verification. OpenRouter chat-image API is locked to 1024×1024 regardless of size hints.
- **warp+inpaint vs whole-image i2i**: warp_inpaint constrains the geometry strongly (visible-pixel preservation), but the model occasionally fails to seam-blend the masked / unmasked boundary at high res — pose_2 in v8a partially collapsed. Whole-image i2i drifts in room identity but never produces this kind of structural failure; per-pose ERPs look more individually plausible at the cost of cross-pose consistency.
- **tilted vs level pose R**: image content is identical between v8c and v8d (corner ERPs are pose-R-independent under whole-image i2i). The only differences are the persp48 split + COLMAP camera matrices. `level` keeps ERP poles aligned with world up/down — cleaner pole compression, no extra tilt stitching at perspective face boundaries.
- **gpt-image-2 as decoder for depth/normal**: per the v8a / v8_nowarp decoder_vs_dap.json, depth MAE in [0,1] = ~0.12–0.20 (some signal but weak), and normal-angle mean is ~74°–90° (essentially random vs analytic). Conclusion: gpt-image-2 cannot be trusted for geometry; DAP-V2 + analytic normals stay as the production source.
- **FastGS on RTX 5070 (sm_120)**: training segfaults beyond ~250k Gaussians. v8a fell back to the densified init point cloud as `gs/output.ply`. v8c / v8d FastGS not yet attempted in this snapshot.

## 6. Provider quick reference


| provider                      | base_url                        | gpt-image-2 status                                    | max ERP          | edits   |
| ----------------------------- | ------------------------------- | ----------------------------------------------------- | ---------------- | ------- |
| OpenAI direct                 | `https://api.openai.com/v1`     | ❌ requires org verification                           | n/a              | n/a     |
| OpenAI direct (gpt-image-1.5) | same                            | ✅ no verification, but capped                         | 1536×1024 (3:2)  | ✅       |
| chatfire.cn                   | `https://api.chatfire.cn/v1`    | ✅ bare model id routes via `default` group            | 2048×1024 native | ✅       |
| super.shangliu.org            | `https://super.shangliu.org/v1` | ⚠️ intermittent silent crash on `images.edit`         | 2048×1024        | partial |
| OpenRouter                    | `https://openrouter.ai/api/v1`  | locked to 1024×1024 chat-image, no `/v1/images/edits` | 1024×1024        | ❌       |


Switch providers by editing the `openai:` block in `config/default.yaml`. Examples are kept commented at the top of that file.

## 7. Asset paths (cafe_v8 family)

### v8a_warp_tilted

- run dir: `outputs/runs/cafe_v8_20260501-130348`
- ERP RGB (9× 2048×1024):
  - `outputs/runs/cafe_v8_20260501-130348/erp/rgb/pose_0.png`
  - `outputs/runs/cafe_v8_20260501-130348/erp/rgb/pose_1.png`
  - `outputs/runs/cafe_v8_20260501-130348/erp/rgb/pose_2.png`
  - ... (6 more)
- DAP depth (.npy): `outputs/runs/cafe_v8_20260501-130348/erp_decoded/pose_*_depth_m.npy`
- analytic normals (.npy): `outputs/runs/cafe_v8_20260501-130348/erp_decoded/pose_*_normal_world.npy`
- init_pcd: `outputs/runs/cafe_v8_20260501-130348/colmap/init_pcd.ply`
- COLMAP: `outputs/runs/cafe_v8_20260501-130348/colmap/`
- persp48: `outputs/runs/cafe_v8_20260501-130348/perspective/pose_*/`
- FastGS PLY: `outputs/runs/cafe_v8_20260501-130348/gs/output.ply` (2.86 MB)

### v8_nowarp_shared

- run dir: `outputs/runs/cafe_v8_nowarp_20260501-132721`
- ERP RGB (9× 2048×1024):
  - `outputs/runs/cafe_v8_nowarp_20260501-132721/erp/rgb/pose_0.png`
  - `outputs/runs/cafe_v8_nowarp_20260501-132721/erp/rgb/pose_1.png`
  - `outputs/runs/cafe_v8_nowarp_20260501-132721/erp/rgb/pose_2.png`
  - ... (6 more)
- DAP depth (.npy): `outputs/runs/cafe_v8_nowarp_20260501-132721/erp_decoded/pose_*_depth_m.npy`
- analytic normals (.npy): `outputs/runs/cafe_v8_nowarp_20260501-132721/erp_decoded/pose_*_normal_world.npy`
- init_pcd: `outputs/runs/cafe_v8_nowarp_20260501-132721/colmap/init_pcd.ply`
- COLMAP: `outputs/runs/cafe_v8_nowarp_20260501-132721/colmap/`
- persp48: `outputs/runs/cafe_v8_nowarp_20260501-132721/perspective/pose_*/`

### v8c_nowarp_tilted

- run dir: `outputs/runs/cafe_v8c_tilted_20260501-185350`
- ERP RGB (9× 2048×1024):
  - `outputs/runs/cafe_v8c_tilted_20260501-185350/erp/rgb/pose_0.png`
  - `outputs/runs/cafe_v8c_tilted_20260501-185350/erp/rgb/pose_1.png`
  - `outputs/runs/cafe_v8c_tilted_20260501-185350/erp/rgb/pose_2.png`
  - ... (6 more)
- DAP depth (.npy): `outputs/runs/cafe_v8c_tilted_20260501-185350/erp_decoded/pose_*_depth_m.npy`
- analytic normals (.npy): `outputs/runs/cafe_v8c_tilted_20260501-185350/erp_decoded/pose_*_normal_world.npy`
- init_pcd: `outputs/runs/cafe_v8c_tilted_20260501-185350/colmap/init_pcd.ply`
- COLMAP: `outputs/runs/cafe_v8c_tilted_20260501-185350/colmap/`
- persp48: `outputs/runs/cafe_v8c_tilted_20260501-185350/perspective/pose_*/`

### v8d_nowarp_level

- run dir: `outputs/runs/cafe_v8d_level_20260501-185350`
- ERP RGB (9× 2048×1024):
  - `outputs/runs/cafe_v8d_level_20260501-185350/erp/rgb/pose_0.png`
  - `outputs/runs/cafe_v8d_level_20260501-185350/erp/rgb/pose_1.png`
  - `outputs/runs/cafe_v8d_level_20260501-185350/erp/rgb/pose_2.png`
  - ... (6 more)
- DAP depth (.npy): `outputs/runs/cafe_v8d_level_20260501-185350/erp_decoded/pose_*_depth_m.npy`
- analytic normals (.npy): `outputs/runs/cafe_v8d_level_20260501-185350/erp_decoded/pose_*_normal_world.npy`
- init_pcd: `outputs/runs/cafe_v8d_level_20260501-185350/colmap/init_pcd.ply`
- COLMAP: `outputs/runs/cafe_v8d_level_20260501-185350/colmap/`
- persp48: `outputs/runs/cafe_v8d_level_20260501-185350/perspective/pose_*/`

## 8. Next steps / pending

- **FastGS training pending** for `cafe_v8_nowarp_20260501-132721`, `cafe_v8c_tilted_20260501-185350`, and `cafe_v8d_level_20260501-185350`. Run sequentially:
  ```
  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8_nowarp_20260501-132721
  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8c_tilted_20260501-185350
  python scripts/run_fastgs_cli.py --run-dir outputs/runs/cafe_v8d_level_20260501-185350
  ```
  Each is ~46 s at 1500 iter on RTX 5070; expect a sm_120 segfault past ~250k Gaussians, in which case `output.ply` falls back to the densified init point cloud.
- **OpenAI-direct gpt-image-2 at 3840×1920** is still blocked by organisation verification. Once the project owner verifies, switch the config to `base_url: ""` + `model: gpt-image-2` + `size: 3840x1920` and rerun cafe_v9. The per-pose API time will jump from ~110 s to 5–10 min and quality should be markedly higher.
- **decoder vs DAP**: the experiment showed gpt-image-2 cannot produce geometry. Either drop the decoder branch from the production pipeline, or repurpose it as a sanity-check viewer (DAP overlay vs LLM overlay).
- **Cross-variant FastGS metrics**: once trained, recompute `outputs/runs/cafe_v8_compare.json` via `scripts/compare_v8_variants.py` and refresh STATE.md / STATE.h5.
- **FastGS sm_120 segfault**: try the `--use_global_gaussian_pool false` flag or upgrade to a CUDA toolkit / FastGS branch with proper Blackwell support; current workaround is the init-pcd fallback.

