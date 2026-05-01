# EGMOR / EGM — 完整工作逻辑与可配置选项

本文档基于仓库当前 **Python 源码** 整理，用于一次性理解「文本 → 360° 全景 → 几何 → 透视图 → COLMAP → 3D Gaussian Splatting」的全链路，以及各阶段可对比、可切换的选项。

---

## 1. 项目在做什么

**目标**：用自然语言描述一个**静态、空室内**场景，生成可在浏览器或第三方工具中漫游的 **3D Gaussian Splatting** 点云（`.ply`）。

**核心思路**：

1. 用大语言模型把短句扩写成结构化场景说明（便于图像模型一致渲染）。
2. 在长方体房间内布置 **1 个中心 + 8 个角点** 共 **9 个相机位姿**，对每个位姿生成一张 **等距圆柱投影（ERP）** 全景 RGB。
3. 对每个位姿的 RGB，用 **Depth-Anything-V2（DAP-V2）** 估计度量深度，再用深度数值微分得到 **解析法线**（与深度严格同像素对齐）。
4. 将 ERP 上的 RGB / 深度 / 法线 **重投影**为多张透视图，写出 **COLMAP** 稀疏模型与初始点云。
5. 可选：对 ERP RGB 做 **4× 超分**（Real-ESRGAN），深度/法线双线性放大，再重新切透视与 COLMAP。
6. 用 **splatfacto（nerfstudio）** 或 **FastGS** 子进程训练 GS，导出 PLY；本地 viewer 或 SuperSplat 等查看。

---

## FAQ：Pose 与首张图、几何「准确性」、Warp、OpenAI 官方 gpt-image-2

### Pose 的位置是不是从第一张图里估计出来的？

**不是。** 9 个相机的 **世界坐标 `xyz` 与旋转 `R`** 在 `erpgen/poses.py` 里由 `**cuboid`（长方体房间）+ `poses.initial` + `poses.generation`（如 `auto_8_corners`）** 解析算出，在调用图像 API **之前** 就已固定，并写入 `poses.json`。这不是视觉 SLAM / COLMAP 从「第一张全景」反推的外参。

### 既然不是从第一张估计的，后面的位置怎么可能「准确」？

这里要分清两种「准确」：


| 含义                         | 本管线怎么做                                                                                                                    | 局限                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **几何上的相机位姿**               | 人为定义的虚拟布局；透视拆分与 COLMAP 使用的位姿来自该布局 + `erp_to_persp`，**不是**从 LLM 图中估计                                                       | 若 `cuboid` 与真实房间尺度不符，度量深度标定与房间物理尺寸会偏离                           |
| **多视角 RGB 语义一致**（看起来像同一间房） | 角点可用 `**warp_inpaint`**：用 **中心位姿的 DAP 度量深度** 把中心 RGB **前向 warp** 到角点 ERP，再用 mask 让模型只填洞；或用 `**i2i`** / 无参考 `**generate**` | LLM 不保证严格遵守球面几何；大视差时 warp **空洞多**，inpaint 仍会漂；纯 `generate` 最不一致 |


**深度**才是「从当前这张 RGB 估计」的：每张 ERP 各自跑 **DAP-V2**，再线性缩放到 `[depth_near_m, depth_far_m]`。法线由深度解析求导得到，与 **该张** RGB 对齐。中心图的深度 **不** 自动等于「真房间」的激光雷达，而是单目相对深度经分位数映射后的 **工程近似**。

### Warp 在一致性里起什么作用？

`erpgen/warp.py` 的 `forward_warp_erp`：把 **源 ERP 每个像素** 按 **源 pose 的深度** 反投影成世界点，再投到 **目标 pose** 的 ERP 像素栅格，用 **range buffer（近处覆盖远处）** 合成目标 ERP；未覆盖处为 **空洞**，经膨胀后作为 `images.edit` 的 **透明编辑区**。这样角点新图在几何上与 **中心 3D 点云（由中心深度诱导）** 绑定，而不是完全独立的文生图。

### OpenAI 官方能用吗？效果如何？速度如何？

- **能用**：`openai.provider: "openai"` 且 `**base_url` 留空**（默认 `api.openai.com`）或指向 **透传官方 `images` API** 的中继时，走 `images.generate` / `images.edit`；`**warp_inpaint` 必须这条路径**（`edit_with_mask` 在 OpenRouter chat 路径不可用）。
- **效果**：几何对齐靠 DAP+warp；画面质量取决于 `gpt-image-2` 与提示词。README 中 cafe 例子的 GS 指标（如 splatfacto PSNR 量级）仅供参考，随场景变化。
- **速度**：DAP 常为 GPU **亚秒～数秒/张**；`gpt-image-2` 取决于分辨率与 `quality`，官方文档称方图通常较快；中继/高分辨率单次 **分钟级** 也可能出现。

### OpenAI 官方 `gpt-image-2` 一次出图（放入 `docs/`）

仓库提供脚本 `**scripts/docs_sample_gpt_image2_official.py`**：仅用 **官方默认主机**（代码里 `OpenAI(api_key=...)`、**不设 `base_url`**），调用 `images.generate`，模型 `**gpt-image-2**`，生成一张与管线风格接近的室内图，保存为：

- `docs/openai_official_gpt_image2_sample.png`
- `docs/openai_official_gpt_image2_sample.json`（记录模型、`size`、prompt、时间戳等）

**在仓库根目录执行（PowerShell 示例）：**

```powershell
$env:OPENAI_API_KEY = "sk-..."   # 你的官方密钥
python scripts/docs_sample_gpt_image2_official.py
```

> **自动化环境说明**：在无 `OPENAI_API_KEY` 的 CI/代理环境里运行脚本只会写入 `openai_official_gpt_image2_sample.json` 且 `"ok": false`，**不会**生成 PNG。你在本机成功运行后，PNG 会出现在 `docs/`，可把 PNG 与更新后的 JSON 一并提交。

**当前元数据**：以仓库内 `docs/openai_official_gpt_image2_sample.json` 为准（成功时含 `model`、`size`、`prompt`、`image_size_px` 等）。

**生成成功后的示例图（本地运行脚本后即存在）：**

OpenAI 官方 gpt-image-2 一次出图（运行 scripts/docs_sample_gpt_image2_official.py 后生成）

---

## 2. 端到端流水线（与脚本对应）

```
用户场景文本 (--scene / config.prompt.scene_default)
        │
        ▼
┌───────────────────────────────────────┐
│ SceneExpander（可选，--no-expand 跳过）│  ← openai 兼容 chat，text_model
│ 输出 JSON → SceneSpec                  │
└─────────────────┬─────────────────────┘
                  ▼
┌───────────────────────────────────────┐
│ build_pose_set                        │  ← cuboid + poses.generation
│ 9 个 Pose（world xyz + R）             │
└─────────────────┬─────────────────────┘
                  ▼
┌───────────────────────────────────────┐
│ run_hybrid_nvs（erpgen/nvs.py）        │  ← ImageClient + DAP
│ 每位姿：LLM RGB → DAP depth → 解析法线 │
└─────────────────┬─────────────────────┘
                  ▼
┌───────────────────────────────────────┐
│ split_all_to_perspectives             │  ← perspective.scheme, fov, out_size
│ 写 perspective/ + cameras.json       │
└─────────────────┬─────────────────────┘
                  ▼
┌───────────────────────────────────────┐
│ build_init_pcd + write_colmap_sparse  │  ← fastgs.init_* 影响点云密度
│ colmap/init_pcd.ply + sparse/         │
└─────────────────┬─────────────────────┘
                  ▼
        （可选）scripts/sr_erp_4x.py
        （可选）scripts/regenerate_persp_4x.py
                  ▼
        scripts/train_gs.py → gs/ 或 gs_4x/
                  ▼
        scripts/serve_viewer.py
```

**主入口**：`scripts/generate_erp.py` 的 `generate()` 完成从配置加载到 COLMAP 写入的 **Stage 1**。

---

## 3. Stage 1 详解：`generate_erp.py`

### 3.1 配置与运行目录

- **加载**：`erpgen/config.py` → `OmegaConf.load` + dotlist 覆盖。
- **运行目录**：`cfg.run.outputs_dir`（默认 `outputs/runs`）+ `run_id`；写入 `resolved_config.yaml`。

### 3.2 位姿 `poses.py`

- **世界系**：右手，**Z 向上**；相机局部 **+X 为视线前方**，`R` 为 **world_from_camera**。
- **ERP 像素 ↔ 射线**：与 `poses.py` 文件头注释一致（中心列为 yaw=0 对应 +X 等）。
- **长方体**：`cuboid.size_xyz`、`cuboid.center_world`、`cuboid.corner_inset`（角点向中心收缩比例）。
- `**cuboid.corner_lookat`**（角点相机朝向）：

  | 取值        | 含义                                           |
  | --------- | -------------------------------------------- |
  | `center`  | 角点相机朝房间中心（默认）                                |
  | `outward` | 朝外（从中心指向角点方向）                                |
  | `random`  | 有界随机 yaw/pitch（可复现性依赖 `prompt.seed` 传入的 RNG） |
  | `level`   | 与中心姿态相同旋转（单位阵），极点与世界上下一致，利于全景模型/warp         |

- `**poses.generation**`：`"auto_8_corners"` 或 **显式 pose 列表**（每项 `xyz_local` + `euler_xyz_deg` 或 `lookat`）。

### 3.3 场景文本 `scene_expander.py` + `prompts.py`

- **SceneExpander**：对 `openai` 兼容端点调用 **chat completions**，要求返回固定 JSON 键：`scene_kind`, `style`, `light`, `occupancy`, `extra_props`。
- `**--no-expand`**：跳过扩展，用 `scene_from_user_input` 把用户字符串包成最小 `SceneSpec`。
- **下游 prompt**：`build_prompt(scene, kind="rgb", pose_idx=...)` 为每位姿生成 ERP 图像提示（含无人物等硬约束，见 `prompts.py`）。

### 3.4 多视图 RGB + 深度 + 法线 `nvs.py`

对 `poses` 中 **每个**位姿顺序执行：

1. **RGB（图像 API）**
  - **位姿 0（中心）**：`ImageClient.generate_rgb` — 纯文生图。
  - **位姿 1–8**：由 `nvs_strategy.corner_method` 决定（见下表）。
2. **若返回尺寸与配置不一致**：双线性 resize 到 `openai.size` 解析出的 `W×H`。
3. **深度**：`estimate_erp_depth(..., mode="direct")` — 当前 NVS **固定 `direct`**（`dap_mode` 形参在 `run_hybrid_nvs` 中为 legacy，未接配置）。
4. **法线**：`normals_from_erp_depth(depth_m, pose_R=pose.R, smooth_radius=1)`。
5. **落盘**：`erp/rgb|depth|normal`、`erp_decoded/pose_*_{depth_m,normal_world}.npy`，角点可选 `erp/warp/` 下保存 warp 图与 mask。

`**nvs_strategy.corner_method` 对比**（与 `ImageClient` 能力绑定）：


| 方法             | 行为                                                                         | 需要 `provider`                   | 说明                                            |
| -------------- | -------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------- |
| `warp_inpaint` | 用中心 RGB+DAP 深度做 **前向 warp** 到当前角点 → 空洞 mask → `**images.edit` + mask** 只填洞 | `**openai`**（走标准 `images.edit`） | `edit_with_mask` 在 `openrouter` 会直接抛错；几何一致性最强 |
| `i2i`          | 整张中心 RGB 作参考，**整图 edit / i2i**，无 mask                                      | `openai` 或 `openrouter`         | 一致性中等                                         |
| `generate`     | 每位姿独立 **文生图**，无参考                                                          | 任意                              | 一致性最弱、最便宜（角点）                                 |


**失败回退**：若 `warp_inpaint` / `i2i` 在某角点抛错，会将 `corner_path_works=False`，**后续角点全部改为 `generate`**，避免反复重试浪费配额。

`**hole_dilate_px**`：空洞 mask 膨胀像素，略增 inpaint 区域，减少边缘接缝。

### 3.5 图像 API `openai_erp.py`

- `**provider: "openai"**`（含官方与 **OpenAI 兼容中继**）：`images.generate`、`images.edit`（含带 mask 的 edit）。支持 `size`、`rgb_quality` 等（具体取决于上游是否识别 `quality` 参数，代码里有 TypeError 降级重试）。
- `**provider: "openrouter"`**：**仅** `chat.completions` + `modalities=["image","text"]`，输出被服务端锁在 **1024×1024**，**不能用** `edit_with_mask`；`supports_native_2x1_ratio` 为 false。
- **缓存**：请求指纹 SHA256 → `cache_dir` 下 PNG，重复调用命中缓存则不再请求网络。

**配置文件中「API 来源」对比示例**（见 `config/default.yaml` 顶部注释）：


| 方案                  | `provider` / `base_url`      | 典型限制                                  |
| ------------------- | ---------------------------- | ------------------------------------- |
| OpenAI 直连           | `openai`，`base_url` 空        | 需有效组织/计费；`gpt-image-2` 可做真 2:1 等      |
| 兼容中继（如 OneAPI / 自建） | `openai`，`base_url` 指向中继     | 依赖中继是否透传 `images.*` 与真实模型             |
| OpenRouter          | `openrouter`                 | 无原生 `images` 路径；分辨率固定；无真 mask inpaint |
| 当前默认 yaml 示例        | `openai` + `api.chatfire.cn` | 项目内注释说明选用原因（2:1、质量等）                  |


### 3.6 深度与法线 `dap.py`

- **模型体量**：`nvs.dap_model_size` → `small` | `base` | `large`（HF 模型 ID 见 `_MODEL_TABLE`）。
- **ERP 处理模式**（库内支持，**NVS 当前仅用 `direct`**）：
  - `direct`：整幅 ERP 直接送 DAP（快，两极略不准）。
  - `cubemap_split`：先拆 cubemap 再拼回（慢，极区更友好）。

深度经 **分位数线性缩放** 映射到 `[depth_near_m, depth_far_m]`（`nvs.depth_near_m` / `depth_far_m`）。

### 3.7 透视切分 `erp_to_persp.py`

对每个 pose，将 ERP 上每个透视像素对应射线转到 ERP 相机系，再双线性采样 RGB/深度/法线；深度转为透视 **z**（沿 face 前向乘以 `cos` 因子）并写 uint16 mm PNG。

**当前 `_faces_for_scheme` 实际支持的 `perspective.scheme`**：


| scheme           | 每 pose 视图数 | 用途简述                                                                                   |
| ---------------- | ---------- | -------------------------------------------------------------------------------------- |
| `cubemap`        | 6          | 经典六面体                                                                                  |
| `persp48_zigzag` | 48         | 12×yaw × 4×pitch，**列优先 zigzag** 顺序，相邻帧只变一个轴，便于当「视频」喂给 SAM 类模型；可写 `frames.txt`（若上层脚本写入） |


**重要说明**：`split_pose_to_perspectives` 的 docstring 仍提到 `persp16`，但 `_faces_for_scheme` 未注册 `persp16`，传入会 `ValueError`。README / `scripts/test_persp16_split.py` 等可能对应历史或其它分支；以 `erp_to_persp.py` 内分支为准。

**可调**：`perspective.fov_deg`（默认 90，越大边缘透视感越强）、`perspective.out_size`（默认 1024）。

### 3.8 COLMAP 与初始点云

- `**init_pcd.py`**：由 ERP 深度 + 位姿反投影，体素下采样；参数来自 `**fastgs.init_max_points`、`fastgs.init_voxel_m**`（虽在 fastgs 命名空间下，Stage 1 也复用）。
- `**colmap_writer.write_colmap_sparse**`：`fastgs.init_from_points3d` 为 true 时把点云写入 COLMAP `points3D.txt`；并复制图像等到 `colmap/`。

### 3.9 可选：Decoder 实验 `decoder_experiment`

若 `decoder_experiment.enabled: true`：对 `experiment_poses` 列表中的索引，调用 `**decode_to_depth` / `decode_to_normal**`（让图像模型从 RGB「画」深度/法线），与 DAP/解析法线对比指标，写入 `decoder_vs_dap.json`。**主流程几何仍以 DAP 为准**。

---

## 4. Stage 1.5：`scripts/sr_erp_4x.py`

- 输入：`erp/rgb/pose_*.png`。
- RGB：**Real-ESRGAN x4**，`erpgen/sr.py` 中带 **水平 wrap padding**，减轻 360° 接缝。
- 深度 / 法线：**双线性 4×** 放大 `erp_decoded/*.npy`，**不**在高分辨率重跑 DAP（节省算力）。
- CLI 可调：`--scale`、`--wrap-pad`、`--tile`、`--no-half`。

---

## 5. Stage 2：`scripts/train_gs.py`

- `**--backend splatfacto`**（默认）：`recon/run_splatfacto.py` → nerfstudio；失败且未 `--no-fallback` 时会 **自动改试 FastGS**。
- `**--backend fastgs`**：`recon/run_fastgs.py`，依赖 `fastgs.repo_path` 指向已 clone 的 FastGS；找不到 train 脚本时可 **fallback** 为稀疏点云 PLY（仍可被 viewer 打开）。
- `**fastgs.iterations`、`fastgs.extra_args` 等**：影响 FastGS 子进程；splatfacto 迭代可在脚本参数覆盖（见 `train_gs.py` 全文）。

---

## 6. 输出目录约定（Stage 1 后）

在 `outputs/runs/<run_id>/` 下典型包括：

- `resolved_config.yaml`、`meta.json`、`poses.json`、`prompts.json`
- `erp/`（rgb、depth、normal PNG）、`erp_decoded/`（float32 npy）
- `perspective/`、`colmap/`（含 `init_pcd.ply`）
- 4× 管线：`erp/rgb_4x/`、`erp_decoded/*_4x.npy`、`perspective_4x/`、`colmap_4x/`（由后续脚本生成，非 `generate_erp` 必选）

---

## 7. 其它脚本（索引）


| 脚本                                   | 作用                                                                                                 |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `regenerate_persp_4x.py`             | 基于 4× ERP 数据重新切透视并重建 COLMAP                                                                        |
| `serve_viewer.py`                    | 本地 HTTP + Spark 风格 viewer                                                                          |
| `build_colmap_for_fastgs.py`         | 另一套建 COLMAP 流程，内含 `persp16_4x4` 等字符串（与主库 `erp_to_persp` 的 scheme 集合**不一定相同**）                      |
| `e2e_test.py`、各类 `test_*.py`         | 冒烟与几何自检                                                                                            |
| `docs_sample_gpt_image2_official.py` | 仅官方 `api.openai.com` 调用 `gpt-image-2` 一次，输出 `docs/openai_official_gpt_image2_sample.png` 与 `.json` |


---

## 8. 与 README 可能不一致之处（阅读源码时注意）

1. `**persp16`**：主路径 `erp_to_persp._faces_for_scheme` **不支持**；README 中三方案描述可能偏旧。
2. `**generate_erp.py` 的 `--mock`**：当前该脚本内 **无 mock 参数**；无 API key 时需自行 mock 或改代码（README 若仍写 mock，以源码为准）。
3. `**nvs.strategy` / `dap_mode`**：README 曾写 `nvs.strategy: hybrid`；当前实现为 `**nvs_strategy.corner_method**` + NVS 内 **固定 DAP `direct`**。

---

## 9. 快速对照：你最常改的配置键


| 键路径                                                               | 作用                                        |
| ----------------------------------------------------------------- | ----------------------------------------- |
| `openai.provider` / `base_url` / `model` / `size` / `rgb_quality` | 图像 API 来源与分辨率、质量                          |
| `openai.text_model`                                               | SceneExpander 用文本模型                       |
| `prompt.seed` / `--scene` / `--no-expand`                         | 场景句、可重复性、是否扩写                             |
| `cuboid.*` / `poses.*`                                            | 房间尺寸与 9 相机布局                              |
| `nvs_strategy.corner_method` / `hole_dilate_px`                   | 角点 RGB 生成策略                               |
| `nvs.dap_model_size` / `depth_near_m` / `depth_far_m`             | DAP 体量与深度范围                               |
| `perspective.scheme` / `fov_deg` / `out_size`                     | 透视切分布局与相机内参尺度                             |
| `decoder_experiment.*`                                            | 是否跑 LLM 深度/法线对比实验                         |
| `fastgs.*`                                                        | FastGS 路径、迭代、初始化点云参数（亦影响 Stage1 init_pcd） |
| `viewer.port` / `bind`                                            | 本地查看器                                     |


---

*文档生成说明：基于仓库内 `erpgen/*.py`、`scripts/generate_erp.py`、`config/default.yaml` 等阅读整理；含 FAQ（Pose / Warp / OpenAI）与 `scripts/docs_sample_gpt_image2_official.py` 官方样张流程。若后续合并 PR 恢复 `persp16` 或 mock CLI，请同步更新本文档。*