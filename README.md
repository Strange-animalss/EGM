# EGM — ERP Gen WorldModel

> 用一段文本生成可漫游的 3D Gaussian Splatting 场景。文本 → 9 张 360° 全景图 (`gpt-image-2`) → 单目深度 (`Depth-Anything-V2`) → 解析法向量 → 4× ERP 超分 (`Real-ESRGAN`) → 透视图切分 → COLMAP → `splatfacto` / `FastGS` 训练 → `.ply`。

## 1. 总览

```
┌──────────────────────┐
│ "an empty cafe"      │  ← 用户场景描述
└─────────┬────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ SceneExpander (gpt-5.5-pro, reasoning=high)  │  PR #1
│ 输出 25-50 项的极详细 SceneSpec               │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ ImageClient (gpt-image-2 via OpenRouter)     │  PR #1
│ 9 个相机姿态 (1 中心 + 8 角点)                │
│ - center: text-to-image                      │
│ - corners: ref-i2i (center 当参考图)         │
│ ⇒ 9 张 1024×512 ERP RGB                      │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ DAP-V2 单目深度 + 解析法向量                  │  PR #2
│ ⇒ 9 个 (H,W) float32 米深度 .npy             │
│ ⇒ 9 个 (H,W,3) float32 世界法线 .npy         │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ Real-ESRGAN x4 ERP 超分（wraparound-safe）   │  PR #3
│ ⇒ 9 张 4096×2048 ERP RGB                     │
│   depth/normal bilinear 4x upsample          │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ Perspective 切分                             │  PR #4
│   cubemap (6) | persp16 (16) | persp48_zigzag (48) │
│ ⇒ 9 × N 张透视图 + cameras.json + COLMAP     │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ GS 训练 (--backend splatfacto | fastgs)      │  PR #5
│ ⇒ gs/output.ply（INRIA 3DGS 格式）           │
└─────────┬────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────┐
│ Spark.js / SuperSplat / brush 漫游           │
└──────────────────────────────────────────────┘
```

## 2. 关键特性


| 特性                   | 说明                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **多 provider**       | `provider: "openai"` 走官方 `/v1/images/{generations,edits}`；`"openrouter"` 走 chat completions + `modalities=["image","text"]` |
| **GPT-5.5-pro 场景扩展** | 输入 `"an empty cafe"`，输出含建筑/家具/材质/光线/装饰的 25-50 项详细 spec，且 `seed` 参数可重复                                                       |
| **DAP-V2 真深度**       | 替代 LLM 画的"伪深度图"。9 次 LLM 调用即可拿到几何严格对齐的 (RGB, depth, normal) 三元组                                                              |
| **解析法向量**            | 从 metric depth 数值微分得到 world-frame normals，跟 depth 数学一致                                                                      |
| **4× ERP 超分**        | Real-ESRGAN_x4plus，左右 wrap-padding 保 360° 闭合                                                                                |
| **3 套切分方案**          | cubemap (6) / persp16 (16) / **persp48_zigzag (48 帧 column-first，可直接当视频喂 SAM3**)                                            |
| **多 GS backend**     | `splatfacto`（nerfstudio，质量稳定）/ `fastgs`（CVPR 2026，10× 快但 sm_120 不稳）                                                         |
| **全管线缓存**            | 每个 API 请求按 (provider, model, prompt, params) SHA256 缓存，重复运行免费                                                               |
| **Mock 模式**          | 无 API key 即可验证全管线                                                                                                           |


## 3. 环境要求

### 必须

- Python 3.10+（实测 3.12.10）
- PyTorch ≥ 2.7（CUDA 12.x build；本仓库验证过 `torch 2.7.1+cu128`）
- 一张 NVIDIA GPU（VRAM ≥ 8 GB，sm_75 / sm_86 / sm_89 / sm_120 都验证过）
- OpenAI 或 OpenRouter API key

### 训练 GS 时额外需要


| 后端                    | Linux                                       | Windows                                                                                       |
| --------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `splatfacto` (gsplat) | `pip install nerfstudio` 一行搞定               | 同左，但首次跑会 JIT 编译 CUDA kernel，需要 **VS 2022 Build Tools 的 C++ workload** + **CUDA Toolkit 12.x** |
| `fastgs`              | `git clone` + `pip install` 它的 3 个 CUDA 子模块 | 同左 + Build Tools + CUDA Toolkit                                                               |


### Windows 专用一键激活脚本

仓库自带 `scripts/win_env.bat`，一行激活完整编译/运行环境（VS C++ + CUDA + `TORCH_CUDA_ARCH_LIST` + UTF-8 控制台）：

```bat
cmd /c scripts\win_env.bat
```

## 4. 安装

```bash
git clone https://github.com/Strange-animalss/EGM.git
cd EGM
pip install -r requirements.txt

# OpenRouter（推荐，国内可访问）
$env:OPENAI_API_KEY = "sk-or-v1-..."

# 或 OpenAI 官方
$env:OPENAI_API_KEY = "sk-..."

# 训练真 GS（任选其一，二者都需要 CUDA + 编译器）
pip install nerfstudio                                       # splatfacto
git clone https://github.com/fastgs/FastGS.git --recursive third_party/FastGS  # FastGS
```

API key 通过环境变量传入；**不要把它写进 `config/default.yaml` 提交**——repo 默认 `api_key: ""`，运行时从 env 读取。

## 5. 配置

`config/default.yaml`：

```yaml
openai:
  provider: "openrouter"          # "openai" | "openrouter"
  model: "openai/gpt-5.4-image-2" # OpenRouter 上 gpt-image-2 的等价 ID
  text_model: "openai/gpt-5.5-pro"
  size: "1024x512"                # ERP 原生分辨率（OpenRouter 输出 1024x1024，2:1 裁剪后是 1024x512）
  base_url: "https://openrouter.ai/api/v1"
  api_key: ""                     # 留空 → 从 OPENAI_API_KEY 读取
  request_timeout_sec: 600
  max_retries: 4
  cache_dir: "outputs/.openai_cache"
  reasoning_effort: "high"

nvs:
  strategy: "hybrid"              # OpenRouter 模式自动降级为 ref-i2i
  dap_model_size: "base"          # small | base | large
  dap_mode: "direct"              # direct | cubemap_split
  depth_near_m: 0.3
  depth_far_m: 12.0

perspective:
  scheme: "persp48_zigzag"        # cubemap | persp16 | persp48_zigzag
  fov_deg: 90.0
  out_size: 1024
  enhance_with_image2: false      # true = 每个 face 都过 image-2 i2i 加细节（贵）

fastgs:
  repo_path: "third_party/FastGS"
  iterations: 7000
```

CLI dotlist 可覆盖任意键：

```bash
python scripts/generate_erp.py --scene "..." cuboid.size_xyz=[6,6,3] perspective.scheme=cubemap
```

## 6. 快速开始（端到端）

### 6.1 一键完整管线

```bash
# 1. 生成 ERP（9 个 pose × 1 个 LLM RGB call + 本地 DAP/normal）
python scripts/generate_erp.py --no-mock --scene "an empty specialty coffee shop, late morning" prompt.seed=42

# 2. 4x 超分 + 重新切分透视图 + 重建 COLMAP
python scripts/sr_erp_4x.py --run-id <RUN_ID>
python scripts/regenerate_persp_4x.py --run-id <RUN_ID>

# 3. 训练 GS（默认 splatfacto）
python scripts/train_gs.py --run-id <RUN_ID> --backend splatfacto \
    --colmap-dir colmap_4x --output-dir gs_4x --iterations 7000

# 4. 启动 viewer
python scripts/serve_viewer.py --run-id <RUN_ID>
```

`<RUN_ID>` 由 `generate_erp.py` 自动生成（如 `run_20260430-160000`），也可用 `--run-id my_run` 自定义。

### 6.2 Windows 一键脚本（推荐）

```bat
cmd /c scripts\win_env.bat
python scripts\generate_erp.py --no-mock --scene "an empty cafe" --run-id cafe_v1
python scripts\sr_erp_4x.py --run-id cafe_v1
python scripts\regenerate_persp_4x.py --run-id cafe_v1
python scripts\train_gs.py --run-id cafe_v1 --backend splatfacto --colmap-dir colmap_4x --output-dir gs_4x
python scripts\serve_viewer.py --run-id cafe_v1
```

### 6.3 Mock 模式（无 API key）

```bash
python scripts/generate_erp.py --mock --scene "any text"
```

合成棋盘房间 + 立方体距离函数深度 + 立方体面法线，全管线打通验证用。

## 7. 各阶段细节

### 7.1 SceneExpander（PR #1）

`erpgen/scene_expander.py` 用 `gpt-5.5-pro` 把短描述展开为：

```json
{
  "scene_kind": "specialty pour-over and espresso coffee shop",
  "style": "industrial Scandinavian: 4 m exposed concrete ceiling with matte black steel I-beams ...",
  "light": "late morning around 10:30 AM, crisp 5600K daylight ...",
  "occupancy": "empty room, no people, no animals",
  "extra_props": "long pale-oak espresso bar, charcoal quartz countertop, ... (50+ items)"
}
```

通过 `--scene` 参数 + `prompt.seed=N` 可重复。`--no-expand` 跳过扩展直接用原文。

### 7.2 ImageClient（PR #1）

`erpgen/openai_erp.py` 透明支持两个 provider：

- **OpenAI 官方**：`client.images.generate()` / `client.images.edits()`
- **OpenRouter**：`client.chat.completions.create(modalities=["image","text"], extra_body=...)`，从 `message.images[0].image_url.url` 解 base64

OpenRouter 没有 mask edit，`edit_with_mask()` 自动降级为"洋红涂层 + 文字"参考；`generate_with_ref()` 是 i2i 无 mask 公共接口。

### 7.3 NVS（PR #2）

`erpgen/nvs.py` 每个 pose 流程：

```
pose_idx == 0 (center):    LLM text-to-image RGB
pose_idx > 0  (corner):    LLM ref-i2i RGB（参考图 = center RGB）
然后：
    DAP depth = infer_erp_depth_metric(rgb)            # erpgen/dap.py
    normal    = analytic_normal_from_depth_erp(depth)  # erpgen/dap.py
```

**所有三元组在同一帧 RGB 上派生，几何严格对齐**。这是从 v1（让 LLM 画三张独立图）的关键改进。

`nvs.strategy` 在 OpenRouter 模式下自动忽略（OpenRouter 不支持真 mask edit），强制走 ref-i2i 路径。

### 7.4 Real-ESRGAN ERP 超分（PR #3）

`erpgen/sr.py`：

- 模型：`RealESRGAN_x4plus.pth`（自动下载到 `third_party/weights/`，约 64 MB）
- 加载：通过 `spandrel`（避免 `basicsr/realesrgan` 跟 torch ≥ 2.4 的 ABI 冲突）
- 360° 闭合：左右各 `wrap_pad_px=128` 像素从对侧 wrap 进来 → 跑 SR → 裁掉 padding
- 评估：`horizontal_seam_score()` 算 first/last column 的 L2 mismatch

depth/normal 不超分，用 numpy bilinear 4x 上采样（连续场不需要 SR 模型）。

### 7.5 Perspective 切分（PR #4）

`erpgen/erp_to_persp.py` 三个方案：


| Scheme           | 视图数/pose                                 | 用途                                            |
| ---------------- | ---------------------------------------- | --------------------------------------------- |
| `cubemap`        | 6                                        | 经典 cubemap 6 面                                |
| `persp16`        | 16（8 yaw × 2 pitch ±30°，FOV 90°）         | 紧凑 GS 训练数据集                                   |
| `persp48_zigzag` | 48（12 yaw × 4 pitch +45°/+15°/-15°/-45°） | column-first zigzag，相邻帧只换一个轴，**可直接当视频喂 SAM3** |


`persp48_zigzag` 帧名 `frame_NNN_yawXX_pitchYY.png`，按全局 zigzag 索引：

```
frame 000: yaw=  0  pitch=+45
frame 001: yaw=  0  pitch=+15
frame 002: yaw=  0  pitch=-15
frame 003: yaw=  0  pitch=-45
frame 004: yaw= 30  pitch=-45    ← column 翻转，相邻只换 yaw
frame 005: yaw= 30  pitch=-15
...
frame 047: yaw=330  pitch=+45
```

旋转矩阵用 `look_at_R(forward, world_up=+Z)` 强制相机无 roll，yaw=0 对应 ERP `+X` 中心列。**采样数学严格正确**：合成 ERP 测试中所有 great circle 在透视图上 R²=1.0 / 残差 0 px（见 `scripts/test_persp_sampling.py`）。

### 7.6 GS 训练（PR #5）

```bash
python scripts/train_gs.py --run-id <RUN_ID> --backend {splatfacto|fastgs}
```


| 后端           | 实现路径                                                                                   | 速度 (432 张 1024² @ 7000 iter) | 质量 (我们 cafe 数据)                    | 备注                                                                                                |
| ------------ | -------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `splatfacto` | `recon/run_splatfacto.py` 包 `ns-train splatfacto` + `scripts/export_splatfacto_ply.py` | ~6.5 min                     | PSNR 34.4 / SSIM 0.92 / LPIPS 0.20 | nerfstudio 的 `ns-export gaussian-splat` 在 Windows + UTF-8 会因 pymeshlab 崩溃，故自带 ckpt → INRIA PLY 导出 |
| `fastgs`     | `recon/run_fastgs.py` 子进程                                                              | 30 秒（1500 iter）              | 收敛不充分，估算 PSNR 25-28 dB             | 7000 iter 在 sm_120 上 segfault；需上游修                                                                |


GS backend 失败时自动回退到 init point cloud（`init_pcd.ply`，DAP 反投影出的彩色稀疏点云），保证 `gs/output.ply` 总是有内容能在 viewer 里看。

### 7.7 SAM3 video 集成

`persp48_zigzag` 切分时每个 pose 目录会同时写一个 `frames.txt`：

```
rgb/frame_000_yaw0_pitch+45.png
rgb/frame_001_yaw0_pitch+15.png
rgb/frame_002_yaw0_pitch-15.png
...
```

可直接 `sam2 video --frames-dir perspective_4x/pose_0/rgb` 或读 `frames.txt` 喂 SAM3 video 推理拿一致 instance ID。

## 8. 输出结构

```
outputs/runs/<RUN_ID>/
├── resolved_config.yaml             # 完整配置快照
├── meta.json                        # run 元数据
├── poses.json                       # 9 个 pose 的 R, t, name
├── prompts.json                     # SceneExpander 输出
│
├── erp/
│   ├── rgb/        pose_0..8.png    # 9 张 1024×512 RGB
│   ├── rgb_4x/     pose_0..8.png    # 9 张 4096×2048 SR
│   ├── depth/      pose_0..8.png    # 1024×512 灰度 depth PNG
│   └── normal/     pose_0..8.png    # 1024×512 normal PNG
│
├── erp_decoded/
│   ├── pose_*_depth_m.npy           # (512,1024) float32 米
│   ├── pose_*_normal_world.npy      # (512,1024,3) float32
│   ├── pose_*_depth_m_4x.npy        # (2048,4096) float32 米
│   └── pose_*_normal_world_4x.npy   # (2048,4096,3) float32
│
├── perspective/                     # 1x 切分（如 cubemap）
├── perspective_4x/                  # 4x 切分（如 persp48_zigzag）
│   ├── cameras.json
│   └── pose_0..8/
│       ├── rgb/    frame_NNN_*.png  # N 张 1024×1024
│       ├── depth/  frame_NNN_*.png  # uint16 mm
│       ├── normal/ frame_NNN_*.png
│       └── frames.txt               # zigzag 顺序帧清单（喂 SAM3 用）
│
├── colmap/        (1x)
├── colmap_4x/     (4x)              # 透视图 + sparse/0/{cameras,images,points3D}.txt
│   └── init_pcd.ply                 # DAP 反投影彩色点云
│
└── gs/  或 gs_4x/  或 gs_4x_fastgs_*/  
    ├── output.ply                   # GS 训练结果（或 init_pcd fallback）
    ├── point_cloud.ply              # init_pcd 副本
    └── train.log
```

## 9. Viewer

```bash
python scripts/serve_viewer.py --run-id <RUN_ID>
```

打开 `http://127.0.0.1:8765/?run=<RUN_ID>&ply=gs_4x/output.ply`。也可手动切换 PLY：

```
http://127.0.0.1:8765/?run=<RUN_ID>&ply=gs_4x_fastgs_no_pose8/output.ply
http://127.0.0.1:8765/?run=<RUN_ID>&ply=colmap_4x/init_pcd.ply
```

或直接拖 `.ply` 进 [SuperSplat](https://playcanvas.com/supersplat/editor) / [brush](https://github.com/ArthurBrussee/brush) 等三方 viewer。

## 10. 常见问题

### 10.1 `pip install gsplat` 后第一次跑训练 hang 在 import 1-2 分钟

正常。gsplat 1.4.x 在 Windows 上是 `py3-none-any` wheel，**首次 import 会 JIT 编译 CUDA kernel**（约 5 min on RTX 5070），编译产物缓存在 `~/.cache/torch_extensions/`，之后秒启。

### 10.2 `subprocess.CalledProcessError: 'where cl' returned 1`

splatfacto/gsplat 找不到 `cl.exe`。运行 `cmd /c scripts\win_env.bat` 后再 `python scripts/train_gs.py`。如果还不行，查 VS 2022 是否装了 "Desktop development with C++" workload。

### 10.3 OpenRouter 输出"上下两张图"的怪 ERP

OpenRouter chat-image API 偶尔把"i2i ref + prompt"理解成"输出 before/after 对比图"。需要：

- 加强 prompt 里"single seamless panorama, no split panels, no comparison"
- 或者直接重新生成那一张
- 工具：`scripts/regen_pose8_via_warp_ref.py` 用其他 pose 的 fused warp 当 ref 重做

### 10.4 透视图"看起来弯弯的"

数学上是对的（`scripts/test_persp_sampling.py` 在合成 ERP 上验证 R²=1.0）。**90° FOV 透视图边缘像素被 `cos²(45°) = 0.5` 压缩**，肉眼觉得鱼眼是数学正确性质。要"看着不弯"可以把 `perspective.fov_deg` 降到 60-70°。

### 10.5 FastGS 在 sm_120 上 segfault

7000 iter 训到 ~50% 时 `_C.rasterize_gaussians` 内部 buffer overflow。**降到 1500 iter 稳定**但收敛不充分（PSNR 比 splatfacto 低 6-9 dB）。要满血质量请用 `--backend splatfacto`。

### 10.6 cafe 场景里凭空冒出一堆人

`SceneExpander` 默认产 `occupancy: "empty room, no people, no animals"`，但用户输入 `"a busy cafe"` 时会被覆盖。`prompt.py` 的 `ERP_HARD_CONSTRAINT` 也加了 NO PEOPLE 约束兜底。如果还有，请在 `--scene` 里显式写 "no people"。

## 11. 项目结构

```
EGM/
├── config/default.yaml             # 全局配置
├── erpgen/                         # 核心库
│   ├── config.py                   # 配置 / run_id
│   ├── poses.py                    # 9 个相机姿态生成
│   ├── prompts.py                  # ERP prompt 拼接（含 NO PEOPLE 硬约束）
│   ├── scene_expander.py           # gpt-5.5-pro 场景扩展（PR #1）
│   ├── openai_erp.py               # provider 抽象（OpenAI / OpenRouter，PR #1）
│   ├── nvs.py                      # 9 pose × (LLM RGB → DAP → normal)（PR #2）
│   ├── dap.py                      # Depth-Anything-V2 + 解析法向量（PR #2）
│   ├── sr.py                       # Real-ESRGAN x4 wraparound-safe SR（PR #3）
│   ├── erp_to_persp.py             # cubemap / persp16 / persp48_zigzag（PR #4）
│   ├── warp.py                     # ERP 前向 warp（NVS 备用）
│   ├── decode.py                   # depth/normal PNG 解码（legacy）
│   ├── init_pcd.py                 # depth 反投影点云
│   ├── colmap_writer.py            # COLMAP txt 输出
│   └── sanity.py                   # 各阶段质量检查
│
├── recon/                          # GS 训练后端
│   ├── run_fastgs.py               # FastGS subprocess（已有）
│   └── run_splatfacto.py           # nerfstudio splatfacto（PR #5）
│
├── scripts/
│   ├── generate_erp.py             # ★ Stage 1 入口
│   ├── sr_erp_4x.py                # Stage 2: ERP 4x 超分
│   ├── regenerate_persp_4x.py      # Stage 3: 重切 + COLMAP
│   ├── train_gs.py                 # Stage 4: GS 训练（multi-backend）
│   ├── export_splatfacto_ply.py    # ckpt → INRIA PLY
│   ├── serve_viewer.py             # Spark.js viewer
│   ├── e2e_test.py                 # 端到端冒烟测试
│   ├── win_env.bat                 # Windows 编译环境一键激活
│   ├── test_*.py                   # 各模块单测 / 冒烟
│   └── ...
│
├── viewer/                         # Spark.js 浏览器端
│   ├── index.html
│   └── main.js
│
├── outputs/runs/<RUN_ID>/          # 每次运行的产物
└── third_party/                    # FastGS / FastGS-Instance / 模型权重（gitignored）
```

## 12. 开发与贡献

### 单元测试

```bash
python scripts/test_openrouter_pipeline.py   # OpenRouter API 端到端
python scripts/test_dap_smoke.py             # DAP-V2 推理 + 法向量
python scripts/test_persp16_split.py         # persp16 切分
python scripts/test_persp48_zigzag.py        # persp48 zigzag 顺序
python scripts/test_persp_sampling.py        # 几何正确性（R²=1.0 grand circle test）
```

### PR 历史（功能切片）


| PR                 | 主题                                                        |
| ------------------ | --------------------------------------------------------- |
| [#1](../../pull/1) | OpenRouter API integration（image + text）                  |
| [#2](../../pull/2) | DAP-V2 metric depth + analytic normals                    |
| [#3](../../pull/3) | Real-ESRGAN x4 ERP super-resolution（wraparound-safe）      |
| [#4](../../pull/4) | persp16 / persp48_zigzag 切分 + orchestrator wiring         |
| [#5](../../pull/5) | splatfacto multi-backend GS training + Windows env helper |


## 13. 已知限制

- gpt-image-2 / OpenRouter 输出的 1024×512 不是数学严格的 ERP（模型不懂球面投影约束），所以 90° FOV 透视图边缘有非 gnomonic 失真。要根治需要换专用全景生成模型（PanoDiffusion / MultiDiffusion / SDXL+ERP-aware ControlNet）。
- 9 个 pose 在 4×4×3 m cuboid 内，视差只有 ~2.5 m。GS 训练对极小角点视差的细节恢复力弱。增加 pose 数（比如 27 = 中心 + 8 角 + 18 棱中点）能改善但 LLM 调用成本翻 3 倍。
- `pose_8` 离 center 最远（2.56 m），即使 fused warp 8 个邻居也有 ~52% hole 喂给 i2i 当参考。新生成的 pose_8 跟其他 ERP 一致性较弱，可能拖累 GS 收敛。
- FastGS 在 RTX 5070 (sm_120) 上跑 7000 iter 会 segfault（上游 ABI 兼容问题），目前限制 1500 iter，质量明显低于 splatfacto。要满血请用 splatfacto 或换 sm_89 GPU。

## License

MIT