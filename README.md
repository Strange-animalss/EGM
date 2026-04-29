# EGM — ERP Gen WorldModel

> 基于 GPT-5.5-pro 深度推理 + gpt-image-2 的 360° 全景生成 → 多视角 NVS → FastGS 三维重建 → Spark.js 交互浏览 完整管线。

## 管线架构

```
用户场景输入 ("cyberpunk bar")
    │
    ▼
┌─────────────────────────────────────────────┐
│  GPT-5.5-pro (reasoning_effort=high)        │
│  深度推理 → 展开为结构化 SceneSpec            │
│  { style, light, occupancy, extra_props }    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  gpt-image-2 × 9 姿态 ERP 生成               │
│  每个姿态: RGB + 彩色深度图 + 彩色法线图       │
│  中心姿态: 独立生成                           │
│  角落×8:  warp + inpainting (hybrid NVS)     │
│  支持 reasoning_effort 深度推理增强            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  解码 & 后处理                               │
│  • 深度: 灰度(白=近,黑=远) → 公制米           │
│  • 法线: RGB编码 → 单位世界法线               │
│  • 可选 DAP 深度校准                          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Cubemap 分割 (6面 × 9姿态 = 54视图)         │
│  写入 COLMAP 稀疏格式 (cameras/images/points3D)│
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  FastGS (CVPR 2026) 训练                    │
│  深度反投影初始化点云 → 跳过 COLMAP SfM       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Spark.js 交互式 3D 查看器                   │
│  THREE.js + @sparkjsdev/spark 渲染           │
└─────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **GPT-5.5-pro 场景展开** | 输入 "cyberpunk bar" → 自动补全风格/光照/道具等细节 |
| **gpt-image-2 深度推理** | 图像生成支持 `reasoning_effort` 参数，提升一致性 |
| **Hybrid NVS** | 中心姿态独立生成 + 8个角落通过 warp+inpaint 保持场景统一 |
| **全管线缓存** | 每个 API 请求按 (model, prompt, params) SHA256 缓存，重复运行免费 |
| **Mock 模式** | 无需 API key 即可验证管线全流程 |
| **DAP 深度校准** | 可选集成 DAP 单目深度估计，线性校准 gpt-image-2 深度图 |

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
pip install -r requirements.txt

# (可选) 克隆 FastGS 以进行真实高斯训练
git clone https://github.com/fastgs/FastGS.git --recursive third_party/FastGS
```

### 2. API 密钥设置

```bash
# 设置 OpenAI API Key (GPT-5.5-pro + gpt-image-2 共用)
export OPENAI_API_KEY="sk-..."
```

详细配置见 `config/default.yaml` 的 `openai` 部分。

### 3. 运行管线

```bash
# 自由场景输入 + GPT-5.5-pro 自动展开 (真实 API)
python scripts/generate_erp.py --scene "a cyberpunk bar at midnight, neon lights"

# 自由场景输入 + 跳过展开 (直接使用原始文本)
python scripts/generate_erp.py --scene "Japanese tea house" --no-expand

# 传统随机采样模式 (从配置池中随机组合)
python scripts/generate_erp.py --config config/default.yaml

# Mock 模式 (无 API key, 验证全管线)
python scripts/generate_erp.py --scene "cozy library" --mock

# 端到端测试 (含重试 + 完整性检查)
python scripts/e2e_test.py --scene "cyberpunk bar" --mock --budget 3
```

### 4. 训练 FastGS

```bash
python scripts/train_gs.py --run-id <run_id>
```

### 5. 启动交互式查看器

```bash
python scripts/serve_viewer.py --run-id <run_id>
# 打开 http://127.0.0.1:8765/index.html
```

## API 配置

所有 API 相关配置位于 `config/default.yaml` → `openai:`：

```yaml
openai:
  # gpt-image-2 图像生成模型
  model: "gpt-image-2"
  size: "3840x1920"         # ERP 2:1 全景图分辨率
  rgb_quality: "high"
  edit_quality: "high"
  api_key_env: "OPENAI_API_KEY"    # 图像模型 API Key 环境变量

  # GPT-5.5-pro 文本模型 (场景展开)
  text_model: "gpt-5.5-pro"
  text_model_api_key_env: "OPENAI_API_KEY"  # 文本模型 API Key (可与图像共用)

  # 深度推理强度
  reasoning_effort: "high"          # GPT-5.5-pro 推理强度 (high/medium/low)
  image_reasoning_effort: "medium"  # gpt-image-2 推理强度 (null=禁用)

  # 重试与缓存
  request_timeout_sec: 180
  max_retries: 4
  retry_backoff_sec: 5.0
  cache_dir: "outputs/.openai_cache"  # API 响应缓存目录
```

### 推理强度说明

| 参数 | 模型 | 作用 |
|------|------|------|
| `reasoning_effort: "high"` | GPT-5.5-pro | 深度思考场景构成，输出更丰富、更连贯的场景描述 |
| `image_reasoning_effort: "medium"` | gpt-image-2 | 生成前深度推理图像构图，提升多姿态一致性 |

设置为 `null` 可禁用对应推理功能（节省 token/时间）。

## Prompt 系统

### 场景展开模式 (默认)

用户输入简短描述 → GPT-5.5-pro 深度推理 → 结构化的 `SceneSpec`：

```json
{
  "scene_kind": "cyberpunk bar",
  "style": "industrial neon-lit, exposed pipes, holographic displays",
  "light": "dim purple and teal neon glow, volumetric fog catching light beams",
  "occupancy": "a few hackers at corner booths, bartender polishing glasses",
  "extra_props": "concrete floor, steel counter, LED strip accents, synthwave poster art"
}
```

展开后的 `SceneSpec` 会被 `scene_description()` 组合为完整描述文本，再拼接待生成类型指令和 ERP 约束。

### 原始文本模式 (`--no-expand`)

跳过 GPT-5.5-pro，直接将用户输入作为场景描述：

```bash
python scripts/generate_erp.py --scene "a dark rainy alley in Neo-Tokyo" --no-expand
```

### 传统随机采样模式 (无 `--scene`)

从 `config/default.yaml` 的 `prompt` 配置池中随机组合：

```yaml
prompt:
  scene_default: "coffee shop"
  style_pool: ["scandinavian", "industrial loft", "vintage parisian", ...]
  light_pool: ["soft morning sun...", "rainy afternoon...", ...]
  occupancy_pool: ["empty", "a couple of patrons...", ...]
  extra_props_pool: ["wooden floor...", "marble counter...", ...]
```

### 每个姿态的 Prompt 组成

```
场景描述 + 视角提示 + 类型指令 + ERP 约束
```

| 类型 | 指令后缀 |
|------|----------|
| `rgb` | (无额外指令) |
| `depth` | "GRAYSCALE DEPTH MAP: white=near, black=far, linear gradient..." |
| `normal` | "WORLD-SPACE SURFACE NORMAL MAP: R=+X, G=+Y, B=+Z..." |

## CLI 参考

### `generate_erp.py`

```
python scripts/generate_erp.py [OPTIONS] [OVERRIDES...]

Options:
  --config PATH       配置文件路径 (默认: config/default.yaml)
  --run-id ID         运行标识 (默认: 自动时间戳)
  --scene TEXT        自由场景描述 ("cyberpunk bar")
  --no-expand         跳过 GPT-5.5-pro 展开
  --mock              强制 Mock 模式
  --no-mock           强制真实 API 模式
  overrides           OmegaConf dotlist 覆盖 (cuboid.size_xyz=[6,6,3])

示例:
  python scripts/generate_erp.py --scene "medieval library" --mock
  python scripts/generate_erp.py --scene "beach cafe" cuboid.size_xyz=[8,8,4]
```

### `e2e_test.py`

```
python scripts/e2e_test.py [OPTIONS] [OVERRIDES...]

Options:
  --config PATH       配置文件路径
  --budget N          最大重试次数 (默认: 5)
  --scene TEXT        场景描述
  --no-expand         跳过 GPT-5.5-pro 展开
  --mock              Mock 模式
  --quiet             静默模式

示例:
  python scripts/e2e_test.py --scene "cyberpunk bar" --mock --budget 3
```

### `train_gs.py`

```
python scripts/train_gs.py [OPTIONS]

Options:
  --config PATH       配置文件
  --run-id ID         指定运行 ID
  --no-fallback       FastGS 不可用时报错 (默认: 回退到初始点云 PLY)
```

### `serve_viewer.py`

```
python scripts/serve_viewer.py [OPTIONS]

Options:
  --config PATH       配置文件
  --run-id ID         指定运行 ID (空=最新)
  --port PORT         HTTP 端口
  --bind ADDR         绑定地址
```

## 输入 / 输出

### 输入

| 输入 | 位置 | 说明 |
|------|------|------|
| 场景描述 | `--scene` CLI 参数 或 `config/default.yaml` | 自由文本或配置池 |
| API Key | 环境变量 `OPENAI_API_KEY` | 图像+文本模型 |
| Cuboid 配置 | `config/default.yaml` → `cuboid` | 空间尺寸/中心/角落策略 |
| 姿态配置 | `config/default.yaml` → `poses` | 相机位置/朝向 |
| 管线参数 | `config/default.yaml` 各节 | NVS 策略/分辨率/训练参数 |

### 输出 (per run)

```
outputs/runs/<run_id>/
├── resolved_config.yaml       # 完整配置快照 (可复现)
├── meta.json                  # 运行元数据
├── poses.json                 # 姿态定义
├── prompts.json               # 场景 SceneSpec
├── erp/
│   ├── rgb/pose_*.png         # 9 个 RGB 全景图 (3840×1920)
│   ├── depth/pose_*.png       # 9 个彩色深度图
│   ├── normal/pose_*.png      # 9 个彩色法线图
│   └── warp/                  # 角落姿态 warp 中间结果
├── erp_decoded/
│   ├── pose_*_depth_m.npy     # 公制深度 (float32)
│   └── pose_*_normal_world.npy # 世界法线 (float32×3)
├── perspective/
│   ├── cameras.json           # COLMAP 格式相机参数
│   └── pose_*/
│       ├── rgb/*.png          # 6 面 cubemap RGB (1024×1024)
│       ├── depth/*.png        # 6 面深度 (uint16 mm)
│       └── normal/*.png       # 6 面法线
├── colmap/
│   ├── init_pcd.ply           # 初始点云 PLY
│   └── sparse/0/
│       ├── cameras.txt        # COLMAP 相机
│       ├── images.txt         # COLMAP 图像姿态 + 特征
│       └── points3D.txt       # COLMAP 3D 点
├── gs/
│   ├── output.ply             # FastGS 高斯模型 (或回退点云)
│   └── train.log              # 训练日志
├── sanity_erp.json            # ERP 质量检查报告
└── sanity_ply.json            # PLY 质量检查报告
```

## 配置参考

完整配置见 `config/default.yaml`。关键节点：

| 配置节 | 用途 |
|--------|------|
| `cuboid` | 空间尺寸、中心坐标、角落内缩、朝向策略 |
| `poses` | 初始姿态、生成姿态策略 (auto_8_corners / 手动列表) |
| `openai` | 模型选择、推理强度、API Key、重试/缓存 |
| `prompt` | 场景默认值、随机池、种子 |
| `nvs` | NVS 策略 (hybrid/independent/warp_inpaint_only)、深度范围、DAP 校准 |
| `perspective` | Cubemap 方案、FOV、输出分辨率 |
| `fastgs` | FastGS 仓库路径、迭代次数、初始点云参数 |
| `viewer` | 查看器端口、绑定地址 |
| `run` | 输出目录、运行 ID |

所有参数均支持命令行 dotlist 覆盖：`cuboid.size_xyz=[8,8,4] mock.enabled=true`

## 部署

### 本地部署

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python scripts/e2e_test.py --scene "your scene" --budget 1
python scripts/serve_viewer.py
```

### 依赖项

```
openai>=1.50       # GPT-5.5-pro + gpt-image-2 API
numpy>=1.24        # 数值计算
Pillow>=10.0       # 图像处理
opencv-python-headless>=4.8  # 图像处理 (可选)
omegaconf>=2.3     # 配置管理
tqdm>=4.66         # 进度条
plyfile>=1.0       # PLY 文件读写
scipy>=1.11        # 科学计算
```

### 可选依赖

| 组件 | 依赖 | 用途 |
|------|------|------|
| FastGS | `third_party/FastGS/` (git clone) | 真实高斯训练 |
| DAP 校准 | PyTorch + CUDA + DAP weights | 深度图精度提升 |
| Spark.js | CDN 引入 (自动) | 3D 查看器渲染 |

## Mock 模式

无 OpenAI API Key 时，使用 `--mock` 标志运行完整管线：

- 合成 RGB 全景图: 基于 prompt hash 的彩色棋盘房间
- 合成深度图: 解析立方体距离函数
- 合成法线图: 立方体面法线投影
- 展开器返回固定模板 SceneSpec

Mock 输出质量有限，但完全覆盖管线所有阶段，适合验证部署正确性和开发调试。

## 项目结构

```
EGM/
├── config/default.yaml       # 全局配置 (所有可调参数)
├── erpgen/                   # 核心库
│   ├── config.py             # 配置加载 / 运行 ID 管理
│   ├── poses.py              # 姿态生成 (cuboid corners + look-at)
│   ├── prompts.py            # Prompt 构建 (场景 + 类型 + ERP 约束)
│   ├── scene_expander.py     # GPT-5.5-pro 场景展开
│   ├── openai_erp.py         # OpenAI API 封装 (生成/编辑/缓存/重试)
│   ├── nvs.py                # Hybrid NVS 调度器
│   ├── warp.py               # ERP 前向投影 + 遮罩处理
│   ├── decode.py             # 深度/法线解码 (公制/世界空间)
│   ├── erp_to_persp.py       # ERP → Cubemap 透视分割
│   ├── init_pcd.py           # 深度反投影初始点云构建
│   ├── colmap_writer.py      # COLMAP 稀疏格式输出
│   └── sanity.py             # 各阶段质量检查
├── recon/
│   └── run_fastgs.py         # FastGS 子进程驱动 + 回退逻辑
├── scripts/
│   ├── generate_erp.py       # Stage 1: ERP 生成 (入口)
│   ├── train_gs.py           # Stage 2: FastGS 训练
│   ├── e2e_test.py           # 端到端测试 (含重试 + 完整性检查)
│   └── serve_viewer.py       # HTTP 服务 + 查看器
├── viewer/
│   ├── index.html            # Spark.js 查看器 (CDN)
│   └── main.js               # 查看器逻辑 (THREE.js + Spark)
├── outputs/runs/<run_id>/    # 每次运行的完整产物
├── third_party/              # 外部依赖 (FastGS 等)
└── requirements.txt          # Python 依赖
```
