# API 与 gpt-image-2 ERP 快速测试报告

- **生成时间（UTC）**：2026-05-01T13:26:59.842126+00:00
- **机器**：黑黑黑  Python 3.12.10
- **凭据**：已解析（可调用）；来源 **`yaml_secrets.local`**（`env` / `yaml_secrets.local` / `key_file` / `none`）

## 1. 摘要

- **ERP 输出**：`outputs/_quick_test/erp_rgb_gpt_image2.png`  请求尺寸 `1536x1024`  实际 `[1536, 1024]`  耗时 **30030.0 ms**

## 2. 子项结果

| 步骤 | ok | 说明 |
|------|----|------|
| official_chat_completions | 是 | model=gpt-4o-mini ms=3433.9 reply=OK |
| official_images_generate_erp | 是 | 1536x1024 -> [1536, 1024] in 30030.0 ms |

## 3. 原始 JSON（节选）

```json
{
  "meta": {
    "utc": "2026-05-01T13:26:59.842126+00:00",
    "node": "黑黑黑",
    "python": "3.12.10",
    "had_key": true,
    "key_source": "yaml_secrets.local",
    "chatfire_cli_flag": false,
    "erp_no_size_flag": false,
    "openai_base_url": "https://api.chatfire.cn/v1",
    "image_model": "gpt-image-2",
    "text_model_ping": "gpt-4o-mini"
  },
  "rows": [
    {
      "name": "official_chat_completions",
      "ok": true,
      "error": null,
      "note": "model=gpt-4o-mini ms=3433.9 reply=OK"
    },
    {
      "name": "official_images_generate_erp",
      "ok": true,
      "note": "1536x1024 -> [1536, 1024] in 30030.0 ms"
    }
  ],
  "erp": {
    "ok": true,
    "path": "outputs/_quick_test/erp_rgb_gpt_image2.png",
    "size_requested": "1536x1024",
    "out_wh": [
      1536,
      1024
    ],
    "ms": 30030.0
  },
  "overall_ok": true
}
```

## 4. 复现命令

```powershell
$env:OPENAI_API_KEY = "<你的密钥>"
python scripts/quick_test_report_and_erp.py
```

## 5. 多通路细测（可选）

更细的 `images.generate` / `images.edit` / OpenRouter 等分项，请运行 `python scripts/test_api_pathways.py --out-json outputs/_api_pathway_test.json`。
