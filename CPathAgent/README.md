# CPathAgent — `run_main.sh` 用法

`run_main.sh` **不吃任何命令列參數**。直接執行：

```bash
bash run_main.sh
```

它會在背景平行啟動多個 segment，各自用同一組固定參數呼叫 `python evaluate_camelyon16.py`，只有 `--start`/`--end` 切片範圍不同（目前 segment 1: `800-1000`、segment 2: `1000-1280`；segment 3 已被註解掉）。每個 segment 的輸出各自寫到 `logs/seg<N>.log`。

若要調整平行度或處理範圍，直接編輯 `run_main.sh` 裡的 `COMMON` 陣列、`--start`/`--end`，或取消註解 segment 3。

## `evaluate_camelyon16.py` 完整參數

### Dataset / IO
| 參數 | 預設值 | 說明 |
|---|---|---|
| `--dataset` | 必填 | `roi_qa_dataset.json` 路徑 |
| `--slide-dir` | 必填 | slide 檔案（.tif/.svs/…）所在目錄 |
| `--qa-set` | `CAMELYON16` | 輸出子目錄名稱 |
| `--output-dir` | `./output` | 輸出根目錄 |
| `--tissue` | `breast` | 傳給 pipeline 的組織型態 |
| `--overwrite` | flag | 即使輸出檔已存在也重跑 |
| `--start` | `0` | 資料集切片起始 index（含） |
| `--end` | 無 | 資料集切片結束 index（不含），不填跑到底 |
| `--save-images` | flag | 儲存縮圖/region/nav overlay 圖片 |
| `--quiet` | flag | 關閉詳細輸出 |

### WSI processing
| 參數 | 預設值 | 說明 |
|---|---|---|
| `--patch-size` | `512` | 縮圖 patch 大小（像素） |
| `--max-nav-steps` | `8` | 最大導覽步數 |

### Model backend
| 參數 | 預設值 | 說明 |
|---|---|---|
| `--backend` | `ollama` | 文字與視覺共用後端，選 `ollama`/`hf`/`openai` |
| `--text-backend` | 無 | 只覆蓋文字任務後端（開啟混合模式） |
| `--vlm-backend` | 無 | 只覆蓋視覺任務後端（開啟混合模式） |
| `--ollama-host` | `http://localhost:11434` | Ollama 服務位址 |
| `--text-ollama` | `qwen3:4b` | Ollama 文字模型 |
| `--vlm-ollama` | `patho_r1` | Ollama 視覺模型 |
| `--text-hf` | `Qwen/Qwen3-4B` | HuggingFace 文字模型 |
| `--vlm-hf` | `patho_r1` | HuggingFace 視覺模型 |
| `--hf-device` | `cuda` | HF 模型執行裝置 |
| `--hf-dtype` | `float16` | `float16`/`bfloat16`/`float32` |
| `--hf-8bit` / `--hf-4bit` | flag | 量化載入 |
| `--hf-flash-attn` | flag | 使用 Flash Attention 2（需裝 flash-attn，僅 fp16/bf16） |
| `--no-sdpa` | flag | 停用 SDPA attention kernel（退回 eager） |
| `--no-tf32` | flag | 停用 tensor core 的 TF32 matmul |
| `--hf-compile` | flag | 對 VLM 套用 `torch.compile`（有暖機成本，之後較快） |
| `--text-openai` | `gpt-4o-mini` | OpenAI 文字模型 |
| `--vlm-openai` | 空字串（預設 gpt-4o） | OpenAI 視覺模型 |
| `--openai-api-key` | 無（預設吃環境變數 `OPENAI_API_KEY`） | OpenAI API key |
| `--openai-base-url` | 無 | 自訂 OpenAI base URL（如 Azure 或相容 API） |
| `--max-tokens` | `2048` | 最大 token 數 |
| `--temperature` | `0.1` | 取樣溫度 |
| `--thinking` | flag | 啟用 Qwen3 思考鏈 token |
