# PathAgent — `inference.sh` 用法

`inference.sh` **不吃任何命令列參數**。直接執行：

```bash
bash inference.sh
```

它會依照腳本內的 `RANGES` 陣列（目前 `"0:60"`、`"60:130"`）平行啟動多個 `python pathagent.py` 進程，每段各自負責一段 `--start_idx`/`--end_idx`，log 分別寫到 `result/CAMELYON16/logs/run_<start>_<end>.log`。

若要調整平行度或處理範圍，直接編輯 `inference.sh` 裡的 `RANGES` 陣列（增減/修改 `"start:end"` 項目）或 `COMMON_ARGS`。

## `pathagent.py` 完整參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--plip_lib_path` | 必填 | PLIP library 目錄路徑 |
| `--qwen_ckpt` | 無 | Qwen checkpoint 路徑（非 OpenAI 後端時才需要） |
| `--plip_ckpt` | 必填 | PLIP checkpoint 路徑 |
| `--patho_r1_ckpt` | 必填 | Patho-R1 checkpoint 路徑 |
| `--openai_api_key` | 無 | OpenAI API key（提供則啟用 OpenAI 文字後端） |
| `--openai_model` | `4o-mini` | OpenAI 模型名稱 |
| `--openai_base_url` | 無 | 自訂 OpenAI 相容 base URL（如 Azure 或本地 proxy） |
| `--descriptions_file` | 必填 | patch 描述 JSON 檔路徑 |
| `--questions_file` | 必填 | 問答/VQA 資料集 JSON 檔路徑 |
| `--feature_dir` | 必填 | 影像特徵所在目錄 |
| `--patch_root` | 必填 | 影像 patch 根目錄 |
| `--save_dir` | 必填 | 結果輸出目錄 |
| `--mask_first_retrieval` | flag | 初次檢索時只用腫瘤 mask 過濾 patch |
| `--mask_root` | 無 | mask json 所在目錄 |
| `--dataset_name` | `wsi_vqa` | 資料集名稱（如 `wsi_vqa`、`slidechat`） |
| `--start_idx` | 無 | 處理範圍起始 index（含，0-based） |
| `--end_idx` | 無 | 處理範圍結束 index（不含） |

## 關掉 / 換掉 Flash Attention

PathAgent **目前沒有使用 Flash Attention**：`pathagent.py` 第 178 行載入 Patho-R1 模型時是寫死 `attn_implementation="sdpa"`，沒有 CLI 參數可以切換，也不需要裝 `flash-attn` 套件。

如果要改用其他 attention 實作（例如想測 `flash_attention_2` 或退到 `eager`），需要直接修改 `pathagent.py` 裡這一行：
```python
patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.patho_r1_ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",   # 改成 "flash_attention_2" 或 "eager"
)
```
