# slide_seek — `run.sh` 用法

`run.sh` **不吃任何命令列參數**。直接執行：

```bash
bash run.sh
```

它是一個背景啟動器：用固定的 `COMMON_ARGS` 呼叫 `python run_qa_eval.py`，若進程失敗會自動重試（最多 30 次，每次間隔 5 秒），log 寫到 `./logs/instance_a.log`。腳本內還有 instance B、instance C 兩個區塊目前被註解掉，可取消註解並搭配 `--start-from`/`--end-at` 來平行跑不同的 case 區間。


使用 `--use-openai` 時需要在 `.env`（或環境變數）中提供 `OPENAI_API_KEY`。

## `run_qa_eval.py` 完整參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--json` | 必填 | `QA.json` 路徑 |
| `--slides` | 必填 | 存放 `.tif or .svs` slide 檔案的目錄 |
| `--output` | `./qa_results` | 輸出目錄 |
| `--slide-id` | 無 | 只跑指定的單一 slide ID（除錯用） |
| `--no-resume` | flag | 停用從上次中斷處繼續 |
| `--start-from` | `0` | 起始 case index（0-based，用於分割平行任務） |
| `--end-at` | 無 | 結束 case index（不含） |
| `--hf-model` | `Qwen/Qwen3-4B` | HuggingFace LLM 模型 |
| `--use-ollama` | flag | 使用 Ollama 後端 |
| `--ollama-url` | `http://192.168.63.184:11434` | Ollama 服務位址 |
| `--ollama-model` | `qwen3:4b` | Ollama 模型 |
| `--use-openai` | flag | 使用 OpenAI 後端 |
| `--openai-model` | `gpt-4o-mini` | OpenAI 模型 |
| `--patho-model` | `WenchuanZhang/Patho-R1-3B` | 病理模型 |
| `--no-vision-supervisor` / `--no_vision_supervisor` | flag（兩種寫法皆可） | 關閉 vision supervisor agent |
| `--device` | `auto` | 執行裝置 |
| `--load-4bit` / `--load-8bit` | flag | 量化載入模型 |
| `--roi-size` | `896` | ROI 大小 |
| `--max-iter` | `10` | supervisor 最大迭代次數 |
| `--max-explorers` | `3` | 最大平行 explorer agent 數 |

## 關掉 / 換掉 Flash Attention

沒有對應的 CLI 參數，`use_flash_attn` 是寫在 `slideseek/config.py` 的 `PathoModelConfig`，**預設為 `True`**：

```python
@dataclass
class PathoModelConfig:
    ...
    use_flash_attn: bool = True   # 改成 False 即可關掉
```

`slideseek/models/patho_model.py` 載入 Patho-R1 模型時本身就有自動退回機制（`flash_attention_2 → sdpa → 預設`，若載入失敗會自動試下一個），所以就算環境沒裝 `flash-attn` 套件通常也能自動 fallback 成 `sdpa` 正常跑；但如果想直接跳過、不要每次都先嘗試 flash-attn，把上面 `use_flash_attn` 改成 `False` 即可強制走 `sdpa`。
