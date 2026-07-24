# Pathology-CoT — `inference.sh` 用法

```bash
bash inference.sh
```

它會用固定參數在背景呼叫一次 `python pathology-o3/think_ln_classify.py`（worker 1），log 寫到 `worker1.log`。腳本裡還有 worker 2 的區塊被註解掉，可取消註解（並調整 `--start`）來額外開一個平行 worker。

> 注意：目前腳本結尾會檢查 `$STATUS2`，但 worker 2 沒啟用時該變數其實沒有被賦值，這是腳本本身既有的小瑕疵，如實記錄現況，若要修正需自行處理。


## `pathology-o3/think_ln_classify.py` 完整參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `-d`, `--data-dir` | `LNCO2` | 存放 case 資料夾的資料目錄 |
| `-o`, `--output` | 必填 | 輸出資料夾路徑 |
| `-m`, `--model` | `Patho-R1-3B` | 使用的視覺模型名稱 |
| `-m_s`, `--model_summary` | `o4-mini` | 摘要用模型名稱 |
| `-n`, `--num-cases` | `5` | 測試 case 數量 |
| `--qa_file` | `annotation_summary_merged_remove_unknown_exclude.csv` | annotation CSV 檔路徑 |
| `--start` | `0` | qa_datas 切片起始 index（含） |
| `--end` | 無 | qa_datas 切片結束 index（不含），不填跑到底 |
