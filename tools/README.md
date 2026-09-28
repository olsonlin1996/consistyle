# tools — 診斷與輔助工具

> 這些工具原本散在 session scratchpad（session 結束即消失），2026-09-28 移入 repo 保存。
> 相關記錄：`../../docs/experiments/`、`../../docs/environment_and_known_issues.md`

---

## `waitjob.sh` — 等待背景工作結束

```bash
PID=$!                    # 啟動時記下
bash tools/waitjob.sh $PID 60
```

以 `/proc/<pid>` 是否存在判斷存活。

**為何不用 `pgrep -f <字串>`**：輪詢指令自身的命令列常含同一字串（工作名稱、log 路徑），
會匹配到自己而永遠回報「還在跑」。此缺陷在 2026-09-21 連續三次造成誤判，最嚴重一次白等
3 小時 20 分。中括號技巧（`name[_]2026`）**同樣不可靠**。

**為何不用 `kill -0`**：`-0` 雖不遞送訊號，但指令名稱是 `kill`，漏打一個字元即成為終止。

⚠️ 程序結束**不等於**成功 —— 呼叫端必須接著查產出檔。

---

## `vanilla_sdxl.py` — 純 SDXL 基準

不含 ConsiStyle 任何機制（無共享注意力、無特徵注入），用於取得「完全不用機制會怎樣」的對照。

設定刻意對齊 `run.py` 的 `load_pipeline()`，使差異只剩「有無 ConsiStyle 機制」。

> **註**：`--run_type batch` 模式**自帶** `original sdxl` 輸出，且為同一次執行、同一組 latents
> 產生，比本腳本更嚴格可比。本腳本適用於 cached 模式或需獨立控制時。

---

## `adherence_screen.py` — 形容詞遵從性篩選

把形容詞用進正式實驗**之前**，先確認基礎模型真的會渲染它。

```bash
python tools/adherence_screen.py           # 全部候選
python tools/adherence_screen.py 2 7 17    # 只跑指定編號
```

檔內含 19 個候選與 2026-09-27 的實測判定，以及五條可複用的經驗法則。

**起因**：D1-3B 指定 `floppy-eared dog`，18 張全數未被遵從，且事後才發現。

---

## 已知的環境變數探針（不在本目錄，但相關）

| 變數 | 作用 | 位置 |
|---|---|---|
| `DUMP_MASKS=<目錄>` | 存下 anchor 遮罩與二值化前的原始注意力圖 | `run.py` |
| `PROBE_INJECT=1` | 累計並印出特徵注入統計 | `attn_utils.py` / `gencli.py` |

兩者**預設關閉**，不影響正常執行（已由「位元相同」驗證）。
