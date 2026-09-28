#!/bin/bash
# 等待指定 PID 結束。
#
# 以 /proc/<pid> 是否存在判斷，不使用 kill -0 ——「-0」雖不遞送訊號，但指令
# 名稱是 kill，漏打一個字元即成為終止，不該出現在會重複執行的腳本裡。
# 也不使用 pgrep -f <字串>：輪詢指令自身的命令列常含同一字串（工作名稱、
# log 路徑），會匹配到自己而永遠回報「還在跑」。
#
#   用法: waitjob.sh <PID> [輪詢秒數]
#
# 退出碼一律為 0；「程序是否還在」由呼叫端接續檢查產出檔判定 —— 程序結束
# 不等於成功（依 research_methodology.md 階段五）。

PID="${1:?需要 PID}"
INTERVAL="${2:-30}"

if [ ! -d "/proc/$PID" ]; then
    echo "[waitjob] PID $PID 已不存在（可能早已結束或啟動失敗）"
    exit 0
fi

echo "[waitjob] 監看 PID $PID，每 ${INTERVAL}s 檢查"
while [ -d "/proc/$PID" ]; do
    sleep "$INTERVAL"
done
echo "[waitjob] PID $PID 已結束"
