#!/usr/bin/env bash
# 在 Linux GPU 机（192.168.50.98 / DGX Spark）上启动 OmniVoice :9877
# 用法: bash start_server_linux.sh [模型路径或HF repo id]
set -u

DIR="${OMNIVOICE_DIR:-$HOME/omnivoice-wsl}"
PY="$DIR/.venv/bin/python"
MODEL="${1:-${OMNIVOICE_MODEL:-k2-fsa/OmniVoice}}"
LOG="$DIR/server.log"

[ -x "$PY" ] || { echo "[失败] 缺 Python 环境: $PY"; exit 1; }
[ -f "$DIR/omni_server.py" ] || { echo "[失败] 缺 omni_server.py"; exit 1; }

if pgrep -f "omni_server.py" >/dev/null 2>&1; then
  echo "[跳过] OmniVoice 已在运行。日志: tail -f $LOG"
  exit 0
fi

cd "$DIR" || exit 1
# 该机器可直连 HF；hf-mirror 会 308 跳回 huggingface.co，反而触发 hub 的元数据校验失败
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-$DIR/gradio-tmp}"
mkdir -p "$GRADIO_TEMP_DIR"

echo "[启动] model=$MODEL  log=$LOG"
nohup "$PY" omni_server.py \
  --model "$MODEL" --device cuda --ip 0.0.0.0 --port 9877 \
  --no-asr --no-flashinfer \
  > "$LOG" 2>&1 < /dev/null &
echo "[OK] PID=$!"

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:9877/ || true)
  if [ "$code" = "200" ]; then
    echo "[OK] OmniVoice 就绪 http://127.0.0.1:9877/"
    exit 0
  fi
  sleep 2
done
echo "[提示] 还在加载模型/下载权重，看日志: tail -f $LOG"
