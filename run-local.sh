#!/usr/bin/env bash
# macOS 本机一键启动「赛博小雅直播版」：GPU 本地数字人 + SSH 转发 TTS + 本地桥 + 舞台页
# 用法: bash run-local.sh            （前台运行，Ctrl+C 停桥）
#       bash run-local.sh --no-browser
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
GPU="${GPU_HOST:-spark-h3}"
BROWSER=1
[ "${1:-}" = "--no-browser" ] && BROWSER=0

# 0) GPU 机上的本地数字人渲染服务 :9880（config.vidu.provider=local 时必需）
RENDER=$("$ROOT/.venv/bin/python" -c \
  "import json;d=json.load(open('$ROOT/config.json'));print(d['vidu']['host'] if d['vidu'].get('provider')=='local' else '')" 2>/dev/null || true)
if [ -n "$RENDER" ]; then
  if curl -s -o /dev/null --max-time 2 "$RENDER/health"; then
    echo "[0/3] 本地渲染服务已就绪 $RENDER"
  else
    echo "[0/3] 拉起 GPU 机本地渲染服务 $RENDER"
    ssh -o BatchMode=yes "$GPU" 'cd ~/local-avatar && bash start.sh' \
      || echo "      [警告] 渲染服务没起来，舞台页会黑屏"
  fi
fi

# 1) 把 GPU 机的 9877 映射到本机 127.0.0.1:9877（桥默认就拨这个地址）
if ! nc -z 127.0.0.1 9877 2>/dev/null; then
  echo "[1/3] 建立 SSH 端口转发 $GPU:9877 -> 127.0.0.1:9877"
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -L 9877:127.0.0.1:9877 "$GPU" &
  SSH_PID=$!
  trap 'kill $SSH_PID 2>/dev/null || true' EXIT
  sleep 2
  if nc -z 127.0.0.1 9877 2>/dev/null; then
    echo "      TTS 转发就绪"
  else
    echo "      [警告] TTS 暂不可用（GPU 侧服务没起来），桥会以无声模式运行"
  fi
else
  echo "[1/3] 127.0.0.1:9877 已在监听，跳过转发"
fi

# 2) 启动桥
echo "[2/3] 启动桥 http://127.0.0.1:8795/stage"
if [ "$BROWSER" = "1" ]; then
  exec "$ROOT/.venv/bin/python" -m src.main
else
  exec "$ROOT/.venv/bin/python" -m src.main --no-browser
fi
