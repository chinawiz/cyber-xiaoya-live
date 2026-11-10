#!/usr/bin/env bash
# 在 GPU 机（192.168.50.98 / DGX Spark）启动本地数字人渲染服务 :9880
set -u
DIR="$HOME/local-avatar"
LOG="$DIR/server.log"
mkdir -p "$DIR"
cd "$DIR" || exit 1

# 用锚定 + 全字匹配，避免 pgrep 把外层 shell 命令行（里面也含这串字符）判成服务进程
if pgrep -f 'python[0-9.]* -u server\.py$' >/dev/null 2>&1; then
  echo "[跳过] 本地渲染服务已在运行。日志: tail -f $LOG"
  exit 0
fi

# MuseTalk 需要 diffusers/transformers，装在它自己的 venv 里（--system-site-packages
# 复用系统 torch，避免再装一份 CUDA 版）。Ditto 也在自己的 venv 里（带 TensorRT）。
PY="python3"
DITTO_VENV="$HOME/ditto-venv/bin/python"
MUSE_VENV="$HOME/MuseTalk/venv/bin/python"
BACKEND="${LOCAL_AVATAR_BACKEND:-ditto}"
if [ "$BACKEND" = "ditto" ] && [ -x "$DITTO_VENV" ]; then
  PY="$DITTO_VENV"
elif [ -x "$MUSE_VENV" ] && [ -f "$HOME/MuseTalk/models/musetalk/pytorch_model.bin" ]; then
  PY="$MUSE_VENV"
fi

# TensorRT 10 在 aarch64 没有 PyPI wheel，是从 deb 解包到 ~/trt 的（不需要 root）
export PYTHONPATH="$HOME/trt/usr/lib/python3.12/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/trt/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# polygraphy/cvt 脚本按 PATH 找，缺了会静默失败
export PATH="$HOME/ditto-venv/bin:$PATH"

[ -f "$DIR/avatar.jpg" ] || echo "[提示] 未找到 $DIR/avatar.jpg，使用代码生成的占位形象；换成自己的正脸照效果更好"

export LOCAL_AVATAR_PUBLIC="${LOCAL_AVATAR_PUBLIC:-http://192.168.50.98:9880}"
export LOCAL_AVATAR_IMAGE="$DIR/avatar.jpg"
export LOCAL_AVATAR_CKPT="${LOCAL_AVATAR_CKPT:-$HOME/models/wav2lip/wav2lip_gan.pth}"
export LOCAL_AVATAR_UPSTREAM="${LOCAL_AVATAR_UPSTREAM:-$HOME/Wav2Lip}"
# 后端：ditto(默认，warp 形变，无贴片感) / musetalk(256² 重绘) / wav2lip(96² 兜底) / mock
# Ditto 会自己决定帧率（模型硬编码 25fps），LOCAL_AVATAR_FPS 只对其它后端生效。
#
# 合成方案（LOCAL_AVATAR_COMPOSE，仅 musetalk/wav2lip 用，默认 blend）：
#   blend        语义蒙版(face-parsing 唇部) + 整块重建图 alpha 合成
#   residual-seg 语义蒙版 + 差值叠加
#   residual     旧方案(椭圆先验)，仅作对照
export LOCAL_AVATAR_FPS="${LOCAL_AVATAR_FPS:-20}"
export LOCAL_AVATAR_BACKEND="$BACKEND"
export LOCAL_AVATAR_MUSETALK_DIR="${LOCAL_AVATAR_MUSETALK_DIR:-$HOME/MuseTalk}"

# Ditto 专有参数：
#   DITTO_STEPS     LMDM 扩散步数（10 = 实测吞吐与画质的平衡点）
#   DITTO_OVERLAP   LMDM 窗口重叠帧数。**这个值同时决定两件事**：
#                   (a) 每次前向摊到多少帧 —— valid = 80 - overlap，越小吞吐越高；
#                   (b) 批次间隔 —— 攒够 valid 帧音频才吐一批，实测 60 → 0.8s 一批。
#                   把它调到批次间隔 × 2 以上，否则攒帧期间时钟一直在走会触发重锚。
#   AV_DELAY        音视频统一延迟（秒）。必须 >= 批次间隔 + 一批的处理时间，
#                   否则脉冲式出帧会让画面周期性落后，靠重锚硬追、观感是突然快进。
# 实测（GB10 / TRT，STEPS=10）：
#   overlap=70 → valid=10 帧/批，a2m 摊到每帧 71ms，只有 14fps
#   overlap=50 → valid=30 帧/批，a2m 24ms/帧，整体 ~21fps（最优）
#   overlap=20 → valid=60 帧/批，批次要攒 2.4s，实测掉到 16fps（批次太稀）
export LOCAL_AVATAR_DITTO_STEPS="${LOCAL_AVATAR_DITTO_STEPS:-10}"
export LOCAL_AVATAR_DITTO_OVERLAP="${LOCAL_AVATAR_DITTO_OVERLAP:-50}"
export LOCAL_AVATAR_AV_DELAY="${LOCAL_AVATAR_AV_DELAY:-2.0}"
export LOCAL_AVATAR_DITTO_DIR="${LOCAL_AVATAR_DITTO_DIR:-$HOME/ditto-talkinghead}"
export LOCAL_AVATAR_DITTO_ROOT="${LOCAL_AVATAR_DITTO_ROOT:-$HOME/ditto-talkinghead/checkpoints/ditto_trt_fast}"

# 可选：手动指定脸框，格式 "y1,y2,x1,x2"（戴墨镜/暗光/侧脸时 Haar 检测不出，用这个兜底）
# 优先级：环境变量 LOCAL_AVATAR_BOX > 同目录 box.txt > 交给服务自己检测
if [ -z "${LOCAL_AVATAR_BOX:-}" ] && [ -f "$DIR/box.txt" ]; then
  LOCAL_AVATAR_BOX=$(tr -d ' \r\n' < "$DIR/box.txt")
fi
if [ -n "${LOCAL_AVATAR_BOX:-}" ]; then export LOCAL_AVATAR_BOX; fi

nohup "$PY" -u server.py > "$LOG" 2>&1 < /dev/null &
echo "[启动] PID=$!  log=$LOG"

# MuseTalk 要 load 3.4G UNet + VAE + whisper，冷启动约 40~90 秒，所以给足 3 分钟
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:9880/health || true)
  if [ "$code" = "200" ]; then
    curl -s http://127.0.0.1:9880/health
    echo
    exit 0
  fi
  sleep 2
done
echo "[失败] 服务未就绪，看日志: tail -50 $LOG"
exit 1
