#!/usr/bin/env python3
"""从本地渲染服务的 MJPEG 流连续抓帧，拼成一张嘴型序列图。

用途：验证口型是否真的随音频变化。截图工具和 TTS 的时机永远对不齐，
直接从流里抓连续帧才能看到口型序列。用法：

    .venv/bin/python tools/local_avatar/grab_stream.py [秒数] [输出目录]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

URL = sys.argv[3] if len(sys.argv) > 3 else "http://192.168.50.98:9880/stream"
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/grab")
OUT.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.trust_env = False                    # 本地地址不要走 HTTP_PROXY
resp = s.get(URL, stream=True, timeout=15)
buf = b""
imgs: list[np.ndarray] = []
t_end = time.time() + SEC
try:
    for chunk in resp.iter_content(32768):
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")
            if a < 0:
                buf = buf[-2:]
                break
            b = buf.find(b"\xff\xd9", a + 2)
            if b < 0:
                buf = buf[a:]
                break
            jpg = buf[a:b + 2]
            buf = buf[b + 2:]
            im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                imgs.append(im)
        if time.time() > t_end:
            break
finally:
    resp.close()

print(f"抓到 {len(imgs)} 帧 / {SEC}s = {len(imgs) / max(SEC, .01):.1f} fps")
if not imgs:
    raise SystemExit("没抓到帧")

h, w = imgs[0].shape[:2]
print(f"帧尺寸 {w}x{h}")

boxf = Path.home() / "local-avatar" / "box.txt"
env_box = os.environ.get("LOCAL_AVATAR_BOX", "").strip()
if env_box:
    y1, y2, x1, x2 = (int(v) for v in env_box.split(","))
elif boxf.is_file():
    y1, y2, x1, x2 = (int(v) for v in boxf.read_text().strip().split(","))
else:
    y1, y2, x1, x2 = int(h * 0.12), int(h * 0.72), int(w * 0.24), int(w * 0.76)

n = 12
# 按"口型动作幅度"挑帧：均匀抽样经常抽到静音段，看不出在不在说话。
# 先算每帧嘴部区域相对首帧的差异，再和均匀抽样混合取。
mouth = [im[y1 + int((y2 - y1) * 0.62):y2, x1:x2].astype(np.float32) for im in imgs]
ref = np.median(np.stack(mouth[:5]), axis=0)
act = np.array([float(np.abs(m - ref).mean()) for m in mouth])
print(f"动作幅度: max={act.max():.2f} p90={np.percentile(act, 90):.2f} "
      f"中位={np.median(act):.2f}")
uni = [int(i * (len(imgs) - 1) / max(1, n - 1)) for i in range(n)]
top = list(np.argsort(act)[-n:][::-1])
idx = sorted(set(uni + top))
tiles = []
for i in idx:
    crop = imgs[i][y1:y2, x1:x2]
    if crop.size == 0:
        continue
    sc = max(2, int(210 / max(1, crop.shape[0])))
    crop = cv2.resize(crop, (crop.shape[1] * sc, crop.shape[0] * sc),
                      interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 120), 2, cv2.LINE_AA)
    tiles.append(crop)

while len(tiles) % 4:
    tiles.append(tiles[-1])
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
sep = np.full((10, rows[0].shape[1], 3), 30, np.uint8)
out = rows[0]
for rw in rows[1:]:
    out = np.vstack([out, sep, rw])
cv2.imwrite(str(OUT / "mouth-seq.png"), out)
print(f"wrote {OUT}/mouth-seq.png ({out.shape[1]}x{out.shape[0]})")
cv2.imwrite(str(OUT / "full.jpg"), imgs[len(imgs) // 2],
            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
