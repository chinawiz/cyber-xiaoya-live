#!/usr/bin/env python3
"""按 0.5s 采样渲染服务状态，看音频到达节奏和渲染进度是否匹配。

要回答的问题：音频是一次性灌进来的（pcm_s 瞬间暴涨）还是实时推过来的？
如果 pcm_s 一上来就跳到全段长度、而 cursor 追不上，那画面就在跳帧。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
URL = "http://192.168.50.98:9880/health"

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 别走系统代理
t0 = time.time()
print(f"{'t':>5} {'pcm_s':>7} {'cursor':>7} {'skipped':>8} {'gen_ms':>7}  帧增量")
prev_frames = None
while time.time() - t0 < SEC:
    with op.open(URL, timeout=5) as r:
        d = json.load(r)
    df = "" if prev_frames is None else f"{d['frames'] - prev_frames:+d}"
    prev_frames = d["frames"]
    print(f"{time.time() - t0:5.1f} {d.get('pcm_s', 0):7.2f} {d.get('cursor', 0):7d} "
          f"{d.get('skipped', 0):8d} {d.get('gen_ms', 0):7.1f}  {df}")
    time.sleep(0.5)
