#!/usr/bin/env python3
"""模拟服务端的实时节奏，复现"流式喂音频时口型不动"。

离线 mseq.py 一次性 push 全段音频，_ensure_feat 的窗口总能凑满 2.5 秒；
线上音频只领先画面 0.2 秒，前瞻窗口凑不满 → 特征永不刷新 → _clip 补零 →
UNet 拿不到音频条件，画面就是一张静止的糊脸。这个脚本按 40ms 一块喂音频、
按 25fps 渲染，能稳定复现线上行为。
"""
from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mstream")
OUT.mkdir(parents=True, exist_ok=True)
SR = 24000


def main() -> None:
    boxf = Path.home() / "local-avatar" / "box.txt"
    if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
        os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()
    from musetalk_renderer import MuseTalkRenderer

    r = MuseTalkRenderer(os.environ.get("LOCAL_AVATAR_IMAGE",
                                       str(Path.home() / "local-avatar" / "avatar.jpg")),
                         os.environ.get("LOCAL_AVATAR_MUSETALK_DIR",
                                        str(Path.home() / "MuseTalk")),
                         os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS",
                                        str(Path.home() / "MuseTalk" / "models")))
    with wave.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        raw = f.readframes(f.getnframes())
    total = len(raw) / (SR * 2)
    print(f"音频 {total:.1f}s  渲染 {total:.0f}s@25fps")

    chunk = int(SR * 2 * 0.04)          # 40ms 一块
    pos = chunk * 13                    # 先灌 520ms，模拟线上 lead=0.5
    pos = min(pos, len(raw))
    y1, y2, x1, x2 = r.box
    my1 = y1 + int((y2 - y1) * 0.55)
    idle = r.idle_low
    frames, acts = [], []
    t0 = time.monotonic()
    n = 0
    while pos < len(raw):
        # 服务端节奏：每帧到达一块 40ms 音频，稳定领先画面 500ms
        r.push_audio(raw[pos:pos + chunk])
        pos += chunk
        t = time.monotonic()
        gen = r._infer(r._clip(r.cursor))
        r.cursor += 1
        r.gen_ms = 0.9 * r.gen_ms + 0.1 * (time.monotonic() - t) * 1000
        d = np.abs(gen.astype(np.float32) - (idle if idle is not None else r.face_low))
        acts.append(float(d.mean()))
        if n % 25 == 0:
            frames.append((n, gen.copy()))
        n += 1
        sleep = 0.04 - (time.monotonic() - t)
        if sleep > 0:
            time.sleep(sleep)

    acts = np.array(acts)
    print(f"共 {n} 帧  耗时 {time.monotonic() - t0:.1f}s  "
          f"gen_ms≈{r.gen_ms:.1f}")
    print(f"|gen-idle| 均值 {acts.mean():.2f}  p50 {np.median(acts):.2f}  "
          f"p90 {np.percentile(acts, 90):.2f}  max {acts.max():.2f}")
    print(f"feat_refresh={r.feat_refresh}  ← 1 表示只编码过一次，之后没再更新")

    tiles = []
    for no, gen in frames[:12]:
        big = cv2.resize(gen, (gen.shape[1] * 2, gen.shape[0] * 2),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"f{no}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 120), 2, cv2.LINE_AA)
        tiles.append(big)
    while len(tiles) % 4:
        tiles.append(tiles[-1])
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sep = np.full((10, rows[0].shape[1], 3), 30, np.uint8)
    out = rows[0]
    for rw in rows[1:]:
        out = np.vstack([out, sep, rw])
    cv2.imwrite(str(OUT / "seq.png"), out)
    print(f"wrote {OUT}/seq.png")


if __name__ == "__main__":
    main()
