#!/usr/bin/env python3
"""单帧三联对比：原图下脸 / 模型生成 / delta 热力图（2 倍放大）。

用来判断模型的原始输出到底长什么样 —— 是被上采样糊掉了，
还是模型本身就生成了一张"别脸"。mdump 的网格图太小，看不清细节。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("/tmp/mdump")
OUT.mkdir(parents=True, exist_ok=True)


def main():
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
    import wave as _wav
    with _wav.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        r.push_audio(f.readframes(f.getnframes()))

    for idx in (6, 12, 24):
        gen = r._infer(r._clip(idx))
        ref = r.face_low.astype(np.uint8)
        delta = gen.astype(np.float32) - r.face_low
        mag = np.abs(delta).mean(axis=2)
        heat = cv2.applyColorMap(np.clip(mag * 5, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)

        tiles = []
        for name, im in (("ref(low)", ref), (f"gen f{idx}", gen), ("delta", heat)):
            big = cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 120), 2, cv2.LINE_AA)
            tiles.append(big)
        sep = np.full((tiles[0].shape[0], 10, 3), 30, np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.hstack([row, sep, t])
        cv2.imwrite(str(OUT / f"triple-f{idx}.png"), row)
        print(f"f{idx}: delta mean={mag.mean():.2f} p95={np.percentile(mag, 95):.1f} "
              f"max={mag.max():.1f}  gen_mean={gen.mean():.1f} ref_mean={ref.mean():.1f}")


if __name__ == "__main__":
    main()
