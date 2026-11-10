#!/usr/bin/env python3
"""逐帧 dump MuseTalk 的原始输出，定位"嘴不动/画面糊"。

不是看合成结果，而是把模型直接吐出来的 128×256 图、delta、
以及它对应的输入参考脸都存下来，逐帧拼成一条时间轴，
这样能一眼判断：是模型没驱动起来，还是合成环节把变化吃掉了。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mdump")
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

    wav = os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")
    import wave as _wav
    with _wav.open(wav) as f:
        pcm = f.readframes(f.getnframes())
    r.push_audio(pcm)
    print(f"audio {r.pcm.seconds:.2f}s")

    frames = [0, 6, 12, 18, 24, 30, 40, 50, 60, 80, 100, 140]
    tiles_raw, tiles_delta = [], []
    stats = []
    for idx in frames:
        img = r._infer(r._clip(idx))
        delta = img.astype(np.float32) - r.face_low
        mag = np.abs(delta).mean()
        peak = float(cv2.GaussianBlur(np.abs(delta).mean(axis=2), (0, 0), 3.2).max())
        stats.append((idx, mag, peak, float(img.mean()), int(img.max())))

        vis = img.copy()
        cv2.putText(vis, f"f{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 120), 2, cv2.LINE_AA)
        tiles_raw.append(vis)

        dv = np.clip(np.abs(delta).mean(axis=2) * 6, 0, 255).astype(np.uint8)
        dv = cv2.cvtColor(dv, cv2.COLOR_GRAY2BGR)
        cv2.putText(dv, f"f{idx} peak{peak:.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 120), 1, cv2.LINE_AA)
        tiles_delta.append(dv)

    # 参考脸（模型输入的下半部分）
    ref = r.face_256.astype(np.uint8)
    ref = cv2.resize(ref, (256, 128))     # 压扁只为便于并排展示

    def grid(tiles, cols=6):
        h, w = tiles[0].shape[:2]
        rows = []
        for i in range(0, len(tiles), cols):
            chunk = tiles[i:i + cols]
            while len(chunk) < cols:
                chunk.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack([np.pad(c, ((0, 0), (0, 4), (0, 0))) for c in chunk]))
        return np.vstack(rows)

    cv2.imwrite(str(OUT / "raw.png"), grid(tiles_raw))
    cv2.imwrite(str(OUT / "delta.png"), grid(tiles_delta))
    cv2.imwrite(str(OUT / "ref.png"), ref)

    print(f"\n{'frame':>6s} {'|delta|mean':>11s} {'peak':>7s} {'img_mean':>9s} {'img_max':>8s}")
    for idx, mag, peak, mean, mx in stats:
        print(f"{idx:>6d} {mag:>11.2f} {peak:>7.1f} {mean:>9.1f} {mx:>8d}")
    print(f"\nwrote {OUT}/raw.png delta.png ref.png")


if __name__ == "__main__":
    main()
