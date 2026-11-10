#!/usr/bin/env python3
"""离线出「口型序列图」：同一段真实语音，逐帧 _infer + _paste，拼成网格。

用来分开两个问题：
  · 模型本身会不会动嘴（看 _infer 的原始输出）
  · 合成后看不看得出来（看 _paste 的成品）
服务端口型不动时，先用这个确认是合成太保守，还是音频没喂进去。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mseq")
OUT.mkdir(parents=True, exist_ok=True)


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
    import wave as _wav
    with _wav.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        r.push_audio(f.readframes(f.getnframes()))
    print(f"audio {r.pcm.seconds:.1f}s  crop={r.crop}")

    y1, y2, x1, x2 = r.box
    my1 = y1 + int((y2 - y1) * 0.55)

    ids = list(range(6, 130, 10))          # 0.24s ~ 5.2s
    tiles = []
    for idx in ids:
        gen = r._infer(r._clip(idx))
        r.compose = "residual"
        jpg = r._paste(gen)
        im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        crop = im[my1:y2, x1:x2]
        sc = max(2, int(200 / max(1, crop.shape[0])))
        crop = cv2.resize(crop, (crop.shape[1] * sc, crop.shape[0] * sc),
                          interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"f{idx}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 120), 2, cv2.LINE_AA)
        tiles.append(crop)
        d = (gen.astype(np.float32) -
             (r.idle_low if r.idle_low is not None else r.face_low))
        print(f"f{idx}: |delta| mean={np.abs(d).mean():5.2f} "
              f"max={np.abs(d).max():6.1f}")

    while len(tiles) % 4:
        tiles.append(tiles[-1])
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sep = np.full((10, rows[0].shape[1], 3), 30, np.uint8)
    out = rows[0]
    for rw in rows[1:]:
        out = np.vstack([out, sep, rw])
    cv2.imwrite(str(OUT / "seq.png"), out)
    print(f"wrote {OUT}/seq.png ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
