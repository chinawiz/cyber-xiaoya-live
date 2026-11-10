#!/usr/bin/env python3
"""合成方案参数扫描：crop 边长 × 合成方式，同一帧横向对比。

要同时看三件事：
  1. 说话帧的嘴部会不会"浮"（边缘/色差/贴片感）
  2. 静音帧会不会把脸画脏（整块替换方案的固有问题：
     即使没有音频条件，模型也会重画一遍下半脸，色偏糊全部落在脸上）
  3. crop 边长对生成质量的影响 —— 脸在 256 输入里的占比越小，模型越陌生

用法：
    EXPANDS=1.12,1.3,1.5 python mblend2.py
输出 /tmp/mscan/<expand>/<compose>.png 及总览 scan.png
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("OUT", "/tmp/mscan"))
OUT.mkdir(parents=True, exist_ok=True)

boxf = Path.home() / "local-avatar" / "box.txt"
if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
    os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()

FRAME = int(os.environ.get("FRAME", "205"))
COMPOSES = [c for c in os.environ.get("COMPOSES", "residual,residual-seg,blend").split(",")]


def main() -> None:
    from musetalk_renderer import MuseTalkRenderer

    wavp = os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")
    with wave.open(wavp) as f:
        pcm = f.readframes(f.getnframes())

    rows = []
    for expand in [float(x) for x in os.environ.get("EXPANDS", "1.12,1.3,1.5").split(",")]:
        os.environ["LOCAL_AVATAR_CROP_EXPAND"] = str(expand)
        r = MuseTalkRenderer(
            os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg")),
            os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk")),
            os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS",
                           str(Path.home() / "MuseTalk" / "models")))
        r.push_audio(pcm)
        if r.idle_low is None:                       # blend 不自动算，这里手动补
            torch = r.torch
            with torch.no_grad():
                z = torch.zeros((1, 10, r.n_layers, 384), dtype=r.dtype, device=r.device)
                r.idle_low = r._infer(z).astype(np.float32)

        gen = r._infer(r._clip(FRAME))
        # 静音帧：全零条件，等价于"这个人闭嘴时模型画成什么样"
        torch = r.torch
        with torch.no_grad():
            z = torch.zeros((1, 10, r.n_layers, 384), dtype=r.dtype, device=r.device)
            gen_idle = r._infer(z)

        by1, by2, bx1, bx2 = r.box
        cy = by1 + int((by2 - by1) * 0.40)
        lx, rx = max(0, bx1 - 26), min(r.base.shape[1], bx2 + 26)

        cells = []
        for tag, g in (("IDLE  original", r.base),):
            cells.append(_cell(g, cy, by2 + 14, lx, rx, tag))
        for compose in COMPOSES:
            r.compose = compose
            for tag, g in ((f"TALK {compose}", gen), (f"IDLE {compose}", gen_idle)):
                jpg = r._paste(g)
                im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                cells.append(_cell(im, cy, by2 + 14, lx, rx, tag))
        r.compose = "blend"
        cv2.imwrite(str(OUT / f"expand{expand:.2f}.png"), np.hstack(cells))
        print(f"expand={expand} crop={r.crop} -> expand{expand:.2f}.png", flush=True)
        rows.append(str(OUT / f"expand{expand:.2f}.png"))
        del r
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # 纵向拼接各 crop 尺寸
    ims = [cv2.imread(p) for p in rows]
    if len(ims) > 1:
        w = min(i.shape[1] for i in ims)
        cv2.imwrite(str(OUT / "scan.png"),
                    np.vstack([i[:, :w] for i in ims]))
    print(f"-> {OUT}/scan.png", flush=True)


def _cell(im: np.ndarray, y0: int, y1: int, x0: int, x1: int, tag: str) -> np.ndarray:
    c = im[y0:y1, x0:x1].copy()
    fx = float(os.environ.get("FX", "3.0"))
    c = cv2.resize(c, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    cv2.putText(c, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)
    return c


if __name__ == "__main__":
    main()
