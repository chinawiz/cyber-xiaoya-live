#!/usr/bin/env python3
"""合成变体对比：把「亮度/形状」与「色度」分开处理，验证能否消掉色斑。

观察：说话帧在嘴角出现墨绿色横条，且嘴周一圈明显比周围糊。
怀疑两件事：
  1. 色斑来自模型自身的色度偏差（被逐通道 gain 进一步放大）；
  2. 糊来自 256 VAE 解码，而细节回填被 lipzone 屏蔽掉了嘴部核心。
所以这里做三组变体：
  A  current      —— 现方案
  B  yonly        —— 亮度取合成结果，色度完全取原图（消灭一切色斑）
  C  yclamp       —— 色度取生成，但相对原图的偏移做限幅（保留牙齿变白的能力）
  D  yonly+usm    —— B 加核心区锐化，补回清晰度

用法：OUT=/tmp/probe3 FRAME=24 python mprobe3.py
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("OUT", "/tmp/mprobe3"))
OUT.mkdir(parents=True, exist_ok=True)

boxf = Path.home() / "local-avatar" / "box.txt"
if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
    os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()

FRAMES = [int(x) for x in os.environ.get("FRAMES", "24,60,120").split(",")]
CLAMP = float(os.environ.get("CLAMP", "5.0"))


def composite(r, gen, chroma="gen", clamp=CLAMP, usm=0.0):
    """按快速路径（只算下半）合成，返回整帧 uint8。"""
    out = r.base.copy()
    x0, y0, x1, y1 = r.crop
    w, h = x1 - x0, y1 - y0
    half = r.half
    roi = out[y0:y1, x0:x1].astype(np.float32)

    sub = gen
    if sub.shape[0] != 16:
        sub = cv2.resize(sub, (256, 256 - r.split), interpolation=cv2.INTER_LANCZOS4)
    gen_lo = cv2.resize(sub.astype(np.float32), (w, half), interpolation=cv2.INTER_LANCZOS4)
    if r.color_fix:
        gen_lo = gen_lo * r.gain_lo + r.bias_lo
    m = r.mask_lo
    lo = gen_lo * m + roi[half:] * (1.0 - m)
    if r.detail_amt > 0:
        lo = lo + r.detail_pack_lo

    if chroma != "gen":
        ycc_gen = cv2.cvtColor(np.clip(lo, 0, 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
        ycc_org = cv2.cvtColor(np.clip(roi[half:], 0, 255).astype(np.float32),
                               cv2.COLOR_BGR2YCrCb)
        d = ycc_gen[..., 1:] - ycc_org[..., 1:]
        if chroma == "clamp":
            d = np.clip(d, -clamp, clamp)
        y = ycc_gen[..., :1]
        cc = ycc_org[..., 1:] if chroma == "yonly" else ycc_org[..., 1:] + d
        lo = cv2.cvtColor(np.concatenate([y, cc], axis=2), cv2.COLOR_YCrCb2BGR)

    if usm > 0:
        core = cv2.resize(r.core[..., 0], (w, half), interpolation=cv2.INTER_LINEAR)[..., None]
        blur = cv2.GaussianBlur(lo, (0, 0), 1.1)
        lo = lo + usm * (lo - blur) * core

    merged = roi.copy()
    merged[half:] = lo
    out[y0:y1, x0:x1] = np.clip(merged, 0, 255).astype(np.uint8)
    return out


def cell(im, tag, y0, y1, x0, x1, fx):
    c = im[y0:y1, x0:x1]
    c = cv2.resize(c, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC).copy()
    cv2.putText(c, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)
    return c


def main() -> None:
    from musetalk_renderer import MuseTalkRenderer

    with wave.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        pcm = f.readframes(f.getnframes())
    r = MuseTalkRenderer(
        os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS", str(Path.home() / "MuseTalk" / "models")))
    r.push_audio(pcm)
    print(f"[probe] crop={r.crop} half={r.half} split={r.split}", flush=True)

    by1, by2, bx1, bx2 = r.box
    y0 = by1 + int((by2 - by1) * 0.42)
    y1 = by2 + int((by2 - by1) * 0.20)
    x0, x1 = max(0, bx1 - 34), min(r.base.shape[1], bx2 + 34)
    fx = float(os.environ.get("FX", "3.6"))

    variants = [
        ("A current", dict(chroma="gen")),
        ("B yonly", dict(chroma="yonly")),
        ("C yclamp", dict(chroma="clamp")),
        ("D yonly+usm", dict(chroma="yonly", usm=0.6)),
    ]
    for fr in FRAMES:
        gen = r._infer(r._clip(fr))
        cells = [cell(r.base, "ORIG", y0, y1, x0, x1, fx)]
        for tag, kw in variants:
            cells.append(cell(composite(r, gen, **kw), tag, y0, y1, x0, x1, fx))
        row = np.hstack(cells)
        cv2.imwrite(str(OUT / f"f{fr}.png"), row)
        print(f"[probe] f{fr} -> {row.shape}", flush=True)


if __name__ == "__main__":
    main()
