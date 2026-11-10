#!/usr/bin/env python3
"""画质对比脚本（离线跑在 GPU 机上，不进服务）。

同一段音频、同一帧预测，分别用 direct / residual / blend 三种贴回策略合成，
输出脸部区域拼图 + 清晰度指标（Laplacian 方差 / 梯度能量），用来选默认参数。

用法：
    python3 qtest.py [输出目录]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from renderer import FPS, Wav2LipRenderer  # noqa: E402

AVATAR = os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg"))
CKPT = os.environ.get("LOCAL_AVATAR_CKPT",
                      str(Path.home() / "models" / "wav2lip" / "wav2lip_gan.pth"))
UPSTREAM = os.environ.get("LOCAL_AVATAR_UPSTREAM", str(Path.home() / "Wav2Lip"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qtest")
OUT.mkdir(parents=True, exist_ok=True)


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def grad_energy(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def main() -> None:
    r = Wav2LipRenderer(AVATAR, CKPT, UPSTREAM)
    y1, y2, x1, x2 = r.box
    print(f"face box = ({y1},{y2},{x1},{x2})  size = {x2 - x1}x{y2 - y1}  "
          f"upscale = {(x2 - x1) / 96:.1f}x")

    # 持续元音（嘴张最大），便于观察嘴部与牙齿区域
    sr = 24000
    t = np.arange(int(sr * 1.0)) / sr
    env = np.minimum(1.0, np.minimum(t / 0.05, 1.0))
    wav = (np.sin(2 * np.pi * 175 * t) * 0.55 * env * 32767).astype("<i2")
    r.push_audio(wav.tobytes())

    pred = r._predict(r._mel())
    variants: dict[str, np.ndarray] = {}

    # 参照：完全不合成的原图
    ref = r.base.copy()

    # (名字, 策略, delta 模糊, 羽化)
    plans = [
        ("direct", "direct", 0.0, 0.14),
        ("res-0.6/0.10", "residual", 0.6, 0.10),
        ("mask-0.6", "residual-mask", 0.6, 0.12),
        ("mask-1.2", "residual-mask", 1.2, 0.12),
        ("mask-0.0", "residual-mask", 0.0, 0.12),
        ("blend-1.2", "blend", 1.2, 0.12),
    ]
    for name, mode, dblur, feather in plans:
        r.compose, r.delta_blur, r.feather = mode, dblur, feather
        patch = np.clip(r._compose(pred), 0, 255)
        out = r.base.copy()
        out[y1:y2, x1:x2] = patch.astype(np.uint8)
        variants[name] = out
        cv2.imwrite(str(OUT / f"frame_{name.replace('/', '_').replace('.', 'p')}.jpg"), out,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    w = 14
    print(f"\n{'variant':{w}s} {'laplacian':>10s} {'grad':>8s} {'mouthΔ':>8s}")
    print(f"{'original':{w}s} {sharpness(ref):10.1f} {grad_energy(ref):8.2f} {'-':>8s}")
    # 嘴部区域（脸框下方 68%~92% 高、45%~92% 宽）与原始帧的平均绝对差 —— 衡量"真动嘴"幅度
    mo_y = (y1 + int((y2 - y1) * 0.68), y1 + int((y2 - y1) * 0.92))
    mo_x = (x1 + int((x2 - x1) * 0.45), x1 + int((x2 - x1) * 0.92))
    for k, v in variants.items():
        d = np.abs(v[mo_y[0]:mo_y[1], mo_x[0]:mo_x[1]].astype(np.float32)
                   - ref[mo_y[0]:mo_y[1], mo_x[0]:mo_x[1]].astype(np.float32))
        print(f"{k:{w}s} {sharpness(v):10.1f} {grad_energy(v):8.2f} {d.mean():8.2f}")

    # 嘴部区域放大拼图：原图 / 各策略
    fy1, fy2 = int((y2 - y1) * 0.66), int((y2 - y1) * 0.94)
    fx1, fx2 = int((x2 - x1) * 0.42), int((x2 - x1) * 0.94)
    my1, my2 = y1 + fy1, y1 + fy2
    mx1, mx2 = x1 + fx1, x1 + fx2
    scale = 3
    tiles = []
    for name, img in [("original", ref)] + list(variants.items()):
        crop = img[my1:my2, mx1:mx2]
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 120), 2, cv2.LINE_AA)
        tiles.append(crop)
    sep = np.full((tiles[0].shape[0], 10, 3), 40, np.uint8)
    row = tiles[0]
    for tp in tiles[1:]:
        row = np.hstack([row, sep, tp])
    cv2.imwrite(str(OUT / "montage_mouth.png"), row)
    print(f"\nwrote {OUT}/montage_mouth.png  ({row.shape[1]}x{row.shape[0]})")


if __name__ == "__main__":
    main()
