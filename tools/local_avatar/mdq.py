#!/usr/bin/env python3
"""诊断：模型「静音重建」与原图的差距有多大。

MuseTalk 的训练目标是「在 latent 里补完嘴部」，理想情况下零音频条件时
它应该原样还原输入的那张脸。如果重建和原图有不小差距（唇形变厚、
唇线位移、整体发虚），那么：
    · 整块替换 → 静音帧也会把脸画脏（色块/位移肉眼可见）
    · 差值叠加 → 差值里混着"重建偏差"，在嘴角生出亮带
无论哪种合成方式都会被这个偏差拖累，所以换合成算法之前必须先量它。
输出 /tmp/mdq/{original,idle,diff,overlay}.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("OUT", "/tmp/mdq"))
OUT.mkdir(parents=True, exist_ok=True)

boxf = Path.home() / "local-avatar" / "box.txt"
if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
    os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()


def main() -> None:
    from musetalk_renderer import MuseTalkRenderer

    r = MuseTalkRenderer(
        os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS",
                       str(Path.home() / "MuseTalk" / "models")))
    x0, y0, x1, y1 = r.crop
    w, h = x1 - x0, y1 - y0

    if r.idle_low is None:
        torch = r.torch
        with torch.no_grad():
            z = torch.zeros((1, 10, r.n_layers, 384), dtype=r.dtype, device=r.device)
            r.idle_low = r._infer(z).astype(np.float32)

    full = r.face_256.astype(np.float32)
    full[r.split:] = r.idle_low if r.split else r.idle_low.reshape(g for g in (-1,))
    if r.split:
        rebuild = cv2.resize(full, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        rebuild = cv2.resize(r.idle_low, (w, h), interpolation=cv2.INTER_LANCZOS4)

    orig = r.face_bgr.astype(np.float32)
    d = np.abs(rebuild - orig)
    # 只看融合区（模型真正会写的地方），以及其中的唇部核心
    m = (r.mask[..., 0] > 0.5)
    core = (r.core[..., 0] > 0.5)
    print(f"[{r.version}] 静音重建 vs 原图: "
          f"全图 mean={d.mean():.2f} p99={np.percentile(d, 99):.1f}", flush=True)
    if m.any():
        print(f"     融合区 mean={d[m].mean():.2f} "
              f"p99={np.percentile(d[m], 99):.1f}", flush=True)
    if core.any():
        print(f"     唇部核心 mean={d[core].mean():.2f} "
              f"p99={np.percentile(d[core], 99):.1f}", flush=True)
    # 结构相似度（灰度），越高越接近原图
    g1 = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(np.clip(rebuild, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    try:
        from skimage.metrics import structural_similarity as ssim
        v = ssim(g1, g2)
        print(f"     SSIM={v:.4f}", flush=True)
    except Exception:  # noqa: BLE001
        pass

    def save(name, img):
        im = cv2.resize(img.astype(np.uint8), None, fx=3.0, fy=3.0,
                        interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(OUT / f"{name}.png"), im)

    save("original", orig)
    save("idle", np.clip(rebuild, 0, 255))
    heat = np.clip(d.mean(axis=2) * 6, 0, 255)
    heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / "diff.png"), cv2.resize(heat, None, fx=3.0, fy=3.0,
                                                  interpolation=cv2.INTER_NEAREST))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
