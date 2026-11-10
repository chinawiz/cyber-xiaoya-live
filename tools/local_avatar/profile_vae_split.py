#!/usr/bin/env python3
"""VAE 局部解码实验：只解码下半张脸，能否把 decode 耗时砍半。

背景：全图 256×256 decode 单帧 44ms，是 25fps 的硬瓶颈。
但 residual 合成只用嘴部变化，嘴在下半张脸 —— latent 32×32 里对应 y 16~32。
裁掉上半 latent 后 decode 面积减半，理论上耗时同步减半。

风险：decoder 是卷积，裁剪边界缺少上下文，会在切口处产生伪影。
所以这里同时看速度和切口处的误差。
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "MuseTalk"))
from musetalk.models.vae import VAE  # noqa: E402

MODELS = str(Path.home() / "MuseTalk" / "models")
AVATAR = str(Path.home() / "local-avatar" / "avatar.jpg")


def timed(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.monotonic()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.monotonic() - t) / n * 1000, out


def main():
    vae = VAE(model_path=os.path.join(MODELS, "sd-vae"))
    vae.vae = vae.vae.half().eval()

    img = cv2.imread(AVATAR)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    lat_full = vae.get_latents_for_unet(img)[:, :4]      # 只要 4 通道的 ref latent

    ms_full, out_full = timed(lambda: vae.decode_latents(lat_full))
    print(f"full 32x32 -> 256x256      {ms_full:6.1f} ms  shape={out_full.shape}")

    for name, sl in [("下半 y16:32 -> 128x256", (slice(16, 32), slice(0, 32))),
                     ("中部 y12:32 -> 160x256", (slice(12, 32), slice(0, 32))),
                     ("嘴部 y16:32 x8:24", (slice(16, 32), slice(8, 24)))]:
        lat = lat_full[:, :, sl[0], sl[1]].contiguous()
        ms, out = timed(lambda: vae.decode_latents(lat))
        h, w = out.shape[:2]
        print(f"{name:<26s} {ms:6.1f} ms  shape={out.shape}  增速={ms_full / ms:.2f}x")

        if sl[0].start == 16 and sl[1].start == 0:
            # 与全图解码的对应区域比对：看切口附近误差有多大
            ref = out_full[128:256, :, :] if out_full.ndim == 4 else out_full[128:256]
            ref = out_full[0][128:256] if out_full.ndim == 4 else out_full[128:256]
            local = out[0] if out.ndim == 4 else out
            d = np.abs(ref.astype(np.float32) - local.astype(np.float32))
            print(f"   切口 y=128 附近 8 行平均误差: {d[:8].mean():.2f}  "
                  f"中部(y=64 起): {d[8:64].mean():.2f}  整体: {d.mean():.2f}")
            # 存一张对照图
            Path("/tmp/mtest").mkdir(exist_ok=True)
            cv2.imwrite("/tmp/mtest/split-full.png", ref)
            cv2.imwrite("/tmp/mtest/split-half.png", local)
            cv2.imwrite("/tmp/mtest/split-diff.png",
                        np.clip(d.mean(axis=2) * 8, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    main()
