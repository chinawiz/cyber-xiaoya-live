#!/usr/bin/env python3
"""VAE 解码加速实验：找出 GB10 上 256×256 decode 的最快配置。

单帧 56ms 里有 44ms 花在 VAE decode 上（UNet 只 11.5ms），
batch=1 的小算子逐个 launch 是主要浪费，所以重点试 kernel 融合 / 内存格式。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "MuseTalk"))
from musetalk.models.vae import VAE  # noqa: E402

MODELS = str(Path.home() / "MuseTalk" / "models")


def bench(vae, lat, name, warm=5, n=20):
    for _ in range(warm):
        vae.decode_latents(lat)
    torch.cuda.synchronize()
    t = time.monotonic()
    for _ in range(n):
        out = vae.decode_latents(lat)
    torch.cuda.synchronize()
    ms = (time.monotonic() - t) / n * 1000
    bad = "NaN!" if not np.isfinite(out).all() else "ok"
    print(f"{name:<28s} {ms:7.1f} ms   {bad}", flush=True)
    return ms


def main():
    torch.backends.cudnn.benchmark = True
    vae = VAE(model_path=os.path.join(MODELS, "sd-vae"))
    vae.vae = vae.vae.half().eval()
    lat = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16) * 0.5

    bench(vae, lat, "baseline fp16")

    vae.vae = vae.vae.to(memory_format=torch.channels_last)
    bench(vae, lat, "channels_last")

    try:
        vae.vae.decoder = torch.compile(vae.vae.decoder, mode="reduce-overhead")
        bench(vae, lat, "compile reduce-overhead", warm=3, n=20)
    except Exception as e:  # noqa: BLE001
        print(f"compile reduce-overhead 失败: {type(e).__name__}: {e}")

    try:
        vae.vae = VAE(model_path=os.path.join(MODELS, "sd-vae")).vae
        vae.vae = vae.vae.half().eval().to(memory_format=torch.channels_last)
        vae.vae.decoder = torch.compile(vae.vae.decoder, mode="default")
        bench(vae, lat, "channels_last + compile", warm=3, n=20)
    except Exception as e:  # noqa: BLE001
        print(f"compile default 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
