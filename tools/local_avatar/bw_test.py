"""测这台机器的显存有效带宽与 FP16 算力，用来判断瓶颈是搬运还是计算。

warp 是 3D grid_sample + 卷积，decode 是卷积上采样，两者都属于访存密集型。
把实测带宽与公开规格对比，就能算出换卡的理论上限。
"""
import time

import torch

dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
prop = torch.cuda.get_device_properties(0)
print(f"SMs={prop.multi_processor_count} "
      f"mem={prop.total_memory / 2**30:.1f}GiB")


def bench_memcopy(size_mb=512, iters=50):
    n = size_mb * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device=dev)
    b = torch.empty(n, dtype=torch.float32, device=dev)
    for _ in range(5):
        a.copy_(b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a.copy_(b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    # 读 + 写
    gbs = 2 * n * 4 / dt / 1e9
    print(f"[带宽] copy {size_mb}MB: {dt * 1000:.2f}ms → {gbs:.0f} GB/s")
    return gbs


def bench_fp16(size=8192, iters=50):
    a = torch.randn(size, size, dtype=torch.float16, device=dev)
    b = torch.randn(size, size, dtype=torch.float16, device=dev)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    flops = 2 * size**3
    tflops = flops / dt / 1e12
    print(f"[算力] FP16 matmul {size}³: {dt * 1000:.1f}ms → {tflops:.1f} TFLOPS")
    return tflops


# 模拟 warp/decode 的实际形态：中等尺寸的张量搬运 + 小卷积
def bench_conv_like(iters=200):
    x = torch.randn(1, 256, 64, 64, dtype=torch.float16, device=dev)
    w = torch.randn(512, 256, 3, 3, dtype=torch.float16, device=dev)
    conv = torch.nn.Conv2d(256, 512, 3, padding=1).half().cuda()
    for _ in range(10):
        conv(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        z = conv(y)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters * 1000
    print(f"[形态] 2×上采样+3x3conv(256→512): {dt:.2f}ms/次")


bench_memcopy()
bench_memcopy(64)
bench_fp16()
bench_conv_like()
