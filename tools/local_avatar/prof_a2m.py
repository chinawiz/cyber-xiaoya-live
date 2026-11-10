"""拆解 Audio2Motion 的 700ms 到底花在哪。

用法：python prof_a2m.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

import ditto_trt_patch  # noqa: E402

ditto_trt_patch.patch()

import core.atomic_components.audio2motion as a2m_mod  # noqa: E402
import core.atomic_components.condition_handler as ch_mod  # noqa: E402
import core.models.lmdm as lmdm_mod  # noqa: E402

STATS: dict[str, list[float]] = {}


def wrap(obj, name, attr="__call__"):
    orig = getattr(obj, attr)

    def f(*a, **k):
        t = time.perf_counter()
        r = orig(*a, **k)
        STATS.setdefault(name, []).append((time.perf_counter() - t) * 1000)
        return r

    setattr(obj, attr, f)


wrap(a2m_mod.Audio2Motion, "a2m.__call__")
wrap(ch_mod.ConditionHandler, "cond_handler")
wrap(lmdm_mod.LMDM, "_call_np")

from stream_pipeline_online import StreamSDK  # noqa: E402

sdk = StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                "./checkpoints/ditto_trt_fast")
sdk.setup(os.path.expanduser("~/local-avatar/avatar.jpg"), "/tmp/prof_a2m.mp4",
          sampling_timesteps=10)

# 直接构造一次 a2m 调用：给它 80 帧音频特征
am = sdk.audio2motion
seq = am.seq_frames
aud_feat = np.random.randn(seq, 1103).astype(np.float32)
cond = sdk.condition_handler(aud_feat, 0)[None]

# 预热
for _ in range(2):
    am(cond.copy(), None)

for _ in range(8):
    t = time.perf_counter()
    am(cond.copy(), None)
    print(f"  一次 a2m: {(time.perf_counter() - t) * 1000:.0f} ms", flush=True)

print()
for k, v in STATS.items():
    a = np.array(v[2:] if len(v) > 4 else v)
    print(f"[{k:16s}] n={len(v):3d} 均值={a.mean():7.1f}ms p50={np.percentile(a, 50):7.1f}")

# 再单独测 lmdm 一次前向
lmdm = sdk.audio2motion.lmdm
x = np.random.randn(1, seq, 265).astype(np.float32)
cf = np.random.randn(1, 265).astype(np.float32)
cc = cond[0][:1].repeat(seq, 0)[None].astype(np.float32)
tc = np.array([999], dtype=np.int64)
for _ in range(3):
    lmdm._one_step(x, cf, cc, tc)
t = time.perf_counter()
n = 20
for _ in range(n):
    lmdm._one_step(x, cf, cc, tc)
print(f"\n_one_step(含 setup+infer): {(time.perf_counter() - t) / n * 1000:.2f} ms/步")
