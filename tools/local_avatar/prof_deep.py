"""把 online 管线拆到最细，定位单帧 26ms 里每一项的占比。

关注点：
1. a2m（Audio2Motion）到底花在哪 —— LMDM 引擎本身 vs Python 编排
2. warp / decode / putback 的真实单帧成本
3. a2m 的摊销系数（一次前向产出多少帧可用）
"""
import os
import sys
import time

import librosa
import numpy as np

sys.path.insert(0, ".")

import core.atomic_components.warp_f3d as m1
import core.atomic_components.decode_f3d as m2
import core.atomic_components.putback as m3
import core.atomic_components.writer as m4
import core.atomic_components.audio2motion as m5
from core.models import lmdm as m6

import ditto_trt_patch
ditto_trt_patch.patch()

STATS = {}
FRAMES = [0]


def wrap(cls, attr, name, extra=None):
    orig = getattr(cls, attr)

    def f(self, *a, **k):
        t = time.perf_counter()
        r = orig(self, *a, **k)
        STATS.setdefault(name, []).append((time.perf_counter() - t) * 1000.0)
        if extra:
            extra(r)
        return r

    setattr(cls, attr, f)


def _count(r):
    FRAMES[0] += 1


wrap(m1.WarpF3D, "__call__", "warp")
wrap(m2.DecodeF3D, "__call__", "decode")
wrap(m3.PutBack, "__call__", "putback")
wrap(m4.VideoWriterByImageIO, "__call__", "write", _count)
wrap(m5.Audio2Motion, "__call__", "a2m")
wrap(m6.LMDM, "__call__", "lmdm")
wrap(m6.LMDM, "_call_np", "  lmdm.np")
wrap(m6.LMDM, "_one_step", "    step")

# 拆 _one_step 内部：setup 与 infer
_orig_one = m6.LMDM._one_step


def _one_step(self, x, cond_frame, cond, time_cond):
    if self.model_type != "tensorrt":
        return _orig_one(self, x, cond_frame, cond, time_cond)
    t0 = time.perf_counter()
    self.model.setup({"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
    t1 = time.perf_counter()
    self.model.infer()
    t2 = time.perf_counter()
    STATS.setdefault("      setup", []).append((t1 - t0) * 1000.0)
    STATS.setdefault("      infer", []).append((t2 - t1) * 1000.0)
    return self.model.buffer["pred_noise"][0], self.model.buffer["x_start"][0]


m6.LMDM._one_step = _one_step

from stream_pipeline_online import StreamSDK  # noqa: E402

STEPS = int(os.environ.get("STEPS", "10"))
OVERLAP = int(os.environ.get("OVERLAP", "50"))
DATAROOT = os.environ.get("DATAROOT", "./checkpoints/ditto_trt_fast")
CFG = os.environ.get("CFG", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/xiaoya.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/prof_deep.mp4"

sdk = StreamSDK(CFG, DATAROOT)
sdk.setup(os.path.expanduser("~/local-avatar/avatar.jpg"), OUT,
          sampling_timesteps=STEPS, overlap_v2=OVERLAP)

a2m = sdk.audio2motion
print(f"[cfg] steps={STEPS} overlap={OVERLAP} seq_frames={a2m.seq_frames} "
      f"valid_clip_len={a2m.valid_clip_len} fuse_length={a2m.fuse_length} "
      f"overlap_v2={a2m.overlap_v2}", flush=True)

audio, sr = librosa.load(WAV, sr=16000)
chunksize = (3, 5, 2)
audio = np.concatenate([np.zeros((chunksize[0] * 640,), np.float32), audio])
split_len = int(sum(chunksize) * 0.04 * 16000) + 80
step = chunksize[1] * 640

for k in STATS:
    STATS[k].clear()
FRAMES[0] = 0

t0 = time.time()
for i in range(0, len(audio), step):
    ch = audio[i:i + split_len]
    if len(ch) < split_len:
        ch = np.pad(ch, (0, split_len - len(ch)))
    sdk.run_chunk(ch, chunksize)
sdk.close()
wall = time.time() - t0

n = FRAMES[0]
audio_len = (len(audio) - chunksize[0] * 640) / 16000
print(f"\n[总] wall={wall:.2f}s 产出 {n} 帧 音频 {audio_len:.2f}s → {n / wall:.1f} fps",
      flush=True)
print(f"[摊销] 每帧预算 40ms@25fps，当前 {wall / n * 1000:.1f}ms/帧\n", flush=True)

print(f"{'项':<14}{'次数':>6}{'均值':>8}{'p50':>8}{'p90':>8}{'最大':>8}{'每帧':>9}")
for k in ["a2m", "lmdm", "  lmdm.np", "    step", "      setup", "      infer",
          "warp", "decode", "putback", "write"]:
    v = STATS.get(k)
    if not v:
        continue
    arr = np.array(v)
    # 每帧成本 = 该项总耗时 / 产出帧数
    per = arr.sum() / max(n, 1)
    print(f"{k:<14}{len(arr):>6}{arr.mean():>8.1f}{np.percentile(arr, 50):>8.1f}"
          f"{np.percentile(arr, 90):>8.1f}{arr.max():>8.1f}{per:>9.2f}")

nc = len(STATS.get("a2m", []))
if nc:
    print(f"\n[a2m 摊销] {nc} 次前向 / {n} 帧 = 每次覆盖 {n / nc:.1f} 帧")
    print(f"[a2m 拆分] 引擎(step总和)={np.sum(STATS['    step']):.0f}ms  "
          f"整体={np.sum(STATS['a2m']):.0f}ms  "
          f"引擎外={np.sum(STATS['a2m']) - np.sum(STATS['    step']):.0f}ms")
