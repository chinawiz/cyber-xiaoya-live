"""测 Ditto online 管线的真实吞吐：每帧产出时刻 vs 该帧对应的音频时间。

用法：
    STEPS=50 DATAROOT=./checkpoints/ditto_trt_custom CFG=./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
        python bench_online.py [wav] [outmp4]
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

STATS = {}
_T0 = [time.time()]
LAT = []          # (帧序号, 产出时刻-起点, 该帧对应音频时刻)
FRAME_TS = []


def wrap(cls, name, extra=None):
    orig = cls.__call__

    def f(self, *a, **k):
        t = time.perf_counter()
        r = orig(self, *a, **k)
        STATS.setdefault(name, []).append((time.perf_counter() - t) * 1000.0)
        if extra:
            extra(r)
        return r

    cls.__call__ = f


wrap(m1.WarpF3D, "warp")
wrap(m2.DecodeF3D, "decode")
wrap(m3.PutBack, "putback")
wrap(m5.Audio2Motion, "a2m")


def _on_frame(r):
    now = time.time() - _T0[0]
    FRAME_TS.append(now)
    idx = len(FRAME_TS) - 1
    LAT.append((idx, now, idx / 25.0))


wrap(m4.VideoWriterByImageIO, "write", _on_frame)

from stream_pipeline_online import StreamSDK  # noqa: E402

if os.environ.get("TRT_PATCH", "0") == "1":
    import ditto_trt_patch  # noqa: E402

    print("[patch] buffer 复用:", ditto_trt_patch.patch(), flush=True)

STEPS = int(os.environ.get("STEPS", "50"))
OVERLAP = int(os.environ.get("OVERLAP", "0"))       # 0 = 用 cfg 默认(70)
DATAROOT = os.environ.get("DATAROOT", "./checkpoints/ditto_trt_custom")
CFG = os.environ.get("CFG", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
WAV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/xiaoya.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bench_online.mp4"

print(f"[cfg] steps={STEPS} overlap={OVERLAP or 'cfg'} data_root={DATAROOT}", flush=True)

t_boot = time.time()
sdk = StreamSDK(CFG, DATAROOT)
print(f"[load] {time.time() - t_boot:.1f}s", flush=True)

_setup_kw = {"sampling_timesteps": STEPS}
if OVERLAP > 0:
    _setup_kw["overlap_v2"] = OVERLAP
sdk.setup(os.path.expanduser("~/local-avatar/avatar.jpg"), OUT, **_setup_kw)
print(f"[setup] {time.time() - t_boot:.1f}s", flush=True)

audio, sr = librosa.load(WAV, sr=16000)
print(f"[audio] {len(audio) / 16000:.2f}s sr={sr}", flush=True)

chunksize = (3, 5, 2)
audio = np.concatenate([np.zeros((chunksize[0] * 640,), np.float32), audio])
split_len = int(sum(chunksize) * 0.04 * 16000) + 80
step = chunksize[1] * 640

_T0[0] = time.time()
for i in range(0, len(audio), step):
    ch = audio[i:i + split_len]
    if len(ch) < split_len:
        ch = np.pad(ch, (0, split_len - len(ch)))
    sdk.run_chunk(ch, chunksize)

t_feed_end = time.time() - _T0[0]
sdk.close()
t_all = time.time() - _T0[0]

n = len(FRAME_TS)
audio_len = (len(audio) - chunksize[0] * 640) / 16000
print(f"[time] feed={t_feed_end:.2f}s total={t_all:.2f}s", flush=True)
print(f"[frames] {n} expect={int(audio_len * 25)} audio={audio_len:.2f}s", flush=True)
if n > 1:
    span = FRAME_TS[-1] - FRAME_TS[0]
    print(f"[fps] 生成段 {span:.2f}s → {(n - 1) / span:.1f} fps | 全程 {n / t_all:.1f} fps",
          flush=True)
    # 实时性：帧产出时刻应 <= 该帧音频时刻（若持续 > ，说明跟不上，延迟会累积）
    over = [(i, t, a) for i, t, a in LAT if t > a + 0.5]
    print(f"[lag] 落后>0.5s 的帧 {len(over)}/{n}", flush=True)
    print(f"[first] 首帧产出 {FRAME_TS[0]:.2f}s（音视频统一延迟要 >= 这个值）", flush=True)
    for i in (5, 25, 50, 100, n // 2, n - 1):
        if 0 <= i < n:
            _, t, a = LAT[i]
            print(f"  frame {i:4d}: 产出 {t:6.2f}s  音频时刻 {a:6.2f}s  lag {t - a:+.2f}s",
                  flush=True)
for k, v in STATS.items():
    arr = np.array(v[10:] if len(v) > 20 else v)
    print(f"[{k:8s}] n={len(v):4d} mean={arr.mean():6.1f}ms p50={np.percentile(arr, 50):6.1f} "
          f"p90={np.percentile(arr, 90):6.1f} max={arr.max():6.1f}", flush=True)
