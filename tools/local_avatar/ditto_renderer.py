#!/usr/bin/env python3
"""Ditto 渲染器 —— 把 antgroup/ditto-talkinghead 的在线管线接进本地渲染服务。

与 MuseTalk/Wav2Lip 的「每帧重绘嘴部再贴回」根本不同：Ditto 在**显式低维运动空间**
生成 21 个隐式关键点序列（LMDM 扩散），再由 warp 网络对**原图外观特征**做三维形变，
decoder 解码回像素、PutBack 按 90% 渐变蒙版贴回原图。像素不是"生成"出来的，
所以肤色/纹理天然锚定原图，不存在"嘴浮在图上"。

三条必须知道的约束（来自上游实现，不是我们的选择）：

1. **帧率硬编码 25fps**。hubert 特征 50Hz，两帧合一次，于是每个 0.04s 音频恰好
   对应一帧（`num_f = ceil(len/16000*25)`）。改帧率等于改训练假设，所以本渲染器的
   `fps` 恒为 25，不跟 LOCAL_AVATAR_FPS 走。
2. **管线是有状态的异步流水线**。`run_chunk` 只往队列塞音频特征块，真正的帧从
   putback 线程回调出来，帧号即音频时间轴（第 i 帧 ↔ [i/25, (i+1)/25)）。
3. **首帧有固定延迟**。要攒够一块 6480 样本 + 一个 LMDM 窗口才吐第一帧，实测
   1.5~1.8s。所以音频统一延后 `av_delay` 秒输出，两边共用一条时钟 —— 观感上是
   "整体延迟"，而不是"音画不同步"。

时基设计（关键，改之前先读完）：
    视频不是"音频推进到哪就渲染哪"，而是把已到达的音频按 1x 播出去，用**同一份
    wall clock** 决定当前该显示第几帧。因为生成速度受 GPU 限制（实测 21fps @
    GB10），比 25fps 慢，所以偶尔会跟不上 —— 这时靠「重锚」把时钟对齐到最新帧，
   表现为画面轻微快进，而不是越拖越远直到嘴型完全对不上。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from renderer import BaseRenderer

SR_IN = 24000            # 桥送来的 PCM：s16le / 24kHz / mono
SR_DITTO = 16000         # hubert 要求 16k
DITTO_FPS = 25           # 模型决定，见文件头
CHUNKSIZE = (3, 5, 2)
CHUNK_LEN = int(sum(CHUNKSIZE) * 0.04 * SR_DITTO) + 80    # 6480
CHUNK_STEP = CHUNKSIZE[1] * 640                          # 3200（0.2s）
PAD_LEN = CHUNKSIZE[0] * 640                             # 1920
GAP_S = 0.35             # 距上一块音频超过这个时长 = 新的一段话
REANCHOR_FRAMES = 12     # 落后超过 0.48s 就重锚时钟（自愈，避免持续漂移）


def _to16k(x: np.ndarray) -> np.ndarray:
    """24k → 16k，优先多相滤波（带抗混叠），退回线性插值。"""
    if x.size == 0:
        return x
    try:
        from scipy.signal import resample_poly
        return resample_poly(x, 2, 3).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * 2 / 3))
        if n <= 0:
            return np.zeros(0, np.float32)
        return np.interp(np.arange(n) * (len(x) / n), np.arange(len(x)), x).astype(np.float32)


class _NullWriter:
    """顶掉上游的 mp4 writer：帧直接走回调给服务，不落盘。"""

    def __call__(self, *a, **k):
        pass

    def close(self):
        pass


def _build_sdk_class(base, on_frame):
    """派生 StreamSDK：把 putback 的产物交给回调，而不是写 mp4。"""

    class _SDK(base):
        def setup(self, *a, **k):
            super().setup(*a, **k)
            self.writer = _NullWriter()

        def _putback_worker(self):
            while not self.stop_event.is_set():
                try:
                    item = self.putback_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    # 让 writer 线程正常收尾，否则 close() 时 join 会卡住
                    self.writer_queue.put(None)
                    break
                frame_idx, render_img = item
                frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
                M_c2o = self.source_info["M_c2o_lst"][frame_idx]
                res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
                on_frame(frame_idx, res_frame_rgb)

        def close(self):
            self.audio2motion_queue.put(None)
            for thread in self.thread_list:
                thread.join()
            if self.worker_exception is not None:
                raise self.worker_exception

    return _SDK


class DittoRenderer(BaseRenderer):
    backend = "ditto"
    fps = DITTO_FPS
    compose = "ditto"

    def __init__(self, avatar_path: str = "", ditto_dir: str = "",
                 data_root: str = "", cfg_pkl: str = ""):
        super().__init__(avatar_path)
        self.backend = "ditto"          # 覆盖 BaseRenderer 设的 "base"

        self.ditto_dir = ditto_dir or str(Path.home() / "ditto-talkinghead")
        self.data_root = data_root or str(Path(self.ditto_dir) / "checkpoints" / "ditto_trt_fast")
        self.cfg_pkl = cfg_pkl or str(Path(self.ditto_dir) / "checkpoints" / "ditto_cfg"
                                      / "v0.4_hubert_cfg_trt_online.pkl")
        self.version = "v0.4"

        def _env(key: str, default, cast=float):
            try:
                return cast(os.environ.get(key, default))
            except (TypeError, ValueError):
                return cast(default)

        self.steps = _env("LOCAL_AVATAR_DITTO_STEPS", 10, int)
        overlap = _env("LOCAL_AVATAR_DITTO_OVERLAP", 0, int)      # 0 = 用 cfg 默认(70)
        self.av_delay = _env("LOCAL_AVATAR_AV_DELAY", 1.7)
        self.sharpen = 0

        if not Path(self.cfg_pkl).is_file():
            raise FileNotFoundError(f"cfg 不存在: {self.cfg_pkl}")
        if not Path(self.data_root).is_dir():
            raise FileNotFoundError(f"引擎目录不存在: {self.data_root}")

        if self.ditto_dir not in sys.path:
            sys.path.insert(0, self.ditto_dir)
        from stream_pipeline_online import StreamSDK      # noqa: E402  上游模块

        # ---- 共享状态（单锁，避免取帧与喂音频互锁）----
        self._lock = threading.Condition()
        self._abuf = np.zeros(PAD_LEN, np.float32)   # 前置静音，与上游 inference.py 对齐
        self._fed = 0                                # 已喂到 _abuf 的哪个位置
        self._total_s = 0.0                          # 已收到音频总时长
        self._last_push = 0.0

        self.frames_q: deque[tuple[int, np.ndarray]] = deque(maxlen=300)
        self.gen_frames = 0
        self.cursor = -1
        self.skipped = 0
        self.lag_frames = 0
        self.gen_ms = 0.0
        self.reanchors = 0
        self._last_gen_at = 0.0

        # 时基：段起点 wall 时刻 + 该段第一帧的全局帧号
        self._seg_t: float | None = None
        self._seg_frame0 = 0

        self._stop = threading.Event()

        # ---- 建 SDK（TRT wrapper 补丁必须在加载引擎前打）----
        t_boot = time.time()
        try:
            import ditto_trt_patch
            ditto_trt_patch.patch()
        except Exception as e:  # noqa: BLE001
            print(f"[ditto] TRT 补丁未生效({type(e).__name__}: {e})，性能会明显偏低",
                  flush=True)

        sdk_cls = _build_sdk_class(StreamSDK, self._on_frame)
        self.sdk = sdk_cls(self.cfg_pkl, self.data_root)

        tmp_out = str(Path("/tmp") / f"ditto_stream_{os.getpid()}.mp4")
        setup_kwargs = {"sampling_timesteps": self.steps}
        if overlap > 0:
            setup_kwargs["overlap_v2"] = overlap
        self.sdk.setup(avatar_path or "example/image.png", tmp_out, **setup_kwargs)
        self.overlap_eff = getattr(self.sdk.audio2motion, "overlap_v2", overlap)

        # 静态底图：没音频/没出帧时显示原图，避免画面黑掉
        src = self.sdk.source_info["img_rgb_lst"][0]
        self._static = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        self.out_h, self.out_w = self._static.shape[:2]
        ok, buf = cv2.imencode(".jpg", self._static, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.jpeg = buf.tobytes() if ok else b""

        self._feeder = threading.Thread(target=self._feed_loop, name="ditto-feeder", daemon=True)
        self._feeder.start()

        print(f"[ditto] ready in {time.time() - t_boot:.1f}s steps={self.steps} "
              f"overlap={self.overlap_eff} out={self.out_w}x{self.out_h} "
              f"av_delay={self.av_delay}s root={os.path.basename(self.data_root)}", flush=True)

    # ────────────────────────── 音频 ──────────────────────────

    def push_audio(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            return
        y = _to16k(x)
        now = time.monotonic()
        new_seg = (self._seg_t is None) or (now - self._last_push) > GAP_S
        with self._lock:
            if new_seg:
                # 新的一段话：时钟重锚，帧号也以「当前已收帧数」为基准，
                # 否则隔了几十秒再说话，旧的 target 会把新帧全冲掉
                self._seg_t = now
                self._seg_frame0 = self.gen_frames
                self.frames_q.clear()
            self._last_push = now
            self._abuf = np.concatenate((self._abuf, y))
            self._total_s += len(y) / SR_DITTO
            self._lock.notify()
        if new_seg:
            print(f"[ditto] 新语音段 frame0={self._seg_frame0}", flush=True)

    def _feed_loop(self) -> None:
        """攒够 6480 样本就喂一块给 hubert；hubert 同步推理，必须独立线程。"""
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and (self._abuf.size - self._fed) < CHUNK_LEN:
                    self._lock.wait(0.2)
                if self._stop.is_set():
                    return
                chunk = self._abuf[self._fed:self._fed + CHUNK_LEN].copy()
                self._fed += CHUNK_STEP
                if self._fed >= SR_DITTO * 2:           # 回收已消费的 2 秒
                    self._abuf = self._abuf[self._fed:]
                    self._fed = 0
            try:
                self.sdk.run_chunk(chunk, CHUNKSIZE)
            except Exception as e:  # noqa: BLE001
                print(f"[ditto] feed error: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.05)

    # ────────────────────────── 帧回调 ──────────────────────────

    def _on_frame(self, idx: int, rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        now = time.monotonic()
        with self._lock:
            self.frames_q.append((idx, bgr))
            self.gen_frames += 1
            self.cursor = idx
            if self._last_gen_at > 0:
                self.gen_ms = self.gen_ms * 0.9 + (now - self._last_gen_at) * 1000 * 0.1
            self._last_gen_at = now

    # ────────────────────────── 出帧 ──────────────────────────

    def _clock_locked(self) -> float:
        if self._seg_t is None:
            return 0.0
        now = time.monotonic()
        # 静音期冻结时钟：没有新音频还让 target 往前跑，就会不断触发重锚
        # （实测 11s 音频 + 2.5s 静音会刷出 20+ 次），冻结后静音期画面干净地停住
        ref = now if (now - self._last_push) <= GAP_S else self._last_push
        return max(0.0, ref - self._seg_t - self.av_delay)

    def render_frame(self) -> bytes:
        with self._lock:
            clock = self._clock_locked()
            target_rel = int(clock * DITTO_FPS)

            # 自愈：生成跟不上时（最新帧还落后目标一大截）把时钟拉回最新帧，
            # 宁可画面轻微快进，也不要音画越差越远
            if self.frames_q:
                newest_rel = self.frames_q[-1][0] - self._seg_frame0
                if target_rel - newest_rel > REANCHOR_FRAMES:
                    self._seg_frame0 = self.frames_q[-1][0]
                    self._seg_t = time.monotonic() - self.av_delay
                    target_rel = 0
                    self.reanchors += 1

            target_abs = self._seg_frame0 + target_rel
            frame = None
            popped = 0
            while self.frames_q and self.frames_q[0][0] <= target_abs:
                _, frame = self.frames_q.popleft()
                popped += 1
            if popped > 1:
                self.skipped += popped - 1      # 时钟一次跨过多帧 = 真的跳帧了
            if self.frames_q:
                self.lag_frames = max(0, target_rel - (self.frames_q[0][0] - self._seg_frame0))
            elif frame is not None:
                self.lag_frames = 0

        if frame is None:
            return self.jpeg                      # 还没到下一帧的显示时刻 → 保持
        out = self._overlay(frame) if self.action else self._encode(frame)
        return out

    def _encode(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else self.jpeg

    def interrupt(self) -> None:
        """桥侧打断（type 7）：丢掉未播音频与排队中的帧，下一句重新起段。"""
        with self._lock:
            self._abuf = np.zeros(PAD_LEN, np.float32)
            self._fed = 0
            self._seg_t = None
            self._total_s = 0.0
            self.frames_q.clear()
            self._lock.notify_all()
        print("[ditto] interrupt: 缓冲已清空", flush=True)

    def extra_health(self) -> dict:
        with self._lock:
            total = self._total_s
            clock = self._clock_locked()
            queued = len(self.frames_q)
        return {"fps": self.fps, "av_delay": self.av_delay, "steps": self.steps,
                "overlap": getattr(self, "overlap_eff", 0), "gen_frames": self.gen_frames,
                "pcm_s": round(total, 2), "gen_ms": round(self.gen_ms, 1),
                "lag_frames": self.lag_frames, "queued": queued,
                "clock": round(clock, 2), "reanchors": self.reanchors}
