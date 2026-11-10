#!/usr/bin/env python3
"""MuseTalk 后端验收脚本（离线跑在 GPU 机上，不进服务）。

回答三个问题：
  1. 权重能不能加载、输出有没有 NaN
  2. 单帧延迟够不够 25fps（40ms 预算）
  3. 画质相对 Wav2Lip 96² 到底提升了多少（Laplacian / 梯度能量 / 嘴部对比图）

用法：
    ~/MuseTalk/venv/bin/python mtest.py [输出目录]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

AVATAR = os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg"))
MUSE = os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk"))
MODELS = os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS", str(Path(MUSE) / "models"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtest")
OUT.mkdir(parents=True, exist_ok=True)


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def grad_energy(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def speech_pcm(sr: int = 24000, sec: float = 3.0) -> bytes:
    """测试音频：优先用真实语音 wav（LOCAL_AVATAR_TEST_WAV）。

    合成正弦虽然能驱动嘴型，但 whisper 对它的特征分布是失真的，
    拿去判断画质会误判。真实中文语音才等价于线上 TTS 的情况。
    """
    wav = os.environ.get("LOCAL_AVATAR_TEST_WAV", "")
    if wav and Path(wav).is_file():
        import wave as _wav
        with _wav.open(wav) as f:
            assert f.getframerate() == sr, f"wav 采样率需 {sr}，实际 {f.getframerate()}"
            return f.readframes(f.getnframes())
    t = np.arange(int(sr * sec)) / sr
    f0 = 165 + 35 * np.sin(2 * np.pi * 2.3 * t)
    sig = np.zeros_like(t)
    for k, amp in enumerate((1.0, 0.5, 0.28, 0.15)):
        sig += amp * np.sin(2 * np.pi * f0 * (k + 1) * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.7 * t)      # 音节包络，制造开合
    env = np.clip(env, 0.05, 1.0)
    return (sig * env * 0.32 * 32767).astype("<i2").tobytes()


def main() -> None:
    # 脸框优先用 box.txt（服务启动时 start.sh 也是这么注入的）。
    # 不设的话 BaseRenderer 会走 Haar 兜底，戴墨镜的肖像检测不到脸，
    # 会退回居中框 —— 裁到的不是脸，后面的画质指标全部失真。
    boxf = Path.home() / "local-avatar" / "box.txt"
    if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
        os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()

    from musetalk_renderer import MuseTalkRenderer

    t0 = time.time()
    r = MuseTalkRenderer(AVATAR, MUSE, MODELS)
    print(f"\n[load] {time.time() - t0:.1f}s  crop={r.crop}  "
          f"face={(r.crop[2] - r.crop[0])}x{(r.crop[3] - r.crop[1])}")

    r.push_audio(speech_pcm())
    print(f"[audio] {r.pcm.seconds:.2f}s buffered")

    # 预热（首帧含 cudnn 算法选择，不计入统计）
    for i in range(3):
        r._infer(r._clip(i))

    # 逐帧推理延迟 + 分项耗时（决定 25fps 够不够）
    import torch
    from einops import rearrange
    n = 20
    lat, unet_ms, vae_ms = [], [], []
    for i in range(n):
        clip = r._clip(i + 3)
        t = time.monotonic()
        img = r._infer(clip)
        lat.append((time.monotonic() - t) * 1000)
        # 分项
        with torch.no_grad():
            x = rearrange(clip, "b c h w -> b (c h) w").to(r.device, dtype=r.dtype)
            x = r.pe(x).to(r.dtype)
            lat_batch = r.latents.to(dtype=r.dtype)
            t1 = time.monotonic()
            pred = r.unet.model(lat_batch, r.timesteps, encoder_hidden_states=x).sample
            t2 = time.monotonic()
            r.vae.decode_latents(pred[:, :, r.split // 8:, :].contiguous()
                                 .to(r.vae.vae.dtype))
            t3 = time.monotonic()
        unet_ms.append((t2 - t1) * 1000)
        vae_ms.append((t3 - t2) * 1000)
    lat = np.array(lat)
    print(f"[perf] infer {lat.mean():.1f}ms mean / {np.median(lat):.1f}ms med / "
          f"{lat.min():.1f}~{lat.max():.1f}ms  → {1000 / lat.mean():.1f} fps")
    print(f"[perf] unet {np.mean(unet_ms):.1f}ms  vae {np.mean(vae_ms):.1f}ms  "
          f"whisper/其它 {lat.mean() - np.mean(unet_ms) - np.mean(vae_ms):.1f}ms")

    # NaN / 全黑检测
    bad = int((img != img).sum()) if np.isnan(img).any() else 0
    print(f"[check] dtype={img.dtype} shape={img.shape} nan={bad} "
          f"min={img.min()} max={img.max()} mean={img.mean():.1f}")

    # 输出帧（第 12 帧，嘴部应处于张开状态）：residual 与 direct 两种贴回方式
    img12 = r._infer(r._clip(12))
    r.compose = "residual"
    frame = r._paste(img12)
    (OUT / "musetalk-frame.jpg").write_bytes(frame)
    r.compose = "direct"
    (OUT / "musetalk-direct.jpg").write_bytes(r._paste(img12))
    r.compose = "residual"
    t = time.monotonic()
    full = r.render_frame()
    print(f"[e2e] render_frame {(time.monotonic() - t) * 1000:.1f}ms  "
          f"cursor={r.cursor} stats={r.stats()}")

    mt = cv2.imread(str(OUT / "musetalk-frame.jpg"))
    mtd = cv2.imread(str(OUT / "musetalk-direct.jpg"))
    base = r.base.copy()

    # 对照：Wav2Lip（若权重在）
    w2l = None
    ck = Path.home() / "models" / "wav2lip" / "wav2lip_gan.pth"
    up = Path.home() / "Wav2Lip"
    if ck.is_file() and up.is_dir():
        try:
            from renderer import Wav2LipRenderer
            w = Wav2LipRenderer(AVATAR, str(ck), str(up))
            w.push_audio(speech_pcm())
            p = w._predict(w._mel())
            out = w.base.copy()
            y1, y2, x1, x2 = w.box
            out[y1:y2, x1:x2] = np.clip(w._compose(p), 0, 255).astype(np.uint8)
            w2l = out
            cv2.imwrite(str(OUT / "wav2lip-frame.jpg"), out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Wav2Lip 对照失败: {type(e).__name__}: {e}")

    rows = [("original", base)]
    if w2l is not None:
        rows.append(("wav2lip 96", w2l))
    rows.append(("musetalk-residual", mt))
    rows.append(("musetalk-direct", mtd))

    print(f"\n{'variant':<16s} {'laplacian':>10s} {'grad':>8s}")
    for name, im in rows:
        print(f"{name:<16s} {sharpness(im):10.1f} {grad_energy(im):8.2f}")

    # 嘴部放大对比（同一区域，最近邻放大 3 倍，方便直接看纹理）
    y1, y2, x1, x2 = r.box
    my1, my2 = y1 + int((y2 - y1) * 0.62), y1 + int((y2 - y1) * 0.96)
    mx1, mx2 = x1 + int((x2 - x1) * 0.30), x1 + int((x2 - x1) * 0.95)
    tiles = []
    for name, im in rows:
        crop = im[my1:my2, mx1:mx2]
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 120), 2, cv2.LINE_AA)
        tiles.append(crop)
    sep = np.full((tiles[0].shape[0], 12, 3), 40, np.uint8)
    row = tiles[0]
    for tp in tiles[1:]:
        row = np.hstack([row, sep, tp])
    cv2.imwrite(str(OUT / "montage-mouth.png"), row)
    print(f"\nwrote {OUT}/musetalk-frame.jpg  和 montage-mouth.png "
          f"({row.shape[1]}x{row.shape[0]})")


if __name__ == "__main__":
    main()
