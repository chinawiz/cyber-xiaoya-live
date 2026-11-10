#!/usr/bin/env python3
"""诊断音频条件链：whisper 特征到底有没有起作用。

症状是「脸能生成但整体失焦、嘴几乎不动」，这不像权重问题，
更像 UNet 拿到的 encoder_hidden_states 是常量/全零 —— 那样它只能
靠 latent 自己重建，结果就是一张糊脸 + 嘴不动。

这里逐层打印统计量，并做「真实特征 vs 全零特征」的输出对比：
两组结果若几乎一致，就说明音频条件彻底失效。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("/tmp/mdump")


def main():
    boxf = Path.home() / "local-avatar" / "box.txt"
    if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
        os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()
    from musetalk_renderer import MuseTalkRenderer

    r = MuseTalkRenderer(os.environ.get("LOCAL_AVATAR_IMAGE",
                                       str(Path.home() / "local-avatar" / "avatar.jpg")),
                         os.environ.get("LOCAL_AVATAR_MUSETALK_DIR",
                                        str(Path.home() / "MuseTalk")),
                         os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS",
                                        str(Path.home() / "MuseTalk" / "models")))
    import wave as _wav
    with _wav.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        r.push_audio(f.readframes(f.getnframes()))

    # 1) whisper encoder 输出结构
    wav = r.pcm.slice(0.0, 3.0)
    print(f"wav slice: n={wav.size} min={wav.min():.3f} max={wav.max():.3f}")
    inp = r.fe(wav, return_tensors="pt", sampling_rate=16000).input_features
    print(f"mel input: shape={tuple(inp.shape)} min={inp.min():.3f} max={inp.max():.3f}")
    with torch.no_grad():
        hs = r.whisper.encoder(inp.to(r.device, dtype=r.dtype),
                               output_hidden_states=True).hidden_states
    print(f"hidden_states: n={len(hs)} each={tuple(hs[0].shape)}")
    for i, h in enumerate(hs):
        print(f"  layer{i}: mean={h.mean():.4f} std={h.std():.4f} absmax={h.abs().max():.3f}")

    feat = r._encode(wav)
    print(f"feat: shape={tuple(feat.shape)} mean={feat.mean():.4f} std={feat.std():.4f}")

    # 2) 特征方差：沿时间轴是否真的在变（恒定 = 坏）
    if feat.shape[1] > 2:
        tvar = feat[0].std(dim=0).mean()
        print(f"沿时间轴标准差（应显著>0）: {tvar:.4f}")

    # 3) 真实特征 vs 全零特征 → 输出差异
    r._ensure_feat(0.0, 2.0)
    c_real = r._clip(20)
    c_zero = torch.zeros_like(c_real)
    img_real = r._infer(c_real)
    img_zero = r._infer(c_zero)
    d = np.abs(img_real.astype(np.float32) - img_zero.astype(np.float32))
    print(f"\n真实特征 vs 零特征 输出差: mean={d.mean():.2f} max={d.max():.1f}")

    # 4) 两帧不同音频 → 差异（判断嘴型是否随音频变化）
    i1 = r._infer(r._clip(6))
    i2 = r._infer(r._clip(40))
    d2 = np.abs(i1.astype(np.float32) - i2.astype(np.float32))
    print(f"第6帧 vs 第40帧 输出差: mean={d2.mean():.2f} max={d2.max():.1f}")

    # 5) 零特征输出 vs 原图（判断"糊"是否是模型本身的重建能力）
    ref = r.face_low.astype(np.float32)
    print(f"零特征输出 vs 原脸 差: {np.abs(img_zero.astype(np.float32) - ref).mean():.2f}")
    print(f"真实输出   vs 原脸 差: {np.abs(img_real.astype(np.float32) - ref).mean():.2f}")

    tiles = []
    for name, im in (("real", img_real), ("zero", img_zero), ("ref", r.face_low)):
        vis = cv2.resize(im.astype(np.uint8), (im.shape[1] * 2, im.shape[0] * 2),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        tiles.append(vis)
    cv2.imwrite(str(OUT / "diag-feat.png"), np.hstack(tiles))


if __name__ == "__main__":
    main()
