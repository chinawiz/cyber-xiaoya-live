#!/usr/bin/env python3
"""融合对比：同一帧、同一份生成结果，分别走旧方案和新方案，横向拼图。

要回答的问题只有一个 —— 嘴部是「浮在脸上」还是「长在脸上」。
所以三张图用完全相同的 gen 张量，只改合成逻辑：
    ORIGINAL      原图（基准，嘴闭合）
    RESIDUAL(old) 旧方案：差值贴回 + 椭圆空间先验
    BLEND(new)    新方案：face-parsing 语义蒙版 + 整块重建图 alpha 合成
这样差异全部来自合成，不掺模型随机性。

另外直接吐出蒙版本身（MASK），确认它盖住的是嘴唇和下巴，而不是一坨椭圆。
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("OUT", "/tmp/mblend"))
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

    # residual 对照需要静音基准
    torch = r.torch
    with torch.no_grad():
        zeros = torch.zeros((1, 10, r.n_layers, 384), dtype=r.dtype, device=r.device)
        r.idle_low = r._infer(zeros).astype(np.float32)

    with wave.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        r.push_audio(f.readframes(f.getnframes()))
    m = r.mask[..., 0]
    print(f"audio={r.pcm.seconds:.1f}s crop={r.crop} "
          f"mask={m.shape} cover={(m > 0.5).mean() * 100:.1f}% "
          f"feather={(m > 0.02).mean() * 100:.1f}%", flush=True)

    # 扫一遍，挑开合幅度最大的几帧 —— 静音帧看不出融合好坏
    cand = []
    for idx in range(10, min(int(r.pcm.seconds * 25) - 10, 260), 3):
        gen = r._infer(r._clip(idx))
        d = np.abs(gen.astype(np.float32) - r.idle_low).mean()
        cand.append((float(d), idx, gen))
    cand.sort(key=lambda t: -t[0])
    print("top frames:", [(round(s, 2), i) for s, i, _ in cand[:5]], flush=True)

    x0, y0, x1, y1 = r.crop
    by1, by2, bx1, bx2 = r.box
    # 取裁窗口：从鼻尖下方到下巴再往下一点，正好覆盖「融合接缝」该出现的位置
    cy = by1 + int((by2 - by1) * 0.42)
    lx = max(0, bx1 - 26)
    rx = min(r.base.shape[1], bx2 + 26)

    for rank, (score, idx, gen) in enumerate(cand[:3]):
        imgs = {}
        for mode in ("residual", "blend"):
            r.compose = mode
            jpg = r._paste(gen)
            imgs[mode] = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        r.compose = "blend"
        row = []
        for name, im in (("ORIGINAL", r.base),
                         ("RESIDUAL  old", imgs["residual"]),
                         ("BLEND  new", imgs["blend"])):
            c = im[cy:by2 + 14, lx:rx].copy()
            c = cv2.resize(c, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (0, 255, 255), 2, cv2.LINE_AA)
            row.append(c)
        cv2.imwrite(str(OUT / f"cmp{rank}_f{idx}_delta{score:.1f}.png"), np.hstack(row))
        # 整帧也留一张，看整体观感
        three = np.hstack([r.base, imgs["residual"], imgs["blend"]])
        cv2.imwrite(str(OUT / f"full{rank}_f{idx}.png"), three)
        print(f"f{idx} delta={score:.2f} -> cmp{rank}_f{idx}.png", flush=True)

    # 蒙版本身（贴在原图 crop 上，红=融合区）
    crop = r.base[y0:y1, x0:x1].copy()
    vis = crop.copy()
    vis[m > 0.5] = (vis[m > 0.5] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
    vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "mask_vis.png"), vis)
    cv2.imwrite(str(OUT / "mask_raw.png"), (m * 255).astype(np.uint8))
    print(f"mask_vis.png / mask_raw.png -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
