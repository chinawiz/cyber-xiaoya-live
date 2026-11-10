#!/usr/bin/env python3
"""蒙版范围扫描：验证「模型把微笑画平」的错误能被收窄到多大范围。

结论待验证假设：
  A 现方案 mask = 嘴唇膨胀 91px（覆盖下半脸 8.8%），模型对嘴角的错误全被贴回来；
  B/C 收窄到嘴唇本身 + 小膨胀，只让模型改嘴，其余像素保持原图。
每组再各测两种色度策略（gen = 用模型色度 / clamp = 限幅）。
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("OUT", "/tmp/probe4"))
OUT.mkdir(parents=True, exist_ok=True)

boxf = Path.home() / "local-avatar" / "box.txt"
if boxf.is_file() and not os.environ.get("LOCAL_AVATAR_BOX"):
    os.environ["LOCAL_AVATAR_BOX"] = boxf.read_text().strip()

FRAMES = [int(x) for x in os.environ.get("FRAMES", "24").split(",")]
DIL = [int(x) for x in os.environ.get("DILS", "0,16,40").split(",")]
CLAMP = float(os.environ.get("CLAMP", "6.0"))
CHROMAS = os.environ.get("CHROMAS", "gen,clamp").split(",")


def make_mask(r, extra_dilate: int, feather_frac: float) -> np.ndarray:
    """在 core（纯唇红）基础上膨胀 + 羽化，返回 (H,W,1) float。"""
    H, W = r.mask.shape[:2]
    m = (r.core[..., 0] > 0.35).astype(np.uint8) * 255
    if extra_dilate > 0:
        k = extra_dilate // 2 * 2 + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kk = max(3, int(feather_frac * W) // 2 * 2 + 1)
    m = cv2.GaussianBlur(m, (kk, kk), 0).astype(np.float32) / 255.0
    return m[..., None]


def main() -> None:
    from musetalk_renderer import MuseTalkRenderer

    with wave.open(os.environ.get("LOCAL_AVATAR_TEST_WAV", "/tmp/xiaoya.wav")) as f:
        pcm = f.readframes(f.getnframes())
    r = MuseTalkRenderer(
        os.environ.get("LOCAL_AVATAR_IMAGE", str(Path.home() / "local-avatar" / "avatar.jpg")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk")),
        os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS", str(Path.home() / "MuseTalk" / "models")))
    r.push_audio(pcm)

    x0, y0, x1, y1 = r.crop
    w, h = x1 - x0, y1 - y0
    half = r.half
    roi = r.base[y0:y1, x0:x1].astype(np.float32)
    print(f"[p4] crop={r.crop} half={half} mask_cover={float(r.mask.mean())*100:.1f}%", flush=True)

    cells = []
    by1, by2, bx1, bx2 = r.box
    cy0 = by1 + int((by2 - by1) * 0.42)
    cy1 = by2 + int((by2 - by1) * 0.20)
    cx0, cx1 = max(0, bx1 - 34), min(r.base.shape[1], bx2 + 34)
    fx = float(os.environ.get("FX", "3.6"))

    def render(mask_full, gen, chroma):
        m_lo = np.ascontiguousarray(mask_full[half:], dtype=np.float32)
        lipzone = cv2.dilate((r.core[..., 0] > 0.35).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        lipzone = cv2.GaussianBlur(lipzone.astype(np.float32), (0, 0), 3)[..., None]
        dw = mask_full * (1.0 - lipzone)
        det = (r.detail[half:] * dw[half:]).astype(np.float32)
        sub = cv2.resize(gen.astype(np.float32), (256, 256 - r.split),
                         interpolation=cv2.INTER_LANCZOS4)
        g = cv2.resize(sub, (w, half), interpolation=cv2.INTER_LANCZOS4)
        g = g * r.gain_lo + r.bias_lo
        lo = g * m_lo + roi[half:] * (1.0 - m_lo) + det
        if chroma != "gen":
            a = cv2.cvtColor(np.clip(lo, 0, 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            b = cv2.cvtColor(np.clip(roi[half:], 0, 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            d = np.clip(a[..., 1:] - b[..., 1:], -CLAMP, CLAMP)
            if chroma in ("ykey", "ygray"):
                # 亮度变化大 = 口腔张开/牙齿露出的位置；亮度几乎不变 = 皮肤和唇面
                dy = np.abs(a[..., :1] - b[..., :1])
                wgt = np.clip((dy - 6.0) / 18.0, 0.0, 1.0)
                wgt = cv2.GaussianBlur(wgt, (0, 0), 2.0)[..., None]
                if chroma == "ykey":
                    cc = b[..., 1:] * (1.0 - wgt) + a[..., 1:] * wgt
                else:
                    # 变化大的地方推向中性：牙齿=亮中性、口腔=暗中性，
                    # 模型在这些区域的色度误差（发绿/发黄）直接丢掉
                    cc = b[..., 1:] * (1.0 - wgt) + 128.0 * wgt
            else:
                cc = b[..., 1:] + d
            lo = cv2.cvtColor(np.concatenate([a[..., :1], cc], axis=2), cv2.COLOR_YCrCb2BGR)
        out = r.base.copy()
        merged = roi.copy()
        merged[half:] = lo
        out[y0:y1, x0:x1] = np.clip(merged, 0, 255).astype(np.uint8)
        return out

    def cell(im, tag):
        c = im[cy0:cy1, cx0:cx1]
        c = cv2.resize(c, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC).copy()
        cv2.putText(c, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return c

    rows = []
    for fr in FRAMES:
      globals_frame = r._infer(r._clip(fr))
      cells = [cell(r.base, "ORIG")]
      for dil in DIL:
          mask_full = r.mask if dil < 0 else make_mask(r, dil, 0.04 if dil <= 16 else 0.06)
          cover = float(mask_full.mean()) * 100
          for chroma in CHROMAS:
              cells.append(cell(render(mask_full, globals_frame, chroma),
                                f"f{fr} d{dil} {chroma} {cover:.1f}%"))
      rows.append(np.hstack(cells))
      print(f"[p4] f{fr} done", flush=True)
    grid = np.vstack(rows) if len(rows) > 1 else rows[0]
    cv2.imwrite(str(OUT / "scan.png"), grid)
    print(f"[p4] -> scan.png {grid.shape}", flush=True)


if __name__ == "__main__":
    main()
