#!/usr/bin/env python3
"""本地数字人渲染器：24kHz PCM 音频流 + 一张正脸照 → 20fps（LOCAL_AVATAR_FPS 可调）口型同步 JPEG 帧。

后端：
- wav2lip（默认，有权重时）：真实唇形同步模型，跑在 GPU 机上
- mock（无权重/加载失败时自动降级）：音频能量驱动口型，用于验证全链路

音频 mel 计算完全自实现（numpy/scipy），不依赖 librosa/numba ——
DGX Spark 是 aarch64 + numpy 2.x，装 librosa 会拖进 numba/llvmlite 依赖地狱。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR_IN = 24000                 # 桥送来的 PCM：s16le / 24kHz / mono
SR_MEL = 16000                # Wav2Lip 训练采样率
# 出帧帧率。25 时单帧预算 40ms，而 MuseTalk 实测 38~41ms，长期贴着上限跑，
# 每段音频都会丢掉约 10% 的帧（音画对不上、嘴唇抽搐）；降到 20 后预算 50ms，
# 留出 20% 余量，实测不再跳帧。可用 LOCAL_AVATAR_FPS 覆盖。
try:
    FPS = max(10, min(30, int(os.environ.get("LOCAL_AVATAR_FPS", "20"))))
except ValueError:
    FPS = 20
N_FFT, HOP, N_MELS = 800, 200, 80
FMIN, FMAX = 55, 7600
MEL_STEP = 16                 # 每个视频帧对应的 mel 帧数（0.2s）
IMG = 96
MAX_ABS, MIN_DB, REF_DB = 4.0, -100.0, 20.0


# ------------------------------ mel 特征（复刻 Wav2Lip/audio.py） ------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_basis() -> np.ndarray:
    """librosa.filters.mel(htk=True, norm='slaney') 的等价实现。"""
    pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    hz = _mel_to_hz(pts)
    bins = np.floor((N_FFT + 1) * hz / SR_MEL).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)
    for i in range(N_MELS):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
        fb[i] *= 2.0 / (hz[i + 2] - hz[i])
    return fb


_MEL = _mel_basis()
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float64)   # librosa hann(sym=False)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """float32 16k 波形 → (80, T) 归一化 mel，数值对齐 Wav2Lip 官方 audio.py。"""
    if wav.size < N_FFT:
        wav = np.pad(wav, (0, N_FFT - wav.size))
    y = np.concatenate(([wav[0]], wav[1:] - 0.97 * wav[:-1]))
    y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    D = np.abs(np.fft.rfft(frames * _WIN, n=N_FFT, axis=1)).T
    S = 20.0 * np.log10(np.maximum(1e-5, np.dot(_MEL, D))) - REF_DB
    S = np.clip((2 * MAX_ABS) * ((S - MIN_DB) / (-MIN_DB)) - MAX_ABS,
                -MAX_ABS, MAX_ABS)
    return S.astype(np.float32)


def resample_24k_to_16k(x: np.ndarray) -> np.ndarray:
    """线性插值重采样（TTS 过来的 100ms 块很短，插值误差可忽略）。"""
    n = int(round(len(x) * SR_MEL / SR_IN))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(np.arange(n) * (len(x) / n), np.arange(len(x)), x).astype(np.float32)


# ------------------------------ 音频环形缓冲 ------------------------------

class AudioRing:
    def __init__(self, seconds: float = 1.5):
        self.max_n = int(seconds * SR_MEL)
        self.data = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.rms = 0.0
        self.last_audio_at = 0.0

    def push(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        y = resample_24k_to_16k(x)
        with self.lock:
            self.data = np.concatenate((self.data, y))[-self.max_n:]
            self.rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            self.last_audio_at = time.time()

    def tail(self) -> np.ndarray:
        with self.lock:
            return self.data.copy()

    def level(self) -> float:
        # 静音超过 0.35s 就闭合嘴巴
        if time.time() - self.last_audio_at > 0.35:
            return 0.0
        with self.lock:
            return min(1.0, self.rms * 6.0)


# ------------------------------ 肖像 / 人脸框 ------------------------------

def _placeholder_face(w: int = 576, h: int = 768) -> np.ndarray:
    """没有肖像图时的占位形象（纯代码绘制，不依赖字体）。"""
    im = Image.new("RGB", (w, h), (28, 32, 46))
    d = ImageDraw.Draw(im)
    for i in range(h):
        v = int(18 + 22 * i / h)
        d.line([(0, i), (w, i)], fill=(v, v + 4, v + 14))
    cx, cy = w // 2, int(h * 0.44)
    d.ellipse([cx - 150, cy - 190, cx + 150, cy + 190], fill=(236, 198, 172))
    d.ellipse([cx - 150, cy - 210, cx + 150, cy - 40], fill=(42, 34, 38))
    d.ellipse([cx - 62, cy - 30, cx - 24, cy + 8], fill=(40, 38, 44))
    d.ellipse([cx + 24, cy - 30, cx + 62, cy + 8], fill=(40, 38, 44))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _face_box(base: np.ndarray) -> tuple[int, int, int, int]:
    """脸框优先级：LOCAL_AVATAR_BOX 显式指定 → Haar 正脸检测 → 画面中心头部区域。

    Haar 对戴墨镜 / 逆光 / 暗背景的正脸经常给不出框，或给出覆盖大半张图的误检框；
    误检框会让 Wav2Lip 把整张图当成脸贴回去，画面直接错位。所以这里加两道闸：
    显式框可覆盖，检测框过大（>62% 画面高）直接判为误检丢弃。
    """
    h, w = base.shape[:2]
    span = max(h, w)

    raw = os.environ.get("LOCAL_AVATAR_BOX", "").strip()
    if raw:
        try:
            y1, y2, x1, x2 = (int(float(v)) for v in raw.replace(" ", "").split(","))
            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w and max(y2 - y1, x2 - x1) < span * 0.95:
                print(f"[renderer] 使用显式脸框 y1={y1} y2={y2} x1={x1} x2={x2}", flush=True)
                return y1, y2, x1, x2
            print(f"[renderer] LOCAL_AVATAR_BOX={raw!r} 越界或过大，已忽略", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[renderer] LOCAL_AVATAR_BOX 解析失败({e})，已忽略", flush=True)

    try:
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        faces = cas.detectMultiScale(g, 1.1, 5, minSize=(80, 80))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            if fh > h * 0.62:
                print(f"[renderer] Haar 框过大({fw}x{fh})，判为误检，改用居中脸框", flush=True)
            else:
                x1 = max(0, int(x - fw * 0.10))
                x2 = min(w, int(x + fw * 1.10))
                y1 = max(0, int(y - fh * 0.14))
                y2 = min(h, int(y + fh * 1.24))
                print(f"[renderer] Haar 脸框 x={x} y={y} {fw}x{fh}", flush=True)
                return y1, y2, x1, x2
        else:
            print("[renderer] 未检出人脸，改用居中脸框（可用 LOCAL_AVATAR_BOX 手动指定）", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[renderer] 人脸检测异常({e})，改用居中脸框", flush=True)
    return int(h * 0.12), int(h * 0.72), int(w * 0.24), int(w * 0.76)


# ------------------------------ 渲染器 ------------------------------

class BaseRenderer:
    def __init__(self, avatar_path: str = ""):
        p = Path(avatar_path) if avatar_path else None
        self.base = cv2.imread(str(p)) if p and p.is_file() else _placeholder_face()
        if self.base is None:
            self.base = _placeholder_face()
        h, w = self.base.shape[:2]
        if max(h, w) > 1000:
            s = 1000 / max(h, w)
            self.base = cv2.resize(self.base, (int(w * s), int(h * s)))
        self.box = _face_box(self.base)
        self.face = cv2.resize(self.base[self.box[0]:self.box[1], self.box[2]:self.box[3]],
                               (IMG, IMG))
        self.ring = AudioRing()
        self.action = ""
        self.action_at = 0.0
        self.jpeg: bytes = b""
        self.lock = threading.Lock()
        self.frames = 0
        self.backend = "base"
        # 画面左上角的 "LOCAL WAV2LIP" 角标：仅排查/取证时开（默认关，避免直播画面有噪音）
        self.watermark = (os.environ.get("LOCAL_AVATAR_WATERMARK", "0").strip().lower()
                          not in ("", "0", "off", "false", "no"))

    def push_audio(self, pcm: bytes) -> None:
        self.ring.push(pcm)

    def set_action(self, text: str) -> None:
        self.action = (text or "").strip()[:60]
        self.action_at = time.time()

    def _overlay(self, frame: np.ndarray) -> bytes:
        """叠加动作指令提示条（云端 type9 的动作在本地降级为画面提示）。"""
        out = frame.copy()
        if self.action and time.time() - self.action_at < 4.0:
            h, w = out.shape[:2]
            cv2.rectangle(out, (int(w * 0.08), int(h * 0.80)),
                          (int(w * 0.92), int(h * 0.89)), (18, 18, 26), -1)
            cv2.putText(out, self.action.encode("ascii", "ignore").decode()[:48],
                        (int(w * 0.10), int(h * 0.86)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (120, 230, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else b""

    def tick(self) -> None:
        jpg = self.render_frame()
        if jpg:
            with self.lock:
                self.jpeg = jpg
                self.frames += 1

    def render_frame(self) -> bytes:
        raise NotImplementedError

    def snapshot(self) -> bytes:
        with self.lock:
            return self.jpeg


class MockRenderer(BaseRenderer):
    """零依赖降级：音频能量驱动嘴型，用于在没有权重时验证整条链路。"""

    def __init__(self, avatar_path: str = ""):
        super().__init__(avatar_path)
        self.backend = "mock"

    def render_frame(self) -> bytes:
        out = self.base.copy()
        h, w = out.shape[:2]
        y1, y2, x1, x2 = self.box
        cx, cy = (x1 + x2) // 2, y1 + int((y2 - y1) * 0.76)
        level = self.ring.level()
        mw = int((x2 - x1) * 0.16)
        mh = max(3, int((y2 - y1) * 0.13 * level))
        cv2.ellipse(out, (cx, cy), (mw, mh), 0, 0, 360, (60, 50, 120), -1)
        if self.watermark:
            cv2.putText(out, "LOCAL MOCK", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (90, 220, 255), 2, cv2.LINE_AA)
        return self._overlay(out)


class Wav2LipRenderer(BaseRenderer):
    """真实 Wav2Lip 唇形同步：每个视频帧取最近 0.2s 音频生成嘴部并贴回肖像。

    画质关键：模型固定输出 96×96，直接放大贴回 500px 级的脸部区域会是 5~6 倍插值，
    整块区域必然糊成一片（连带把鼻梁、皮肤纹理一起磨平，还留下矩形色差）。
    所以默认走 residual 合成：只把模型算出的「低分辨率变化量」(pred - 输入) 放大后
    叠加到**原始高清脸部**上 —— 皮肤/唇线/睫毛的高频全部保留，只新增嘴部运动，
    这在视觉上比整块贴回清晰得多。direct 保留为对照实现。
    """

    def __init__(self, avatar_path: str = "", checkpoint: str = "", upstream: str = ""):
        super().__init__(avatar_path)
        import torch
        self.torch = torch
        sys.path.insert(0, upstream)
        from models.wav2lip import Wav2Lip  # noqa: E402  上游仓库的模型定义
        model = Wav2Lip()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ck.get("state_dict", ck)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        self.model = model.eval().to("cuda")
        self.device = "cuda"
        self.backend = "wav2lip"

        # 贴回策略与锐化强度（可用环境变量调，默认值经实拍对比选定）
        # residual-mask：只把模型"真的改动过"的嘴部区域叠回高清原脸，其余像素不动 —— 
        # qtest.py 实测 Laplacian 83.4（原图 72.3 / direct 62.5），且无方块色差。
        self.compose = (os.environ.get("LOCAL_AVATAR_COMPOSE", "residual-mask").strip().lower()
                        or "residual-mask")
        try:
            self.sharpen = float(os.environ.get("LOCAL_AVATAR_SHARPEN", "0.55"))
        except ValueError:
            self.sharpen = 0.55
        try:
            self.delta_blur = float(os.environ.get("LOCAL_AVATAR_DELTA_BLUR", "0.6"))
        except ValueError:
            self.delta_blur = 0.6
        try:
            self.feather = float(os.environ.get("LOCAL_AVATAR_FEATHER", "0.12"))
        except ValueError:
            self.feather = 0.12

        y1, y2, x1, x2 = self.box
        self.face_hr = self.base[y1:y2, x1:x2].astype(np.float32)   # 原始高清脸
        self.face_lr = self.face.astype(np.float32)                 # 96×96 模型输入
        print(f"[renderer] Wav2Lip loaded on {torch.cuda.get_device_name(0)} "
              f"compose={self.compose} sharpen={self.sharpen} face={x2 - x1}x{y2 - y1}",
              flush=True)

    @staticmethod
    def _feather_mask(ph: int, pw: int, ratio: float = 0.14) -> np.ndarray:
        """四周渐隐的 alpha 蒙版，用于把模型输出的脸柔和贴回肖像，消掉硬边接缝。"""
        m = np.ones((ph, pw), np.float32)
        fy = min(max(2, int(ph * ratio)), max(2, ph // 2))
        fx = min(max(2, int(pw * ratio)), max(2, pw // 2))
        ramp_y = np.hanning(fy * 2)[:fy].astype(np.float32)
        ramp_x = np.hanning(fx * 2)[:fx].astype(np.float32)
        ramp_y /= max(ramp_y.max(), 1e-6)
        ramp_x /= max(ramp_x.max(), 1e-6)
        m[:fy, :] *= ramp_y[:, None]
        m[-fy:, :] *= ramp_y[::-1][:, None]
        m[:, :fx] *= ramp_x[None, :]
        m[:, -fx:] *= ramp_x[None, ::-1]
        return m[..., None]

    def _mel(self) -> np.ndarray:
        """最近 0.2s 音频 → (80,16) mel，喂给模型。"""
        wav = self.ring.tail()
        if wav.size < N_FFT:
            return np.zeros((N_MELS, MEL_STEP), dtype=np.float32)
        mel = melspectrogram(wav[-3200:])
        if mel.shape[1] < MEL_STEP:
            mel = np.pad(mel, ((0, 0), (MEL_STEP - mel.shape[1], 0)))
        return mel[:, -MEL_STEP:]

    def _predict(self, mel: np.ndarray) -> np.ndarray:
        """模型推理 → (96,96,3) float32 / 0~255 的嘴型帧。"""
        torch = self.torch
        face = self.face
        masked = face.copy()
        # 与 Wav2Lip 官方 inference.py 对齐：遮的是「下半部分行」(img_size//2:)，不是右半边列。
        # 之前写成 [:, IMG//2:] 会遮掉半张脸的右半边，模型失去下半脸上下文，嘴部会糊成一片。
        masked[IMG // 2:, :] = 0
        x = np.concatenate((masked, face), axis=2) / 255.0
        xf = torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).float().to(self.device)
        mf = torch.from_numpy(mel[None, None]).float().to(self.device)   # (1,1,80,16)
        with torch.no_grad():
            pred = self.model(mf, xf)
        return pred.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0

    def _delta_mask(self, delta: np.ndarray) -> np.ndarray:
        """按"模型到底改了哪里"生成 0~1 的平滑蒙版（96×96）。

        Wav2Lip 的整块输出还会顺带改肤色/亮度/边缘，这些低分辨率误差放大后就是
        脸颊上的方块色差。但我们真正要的只有**嘴部开合**——所以直接用变化量的
        幅度分布做蒙版：嘴在哪，模型就在哪改得多，蒙版自然落在嘴上；
        其余皮肤、头发、墨镜保持原图一个像素都不动。
        """
        mag = np.abs(delta).mean(axis=2)
        mag = cv2.GaussianBlur(mag, (0, 0), 2.2)
        peak = float(mag.max())
        if peak < 2.5:                       # 静音/几乎没动嘴 → 整帧保持原图
            return np.zeros((IMG, IMG), np.float32)
        m = np.clip((mag - 0.12 * peak) / (0.50 * peak), 0.0, 1.0)
        return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.6)

    def _compose(self, pred: np.ndarray) -> np.ndarray:
        """把 96×96 预测合成为脸框大小的 BGR float32 贴片。"""
        y1, y2, x1, x2 = self.box
        w, h = x2 - x1, y2 - y1
        hr = self.face_hr

        direct = None
        if self.compose in ("direct", "blend"):
            direct = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if self.compose == "direct":
            alpha = self._feather_mask(h, w, 0.14)
            return direct * alpha + hr * (1.0 - alpha)

        # residual：低分辨率变化量（嘴部开合/齿舌）放大后叠加到高清原脸
        delta = pred - self.face_lr
        if self.compose in ("residual-mask", "mask"):
            mask = cv2.resize(self._delta_mask(delta), (w, h),
                              interpolation=cv2.INTER_LINEAR)[..., None]
            if self.delta_blur > 0:
                delta = cv2.GaussianBlur(delta, (0, 0), self.delta_blur)
            merged = hr + cv2.resize(delta, (w, h),
                                     interpolation=cv2.INTER_LANCZOS4) * mask
        else:
            if self.delta_blur > 0:
                delta = cv2.GaussianBlur(delta, (0, 0), self.delta_blur)
            merged = hr + cv2.resize(delta, (w, h), interpolation=cv2.INTER_LANCZOS4)
            alpha = self._feather_mask(h, w, self.feather)   # 四边收一下，消接缝
            merged = merged * alpha + hr * (1.0 - alpha)

        if self.compose == "blend":
            merged = merged * 0.72 + direct * 0.28
        if self.sharpen > 0:
            blur = cv2.GaussianBlur(merged, (0, 0), 1.1)
            merged = merged + self.sharpen * (merged - blur)
        return merged

    def render_frame(self) -> bytes:
        pred = self._predict(self._mel())
        out = self.base.copy()
        y1, y2, x1, x2 = self.box
        out[y1:y2, x1:x2] = np.clip(self._compose(pred), 0, 255).astype(np.uint8)
        if self.watermark:
            # 默认关闭：这行字会压在画面左上角（统计条也在那），直播观感上是噪音。
            # 需要取证/排查时用 LOCAL_AVATAR_WATERMARK=1 打开。
            cv2.putText(out, "LOCAL WAV2LIP", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (90, 220, 255), 2, cv2.LINE_AA)
        return self._overlay(out)


def build_renderer(avatar: str = "", checkpoint: str = "", upstream: str = "") -> BaseRenderer:
    """后端选择：LOCAL_AVATAR_BACKEND = ditto / musetalk / wav2lip / mock / auto。

    - ditto（推荐）：warp 原图特征做形变，肤色纹理锚定原图，没有「嘴浮在图上」问题；
      原生 25fps，需 GPU 机上有 ~/ditto-talkinghead + TensorRT 引擎
    - musetalk：256² 潜空间重绘，融合再好也会有贴片感，留作对照
    - wav2lip：96²，仅在无 TRT 环境时兜底
    auto 依次尝试 ditto → musetalk → wav2lip → mock。
    """
    backend = (os.environ.get("LOCAL_AVATAR_BACKEND", "ditto") or "ditto").strip().lower()
    muse_dir = os.environ.get("LOCAL_AVATAR_MUSETALK_DIR", str(Path.home() / "MuseTalk"))
    muse_models = os.environ.get("LOCAL_AVATAR_MUSETALK_MODELS",
                                 str(Path(muse_dir) / "models"))
    ditto_dir = os.environ.get("LOCAL_AVATAR_DITTO_DIR", str(Path.home() / "ditto-talkinghead"))
    ditto_root = os.environ.get("LOCAL_AVATAR_DITTO_ROOT",
                                str(Path(ditto_dir) / "checkpoints" / "ditto_trt_fast"))
    ditto_cfg = os.environ.get("LOCAL_AVATAR_DITTO_CFG", "")

    if backend in ("ditto", "dt", "auto"):
        try:
            from ditto_renderer import DittoRenderer
            return DittoRenderer(avatar, ditto_dir, ditto_root, ditto_cfg)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if backend != "auto":
                raise
            print(f"[renderer] Ditto 不可用，回退: {msg}", flush=True)

    if backend in ("musetalk", "muse", "mt", "auto"):
        try:
            from musetalk_renderer import MuseTalkRenderer
            return MuseTalkRenderer(avatar, muse_dir, muse_models)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if backend != "auto":
                raise
            print(f"[renderer] MuseTalk 不可用，回退: {msg}", flush=True)

    if checkpoint and Path(checkpoint).is_file() and upstream and Path(upstream).is_dir():
        try:
            return Wav2LipRenderer(avatar, checkpoint, upstream)
        except Exception as e:  # noqa: BLE001
            print(f"[renderer] Wav2Lip 加载失败，降级 mock: {type(e).__name__}: {e}", flush=True)
    else:
        print(f"[renderer] 未找到 Wav2Lip 权重/上游代码，使用 mock 后端", flush=True)
    return MockRenderer(avatar)
